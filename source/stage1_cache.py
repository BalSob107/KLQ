# Six spaces per block (all via forward-PRE hooks on the consuming module):
#   res  : input_layernorm input        (residual stream, pre-attn)  [D]
#   post : post_attention_layernorm in  (residual stream, pre-mlp)   [D]
#   qkv  : q_proj input  (= LN1 output — what q/k/v actually read)   [D]
#   ctx  : o_proj input  (attention context)                         [D]
#   mlp  : gate_proj input (= LN2 output — what gate/up read)        [D]
#   int  : down_proj input (MLP intermediate)                        [DI]
#
# Per block: accumulate float64 sums + second moments over the calib
# windows, C = S2/n - mu mu^T, eigh, store, discard S2 before next block
# (keeping all blocks' S2 at once would be ~5 GB fp64 for Qwen's DI=4864).
#
# Eigh is done in fp32 on GPU after fp64 accumulation: the tiny-eigenvalue
# directions get slightly noisier vectors, but those are the dead/sheath
# directions. Spike directions (the ones that get bits) are fine.
#
# Output: spaces_{tag}.pt = {space: {"mu": (NB,d), "evals": (NB,d),
#                                    "evecs": (NB,d,d)}}  [fp32, CPU]
#   Note: torch.linalg.eigh returns ASCENDING eigenvalues, so "top"
#   directions are the LAST indices.
#
# Usage:  python stage1_cache.py --model qwen --nwin 8 --seq 512 --pb 8

import argparse
import time
import torch
import os

from config import MODELS, paths
from common import get_teacher, blocks, load_windows

EIGH_DTYPE = torch.float32

SPACES = ["res", "post", "qkv", "ctx", "mlp", "int"]


def mods_of(L):
    return {
        "res":  L.input_layernorm,
        "post": L.post_attention_layernorm,
        "qkv":  L.self_attn.q_proj,
        "ctx":  L.self_attn.o_proj,
        "mlp":  L.mlp.gate_proj,
        "int":  L.mlp.down_proj,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--nwin", type=int, default=64,
                    help="nwin*seq = covariance samples; keep >= ~5x the "
                         "largest space dim (Qwen int = 4864 -> nwin 64)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--pb", type=int, default=8, help="Parallel batches.")
    ap.add_argument("--seed", type=int, default=12345)   
    ap.add_argument("--wikitext", action="store_true",
                    help="calibrate on wikitext-2-raw-v1 train (QuIP#/VPTQ "
                         "distribution) instead of VAL_BIN")
    a = ap.parse_args()

    cfg = MODELS[a.model]
    P = paths(a.model)
    model = get_teacher(cfg)
    src = "wikitext" if a.wikitext else None
    wins, starts = load_windows(cfg, a.nwin, a.seq, a.seed,
                                segment="calib", source=src)
    print(f"cache windows: {a.nwin} x {a.seq} tokens, seed {a.seed}, "
          f"source {src or 'val_bin'}, starts {starts.tolist()}")

    cache = {s: {"mu": [], "evals": [], "evecs": []} for s in SPACES}

    for t in range(cfg["NB"]):
        t0 = time.time()
        S, v, n = {}, {}, {}
        handles = []

        def mk(name):
            def h(module, args):
                x = args[0].detach()
                x = x.reshape(-1, x.shape[-1]).double()
                if name not in S:
                    d = x.shape[-1]
                    S[name] = torch.zeros(d, d, dtype=torch.float64,
                                          device=x.device)
                    v[name] = torch.zeros(d, dtype=torch.float64,
                                          device=x.device)
                    n[name] = 0
                S[name] += x.T @ x
                v[name] += x.sum(0)
                n[name] += x.shape[0]
            return h

        for name, m in mods_of(blocks(model)[t]).items():
            handles.append(m.register_forward_pre_hook(mk(name)))

        for s in range(0, a.nwin, a.pb):
            ids = torch.tensor(wins[s:s + a.pb], dtype=torch.long,
                               device="cuda")
            model.model(input_ids=ids)

        for h_ in handles:
            h_.remove()

        for name in SPACES:
            mu = v[name] / n[name]
            C = S[name] / n[name] - torch.outer(mu, mu)
            C = 0.5 * (C + C.T)
            ev, V = torch.linalg.eigh(C.to(EIGH_DTYPE))
            cache[name]["mu"].append(mu.float().cpu())
            cache[name]["evals"].append(ev.clamp_min(0).float().cpu())
            cache[name]["evecs"].append(V.float().cpu())
            del S[name], v[name]

        print(f"block {t + 1}/{cfg['NB']}  {time.time() - t0:.1f}s",
              flush=True)
        torch.cuda.empty_cache()

    # memory-safe stacking: torch.stack would double the peak (list + new
    # contiguous tensor). llama int evecs are 11008^2*4*32 = 15.5 GB fp32.
    # Preallocate, copy block-by-block, and free each source tensor.
    # Oversized spaces (raw fp32 > 4 GB) are stored fp16: their only
    # consumer is step2's per-direction sweep, which is fp16-safe, and the
    # tail directions are estimation noise anyway. Threshold leaves the
    # 135M/qwen caches bit-identical to before.
    out = {}
    for s in SPACES:
        out[s] = {}
        for k, vv in cache[s].items():
            store_dt = vv[0].dtype
            if k == "evecs" and vv[0].numel() * 4 * len(vv) > (4 << 30):
                store_dt = torch.float16
                print(f"[step1] {s} evecs: "
                      f"{vv[0].numel() * 4 * len(vv) / 2**30:.1f} GB fp32 "
                      f"-> storing fp16")
            buf = torch.empty((len(vv),) + tuple(vv[0].shape),
                              dtype=store_dt)
            for i, ten in enumerate(vv):
                buf[i].copy_(ten)
                vv[i] = None
            out[s][k] = buf
        cache[s] = None
    torch.save(out, P["spaces"])
    print(f"saved -> {P['spaces']}")
    for s in SPACES:
        print(f"  {s:5s} evals {tuple(out[s]['evals'].shape)} "
              f"evecs {tuple(out[s]['evecs'].shape)}")


if __name__ == "__main__":
    main()
