# For each space, each block, each measured direction i:
#   perturb the module input by  delta_i = sqrt(evals[i]) * evecs[:, i]
#   (one sigma along that eigendirection, added to every token),
#   re-forward the eval windows, record mean KL(base || pert).
#
# Direction convention: evals are ASCENDING (see stage1), so the expensive
# top directions are the last indices.
#
# Estimated forwards are printed before the sweep starts. 
#
# Most compute intensive part of the process.
#
# Output: sens_{space}_{tag}.pt = {"kl": (NB,d) NaN where unmeasured,
#         "measured": (NB,d) bool, "meta": {...}}
#
# Usage:  python stage2_sens.py --model qwen --spaces res
#         python stage2_sens.py --model qwen --spaces qkv,ctx,mlp,int

import argparse
import os
import time
import torch

from config import MODELS, paths
from common import (get_teacher, blocks, load_windows,
                    forward_hidden, hidden_logprobs, kl_mean)

MODS = {
    "res":  lambda L: L.input_layernorm,
    "post": lambda L: L.post_attention_layernorm,
    "qkv":  lambda L: L.self_attn.q_proj,
    "ctx":  lambda L: L.self_attn.o_proj,
    "mlp":  lambda L: L.mlp.gate_proj,
    "int":  lambda L: L.mlp.down_proj,
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--spaces", default="res")
    ap.add_argument("--nwin", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--pb", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)    
    ap.add_argument("--topk", type=int, default=2048)  
    ap.add_argument("--tail", type=int, default=512)  
    ap.add_argument("--full", action="store_true",
                    help="force a full D-dim sweep regardless of dim")
    ap.add_argument("--wikitext", action="store_true",
                    help="use wikitext-2 train tokens (must match stage1)")
    a = ap.parse_args()

    cfg = MODELS[a.model]
    P = paths(a.model)
    model = get_teacher(cfg)
    cache = torch.load(P["spaces"], map_location="cpu")

    wins, starts = load_windows(cfg, a.nwin, a.seq, a.seed,
                                segment="val",
                                source="wikitext" if a.wikitext else None)
    ids_all = torch.tensor(wins, dtype=torch.long, device="cuda")
    print(f"sens windows: {a.nwin} x {a.seq}, seed {a.seed}, "
          f"starts {starts.tolist()}")

    print("computing base log-probs...", flush=True)
    base_lp = hidden_logprobs(model, forward_hidden(model, ids_all))

    for space in [s.strip() for s in a.spaces.split(",")]:
        ev_all = cache[space]["evals"]
        V_all = cache[space]["evecs"]
        NB, d = ev_all.shape

        # direction plan (same for every block so 'measured' masks align).
        # Small dims: full sweep. Large dims: top-k by eigenvalue + seeded
        # random tail -- the tail keeps the plan unbiased (low-variance
        # directions CAN have high KL; see qkv spearman 0.458 on qwen),
        # while the top-k guarantees precise prices near the waterline.
        if a.full or d <= 2048:
            dirs = torch.arange(d)
        else:
            top = torch.arange(d - a.topk, d)
            tail = torch.randperm(d - a.topk,
                                  generator=torch.Generator().manual_seed(7)
                                  )[:a.tail]
            dirs = torch.cat([top, tail]).sort().values
        n_forwards = NB * len(dirs)
        print(f"\n=== {space}: dim {d}, {len(dirs)} dirs/block, "
              f"{n_forwards} forwards total ===", flush=True)

        KL = torch.full((NB, d), float("nan"))
        measured = torch.zeros(NB, d, dtype=torch.bool)

        for t in range(NB):
            ev = ev_all[t].cuda()
            V = V_all[t].cuda()
            mod = MODS[space](blocks(model)[t])
            state = {"i": None}

            def h(module, args, state=state, V=V, ev=ev):
                i = state["i"]
                if i is None:
                    return None
                delta = ev[i].sqrt() * V[:, i]          # (d,) fp32
                return (args[0] + delta.to(args[0].dtype),) + args[1:]

            handle = mod.register_forward_pre_hook(h)
            t0 = time.time()
            for cnt, i in enumerate(dirs.tolist()):
                state["i"] = i
                lp = hidden_logprobs(model, forward_hidden(model, ids_all))
                KL[t, i] = kl_mean(base_lp, lp)
                del lp
                if cnt % 64 == 0 or cnt == len(dirs) - 1:
                    el = time.time() - t0
                    rate = (cnt + 1) / el
                    print(f"{space} L{t:02d} dir {cnt + 1}/{len(dirs)}  "
                          f"{el / (cnt + 1) * 1000:.0f} ms/dir  "
                          f"block ETA {(len(dirs) - cnt - 1) / rate / 60:.1f} min",
                          flush=True)
            handle.remove()
            measured[t, dirs] = True
            del ev, V
            torch.cuda.empty_cache()

        out = dict(kl=KL, measured=measured, space=space,
                   meta=dict(model=a.model, nwin=a.nwin, seq=a.seq,
                             seed=a.seed, topk=a.topk, tail=a.tail))
        f = os.path.join(P["sens_dir"], f"sens_{space}_{P['tag']}.pt")
        torch.save(out, f)
        print(f"saved -> {f}", flush=True)


if __name__ == "__main__":
    main()
