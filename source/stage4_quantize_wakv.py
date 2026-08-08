#   Phase 0  fresh teacher, fp16 baseline (fineweb val [+ WikiText-2])
#   Phase 1  W: sequential re-capture KLQ (common.quantize_weights)
#   Phase 2  KV: per-(layer, KV-head) eigenbases + measured-KL sweep
#   Phase 3  A: eigenbasis quant hooks on both LN inputs (res, post),
#            per-direction global scales recalibrated UNDER the quantized
#            weights; KV: per-head eigenbasis quant hooks on k/v_proj
#            outputs (pre-RoPE, same convention as the 135M KV work)
#   Phase 4  eval + report: W true bpw / A avg bits / KV avg bits, ppl,
#            x fp16 (plus WikiText-2 headline with --wikitext)
#
# Usage:
#   python stage4_quantize_wakv.py --model qwen --w_avg 4.25 --a_avg 4.5 --kv_avg 4.5
#   python stage4_quantize_wakv.py --model qwen --w_avg 4.25 --wikitext
#   python stage4_quantize_wakv.py --model qwen --no_kv --no_a     (W-only sanity)

import argparse
import json
import os
import time
import torch

from config import MODELS, paths
from common import (get_teacher, blocks, load_windows, waterfill, ppl_eval,
                    forward_hidden, hidden_logprobs, kl_mean,
                    quantize_weights)


# ---------------- KV eigenbases + measured KL ----------------

@torch.no_grad()
def kv_eigen(model, cfg, nwin, seq, pb, seed):
    """Per (layer, KV-head) mu/evals/evecs of k_proj/v_proj outputs."""
    H, Dh = cfg["H_KV"], cfg["HEAD_DIM"]
    wins, _ = load_windows(cfg, nwin, seq, seed, segment="calib")
    eig = {nm: {"mu": [], "evals": [], "evecs": []} for nm in ("k", "v")}
    for t in range(cfg["NB"]):
        L = blocks(model)[t]
        acc = {}

        def mk(name):
            def h(module, args, out):
                y = out.reshape(-1, H, Dh).double()
                if name not in acc:
                    acc[name] = [torch.zeros(H, Dh, Dh, dtype=torch.float64,
                                             device=y.device),
                                 torch.zeros(H, Dh, dtype=torch.float64,
                                             device=y.device), 0]
                S, v, _ = acc[name]
                S += torch.einsum("nhd,nhe->hde", y, y)
                v += y.sum(0)
                acc[name][2] += y.shape[0]
            return h

        hs = [L.self_attn.k_proj.register_forward_hook(mk("k")),
              L.self_attn.v_proj.register_forward_hook(mk("v"))]
        for s in range(0, nwin, pb):
            ids = torch.tensor(wins[s:s + pb], dtype=torch.long,
                               device="cuda")
            model.model(input_ids=ids)
        for h_ in hs:
            h_.remove()
        for nm in ("k", "v"):
            S, v, n = acc[nm]
            mu = v / n
            C = S / n - torch.einsum("hd,he->hde", mu, mu)
            ev, V = torch.linalg.eigh((0.5 * (C + C.transpose(-1, -2)))
                                      .float())
            eig[nm]["mu"].append(mu.float().cpu())
            eig[nm]["evals"].append(ev.clamp_min(0).float().cpu())
            eig[nm]["evecs"].append(V.float().cpu())
    return {nm: {k: torch.stack(vv) for k, vv in eig[nm].items()}
            for nm in ("k", "v")}          # tensors (NB, H, ...)


@torch.no_grad()
def fresh_a_bases(model, cfg, P, nwin, seq, pb, seed):
    """Phase 2.5: re-capture res/post eigenspaces under the CURRENT
    (weight-quantized) model. The step1 cache is fp16-teacher geometry;
    at low weight budgets (W3-VQ and below) the activation basis drifts
    enough that fp16 axes are wrong axes. Prices still come from the
    fp16 sens sweep (re-indexed by direction), same caveat as weights."""
    cache = torch.load(P["spaces"], map_location="cpu")
    D = cfg["D"]
    wins, _ = load_windows(cfg, nwin, seq, seed, segment="calib")
    for sp, attr in (("res", "input_layernorm"),
                     ("post", "post_attention_layernorm")):
        for t in range(cfg["NB"]):
            mod = getattr(blocks(model)[t], attr)
            S = torch.zeros(D, D, dtype=torch.float64, device="cuda")
            v = torch.zeros(D, dtype=torch.float64, device="cuda")
            n = [0]

            def h(module, args, S=S, v=v, n=n):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).double()
                S += x.T @ x
                v += x.sum(0)
                n[0] += x.shape[0]

            hd = mod.register_forward_pre_hook(h)
            for s in range(0, nwin, pb):
                ids = torch.tensor(wins[s:s + pb], dtype=torch.long,
                                   device="cuda")
                model.model(input_ids=ids)
            hd.remove()
            mu = v / n[0]
            C = S / n[0] - torch.outer(mu, mu)
            C = 0.5 * (C + C.T)
            ev, V = torch.linalg.eigh(C.float())
            cache[sp]["mu"][t] = mu.float().cpu()
            cache[sp]["evals"][t] = ev.clamp_min(0).float().cpu()
            cache[sp]["evecs"][t] = V.float().cpu()
            print(f"  fresh {sp} basis block {t + 1}/{cfg['NB']}",
                  flush=True)
    return cache


@torch.no_grad()
def transfer_sens(P, a_cache, sens, sp, power=2.0):
    """Phase 2.6 (cheap): transfer fp16 KL prices onto the FRESH bases
    via basis matching, no new forwards. For each fresh direction j,
    price = weighted average of old prices with weights |<v_new_j,
    v_old_i>|^power (normalized, unmeasured old dirs masked out)."""
    old = torch.load(P["spaces"], map_location="cpu")[sp]
    V_old = old["evecs"].cuda().float()          # (NB, D, D), cols asc
    V_new = a_cache[sp]["evecs"].cuda().float()
    kl_old = sens["kl"].cuda().float()           # (NB, D), NaN unmeasured
    M = torch.einsum("ndi,ndj->nij", V_new, V_old).abs()   # new x old
    w = M.pow(power) * (~kl_old.isnan()).float().unsqueeze(1)
    kl_new = (w @ kl_old.nan_to_num(0.0).unsqueeze(-1)).squeeze(-1) \
        / w.sum(-1).clamp_min(1e-12)
    D = V_new.shape[-1]
    top = slice(D - 16, D)
    best = M[:, :, :].amax(-1)                   # best old match per new
    print(f"  transfer {sp}: best-match |overlap| of fresh top-16 dirs "
          f"(mean over blocks): {best[:, top].mean():.3f}   "
          f"block0 {best[0, top].mean():.3f}  "
          f"block-1 {best[-1, top].mean():.3f}", flush=True)
    out = dict(sens)
    out["kl"] = kl_new.cpu()
    out["meta"] = dict(sens.get("meta", {}), transferred=True,
                       power=power)
    return out


@torch.no_grad()
def fresh_a_sens(model, cfg, P, cache, sp, attr, topk, tail,
                 nwin=2, seq=256, pb=2, seed=0, save=None):
    """Phase 2.75: re-run the causal-KL probe for res/post UNDER the
    quantized model, along the FRESH bases from phase 2.5.

    fp16 prices re-indexed onto fresh axes are misassigned when the basis
    rotates (measured: top-16 subspace overlap ~0.5 at W3-VQ late blocks),
    so both axes AND prices must come from the same geometry. Same probe
    as step2: delta_i = sqrt(evals[i]) * evecs[:, i] added to the LN
    input of the quantized model; KL against the quantized model's own
    base log-probs. Direction plan mirrors stage2 (top-k by fresh
    eigenvalue + seeded random tail; NaN -> tail median via tail_fill).
    Cached to disk keyed by the weight config so reruns skip it."""
    if save is not None and os.path.exists(save):
        print(f"  fresh {sp} sens: loading cached "
              f"{os.path.basename(save)}")
        return torch.load(save, map_location="cpu")
    D = cfg["D"]
    wins, _ = load_windows(cfg, nwin, seq, seed, segment="val")
    ids = torch.tensor(wins, dtype=torch.long, device="cuda")
    base = hidden_logprobs(model, forward_hidden(model, ids))
    if D <= 2048:
        dirs = torch.arange(D)
    else:
        top = torch.arange(D - topk, D)
        tl = torch.randperm(D - topk,
                            generator=torch.Generator().manual_seed(7)
                            )[:tail]
        dirs = torch.cat([top, tl]).sort().values
    KL = torch.full((cfg["NB"], D), float("nan"))
    measured = torch.zeros(cfg["NB"], D, dtype=torch.bool)
    print(f"  fresh {sp} sens: {len(dirs)} dirs/block x {cfg['NB']} blocks"
          f" = {len(dirs) * cfg['NB']} forwards", flush=True)
    t00 = time.time()
    for t in range(cfg["NB"]):
        ev = cache[sp]["evals"][t].cuda()
        V = cache[sp]["evecs"][t].cuda()
        mod = getattr(blocks(model)[t], attr)
        state = {"i": None}

        def h(module, args, state=state, V=V, ev=ev):
            i = state["i"]
            if i is None:
                return None
            delta = ev[i].sqrt() * V[:, i]
            return (args[0] + delta.to(args[0].dtype),) + args[1:]

        handle = mod.register_forward_pre_hook(h)
        t0 = time.time()
        for cnt, i in enumerate(dirs.tolist()):
            state["i"] = i
            lp = hidden_logprobs(model, forward_hidden(model, ids))
            KL[t, i] = kl_mean(base, lp)
            del lp
            if cnt % 128 == 0 or cnt == len(dirs) - 1:
                el = time.time() - t0
                rate = (cnt + 1) / el
                print(f"  fresh {sp} L{t:02d} dir {cnt + 1}/{len(dirs)}  "
                      f"{el / (cnt + 1) * 1000:.0f} ms/dir  block ETA "
                      f"{(len(dirs) - cnt - 1) / rate / 60:.1f} min",
                      flush=True)
        handle.remove()
        measured[t, dirs] = True
        del ev, V
        torch.cuda.empty_cache()
    el = time.time() - t00
    print(f"  fresh {sp} sens done in {el / 60:.1f} min", flush=True)
    # NaN (unmeasured tail) -> median of the measured tail directions,
    # same fill step2 documents; keeps the random-tail plan unbiased.
    if D > 2048:
        tl_ix = torch.randperm(D - topk,
                               generator=torch.Generator().manual_seed(7)
                               )[:tail]
        for t in range(cfg["NB"]):
            tm = KL[t, tl_ix]
            fill = tm[~tm.isnan()].median()
            KL[t] = torch.where(KL[t].isnan(), fill, KL[t])
    kl_filled = KL.nan_to_num(0.0)
    out = dict(kl=kl_filled, measured=measured, space=sp,
               meta=dict(fresh=True, topk=topk, tail=tail,
                         nwin=nwin, seq=seq, seed=seed))
    if save is not None:
        torch.save(out, save)
        print(f"  saved -> {save}", flush=True)
    return out


@torch.no_grad()
def kv_sens(model, cfg, eig, nwin, seq, pb, seed, save):
    """Mean KL(base||pert) per (layer, head, dir); cached to disk."""
    if os.path.exists(save):
        print(f"  KV sens: loading cached {os.path.basename(save)}")
        return torch.load(save, map_location="cpu")
    H, Dh = cfg["H_KV"], cfg["HEAD_DIM"]
    wins, _ = load_windows(cfg, nwin, seq, seed, segment="val")
    ids = torch.tensor(wins, dtype=torch.long, device="cuda")
    base = hidden_logprobs(model, forward_hidden(model, ids))
    KL = {nm: torch.zeros(cfg["NB"], H, Dh) for nm in ("k", "v")}
    t00 = time.time()
    for t in range(cfg["NB"]):
        L = blocks(model)[t]
        for nm, proj in (("k", L.self_attn.k_proj),
                         ("v", L.self_attn.v_proj)):
            V = eig[nm]["evecs"][t].cuda()
            ev = eig[nm]["evals"][t].cuda()
            state = {"h": None, "i": None}

            def h(module, args, out, state=state, V=V, ev=ev):
                if state["i"] is None:
                    return out
                y = out.reshape(*out.shape[:-1], H, Dh)
                d = ev[state["h"], state["i"]].sqrt() \
                    * V[state["h"], :, state["i"]]
                y = y.clone()
                y[..., state["h"], :] += d.to(y.dtype)
                return y.reshape(out.shape)

            handle = proj.register_forward_hook(h)
            for hh in range(H):
                for i in range(Dh):
                    state.update(h=hh, i=i)
                    lp = hidden_logprobs(model, forward_hidden(model, ids))
                    KL[nm][t, hh, i] = kl_mean(base, lp)
                    del lp
            handle.remove()
        el = time.time() - t00
        done = (t + 1) / cfg["NB"]
        print(f"  KV sens: block {t + 1}/{cfg['NB']}  "
              f"ETA {el / done * (1 - done) / 60:.1f} min", flush=True)
    torch.save(KL, save)
    return KL


# ---------------- quantization hooks ----------------

def make_act_hook(V, mu, s, bits, sink=0):
    """Eigenbasis per-direction RTN on a LN input (residual stream).
    The first `sink` token positions are left fp16: attention sinks
    carry the extreme activations, and at 3 bits a single sink token
    both gets destroyed and (via absmax calibration) inflates the grid
    for every normal token."""
    n = 2.0 ** bits.float()

    def h(module, args):
        x = args[0]
        z = (x.float() - mu) @ V
        zn = z / s
        q = torch.clamp(torch.round((zn + 1) * 0.5 * (n - 1)),
                        torch.zeros_like(n), n - 1)
        zq = (q / (n - 1) * 2 - 1) * s
        yq = (zq @ V.T + mu).to(x.dtype)
        if sink > 0:
            yq[:, :sink] = x[:, :sink]
        return (yq,) + args[1:]
    return h


def make_kv_hook(V, mu, s, bits, sink):
    """Per-head eigenbasis RTN on k/v_proj output (pre-RoPE)."""
    H = V.shape[0]
    n = 2.0 ** bits.float()

    def h(module, args, out):
        shp = out.shape
        y = out.reshape(*shp[:-1], H, -1).float()
        z = torch.einsum("bthd,hde->bthe", y - mu, V)
        zn = z / s
        q = torch.clamp(torch.round((zn + 1) * 0.5 * (n - 1)),
                        torch.zeros_like(n), n - 1)
        zq = (q / (n - 1) * 2 - 1) * s
        # fold back: yq[d] = sum_e zq[e] * V[d,e]  (= zq @ V^T)
        # output index pairs with V's ROW index ("hde") — using "hed"
        # here is the double-rotation bug (zq @ V), KV-uniform-16 ~6000 ppl
        yq = torch.einsum("bthe,hde->bthd", zq, V) + mu
        if sink > 0:
            yq[:, :sink] = y[:, :sink]
        return yq.to(out.dtype).reshape(shp)
    return h


@torch.no_grad()
def calib_ranges(model, cfg, P, eig_kv, nwin, seq, pb, seed, cache=None,
                 a_sink=0, kv_sink=0):
    a_sink_global = [kv_sink]
    """Per-direction absmax for res/post (LN inputs) and k/v (per head),
    measured under the CURRENT (already weight-quantized) model."""
    if cache is None:
        cache = torch.load(P["spaces"], map_location="cpu")
    H, Dh = cfg["H_KV"], cfg["HEAD_DIM"]
    wins, _ = load_windows(cfg, nwin, seq, seed, segment="calib")
    NB = cfg["NB"]
    D = cfg["D"]
    mx = {"res": torch.zeros(NB, D), "post": torch.zeros(NB, D),
          "k": torch.zeros(NB, H, Dh), "v": torch.zeros(NB, H, Dh)}
    Vr = cache["res"]["evecs"].cuda()
    mur = cache["res"]["mu"].cuda()
    Vp = cache["post"]["evecs"].cuda()
    mup = cache["post"]["mu"].cuda()
    Vk = eig_kv["k"]["evecs"].cuda()
    muk = eig_kv["k"]["mu"].cuda()
    Vv = eig_kv["v"]["evecs"].cuda()
    muv = eig_kv["v"]["mu"].cuda()
    handles = []
    for t in range(NB):
        L = blocks(model)[t]

        def mk_ln(t, sp, V, mu):
            def h(module, args):
                x = args[0].detach()
                if a_sink > 0:      # sink positions keep fp16 in the
                    x = x[:, a_sink:]  # hook; don't let them set ranges
                z = (x.float().reshape(-1, D) - mu[t]) @ V[t]
                mx[sp][t] = torch.maximum(mx[sp][t].cuda(),
                                          z.abs().amax(0)).cpu()
            return h

        def mk_kv(t, nm, proj, V, mu):
            def h(module, args, out):
                o = out.detach()
                if a_sink_global[0] > 0:   # KV hook keeps sink fp16;
                    o = o[:, a_sink_global[0]:]  # don't let it set ranges
                y = o.reshape(-1, H, Dh).float()
                z = torch.einsum("nhd,hde->nhe", y - mu[t], V[t])
                mx[nm][t] = torch.maximum(mx[nm][t].cuda(),
                                          z.abs().amax(0)).cpu()
            return h

        handles += [
            L.input_layernorm.register_forward_pre_hook(
                mk_ln(t, "res", Vr, mur)),
            L.post_attention_layernorm.register_forward_pre_hook(
                mk_ln(t, "post", Vp, mup)),
            L.self_attn.k_proj.register_forward_hook(
                mk_kv(t, "k", None, Vk, muk)),
            L.self_attn.v_proj.register_forward_hook(
                mk_kv(t, "v", None, Vv, muv)),
        ]
    for s in range(0, nwin, pb):
        ids = torch.tensor(wins[s:s + pb], dtype=torch.long, device="cuda")
        model.model(input_ids=ids)
    for h_ in handles:
        h_.remove()
    scales = {sp: mx[sp].clamp_min(1e-8) for sp in mx}
    return scales, cache


def ppl_wikitext(model, cfg, seqlen=2048, pb=1):
    """WikiText-2 test PPL, CoQuant protocol (they report 13.07 fp16)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    nwin = ids.numel() // seqlen
    ids = ids[:nwin * seqlen].reshape(nwin, seqlen)
    nll, ntok = 0.0, 0
    for s in range(0, nwin, pb):
        b = ids[s:s + pb].cuda()
        lp = hidden_logprobs(model, forward_hidden(model, b))
        tgt = b[:, 1:]
        nll += -torch.gather(lp[:, :-1].float(), 2,
                             tgt.unsqueeze(-1)).sum().item()
        ntok += tgt.numel()
        if s % (20 * pb) == 0:
            print(f"  wikitext: window {s}/{nwin}", flush=True)
    import math
    return math.exp(nll / ntok)


# ---------------- main ----------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--price", default="kl",
                    choices=["kl", "uniform", "variance"])
    ap.add_argument("--w_avg", type=float, default=4.25)  # -> true ~4.5, this number does not take grouping into account.
    ap.add_argument("--a_avg", type=float, default=4.5)
    ap.add_argument("--kv_avg", type=float, default=4.5)
    ap.add_argument("--k_avg", type=float, default=None,
                    help="K-only budget (overrides --kv_avg for K)")
    ap.add_argument("--v_avg", type=float, default=None,
                    help="V-only budget (overrides --kv_avg for V)")
    ap.add_argument("--bmin", type=int, default=3)        # W floor
    ap.add_argument("--a_bmin", type=int, default=2)      # A/KV floor
    ap.add_argument("--sink", type=int, default=4)          # KV sink
    ap.add_argument("--a_sink", type=int, default=0,
                    help="fp16 sink positions for A hooks + excluded "
                         "from A range calibration; try 4 at 3 bits")
    ap.add_argument("--vq", action="store_true",
                    help="additive VQ (2x256x8, 2.0 bits/scalar) for W "
                         "directions allocated 1.5-2.5 bits")
    ap.add_argument("--no_a", action="store_true")
    ap.add_argument("--fresh_a", action="store_true",
                    help="re-capture res/post bases under quantized weights "
                         "(phase 2.5) instead of using fp16 stage1 spaces")
    ap.add_argument("--fresh_w", action="store_true",
                    help="re-measure each block's weight-space KL prices "
                         "on the partially quantized model during phase "
                         "1 (default: fp16 stage2 caches). Expensive: "
                         "~4 spaces x ~2k dirs x NB extra forwards; "
                         "regime-dependent, see fresh_sens findings")
    ap.add_argument("--transfer_sens", action="store_true",
                    help="transfer fp16 KL prices onto the fresh bases "
                         "by basis matching (phase 2.6, ~free, no new "
                         "forwards); implies --fresh_a. Cheaper "
                         "alternative to --fresh_sens")
    ap.add_argument("--fresh_sens", action="store_true",
                    help="also re-measure res/post KL prices under the "
                         "quantized model along the fresh bases (phase "
                         "2.75); implies --fresh_a. Needed at low weight "
                         "budgets where the basis rotates")
    ap.add_argument("--sens_topk", type=int, default=1536,
                    help="fresh-sens top directions per block (D > 2048)")
    ap.add_argument("--sens_tail", type=int, default=512,
                    help="fresh-sens random tail dirs per block (D > 2048)")
    ap.add_argument("--sens_nwin", type=int, default=2,
                    help="fresh-sens eval windows")
    ap.add_argument("--sens_seq", type=int, default=256,
                    help="fresh-sens window length")
    ap.add_argument("--scale_floor", type=float, default=0.0,
                    help="floor A ranges at this many sigmas of the "
                         "direction's eigenvalue (0 = off; 4 recommended "
                         "at low bits)")
    ap.add_argument("--no_kv", action="store_true")
    ap.add_argument("--no_w", action="store_true")
    ap.add_argument("--wikitext", action="store_true")
    ap.add_argument("--calib_nwin", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--pb", type=int, default=8)
    ap.add_argument("--eval_nwin", type=int, default=16)
    a = ap.parse_args()

    cfg = MODELS[a.model]
    P = paths(a.model)
    model = get_teacher(cfg)
    H, Dh = cfg["H_KV"], cfg["HEAD_DIM"]

    print("== phase 0: fp16 baselines ==")
    base = ppl_eval(model, cfg, nwin=a.eval_nwin, seq=a.seq, seed=0)
    print(f"fp16 fineweb-val ppl: {base:.2f}")
    base_wt = None
    if a.wikitext:
        base_wt = ppl_wikitext(model, cfg)
        print(f"fp16 wikitext-2 ppl: {base_wt:.2f}  (CoQuant row: 13.07)")

    wres = None
    if not a.no_w:
        print("== phase 1: weights (sequential re-capture KLQ) ==")
        wres = quantize_weights(model, cfg, P, price=a.price, avg=a.w_avg,
                                bmin=a.bmin, nwin=a.calib_nwin, seq=a.seq,
                                pb=a.pb, fresh_kl=a.fresh_w,
                                sens_nwin=a.sens_nwin,
                                sens_seq=a.sens_seq,
                                vq=dict(d=8, nbooks=2, levels=256,
                                        thresh=2.5) if a.vq else None)
        print(f"W true bpw: {wres['true_bpw']:.3f}")

    sens_res = torch.load(os.path.join(
        P["sens_dir"], f"sens_res_{P['tag']}.pt"), map_location="cpu")
    sens_post = torch.load(os.path.join(
        P["sens_dir"], f"sens_post_{P['tag']}.pt"), map_location="cpu")

    # W-config tag: any sens/prices measured on the damaged model are
    # only valid for THIS weight configuration (the A-side lesson).
    wtag = (f"fresh_w{a.w_avg}_{a.price}_bmin{a.bmin}"
            f"{'vq' if a.vq else 'rtn'}"
            f"{'_fw' if a.fresh_w else ''}" if not a.no_w else "now")

    eig_kv, kvsens = None, None
    if not a.no_kv:
        print("== phase 2: KV eigenbases + measured KL ==")
        eig_kv = kv_eigen(model, cfg, a.calib_nwin, a.seq, a.pb, 12345)
        # default: one shared KV sens cache per model (pre-rekey
        # behavior). Per-W-config KV prices only when a fresh flag is
        # active -- the regime where damaged-model prices matter.
        fresh_mode = a.fresh_sens or a.fresh_w or a.transfer_sens
        kv_sens_name = (f"sens_kv_{wtag}_{P['tag']}.pt" if fresh_mode
                        else f"sens_kv_{P['tag']}.pt")
        kvsens = kv_sens(model, cfg, eig_kv, nwin=2, seq=a.seq, pb=2,
                         seed=0,
                         save=os.path.join(P["sens_dir"], kv_sens_name))
    else:  # KV hooks still need eigenbases for scale calibration
        eig_kv = kv_eigen(model, cfg, a.calib_nwin, a.seq, a.pb, 12345)

    a_cache = None
    if (a.fresh_a or a.fresh_sens or a.transfer_sens) and not a.no_a:
        print("== phase 2.5: fresh res/post bases under quantized W ==")
        a_cache = fresh_a_bases(model, cfg, P, a.calib_nwin, a.seq,
                                a.pb, 12345)

    if a.transfer_sens and not a.no_a:
        print("== phase 2.6: price transfer onto fresh bases ==")
        sens_res = transfer_sens(P, a_cache, sens_res, "res")
        sens_post = transfer_sens(P, a_cache, sens_post, "post")

    if a.fresh_sens and not a.no_a:
        print("== phase 2.75: fresh res/post KL prices under quantized "
              "W, along fresh bases ==")
        for sp, attr in (("res", "input_layernorm"),
                         ("post", "post_attention_layernorm")):
            save = os.path.join(
                P["sens_dir"],
                f"sens_{sp}_{wtag}_n{a.sens_nwin}s{a.sens_seq}"
                f"_{P['tag']}.pt")
            fs = fresh_a_sens(model, cfg, P, a_cache, sp, attr,
                              a.sens_topk, a.sens_tail,
                              nwin=a.sens_nwin, seq=a.sens_seq, pb=2,
                              seed=0, save=save)
            if sp == "res":
                sens_res = fs
            else:
                sens_post = fs

    print("== phase 3: attach A + KV quant hooks ==")
    scales, cache = calib_ranges(model, cfg, P, eig_kv, a.calib_nwin,
                                 a.seq, a.pb, 12345, cache=a_cache,
                                 a_sink=a.a_sink, kv_sink=a.sink)
    if a.scale_floor > 0 and not a.no_a:
        # near-null tail directions get garbage absmax scales (fp32 eigh
        # noise, step8: 50%+ clip rates on d0-d74 late blocks); floor at
        # scale_floor * sigma of the direction's own eigenvalue. Spike
        # directions (absmax >> 4 sigma) are untouched.
        for sp in ("res", "post"):
            lam = cache[sp]["evals"].clamp_min(0)
            floor = a.scale_floor * lam.sqrt()
            n_floored = int((scales[sp] < floor).sum())
            scales[sp] = torch.maximum(scales[sp], floor)
            print(f"  scale floor {a.scale_floor:.1f}*sigma: {sp} "
                  f"{n_floored}/{scales[sp].numel()} ranges raised")

    def price_of(kl, ev):
        if a.price == "kl":
            return kl.float().clamp_min(1e-12)
        if a.price == "variance":
            return ev.float().clamp_min(1e-12)
        return torch.ones_like(kl, dtype=torch.float)

    handles = []
    abits_report, kvbits_report = {}, {}
    if not a.no_a:
        for sp, sens, Vc in (("res", sens_res, "res"),
                             ("post", sens_post, "post")):
            pr = price_of(sens["kl"], cache[sp]["evals"])
            bits = waterfill(pr.flatten(), a.a_avg, a.a_bmin, 12
                             ).reshape(pr.shape)
            abits_report[sp] = float(bits.mean())
            for t in range(cfg["NB"]):
                L = blocks(model)[t]
                mod = (L.input_layernorm if sp == "res"
                       else L.post_attention_layernorm)
                handles.append(mod.register_forward_pre_hook(
                    make_act_hook(cache[sp]["evecs"][t].cuda(),
                                  cache[sp]["mu"][t].cuda(),
                                  scales[sp][t].cuda(), bits[t].cuda(),
                                  a.a_sink)))
    if not a.no_kv:
        for nm, proj_attr in (("k", "k_proj"), ("v", "v_proj")):
            avg_nm = (a.k_avg if nm == "k" else a.v_avg)
            avg_nm = a.kv_avg if avg_nm is None else avg_nm
            pr = price_of(kvsens[nm], eig_kv[nm]["evals"])
            bits = waterfill(pr.flatten(), avg_nm, a.a_bmin, 12
                             ).reshape(pr.shape)
            kvbits_report[nm] = float(bits.mean())
            for t in range(cfg["NB"]):
                proj = getattr(blocks(model)[t].self_attn, proj_attr)
                handles.append(proj.register_forward_hook(
                    make_kv_hook(eig_kv[nm]["evecs"][t].cuda(),
                                 eig_kv[nm]["mu"][t].cuda(),
                                 scales[nm][t].cuda(), bits[t].cuda(),
                                 a.sink)))

    print("== phase 4: eval ==")
    ppl = ppl_eval(model, cfg, nwin=a.eval_nwin, seq=a.seq, seed=0)
    line = (f"W{'-' if wres is None else format(wres['true_bpw'], '.2f')}"
            f"/A{'-' if not abits_report else format(sum(abits_report.values()) / len(abits_report), '.2f')}"
            f"/KV{'-' if not kvbits_report else format(sum(kvbits_report.values()) / len(kvbits_report), '.2f')}"
            f"  {a.price}  ppl {ppl:.2f}  "
            f"(fp16 {base:.2f}, x{ppl / base:.3f})")
    print("\n" + line)
    res = dict(model=a.model, price=a.price, w_avg=a.w_avg, a_avg=a.a_avg,
               kv_avg=a.kv_avg, bmin=a.bmin, a_bmin=a.a_bmin, sink=a.sink,
               true_w_bpw=None if wres is None else wres["true_bpw"],
               a_bits=abits_report, kv_bits=kvbits_report,
               base_ppl=base, ppl=ppl, ratio=ppl / base)
    if a.wikitext:
        ppl_wt = ppl_wikitext(model, cfg)
        res.update(base_wikitext=base_wt, wikitext=ppl_wt,
                   wikitext_ratio=ppl_wt / base_wt)
        print(f"wikitext-2: {ppl_wt:.2f} (fp16 {base_wt:.2f}, "
              f"x{ppl_wt / base_wt:.3f})   [CoQuant 4.5/4.5/4.5 = 17.76]")
    f = os.path.join(P["sens_dir"],
                     f"step5_{a.price}_w{a.w_avg}_a{a.a_avg}_kv{a.kv_avg}"
                     f"_{P['tag']}.json")
    with open(f, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"saved -> {f}")


if __name__ == "__main__":
    main()
