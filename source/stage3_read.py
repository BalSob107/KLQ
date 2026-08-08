# Not vital for quantization, useful for model analysis.
# Usage:
#   python stage3_read.py --model qwen --spaces res
#   python stage3_read.py --model qwen --spaces res,qkv,int

import argparse
import os
import numpy as np
import torch

from config import MODELS, paths


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() /
                 np.sqrt((ra ** 2).sum() * (rb ** 2).sum() + 1e-30))


def eff_rank(ev):
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum() + 1e-30)


def report_space(P, space, tag, topk_tbl=15):
    f = os.path.join(P["sens_dir"], f"sens_{space}_{tag}.pt")
    if not os.path.exists(f):
        print(f"\n=== {space}: no sens file ({f}), skipping ===")
        return
    sens = torch.load(f, map_location="cpu")
    cache = torch.load(P["spaces"], map_location="cpu")
    KL = sens["kl"].numpy()                     # (NB, d), NaN = unmeasured
    ev = cache[space]["evals"].numpy()          # (NB, d), ascending
    NB, d = KL.shape
    measured = sens["measured"].numpy()

    print(f"\n{'=' * 70}\n=== {space}  (dim {d}, {NB} blocks)                                     ===\n{'=' * 70}")

    # ---- 1. eigenspectrum -------------------------------------------------
    print("\n[eigenspectrum]  per-layer: log10(lmax/lmed) | eff_rank/d | "
          "var% in top 1% | var% in top 16")
    for t in range(NB):
        e = ev[t]
        epos = e[e > 0]
        lmed = np.median(epos)
        cum = e[::-1].cumsum()[::-1] / e.sum()   # cumvar from the top down
        k1 = max(1, d // 100)
        v1 = 1 - cum[k1 - 1] / 1 if False else e[::-1][:k1].sum() / e.sum()
        v16 = e[::-1][:16].sum() / e.sum()
        print(f"  L{t:02d}  {np.log10(e[-1] / lmed):6.2f}   "
              f"{eff_rank(e) / d:7.3f}   {100 * v1:6.1f}%   {100 * v16:6.1f}%")

    # ---- 2. KL profile ----------------------------------------------------
    print("\n[KL]  per-layer: max | median | log10(max/med) | "
          "top-16 share of total KL")
    for t in range(NB):
        k = KL[t][~np.isnan(KL[t])]
        kpos = k[k > 0]
        ks = np.sort(k)[::-1]
        share = ks[:16].sum() / ks.sum()
        print(f"  L{t:02d}  {k.max():9.5f}  {np.median(kpos):9.6f}  "
              f"{np.log10(k.max() / np.median(kpos)):6.2f}   "
              f"{100 * share:6.1f}%")

    g = KL.copy()
    g[np.isnan(g)] = -1
    flat = np.argsort(g, axis=None)[::-1][:topk_tbl]
    print(f"\n[KL] global top-{topk_tbl} (layer, dir, KL, eigenvalue):")
    for ix in flat:
        t, i = divmod(ix, d)
        print(f"  L{t:02d} dir {i:5d}  KL {KL[t, i]:9.5f}  ev {ev[t, i]:.4g}")

    # ---- 3. spearman: would variance pricing work? ------------------------
    print("\n[spearman(KL, eigenvalue)] per layer (measured dirs only):")
    rhos = []
    for t in range(NB):
        m = measured[t] & (KL[t] > 0)
        r = spearman(KL[t][m], ev[t][m]) if m.sum() > 8 else float("nan")
        rhos.append(r)
        print(f"  L{t:02d}  {r:+.3f}")
    print(f"  mean {np.nanmean(rhos):+.3f}   ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--spaces", default="res")
    a = ap.parse_args()

    cfg = MODELS[a.model]
    P = paths(a.model)
    spaces = [s.strip() for s in a.spaces.split(",")]
    for s in spaces:
        report_space(P, s, P["tag"])


if __name__ == "__main__":
    main()
