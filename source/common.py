import numpy as np
import torch
from transformers import AutoModelForCausalLM
import os

def get_teacher(cfg, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf"], torch_dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def blocks(model):
    return model.model.layers


VAL_TOKENS = 100_000   # held-out tail of val.bin. sens/eval draw only from
                       # here, calibration never sees it


def get_tokens(cfg, source=None):
    """Token stream for calibration/val. source=None -> cfg['VAL_BIN'];
    source='wikitext' -> wikitext-2-raw-v1 TRAIN split, tokenized once with
    the model's own tokenizer and cached to ROOT/data/wikitext_{tag}.bin"""
    if source != "wikitext":
        dt = np.uint32 if cfg["VAL_DTYPE"] == "uint32" else np.uint16
        return np.memmap(cfg["VAL_BIN"], dtype=dt, mode="r")
    dt = np.uint32 if cfg["VAL_DTYPE"] == "uint32" else np.uint16
    path = os.path.join(os.path.dirname(cfg["VAL_BIN"]),
                        f"wikitext_{cfg['tag']}.bin")
    if not os.path.exists(path):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        print(f"[wikitext] building {path} (one-time)...")
        tok = AutoTokenizer.from_pretrained(cfg["hf"])
        eos = tok.eos_token_id
        assert eos is not None and eos < np.iinfo(dt).max
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        buf = []
        for s in range(0, len(ds), 512):
            texts = [t for t in ds[s:s + 512]["text"] if t.strip()]
            ids = tok(texts, add_special_tokens=False)["input_ids"]
            for seq_ids in ids:
                buf.extend(seq_ids)
                buf.append(eos)
        arr = np.asarray(buf, dtype=dt)
        assert int(arr.max()) == max(buf) < np.iinfo(dt).max
        arr.tofile(path)
        print(f"[wikitext] wrote {len(arr)} tokens -> {path}")
    return np.memmap(path, dtype=dt, mode="r")


def load_windows(cfg, nwin, seq, seed, segment="calib", source=None):
    """Windows from val.bin with a hard calib/val split:
      segment="calib" -> starts in [0, len - VAL_TOKENS - seq)   (stage1)
      segment="val"   -> starts in the last VAL_TOKENS tokens    (stage2, ppl)."""
    val = get_tokens(cfg, source)
    n_starts = len(val) - seq - 1
    if segment == "calib":
        off, pool = 0, max(1, n_starts - VAL_TOKENS)
    elif segment == "val":
        off = max(0, n_starts - VAL_TOKENS)
        pool = n_starts - off
    else:
        off, pool = 0, n_starts
    assert pool >= nwin, (f"val.bin too small: {pool} possible starts in "
                          f"'{segment}' segment, need {nwin}")
    rng = np.random.default_rng(seed)
    starts = np.sort(off + rng.choice(pool, size=nwin, replace=False))
    out = np.stack([val[s:s + seq] for s in starts]).astype(np.int64)
    return out, starts

@torch.no_grad()
def forward_hidden(model, ids):
    return model.model(input_ids=ids).last_hidden_state


@torch.no_grad()
def hidden_logprobs(model, hidden, chunk=256):
    """(B,T,D) hidden -> (B,T,V) fp16 log-probs, computed in row chunks."""
    W = model.get_output_embeddings().weight          # tied for both models
    B, T, D = hidden.shape
    V = W.shape[0]
    out = torch.empty(B, T, V, dtype=torch.float16, device=hidden.device)
    h = hidden.reshape(B * T, D)
    o = out.reshape(B * T, V)
    for s in range(0, B * T, chunk):
        z = h[s:s + chunk] @ W.T                       # fp16 matmul
        o[s:s + chunk] = torch.log_softmax(z.float(), dim=-1).half()
    return out


def kl_mean(base_lp, pert_lp, chunk=4096):
    """Mean over tokens of KL(base || pert), computed chunkwise in fp32."""
    B, T, V = base_lp.shape
    a = base_lp.reshape(-1, V)
    b = pert_lp.reshape(-1, V)
    tot = 0.0
    for s in range(0, a.shape[0], chunk):
        af = a[s:s + chunk].float()
        bf = b[s:s + chunk].float()
        tot += (af.exp() * (af - bf)).sum().item()
    return tot / a.shape[0]


@torch.no_grad()
def ppl_eval(model, cfg, nwin=8, seq=512, seed=0, pb=2, source=None):
    """Teacher-forced PPL via chunked lm_head (no giant logits tensor)."""
    wins, _ = load_windows(cfg, nwin, seq, seed, segment="val", source=source)
    nll, ntok = 0.0, 0
    for s in range(0, nwin, pb):
        ids = torch.tensor(wins[s:s + pb], dtype=torch.long, device="cuda")
        lp = hidden_logprobs(model, forward_hidden(model, ids))
        tgt = ids[:, 1:]
        nll += -torch.gather(lp[:, :-1].float(), 2,
                             tgt.unsqueeze(-1)).sum().item()
        ntok += tgt.numel()
    return float(np.exp(nll / ntok))


def waterfill(price, avg, bmin=1, bmax=12):
    """b_i = clip(0.5*log2(price_i) - theta, bmin, bmax); bisect theta so
    mean(b) == avg. price: (n,) positive tensor."""
    price = price.float().clamp_min(1e-12)
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        b = (0.5 * torch.log2(price) - mid).clamp(bmin, bmax)
        if b.mean() > avg:
            lo = mid
        else:
            hi = mid
    return (0.5 * torch.log2(price) - 0.5 * (lo + hi)).clamp(bmin, bmax)


@torch.no_grad()
def _kmeans(X, K, iters=15, seed=0, chunk=1 << 16):
    """Lloyd's algorithm on (N, d) fp32 cuda -> (K, d) centroids, (N,) assign."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:K]].clone()
    a = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    for _ in range(iters):
        for st in range(0, X.shape[0], chunk):
            a[st:st + chunk] = torch.cdist(X[st:st + chunk], C).pow_(2).argmin(1)
        Cnew = torch.zeros_like(C)
        cnt = torch.zeros(K, device=X.device)
        Cnew.index_add_(0, a, X)
        cnt.index_add_(0, a, torch.ones(X.shape[0], device=X.device))
        C = torch.where(cnt.unsqueeze(1) > 0,
                        Cnew / cnt.unsqueeze(1).clamp_min(1), C)
    return C, a


@torch.no_grad()
def fit_additive_vq(Z, nbooks=2, levels=256, iters=15, refine=2):
    """Greedy additive VQ: Z = sum_b C_b[idx_b] + residual. Block-coordinate
    refinement after the greedy pass. Z: (N, d) fp32 cuda."""
    books = []
    R = Z.clone()
    for b in range(nbooks):
        C, a = _kmeans(R, levels, iters, seed=b)
        books.append(C)
        R -= C[a]
    for _ in range(refine):
        for b in range(nbooks):
            # residual for book b = Z - (greedy contribution of other books)
            R = Z.clone()
            for j, C in enumerate(books):
                if j == b:
                    continue
                a = torch.empty(Z.shape[0], dtype=torch.long, device=Z.device)
                for st in range(0, Z.shape[0], 1 << 16):
                    a[st:st + 1 << 16] = torch.cdist(
                        R[st:st + 1 << 16], C).pow_(2).argmin(1)
                R -= C[a]
            C, a = _kmeans(R, levels, iters, seed=100 + b)
            books[b] = C
    return books


@torch.no_grad()
def encode_additive(Z, books, chunk=1 << 16):
    """Greedy encode: returns list of (N,) index tensors, sum of selected
    codevectors approximates Z."""
    idxs, R = [], Z.clone()
    for C in books:
        a = torch.empty(Z.shape[0], dtype=torch.long, device=Z.device)
        for st in range(0, Z.shape[0], chunk):
            a[st:st + chunk] = torch.cdist(R[st:st + chunk], C).pow_(2).argmin(1)
        idxs.append(a)
        R -= C[a]
    return idxs


@torch.no_grad()
def rtn_eigenbasis(W, Vg, bits, group=None, vq=None):
    """Rotate to eigenbasis, per-(group, direction) absmax-scaled RTN on a
    full-range grid, fold back. bits: (in_d,) per-direction allocation.
    vq (optional dict(d=8, nbooks=2, levels=256, thresh=2.5)): directions
    with 1.5 <= bits < thresh are quantized with additive vector
    quantization at exactly nbooks*log2(levels)/d bits/scalar (one codebook
    set fitted per matrix on all low directions' normalized blocks).
    Returns Wq, or (Wq, bits_eff) when vq is active."""
    n = 2.0 ** bits.float()
    Wp = W.float() @ Vg
    out_d, in_d = Wp.shape
    if group is None:
        group = 128 if out_d % 128 == 0 else 64
    G = out_d // group
    Wg = Wp.reshape(G, group, in_d)
    s = Wg.abs().amax(1, keepdim=True).clamp_min(1e-8)
    zn = Wg / s
    q = torch.clamp(torch.round((zn + 1) * 0.5 * (n - 1)),
                    torch.zeros_like(n), n - 1)
    Wq = (q / (n - 1) * 2 - 1) * s

    # 1-bit fix: levels at +-mean|x| (Lloyd-optimal for symmetric data)
    one = (bits < 1.5).float().reshape(1, 1, in_d)
    if one.any():
        lvl = Wg.abs().mean(1, keepdim=True).clamp_min(1e-8)
        Wq = Wq * (1 - one) + torch.sign(zn) * lvl * one

    if vq is not None:
        d = vq.get("d", 8)
        nb = vq.get("nbooks", 2)
        lv = vq.get("levels", 256)
        thr = vq.get("thresh", 2.5)
        assert group % d == 0, "VQ block size must divide the group size"
        low = (bits >= 1.5) & (bits < thr)
        L = int(low.sum())
        nb_blocks = L * G * (group // d)
        if L > 0 and nb_blocks >= 4 * lv:
            Z = zn[:, :, low].permute(2, 0, 1).reshape(L * G, group)
            Z = Z.reshape(-1, d)                       # (L*G*g/d, d)
            books = fit_additive_vq(Z, nb, lv,
                                    vq.get("iters", 15),
                                    vq.get("refine", 2))
            idxs = encode_additive(Z, books)
            Zq = torch.zeros_like(Z)
            for C, a in zip(books, idxs):
                Zq += C[a]
            Zq = Zq.reshape(L, G, group).permute(1, 2, 0)  # (G, g, L)
            Wq[:, :, low] = Zq * s[:, :, low]
            rate = nb * (lv.bit_length() - 1) / d
            bits_eff = torch.where(low, torch.full_like(bits, rate), bits)
        else:
            bits_eff = bits
        Wq = Wq.reshape(out_d, in_d) @ Vg.T
        return Wq, bits_eff

    Wq = Wq.reshape(out_d, in_d) @ Vg.T
    return Wq


MATS = {
    "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv",
    "o_proj": "ctx",
    "gate_proj": "mlp", "up_proj": "mlp",
    "down_proj": "int",
}


def get_mat(L, name):
    return (getattr(L.self_attn, name) if hasattr(L.self_attn, name)
            else getattr(L.mlp, name))


def tail_fill(kl, d, D, meta=None):
    if d <= D:
        return kl.nan_to_num(0.0)
    topk = (meta or {}).get("topk", 512)
    tail = (meta or {}).get("tail", 256)
    tail_ix = torch.randperm(d - topk,
                             generator=torch.Generator().manual_seed(7)
                             )[:tail]
    tm = kl[tail_ix]
    fill = tm[~tm.isnan()].median()
    return torch.where(kl.isnan(), fill, kl)


@torch.no_grad()
def _fresh_space_kl(model, mod, ev, V, base_lp, ids, d, cfg,
                    topk=2048, tail=512):
    """Measured KL per direction for one matrix input space, on the
    CURRENT model state, along the basis (V, ev) just captured for it.
    Probe: delta_i = sqrt(ev_i) v_i added to the module input; KL
    against base_lp. Plan mirrors step2 (full if d<=2048 else top-k +
    seeded random tail), unmeasured tail -> measured-tail median."""
    if d <= 2048:
        dirs = torch.arange(d)
    else:
        top = torch.arange(d - topk, d)
        tl = torch.randperm(d - topk,
                            generator=torch.Generator().manual_seed(7)
                            )[:tail]
        dirs = torch.cat([top, tl]).sort().values
    KL = torch.full((d,), float("nan"), device="cuda")
    state = {"i": None}

    def h(module, args, state=state, V=V, ev=ev):
        i = state["i"]
        if i is None:
            return None
        delta = ev[i].sqrt() * V[:, i]
        return (args[0] + delta.to(args[0].dtype),) + args[1:]

    handle = mod.register_forward_pre_hook(h)
    for i in dirs.tolist():
        state["i"] = i
        lp = hidden_logprobs(model, forward_hidden(model, ids))
        KL[i] = kl_mean(base_lp, lp)
        del lp
    handle.remove()
    if d > 2048:
        tl_ix = torch.randperm(d - topk,
                               generator=torch.Generator().manual_seed(7)
                               )[:tail]
        tm = KL[tl_ix.cuda()]
        fill = tm[~tm.isnan()].median()
        KL = torch.where(KL.isnan(), fill.cuda(), KL)
    return KL.nan_to_num(0.0)


@torch.no_grad()
def quantize_weights(model, cfg, P, price="kl", avg=4.5, group=64,
                     bmin=3, bmax=12, nwin=8, seq=512, pb=8, seed=12345,
                     vq=None, verbose=True, fresh_kl=False,
                     sens_nwin=2, sens_seq=256):
    """Sequential re-capture + per-matrix water-filled eigenbasis RTN.
    Returns dict(tot_b, tot_p, rows, true_bpw). Quantizes in place.
    fresh_kl=True re-measures each block's KL prices on the partially
    quantized model (the A-side phase-2.75 lesson, applied to W);
    default False keeps the fp16 step2 caches (regime-dependent: needed
    only at very low budgets, expensive: ~4 spaces x ~2k dirs x NB
    extra forwards)."""
    import os
    import time
    sens = {sp: torch.load(os.path.join(P["sens_dir"],
                                        f"sens_{sp}_{P['tag']}.pt"),
                           map_location="cpu")
            for sp in ["qkv", "ctx", "mlp", "int"]}
    wins, _ = load_windows(cfg, nwin, seq, seed, segment="calib")
    mods_of = {
        "qkv": lambda L: L.self_attn.q_proj,
        "ctx": lambda L: L.self_attn.o_proj,
        "mlp": lambda L: L.mlp.gate_proj,
        "int": lambda L: L.mlp.down_proj,
    }
    tot_b, tot_p, rows = 0.0, 0, []
    for t in range(cfg["NB"]):
        t0 = time.time()
        L = blocks(model)[t]
        S, v, n = {}, {}, {}
        handles = []

        def mk(name):
            def h(module, args):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).double()
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

        for name, m in mods_of.items():
            handles.append(m(L).register_forward_pre_hook(mk(name)))
        for s in range(0, nwin, pb):
            ids = torch.tensor(wins[s:s + pb], dtype=torch.long,
                               device="cuda")
            model.model(input_ids=ids)
        for h_ in handles:
            h_.remove()

        if fresh_kl:
            # base log-probs of the current (partially quantized) model
            # probes below are measured against this, per space, along
            # that space's freshly captured eigenbasis.
            vwins, _ = load_windows(cfg, sens_nwin, sens_seq, 0,
                                    segment="val")
            vids = torch.tensor(vwins, dtype=torch.long, device="cuda")
            base_lp = hidden_logprobs(model, forward_hidden(model, vids))

        for mname, space in MATS.items():
            mod = get_mat(L, mname)
            W = mod.weight.data
            mu = v[space] / n[space]
            C = S[space] / n[space] - torch.outer(mu, mu)
            C = 0.5 * (C + C.T)
            ev, V = torch.linalg.eigh(C.float())
            ev = ev.clamp_min(0)
            d = V.shape[0]
            if fresh_kl:
                kl = _fresh_space_kl(model, mod, ev, V, base_lp, vids,
                                     d, cfg)
            else:
                kl = tail_fill(sens[space]["kl"][t].cuda(), d, cfg["D"],
                               sens[space].get("meta"))
            read2 = (W.float() @ V).pow(2).sum(0)

            # price functions
            if price == "kl":
                pr = kl * read2
            elif price == "variance":
                pr = ev * read2
            else:
                pr = torch.ones_like(read2)
            bits = waterfill(pr, avg, bmin, bmax)
            if vq is not None:
                Wq, bits_eff = rtn_eigenbasis(W, V, bits, group=group, vq=vq)
            else:
                Wq = rtn_eigenbasis(W, V, bits, group=group)
                bits_eff = bits
            mod.weight.data.copy_(Wq.to(W.dtype))
            out_d, in_d = W.shape
            tot_b += float(bits_eff.sum()) * out_d
            tot_p += out_d * in_d
            rows.append((t, mname, float(bits.mean()), int(bits.min()),
                         int(bits.max())))
            del Wq, V, C
        for space in list(S):
            S.pop(space, None)
            v.pop(space, None)
        torch.cuda.empty_cache()
        if verbose:
            print(f"  W: block {t + 1}/{cfg['NB']}  "
                  f"{time.time() - t0:.1f}s", flush=True)
    return dict(tot_b=tot_b, tot_p=tot_p, rows=rows,
                true_bpw=tot_b / tot_p + 16.0 / group)
