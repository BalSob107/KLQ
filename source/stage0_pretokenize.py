# Presets (reproduce the pipeline's own numbers):
#   python stage0_pretokenization.py --model llama3_1b --preset wikitext2
#
# Generic sources:
#   python stage0_pretokenization.py --model qwen --parquet dump.parquet
#   python stage0_pretokenization.py --model qwen --txt corpus.txt
#   python stage0_pretokenization.py --model qwen --hf <dataset> \
#       --split train --field text
#
# Dtype follows the model config (uint32 for >64k vocab, else uint16).

import argparse
import os
import numpy as np

from config import MODELS


def iter_texts(a):
    if a.parquet:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(a.parquet)
        cols = pf.schema_arrow.names
        assert a.field in cols, f"column '{a.field}' not in parquet {cols}"
        for rg in range(pf.num_row_groups):          # row groups keep RAM flat
            for t in pf.read_row_group(rg, columns=[a.field]
                                       ).column(a.field).to_pylist():
                t = t.strip()
                if t:
                    yield t
    elif a.txt:
        with open(a.txt, "r", encoding="utf-8") as f:
            for para in f.read().split("\n"):
                para = para.strip()
                if para:
                    yield para
    else:
        from datasets import load_dataset
        if a.preset == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="train")
        elif a.hf == "wikitext":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                              split=a.split)
        else:
            ds = load_dataset(a.hf, split=a.split)
        for row in ds:
            t = row[a.field].strip()
            if t:
                yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, choices=["wikitext2"],
                    help="named recipe reproducing the pipeline's own "
                         "calibration data (writes wikitext_{tag}.bin)")
    ap.add_argument("--parquet", default=None,
                    help="parquet extract (preferred for large corpora)")
    ap.add_argument("--txt", default=None, help="raw text file")
    ap.add_argument("--hf", default="wikitext", help="HF dataset fallback")
    ap.add_argument("--split", default="train")
    ap.add_argument("--field", default="text")
    ap.add_argument("--tokens", type=int, default=4_000_000,
                    help="target token count; the last VAL_TOKENS tokens "
                         "of the file are the val segment (positional "
                         "split), so keep generous headroom")
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=512,
                    help="texts per tokenizer batch")
    a = ap.parse_args()

    cfg = MODELS[a.model]
    dt = np.uint32 if cfg["VAL_DTYPE"] == "uint32" else np.uint16
    if a.out:
        out = a.out
    elif a.preset == "wikitext2":
        out = os.path.join(os.path.dirname(cfg["VAL_BIN"]),
                           f"wikitext_{cfg['tag']}.bin")
    else:
        out = cfg["VAL_BIN"]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    eos = tok.eos_token_id
    assert eos is not None and eos < np.iinfo(dt).max, \
        f"eos id {eos} does not fit {dt}"

    buf = np.empty(a.tokens + 1024, dtype=dt)
    n = 0
    batch = []

    def flush(batch):
        nonlocal n, buf
        if not batch:
            return False
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            ids = ids + [eos]
            m = len(ids)
            if n + m > len(buf):
                nb = np.empty(len(buf) * 2, dtype=dt)
                nb[:n] = buf[:n]
                buf = nb
            buf[n:n + m] = ids
            n += m
        print(f"\r{n:,} tokens", end="", flush=True)
        return n >= a.tokens

    for text in iter_texts(a):
        batch.append(text)
        if len(batch) >= a.chunk:
            if flush(batch):
                break
            batch = []
    if batch and n < a.tokens:
        flush(batch)

    buf[:n].tofile(out)
    mx = int(buf[:n].max()) if n else 0
    print(f"\nsaved {n:,} tokens ({dt.__name__}) -> {out}")
    print(f"max token id {mx} (vocab {cfg['VOCAB']})  "
          f"file {os.path.getsize(out) / 1e6:.1f} MB")
    assert mx < cfg["VOCAB"]


if __name__ == "__main__":
    main()
