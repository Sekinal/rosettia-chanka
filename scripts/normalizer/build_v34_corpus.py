"""Assemble the v34 MT training corpus from the gated-normalized AmericasNLP
quy side + untouched Spanish + the (already-clean) v30 corpus.

Inputs:
  - amnlp normalized jsonl: {idx, chanka(original), spanish, chanka_normalized,
    normalizer_changed, ...}  (idx aligns to the filtered v34 input)
  - v30 corpus parquet (HF): reviewed_spanish / reviewed_chanka_quechua

Output: a parallel training JSONL with fields {spanish, chanka, source} where
`chanka` is the MINEDU-normalized target. Leakage vs AmericasNLP 2021 test was
already enforced at download (0 overlap); we re-assert here defensively.
"""
import argparse
import json
import os
import re
from pathlib import Path


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amnlp-normalized", required=True)
    ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--v30-parquet", default=None, help="optional local v30 parquet")
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    # Leakage guard set (defensive — download already verified 0 overlap)
    test_keys = set()
    if Path(args.test_quy).exists():
        test_keys = {norm_key(l) for l in open(args.test_quy) if l.strip()}

    rows = []
    n_amnlp = n_leak = n_empty = 0
    for line in open(args.amnlp_normalized):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        quy = (d.get("chanka_normalized") or "").strip()
        spa = (d.get("spanish") or "").strip()
        if not quy or not spa:
            n_empty += 1
            continue
        if norm_key(quy) in test_keys:
            n_leak += 1
            continue
        rows.append({"spanish": spa, "chanka": quy, "source": "amnlp_train_normalized"})
        n_amnlp += 1

    n_v30 = 0
    if args.v30_parquet and Path(args.v30_parquet).exists():
        import polars as pl
        df = pl.read_parquet(args.v30_parquet)
        for r in df.iter_rows(named=True):
            spa = (r.get("reviewed_spanish") or "").strip()
            quy = (r.get("reviewed_chanka_quechua") or "").strip()
            if spa and quy:
                rows.append({"spanish": spa, "chanka": quy, "source": "v30_manual"})
                n_v30 += 1

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"AmericasNLP normalized kept: {n_amnlp}  (leak-dropped {n_leak}, empty {n_empty})")
    print(f"v30 manual added:            {n_v30}")
    print(f"TOTAL v34 corpus:            {len(rows)} -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
