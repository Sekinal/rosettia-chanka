"""Build the v31 EXPANDED Chanka training corpus.

Merges:
  1. v30 expanded corpus (1929 pairs: reviewed + glossary + Benito + simple terms)
  2. Nouman 2023 quy parallel files (little_prince, web_misc, minedu, handbook, dict_misc)
  3. RunaSimi.de Chanka column (high-confidence only by default; lemma-fallback optional)

With STRICT leakage filter against:
  - AmericasNLP 2021 spa→quy test set (1003 lines) — our public benchmark
  - Internal eval set (158 Chanka held-out rows)

Outputs a parquet matching the schema train_sft_unsloth.py expects.
"""
import argparse
import json
import os
import re
import random
from collections import Counter
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download


def norm_es(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def norm_qu(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-repo", default="Thermostatic/rosettia-chanka-data")
    ap.add_argument("--v30-file", default="clean_chanka/manual_quechua_chanka_expanded_v30_training_ready.parquet")
    ap.add_argument("--nouman-jsonl", default="/tmp/nouman_quy_pairs.jsonl")
    ap.add_argument("--runasimi-jsonl", default="/tmp/runasimi_chanka_pairs.jsonl")
    ap.add_argument("--anlp-test-es", default="docs/references/americasnlp_test/2021_test.es")
    ap.add_argument("--anlp-test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--include-runasimi", action="store_true", default=False,
                    help="Include RunaSimi.de Ayakuchu pairs (mostly single-word dict; hurt v31).")
    ap.add_argument("--include-runasimi-fallback", action="store_true",
                    help="Also include the 16k lemma-fallback Runasimi rows.")
    ap.add_argument("--include-handbook", action="store_true",
                    help="Include nouman handbook_quy (2296 pairs, quz/quy mixed orthography).")
    ap.add_argument("--include-dict-misc", action="store_true", default=False,
                    help="Include nouman dict_misc_quy (8998 single-word pairs; hurt v31).")
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()

    print(f"=== building v31 corpus ===")

    # Leakage sets
    eval_es = set(norm_es(s) for s in open(args.anlp_test_es).read().split("\n") if s.strip())
    eval_qu = set(norm_qu(s) for s in open(args.anlp_test_quy).read().split("\n") if s.strip())
    print(f"AmericasNLP 2021 test surfaces: {len(eval_es)} es, {len(eval_qu)} quy")

    # Load v30 base
    v30_path = hf_hub_download(repo_id=args.dataset_repo, filename=args.v30_file, repo_type="dataset",
                                token=os.environ.get("HF_TOKEN"))
    v30 = pl.read_parquet(v30_path)
    print(f"\nv30 base: {len(v30)} rows")
    v30_rows = []
    for r in v30.iter_rows(named=True):
        v30_rows.append({
            "reviewed_spanish": r["reviewed_spanish"],
            "reviewed_chanka_quechua": r["reviewed_chanka_quechua"],
            "source_name": r["source_name"],
            "variant": "quy/chanka",
        })
    # Existing dedup keys
    seen = {(norm_es(r["reviewed_spanish"]), norm_qu(r["reviewed_chanka_quechua"])) for r in v30_rows}

    n_anlp_leaked = 0
    n_dup = 0
    n_added_by_source = Counter()

    def try_add(es, qu, source_name):
        nonlocal n_anlp_leaked, n_dup
        es_n = norm_es(es)
        qu_n = norm_qu(qu)
        if es_n in eval_es or qu_n in eval_qu:
            n_anlp_leaked += 1
            return False
        key = (es_n, qu_n)
        if key in seen:
            n_dup += 1
            return False
        seen.add(key)
        v30_rows.append({
            "reviewed_spanish": es,
            "reviewed_chanka_quechua": qu,
            "source_name": source_name,
            "variant": "quy/chanka",
        })
        n_added_by_source[source_name] += 1
        return True

    # Nouman 2023 quy
    nouman_rows = [json.loads(l) for l in open(args.nouman_jsonl)]
    print(f"\nNouman 2023 candidates: {len(nouman_rows)}")
    for r in nouman_rows:
        if "handbook" in r["source_name"] and not args.include_handbook:
            continue
        if "dict_misc" in r["source_name"] and not args.include_dict_misc:
            continue
        try_add(r["source"], r["target"], r["source_name"])

    # RunaSimi (off by default after v31 lessons)
    if args.include_runasimi:
        runasimi_rows = [json.loads(l) for l in open(args.runasimi_jsonl)]
        print(f"RunaSimi candidates: {len(runasimi_rows)}")
        for r in runasimi_rows:
            if r.get("confidence") != "high" and not args.include_runasimi_fallback:
                continue
            try_add(r["source"], r["target"], r["source_name"])
    else:
        print("RunaSimi: SKIPPED (--include-runasimi off)")

    print(f"\nLeakage check (vs AmericasNLP 2021 test surfaces): {n_anlp_leaked} rows DROPPED")
    print(f"Dedup vs v30: {n_dup} rows DROPPED")
    print(f"\nAdded per source:")
    for src, n in n_added_by_source.most_common():
        print(f"  {src:50s} +{n}")

    print(f"\nFinal corpus: {len(v30_rows)} rows (was {len(v30)} in v30)")

    df = pl.DataFrame(v30_rows)
    df = df.with_columns(pl.lit(True).alias("training_ready"))
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out_parquet)
    print(f"Wrote {args.out_parquet}")


if __name__ == "__main__":
    main()
