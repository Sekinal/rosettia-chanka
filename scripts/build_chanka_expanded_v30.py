"""Build the expanded Chanka training corpus for v30.

Merges:
  1. Existing 1055 reviewed pairs (manual_quechua_chanka_parallel_training_ready_augmented.parquet)
  2. 423 glossary entries (split atomically — each "Saqiy, wischuy" → 2 separate pairs)
  3. 219 simple term pairs (single-word Chanka equivalents)
  4. 81 alternative split parallel rows
  5. 382 Benito 2018 Chanka dict pairs (parsed locally)

Outputs a Parquet file matching the schema expected by train_sft_unsloth.py:
  - reviewed_spanish, reviewed_chanka_quechua, source_name, variant, training_ready

Strict eval-leakage filter: drop any pair whose Spanish source duplicates an
existing eval-set source (158 rows from the 0.15 split with seed=3407).
"""
import argparse
import json
import re
import random
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download


def norm_es(s: str) -> str:
    """Normalize Spanish for de-dup / leakage filter."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def load_existing_chanka(repo: str) -> list[dict]:
    """Load the 1055 reviewed Chanka pairs as-is."""
    path = hf_hub_download(repo_id=repo, filename="clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet", repo_type="dataset")
    f = pl.read_parquet(path)
    out = []
    for r in f.iter_rows(named=True):
        es = str(r.get("reviewed_spanish", "")).strip()
        qu = str(r.get("reviewed_chanka_quechua", "")).strip()
        if es and qu:
            out.append({
                "reviewed_spanish": es,
                "reviewed_chanka_quechua": qu,
                "source_name": "manual_chanka_2014_reviewed",
                "variant": "quy/chanka",
            })
    print(f"  base reviewed: {len(out)} pairs")
    return out


def load_glossary_entries(repo: str) -> list[dict]:
    """Load 423 rich glossary entries. Split multi-translation cells into atomic pairs."""
    path = hf_hub_download(repo_id=repo, filename="clean_chanka/manual_quechua_chanka_glossary_entries.parquet", repo_type="dataset")
    f = pl.read_parquet(path)
    out = []
    for r in f.iter_rows(named=True):
        es = str(r.get("source_term") or "").strip()
        qu_raw = str(r.get("target_text") or "").strip()
        if not es or not qu_raw:
            continue
        # qu_raw can be like "Saqiy, wischuy. Abandonado: saqisqa, wischusqa"
        # Split on `;`, `.`, drop bracketed clarifiers; only keep tokens that look Chanka.
        # First, take the part BEFORE any colon (the primary translation; the part after is often a derived form).
        main_part = qu_raw.split(":")[0].split(".")[0]
        for option in re.split(r"[,;/]", main_part):
            option = option.strip()
            # Drop bracketed clarifiers
            option = re.sub(r"\s*\([^)]*\)\s*", " ", option).strip()
            if not option:
                continue
            # Single-word Chanka usually
            first_word = option.split()[0].lower()
            if not first_word:
                continue
            # Skip if contains accented Spanish vowels
            if re.search(r"[áéíóú]", first_word):
                continue
            out.append({
                "reviewed_spanish": es.lower().strip(),
                "reviewed_chanka_quechua": first_word,
                "source_name": "manual_chanka_2014_glossary_entries",
                "variant": "quy/chanka",
            })
    print(f"  glossary_entries: {len(out)} pairs")
    return out


def load_glossary_simple_terms(repo: str) -> list[dict]:
    """Load 219 simple term pairs."""
    path = hf_hub_download(repo_id=repo, filename="clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet", repo_type="dataset")
    f = pl.read_parquet(path)
    out = []
    for r in f.iter_rows(named=True):
        es = str(r.get("source_term") or "").strip()
        qu = str(r.get("target_text") or "").strip()
        if es and qu:
            out.append({
                "reviewed_spanish": es.lower().strip(),
                "reviewed_chanka_quechua": qu.strip(),
                "source_name": "manual_chanka_2014_glossary_simple_terms",
                "variant": "quy/chanka",
            })
    print(f"  glossary_simple_terms: {len(out)} pairs")
    return out


def load_alternative_splits(repo: str) -> list[dict]:
    """Load 81 alternative-split parallel rows."""
    path = hf_hub_download(repo_id=repo, filename="clean_chanka/manual_quechua_chanka_parallel_alternative_splits.parquet", repo_type="dataset")
    f = pl.read_parquet(path)
    out = []
    for r in f.iter_rows(named=True):
        es = str(r.get("reviewed_spanish") or r.get("spanish") or "").strip()
        qu = str(r.get("reviewed_chanka_quechua") or r.get("chanka_quechua") or "").strip()
        if es and qu and r.get("training_ready"):
            out.append({
                "reviewed_spanish": es,
                "reviewed_chanka_quechua": qu,
                "source_name": "manual_chanka_2014_alternative_splits",
                "variant": "quy/chanka",
            })
    print(f"  alternative_splits: {len(out)} pairs")
    return out


def load_benito(path: str) -> list[dict]:
    """Load Benito 2018 Chanka dict (parsed earlier)."""
    out = []
    for line in open(path):
        d = json.loads(line)
        out.append({
            "reviewed_spanish": d["source"].strip(),
            "reviewed_chanka_quechua": d["target"].strip(),
            "source_name": d.get("source_name", "benito_2018_chanka_dict"),
            "variant": "quy/chanka",
        })
    print(f"  benito_2018: {len(out)} pairs")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-repo", default="Thermostatic/rosettia-chanka-data")
    ap.add_argument("--benito-jsonl", default="/tmp/benito_pairs.jsonl")
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    args = ap.parse_args()

    print("Loading sources...")
    base = load_existing_chanka(args.dataset_repo)
    gloss = load_glossary_entries(args.dataset_repo)
    simple = load_glossary_simple_terms(args.dataset_repo)
    splits = load_alternative_splits(args.dataset_repo)
    benito = load_benito(args.benito_jsonl)

    all_rows = base + gloss + simple + splits + benito
    print(f"Total combined: {len(all_rows)} rows")

    # Determine eval-set Spanish sources from the base (deterministic 0.15 split with seed=3407)
    rng = random.Random(args.seed)
    shuffled = base[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * args.validation_fraction))
    eval_es = {norm_es(r["reviewed_spanish"]) for r in shuffled[:eval_size]}
    print(f"Eval-set Spanish surfaces to exclude: {len(eval_es)}")

    # Dedupe by (norm_es, qu) and drop any row whose normalized Spanish is in eval set
    # EXCEPT for the original base rows (the trainer's own split will handle base-set eval/train)
    seen = set()
    out_rows = []
    n_eval_excluded = 0
    n_dup_excluded = 0
    for r in all_rows:
        es_n = norm_es(r["reviewed_spanish"])
        qu_n = r["reviewed_chanka_quechua"].strip().lower()
        key = (es_n, qu_n)
        if key in seen:
            n_dup_excluded += 1
            continue
        seen.add(key)
        # Don't filter out base rows (their split handles itself)
        if r["source_name"] != "manual_chanka_2014_reviewed":
            if es_n in eval_es:
                n_eval_excluded += 1
                continue
        out_rows.append(r)

    print(f"After dedup: {len(out_rows)} (dropped {n_dup_excluded} dup, {n_eval_excluded} eval-leak)")

    # Source breakdown
    from collections import Counter
    src_counts = Counter(r["source_name"] for r in out_rows)
    print("\nFinal breakdown:")
    for src, n in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:50s} {n:6d}")

    # Write parquet matching the schema train_sft_unsloth.py expects
    df = pl.DataFrame(out_rows)
    # Add a training_ready column for compatibility
    df = df.with_columns(pl.lit(True).alias("training_ready"))
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out_parquet)
    print(f"\nWrote {args.out_parquet}  ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
