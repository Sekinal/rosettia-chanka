"""Download the high-confidence Chanka parallel data from nouman-10/MT-SharedTask.

Outputs a JSONL with (source, target) parallel pairs from:
  - extra/little_prince_que (1,312 pairs)
  - extra/web_misc_quy (985 pairs)
  - original/minedu_quy (643 pairs)
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path


BASE = "https://raw.githubusercontent.com/nouman-10/MT-SharedTask/main/data/parallel-data/quechua-spanish"


SOURCES = [
    # NOTE: Little Prince has SWAPPED labels — .es file actually contains Chanka, .quy contains Spanish.
    ("extra/little_prince_que/little_prince.es", "extra/little_prince_que/little_prince.quy", "little_prince_quy"),
    ("extra/web_misc_quy/web_misc.quy", "extra/web_misc_quy/web_misc.es", "web_misc_quy"),
    ("original/minedu_quy/minedu.quy", "original/minedu_quy/minedu.es", "minedu_quy"),
    ("extra/handbook_quy/handbook.quy", "extra/handbook_quy/handbook.es", "handbook_quy_quz_mixed"),
    ("original/dict_misc_quy/dict_misc.quy", "original/dict_misc_quy/dict_misc.es", "dict_misc_quy"),
]


def fetch(path: str, cache_dir: Path) -> list[str]:
    url = f"{BASE}/{path}"
    cache = cache_dir / path.replace("/", "_")
    if not cache.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, cache)
        except Exception as e:
            return []
    return cache.read_text(encoding="utf-8").splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/tmp/nouman_mt_data")
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    out = []

    for qu_path, es_path, source_name in SOURCES:
        print(f"Fetching {source_name}...")
        qu_lines = fetch(qu_path, cache)
        es_lines = fetch(es_path, cache)
        if not qu_lines or not es_lines:
            print(f"  FAIL — empty download")
            continue
        if len(qu_lines) != len(es_lines):
            print(f"  WARN unequal lengths: es={len(es_lines)}, quy={len(qu_lines)}")
            n = min(len(qu_lines), len(es_lines))
            qu_lines = qu_lines[:n]
            es_lines = es_lines[:n]
        n_kept = 0
        for es, qu in zip(es_lines, qu_lines):
            # Strip BOM and whitespace
            es = es.lstrip("﻿").strip()
            qu = qu.lstrip("﻿").strip()
            if not es or not qu:
                continue
            # Drop trailing punctuation-only differences for dict entries
            out.append({
                "source": es,
                "target": qu,
                "source_name": f"nouman_2023_{source_name}",
                "variant": "quy/chanka",
            })
            n_kept += 1
        print(f"  kept {n_kept} pairs")

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTotal: {len(out)} pairs written to {args.out_jsonl}")


if __name__ == "__main__":
    main()
