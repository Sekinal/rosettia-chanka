"""Build a mixed JSONL SFT file from clean Chanka anchors and pseudo-labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_gspo_chanka_unsloth as gspo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--pseudo-jsonl", type=Path, action="append", default=[], help="Pseudo-label JSONL. Repeatable.")
    parser.add_argument("--pseudo-source-field", default="source")
    parser.add_argument("--pseudo-target-field", default="prediction")
    parser.add_argument("--pseudo-reference-field", default="reference")
    parser.add_argument("--max-pseudo-samples", type=int, default=None)
    parser.add_argument(
        "--no-clean-anchors",
        action="store_true",
        help="Write only pseudo-label rows. By default clean Chanka train anchors are included first.",
    )
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--dataset-file", default=gspo.CHANKA_FILE)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-clean-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc


def clean_anchor_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = gspo.load_chanka_rows(args.dataset_repo, args.dataset_file)
    train_rows, _ = gspo.split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_samples=args.max_clean_train_samples,
        max_eval_samples=None,
    )
    anchors: list[dict[str, str]] = []
    for row in train_rows:
        anchors.append(
            {
                "source": gspo.normalize_text(row["source"]),
                "target": gspo.normalize_text(row["target"]),
                "reference": gspo.normalize_text(row["target"]),
                "source_name": row.get("source_name", args.dataset_file),
                "variant": row.get("variant", "quy/chanka"),
                "label_type": "clean_anchor",
            }
        )
    return anchors


def pseudo_label_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in args.pseudo_jsonl:
        for record in iter_jsonl(path):
            source = gspo.normalize_text(str(record.get(args.pseudo_source_field, "")))
            target = gspo.normalize_text(str(record.get(args.pseudo_target_field, "")))
            if not source or not target:
                continue
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "reference": gspo.normalize_text(str(record.get(args.pseudo_reference_field, ""))),
                    "source_name": str(record.get("source_name") or path.name),
                    "variant": str(record.get("variant") or "quy/chanka_pseudo"),
                    "label_type": "pseudo_mbr",
                }
            )
            if args.max_pseudo_samples is not None and len(rows) >= args.max_pseudo_samples:
                return rows
    return rows


def dedupe_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["source"].lower(), row["target"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not args.no_clean_anchors:
        rows.extend(clean_anchor_rows(args))
    rows.extend(pseudo_label_rows(args))
    rows = dedupe_rows(rows)
    if not rows:
        raise RuntimeError("No rows available for mixed SFT JSONL.")
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_jsonl(args.output_jsonl, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label_type"]] = counts.get(row["label_type"], 0) + 1
    print(json.dumps({"output_jsonl": str(args.output_jsonl), "rows": len(rows), "label_type_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
