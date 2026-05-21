"""Summarize GSPO canary run metrics into JSONL and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


CORE_METRICS = (
    "trainer_eval_reward",
    "chrf++",
    "bleu",
    "token_f1",
    "exact_source_copy_rate",
    "source_copy_ratio",
    "spanish_leakage_penalty",
    "ter",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--summary-jsonl", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    return parser.parse_args(argv)


def load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    payload["metrics_path"] = str(path)
    return payload


def collect_metrics(sweep_dir: Path) -> list[dict[str, Any]]:
    records = [load_metrics(path) for path in sorted(sweep_dir.glob("*/chanka_gspo/final_metrics.json"))]
    return sorted(
        records,
        key=lambda record: (
            float(record.get("trainer_eval_reward", 0.0)),
            float(record.get("chrf++", 0.0)),
            -float(record.get("source_copy_ratio", 100.0)),
        ),
        reverse=True,
    )


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GSPO Canary Summary",
        "",
        "| Rank | Profile | Eval reward | chrF++ | BLEU | token F1 | source copy % | exact copy % | leakage % |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, record in enumerate(records, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(record.get("reward_profile", "unknown")),
                    format_value(record.get("trainer_eval_reward", 0.0)),
                    format_value(record.get("chrf++", 0.0)),
                    format_value(record.get("bleu", 0.0)),
                    format_value(record.get("token_f1", 0.0)),
                    format_value(record.get("source_copy_ratio", 0.0)),
                    format_value(record.get("exact_source_copy_rate", 0.0)),
                    format_value(record.get("spanish_leakage_penalty", 0.0)),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    records = collect_metrics(args.sweep_dir)
    summary_jsonl = args.summary_jsonl or (args.sweep_dir / "summary.jsonl")
    summary_md = args.summary_md or (args.sweep_dir / "summary.md")
    write_jsonl(records, summary_jsonl)
    write_markdown(records, summary_md)
    print(f"Wrote {len(records)} records to {summary_jsonl}")
    print(f"Wrote Markdown summary to {summary_md}")
    if records:
        best = records[0]
        print(
            "Best profile: "
            f"{best.get('reward_profile')} "
            f"reward={best.get('trainer_eval_reward')} "
            f"chrf++={best.get('chrf++')} "
            f"copy={best.get('source_copy_ratio')}"
        )


if __name__ == "__main__":
    main()
