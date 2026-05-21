"""Summarize external GSPO checkpoint evaluation metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_gspo_canaries import format_value, selection_score


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--summary-jsonl", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    return parser.parse_args(argv)


def checkpoint_label(record: dict[str, Any]) -> str:
    adapter_path = str(record.get("adapter_path", "unknown"))
    return Path(adapter_path).name if adapter_path != "unknown" else "unknown"


def load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    payload["metrics_path"] = str(path)
    payload["checkpoint_label"] = path.name.removesuffix("_metrics.json")
    payload["adapter_label"] = checkpoint_label(payload)
    payload["selection_score"] = selection_score(payload)
    return payload


def collect_metrics(eval_dir: Path) -> list[dict[str, Any]]:
    records = [load_metrics(path) for path in sorted(eval_dir.glob("*_metrics.json"))]
    return sorted(
        records,
        key=lambda record: (
            float(record.get("selection_score", 0.0)),
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


def write_markdown(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GSPO Checkpoint Eval Summary",
        "",
        "Ranking uses `selection_score` from external corpus metrics. It is a triage score, not a Chanka quality oracle.",
        "",
        "| Rank | Checkpoint | Selection | chrF++ | BLEU | token F1 | source copy % | exact copy % | leakage % | artifact % | TER |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, record in enumerate(records, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(record.get("checkpoint_label", record.get("adapter_label", "unknown"))),
                    format_value(record.get("selection_score", 0.0)),
                    format_value(record.get("chrf++", 0.0)),
                    format_value(record.get("bleu", 0.0)),
                    format_value(record.get("token_f1", 0.0)),
                    format_value(record.get("source_copy_ratio", 0.0)),
                    format_value(record.get("exact_source_copy_rate", 0.0)),
                    format_value(record.get("spanish_leakage_penalty", 0.0)),
                    format_value(record.get("chat_artifact_penalty", 0.0)),
                    format_value(record.get("ter", 0.0)),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    records = collect_metrics(args.eval_dir)
    summary_jsonl = args.summary_jsonl or (args.eval_dir / "summary.jsonl")
    summary_md = args.summary_md or (args.eval_dir / "summary.md")
    write_jsonl(records, summary_jsonl)
    write_markdown(records, summary_md)
    print(f"Wrote {len(records)} records to {summary_jsonl}")
    print(f"Wrote Markdown summary to {summary_md}")
    if records:
        best = records[0]
        print(
            "Best checkpoint: "
            f"{best.get('checkpoint_label')} "
            f"selection={best.get('selection_score')} "
            f"chrf++={best.get('chrf++')} "
            f"copy={best.get('source_copy_ratio')}"
        )


if __name__ == "__main__":
    main()
