"""Summarize DeepSeekMath-style translation cycle manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


CORE_METRICS = (
    "chrf++",
    "bleu",
    "token_f1",
    "ter",
    "self_verification_required_format_rate",
    "self_verification_false_confidence_rate",
    "self_verification_missing_score_rate",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing cycle_manifest.json files.")
    parser.add_argument("--summary-jsonl", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    return parser.parse_args(argv)


def metric(metrics: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(metrics, dict):
        return default
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def cycle_score(record: dict[str, Any]) -> float:
    metrics = record.get("metrics")
    promotion = record.get("promotion")
    promoted_bonus = 10.0 if bool(record.get("promoted")) else 0.0
    hardcases = record.get("output_hardcases") if isinstance(record.get("output_hardcases"), dict) else {}
    hardcase_signal = min(5.0, 0.02 * float(hardcases.get("valid_records", 0) or 0))
    return (
        promoted_bonus
        + (0.45 * metric(metrics, "chrf++"))
        + (0.15 * metric(metrics, "bleu"))
        + (0.25 * metric(metrics, "token_f1"))
        - (0.03 * metric(metrics, "ter", 100.0))
        - (0.05 * metric(metrics, "self_verification_false_confidence_rate"))
        - (0.04 * metric(metrics, "self_verification_missing_score_rate"))
        + hardcase_signal
        - (2.0 if isinstance(promotion, dict) and promotion.get("reasons") else 0.0)
    )


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    payload["manifest_path"] = str(path)
    payload["cycle_score"] = cycle_score(payload)
    return payload


def collect_manifests(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("**/cycle_manifest.json"))
    records = [load_manifest(path) for path in paths]
    return sorted(
        records,
        key=lambda record: (
            bool(record.get("promoted")),
            float(record.get("cycle_score", 0.0)),
            metric(record.get("metrics"), "chrf++"),
            -metric(record.get("metrics"), "self_verification_false_confidence_rate"),
        ),
        reverse=True,
    )


def write_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DeepSeekMath Cycle Summary",
        "",
        "Ranking prioritizes promoted cycles, then external translation metrics and calibration. Failed cycles can still be useful if they mined many hardcases.",
        "",
        "| Rank | Promoted | Score | Stamp | Stage | chrF++ | BLEU | token F1 | TER | format % | false-conf % | missing-score % | input hardcases | output hardcases |",
        "| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, record in enumerate(records, start=1):
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        input_hardcases = record.get("input_hardcases") if isinstance(record.get("input_hardcases"), dict) else {}
        output_hardcases = record.get("output_hardcases") if isinstance(record.get("output_hardcases"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    "yes" if record.get("promoted") else "no",
                    format_value(float(record.get("cycle_score", 0.0))),
                    str(record.get("stamp", "unknown")),
                    str(record.get("stage", "gspo")),
                    format_value(metric(metrics, "chrf++")),
                    format_value(metric(metrics, "bleu")),
                    format_value(metric(metrics, "token_f1")),
                    format_value(metric(metrics, "ter")),
                    format_value(metric(metrics, "self_verification_required_format_rate")),
                    format_value(metric(metrics, "self_verification_false_confidence_rate")),
                    format_value(metric(metrics, "self_verification_missing_score_rate")),
                    str(input_hardcases.get("valid_records", 0)),
                    str(output_hardcases.get("valid_records", 0)),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    records = collect_manifests(args.root)
    summary_jsonl = args.summary_jsonl or (args.root / "cycle_summary.jsonl")
    summary_md = args.summary_md or (args.root / "cycle_summary.md")
    write_jsonl(records, summary_jsonl)
    write_markdown(records, summary_md)
    print(f"Wrote {len(records)} cycle records to {summary_jsonl}")
    print(f"Wrote Markdown summary to {summary_md}")
    if records:
        best = records[0]
        print(
            "Best cycle: "
            f"{best.get('stamp')} promoted={best.get('promoted')} "
            f"chrf++={metric(best.get('metrics'), 'chrf++')} "
            f"score={best.get('cycle_score')}"
        )


if __name__ == "__main__":
    main()
