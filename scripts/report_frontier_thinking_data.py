"""Report quality and failure patterns for frontier thinking SFT data."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from scripts import check_frontier_thinking_data as gate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=5)
    return parser.parse_args(argv)


def iter_jsonl(path: Path | None) -> Iterable[dict[str, Any]]:
    if path is None or not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("summary", payload)


def audit_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    reasons: collections.Counter[str] = collections.Counter()
    passed = 0
    for record in records:
        audit = record.get("frontier_audit")
        if not isinstance(audit, dict):
            continue
        passed += int(bool(audit.get("pass")))
        try:
            scored.append(float(audit.get("score", 0.0)))
        except (TypeError, ValueError):
            pass
        reason = str(audit.get("reason") or "").strip()
        if reason:
            reasons[reason] += 1
    return {
        "audited_rows": sum(1 for record in records if isinstance(record.get("frontier_audit"), dict)),
        "audit_pass_rows": passed,
        "avg_audit_score": sum(scored) / max(1, len(scored)),
        "audit_reasons": dict(reasons.most_common(20)),
    }


def failure_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reasons = collections.Counter(str(record.get("reason") or "unknown") for record in records)
    audit_reasons = collections.Counter()
    for record in records:
        audit = record.get("audit")
        if isinstance(audit, dict) and audit.get("reason"):
            audit_reasons[str(audit["reason"])] += 1
    return {
        "failed_rows": len(records),
        "failure_reasons": dict(reasons.most_common(20)),
        "audit_rejection_reasons": dict(audit_reasons.most_common(20)),
    }


def sample_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": record.get("source"),
        "reference": record.get("reference"),
        "frontier_analysis": record.get("frontier_analysis"),
        "frontier_score": record.get("frontier_score"),
        "audit": record.get("frontier_audit"),
    }


def sample_failure(record: dict[str, Any]) -> dict[str, Any]:
    parsed = record.get("parsed") if isinstance(record.get("parsed"), dict) else {}
    audit = record.get("audit") if isinstance(record.get("audit"), dict) else None
    return {
        "source": record.get("source"),
        "reference": record.get("reference"),
        "reason": record.get("reason"),
        "parsed_analysis": parsed.get("analysis"),
        "audit": audit,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    accepted = list(iter_jsonl(args.output_jsonl))
    failures = list(iter_jsonl(args.failures_jsonl))
    counts = gate.load_counts(
        argparse.Namespace(
            summary_json=args.summary_json,
            output_jsonl=args.output_jsonl,
            failures_jsonl=args.failures_jsonl,
        )
    )
    primitives = gate.primitive_counts(args.output_jsonl)
    metrics = gate.gate_metrics(counts, primitives, min_tags_per_row=2)
    return {
        "summary": load_summary(args.summary_json),
        "counts": counts,
        "gate_metrics": metrics,
        "primitive_tag_counts": primitives.get("primitive_tag_counts", {}),
        "primitive_row_tag_counts": primitives.get("primitive_row_tag_counts", []),
        "audit": audit_summary(accepted),
        "failures": failure_summary(failures),
        "accepted_samples": [sample_record(record) for record in accepted[: max(0, args.sample_size)]],
        "failure_samples": [sample_failure(record) for record in failures[: max(0, args.sample_size)]],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = report["gate_metrics"]
    lines = [
        "# Frontier Thinking Data Report",
        "",
        f"- accepted rows: {metrics['written_rows']:.0f}",
        f"- failed rows: {metrics['failed_rows']:.0f}",
        f"- accept rate: {metrics['accept_rate']:.4f}",
        f"- avg primitive tags: {metrics['avg_primitive_tags']:.4f}",
        f"- distinct primitives: {metrics['distinct_primitives']:.0f}",
        "",
        "## Primitive Tags",
        "",
    ]
    for tag, count in sorted(report["primitive_tag_counts"].items()):
        lines.append(f"- `{tag}`: {count}")
    lines.extend(["", "## Failure Reasons", ""])
    for reason, count in report["failures"]["failure_reasons"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Accepted Samples", ""])
    for sample in report["accepted_samples"]:
        lines.append(f"- `{sample.get('source')}` -> {sample.get('frontier_analysis')}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    if args.report_md:
        write_markdown(report, args.report_md)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
