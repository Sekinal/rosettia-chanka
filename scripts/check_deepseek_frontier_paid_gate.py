"""Require a passed paid frontier data gate before SFT/GSPO continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paid-gate-json", type=Path, required=True)
    parser.add_argument("--frontier-report-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def check_paid_gate(paid_gate_json: Path, frontier_report_json: Path | None = None) -> dict[str, Any]:
    reasons: list[str] = []
    paid_gate = load_json(paid_gate_json)
    report = load_json(frontier_report_json) if frontier_report_json is not None else None

    if paid_gate is None:
        reasons.append(f"paid gate JSON missing or invalid: {paid_gate_json}")
    elif paid_gate.get("passed") is not True:
        reasons.append("paid gate did not pass")

    if frontier_report_json is not None:
        if report is None:
            reasons.append(f"frontier report JSON missing or invalid: {frontier_report_json}")
        else:
            metrics = report.get("gate_metrics")
            if not isinstance(metrics, dict):
                reasons.append("frontier report missing gate_metrics")
            elif float(metrics.get("written_rows", 0) or 0) <= 0:
                reasons.append("frontier report has no accepted rows")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "paid_gate_json": str(paid_gate_json),
        "frontier_report_json": str(frontier_report_json) if frontier_report_json else None,
        "paid_gate_passed": paid_gate.get("passed") if isinstance(paid_gate, dict) else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_paid_gate(args.paid_gate_json, args.frontier_report_json)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
