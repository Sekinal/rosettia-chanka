"""Gate frontier-thinking datasets before SFT.

This prevents spending GPU time on tiny or low-yield synthetic thinking sets.
The gate can read the builder summary JSON, or count accepted/failure JSONL
files directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    parser.add_argument("--min-written-rows", type=int, default=64)
    parser.add_argument("--min-accept-rate", type=float, default=0.5)
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def load_counts(args: argparse.Namespace) -> dict[str, int]:
    if args.summary_json is not None and args.summary_json.exists():
        payload = json.loads(args.summary_json.read_text())
        summary = payload.get("summary", payload)
        written = int(summary.get("total_written_rows", summary.get("written_rows", summary.get("new_written_rows", 0))))
        failed = int(summary.get("total_failed_rows", summary.get("failed_rows", summary.get("new_failed_rows", 0))))
        requested = int(summary.get("requested_rows", written + failed))
        return {"written": written, "failed": failed, "requested": requested}
    written = count_jsonl(args.output_jsonl)
    failed = count_jsonl(args.failures_jsonl)
    return {"written": written, "failed": failed, "requested": written + failed}


def gate_metrics(counts: dict[str, int]) -> dict[str, float]:
    attempted = counts["written"] + counts["failed"]
    accept_rate = counts["written"] / max(1, attempted)
    return {
        "written_rows": float(counts["written"]),
        "failed_rows": float(counts["failed"]),
        "requested_rows": float(counts["requested"]),
        "attempted_rows": float(attempted),
        "accept_rate": accept_rate,
    }


def check_gate(metrics: dict[str, float], min_written_rows: int, min_accept_rate: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics["written_rows"] < min_written_rows:
        reasons.append(f"written_rows {metrics['written_rows']:.0f} < {min_written_rows}")
    if metrics["accept_rate"] < min_accept_rate:
        reasons.append(f"accept_rate {metrics['accept_rate']:.4f} < {min_accept_rate:.4f}")
    return not reasons, reasons


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = gate_metrics(load_counts(args))
    passed, reasons = check_gate(metrics, args.min_written_rows, args.min_accept_rate)
    report = {
        **metrics,
        "min_written_rows": args.min_written_rows,
        "min_accept_rate": args.min_accept_rate,
        "passed": passed,
        "reasons": reasons,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
