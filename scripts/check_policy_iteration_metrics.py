"""Gate policy iterations by translation quality and self-verification calibration.

GSPO reward can improve while corpus translation quality collapses. This gate
marks whether a policy iteration is promotable by comparing final metrics to
absolute floors and, optionally, to a baseline metrics JSON from the previous
policy/SFT run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


HIGHER_IS_BETTER = {
    "chrf++",
    "bleu",
    "token_f1",
    "self_verification_required_format_rate",
    "self_verification_thinking_format_rate",
}
LOWER_IS_BETTER = {
    "ter",
    "self_verification_false_confidence_rate",
    "self_verification_missing_score_rate",
    "self_verification_avg_score_gap",
    "spanish_leakage_penalty",
    "chat_artifact_penalty",
    "exact_source_copy_rate",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-chrf", type=float, default=35.0)
    parser.add_argument("--min-bleu", type=float, default=8.0)
    parser.add_argument("--min-token-f1", type=float, default=15.0)
    parser.add_argument("--max-ter", type=float, default=120.0)
    parser.add_argument("--min-format-rate", type=float, default=50.0)
    parser.add_argument("--max-false-confidence-rate", type=float, default=95.0)
    parser.add_argument("--max-missing-score-rate", type=float, default=50.0)
    parser.add_argument(
        "--min-chrf-delta",
        type=float,
        default=-1.0,
        help="Allowed candidate-baseline chrF++ delta. Negative values allow small regressions.",
    )
    parser.add_argument(
        "--min-bleu-delta",
        type=float,
        default=-1.0,
        help="Allowed candidate-baseline BLEU delta. Negative values allow small regressions.",
    )
    parser.add_argument(
        "--min-token-f1-delta",
        type=float,
        default=-1.0,
        help="Allowed candidate-baseline token-F1 delta. Negative values allow small regressions.",
    )
    parser.add_argument(
        "--max-false-confidence-delta",
        type=float,
        default=5.0,
        help="Allowed candidate-baseline false-confidence increase.",
    )
    return parser.parse_args(argv)


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return metric(candidate, key) - metric(baseline, key)


def check_absolute(candidate: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if metric(candidate, "chrf++") < args.min_chrf:
        reasons.append(f"chrf++ {metric(candidate, 'chrf++'):.4f} < {args.min_chrf:.4f}")
    if metric(candidate, "bleu") < args.min_bleu:
        reasons.append(f"bleu {metric(candidate, 'bleu'):.4f} < {args.min_bleu:.4f}")
    if metric(candidate, "token_f1") < args.min_token_f1:
        reasons.append(f"token_f1 {metric(candidate, 'token_f1'):.4f} < {args.min_token_f1:.4f}")
    if metric(candidate, "ter", default=999.0) > args.max_ter:
        reasons.append(f"ter {metric(candidate, 'ter', default=999.0):.4f} > {args.max_ter:.4f}")
    if metric(candidate, "self_verification_required_format_rate") < args.min_format_rate:
        reasons.append(
            "self_verification_required_format_rate "
            f"{metric(candidate, 'self_verification_required_format_rate'):.4f} < {args.min_format_rate:.4f}"
        )
    if metric(candidate, "self_verification_false_confidence_rate") > args.max_false_confidence_rate:
        reasons.append(
            "self_verification_false_confidence_rate "
            f"{metric(candidate, 'self_verification_false_confidence_rate'):.4f} > "
            f"{args.max_false_confidence_rate:.4f}"
        )
    if metric(candidate, "self_verification_missing_score_rate") > args.max_missing_score_rate:
        reasons.append(
            "self_verification_missing_score_rate "
            f"{metric(candidate, 'self_verification_missing_score_rate'):.4f} > {args.max_missing_score_rate:.4f}"
        )
    return reasons


def check_baseline(candidate: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    chrf_delta = delta(candidate, baseline, "chrf++")
    bleu_delta = delta(candidate, baseline, "bleu")
    token_f1_delta = delta(candidate, baseline, "token_f1")
    false_confidence_delta = delta(candidate, baseline, "self_verification_false_confidence_rate")
    if chrf_delta < args.min_chrf_delta:
        reasons.append(f"chrf++ delta {chrf_delta:.4f} < {args.min_chrf_delta:.4f}")
    if bleu_delta < args.min_bleu_delta:
        reasons.append(f"bleu delta {bleu_delta:.4f} < {args.min_bleu_delta:.4f}")
    if token_f1_delta < args.min_token_f1_delta:
        reasons.append(f"token_f1 delta {token_f1_delta:.4f} < {args.min_token_f1_delta:.4f}")
    if false_confidence_delta > args.max_false_confidence_delta:
        reasons.append(
            "self_verification_false_confidence_rate delta "
            f"{false_confidence_delta:.4f} > {args.max_false_confidence_delta:.4f}"
        )
    return reasons


def report_for(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reasons = check_absolute(candidate, args)
    deltas: dict[str, float] = {}
    if baseline is not None:
        deltas = {
            key: delta(candidate, baseline, key)
            for key in sorted(HIGHER_IS_BETTER.union(LOWER_IS_BETTER))
            if key in candidate or key in baseline
        }
        reasons.extend(check_baseline(candidate, baseline, args))
    return {
        "promoted": not reasons,
        "reasons": reasons,
        "candidate": {key: metric(candidate, key) for key in sorted(HIGHER_IS_BETTER.union(LOWER_IS_BETTER)) if key in candidate},
        "baseline": (
            {key: metric(baseline, key) for key in sorted(HIGHER_IS_BETTER.union(LOWER_IS_BETTER)) if key in baseline}
            if baseline is not None
            else None
        ),
        "deltas": deltas,
        "thresholds": {
            "min_chrf": args.min_chrf,
            "min_bleu": args.min_bleu,
            "min_token_f1": args.min_token_f1,
            "max_ter": args.max_ter,
            "min_format_rate": args.min_format_rate,
            "max_false_confidence_rate": args.max_false_confidence_rate,
            "max_missing_score_rate": args.max_missing_score_rate,
            "min_chrf_delta": args.min_chrf_delta,
            "min_bleu_delta": args.min_bleu_delta,
            "min_token_f1_delta": args.min_token_f1_delta,
            "max_false_confidence_delta": args.max_false_confidence_delta,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate = load_metrics(args.candidate_json)
    baseline = load_metrics(args.baseline_json) if args.baseline_json else None
    report = report_for(candidate, baseline, args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
