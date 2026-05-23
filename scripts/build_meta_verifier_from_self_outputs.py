"""Build meta-verifier rows from real self-verifying translation outputs.

Cold-start meta-verifier data uses synthetic faithful/flawed analyses. This
script is the next DeepSeekMath-style loop: run a self-verifying generator,
compare its final translation and self-analysis to clean references, and turn
real self-analysis failures into meta-verifier training examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_gspo_chanka_unsloth as gspo
from scripts import train_verifier_chanka_unsloth as verifier


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-predictions-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--min-quality-gap", type=float, default=0.20)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args(argv)


def self_score_from_payload(payload: dict[str, Any]) -> float | None:
    parsed = payload.get("self_verification")
    if isinstance(parsed, dict):
        value = parsed.get("self_score")
        if isinstance(value, int | float):
            return max(0.0, min(1.0, float(value)))
    raw = str(payload.get("raw_prediction") or payload.get("prediction") or "")
    value = gspo.parse_self_verification_output(raw)["self_score"]
    return float(value) if isinstance(value, float) else None


def analysis_from_payload(payload: dict[str, Any]) -> str:
    parsed = payload.get("self_verification")
    if isinstance(parsed, dict):
        analysis = str(parsed.get("analysis") or "")
        if analysis:
            return gspo.normalize_text(analysis)
    raw = str(payload.get("raw_prediction") or payload.get("prediction") or "")
    return str(gspo.parse_self_verification_output(raw)["analysis"])


def candidate_from_payload(payload: dict[str, Any]) -> str:
    parsed = payload.get("self_verification")
    if isinstance(parsed, dict):
        translation = str(parsed.get("translation") or "")
        if translation:
            return gspo.normalize_text(translation)
    return gspo.normalize_text(str(payload.get("prediction") or ""))


def meta_label_for_self_output(
    source: str,
    reference: str,
    candidate: str,
    analysis: str,
    self_score: float | None,
    min_quality_gap: float,
) -> tuple[str, dict[str, float | str]]:
    true_score = gspo.bounded_translation_quality_score(candidate, reference, source)
    proxy_meta = gspo.self_analysis_meta_score(analysis, self_score, true_score)
    gap = abs((self_score if self_score is not None else 0.0) - true_score)
    if self_score is None:
        score = 0.0
        severity = "critical"
        rationale = "missing_self_score"
    elif gap >= min_quality_gap and self_score > true_score:
        score = min(proxy_meta, 0.20)
        severity = "critical" if gap >= 0.40 else "major"
        rationale = "false_confidence_self_score_too_high"
    elif gap >= min_quality_gap and self_score < true_score:
        score = min(proxy_meta, 0.45)
        severity = "major"
        rationale = "underconfident_or_hallucinated_issues"
    elif proxy_meta < 0.65:
        score = proxy_meta
        severity = "major"
        rationale = "analysis_does_not_match_translation_quality"
    else:
        score = max(0.80, proxy_meta)
        severity = "none" if score >= 0.90 else "minor"
        rationale = "self_analysis_matches_translation_quality"
    return (
        verifier.verifier_target(score, severity, rationale),
        {
            "true_score": true_score,
            "self_score": self_score if self_score is not None else -1.0,
            "self_score_gap": gap if self_score is not None else 1.0,
            "proxy_meta_score": proxy_meta,
            "rationale": rationale,
        },
    )


def build_records(paths: Sequence[Path], min_quality_gap: float, max_records: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    missing_format = 0
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                source = gspo.normalize_text(str(payload.get("source") or ""))
                reference = gspo.normalize_text(str(payload.get("reference") or payload.get("target") or ""))
                candidate = candidate_from_payload(payload)
                analysis = analysis_from_payload(payload)
                if not source or not reference or not candidate:
                    continue
                self_score = self_score_from_payload(payload)
                if self_score is None or not analysis:
                    missing_format += 1
                label, diagnostics = meta_label_for_self_output(
                    source,
                    reference,
                    candidate,
                    analysis,
                    self_score,
                    min_quality_gap,
                )
                key = (source, reference, candidate, analysis)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "source": source,
                        "reference": reference,
                        "candidate": candidate,
                        "analysis": analysis,
                        "label": label,
                        "diagnostics": diagnostics,
                    }
                )
                if max_records is not None and len(records) >= max_records:
                    break
        if max_records is not None and len(records) >= max_records:
            break
    summary = {
        "input_files": [str(path) for path in paths],
        "records": len(records),
        "missing_or_incomplete_self_verification": missing_format,
        "min_quality_gap": min_quality_gap,
    }
    return records, summary


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    records, summary = build_records(args.self_predictions_jsonl, args.min_quality_gap, args.max_records)
    write_jsonl(args.output_jsonl, records)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
