"""Gate frontier-thinking datasets before SFT.

This prevents spending GPU time on tiny or low-yield synthetic thinking sets.
The gate can read the builder summary JSON, or count accepted/failure JSONL
files directly.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

PRIMITIVES = (
    "[SIGNIFICADO]",
    "[GRAMATICA]",
    "[ENTIDADES]",
    "[TERMINOLOGIA]",
    "[ANTI_COPIA]",
)

TAG_WORDS = {tag.strip("[]").lower() for tag in PRIMITIVES}
STOPWORDS = {
    "a",
    "al",
    "de",
    "del",
    "el",
    "en",
    "es",
    "la",
    "las",
    "lo",
    "los",
    "por",
    "que",
    "se",
    "un",
    "una",
    "y",
}
TRANSLATION_SPECIFIC_TERMS = {
    "adicion",
    "anticopia",
    "calco",
    "caso",
    "chanka",
    "copia",
    "entidad",
    "entidades",
    "espanol",
    "final",
    "forma",
    "frase",
    "gramatica",
    "literal",
    "mantiene",
    "natural",
    "nombre",
    "numero",
    "omite",
    "preserva",
    "quechua",
    "referencia",
    "significado",
    "sufijo",
    "termino",
    "terminologia",
    "traduccion",
    "verbo",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    parser.add_argument("--min-written-rows", type=int, default=64)
    parser.add_argument("--min-accept-rate", type=float, default=0.5)
    parser.add_argument(
        "--min-primitive-tags-per-row",
        type=int,
        default=0,
        help="Require this many primitive tags on average per accepted row. Disabled at 0.",
    )
    parser.add_argument(
        "--min-primitive-row-rate",
        type=float,
        default=0.0,
        help="Require this fraction of accepted rows to have at least --min-primitive-tags-per-row tags. Disabled at 0.",
    )
    parser.add_argument(
        "--min-distinct-primitives",
        type=int,
        default=0,
        help="Require at least this many distinct primitive tags across accepted rows. Disabled at 0.",
    )
    parser.add_argument(
        "--min-expected-primitive-coverage",
        type=float,
        default=0.0,
        help="Require this fraction of rows with expected_primitives to include all expected tags. Disabled at 0.",
    )
    parser.add_argument(
        "--min-analysis-words",
        type=int,
        default=0,
        help="Require accepted rows to have at least this many non-tag analysis words. Disabled at 0.",
    )
    parser.add_argument(
        "--min-analysis-word-row-rate",
        type=float,
        default=0.0,
        help="Require this fraction of accepted rows to meet --min-analysis-words. Disabled at 0.",
    )
    parser.add_argument(
        "--min-specific-analysis-rate",
        type=float,
        default=0.0,
        help=(
            "Require this fraction of accepted rows to include row-specific source/reference tokens "
            "or translation-specific technical terms. Disabled at 0."
        ),
    )
    parser.add_argument(
        "--min-audited-row-rate",
        type=float,
        default=0.0,
        help="Require this fraction of accepted rows to include a frontier_audit object. Disabled at 0.",
    )
    parser.add_argument(
        "--min-audit-pass-rate",
        type=float,
        default=0.0,
        help="Require this fraction of accepted rows to have frontier_audit.pass=true. Disabled at 0.",
    )
    parser.add_argument(
        "--min-avg-audit-score",
        type=float,
        default=0.0,
        help="Require average frontier_audit.score across audited rows. Disabled at 0.",
    )
    parser.add_argument(
        "--min-reference-final-match-rate",
        type=float,
        default=0.0,
        help=(
            "Require this fraction of accepted rows to have Traduccion final exactly match the reviewed "
            "reference after whitespace normalization. Disabled at 0."
        ),
    )
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


def strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    ).lower()


def tokens(text: str) -> list[str]:
    normalized = strip_accents(text)
    return [
        token
        for token in re.findall(r"[a-záéíóúüñ]+", normalized, flags=re.IGNORECASE)
        if token and token not in TAG_WORDS and token not in STOPWORDS
    ]


def analysis_text(record: dict[str, Any]) -> str:
    text = str(record.get("frontier_analysis") or "")
    if text:
        return text
    target = str(record.get("target") or "")
    match = re.search(
        r"analisis de traduccion\s*:\s*(.*?)(?:traduccion final\s*:|autoevaluacion\s*:|puntaje\s*:|$)",
        target,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return target


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def final_translation_text(record: dict[str, Any]) -> str:
    target = str(record.get("target") or "")
    match = re.search(
        r"traduccion final\s*:\s*(.*?)(?:autoevaluacion\s*:|puntaje\s*:|$)",
        target,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return normalize_text(match.group(1))


def final_matches_reference(record: dict[str, Any]) -> bool:
    reference = normalize_text(str(record.get("reference") or ""))
    final_translation = final_translation_text(record)
    return bool(reference) and final_translation == reference


def is_specific_analysis(record: dict[str, Any], analysis_tokens: set[str]) -> bool:
    if analysis_tokens.intersection(TRANSLATION_SPECIFIC_TERMS):
        return True
    source_reference_tokens = {
        token
        for token in tokens(f"{record.get('source') or ''} {record.get('reference') or ''}")
        if len(token) >= 4
    }
    return bool(analysis_tokens.intersection(source_reference_tokens))


def primitive_counts(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "primitive_rows": 0,
            "primitive_tag_total": 0,
            "primitive_tag_counts": {},
            "primitive_row_tag_counts": [],
            "analysis_word_counts": [],
            "specific_analysis_rows": 0,
            "audited_rows": 0,
            "audit_pass_rows": 0,
            "audit_scores": [],
            "reference_final_match_rows": 0,
        }
    tag_counts: collections.Counter[str] = collections.Counter()
    missing_expected_counts: collections.Counter[str] = collections.Counter()
    row_tag_counts: list[int] = []
    analysis_word_counts: list[int] = []
    specific_analysis_rows = 0
    expected_rows = 0
    expected_covered_rows = 0
    audited_rows = 0
    audit_pass_rows = 0
    audit_scores: list[float] = []
    reference_final_match_rows = 0
    for record in iter_jsonl(path):
        text = analysis_text(record)
        present = [tag for tag in PRIMITIVES if tag in text]
        row_tag_counts.append(len(present))
        tag_counts.update(present)
        row_analysis_tokens = set(tokens(text))
        analysis_word_counts.append(len(row_analysis_tokens))
        specific_analysis_rows += int(is_specific_analysis(record, row_analysis_tokens))
        expected = record.get("expected_primitives")
        if isinstance(expected, list) and expected:
            expected_rows += 1
            missing = [str(tag) for tag in expected if str(tag) not in text]
            if missing:
                missing_expected_counts.update(missing)
            else:
                expected_covered_rows += 1
        audit = record.get("frontier_audit")
        if isinstance(audit, dict):
            audited_rows += 1
            audit_pass_rows += int(bool(audit.get("pass")))
            try:
                audit_scores.append(float(audit.get("score", 0.0)))
            except (TypeError, ValueError):
                pass
        reference_final_match_rows += int(final_matches_reference(record))
    return {
        "primitive_rows": len(row_tag_counts),
        "primitive_tag_total": sum(row_tag_counts),
        "primitive_tag_counts": dict(tag_counts),
        "primitive_row_tag_counts": row_tag_counts,
        "analysis_word_counts": analysis_word_counts,
        "specific_analysis_rows": specific_analysis_rows,
        "expected_primitive_rows": expected_rows,
        "expected_primitive_covered_rows": expected_covered_rows,
        "missing_expected_primitive_counts": dict(missing_expected_counts),
        "audited_rows": audited_rows,
        "audit_pass_rows": audit_pass_rows,
        "audit_scores": audit_scores,
        "reference_final_match_rows": reference_final_match_rows,
    }


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


def gate_metrics(
    counts: dict[str, int],
    primitives: dict[str, Any] | None = None,
    min_tags_per_row: int = 0,
    min_analysis_words: int = 0,
) -> dict[str, float]:
    attempted = counts["written"] + counts["failed"]
    accept_rate = counts["written"] / max(1, attempted)
    primitives = primitives or {}
    primitive_rows = int(primitives.get("primitive_rows", 0))
    row_tag_counts = list(primitives.get("primitive_row_tag_counts", []))
    analysis_word_counts = list(primitives.get("analysis_word_counts", []))
    primitive_tag_total = int(primitives.get("primitive_tag_total", 0))
    primitive_tag_counts = dict(primitives.get("primitive_tag_counts", {}))
    specific_analysis_rows = int(primitives.get("specific_analysis_rows", 0))
    expected_rows = int(primitives.get("expected_primitive_rows", 0))
    expected_covered_rows = int(primitives.get("expected_primitive_covered_rows", 0))
    audited_rows = int(primitives.get("audited_rows", 0))
    audit_pass_rows = int(primitives.get("audit_pass_rows", 0))
    audit_scores = list(primitives.get("audit_scores", []))
    reference_final_match_rows = int(primitives.get("reference_final_match_rows", 0))
    metrics = {
        "written_rows": float(counts["written"]),
        "failed_rows": float(counts["failed"]),
        "requested_rows": float(counts["requested"]),
        "attempted_rows": float(attempted),
        "accept_rate": accept_rate,
        "primitive_rows": float(primitive_rows),
        "avg_primitive_tags": primitive_tag_total / max(1, primitive_rows),
        "distinct_primitives": float(len(primitive_tag_counts)),
        "expected_primitive_rows": float(expected_rows),
        "expected_primitive_covered_rows": float(expected_covered_rows),
        "expected_primitive_coverage": expected_covered_rows / max(1, expected_rows),
        "avg_analysis_words": sum(analysis_word_counts) / max(1, len(analysis_word_counts)),
        "specific_analysis_rate": specific_analysis_rows / max(1, primitive_rows),
        "audited_rows": float(audited_rows),
        "audit_pass_rows": float(audit_pass_rows),
        "audited_row_rate": audited_rows / max(1, primitive_rows),
        "audit_pass_rate": audit_pass_rows / max(1, primitive_rows),
        "avg_audit_score": sum(audit_scores) / max(1, len(audit_scores)),
        "reference_final_match_rows": float(reference_final_match_rows),
        "reference_final_match_rate": reference_final_match_rows / max(1, primitive_rows),
    }
    if min_tags_per_row > 0:
        rows_with_min_tags = sum(1 for count in row_tag_counts if count >= min_tags_per_row)
        metrics["primitive_row_rate"] = rows_with_min_tags / max(1, primitive_rows)
    else:
        metrics["primitive_row_rate"] = 0.0
    if min_analysis_words > 0:
        rows_with_min_analysis_words = sum(1 for count in analysis_word_counts if count >= min_analysis_words)
        metrics["analysis_word_row_rate"] = rows_with_min_analysis_words / max(1, len(analysis_word_counts))
    else:
        metrics["analysis_word_row_rate"] = 0.0
    for tag in PRIMITIVES:
        metrics[f"primitive_{tag.strip('[]').lower()}_rows"] = float(primitive_tag_counts.get(tag, 0))
    return metrics


def check_gate(
    metrics: dict[str, float],
    min_written_rows: int,
    min_accept_rate: float,
    min_primitive_tags_per_row: int = 0,
    min_primitive_row_rate: float = 0.0,
    min_distinct_primitives: int = 0,
    min_expected_primitive_coverage: float = 0.0,
    min_analysis_words: int = 0,
    min_analysis_word_row_rate: float = 0.0,
    min_specific_analysis_rate: float = 0.0,
    min_audited_row_rate: float = 0.0,
    min_audit_pass_rate: float = 0.0,
    min_avg_audit_score: float = 0.0,
    min_reference_final_match_rate: float = 0.0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics["written_rows"] < min_written_rows:
        reasons.append(f"written_rows {metrics['written_rows']:.0f} < {min_written_rows}")
    if metrics["accept_rate"] < min_accept_rate:
        reasons.append(f"accept_rate {metrics['accept_rate']:.4f} < {min_accept_rate:.4f}")
    if min_primitive_tags_per_row > 0 and metrics["avg_primitive_tags"] < min_primitive_tags_per_row:
        reasons.append(
            f"avg_primitive_tags {metrics['avg_primitive_tags']:.4f} < {min_primitive_tags_per_row:.4f}"
        )
    if min_primitive_row_rate > 0 and metrics["primitive_row_rate"] < min_primitive_row_rate:
        reasons.append(f"primitive_row_rate {metrics['primitive_row_rate']:.4f} < {min_primitive_row_rate:.4f}")
    if min_distinct_primitives > 0 and metrics["distinct_primitives"] < min_distinct_primitives:
        reasons.append(f"distinct_primitives {metrics['distinct_primitives']:.0f} < {min_distinct_primitives}")
    if min_expected_primitive_coverage > 0 and metrics["expected_primitive_coverage"] < min_expected_primitive_coverage:
        reasons.append(
            "expected_primitive_coverage "
            f"{metrics['expected_primitive_coverage']:.4f} < {min_expected_primitive_coverage:.4f}"
        )
    if min_analysis_words > 0 and metrics["avg_analysis_words"] < min_analysis_words:
        reasons.append(f"avg_analysis_words {metrics['avg_analysis_words']:.4f} < {min_analysis_words:.4f}")
    if min_analysis_word_row_rate > 0 and metrics["analysis_word_row_rate"] < min_analysis_word_row_rate:
        reasons.append(
            f"analysis_word_row_rate {metrics['analysis_word_row_rate']:.4f} < {min_analysis_word_row_rate:.4f}"
        )
    if min_specific_analysis_rate > 0 and metrics["specific_analysis_rate"] < min_specific_analysis_rate:
        reasons.append(
            f"specific_analysis_rate {metrics['specific_analysis_rate']:.4f} < {min_specific_analysis_rate:.4f}"
        )
    if min_audited_row_rate > 0 and metrics["audited_row_rate"] < min_audited_row_rate:
        reasons.append(f"audited_row_rate {metrics['audited_row_rate']:.4f} < {min_audited_row_rate:.4f}")
    if min_audit_pass_rate > 0 and metrics["audit_pass_rate"] < min_audit_pass_rate:
        reasons.append(f"audit_pass_rate {metrics['audit_pass_rate']:.4f} < {min_audit_pass_rate:.4f}")
    if min_avg_audit_score > 0 and metrics["avg_audit_score"] < min_avg_audit_score:
        reasons.append(f"avg_audit_score {metrics['avg_audit_score']:.4f} < {min_avg_audit_score:.4f}")
    if min_reference_final_match_rate > 0 and metrics["reference_final_match_rate"] < min_reference_final_match_rate:
        reasons.append(
            "reference_final_match_rate "
            f"{metrics['reference_final_match_rate']:.4f} < {min_reference_final_match_rate:.4f}"
        )
    return not reasons, reasons


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = gate_metrics(
        load_counts(args),
        primitive_counts(args.output_jsonl),
        min_tags_per_row=args.min_primitive_tags_per_row,
        min_analysis_words=args.min_analysis_words,
    )
    passed, reasons = check_gate(
        metrics,
        args.min_written_rows,
        args.min_accept_rate,
        args.min_primitive_tags_per_row,
        args.min_primitive_row_rate,
        args.min_distinct_primitives,
        args.min_expected_primitive_coverage,
        args.min_analysis_words,
        args.min_analysis_word_row_rate,
        args.min_specific_analysis_rate,
        args.min_audited_row_rate,
        args.min_audit_pass_rate,
        args.min_avg_audit_score,
        args.min_reference_final_match_rate,
    )
    report = {
        **metrics,
        "min_written_rows": args.min_written_rows,
        "min_accept_rate": args.min_accept_rate,
        "min_primitive_tags_per_row": args.min_primitive_tags_per_row,
        "min_primitive_row_rate": args.min_primitive_row_rate,
        "min_distinct_primitives": args.min_distinct_primitives,
        "min_expected_primitive_coverage": args.min_expected_primitive_coverage,
        "min_analysis_words": args.min_analysis_words,
        "min_analysis_word_row_rate": args.min_analysis_word_row_rate,
        "min_specific_analysis_rate": args.min_specific_analysis_rate,
        "min_audited_row_rate": args.min_audited_row_rate,
        "min_audit_pass_rate": args.min_audit_pass_rate,
        "min_avg_audit_score": args.min_avg_audit_score,
        "min_reference_final_match_rate": args.min_reference_final_match_rate,
        "passed": passed,
        "reasons": reasons,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
