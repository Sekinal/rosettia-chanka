"""Build confidence-filtered MBR pseudo-label JSONL from multi-candidate predictions."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import mbr_candidate_predictions as mbr
from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_gspo_chanka_unsloth as gspo
from scripts.summarize_gspo_canaries import selection_score


@dataclass(frozen=True)
class MbrSelection:
    candidate: oracle_rerank.Candidate
    score: float
    margin: float
    mean_peer_utility: float
    candidate_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--min-candidates", type=int, default=4)
    parser.add_argument("--min-mbr-score", type=float, default=0.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--min-mean-peer-utility", type=float, default=0.0)
    parser.add_argument("--max-source-copy-ratio", type=float, default=0.60)
    parser.add_argument("--max-spanish-leakage-penalty", type=float, default=0.25)
    parser.add_argument("--max-chat-artifact-penalty", type=float, default=0.0)
    parser.add_argument("--allow-exact-source-copy", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args(argv)


def mean_peer_utility(candidate: oracle_rerank.Candidate, group: Sequence[oracle_rerank.Candidate]) -> float:
    peers = [item for item in group if item.prediction != candidate.prediction]
    if not peers:
        peers = list(group)
    if not peers:
        return 0.0
    return sum(mbr.pairwise_utility(candidate.prediction, peer.prediction) for peer in peers) / len(peers)


def rank_group(group: Sequence[oracle_rerank.Candidate]) -> MbrSelection:
    scored = sorted(
        ((candidate, mbr.mbr_score(candidate, group)) for candidate in group),
        key=lambda item: (item[1], -item[0].candidate_index),
        reverse=True,
    )
    if not scored:
        raise ValueError("Cannot rank an empty candidate group")
    top_candidate, top_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else top_score
    return MbrSelection(
        candidate=top_candidate,
        score=top_score,
        margin=max(0.0, top_score - second_score),
        mean_peer_utility=mean_peer_utility(top_candidate, group),
        candidate_count=len(group),
    )


def passes_filters(selection: MbrSelection, args: argparse.Namespace) -> bool:
    candidate = selection.candidate
    if selection.candidate_count < args.min_candidates:
        return False
    if selection.score < args.min_mbr_score:
        return False
    if selection.margin < args.min_margin:
        return False
    if selection.mean_peer_utility < args.min_mean_peer_utility:
        return False
    if gspo.source_copy_ratio(candidate.prediction, candidate.source) > args.max_source_copy_ratio:
        return False
    if gspo.spanish_leakage_penalty(candidate.prediction) > args.max_spanish_leakage_penalty:
        return False
    if gspo.chat_artifact_penalty(candidate.prediction) > args.max_chat_artifact_penalty:
        return False
    if not args.allow_exact_source_copy and gspo.exact_source_copy(candidate.prediction, candidate.source):
        return False
    return True


def pseudo_record(selection: MbrSelection) -> dict[str, Any]:
    candidate = selection.candidate
    return {
        "source": candidate.source,
        "target": candidate.prediction,
        "prediction": candidate.prediction,
        "reference": candidate.reference,
        "source_name": candidate.source_name,
        "variant": candidate.variant,
        "label_type": "pseudo_mbr_confident",
        "candidate_index": candidate.candidate_index,
        "mbr_score": selection.score,
        "mbr_margin": selection.margin,
        "mbr_mean_peer_utility": selection.mean_peer_utility,
        "mbr_candidate_count": selection.candidate_count,
        "source_copy_ratio": gspo.source_copy_ratio(candidate.prediction, candidate.source),
        "spanish_leakage_penalty": gspo.spanish_leakage_penalty(candidate.prediction),
        "chat_artifact_penalty": gspo.chat_artifact_penalty(candidate.prediction),
    }


def build_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = oracle_rerank.load_candidates(args.predictions_jsonl)
    groups = oracle_rerank.group_candidates(candidates)
    selections = [rank_group(group) for group in groups if group]
    filtered = [selection for selection in selections if passes_filters(selection, args)]
    if args.max_rows is not None:
        filtered = filtered[: args.max_rows]
    records = [pseudo_record(selection) for selection in filtered]

    selected_candidates = [selection.candidate for selection in filtered]
    metrics: dict[str, Any] = {
        "predictions_jsonl": str(args.predictions_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "input_groups": len(groups),
        "input_candidates": len(candidates),
        "kept_rows": len(records),
        "kept_rate": len(records) / max(1, len(groups)),
        "filters": {
            "min_candidates": args.min_candidates,
            "min_mbr_score": args.min_mbr_score,
            "min_margin": args.min_margin,
            "min_mean_peer_utility": args.min_mean_peer_utility,
            "max_source_copy_ratio": args.max_source_copy_ratio,
            "max_spanish_leakage_penalty": args.max_spanish_leakage_penalty,
            "max_chat_artifact_penalty": args.max_chat_artifact_penalty,
            "allow_exact_source_copy": args.allow_exact_source_copy,
            "max_rows": args.max_rows,
        },
        "mean_mbr_score": statistics.fmean(selection.score for selection in filtered) if filtered else 0.0,
        "mean_mbr_margin": statistics.fmean(selection.margin for selection in filtered) if filtered else 0.0,
        "mean_peer_utility": statistics.fmean(selection.mean_peer_utility for selection in filtered) if filtered else 0.0,
    }
    if selected_candidates:
        quality = oracle_rerank.metrics_for_selection(selected_candidates, "confident_mbr", args.predictions_jsonl, len(candidates))
        quality["selection_score"] = selection_score(quality)
        metrics["hidden_reference_quality"] = quality
    return records, metrics


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    quality = metrics.get("hidden_reference_quality", {})
    lines = [
        "# Confident MBR Pseudo-Labels",
        "",
        f"- Input groups: `{metrics['input_groups']}`",
        f"- Kept rows: `{metrics['kept_rows']}` (`{100 * metrics['kept_rate']:.2f}%`)",
        f"- Mean MBR score: `{metrics['mean_mbr_score']:.4f}`",
        f"- Mean MBR margin: `{metrics['mean_mbr_margin']:.4f}`",
        f"- Mean peer utility: `{metrics['mean_peer_utility']:.4f}`",
        "",
    ]
    if quality:
        lines.extend(
            [
                "| Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| "
                + " | ".join(
                    [
                        f"{quality.get('selection_score', 0.0):.4f}",
                        f"{quality.get('chrf++', 0.0):.4f}",
                        f"{quality.get('bleu', 0.0):.4f}",
                        f"{quality.get('token_f1', 0.0):.4f}",
                        f"{quality.get('source_copy_ratio', 0.0):.4f}",
                        f"{quality.get('spanish_leakage_penalty', 0.0):.4f}",
                        f"{quality.get('ter', 0.0):.4f}",
                    ]
                )
                + " |",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    records, metrics = build_records(args)
    write_jsonl(args.output_jsonl, records)
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if args.summary_md:
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary_md, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
