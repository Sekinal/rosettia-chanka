"""Train a reference-free ensemble over candidate selector scores.

The ensemble combines already deployable scorers, such as the numeric feature
reranker, the text-aware hashed reranker, and MBR consensus. References are
used only to fit the blend weights on training candidate pools.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_feature_candidate_reranker as feature_reranker
from scripts import train_text_candidate_reranker as text_reranker


SIGNAL_NAMES = [
    "feature_score",
    "text_score",
    "mbr_score",
    "mbr_rank",
    "length_consensus",
    "duplicate_rate",
    "candidate_index_fraction",
]


@dataclass(frozen=True)
class EnsembleRow:
    row: feature_reranker.CandidateFeatures
    signals: dict[str, float]


@dataclass(frozen=True)
class EnsembleModel:
    signal_names: list[str]
    weights: dict[str, float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--eval-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--feature-weights-json", type=Path, required=True)
    parser.add_argument("--text-model-json", type=Path, required=True)
    parser.add_argument("--ensemble-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="score_ensemble")
    parser.add_argument("--search-iterations", type=int, default=5000)
    parser.add_argument("--initial-noise", type=float, default=0.45)
    parser.add_argument("--min-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def load_feature_model(path: Path) -> feature_reranker.FeatureModel:
    return feature_reranker.model_from_json(json.loads(path.read_text()))


def load_text_model(path: Path) -> text_reranker.TextRankerModel:
    return text_reranker.model_from_json(json.loads(path.read_text()))


def zscore(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 1.0
    if std <= 1e-9:
        std = 1.0
    return [(value - mean) / std for value in values]


def build_ensemble_groups(
    jsonl_paths: Sequence[Path],
    feature_model: feature_reranker.FeatureModel,
    text_model: text_reranker.TextRankerModel,
) -> list[list[EnsembleRow]]:
    raw_groups = feature_reranker.load_groups(jsonl_paths)
    feature_groups = feature_reranker.featurize_groups(raw_groups)
    sparse_groups = text_reranker.sparse_groups_for_model(feature_groups, text_model)
    ensemble_groups: list[list[EnsembleRow]] = []
    for feature_group, sparse_group in zip(feature_groups, sparse_groups, strict=True):
        raw_signals: dict[str, list[float]] = {
            "feature_score": [feature_reranker.candidate_score(row, feature_model) for row in feature_group],
            "text_score": [text_reranker.model_score(row.features, text_model.weights) for row in sparse_group],
            "mbr_score": [row.raw["mbr_score"] for row in feature_group],
            "mbr_rank": [row.raw["mbr_rank"] for row in feature_group],
            "length_consensus": [row.raw["length_consensus"] for row in feature_group],
            "duplicate_rate": [row.raw["duplicate_rate"] for row in feature_group],
            "candidate_index_fraction": [row.raw["candidate_index_fraction"] for row in feature_group],
        }
        normalized = {name: zscore(values) for name, values in raw_signals.items()}
        group_rows: list[EnsembleRow] = []
        for index, row in enumerate(feature_group):
            group_rows.append(
                EnsembleRow(
                    row=row,
                    signals={name: normalized[name][index] for name in SIGNAL_NAMES},
                )
            )
        ensemble_groups.append(group_rows)
    return ensemble_groups


def ensemble_score(row: EnsembleRow, model: EnsembleModel) -> float:
    return sum(model.weights.get(name, 0.0) * row.signals[name] for name in model.signal_names)


def select_ensemble(groups: Sequence[Sequence[EnsembleRow]], model: EnsembleModel) -> list[EnsembleRow]:
    selected: list[EnsembleRow] = []
    for group in groups:
        if not group:
            continue
        selected.append(
            max(
                group,
                key=lambda row: (
                    ensemble_score(row, model),
                    -row.row.candidate.candidate_index,
                ),
            )
        )
    return selected


def mean_oracle_objective(groups: Sequence[Sequence[EnsembleRow]], model: EnsembleModel) -> float:
    selected = select_ensemble(groups, model)
    if not selected:
        return 0.0
    return statistics.fmean(row.row.oracle_score for row in selected)


def initial_weights() -> dict[str, float]:
    return {
        "feature_score": 0.6,
        "text_score": 0.8,
        "mbr_score": 0.15,
        "mbr_rank": 0.05,
        "length_consensus": 0.05,
        "duplicate_rate": 0.0,
        "candidate_index_fraction": 0.0,
    }


def train_ensemble_model(
    train_groups: Sequence[Sequence[EnsembleRow]],
    seed: int,
    search_iterations: int,
    initial_noise: float,
    min_noise: float,
) -> tuple[EnsembleModel, dict[str, Any]]:
    signal_names = SIGNAL_NAMES[:]
    best = EnsembleModel(signal_names, initial_weights())
    best_objective = mean_oracle_objective(train_groups, best)
    rng = random.Random(seed)
    accepted = 0

    for step in range(max(0, search_iterations)):
        progress = step / max(1, search_iterations - 1)
        noise = max(min_noise, initial_noise * (1.0 - progress))
        weights = dict(best.weights)
        if rng.random() < 0.8:
            signal = rng.choice(signal_names)
            weights[signal] = weights.get(signal, 0.0) + rng.gauss(0.0, noise)
        else:
            for signal in rng.sample(signal_names, k=min(3, len(signal_names))):
                weights[signal] = weights.get(signal, 0.0) + rng.gauss(0.0, noise * 0.5)
        proposal = EnsembleModel(signal_names, weights)
        objective = mean_oracle_objective(train_groups, proposal)
        if objective > best_objective:
            best = proposal
            best_objective = objective
            accepted += 1

    diagnostics = {
        "training_objective": "score_ensemble_hillclimb",
        "accepted_updates": accepted,
        "initial_objective": mean_oracle_objective(train_groups, EnsembleModel(signal_names, initial_weights())),
        "best_objective": best_objective,
        "search_iterations": search_iterations,
        "train_groups": len(train_groups),
        "train_candidates": sum(len(group) for group in train_groups),
    }
    return best, diagnostics


def model_to_json(model: EnsembleModel, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal_names": model.signal_names,
        "weights": model.weights,
        "diagnostics": diagnostics,
    }


def model_from_json(payload: dict[str, Any]) -> EnsembleModel:
    return EnsembleModel(
        signal_names=list(payload["signal_names"]),
        weights={name: float(value) for name, value in payload["weights"].items()},
    )


def evaluate_groups(
    groups: Sequence[Sequence[EnsembleRow]],
    model: EnsembleModel,
    predictions_jsonl: Path,
    output_dir: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    feature_groups = [[row.row for row in group] for group in groups]
    raw_groups = [[row.candidate for row in group] for group in feature_groups]
    total_candidates = sum(len(group) for group in raw_groups)
    first = [group[0] for group in feature_groups if group]
    ensemble = [row.row for row in select_ensemble(groups, model)]
    mbr_selected = [
        feature_reranker.CandidateFeatures(candidate, {}, oracle_rerank.candidate_oracle_score(candidate))
        for candidate in feature_reranker.mbr.select_mbr(raw_groups)
    ]
    oracle_selected = [
        feature_reranker.CandidateFeatures(candidate, {}, oracle_rerank.candidate_oracle_score(candidate))
        for candidate in oracle_rerank.select_oracle(raw_groups)
    ]
    selections = [
        ("first", first),
        ("ensemble", ensemble),
        ("mbr", mbr_selected),
        ("oracle", oracle_selected),
    ]
    records = [
        feature_reranker.metrics_for_selected(selected_rows, method, predictions_jsonl, total_candidates)
        for method, selected_rows in selections
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for (method, selected_rows), record in zip(selections, records, strict=True):
        oracle_rerank.write_predictions(
            output_dir / f"{prefix}_{method}_predictions.jsonl",
            [row.candidate for row in selected_rows],
        )
        oracle_rerank.write_metrics(output_dir / f"{prefix}_{method}_metrics.json", record)
    oracle_rerank.write_metrics(output_dir / f"{prefix}_summary.json", {"records": records})
    feature_reranker.write_summary(output_dir / f"{prefix}_summary.md", records)
    return records


def main() -> None:
    args = parse_args()
    feature_model = load_feature_model(args.feature_weights_json)
    text_model = load_text_model(args.text_model_json)
    eval_groups = build_ensemble_groups(args.eval_jsonl, feature_model, text_model)
    if args.ensemble_json:
        payload = json.loads(args.ensemble_json.read_text())
        model = model_from_json(payload)
        diagnostics = dict(payload.get("diagnostics", {}))
    else:
        if not args.train_jsonl:
            raise ValueError("--train-jsonl is required unless --ensemble-json is supplied")
        train_groups = build_ensemble_groups(args.train_jsonl, feature_model, text_model)
        model, diagnostics = train_ensemble_model(
            train_groups,
            seed=args.seed,
            search_iterations=args.search_iterations,
            initial_noise=args.initial_noise,
            min_noise=args.min_noise,
        )

    diagnostics["eval_groups"] = len(eval_groups)
    diagnostics["eval_candidates"] = sum(len(group) for group in eval_groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.prefix}_ensemble.json"
    model_path.write_text(json.dumps(model_to_json(model, diagnostics), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    records = evaluate_groups(eval_groups, model, args.eval_jsonl[0], args.output_dir, args.prefix)
    print(
        json.dumps(
            {
                "ensemble_json": str(model_path),
                "diagnostics": diagnostics,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
