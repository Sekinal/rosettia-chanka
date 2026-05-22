"""Train a deployable text-aware candidate reranker.

This is a lightweight reference-free selector: references are used only during
training to identify oracle winners. At inference/evaluation time the model
scores only the Spanish source, the Chanka candidate, and group-level features.

The model is an online pairwise logistic ranker over hashed sparse features:
manual reranker features, candidate word/character n-grams, and source-token to
candidate-token cross features. It intentionally avoids sklearn/xgboost so it
can run in the current project environment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_feature_candidate_reranker as feature_reranker
from scripts import train_gspo_chanka_unsloth as gspo


TOKEN_RE = re.compile(r"[\wáéíóúüñÁÉÍÓÚÜÑʼ’'-]+", re.UNICODE)


@dataclass(frozen=True)
class SparseRow:
    row: feature_reranker.CandidateFeatures
    features: dict[int, float]


@dataclass(frozen=True)
class TextRankerModel:
    hash_size: int
    feature_names: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    weights: dict[int, float]
    config: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--eval-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--model-json", type=Path, default=None, help="Existing text-ranker model to evaluate.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="text_reranker")
    parser.add_argument("--hash-size", type=int, default=1 << 20)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--margin", type=float, default=0.05, help="Skip negatives too close to the oracle winner.")
    parser.add_argument("--max-negatives-per-group", type=int, default=12)
    parser.add_argument("--char-ngram-min", type=int, default=3)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--max-source-tokens", type=int, default=8)
    parser.add_argument("--max-candidate-tokens", type=int, default=14)
    parser.add_argument("--include-manual-features", action="store_true", default=True)
    parser.add_argument("--no-include-manual-features", action="store_false", dest="include_manual_features")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def stable_hash(text: str, hash_size: int) -> int:
    return zlib.crc32(text.encode("utf-8")) % hash_size


def add_feature(features: dict[int, float], key: str, value: float, hash_size: int) -> None:
    if value == 0.0:
        return
    index = stable_hash(key, hash_size)
    features[index] = features.get(index, 0.0) + value


def text_tokens(text: str) -> list[str]:
    normalized = gspo.normalize_text(text).lower()
    return [match.group(0).strip("-'ʼ’") for match in TOKEN_RE.finditer(normalized) if match.group(0).strip("-'ʼ’")]


def content_tokens(text: str, limit: int) -> list[str]:
    tokens = [
        token
        for token in text_tokens(text)
        if len(token) >= 3 and token not in gspo.SPANISH_STOPWORDS
    ]
    return tokens[:limit]


def char_ngrams(text: str, min_n: int, max_n: int) -> Iterable[str]:
    normalized = gspo.normalize_text(text).lower()
    padded = f" {normalized} "
    for size in range(min_n, max_n + 1):
        if len(padded) < size:
            continue
        for index in range(0, len(padded) - size + 1):
            ngram = padded[index : index + size]
            if ngram.strip():
                yield ngram


def word_ngrams(tokens: Sequence[str], max_n: int) -> Iterable[str]:
    for size in range(1, max_n + 1):
        if len(tokens) < size:
            continue
        for index in range(0, len(tokens) - size + 1):
            yield " ".join(tokens[index : index + size])


def sparse_features_for_row(
    row: feature_reranker.CandidateFeatures,
    model: TextRankerModel,
) -> dict[int, float]:
    config = model.config
    candidate = row.candidate
    features: dict[int, float] = {}

    if config.get("include_manual_features", True):
        for name in model.feature_names:
            value = (row.raw[name] - model.means[name]) / model.stds[name]
            add_feature(features, f"manual:{name}", value, model.hash_size)

    prediction_tokens = text_tokens(candidate.prediction)
    source_tokens = content_tokens(candidate.source, int(config["max_source_tokens"]))
    candidate_tokens = [
        token
        for token in prediction_tokens
        if len(token) >= 2
    ][: int(config["max_candidate_tokens"])]

    for token in candidate_tokens:
        add_feature(features, f"cand_tok:{token}", 1.0, model.hash_size)
    for ngram in word_ngrams(candidate_tokens, int(config["word_ngram_max"])):
        add_feature(features, f"cand_wng:{ngram}", 1.0, model.hash_size)
    for ngram in char_ngrams(
        candidate.prediction,
        int(config["char_ngram_min"]),
        int(config["char_ngram_max"]),
    ):
        add_feature(features, f"cand_cng:{ngram}", 0.25, model.hash_size)

    for source_token in source_tokens:
        add_feature(features, f"src_tok:{source_token}", 0.5, model.hash_size)
        for candidate_token in candidate_tokens:
            add_feature(features, f"src_cand:{source_token}->{candidate_token}", 1.0, model.hash_size)

    return features


def model_score(features: dict[int, float], weights: dict[int, float]) -> float:
    return sum(weights.get(index, 0.0) * value for index, value in features.items())


def subtract_features(left: dict[int, float], right: dict[int, float]) -> dict[int, float]:
    diff = dict(left)
    for index, value in right.items():
        diff[index] = diff.get(index, 0.0) - value
        if abs(diff[index]) < 1e-12:
            del diff[index]
    return diff


def sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def load_feature_groups(paths: Sequence[Path]) -> list[list[feature_reranker.CandidateFeatures]]:
    groups = feature_reranker.load_groups(paths)
    return feature_reranker.featurize_groups(groups)


def model_shell(
    train_groups: Sequence[Sequence[feature_reranker.CandidateFeatures]],
    args: argparse.Namespace,
) -> TextRankerModel:
    feature_names = feature_reranker.BASE_FEATURE_NAMES[:]
    means, stds = feature_reranker.normalization_stats(train_groups, feature_names)
    config = {
        "char_ngram_min": args.char_ngram_min,
        "char_ngram_max": args.char_ngram_max,
        "word_ngram_max": args.word_ngram_max,
        "max_source_tokens": args.max_source_tokens,
        "max_candidate_tokens": args.max_candidate_tokens,
        "include_manual_features": args.include_manual_features,
    }
    return TextRankerModel(
        hash_size=args.hash_size,
        feature_names=feature_names,
        means=means,
        stds=stds,
        weights={},
        config=config,
    )


def sparse_groups_for_model(
    groups: Sequence[Sequence[feature_reranker.CandidateFeatures]],
    model: TextRankerModel,
) -> list[list[SparseRow]]:
    return [
        [SparseRow(row, sparse_features_for_row(row, model)) for row in group]
        for group in groups
        if group
    ]


def oracle_best_index(group: Sequence[SparseRow]) -> int:
    return max(
        range(len(group)),
        key=lambda index: (
            group[index].row.oracle_score,
            -group[index].row.candidate.candidate_index,
        ),
    )


def negative_indices(
    group: Sequence[SparseRow],
    best_index: int,
    margin: float,
    limit: int,
    rng: random.Random,
) -> list[int]:
    best_score = group[best_index].row.oracle_score
    negatives = [
        index
        for index, sparse_row in enumerate(group)
        if index != best_index and sparse_row.row.oracle_score <= best_score - margin
    ]
    negatives.sort(key=lambda index: group[index].row.oracle_score, reverse=True)
    if limit > 0 and len(negatives) > limit:
        hard = negatives[: max(1, limit // 2)]
        tail = negatives[max(1, limit // 2) :]
        rng.shuffle(tail)
        negatives = hard + tail[: limit - len(hard)]
    return negatives


def train_text_ranker(
    train_groups: Sequence[Sequence[feature_reranker.CandidateFeatures]],
    args: argparse.Namespace,
) -> tuple[TextRankerModel, dict[str, Any]]:
    model = model_shell(train_groups, args)
    sparse_groups = sparse_groups_for_model(train_groups, model)
    weights: dict[int, float] = {}
    rng = random.Random(args.seed)
    updates = 0
    skipped_pairs = 0

    for _epoch in range(max(0, args.epochs)):
        rng.shuffle(sparse_groups)
        for group in sparse_groups:
            best_index = oracle_best_index(group)
            for negative_index in negative_indices(
                group,
                best_index,
                args.margin,
                args.max_negatives_per_group,
                rng,
            ):
                diff = subtract_features(group[best_index].features, group[negative_index].features)
                score = model_score(diff, weights)
                gradient_scale = 1.0 - sigmoid(score)
                if gradient_scale < 1e-5:
                    skipped_pairs += 1
                    continue
                touched = list(diff)
                for index in touched:
                    weights[index] = weights.get(index, 0.0) * (1.0 - args.learning_rate * args.l2)
                for index, value in diff.items():
                    weights[index] = weights.get(index, 0.0) + args.learning_rate * gradient_scale * value
                    if abs(weights[index]) < 1e-10:
                        del weights[index]
                updates += 1

    trained = TextRankerModel(
        hash_size=model.hash_size,
        feature_names=model.feature_names,
        means=model.means,
        stds=model.stds,
        weights=weights,
        config=model.config,
    )
    diagnostics = {
        "training_objective": "pairwise_hashed_text_logistic",
        "train_groups": len(sparse_groups),
        "train_candidates": sum(len(group) for group in sparse_groups),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "margin": args.margin,
        "max_negatives_per_group": args.max_negatives_per_group,
        "updates": updates,
        "skipped_pairs": skipped_pairs,
        "nonzero_weights": len(weights),
        "train_mean_oracle": mean_oracle_for_sparse_selection(sparse_groups, trained),
    }
    return trained, diagnostics


def select_sparse(groups: Sequence[Sequence[SparseRow]], model: TextRankerModel) -> list[SparseRow]:
    selected: list[SparseRow] = []
    for group in groups:
        if not group:
            continue
        selected.append(
            max(
                group,
                key=lambda sparse_row: (
                    model_score(sparse_row.features, model.weights),
                    -sparse_row.row.candidate.candidate_index,
                ),
            )
        )
    return selected


def mean_oracle_for_sparse_selection(groups: Sequence[Sequence[SparseRow]], model: TextRankerModel) -> float:
    selected = select_sparse(groups, model)
    if not selected:
        return 0.0
    return sum(row.row.oracle_score for row in selected) / len(selected)


def model_to_json(model: TextRankerModel, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "hash_size": model.hash_size,
        "feature_names": model.feature_names,
        "means": model.means,
        "stds": model.stds,
        "weights": [[index, value] for index, value in sorted(model.weights.items())],
        "config": model.config,
        "diagnostics": diagnostics,
    }


def model_from_json(payload: dict[str, Any]) -> TextRankerModel:
    return TextRankerModel(
        hash_size=int(payload["hash_size"]),
        feature_names=list(payload["feature_names"]),
        means={key: float(value) for key, value in payload["means"].items()},
        stds={key: float(value) for key, value in payload["stds"].items()},
        weights={int(index): float(value) for index, value in payload["weights"]},
        config=dict(payload["config"]),
    )


def evaluate_sparse_groups(
    groups: Sequence[Sequence[SparseRow]],
    model: TextRankerModel,
    predictions_jsonl: Path,
    output_dir: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    feature_groups = [[sparse_row.row for sparse_row in group] for group in groups]
    raw_groups = [[row.candidate for row in group] for group in feature_groups]
    total_candidates = sum(len(group) for group in raw_groups)
    first = [group[0] for group in feature_groups if group]
    text_selected = [sparse_row.row for sparse_row in select_sparse(groups, model)]
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
        ("text", text_selected),
        ("mbr", mbr_selected),
        ("oracle", oracle_selected),
    ]
    records = [
        feature_reranker.metrics_for_selected(selected_rows, method, predictions_jsonl, total_candidates)
        for method, selected_rows in selections
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for (method, selected_rows), record in zip(selections, records, strict=True):
        selected = [row.candidate for row in selected_rows]
        oracle_rerank.write_predictions(output_dir / f"{prefix}_{method}_predictions.jsonl", selected)
        oracle_rerank.write_metrics(output_dir / f"{prefix}_{method}_metrics.json", record)
    oracle_rerank.write_metrics(output_dir / f"{prefix}_summary.json", {"records": records})
    feature_reranker.write_summary(output_dir / f"{prefix}_summary.md", records)
    return records


def main() -> None:
    args = parse_args()
    eval_groups = load_feature_groups(args.eval_jsonl)
    diagnostics: dict[str, Any]
    if args.model_json:
        payload = json.loads(args.model_json.read_text())
        model = model_from_json(payload)
        diagnostics = dict(payload.get("diagnostics", {}))
    else:
        if not args.train_jsonl:
            raise ValueError("--train-jsonl is required unless --model-json is supplied")
        train_groups = load_feature_groups(args.train_jsonl)
        model, diagnostics = train_text_ranker(train_groups, args)

    sparse_eval_groups = sparse_groups_for_model(eval_groups, model)
    diagnostics["eval_groups"] = len(sparse_eval_groups)
    diagnostics["eval_candidates"] = sum(len(group) for group in sparse_eval_groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.prefix}_model.json"
    model_path.write_text(json.dumps(model_to_json(model, diagnostics), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    records = evaluate_sparse_groups(
        sparse_eval_groups,
        model,
        predictions_jsonl=args.eval_jsonl[0],
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    result = {
        "model_json": str(model_path),
        "diagnostics": diagnostics,
        "records": records,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
