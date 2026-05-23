"""Generate Chanka translations with K-sampling plus score-ensemble reranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_gspo_checkpoint as evaluate
from scripts import generate_feature_reranked_translations as feature_generate
from scripts import train_feature_candidate_reranker as feature_reranker
from scripts import train_gspo_chanka_unsloth as gspo
from scripts import train_score_ensemble_reranker as ensemble_reranker
from scripts import train_text_candidate_reranker as text_reranker


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--feature-weights-json", type=Path, required=True)
    parser.add_argument("--text-model-json", type=Path, required=True)
    parser.add_argument("--ensemble-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, default=None)
    parser.add_argument("--source", action="append", default=[], help="Spanish source text. Repeatable.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--strip-chat-artifacts", action="store_true")
    parser.add_argument("--terminology-file", default=None)
    parser.add_argument("--terminology-top-k", type=int, default=1)
    parser.add_argument("--terminology-min-source-chars", type=int, default=3)
    return parser.parse_args(argv)


def select_translations(
    groups: Sequence[Sequence[feature_generate.oracle_rerank.Candidate]],
    feature_model: feature_reranker.FeatureModel,
    text_model: text_reranker.TextRankerModel,
    ensemble_model: ensemble_reranker.EnsembleModel,
) -> list[feature_generate.oracle_rerank.Candidate]:
    feature_groups = feature_reranker.featurize_groups(groups)
    sparse_groups = text_reranker.sparse_groups_for_model(feature_groups, text_model)
    ensemble_groups: list[list[ensemble_reranker.EnsembleRow]] = []
    for feature_group, sparse_group in zip(feature_groups, sparse_groups, strict=True):
        raw_signals = {
            "feature_score": [feature_reranker.candidate_score(row, feature_model) for row in feature_group],
            "text_score": [text_reranker.model_score(row.features, text_model.weights) for row in sparse_group],
            "mbr_score": [row.raw["mbr_score"] for row in feature_group],
            "mbr_rank": [row.raw["mbr_rank"] for row in feature_group],
            "length_consensus": [row.raw["length_consensus"] for row in feature_group],
            "duplicate_rate": [row.raw["duplicate_rate"] for row in feature_group],
            "candidate_index_fraction": [row.raw["candidate_index_fraction"] for row in feature_group],
        }
        normalized = {name: ensemble_reranker.zscore(values) for name, values in raw_signals.items()}
        ensemble_groups.append(
            [
                ensemble_reranker.EnsembleRow(
                    row=row,
                    signals={name: normalized[name][index] for name in ensemble_reranker.SIGNAL_NAMES},
                )
                for index, row in enumerate(feature_group)
            ]
        )
    return [row.row.candidate for row in ensemble_reranker.select_ensemble(ensemble_groups, ensemble_model)]


def main() -> None:
    args = parse_args()
    if args.num_return_sequences > 1 and args.batch_size != 1:
        print("Using batch-size 1 is recommended for decoder-only K-sampling; continuing anyway.", file=sys.stderr)

    from unsloth import FastLanguageModel

    source_rows = feature_generate.load_source_rows(args.input_path, args.source)
    terminology_entries = (
        gspo.load_terminology_entries(args.dataset_repo, args.terminology_file, args.terminology_min_source_chars)
        if args.terminology_file
        else []
    )
    feature_model = feature_reranker.model_from_json(json.loads(args.feature_weights_json.read_text()))
    text_model = text_reranker.model_from_json(json.loads(args.text_model_json.read_text()))
    ensemble_model = ensemble_reranker.model_from_json(json.loads(args.ensemble_json.read_text()))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    FastLanguageModel.for_inference(model)

    generated_rows, predictions, _generated_terms, _generated_few_shots = evaluate.generate_predictions_with_progress(
        model=model,
        tokenizer=tokenizer,
        rows=source_rows,
        max_completion_length=args.max_completion_length,
        batch_size=args.batch_size,
        num_return_sequences=args.num_return_sequences,
        do_sample=args.num_return_sequences > 1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        progress_every=args.progress_every,
        strip_chat_artifacts=args.strip_chat_artifacts,
        terminology_entries=terminology_entries,
        terminology_top_k=args.terminology_top_k,
    )
    groups = feature_generate.candidate_groups_from_generation(generated_rows, predictions, args.num_return_sequences)
    selected = select_translations(groups, feature_model, text_model, ensemble_model)
    feature_generate.write_selected(args.output_jsonl, selected)
    if args.candidates_jsonl:
        feature_generate.write_candidates(args.candidates_jsonl, groups)
    print(json.dumps({"sources": len(source_rows), "written": len(selected)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
