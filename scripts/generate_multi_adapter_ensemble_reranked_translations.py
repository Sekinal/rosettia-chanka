"""Generate Chanka translations from multiple adapters, then rerank.

This is the deployable version of the current best held-out profile: generate
candidate pools from more than one model/checkpoint, dedupe them per source,
and select with a trained reference-free reranker.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_gspo_checkpoint as evaluate
from scripts import generate_ensemble_reranked_translations as ensemble_generate
from scripts import generate_feature_reranked_translations as feature_generate
from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_feature_candidate_reranker as feature_reranker
from scripts import train_gspo_chanka_unsloth as gspo
from scripts import train_score_ensemble_reranker as ensemble_reranker
from scripts import train_text_candidate_reranker as text_reranker


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, action="append", required=True)
    parser.add_argument("--text-model-json", type=Path, required=True)
    parser.add_argument("--feature-weights-json", type=Path, default=None)
    parser.add_argument("--ensemble-json", type=Path, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["text", "ensemble"],
        default="ensemble",
        help="Use the text reranker directly, or blend feature/text/MBR signals with a score ensemble.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, default=None)
    parser.add_argument("--source", action="append", default=[], help="Spanish source text. Repeatable.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        action="append",
        default=None,
        help="Samples per adapter. Repeat once per --adapter-path, or provide one value to broadcast.",
    )
    parser.add_argument("--temperature", type=float, action="append", default=None)
    parser.add_argument("--top-p", type=float, action="append", default=None)
    parser.add_argument("--top-k", type=int, action="append", default=None)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--strip-chat-artifacts", action="store_true")
    parser.add_argument("--terminology-file", default=None)
    parser.add_argument("--terminology-top-k", type=int, default=1)
    parser.add_argument("--terminology-min-source-chars", type=int, default=3)
    parser.add_argument(
        "--few-shot-top-k",
        type=int,
        default=0,
        help="Retrieve this many similar clean train examples and include them in generation prompts.",
    )
    parser.add_argument("--few-shot-max-candidates", type=int, default=128)
    parser.add_argument("--few-shot-dataset-file", default=gspo.CHANKA_FILE)
    return parser.parse_args(argv)


def broadcast(values: Sequence[object] | None, count: int, default: object) -> list[object]:
    if not values:
        return [default for _ in range(count)]
    if len(values) == 1:
        return [values[0] for _ in range(count)]
    if len(values) != count:
        raise ValueError(f"Expected either 1 or {count} values, got {len(values)}")
    return list(values)


def candidate_key(candidate: oracle_rerank.Candidate) -> tuple[str, str | None, str | None]:
    return (candidate.source, candidate.source_name, candidate.variant)


def merge_candidate_groups(
    pool_groups: Sequence[Sequence[Sequence[oracle_rerank.Candidate]]],
) -> list[list[oracle_rerank.Candidate]]:
    grouped: dict[tuple[str, str | None, str | None], list[oracle_rerank.Candidate]] = {}
    order: list[tuple[str, str | None, str | None]] = []
    seen_predictions: dict[tuple[str, str | None, str | None], set[str]] = {}

    for groups in pool_groups:
        for group in groups:
            if not group:
                continue
            key = candidate_key(group[0])
            if key not in grouped:
                grouped[key] = []
                seen_predictions[key] = set()
                order.append(key)
            for candidate in group:
                normalized = gspo.normalize_text(candidate.prediction).lower()
                if normalized in seen_predictions[key]:
                    continue
                seen_predictions[key].add(normalized)
                grouped[key].append(
                    oracle_rerank.Candidate(
                        source=candidate.source,
                        reference="",
                        prediction=gspo.normalize_text(candidate.prediction),
                        source_name=candidate.source_name,
                        variant=candidate.variant,
                        candidate_index=len(grouped[key]),
                        pool_path=candidate.pool_path,
                    )
                )

    return [grouped[key] for key in order]


def generate_groups_for_adapter(
    adapter_path: Path,
    source_rows: Sequence[dict[str, str]],
    terminology_entries: Sequence[tuple[str, str]],
    few_shot_examples: Sequence[dict[str, str]],
    args: argparse.Namespace,
    num_return_sequences: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[list[oracle_rerank.Candidate]]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
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

    try:
        generated_rows, predictions, _generated_terms, _generated_few_shots = evaluate.generate_predictions_with_progress(
            model=model,
            tokenizer=tokenizer,
            rows=source_rows,
            max_completion_length=args.max_completion_length,
            batch_size=args.batch_size,
            num_return_sequences=num_return_sequences,
            do_sample=num_return_sequences > 1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            progress_every=args.progress_every,
            strip_chat_artifacts=args.strip_chat_artifacts,
            terminology_entries=terminology_entries,
            terminology_top_k=args.terminology_top_k,
            few_shot_examples=few_shot_examples,
            few_shot_top_k=args.few_shot_top_k,
            few_shot_max_candidates=args.few_shot_max_candidates,
        )
        return feature_generate.candidate_groups_from_generation(
            generated_rows,
            predictions,
            num_return_sequences,
            pool_path=str(adapter_path),
        )
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    if args.selection_mode == "ensemble" and (args.feature_weights_json is None or args.ensemble_json is None):
        raise ValueError("--selection-mode ensemble requires --feature-weights-json and --ensemble-json")
    adapter_count = len(args.adapter_path)
    num_return_sequences = [int(value) for value in broadcast(args.num_return_sequences, adapter_count, 16)]
    temperatures = [float(value) for value in broadcast(args.temperature, adapter_count, 0.65)]
    top_ps = [float(value) for value in broadcast(args.top_p, adapter_count, 0.90)]
    top_ks = [int(value) for value in broadcast(args.top_k, adapter_count, 50)]
    if any(value > 1 for value in num_return_sequences) and args.batch_size != 1:
        print("Using batch-size 1 is recommended for decoder-only K-sampling; continuing anyway.", file=sys.stderr)

    source_rows = feature_generate.load_source_rows(args.input_path, args.source)
    terminology_entries = (
        gspo.load_terminology_entries(args.dataset_repo, args.terminology_file, args.terminology_min_source_chars)
        if args.terminology_file
        else []
    )
    if args.few_shot_top_k > 0:
        few_shot_rows, _eval_rows = gspo.split_rows(
            gspo.load_chanka_rows(args.dataset_repo, args.few_shot_dataset_file),
            validation_fraction=0.15,
            seed=3407,
            max_train_samples=None,
            max_eval_samples=None,
        )
    else:
        few_shot_rows = []
    text_model = text_reranker.model_from_json(json.loads(args.text_model_json.read_text()))
    if args.selection_mode == "ensemble":
        feature_model = feature_reranker.model_from_json(json.loads(args.feature_weights_json.read_text()))
        ensemble_model = ensemble_reranker.model_from_json(json.loads(args.ensemble_json.read_text()))
    else:
        feature_model = None
        ensemble_model = None

    all_groups = []
    for index, adapter_path in enumerate(args.adapter_path):
        print(
            json.dumps(
                {
                    "adapter_index": index,
                    "adapter_path": str(adapter_path),
                    "num_return_sequences": num_return_sequences[index],
                    "temperature": temperatures[index],
                    "top_p": top_ps[index],
                    "top_k": top_ks[index],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        all_groups.append(
            generate_groups_for_adapter(
                adapter_path,
                source_rows,
                terminology_entries,
                few_shot_rows,
                args,
                num_return_sequences[index],
                temperatures[index],
                top_ps[index],
                top_ks[index],
            )
        )

    merged_groups = merge_candidate_groups(all_groups)
    if args.selection_mode == "text":
        selected = [
            sparse_row.row.candidate
            for sparse_row in text_reranker.select_sparse(
                text_reranker.sparse_groups_for_model(
                    feature_reranker.featurize_groups(merged_groups),
                    text_model,
                ),
                text_model,
            )
        ]
    else:
        assert feature_model is not None and ensemble_model is not None
        selected = ensemble_generate.select_translations(merged_groups, feature_model, text_model, ensemble_model)
    feature_generate.write_selected(args.output_jsonl, selected)
    if args.candidates_jsonl:
        feature_generate.write_candidates(args.candidates_jsonl, merged_groups)
    print(
        json.dumps(
            {
                "sources": len(source_rows),
                "adapter_count": adapter_count,
                "candidate_groups": len(merged_groups),
                "total_candidates": sum(len(group) for group in merged_groups),
                "written": len(selected),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
