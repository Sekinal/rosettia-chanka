"""Generate Chanka translations with K-sampling plus text-aware reranking."""

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
from scripts import train_text_candidate_reranker as text_reranker


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
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
    model: text_reranker.TextRankerModel,
) -> list[feature_generate.oracle_rerank.Candidate]:
    feature_groups = feature_reranker.featurize_groups(groups)
    sparse_groups = text_reranker.sparse_groups_for_model(feature_groups, model)
    return [sparse_row.row.candidate for sparse_row in text_reranker.select_sparse(sparse_groups, model)]


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
    text_model = text_reranker.model_from_json(json.loads(args.model_json.read_text()))

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

    generated_rows, predictions, _generated_terms = evaluate.generate_predictions_with_progress(
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
    selected = select_translations(groups, text_model)
    feature_generate.write_selected(args.output_jsonl, selected)
    if args.candidates_jsonl:
        feature_generate.write_candidates(args.candidates_jsonl, groups)
    print(json.dumps({"sources": len(source_rows), "written": len(selected)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
