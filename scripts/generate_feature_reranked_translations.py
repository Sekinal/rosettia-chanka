"""Generate Chanka translations with K-sampling plus feature reranking.

This is the reference-free deployment path for the current best selector:
generate multiple sampled candidates from an adapter, then select one candidate
with weights trained by ``train_feature_candidate_reranker.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_gspo_checkpoint as evaluate
from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_feature_candidate_reranker as feature_reranker
from scripts import train_gspo_chanka_unsloth as gspo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--weights-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, default=None)
    parser.add_argument("--source", action="append", default=[], help="Spanish source text. Repeatable.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Plain text file with one source per line, or JSONL with a 'source' field.",
    )
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--strip-chat-artifacts", action="store_true")
    parser.add_argument(
        "--terminology-file",
        default=None,
        help="Optional dataset-repo parquet glossary used in prompts and any terminology-aware feature weights.",
    )
    parser.add_argument("--terminology-top-k", type=int, default=1)
    parser.add_argument("--terminology-min-source-chars", type=int, default=3)
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc


def load_source_rows(path: Path | None, inline_sources: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in inline_sources:
        normalized = gspo.normalize_text(source)
        if normalized:
            rows.append({"source": normalized, "source_name": "inline", "variant": "quy/chanka"})

    if path is not None:
        if path.suffix.lower() == ".jsonl":
            for record in iter_jsonl(path):
                source = gspo.normalize_text(str(record.get("source", "")))
                if source:
                    rows.append(
                        {
                            "source": source,
                            "source_name": str(record.get("source_name") or path.name),
                            "variant": str(record.get("variant") or "quy/chanka"),
                        }
                    )
        else:
            with path.open() as handle:
                for line in handle:
                    source = gspo.normalize_text(line)
                    if source:
                        rows.append({"source": source, "source_name": path.name, "variant": "quy/chanka"})

    if not rows:
        raise ValueError("No source rows were provided.")
    return rows


def candidate_groups_from_generation(
    generated_rows: Sequence[dict[str, str]],
    predictions: Sequence[str],
    num_return_sequences: int,
    pool_path: str | None = None,
) -> list[list[oracle_rerank.Candidate]]:
    if len(generated_rows) != len(predictions):
        raise ValueError("generated_rows and predictions must have the same length")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be >= 1")

    groups: list[list[oracle_rerank.Candidate]] = []
    for start in range(0, len(predictions), num_return_sequences):
        row_slice = generated_rows[start : start + num_return_sequences]
        prediction_slice = predictions[start : start + num_return_sequences]
        if not row_slice:
            continue
        source = row_slice[0]["source"]
        group: list[oracle_rerank.Candidate] = []
        for index, (row, prediction) in enumerate(zip(row_slice, prediction_slice, strict=True)):
            group.append(
                oracle_rerank.Candidate(
                    source=source,
                    reference="",
                    prediction=prediction,
                    source_name=row.get("source_name"),
                    variant=row.get("variant"),
                    candidate_index=index,
                    pool_path=pool_path,
                )
            )
        groups.append(group)
    return groups


def select_translations(
    groups: Sequence[Sequence[oracle_rerank.Candidate]],
    model: feature_reranker.FeatureModel,
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> list[oracle_rerank.Candidate]:
    feature_groups = feature_reranker.featurize_groups(groups, terminology_entries, terminology_top_k)
    return [row.candidate for row in feature_reranker.select_feature(feature_groups, model)]


def write_selected(path: Path, selected: Sequence[oracle_rerank.Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for candidate in selected:
            handle.write(
                json.dumps(
                    {
                        "source": candidate.source,
                        "prediction": candidate.prediction,
                        "candidate_index": candidate.candidate_index,
                        "source_name": candidate.source_name,
                        "variant": candidate.variant,
                        "pool_path": candidate.pool_path,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def write_candidates(path: Path, groups: Sequence[Sequence[oracle_rerank.Candidate]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for group in groups:
            for candidate in group:
                handle.write(
                    json.dumps(
                        {
                            "source": candidate.source,
                            "prediction": candidate.prediction,
                            "candidate_index": candidate.candidate_index,
                            "source_name": candidate.source_name,
                            "variant": candidate.variant,
                            "pool_path": candidate.pool_path,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


def main() -> None:
    args = parse_args()
    if args.num_return_sequences > 1 and args.batch_size != 1:
        print("Using batch-size 1 is recommended for decoder-only K-sampling; continuing anyway.", file=sys.stderr)

    from unsloth import FastLanguageModel

    source_rows = load_source_rows(args.input_path, args.source)
    terminology_entries = (
        gspo.load_terminology_entries(args.dataset_repo, args.terminology_file, args.terminology_min_source_chars)
        if args.terminology_file
        else []
    )
    weights_payload = json.loads(args.weights_json.read_text())
    feature_model = feature_reranker.model_from_json(weights_payload)

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
    groups = candidate_groups_from_generation(
        generated_rows,
        predictions,
        args.num_return_sequences,
        pool_path=str(args.adapter_path),
    )
    selected = select_translations(groups, feature_model, terminology_entries, args.terminology_top_k)
    write_selected(args.output_jsonl, selected)
    if args.candidates_jsonl:
        write_candidates(args.candidates_jsonl, groups)
    print(json.dumps({"sources": len(source_rows), "written": len(selected)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
