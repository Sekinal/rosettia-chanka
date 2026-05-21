"""Evaluate a trained GSPO/SFT adapter on the clean Chanka validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_gspo_chanka_unsloth as gspo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--dataset-file", default=gspo.CHANKA_FILE)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=["eval", "train", "all"],
        default="eval",
        help="Dataset split to generate on. Use train/all for verifier candidate mining, eval for metrics.",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Number of generations per source row. Values >1 are intended for sampled verifier candidates.",
    )
    parser.add_argument("--do-sample", action="store_true", help="Use stochastic decoding for candidate mining.")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--strip-chat-artifacts", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def generate_predictions_with_progress(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    max_completion_length: int,
    batch_size: int,
    num_return_sequences: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    progress_every: int,
    strip_chat_artifacts: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    import torch

    if num_return_sequences > 1 and not do_sample:
        raise ValueError("--num-return-sequences > 1 requires --do-sample for generation diversity.")

    generated_rows: list[dict[str, str]] = []
    predictions: list[str] = []
    model.eval()
    total = len(rows)
    batch_size = max(1, batch_size)
    num_return_sequences = max(1, num_return_sequences)
    for start in range(0, total, batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                gspo.prompt_messages(row["source"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch_rows
        ]
        inputs = tokenizer(text=prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                num_return_sequences=num_return_sequences,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        for output_index in range(output_ids.shape[0]):
            row = batch_rows[output_index // num_return_sequences]
            completion_ids = output_ids[output_index, prompt_length:]
            prediction = tokenizer.decode(completion_ids, skip_special_tokens=True)
            if strip_chat_artifacts:
                prediction = gspo.strip_chat_artifacts(prediction)
            else:
                prediction = gspo.normalize_text(prediction)
            generated_rows.append(row)
            predictions.append(prediction)
        completed = min(total, start + len(batch_rows))
        if progress_every > 0 and (completed == total or completed % progress_every == 0):
            print(f"generated {completed}/{total} predictions", flush=True)
    return generated_rows, predictions


def main() -> None:
    args = parse_args()

    from unsloth import FastLanguageModel

    rows = gspo.load_chanka_rows(args.dataset_repo, args.dataset_file)
    train_rows, eval_rows = gspo.split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    if args.split == "eval":
        generation_rows = eval_rows
    elif args.split == "train":
        generation_rows = train_rows
    else:
        generation_rows = [*eval_rows, *train_rows]

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

    generated_rows, predictions = generate_predictions_with_progress(
        model,
        tokenizer,
        generation_rows,
        args.max_completion_length,
        args.batch_size,
        args.num_return_sequences,
        args.do_sample,
        args.temperature,
        args.top_p,
        args.top_k,
        args.progress_every,
        args.strip_chat_artifacts,
    )
    references = [row["target"] for row in generated_rows]
    sources = [row["source"] for row in generated_rows]
    metrics = gspo.corpus_metrics(predictions, references, sources)
    metrics["eval_rows"] = len(generated_rows)
    metrics["source_rows"] = len(generation_rows)
    metrics["split"] = args.split
    metrics["num_return_sequences"] = args.num_return_sequences
    metrics["adapter_path"] = str(args.adapter_path)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.predictions_jsonl:
        args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_jsonl.open("w") as handle:
            for row, prediction in zip(generated_rows, predictions, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "source": row["source"],
                            "reference": row["target"],
                            "prediction": prediction,
                            "source_name": row.get("source_name"),
                            "variant": row.get("variant"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
