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
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def generate_predictions_with_progress(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    max_completion_length: int,
    batch_size: int,
    progress_every: int,
) -> list[str]:
    import torch

    predictions: list[str] = []
    model.eval()
    total = len(rows)
    batch_size = max(1, batch_size)
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
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        for row_index in range(len(batch_rows)):
            completion_ids = output_ids[row_index, prompt_length:]
            predictions.append(gspo.normalize_text(tokenizer.decode(completion_ids, skip_special_tokens=True)))
        completed = min(total, start + len(batch_rows))
        if progress_every > 0 and (completed == total or completed % progress_every == 0):
            print(f"generated {completed}/{total} predictions", flush=True)
    return predictions


def main() -> None:
    args = parse_args()

    from unsloth import FastLanguageModel

    rows = gspo.load_chanka_rows(args.dataset_repo, args.dataset_file)
    _, eval_rows = gspo.split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_samples=None,
        max_eval_samples=args.max_eval_samples,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    FastLanguageModel.for_inference(model)

    predictions = generate_predictions_with_progress(
        model,
        tokenizer,
        eval_rows,
        args.max_completion_length,
        args.batch_size,
        args.progress_every,
    )
    references = [row["target"] for row in eval_rows]
    sources = [row["source"] for row in eval_rows]
    metrics = gspo.corpus_metrics(predictions, references, sources)
    metrics["eval_rows"] = len(eval_rows)
    metrics["adapter_path"] = str(args.adapter_path)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.predictions_jsonl:
        args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_jsonl.open("w") as handle:
            for row, prediction in zip(eval_rows, predictions, strict=True):
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
