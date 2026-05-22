"""Evaluate external translation baselines on the clean Chanka split."""

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
    parser.add_argument("--backend", choices=["nllb", "seq2seq-chat", "causal-chat", "translategemma"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--dataset-file", default=gspo.CHANKA_FILE)
    parser.add_argument("--split", choices=["eval", "train", "all"], default="eval")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--source-lang", default="spa_Latn")
    parser.add_argument("--target-lang", default="quy_Latn")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, default=None)
    return parser.parse_args(argv)


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = gspo.load_chanka_rows(args.dataset_repo, args.dataset_file)
    train_rows, eval_rows = gspo.split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    if args.split == "eval":
        return eval_rows
    if args.split == "train":
        return train_rows
    return [*eval_rows, *train_rows]


def torch_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def generate_nllb(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch_dtype(args.torch_dtype),
    )
    tokenizer.src_lang = args.source_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)
    if forced_bos_token_id is None or forced_bos_token_id < 0:
        raise ValueError(f"Target language token not found: {args.target_lang}")

    predictions: list[str] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        inputs = tokenizer(
            [row["source"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=args.max_new_tokens,
            )
        predictions.extend(gspo.normalize_text(text) for text in tokenizer.batch_decode(outputs, skip_special_tokens=True))
        print(f"generated {min(len(rows), start + len(batch))}/{len(rows)}", flush=True)
    return predictions


def seq2seq_prompt(source: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Translate the following Spanish text into Quechua Chanka. "
                "Return only the translation, with no explanation.\n\n"
                f"Spanish: {source}"
            ),
        }
    ]


def causal_translation_prompt(source: str) -> str:
    return (
        "Translate the following Spanish text into Quechua Chanka. "
        "Only output the translated result without any explanation:\n\n"
        f"{source}"
    )


def generate_seq2seq_chat(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch_dtype(args.torch_dtype),
    )
    predictions: list[str] = []
    for row in rows:
        if getattr(tokenizer, "chat_template", None):
            inputs = tokenizer.apply_chat_template(
                seq2seq_prompt(row["source"]),
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).to(model.device)
        else:
            prompt = (
                "Translate Spanish to Quechua Chanka. Return only the translation.\n"
                f"Spanish: {row['source']}\nQuechua Chanka:"
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:] if outputs.shape[-1] > prompt_len else outputs[0]
        predictions.append(gspo.strip_chat_artifacts(tokenizer.decode(generated, skip_special_tokens=True)))
        print(f"generated {len(predictions)}/{len(rows)}", flush=True)
    return predictions


def generate_causal_chat(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch_dtype(args.torch_dtype),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    predictions: list[str] = []
    for row in rows:
        prompt = causal_translation_prompt(row["source"])
        messages = [{"role": "user", "content": prompt}]
        if getattr(tokenizer, "chat_template", None):
            chat_template_kwargs = {
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
            }
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **chat_template_kwargs,
                ).to(model.device)
            except TypeError:
                inputs = tokenizer.apply_chat_template(messages, **chat_template_kwargs).to(model.device)
        else:
            fallback_prompt = f"{prompt}\n\nQuechua Chanka:"
            inputs = tokenizer(fallback_prompt, return_tensors="pt", truncation=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        predictions.append(gspo.strip_chat_artifacts(tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)))
        print(f"generated {len(predictions)}/{len(rows)}", flush=True)
    return predictions


def generate_translategemma(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[str]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch_dtype(args.torch_dtype),
    )
    predictions: list[str] = []
    for row in rows:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": args.source_lang,
                        "target_lang_code": args.target_lang,
                        "text": row["source"],
                    }
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        prompt_len = inputs["input_ids"].shape[-1]
        predictions.append(gspo.strip_chat_artifacts(processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)))
        print(f"generated {len(predictions)}/{len(rows)}", flush=True)
    return predictions


def main() -> None:
    args = parse_args()
    rows = selected_rows(args)
    if args.backend == "nllb":
        predictions = generate_nllb(args, rows)
    elif args.backend == "seq2seq-chat":
        predictions = generate_seq2seq_chat(args, rows)
    elif args.backend == "causal-chat":
        predictions = generate_causal_chat(args, rows)
    else:
        predictions = generate_translategemma(args, rows)

    references = [row["target"] for row in rows]
    sources = [row["source"] for row in rows]
    metrics = gspo.corpus_metrics(predictions, references, sources)
    metrics.update(
        {
            "backend": args.backend,
            "model_id": args.model_id,
            "split": args.split,
            "eval_rows": len(rows),
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))

    if args.predictions_jsonl:
        args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_jsonl.open("w") as handle:
            for row, prediction in zip(rows, predictions, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "prediction": prediction,
                            "reference": row["target"],
                            "source": row["source"],
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
