"""Fine-tune Qwen3.5-4B as an explainable Chanka orthographic normalizer.

Input format per training row (JSONL with keys 'input', 'trace', 'normalized'):

  System: "You are a Chanka Quechua orthographic normalizer..."
  User:   "Normalize: <noisy sentence>"
  Asst:   "<think>\n<trace>\n</think>\n\nNormalized: <clean sentence>"

The trace is the multi-teacher-voted reasoning trace cataloging per-token
spec rule citations. The model learns to emit <think>...</think> before
producing the final normalization.

Implementation mirrors scripts/train_sft_unsloth.py but with the dedicated
seq2seq normalization format (no broad/chanka stage switching, no glossary
RAG, fixed Qwen3.5-4B base).
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are an expert Chanka (Ayacucho) Quechua orthographic normalizer "
    "following the MINEDU 2021 standard. For each input, emit a <think>...</think> "
    "trace that lists every token left-to-right with the spec rule cited (R1-R7, "
    "S1-S8, L0-L3, §6.5, §6.6, §8.5, §8.6), then a 'Normalized:' line with the "
    "canonical sentence."
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def format_row(row: dict[str, Any]) -> dict[str, str]:
    """Convert a gold-merge row into (source, target) for the trainer."""
    trace = row.get("trace", "").strip()
    norm = row.get("normalized", "").strip()
    # Final assistant turn
    if trace:
        target = f"<think>\n{trace}\n</think>\n\nNormalized: {norm}"
    else:
        target = f"<think>\nAll tokens already MINEDU-compliant. No changes required.\n</think>\n\nNormalized: {norm}"
    user_msg = f"Normalize this Chanka sentence per the MINEDU 2021 spec:\n\n{row['input']}"
    return {"system": SYSTEM_PROMPT, "user": user_msg, "assistant": target}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-jsonl", required=True)
    ap.add_argument("--model-id", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--lora-alpha", type=int, default=256)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--num-train-epochs", type=float, default=3.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=64)
    ap.add_argument("--save-steps", type=int, default=64)
    ap.add_argument("--save-total-limit", type=int, default=0)
    ap.add_argument("--logging-steps", type=int, default=8)
    ap.add_argument("--validation-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resume-from-checkpoint", type=str, default=None)
    ap.add_argument("--gradient-checkpointing", choices=["unsloth", "true", "false"], default="unsloth")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load + split
    rows = [format_row(r) for r in iter_jsonl(Path(args.gold_jsonl))]
    rng.shuffle(rows)
    eval_size = max(8, int(len(rows) * args.validation_fraction))
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]
    print(f"Loaded {len(rows)} rows -> train={len(train_rows)} eval={len(eval_rows)}")

    # Lazy import unsloth
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    gc_arg: Any = "unsloth"
    if args.gradient_checkpointing == "true": gc_arg = True
    elif args.gradient_checkpointing == "false": gc_arg = False

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing=gc_arg,
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    def to_chat(row: dict[str, str]) -> dict[str, str]:
        msgs = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["assistant"]},
        ]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_ds = Dataset.from_list(train_rows).map(to_chat, remove_columns=["system", "user", "assistant"])
    eval_ds = Dataset.from_list(eval_rows).map(to_chat, remove_columns=["system", "user", "assistant"])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=targs,
        packing=False,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
