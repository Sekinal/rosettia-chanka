"""LoRA SFT for NLLB-200 on the same leak-free direct Chanka pairs we used for Qwen3.5.

NLLB is M2M100 encoder-decoder. Unsloth recognizes M2M_100 but loads with the wrong
AutoModel class, so we use plain HF transformers + peft. Same data slice as our
Qwen3.5 recipe (841 leak-free direct rows).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, "/root/rosettia-chanka")
from scripts import train_gspo_chanka_unsloth as gspo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-lang", default="spa_Latn")
    parser.add_argument("--target-lang", default="quy_Latn")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=32)
    parser.add_argument("--save-total-limit", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    print(f"loading model {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.src_lang = args.source_lang
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, dtype=torch.bfloat16)

    # NLLB uses forced_bos_token_id for target language during generation
    target_bos = tokenizer.convert_tokens_to_ids(args.target_lang)
    if target_bos is None or target_bos == tokenizer.unk_token_id:
        raise ValueError(f"target_lang token {args.target_lang} not in NLLB tokenizer")
    model.generation_config.forced_bos_token_id = target_bos

    # Apply LoRA — for M2M100 the projections live under encoder.layers.* and
    # decoder.layers.*. PEFT supports a target_modules list; q/k/v/o/fc1/fc2.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable={trainable:,} of {total:,} ({100*trainable/total:.2f}%)", flush=True)

    # Load JSONL — direct rows only
    rows = []
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("prompt_mode", "direct") != "direct":
                continue
            rows.append({"source": r["source"], "target": r["target"]})
    print(f"loaded {len(rows)} direct rows", flush=True)

    # Cross-check leakage against Chanka eval split
    chanka_rows = gspo.load_chanka_rows(gspo.DATASET_REPO, gspo.CHANKA_FILE)
    _, eval_rows = gspo.split_rows(chanka_rows, validation_fraction=0.15, seed=3407,
                                    max_train_samples=None, max_eval_samples=None)
    eval_sources = {r["source"] for r in eval_rows}
    before = len(rows)
    rows = [r for r in rows if r["source"] not in eval_sources]
    print(f"after eval-leak filter: {len(rows)} (dropped {before - len(rows)})", flush=True)
    for r in rows:
        assert r["source"] not in eval_sources

    # Train/val split (internal — does NOT affect the eval set)
    import random
    rng = random.Random(args.seed)
    shuffled = rows[:]; rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.validation_fraction))
    val_rows = shuffled[:n_val]
    train_rows = shuffled[n_val:]
    print(f"train {len(train_rows)} / val {len(val_rows)}", flush=True)

    def preprocess(batch):
        tokenizer.src_lang = args.source_lang
        tokenizer.tgt_lang = args.target_lang
        model_inputs = tokenizer(
            text=batch["source"],
            text_target=batch["target"],
            max_length=args.max_source_length,
            truncation=True,
            padding=False,
        )
        return model_inputs

    train_ds = Dataset.from_list(train_rows).map(preprocess, batched=True,
                                                   remove_columns=["source", "target"])
    val_ds = Dataset.from_list(val_rows).map(preprocess, batched=True,
                                              remove_columns=["source", "target"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,  # speed; eval loss is enough for stopping
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("starting training", flush=True)
    trainer.train()
    print("training done", flush=True)

    final_dir = args.output_dir / "final_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"saved final adapter to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
