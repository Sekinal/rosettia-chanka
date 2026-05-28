"""V-STaR step 3: DPO-train a verifier from preference pairs.

Verifier is a LoRA adapter on top of v25b merged base. DPO objective increases
log P_verifier(winner|source) - log P_verifier(loser|source). At inference, the
verifier scores K candidates by their log-likelihood; pick the highest.
"""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="merged 9B base model (v25b merged)")
    ap.add_argument("--pairs-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num-epochs", type=float, default=2.0)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=192)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading base {args.base} via Unsloth...", flush=True)
    from unsloth import FastLanguageModel, PatchDPOTrainer
    PatchDPOTrainer()
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_length,
    )

    print(f"[{time.strftime('%H:%M:%S')}] loading {args.pairs_jsonl}...", flush=True)
    from datasets import Dataset
    rows = [json.loads(l) for l in open(args.pairs_jsonl)]
    ds = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in rows
    ])
    split = ds.train_test_split(test_size=0.1, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"[{time.strftime('%H:%M:%S')}] DPO pairs: train={len(train_ds)} eval={len(eval_ds)}", flush=True)

    from trl import DPOTrainer, DPOConfig

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    dpo_cfg = DPOConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        eval_strategy="steps",
        eval_steps=16,
        save_strategy="steps",
        save_steps=16,
        save_total_limit=0,
        logging_steps=4,
        seed=args.seed,
        bf16=True,
        warmup_ratio=0.05,
        weight_decay=0.0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    print(f"[{time.strftime('%H:%M:%S')}] DONE. Adapter saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
