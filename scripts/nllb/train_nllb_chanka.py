"""Fine-tune NLLB-200 for Spanish→Chanka Quechua (spa_Latn → quy_Latn).

NLLB-200 natively supports `quy_Latn` (Ayacucho Quechua) — unlike Qwen, it has
real Quechua pretraining exposure, which is the main lever for breaking past the
~40 ChrF plateau we hit fine-tuning a Quechua-naive decoder-only model.

Trains on the normalized v34 corpus (jsonl: {spanish, chanka}). LoRA by default
(fits the 3.3B on one 80GB card with headroom); --full for full fine-tune.
"""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="facebook/nllb-200-3.3B",
                    help="facebook/nllb-200-{distilled-600M,1.3B,3.3B}")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--src-field", default="spanish")
    ap.add_argument("--tgt-field", default="chanka")
    ap.add_argument("--src-lang", default="spa_Latn")
    ap.add_argument("--tgt-lang", default="quy_Latn")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)        # LoRA wants higher LR
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import torch, random
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              DataCollatorForSeq2Seq)
    from datasets import Dataset

    rows = [json.loads(l) for l in open(args.train_jsonl) if l.strip()]
    random.Random(args.seed).shuffle(rows)
    n_val = max(64, int(len(rows) * args.val_fraction))
    val, train = rows[:n_val], rows[n_val:]
    print(f"train={len(train)} val={len(val)}")

    tok = AutoTokenizer.from_pretrained(args.model_id, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)

    if not args.full:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                        task_type="SEQ_2_SEQ_LM")
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()

    def preprocess(batch):
        model_in = tok(batch[args.src_field], max_length=args.max_len, truncation=True)
        labels = tok(text_target=batch[args.tgt_field], max_length=args.max_len, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in

    train_ds = Dataset.from_list(train).map(preprocess, batched=True, remove_columns=list(train[0].keys()))
    val_ds = Dataset.from_list(val).map(preprocess, batched=True, remove_columns=list(val[0].keys()))
    collator = DataCollatorForSeq2Seq(tok, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=None,
        logging_steps=50, bf16=True, seed=args.seed,
        predict_with_generate=False, report_to=[],
    )
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds,
                             eval_dataset=val_ds, data_collator=collator, tokenizer=tok)
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print("done ->", args.output_dir)


if __name__ == "__main__":
    main()
