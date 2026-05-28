"""Eval a single LoRA checkpoint on the broad held-out split (seed=3407, validation_fraction=0.02).

Reports chrF++ and BLEU via sacrebleu. The held-out rows are the SAME rows the trainer
excluded from training (deterministic shuffle + take first 2%), so the metric is provably
on data the model never saw.
"""
import argparse
import json
import random
import time
from pathlib import Path

import polars as pl
import sacrebleu
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BROAD_FILES = [
    "broad_quechua/somosnlp_spanish_to_quechua_high_quality_sft.parquet",
    "americasnlp/americasnlp_quechua_spanish_high_quality_real_sft.parquet",
]
INSTRUCTION = "Traduce del español al quechua. Conserva el significado y no copies el español."
SYSTEM = "Eres un traductor profesional español-quechua."


def load_broad_rows(repo: str, cache_dir: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fn in BROAD_FILES:
        path = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset", cache_dir=cache_dir)
        frame = pl.read_parquet(path)
        sc = "es"
        tc = "quy" if "quy" in frame.columns else "qu"
        for r in frame.select([sc, tc]).iter_rows(named=True):
            s = str(r[sc]).strip()
            t = str(r[tc]).strip()
            if s and t:
                rows.append({"source": s, "target": t})
    return rows


def split_eval(rows: list[dict[str, str]], seed: int, frac: float, n: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * frac))
    eval_rows = shuffled[:eval_size]
    return eval_rows[:n]


def build_prompt(tok, source: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{INSTRUCTION}\n\nEspañol: {source}"},
    ]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def evaluate(model, tok, rows, batch_size: int, max_new: int) -> tuple[float, float, list[str]]:
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    preds: list[str] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        prompts = [build_prompt(tok, r["source"]) for r in batch]
        inputs = tok(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=pad_id,
            )
        prompt_len = inputs.input_ids.shape[1]
        for j in range(len(batch)):
            new_tokens = out[j, prompt_len:]
            txt = tok.decode(new_tokens, skip_special_tokens=True).strip()
            preds.append(txt)
    refs = [r["target"] for r in rows]
    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds, [refs]).score
    bleu = sacrebleu.BLEU().corpus_score(preds, [refs]).score
    return chrfpp, bleu, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model id or path")
    ap.add_argument("--ckpt", required=True, help="LoRA adapter dir to evaluate")
    ap.add_argument("--repo", default="Thermostatic/rosettia-chanka-data")
    ap.add_argument("--cache-dir", default="/root/.cache/hf-rosettia")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--validation-fraction", type=float, default=0.02)
    ap.add_argument("--n-rows", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-predictions-jsonl", default=None)
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading broad rows...", flush=True)
    all_rows = load_broad_rows(args.repo, args.cache_dir)
    eval_rows = split_eval(all_rows, args.seed, args.validation_fraction, args.n_rows)
    print(
        f"[{time.strftime('%H:%M:%S')}] total broad rows={len(all_rows):,}, "
        f"held-out eval rows used={len(eval_rows):,}",
        flush=True,
    )

    is_full_model = (Path(args.ckpt) / "config.json").exists() and not (Path(args.ckpt) / "adapter_config.json").exists()

    if is_full_model:
        print(f"[{time.strftime('%H:%M:%S')}] loading FULL model from {args.ckpt} via Unsloth (matching evaluate_gspo_checkpoint.py)...", flush=True)
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=str(args.ckpt),
            max_seq_length=512,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.pad_token_id = tok.eos_token_id
        FastLanguageModel.for_inference(model)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] loading base {args.base}...", flush=True)
        tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        print(f"[{time.strftime('%H:%M:%S')}] loading adapter {args.ckpt}...", flush=True)
        model = PeftModel.from_pretrained(base, args.ckpt)
    model.eval()

    t0 = time.time()
    chrfpp, bleu, preds = evaluate(model, tok, eval_rows, args.batch_size, args.max_new)
    elapsed = time.time() - t0

    step = int(args.ckpt.rstrip("/").split("checkpoint-")[-1]) if "checkpoint-" in args.ckpt else -1
    rec = {
        "ckpt": args.ckpt,
        "step": step,
        "chrf++": chrfpp,
        "bleu": bleu,
        "n_rows": len(eval_rows),
        "elapsed_sec": elapsed,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rec, indent=2))
    print(
        f"[{time.strftime('%H:%M:%S')}] {args.ckpt}: chrF++={chrfpp:.3f} BLEU={bleu:.3f} ({elapsed:.0f}s)",
        flush=True,
    )

    if args.out_predictions_jsonl:
        Path(args.out_predictions_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_predictions_jsonl, "w") as f:
            for r, p in zip(eval_rows, preds):
                f.write(json.dumps({"source": r["source"], "target": r["target"], "prediction": p}) + "\n")


if __name__ == "__main__":
    main()
