"""Evaluate a v26 STaR checkpoint on the 158-row clean Chanka held-out.

Loads the adapter via Unsloth, generates raw output (reasoning + translation),
extracts the Traducción: segment, computes chrF++/BLEU.
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


CHANKA_FILE = "clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet"
SYSTEM = "Eres un traductor profesional español-quechua chanka."
INSTR = (
    "Traduce del español al quechua chanka. Usa una traducción directa, "
    "fiel y apropiada para contexto judicial."
)
TRANSLATION_MARKERS = ["Traducción:", "Traduccion:", "Translation:", "Final:"]


def load_chanka_rows(repo: str, filename: str, cache_dir: str):
    path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", cache_dir=cache_dir)
    frame = pl.read_parquet(path)
    rows = []
    for r in frame.select(["reviewed_spanish", "reviewed_chanka_quechua"]).iter_rows(named=True):
        s = str(r["reviewed_spanish"]).strip()
        t = str(r["reviewed_chanka_quechua"]).strip()
        if s and t:
            rows.append({"source": s, "target": t})
    return rows


def split_eval(rows, seed=3407, frac=0.15):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * frac))
    return shuffled[:eval_size]


def build_prompt(tok, source: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{INSTR}\n\nEspañol: {source}"},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_translation(raw: str) -> str:
    for marker in TRANSLATION_MARKERS:
        idx = raw.find(marker)
        if idx >= 0:
            tail = raw[idx + len(marker):].strip()
            for stop in ["Razonamiento:", "Analisis:", "Puntaje:", "\n\n"]:
                stop_idx = tail.find(stop)
                if stop_idx >= 0:
                    tail = tail[:stop_idx].strip()
                    break
            return tail
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[-1] if lines else raw.strip()


def evaluate(model, tok, rows, batch_size: int, max_new: int):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    preds_raw = []
    preds_extracted = []
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
            preds_raw.append(txt)
            preds_extracted.append(extract_translation(txt))
    refs = [r["target"] for r in rows]
    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds_extracted, [refs]).score
    bleu = sacrebleu.BLEU().corpus_score(preds_extracted, [refs]).score
    marker_hit = sum(1 for r in preds_raw if any(m in r for m in TRANSLATION_MARKERS)) / max(1, len(preds_raw))
    return chrfpp, bleu, marker_hit, preds_raw, preds_extracted, refs, [r["source"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--dataset-repo", default="Thermostatic/rosettia-chanka-data")
    ap.add_argument("--cache-dir", default="/root/.cache/hf-rosettia")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-predictions-jsonl", default=None)
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading Chanka eval rows...", flush=True)
    rows = load_chanka_rows(args.dataset_repo, CHANKA_FILE, args.cache_dir)
    eval_rows = split_eval(rows)
    print(f"[{time.strftime('%H:%M:%S')}] eval rows: {len(eval_rows)}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading adapter {args.adapter}...", flush=True)
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=args.max_seq_length,
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

    t0 = time.time()
    chrfpp, bleu, marker_hit, preds_raw, preds_ext, refs, sources = evaluate(
        model, tok, eval_rows, args.batch_size, args.max_new
    )
    elapsed = time.time() - t0

    step = int(str(args.adapter).rstrip("/").split("checkpoint-")[-1]) if "checkpoint-" in str(args.adapter) else -1
    rec = {
        "adapter": str(args.adapter),
        "step": step,
        "chrf++": chrfpp,
        "bleu": bleu,
        "marker_hit_rate": marker_hit,
        "n_rows": len(eval_rows),
        "elapsed_sec": elapsed,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rec, indent=2))
    print(
        f"[{time.strftime('%H:%M:%S')}] {args.adapter}: chrF++={chrfpp:.3f} BLEU={bleu:.3f} marker_hit={marker_hit:.2%} ({elapsed:.0f}s)",
        flush=True,
    )

    if args.out_predictions_jsonl:
        Path(args.out_predictions_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_predictions_jsonl, "w") as f:
            for src, ref, raw, pred in zip(sources, refs, preds_raw, preds_ext):
                f.write(json.dumps({
                    "source": src,
                    "reference": ref,
                    "raw_prediction": raw,
                    "extracted_translation": pred,
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
