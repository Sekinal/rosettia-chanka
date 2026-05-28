"""STaR-style self-grounded reasoning trace generation for Chanka translation.

Pipeline:
  1. Load v25 (best chanka-stage adapter) via Unsloth.
  2. Greedy-translate the 897 train Chanka sources.
  3. Compute per-row chrF++ vs reference.
  4. For matches (chrF > 70): use v25's own translation as the target.
     For misses: use the reference as the target (post-hoc rationalization).
  5. Prompt v25 again with (source, target) to generate a 2-3 sentence
     Spanish reasoning about why the target is correct.
  6. Save JSONL: {source, target, reasoning, v25_translation, match_chrf}.

Output is consumed by v26_star_build_jsonl.py to produce the training file.
"""
import argparse
import json
import time
from pathlib import Path

import polars as pl
import sacrebleu
import torch
from huggingface_hub import hf_hub_download


CHANKA_FILE = "clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet"
SYSTEM = "Eres un traductor profesional español-quechua chanka."
TRANSLATE_INSTR = (
    "Traduce del español al quechua chanka. Usa una traducción directa, "
    "fiel y apropiada para contexto judicial."
)
REASON_INSTR = (
    "Eres un experto en lingüística del quechua chanka. Te muestro una "
    "traducción correcta. Explica en 2-3 oraciones por qué la traducción "
    "es correcta, mencionando morfología, raíces verbales o sufijos clave."
)


def load_chanka_rows(repo_id: str, filename: str, cache_dir: str) -> list[dict[str, str]]:
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=cache_dir)
    frame = pl.read_parquet(path)
    sc = "reviewed_spanish"
    tc = "reviewed_chanka_quechua"
    rows = []
    for r in frame.select([sc, tc]).iter_rows(named=True):
        s = str(r[sc]).strip()
        t = str(r[tc]).strip()
        if s and t:
            rows.append({"source": s, "target": t})
    return rows


def split_eval(rows, seed=3407, frac=0.15):
    import random
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * frac))
    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    return train_rows, eval_rows


def build_translate_prompt(tok, source: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{TRANSLATE_INSTR}\n\nEspañol: {source}"},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_reason_prompt(tok, source: str, target: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"{REASON_INSTR}\n\n"
                f"Español: {source}\n"
                f"Traducción correcta al quechua chanka: {target}\n\n"
                "Razonamiento (2-3 oraciones en español):"
            ),
        },
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def batch_generate(model, tok, prompts, max_new, batch_size, do_sample=False, temperature=0.0):
    out_texts = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tok(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.no_grad():
            kwargs = {
                "max_new_tokens": max_new,
                "do_sample": do_sample,
                "pad_token_id": pad_id,
            }
            if do_sample:
                kwargs["temperature"] = temperature
            out = model.generate(**inputs, **kwargs)
        prompt_len = inputs.input_ids.shape[1]
        for j in range(len(batch)):
            new_tokens = out[j, prompt_len:]
            txt = tok.decode(new_tokens, skip_special_tokens=True).strip()
            out_texts.append(txt)
        if (i // batch_size) % 10 == 0:
            print(f"  generated {i + len(batch)}/{len(prompts)}", flush=True)
    return out_texts


def chrf_score(pred: str, ref: str) -> float:
    return sacrebleu.CHRF(word_order=2).sentence_score(pred, [ref]).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="v25 best adapter path")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--max-new-translate", type=int, default=96)
    ap.add_argument("--max-new-reason", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--match-chrf-threshold", type=float, default=70.0)
    ap.add_argument("--dataset-repo", default="Thermostatic/rosettia-chanka-data")
    ap.add_argument("--cache-dir", default="/root/.cache/hf-rosettia")
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading Chanka rows...", flush=True)
    rows = load_chanka_rows(args.dataset_repo, CHANKA_FILE, args.cache_dir)
    train_rows, eval_rows = split_eval(rows)
    print(f"[{time.strftime('%H:%M:%S')}] train={len(train_rows)} eval={len(eval_rows)}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading v25 adapter from {args.adapter} via Unsloth...", flush=True)
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

    # Step 1+2: translate
    print(f"[{time.strftime('%H:%M:%S')}] translating {len(train_rows)} train sources...", flush=True)
    t_prompts = [build_translate_prompt(tok, r["source"]) for r in train_rows]
    t0 = time.time()
    translations = batch_generate(model, tok, t_prompts, args.max_new_translate, args.batch_size)
    print(f"[{time.strftime('%H:%M:%S')}] translation done in {time.time()-t0:.0f}s", flush=True)

    # Step 3: per-row chrF
    chrf_vals = [chrf_score(p, r["target"]) for p, r in zip(translations, train_rows)]
    n_match = sum(1 for c in chrf_vals if c >= args.match_chrf_threshold)
    print(f"[{time.strftime('%H:%M:%S')}] match rate at chrF>={args.match_chrf_threshold}: "
          f"{n_match}/{len(train_rows)} = {n_match/len(train_rows)*100:.1f}%", flush=True)

    # Step 4: pick training target
    star_targets = []
    for tr, chrf, row in zip(translations, chrf_vals, train_rows):
        target = tr if chrf >= args.match_chrf_threshold else row["target"]
        star_targets.append(target)

    # Step 5: generate reasoning prompts
    print(f"[{time.strftime('%H:%M:%S')}] generating reasoning for {len(train_rows)} pairs...", flush=True)
    r_prompts = [build_reason_prompt(tok, r["source"], t) for r, t in zip(train_rows, star_targets)]
    t0 = time.time()
    reasonings = batch_generate(model, tok, r_prompts, args.max_new_reason, args.batch_size)
    print(f"[{time.strftime('%H:%M:%S')}] reasoning done in {time.time()-t0:.0f}s", flush=True)

    # Save
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for row, tr, chrf, target, reason in zip(
            train_rows, translations, chrf_vals, star_targets, reasonings
        ):
            rec = {
                "source": row["source"],
                "reference": row["target"],
                "v25_translation": tr,
                "v25_chrf_vs_reference": round(chrf, 3),
                "star_target": target,
                "is_match": chrf >= args.match_chrf_threshold,
                "reasoning": reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
