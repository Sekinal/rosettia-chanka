"""v28 smoke: inference-only test of retrieval-augmented prompting on 4B v13.

Goal: does just adding dictionary lookups to the prompt (no training) lift chrF?
If yes, full v28 (GSPO + RAG + reasoning) is worth building. If no, redesign.

Pipeline:
  1. Load 4B base + v13 LoRA (alpha=160)
  2. For each of 158 held-out sources:
     a. Tokenize source; find content-word matches in cuzco_dictionary
     b. Format prompt: [Diccionario block] + [source]
     c. Generate translation
  3. Score chrF/BLEU. Compare to no-retrieval baseline (which we already have: 56.94).
"""
import argparse
import json
import re
import random
import time
import unicodedata
from pathlib import Path


def normalize(s: str) -> str:
    # Lowercase + strip accents for matching
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s


def load_chanka_rows(repo, filename, cache_dir):
    import polars as pl
    from huggingface_hub import hf_hub_download
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


def cuzco_to_chanka(token: str) -> str:
    """Apply Cuzco→Chanka phonological transforms.

    Based on standard descriptions of Chanka (Ayacucho) Quechua vs Cuzco-Collao:
    - Drop ejective apostrophes: k'a → ka, p'a → pa, t'a → ta, q'a → qa, ch' → ch, s' → s
    - Drop aspirate h after stop: kha → ka, pha → pa, tha → ta, qha → qa, chha → cha
    - Chanka 2sg pronoun: qan → qam (final consonant differs)
    - Cuzco /o/ → Chanka /u/ where allophonic (mostly in close vowel contexts)
    - Lowercase first letter for non-proper nouns
    """
    if not token:
        return token
    s = token
    # Lowercase the first character unless it looks like a proper noun (all-caps initial cluster)
    if s and s[0].isupper() and not s[:2].isupper():
        s = s[0].lower() + s[1:]
    # Ejectives
    for stop in ["k", "p", "t", "q", "ch", "s"]:
        s = s.replace(stop + "'", stop)
    # Aspirates: kh, ph, th, qh, chh before a vowel
    import re
    s = re.sub(r"\b(k|p|t|q|ch|s)h([aeiou])", r"\1\2", s)
    # Cuzco /o/ → Chanka /u/ (allophonic — Chanka uses /u/ everywhere)
    s = s.replace("o", "u").replace("O", "U")
    # Final /n/ → /m/ for the 2sg pronoun: qan → qam (a known difference)
    if s.lower() == "qan":
        s = "qam"
    return s


def load_dictionary(path: str, apply_chanka_transform: bool = True) -> list[dict]:
    """Load lemmas from cuzco dictionary; apply Cuzco→Chanka orthographic transforms when needed."""
    d = json.load(open(path))
    entries = []
    for e in d.get("lemmas", []):
        sp = e.get("es", "")
        qu = e.get("qu_cuzco", "")
        note = (e.get("chanka_note") or "").strip()
        if sp and qu:
            if apply_chanka_transform:
                qu = cuzco_to_chanka(qu)
            entries.append({
                "es": sp,
                "es_norm": normalize(sp),
                "qu": qu,
                "note": note,
            })
    return entries


def retrieve_for_source(source: str, entries: list[dict], top_k: int) -> list[dict]:
    """Return up to top_k dictionary entries whose Spanish lemma appears in the source."""
    src_norm = normalize(source)
    src_tokens = set(re.findall(r"\w+", src_norm))
    matches = []
    for e in entries:
        es_norm = e["es_norm"]
        # Match if any token of the dict entry appears as a token in the source
        entry_tokens = set(re.findall(r"\w+", es_norm))
        if entry_tokens & src_tokens:
            matches.append(e)
        elif len(es_norm) >= 4 and es_norm in src_norm:
            matches.append(e)
    # Dedupe by Chanka form
    seen = set()
    unique = []
    for e in matches:
        key = e["qu"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    # Sort by longer Spanish lemma first (more specific)
    unique.sort(key=lambda e: -len(e["es"]))
    return unique[:top_k]


def format_retrieval_block(entries: list[dict]) -> str:
    if not entries:
        return ""
    lines = ["Diccionario (entradas relevantes):"]
    for e in entries:
        if e.get("note"):
            lines.append(f"  • {e['es']} → {e['qu']} ({e['note']})")
        else:
            lines.append(f"  • {e['es']} → {e['qu']}")
    return "\n".join(lines)


def build_prompt(tok, source: str, retrieval_block: str) -> str:
    user_content = (
        "Traduce del español al quechua chanka. Usa una traducción directa, "
        "fiel y apropiada para contexto judicial."
    )
    if retrieval_block:
        user_content = f"{retrieval_block}\n\n{user_content}"
    user_content += f"\n\nEspañol: {source}"
    msgs = [
        {"role": "system", "content": "Eres un traductor profesional español-quechua chanka."},
        {"role": "user", "content": user_content},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="merged 4B Chanka base (full_sft_sweeps ckpt-36)")
    ap.add_argument("--lora-adapter", required=True, help="v13 alpha160 champion LoRA")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--dictionary", required=True, help="cuzco_dictionary_es_to_quechua_lookup.json")
    ap.add_argument("--retrieval-top-k", type=int, default=8)
    ap.add_argument("--no-retrieval", action="store_true", help="ablation: run without retrieval block")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--dataset-repo", default="Thermostatic/rosettia-chanka-data")
    ap.add_argument("--chanka-file", default="clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet")
    ap.add_argument("--cache-dir", default="/root/.cache/hf-rosettia")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-metrics-json", required=True)
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading dictionary...", flush=True)
    entries = load_dictionary(args.dictionary)
    print(f"[{time.strftime('%H:%M:%S')}] dict entries: {len(entries)}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading Chanka eval rows...", flush=True)
    rows = load_chanka_rows(args.dataset_repo, args.chanka_file, args.cache_dir)
    eval_rows = split_eval(rows)
    print(f"[{time.strftime('%H:%M:%S')}] eval rows: {len(eval_rows)}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"[{time.strftime('%H:%M:%S')}] [Unsloth] loading 4B base + v13 LoRA adapter...", flush=True)
    import torch
    from unsloth import FastLanguageModel
    model, _tok = FastLanguageModel.from_pretrained(
        model_name=str(args.lora_adapter),  # loads adapter + its base_model from adapter_config
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _tok.padding_side = "left"
    model.generation_config.eos_token_id = _tok.eos_token_id
    model.generation_config.pad_token_id = _tok.eos_token_id
    FastLanguageModel.for_inference(model)
    tok = _tok  # use the model's tokenizer for the chat template

    print(f"[{time.strftime('%H:%M:%S')}] building prompts with retrieval...", flush=True)
    prompts = []
    retrievals = []
    for row in eval_rows:
        if args.no_retrieval:
            rb = ""
            ret_entries = []
        else:
            ret_entries = retrieve_for_source(row["source"], entries, args.retrieval_top_k)
            rb = format_retrieval_block(ret_entries)
        retrievals.append(ret_entries)
        prompts.append(build_prompt(tok, row["source"], rb))

    print("\n=== sample prompt ===")
    print(prompts[0])
    print("=== /sample prompt ===\n")

    print(f"[{time.strftime('%H:%M:%S')}] generating {len(prompts)} predictions...", flush=True)
    t0 = time.time()
    preds = []
    bs = 8
    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        inputs = tok(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        plen = inputs.input_ids.shape[1]
        for j in range(len(batch)):
            new_tokens = out[j, plen:]
            preds.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
        if (i // bs) % 4 == 0:
            print(f"  generated {i + len(batch)}/{len(prompts)}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] gen done in {time.time()-t0:.0f}s", flush=True)

    refs = [r["target"] for r in eval_rows]
    srcs = [r["source"] for r in eval_rows]

    import sacrebleu
    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds, [refs]).score
    bleu = sacrebleu.BLEU().corpus_score(preds, [refs]).score

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for src, ref, pred, ret in zip(srcs, refs, preds, retrievals):
            f.write(json.dumps({
                "source": src,
                "reference": ref,
                "prediction": pred,
                "n_retrieved": len(ret),
                "retrieved_entries": [{"es": e["es"], "qu": e["qu"]} for e in ret],
            }, ensure_ascii=False) + "\n")

    n_with_retrieval = sum(1 for r in retrievals if r)
    avg_retrieved = sum(len(r) for r in retrievals) / max(1, len(retrievals))
    metrics = {
        "n_rows": len(eval_rows),
        "no_retrieval": args.no_retrieval,
        "retrieval_top_k": args.retrieval_top_k,
        "n_sources_with_retrieval": n_with_retrieval,
        "avg_retrieved_per_source": avg_retrieved,
        "chrf++": chrfpp,
        "bleu": bleu,
    }
    Path(args.out_metrics_json).write_text(json.dumps(metrics, indent=2))

    print()
    print(f"=== v28 RAG smoke results ===")
    print(f"  no_retrieval={args.no_retrieval}")
    print(f"  chrF++={chrfpp:.3f}  BLEU={bleu:.3f}")
    print(f"  retrieval coverage: {n_with_retrieval}/{len(eval_rows)} sources, "
          f"avg {avg_retrieved:.2f} entries per source")


if __name__ == "__main__":
    main()
