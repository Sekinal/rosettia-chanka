"""V-STaR step 4: Best-of-K inference on the 158-row Chanka held-out.

  1. Generate K candidates per held-out source via vLLM (using v25 generator)
  2. Score each candidate's log-likelihood under the verifier LoRA (PEFT model)
  3. Pick the candidate with highest verifier log-likelihood
  4. Report chrF/BLEU of the verifier-selected vs:
     - random pick (sanity baseline)
     - oracle (best-by-chrF, upper bound)
     - v25 single greedy (status quo baseline)
"""
import argparse
import json
import random
import time
from pathlib import Path


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


def build_prompt(tok, source):
    msgs = [
        {"role": "system", "content": "Eres un traductor profesional español-quechua chanka."},
        {"role": "user", "content": (
            "Traduce del español al quechua chanka. Usa una traducción directa, "
            f"fiel y apropiada para contexto judicial.\n\nEspañol: {source}"
        )},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="v25b merged base")
    ap.add_argument("--gen-lora", required=True, help="generator LoRA (v25 alpha1024)")
    ap.add_argument("--verifier-lora", required=True, help="verifier LoRA (v27 DPO output)")
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.80)
    ap.add_argument("--lora-rank-max", type=int, default=256)
    ap.add_argument("--dataset-repo", default="Thermostatic/rosettia-chanka-data")
    ap.add_argument("--chanka-file", default="clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet")
    ap.add_argument("--cache-dir", default="/root/.cache/hf-rosettia")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-metrics-json", required=True)
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading Chanka eval rows...", flush=True)
    rows = load_chanka_rows(args.dataset_repo, args.chanka_file, args.cache_dir)
    eval_rows = split_eval(rows)
    print(f"[{time.strftime('%H:%M:%S')}] eval rows: {len(eval_rows)}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    # ---- Step 1: generate K candidates with vLLM + generator LoRA ----
    print(f"[{time.strftime('%H:%M:%S')}] [vLLM] loading model + generator LoRA...", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(
        model=args.base,
        enable_lora=True,
        max_lora_rank=args.lora_rank_max,
        dtype="bfloat16",
        max_model_len=args.max_seq_length,
        gpu_memory_utilization=args.gpu_mem_frac,
        trust_remote_code=True,
    )
    gen_lora = LoRARequest("generator", 1, args.gen_lora)
    sampling = SamplingParams(
        n=args.n_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    prompts = [build_prompt(tok, r["source"]) for r in eval_rows]
    print(f"[{time.strftime('%H:%M:%S')}] generating {len(prompts)} × {args.n_candidates} candidates...", flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, sampling, lora_request=gen_lora)
    print(f"[{time.strftime('%H:%M:%S')}] gen done in {time.time()-t0:.0f}s", flush=True)

    all_candidates = []
    for output in outs:
        all_candidates.append([o.text.strip() for o in output.outputs])

    # ---- Free vLLM, load verifier via PEFT for log-likelihood scoring ----
    print(f"[{time.strftime('%H:%M:%S')}] releasing vLLM, loading verifier via Unsloth...", flush=True)
    import gc, torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    from unsloth import FastLanguageModel
    ver_model, ver_tok = FastLanguageModel.from_pretrained(
        model_name=args.verifier_lora,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if ver_tok.pad_token is None:
        ver_tok.pad_token = ver_tok.eos_token
    ver_tok.padding_side = "right"
    FastLanguageModel.for_inference(ver_model)

    # ---- Step 2: score each candidate's log-likelihood under verifier ----
    print(f"[{time.strftime('%H:%M:%S')}] scoring candidates under verifier...", flush=True)

    @torch.no_grad()
    def verifier_logprob(prompt: str, completion: str) -> float:
        full = prompt + completion
        prompt_ids = ver_tok(text=prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        full_ids = ver_tok(text=full, return_tensors="pt", add_special_tokens=False).input_ids[0]
        comp_len = full_ids.shape[0] - prompt_ids.shape[0]
        if comp_len <= 0:
            return 0.0
        ids = full_ids.unsqueeze(0).to(ver_model.device)
        logits = ver_model(ids).logits  # [1, T, V]
        # log-prob of token t given context up to t-1: logits[t-1] predicting ids[t]
        targets = ids[:, 1:]
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
        comp_lp = token_lp[:, -comp_len:].sum().item()
        return comp_lp / max(1, comp_len)  # length-normalized

    selected_verifier = []
    selected_random = []
    selected_oracle = []
    selected_greedy = []
    all_scored_rows = []

    rng = random.Random(args.dataset_repo.encode().hex()[:8] if isinstance(args.dataset_repo, str) else 42)
    import sacrebleu
    chrf_metric = sacrebleu.CHRF(word_order=2)

    for i, (row, cands) in enumerate(zip(eval_rows, all_candidates)):
        prompt = prompts[i]
        # verifier scores
        scores = [verifier_logprob(prompt, c) for c in cands]
        chrfs = [chrf_metric.sentence_score(c, [row["target"]]).score for c in cands]
        ver_idx = max(range(len(cands)), key=lambda i: scores[i])
        rand_idx = rng.randrange(len(cands))
        oracle_idx = max(range(len(cands)), key=lambda i: chrfs[i])
        selected_verifier.append(cands[ver_idx])
        selected_random.append(cands[rand_idx])
        selected_oracle.append(cands[oracle_idx])
        # Greedy proxy: pick the candidate with the most-common output (mode); approximation
        from collections import Counter
        most_common = Counter(cands).most_common(1)[0][0]
        selected_greedy.append(most_common)
        rec = {
            "source": row["source"],
            "reference": row["target"],
            "candidates": cands,
            "verifier_scores": scores,
            "chrf_scores": chrfs,
            "verifier_pick": cands[ver_idx],
            "oracle_pick": cands[oracle_idx],
            "greedy_proxy_pick": most_common,
            "ver_idx": ver_idx,
            "oracle_idx": oracle_idx,
        }
        all_scored_rows.append(rec)
        if (i + 1) % 16 == 0:
            print(f"  scored {i+1}/{len(eval_rows)}", flush=True)

    refs = [r["target"] for r in eval_rows]
    metrics = {
        "n_rows": len(eval_rows),
        "n_candidates": args.n_candidates,
        "verifier_pick": {
            "chrf++": sacrebleu.CHRF(word_order=2).corpus_score(selected_verifier, [refs]).score,
            "bleu": sacrebleu.BLEU().corpus_score(selected_verifier, [refs]).score,
        },
        "random_pick": {
            "chrf++": sacrebleu.CHRF(word_order=2).corpus_score(selected_random, [refs]).score,
            "bleu": sacrebleu.BLEU().corpus_score(selected_random, [refs]).score,
        },
        "oracle_pick": {
            "chrf++": sacrebleu.CHRF(word_order=2).corpus_score(selected_oracle, [refs]).score,
            "bleu": sacrebleu.BLEU().corpus_score(selected_oracle, [refs]).score,
        },
        "mode_pick": {
            "chrf++": sacrebleu.CHRF(word_order=2).corpus_score(selected_greedy, [refs]).score,
            "bleu": sacrebleu.BLEU().corpus_score(selected_greedy, [refs]).score,
        },
    }

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in all_scored_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(args.out_metrics_json).write_text(json.dumps(metrics, indent=2))

    print()
    print("=== V-STaR Best-of-K results ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}: chrF++={v['chrf++']:.3f}  BLEU={v['bleu']:.3f}")


if __name__ == "__main__":
    main()
