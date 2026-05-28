"""V-STaR step 1: generate K candidates per training source via vLLM.

Uses temperature sampling to produce diverse candidates. Each candidate's chrF
vs reference is computed for downstream preference-pair construction.

Output JSONL: one row per (source, candidate_i), with chrF score.
"""
import argparse
import json
import random
import time
from pathlib import Path


def load_chanka_rows(repo: str, filename: str, cache_dir: str):
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


def split_train(rows, seed=3407, frac=0.15):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * frac))
    return shuffled[eval_size:]  # train portion only


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
    ap.add_argument("--base", required=True, help="merged model dir (LoRA base)")
    ap.add_argument("--lora-adapter", required=True)
    ap.add_argument("--lora-rank", type=int, default=256)
    ap.add_argument("--n-candidates", type=int, default=8, help="K candidates per source")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--dataset-repo", default="Thermostatic/rosettia-chanka-data")
    ap.add_argument("--chanka-file", default="clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet")
    ap.add_argument("--cache-dir", default="/root/.cache/hf-rosettia")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--limit", type=int, default=None, help="limit train rows for smoke testing")
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading Chanka rows...", flush=True)
    rows = load_chanka_rows(args.dataset_repo, args.chanka_file, args.cache_dir)
    train_rows = split_train(rows)
    if args.limit:
        train_rows = train_rows[: args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] train rows: {len(train_rows)}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM model from {args.base}...", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.base,
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        dtype="bfloat16",
        max_model_len=args.max_seq_length,
        gpu_memory_utilization=args.gpu_mem_frac,
        trust_remote_code=True,
    )
    lora = LoRARequest("v25_adapter", 1, args.lora_adapter)

    sampling = SamplingParams(
        n=args.n_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new,
        stop=["<|im_end|>", "<|endoftext|>"],
        # NOTE: not passing seed — let vLLM use random for each sample so K candidates diverge.
    )

    print(f"[{time.strftime('%H:%M:%S')}] building prompts...", flush=True)
    prompts = [build_prompt(tok, r["source"]) for r in train_rows]
    print(f"[{time.strftime('%H:%M:%S')}] generating {len(prompts)} × {args.n_candidates} = {len(prompts)*args.n_candidates} candidates...", flush=True)

    t0 = time.time()
    outs = llm.generate(prompts, sampling, lora_request=lora)
    elapsed = time.time() - t0
    n_total = len(prompts) * args.n_candidates
    print(f"[{time.strftime('%H:%M:%S')}] generation done in {elapsed:.0f}s ({n_total/elapsed:.1f} cand/s)", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] scoring chrF...", flush=True)
    import sacrebleu
    chrf_metric = sacrebleu.CHRF(word_order=2)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for row, output in zip(train_rows, outs):
            candidates = [o.text.strip() for o in output.outputs]
            chrf_scores = [chrf_metric.sentence_score(c, [row["target"]]).score for c in candidates]
            rec = {
                "source": row["source"],
                "reference": row["target"],
                "candidates": candidates,
                "chrf_scores": chrf_scores,
                "best_idx": max(range(len(candidates)), key=lambda i: chrf_scores[i]),
                "worst_idx": min(range(len(candidates)), key=lambda i: chrf_scores[i]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {args.out_jsonl}", flush=True)

    # Summary stats
    all_best = []
    all_worst = []
    all_oracle_gap = []
    for line in open(args.out_jsonl):
        d = json.loads(line)
        scores = d["chrf_scores"]
        all_best.append(max(scores))
        all_worst.append(min(scores))
        all_oracle_gap.append(max(scores) - min(scores))
    import statistics
    print(f"\n=== summary ===")
    print(f"avg best-of-{args.n_candidates} chrF: {statistics.mean(all_best):.2f}")
    print(f"avg worst-of-{args.n_candidates} chrF: {statistics.mean(all_worst):.2f}")
    print(f"avg oracle gap (best - worst): {statistics.mean(all_oracle_gap):.2f}")


if __name__ == "__main__":
    main()
