"""Run the trained Chanka normalizer over a corpus and produce normalized text.

For each input row, generates `<think>...</think>\\n\\nNormalized: <out>` and
extracts the normalized portion as the canonical Chanka form.

Usage:
  python apply_normalizer_vllm.py --base unsloth/Qwen3.5-4B --adapter <path> \\
    --input-jsonl <corpus.jsonl> --input-field chanka \\
    --out-jsonl <out.jsonl>
"""
import argparse
import json
import re
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert Chanka (Ayacucho) Quechua orthographic normalizer "
    "following the MINEDU 2021 standard. For each input, emit a <think>...</think> "
    "trace that lists every token left-to-right with the spec rule cited (R1-R7, "
    "S1-S8, L0-L3, §6.5, §6.6, §8.5, §8.6), then a 'Normalized:' line with the "
    "canonical sentence."
)


def make_chat_prompt(tokenizer, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_normalized(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"Normalized\s*:\s*(.+?)(?:\n|$)", text, flags=re.DOTALL)
    return m.group(1).strip().strip('"') if m else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--input-field", default="chanka", help="Field name in input JSONL")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128, help="vLLM ingests internally, but limit batch for memory.")
    args = ap.parse_args()

    rows = []
    for line in open(args.input_jsonl):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows.")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.base,
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem_frac,
        trust_remote_code=True,
        enforce_eager=True,
    )
    lora = LoRARequest("normalizer", 1, args.adapter)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new,
                              stop=["<|im_end|>", "<|endoftext|>"])

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    t0 = time.time()
    with open(args.out_jsonl, "w") as out:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [
                make_chat_prompt(tok, SYSTEM_PROMPT,
                                 f"Normalize this Chanka sentence per the MINEDU 2021 spec:\n\n{r[args.input_field]}")
                for r in batch
            ]
            outs = llm.generate(prompts, sampling, lora_request=lora)
            for r, o in zip(batch, outs):
                raw = o.outputs[0].text
                normalized = extract_normalized(raw)
                rec = dict(r)
                rec["original"] = r[args.input_field]
                rec[args.input_field + "_normalized"] = normalized
                rec["normalizer_changed"] = (normalized != r[args.input_field])
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += len(batch)
            elapsed = time.time() - t0
            print(f"[{n_done:6d}/{len(rows)}]  {elapsed:.0f}s  {n_done/elapsed:.1f} rows/s", flush=True)

    print(f"\nDone. Wrote {n_done} rows to {args.out_jsonl} in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
