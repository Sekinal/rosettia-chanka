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


def _safe_token_transform(tok: str) -> str:
    """Apply ONLY the bulletproof, reversible character-level Chanka ops to one
    token: strip apostrophes (R1), de-aspirate (R2), e→i / o→u (R3), l→ll before
    q (R6). No morphology, no suffix edits, no splits. This is the ONLY edit the
    model is permitted to trigger on a token."""
    out = tok.replace("'", "").replace("’", "")
    out = re.sub(r"chh", "ch", out); out = re.sub(r"kh", "k", out)
    out = re.sub(r"ph", "p", out);  out = re.sub(r"qh", "q", out)
    out = re.sub(r"th", "t", out)
    out = (out.replace("e", "i").replace("o", "u")
              .replace("E", "I").replace("O", "U"))
    out = re.sub(r"(?<!l)l(q)", r"ll\1", out)
    return out


def verified_normalize(original: str, proposed: str) -> tuple[str, str | None]:
    """Token-aligned verification. The model SELECTS which tokens to change
    (its context judgment protects loans / proper nouns), but each accepted
    edit must EQUAL the deterministic safe transform of that token. Any edit
    that isn't a known-safe transform is a corruption → keep the original token.

    Requires same token count (no splits/merges). If counts differ, reject the
    whole sentence (keep original)."""
    o_toks, p_toks = original.split(), proposed.split()
    if len(o_toks) != len(p_toks):
        return original, "token_count_mismatch"
    out = []
    vetoed = 0
    for ot, pt in zip(o_toks, p_toks):
        if ot == pt:
            out.append(ot)
            continue
        # model wants to change this token — only allow the safe transform
        if pt == _safe_token_transform(ot):
            out.append(pt)
        else:
            out.append(ot)  # corruption (suffix swap, q/k, hallucination) → keep
            vetoed += 1
    result = " ".join(out)
    return result, (f"vetoed_{vetoed}_tokens" if vetoed else None)


def safe_accept(original: str, proposed: str) -> tuple[str, str | None]:
    """Veto corrupting normalizations. Returns (accepted_text, reject_reason).

    The model (v43b) reaches 96.9% on clean single-rule inputs but on real
    corpus text it sometimes (a) INSERTS Cuzco apostrophes into clean Chanka,
    (b) over-splits agglutinated words, or (c) hallucinates extra text. All
    three are deterministically detectable; on any of them we keep the original.
    """
    o, p = original.strip(), proposed.strip()
    if o == p:
        return o, None
    # (a) Apostrophe insertion — Chanka can only ever LOSE apostrophes.
    n_apos_o = o.count("'") + o.count("’")
    n_apos_p = p.count("'") + p.count("’")
    if n_apos_p > n_apos_o:
        return o, "apostrophe_inserted"
    # (b) Token-count increase (over-split). Genuine compound splits are rare;
    #     a normalizer pass that ADDS tokens is almost always wrong on Chanka
    #     running text. Allow a decrease (nisqa-strip merges) but not increase.
    if len(p.split()) > len(o.split()):
        return o, "token_count_increased"
    # (c) Hallucination / length explosion — accept only modest length deltas.
    if len(p) > len(o) + 3:
        return o, "length_explosion"
    # (d) Aspirated-digraph INSERTION (model added h to make chh/kh/ph/qh/th).
    def n_aspir(s): return len(re.findall(r"(chh|kh|ph|qh|th)", s))
    if n_aspir(p) > n_aspir(o):
        return o, "aspirate_inserted"
    return p, None


def extract_normalized(text: str, fallback: str | None = None) -> str:
    """Pull the 'Normalized: ...' line. If the model emitted only a trace
    (no Normalized: line), fall back to the original input rather than dumping
    raw trace text into the corpus."""
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"Normalized\s*:\s*(.+?)(?:\n|$)", body, flags=re.DOTALL)
    if m:
        return m.group(1).strip().strip('"')
    # No Normalized: line. If a trace block is still present, the model never
    # produced an answer → safest is identity (preserve input).
    if "<think>" in text or "Tokens:" in text or fallback is not None:
        return fallback if fallback is not None else body.strip()
    return body.strip()


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
                proposed = extract_normalized(raw, fallback=r[args.input_field])
                normalized, reject_reason = safe_accept(r[args.input_field], proposed)
                rec = dict(r)
                rec["safety_reject_reason"] = reject_reason
                rec["model_proposed"] = proposed
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
