"""V-STaR step 2: build (source, winner, loser) preference pairs from candidates.

Filters:
  - Drop sources where best-worst chrF gap < min_gap (no signal)
  - For each kept source: emit P pairs by sampling (winner, loser) from chrF-ranked candidates
  - Winner = highest chrF candidate; loser = any candidate with chrF lower by ≥ min_gap

Output JSONL row format compatible with TRL DPOTrainer:
  {prompt: <full_chat_template_prompt>, chosen: <winner>, rejected: <loser>, ...}
"""
import argparse
import json
import random
from pathlib import Path


SYSTEM = "Eres un traductor profesional español-quechua chanka."
INSTR = (
    "Traduce del español al quechua chanka. Usa una traducción directa, "
    "fiel y apropiada para contexto judicial."
)


def build_prompt(tok, source):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{INSTR}\n\nEspañol: {source}"},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-jsonl", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--min-gap", type=float, default=5.0, help="min chrF gap to count as a learning pair")
    ap.add_argument("--pairs-per-source", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows = [json.loads(line) for line in open(args.candidates_jsonl)]
    print(f"Loaded {len(rows)} sources with candidates")

    n_kept_sources = 0
    n_pairs = 0
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out_jsonl, "w")
    for row in rows:
        cands = row["candidates"]
        scores = row["chrf_scores"]
        if max(scores) - min(scores) < args.min_gap:
            continue
        # Rank by chrF descending
        ranked = sorted(range(len(cands)), key=lambda i: -scores[i])
        winners = [(cands[i], scores[i]) for i in ranked if scores[i] >= max(scores) - 0.5]  # near-best
        losers = [(cands[i], scores[i]) for i in ranked if scores[i] <= max(scores) - args.min_gap]
        if not winners or not losers:
            continue
        n_kept_sources += 1
        prompt = build_prompt(tok, row["source"])
        # Sample up to pairs_per_source unique (winner, loser) combinations
        seen = set()
        for _ in range(args.pairs_per_source * 4):  # try more, dedupe
            w_text, w_score = rng.choice(winners)
            l_text, l_score = rng.choice(losers)
            key = (w_text, l_text)
            if key in seen or w_text == l_text:
                continue
            seen.add(key)
            rec = {
                "prompt": prompt,
                "chosen": w_text,
                "rejected": l_text,
                "source": row["source"],
                "reference": row["reference"],
                "chosen_chrf": w_score,
                "rejected_chrf": l_score,
                "chrf_gap": w_score - l_score,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_pairs += 1
            if len(seen) >= args.pairs_per_source:
                break
    out_f.close()

    print(f"Kept {n_kept_sources}/{len(rows)} sources after min-gap filter")
    print(f"Wrote {n_pairs} preference pairs to {args.out_jsonl}")


if __name__ == "__main__":
    main()
