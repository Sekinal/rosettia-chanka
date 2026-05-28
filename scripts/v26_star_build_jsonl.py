"""Format v26 STaR traces into a training JSONL.

Two output modes:
  - --target-format combined: target = "Razonamiento: <reasoning>\\n\\nTraducción: <chanka>"
    Standard SFT learns to produce reasoning AND translation. At inference, parse out
    the post-"Traducción:" segment.
  - --target-format translation_only: target = chanka only, reasoning goes into the user
    prompt as context. No loss on reasoning. Eval is plain.

Default is "combined" (true STaR — model generates its own reasoning then conditions on it).
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="v26 traces JSONL from v26_star_generate_traces.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-format", choices=["combined", "translation_only"], default="combined")
    ap.add_argument(
        "--min-reasoning-chars",
        type=int,
        default=30,
        help="Drop rows whose reasoning is shorter than this (degenerate generations).",
    )
    args = ap.parse_args()

    rows_in = [json.loads(line) for line in open(args.input)]
    n_dropped = 0
    rows_out = []
    for r in rows_in:
        reasoning = r["reasoning"].strip()
        if len(reasoning) < args.min_reasoning_chars:
            n_dropped += 1
            continue
        target = r["star_target"]
        if args.target_format == "combined":
            train_target = f"Razonamiento: {reasoning}\n\nTraducción: {target}"
        else:
            train_target = target
        rec = {
            "source": r["source"],
            "target": train_target,
            "source_name": "v26_star",
            "variant": "quy/chanka",
            "v25_chrf_vs_reference": r["v25_chrf_vs_reference"],
            "is_match": r["is_match"],
            "raw_reasoning": reasoning,
            "raw_translation_target": target,
        }
        rows_out.append(rec)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in rows_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_match = sum(1 for r in rows_in if r.get("is_match"))
    print(f"input: {len(rows_in)} rows ({n_match} matches, {len(rows_in)-n_match} misses)")
    print(f"dropped {n_dropped} rows with short reasoning")
    print(f"wrote {len(rows_out)} rows to {args.output} (target-format={args.target_format})")


if __name__ == "__main__":
    main()
