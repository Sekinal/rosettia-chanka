"""Run DeepSeek-V3 as a normalizer teacher with chain-of-thought reasoning.

Each input sentence produces a (trace, normalized) pair following the
strict output format in scripts/normalizer/teacher_brief.md.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


def load_brief(path: str) -> str:
    return Path(path).read_text()


def parse_response(text: str) -> tuple[str, str]:
    """Extract (think_trace, normalized_output) from the model's response."""
    think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think = think_m.group(1).strip() if think_m else ""
    # Look for "Normalized: ..." line after </think>
    after = text[think_m.end():] if think_m else text
    norm_m = re.search(r"Normalized\s*:\s*(.*?)(?:\n\n|\Z)", after, re.DOTALL)
    normalized = norm_m.group(1).strip() if norm_m else after.strip()
    # Strip surrounding quotes if the model wrapped the answer
    if normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]
    return think, normalized


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", default="scripts/normalizer/teacher_brief.md")
    ap.add_argument("--input-jsonl", required=True,
                    help="JSONL with a row per sentence; field 'input' = sentence to normalize.")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--api-base", default="https://api.deepseek.com")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY env var.")

    client = OpenAI(api_key=api_key, base_url=args.api_base)
    brief = load_brief(args.brief)

    rows = [json.loads(l) for l in open(args.input_jsonl)]
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"Labeling {len(rows)} rows with {args.model}…", flush=True)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as out:
        for i, row in enumerate(rows):
            sent = row["input"]
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": brief},
                        {"role": "user", "content": f"Normalize this Chanka sentence per the brief:\n\n{sent}"},
                    ],
                    temperature=args.temperature,
                    max_tokens=2048,
                )
                raw = resp.choices[0].message.content
                think, normalized = parse_response(raw)
                rec = {
                    "idx": row.get("idx", i),
                    "source": row.get("source"),
                    "input": sent,
                    "expected": row.get("expected"),
                    "trace": think,
                    "normalized": normalized,
                    "teacher": f"deepseek/{args.model}",
                    "elapsed_sec": time.time() - t0,
                    "raw": raw,
                }
            except Exception as e:
                rec = {
                    "idx": row.get("idx", i),
                    "source": row.get("source"),
                    "input": sent,
                    "expected": row.get("expected"),
                    "trace": "",
                    "normalized": "",
                    "teacher": f"deepseek/{args.model}",
                    "error": str(e),
                }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i+1:3}/{len(rows)}] {rec.get('elapsed_sec','?'):.1f}s   {sent[:60]}…", flush=True)

    print(f"\nWrote {args.out_jsonl}")


if __name__ == "__main__":
    main()
