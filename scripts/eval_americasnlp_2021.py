"""Evaluate a v30 (or any) Chanka adapter against AmericasNLP 2021 Spanish→quy test set.

This is the cleanest public benchmark we have for Spanish→Chanka MT:
  - 1003 test sentences released by the AmericasNLP shared task organizers
  - Verified ZERO overlap with our training data (americasnlp+somosnlp broad + reviewed chanka)
  - Same eval metric (chrF++) as the shared task

Usage:
  python eval_americasnlp_2021.py \
    --adapter outputs/v30c_9b_compact_mixed_expanded_20260527-m2/checkpoint-XYZ \
    --out-json outputs/eval_americasnlp_2021/v30c_ckptXYZ.json
"""
import argparse
import json
import time
from pathlib import Path

import sacrebleu
import torch


SYSTEM = "Eres un traductor profesional español-quechua chanka."
INSTR = (
    "Traduce del español al quechua chanka. Usa una traducción directa, "
    "fiel y apropiada para contexto judicial."
)


def build_prompt(tok, source: str) -> str:
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
    ap.add_argument("--adapter", required=True, help="LoRA adapter path (e.g. v30c best ckpt)")
    ap.add_argument("--test-es", default="docs/references/americasnlp_test/2021_test.es")
    ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-predictions-jsonl", default=None)
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading test data...", flush=True)
    src_lines = open(args.test_es).read().strip().split("\n")
    ref_lines = open(args.test_quy).read().strip().split("\n")
    assert len(src_lines) == len(ref_lines), f"{len(src_lines)} != {len(ref_lines)}"
    print(f"[{time.strftime('%H:%M:%S')}] test rows: {len(src_lines)}")

    print(f"[{time.strftime('%H:%M:%S')}] loading adapter via Unsloth...", flush=True)
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

    print(f"[{time.strftime('%H:%M:%S')}] generating {len(src_lines)} translations...", flush=True)
    t0 = time.time()
    preds = []
    pad_id = tok.pad_token_id or tok.eos_token_id
    for i in range(0, len(src_lines), args.batch_size):
        batch = src_lines[i : i + args.batch_size]
        prompts = [build_prompt(tok, s) for s in batch]
        inputs = tok(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False, pad_token_id=pad_id)
        plen = inputs.input_ids.shape[1]
        for j in range(len(batch)):
            new_tokens = out[j, plen:]
            preds.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
        if (i // args.batch_size) % 8 == 0:
            print(f"  {i + len(batch)}/{len(src_lines)}", flush=True)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] generation done in {elapsed:.0f}s", flush=True)

    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds, [ref_lines]).score
    bleu = sacrebleu.BLEU().corpus_score(preds, [ref_lines]).score
    # Also compute sentence-level chrF distribution
    sent_chrf = [sacrebleu.CHRF(word_order=2).sentence_score(p, [r]).score for p, r in zip(preds, ref_lines)]
    import statistics
    rec = {
        "adapter": str(args.adapter),
        "benchmark": "AmericasNLP 2021 spanish→quy test (1003 lines)",
        "n_rows": len(src_lines),
        "chrf++": chrfpp,
        "bleu": bleu,
        "sent_chrf_mean": statistics.mean(sent_chrf),
        "sent_chrf_median": statistics.median(sent_chrf),
        "sent_chrf_p10": statistics.quantiles(sent_chrf, n=10)[0],
        "sent_chrf_p90": statistics.quantiles(sent_chrf, n=10)[8],
        "elapsed_sec": elapsed,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rec, indent=2))
    print(f"\n=== AmericasNLP 2021 spanish→quy results ===")
    print(f"  chrF++={chrfpp:.3f}  BLEU={bleu:.3f}")
    print(f"  sentence chrF: mean={rec['sent_chrf_mean']:.2f}  median={rec['sent_chrf_median']:.2f}")
    print(f"  P10={rec['sent_chrf_p10']:.2f}  P90={rec['sent_chrf_p90']:.2f}")

    if args.out_predictions_jsonl:
        with open(args.out_predictions_jsonl, "w") as f:
            for src, ref, pred in zip(src_lines, ref_lines, preds):
                f.write(json.dumps({
                    "source": src, "reference": ref, "prediction": pred,
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
