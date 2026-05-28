"""Fast vLLM eval against AmericasNLP 2021 spanish→quy test (1003 lines).

Same as eval_americasnlp_2021.py but uses vLLM (5-15x faster).

Loads the adapter's `base_model_name_or_path` automatically from adapter_config.json
so caller only specifies --adapter.
"""
import argparse
import json
import time
from pathlib import Path


SYSTEM = "Eres un traductor profesional español-quechua chanka."
INSTR = (
    "Traduce del español al quechua chanka. Usa una traducción directa, "
    "fiel y apropiada para contexto judicial."
)


def main():
    def parse_args():
        ap = argparse.ArgumentParser()
        ap.add_argument("--base", required=True, help="merged base model dir (the LoRA's base_model_name_or_path)")
        ap.add_argument("--adapter", required=True, help="LoRA adapter path")
        ap.add_argument("--lora-rank", type=int, default=256)
        ap.add_argument("--test-es", default="docs/references/americasnlp_test/2021_test.es")
        ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
        ap.add_argument("--max-seq-length", type=int, default=512)
        ap.add_argument("--max-new", type=int, default=128)
        ap.add_argument("--gpu-mem-frac", type=float, default=0.80)
        ap.add_argument("--out-json", required=True)
        ap.add_argument("--out-predictions-jsonl", default=None)
        return ap.parse_args()

    args = parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading test data...", flush=True)
    src_lines = open(args.test_es).read().strip().split("\n")
    ref_lines = open(args.test_quy).read().strip().split("\n")
    assert len(src_lines) == len(ref_lines), f"{len(src_lines)} != {len(ref_lines)}"
    print(f"[{time.strftime('%H:%M:%S')}] test rows: {len(src_lines)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM (base={args.base}, lora={args.adapter})...", flush=True)
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
    lora = LoRARequest("adapter", 1, args.adapter)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    prompts = []
    for s in src_lines:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{INSTR}\n\nEspañol: {s}"},
        ]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)

    print(f"[{time.strftime('%H:%M:%S')}] generating {len(prompts)} translations...", flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, sampling, lora_request=lora)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] gen done in {elapsed:.0f}s ({len(prompts)/elapsed:.1f} sent/s)", flush=True)

    preds = [o.outputs[0].text.strip() for o in outs]

    import sacrebleu
    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds, [ref_lines]).score
    bleu = sacrebleu.BLEU().corpus_score(preds, [ref_lines]).score
    sent_chrf = [sacrebleu.CHRF(word_order=2).sentence_score(p, [r]).score for p, r in zip(preds, ref_lines)]
    import statistics
    rec = {
        "adapter": str(args.adapter),
        "base": str(args.base),
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
                f.write(json.dumps({"source": src, "reference": ref, "prediction": pred}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
