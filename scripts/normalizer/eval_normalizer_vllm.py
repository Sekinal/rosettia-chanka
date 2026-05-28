"""Evaluate a fine-tuned Chanka normalizer (Qwen3.5-4B LoRA) on:
  1. The §7 gold wrong→right list from the spec (deterministic accuracy).
  2. A held-out subset of the multi-teacher gold (output-level exact match).
  3. Optional: round-trip identity on a MINEDU-compliant corpus (no changes).

Uses vLLM with LoRA adapter.
"""
import argparse
import json
import re
import time
from pathlib import Path


# §7 wrong→right gold list from the spec doc. Each entry is the spec-prescribed
# normalization that the model MUST produce. Each row gives one sentence-shaped
# input.
GOLD_PAIRS = [
    # R1 — strip apostrophes
    ("p'acha sumaqmi", "pacha sumaqmi"),
    ("t'anta munanim", "tanta munanim"),
    ("mayt'u rikch'aymi", "maytu rikchaymi"),
    ("q'illu pacham", "qillu pacham"),
    ("k'aspi yaykurqa", "kaspi yaykurqa"),
    ("ch'uklla wasiykim", "chuklla wasiykim"),
    ("sach'a hatunmi", "sacha hatunmi"),
    ("hap'iy ñawiykita", "hapiy ñawiykita"),
    ("mut'i mikuni", "muti mikuni"),
    ("llamk'ay sumaqmi", "llamkay sumaqmi"),
    ("mink'a allinmi", "minka allinmi"),
    # R2 — de-aspirate
    ("phullu wayrachkan", "pullu wayrachkan"),
    ("thuqay manam allinchu", "tuqay manam allinchu"),
    ("chhalla mikuni", "chala mikuni"),
    ("qhari hamuchkan", "qari hamuchkan"),
    ("mikhuy sumaqmi", "mikuy sumaqmi"),
    ("khuchi puriykuchkan", "kuchi puriykuchkan"),
    ("llanthu hamuchkan", "llantu hamuchkan"),
    ("qhaway taytaykita", "qaway taytaykita"),
    # R3 — 3-vowel collapse
    ("ñoqa hamurqani", "ñuqa hamurqani"),
    ("qelqay sumaqmi", "qillqay sumaqmi"),
    ("qosqo llaqtam", "qusqu llaqtam"),
    ("sonqo nanachkawan", "sunqu nanachkawan"),
    ("qollqe pisillam", "qullqi pisillam"),
    ("onqoy hamuchkan", "unquy hamuchkan"),
    ("lloqsiy wasimanta", "lluqsiy wasimanta"),
    # R4 — forbidden grapheme replace
    ("jam munankichu", "qam munankichu"),
    ("jucha kachkan", "hucha kachkan"),
    ("jatun runam", "hatun runam"),
    ("quilla puriykuchkan", "killa puriykuchkan"),
    ("camay sumaqmi", "kamay sumaqmi"),
    ("fukuy phukuy", "pukuy pukuy"),
    # R5 — diphthong breaking
    ("huaita allinmi", "wayta allinmi"),
    ("yahuar miskimi", "yawar miskimi"),
    ("huata sumaqmi", "wata sumaqmi"),
    # R6 — ll before q
    ("qulqi pisilla", "qullqi pisilla"),
    ("walqa sumaqmi", "wallqa sumaqmi"),
    ("alqu hamuchkan", "allqu hamuchkan"),
    # R7 — m/n before p
    ("qan hamunki", "qam hamunki"),
    ("allim ruwarqanki", "allin ruwarqanki"),
    # L1c — recent loans keep Spanish; strip nisqa
    ("Televisor nisqatam rantirqani", "Televisortam rantirqani"),
    ("computadora nisqapi llamkani", "computadorapi llamkani"),
    # L1b — refonologized loans canonical
    ("vaca hatunmi", "waka hatunmi"),
    ("caballo allinmi", "kawallu allinmi"),
    # L2 — proper nouns preserved
    ("Pedrochaqa ripunqam", "Pedrochaqa ripunqam"),
    # L3 — toponym fixed
    ("Cuzco llaqtam sumaq", "Qusqu llaqtam sumaq"),
    # S2 — suffix preservation
    ("mamanchix kuchkan", "mamanchik kuchkan"),
    # §8.5 conservative fallback — religious proper nouns preserved
    ("Jehová Diosqa rimarqan", "Jehová Diosqa rimarqan"),
    ("Jesus hamurqan llaqtaman", "Jesus hamurqan llaqtaman"),
    ("Dios munakuwanchik", "Dios munakuwanchik"),
    # §8.5 — acronyms preserved
    ("MINEDU kamachikuq", "MINEDU kamachikuq"),
    ("UNESCO yanapan", "UNESCO yanapan"),
    # §L0 — nisqa NOT stripped after foreign technical
    ("Fe nisqaqa qillaymi", "Fe nisqaqa qillaymi"),
    ("Falco sparverius nisqaqa pisqum", "Falco sparverius nisqaqa pisqum"),
    # §8.6 — Spanish multi-word span preserved
    ('Atuqsi nirqa "Dios mío" nispa', 'Atuqsi nirqa "Dios mío" nispa'),
    # §L3 — Quechua surname in Spanish form preserved (the L3 carve-out we added)
    ("Tayta Huamanmi rimarqan", "Tayta Huamanmi rimarqan"),
    ("Quispe wawqiymi", "Quispe wawqiymi"),
    # §6.5 — numerals unchanged
    ("20 wawakuna hamurqaku", "20 wawakuna hamurqaku"),
    ("iskay chunka wawakuna hamurqaku", "iskay chunka wawakuna hamurqaku"),
    # No-change case: already MINEDU-compliant
    ("Allillanmi taytáy", "Allillanmi taytáy"),
    ("Ñuqa kachkani wasipi", "Ñuqa kachkani wasipi"),
    # Multi-rule combined
    ("Ñoqa qosqopi kachkani", "Ñuqa qusqupi kachkani"),
    ("ñoqamanta phullu sumaqmi", "ñuqamanta pullu sumaqmi"),
    ("Tayta Inti qhawayta munachkan", "Tayta Inti qawayta munachkan"),
]


def _safe_token_transform(tok: str) -> str:
    out = tok.replace("'", "").replace("’", "")
    for a, b in [("chh", "ch"), ("kh", "k"), ("ph", "p"), ("qh", "q"), ("th", "t")]:
        out = out.replace(a, b)
    out = out.replace("e", "i").replace("o", "u").replace("E", "I").replace("O", "U")
    import re as _re
    out = _re.sub(r"(?<!l)l(q)", r"ll\1", out)
    return out


def count_corruptions(original: str, proposed: str) -> int:
    """Count tokens the model changed to something that is NOT the safe transform
    (i.e. morphology corruption, suffix swap, hallucination, apostrophe/aspirate
    insertion). Token-count mismatch counts as a full-sentence corruption."""
    o, p = original.split(), proposed.split()
    if len(o) != len(p):
        return max(len(o), len(p))  # split/merge/hallucination
    bad = 0
    for ot, pt in zip(o, p):
        if ot != pt and pt != _safe_token_transform(ot):
            bad += 1
    return bad


def make_chat_prompt(tokenizer, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_normalized(text: str) -> str:
    """Parse 'Normalized: ...' line out of the assistant response."""
    # Strip <think>...</think> block if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"Normalized\s*:\s*(.+?)(?:\n|$)", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip().strip('"')
    return text.strip()


SYSTEM_PROMPT = (
    "You are an expert Chanka (Ayacucho) Quechua orthographic normalizer "
    "following the MINEDU 2021 standard. For each input, emit a <think>...</think> "
    "trace that lists every token left-to-right with the spec rule cited (R1-R7, "
    "S1-S8, L0-L3, §6.5, §6.6, §8.5, §8.6), then a 'Normalized:' line with the "
    "canonical sentence."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Qwen3.5-4B base model dir (or HF id)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir")
    ap.add_argument("--gold-jsonl", default=None, help="Held-out gold JSONL (input,trace,normalized)")
    ap.add_argument("--n-heldout", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-jsonl", default=None)
    ap.add_argument("--clean-ref-file", default="docs/references/americasnlp_test/2021_test.quy",
                    help="Clean MINEDU-compliant refs for the precision/corruption metric.")
    ap.add_argument("--n-clean", type=int, default=150)
    args = ap.parse_args()

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
        enforce_eager=True,  # skip torch.compile to avoid hang on Qwen3.5 multimodal
    )
    lora = LoRARequest("normalizer", 1, args.adapter)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new,
                              stop=["<|im_end|>", "<|endoftext|>"])

    # ============================================================
    # 1) Spec §7 gold list accuracy
    # ============================================================
    prompts = [make_chat_prompt(tok, SYSTEM_PROMPT,
                                f"Normalize this Chanka sentence per the MINEDU 2021 spec:\n\n{src}")
               for src, _ in GOLD_PAIRS]
    t0 = time.time()
    outs = llm.generate(prompts, sampling, lora_request=lora)
    elapsed_spec = time.time() - t0

    spec_results = []
    spec_correct = 0
    for (src, expected), out in zip(GOLD_PAIRS, outs):
        raw = out.outputs[0].text
        pred = extract_normalized(raw)
        ok = pred.strip() == expected.strip()
        spec_correct += int(ok)
        spec_results.append({"input": src, "expected": expected, "predicted": pred, "ok": ok, "raw": raw})
    spec_acc = spec_correct / len(GOLD_PAIRS)
    print(f"\n=== §7 GOLD LIST: {spec_correct}/{len(GOLD_PAIRS)} = {spec_acc*100:.1f}% ===")

    # ============================================================
    # 2) Held-out gold accuracy
    # ============================================================
    heldout_results = []
    heldout_acc = None
    if args.gold_jsonl:
        rows = []
        with open(args.gold_jsonl) as f:
            for line in f:
                if not line.strip(): continue
                rows.append(json.loads(line))
        rows = rows[: args.n_heldout]
        prompts = [make_chat_prompt(tok, SYSTEM_PROMPT,
                                    f"Normalize this Chanka sentence per the MINEDU 2021 spec:\n\n{r['input']}")
                   for r in rows]
        t0 = time.time()
        outs = llm.generate(prompts, sampling, lora_request=lora)
        elapsed_heldout = time.time() - t0
        ho_correct = 0
        for r, out in zip(rows, outs):
            raw = out.outputs[0].text
            pred = extract_normalized(raw)
            expected = r["normalized"]
            ok = pred.strip() == expected.strip()
            ho_correct += int(ok)
            heldout_results.append({"input": r["input"], "expected": expected, "predicted": pred,
                                    "ok": ok, "source": r.get("source"), "raw": raw})
        heldout_acc = ho_correct / len(rows) if rows else 0.0
        print(f"=== HELD-OUT GOLD: {ho_correct}/{len(rows)} = {heldout_acc*100:.1f}% ===")
        print(f"    gen time: {elapsed_heldout:.0f}s")

    # ============================================================
    # 3) PRECISION: corruption rate on clean held-out text (AmericasNLP refs).
    #    These are MINEDU-compliant and NEVER in training, so any non-safe-
    #    transform change is a corruption. This is the metric the §7 list missed.
    # ============================================================
    corruption_rate = None
    n_corrupt_sents = None
    if args.clean_ref_file and Path(args.clean_ref_file).exists():
        clean = [l.strip() for l in open(args.clean_ref_file) if l.strip()][: args.n_clean]
        prompts = [make_chat_prompt(tok, SYSTEM_PROMPT,
                                    f"Normalize this Chanka sentence per the MINEDU 2021 spec:\n\n{s}")
                   for s in clean]
        outs = llm.generate(prompts, sampling, lora_request=lora)
        total_corrupt = 0
        n_corrupt_sents = 0
        clean_results = []
        for s, out in zip(clean, outs):
            pred = extract_normalized(out.outputs[0].text)
            c = count_corruptions(s, pred)
            total_corrupt += c
            if c > 0:
                n_corrupt_sents += 1
                clean_results.append({"input": s, "predicted": pred, "n_corrupt": c})
        corruption_rate = n_corrupt_sents / len(clean) if clean else 0.0
        print(f"=== PRECISION (clean {len(clean)} refs): {n_corrupt_sents} corrupted "
              f"({corruption_rate*100:.1f}%), {total_corrupt} bad tokens ===")

    # Combined score: recall (spec) minus corruption penalty. Higher = better.
    combined = spec_acc - (corruption_rate if corruption_rate is not None else 0.0)

    # Save
    rec = {
        "adapter": args.adapter,
        "base": args.base,
        "spec_gold_acc": spec_acc,
        "spec_gold_correct": spec_correct,
        "spec_gold_total": len(GOLD_PAIRS),
        "heldout_acc": heldout_acc,
        "heldout_correct": ho_correct if args.gold_jsonl else None,
        "heldout_total": len(rows) if args.gold_jsonl else 0,
        "corruption_rate": corruption_rate,
        "n_corrupt_sents": n_corrupt_sents,
        "combined_score": combined,
        "elapsed_spec_sec": elapsed_spec,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(args.out_json, "w"), indent=2)
    if args.out_jsonl:
        with open(args.out_jsonl, "w") as f:
            for r in spec_results:
                f.write(json.dumps({"kind": "spec", **r}, ensure_ascii=False) + "\n")
            for r in heldout_results:
                f.write(json.dumps({"kind": "heldout", **r}, ensure_ascii=False) + "\n")

    print(f"\nSaved {args.out_json}")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
