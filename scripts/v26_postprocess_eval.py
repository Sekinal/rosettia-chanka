"""Re-score v26 predictions by extracting post-'Traducción:' segment.

Takes a predictions JSONL from evaluate_gspo_checkpoint.py and the matching
reference JSONL, extracts the Chanka after the 'Traducción:' marker, computes
chrF++/BLEU/tokF1, and writes a new metrics JSON.
"""
import argparse
import json
import re
from pathlib import Path

import sacrebleu


TRANSLATION_MARKERS = [
    "Traducción:",
    "Traduccion:",
    "Translation:",
    "Final:",
]


def extract_translation(raw: str) -> str:
    for marker in TRANSLATION_MARKERS:
        idx = raw.find(marker)
        if idx >= 0:
            tail = raw[idx + len(marker):].strip()
            # Truncate at next reasoning-like marker if present.
            for stop in ["Razonamiento:", "Analisis:", "Puntaje:", "\n\n"]:
                stop_idx = tail.find(stop)
                if stop_idx >= 0:
                    tail = tail[:stop_idx].strip()
                    break
            return tail
    # No marker found — use the last non-empty line as fallback.
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[-1] if lines else raw.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-jsonl", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-extracted-jsonl", default=None)
    args = ap.parse_args()

    refs = []
    preds_raw = []
    preds_extracted = []
    sources = []
    for line in open(args.predictions_jsonl):
        d = json.loads(line)
        # The predictions schema from evaluate_gspo_checkpoint.py varies; try common keys.
        src = d.get("spanish_source") or d.get("source") or ""
        ref = d.get("reference") or d.get("target") or d.get("chanka_reference") or ""
        raw = d.get("prediction") or d.get("generated") or d.get("raw_prediction") or ""
        if not ref:
            continue
        sources.append(src)
        refs.append(ref)
        preds_raw.append(raw)
        preds_extracted.append(extract_translation(raw))

    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(preds_extracted, [refs]).score
    bleu = sacrebleu.BLEU().corpus_score(preds_extracted, [refs]).score

    rec = {
        "predictions_jsonl": args.predictions_jsonl,
        "n_rows": len(refs),
        "chrf++": chrfpp,
        "bleu": bleu,
        "marker_hit_rate": sum(
            1 for r in preds_raw if any(m in r for m in TRANSLATION_MARKERS)
        ) / max(1, len(preds_raw)),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rec, indent=2))
    print(f"chrF++={chrfpp:.3f} BLEU={bleu:.3f} marker_hit_rate={rec['marker_hit_rate']:.2%}")

    if args.out_extracted_jsonl:
        with open(args.out_extracted_jsonl, "w") as f:
            for src, ref, raw, pred in zip(sources, refs, preds_raw, preds_extracted):
                f.write(json.dumps({
                    "source": src,
                    "reference": ref,
                    "raw_prediction": raw,
                    "extracted_translation": pred,
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
