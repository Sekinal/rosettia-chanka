#!/usr/bin/env bash
# Sequential external translation-model benchmarks on the same 158-row Chanka eval.
# Compares NLLB / TranslateGemma / Gemma 4 / Hy-MT2 to our v13 chrF++ 55.76.
# Waits for any in-flight evaluate_gspo_checkpoint job before starting.

set -u
cd /root/rosettia-chanka
mkdir -p outputs/external_baselines_20260525

while pgrep -f evaluate_gspo_checkpoint > /dev/null 2>&1; do
  echo "$(date -u +%FT%TZ) waiting for current GPU eval to finish..."
  sleep 60
done
echo "$(date -u +%FT%TZ) starting external baselines"

run_one() {
  local NAME="$1"; shift
  echo "=== $NAME starting $(date -u +%FT%TZ) ==="
  .venv/bin/python "$@" 2>&1 | tail -3
  echo "=== $NAME done $(date -u +%FT%TZ) ==="
}

OUT=outputs/external_baselines_20260525

# NLLB-200 600M / 1.3B / 3.3B
for SIZE in 600M 1.3B 3.3B; do
  if [[ "$SIZE" == "3.3B" ]]; then
    MODEL=facebook/nllb-200-3.3B
    BATCH=2
  else
    MODEL=facebook/nllb-200-distilled-$SIZE
    BATCH=4
  fi
  run_one "nllb_$SIZE" scripts/evaluate_translation_baseline.py \
    --backend nllb --model-id "$MODEL" \
    --target-language-name "Quechua Ayacucho" \
    --source-lang spa_Latn --target-lang quy_Latn \
    --split eval --max-new-tokens 96 --batch-size $BATCH \
    --output-json "$OUT/nllb_${SIZE}_metrics.json" \
    --predictions-jsonl "$OUT/nllb_${SIZE}_predictions.jsonl" \
    || echo "nllb $SIZE FAILED"
done

# TranslateGemma 4B IT
run_one "translategemma_4b" scripts/evaluate_translation_baseline.py \
  --backend translategemma --model-id google/translategemma-4b-it \
  --target-language-name "Quechua Chanka" \
  --source-lang spa_Latn --target-lang quy_Latn \
  --split eval --max-new-tokens 96 --batch-size 4 \
  --output-json "$OUT/translategemma_4b_metrics.json" \
  --predictions-jsonl "$OUT/translategemma_4b_predictions.jsonl" \
  || echo "translategemma FAILED"

# Gemma 4 E4B IT
run_one "gemma4_e4b" scripts/evaluate_translation_baseline.py \
  --backend causal-chat --model-id google/gemma-4-E4B-it \
  --target-language-name "Quechua Chanka" \
  --split eval --max-new-tokens 96 --batch-size 4 \
  --output-json "$OUT/gemma4_e4b_metrics.json" \
  --predictions-jsonl "$OUT/gemma4_e4b_predictions.jsonl" \
  || echo "gemma4 FAILED"

# Hy-MT2 7B
run_one "hymt2_7b" scripts/evaluate_translation_baseline.py \
  --backend causal-chat --model-id tencent/Hy-MT2-7B \
  --target-language-name "Quechua Chanka" \
  --prompt-style hymt2 \
  --split eval --max-new-tokens 96 --batch-size 2 \
  --output-json "$OUT/hymt2_7b_metrics.json" \
  --predictions-jsonl "$OUT/hymt2_7b_predictions.jsonl" \
  || echo "hymt2 FAILED"

# T5Gemma 270M
run_one "t5gemma_270m" scripts/evaluate_translation_baseline.py \
  --backend seq2seq-chat --model-id google/t5gemma-2-270m-270m \
  --target-language-name "Quechua Chanka" \
  --split eval --max-new-tokens 96 --batch-size 4 \
  --output-json "$OUT/t5gemma_270m_metrics.json" \
  --predictions-jsonl "$OUT/t5gemma_270m_predictions.jsonl" \
  || echo "t5gemma FAILED"

echo ""
echo "==================== RESULTS SUMMARY ===================="
for f in "$OUT"/*_metrics.json; do
  [ -f "$f" ] || continue
  printf '%-30s ' "$(basename "$f" .json)"
  .venv/bin/python -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(f"chrF++ {d[\"chrf++\"]:.3f}  BLEU {d[\"bleu\"]:.3f}  tokF1 {d[\"token_f1\"]:.3f}  TER {d[\"ter\"]:.3f}")
' "$f" 2>&1 || echo "(parse fail)"
done
echo "========================================================="
