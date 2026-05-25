#!/usr/bin/env bash
set -u
cd /root/rosettia-chanka
OUT=outputs/external_baselines_20260525
mkdir -p "$OUT"

echo "=== TranslateGemma 4B (BCP-47 fix) ==="
.venv/bin/python scripts/evaluate_translation_baseline.py \
  --backend translategemma --model-id google/translategemma-4b-it \
  --target-language-name "Quechua Chanka" \
  --source-lang spa_Latn --target-lang quy_Latn \
  --split eval --max-new-tokens 96 --batch-size 4 \
  --output-json "$OUT/translategemma_4b_v2_metrics.json" \
  --predictions-jsonl "$OUT/translategemma_4b_v2_predictions.jsonl" || echo "TG failed"

echo "=== T5Gemma 2-1B ==="
.venv/bin/python scripts/evaluate_translation_baseline.py \
  --backend seq2seq-chat --model-id google/t5gemma-2-1b-1b \
  --target-language-name "Quechua Chanka" \
  --split eval --max-new-tokens 96 --batch-size 4 \
  --output-json "$OUT/t5gemma_2_1b_metrics.json" \
  --predictions-jsonl "$OUT/t5gemma_2_1b_predictions.jsonl" || echo "T5G 1B failed"

echo
echo "=== Results ==="
for f in "$OUT"/translategemma_4b_v2_metrics.json "$OUT"/t5gemma_2_1b_metrics.json; do
  [ -f "$f" ] || { echo "$(basename "$f") MISSING"; continue; }
  printf "%-32s " "$(basename "$f" .json)"
  .venv/bin/python -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('chrF++', round(d['chrf++'],3), 'BLEU', round(d['bleu'],3), 'tokF1', round(d['token_f1'],3), 'TER', round(d['ter'],3))
" "$f"
done
