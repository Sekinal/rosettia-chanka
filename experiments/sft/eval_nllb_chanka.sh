#!/usr/bin/env bash
set -u
cd /root/rosettia-chanka
OUT=outputs/nllb_lora_chanka_20260525-1.3B-m1
for CKPT in 128 192 256 320 384 448 512 final_lora; do
  ADAPTER_DIR=$OUT/checkpoint-$CKPT
  if [ "$CKPT" = "final_lora" ]; then
    ADAPTER_DIR=$OUT/final_lora
  fi
  [ -d "$ADAPTER_DIR" ] || continue
  .venv/bin/python scripts/evaluate_translation_baseline.py \
    --backend nllb --model-id facebook/nllb-200-distilled-1.3B \
    --adapter-path "$ADAPTER_DIR" \
    --target-language-name "Quechua Chanka" \
    --source-lang spa_Latn --target-lang quy_Latn \
    --split eval --max-new-tokens 96 --batch-size 4 \
    --output-json "$OUT/eval_ckpt${CKPT}_metrics.json" \
    --predictions-jsonl "$OUT/eval_ckpt${CKPT}_predictions.jsonl" \
    > /tmp/nllb_eval_${CKPT}.log 2>&1 || echo "ckpt-$CKPT failed"
  printf "nllb-1.3B-lora ckpt-%-12s " "$CKPT"
  .venv/bin/python -c "
import json
d = json.load(open('$OUT/eval_ckpt${CKPT}_metrics.json'))
print('chrF++', round(d['chrf++'],3), 'BLEU', round(d['bleu'],3), 'tokF1', round(d['token_f1'],3), 'TER', round(d['ter'],3))
"
done
