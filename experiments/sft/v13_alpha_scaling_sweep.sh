#!/usr/bin/env bash
# v13 LoRA alpha-scaling probe: copy v13 ckpt-32 adapter dir, modify lora_alpha
# in adapter_config.json, run held-out eval. Tests whether SFT signal is
# undertrained (higher alpha helps) or saturated (lower alpha helps).
set -u
cd /root/rosettia-chanka

SRC=outputs/compact_mixed_format_sft_20260525-4b-refine-v13-from-v12ckpt128/checkpoint-32
OUT_DIR=outputs/v13_alpha_scaling_sweep_20260525
mkdir -p "$OUT_DIR"

# v13 was trained with r=64, alpha=128 → effective scale 2.0
# Try scales 0.5, 0.75, 1.0 (orig), 1.25, 1.5, 1.75, 2.0
for SCALE in 0.5 0.75 1.0 1.25 1.5 1.75 2.0; do
  NEW_ALPHA=$(.venv/bin/python -c "print(int(128 * $SCALE))")
  TAG="alpha${NEW_ALPHA}"
  ADAPTER="$OUT_DIR/v13_$TAG"
  echo "=== scale=$SCALE alpha=$NEW_ALPHA -> $ADAPTER ==="
  rm -rf "$ADAPTER"; cp -r "$SRC" "$ADAPTER"
  .venv/bin/python -c "
import json, sys
p = sys.argv[1] + '/adapter_config.json'
d = json.load(open(p))
d['lora_alpha'] = $NEW_ALPHA
json.dump(d, open(p,'w'), indent=2)
print('updated', p, 'lora_alpha=', d['lora_alpha'])
" "$ADAPTER"
  .venv/bin/python scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$ADAPTER" \
    --output-json "$OUT_DIR/${TAG}_metrics.json" \
    --predictions-jsonl "$OUT_DIR/${TAG}_predictions.jsonl" \
    --terminology-file clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet \
    --terminology-top-k 1 --max-completion-length 96 --split eval \
    > "$OUT_DIR/${TAG}_eval.log" 2>&1
  printf "%-25s " "$TAG"
  .venv/bin/python -c "
import json
d=json.load(open('$OUT_DIR/${TAG}_metrics.json'))
print('chrF++', round(d['chrf++'],3), 'BLEU', round(d['bleu'],3), 'tokF1', round(d['token_f1'],3), 'TER', round(d['ter'],3))
"
done

echo
echo "==================== ALPHA SCALING SUMMARY ===================="
for f in "$OUT_DIR"/*_metrics.json; do
  printf "%-25s " "$(basename "$f" .json)"
  .venv/bin/python -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('chrF++', round(d['chrf++'],3), 'BLEU', round(d['bleu'],3), 'tokF1', round(d['token_f1'],3), 'TER', round(d['ter'],3))
" "$f"
done | sort -V
