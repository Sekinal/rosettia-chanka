#!/usr/bin/env bash
# Iterate normalizer training: train → eval → decide → next.
# Stops when spec_gold_acc >= TARGET or after N iterations.
set -u
cd /root/rosettia-chanka
mkdir -p outputs/logs outputs/normalizer_eval

TARGET=${TARGET:-0.97}      # 97% gold accuracy = good enough
MAX_ITERS=${MAX_ITERS:-4}

CU13=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
NVTL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib
NCCL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$CU13:$NVTL:$NCCL:${LD_LIBRARY_PATH:-}

# Define recipes (label r alpha lr epochs)
RECIPES=(
  "v40a 128 256 2e-5 3.0"
  "v40b 256 512 2e-5 3.0"
  "v40c 128 256 1e-5 5.0"
  "v40d 256 512 3e-5 4.0"
)

for IDX in $(seq 0 $((MAX_ITERS-1))); do
  PARTS=(${RECIPES[$IDX]})
  TAG=${PARTS[0]}; R=${PARTS[1]}; ALPHA=${PARTS[2]}; LR=${PARTS[3]}; EPOCHS=${PARTS[4]}
  OUT_DIR=outputs/${TAG}_normalizer_20260528
  echo "=== ITER $IDX  $TAG  r=$R alpha=$ALPHA lr=$LR epochs=$EPOCHS  $(date -u +%FT%TZ) ==="

  # Train if not already done
  if [ ! -d "$OUT_DIR" ] || [ -z "$(ls $OUT_DIR/checkpoint-* 2>/dev/null)" ]; then
    .venv/bin/python scripts/normalizer/train_normalizer_unsloth.py \
      --gold-jsonl data/normalizer_gold.jsonl \
      --model-id unsloth/Qwen3.5-4B \
      --output-dir $OUT_DIR \
      --lora-r $R --lora-alpha $ALPHA \
      --max-seq-length 2048 \
      --per-device-train-batch-size 2 --gradient-accumulation-steps 4 \
      --learning-rate $LR \
      --num-train-epochs $EPOCHS \
      --eval-steps 64 --save-steps 64 --save-total-limit 0 \
      > outputs/logs/${TAG}_normalizer.log 2>&1
  else
    echo "  -> training already exists, skipping to eval"
  fi

  # Eval all ckpts
  bash experiments/sft/normalizer_eval_all_ckpts.sh $OUT_DIR

  # Find best spec_gold_acc
  BEST=$(.venv/bin/python -c "
import json, glob
best = (0.0, None)
for p in glob.glob(\"outputs/normalizer_eval/${TAG}_normalizer_20260528/ckpt_*.json\"):
    d=json.load(open(p))
    if d[\"spec_gold_acc\"] > best[0]:
        best = (d[\"spec_gold_acc\"], p)
print(best[0], best[1] or \"NONE\")
")
  BEST_ACC=$(echo $BEST | cut -d" " -f1)
  BEST_FILE=$(echo $BEST | cut -d" " -f2-)
  printf "=== ITER $IDX  $TAG best spec_gold_acc=%.4f  (%s) ===\n" $BEST_ACC "$BEST_FILE"

  # Check stop condition
  STOP=$(.venv/bin/python -c "print(1 if $BEST_ACC >= $TARGET else 0)")
  if [ "$STOP" = "1" ]; then
    echo "TARGET HIT.  $TAG  spec_gold_acc=$BEST_ACC >= $TARGET"
    echo "BEST_NORMALIZER=$BEST_FILE"
    break
  fi
done

echo "=== iterate end $(date -u +%FT%TZ) ==="
