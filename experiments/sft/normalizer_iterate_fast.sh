#!/usr/bin/env bash
# Fast iteration: 1 epoch per recipe, larger effective batch, eval after.
set -u
cd /root/rosettia-chanka
mkdir -p outputs/logs outputs/normalizer_eval

TARGET=${TARGET:-0.97}
MAX_ITERS=${MAX_ITERS:-5}

CU13=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
NVTL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib
NCCL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$CU13:$NVTL:$NCCL:${LD_LIBRARY_PATH:-}

# Recipes: tag r alpha lr epochs bs gas
RECIPES=(
  "v40a 128 256 2e-5 1.0 4 2"
  "v40b 128 256 2e-5 3.0 4 2"
  "v40c 256 512 2e-5 3.0 2 4"
  "v40d 256 512 1e-5 5.0 2 4"
  "v40e 128 256 5e-5 3.0 4 2"
)

for IDX in $(seq 0 $((MAX_ITERS-1))); do
  PARTS=(${RECIPES[$IDX]})
  TAG=${PARTS[0]}; R=${PARTS[1]}; ALPHA=${PARTS[2]}; LR=${PARTS[3]}; EPOCHS=${PARTS[4]}; BS=${PARTS[5]}; GAS=${PARTS[6]}
  OUT_DIR=outputs/${TAG}_normalizer_20260528
  echo "=== ITER $IDX  $TAG  r=$R alpha=$ALPHA lr=$LR epochs=$EPOCHS bs=$BS gas=$GAS  $(date -u +%FT%TZ) ==="

  if [ ! -d "$OUT_DIR" ] || [ -z "$(ls $OUT_DIR/checkpoint-* 2>/dev/null)" ]; then
    rm -rf $OUT_DIR
    .venv/bin/python scripts/normalizer/train_normalizer_unsloth.py \
      --gold-jsonl data/normalizer_gold.jsonl \
      --model-id unsloth/Qwen3.5-4B \
      --output-dir $OUT_DIR \
      --lora-r $R --lora-alpha $ALPHA \
      --max-seq-length 2048 \
      --per-device-train-batch-size $BS --gradient-accumulation-steps $GAS \
      --learning-rate $LR \
      --num-train-epochs $EPOCHS \
      --eval-steps 64 --save-steps 64 --save-total-limit 0 \
      > outputs/logs/${TAG}_normalizer.log 2>&1
    if [ $? -ne 0 ]; then
      echo "TRAIN_FAILED for $TAG"; tail -20 outputs/logs/${TAG}_normalizer.log; continue
    fi
  fi

  echo "=== ITER $IDX  $TAG eval-all-ckpts $(date -u +%FT%TZ) ==="
  bash experiments/sft/normalizer_eval_all_ckpts.sh $OUT_DIR

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

  STOP=$(.venv/bin/python -c "print(1 if $BEST_ACC >= $TARGET else 0)")
  if [ "$STOP" = "1" ]; then
    echo "TARGET HIT.  $TAG  spec_gold_acc=$BEST_ACC >= $TARGET"
    echo "BEST_NORMALIZER=$BEST_FILE"
    break
  fi
done

echo "=== iterate end $(date -u +%FT%TZ) ==="
