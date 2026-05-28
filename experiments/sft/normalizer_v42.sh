#!/usr/bin/env bash
# v42: tighter LR + fewer epochs at the v41a sweet spot
set -u
cd /root/rosettia-chanka
mkdir -p outputs/logs outputs/normalizer_eval
TARGET=${TARGET:-0.95}
CU13=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
NVTL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib
NCCL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$CU13:$NVTL:$NCCL:${LD_LIBRARY_PATH:-}

RECIPES=(
  "v42a 256 512 5e-5 2.0 4 2"
  "v42b 512 1024 5e-5 2.0 4 2"
  "v42c 256 512 1e-4 2.0 4 2"
)
for IDX in $(seq 0 2); do
  PARTS=(${RECIPES[$IDX]})
  TAG=${PARTS[0]}; R=${PARTS[1]}; ALPHA=${PARTS[2]}; LR=${PARTS[3]}; EPOCHS=${PARTS[4]}; BS=${PARTS[5]}; GAS=${PARTS[6]}
  OUT_DIR=outputs/${TAG}_normalizer_20260528
  echo "=== ITER $IDX  $TAG  r=$R alpha=$ALPHA lr=$LR epochs=$EPOCHS  $(date -u +%FT%TZ) ==="
  rm -rf $OUT_DIR
  .venv/bin/python scripts/normalizer/train_normalizer_unsloth.py \
    --gold-jsonl data/normalizer_gold_filtered.jsonl \
    --model-id unsloth/Qwen3.5-4B \
    --output-dir $OUT_DIR \
    --lora-r $R --lora-alpha $ALPHA \
    --max-seq-length 2048 \
    --per-device-train-batch-size $BS --gradient-accumulation-steps $GAS \
    --learning-rate $LR \
    --num-train-epochs $EPOCHS \
    --eval-steps 96 --save-steps 96 --save-total-limit 0 \
    > outputs/logs/${TAG}_normalizer.log 2>&1
  if [ $? -ne 0 ]; then echo "TRAIN_FAILED for $TAG"; tail -20 outputs/logs/${TAG}_normalizer.log; continue; fi
  echo "=== ITER $IDX  $TAG eval-all-ckpts $(date -u +%FT%TZ) ==="
  bash experiments/sft/normalizer_eval_all_ckpts.sh $OUT_DIR
  BEST=$(.venv/bin/python -c "
import json, glob
best = (0.0, None)
for p in glob.glob(\"outputs/normalizer_eval/${TAG}_normalizer_20260528/ckpt_*.json\"):
    d=json.load(open(p))
    if d.get(\"spec_gold_acc\", 0) > best[0]:
        best = (d[\"spec_gold_acc\"], p)
print(best[0], best[1] or \"NONE\")
")
  BEST_ACC=$(echo $BEST | cut -d" " -f1); BEST_FILE=$(echo $BEST | cut -d" " -f2-)
  printf "=== ITER $IDX  $TAG best spec_gold_acc=%.4f  (%s) ===\n" $BEST_ACC "$BEST_FILE"
  STOP=$(.venv/bin/python -c "print(1 if $BEST_ACC >= $TARGET else 0)")
  if [ "$STOP" = "1" ]; then echo "TARGET HIT.  BEST_NORMALIZER=$BEST_FILE"; break; fi
done
echo "=== v42 iterate end $(date -u +%FT%TZ) ==="
