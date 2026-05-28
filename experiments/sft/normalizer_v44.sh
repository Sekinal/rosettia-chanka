#!/usr/bin/env bash
# v44: precision-augmented data (v30 verified-gated preservation + hard negatives
# + corrections). Selects checkpoints by COMBINED score (spec recall - corruption).
set -u
cd /root/rosettia-chanka
mkdir -p outputs/logs outputs/normalizer_eval
CU13=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
NVTL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib
NCCL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$CU13:$NVTL:$NCCL:${LD_LIBRARY_PATH:-}

# tag r alpha lr epochs bs gas
RECIPES=(
  "v44a 256 512 1e-4 3.0 4 2"
  "v44b 256 512 1e-4 2.0 4 2"
  "v44c 256 512 5e-5 4.0 4 2"
)
for IDX in 0 1 2; do
  PARTS=(${RECIPES[$IDX]})
  TAG=${PARTS[0]}; R=${PARTS[1]}; ALPHA=${PARTS[2]}; LR=${PARTS[3]}; EPOCHS=${PARTS[4]}; BS=${PARTS[5]}; GAS=${PARTS[6]}
  OUT_DIR=outputs/${TAG}_normalizer_20260528
  echo "=== ITER $IDX  $TAG  r=$R alpha=$ALPHA lr=$LR epochs=$EPOCHS  $(date -u +%FT%TZ) ==="
  rm -rf $OUT_DIR
  .venv/bin/python scripts/normalizer/train_normalizer_unsloth.py \
    --gold-jsonl data/normalizer_gold_v44.jsonl \
    --model-id unsloth/Qwen3.5-4B \
    --output-dir $OUT_DIR \
    --lora-r $R --lora-alpha $ALPHA --max-seq-length 2048 \
    --per-device-train-batch-size $BS --gradient-accumulation-steps $GAS \
    --learning-rate $LR --num-train-epochs $EPOCHS \
    --eval-steps 128 --save-steps 128 --save-total-limit 0 \
    > outputs/logs/${TAG}_normalizer.log 2>&1
  if [ $? -ne 0 ]; then echo "TRAIN_FAILED for $TAG"; tail -20 outputs/logs/${TAG}_normalizer.log; continue; fi

  echo "=== ITER $IDX  $TAG eval-all-ckpts $(date -u +%FT%TZ) ==="
  EVAL_DIR=outputs/normalizer_eval/${TAG}_normalizer_20260528
  mkdir -p $EVAL_DIR
  HELDOUT=/tmp/normalizer_heldout_eval.jsonl
  tail -100 data/normalizer_gold_v44.jsonl > $HELDOUT
  for CKPT in $OUT_DIR/checkpoint-*; do
    [ -d "$CKPT" ] || continue
    N=$(basename $CKPT | sed "s/checkpoint-//")
    OUT=$EVAL_DIR/ckpt_${N}.json
    [ -f $OUT ] && continue
    .venv/bin/python scripts/normalizer/eval_normalizer_vllm.py \
      --base unsloth/Qwen3.5-4B --adapter $CKPT \
      --gold-jsonl $HELDOUT --n-heldout 100 \
      --clean-ref-file docs/references/americasnlp_test/2021_test.quy --n-clean 150 \
      --lora-rank 512 --gpu-mem-frac 0.85 \
      --out-json $OUT --out-jsonl $EVAL_DIR/ckpt_${N}.preds.jsonl \
      > outputs/logs/eval_${TAG}_ckpt${N}.log 2>&1
    printf "$TAG ckpt-%-4s " $N
    .venv/bin/python -c "
import json
d=json.load(open('$OUT'))
cr = d.get('corruption_rate')
crs = f'{cr*100:.1f}%' if cr is not None else 'NA'
print(f\"spec={d['spec_gold_acc']*100:.1f}%  corruption={crs}  combined={d.get('combined_score',0):.3f}\")
"
  done

  BEST=$(.venv/bin/python -c "
import json, glob
best=(-9,'')
for p in glob.glob('outputs/normalizer_eval/${TAG}_normalizer_20260528/ckpt_*.json'):
    d=json.load(open(p))
    s=d.get('combined_score',-9)
    if s>best[0]: best=(s,p)
print(best[0], best[1] or 'NONE')
")
  BEST_SCORE=$(echo $BEST | awk '{print $1}'); BEST_FILE=$(echo $BEST | awk '{print $2}')
  printf "=== ITER %s  %s best combined=%.4f  (%s) ===\n" "$IDX" "$TAG" "$BEST_SCORE" "$BEST_FILE"
done
echo "=== v44 iterate end $(date -u +%FT%TZ) ==="
