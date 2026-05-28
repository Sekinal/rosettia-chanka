#!/usr/bin/env bash
# Eval every checkpoint in a normalizer training dir, write per-ckpt JSON.
set -u
cd /root/rosettia-chanka
TRAIN_DIR=$1
EVAL_DIR=outputs/normalizer_eval/$(basename $TRAIN_DIR)
mkdir -p $EVAL_DIR

CU13=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
NVTL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib
NCCL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$CU13:$NVTL:$NCCL:${LD_LIBRARY_PATH:-}

# Hold-out 100 rows from the END of the gold (training takes random shuffle of front)
HELDOUT=/tmp/normalizer_heldout_eval.jsonl
tail -100 data/normalizer_gold.jsonl > $HELDOUT

echo "=== eval start $(date -u +%FT%TZ) ==="
for CKPT in $TRAIN_DIR/checkpoint-*; do
  [ -d "$CKPT" ] || continue
  N=$(basename $CKPT | sed "s/checkpoint-//")
  OUT=$EVAL_DIR/ckpt_${N}.json
  if [ -f $OUT ]; then
    echo "skip $N (already evaluated)"
    continue
  fi
  echo "=== ckpt-$N eval $(date -u +%FT%TZ) ==="
  .venv/bin/python scripts/normalizer/eval_normalizer_vllm.py \
    --base unsloth/Qwen3.5-4B \
    --adapter $CKPT \
    --gold-jsonl $HELDOUT --n-heldout 100 \
    --lora-rank 128 --gpu-mem-frac 0.85 \
    --out-json $OUT \
    --out-jsonl $EVAL_DIR/ckpt_${N}.preds.jsonl \
    > outputs/logs/eval_$(basename $TRAIN_DIR)_ckpt${N}.log 2>&1
  printf "ckpt-%-4s " $N
  .venv/bin/python -c "
import json
d=json.load(open(\"$OUT\"))
print(f\"spec_gold={d[\"spec_gold_correct\"]}/{d[\"spec_gold_total\"]} ({d[\"spec_gold_acc\"]*100:.1f}%)  heldout={d[\"heldout_correct\"]}/{d[\"heldout_total\"]} ({d[\"heldout_acc\"]*100:.1f}%)\"
)
"
done
echo "=== eval done $(date -u +%FT%TZ) ==="
