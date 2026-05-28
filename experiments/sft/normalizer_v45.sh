#!/usr/bin/env bash
set -u
cd /root/rosettia-chanka
mkdir -p outputs/logs outputs/normalizer_eval
CU13=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
NVTL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib
NCCL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$CU13:$NVTL:$NCCL:${LD_LIBRARY_PATH:-}
RECIPES=("v45a 256 512 1e-4 3.0 4 2" "v45b 256 512 1e-4 2.0 4 2")
for IDX in 0 1; do
  PARTS=(${RECIPES[$IDX]}); TAG=${PARTS[0]}; R=${PARTS[1]}; ALPHA=${PARTS[2]}; LR=${PARTS[3]}; EPOCHS=${PARTS[4]}; BS=${PARTS[5]}; GAS=${PARTS[6]}
  OUT=outputs/${TAG}_normalizer_20260528
  echo "=== ITER $IDX $TAG r=$R a=$ALPHA lr=$LR ep=$EPOCHS $(date -u +%FT%TZ) ==="
  rm -rf $OUT
  .venv/bin/python scripts/normalizer/train_normalizer_unsloth.py \
    --gold-jsonl data/normalizer_gold_v45.jsonl --model-id unsloth/Qwen3.5-4B \
    --output-dir $OUT --lora-r $R --lora-alpha $ALPHA --max-seq-length 2048 \
    --per-device-train-batch-size $BS --gradient-accumulation-steps $GAS \
    --learning-rate $LR --num-train-epochs $EPOCHS \
    --eval-steps 128 --save-steps 128 --save-total-limit 0 > outputs/logs/${TAG}_normalizer.log 2>&1 || { echo "TRAIN_FAILED $TAG"; continue; }
  EVAL=outputs/normalizer_eval/${TAG}_normalizer_20260528; mkdir -p $EVAL
  tail -100 data/normalizer_gold_v45.jsonl > /tmp/ho_v45.jsonl
  for CK in $OUT/checkpoint-*; do
    [ -d "$CK" ] || continue; N=$(basename $CK|sed s/checkpoint-//)
    .venv/bin/python scripts/normalizer/eval_normalizer_vllm.py --base unsloth/Qwen3.5-4B --adapter $CK \
      --gold-jsonl /tmp/ho_v45.jsonl --n-heldout 100 \
      --clean-ref-file data/normalizer_precision_holdout.txt --n-clean 200 \
      --lora-rank 512 --gpu-mem-frac 0.85 --out-json $EVAL/ckpt_${N}.json \
      --out-jsonl $EVAL/ckpt_${N}.preds.jsonl > outputs/logs/eval_${TAG}_ck${N}.log 2>&1
    printf "$TAG ckpt-%-5s " $N
    .venv/bin/python -c "import json;d=json.load(open('$EVAL/ckpt_${N}.json'));cr=d.get('corruption_rate');print('spec=%.1f%%'%(d['spec_gold_acc']*100),'corruption=%s'%(f'{cr*100:.1f}%' if cr is not None else 'NA'),'combined=%.3f'%d.get('combined_score',0))"
  done
  BEST=$(.venv/bin/python -c "import json,glob;b=(-9,'');[b:=(max(b,(json.load(open(p)).get('combined_score',-9),p))) for p in glob.glob('$EVAL/ckpt_*.json')];print(b[0],b[1])")
  echo "=== $TAG best combined=$(echo $BEST|awk '{print $1}') @ $(echo $BEST|awk '{print $2}') ==="
done
echo "=== v45 done $(date -u +%FT%TZ) ==="
