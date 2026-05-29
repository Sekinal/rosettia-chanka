#!/usr/bin/env bash
# v34: the normalizer payoff run. Chanka LoRA on the 109,543-pair MINEDU-unified
# corpus (107,614 gated-normalized AmericasNLP + 1,929 v30 manual), then merge
# and eval on AmericasNLP 2021 test vs v30's ChrF 40.55.
set -u
cd /root/rosettia-chanka
mkdir -p outputs/logs outputs/eval_self_verifiable_session outputs/eval_americasnlp_2021

CU13=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
NVTL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib
NCCL=$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$CU13:$NVTL:$NCCL:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in env before running}

V24F_BASE=outputs/v24f_9b_broad_20260526-m2/broad/checkpoint-2688
V34_CORPUS=clean_chanka/v34_training_corpus.parquet
V34A_DIR=outputs/v34a_9b_chanka_lora_20260528-m2

echo "=== v34a: Chanka LoRA 2 epochs on 109,543 MINEDU-unified pairs $(date -u +%FT%TZ) ==="
# ~93k train rows / eff_batch 8 = ~11,600 steps/epoch; 2 epochs ~= 23,200 steps. Eval/save every 2000.
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage chanka \
  --chanka-file "$V34_CORPUS" \
  --model-id unsloth/Qwen3.5-9B \
  --training-mode lora --adapter-method lora \
  --adapter-path "$V24F_BASE" \
  --lora-r 256 --lora-alpha 512 \
  --max-seq-length 256 \
  --per-device-train-batch-size 4 --gradient-accumulation-steps 2 \
  --learning-rate 2e-5 \
  --num-train-epochs 2.0 \
  --eval-steps 2000 --save-steps 2000 --save-total-limit 0 \
  --output-dir $V34A_DIR \
  > outputs/logs/v34a_chanka_lora.log 2>&1
echo "v34a done. Saved ckpts:"; ls $V34A_DIR/chanka/ 2>/dev/null | grep checkpoint

# --- eval each ckpt on the 158-row judicial held-out (fast chrF++ signal) ---
for CKPT_DIR in $V34A_DIR/chanka/checkpoint-*; do
  [ -d "$CKPT_DIR" ] || continue
  CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
  OUT=outputs/eval_self_verifiable_session/v34a_ckpt${CKPT}.json
  [ -f "$OUT" ] && continue
  .venv/bin/python scripts/evaluate_gspo_checkpoint.py \
    --adapter-path $CKPT_DIR --output-json $OUT \
    --predictions-jsonl outputs/eval_self_verifiable_session/v34a_ckpt${CKPT}.jsonl \
    --terminology-file clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet \
    --terminology-top-k 1 --max-completion-length 96 --split eval \
    > /tmp/v34a_eval_$CKPT.log 2>&1
  printf "v34a ckpt-%-6s " $CKPT
  .venv/bin/python -c "import json;d=json.load(open('$OUT'));print('chrF++',round(d['chrf++'],3),'BLEU',round(d['bleu'],3))"
done

BEST_V34A=$(.venv/bin/python -c "
import json, glob
best=max(glob.glob('outputs/eval_self_verifiable_session/v34a_ckpt*.json'), key=lambda p: json.load(open(p))['chrf++'])
print(best.split('ckpt')[1].split('.')[0])
")
V34A_ADAPTER=$V34A_DIR/chanka/checkpoint-$BEST_V34A
echo "=== best v34a: ckpt-$BEST_V34A @ $V34A_ADAPTER ==="

# --- merge best ---
echo "=== v34b: MERGE best v34a $(date -u +%FT%TZ) ==="
V34B_DIR=outputs/merged_full_models/20260528-v34b-9b-chanka-unified-merged
rm -rf $V34B_DIR
.venv/bin/python scripts/export_unsloth_merged_model.py \
  --adapter-path "$V34A_ADAPTER" --output-dir "$V34B_DIR" \
  --max-seq-length 256 --save-method merged_16bit \
  > outputs/logs/v34b_merge.log 2>&1
du -sh "$V34B_DIR" || { echo MERGE_FAILED; exit 1; }

# --- eval on AmericasNLP 2021 test (base-only, official ChrF word_order=0) ---
echo "=== v34 eval on AmericasNLP 2021 $(date -u +%FT%TZ) ==="
.venv/bin/python scripts/eval_americasnlp_2021_base_only.py \
  --model "$V34B_DIR" \
  --out-json outputs/eval_americasnlp_2021/v34b_final.json \
  --out-predictions-jsonl outputs/eval_americasnlp_2021/v34b_final.predictions.jsonl \
  --gpu-mem-frac 0.85 \
  > outputs/logs/v34b_anlp.log 2>&1
cat outputs/eval_americasnlp_2021/v34b_final.json
.venv/bin/python -c "
import json, sacrebleu
P=[];R=[]
for l in open('outputs/eval_americasnlp_2021/v34b_final.predictions.jsonl'):
    d=json.loads(l);P.append(d['prediction']);R.append(d['reference'])
print('OFFICIAL ChrF (word_order=0):', round(sacrebleu.corpus_chrf(P,[R],word_order=0).score,3))
print('vs v30 SOTA 40.55, Helsinki 39.40')
"
echo "=== v34 chain DONE $(date -u +%FT%TZ) ==="
