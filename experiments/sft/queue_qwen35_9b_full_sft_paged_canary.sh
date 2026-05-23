#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-9b-paged-full-sft-canary)}"

BEST_CHECKPOINT="${BEST_CHECKPOINT:-outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64/chanka/chanka/checkpoint-192}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-outputs/merged_full_models/20260523-qwen35-9b-direct-full-sft-smoke-merged16}"
WORK_DIR="${WORK_DIR:-outputs/qwen35_9b_full_sft/${STAMP}}"
FULL_RUN_DIR="${FULL_RUN_DIR:-${WORK_DIR}/full_chanka_paged_lr5e-7_8steps}"
EVAL_DIR="${EVAL_DIR:-${WORK_DIR}/checkpoint_eval}"
FAIL_MARKER="${FAIL_MARKER:-${WORK_DIR}/full_sft_failed.txt}"

TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"
OPTIM="${OPTIM:-paged_adamw_8bit}"
MAX_STEPS="${MAX_STEPS:-8}"
EVAL_STEPS="${EVAL_STEPS:-4}"
SAVE_STEPS="${SAVE_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

cd "$ROOT_DIR"
mkdir -p "$WORK_DIR" "$EVAL_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${MERGED_MODEL_DIR}/config.json" ]]; then
  "$PYTHON" scripts/export_unsloth_merged_model.py \
    --adapter-path "$BEST_CHECKPOINT" \
    --output-dir "$MERGED_MODEL_DIR" \
    --max-seq-length 128 \
    --save-method merged_16bit
fi

if [[ ! -f "${FULL_RUN_DIR}/chanka/final_full_model/config.json" && ! -f "$FAIL_MARKER" ]]; then
  set +e
  "$PYTHON" scripts/train_sft_unsloth.py \
    --stage chanka \
    --model-id "$MERGED_MODEL_DIR" \
    --training-mode full \
    --output-dir "$FULL_RUN_DIR" \
    --max-seq-length 128 \
    --max-steps "$MAX_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit 1 \
    --logging-steps 2 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio "${WARMUP_RATIO:-0.05}" \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "$TERMINOLOGY_TOP_K" \
    --optim "$OPTIM"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    {
      echo "$(date -u +%FT%TZ) 9B paged full-SFT canary failed with exit status $status"
      echo "Optimizer: $OPTIM"
      echo "Merged model: $MERGED_MODEL_DIR"
    } | tee "$FAIL_MARKER"
    exit 0
  fi
fi

mapfile -t CHECKPOINTS < <(
  {
    find "${FULL_RUN_DIR}/chanka" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null
    find "${FULL_RUN_DIR}/chanka" -maxdepth 1 -type d -name 'final_full_model' 2>/dev/null
  } | sort -V
)

for checkpoint in "${CHECKPOINTS[@]}"; do
  name="$(basename "$checkpoint")"
  output_dir="${EVAL_DIR}/${name}"
  mkdir -p "$output_dir"
  if [[ -f "${output_dir}/metrics.json" ]]; then
    echo "Skipping existing full-SFT eval: $checkpoint"
    continue
  fi
  "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$checkpoint" \
    --output-json "${output_dir}/metrics.json" \
    --predictions-jsonl "${output_dir}/predictions.jsonl" \
    --batch-size 1 \
    --max-completion-length 80 \
    --strip-chat-artifacts \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "$TERMINOLOGY_TOP_K" \
    --progress-every 32
done

"$PYTHON" scripts/write_nested_metrics_summary.py "$EVAL_DIR"
df -h /root
