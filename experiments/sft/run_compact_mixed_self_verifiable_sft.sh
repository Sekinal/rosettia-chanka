#!/usr/bin/env bash
# DeepSeekMath-V2-adapted compact-mixed SFT for Spanish -> Chanka Quechua.
#
# Builds on the 4B full-SFT base (checkpoint-36) and learns a LoRA on a
# mixed dataset of (a) direct translation rows and (b) compact-thinking
# self-verification rows. At inference time, the standard direct prompt
# already beats the 4B base on chrF++/BLEU/token-F1/TER; the
# self-verification mode remains useful as a training-time auxiliary task.
#
# Default settings reproduce the v2 long-run that improves over the
# 32-step v1 (ckpt-32). Override env vars to ablate.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-compact-mixed-self-verifiable)}"

BASE_MODEL="${BASE_MODEL:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36}"
DATA_DIR="${DATA_DIR:-outputs/self_verifiable_data_20260525-compact-mixed}"
INPUT_JSONL="${INPUT_JSONL:-${DATA_DIR}/self_verifiable_compact_mixed_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/compact_mixed_self_verifiable_sft_${STAMP}}"

MAX_STEPS="${MAX_STEPS:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
EVAL_STEPS="${EVAL_STEPS:-16}"
SAVE_STEPS="${SAVE_STEPS:-16}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1792}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-256}"
PER_DEVICE_TRAIN_BATCH="${PER_DEVICE_TRAIN_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR" outputs/logs

# Build the mixed JSONL if it is missing. The build script writes both
# direct and compact rows from the reviewed Chanka pairs.
if [[ ! -f "$INPUT_JSONL" ]]; then
  "$PYTHON" scripts/build_self_verifiable_translation_data.py \
    --output-dir "$DATA_DIR"
fi

"$PYTHON" scripts/train_jsonl_sft_unsloth.py \
  --jsonl "$INPUT_JSONL" \
  --target-field target \
  --model-id "$BASE_MODEL" \
  --training-mode lora \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --max-eval-samples "$MAX_EVAL_SAMPLES" \
  --max-steps "$MAX_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --eval-steps "$EVAL_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit 0 \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-seq-length 128 \
  --terminology-top-k 1 \
  --logging-steps 4 \
  --output-dir "$OUTPUT_DIR" \
  "$@"

# Held-out direct-prompt eval against the chanka validation split.
EVAL_OUT="${OUTPUT_DIR}/direct_eval"
mkdir -p "$EVAL_OUT"
"$PYTHON" scripts/evaluate_gspo_checkpoint.py \
  --adapter-path "$OUTPUT_DIR/final_lora" \
  --output-json "${EVAL_OUT}/final_lora_direct_metrics.json" \
  --predictions-jsonl "${EVAL_OUT}/final_lora_direct_predictions.jsonl" \
  --terminology-file clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet \
  --terminology-top-k 1 \
  --max-completion-length 96 \
  --split eval

cat "${EVAL_OUT}/final_lora_direct_metrics.json"
