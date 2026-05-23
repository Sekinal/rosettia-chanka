#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-curriculum-eval)}"
RUN_ROOT="${RUN_ROOT:-outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64}"
CHANKA_RUN_DIR="${CHANKA_RUN_DIR:-${RUN_ROOT}/chanka/chanka}"
EVAL_DIR="${EVAL_DIR:-outputs/gspo_checkpoint_evals/${STAMP}}"
WAIT_PATTERN="${WAIT_PATTERN:-train_sft_unsloth.py.*Qwen3.5-9B}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-80}"
PROGRESS_EVERY="${PROGRESS_EVERY:-32}"

mkdir -p "$EVAL_DIR"

while pgrep -f "$WAIT_PATTERN" >/dev/null; do
  echo "$(date -u +%FT%TZ) waiting for training processes matching: $WAIT_PATTERN"
  sleep 300
done

if [[ ! -d "$CHANKA_RUN_DIR" ]]; then
  echo "Missing Chanka run directory: $CHANKA_RUN_DIR" >&2
  exit 1
fi

mapfile -t CHECKPOINTS < <(
  {
    find "$CHANKA_RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null
    find "$CHANKA_RUN_DIR" -maxdepth 1 -type d -name 'final_lora' 2>/dev/null
  } | sort -V
)

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No checkpoints found under $CHANKA_RUN_DIR" >&2
  exit 1
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
  name="$(basename "$checkpoint")"
  output_dir="${EVAL_DIR}/${name}"
  mkdir -p "$output_dir"
  if [[ -f "${output_dir}/metrics.json" ]]; then
    echo "Skipping existing eval: $checkpoint"
    continue
  fi
  echo "$(date -u +%FT%TZ) evaluating $checkpoint"
  "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$checkpoint" \
    --output-json "${output_dir}/metrics.json" \
    --predictions-jsonl "${output_dir}/predictions.jsonl" \
    --batch-size "$BATCH_SIZE" \
    --max-completion-length "$MAX_COMPLETION_LENGTH" \
    --strip-chat-artifacts \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "$TERMINOLOGY_TOP_K" \
    --progress-every "$PROGRESS_EVERY"
done

"$PYTHON" scripts/write_nested_metrics_summary.py "$EVAL_DIR"
