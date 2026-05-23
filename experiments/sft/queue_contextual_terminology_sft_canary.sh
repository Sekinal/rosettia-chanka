#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-contextual-terminology-sft)}"

BASE_ADAPTER="${BASE_ADAPTER:-outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8}"
MIXED_JSONL="${MIXED_JSONL:-outputs/mbr_self_training_data/20260522-k16-full-newbest-noterm-t065-p090/mixed_clean512_confident_margin000_target.jsonl}"
WAIT_FOR_FILE="${WAIT_FOR_FILE:-outputs/gspo_checkpoint_evals/20260523-qwen35-9b-term-chanka/summary.json}"

WORK_DIR="${WORK_DIR:-outputs/contextual_terminology_sft/${STAMP}}"
CONTEXT_JSONL="${CONTEXT_JSONL:-${WORK_DIR}/contextual_terminology_target.jsonl}"
SFT_DIR="${SFT_DIR:-${WORK_DIR}/sft_lr1e-7_16steps}"
EVAL_DIR="${EVAL_DIR:-${WORK_DIR}/checkpoint_eval}"

TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"
MAX_TERMS="${MAX_TERMS:-96}"
TEMPLATES_PER_TERM="${TEMPLATES_PER_TERM:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
MAX_STEPS="${MAX_STEPS:-16}"
EVAL_STEPS="${EVAL_STEPS:-8}"
SAVE_STEPS="${SAVE_STEPS:-8}"

mkdir -p "$WORK_DIR" "$EVAL_DIR"

while [[ -n "$WAIT_FOR_FILE" && ! -f "$WAIT_FOR_FILE" ]]; do
  echo "$(date -u +%FT%TZ) waiting before contextual terminology SFT: $WAIT_FOR_FILE"
  sleep 300
done

if [[ ! -f "$CONTEXT_JSONL" ]]; then
  "$PYTHON" scripts/build_contextual_terminology_jsonl.py \
    --output-jsonl "$CONTEXT_JSONL" \
    --terminology-file "$TERMINOLOGY_FILE" \
    --templates-per-term "$TEMPLATES_PER_TERM" \
    --max-terms "$MAX_TERMS"
fi

if [[ ! -f "${SFT_DIR}/final_lora/adapter_config.json" ]]; then
  "$PYTHON" scripts/train_jsonl_sft_unsloth.py \
    --jsonl "$MIXED_JSONL" \
    --jsonl "$CONTEXT_JSONL" \
    --target-field target \
    --adapter-path "$BASE_ADAPTER" \
    --output-dir "$SFT_DIR" \
    --max-seq-length 128 \
    --max-steps "$MAX_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --logging-steps 4 \
    --per-device-train-batch-size 4 \
    --per-device-eval-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio 0.03 \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "$TERMINOLOGY_TOP_K"
fi

mapfile -t CHECKPOINTS < <(
  {
    find "$SFT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null
    find "$SFT_DIR" -maxdepth 1 -type d -name 'final_lora' 2>/dev/null
  } | sort -V
)

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No contextual terminology checkpoints found under $SFT_DIR" >&2
  exit 1
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
  name="$(basename "$checkpoint")"
  output_dir="${EVAL_DIR}/${name}"
  mkdir -p "$output_dir"
  if [[ -f "${output_dir}/metrics.json" ]]; then
    echo "Skipping existing contextual terminology eval: $checkpoint"
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

"$PYTHON" - <<'PY' "$EVAL_DIR" "$CONTEXT_JSONL" "$WORK_DIR"
import json
import sys
from pathlib import Path

from scripts.write_nested_metrics_summary import collect_records

eval_dir = Path(sys.argv[1])
context_jsonl = Path(sys.argv[2])
work_dir = Path(sys.argv[3])
records = collect_records(eval_dir, [])
summary = {
    "context_jsonl": str(context_jsonl),
    "context_rows": sum(1 for _ in context_jsonl.open()),
    "records": records,
}
summary_path = work_dir / "summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"best": records[0] if records else None, "summary_json": str(summary_path)}, ensure_ascii=False, indent=2))
PY
