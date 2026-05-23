#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-4b-full-sft-lr-sweep)}"

SOURCE_ADAPTER="${SOURCE_ADAPTER:-outputs/qwen35_4b_curriculum/20260522-broad512-chanka-r64-a128-s128-256steps/chanka/checkpoint-224}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-outputs/merged_full_models/20260522-qwen35-4b-broad512-chanka224-merged16}"
WORK_DIR="${WORK_DIR:-outputs/full_sft_sweeps/${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-${WORK_DIR}/checkpoint_eval}"

TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"
LR_LIST="${LR_LIST:-5e-7 1e-6 2e-6}"
MAX_STEPS="${MAX_STEPS:-96}"
EVAL_STEPS="${EVAL_STEPS:-12}"
SAVE_STEPS="${SAVE_STEPS:-12}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"

# By default this waits until the higher-priority 9B merge -> full-SFT chain has
# either produced metrics or failed its memory smoke. Override both to empty
# strings to run immediately.
WAIT_FOR_SUCCESS="${WAIT_FOR_SUCCESS:-outputs/qwen35_9b_full_sft/20260523-qwen35-9b-merge-full-sft/checkpoint_eval/summary.json}"
WAIT_FOR_FAILURE="${WAIT_FOR_FAILURE:-outputs/qwen35_9b_full_sft/20260523-qwen35-9b-merge-full-sft/full_sft_smoke_failed.txt}"

mkdir -p "$WORK_DIR" "$EVAL_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${WAIT_FOR_SUCCESS}${WAIT_FOR_FAILURE}" ]]; then
  while [[ ! -f "$WAIT_FOR_SUCCESS" && ! -f "$WAIT_FOR_FAILURE" ]]; do
    echo "$(date -u +%FT%TZ) waiting before 4B full-SFT LR sweep"
    echo "  success marker: $WAIT_FOR_SUCCESS"
    echo "  failure marker: $WAIT_FOR_FAILURE"
    sleep 300
  done
fi

if [[ ! -f "${MERGED_MODEL_DIR}/config.json" ]]; then
  "$PYTHON" scripts/export_unsloth_merged_model.py \
    --adapter-path "$SOURCE_ADAPTER" \
    --output-dir "$MERGED_MODEL_DIR" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --save-method merged_16bit
fi

echo "$SOURCE_ADAPTER" > "${WORK_DIR}/source_adapter.txt"
echo "$MERGED_MODEL_DIR" > "${WORK_DIR}/merged_model.txt"

for lr in $LR_LIST; do
  lr_label="${lr//./p}"
  lr_label="${lr_label//-e-/em}"
  lr_label="${lr_label//e-/em}"
  run_dir="${WORK_DIR}/lr_${lr_label}_${MAX_STEPS}steps"
  eval_dir="${EVAL_ROOT}/lr_${lr_label}"
  mkdir -p "$eval_dir"

  if [[ ! -f "${run_dir}/chanka/final_full_model/config.json" ]]; then
    "$PYTHON" scripts/train_sft_unsloth.py \
      --stage chanka \
      --model-id "$MERGED_MODEL_DIR" \
      --training-mode full \
      --output-dir "$run_dir" \
      --max-seq-length "$MAX_SEQ_LENGTH" \
      --max-steps "$MAX_STEPS" \
      --eval-steps "$EVAL_STEPS" \
      --save-steps "$SAVE_STEPS" \
      --save-total-limit "$SAVE_TOTAL_LIMIT" \
      --logging-steps 4 \
      --per-device-train-batch-size 1 \
      --per-device-eval-batch-size 1 \
      --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
      --learning-rate "$lr" \
      --warmup-ratio 0.05 \
      --terminology-file "$TERMINOLOGY_FILE" \
      --terminology-top-k "$TERMINOLOGY_TOP_K"
  fi

  mapfile -t checkpoints < <(
    {
      find "${run_dir}/chanka" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null
      find "${run_dir}/chanka" -maxdepth 1 -type d -name 'final_full_model' 2>/dev/null
    } | sort -V
  )

  for checkpoint in "${checkpoints[@]}"; do
    name="$(basename "$checkpoint")"
    output_dir="${eval_dir}/${name}"
    mkdir -p "$output_dir"
    if [[ -f "${output_dir}/metrics.json" ]]; then
      echo "Skipping existing 4B full-SFT eval: $checkpoint"
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

  "$PYTHON" scripts/write_nested_metrics_summary.py "$eval_dir" --metric-field source_copy_ratio
done

"$PYTHON" - <<'PY' "$EVAL_ROOT" "${WORK_DIR}/summary.json"
import json
import sys
from pathlib import Path

eval_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
records = []
for lr_dir in sorted(eval_root.glob("lr_*")):
    path = lr_dir / "summary.json"
    if not path.exists():
        continue
    for row in json.loads(path.read_text()).get("records", []):
        records.append({**row, "learning_rate_label": lr_dir.name, "eval_dir": str(lr_dir)})
records.sort(
    key=lambda row: (
        row.get("selection_score") is not None,
        float(row.get("selection_score") or -1.0),
        float(row.get("bleu") or 0.0),
    ),
    reverse=True,
)
summary_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"summary_json": str(summary_path), "best": records[0] if records else None}, ensure_ascii=False, indent=2))
PY
