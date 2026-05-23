#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-9b-term-chanka)}"

RUN_ROOT="${RUN_ROOT:-outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64}"
BROAD_CHECKPOINT="${BROAD_CHECKPOINT:-${RUN_ROOT}/broad/broad/checkpoint-256}"
WAIT_FOR_SUCCESS="${WAIT_FOR_SUCCESS:-outputs/qwen35_9b_full_sft/${STAMP_FULL_WAIT:-20260523-qwen35-9b-merge-full-sft}/checkpoint_eval/summary.json}"
WAIT_FOR_FAILURE="${WAIT_FOR_FAILURE:-outputs/qwen35_9b_full_sft/${STAMP_FULL_WAIT:-20260523-qwen35-9b-merge-full-sft}/full_sft_smoke_failed.txt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwen35_9b_terminology_chanka_sft/${STAMP}}"
EVAL_DIR="${EVAL_DIR:-outputs/gspo_checkpoint_evals/${STAMP}}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"

MAX_STEPS="${MAX_STEPS:-192}"
EVAL_STEPS="${EVAL_STEPS:-32}"
SAVE_STEPS="${SAVE_STEPS:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"

mkdir -p "$OUTPUT_ROOT" "$EVAL_DIR"

while [[ ! -d "$BROAD_CHECKPOINT" ]]; do
  echo "$(date -u +%FT%TZ) waiting for broad checkpoint: $BROAD_CHECKPOINT"
  sleep 300
done

while [[ ! -f "$WAIT_FOR_SUCCESS" && ! -f "$WAIT_FOR_FAILURE" ]]; do
  echo "$(date -u +%FT%TZ) waiting for 9B rerank/full-SFT chain before terminology SFT"
  echo "  success marker: $WAIT_FOR_SUCCESS"
  echo "  failure marker: $WAIT_FOR_FAILURE"
  sleep 300
done

if [[ ! -f "${OUTPUT_ROOT}/chanka/final_lora/adapter_config.json" ]]; then
  "$PYTHON" scripts/train_sft_unsloth.py \
    --stage chanka \
    --model-id unsloth/Qwen3.5-9B \
    --adapter-path "$BROAD_CHECKPOINT" \
    --output-dir "$OUTPUT_ROOT" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --max-steps "$MAX_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --logging-steps 16 \
    --per-device-train-batch-size 2 \
    --per-device-eval-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio 0.05 \
    --lora-r 64 \
    --lora-alpha 128 \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "$TERMINOLOGY_TOP_K"
fi

mapfile -t CHECKPOINTS < <(
  {
    find "${OUTPUT_ROOT}/chanka" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null
    find "${OUTPUT_ROOT}/chanka" -maxdepth 1 -type d -name 'final_lora' 2>/dev/null
  } | sort -V
)

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No terminology Chanka checkpoints found under ${OUTPUT_ROOT}/chanka" >&2
  exit 1
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
  name="$(basename "$checkpoint")"
  output_dir="${EVAL_DIR}/${name}"
  mkdir -p "$output_dir"
  if [[ -f "${output_dir}/metrics.json" ]]; then
    echo "Skipping existing terminology-SFT eval: $checkpoint"
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

"$PYTHON" - <<'PY' "$EVAL_DIR"
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
rows = []
for metrics_path in sorted(eval_dir.glob("*/metrics.json")):
    metrics = json.loads(metrics_path.read_text())
    rows.append(
        {
            "checkpoint": metrics_path.parent.name,
            "selection_score": metrics.get("selection_score"),
            "chrf++": metrics.get("chrf++"),
            "bleu": metrics.get("bleu"),
            "token_f1": metrics.get("token_f1"),
            "ter": metrics.get("ter"),
            "metrics_json": str(metrics_path),
        }
    )
rows.sort(key=lambda row: (row["selection_score"] is not None, row["selection_score"] or -1), reverse=True)
summary_path = eval_dir / "summary.json"
summary_path.write_text(json.dumps({"records": rows}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"best": rows[0] if rows else None, "summary_json": str(summary_path)}, ensure_ascii=False, indent=2))
PY
