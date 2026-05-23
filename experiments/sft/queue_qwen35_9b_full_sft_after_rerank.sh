#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-9b-merge-full-sft)}"

RUN_ROOT="${RUN_ROOT:-outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64}"
CHANKA_RUN_DIR="${CHANKA_RUN_DIR:-${RUN_ROOT}/chanka/chanka}"
CHECKPOINT_EVAL_DIR="${CHECKPOINT_EVAL_DIR:-outputs/gspo_checkpoint_evals/20260522-qwen35-curriculum-eval}"
WAIT_FOR_FILE="${WAIT_FOR_FILE:-outputs/qwen35_9b_candidate_rerank/20260523-qwen35-9b-fewshot2-candidate-rerank/score_ensemble/ensemble_current_4b_9b_summary.json}"

WORK_DIR="${WORK_DIR:-outputs/qwen35_9b_full_sft/${STAMP}}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-outputs/merged_full_models/${STAMP}-best-merged16}"
FULL_RUN_DIR="${FULL_RUN_DIR:-${WORK_DIR}/full_chanka_lr5e-7_32steps}"
SMOKE_DIR="${SMOKE_DIR:-${WORK_DIR}/smoke_full_chanka_lr5e-7_2steps}"
EVAL_DIR="${EVAL_DIR:-${WORK_DIR}/checkpoint_eval}"

TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
MAX_STEPS="${MAX_STEPS:-32}"
EVAL_STEPS="${EVAL_STEPS:-8}"
SAVE_STEPS="${SAVE_STEPS:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"

mkdir -p "$WORK_DIR" "$EVAL_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

while [[ ! -f "${CHECKPOINT_EVAL_DIR}/summary.json" ]]; do
  echo "$(date -u +%FT%TZ) waiting for checkpoint eval summary: ${CHECKPOINT_EVAL_DIR}/summary.json"
  sleep 300
done

while [[ ! -f "$WAIT_FOR_FILE" ]]; do
  echo "$(date -u +%FT%TZ) waiting for rerank summary before full SFT: $WAIT_FOR_FILE"
  sleep 300
done

BEST_CHECKPOINT="$(
  "$PYTHON" - <<'PY' "$CHECKPOINT_EVAL_DIR" "$CHANKA_RUN_DIR"
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
chanka_dir = Path(sys.argv[2])
summary = json.loads((eval_dir / "summary.json").read_text())
records = [row for row in summary.get("records", []) if row.get("selection_score") is not None]
if not records:
    raise SystemExit("No scored checkpoint records found")
best = max(records, key=lambda row: row["selection_score"])
checkpoint = chanka_dir / best["checkpoint"]
if not checkpoint.exists():
    raise SystemExit(f"Selected checkpoint does not exist: {checkpoint}")
print(checkpoint)
PY
)"

echo "$(date -u +%FT%TZ) selected 9B checkpoint for merge/full SFT: $BEST_CHECKPOINT"
echo "$BEST_CHECKPOINT" > "${WORK_DIR}/selected_checkpoint.txt"

if [[ ! -f "${MERGED_MODEL_DIR}/config.json" ]]; then
  "$PYTHON" scripts/export_unsloth_merged_model.py \
    --adapter-path "$BEST_CHECKPOINT" \
    --output-dir "$MERGED_MODEL_DIR" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --save-method merged_16bit
fi

if [[ ! -f "${SMOKE_DIR}/chanka/final_full_model/config.json" ]]; then
  echo "$(date -u +%FT%TZ) running 9B full-SFT memory smoke"
  if ! "$PYTHON" scripts/train_sft_unsloth.py \
    --stage chanka \
    --model-id "$MERGED_MODEL_DIR" \
    --training-mode full \
    --output-dir "$SMOKE_DIR" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --max-train-samples 32 \
    --max-eval-samples 8 \
    --max-steps 2 \
    --eval-steps 1 \
    --save-steps 1 \
    --logging-steps 1 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio 0.05; then
    echo "$(date -u +%FT%TZ) 9B full-SFT smoke failed; likely memory-bound on this GPU" | tee "${WORK_DIR}/full_sft_smoke_failed.txt"
    exit 0
  fi
fi

if [[ ! -f "${FULL_RUN_DIR}/chanka/final_full_model/config.json" ]]; then
  "$PYTHON" scripts/train_sft_unsloth.py \
    --stage chanka \
    --model-id "$MERGED_MODEL_DIR" \
    --training-mode full \
    --output-dir "$FULL_RUN_DIR" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --max-steps "$MAX_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --logging-steps 4 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio 0.05
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
