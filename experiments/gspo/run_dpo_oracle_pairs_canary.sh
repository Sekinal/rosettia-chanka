#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date -u +%Y%m%d-dpo-oracle-pairs)}"
PYTHON="${PYTHON:-.venv/bin/python}"
PREDICTIONS_JSONL="${PREDICTIONS_JSONL:-outputs/verifier_candidate_mining/20260522-train-k16-full-newbest-noterm-t065-p090/train_k16_predictions.jsonl}"
DATA_DIR="${DATA_DIR:-outputs/dpo_preference_data/${STAMP}}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-outputs/dpo_preference_runs/${STAMP}}"
EVAL_DIR="${EVAL_DIR:-outputs/gspo_checkpoint_evals/${STAMP}}"
BASE_ADAPTER="${BASE_ADAPTER:-outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"

mkdir -p "$DATA_DIR" "$DPO_OUTPUT_DIR" "$EVAL_DIR"

"$PYTHON" scripts/build_oracle_preference_pairs.py \
  --predictions-jsonl "$PREDICTIONS_JSONL" \
  --output-jsonl "$DATA_DIR/oracle_preference_pairs.jsonl" \
  --metrics-json "$DATA_DIR/oracle_preference_pairs_metrics.json" \
  --summary-md "$DATA_DIR/oracle_preference_pairs_summary.md" \
  --min-candidates "${MIN_CANDIDATES:-8}" \
  --min-margin "${MIN_MARGIN:-0.50}" \
  --rejected-strategy "${REJECTED_STRATEGY:-hard}"

"$PYTHON" scripts/train_dpo_unsloth.py \
  --jsonl "$DATA_DIR/oracle_preference_pairs.jsonl" \
  --adapter-path "$BASE_ADAPTER" \
  --reference-adapter-path "$BASE_ADAPTER" \
  --terminology-file "$TERMINOLOGY_FILE" \
  --terminology-top-k "${TERMINOLOGY_TOP_K:-1}" \
  --max-steps "${DPO_MAX_STEPS:-16}" \
  --eval-steps "${DPO_EVAL_STEPS:-8}" \
  --save-steps "${DPO_SAVE_STEPS:-8}" \
  --max-train-samples "${DPO_MAX_TRAIN_SAMPLES:-256}" \
  --max-eval-samples "${DPO_MAX_EVAL_SAMPLES:-64}" \
  --learning-rate "${DPO_LEARNING_RATE:-1e-7}" \
  --beta "${DPO_BETA:-0.05}" \
  --per-device-train-batch-size "${DPO_TRAIN_BATCH_SIZE:-1}" \
  --per-device-eval-batch-size "${DPO_EVAL_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${DPO_GRADIENT_ACCUMULATION_STEPS:-8}" \
  --max-seq-length "${DPO_MAX_SEQ_LENGTH:-192}" \
  --max-prompt-length "${DPO_MAX_PROMPT_LENGTH:-128}" \
  --max-completion-length "${DPO_MAX_COMPLETION_LENGTH:-64}" \
  --output-dir "$DPO_OUTPUT_DIR"

for item in checkpoint-8 checkpoint-16 final_dpo_lora; do
  adapter="$DPO_OUTPUT_DIR/$item"
  [[ -d "$adapter" ]] || continue
  label="${item}_terminology_top1"
  "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$adapter" \
    --batch-size 1 \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "${TERMINOLOGY_TOP_K:-1}" \
    --output-json "$EVAL_DIR/${label}_metrics.json" \
    --predictions-jsonl "$EVAL_DIR/${label}_predictions.jsonl"
done

"$PYTHON" scripts/summarize_gspo_checkpoint_evals.py "$EVAL_DIR"
