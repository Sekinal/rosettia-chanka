#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M)}"
BEST_ADAPTER="${BEST_ADAPTER:-outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora}"
CANARY_OUTPUT_DIR="${CANARY_OUTPUT_DIR:-outputs/gspo_paper_profiles/reference_rerank_vibe_current_best_${STAMP}}"
EVAL_DIR="${EVAL_DIR:-outputs/gspo_checkpoint_evals/${STAMP}-reference-rerank-canary}"

cd "$ROOT_DIR"
source .venv/bin/activate

python scripts/train_gspo_chanka_unsloth.py \
  --reward-profile reference_rerank_vibe_v1 \
  --adapter-path "$BEST_ADAPTER" \
  --output-dir "$CANARY_OUTPUT_DIR" \
  --learning-rate "${GSPO_LEARNING_RATE:-5e-7}" \
  --warmup-ratio "${GSPO_WARMUP_RATIO:-0.01}" \
  --num-train-epochs "${GSPO_NUM_TRAIN_EPOCHS:-1}" \
  --per-device-train-batch-size "${GSPO_TRAIN_BATCH_SIZE:-8}" \
  --per-device-eval-batch-size "${GSPO_EVAL_BATCH_SIZE:-8}" \
  --gradient-accumulation-steps "${GSPO_GRADIENT_ACCUMULATION_STEPS:-1}" \
  --num-generations "${GSPO_NUM_GENERATIONS:-8}" \
  --logging-steps "${GSPO_LOGGING_STEPS:-4}" \
  --no-log-completions \
  --temperature "${GSPO_TEMPERATURE:-0.85}" \
  --top-p "${GSPO_TOP_P:-0.95}" \
  --max-steps "${GSPO_MAX_STEPS:-24}" \
  --max-train-samples "${GSPO_MAX_TRAIN_SAMPLES:-256}" \
  --max-eval-samples "${GSPO_MAX_EVAL_SAMPLES:-64}" \
  --eval-steps "${GSPO_EVAL_STEPS:-8}" \
  --save-steps "${GSPO_SAVE_STEPS:-8}"

mkdir -p "$EVAL_DIR"
run_root="$CANARY_OUTPUT_DIR/chanka_gspo"
for item in checkpoint-8 checkpoint-16 checkpoint-24 final_gspo_lora; do
  adapter_path="$run_root/$item"
  [[ -d "$adapter_path" ]] || continue
  label="reference_rerank_${item//-/_}"
  python scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$adapter_path" \
    --output-json "$EVAL_DIR/${label}_metrics.json" \
    --predictions-jsonl "$EVAL_DIR/${label}_predictions.jsonl" \
    --batch-size 1
done

python scripts/summarize_gspo_checkpoint_evals.py "$EVAL_DIR"
cat "$EVAL_DIR/summary.md"
