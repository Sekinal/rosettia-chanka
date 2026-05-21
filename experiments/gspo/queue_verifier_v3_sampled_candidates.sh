#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M)}"
BEST_ADAPTER="${BEST_ADAPTER:-outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora}"
CANDIDATE_DIR="${CANDIDATE_DIR:-outputs/verifier_candidate_mining/${STAMP}}"
VERIFIER_OUTPUT_DIR="${VERIFIER_OUTPUT_DIR:-outputs/chanka_translation_verifier_sampled_candidates_v3_${STAMP}}"
VERIFIER_SELECTED_DIR="${VERIFIER_SELECTED_DIR:-outputs/verifier_selected_checkpoints}"
CANARY_OUTPUT_DIR="${CANARY_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_verifier_v3_sampled_candidates_on_best_${STAMP}}"
EVAL_DIR="${EVAL_DIR:-outputs/gspo_checkpoint_evals/${STAMP}-verifier-v3-canary}"

cd "$ROOT_DIR"
source .venv/bin/activate

while pgrep -f "python scripts/train_gspo_chanka_unsloth.py|python scripts/evaluate_gspo_checkpoint.py|python scripts/train_verifier_chanka_unsloth.py" >/dev/null; do
  date -u
  pgrep -af "python scripts/train_gspo_chanka_unsloth.py|python scripts/evaluate_gspo_checkpoint.py|python scripts/train_verifier_chanka_unsloth.py" || true
  sleep "$POLL_SECONDS"
done

mkdir -p "$CANDIDATE_DIR" "$VERIFIER_SELECTED_DIR" "$EVAL_DIR"

read -r -a candidate_specs <<< "${CANDIDATE_SPECS:-t075_p090:0.75:0.90 t095_p095:0.95:0.95}"
candidate_jsonl=()
for spec in "${candidate_specs[@]}"; do
  IFS=: read -r label temperature top_p <<< "$spec"
  metrics_path="$CANDIDATE_DIR/${label}_metrics.json"
  predictions_path="$CANDIDATE_DIR/${label}_predictions.jsonl"
  if [[ ! -f "$predictions_path" ]]; then
    python scripts/evaluate_gspo_checkpoint.py \
      --adapter-path "$BEST_ADAPTER" \
      --split "${CANDIDATE_SPLIT:-all}" \
      --num-return-sequences "${CANDIDATE_NUM_RETURN_SEQUENCES:-4}" \
      --do-sample \
      --temperature "$temperature" \
      --top-p "$top_p" \
      --top-k "${CANDIDATE_TOP_K:-50}" \
      --batch-size "${CANDIDATE_BATCH_SIZE:-8}" \
      --max-completion-length "${CANDIDATE_MAX_COMPLETION_LENGTH:-80}" \
      --output-json "$metrics_path" \
      --predictions-jsonl "$predictions_path"
  fi
  candidate_jsonl+=( "$predictions_path" )
done

candidate_args=()
for path in "${candidate_jsonl[@]}"; do
  candidate_args+=(--candidate-jsonl "$path")
done

set +e
python scripts/train_verifier_chanka_unsloth.py \
  --output-dir "$VERIFIER_OUTPUT_DIR" \
  "${candidate_args[@]}" \
  --candidate-max-examples "${CANDIDATE_MAX_EXAMPLES:-12000}" \
  --num-train-epochs "${VERIFIER_NUM_TRAIN_EPOCHS:-2}" \
  --max-steps "${VERIFIER_MAX_STEPS:-1200}" \
  --learning-rate "${VERIFIER_LEARNING_RATE:-2e-5}" \
  --warmup-ratio "${VERIFIER_WARMUP_RATIO:-0.03}" \
  --lora-r "${VERIFIER_LORA_R:-128}" \
  --lora-alpha "${VERIFIER_LORA_ALPHA:-256}" \
  --per-device-train-batch-size "${VERIFIER_TRAIN_BATCH_SIZE:-4}" \
  --per-device-eval-batch-size "${VERIFIER_EVAL_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${VERIFIER_GRADIENT_ACCUMULATION_STEPS:-4}" \
  --evals-per-epoch "${VERIFIER_EVALS_PER_EPOCH:-10}" \
  --logging-steps "${VERIFIER_LOGGING_STEPS:-10}"
verifier_status="$?"
set -e

verifier_adapter="$VERIFIER_OUTPUT_DIR/final_verifier_lora"
if [[ ! -d "$verifier_adapter" ]]; then
  verifier_adapter="$(
    python - "$VERIFIER_OUTPUT_DIR" "$VERIFIER_SELECTED_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
selected = Path(sys.argv[2])
best_path = None
best_loss = math.inf
for search_root in (root, selected):
    for state_path in sorted(search_root.glob("**/trainer_state.json")):
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            continue
        for row in state.get("log_history", []):
            if "eval_loss" not in row:
                continue
            loss = float(row["eval_loss"])
            if loss < best_loss:
                best_loss = loss
                best_path = state_path.parent

if best_path is None:
    checkpoints = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    best_path = checkpoints[-1] if checkpoints else None
if best_path is None:
    raise SystemExit("no verifier checkpoint found")
print(best_path)
PY
  )"
fi

if [[ "$verifier_status" -ne 0 ]]; then
  echo "Verifier training exited with status $verifier_status; continuing with checkpoint: $verifier_adapter" >&2
fi

env \
  ROOT_DIR="$ROOT_DIR" \
  GSPO_REWARD_PROFILE=learned_verifier_vibe_2511 \
  VERIFIER_ADAPTER_PATH="$verifier_adapter" \
  VERIFIER_REWARD_BATCH_SIZE="${VERIFIER_REWARD_BATCH_SIZE:-2}" \
  SFT_ADAPTER_PATH="$BEST_ADAPTER" \
  GSPO_OUTPUT_DIR="$CANARY_OUTPUT_DIR" \
  LEARNING_RATE="${GSPO_LEARNING_RATE:-7e-7}" \
  WARMUP_RATIO="${GSPO_WARMUP_RATIO:-0.01}" \
  NUM_TRAIN_EPOCHS="${GSPO_NUM_TRAIN_EPOCHS:-2}" \
  TRAIN_BATCH_SIZE="${GSPO_TRAIN_BATCH_SIZE:-4}" \
  EVAL_BATCH_SIZE="${GSPO_EVAL_BATCH_SIZE:-4}" \
  GRADIENT_ACCUMULATION_STEPS="${GSPO_GRADIENT_ACCUMULATION_STEPS:-2}" \
  NUM_GENERATIONS="${GSPO_NUM_GENERATIONS:-4}" \
  LOGGING_STEPS="${GSPO_LOGGING_STEPS:-4}" \
  TEMPERATURE="${GSPO_TEMPERATURE:-0.75}" \
  TOP_P="${GSPO_TOP_P:-0.90}" \
  MAX_STEPS="${GSPO_MAX_STEPS:-24}" \
  MAX_TRAIN_SAMPLES="${GSPO_MAX_TRAIN_SAMPLES:-256}" \
  MAX_EVAL_SAMPLES="${GSPO_MAX_EVAL_SAMPLES:-64}" \
  EVAL_STEPS="${GSPO_EVAL_STEPS:-8}" \
  SAVE_STEPS="${GSPO_SAVE_STEPS:-8}" \
  bash experiments/gspo/run_2511_learned_verifier_gspo.sh

run_root="$CANARY_OUTPUT_DIR/chanka_gspo"
for item in checkpoint-8 checkpoint-16 checkpoint-24 final_gspo_lora; do
  adapter_path="$run_root/$item"
  [[ -d "$adapter_path" ]] || continue
  label="verifier_v3_sampled_${item//-/_}"
  python scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$adapter_path" \
    --output-json "$EVAL_DIR/${label}_metrics.json" \
    --predictions-jsonl "$EVAL_DIR/${label}_predictions.jsonl" \
    --batch-size 1
done

python scripts/summarize_gspo_checkpoint_evals.py "$EVAL_DIR"
cat "$EVAL_DIR/summary.md"
