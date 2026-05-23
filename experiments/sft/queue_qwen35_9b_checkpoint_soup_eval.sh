#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-9b-checkpoint-soup)}"

RUN_ROOT="${RUN_ROOT:-outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64}"
CHANKA_RUN_DIR="${CHANKA_RUN_DIR:-${RUN_ROOT}/chanka/chanka}"
CHECKPOINT_EVAL_DIR="${CHECKPOINT_EVAL_DIR:-outputs/gspo_checkpoint_evals/20260522-qwen35-curriculum-eval}"
WAIT_FOR_FILE="${WAIT_FOR_FILE:-outputs/qwen35_9b_candidate_rerank/20260523-qwen35-9b-fewshot2-candidate-rerank/score_ensemble/ensemble_current_4b_9b_summary.json}"

WORK_DIR="${WORK_DIR:-outputs/qwen35_9b_checkpoint_soups/${STAMP}}"
SOUP_DIR="${SOUP_DIR:-${WORK_DIR}/top3_weighted_soup}"
EVAL_DIR="${EVAL_DIR:-${WORK_DIR}/eval}"
SOUP_SIZE="${SOUP_SIZE:-3}"
WEIGHTS="${WEIGHTS:-0.50,0.30,0.20}"

TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-80}"
PROGRESS_EVERY="${PROGRESS_EVERY:-32}"

mkdir -p "$WORK_DIR" "$EVAL_DIR"

while [[ ! -f "${CHECKPOINT_EVAL_DIR}/summary.json" ]]; do
  echo "$(date -u +%FT%TZ) waiting for checkpoint eval summary: ${CHECKPOINT_EVAL_DIR}/summary.json"
  sleep 300
done

while [[ -n "$WAIT_FOR_FILE" && ! -f "$WAIT_FOR_FILE" ]]; do
  echo "$(date -u +%FT%TZ) waiting before checkpoint soup to avoid GPU contention: $WAIT_FOR_FILE"
  sleep 300
done

SELECTION_JSON="${WORK_DIR}/selected_checkpoints.json"
"$PYTHON" - <<'PY' "$CHECKPOINT_EVAL_DIR" "$CHANKA_RUN_DIR" "$SOUP_SIZE" "$WEIGHTS" "$SELECTION_JSON"
import json
import sys
from pathlib import Path

from scripts.summarize_gspo_canaries import selection_score

eval_dir = Path(sys.argv[1])
chanka_dir = Path(sys.argv[2])
soup_size = int(sys.argv[3])
raw_weights = [float(item) for item in sys.argv[4].split(",") if item.strip()]
output_path = Path(sys.argv[5])

summary = json.loads((eval_dir / "summary.json").read_text())
records = []
for row in summary.get("records", []):
    score = row.get("selection_score")
    if score is None and row.get("metrics_json"):
        score = selection_score(json.loads(Path(row["metrics_json"]).read_text()))
        row = {**row, "selection_score": score}
    if score is not None:
        records.append(row)
records.sort(key=lambda row: row["selection_score"], reverse=True)

selected = []
for row in records:
    checkpoint = chanka_dir / row["checkpoint"]
    if (checkpoint / "adapter_model.safetensors").exists() and (checkpoint / "adapter_config.json").exists():
        selected.append({**row, "adapter_path": str(checkpoint)})
    if len(selected) >= soup_size:
        break

if not selected:
    raise SystemExit("No compatible LoRA checkpoints found for soup")

weights = raw_weights[: len(selected)]
if len(weights) < len(selected):
    weights.extend([1.0] * (len(selected) - len(weights)))
total = sum(weights)
weights = [weight / total for weight in weights]

payload = {"selected": selected, "weights": weights}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

if [[ ! -f "${SOUP_DIR}/adapter_model.safetensors" ]]; then
  mapfile -t MERGE_ARGS < <(
    "$PYTHON" - <<'PY' "$SELECTION_JSON"
import json
import shlex
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
for row in payload["selected"]:
    print("--adapter")
    print(shlex.quote(row["adapter_path"]))
for weight in payload["weights"]:
    print("--weight")
    print(str(weight))
PY
  )
  "$PYTHON" scripts/merge_lora_adapters.py \
    "${MERGE_ARGS[@]}" \
    --output-dir "$SOUP_DIR" \
    --name "qwen35_9b_top${SOUP_SIZE}_weighted_checkpoint_soup"
fi

if [[ ! -f "${EVAL_DIR}/metrics.json" ]]; then
  "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$SOUP_DIR" \
    --output-json "${EVAL_DIR}/metrics.json" \
    --predictions-jsonl "${EVAL_DIR}/predictions.jsonl" \
    --batch-size "$BATCH_SIZE" \
    --max-completion-length "$MAX_COMPLETION_LENGTH" \
    --strip-chat-artifacts \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "$TERMINOLOGY_TOP_K" \
    --progress-every "$PROGRESS_EVERY"
fi

"$PYTHON" - <<'PY' "$EVAL_DIR" "$SELECTION_JSON" "$SOUP_DIR" "$WORK_DIR"
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
soup_dir = Path(sys.argv[3])
work_dir = Path(sys.argv[4])
metrics = json.loads((eval_dir / "metrics.json").read_text())
selection = json.loads(selection_path.read_text())
score = metrics.get("selection_score")
if score is None:
    from scripts.summarize_gspo_canaries import selection_score

    score = selection_score(metrics)
summary = {
    "soup_dir": str(soup_dir),
    "selected": selection["selected"],
    "weights": selection["weights"],
    "metrics": {
        "selection_score": score,
        "chrf++": metrics.get("chrf++"),
        "bleu": metrics.get("bleu"),
        "token_f1": metrics.get("token_f1"),
        "ter": metrics.get("ter"),
    },
}
(work_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
