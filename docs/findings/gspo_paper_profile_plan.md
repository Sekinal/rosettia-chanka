# GSPO Paper Profile Plan

Date: 2026-05-21

Goal: test paper-inspired GSPO reward profiles independently for general Spanish to Chanka Quechua translation. The reviewed legal Chanka corpus is used as a clean dialect anchor and reward/eval source, not as a legal-only product objective.

## Shared Setup

- Base adapter: `outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400`.
- Base model: `unsloth/Qwen3.5-2B`.
- Training script: `scripts/train_gspo_chanka_unsloth.py`.
- Prompt: general Spanish to Quechua Chanka translation. It asks for faithful, natural translation and explicitly discourages Spanish copying.
- Default comparison length: one epoch unless `MAX_STEPS` is set for a smoke run.
- Selection metric: checkpoint reward plus qualitative samples, not final-step loss alone.
- Batching: Unsloth GRPO is sensitive to `num_generations`. Keep per-device train and eval batch sizes divisible by `NUM_GENERATIONS`; do not rely on gradient accumulation to fix a non-divisible microbatch.

## Profiles

### 2411.05986 Fine-Grained Severity

Script: `experiments/gspo/run_2411_fine_grained_severity.sh`

Paper idea: sentence-level MT rewards are sparse; token/error-span severity rewards give denser and more stable training signals.

Implementation proxy:

- score chrF++, BLEU, token F1, length sanity, and entity preservation;
- subtract a severity map for source copying, Spanish leakage, low overlap, bad length, missing entities, and repeated n-grams.

### 2310.10482 Severity Proxy

Script: `experiments/gspo/run_2310_severity_proxy.sh`

Paper idea: xCOMET combines sentence quality with transparent error span/severity detection.

Implementation note: we do not have, and should not claim to have, a real Chanka-capable xCOMET model. This run is only a transparent severity-weighted ablation inspired by the error-span/severity framing.

- keep a sentence-level score;
- subtract a stronger severity penalty, treating source-copy and leakage as explicit error spans.

### 2511.22570 Self-Verifier

Script: `experiments/gspo/run_2511_self_verifier.sh`

Paper idea: a separate verifier can train a generator by identifying issues and scoring outputs; a meta-verifier can improve faithfulness.

Implementation proxy:

- reward meaning similarity, anti-copy behavior, anti-leakage behavior, length sanity, entity preservation, and repetition control;
- penalize exact source copying heavily;
- treat the reward as a verifier rubric until a learned verifier is available.

Verifier training script: `scripts/train_verifier_chanka_unsloth.py`

Chained script: `experiments/gspo/run_2511_train_verifier_then_gspo.sh`

Learned-verifier script: `experiments/gspo/run_2511_learned_verifier_gspo.sh`

The verifier path bootstraps labels from clean Chanka pairs plus synthetic corruptions: correct reference translation, source copy, incomplete translation, Spanish leakage, word-order/fluency damage, repetition damage, fluent unrelated Chanka translations, mixed-reference translations, and unsupported Chanka additions. It trains a separate JSON-scoring LoRA verifier with validation enabled. The `learned_verifier_2511` reward profile can then load the verifier LoRA directly in the GSPO reward loop and blend its score with hard anti-copy, anti-leakage, repetition, and reference guards. This is much slower and uses more memory than metric-only rewards, but it is the preferred DeepSeek-style branch now that training cost is not the deciding constraint.

### 2511.06221 VibeThinker

Script: `experiments/gspo/run_2511_vibethinker.sh`

Paper idea: preserve a broad output spectrum, then amplify the best signal with RL. Diversity matters because pass@K gains come from exploring multiple valid candidates.

Implementation proxy:

- use `NUM_GENERATIONS=8` by default;
- add a small within-source diversity bonus;
- penalize duplicate generations and repeated n-grams.

## Run Order

Use smoke runs first:

```bash
MAX_STEPS=8 MAX_TRAIN_SAMPLES=32 MAX_EVAL_SAMPLES=8 experiments/gspo/run_2411_fine_grained_severity.sh
MAX_STEPS=8 MAX_TRAIN_SAMPLES=32 MAX_EVAL_SAMPLES=8 experiments/gspo/run_2310_severity_proxy.sh
MAX_STEPS=8 MAX_TRAIN_SAMPLES=32 MAX_EVAL_SAMPLES=8 experiments/gspo/run_2511_self_verifier.sh
MAX_STEPS=8 MAX_TRAIN_SAMPLES=32 MAX_EVAL_SAMPLES=8 experiments/gspo/run_2511_vibethinker.sh
VERIFIER_MAX_STEPS=8 GSPO_MAX_STEPS=8 MAX_TRAIN_SAMPLES=32 MAX_EVAL_SAMPLES=8 experiments/gspo/run_2511_train_verifier_then_gspo.sh
```

Then run the best two full one-epoch profiles and compare against the existing GSPO baseline. Prioritize `self_verifier_2511` and the chained verifier path over the severity proxy if GPU time is not the constraint.

For the finding phase, use a canary sweep instead of single-step smokes:

```bash
MAX_STEPS=24 MAX_TRAIN_SAMPLES=256 MAX_EVAL_SAMPLES=64 experiments/gspo/run_canary_sweep.sh
```

The canary sweep runs individual paper profiles plus mixed rewards:

- `mix_severity_verifier`: fine-grained severity + severity proxy + verifier rubric;
- `mix_verifier_vibe`: verifier rubric + VibeThinker diversity, with eight generations;
- `mix_all_strict`: all paper-inspired signals plus stronger copy/leakage guards;
- `rosettia_guard_v1` and `rosettia_guard_v2`: project-specific rewards designed to prioritize reference overlap, anti-copy behavior, anti-leakage behavior, length sanity, entity preservation, and repetition control.
- `learned_verifier_2511`: loads a trained verifier LoRA and uses its JSON score as the dominant reward, with project guards to prevent source-copy and Spanish-leakage reward hacking.

Each run writes `final_metrics.json`, and `scripts/summarize_gspo_canaries.py` produces `summary.jsonl` plus `summary.md`. The summary ranks by a profile-comparable `selection_score` based on external corpus metrics: chrF++, BLEU, token F1, length sanity, source-copy rate, exact-copy rate, Spanish leakage, and TER. Do not rank mixed profiles by trainer eval reward alone because each reward profile has a different scale.

Once a sweep completes, launch the top-ranked full contender with:

```bash
SWEEP_DIR=outputs/gspo_canary_sweeps/20260521-paper-mix-v1 experiments/gspo/run_best_canary_full.sh
```

The full-contender launcher reruns the summary, selects the first `summary.jsonl` profile unless `PROFILE` is set, and defaults to two epochs over the full clean Chanka corpus with validation/checkpoints every 28 steps.

## Caveats

- These are proxy implementations, not exact reproductions. We do not have a Chanka-capable xCOMET model, and the severity proxy should not be described as xCOMET. It is only a cheap ablation/guardrail; serious reward-model work should prioritize the learned Chanka verifier.
- BLEU/chrF alone are not sufficient. The previous GSPO final adapter showed source-copy risk, so qualitative samples and copy/leakage metrics must be reviewed.
- The VibeThinker profile is slower because it defaults to eight generations and therefore needs `TRAIN_BATCH_SIZE=8` / `EVAL_BATCH_SIZE=8` unless `NUM_GENERATIONS` is lowered.

## Smoke Results

Remote: `root@216.81.248.197 -p 20299`, path `/root/rosettia-chanka`, GPU `NVIDIA L40S`.

Date: 2026-05-21

All smoke runs used tiny slices (`MAX_STEPS=1`, `MAX_TRAIN_SAMPLES=8`, `MAX_EVAL_SAMPLES=8`) only to validate wiring, batching, reward execution, checkpointing, validation, and final metric writing. These numbers are not model-quality claims.

| Run | Status | Eval reward | Notes |
| --- | --- | ---: | --- |
| `run_2411_fine_grained_severity.sh` | passed | -0.0557 | Original smoke before renaming/batch tightening. |
| `run_2310_severity_proxy.sh` | passed | -0.0327 | Renamed from xCOMET proxy to avoid overclaiming; final metrics wrote successfully. |
| `run_2511_self_verifier.sh` | passed | 0.4377 | Rubric-style verifier reward only; no learned verifier in-loop yet. |
| `run_2511_vibethinker.sh` | passed after batch fix | 0.2521 | Original `TRAIN_BATCH_SIZE=2`, `NUM_GENERATIONS=8` crashed with a reshape error; fixed default is `TRAIN_BATCH_SIZE=8`. |
| `run_2511_train_verifier_then_gspo.sh` | passed | 0.4377 for GSPO half | Verifier SFT step eval loss was 2.238 on the 8-example smoke; a final post-train `evaluate()` returned `nan`, so full verifier runs should use larger validation slices and inspect checkpoint eval logs. |

Common final corpus metrics on the tiny 8-row smoke slice: chrF++ 32.718, BLEU 3.129, token F1 12.292, exact source-copy rate 0.0%, source-copy ratio 3.125%, Spanish leakage penalty 0.0%.
