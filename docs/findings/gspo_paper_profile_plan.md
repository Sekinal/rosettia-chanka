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

## Canary Results

Remote: `root@216.81.248.197 -p 20299`, path `/root/rosettia-chanka`, GPU `NVIDIA L40S`.

Date: 2026-05-21

Canary sweep: `outputs/gspo_canary_sweeps/20260521-paper-mix-v1`

The sweep ranked candidates by external corpus metrics rather than raw trainer reward. `vibethinker_2511` was the best canary with selection score `21.0148`, chrF++ `37.0672`, BLEU `6.4582`, token F1 `18.2594`, source-copy ratio `4.0551%`, exact-copy rate `1.5625%`, and Spanish leakage `0.3906%`.

The first full VibeThinker contender reached `checkpoint-196` with eval reward `0.30345`, but the process later stopped using GPU SM while still holding GPU memory. It had been launched with completion-table logging every two steps. The serious run was restarted from that GSPO checkpoint as `outputs/gspo_full_contenders/vibethinker_2511_continued_no_tables_20260521-083900` with `--no-log-completions` and `MAX_STEPS=1598`, preserving the useful checkpoint progress while avoiding the logging bottleneck. The restarted run produced `checkpoint-28` with eval reward `0.31191` and `checkpoint-56` with eval reward `0.30850`.

After `checkpoint-56`, the VibeThinker run was relaunched with true trainer checkpoint resume via `--resume-from-checkpoint` and validation/checkpoint cadence changed from 28 to 112 steps. TRL loaded the checkpoint successfully, but preserved the checkpoint's old 28-step eval/save cadence despite the new arguments. That run produced `checkpoint-84` with eval reward `0.30904` and `checkpoint-112` with eval reward `0.31448`.

To actually reduce validation overhead, phase 2 was launched as a fresh adapter-seeded continuation from `checkpoint-112`: `outputs/gspo_full_contenders/vibethinker_2511_phase2_from112_eval112_20260521-0909`. It uses `EVAL_STEPS=112`, `SAVE_STEPS=112`, `MAX_STEPS=1486`, `WARMUP_RATIO=0.01`, and `--no-log-completions`.

Hard-negative verifier SFT is running at `outputs/chanka_translation_verifier_hard_r128`. As of checkpoint `1083`, the best validation loss remains `0.002326` from checkpoint `912`; current eval loss was `0.002389`. Treat additional verifier loss gains as secondary to whether the learned verifier improves GSPO outcomes.

Update as of 2026-05-21 10:17 UTC:

- The hard-negative verifier SFT reached `checkpoint-1383` but crashed during final save with CUDA illegal memory access. Best validation checkpoint is `checkpoint-1368` with eval loss `0.0022854567505419254`; checkpoint `1383` eval loss was `0.002302516484633088`. On the remote only, `final_verifier_lora` is a symlink to `checkpoint-1368`.
- Learned verifier from the broad SFT adapter alone did not beat Vibe: `outputs/gspo_paper_profiles/2511_learned_verifier_hard_canary_20260521-0936` ended with chrF++ `35.77395`, BLEU `6.19078`, token F1 `17.04413`, exact-copy `1.5625%`, source-copy `4.0551%`, leakage `0.3906%`, TER `102.89017`, and trainer eval reward `0.23386`.
- Learned verifier applied on top of the Vibe checkpoint did beat the previous best canary externally: `outputs/gspo_paper_profiles/2511_learned_verifier_on_vibe112_canary_20260521-1000` ended with chrF++ `38.20628`, BLEU `7.28880`, token F1 `18.97321`, exact-copy `1.5625%`, source-copy `4.1667%`, leakage `0.3906%`, TER `96.53179`, and trainer eval reward `0.24287`.
- New mix-and-match profile `learned_verifier_vibe_2511` combines the hard-negative verifier reward with the VibeThinker group diversity bonus. This tests whether the current best recipe is improved by mixing the verifier reward itself with Vibe exploration, rather than only seeding learned-verifier GSPO from a Vibe checkpoint.
- `scripts/evaluate_gspo_checkpoint.py` evaluates arbitrary saved GSPO checkpoints on the same clean Chanka validation split and writes external metrics plus optional predictions. Use this for checkpoint selection instead of trusting final-step metrics only.
- Queue script `experiments/gspo/queue_learned_verifier_vibe_canaries.sh` waits until current GSPO jobs finish, evaluates selected full-run checkpoints externally, then runs 4-generation and 8-generation canaries for `learned_verifier_vibe_2511` from the Vibe checkpoint with the hard verifier.
- The serious current contender is `outputs/gspo_full_contenders/learned_verifier_on_vibe112_full_20260521-1016`, seeded from `outputs/gspo_full_contenders/vibethinker_2511_continued_no_tables_20260521-083900/chanka_gspo/checkpoint-112` with verifier `outputs/chanka_translation_verifier_hard_r128/checkpoint-1368`. It uses `MAX_STEPS=224`, `EVAL_STEPS=56`, `SAVE_STEPS=56`, `LEARNING_RATE=7e-7`, `WARMUP_RATIO=0.01`, `NUM_GENERATIONS=4`, verifier reward batch size `2`, and no completion-table logging.
- Full learned-verifier-on-Vibe checkpoints: `checkpoint-56` eval reward `0.25755`, eval loss `0.02497`, eval runtime `762.2s`; `checkpoint-112` eval reward `0.26144`, eval loss `0.04412`, eval runtime `768.6s`. This remains above the learned-verifier-on-Vibe canary trainer eval reward `0.24287`, so keep the full run alive and compare final external metrics once available. A copy of `checkpoint-112` was preserved on the remote at `outputs/gspo_selected_checkpoints/learned_verifier_on_vibe112_checkpoint_112`.
- The Vibe phase-2 continuation `outputs/gspo_full_contenders/vibethinker_2511_phase2_from112_eval112_20260521-0909` is also still running. It peaked so far at `checkpoint-896` with eval reward `0.33324`, then dropped to `0.33250` at `checkpoint-1008` and `0.32695` at `checkpoint-1120`. A copy of `checkpoint-896` was preserved on the remote at `outputs/gspo_selected_checkpoints/vibethinker_phase2_checkpoint_896` before save rotation could delete it. Reward scales are profile-specific; final external metrics matter most.

## Caveats

- These are proxy implementations, not exact reproductions. We do not have a Chanka-capable xCOMET model, and the severity proxy should not be described as xCOMET. It is only a cheap ablation/guardrail; serious reward-model work should prioritize the learned Chanka verifier.
- Do not use the xCOMET-inspired proxy as the optimization target. It is useful only for triage and transparent failure accounting. Since a real Chanka-capable xCOMET judge is unavailable, the DeepSeek-style trained verifier plus external lexical sanity metrics is the more defensible reward direction.
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
