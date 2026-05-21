# GSPO Paper Profile Plan

Date: 2026-05-21

Goal: test paper-inspired GSPO reward profiles independently for general Spanish to Chanka Quechua translation. The reviewed legal Chanka corpus is used as a clean dialect anchor and reward/eval source, not as a legal-only product objective.

## Shared Setup

- Base adapter: `outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400`.
- Base model: `unsloth/Qwen3.5-2B`.
- Training script: `scripts/train_gspo_chanka_unsloth.py`.
- Prompt: general Spanish to Quechua Chanka translation. It asks for faithful, natural translation and explicitly discourages Spanish copying.
- Default comparison length: one epoch unless `MAX_STEPS` is set for a smoke run.
- Selection metric: checkpoint reward plus qualitative samples, not final-step loss alone. xCOMET-like proxies are useful only as diagnostics/regression guards; they are not a substitute for a Chanka-capable evaluator.
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

The verifier path bootstraps labels from clean Chanka pairs plus synthetic corruptions: correct reference translation, source copy, incomplete translation, Spanish leakage, word-order/fluency damage, repetition damage, fluent unrelated Chanka translations, mixed-reference translations, and unsupported Chanka additions. It trains a separate JSON-scoring LoRA verifier with validation enabled. The `learned_verifier_2511` reward profile can then load the verifier LoRA directly in the GSPO reward loop and blend its score with hard anti-copy, anti-leakage, repetition, and reference guards. This is much slower and uses more memory than metric-only rewards, but it is the preferred DeepSeek-style branch now that training cost is not the deciding constraint. If the current verifier is not discriminative enough, the next escalation is a larger or pairwise/listwise verifier trained from multiple candidate translations per source, with hard negatives sampled from every strong GSPO checkpoint. `scripts/train_verifier_chanka_unsloth.py --candidate-jsonl <predictions.jsonl>` can already ingest checkpoint-eval predictions as real hard-negative verifier examples for that second pass.

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
- Full learned-verifier-on-Vibe checkpoints: `checkpoint-56` eval reward `0.25755`, eval loss `0.02497`, eval runtime `762.2s`; `checkpoint-112` eval reward `0.26144`, eval loss `0.04412`, eval runtime `768.6s`; `checkpoint-168` eval reward `0.25736`, eval loss `0.02710`, eval runtime `717.1s`. Checkpoint `112` remains the best learned-verifier full checkpoint so far and was preserved on the remote at `outputs/gspo_selected_checkpoints/learned_verifier_on_vibe112_checkpoint_112`.
- The Vibe phase-2 continuation `outputs/gspo_full_contenders/vibethinker_2511_phase2_from112_eval112_20260521-0909` was stopped after sustained decline. It peaked at `checkpoint-896` with eval reward `0.33324`, then dropped to `0.33250` at `checkpoint-1008`, `0.32695` at `checkpoint-1120`, and `0.32441` at `checkpoint-1232`. A copy of `checkpoint-896` was preserved on the remote at `outputs/gspo_selected_checkpoints/vibethinker_phase2_checkpoint_896` before save rotation could delete it. Reward scales are profile-specific; final external metrics matter most.
- A live external evaluation attempt of preserved checkpoints was stopped because it was competing with the learned-verifier full run and had not written metrics yet. Let the queued post-training checkpoint evaluation run after `train_gspo_chanka_unsloth.py` exits, when it will not compete with the full contender.

## Caveats

- These are proxy implementations, not exact reproductions. We do not have a Chanka-capable xCOMET model, and the severity proxy should not be described as xCOMET. It is only a cheap ablation/guardrail; serious reward-model work should prioritize the learned Chanka verifier.
- Do not use the xCOMET-inspired proxy as the optimization target. It is useful only for triage and transparent failure accounting. Since a real Chanka-capable xCOMET judge is unavailable, the DeepSeek-style trained verifier plus external lexical sanity metrics is the more defensible reward direction.
- Treat any xCOMET-like proxy as a cheap filter, not as the scoreboard. If we will not have real xCOMET for Chanka, the higher-upside path is to train stronger verifier models from real model outputs, pairwise/listwise preference labels, and hard negative categories: chat artifacts, source copying, Spanish leakage, wrong-language outputs, unsupported additions, semantic drift, and fluent unrelated Chanka. The verifier should be audited against held-out references and qualitative examples; if it disagrees with clear human/clean-reference judgments, retrain the verifier rather than chasing its score.
- BLEU/chrF alone are not sufficient. The previous GSPO final adapter showed source-copy risk, so qualitative samples and copy/leakage metrics must be reviewed.
- The VibeThinker profile is slower because it defaults to eight generations and therefore needs `TRAIN_BATCH_SIZE=8` / `EVAL_BATCH_SIZE=8` unless `NUM_GENERATIONS` is lowered.

## Artifact Guard Follow-Up

Update as of 2026-05-21 12:26 UTC:

- The saved/reloaded full learned-verifier-on-Vibe checkpoints failed artifact-aware evaluation: chrF++ around `20.2`, BLEU around `0.44`, TER around `676`, and chat artifact penalty around `58`. These checkpoints should not be used as final contenders even though their in-memory training metrics looked much better.
- The artifact-penalized `learned_verifier_vibe_2511` 4-generation canary finished at `outputs/gspo_paper_profiles/2511_artifact_guard_learned_verifier_vibe_4gen_canary_20260521-115915b`. Its in-memory final metrics were chrF++ `38.19053`, BLEU `7.22303`, token F1 `19.49405`, source-copy ratio `4.1667%`, exact-source-copy `1.5625%`, Spanish leakage `0.390625%`, chat artifact penalty `0.0`, TER `97.10983`, and trainer eval reward `0.24584`.
- Do not accept the 4-generation canary on in-memory metrics alone. The required next check is saved-adapter reload evaluation with `scripts/evaluate_gspo_checkpoint.py --batch-size 1`, first raw and then with `--strip-chat-artifacts`, so we can distinguish a genuinely cleaner adapter from a serving-time salvage artifact.
- The 8-generation artifact-penalized canary started immediately after the 4-generation run at `outputs/gspo_paper_profiles/2511_artifact_guard_learned_verifier_vibe_8gen_canary_20260521-115915b`.

Update as of 2026-05-21 13:12 UTC:

- The apparent reload artifact collapse was caused primarily by EOS mismatch, not by the 4-generation artifact-guard adapter itself. Reloaded Qwen3.5 adapters had tokenizer/processor EOS `<|im_end|>` id `248046`, but model generation config EOS id `248044`; generation did not stop at `<|im_end|>` unless `eos_token_id=tokenizer.eos_token_id` was passed explicitly.
- `scripts/evaluate_gspo_checkpoint.py` and `scripts/train_gspo_chanka_unsloth.py` now align `model.generation_config.eos_token_id` and pass `eos_token_id=tokenizer.eos_token_id` in direct generation calls, including learned-verifier generation.
- After the EOS fix, raw saved-adapter reload for the 4-generation artifact-guard canary with `--batch-size 1` produced chrF++ `37.99378`, BLEU `8.22530`, token F1 `22.55735`, source-copy ratio `2.9114%`, exact-source-copy `0.6329%`, Spanish leakage `0.7911%`, chat artifact penalty `0.0`, and TER `92.62673` over 158 eval rows. This exactly matched the previous trim/salvage metrics, so trimming is no longer required for this adapter.
- Batched reload with `--batch-size 8` is clean but not identical: chrF++ `37.58967`, BLEU `5.69811`, token F1 `22.45106`, chat artifact penalty `0.0`, TER `93.54839`. Use batch-size 1 as the canonical comparison for checkpoint selection, and batch-size 8 only for faster triage.
- The 8-generation artifact-guard canary was stopped early after checkpoint 16 because its reward stayed below 4-generation: checkpoint 8 reward `0.23739`, checkpoint 16 reward `0.23563`. The 4-generation artifact-guard adapter is the current stronger contender.

Update as of 2026-05-21 13:55 UTC:

- Re-ranking saved contenders after the EOS fix changed the decision point. `outputs/gspo_checkpoint_evals/20260521-eosfix-ranking/summary.md` ranks `vibethinker_phase2_checkpoint_896` above the earlier learned-verifier-on-Vibe112 checkpoints, and a new Vibe896-seeded learned-verifier+Vibe canary above both.
- Canonical batch-size-1 evaluation for `outputs/gspo_selected_checkpoints/vibethinker_phase2_checkpoint_896`: selection `25.6181`, chrF++ `40.2643`, BLEU `5.8940`, token F1 `25.8484`, source-copy `2.5105%`, exact-copy `0.6329%`, leakage `0.7911%`, artifact `0.0%`, TER `88.7097`.
- Canary `outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146` starts from `vibethinker_phase2_checkpoint_896` and uses `learned_verifier_vibe_2511`. Its saved-adapter batch-size-1 evaluation is now rank 1: selection `26.3808`, chrF++ `40.9703`, BLEU `8.1555`, token F1 `26.6169`, source-copy `2.8270%`, exact-copy `0.6329%`, leakage `0.6329%`, artifact `0.0%`, TER `88.7097`.
- Because that canary beat the previous best saved adapter, a full serious contender was launched at `outputs/gspo_full_contenders/learned_verifier_vibe_on_vibe896_full_20260521-135419`, log `outputs/learned_verifier_vibe_on_vibe896_full_20260521-135419.log`. Configuration: seed adapter `outputs/gspo_selected_checkpoints/vibethinker_phase2_checkpoint_896`, reward `learned_verifier_vibe_2511`, verifier `outputs/chanka_translation_verifier_hard_r128/checkpoint-1368`, 897 train rows, 158 validation rows, `MAX_STEPS=224`, `EVAL_STEPS=56`, `SAVE_STEPS=56`, LR `7e-7`, warmup ratio `0.01`, `NUM_GENERATIONS=4`, train/eval batch `4`, gradient accumulation `2`.
- Update as of 2026-05-21 14:55 UTC: the full serious contender reached `checkpoint-168` and it was preserved at `outputs/gspo_selected_checkpoints/learned_verifier_vibe_on_vibe896_checkpoint_168`. Trainer eval reward was `0.27686325638684667`, eval loss `0.01147084217518568`; checkpoint `112` still has the stronger trainer reward at `0.2869656496221506`. External batch-size-1 checkpoint evaluation remains the selection authority.
- Update as of 2026-05-21 15:15 UTC: the full serious contender finished and `checkpoint-224` was preserved at `outputs/gspo_selected_checkpoints/learned_verifier_vibe_on_vibe896_checkpoint_224`. In-memory final metrics were chrF++ `40.12930518900551`, BLEU `6.0075693712342275`, token F1 `25.644966847498495`, source-copy ratio `2.668776371308017%`, exact-copy `0.6329113924050633%`, Spanish leakage `0.7911392405063291%`, artifact `0.0%`, TER `89.86175115207374`, and trainer eval reward `0.2858995793481605`. The external batch-size-1 checkpoint evaluation queue `outputs/gspo_checkpoint_evals/20260521-vibe896-full-eval` is running and remains the selection authority.
- External batch-size-1 eval `outputs/gspo_checkpoint_evals/20260521-vibe896-full-eval` completed. The full run was clean but did not improve on the shorter canary. Ranking: `learned_verifier_vibe_on_vibe896_canary` selection `26.3808`, chrF++ `40.9703`, BLEU `8.1555`, token F1 `26.6169`; `vibethinker_phase2_checkpoint_896` selection `25.6181`; full-run `checkpoint_56` selection `25.3947`; full-run `checkpoint_224` and `final` selection `25.3835`; full-run `checkpoint_168` selection `25.2477`; full-run `checkpoint_112` selection `25.2292`. This means the next verifier-v2 GSPO canary should seed from the earlier canary, not from the longer full run.
- The current learned verifier is not yet the full stronger DeepSeek-style loop. It is verifier v1 trained mostly from synthetic corruptions plus hard negatives. Verifier v2 is queued via `experiments/gspo/queue_verifier_v2_from_real_candidates.sh`: after the full Vibe896 run and its external checkpoint evaluations finish, it trains a new verifier from real model prediction JSONLs, then runs a 24-step `learned_verifier_vibe_2511` canary from the best externally ranked adapter. Remote stamp/log: `20260521-verifier-v2-real`, `outputs/queue_verifier_v2_from_real_candidates_20260521.log`.
- Commit `5679ace` hardened the verifier-v2 queue so a final-save crash does not discard a useful verifier. If `final_verifier_lora` is missing, it selects the checkpoint with the best logged `eval_loss` and uses that adapter for the canary.
- Verifier-v2 real-candidate training has started at `outputs/chanka_translation_verifier_real_candidates_v2_20260521-verifier-v2-real`. The candidate JSONLs deduped down to 305 real model-output verifier examples, giving 8,979 total verifier examples and a 7,633 train / 1,346 validation split. Schedule: 4 epochs, 1,912 optimizer steps, eval every 59 steps, LoRA r128/a256, LR `2e-5`, batch 4, gradient accumulation 4. This is the first actual real-rollout verifier pass; the canary after it will be the evidence for whether the DeepSeek-style loop improves the translator.
- Verifier-v2 is improving early: checkpoint `59` eval loss `0.04840836673974991`, checkpoint `118` `0.0312885046005249`, checkpoint `177` `0.020037654787302017`, checkpoint `236` `0.021803665906190872`, checkpoint `295` `0.017204049974679947`, checkpoint `354` `0.015418198890984058`, checkpoint `413` `0.01523187942802906`, checkpoint `472` `0.015611322596669197`. Current best is checkpoint `413`; best-so-far checkpoints are preserved under `outputs/verifier_selected_checkpoints/` in case later epochs overfit or final save fails.
- Later verifier-v2 checkpoints continued improving: checkpoint `531` eval loss `0.013897795230150223`, checkpoint `590` `0.014394257217645645`, checkpoint `649` `0.013841227628290653`, and checkpoint `708` `0.01347315777093172`. Commit `89fa4e1` adds `experiments/gspo/watch_verifier_best_checkpoint.sh` and expands queue fallback to search `outputs/verifier_selected_checkpoints/`. The watcher is running on the remote and maintains a stable best symlink at `outputs/verifier_selected_checkpoints/chanka_translation_verifier_real_candidates_v2_20260521-verifier-v2-real_best`.
- If verifier-v2 does not improve the current canary, the next verifier-data escalation should be sampled train/all candidate mining rather than another eval-only pass. Commit `382e88a` adds this to `scripts/evaluate_gspo_checkpoint.py` with `--split train|eval|all`, `--max-train-samples`, `--do-sample`, and `--num-return-sequences`. Generate multiple sampled translations from the current best adapter, then feed that JSONL into `scripts/train_verifier_chanka_unsloth.py --candidate-jsonl` for verifier-v3.

Update as of 2026-05-21 16:35 UTC:

- The DeepSeek-style real-rollout verifier pass has now been tried end to end. Verifier-v2 was trained from the clean/corruption verifier pool plus real model prediction JSONLs. After dedupe, only 305 real model-output examples were added, for 8,979 verifier examples total.
- Verifier-v2 was stopped after overfit signs. Best preserved checkpoint was `outputs/verifier_selected_checkpoints/chanka_translation_verifier_real_candidates_v2_20260521-verifier-v2-real_checkpoint-944` with eval loss `0.012264552526175976`; later checkpoints worsened to `0.012620210647583008` at checkpoint `1003` and `0.013034608215093613` at checkpoint `1062`.
- The verifier-v2 GSPO canary ran at `outputs/gspo_paper_profiles/2511_verifier_v2_real_candidates_on_best_20260521-verifier-v2-real`, seeded from the current best `learned_verifier_vibe_on_vibe896_4gen_canary` adapter and using `learned_verifier_vibe_2511` with the verifier-v2 best symlink.
- Canonical batch-size-1 eval `outputs/gspo_checkpoint_evals/20260521-verifier-v2-real-verifier-v2-canary` did not beat the previous best. Ranking: checkpoint `24`/final selection `25.4932`, chrF++ `40.1668`, BLEU `5.9356`, token F1 `25.6148`, source-copy `2.5105%`, exact-copy `0.6329%`, leakage `0.7911%`, artifact `0.0%`, TER `88.9401`; checkpoint `16` selection `25.1987`; checkpoint `8` selection `25.1591`.
- Current best remains `learned_verifier_vibe_on_vibe896_4gen_canary` from `outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora`, with selection `26.3808`, chrF++ `40.9703`, BLEU `8.1555`, token F1 `26.6169`, source-copy `2.8270%`, exact-copy `0.6329%`, leakage `0.6329%`, artifact `0.0%`, TER `88.7097`.
- Interpretation: the verifier idea is not disproven, but this v2 data mix was too small on real candidates to move the policy. The next serious verifier attempt should mine sampled train/all candidates from the current best adapter (`--split train` or `--split all`, `--do-sample`, multiple returns) and train a verifier-v3 from a much larger real-output pool. Repeating eval-only candidate mining is unlikely to help.

Update as of 2026-05-21 17:27 UTC:

- Added `experiments/gspo/queue_verifier_v3_sampled_candidates.sh` for the next verifier escalation. It mines sampled real candidates from the current best adapter, trains a stronger verifier on the larger candidate pool, runs a 24-step canary, and batch-size-1 evaluates checkpoint `8`, `16`, `24`, and final.
- Remote verifier-v3 queue launched with `STAMP=20260521-verifier-v3-sampled`, log `outputs/queue_verifier_v3_sampled_candidates_20260521.log`.
- Defaults: current best adapter as seed; candidate mining over `--split all`; two sampling settings (`temperature=0.75, top_p=0.90` and `temperature=0.95, top_p=0.95`); `4` return sequences per source; verifier max candidate examples `12000`; verifier max steps `1200`; GSPO canary max steps `24`.
- Expected artifacts: candidates under `outputs/verifier_candidate_mining/20260521-verifier-v3-sampled`, verifier under `outputs/chanka_translation_verifier_sampled_candidates_v3_20260521-verifier-v3-sampled`, canary under `outputs/gspo_paper_profiles/2511_verifier_v3_sampled_candidates_on_best_20260521-verifier-v3-sampled`, eval summary under `outputs/gspo_checkpoint_evals/20260521-verifier-v3-sampled-verifier-v3-canary`.

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
