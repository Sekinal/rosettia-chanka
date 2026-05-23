# Full SFT With Unsloth

## 2026-05-22 Full-Finetuning Support

Purpose: test whether improving the base model itself via Unsloth full finetuning can beat continued LoRA SFT/GSPO.

Code added:

- `scripts/train_sft_unsloth.py` now supports `--training-mode lora|full`.
- `scripts/train_jsonl_sft_unsloth.py` also supports `--training-mode lora|full` for pseudo-label/self-training rows.
- Full mode loads Unsloth with `full_finetuning=True`, does not call `get_peft_model`, uses lower default learning rates, smaller per-device batches, and saves to `final_full_model`.
- `scripts/export_unsloth_merged_model.py` exports an existing Unsloth LoRA adapter to a merged full model with `save_pretrained_merged(..., save_method="merged_16bit")`.

Official-docs check:

- Unsloth full finetuning is enabled through `full_finetuning=True`.
- Only one training mode should be active at a time, so the scripts now treat full finetuning and LoRA as mutually exclusive paths.

## Direct Full SFT From Raw Qwen3.5-2B

Smoke:

- Output: `outputs/full_sft_canaries/20260522-qwen35-2b-full-sft-smoke`
- Setup: raw `unsloth/Qwen3.5-2B`, clean Chanka, 32 train rows, 8 eval rows, 2 steps, LR `1e-6`, effective batch `8`.
- Result: completed and saved `final_full_model`; final eval loss `3.1466`.
- Tiny 16-row generation eval: chrF++ `8.9113`, BLEU `1.6603`, token F1 `0.0`, TER `135.8974`.

16-step canary:

- Output: `outputs/full_sft_canaries/20260522-qwen35-2b-full-sft-16step`
- Setup: raw `unsloth/Qwen3.5-2B`, full clean Chanka train split, 128 eval rows, 16 steps, LR `1e-6`, effective batch `8`.
- Eval loss: step 8 `3.040`, step 16/final `3.0149`.
- 128-row generation eval: chrF++ `10.1459`, BLEU `1.2714`, token F1 `2.3668`, TER `139.8281`.

Decision:

- Direct full SFT from raw Qwen is technically stable on the L40S but not useful at this scale. It learns Chanka-looking text slowly and remains far behind the LoRA/GSPO checkpoint.

## Merge Current Best LoRA, Then Full SFT

Merged model:

- Adapter source: `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8`
- Merged model: `outputs/merged_full_models/20260522-current-best-checkpoint8-merged16`
- Export command used `save_method=merged_16bit`.
- 128-row greedy terminology eval before full SFT: chrF++ `41.7751`, BLEU `7.3813`, token F1 `26.7573`, TER `89.3983`.

Merged-state full SFT canary:

- Output: `outputs/full_sft_canaries/20260522-current-best-merged-full-sft-16step`
- Setup: model id `outputs/merged_full_models/20260522-current-best-checkpoint8-merged16`, full clean Chanka train split, 128 eval rows, 16 steps, LR `1e-6`, effective batch `8`.
- Eval loss: step 8 `1.029`, step 16/final `1.0286`.
- 128-row greedy terminology eval after full SFT: chrF++ `41.7152`, BLEU `7.3915`, token F1 `26.7486`, TER `89.3983`.

Decision:

- Merge-then-full-SFT is the correct full-finetuning path; the loss scale is healthy and no instability appeared.
- The 16-step canary did not improve generation metrics. It was almost neutral, with a tiny BLEU increase but lower chrF++ and token F1.
- Next full-SFT experiment should not be raw-base full tuning. Try merged current best with either a slightly higher LR (`2e-6`), longer horizon with early evals, or JSONL terminology/MBR data via `train_jsonl_sft_unsloth.py --training-mode full`.
- Keep all merged/full model artifacts out of git.

## 2026-05-22 Merged Full SFT On JSONL Terminology/MBR Mix

Purpose: test the most relevant full-finetuning setup: start from the merged current-best LoRA model, then continue full SFT on the same terminology-conditioned MBR/clean JSONL mixture that produced the current best LoRA checkpoint.

Setup:

- Model id: `outputs/merged_full_models/20260522-current-best-checkpoint8-merged16`
- JSONL: `outputs/mbr_self_training_data/20260522-k16-full-newbest-noterm-t065-p090/mixed_clean512_confident_margin000_target.jsonl`
- Output: `outputs/full_sft_canaries/20260522-current-best-merged-jsonl-term-full-sft-lr2e-6-32step`
- Terminology prompt: `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`, top-k `1`
- Rows after filtering: 1,332 total, 1,133 train, 199 validation
- Terminology-matched rows: 143 train, 25 validation
- LR: `2e-6`
- Max steps: `32`
- Batch: per-device `1`, gradient accumulation `8`
- Validation/save: every `8` optimizer steps

Trainer eval loss:

| Checkpoint | eval loss |
| --- | ---: |
| checkpoint-8 | 0.6429 |
| checkpoint-16 | 0.6417 |
| checkpoint-24 | 0.6409 |
| checkpoint-32/final | 0.6417 |

Held-out clean Chanka terminology-prompt eval:

| Checkpoint | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-16 | 26.3055 | 41.0008 | 6.7101 | 26.8596 | 2.6688 | 0.4747 | 88.0184 |
| checkpoint-24 | 26.4243 | 41.2338 | 6.8438 | 26.9119 | 2.6688 | 0.4747 | 88.4793 |
| checkpoint-32/final | 26.5312 | 41.2908 | 6.7858 | 27.2479 | 2.6688 | 0.3165 | 88.2488 |

K8 candidate-pool check from checkpoint-32:

- Candidate output: `outputs/full_sft_canaries/20260522-current-best-merged-jsonl-term-full-sft-lr2e-6-32step-k8-candidates/k8_predictions.jsonl`
- First candidate: selection `25.0974`, chrF++ `39.0863`, BLEU `10.2628`, token F1 `24.0793`, TER `90.3226`
- MBR: selection `26.4580`, chrF++ `41.8282`, BLEU `7.3718`, token F1 `25.4410`, TER `91.9355`
- Oracle: selection `35.7023`, chrF++ `50.4212`, BLEU `16.3823`, token F1 `38.7261`, TER `75.1152`

Decision:

- This is a negative result for deployment. It does not beat the current LoRA checkpoint (`checkpoint-8` from `20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps`) and is far below the conservative K32 score ensemble.
- It also is not useful as a candidate diversity generator: its K8 oracle is below the previous conservative and mixed candidate pools.
- Full SFT is viable mechanically, but current evidence says it should be reserved for larger/model-scale experiments or a better data mix. Do not scale this exact JSONL full-SFT recipe.

## Current Full-SFT Recommendation

Full fine-tuning with Unsloth is worth keeping in the toolkit, but not as the next blind experiment.

Current evidence:

- Raw-base full SFT is too slow and weak at our data scale.
- Merged-current-best full SFT is stable but neutral on clean Chanka.
- Merged-current-best JSONL full SFT is negative.
- A Qwen3.5 4B JSONL-only LoRA canary is also negative after fixing no-thinking prompt formatting, so full-tuning 4B on that same data mix would likely spend more GPU without solving the data/curriculum issue.
- Qwen3.5 4B broad -> clean Chanka LoRA is the first 4B SFT result worth keeping: best checkpoint `outputs/qwen35_4b_curriculum/20260522-broad512-chanka-r64-a128-s128-256steps/chanka/checkpoint-224` scored selection `30.3580`, chrF++ `43.2858`, BLEU `16.3507`, token F1 `30.1583`.
- Merge -> full SFT from that 4B LoRA checkpoint is feasible and slightly improves the external triage score: best checkpoint `outputs/full_sft_canaries/20260522-qwen35-4b-merged-broad512-chanka224-full-chanka-lr1e-6-64steps/chanka/checkpoint-48` scored selection `30.4393`, chrF++ `43.3102`, BLEU `16.2178`, token F1 `30.6209`.
- The same 4B full-SFT checkpoint became highly useful as a candidate generator. Current K32 candidates plus 4B-full K16 candidates, with a matching-distribution score ensemble, reached selection `35.5481`, chrF++ `48.1624`, BLEU `24.0635`, token F1 `35.9579`, TER `70.5069`. This is the best held-out profile so far.

Next best SFT path:

1. Train larger Qwen variants through the full broad Quechua -> clean Chanka curriculum with no-thinking chat templates.
2. Evaluate generation quality at checkpoints, not just trainer eval loss.
3. Only if a larger LoRA checkpoint approaches or beats the current 2B adapter, merge it and run a short full-SFT continuation from that strong state.
4. Evaluate larger full-SFT checkpoints both greedily and as candidate generators. The 4B result shows the reranked candidate pool can improve even when the single-model greedy metrics only move slightly.

Queued focused 4B FFT sweep:

- `experiments/sft/queue_qwen35_4b_full_sft_lr_sweep.sh` reuses the proven 4B broad -> Chanka LoRA checkpoint `outputs/qwen35_4b_curriculum/20260522-broad512-chanka-r64-a128-s128-256steps/chanka/checkpoint-224`.
- It merges the adapter to `outputs/merged_full_models/20260522-qwen35-4b-broad512-chanka224-merged16` if needed, then runs terminology-conditioned full-SFT continuations for `LR_LIST="5e-7 1e-6 2e-6"` by default, `MAX_STEPS=96`, eval/save every `12` steps.
- It waits behind the 9B merge -> full-SFT chain by default so it does not steal the L40S from the current scale experiment. Set `WAIT_FOR_SUCCESS="" WAIT_FOR_FAILURE=""` to run it immediately.
- This is a targeted sweep around the only full-finetuning recipe that has shown signal. It should replace ad hoc raw-base FFT attempts.

Details: `docs/findings/qwen35_4b_curriculum_sft.md`.

## Queued 9B Merge -> Full SFT

Added `experiments/sft/queue_qwen35_9b_full_sft_after_rerank.sh`.

Purpose: test whether the same merge -> full-SFT refinement that slightly helped the 4B model is useful after the 9B broad -> clean Chanka LoRA curriculum.

Design:

- Wait for the 9B checkpoint eval summary and the few-shot 9B rerank summary so this does not compete with active GPU candidate-generation jobs.
- Select the best 9B Chanka LoRA checkpoint by held-out external `selection_score`.
- Merge the selected LoRA to a 16-bit full model using `scripts/export_unsloth_merged_model.py`.
- Run a tiny terminology-conditioned full-SFT memory smoke before spending GPU on the real canary. If 9B full fine-tuning does not fit on the L40S, the script writes `full_sft_smoke_failed.txt` and exits cleanly.
- If the smoke passes, run a 32-step terminology-conditioned Chanka full-SFT canary at LR `5e-7`, with validation/checkpointing every 8 steps.
- Evaluate every full-SFT checkpoint and `final_full_model` with terminology top-1 prompting.

Caveat: 9B FFT may be memory-bound on the current L40S despite Unsloth support. If it fails here, keep the recipe for the larger RTX PRO 6000 and do not treat the failure as evidence against full fine-tuning itself.

Update: the queued 9B full-SFT smoke and real canary now train with `--terminology-file clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet --terminology-top-k 1`, not just evaluate that way. This removes the prompt mismatch from the original queued plan.

9B checkpoint status:

- The broad -> Chanka 9B LoRA curriculum completed at `outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64`.
- Best standalone checkpoint is `checkpoint-192` / `final_lora`: selection `28.3241`, chrF++ `41.6747`, BLEU `12.3257`, token F1 `29.2655`, TER `84.7926`.
- This is weaker than the 4B full-SFT checkpoint as a standalone model, so the immediate value of 9B is candidate diversity and possible merge -> full-SFT refinement, not direct deployment yet.
- `scripts/write_nested_metrics_summary.py` was added because the 9B eval summary initially lacked computed `selection_score` fields. Downstream queues now compute fallback selection scores from `metrics.json`.
- The no-few-shot 9B candidate-rerank queue was restarted on 2026-05-23 and selected `checkpoint-192`. The 9B full-SFT queue remains intentionally behind the no-few-shot and few-shot candidate-rerank passes.

## Terminology-Conditioned SFT Prompt Matching

Added terminology-conditioned prompts to `scripts/train_sft_unsloth.py`.

Why: the best inference profiles use terminology top-1 prompting from `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`, but most plain Chanka SFT runs trained only on the base translation prompt. That creates a train/eval prompt mismatch exactly where we are extracting gains from glossary hints.

New controls:

- `--terminology-file`
- `--terminology-top-k`
- `--terminology-min-source-chars`

The implementation mirrors the existing eval/JSONL logic: longest source-term matches first, dedupe by target term, and include the glossary block only when a row has a relevant match.

Queued follow-up: `experiments/sft/queue_qwen35_9b_terminology_chanka_sft.sh` waits for the active 9B rerank/full-SFT chain, then reruns the clean Chanka continuation from the 9B broad checkpoint with terminology top-1 prompts and evaluates all checkpoints with terminology top-1 prompting.

## 2026-05-23 Focus Back On Base-Model SFT

The base-model improvement path is now prioritized over more reranking-only plumbing.

Actions:

- Paused the downstream waiting queue wrappers on the L40S host so they do not automatically take the GPU after the 9B selector-recovery marker appears.
- Kept the 9B selector recovery running because it reuses already-generated candidate pools and records a clean result.
- Launched the focused 4B merged-model full-SFT LR sweep immediately by pointing both wait markers at `outputs/logs/manual_start_marker`.

Active run:

- Script: `experiments/sft/queue_qwen35_4b_full_sft_lr_sweep.sh`
- Log: `outputs/logs/qwen35_4b_full_sft_lr_sweep_focus_20260523.log`
- Output root: `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-sweep-focus`
- Starting model: `outputs/merged_full_models/20260522-qwen35-4b-broad512-chanka224-merged16`
- Training mode: Unsloth full fine-tuning, bf16, `4,539,265,536` trainable parameters.
- Data: clean Chanka stage, 897 train rows, 158 validation rows.
- Prompting: terminology-conditioned top-1 training prompts from `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`.
- LR sweep: `5e-7 1e-6 2e-6`, `96` optimizer steps each, eval/save every `12` steps.

Early signal:

- The `5e-7` run fits on the L40S and started normally.
- First checkpoint eval at step 12: `eval_loss = 0.9906`.
- The original sweep crashed at checkpoint 48 because `/root` filled while saving full-model optimizer states, not because the training step failed. Deleted only full-SFT `optimizer.pt` files to recover 131 GB of free space.
- `scripts/train_sft_unsloth.py` and `scripts/train_jsonl_sft_unsloth.py` now default `--save-only-model` to true for `--training-mode full`, avoiding 12 GB optimizer-state files per 4B FFT checkpoint.

Recovered `5e-7` held-out generation metrics:

| Checkpoint | Selection | chrF++ | BLEU | token F1 | TER |
| --- | ---: | ---: | ---: | ---: | ---: |
| checkpoint-12 | 30.1047 | 42.7808 | 16.4394 | 29.8419 | 84.5622 |
| checkpoint-24 | 30.5455 | 43.4370 | 15.3563 | 31.0685 | 84.7926 |
| checkpoint-36 | 30.1001 | 42.9743 | 15.0028 | 30.5275 | 85.7143 |
| checkpoint-48 | 29.6333 | 42.5131 | 14.3296 | 30.3165 | 88.4793 |

Decision:

- `checkpoint-24` is the best 4B full-SFT checkpoint so far by selection, chrF++, and token F1, beating the previous 4B full-SFT best (`30.4393` selection, chrF++ `43.3102`, token F1 `30.6209`) but losing BLEU (`15.3563` vs `16.2178`).
- The run peaks early; longer `5e-7` full SFT is not useful.
- Follow-up run `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups` is testing `1e-6` and `2e-6` for only `48` steps with model-only checkpoint saves.

Caveat:

- Loading the merged local model emitted a Transformers tokenizer warning suggesting `fix_mistral_regex=True`. Do not overinterpret one canary if generation metrics look strange; the next script hardening pass should test passing that tokenizer fix flag through Unsloth when loading merged full models.
- Directly passing `fix_mistral_regex=True` to this Transformers/Unsloth tokenizer path currently crashes on the merged tokenizer, so this is not safe to patch blindly.

## 2026-05-23 Follow-Up 4B Full-SFT LR Sweep

The focused follow-up found a real base-model improvement.

Setup:

- Output root: `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups`
- Starting model: `outputs/merged_full_models/20260522-qwen35-4b-broad512-chanka224-merged16`
- Data: clean Chanka SFT with terminology top-1 prompts.
- LRs: `1e-6` and `2e-6`, `48` optimizer steps each, eval/save every `12` steps.
- Important caveat: this active run started before `--save-total-limit 0` was added, so trainer checkpoint pruning deleted some intermediate checkpoints, including `1e-6/checkpoint-24`.

Held-out generation metrics:

| LR/checkpoint | Selection | chrF++ | BLEU | token F1 | TER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1e-6/checkpoint-12` | 30.1318 | 42.8587 | 15.1295 | 30.6646 | 85.0230 |
| `1e-6/checkpoint-36` | 30.0566 | 42.9153 | 15.4865 | 30.4838 | 87.5576 |
| `1e-6/checkpoint-48` | 29.8359 | 42.7666 | 15.5218 | 30.0332 | 88.4793 |
| `2e-6/checkpoint-12` | 29.9115 | 42.9988 | 14.6143 | 30.3467 | 88.2488 |
| `2e-6/checkpoint-36` | 31.1999 | 44.4295 | 18.0014 | 29.9850 | 81.5668 |
| `2e-6/checkpoint-48` | 30.6535 | 43.8315 | 15.9186 | 30.4944 | 84.1014 |
| `2e-6/final_full_model` | 30.6535 | 43.8315 | 15.9186 | 30.4944 | 84.1014 |

Decision:

- `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36` is now the best standalone 4B full-SFT checkpoint by selection, chrF++, BLEU, and TER.
- This improves over the previous 4B full-SFT best (`30.4393` selection, chrF++ `43.3102`, BLEU `16.2178`, token F1 `30.6209`, TER `85.2535`) and over the recovered `5e-7/checkpoint-24` on selection/chrF++/BLEU/TER, though `5e-7/checkpoint-24` still had higher token F1.
- Full fine-tuning is useful, but only as a short late-stage refinement after a strong broad Quechua -> clean Chanka LoRA curriculum is merged. Direct raw-base full SFT and JSONL-only full SFT remain negative.
- A candidate-pool harvest is running from `2e-6/checkpoint-36`: log `outputs/logs/qwen35_4b_fft2e6ckpt36_candidate_rerank_20260523.log`, output root `outputs/qwen35_4b_full_sft_candidate_rerank/20260523-qwen35-4b-fft2e6ckpt36-candidate-rerank`.
- Eval-side K16 sampling from this checkpoint has real selector headroom: first candidate selection `29.1882`, chrF++ `42.5822`, BLEU `16.4238`, TER `81.7972`; oracle selection `41.3831`, chrF++ `54.6467`, BLEU `25.0224`, TER `63.5945`. This is below the current multi-model oracle ceiling, but strong enough to justify the active merged-pool listwise rerank.

## 2026-05-23 9B Full-SFT Optimizer Canary

Purpose: test whether the completed Qwen3.5 9B broad -> clean Chanka LoRA checkpoint can be merged and improved through full fine-tuning on the L40S host.

Code changes:

- `scripts/train_sft_unsloth.py` and `scripts/train_jsonl_sft_unsloth.py` now expose `--optim`.
- Default remains `adamw_8bit`.
- For large full-SFT memory canaries, use `--optim paged_adamw_8bit`.
- Added queue scripts:
  - `experiments/sft/queue_qwen35_9b_full_sft_direct_smoke.sh`
  - `experiments/sft/queue_qwen35_9b_full_sft_paged_canary.sh`

Direct smoke:

- Starting adapter: `outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64/chanka/chanka/checkpoint-192`
- Merged model: `outputs/merged_full_models/20260523-qwen35-9b-direct-full-sft-smoke-merged16`
- Regular `adamw_8bit` full SFT OOMed on the first optimizer step on the 44 GB L40S.
- `paged_adamw_8bit` completed the 2-step memory smoke, proving 9B full SFT is mechanically possible on this host only with the paged optimizer.

Paged canary:

- Output root: `outputs/qwen35_9b_full_sft/20260523-qwen35-9b-paged-full-sft-canary`
- Training: clean Chanka SFT, terminology top-1 prompts, LR `5e-7`, `8` optimizer steps, eval/save every `4` steps, `paged_adamw_8bit`.
- Trainer eval loss: step 4 about `1.008`; final about `1.00676`.

Held-out terminology-prompt generation metrics:

| Checkpoint | Selection | chrF++ | BLEU | token F1 | TER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-4` | 28.0082 | 41.4058 | 11.6963 | 29.0952 | 85.9447 |
| `final_full_model` | 28.4456 | 41.9470 | 13.0248 | 28.8405 | 81.3364 |

Decision:

- This is a negative quality result. It is slightly better than the 9B LoRA checkpoint on BLEU/TER, but it remains below the best standalone 4B full-SFT checkpoint (`31.1999` selection, chrF++ `44.4295`, BLEU `18.0014`) and far below the multi-model listwise reranked system.
- Keep the paged optimizer path for future 9B/30B memory probes, but do not scale this exact 9B full-SFT recipe.
- The remote full-model artifacts were deleted after metrics were captured to recover disk; only `checkpoint_eval` metrics/predictions and the log were kept.

Code hygiene:

- `scripts/train_sft_unsloth.py` and `scripts/train_jsonl_sft_unsloth.py` now expose `--save-total-limit`; use `--save-total-limit 0` for full-SFT sweeps when model-only checkpointing is enabled.
- `experiments/sft/queue_qwen35_4b_full_sft_lr_sweep.sh` defaults `SAVE_TOTAL_LIMIT=0` so future full-SFT sweeps do not silently prune early checkpoints before external generation eval.
- `experiments/sft/queue_qwen35_4b_full_sft_candidate_rerank.sh` can take `FULL_SFT_CHECKPOINT=...` to harvest candidates from a known best checkpoint directly instead of selecting from one sweep summary.
- `scripts/write_nested_metrics_summary.py` now handles both nested `*/metrics.json` and flat `*_metrics.json` layouts. The first GSPO retry summary was empty only because this queue writes flat metric files.

## 2026-05-23 Active 4B Full-SFT Refinement

Launched a longer low-LR full fine-tune from the current best standalone 4B full-SFT checkpoint.

Setup:

- Remote host: `root@216.81.248.197 -p 20299`
- PID at launch: `254147`
- Log: `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-refine-from-ckpt36-lr5e-7-72steps/run.log`
- Starting model: `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36`
- Output root: `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-refine-from-ckpt36-lr5e-7-72steps`
- Training: clean Chanka SFT with terminology top-1 prompts.
- Full fine-tuning, all `4,539,265,536` parameters trainable.
- Optimizer: `paged_adamw_8bit`
- LR: `5e-7`
- Steps: `72`
- Eval/save every `8` steps, `save_only_model=True`, keeping `4` checkpoints.

Reason:

- The current best standalone base is the `2e-6/checkpoint-36` 4B full-SFT model: selection `31.1999`, chrF++ `44.4295`, BLEU `18.0014`, token F1 `29.9850`, TER `81.5668`.
- This run tests whether a slower follow-up phase from that checkpoint can preserve BLEU while improving chrF++/TER or reduce overfit relative to the 48-step `2e-6` sweep.
- Keep validation-generation eval after training; trainer eval loss alone is not enough for model selection.

GSPO follow-up:

- `scripts/train_gspo_chanka_unsloth.py` now supports `--attach-lora`, `--lora-r`, `--lora-alpha`, and `--lora-dropout`. This is intended for RL from a merged/full checkpoint without full-parameter RL.
- `experiments/gspo/queue_qwen35_4b_full_sft_gspo_canary.sh` queues a small LoRA-on-full-checkpoint GSPO canary from `2e-6/checkpoint-36`, using the current learned-verifier-vibe reward profile and terminology top-1 prompts.
- First two launches failed before training because Unsloth GRPO requires both train and eval per-device batch sizes to be divisible by `num_generations`. The queue script now defaults train/eval batch size to `4` when `NUM_GENERATIONS=4`.
- Remote retry log: `outputs/logs/qwen35_4b_fft2e6ckpt36_lora_gspo_retry2_20260523.log`. It has passed the batching guard and is training from `2e-6/checkpoint-36`.
- Result: the canary completed, but it did not beat the full-SFT base. Trainer-side eval reward peaked at step 8 (`0.1487`) and fell by step 16 (`0.1349`); all trainer eval completions hit the 80-token cap with no terminations, so reward is not trustworthy on its own.
- External held-out eval with terminology top-1:

| Checkpoint | Selection | chrF++ | BLEU | token F1 | TER |
| --- | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-8` | 29.5960 | 43.0832 | 14.9256 | 28.9678 | 86.1751 |
| `checkpoint-16` | 30.9849 | 44.3754 | 18.1300 | 29.2240 | 81.7972 |
| `final_gspo_lora` | 30.9849 | 44.3754 | 18.1300 | 29.2240 | 81.7972 |

- Decision: LoRA-GSPO from the best 4B full-SFT checkpoint is mechanically viable but negative as run. It slightly improves BLEU over the full-SFT base (`18.13` vs `18.00`) but loses selection (`30.98` vs `31.20`), token F1 (`29.22` vs `29.99`), and TER (`81.80` vs `81.57`). Do not scale this exact reward/length recipe; fix termination/length reward first or move GSPO back to a stronger adapter/checkpoint with a better candidate selector target.

Candidate-rerank follow-up:

- K16 candidates generated from `2e-6/checkpoint-36` have strong oracle headroom but did not improve the deployable selector when added to the current K32+old-4B pool.
- Merged pool listwise text result: selection `38.3355`, chrF++ `51.6498`, BLEU `23.9115`, token F1 `39.6077`, TER `69.1244`.
- Merged pool oracle: selection `51.1497`, chrF++ `64.5949`, BLEU `37.1959`, token F1 `56.1489`, TER `50.0000`.
- Decision: keep `2e-6/checkpoint-36` as the best standalone 4B full-SFT base and as a useful diversity source, but do not replace the current deployable K32+old-4B listwise profile until selector training can harvest the extra oracle headroom.
