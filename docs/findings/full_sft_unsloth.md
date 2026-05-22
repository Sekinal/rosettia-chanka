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

Details: `docs/findings/qwen35_4b_curriculum_sft.md`.
