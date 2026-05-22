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
