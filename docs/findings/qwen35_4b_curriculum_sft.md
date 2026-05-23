# Qwen3.5 4B Curriculum SFT

Date: 2026-05-22

Purpose: test whether the weak Qwen3.5 4B JSONL-only result was a model problem or a curriculum problem, and whether Unsloth full finetuning can improve a strong merged 4B checkpoint.

## Broad -> Clean Chanka LoRA

Broad stage:

- Model: `unsloth/Qwen3.5-4B`
- Output: `outputs/qwen35_4b_curriculum/20260522-broad-r64-a128-s256-512steps/broad`
- Data: broad Quechua/Spanish SFT stage from `scripts/train_sft_unsloth.py --stage broad`
- Rows: 169,877 loaded, 166,480 train, 3,397 validation
- LoRA: r64 / alpha128
- Sequence length: 256
- Max steps: 512
- Effective batch: 16
- LR: `5e-5`
- Validation/save: every 128 steps

Broad eval loss:

| Checkpoint | eval loss |
| --- | ---: |
| checkpoint-128 | 1.7978 |
| checkpoint-256 | 1.5536 |
| checkpoint-384 | 1.4371 |
| checkpoint-512 | 1.3870 |

The final post-save eval printed `nan`, so `checkpoint-512` is the reliable continuation point.

Clean Chanka continuation:

- Adapter start: `outputs/qwen35_4b_curriculum/20260522-broad-r64-a128-s256-512steps/broad/checkpoint-512`
- Output: `outputs/qwen35_4b_curriculum/20260522-broad512-chanka-r64-a128-s128-256steps/chanka`
- Rows: 1,055 loaded, 897 train, 158 validation
- LoRA: r64 / alpha128
- Sequence length: 128
- Max steps: 256
- Effective batch: 8
- LR: `2e-5`
- Validation/save: every 32 steps

Chanka eval loss:

| Checkpoint | eval loss |
| --- | ---: |
| checkpoint-64 | 1.209 |
| checkpoint-96 | 1.122 |
| checkpoint-128 | 1.085 |
| checkpoint-160 | 1.056 |
| checkpoint-224 | 1.012 |
| checkpoint-256/final | 1.020 |

External held-out eval with terminology top-1:

| Checkpoint | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-224 | 30.3580 | 43.2858 | 16.3507 | 30.1583 | 2.1624 | 0.0000 | 85.0230 |
| final_lora | 30.3580 | 43.2858 | 16.3507 | 30.1583 | 2.1624 | 0.0000 | 85.0230 |
| checkpoint-256 | 29.4458 | 42.5682 | 15.4622 | 28.9673 | 2.1624 | 0.0000 | 88.4793 |
| checkpoint-192 | 29.2875 | 42.4893 | 16.2026 | 27.1505 | 2.2152 | 0.0000 | 82.0276 |

Decision:

- The 4B model is not the problem. Raw 4B JSONL-only SFT was the wrong data order.
- Broad -> clean Chanka curriculum gives the best 4B result so far and a strong diversity candidate.
- It still trails the current K32 ensemble on chrF++ (`43.2858` vs `43.8064`), but beats it on BLEU (`16.3507` vs `13.2505`) and selection (`30.3580` vs `29.7584`).

## Merge -> Full SFT Refinement

Merged model:

- Adapter source: `outputs/qwen35_4b_curriculum/20260522-broad512-chanka-r64-a128-s128-256steps/chanka/checkpoint-224`
- Merged model: `outputs/merged_full_models/20260522-qwen35-4b-broad512-chanka224-merged16`
- Export: `scripts/export_unsloth_merged_model.py --save-method merged_16bit`

Full-SFT canary:

- Model id: `outputs/merged_full_models/20260522-qwen35-4b-broad512-chanka224-merged16`
- Output: `outputs/full_sft_canaries/20260522-qwen35-4b-merged-broad512-chanka224-full-chanka-lr1e-6-64steps/chanka`
- Training mode: `--training-mode full`
- Trainable parameters: 4,539,265,536 / 4,539,265,536
- Data: clean Chanka stage, 897 train / 158 validation
- Sequence length: 128
- Max steps: 64
- Effective batch: 8
- LR: `1e-6`
- Validation/save: every 16 steps
- Note: `save_total_limit=3` retained checkpoint 16, 48, 64, and `final_full_model`.

Trainer eval loss:

| Checkpoint | eval loss |
| --- | ---: |
| checkpoint-16 | 0.9938 |
| checkpoint-32 | 0.9946 |
| checkpoint-48 | 0.9960 |
| checkpoint-64/final | 0.9952 |

External held-out eval with terminology top-1:

| Checkpoint | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-48 | 30.4393 | 43.3102 | 16.2178 | 30.6209 | 2.1624 | 0.0000 | 85.2535 |
| checkpoint-64 | 30.0848 | 43.0061 | 15.2830 | 30.5873 | 2.1308 | 0.0000 | 88.0184 |
| final_full_model | 30.0848 | 43.0061 | 15.2830 | 30.5873 | 2.1308 | 0.0000 | 88.0184 |
| checkpoint-16 | 30.0356 | 43.0186 | 15.3739 | 30.2638 | 2.4789 | 0.0000 | 85.0230 |

Decision:

- Full finetuning is feasible for Qwen3.5 4B on the L40S with Unsloth bf16 full mode and 8-bit Adam.
- Full SFT gives a tiny refinement over the 4B LoRA checkpoint in selection (`30.4393` vs `30.3580`) and token F1 (`30.6209` vs `30.1583`), with a small chrF++ gain (`43.3102` vs `43.2858`) and a small BLEU drop (`16.2178` vs `16.3507`).
- It is not yet a deployable replacement for the K32 ensemble because chrF++ is still lower (`43.3102` vs `43.8064`).
- The useful recipe is not raw full SFT. Use full SFT only after a strong LoRA curriculum checkpoint has been merged.

## Next Steps

- Use `checkpoint-224` LoRA and `checkpoint-48` full-SFT as candidate generators in a mixed pool with the current 2B deployable model; the diversity is likely more valuable than greedy replacement. Done: the K32 current + K16 4B-full pool with matched score ensemble reached selection `35.5481`, chrF++ `48.1624`, BLEU `24.0635`, token F1 `35.9579`, TER `70.5069`.
- Try the same broad -> clean curriculum on Qwen3.5 9B if memory allows.
- If full SFT is revisited, test short low-LR refinements from the best 9B/35B-A3B merged checkpoint rather than scaling the 4B full run blindly.

## Candidate-Generator Result

The 4B full-SFT checkpoint is not a greedy replacement for the conservative K32 ensemble, but it is the strongest candidate-diversity source so far.

Merged eval pool:

- Current deployable K32 candidates plus Qwen3.5 4B full-SFT K16 candidates.
- Eval rows: 158 groups, 4,279 deduped candidates, mean 27.08 candidates/group.
- Matching train pool: 886 groups, 23,786 candidates.

Best matched selector:

- Ensemble: `outputs/score_ensemble_reranker_evals/20260522-ensemble-train-current-k32-plus-qwen35-4b-full-k16-term/ensemble_current_k32_plus_4bfull_k16_ensemble.json`
- Metrics: selection `35.5481`, chrF++ `48.1624`, BLEU `24.0635`, token F1 `35.9579`, source-copy `2.4156%`, leakage `0.0%`, TER `70.5069`.
- Oracle: selection `50.0788`, chrF++ `63.5158`, BLEU `35.7333`, token F1 `54.9308`, TER `51.3825`.

Takeaway:

- The 4B curriculum/full-SFT path is worth scaling, but primarily as a multi-candidate generator.
- The next base-model improvement experiment should be Qwen3.5 9B broad -> clean Chanka LoRA first; only merge and full-SFT it if the LoRA checkpoint has strong external generation metrics.

## Qwen3.5 9B Curriculum Run

Started on 2026-05-22 / 2026-05-23 UTC to scale the successful 4B recipe.

Setup:

- Model: `unsloth/Qwen3.5-9B`
- Output root: `outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64`
- Broad stage: 166,480 train / 3,397 validation rows, max sequence length `256`, LoRA r64/alpha128, LR `5e-5`, effective batch `16`, max steps `256`, eval/save every `64` steps.
- Chanka stage: planned from broad `checkpoint-256`, max sequence length `128`, LoRA r64/alpha128, LR `2e-5`, effective batch `8`, max steps `192`, eval/save every `32` steps.

Early broad status:

| Checkpoint | eval loss | Notes |
| --- | ---: | --- |
| checkpoint-64 | 1.8167 | Saved successfully; train loss logs decreased from `2.68` at step 16 to `1.891` at step 64. |

Automation:

- `experiments/sft/queue_qwen35_curriculum_eval.sh` waits for the 9B curriculum to finish, then evaluates all Chanka checkpoints with terminology top-1.
- `experiments/sft/queue_qwen35_9b_candidate_rerank.sh` waits for that checkpoint-eval summary, selects the best 9B Chanka checkpoint, generates train/eval K16 candidate pools, merges them with the current K32 + 4B-full K16 pools, then trains matching feature/text/ensemble selectors.
- The 9B candidate queue supports retrieval-augmented prompts through `FEW_SHOT_TOP_K` / `FEW_SHOT_MAX_CANDIDATES`, but defaults to no few-shot examples so the first 9B pass isolates model-scale candidate diversity.
- `experiments/sft/queue_qwen35_9b_fewshot_candidate_rerank.sh` chains after the no-few-shot 9B rerank summary and reruns the candidate-rerank pipeline with `FEW_SHOT_TOP_K=2` by default. This gives a sequential scale-only vs retrieval-augmented comparison without concurrent GPU contention.
- `experiments/sft/queue_qwen35_9b_full_sft_after_rerank.sh` chains after the few-shot rerank summary, merges the best 9B LoRA checkpoint to a 16-bit full model, runs a 2-step FFT smoke, then runs a 32-step Chanka full-SFT canary if memory permits. This keeps full fine-tuning behind the current scale/retrieval experiments and avoids GPU contention.

Decision so far:

- The 9B run is mechanically feasible on the L40S in 16-bit LoRA mode.
- Do not judge it from broad eval loss alone. The important evidence will be Chanka held-out generation and whether its K16 candidate pool improves the current multi-model oracle/selector profile.
