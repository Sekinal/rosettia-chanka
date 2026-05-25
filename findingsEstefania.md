# Findings - Estefania

## 2026-05-20: Initial Qwen3.5 2B Adapter Comparison Plan

Goal: compare LoRA, DoRA, and rank-stabilized LoRA for Spanish to Chanka Quechua SFT before scaling to larger models or longer runs.

Starting model:

- `unsloth/Qwen3.5-2B`
- Reason: cheaper first-pass smoke tests and ablations than the previous 4B default.

Adapter methods to compare:

- `lora`: standard LoRA baseline.
- `dora`: weight-decomposed LoRA, exposed as `--adapter-method dora`.
- `rslora`: rank-stabilized LoRA, exposed as `--adapter-method rslora`.

Overfitting guardrails:

- Use a validation split for every run.
- Select the best checkpoint by `eval_loss`.
- Evaluate and save by step cadence, not only at epoch boundaries.
- For Chanka runs, the script defaults to `--evals-per-epoch 8` when explicit `--eval-steps` and `--save-steps` are not provided.
- Keep seed, train/eval split, model id, LoRA rank, learning rate, batch size, and max sequence length fixed across adapter methods.

Smoke test target:

```bash
python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id unsloth/Qwen3.5-2B \
  --adapter-method lora \
  --max-train-samples 64 \
  --max-eval-samples 16 \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 1 \
  --output-dir outputs/smoke_qwen35_2b_lora
```

Repeat with `--adapter-method dora` and `--adapter-method rslora` after the LoRA smoke run passes.

Open validation item:

- Confirm on the remote training environment that the installed Unsloth version accepts `use_dora=True`. `use_rslora` is documented by Unsloth; DoRA may depend on the installed Unsloth/PEFT surface.

## 2026-05-20: Remote Unit Tests And Adapter Smoke Tests

Remote host:

- `root@216.81.248.197 -p 20299`
- GPU: NVIDIA L40S, 46 GB VRAM.
- Training stack installed in `/root/rosettia-chanka/.venv`.
- Unsloth: `2026.5.5`
- Torch: `2.10.0+cu128`
- Transformers: `5.5.0`
- Flash Attention 2 was unavailable/broken; Unsloth used XFormers fallback.

Unit tests:

```bash
python -m unittest discover -s tests -v
```

Result: 5 tests passed.

Smoke command pattern:

```bash
python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id unsloth/Qwen3.5-2B \
  --adapter-method METHOD \
  --max-train-samples 32 \
  --max-eval-samples 8 \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 1 \
  --output-dir outputs/smoke_qwen35_2b_METHOD
```

Smoke results:

| Method | Status | Trainable parameters | Step 1 eval loss | Step 2 eval loss | Final eval loss | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LoRA | passed | 87,293,952 | 3.258 | 2.805 | 2.899 | 48.75s |
| DoRA | passed | 87,681,024 | 3.254 | 2.807 | 2.981 | 60.72s |
| rsLoRA | passed | 87,293,952 | 3.258 | 5.253 | 3.444 | 27.88s |

Notes:

- DoRA compatibility is confirmed with the installed Unsloth/PEFT stack.
- The smoke runs are only infrastructure checks, not quality comparisons.
- At rank 128 / alpha 256, the rsLoRA two-step smoke showed a sharp step-2 eval-loss jump. For the first real comparison, consider reducing rank/alpha or running enough steps to see whether this is noise or instability.
- LoRA and DoRA behaved similarly over two steps; DoRA was slower and trained slightly more parameters.

## 2026-05-20: Conservative Rank/Alpha Smoke Matrix

Purpose: test lower-rank adapter configurations without overloading the model or allowing overfitting.

Guardrails:

- Chanka-only smoke runs.
- `max_train_samples=32`, `max_eval_samples=8`.
- `max_steps=2`.
- `eval_steps=1`, `save_steps=1`.
- No full epochs.
- Same seed and dataset split behavior as the previous smoke tests.

Command pattern:

```bash
python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id unsloth/Qwen3.5-2B \
  --adapter-method METHOD \
  --lora-r RANK \
  --lora-alpha ALPHA \
  --max-train-samples 32 \
  --max-eval-samples 8 \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 1 \
  --output-dir outputs/rank_alpha_smoke/METHOD_rRANK_aALPHA
```

Results:

| Method | Rank | Alpha | Trainable parameters | Step 1 eval loss | Step 2 eval loss | Final eval loss | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 32 | 64 | 21,823,488 | 3.258 | 3.041 | 3.260 | 27.82s |
| DoRA | 32 | 64 | 22,210,560 | 3.254 | 3.010 | nan | 46.09s |
| rsLoRA | 32 | 64 | 21,823,488 | 3.258 | 2.846 | 2.911 | 28.21s |
| LoRA | 64 | 128 | 43,646,976 | 3.258 | 2.892 | 3.099 | 28.81s |
| DoRA | 64 | 128 | 44,034,048 | 3.254 | 2.864 | 3.059 | 45.14s |
| rsLoRA | 64 | 128 | 43,646,976 | 3.258 | 3.501 | 3.444 | 27.37s |

Interpretation:

- Lower ranks reduce trainable parameters substantially: r32 trains about 1% of the model, r64 trains about 1.9%, and the earlier r128 setup trained about 3.8%.
- rsLoRA r32/a64 looked stable in this tiny smoke and did not reproduce the r128/a256 spike.
- rsLoRA r64/a128 drifted upward by step 2, so rsLoRA should stay at r32/a64 for the next cautious run.
- LoRA r64/a128 and DoRA r64/a128 both behaved reasonably over two steps, but DoRA is slower.
- DoRA r32/a64 had normal per-step eval losses but returned `nan` on the final explicit evaluation after training. Treat that run as a caution flag and rely on per-step evals until repeated.

Next cautious candidate configurations:

- LoRA r64/a128 as a standard baseline.
- DoRA r64/a128 if extra runtime is acceptable.
- rsLoRA r32/a64 only, not r64/a128 or r128/a256 yet.

## 2026-05-20: 30-Step Gradient Comparison

Purpose: compare the plausible adapter configurations over more than a two-step smoke test, while still avoiding full-epoch overfitting or high-cost runs.

Guardrails:

- Chanka-only run.
- Full current Chanka split: 897 train rows, 158 validation rows.
- `max_steps=30`, which is about 0.27 epoch with the current effective batch size.
- `eval_steps=5`, `save_steps=5`, `logging_steps=1`.
- Same model: `unsloth/Qwen3.5-2B`.
- Same optimizer schedule and validation split behavior.

Command pattern:

```bash
python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id unsloth/Qwen3.5-2B \
  --adapter-method METHOD \
  --lora-r RANK \
  --lora-alpha ALPHA \
  --max-steps 30 \
  --eval-steps 5 \
  --save-steps 5 \
  --logging-steps 1 \
  --output-dir outputs/gradient_compare/METHOD_rRANK_aALPHA
```

Results:

| Method | Rank | Alpha | Best step eval loss | Final explicit eval loss | Train loss | Runtime | Avg grad norm | Grad norm range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA | 64 | 128 | 2.426 | 2.618 | 2.702 | 341.4s | 8.519 | 6.604-13.49 |
| DoRA | 64 | 128 | 2.419 | 2.547 | 2.693 | 376.4s | 8.539 | 6.638-13.63 |
| rsLoRA | 32 | 64 | 2.250 | 2.276 | 2.540 | 333.4s | 29.201 | 20.67-54.36 |

Eval-loss curves at steps 5/10/15/20/25/30 plus final explicit eval:

- LoRA r64/a128: 2.800, 2.647, 2.539, 2.472, 2.437, 2.426, final 2.618.
- DoRA r64/a128: 2.783, 2.643, 2.528, 2.464, 2.430, 2.419, final 2.547.
- rsLoRA r32/a64: 2.652, 2.463, 2.358, 2.304, 2.272, 2.250, final 2.276.

Gradient observations:

- LoRA and DoRA have nearly identical gradient norm behavior. Both average about 8.5 with maximums around 13.5.
- DoRA does not show a meaningful gradient-stability advantage over LoRA in this run.
- DoRA's best eval loss was only 0.007 better than LoRA at step 30, while runtime was about 10% slower.
- rsLoRA r32/a64 has much larger gradient norms, averaging about 29 and peaking at 54.36. It still improved eval loss smoothly here, but the gradient scale is very different and should be watched closely before longer runs.

Current interpretation:

- DoRA is not clearly worth it yet. Its tiny eval-loss improvement over LoRA does not justify the slower runtime unless later qualitative translation checks show a real gain.
- LoRA r64/a128 remains the safest standard baseline.
- rsLoRA r32/a64 deserves a controlled follow-up because its eval loss was best, but it should use frequent evals and gradient logging because the gradients are much larger.
- Do not increase rsLoRA rank/alpha yet.

## 2026-05-20: First Full LoRA Epoch

Purpose: start the main LoRA baseline after the short adapter comparison, with frequent validation/checkpointing before the epoch completes to watch for overfitting.

Configuration:

- Remote host only: `root@216.81.248.197:20299`.
- Model: `unsloth/Qwen3.5-2B`.
- Stage: Chanka-only.
- Split: 897 train rows, 158 validation rows.
- Adapter: LoRA r64/a128.
- Training length: 1 epoch, 113 optimizer steps.
- Evaluation/checkpoint cadence: every 10 steps, plus final epoch checkpoint.
- Logging cadence: every optimizer step.
- Output: `outputs/qwen35_2b_chanka_lora_r64_a128/chanka`.
- Final adapter: `outputs/qwen35_2b_chanka_lora_r64_a128/chanka/final_lora`.
- Log: `outputs/qwen35_2b_chanka_lora_r64_a128/logs/train.log`.

Command:

```bash
python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id unsloth/Qwen3.5-2B \
  --adapter-method lora \
  --lora-r 64 \
  --lora-alpha 128 \
  --num-train-epochs 1 \
  --eval-steps 10 \
  --save-steps 10 \
  --logging-steps 1 \
  --output-dir outputs/qwen35_2b_chanka_lora_r64_a128
```

Eval-loss curve:

| Step | Epoch | Eval loss |
| ---: | ---: | ---: |
| 10 | 0.089 | 2.653 |
| 20 | 0.178 | 2.434 |
| 30 | 0.268 | 2.321 |
| 40 | 0.357 | 2.246 |
| 50 | 0.446 | 2.193 |
| 60 | 0.535 | 2.120 |
| 70 | 0.624 | 2.086 |
| 80 | 0.714 | 2.072 |
| 90 | 0.803 | 2.040 |
| 100 | 0.892 | 2.026 |
| 110 | 0.981 | 2.017 |
| 113 | 1.000 | 2.016 |
| Final explicit eval | 1.000 | 2.068 |

Metrics:

- Best in-training eval loss: 2.016 at step 113.
- Final explicit eval loss after saving `final_lora`: 2.068.
- Train loss: 2.369.
- Runtime: 947s.
- Train throughput: 0.119 steps/s.
- Average gradient norm: 8.638.
- Gradient norm range: 6.713-24.86.
- The only gradient norm above the earlier LoRA comparison range was the final optimizer step: 24.86 at step 113 with learning rate `1.869e-07`.
- Retained checkpoints: `checkpoint-100`, `checkpoint-110`, `checkpoint-113`; older checkpoints were pruned by the trainer save limit.

Interpretation:

- The first full LoRA epoch completed cleanly and validation loss improved throughout the epoch.
- The improvements became smaller after about step 80, but the curve still moved from 2.072 to 2.016 between steps 80 and 113.
- The final explicit eval loss was higher than the last in-training eval, so use `checkpoint-113` and `final_lora` as close candidates rather than assuming the final saved adapter is strictly better.
- Gradient behavior remained mostly consistent with the 30-step LoRA baseline. The final-step spike should be noted, but it did not resemble the much larger rsLoRA gradient scale observed earlier.

## 2026-05-20: Sequence Length and Batch VRAM Check

Purpose: measure the actual formatted Chanka sequence lengths and choose a less wasteful batch/context setup before continuing training.

Important workflow correction:

- The `chanka` stage uses the reviewed judicial Chanka dataset: `clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet`.
- The `broad` stage uses non-judicial broad data: SomosNLP plus AmericasNLP.
- The LoRA/DoRA/rsLoRA and VRAM runs above used `--stage chanka`, so treat them as calibration runs for adapter behavior, context length, batch size, and overfitting risk, not as the SFT training sequence.
- Main SFT should start with `--stage broad` on SomosNLP + AmericasNLP.
- Reviewed judicial Chanka data is reserved for final GSPO, not SFT.
- The SFT script now rejects `--stage chanka` from the CLI to prevent accidental judicial-data SFT.

Sequence-length analysis:

- Tokenizer/model: `unsloth/Qwen3.5-2B`.
- Text measured after applying the training chat template.
- Train split: 897 rows, mean 78.54 tokens, median 78, p90 88, p95 91, p99 102, max 115.
- Eval split: 158 rows, mean 77.99 tokens, median 77, p90 85, p95 90, p99 94, max 97.
- All Chanka rows: 1,055 rows, mean 78.45 tokens, median 78, p90 88, p95 91, p99 99, max 115.
- No rows exceeded 128 tokens. The previous `max_seq_length=1024` was far larger than needed for this dataset.

Short VRAM probes at `max_seq_length=128`, LoRA r64/a128, 5 optimizer steps:

| Per-device batch | Peak VRAM |
| ---: | ---: |
| 2 | 5,359 MiB |
| 4 | 5,707 MiB |
| 8 | 6,233 MiB |
| 16 | 7,389 MiB |

Full one-epoch run selected for continuity:

- Model: `unsloth/Qwen3.5-2B`.
- Adapter: LoRA r64/a128.
- Stage: Chanka-only.
- `max_seq_length=128`.
- Per-device train/eval batch size: 8.
- Gradient accumulation: 1.
- Effective batch size: 8, matching the earlier `batch=1, grad_accum=8` run.
- Training length: 1 epoch, 113 optimizer steps.
- Output: `outputs/qwen35_2b_chanka_lora_r64_a128_seq128_b8/chanka`.
- Final adapter: `outputs/qwen35_2b_chanka_lora_r64_a128_seq128_b8/chanka/final_lora`.
- VRAM log: `outputs/qwen35_2b_chanka_lora_r64_a128_seq128_b8/logs/vram.csv`.

Full-run metrics:

- Peak VRAM: 6,313 MiB on an NVIDIA L40S with 44.392 GiB reported by Unsloth.
- Average sampled VRAM: 5,161 MiB.
- p95 sampled VRAM: 6,193 MiB.
- Average sampled GPU utilization: 16.4%, max 46%.
- Train runtime: 206.9s.
- Train throughput: 0.546 steps/s, 4.335 samples/s.
- Train loss: 2.376.
- Best in-training eval loss: 2.058 at epoch 1.0.
- Final explicit eval loss after saving `final_lora`: 2.108.
- Average gradient norm: 8.646.
- Gradient norm range: 6.733-24.93.
- The final optimizer step again had the only large gradient reading: 24.93.

Eval-loss curve:

| Step | Epoch | Eval loss |
| ---: | ---: | ---: |
| 10 | 0.089 | 2.720 |
| 20 | 0.177 | 2.489 |
| 30 | 0.266 | 2.367 |
| 40 | 0.354 | 2.291 |
| 50 | 0.443 | 2.237 |
| 60 | 0.531 | 2.165 |
| 70 | 0.620 | 2.129 |
| 80 | 0.708 | 2.111 |
| 90 | 0.797 | 2.082 |
| 100 | 0.885 | 2.068 |
| 110 | 0.974 | 2.059 |
| 113 | 1.000 | 2.058 |
| Final explicit eval | 1.000 | 2.108 |

Interpretation:

- Set the Chanka default `max_seq_length` to 128. It covers the current maximum formatted row, not only p99, while removing the 1024-token outlier risk and wasted memory.
- Set the Chanka default per-device train/eval batch size to 8 and gradient accumulation to 1. This keeps effective batch size 8 for continuity with earlier LoRA results while using only about 6.3 GiB peak VRAM.
- Set the Chanka default LoRA rank/alpha to r64/a128, matching the measured stable baseline.
- Batch 16 also fits comfortably, but it changes the effective batch size if gradient accumulation remains 1. Use batch 16 only after a controlled comparison or if throughput becomes more important than exact continuity.
- The seq128/batch8 run was about 4.6x faster than the earlier seq1024/batch1+accum8 run: 206.9s vs 947s for one epoch.
- Validation behavior remained similar to the seq1024 run. The shorter context did not truncate any Chanka examples.

## 2026-05-20: Broad SFT Setup

Purpose: switch main SFT training to the non-judicial broad data only, reserving reviewed Chanka judicial data for GSPO.

Dataset routing:

- SFT `--stage broad`: SomosNLP + AmericasNLP.
- Chanka reviewed judicial data: GSPO only, not SFT.
- `scripts/train_sft_unsloth.py` rejects `--stage chanka` at the CLI.

Broad sequence-length analysis:

- Loaded rows: 169,877.
- SomosNLP rows: 82,873.
- AmericasNLP rows: 87,004.
- Train split: 166,480 rows, mean 120.09 tokens, median 115, p90 158, p95 174, p99 216, p99.9 286, max 443.
- Eval split: 3,397 rows, mean 120.09 tokens, median 115, p90 158, p95 175, p99 216, p99.9 304, max 367.
- All broad rows: mean 120.09 tokens, median 115, p90 158, p95 174, p99 216, p99.9 287, max 443.
- No broad rows exceeded 512 tokens.

Short broad VRAM probes at `max_seq_length=512`, LoRA r64/a128, 5 optimizer steps, small eval subset:

| Per-device batch | Peak VRAM |
| ---: | ---: |
| 2 | 5,577 MiB |
| 4 | 5,845 MiB |
| 8 | 7,307 MiB |
| 16 | 6,703 MiB |

Decision:

- Set broad SFT default `max_seq_length` to 512. It covers the current max formatted broad row.
- Set broad SFT default per-device train/eval batch size to 8.
- Set broad SFT default gradient accumulation to 2, preserving effective batch size 16 from the earlier broad default while using GPU memory more efficiently.
- Use LoRA r64/a128 for the first broad SFT run.

## 2026-05-20: Broad LoRA SFT Batch-Size Checkpoint

Purpose: cautiously increase the physical batch size for broad SFT without changing the effective batch size.

Active run:

- Model: `unsloth/Qwen3.5-2B`.
- Dataset stage: `broad` only, SomosNLP + AmericasNLP.
- Adapter: LoRA, r64/a128.
- Sequence length: 512.
- Per-device train/eval batch size: 16.
- Gradient accumulation: 1.
- Effective batch size: 16, matching the earlier batch8/grad-accum2 setting.
- Total optimizer steps for one epoch: 10,405.
- Output directory: `outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1`.

Checkpoint at step 1300:

- Train loss at step 1300: 1.2737.
- Eval loss at step 1300: 1.2101.
- Eval runtime: 64.9s over 3,397 eval rows.
- Recent gradient norms before eval: mostly 2.65-3.46.
- Recent train losses before eval: mostly 1.14-1.31.
- Sampled VRAM peak through first eval/checkpoint: 10,897 MiB.
- Sampled VRAM p95 through first eval/checkpoint: 8,357 MiB.
- Sampled average VRAM through first eval/checkpoint: 7,548 MiB.

Interpretation:

- Batch 16 is safe on the L40S for this setup. It leaves large VRAM headroom even during eval/checkpointing.
- Keeping gradient accumulation at 1 preserves the effective batch size of 16, so the change improves throughput without intentionally changing optimization behavior.
- Gradients are not showing instability at this checkpoint. They rose during LR warmup, then settled near 3.0 by step 1300.
- Do not increase the batch again mid-run. Wait for more eval checkpoints to confirm the trend and avoid conflating batch changes with the one-epoch baseline.

Checkpoint at step 2600:

- Train loss at step 2600: 1.0449.
- Eval loss at step 2600: 1.0114.
- Eval runtime: 63.5s over 3,397 eval rows.
- Recent gradient norms before eval: mostly 2.34-3.00.
- Recent train losses before eval: mostly 0.96-1.17.
- Sampled VRAM peak through second eval/checkpoint: 11,777 MiB.
- Sampled VRAM p95 through second eval/checkpoint: 8,357 MiB.
- Sampled average VRAM through second eval/checkpoint: 7,578 MiB.

Interpretation after step 2600:

- Batch 16 continues to look stable. Eval loss improved from 1.2101 to 1.0114 between the first two checkpoints.
- Gradient norms decreased rather than spiking, which supports continuing the one-epoch baseline with this batch setting.
- VRAM headroom remains large; the peak is still only about 26.5% of the 44.4 GiB card.

Checkpoint at step 3900:

- Train loss at step 3900: 0.8581.
- Eval loss at step 3900: 0.9126.
- Eval runtime: 63.9s over 3,397 eval rows.
- Recent gradient norms before eval: mostly 2.13-3.11.
- Recent train losses before eval: mostly 0.84-1.09.
- Sampled VRAM peak through third eval/checkpoint: 11,777 MiB.
- Sampled VRAM p95 through third eval/checkpoint: 8,597 MiB.
- Sampled average VRAM through third eval/checkpoint: 7,639 MiB.

Interpretation after step 3900:

- Eval loss continues to improve, from 1.2101 to 1.0114 to 0.9126 across the first three checkpoints.
- Gradient norms remain controlled, with no evidence that batch 16 is destabilizing the run.
- VRAM peak did not increase beyond the step-2600 peak.

Checkpoint at step 5200:

- Train loss at step 5200: 0.7877.
- Eval loss at step 5200: 0.8391.
- Eval runtime: 62.7s over 3,397 eval rows.
- Recent gradient norms before eval: mostly 2.36-3.15.
- Recent train losses before eval: mostly 0.78-0.92.
- Sampled VRAM peak through halfway eval/checkpoint: 11,777 MiB.
- Sampled VRAM p95 through halfway eval/checkpoint: 8,517 MiB.
- Sampled average VRAM through halfway eval/checkpoint: 7,655 MiB.

Interpretation after step 5200:

- Eval loss continues to improve at the halfway point: 1.2101 -> 1.0114 -> 0.9126 -> 0.8391.
- No batch-related instability is visible in gradients. The gradient norms remain in the same stable range.
- VRAM peak remains unchanged, so batch 16 is clearly safe from a memory perspective on this card.

Checkpoint at step 6500:

- Train loss at step 6500: 0.7315.
- Eval loss at step 6500: 0.7772.
- Eval runtime: 63.9s over 3,397 eval rows.
- Recent gradient norms before eval: mostly 2.22-3.09.
- Recent train losses before eval: mostly 0.68-0.98.
- Sampled VRAM peak through fifth eval/checkpoint: 11,777 MiB.
- Sampled VRAM p95 through fifth eval/checkpoint: 8,597 MiB.
- Sampled average VRAM through fifth eval/checkpoint: 7,711 MiB.

Interpretation after step 6500:

- Eval loss is still improving: 1.2101 -> 1.0114 -> 0.9126 -> 0.8391 -> 0.7772.
- Gradient norms remain controlled and similar to earlier checkpoints, so LoRA batch 16/GA1 continues to look stable.
- VRAM remains far below the L40S capacity, but the run should finish as-is to keep the one-epoch baseline clean.

Checkpoint at step 7800:

- Train loss at step 7800: 0.7456.
- Eval loss at step 7800: 0.7253.
- Eval runtime: 62.5s over 3,397 eval rows.
- Recent gradient norms before eval: mostly 2.26-3.11.
- Recent train losses before eval: mostly 0.61-0.85.
- Sampled VRAM peak through sixth eval/checkpoint: 11,777 MiB.
- Sampled VRAM p95 through sixth eval/checkpoint: 8,597 MiB.
- Sampled average VRAM through sixth eval/checkpoint: 7,728 MiB.

Interpretation after step 7800:

- Eval loss continues to improve: 1.2101 -> 1.0114 -> 0.9126 -> 0.8391 -> 0.7772 -> 0.7253.
- Gradient norms are still stable, with no spike pattern suggesting the batch increase is harming optimization.
- Batch 16/GA1 remains safe on memory; no new VRAM peak appeared at this checkpoint.

Checkpoint at step 9100:

- Train loss at step 9100: 0.7145.
- Eval loss at step 9100: 0.6837.
- Eval runtime: 63.2s over 3,397 eval rows.
- Recent gradient norms before eval: mostly 2.10-3.29.
- Recent train losses before eval: mostly 0.60-0.72.
- Sampled VRAM peak through seventh eval/checkpoint: 11,777 MiB.
- Sampled VRAM p95 through seventh eval/checkpoint: 8,595 MiB.
- Sampled average VRAM through seventh eval/checkpoint: 7,747 MiB.

Interpretation after step 9100:

- Eval loss is still improving: 1.2101 -> 1.0114 -> 0.9126 -> 0.8391 -> 0.7772 -> 0.7253 -> 0.6837.
- Gradient norms remain bounded and do not show a worsening trend as the LR decays toward the end of the epoch.
- Memory remains stable; batch 16/GA1 is not close to OOM on the L40S.

End-of-epoch result:

- Training completed successfully: 10,405 / 10,405 optimizer steps.
- Trainer runtime: 9,910s.
- Trainer throughput: 16.8 train samples/s, 1.05 train steps/s.
- Trainer mean train loss over the epoch: 0.9583.
- Train loss at final step 10405: 0.5955.
- Eval loss at step 10400: 0.6610.
- Eval loss at step 10405: 0.6611.
- Best checkpoint by eval loss: `outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400`.
- Saved final adapter: `outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/final_lora`.
- Post-save final adapter eval loss: 0.6875.
- Final sampled VRAM peak: 11,895 MiB.
- Final sampled VRAM p95: 8,597 MiB.
- Final sampled average VRAM: 7,738 MiB.

Final interpretation for LoRA r64/a128, seq512, batch16/GA1:

- Batch 16 is safe on the L40S for this setup. Peak VRAM stayed under 12 GiB on a 44.4 GiB card.
- The larger per-device batch did not destabilize training. Gradient norms stayed mostly in the 2.1-3.6 range through the epoch, with no runaway spikes.
- Validation loss improved at every scheduled checkpoint: 1.2101 -> 1.0114 -> 0.9126 -> 0.8391 -> 0.7772 -> 0.7253 -> 0.6837 -> 0.6610/0.6611.
- Because eval loss was still improving at the end of one epoch, there is no evidence of overfitting in this one-epoch LoRA baseline. The next comparison should keep this batch/sequence setting fixed while testing DoRA and rsLoRA, so adapter method is the main variable.

## 2026-05-20: Chanka GSPO From Broad LoRA Checkpoint

Purpose: establish the completed broad LoRA adapter as the checkpoint for Chanka-only preference-style training, using the reviewed Chanka judicial dataset for GSPO/RL and not for SFT.

Starting checkpoint:

- Broad LoRA source: `outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400`.
- Broad baseline best eval loss: 0.6610 at step 10400.
- Model: `unsloth/Qwen3.5-2B`.
- Adapter: LoRA r64/a128 inherited from the broad checkpoint.

Dataset routing:

- GSPO Chanka data: `clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet`.
- Loaded rows: 1,055.
- Train split: 897 rows.
- Validation split: 158 rows.
- The Chanka target is used only inside reward/evaluation functions. The prompt does not include supervised labels.
- The prompt asks for Spanish to Quechua Chanka judicial translation.

Implementation note:

- The installed TRL version exposes `GRPOTrainer`, not a separate `GSPOTrainer`.
- The run uses TRL's sequence-level importance sampling mode through `importance_sampling_level="sequence"`, which is the GSPO-style configuration available in this stack.

Run configuration:

- Output: `outputs/qwen35_2b_chanka_gspo_seq128_g4_b1_ga4_eval112_jsonl`.
- Final adapter: `outputs/qwen35_2b_chanka_gspo_seq128_g4_b1_ga4_eval112_jsonl/chanka_gspo/final_gspo_lora`.
- Scalar logs: `outputs/qwen35_2b_chanka_gspo_seq128_g4_b1_ga4_eval112_jsonl/chanka_gspo/scalar_logs.jsonl`.
- Final metrics: `outputs/qwen35_2b_chanka_gspo_seq128_g4_b1_ga4_eval112_jsonl/chanka_gspo/final_metrics.json`.
- Max sequence length: 128.
- Max prompt length: 96.
- Max completion length: 80.
- Train batch size: 1.
- Gradient accumulation: 4.
- Number of generations: 4.
- Eval batch size: 4.
- Learning rate: `5e-6`.
- Epochs: 1.
- Eval/checkpoint cadence: every 112 trainer steps, approximately 8 evals per epoch.
- KL beta: 0.0. This avoids the Unsloth DAPO warning; KL is therefore logged as 0 by design.
- Loss type: DAPO.
- Reward scaling: group.

Reward and metrics:

- chrF++ from SacreBLEU `CHRF(word_order=2)`.
- chrF2 from SacreBLEU `CHRF(word_order=0)`.
- BLEU with effective order.
- TER.
- Token F1.
- Length-ratio score.
- Spanish leakage penalty.
- Exact source-copy rate.
- Reward composition: `0.62*chrF++ + 0.18*BLEU + 0.12*token_f1 + 0.08*length_score - Spanish leakage penalty`.

Validation reward curve:

| Step | Eval reward | Eval loss |
| ---: | ---: | ---: |
| 112 | 0.323243 | 0.026400 |
| 224 | 0.330221 | 0.014996 |
| 336 | 0.342346 | 0.006396 |
| 448 | 0.349474 | 0.013489 |
| 560 | 0.348714 | 0.017320 |
| 672 | 0.369095 | 0.018173 |
| 784 | 0.363374 | 0.002965 |
| 896 | 0.359266 | 0.024370 |
| 897 | 0.362562 | 0.017255 |

Best GSPO checkpoint:

- Best reward checkpoint: `outputs/qwen35_2b_chanka_gspo_seq128_g4_b1_ga4_eval112_jsonl/chanka_gspo/checkpoint-672`.
- Best eval reward: 0.369095.
- Eval reward std: 0.061147.
- Eval completion length: 12.3924.
- Eval clipped completion ratio: 0.0.
- Eval KL: 0.0.
- Eval clip-ratio region mean: 0.0.

Final validation corpus metrics for the saved final adapter:

| Metric | Value |
| --- | ---: |
| chrF++ | 42.2119 |
| chrF2 | 47.8233 |
| BLEU | 8.1688 |
| TER | 87.7880 |
| Token F1 | 23.5112 |
| Length-ratio score | 79.3510 |
| Spanish leakage penalty | 0.0 |
| Exact source-copy rate | 0.6329 |
| Trainer eval reward | 0.362562 |

Runtime, VRAM, and gradients:

- Trainer runtime: 6,117s, about 1h42m.
- Train throughput: 0.147 samples/s and 0.147 steps/s.
- Train loss: 0.01553.
- Peak sampled VRAM: 15,831 MiB.
- Average sampled VRAM: 7,841 MiB.
- Train gradient norm mean: 14.612.
- Train gradient norm p95: 31.438.
- Train gradient norm max: 82.524.
- Gradients stayed finite, with occasional spikes. There were no NaNs or crashes.
- Completion truncation stayed at 0.0 on evals, so `max_completion_length=80` is not clipping the current outputs.

Interpretation:

- The GSPO run improved reward through the first two thirds of the epoch, peaking at step 672, then dipped slightly. Use `checkpoint-672` as the Chanka GSPO checkpoint for the next comparison rather than assuming the final adapter is best.
- VRAM has large headroom on the L40S, but runtime and eval generation cost are the practical bottlenecks for this GSPO setup.
- The gradient spikes are worth watching before enabling DoRA for GSPO. They remained finite under LoRA, but DoRA should be compared only after this LoRA checkpoint is preserved.
- The exact source-copy rate in final corpus metrics is high enough to inspect qualitative samples before trusting the reward alone. The next run should report a small fixed sample set of source, reference, and generation at every checkpoint.
- Keep Chanka judicial data restricted to GSPO/RL. Do not use it for SFT.

## 2026-05-25: Qwen3.5 9B Remote Smoke Tests

Purpose: start testing Qwen3.5 9B on the A100 host before committing to longer runs.

Remote execution target:

- Host: `root@154.54.100.193 -p 20295`.
- GPU: NVIDIA A100-SXM4-80GB, 81,920 MiB.
- Repo path: `/root/rosettia-chanka`.
- Environment: remote `.venv` with Torch 2.10.0+cu128, Transformers 5.5.0, TRL 0.24.0, Unsloth 2026.5.7, bitsandbytes 0.49.2.
- Flash Attention 2 was unavailable; Unsloth used XFormers.

Remote validation:

- `python -m unittest discover -s tests -v`
- Result: 299 tests passed.

Broad-stage 9B LoRA smoke:

- Model: `unsloth/Qwen3.5-9B`.
- Output: `outputs/qwen35_9b_tests/20260525-broad-smoke`.
- Final adapter: `outputs/qwen35_9b_tests/20260525-broad-smoke/broad/final_lora`.
- Stage: broad.
- Train/eval rows: 32/8.
- Max sequence length: 256.
- LoRA r/alpha: 64/128.
- Effective batch: 8 via batch 1 and grad accumulation 8.
- Max steps: 2.
- Trainable parameters: 116,391,936 / 9,526,205,680, or 1.22%.
- Step 1 train loss / grad norm: 3.194 / 4.033.
- Step 1 eval loss: 2.846.
- Step 2 train loss / grad norm: 2.534 / 3.608.
- Step 2 eval loss: 2.746.
- Final explicit eval loss: 2.727.
- Train runtime: 68.18s.
- Peak sampled VRAM: 19,871 MiB.
- p95 sampled VRAM: 19,801 MiB.
- Average sampled VRAM: 10,998 MiB.

Chanka terminology continuation smoke:

- Starting adapter: `outputs/qwen35_9b_tests/20260525-broad-smoke/broad/final_lora`.
- Output: `outputs/qwen35_9b_tests/20260525-chanka-term-smoke`.
- Final adapter: `outputs/qwen35_9b_tests/20260525-chanka-term-smoke/chanka/final_lora`.
- Stage: Chanka, terminology top-1 prompt.
- Terminology file: `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`.
- Terminology entries loaded: 128.
- Train/eval rows: 32/8.
- Terminology-matched train/eval rows: 3/0.
- Max sequence length: 128.
- Effective batch: 8 via batch 1 and grad accumulation 8.
- Max steps: 2.
- Step 1 train loss / grad norm: 2.452 / 7.720.
- Step 1 eval loss: 2.330.
- Step 2 train loss / grad norm: 2.366 / 5.781.
- Step 2 eval loss: 2.214.
- Final explicit eval loss: 2.160.
- Train runtime: 42.0s.
- Peak sampled VRAM: 19,801 MiB.
- p95 sampled VRAM: 19,801 MiB.
- Average sampled VRAM: 15,612 MiB.

Interpretation:

- Qwen3.5 9B LoRA is mechanically safe on the A100 with large VRAM headroom at batch 1 / GA 8.
- Gradients were finite in both 2-step smokes. The Chanka terminology continuation has higher gradient norms than the broad smoke but no immediate instability.
- The A100 can support a more serious 9B run. The next useful test should not be another tiny smoke; it should either run a longer broad curriculum slice with external generation evaluation, or reproduce the current best deployable experiment shape by generating a 9B candidate pool and training a matched listwise selector.
- These smoke metrics are infrastructure checks only. They are not quality evidence against or for 9B yet.

## 2026-05-25: Qwen3.5 9B Broad LoRA, 2 Epochs

Purpose: test whether training for more than one epoch on the 4,096-row broad slice continues improving or begins to overfit.

Run:

- Remote execution target: `root@154.54.100.193 -p 20295`.
- Output: `outputs/qwen35_9b_tests/20260525-broad4096-seq512-b2-ga8-2epochs`.
- Final adapter: `outputs/qwen35_9b_tests/20260525-broad4096-seq512-b2-ga8-2epochs/broad/final_lora`.
- Model: `unsloth/Qwen3.5-9B`.
- Stage: broad.
- Train/eval rows: 4,096/512.
- Epochs: 2.0, 512 optimizer steps.
- Max sequence length: 512.
- LoRA r/alpha: 64/128.
- Effective batch: 16 via per-device batch 2 and gradient accumulation 8.
- Learning rate: 2e-5.
- Eval/checkpoint cadence: every 32 steps.
- Retained checkpoints: `checkpoint-416`, `checkpoint-448`, `checkpoint-480`, `checkpoint-512`.

Eval loss curve:

- Step 32, epoch 0.125: 2.280.
- Step 64, epoch 0.250: 2.010.
- Step 96, epoch 0.375: 1.869.
- Step 128, epoch 0.500: 1.763.
- Step 160, epoch 0.625: 1.697.
- Step 192, epoch 0.750: 1.637.
- Step 224, epoch 0.875: 1.592.
- Step 256, epoch 1.000: 1.547.
- Step 288, epoch 1.125: 1.525.
- Step 320, epoch 1.250: 1.507.
- Step 352, epoch 1.375: 1.481.
- Step 384, epoch 1.500: 1.466.
- Step 416, epoch 1.625: 1.451.
- Step 448, epoch 1.750: 1.441.
- Step 480, epoch 1.875: 1.432.
- Step 512, epoch 2.000: 1.427.
- Final explicit eval loss: 1.4276.

Training and stability:

- Train runtime: 6,072s, about 101.2 minutes.
- Final train loss log: 1.306.
- Mean train loss across logs: 1.6422.
- Mean grad norm: 5.5825.
- p95 grad norm: 7.213.
- Max grad norm: 8.978.
- Gradients stayed finite. There was one visible high spike near step 448, but the following gradient norms returned to the 6-8 range.

VRAM:

- Samples: 3,030.
- Peak sampled VRAM: 21,049 MiB.
- p95 sampled VRAM: 20,257 MiB.
- Average sampled VRAM: 20,017 MiB.
- GPU memory after run: 0 MiB used.

Interpretation:

- More than one epoch helped on this validation slice. Eval loss improved continuously through the final checkpoint, from 1.547 at 1 epoch to 1.427 at 2 epochs.
- The improvement is slowing. The second epoch gained about 0.120 eval loss total, but the final quarter epoch gained only about 0.014.
- Use `checkpoint-512` or `final_lora` as the current broad-stage 9B LoRA checkpoint unless external generation metrics contradict it.
- This is still broad SFT only. It should not be treated as the Chanka/GSPO result, and judicial Chanka data should remain reserved for GSPO/RL.
- The next serious comparison should add generation metrics on fixed validation samples, especially chrF++ and BLEU, before deciding whether longer broad SFT is worth more compute.

## 2026-05-25: Qwen3.5 9B Broad Checkpoint -> Chanka GSPO Canary

Purpose: test whether the 9B broad LoRA checkpoint can beat the best standalone 4B full-SFT checkpoint after a Chanka GSPO pass. Baseline target from the 4B checkpoint is chrF++ `44.4295` and BLEU `18.0014`.

Run:

- Remote execution target: `root@154.54.100.193 -p 20295`.
- Starting adapter: `outputs/qwen35_9b_tests/20260525-broad4096-seq512-b2-ga8-2epochs/broad/final_lora`.
- Output: `outputs/qwen35_9b_tests/20260525-chanka-gspo-ref-rerank-24steps`.
- Reward profile: `reference_rerank_vibe_v1`.
- Note: the hard learned-verifier checkpoint used by the older 4B GSPO launcher was not present on this A100 host, so this run used the reference-rerank reward instead of `learned_verifier_vibe_2511`.
- Dataset: clean Chanka judicial parallel corpus, used only for GSPO/RL.
- Train/eval rows during GSPO: 384/64.
- Max steps: 24.
- Eval/checkpoint cadence: every 8 steps.
- Max prompt/completion length: 96/80.
- Terminology: `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`, top-k 1.
- Terminology-matched rows: 48 train / 8 validation.
- Per-device train/eval batch: 4/4.
- Gradient accumulation: 4.
- Num generations: 4.
- Learning rate: 3e-7, warmup 0.
- Overlong guard: max words 48, ratio threshold 2.75.

Trainer reward:

- Step 8 eval reward: 0.1613.
- Step 16 eval reward: 0.1620.
- Step 24 eval reward: 0.1634.
- Train reward logs: 0.1420, 0.1573, 0.1722, 0.1977, 0.1620, 0.1764.
- Mean grad norm: 14.6667.
- Max grad norm: 18.89.
- Completions were short, about 15-19 tokens, with 0 clipping.

VRAM:

- Peak sampled VRAM: 39,059 MiB.
- p95 sampled VRAM: 38,871 MiB.
- Average sampled VRAM: 25,583 MiB.
- GPU memory after run: 0 MiB used.

Canonical full held-out reload eval, 158 rows, batch size 1:

| Adapter | chrF++ | BLEU | token F1 | TER |
| --- | ---: | ---: | ---: | ---: |
| 9B broad seed, before GSPO | 27.5314 | 2.6680 | 12.5258 | 117.7419 |
| GSPO checkpoint-8 | 27.2250 | 2.4716 | 12.0345 | 121.8894 |
| GSPO checkpoint-16 | 27.7565 | 2.6741 | 12.2937 | 118.4332 |
| GSPO checkpoint-24 / final | 27.6404 | 2.5716 | 12.0852 | 119.8157 |
| 4B full-SFT baseline checkpoint | 44.4295 | 18.0014 | 29.9850 | 81.5668 |

Interpretation:

- This 9B broad-only -> GSPO canary does not beat the 4B checkpoint. It is far below the 4B baseline on chrF++ and BLEU.
- GSPO barely changed the broad 9B seed: best chrF++ moved from 27.5314 to 27.7565, while BLEU moved from 2.6680 to 2.6741 and token F1 decreased.
- The problem is not VRAM or runtime. The A100 has enough memory. The weak point is the starting policy: broad-only 9B has not learned enough Chanka surface form to make a short reference-rerank GSPO pass competitive.
- Do not scale this exact 9B broad-only GSPO recipe. A fairer next 9B contender needs either a non-judicial clean Chanka/terminology SFT seed if allowed, a stronger candidate-rerank path, or the learned-verifier checkpoint restored on this host before trying the older `learned_verifier_vibe_2511` recipe.
