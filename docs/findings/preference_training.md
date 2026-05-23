# Preference Training Findings

## 2026-05-22 Oracle-Pair DPO Canary

Purpose: test whether the K16 candidate-pool oracle gap can be converted into a policy update with direct preference optimization. This uses hidden references only on training candidate pools to label `chosen` vs `rejected`; held-out validation remains untouched.

Code added:

- `scripts/build_oracle_preference_pairs.py`: builds preference-pair JSONL from multi-candidate prediction pools. It ranks candidates with the existing reference-aware oracle score, filters weak/unsafe chosen outputs, and writes `source`, `chosen`, `rejected`, scores, margins, and candidate indices.
- `scripts/train_dpo_unsloth.py`: Unsloth + TRL `DPOTrainer` continuation from an existing LoRA adapter, with a separately loaded frozen reference adapter, validation enabled, and optional terminology-conditioned prompts.
- `experiments/gspo/run_dpo_oracle_pairs_canary.sh`: end-to-end builder, DPO train, terminology top-1 held-out eval, and summary launcher.

Preference source:

- Candidate pool: `outputs/verifier_candidate_mining/20260522-train-k16-full-newbest-noterm-t065-p090/train_k16_predictions.jsonl`
- Base adapter: `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8`
- Eval mode: terminology top-1 using `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`
- Standing best for comparison: selection `27.0396`, chrF++ `41.4823`, BLEU `9.7158`, token F1 `27.0736`, TER `87.3272`.

Margin sweep for pair construction:

| Min margin | Kept pairs | Chosen selection | Chosen chrF++ | Chosen BLEU | Rejected selection | Rejected chrF++ | Rejected BLEU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.50` | `4` | `90.0000` | `100.0000` | `100.0000` | `25.0974` | `44.1160` | `18.1230` |
| `0.20` | `84` | `71.6098` | `84.6516` | `64.7267` | `38.7622` | `58.0266` | `9.8707` |
| `0.10` | `162` | `60.0447` | `73.9412` | `45.6199` | `36.4412` | `53.4859` | `13.1008` |
| `0.05` | `261` | `51.2768` | `66.3408` | `32.8917` | `33.9247` | `50.7641` | `10.4055` |
| `0.00` | `856` | `35.9058` | `50.6811` | `16.6124` | `29.8166` | `44.5208` | `9.7816` |

DPO canary results:

| Run | Pairs | LR | Steps | Best adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260522-dpo-oracle-pairs-v1` | `4` | `1e-7` | `16` | `checkpoint-8` | `26.0833` | `40.9402` | `6.7986` | `26.0665` | `2.6688` | `0.4747` | `88.7097` |
| `20260522-dpo-oracle-pairs-margin010` | `162` | `1e-7` | `16` | `checkpoint-16` | `26.2328` | `41.0748` | `6.9083` | `26.2599` | `2.6688` | `0.4747` | `88.0184` |
| `20260522-dpo-oracle-pairs-margin010-lr5e-7` | `162` | `5e-7` | `8` | `checkpoint-8` | `26.8529` | `41.2099` | `9.5910` | `26.8049` | `2.8270` | `0.4747` | `87.0968` |
| `20260522-dpo-oracle-pairs-margin010-lr1e-6` | `162` | `1e-6` | `8` | `checkpoint-8` | `26.1011` | `40.7399` | `6.6965` | `26.4165` | `2.6688` | `0.4747` | `88.2488` |

Qualitative pattern:

- DPO keeps outputs fluent and mostly artifact-free.
- The best high-LR run improves TER slightly versus the standing best (`87.0968` vs `87.3272`) but loses selection, chrF++, BLEU, and token F1.
- Some generated translations become acceptable paraphrases that score poorly against the single clean reference, and some regressions remain obvious. Example from the `1e-7` margin-0.10 run: `En la fiesta -> Fiestapi`, which is a Spanish-root leakage/paraphrase instead of the reference `Raymipim`.
- TRL DPO internal eval was weak for the 162-pair runs: reward margins stayed near zero and eval preference accuracy hovered around `0.54` to `0.58`; the `1e-6` run showed stronger overfit/instability.

Conclusion:

- DPO is now wired and reproducible, but naive oracle-pair DPO from this K16 train pool did not beat the current MBR/terminology SFT best.
- The close `5e-7` result suggests preference optimization is not hopeless, but the labels/objective need work before scaling.
- Next preference attempts should avoid plain independent DPO pairs from single-reference oracle labels. Better candidates:
  - listwise or contrastive training over the full K16 set,
  - DPO pairs where `chosen` is MBR/consensus plus clean-reference anchors, not pure oracle-only labels,
  - preference pairs mined from rows where MBR agrees with oracle,
  - or a discriminative reranker with lexical/terminology features rather than a generative JSON scorer.

## 2026-05-23 4B Full-SFT GSPO Termination Follow-up

Purpose: debug the negative LoRA-GSPO continuation from the best Qwen3.5 4B full-SFT checkpoint. The previous run nudged BLEU but lost selection/token F1, and trainer logs showed every sampled completion reached the 80-token cap.

Code changes:

- `scripts/train_gspo_chanka_unsloth.py` now has optional overlong-completion reward guards:
  - `--overlong-max-words`
  - `--overlong-ratio-threshold`
  - `--overlong-penalty-weight`
- The guard is wired into both learned-verifier rewards and non-verifier reward profiles.
- Final in-script metrics now report average prediction/reference word counts and the overlong penalty.
- GRPOConfig now receives explicit `generation_kwargs` with tokenizer EOS/PAD IDs.
- `experiments/gspo/queue_qwen35_4b_full_sft_gspo_canary.sh` exposes `MAX_PROMPT_LENGTH`, `MAX_COMPLETION_LENGTH`, and the overlong guard knobs.

Runs:

| Run | RL max completion | Train/eval rows | Trainer clipped ratio | External selection | chrF++ | BLEU | token F1 | TER | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `20260523-qwen35-4b-full-sft-lora-gspo-overlong-guard` | `80` | `128/64` | `1.0` | `29.6703` | `43.0678` | `14.8357` | `28.9377` | `86.4055` | Reward guard alone is negative versus the 4B full-SFT base. |
| `20260523-qwen35-4b-full-sft-lora-gspo-eos-overlong-smoke` | `80` | `64/16` | `1.0` | not kept | not kept | not kept | not kept | not kept | Explicit EOS/PAD generation kwargs did not change clipping; full external eval was stopped after the diagnostic. |
| `20260523-qwen35-4b-full-sft-lora-gspo-max32-smoke` | `32` | `64/16` | `1.0` | `29.7981` | `43.1093` | `14.9865` | `29.1185` | `85.7143` | Still clipped, but validation runtime dropped and reward became less negative. |

Comparison targets:

- Best standalone 4B full-SFT base remains `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36`: selection `31.1999`, chrF++ `44.4295`, BLEU `18.0014`, token F1 `29.9850`, TER `81.5668`.
- Best deployable reranked profile remains current K32 + 4B full-SFT K16 with listwise text selection: selection `38.9423`, chrF++ `51.8586`, BLEU `26.0207`, token F1 `40.3340`, TER `67.7419`.

Conclusion:

- Overlong reward guards are useful instrumentation but not sufficient to improve this GSPO recipe.
- Passing EOS/PAD through `generation_kwargs` does not fix TRL/Unsloth GRPO termination for this Qwen3.5 setup; sampled completions still clip at the configured max.
- Shortening RL `max_completion_length` to `32` is the practical workaround for future GSPO canaries: it is faster and slightly better than the 80-token guarded run, but still below the base checkpoint.
- Do not scale 4B full-SFT GSPO until either the sampler termination issue is actually fixed or a max-32 recipe shows a real external gain over the base.
