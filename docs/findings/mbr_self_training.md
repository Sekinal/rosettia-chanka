# MBR Self-Training Findings

## 2026-05-22 K8 Train-512 Canary

Purpose: test whether reference-free MBR-selected translations can be recycled as pseudo-labels for a small SFT continuation from the current best Chanka adapter.

Code added:

- `scripts/train_jsonl_sft_unsloth.py`: generic JSONL SFT trainer for local/pseudo-labeled translation rows.
- `tests/test_train_jsonl_sft_unsloth.py`: loading, filtering, prompt-formatting, and step-schedule tests.

Pseudo-label data:

- Source candidates: `outputs/candidate_reranker_runs/20260522-source-only-reranker-v1/train_candidates_predictions.jsonl`
- MBR selection output: `outputs/mbr_self_training_data/20260522-k8-train512/mbr_train_mbr_predictions.jsonl`
- Train candidate groups: 510
- MBR pseudo-label quality against hidden references:
  - selection `26.2086`
  - chrF++ `40.6849`
  - BLEU `8.0435`
  - token F1 `24.3294`
  - source-copy `1.8966%`
  - leakage `0.0980%`
  - TER `92.0924`

SFT setup:

- Start adapter: `outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora`
- Output: `outputs/mbr_self_training_sft/20260522-k8-train512-from-current-best-lr2e-6-64steps`
- Input rows after filters: 501
- Split: 426 train / 75 validation
- LR: `2e-6`
- Max steps: `64`
- Batch: per-device `4`, gradient accumulation `2`
- Validation/save: every `8` optimizer steps
- Filters: exact-copy rejection, source-copy ratio <= `0.60`, Spanish leakage penalty <= `0.25`, no chat artifacts.

Held-out clean Chanka eval:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-48` | 25.8330 | 41.0508 | 6.1758 | 25.4893 | 2.6160 | 0.6329 | 89.6313 |
| `checkpoint-56` | 25.9035 | 40.9072 | 6.1663 | 26.1799 | 2.6477 | 0.6329 | 90.0922 |
| `checkpoint-64` | 25.7463 | 40.7202 | 5.9496 | 25.8876 | 2.6477 | 0.6329 | 89.8618 |
| `final_lora` | 25.7463 | 40.7202 | 5.9496 | 25.8876 | 2.6477 | 0.6329 | 89.8618 |

Conclusion:

- This exact small MBR self-training canary is a negative result.
- It preserves chrF++ reasonably well and recovers token F1 somewhat at step 56, but BLEU drops sharply versus the standing best.
- Do not scale this exact recipe. If revisiting MBR self-training, use a stronger pseudo-label mix:
  - larger K16 conservative MBR pool,
  - confidence/consensus thresholds,
  - clean-reference anchors mixed into training,
  - lower LR or fewer steps,
  - or train a selector/verifier from MBR/oracle features instead of directly imitating MBR pseudo-labels.

## 2026-05-22 Clean-Anchor + MBR K8 Canary

Purpose: test the main flaw in the pseudo-only run by mixing clean Chanka reference anchors with MBR pseudo-labels.

Code added:

- `scripts/build_mixed_sft_jsonl.py`: builds one JSONL file with a common `target` field from clean Chanka train anchors plus pseudo-label JSONL files.
- `tests/test_build_mixed_sft_jsonl.py`: covers pseudo target normalization, clean-anchor precedence over duplicate pseudo rows, and JSONL writing.

Mixed data:

- Pseudo source: `outputs/mbr_self_training_data/20260522-k8-train512/mbr_train_mbr_predictions.jsonl`
- Mixed output: `outputs/mbr_self_training_data/20260522-k8-train512/mixed_clean_anchors_mbr_target.jsonl`
- Builder rows: 1,003 total
  - 510 clean anchors
  - 493 MBR pseudo-labels
- Trainer rows after copy/leakage/artifact filters: 998
  - 849 train
  - 149 validation

SFT setup:

- Start adapter: `outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora`
- Output: `outputs/mbr_self_training_sft/20260522-k8-train512-mixed-clean512-from-current-best-lr1e-6-64steps`
- LR: `1e-6`
- Max steps: `64`
- Batch: per-device `4`, gradient accumulation `2`
- Validation/save: every `8` optimizer steps

Held-out clean Chanka eval:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-48` | 25.9369 | 40.5448 | 8.1424 | 25.7731 | 2.8270 | 0.6329 | 89.1705 |
| `checkpoint-56` | 25.6320 | 40.4837 | 5.8668 | 25.8258 | 2.6688 | 0.6329 | 89.1705 |
| `checkpoint-64` | 25.0994 | 39.8556 | 5.6166 | 25.4461 | 2.7215 | 0.6329 | 90.3226 |
| `final_lora` | 25.6320 | 40.4837 | 5.8668 | 25.8258 | 2.6688 | 0.6329 | 89.1705 |

Terminology-prompt eval on best mixed checkpoint:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-48` + terminology top-1 | 26.2287 | 40.6787 | 8.6205 | 26.1684 | 2.8270 | 0.4747 | 88.2488 |

Conclusion:

- Clean anchors fixed the severe BLEU collapse seen in pseudo-only self-training, but still did not beat the standing best overall.
- The terminology-prompt wrapper on mixed `checkpoint-48` gives the highest BLEU seen in these triage runs (`8.6205`), but selection and chrF++ remain below the previous terminology-prompt best.
- Do not scale this exact K8 mixed recipe. The next MBR/self-training variant should use higher-confidence pseudo-labels, likely from the K16 conservative pool or a consensus-thresholded builder, before any longer SFT continuation.

## 2026-05-22 K16 Confident MBR + Clean Anchors

Purpose: test the next MBR/self-training variant from the K8 mixed result: use a K16 conservative candidate pool and train only on high-confidence MBR pseudo-labels.

Code added:

- `scripts/build_confident_mbr_pseudo_labels.py`: selects MBR candidates only when they pass score-margin, peer-utility, copy, leakage, exact-copy, and artifact filters. It writes `target` plus diagnostics so it can feed the JSONL SFT trainer.
- `tests/test_build_confident_mbr_pseudo_labels.py`: covers ranking, filtering, and emitted diagnostic fields.

K16 train candidate pool:

- Candidate generation: `outputs/verifier_candidate_mining/20260522-train-k16-t065-p090/train_k16_predictions.jsonl`
- Source adapter: `outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora`
- Split: `train`, max train samples `512`
- Sampling: `num_return_sequences=16`, temperature `0.65`, top-p `0.90`, top-k `50`
- Full sampled candidate metrics across all 8,192 candidates: chrF++ `40.2802`, BLEU `7.8637`, token F1 `24.6964`
- Unfiltered K16 MBR selection over 510 groups:
  - selection `27.6782`
  - chrF++ `42.5243`
  - BLEU `9.0977`
  - token F1 `26.1104`
  - source-copy `2.3506%`
  - leakage `0.1471%`
  - TER `88.1036`
- Oracle over the same K16 train pool:
  - selection `36.3301`
  - chrF++ `51.4576`
  - BLEU `15.9508`
  - token F1 `37.1061`

Confidence sweep:

All rows used `min_candidates=8`, `min_mean_peer_utility=0.20`, max source-copy ratio `0.60`, max Spanish leakage penalty `0.25`, and no chat artifacts.

| Min MBR margin | Kept rows | Kept rate | Selection | chrF++ | BLEU | token F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 483 | 94.7% | 28.4200 | 43.0887 | 9.1948 | 27.3629 |
| 0.005 | 225 | 44.1% | 27.1629 | 41.6430 | 9.9757 | 25.2411 |
| 0.010 | 171 | 33.5% | 27.5078 | 41.9927 | 11.3886 | 25.2158 |
| 0.020 | 115 | 22.5% | 28.4099 | 43.2937 | 13.4162 | 25.5096 |
| 0.030 | 82 | 16.1% | 27.6520 | 42.3816 | 13.2851 | 24.1641 |

SFT setup:

- Pseudo-labels used: margin `0.020`
- Mixed data: `outputs/mbr_self_training_data/20260522-k16-train512-t065-p090/mixed_clean512_confident_margin002_target.jsonl`
- Builder rows: 621 total
  - 510 clean anchors
  - 111 confident MBR pseudo-labels after dedupe
- Trainer rows after filters: 618 total
  - 526 train
  - 92 validation
- Output: `outputs/mbr_self_training_sft/20260522-k16-confident-margin002-clean512-lr8e-7-48steps`
- LR: `8e-7`
- Max steps: `48`
- Batch: per-device `4`, gradient accumulation `2`
- Validation/save: every `8` optimizer steps

Held-out clean Chanka eval:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-32` | 25.9900 | 40.9140 | 6.3575 | 26.1317 | 2.6688 | 0.4747 | 89.1705 |
| `checkpoint-40` | 26.3578 | 41.0006 | 8.5802 | 26.0488 | 2.8270 | 0.4747 | 88.9401 |
| `checkpoint-48` | 25.6825 | 40.3648 | 6.1680 | 25.9207 | 2.6688 | 0.4747 | 89.4009 |
| `final_lora` | 25.6825 | 40.3648 | 6.1680 | 25.9207 | 2.6688 | 0.4747 | 89.4009 |

Terminology-prompt eval on best confident checkpoint:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-40` + terminology top-1 | 26.4760 | 41.0961 | 8.9872 | 25.9870 | 2.8270 | 0.4747 | 88.4793 |

Conclusion:

- This is the first self-training variant that edges past the previous deployable best when paired with terminology prompting.
- This became the deployable best at the time, but is superseded by the full-train terminology-MBR follow-up below.
- Superseded adapter: `checkpoint-40` from `20260522-k16-confident-margin002-clean512-lr8e-7-48steps` with terminology top-1.
- It improves selection slightly over the previous terminology-prompt best (`26.4760` vs `26.4615`) and improves BLEU more clearly (`8.9872` vs `8.6118`), while chrF++ is also higher (`41.0961` vs `41.0248`).
- The final checkpoint regresses; use `checkpoint-40`, not `final_lora`.
- Next step should expand the high-confidence K16 pool beyond 512 train rows or try margin `0.000`/`0.020` as separate mixtures. Margin `0.020` is high-BLEU but only 111 pseudo-labels after dedupe; margin `0.000` has better hidden-reference selection/F1 and far more rows.

### Margin 0.000 Follow-Up

Purpose: test whether the denser K16 confident set works better than the high-margin `0.020` subset. The margin `0.000` subset had the best hidden-reference selection/F1 in the confidence sweep and many more rows, but lower BLEU than margin `0.020`.

SFT setup:

- Pseudo-labels used: margin `0.000`
- Mixed data: `outputs/mbr_self_training_data/20260522-k16-train512-t065-p090/mixed_clean512_confident_margin000_target.jsonl`
- Builder rows: 971 total
  - 510 clean anchors
  - 461 confident MBR pseudo-labels after dedupe
- Trainer rows after filters: 968 total
  - 823 train
  - 145 validation
- Output: `outputs/mbr_self_training_sft/20260522-k16-confident-margin000-clean512-lr8e-7-56steps`
- LR: `8e-7`
- Max steps: `56`
- Batch: per-device `4`, gradient accumulation `2`
- Validation/save: every `8` optimizer steps

Held-out clean Chanka eval:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-40` | 25.5853 | 40.4324 | 6.2136 | 25.8660 | 2.6688 | 0.7911 | 89.6313 |
| `checkpoint-48` | 25.4862 | 40.1338 | 6.0282 | 25.8358 | 2.6688 | 0.7911 | 89.4009 |
| `checkpoint-56` | 25.6758 | 40.2917 | 8.5100 | 25.1140 | 2.8270 | 0.7911 | 89.1705 |
| `final_lora` | 25.6758 | 40.2917 | 8.5100 | 25.1140 | 2.8270 | 0.7911 | 89.1705 |

Conclusion:

- Margin `0.000` is worse than margin `0.020` despite much stronger hidden-reference training-set selection/F1.
- The denser pseudo-label mix appears to increase Spanish leakage and hurts held-out selection, chrF++, and token F1.
- Do not use this run as the deployable checkpoint. At the time, `20260522-k16-confident-margin002-clean512-lr8e-7-48steps/checkpoint-40` with terminology top-1 remained best, but it is now superseded by the full-train terminology-MBR follow-up below.
- The next high-upside self-training run should expand the K16 high-margin pool over more train rows, not lower the confidence threshold on the same 512-row pool.

### Full-Train Terminology-MBR Follow-Up

Purpose: expand the K16 MBR pseudo-label pool over the full training split, using the previous deployable best adapter plus terminology prompting during candidate generation.

Candidate generation:

- Adapter: `outputs/mbr_self_training_sft/20260522-k16-confident-margin002-clean512-lr8e-7-48steps/checkpoint-40`
- Terminology prompting: `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`, top-k `1`
- Split: full train split, 897 rows
- Sampling: K16, temperature `0.65`, top-p `0.90`, top-k `50`
- Candidate pool: `outputs/verifier_candidate_mining/20260522-train-k16-full-currentbest-term-t065-p090/train_k16_predictions.jsonl`

Reference-aware pool diagnostics:

| Selector/filter | Rows | Selection | chrF++ | BLEU | token F1 | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unfiltered MBR | 897 | 27.2143 | 41.7980 | 8.9381 | 25.4103 | 0.0988 | 89.2220 |
| Oracle | 897 | 36.2968 | 51.0586 | 17.2767 | 36.8458 | - | - |
| Filter-only margin `0.000` | 836 | 28.0242 | 42.4862 | 9.0517 | 26.6908 | 0.1047 | - |
| Margin `0.010` | 322 | 24.6006 | 39.0731 | 3.3570 | 23.7600 | - | - |
| Margin `0.020` | 208 | 25.1325 | 39.6352 | 3.2470 | 24.7401 | - | - |
| Margin `0.030` | 133 | 25.4397 | 40.1825 | 3.3205 | 25.9127 | - | - |

Observation: high-margin filtering behaved badly on this terminology-mined pool. Unlike the earlier non-terminology 512-row pool, the margin threshold mostly selected low-BLEU rows. The only viable follow-up was the filter-only margin `0.000` mixture.

SFT setup:

- Pseudo-labels used: full-train terminology MBR, filter-only margin `0.000`
- Mixed data: `outputs/mbr_self_training_data/20260522-k16-full-currentbest-term-t065-p090/mixed_clean512_confident_margin000_target.jsonl`
- Builder rows: 1296 total
  - 510 clean anchors
  - 786 pseudo-labels after dedupe
- Trainer rows after filters: 1293 total
  - 1100 train
  - 193 validation
- Output: `outputs/mbr_self_training_sft/20260522-k16-fullterm-margin000-clean512-lr4e-7-32steps`
- Starting adapter: `outputs/mbr_self_training_sft/20260522-k16-confident-margin002-clean512-lr8e-7-48steps/checkpoint-40`
- LR: `4e-7`
- Max steps: `32`
- Batch: per-device `4`, gradient accumulation `2`
- Validation/save: every `8` optimizer steps

Held-out clean Chanka eval:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-16` | 25.8105 | 40.5802 | 6.3858 | 25.9609 | 2.6688 | 0.4747 | 89.1705 |
| `checkpoint-24` | 25.9777 | 40.7881 | 6.5035 | 26.1724 | 2.6688 | 0.4747 | 88.9401 |
| `checkpoint-32` | 26.4340 | 40.8884 | 9.0799 | 26.2899 | 2.8270 | 0.4747 | 88.4793 |
| `final_lora` | 25.9777 | 40.7881 | 6.5035 | 26.1724 | 2.6688 | 0.4747 | 88.9401 |

Terminology-prompt eval on the close checkpoint:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-32` + terminology top-1 | 26.5515 | 40.8942 | 9.4839 | 26.3462 | 2.8270 | 0.3165 | 88.0184 |

Conclusion:

- This became the deployable best at the time, but is superseded by the non-terminology candidate-pool follow-up below.
- Superseded adapter: `outputs/mbr_self_training_sft/20260522-k16-fullterm-margin000-clean512-lr4e-7-32steps/checkpoint-32` with terminology top-1 inference.
- It beats the previous deployable best (`26.5515` vs `26.4760` selection) mainly through BLEU (`9.4839` vs `8.9872`) and lower TER (`88.0184` vs `88.4793`), while chrF++ is slightly lower than the previous checkpoint-40 terminology eval (`40.8942` vs `41.0961`).
- Do not use `final_lora`; it reverted to the checkpoint-24 behavior.
- The broader terminology-mined pseudo-label pool is useful only with a very low continuation LR and early checkpoint selection. More steps or stricter MBR-margin filtering are not justified by this result.

### Full-Train New-Best Non-Terminology MBR Follow-Up

Purpose: test whether the improved checkpoint from the full-train terminology-MBR run can generate a better non-terminology K16 pseudo-label pool. This isolates the policy distribution from terminology-prompt candidate generation, because high-margin filtering behaved badly on the terminology-mined pool.

Candidate generation:

- Adapter: `outputs/mbr_self_training_sft/20260522-k16-fullterm-margin000-clean512-lr4e-7-32steps/checkpoint-32`
- Terminology prompting: none
- Split: full train split, 897 rows
- Sampling: K16, temperature `0.65`, top-p `0.90`, top-k `50`
- Candidate pool: `outputs/verifier_candidate_mining/20260522-train-k16-full-newbest-noterm-t065-p090/train_k16_predictions.jsonl`

Reference-aware pool diagnostics:

| Selector/filter | Rows/groups | Selection | chrF++ | BLEU | token F1 | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All candidates | 14,352 candidates | - | 39.8951 | 8.9216 | 23.7855 | 0.2375 | 90.5610 |
| First candidate | 886 groups | 25.6169 | 39.8136 | 8.0786 | 24.2197 | 0.2822 | 89.4652 |
| Unfiltered MBR | 886 groups | 27.0672 | 41.9230 | 8.2642 | 25.0538 | 0.1411 | 89.5057 |
| Oracle | 886 groups | 36.1985 | 51.1360 | 17.1928 | 36.4736 | 0.0282 | 70.7861 |
| Filter-only margin `0.000` | 867 kept | 27.3825 | 41.9044 | 8.1874 | 24.9884 | 0.0000 | 89.5436 |
| Margin `0.005` | 408 kept | 23.0017 | 37.4351 | 3.6855 | 19.3891 | 0.0000 | 97.7835 |
| Margin `0.010` | 323 kept | 23.3561 | 37.9059 | 4.3072 | 20.1528 | 0.0000 | 98.4530 |
| Margin `0.020` | 212 kept | 23.0346 | 37.8563 | 4.8105 | 19.0160 | 0.0000 | 99.3443 |
| Margin `0.030` | 123 kept | 23.2715 | 38.6491 | 4.8563 | 18.9378 | 0.0000 | 101.2012 |

Observation: high MBR-score margin again selected bad rows. The only useful subset was filter-only margin `0.000`, which removed copy/leakage/artifact failures while keeping most groups.

SFT setup:

- Pseudo-labels used: full-train new-best non-terminology MBR, filter-only margin `0.000`
- Mixed data: `outputs/mbr_self_training_data/20260522-k16-full-newbest-noterm-t065-p090/mixed_clean512_confident_margin000_target.jsonl`
- Builder rows: 1332 total
  - 510 clean anchors
  - 822 pseudo-labels after dedupe
- Trainer rows after filters: 1332 total
  - 1133 train
  - 199 validation
- Output: `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-lr3e-7-32steps`
- Starting adapter: `outputs/mbr_self_training_sft/20260522-k16-fullterm-margin000-clean512-lr4e-7-32steps/checkpoint-32`
- LR: `3e-7`
- Max steps: `32`
- Batch: per-device `4`, gradient accumulation `2`
- Validation/save: every `8` optimizer steps

Held-out clean Chanka eval:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-16` | 25.5624 | 40.4161 | 5.9737 | 25.5817 | 2.6688 | 0.4747 | 89.6313 |
| `checkpoint-24` | 26.6394 | 41.1437 | 9.1801 | 26.5728 | 2.8270 | 0.4747 | 88.0184 |
| `checkpoint-32` | 26.3009 | 41.1277 | 6.8091 | 26.4447 | 2.6688 | 0.4747 | 87.7880 |
| `final_lora` | 26.3009 | 41.1277 | 6.8091 | 26.4447 | 2.6688 | 0.4747 | 87.7880 |

Terminology-prompt eval:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-24` + terminology top-1 | 26.8011 | 41.3017 | 9.5399 | 26.6291 | 2.8270 | 0.4747 | 87.5576 |
| `checkpoint-32` + terminology top-1 | 26.3438 | 41.1735 | 7.1561 | 26.2071 | 2.6688 | 0.4747 | 87.3272 |

Conclusion:

- This is the new deployable best as of 2026-05-22.
- Use `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-lr3e-7-32steps/checkpoint-24` with terminology top-1 inference.
- It improves over the previous deployable best on selection (`26.8011` vs `26.5515`), chrF++ (`41.3017` vs `40.8942`), BLEU (`9.5399` vs `9.4839`), token F1 (`26.6291` vs `26.3462`), and TER (`87.5576` vs `88.0184`).
- Do not use `checkpoint-32` or `final_lora`; both lose the BLEU gain from checkpoint-24.
- Future MBR self-training should stop treating MBR margin as confidence on these pools. The better rule so far is broad filter-only MBR plus low-LR continuation and early checkpoint selection.

### Terminology-Aware JSONL SFT Follow-Up

Purpose: train with the same glossary-conditioned prompt shape used at deployment. `scripts/train_jsonl_sft_unsloth.py` now accepts `--terminology-file`, `--terminology-top-k`, and `--terminology-min-source-chars`, reusing the same glossary loader and source-term matcher as GSPO/evaluation.

SFT setup:

- Starting adapter: `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-lr3e-7-32steps/checkpoint-24`
- Mixed data: `outputs/mbr_self_training_data/20260522-k16-full-newbest-noterm-t065-p090/mixed_clean512_confident_margin000_target.jsonl`
- Terminology file: `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`
- Terminology top-k: `1`
- Loaded rows after filtering: 1332
  - 1133 train
  - 199 validation
- Terminology-matched rows:
  - 143 train
  - 25 validation
- Output: `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps`
- LR: `2e-7`
- Max steps: `24`
- Batch: per-device `4`, gradient accumulation `2`
- Validation/save: every `8` optimizer steps

Held-out clean Chanka terminology-prompt eval:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-8` + terminology top-1 | 27.0396 | 41.4823 | 9.7158 | 27.0736 | 2.8270 | 0.4747 | 87.3272 |
| `checkpoint-16` + terminology top-1 | 26.1284 | 40.9647 | 6.6537 | 26.2297 | 2.6688 | 0.4747 | 88.9401 |
| `checkpoint-24` + terminology top-1 | 26.6986 | 41.2255 | 9.4832 | 26.4181 | 2.8270 | 0.4747 | 87.7880 |
| `final_lora` + terminology top-1 | 26.6986 | 41.2255 | 9.4832 | 26.4181 | 2.8270 | 0.4747 | 87.7880 |

Raw no-terminology eval for the best checkpoint:

| Adapter | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint-8` | 26.5490 | 41.1692 | 9.0400 | 26.3091 | 2.8270 | 0.4747 | 88.7097 |

Conclusion:

- This is the new deployable best as of 2026-05-22.
- Use `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8` with terminology top-1 inference.
- It improves over the previous deployable best on selection (`27.0396` vs `26.8011`), chrF++ (`41.4823` vs `41.3017`), BLEU (`9.7158` vs `9.5399`), token F1 (`27.0736` vs `26.6291`), and TER (`87.3272` vs `87.5576`).
- The gain is from the combination of terminology-aware SFT and terminology-aware inference. The same checkpoint without terminology prompting is not better than the previous deployable wrapper.
- As before, early checkpoint selection matters: checkpoint-8 is best, while checkpoint-16 regresses and checkpoint-24/final are below checkpoint-8.
