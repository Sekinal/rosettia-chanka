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
- New deployable best for now: `checkpoint-40` from `20260522-k16-confident-margin002-clean512-lr8e-7-48steps` with terminology top-1.
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
- Do not use this run as the deployable checkpoint. Keep `20260522-k16-confident-margin002-clean512-lr8e-7-48steps/checkpoint-40` with terminology top-1 as the current best.
- The next high-upside self-training run should expand the K16 high-margin pool over more train rows, not lower the confidence threshold on the same 512-row pool.
