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
