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

