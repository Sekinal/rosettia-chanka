# TranslateGemma Takeaways

Source: `/home/ieqr/Downloads/2601.09012v3.pdf`, reviewed on 2026-05-22.

## What The Paper Does

TranslateGemma is a Gemma 3 translation-specialized family trained with:

- SFT on a large mixture of human and Gemini-generated parallel data.
- Synthetic source selection: compare greedy vs sampled Gemini outputs with MetricX-QE, then select sources where sampling helps most.
- 128-sample generation followed by MetricX-QE filtering to keep the best synthetic translations.
- Two generated lengths: sentence-level and blobs up to 512 tokens.
- A formatting filter to remove malformed translations.
- 30% generic instruction-following data in SFT to reduce translation-task overfit.
- RL on the SFT model with an ensemble of reward signals:
  - MetricX-QE, no reference.
  - AutoMQM-QE, no reference, with token/span-level error signals.
  - chrF, with reference, scaled by 2.
  - Naturalness autorater.
  - Generalist reward model.
- Token-level reward/advantage signals added to sequence-level rewards for better credit assignment.
- A stable translation prompt used consistently for synthetic generation, SFT, evaluation, and inference.

## What Transfers To RosettIA-Chanka

Useful immediately:

- Multi-sample generation plus QE/reranking. Our K16/K32 candidate pools are the local version of their 128-sample QE decoding.
- Source selection by “sampling improves over greedy.” For us, rows with a large oracle-over-first or oracle-over-MBR gap should drive synthetic data, verifier labels, and manual review.
- Prompt consistency. Our terminology-prompt SFT and terminology-prompt inference helped; future data generation should use the same prompt shape intended for deployment.
- Filtering generated data for formatting/copy/leakage before SFT.
- Mixing some broad/general data during SFT to avoid overfitting the tiny Chanka manual.

Useful later:

- Train a stronger QE/verifier from real candidate pools, ideally with span/error-style labels rather than scalar JSON scores only.
- Add token-level or span-level reward shaping for known errors: Spanish-root leakage, missing negation, number errors, wrong legal/family terminology, and unsupported additions.
- Generate both short sentence examples and longer context blobs once we have enough reliable sources.

## Current Local Experiment Inspired By The Paper

The direct adaptation currently under test is mixed candidate decoding:

- Conservative pool: K16, temperature `0.65`, top-p `0.90`, terminology top-1.
- Exploratory pool: K16, temperature `0.95`, top-p `0.95`, terminology top-1.
- Merge by source/reference, dedupe normalized candidates, then evaluate MBR, feature reranking, and oracle.

Held-out result:

- Mixed eval pool: `outputs/rerank_candidate_evals/20260522-current-deployable-mixed-k32-term/candidates_predictions.jsonl`
- Mean candidates/group after dedupe: `22.1582`.
- Oracle: selection `41.3016`, chrF++ `55.0908`, BLEU `22.6396`, token F1 `45.0686`, TER `67.2811`.
- Existing K16-conservative feature weights do **not** transfer to this mixed pool: selection `27.4724`, chrF++ `41.7903`, BLEU `6.1268`.
- MBR on the mixed pool is better than old feature weights but still below current best: selection `27.6564`, chrF++ `42.3236`, BLEU `7.1796`.
- Matching mixed-pool feature refit also does not beat current best: selection `28.3403`, chrF++ `42.0763`, BLEU `9.3382`, TER `85.2535`.

Decision so far:

- The mixed candidate pool has much higher oracle headroom, but needs matching train-pool reranker weights or a stronger selector.
- Do not deploy mixed K32 with the old K16 feature weights.
- If matching mixed-pool training cannot harvest the oracle gain, the next paper-aligned step is training a QE/listwise selector from rows where sampling improves strongly over greedy.

## Frontier-Style Merge Experiments To Try

User-requested experimental direction: try mergekit/frankenstein merging after the current mixed-pool reranker work.

Recommended order:

1. LoRA adapter averaging among compatible Qwen3.5-2B checkpoints:
   - current best terminology-aware JSONL SFT checkpoint-8,
   - high-BLEU low-LR refinement checkpoint,
   - learned-verifier-vibe canary checkpoint,
   - maybe the DPO checkpoint with better TER.
2. If LoRA averaging helps or at least gives diverse behavior, merge candidate adapters into full Qwen3.5-2B models and test `mergekit` linear/task-arithmetic/TIES-style merges.
3. Only then try `mergekit-moe` or a frankenstein/MoE setup. Treat this as exploratory: it may produce candidate diversity even if greedy metrics do not improve.

Evaluation requirements:

- Verify EOS/chat-template behavior after every merge. Earlier saved-adapter reload failures were caused by EOS mismatch, so this is a real risk.
- Evaluate greedy, K16 feature-reranked, and mixed-pool oracle/feature results. A merge that is worse greedily may still be useful as an additional candidate generator.
- Do not commit merged model weights or generated outputs.

## 2026-05-22 LoRA Soup Probe

Utility added:

- `scripts/merge_lora_adapters.py`: averages compatible PEFT LoRA adapter tensors and writes a new adapter directory. This is a lightweight precursor to full `mergekit` model merges.

First soup:

- Output: `outputs/lora_soups/20260522-termtrain-soup-8-16-24`
- Adapters:
  - `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8`
  - `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-16`
  - `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-24`
- Weights: `0.50/0.30/0.20`

Greedy terminology top-1 eval:

- chrF++ `41.0070`, BLEU `9.1150`, token F1 `26.4196`, TER `88.9401`
- This is below the underlying checkpoint-8 eval, so the soup should not replace the current adapter.

Candidate-generator eval:

- Soup K8 sampled pool: `outputs/rerank_candidate_evals/20260522-termtrain-soup-8-k8-t065-p090-term/candidates_predictions.jsonl`
- Current K16 + soup K8 merged pool: `outputs/rerank_candidate_evals/20260522-current-k16-plus-soup-k8-term/candidates_predictions.jsonl`
- Merged pool stats after dedupe: `158` groups, `2,160` candidates, mean `13.6709` candidates/group.

| Method | Selection | chrF++ | BLEU | token F1 | TER |
| --- | ---: | ---: | ---: | ---: | ---: |
| current K16 feature best | 28.5647 | 42.9352 | 11.0872 | 27.7860 | 84.1014 |
| current K16 oracle | 38.3462 | 52.6971 | 19.7691 | 41.8353 | 69.8157 |
| current K16 + soup K8, current feature weights | 27.9722 | 42.4683 | 8.8622 | 27.8317 | 87.3272 |
| current K16 + soup K8 oracle | 39.5617 | 54.0888 | 20.8652 | 43.1052 | 67.9724 |

Takeaway:

- The LoRA soup is worse greedily but adds useful diversity: oracle selection improves by `+1.2155`, chrF++ by `+1.3916`, BLEU by `+1.0961`, and TER by `-1.8433` over current K16 oracle.
- The deployable feature selector cannot harvest that gain yet. This supports the frontier-style merge direction as candidate generation diversity, while reinforcing that the real bottleneck is selection/QE.
