# Candidate Reranking Findings

## 2026-05-22 Feature Reranker

Purpose: test a deployable reference-free reranker for sampled candidate pools. Hidden references are used only to fit feature weights on training candidate pools; inference uses the Spanish source, candidate translations, MBR consensus features, and automatic safety features.

Code added:

- `scripts/train_feature_candidate_reranker.py`: trains/evaluates a lightweight linear feature reranker from candidate JSONL pools.
- `tests/test_train_feature_candidate_reranker.py`: covers feature extraction, feature-based selection, and objective improvement on a toy group.

Features:

- MBR consensus score/rank/gap.
- Duplicate rate and length consensus.
- Source-copy, exact-copy, Spanish leakage, chat artifact, and repetition penalties.
- Candidate index and target/source length features.

First evaluation, using weights trained on an older train pool and evaluated on the older K16 held-out pool:

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 22.4047 | 36.9895 | 7.7949 | n/a | n/a | n/a | n/a |
| feature | 25.6484 | 40.3494 | 6.0580 | 24.1054 | 1.7300 | 0.0000 | 88.9401 |
| MBR | 25.8597 | 40.8152 | 8.1692 | 24.1345 | 1.9673 | 0.1582 | 93.5484 |
| oracle | 36.7495 | 50.8698 | 18.9527 | 39.5376 | n/a | 0.0000 | n/a |

Conclusion from the older pool: features were better than first-candidate sampling but worse than MBR.

Fresh current-deployable K16 terminology-prompt pool:

- Adapter: `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8`
- Candidate generation: K16, temperature `0.65`, top-p `0.90`, top-k `50`, terminology top-1 prompt.
- Candidate file: `outputs/rerank_candidate_evals/20260522-current-deployable-k16-t065-p090-term/candidates_predictions.jsonl`

Reference-free MBR on this fresh pool:

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 25.1206 | 39.9330 | 8.1285 | 23.9645 | 2.5352 | 0.2215 | 93.7788 |
| MBR | 27.0751 | 41.7348 | 8.1092 | 26.8743 | 2.1414 | 0.3165 | 89.6313 |
| oracle | 38.3462 | 52.6971 | 19.7691 | 41.8353 | 2.6899 | 0.0000 | 69.8157 |

This MBR run slightly improves selection and chrF++ over the previous deployable best, but BLEU drops from `9.7158` to `8.1092`.

Feature weights from the older train pool evaluated on the fresh current-deployable K16 terminology-prompt pool:

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| feature | 28.0240 | 42.2514 | 9.4148 | 28.0195 | 1.9831 | 0.0000 | 85.4839 |
| MBR | 27.0751 | 41.7348 | 8.1092 | 26.8743 | 2.1414 | 0.3165 | 89.6313 |
| oracle | 38.3462 | 52.6971 | 19.7691 | 41.8353 | 2.6899 | 0.0000 | 69.8157 |

Matching current-deployable train-pool refit:

- Train candidate file: `outputs/verifier_candidate_mining/20260522-train-k16-current-deployable-term-t065-p090/train_k16_predictions.jsonl`
- Train candidates: `14,352`
- Train groups after candidate grouping: `886`
- Search iterations: `2,500`
- Accepted updates: `28`
- Eval candidate file: `outputs/rerank_candidate_evals/20260522-current-deployable-k16-t065-p090-term/candidates_predictions.jsonl`
- Weights path on remote: `outputs/feature_candidate_reranker_evals/20260522-feature-reranker-train-current-deployable-term-eval-current-deployable-k16/feature_k16_current_term_weights.json`

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER | non-first % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 25.1206 | 39.9330 | 8.1285 | 23.9645 | 2.5352 | 0.2215 | 93.7788 | 0.0000 |
| feature | 28.5647 | 42.9352 | 11.0872 | 27.7860 | 2.0359 | 0.0000 | 84.1014 | 98.1013 |
| MBR | 27.0751 | 41.7348 | 8.1092 | 26.8743 | 2.1414 | 0.3165 | 89.6313 | 86.7089 |
| oracle | 38.3462 | 52.6971 | 19.7691 | 41.8353 | 2.6899 | 0.0000 | 69.8157 | 86.7089 |

Decision:

- The matching-pool feature refit is now the current best deployable selector by all headline metrics: selection, chrF++, BLEU, token F1, leakage, and TER.
- It also beats the previous greedy/terminology best BLEU (`11.0872` vs `9.7158`), unlike the older-weight feature run.
- The oracle gap remains large. The model distribution contains far better translations; selector quality is now a high-upside path.

Next checks:

- Add glossary-root leakage features such as `fiesta -> raymi` versus `fiestapi`, because current Spanish-leakage metrics miss Spanish roots with Quechua suffixes.
- Consider a listwise/discriminative reranker after the feature baseline is stable.
