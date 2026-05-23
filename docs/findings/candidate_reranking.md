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

## 2026-05-22 Source-Root And Glossary Feature Ablations

Purpose: test whether the feature reranker can directly penalize Spanish roots with Quechua suffixes, e.g. `En la fiesta -> Fiestapi`, and reward coverage of matched Chanka glossary targets.

Code update:

- `source_root_copy_ratio` detects source content-token prefix copying such as `fiesta -> Fiestapim`.
- `terminology_target_coverage` and `terminology_source_root_leakage` can use the simple glossary to score whether matched source terms are translated with the expected Chanka term.
- These features are opt-in via `--include-source-root-copy` and `--include-terminology-features`. The default feature set still reproduces the current best result.

Default reproduction after the opt-in refactor:

- Output: `outputs/feature_candidate_reranker_evals/20260522-feature-reranker-default-repro-after-feature-ablation`
- Feature result: selection `28.5647`, chrF++ `42.9352`, BLEU `11.0872`, token F1 `27.7860`, TER `84.1014`.

Source-root copy feature only:

- Output: `outputs/feature_candidate_reranker_evals/20260522-feature-reranker-source-root-copy-eval-current-deployable-k16`
- Feature result: selection `28.1603`, chrF++ `42.8616`, BLEU `9.7972`, token F1 `27.3957`, TER `85.7143`.
- This is still stronger than MBR but worse than the current default feature reranker.

Glossary terminology features:

- Output: `outputs/feature_candidate_reranker_evals/20260522-feature-reranker-train-current-deployable-term-features-eval-current-deployable-k16`
- Feature result: selection `27.8330`, chrF++ `42.4529`, BLEU `8.9455`, token F1 `26.9195`, TER `84.5622`.
- The fitted model learned a negative weight for `terminology_target_coverage`, which is a sign that the current glossary matches are too sparse or distribution-mismatched for this held-out pool.

Decision:

- Do not enable source-root or glossary features for the current deployable reranker.
- Keep the features in code as opt-in diagnostics for future richer terminology pools.
- The better fix for `Fiestapi` is likely data generation or candidate generation with a glossary that actually includes `fiesta -> raymi`; the K16 candidate pool for `En la fiesta` contained only `Fiestapi`/`Fiestapim` variants, so no selector could choose `Raymipim`.

## 2026-05-22 Reference-Free Inference Wrapper

Code added:

- `scripts/generate_feature_reranked_translations.py`: loads a LoRA adapter, generates K sampled candidates for arbitrary Spanish source strings, applies saved feature-reranker weights, and writes selected Chanka translations without references.
- `tests/test_generate_feature_reranked_translations.py`: covers source loading, candidate grouping, and feature-model selection without requiring a GPU.

Current best deployment command pattern:

```bash
.venv/bin/python scripts/generate_feature_reranked_translations.py \
  --adapter-path outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8 \
  --weights-json outputs/feature_candidate_reranker_evals/20260522-feature-reranker-train-current-deployable-term-eval-current-deployable-k16/feature_k16_current_term_weights.json \
  --input-path sources.txt \
  --output-jsonl predictions.jsonl \
  --candidates-jsonl candidates.jsonl \
  --num-return-sequences 16 \
  --temperature 0.65 \
  --top-p 0.90 \
  --top-k 50 \
  --terminology-file clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet \
  --terminology-top-k 1
```

Remote smoke:

- Output: `outputs/manual_inference_smokes/20260522-feature-reranked-smoke`
- Adapter and weights matched the current best, but K was reduced to `4` for speed.
- Inputs:
  - `Yo vivo en Quinua`
  - `Tengo 45 años`
- Selected outputs:
  - `Kunamantaqu kawsakuni Quinua llaqtapi`
  - `Tawa chunka pichqayoq watayuqmi kani`

Notes:

- The wrapper is deployment plumbing, not a new quality result.
- Use K16 for quality comparisons; K4 was only a smoke check.
- The first smoke row shows why we still need qualitative review: one of the unselected candidates, `Ñuqaqa Quinua llaqtapim kawsani`, looks better than the selected row. The held-out K16 metrics remain the authoritative aggregate evidence for now.

## 2026-05-22 Mixed Candidate Pool

Purpose: adapt the TranslateGemma multi-sample/QE idea by combining conservative and exploratory candidate pools before reranking.

Code added:

- `scripts/merge_candidate_prediction_pools.py`: merges multiple candidate JSONL files by source/reference/source_name/variant, dedupes normalized predictions, and reindexes candidates within each merged group.
- `tests/test_merge_candidate_prediction_pools.py`: covers dedupe, reindexing, and summary counts.

Held-out pools:

- Conservative: `outputs/rerank_candidate_evals/20260522-current-deployable-k16-t065-p090-term/candidates_predictions.jsonl`
- Exploratory: `outputs/rerank_candidate_evals/20260522-current-deployable-k16-t095-p095-term/candidates_predictions.jsonl`
- Merged: `outputs/rerank_candidate_evals/20260522-current-deployable-mixed-k32-term/candidates_predictions.jsonl`
- Merged pool stats after dedupe: `158` groups, `3,501` candidates, mean `22.1582` candidates/group.

Held-out mixed-pool results:

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 25.1206 | 39.9330 | 8.1285 | 23.9645 | 2.5352 | 0.2215 | 93.7788 |
| MBR | 27.6564 | 42.3236 | 7.1796 | 27.4644 | 1.6139 | 0.1582 | 91.7051 |
| old K16 feature weights | 27.4724 | 41.7903 | 6.1268 | 27.3172 | 1.1498 | 0.0000 | 88.2488 |
| mixed-pool feature refit | 28.3403 | 42.0763 | 9.3382 | 27.1946 | 0.8966 | 0.0000 | 85.2535 |
| oracle | 41.3016 | 55.0908 | 22.6396 | 45.0686 | 1.9409 | 0.0000 | 67.2811 |

Train mixed pool:

- Conservative train: `outputs/verifier_candidate_mining/20260522-train-k16-current-deployable-term-t065-p090/train_k16_predictions.jsonl`
- Exploratory train: `outputs/verifier_candidate_mining/20260522-train-k16-current-deployable-term-t095-p095/train_k16_predictions.jsonl`
- Merged train: `outputs/verifier_candidate_mining/20260522-train-mixed-k32-current-deployable-term/train_mixed_k32_predictions.jsonl`
- Merged train stats after dedupe: `886` groups, `19,485` candidates, mean `21.9921` candidates/group.
- Mixed feature weights: `outputs/feature_candidate_reranker_evals/20260522-feature-reranker-train-mixed-k32-eval-mixed-k32-term/feature_mixed_k32_weights.json`

Decision:

- Do not deploy mixed K32 with the current linear feature reranker. It underperforms the current K16 feature best (`28.5647` selection, chrF++ `42.9352`, BLEU `11.0872`).
- The mixed pool is still extremely valuable: oracle reaches chrF++ `55.0908` and BLEU `22.6396`, much closer to the project target.
- Next selector work should be listwise/QE-style, not another linear feature refit. Good labels are rows where exploratory sampling adds a better oracle candidate than conservative K16.

## 2026-05-22 Listwise Feature Reranker Ablation

Code added:

- `scripts/train_feature_candidate_reranker.py --training-objective listwise`
- `--listwise-target soft|best`, `--listwise-epochs`, `--listwise-learning-rate`, `--listwise-temperature`, `--listwise-l2`

Mixed K32 train/eval results:

| Objective | Output | Selection | chrF++ | BLEU | token F1 | TER |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| soft listwise | `outputs/feature_candidate_reranker_evals/20260522-listwise-soft-train-mixed-k32-eval-mixed-k32-term` | 27.3805 | 41.1811 | 9.0921 | 25.8811 | 82.9493 |
| hard-best listwise | `outputs/feature_candidate_reranker_evals/20260522-listwise-best-train-mixed-k32-eval-mixed-k32-term` | 26.8655 | 40.5992 | 8.1170 | 25.6349 | 81.5668 |

Decision:

- The listwise linear objective is not a deployable improvement. It improves TER relative to MBR, but it gives up too much chrF++/BLEU/selection score.
- The negative result points to feature expressiveness, not just optimization. A stronger selector likely needs candidate text modeling or richer learned QE/verifier features, not another objective over the same linear feature set.

## 2026-05-22 K32 Conservative Sampling

Purpose: test whether the current feature selector can improve by sampling more candidates from the same conservative distribution, without mixing in noisier exploratory/soup candidates.

Held-out K32 pool:

- Candidate file: `outputs/rerank_candidate_evals/20260522-current-deployable-k32-t065-p090-term/candidates_predictions.jsonl`
- Adapter: `outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8`
- Decoding: K32, temperature `0.65`, top-p `0.90`, top-k `50`, terminology top-1.

K16 feature weights transferred to the K32 pool:

- Output: `outputs/feature_candidate_reranker_evals/20260522-current-k32-term-current-k16-weights`
- Weights: `outputs/feature_candidate_reranker_evals/20260522-feature-reranker-train-current-deployable-term-eval-current-deployable-k16/feature_k16_current_term_weights.json`

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous K16 feature best | 28.5647 | 42.9352 | 11.0872 | 27.7860 | 2.0359 | 0.0000 | 84.1014 |
| K32 + transferred K16 feature weights | 28.6939 | 43.3398 | 9.2082 | 28.4464 | 2.0992 | 0.0000 | 82.9493 |
| K32 MBR | 27.1236 | 42.0596 | 7.0919 | 27.1913 | 1.9937 | 0.3165 | 89.4009 |
| K32 oracle | 40.1555 | 54.4572 | 22.0831 | 43.9097 | 2.5814 | 0.0000 | 66.8203 |

Matching K32 train-pool refit:

- Train pool: `outputs/verifier_candidate_mining/20260522-train-k32-current-deployable-term-t065-p090/train_k32_predictions.jsonl`
- Train stats: `897` source rows, `28,704` candidates.
- Output: `outputs/feature_candidate_reranker_evals/20260522-feature-reranker-train-k32-current-deployable-term-eval-k32`
- Feature result: selection `28.3664`, chrF++ `43.1674`, BLEU `7.3568`, token F1 `28.3078`, TER `84.3318`.

Decision:

- K32 with the existing K16 feature weights is the new best deployable setting for selection, chrF++, token F1, and TER.
- K16 remains the better BLEU setting (`11.0872` vs `9.2082`). Because the project wants much higher BLEU too, this is not a complete replacement; keep both profiles depending on metric priority.
- Matched K32 refitting over-selected late/high-index candidates and hurt BLEU, so keep using the K16-trained feature weights for K32 inference.
- The K32 oracle is higher than K16 oracle but lower than mixed K32 oracle, so same-distribution K32 is a safer deployable gain while mixed/exploratory pools remain a selector research target.

## 2026-05-22 Text-Aware Hashed Reranker

Purpose: add source/candidate text modeling to the selector without needing sklearn/xgboost. The model is deployable: references are used only to train pairwise oracle-winner preferences, while inference uses only source text, candidate text, and reference-free group features.

Code added:

- `scripts/train_text_candidate_reranker.py`: online pairwise logistic ranker over hashed sparse features.
- `scripts/generate_text_reranked_translations.py`: K-sampling deployment wrapper for the text-aware reranker.
- `tests/test_train_text_candidate_reranker.py`
- `tests/test_generate_text_reranked_translations.py`

Feature families:

- Existing numeric feature-reranker signals.
- Candidate word n-grams and character n-grams.
- Source-token to candidate-token cross features, e.g. a learnable `fiesta -> raymipim` style association.

Same-distribution K32 result:

- Train pool: `outputs/verifier_candidate_mining/20260522-train-k32-current-deployable-term-t065-p090/train_k32_predictions.jsonl`
- Eval pool: `outputs/rerank_candidate_evals/20260522-current-deployable-k32-t065-p090-term/candidates_predictions.jsonl`
- Model: `outputs/text_candidate_reranker_evals/20260522-text-ranker-train-k32-eval-k32-term/text_ranker_k32_model.json`

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous K16 feature best | 28.5647 | 42.9352 | 11.0872 | 27.7860 | 2.0359 | 0.0000 | 84.1014 |
| K32 numeric feature best | 28.6939 | 43.3398 | 9.2082 | 28.4464 | 2.0992 | 0.0000 | 82.9493 |
| K32 text-aware reranker | 28.7980 | 43.0524 | 11.9903 | 28.8624 | 3.1646 | 0.0000 | 82.9493 |
| K32 oracle | 40.1555 | 54.4572 | 22.0831 | 43.9097 | 2.5814 | 0.0000 | 66.8203 |

K16 text-ranker result:

- Output: `outputs/text_candidate_reranker_evals/20260522-text-ranker-train-k16-eval-k16-term`
- Selection `28.5400`, chrF++ `42.2006`, BLEU `11.2813`, token F1 `29.4821`, TER `83.1797`.
- Useful as a high-token-F1/BLEU-ish profile, but it does not beat K32 text overall.

Transfer checks:

- K32-trained text model on mixed K32: selection `27.7533`, chrF++ `41.1096`, BLEU `11.8041`, TER `84.5622`.
- K32-trained text model on current K16 + soup K8: selection `27.3980`, chrF++ `40.8253`, BLEU `12.2599`, TER `84.1014`.
- These transfer runs improve BLEU but lose too much chrF++/selection. The text model should be used on same-distribution conservative K32 candidates for now.

Deployment smoke:

- Output: `outputs/manual_inference_smokes/20260522-text-reranked-smoke`
- K4 smoke command used the K32-trained text model and current adapter.
- `Yo vivo en Quinua` -> `Quinua llaqtapim kawsani`
- `Tengo 45 años` -> `45 watayuqmi kani`

Decision:

- K32 text-aware reranking is the new best deployable overall profile because it improves selection, BLEU, token F1, and TER over the previous K16 feature best.
- K32 numeric feature reranking remains the chrF++ best (`43.3398` vs text `43.0524`).
- The remaining oracle gap is still large: K32 oracle reaches selection `40.1555`, chrF++ `54.4572`, BLEU `22.0831`. Text-aware selection is progress, not the finish line.

## 2026-05-22 Score Ensemble Reranker

Purpose: combine the strengths of the numeric feature selector and the text-aware selector instead of choosing between chrF++ and BLEU profiles.

Code added:

- `scripts/train_score_ensemble_reranker.py`: hill-climbs a reference-free score blend over normalized feature-score, text-score, MBR score/rank, length consensus, duplicate rate, and candidate index.
- `scripts/generate_ensemble_reranked_translations.py`: K-sampling deployment wrapper for the score ensemble.
- `tests/test_train_score_ensemble_reranker.py`
- `tests/test_generate_ensemble_reranked_translations.py`

Held-out K32 result:

- Feature model: `outputs/feature_candidate_reranker_evals/20260522-feature-reranker-train-current-deployable-term-eval-current-deployable-k16/feature_k16_current_term_weights.json`
- Text model: `outputs/text_candidate_reranker_evals/20260522-text-ranker-train-k32-eval-k32-term/text_ranker_k32_model.json`
- Ensemble: `outputs/score_ensemble_reranker_evals/20260522-ensemble-feature-k16-text-k32-eval-k32-term/ensemble_k32_ensemble.json`

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous K16 feature best | 28.5647 | 42.9352 | 11.0872 | 27.7860 | 2.0359 | 0.0000 | 84.1014 |
| K32 numeric feature best | 28.6939 | 43.3398 | 9.2082 | 28.4464 | 2.0992 | 0.0000 | 82.9493 |
| K32 text-aware reranker | 28.7980 | 43.0524 | 11.9903 | 28.8624 | 3.1646 | 0.0000 | 82.9493 |
| K32 score ensemble | 29.7584 | 43.8064 | 13.2505 | 30.4677 | 3.1118 | 0.0000 | 81.3364 |
| K32 oracle | 40.1555 | 54.4572 | 22.0831 | 43.9097 | 2.5814 | 0.0000 | 66.8203 |

Deployment smoke:

- Output: `outputs/manual_inference_smokes/20260522-ensemble-reranked-smoke`
- K4 smoke selected:
  - `Yo vivo en Quinua` -> `Quinua llaqtapim kawsani`
  - `Tengo 45 años` -> `Tawa chunka pichqayoq watam kani`

Decision:

- K32 score ensemble is the new best deployable profile by selection, chrF++, BLEU, token F1, and TER.
- It is still below K32 oracle by a wide margin, so the next selector work should keep moving toward richer QE/verifier signals or better candidate pools rather than more shallow numerical blends.

## 2026-05-22 Mixed-Distribution Text And Ensemble Selectors

Purpose: test whether the mixed/exploratory pool's larger oracle headroom can be harvested by training selectors on the matching mixed distribution rather than transferring conservative K32 selectors.

Mixed text-aware reranker:

- Train pool: `outputs/verifier_candidate_mining/20260522-train-mixed-k32-current-deployable-term/train_mixed_k32_predictions.jsonl`
- Eval pool: `outputs/rerank_candidate_evals/20260522-current-deployable-mixed-k32-term/candidates_predictions.jsonl`
- Model: `outputs/text_candidate_reranker_evals/20260522-text-ranker-train-mixed-k32-eval-mixed-k32-term/text_ranker_mixed_k32_model.json`
- Metrics: selection `29.8740`, chrF++ `43.2738`, BLEU `11.7439`, token F1 `30.9609`, source-copy `2.0570%`, leakage `0.0%`, TER `81.5668`.

Mixed score ensemble:

- Feature weights: `outputs/feature_candidate_reranker_evals/20260522-feature-reranker-train-mixed-k32-eval-mixed-k32-term/feature_mixed_k32_weights.json`
- Text model: `outputs/text_candidate_reranker_evals/20260522-text-ranker-train-mixed-k32-eval-mixed-k32-term/text_ranker_mixed_k32_model.json`
- Ensemble: `outputs/score_ensemble_reranker_evals/20260522-ensemble-feature-mixed-text-mixed-eval-mixed-k32-term/ensemble_mixed_k32_ensemble.json`
- Metrics: selection `29.3974`, chrF++ `42.4715`, BLEU `11.5927`, token F1 `30.4566`, source-copy `2.1624%`, leakage `0.0%`, TER `81.1060`.

Comparison:

| Selector | Pool | Selection | chrF++ | BLEU | token F1 | TER |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| conservative K32 score ensemble | conservative K32 | 29.7584 | 43.8064 | 13.2505 | 30.4677 | 81.3364 |
| mixed text-aware reranker | mixed K32 | 29.8740 | 43.2738 | 11.7439 | 30.9609 | 81.5668 |
| mixed score ensemble | mixed K32 | 29.3974 | 42.4715 | 11.5927 | 30.4566 | 81.1060 |
| mixed oracle | mixed K32 | 41.3016 | 55.0908 | 22.6396 | 45.0686 | 67.2811 |

Decision:

- Mixed text-aware reranking has the best selection and token F1 seen so far, but it gives up too much chrF++ and BLEU compared with the conservative K32 ensemble.
- Mixed score ensemble has the best TER seen so far, but also trails conservative K32 ensemble on selection/chrF++/BLEU.
- Keep conservative K32 score ensemble as the best overall deployable profile. Mixed-distribution training is promising but needs a stronger selector/QE model before using the higher-headroom mixed pool for deployment.

## 2026-05-22 Qwen3.5 4B Full-SFT Candidate Pool

Purpose: test whether the strong Qwen3.5 4B broad -> clean Chanka curriculum, followed by short full-SFT refinement, is useful as a candidate generator for the existing reranking stack.

Candidate pool:

- Current deployable K32 pool: `outputs/rerank_candidate_evals/20260522-current-deployable-k32-t065-p090-term/candidates_predictions.jsonl`
- 4B full-SFT K16 pool: `outputs/rerank_candidate_evals/20260522-qwen35-4b-full-chanka48-k16-t065-p090-term/candidates_predictions.jsonl`
- Merged eval pool: `outputs/rerank_candidate_evals/20260522-current-k32-plus-qwen35-4b-full-k16-term/candidates_predictions.jsonl`
- Matching merged train pool: `outputs/verifier_candidate_mining/20260522-train-current-k32-plus-qwen35-4b-full-k16-term/train_current_k32_plus_4bfull_k16_predictions.jsonl`

Pool diagnostics:

- Eval groups: 158
- Eval records after dedupe: 4,279
- Mean candidates/group: 27.08
- Oracle: selection `50.0788`, chrF++ `63.5158`, BLEU `35.7333`, token F1 `54.9308`, TER `51.3825`

The old K32 score ensemble transferred poorly to this new pool: selection `29.4916`, chrF++ `42.8544`, BLEU `12.9017`, TER `79.0323`. The pool has much more headroom, but the selector must be trained on the matching candidate distribution.

Matching-distribution selectors:

- Feature weights: `outputs/feature_candidate_reranker_evals/20260522-feature-listwise-current-k32-plus-qwen35-4b-full-k16-term/feature_listwise_current_k32_plus_4bfull_k16_weights.json`
- Text model: `outputs/text_candidate_reranker_evals/20260522-text-train-current-k32-plus-qwen35-4b-full-k16-term/text_current_k32_plus_4bfull_k16_model.json`
- Score ensemble: `outputs/score_ensemble_reranker_evals/20260522-ensemble-train-current-k32-plus-qwen35-4b-full-k16-term/ensemble_current_k32_plus_4bfull_k16_ensemble.json`
- Deployment wrapper: `scripts/generate_multi_adapter_ensemble_reranked_translations.py`

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous conservative K32 ensemble | 29.7584 | 43.8064 | 13.2505 | 30.4677 | 3.1118 | 0.0000 | 81.3364 |
| merged-pool feature selector | 30.4668 | 43.8848 | 14.8210 | 29.4766 | 2.0042 | 0.0000 | 76.4977 |
| merged-pool text reranker | 33.9542 | 46.9533 | 21.3086 | 34.0165 | 2.6793 | 0.0000 | 72.8111 |
| merged-pool score ensemble | 35.5481 | 48.1624 | 24.0635 | 35.9579 | 2.4156 | 0.0000 | 70.5069 |
| merged-pool oracle | 50.0788 | 63.5158 | 35.7333 | 54.9308 | 2.0992 | 0.0000 | 51.3825 |

Decision:

- This is the new best held-out profile by every headline metric.
- Full-SFT did not become the best greedy model, but it did create high-value candidate diversity.
- The current best deployable direction is multi-model candidate generation plus a matching-distribution text/score reranker.
- The remaining oracle gap is still huge, so more candidate diversity from 9B/35B-A3B SFT or a stronger QE/listwise selector is more promising than another plain continuation from the 2B policy.
- Retrieval-augmented prompts are now available in `scripts/evaluate_gspo_checkpoint.py` via `--few-shot-top-k`, and in the multi-adapter deployment wrapper via the same flag. The retriever uses clean train rows only and excludes exact source matches, so eval references are not placed in the prompt. This is intended for a controlled follow-up after the 9B no-few-shot candidate pass.

## 2026-05-23 Listwise Text Reranker

Purpose: adapt the text-aware selector to the actual listwise candidate-choice problem, closer to QE/listwise reranking papers. Pairwise training was already useful, but it learned independent winner-vs-loser preferences instead of a distribution over all candidates in a group.

Code:

- `scripts/train_text_candidate_reranker.py --training-objective listwise`
- `scripts/generate_multi_adapter_ensemble_reranked_translations.py --selection-mode text`
- Unit tests: `tests/test_train_text_candidate_reranker.py`, `tests/test_generate_multi_adapter_ensemble_reranked_translations.py`

Run:

- Train pool: `outputs/verifier_candidate_mining/20260522-train-current-k32-plus-qwen35-4b-full-k16-term/train_current_k32_plus_4bfull_k16_predictions.jsonl`
- Eval pool: `outputs/rerank_candidate_evals/20260522-current-k32-plus-qwen35-4b-full-k16-term/candidates_predictions.jsonl`
- Output: `outputs/text_candidate_reranker_evals/20260523-text-listwise-current-k32-plus-qwen35-4b-full-k16-term`
- Settings: `--training-objective listwise --epochs 8 --learning-rate 0.05 --listwise-temperature 0.06`

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous merged-pool score ensemble | 35.5481 | 48.1624 | 24.0635 | 35.9579 | 2.4156 | 0.0000 | 70.5069 |
| listwise text reranker | 38.9423 | 51.8586 | 26.0207 | 40.3340 | 1.9409 | 0.0000 | 67.7419 |
| listwise-text score ensemble | 38.4376 | 51.5307 | 25.3485 | 39.6017 | 2.0464 | 0.0000 | 69.1244 |
| oracle | 50.0788 | 63.5158 | 35.7333 | 54.9308 | 2.0992 | 0.0000 | 51.3825 |

Decision:

- This is the new best deployable held-out profile by selection, chrF++, BLEU, token F1, source-copy, and TER.
- The old score ensemble hurts this stronger text selector, so deploy this pool with `--selection-mode text` rather than the ensemble blend.
- The remaining gap to oracle is still large, but chrF++ `51.8586` and BLEU `26.0207` are the strongest verified metrics so far and move materially toward the chrF++ 60 target.

Deployment shape:

```bash
.venv/bin/python scripts/generate_multi_adapter_ensemble_reranked_translations.py \
  --selection-mode text \
  --adapter-path outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8 \
  --adapter-path outputs/full_sft_canaries/20260522-qwen35-4b-merged-broad512-chanka224-full-chanka-lr1e-6-64steps/chanka/checkpoint-48 \
  --num-return-sequences 32 \
  --num-return-sequences 16 \
  --text-model-json outputs/text_candidate_reranker_evals/20260523-text-listwise-current-k32-plus-qwen35-4b-full-k16-term/text_listwise_current_4bfull_model.json \
  --terminology-file clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet \
  --strip-chat-artifacts \
  --input-path sources.txt \
  --output-jsonl predictions.jsonl
```

## 2026-05-23 9B Eval-Only Candidate Complement

Purpose: check whether Qwen3.5 9B adds useful candidate diversity before waiting for the full train candidate pool and matched selector training.

Inputs:

- Base eval pool: `outputs/rerank_candidate_evals/20260522-current-k32-plus-qwen35-4b-full-k16-term/candidates_predictions.jsonl`
- 9B eval pool: `outputs/qwen35_9b_candidate_rerank/20260523-qwen35-9b-candidate-rerank/eval_qwen35_9b_k16_predictions.jsonl`
- Merged eval-only pool: `outputs/qwen35_9b_candidate_rerank/20260523-qwen35-9b-candidate-rerank/eval_only_diagnostics/eval_current_4b_9b_k16.jsonl`
- Records after dedupe: `5,963`, mean candidates/group `37.74`, max candidates/group `64`.

Eval-only diagnostics:

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current K32+4B listwise text | 38.9423 | 51.8586 | 26.0207 | 40.3340 | 1.9409 | 0.0000 | 67.7419 |
| 9B-augmented transfer text | 37.6669 | 50.6379 | 24.9145 | 38.5393 | 2.2574 | 0.0000 | 68.2028 |
| 9B-augmented MBR | 32.1510 | 46.3562 | 11.3516 | 33.3208 | 1.7827 | 0.0000 | 80.8756 |
| current K32+4B oracle | 50.0788 | 63.5158 | 35.7333 | 54.9308 | 2.0992 | 0.0000 | 51.3825 |
| 9B-augmented oracle | 51.6170 | 64.8936 | 37.1419 | 56.9913 | 2.0992 | 0.0000 | 48.3871 |

Decision:

- 9B adds real candidate diversity: oracle improves by `+1.3778` chrF++, `+1.4086` BLEU, and `-2.9954` TER over the current K32+4B pool.
- The existing listwise text selector does not transfer to the 9B-augmented distribution; it drops below the non-9B listwise profile.
- Wait for the 9B train pool and train a matched listwise text selector before deciding whether 9B improves deployable quality.

Matched-selector update:

- Train pool: `outputs/qwen35_9b_candidate_rerank/20260523-qwen35-9b-candidate-rerank/train_current_k32_4b_k16_9b_k16.jsonl`
- Eval pool: `outputs/qwen35_9b_candidate_rerank/20260523-qwen35-9b-candidate-rerank/eval_current_k32_4b_k16_9b_k16.jsonl`
- Text output: `outputs/qwen35_9b_candidate_rerank/20260523-qwen35-9b-candidate-rerank/text_ranker_listwise/text_current_4b_9b_model.json`
- Feature output: `outputs/qwen35_9b_candidate_rerank/20260523-qwen35-9b-candidate-rerank/feature_listwise/feature_current_4b_9b_weights.json`

| Matched 9B-augmented method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matched listwise text | 38.1557 | 51.2895 | 23.4968 | 39.9106 | 2.0992 | 0.0000 | 67.9724 |
| matched listwise feature | 31.8334 | 44.8211 | 13.6102 | 33.4293 | 2.3101 | 0.0000 | 74.1935 |
| matched score ensemble | 37.8162 | 50.7433 | 23.7969 | 39.3138 | 2.0992 | 0.0000 | 67.5115 |
| 9B-augmented oracle | 51.6170 | 64.8936 | 37.1419 | 56.9913 | 2.0992 | 0.0000 | 48.3871 |

Decision:

- The matched text selector recovers most of the transfer loss but still trails the current K32+4B listwise text profile (`38.9423` selection, chrF++ `51.8586`, BLEU `26.0207`).
- The feature selector is not competitive on this pool.
- The score ensemble improves TER over matched text but still trails on selection, chrF++, BLEU, and token F1.
- 9B remains useful as evidence of extra oracle headroom, but it is not a deployable win until the selector improves or the 9B base checkpoint is improved through better SFT.

## 2026-05-23 Tuned MBR Consensus Selector

Purpose: adapt the Kaggle Deep Past MBR idea more directly for our candidate pools: select the candidate with highest consensus against other candidates, then tune the reference-free signal weights on a train candidate pool.

Code:

- `scripts/train_mbr_consensus_selector.py`
- Unit tests: `tests/test_train_mbr_consensus_selector.py`

Implementation notes:

- Eval-time selection is reference-free.
- Training uses references only to tune weights on a separate train candidate pool.
- Signals include fast chrF-like char n-gram overlap, BLEU-like token n-gram precision, token F1, token precision, Jaccard, length consensus, duplicate rate, copy/leakage/artifact/repetition guards, and candidate index.
- The default utility is `--utility-mode fast`; sacrebleu pairwise utility was too slow for full candidate pools.
- Weight search is clamped so copy/leakage/artifact/repetition/exact-copy/candidate-index signals cannot become positive rewards, and overlap/consensus signals cannot become penalties.

Current K32 + 4B full-SFT pool result:

- Train pool: `outputs/verifier_candidate_mining/20260522-train-current-k32-plus-qwen35-4b-full-k16-term/train_current_k32_plus_4bfull_k16_predictions.jsonl`
- Eval pool: `outputs/rerank_candidate_evals/20260522-current-k32-plus-qwen35-4b-full-k16-term/candidates_predictions.jsonl`
- Output: `outputs/rerank_candidate_evals/20260523-current-k32-plus-4bfull-tuned-mbr-clamped`
- Settings: `--search-iterations 1000 --max-peers 4 --utility-mode fast`

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 25.9309 | 40.7905 | 8.9543 | 25.1761 | 2.7637 | 0.3165 | 92.1659 |
| default MBR | 29.8908 | 44.3650 | 10.6186 | 28.7035 | 1.5401 | 0.0000 | 84.7926 |
| tuned fast MBR | 29.9682 | 44.0250 | 10.6333 | 29.6208 | 1.8038 | 0.0000 | 82.7189 |
| score ensemble | 35.5481 | 48.1624 | 24.0635 | 35.9579 | 2.4156 | 0.0000 | 70.5069 |
| oracle | 50.0788 | 63.5158 | 35.7333 | 54.9308 | 2.0992 | 0.0000 | 51.3825 |

Decision:

- Tuned fast MBR is a small improvement over default MBR on selection, token F1, and TER, but it does not approach the text/score ensemble.
- It should be kept as a cheap deployable diagnostic and possible future ensemble signal, not as the primary selector.
- The next selector work should target stronger learned QE/listwise reranking over the high-oracle candidate pools.

Example inference shape:

```bash
.venv/bin/python scripts/generate_multi_adapter_ensemble_reranked_translations.py \
  --adapter-path outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8 \
  --adapter-path outputs/full_sft_canaries/20260522-qwen35-4b-merged-broad512-chanka224-full-chanka-lr1e-6-64steps/chanka/checkpoint-48 \
  --num-return-sequences 32 \
  --num-return-sequences 16 \
  --feature-weights-json outputs/feature_candidate_reranker_evals/20260522-feature-listwise-current-k32-plus-qwen35-4b-full-k16-term/feature_listwise_current_k32_plus_4bfull_k16_weights.json \
  --text-model-json outputs/text_candidate_reranker_evals/20260522-text-train-current-k32-plus-qwen35-4b-full-k16-term/text_current_k32_plus_4bfull_k16_model.json \
  --ensemble-json outputs/score_ensemble_reranker_evals/20260522-ensemble-train-current-k32-plus-qwen35-4b-full-k16-term/ensemble_current_k32_plus_4bfull_k16_ensemble.json \
  --terminology-file clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet \
  --strip-chat-artifacts \
  --input-path sources.txt \
  --output-jsonl predictions.jsonl
```

## 2026-05-23 New 4B Full-SFT Candidate Complement

Purpose: test whether the improved standalone 4B full-SFT checkpoint from the `2e-6` LR follow-up adds deployable candidate quality beyond the current K32+old-4B pool.

Inputs:

- Current deployable eval pool: `outputs/rerank_candidate_evals/20260522-current-k32-plus-qwen35-4b-full-k16-term/candidates_predictions.jsonl`
- New 4B full-SFT checkpoint: `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36`
- New 4B K16 eval pool: `outputs/qwen35_4b_full_sft_candidate_rerank/20260523-qwen35-4b-fft2e6ckpt36-candidate-rerank/eval_qwen35_4b_full_sft_k16_predictions.jsonl`
- Merged eval pool: `outputs/qwen35_4b_full_sft_candidate_rerank/20260523-qwen35-4b-fft2e6ckpt36-candidate-rerank/eval_current_4b_old_4b_new_k16.jsonl`
- Text selector: `outputs/qwen35_4b_full_sft_candidate_rerank/20260523-qwen35-4b-fft2e6ckpt36-candidate-rerank/text_ranker_listwise/text_current_4b_old_4b_new_summary.json`

Standalone new-4B K16 eval:

| Method | Selection | chrF++ | BLEU | token F1 | TER |
| --- | ---: | ---: | ---: | ---: | ---: |
| first candidate | 29.1882 | 42.5822 | 16.4238 | 28.3610 | 81.7972 |
| oracle | 41.3831 | 54.6467 | 25.0224 | 42.1051 | 63.5945 |

Merged-pool eval:

| Method | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first | 25.9309 | 40.7905 | 8.9543 | 25.1761 | 2.7637 | 0.3165 | 92.1659 |
| MBR | 30.9242 | 45.4973 | 11.0815 | 29.9680 | 1.7119 | 0.0000 | 83.6406 |
| matched listwise text | 38.3355 | 51.6498 | 23.9115 | 39.6077 | 1.9409 | 0.0000 | 69.1244 |
| oracle | 51.1497 | 64.5949 | 37.1959 | 56.1489 | 2.0992 | 0.0000 | 50.0000 |

Decision:

- The new 4B full-SFT checkpoint is the best standalone 4B base so far and has a useful K16 oracle.
- Adding it to the current deployable pool raises the oracle slightly over the old K32+4B pool, but the matched listwise text selector does not beat the current deployable result (`38.9423` selection, chrF++ `51.8586`, BLEU `26.0207`).
- This is a selector problem more than a generation problem. Keep the checkpoint for future reranker/verifier work, but do not change the default deployment recipe yet.
