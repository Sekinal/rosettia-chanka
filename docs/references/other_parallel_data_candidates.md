# Other Quechua-Spanish Data Candidates

Research date: 2026-05-20

Scope: additional data sources for the tiered RosettIA workflow after the current clean Chanka corpus, SomosNLP broad corpus, and AmericasNLP 2024 real-source tier.

## Recommendation

Use the following next:

1. **FLORES-200 / FLORES+ (`spa_Latn` <-> `quy_Latn`)** for evaluation only.
2. **PanLex Chanka/Ayacucho Quechua lexical pairs** for terminology constraints, glossary augmentation, and reward features.
3. **OPUS tiny generic Quechua pairs** only as a small smoke-test/lexical sanity source.
4. **IWSLT 2025/2026** only for restricted experiments or benchmarking because of NC/ND licensing.

Do not add Cuzco/Collao-specific HF datasets to the main broad SFT stage unless we explicitly decide to do cross-variant adaptation experiments.

## High-Value Candidates

### FLORES-200 / FLORES+

Sources:

- https://huggingface.co/datasets/facebook/flores
- https://huggingface.co/datasets/openlanguagedata/flores_plus

Findings:

- Supports `quy_Latn` and `spa_Latn`.
- HF dataset card for `facebook/flores` reports license `cc-by-sa-4.0`.
- `openlanguagedata/flores_plus` also reports `cc-by-sa-4.0`, but is gated and explicitly asks users not to re-host evaluation content where crawlers can pick it up.

Use:

- **Evaluation only**, not training.
- Keep dev/devtest out of training data to avoid contamination.
- Good for measuring generic Spanish <-> Ayacucho Quechua translation, but not judicial-domain Chanka.

Decision: add an eval-preparation script later if we need standard benchmark metrics.

### PanLex

Sources:

- License: https://panlex.org/license
- Chanka/Ayacucho example: https://app.panlex.org/panlinx/ex/7069142

Findings:

- PanLex data snapshots are under CC0 1.0 Universal.
- Search results expose Chanka/Ayacucho Quechua entries.
- This is lexical translation data, not sentence-parallel corpus.

Use:

- Terminology constraints for decoding/evaluation.
- Reward features for judicial Chanka vocabulary preservation.
- Glossary expansion and lexicon filtering.

Decision: strong candidate, but keep separate from sentence SFT.

### AmericasNLP 2025 Quechua-Spanish

Sources:

- Task page: https://turing.iimas.unam.mx/americasnlp/2025_st_1.html
- Data repo: https://github.com/AmericasNLP/americasnlp2025/tree/main/ST1_MachineTranslation/data/quechua-spanish

Findings:

- The 2025 shared task includes Quechua (`quy`) and both translation directions.
- The `quechua-spanish` data directory has the same public counts observed in the 2024 directory for train/dev/dict/extra/synthetic:
  - `train.es` / `train.quy`: 125,008 lines.
  - `dev.es` / `dev.quy`: 996 lines.
  - `dict.es` / `dict.quy`: 9,643 lines.
  - `extra.tsv`: 6,469 rows.
  - `synthetic.tsv`: 60,399 rows.
- GitHub repo metadata reports no top-level license.

Use:

- Do **not** duplicate our AmericasNLP 2024 tier unless content differs after checksum comparison.
- The 2025 page is useful as a current shared-task reference and evaluation framing.

Decision: low immediate priority; likely duplicate of our current AmericasNLP real-source tier.

## Lower-Value / Restricted Candidates

### IWSLT 2025 / 2026 Quechua-Spanish

Sources:

- 2025 repo: https://github.com/Llamacha/IWSLT2025_Quechua_data
- 2026 repo: https://github.com/johneortega/IWSLT2026_Quechua_data
- IWSLT 2026 track page: https://iwslt.org/2026/low-resource

Findings:

- Useful Quechua-Spanish speech-translation resources.
- Includes Chanka/Ayacucho (`quy`) and Collao/Cusco (`quz`) material.
- 2025 repo states CC BY-NC-ND 3.0.

Use:

- Restricted experiments and benchmarking only.
- Do not include in a permissive public training release.

Decision: useful if we do a speech translation angle or compare against IWSLT baselines, but not primary SFT.

### OPUS Generic Quechua

Source:

- OPUS API: https://opus.nlpl.eu/opusapi/

Findings from OPUS API:

- `es` -> `qu`:
  - Tatoeba: 170 alignment pairs.
  - Wikimedia: 280 alignment pairs.
  - Ubuntu: tiny/unclear count.
- `es` -> `quz`:
  - GNOME: 7 alignment pairs.
- Direct `es`/`spa` -> `quy` returned no corpora via OPUS API.

Use:

- Too small for meaningful SFT.
- Could be used as a smoke-test or tiny lexical sanity set.

Decision: low priority.

### Cuzco/Collao Hugging Face Datasets

Sources:

- https://huggingface.co/datasets/pollitoconpapass/cuzco-quechua-translation-spanish
- https://huggingface.co/datasets/Zeal-Nir/cuzco-quechua-2-spanish-dataset

Findings:

- `pollitoconpapass/cuzco-quechua-translation-spanish` exposes columns `spa` and `quz`; sizes are roughly 106k train, 15k validation, 13k test.
- `Zeal-Nir/cuzco-quechua-2-spanish-dataset` exposes `quechua` and `spanish`; roughly 619 train and 16.7k eval examples.
- These are Cuzco/Collao-oriented, not Chanka/Ayacucho.
- Visible cards did not expose clear licenses during initial review.

Use:

- Only for explicit cross-variant adaptation ablations.
- Do not mix with Chanka-targeted training by default.

Decision: skip for now.

### Common Voice Ayacucho Quechua

Source:

- https://prod.datacollective.mozillafoundation.org/datasets/cmflnuzw6kz1l5ki9698u95lp

Findings:

- Ayacucho Quechua / Chanka (`quy`) speech dataset.
- License CC0.
- Not Spanish parallel text.

Use:

- ASR, pronunciation, or transcript-style monolingual text prompts.
- Not useful for our current Spanish-Chanka text translation SFT stage.

Decision: keep out of current text MT data mixture.

## Current Training Mixture Guidance

For Qwen/Unsloth SFT:

1. Broad translation adaptation:
   - `somosnlp_spanish_to_quechua_high_quality_sft.parquet`
   - `americasnlp_quechua_spanish_high_quality_real_sft.parquet`
2. Optional evaluation:
   - `americasnlp_quechua_spanish_eval.parquet`
   - FLORES `spa_Latn`/`quy_Latn` after access/prep.
3. Chanka/domain adaptation:
   - `manual_quechua_chanka_parallel_training_ready_augmented.parquet`
4. Reward/terminology support:
   - current glossary artifacts;
   - PanLex Chanka/Ayacucho terms once extracted.
