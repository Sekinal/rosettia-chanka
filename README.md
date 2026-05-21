# RosettIA - Quechua

RosettIA is a SomosNLP 2026 hackathon project focused on open-source translation resources for low-resource languages, starting with **Chanka Quechua**.

The first milestone is a Spanish <-> Chanka Quechua parallel corpus extracted from public, reusable source material with explicit provenance.

## Public Dataset

Processed dataset artifacts are published on Hugging Face:

https://huggingface.co/datasets/Thermostatic/rosettia-chanka-data

The source PDF is not uploaded. The release includes:

- `clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet`: 1,055 reviewed Spanish-Chanka Quechua pairs.
- `clean_chanka/manual_quechua_chanka_parallel_reviewed.parquet`: reviewed extraction rows with decisions and flags.
- `clean_chanka/*glossary*.parquet`: glossary and simple term-pair artifacts.
- `broad_quechua/somosnlp_spanish_to_quechua_high_quality_sft.parquet`: 82,873 strict filtered broad Spanish-Quechua pairs derived from the public SomosNLP dataset.

Treat `clean_chanka/` and `broad_quechua/` as separate training tiers. The broad tier is not Chanka-verified.

## Setup

```bash
uv sync
```

## Repository Layout

- `data/raw/`: local source files such as PDFs. Ignored by git.
- `data/interim/`: local extraction artifacts. Ignored by git.
- `data/processed/`: local generated datasets. Ignored by git.
- `docs/references/`: source notes, license notes, and extraction caveats.
- `scripts/`: reproducible data preparation scripts.
- `src/rosettia/`: importable project code.
- `tests/`: test suite placeholder.

## Current Source

The first source document is:

`data/raw/source_documents/manual_quechua_chanka_administracion_justicia_2014.pdf`

It was renamed from `document.pdf` and moved into the local raw-data directory. The PDF states that reproduction is permitted as long as the source is cited. See `docs/references/manual_quechua_chanka_administracion_justicia_2014.md`.

## Extract Dataset

```bash
uv run python scripts/extract_manual_parallel_corpus.py
```

This writes local, git-ignored artifacts:

- `data/processed/manual_quechua_chanka_parallel_candidates.csv`
- `data/processed/manual_quechua_chanka_parallel_candidates.parquet`
- `data/processed/manual_quechua_chanka_parallel_auto_passed.csv`
- `data/processed/manual_quechua_chanka_parallel_auto_passed.parquet`

The current extraction is a table-based pass over role-marked bilingual rows. It preserves raw table cells, writes cleaned Spanish/Chanka Quechua text, and marks rows with context notes, alternatives, or extraction caveats as `needs_review = true`.

## Review Dataset

```bash
uv run python scripts/review_manual_parallel_corpus.py
```

This writes local, git-ignored reviewed artifacts:

- `data/processed/manual_quechua_chanka_parallel_reviewed.csv`
- `data/processed/manual_quechua_chanka_parallel_reviewed.parquet`
- `data/processed/manual_quechua_chanka_parallel_training_ready.csv`
- `data/processed/manual_quechua_chanka_parallel_training_ready.parquet`
- `data/interim/manual_review_report.md`

The current reviewed pass includes 1,011 rows and excludes one visually confirmed semantic mismatch. The `training_ready` subset currently has 974 pairs; rows with slash alternatives are kept in the reviewed file but held out of the training-ready file until they are split or normalized.

## Split Alternatives

```bash
uv run python scripts/split_manual_alternatives.py
```

This writes local, git-ignored artifacts:

- `data/processed/manual_quechua_chanka_parallel_alternative_splits.csv`
- `data/processed/manual_quechua_chanka_parallel_alternative_splits.parquet`
- `data/processed/manual_quechua_chanka_parallel_training_ready_augmented.csv`
- `data/processed/manual_quechua_chanka_parallel_training_ready_augmented.parquet`

The current split pass turns 37 slash-alternative rows into 81 clean rows. The augmented training-ready set has 1,055 pairs and no remaining slash markers.

## Extract Glossary

```bash
uv run python scripts/extract_manual_glossary.py
```

This writes local, git-ignored terminology artifacts:

- `data/processed/manual_quechua_chanka_glossary_entries.csv`
- `data/processed/manual_quechua_chanka_glossary_entries.parquet`
- `data/processed/manual_quechua_chanka_glossary_simple_terms.csv`
- `data/processed/manual_quechua_chanka_glossary_simple_terms.parquet`

The glossary is kept separate from sentence-pair training data. It is useful for terminology, retrieval, evaluation, and later augmentation.

## Preprocess Dirty Broad Quechua SFT Data

The SomosNLP 2022 Spanish-to-Quechua Hugging Face dataset is useful as broad, dirty translation SFT data before the clean Chanka judicial fine-tune. It is **not** treated as Chanka-verified data.

```bash
uv run python scripts/preprocess_somosnlp_dirty_parallel.py
```

This downloads the Hugging Face Parquet split files into `data/raw/huggingface/somosnlp_spanish_to_quechua/` and writes local, git-ignored artifacts:

- `data/processed/somosnlp_spanish_to_quechua_scored.csv`
- `data/processed/somosnlp_spanish_to_quechua_scored.parquet`
- `data/processed/somosnlp_spanish_to_quechua_high_quality_sft.csv`
- `data/processed/somosnlp_spanish_to_quechua_high_quality_sft.parquet`
- `data/processed/somosnlp_spanish_to_quechua_broad_sft.csv`
- `data/processed/somosnlp_spanish_to_quechua_broad_sft.parquet`
- `data/interim/somosnlp_dirty_preprocess_report.md`

The primary SFT subset is strict: it keeps only normalized pairs with no filter flags. A broader backup subset is also written for ablations. See `docs/references/somosnlp_spanish_to_quechua_dirty.md`.

## Preprocess AmericasNLP Quechua-Spanish Data

AmericasNLP 2024 Quechua-Spanish is another broad adaptation tier with `quy` resources. It is still not reviewed Chanka judicial-domain data.

```bash
uv run python scripts/preprocess_americasnlp_quechua_spanish.py
```

This downloads source files from the AmericasNLP GitHub repo and writes local, git-ignored artifacts:

- `data/processed/americasnlp_quechua_spanish_scored.parquet`
- `data/processed/americasnlp_quechua_spanish_high_quality_real_sft.parquet`
- `data/processed/americasnlp_quechua_spanish_eval.parquet`
- `data/interim/americasnlp_quechua_spanish_preprocess_report.md`

Current run counts:

- Raw aligned rows loaded: 142,116.
- Unique normalized pairs: 135,566.
- High-quality real SFT rows: 87,004.
- Evaluation rows: 996.

Use the real-only SFT file for now. Synthetic/backtranslated data is opt-in for local ablations via `--include-synthetic`; it is not part of the public release. See `docs/references/americasnlp_quechua_spanish.md`.

## Train With Unsloth SFT

SFT scripts for Qwen3.5 LoRA-style adapter training are in `scripts/train_sft_unsloth.py`. The current default model is `unsloth/Qwen3.5-2B` for cheaper early tests. The script supports `--adapter-method lora|dora|rslora` so LoRA, DoRA, and rank-stabilized LoRA can be compared under the same split and evaluation cadence. The Chanka pass can continue from the broad adapter with `--adapter-path`.

The workflow uses 16-bit LoRA, not QLoRA, and every run includes a validation split with best-checkpoint selection by `eval_loss`.

See `docs/unsloth_sft.md`.

## Data Policy

Do not commit source PDFs, extracted text, CSV files, Parquet files, or other data artifacts. Keep data local unless the team explicitly prepares a release package with the right citation and license language.
