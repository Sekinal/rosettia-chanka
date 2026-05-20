# SomosNLP Spanish-to-Quechua Dirty Corpus

Source dataset: https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua

Research date: 2026-05-19

## Role In This Project

Use this dataset only as **broad, dirty Spanish-Quechua SFT data** for an initial translation-adaptation stage. Do not label it as Chanka and do not mix it directly into the reviewed Chanka judicial corpus.

The Hugging Face dataset page reports:

- Task: translation.
- Languages: Spanish and Quechua.
- Format: Parquet.
- Rows: about 128k total, with train/validation/test splits.
- Columns observed in the dataset viewer: `es` and `qu`.

## Caveats

- The dataset is not Chanka-verified.
- The dataset appears broad/noisy and includes short fragments, headings, religious-domain text, date fragments, punctuation/numeric rows, and possible non-literal pairs.
- The dataset page did not expose a clear license in the visible card metadata during initial review. Treat outputs as internal training candidates until licensing/provenance is confirmed.
- Keep this data in a separate tier from clean Chanka data:
  - `tier_3_broad_quechua_parallel`
  - not `tier_1_chanka_reviewed`

## Preprocessing

Run:

```bash
uv run python scripts/preprocess_somosnlp_dirty_parallel.py
```

The script downloads the three Parquet split files into `data/raw/huggingface/somosnlp_spanish_to_quechua/`, scores each normalized pair, and writes:

- `data/processed/somosnlp_spanish_to_quechua_scored.parquet`
- `data/processed/somosnlp_spanish_to_quechua_high_quality_sft.parquet`
- `data/processed/somosnlp_spanish_to_quechua_broad_sft.parquet`
- `data/interim/somosnlp_dirty_preprocess_report.md`

All outputs are ignored by git.

## Filter Intent

The default SFT filter removes or demotes:

- empty sides;
- exact copies;
- punctuation/number-only rows;
- bracketed metadata rows;
- very short fragments;
- extreme Spanish/Quechua length-ratio rows;
- high-overlap copy-risk rows;
- rows with weak Spanish or Quechua signal.

The primary high-quality SFT output keeps only rows with no filter flags. The broader SFT output keeps rows with high enough scores and no hard rejection flags, and is mainly intended for ablations. For the first broad SFT stage, prefer `somosnlp_spanish_to_quechua_high_quality_sft.parquet`.
