# AmericasNLP 2024 Quechua-Spanish

Source: https://github.com/AmericasNLP/americasnlp2024/tree/master/ST1_MachineTranslation/data/quechua-spanish

Research date: 2026-05-20

## Role In This Project

Use this as another **broad Quechua-Spanish adaptation tier**, separate from the reviewed Chanka judicial corpus.

The AmericasNLP README says the development and test sets are translated into standard Southern Quechua, and that datasets are provided for Cuzco (`quz`) and Ayacucho (`quy`) variants. The `quechua-spanish` directory includes `quy` files.

Observed files:

- `train.es` / `train.quy`: 125,008 aligned lines.
- `dev.es` / `dev.quy`: 996 aligned lines.
- `test.es`: 1,003 Spanish-only input lines in the public repo.
- `dict.es` / `dict.quy`: 9,643 aligned dictionary/example lines.
- `extra.tsv`: 6,469 additional aligned rows.
- `synthetic.tsv`: 60,399 synthetic/backtranslated rows.

## Caveats

- This is not reviewed Chanka judicial-domain data.
- The training data is broad shared-task data. The README says only JW300 (`quy`) is included in `train.es` / `train.quy`.
- Repository metadata did not expose a top-level GitHub license during initial review. Keep license/provenance visible before redistribution.
- Keep real and synthetic sources separate in experiments.

## Preprocessing

Run:

```bash
uv run python scripts/preprocess_americasnlp_quechua_spanish.py
```

The script downloads source files into `data/raw/github/americasnlp2024/quechua-spanish/`, scores/deduplicates rows, and writes:

- `data/processed/americasnlp_quechua_spanish_scored.parquet`
- `data/processed/americasnlp_quechua_spanish_high_quality_real_sft.parquet`
- `data/processed/americasnlp_quechua_spanish_high_quality_with_synthetic_sft.parquet`
- `data/processed/americasnlp_quechua_spanish_eval.parquet`
- `data/interim/americasnlp_quechua_spanish_preprocess_report.md`

All generated outputs are ignored by git.
