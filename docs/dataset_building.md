# Dataset Building Notes

## Current Corpus Pass

Source:

- `Manual para el empleo del Quechua Chanka en la administración de justicia`
- Publisher: Ministerio de Cultura, Peru
- First edition: Lima, April 2014
- Local PDF path: `data/raw/source_documents/manual_quechua_chanka_administracion_justicia_2014.pdf`

Generated artifacts:

- `data/interim/extracted_text/manual_quechua_chanka_administracion_justicia_2014.layout.txt`
- `data/processed/manual_quechua_chanka_parallel_candidates.csv`
- `data/processed/manual_quechua_chanka_parallel_candidates.parquet`

The current parser uses PyMuPDF table detection and Polars. It extracts rows with explicit role markers from the bilingual tables:

- `A`: authority
- `C`: citizen
- `A/C`: either side

## Current Result

The current run generated **1,012 draft Spanish-Chanka Quechua parallel candidates** from PDF pages 17-118.

Of these, **800 rows auto-pass table extraction checks** and **212 rows are marked `needs_review = true`**. The extractor writes a semicolon-separated `quality_flags` column for prioritizing manual review.

The flagged count grew from the initial 90 because manual review found 122 additional Quechua alternatives in merged PDF cells that had blank role/source cells. These are now included by filling role/source from the previous row and flagging them for review.

Generated clean subset:

- `data/processed/manual_quechua_chanka_parallel_auto_passed.csv`
- `data/processed/manual_quechua_chanka_parallel_auto_passed.parquet`

Reviewed outputs:

- `data/processed/manual_quechua_chanka_parallel_reviewed.csv`
- `data/processed/manual_quechua_chanka_parallel_reviewed.parquet`
- `data/processed/manual_quechua_chanka_parallel_training_ready.csv`
- `data/processed/manual_quechua_chanka_parallel_training_ready.parquet`
- `data/interim/manual_review_report.md`

Current reviewed counts:

- Total extracted rows reviewed: 1,012
- Included after review: 1,011
- Training-ready after review: 974
- Kept but needs split/normalization before training: 37
- Excluded after review: 1

## Alternative Splits

The 37 reviewed slash-alternative rows were inspected and split with `scripts/split_manual_alternatives.py`.

Current split counts:

- Slash-alternative source rows: 37
- Split rows generated: 81
- Training-ready split rows: 81
- Augmented training-ready pairs: 1,055
- Remaining slash markers in augmented training set: 0

Generated artifacts:

- `data/processed/manual_quechua_chanka_parallel_alternative_splits.csv`
- `data/processed/manual_quechua_chanka_parallel_alternative_splits.parquet`
- `data/processed/manual_quechua_chanka_parallel_training_ready_augmented.csv`
- `data/processed/manual_quechua_chanka_parallel_training_ready_augmented.parquet`

## Glossary Extraction

The vocabulary sections are valuable, but they are terminology data, not the same kind of parallel sentence/dialogue data. They are extracted separately with `scripts/extract_manual_glossary.py`.

Current glossary counts:

- Total glossary entries: 423
- Simple term pairs: 219
- Rich entries or entries with examples/notes/alternatives: 204

Generated artifacts:

- `data/processed/manual_quechua_chanka_glossary_entries.csv`
- `data/processed/manual_quechua_chanka_glossary_entries.parquet`
- `data/processed/manual_quechua_chanka_glossary_simple_terms.csv`
- `data/processed/manual_quechua_chanka_glossary_simple_terms.parquet`

## Visual Verification

No OCR or Tesseract was used for validation. Ambiguous pages were rendered as images and checked visually against the PDF layout.

Pages visually checked during the first validation pass:

- Page 18: confirmed table extraction fixes the previous row bleed around formal greetings and "Le recibo".
- Page 35: confirmed rows with context notes and multiple alternatives should remain flagged instead of silently auto-passing.
- Page 52: confirmed an apparent duplicated target token is present in the PDF text/layout and should remain flagged.
- Page 52 follow-up: reviewed output corrects `¿Hayka watataq kuska tiyarankichik?tiyarankichik?` to `¿Hayka watataq kuska tiyarankichik?`.
- Page 54: confirmed a blank Spanish cell is a second Quechua alternative for "El es un borracho"; the extractor fills it from the previous row and flags it.
- Page 55: confirmed a blank Spanish cell is a second Quechua alternative for "Quiere dar poquisimo dinero"; the extractor fills it from the previous row and flags it.
- Page 104: excluded one semantic mismatch where the Spanish cell says `Mi hija me aviso` but the Quechua cell is a question about how something was seen.
- Page 118: confirmed table extraction fixes the previous bleed from the omnibus/colectivo note and the final emergency rows.

## Known Extraction Problems

- Rows with contextual parentheses are cleaned for the `spanish` and `chanka_quechua` fields, while the original text remains in `raw_spanish` and `raw_chanka_quechua`.
- Some rows contain multiple Spanish or Quechua alternatives in one cell; these are flagged for review rather than expanded automatically.
- Some merged table cells produce blank role/source cells for additional Quechua alternatives; these are filled from the previous row and flagged with `role_filled_from_previous_row` and `source_filled_from_previous_row`.
- Some table rows contain multiple Spanish or Quechua alternatives in one cell; reviewed outputs keep them but mark them as not `training_ready` until they are split or normalized.
- The augmented training-ready set includes split slash-alternative rows. Prefer this over the non-augmented training-ready set when a larger sentence-pair corpus is useful.
- Glossary artifacts should not be mixed into sentence-pair training without a deliberate formatting strategy.
- Glossary-style entries are extracted separately as terminology artifacts.

## Next Dataset Steps

- Add automated quality flags for suspicious rows: long text, repeated words, parenthetical notes, Spanish-looking target text, or multiple alternatives.
- Create a review CSV with `approved`, `corrected_spanish`, `corrected_chanka_quechua`, and `review_notes` columns.
- Add a second extractor for glossary entries.
- Decide final language tags after confirming the best standard tag for Chanka Quechua in the intended model/dataset ecosystem.
