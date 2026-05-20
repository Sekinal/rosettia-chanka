# Manual para el empleo del Quechua Chanka en la administración de justicia

## Source

- File: `data/raw/source_documents/manual_quechua_chanka_administracion_justicia_2014.pdf`
- Title: `Manual para el empleo del Quechua Chanka en la administración de justicia`
- Publisher: Ministerio de Cultura, Peru
- First edition: Lima, April 2014
- ISBN: `978-612-4126-20-8`
- Pages: 150
- Content author listed in PDF: Wilfredo Ardito Vega
- Collaborators listed in PDF: Eber Llacctarimay Quispe, Cinthya Palomino Cordova, Juan Galiano Roman, Apolinario Ciriaco Saldivar Bolivar, Clodomiro Landeo Lagos, Salvador Alvarado Tovar, Carlos Andres Vera Vasquez

## License / Reuse Note

The PDF states: `Se permite la reproduccion de esta obra siempre y cuando se cite la fuente.`

Working interpretation for the dataset pipeline: reuse appears permitted with source citation, but preserve the citation and keep this document isolated as a source-specific dataset until the team confirms final release wording.

## Extraction Notes

The manual contains bilingual Spanish and Chanka Quechua phrase tables for justice-administration scenarios. The first extraction pass targets only rows with explicit role markers:

- `A`: autoridad
- `C`: ciudadano / ciudadana
- `A/C`: expressions used by either side

The parser uses PyMuPDF table detection to extract the role, Spanish, and Chanka Quechua cells. It preserves raw cells and also writes cleaned text with parenthetical notes removed. Rows with context notes, alternatives, filled merged cells, or other caveats are marked `needs_review = true`.

## Dataset Artifacts

- Layout text extraction: `data/interim/extracted_text/manual_quechua_chanka_administracion_justicia_2014.layout.txt`
- Draft CSV candidates: `data/processed/manual_quechua_chanka_parallel_candidates.csv`
- Draft Parquet candidates: `data/processed/manual_quechua_chanka_parallel_candidates.parquet`
- Auto-passed CSV subset: `data/processed/manual_quechua_chanka_parallel_auto_passed.csv`
- Auto-passed Parquet subset: `data/processed/manual_quechua_chanka_parallel_auto_passed.parquet`
- Reviewed CSV: `data/processed/manual_quechua_chanka_parallel_reviewed.csv`
- Reviewed Parquet: `data/processed/manual_quechua_chanka_parallel_reviewed.parquet`
- Training-ready CSV: `data/processed/manual_quechua_chanka_parallel_training_ready.csv`
- Training-ready Parquet: `data/processed/manual_quechua_chanka_parallel_training_ready.parquet`
- Alternative splits CSV: `data/processed/manual_quechua_chanka_parallel_alternative_splits.csv`
- Alternative splits Parquet: `data/processed/manual_quechua_chanka_parallel_alternative_splits.parquet`
- Augmented training-ready CSV: `data/processed/manual_quechua_chanka_parallel_training_ready_augmented.csv`
- Augmented training-ready Parquet: `data/processed/manual_quechua_chanka_parallel_training_ready_augmented.parquet`
- Glossary entries CSV: `data/processed/manual_quechua_chanka_glossary_entries.csv`
- Glossary entries Parquet: `data/processed/manual_quechua_chanka_glossary_entries.parquet`
- Simple glossary terms CSV: `data/processed/manual_quechua_chanka_glossary_simple_terms.csv`
- Simple glossary terms Parquet: `data/processed/manual_quechua_chanka_glossary_simple_terms.parquet`

## Known Caveats

- This is a domain-specific justice/administration manual, so the corpus will be useful for legal-service translation but should not be treated as general-domain Chanka Quechua.
- Some Quechua terms are intentionally borrowed or adapted from Spanish because the manual preserves common local usage.
- The first pass does not yet extract glossary-style entries that lack `A`/`C` role markers.
