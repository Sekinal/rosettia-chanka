# Chanka Quechua Source Inventory

Research date: 2026-05-18

Scope: sources that are explicitly Chanka Quechua, Ayacucho Chanka, or `quy` where possible. Sources that mix Chanka with Collao or generic Quechua are flagged.

## Triage Summary

Best immediate targets:

1. **MINEDU institutional repository records tagged `Quechua Chanka`**: broadest source pool, many DSpace records expose `dc.rights.uri` as `https://creativecommons.org/licenses/by/4.0/`. Mostly educational PDFs; valuable for monolingual Chanka, glossary extraction, and some bilingual/Spanish-instruction pairs.
2. **Official government translated pairs on gob.pe**: smaller but often easier to align because Spanish and Chanka PDFs exist side-by-side.
3. **ANA bilingual water law PDF**: high-value bilingual legal text, but license is CC BY-NC-ND 4.0, so keep separate from a permissive training release unless the team accepts that constraint.
4. **IWSLT/Common Voice datasets**: useful for baselines and speech/ASR, but not all are clean Chanka-only and several are non-commercial/no-derivatives.

Avoid for now:

- Scraped mirrors such as Scribd, Studocu, PDFCoffee, or random textbook aggregators when an official MINEDU/gob.pe/DSpace source exists.
- Glosbe or other translation-memory sites until license/provenance is clear.
- Generic Quechua resources without variety metadata.

## Priority A: MINEDU Chanka Repository

Main browse endpoint:

- https://repositorio.minedu.gob.pe/browse?type=subject&value=Quechua%20Chanka&rpp=100

Observed count from the subject browse: **109 Chanka-tagged items**. The inventory includes textbooks, workbooks, communication sheets, literature, stories, mathematics sheets, saberes/haceres materials, guides, and dictionaries.

Important licensing pattern:

- Several item pages expose:
  - `dc.rights info:eu-repo/semantics/openAccess`
  - `dc.rights.uri https://creativecommons.org/licenses/by/4.0/`
- Some PDFs still include printed boilerplate like "Todos los derechos reservados". Treat DSpace metadata and embedded PDF statements as a conflict to record per source before release.

Representative high-value records:

- https://repositorio.minedu.gob.pe/handle/20.500.12799/10204 - `Llaqtaypa Kawsayninkuna. Saberes de los pueblos - 4° Primaria - Quechua chanka`; DSpace metadata says CC BY 4.0/open access.
- https://repositorio.minedu.gob.pe/handle/20.500.12799/10044 - `Llaqtanchikpa kawsayninkuna - Chanka. Saberes de los pueblos 1 - 1° Secundaria - Quechua chanka`; DSpace metadata says CC BY 4.0/open access.
- https://repositorio.minedu.gob.pe/handle/20.500.12799/11888 - `Pukllaspa yachasunchik. Cuaderno de Trabajo de 4 Años - Quechua Chanka`; DSpace metadata says CC BY 4.0/open access.
- https://repositorio.minedu.gob.pe/handle/20.500.12799/10380 - `Yachakuqkunapa Simi Qullqa. Ayakuchu Chanka Qichwa simipi`; bilingual school dictionary/vocabulary, likely very useful for lexicon extraction.
- https://repositorio.minedu.gob.pe/handle/20.500.12799/10453 - `Reflexionando sobre nuestra lengua Ayacucho Chanka Qichwa Simi`; good candidate for grammar/orthography notes.

MINEDU gob.pe collection:

- Page: https://www.gob.pe/institucion/minedu/informes-publicaciones/4544363-biblioteca-quechua-chanka
- Direct PDFs:
  - https://cdn.www.gob.pe/uploads/document/file/5008648/literatura-comunicacion-quechua-chanka.pdf?v=1692307118
  - https://cdn.www.gob.pe/uploads/document/file/5008779/saberes-pueblos-quechua-chanka-comprimido.pdf?v=1692307118

Extraction notes:

- These are mostly not sentence-aligned Spanish-Chanka documents.
- Use them first for Chanka monolingual SFT, dictionary/term extraction, and possible instruction-pair extraction where Spanish instructions and Chanka content coexist.
- Build a `source_registry` table before download: source URL, PDF URL, title, institution, year, license metadata, embedded PDF rights text, variant, extraction mode, allowed use.

## Priority B: Official Translated/Bilingual PDFs

### Indecopi Environmental Advertising Guide

- Page: https://www.gob.pe/institucion/indecopi/informes-publicaciones/6811037-guia-de-publicidad-ambiental
- Spanish PDF: https://cdn.www.gob.pe/uploads/document/file/8129780/6811037-guia-de-publicidad-ambiental-espanol.pdf?v=1755891974
- Chanka PDF: https://cdn.www.gob.pe/uploads/document/file/8129783/6811037-guia-de-publicidad-ambiental-quechua-chanka.pdf?v=1755891976

Assessment: good parallel candidate. Same publication in Spanish and Chanka, official source, probably alignable by headings/pages. License terms need checking inside PDFs/page metadata.

### MINJUS TTAIP Accessible Materials

- Page: https://www.gob.pe/institucion/minjus/informes-publicaciones/7886849
- Chanka PDFs found:
  - https://cdn.www.gob.pe/uploads/document/file/9636341/7886849-ttaip-procedimiento-de-aip-quechua-chanka.pdf?v=1773850713
  - https://cdn.www.gob.pe/uploads/document/file/9636349/7886849-ttaip-procedimiento-de-apelacion-por-denegatoria-aip-quechua-chanka.pdf?v=1773850714
  - https://cdn.www.gob.pe/uploads/document/file/9649909/7886849-1843924-ttaip-procedimiento-sancionador-contra-funcionarios-y-servidores-publicos-quechua-chanka.pdf?v=1773938454
  - https://cdn.www.gob.pe/uploads/document/file/9649918/7886849-1843924-ttaip-recurso-de-apelacion-contra-funcionarios-quechua-chanka.pdf?v=1773938454

Assessment: small but likely high-quality public-service text. The page has Spanish-paired or bilingual variants for some materials, but naming is inconsistent (`espanol-aimara`, `espanol-ticuna`). Inspect layouts before assuming a clean Spanish counterpart.

### SUNASS Rural Water/Sanitation Claims

- Manual page: https://www.gob.pe/institucion/sunass/informes-publicaciones/5189527-como-reclamar-por-los-servicios-de-agua-y-alcantarillado-en-el-ambito-rural-manuales-en-5-lenguas-originarias
- Chanka manual: https://cdn.www.gob.pe/uploads/document/file/5853794/5189527-manual-en-quechua-chanka.pdf?v=1707837622
- Trifold page: https://www.gob.pe/institucion/sunass/informes-publicaciones/6485958-cartillas-informativas-sobre-el-reglamento-de-reclamos-en-el-ambito-rural-5-lenguas-originarias
- Chanka trifold: https://cdn.www.gob.pe/uploads/document/file/7652957/6485958-triptico-en-lengua-quechua-chanka.pdf?v=1739914194

Assessment: useful domain coverage. The page does not expose a clean Spanish PDF next to Chanka, so direct parallel extraction may require finding the Spanish base regulation/cartilla separately.

### CONTIGO / UN Committee Dictamen

- Page: https://www.gob.pe/institucion/contigo/informes-publicaciones/5799694-dictamen-traducida-al-quechua-chanka-aprobado-por-el-comite-en-relacion-con-el-protocolo-facultativo-de-la-convencion-sobre-los-derechos-del-nino-relativo-a-un-procedimiento-de-comunicaciones-respecto-de-la-comunicacion-num-136-2021
- Spanish/source PDF: https://cdn.www.gob.pe/uploads/document/file/6675113/5799694-dictamen-adoptado-por-el-comite-de-los-derechos-del-nino-de-de-naciones-unidas-el-15-de-mayo-de-2023.pdf?v=1721409191
- Chanka translation PDF: https://cdn.www.gob.pe/uploads/document/file/6675114/5799694-dictamen-camila-traducido-en-quechua-docx.pdf?v=1721409192

Assessment: small, official, likely alignable legal/human-rights parallel text. Check if the Spanish/source text is exactly the same base version used for translation.

### ANA Water Law

- Aggregated metadata: https://alicia.concytec.gob.pe/vufind/Record/ANAI_a48ae2c31d60606834d4be6c13cfb84e
- Direct PDF from metadata: https://repositorio.ana.gob.pe/bitstream/20.500.12543/5783/1/ANA0004324.pdf
- Metadata says: bilingual Spanish / Quechua Chanka, open access, `https://creativecommons.org/licenses/by-nc-nd/4.0/`.

Assessment: very valuable alignment target because it is explicitly bilingual. However, CC BY-NC-ND is not ideal for a permissively released training corpus. Keep in a restricted/research-only subset unless license strategy changes.

Related older mixed-variety source:

- https://repositorio.ana.gob.pe/handle/20.500.12543/436 - bilingual Spanish / Quechua Chanka-Collao. Useful only if the team accepts mixed variety or can isolate Chanka.

### Congreso Technical Sheets

- Example search result: https://leyes.congreso.gob.pe/Documentos/2016_2021/ADLP/Ficha_Tecnica_Quechua/30435-FTQ.pdf
- Search pattern: `site:leyes.congreso.gob.pe/Documentos "Traducido al: Quechua Chanka" "FTQ.pdf"`

Assessment: many small official Chanka technical sheets for laws. They are not obviously sentence-parallel with Spanish in the same file, but can be paired to Spanish law metadata/summaries with work. Good for legal terminology, less good for high-confidence parallel pairs.

## Priority C: Existing Datasets

### IWSLT 2026 Quechua-Spanish

- Track page: https://iwslt.org/2026/low-resource
- Dataset repo: https://github.com/johneortega/IWSLT2026_Quechua_data

Important facts:

- IWSLT says the Quechua data includes Ayacucho/Chanka (`quy`) and Cusco/Collao (`quz`) and labels the task as `que-spa`.
- Repo includes speech-aligned Spanish translations, 48 hours of transcribed Quechua audio without translations, synthetic/post-edited translations, Collao ASR/emotion data, and additional MT text from JW300/Hinantín.
- Repo license section says the work is Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported.

Assessment: good benchmark and baseline source. Not clean enough for our Chanka-only permissive release unless filtered by variety and license constraints are accepted.

### IWSLT 2025 Quechua-Spanish

- Repo: https://github.com/Llamacha/IWSLT2025_Quechua_data

Assessment: very similar to IWSLT 2026. Useful for baselines; license is CC BY-NC-ND 3.0.

### Common Voice Ayacucho Quechua

- Page: https://prod.datacollective.mozillafoundation.org/datasets/cmflnuzw6kz1l5ki9698u95lp
- Locale: `quy`
- License: CC-0
- Size observed: 1,300 clips / 3.1 hours; text corpus has 1,041 sentences.

Assessment: good for Chanka ASR/text prompts and pronunciation work, not a Spanish parallel corpus.

## Lower-Priority / Needs Care

- Glosbe Ayacucho Quechua-Spanish: useful for manual lookup, but do not scrape into a dataset until licensing/provenance is clear.
- Wiktionary/Wikipedia Quechua: CC BY-SA, but not reliably Chanka-specific. Use only as weak lexicon/monolingual data with variety flags.
- PUCP Lexis review of Soto Ruiz dictionary: the review article is CC BY 4.0, but it is a review, not the dictionary itself. The dictionary is valuable bibliographically, but full-text reuse needs separate rights.
- ISBN/WorldCat/Glottolog pages: useful for bibliographic discovery, not direct extractable training data.

## Recommended Next Step

Implement a source registry and downloader before extracting more text:

1. Create `data/interim/source_registry_chanka.{csv,parquet}` with one row per source/document.
2. Include columns: `source_id`, `title`, `institution`, `source_page_url`, `download_url`, `year`, `variant`, `license_text`, `license_url`, `rights_conflict`, `scrape_priority`, `extraction_mode`, `notes`.
3. Download only into `data/raw/source_documents/` so git stays clean.
4. Process Priority B first for parallel pairs, then Priority A for monolingual/glossary and selective bilingual extraction.
