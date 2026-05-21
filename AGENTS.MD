# AGENTS.MD

## Task Context

We are participating in the SomosNLP 2026 hackathon with the project **RosettIA - Quechua**.

The immediate task that created this file was to establish a durable working memory for current and future agent sessions: store useful findings, caveats, project intent, and operational notes that may matter later.

## Hackathon Context

- Event: **#HACKATHONSomosNLP 2026**, 5th edition.
- Main theme: post-training of LLMs.
- Suggested technical directions from organizers:
  - Reinforcement learning.
  - Preference alignment.
  - Test-time learning.
  - DPO/alignment using generated preference data.
- Cultural direction: projects are open, but organizers encourage links to cultures of LATAM and the Iberian Peninsula.
- Launch date from announcement: **Monday, May 11, 2026**.
- Workshop: Sergio Paniego, MLE at Hugging Face, on post-training techniques and libraries.
- Main info page: https://somosnlp.org/hackathon
- Hugging Face org invite from announcement: https://huggingface.co/organizations/somosnlp-hackathon-2026/share/DNcqoZrtSmEkyLLOiSYTQCzkcrquceDoVY
- Registration link from announcement: https://hackathon-somosnlp-2026.eventbrite.com/
- Workshop recording playlist from announcement: https://www.youtube.com/playlist?list=PLTA-KAy8nxaCGGYz5CWiLZNzc31ilPDyI
- Book/resource mentioned in workshop: https://huggingface.co/spaces/nanotron/ultrascale-playbook

## Project: RosettIA - Quechua

Core idea from the team thread:

> Build open-source translation technology for low-resource languages, starting with Quechua, the most widely spoken Indigenous language in the Americas. The long-term goal is to adapt the techniques to additional languages and reduce the technological lag affecting speakers of these languages.

Motivation:

- Low-resource translation remains underdeveloped despite rapid progress in LLMs.
- Quechua is a high-impact starting point because of its speaker population and cultural importance.
- The project should support language survival and inclusion for people who primarily or only speak Indigenous languages.
- Keep the work open-source and reusable for future languages where possible.

Known team members from the project announcement:

- Thermostatic
- NiiaTheBugg

Current language scope:

- Default target variety for now: **Chanka Quechua**.
- Reason: the team currently considers this the safest path because of licensing concerns around other available Quechua resources.
- Future agents should not broaden the scope to all Quechua varieties unless the team explicitly approves it and the relevant dataset/model licenses have been checked.

## Current Repository State

As of 2026-05-18, the repo is a Python `>=3.12` project managed with `uv`.

Current structure:

- `data/raw/`: local source files such as PDFs. Ignored by git.
- `data/interim/`: local extraction artifacts. Ignored by git.
- `data/processed/`: local generated datasets. Ignored by git.
- `docs/references/`: source notes, license notes, and extraction caveats.
- `scripts/`: reproducible data preparation scripts.
- `src/rosettia/`: importable project code.
- `tests/`: test suite placeholder.

Important git convention:

- Do **not** commit data artifacts: PDFs, extracted text, CSV, Parquet, generated datasets, model files, or other large/local data. `data/**` is ignored except `.gitkeep` placeholders.
- Public processed dataset release: https://huggingface.co/datasets/Thermostatic/rosettia-chanka-data
- The HF dataset release intentionally does not include the source PDF. It includes processed Parquet artifacts plus metadata only.

## Working Principles For Future Agents

- Preserve user/team intent: this is a hackathon project about Quechua translation and low-resource language inclusion, not a generic translation demo.
- Prefer small, well-documented steps that can survive handoff between sessions.
- Record important findings here when they affect future work: datasets, licensing constraints, model choices, evaluation pitfalls, commands, credentials setup expectations, and known blockers.
- Avoid inventing facts about Quechua varieties, datasets, or licensing. Verify with primary sources or dataset cards when possible.
- Treat Quechua as a language family/continuum with regional varieties. Be explicit about which variety or dataset a model/evaluation targets.
- For now, assume Chanka Quechua is the intended target variety and flag any resource that mixes varieties or does not clearly identify the variety.
- Be careful with Spanish-centric assumptions. Quechua translation may involve Spanish, English, or other pivot languages, but the project goal is not just Spanish NLP.
- Favor reproducible pipelines over one-off notebooks when practical.
- Prefer modern libraries and columnar data tooling. Use **Polars** for tabular dataset work unless there is a strong reason not to.
- Keep licensing and consent visible, especially for Indigenous-language data.
- When using Hugging Face datasets/models, read model cards and dataset cards before integrating them.
- When adding training or post-training code, include a minimal way to run a smoke test locally.

## Likely Technical Directions To Explore

These are candidate directions, not commitments:

- Baseline translation model or pipeline for Quechua <-> Spanish and/or Quechua <-> English.
- SFT data collection or conversion from public parallel corpora.
- Preference dataset construction for translation quality, cultural appropriateness, and dialect/variant sensitivity.
- DPO or another preference-alignment method on a small open model if GPU credits are available.
- RLVR-style reward design if a reliable automatic verifier or evaluator can be built.
- Synthetic data generation with careful human review or filtering.
- Evaluation with automatic metrics plus qualitative review by humans familiar with the target variety.

## Caveats

- The hackathon organizers mentioned GPU credits after contributing to the preference dataset. Future agents should confirm the current process before assuming access.
- Announcements and Discord channel names came from user-provided text, not direct live verification.
- Current source PDF: `data/raw/source_documents/manual_quechua_chanka_administracion_justicia_2014.pdf`.
- Current source note: `docs/references/manual_quechua_chanka_administracion_justicia_2014.md`.
- Current extraction script: `scripts/extract_manual_parallel_corpus.py`.
- Current extraction result: 1,012 draft Spanish-Chanka Quechua candidates from pages 17-118, of which 800 auto-pass table extraction checks and 212 are flagged for review. Generated files are local and git-ignored.
- Current reviewed result: 1,011 included rows and 974 training-ready pairs. One row from page 104 is excluded because visual review showed a semantic mismatch between Spanish and Quechua cells. Rows with slash alternatives are kept in the reviewed artifact but excluded from the training-ready subset until split/normalized.
- Current split-alternative result: 37 slash-alternative rows split into 81 clean rows. Augmented training-ready corpus has 1,055 pairs and no slash markers.
- Current glossary result: 423 terminology entries from the vocabulary sections, including 219 simple term pairs. Keep glossary artifacts separate from sentence-pair training unless a deliberate formatting strategy is chosen.
- Current external-source research note: `docs/source_inventory_chanka.md`.
- Current broad dirty SFT source note: `docs/references/somosnlp_spanish_to_quechua_dirty.md`.
- Current broad dirty SFT preprocessing script: `scripts/preprocess_somosnlp_dirty_parallel.py`.
- The SomosNLP 2022 Spanish-to-Quechua Hugging Face dataset should be treated as broad, noisy Quechua translation data only. It is not Chanka-verified, and the visible dataset card did not expose a clear license during initial review. Keep its outputs in a separate tier from reviewed Chanka data.
- Current SomosNLP dirty preprocessing run: 128,434 raw rows, 123,839 unique normalized pairs, 82,873 strict high-quality SFT rows, and 111,699 broad backup SFT rows. Generated outputs are git-ignored under `data/`.
- Current AmericasNLP source note: `docs/references/americasnlp_quechua_spanish.md`.
- Current AmericasNLP preprocessing script: `scripts/preprocess_americasnlp_quechua_spanish.py`.
- AmericasNLP 2024 Quechua-Spanish can be used as another broad adaptation tier with `quy` data, but it is not reviewed Chanka judicial-domain data. Keep synthetic/backtranslated data opt-in only.
- Current AmericasNLP real-source preprocessing run: 142,116 raw aligned rows, 135,566 unique normalized pairs, 87,004 high-quality real SFT rows, and 996 eval rows. Generated outputs are git-ignored under `data/`. Synthetic/backtranslated data is opt-in only via `--include-synthetic`.
- Current broader data-source research note: `docs/references/other_parallel_data_candidates.md`. Best next candidates are FLORES `spa_Latn`/`quy_Latn` for evaluation, PanLex Chanka/Ayacucho CC0 lexical data for terminology/reward features, and IWSLT only for restricted experiments because of NC/ND licensing.
- Current HF dataset release: `Thermostatic/rosettia-chanka-data`, public. Uploaded files include clean Chanka reviewed/training/glossary Parquets and broad Quechua filtered/scored Parquets. No PDF was uploaded.
- Current SFT training script: `scripts/train_sft_unsloth.py`.
- Current SFT training note: `docs/unsloth_sft.md`.
- Unsloth official Qwen3.5 docs say QLoRA/4-bit training is not recommended for Qwen3.5. Use 16-bit LoRA with `load_in_4bit=False`, `load_in_16bit=True`, `use_gradient_checkpointing="unsloth"`, and validation enabled. The script creates eval splits, selects best checkpoints by `eval_loss`, and can continue Chanka SFT from the broad LoRA with `--adapter-path`.
- Remote SFT setup on `root@91.224.44.81 -p 20899`: repo path `/root/rosettia-chanka`, venv `.venv`, GPU `NVIDIA GeForce RTX 3090` with Torch `2.10.0+cu128`, Unsloth `2026.5.5`, Polars `1.40.1`. Smoke test passed on 2026-05-20 with `--stage broad --max-train-samples 32 --max-eval-samples 8 --max-steps 2 --eval-steps 1 --save-steps 1 --output-dir outputs/smoke`; validation ran during training, checkpoints were saved, and `outputs/smoke/broad/final_lora` was produced. Flash Attention 2 was not installed, so Unsloth used XFormers fallback.
- GitHub issue #2 requested creating `brancheef` in the owner repo from `niiaespinosa/rosettia-chanka:brancheef`. The branch was created, then merged into `main` after the user clarified they wanted it merged too.
- Current GSPO paper-profile plan: `docs/findings/gspo_paper_profile_plan.md`. Independent scripts live under `experiments/gspo/`: `run_2411_fine_grained_severity.sh`, `run_2310_severity_proxy.sh`, `run_2511_self_verifier.sh`, `run_2511_vibethinker.sh`, and `run_2511_train_verifier_then_gspo.sh`.
- Do not present the 2310 severity-proxy run as real xCOMET. We do not have a Chanka-capable xCOMET model. Treat it only as a low-cost transparent severity ablation.
- DeepSeek-style verifier work is now the serious reward-model path. `scripts/train_verifier_chanka_unsloth.py` trains a separate JSON-scoring verifier LoRA from clean Chanka pairs plus synthetic corruptions, with validation enabled. The first chained script trains the verifier, then runs the `self_verifier_2511` GSPO profile; direct learned-verifier reward integration is still pending.
- Remote smoke on `root@216.81.248.197 -p 20299` passed for the renamed severity proxy, repaired VibeThinker batching, and chained verifier+GSPO script. VibeThinker needs per-device `TRAIN_BATCH_SIZE=8` when `NUM_GENERATIONS=8`; the previous `TRAIN_BATCH_SIZE=2` crashed in eval reward reshaping.
- Current GSPO canary sweep script: `experiments/gspo/run_canary_sweep.sh`. It tests individual paper profiles plus mixed profiles `mix_severity_verifier`, `mix_verifier_vibe`, `mix_all_strict`, `rosettia_guard_v1`, and `rosettia_guard_v2`, then summarizes metrics with `scripts/summarize_gspo_canaries.py`.
- Web search on 2026-05-18 found a strong Chanka source pool:
  - MINEDU DSpace subject browse has 109 `Quechua Chanka` items; many records expose CC BY 4.0/open access metadata, but embedded PDF rights text may conflict and must be recorded per source.
  - Best near-term parallel candidates are official gob.pe translated PDF pairs from Indecopi, MINJUS/TTAIP, SUNASS, CONTIGO, and ANA.
  - ANA water law is explicitly bilingual Spanish/Quechua Chanka but metadata indicates CC BY-NC-ND 4.0, so keep separate from any permissive release unless the team accepts the restriction.
  - IWSLT 2026/2025 Quechua-Spanish data is useful for baselines but mixes Chanka (`quy`) and Collao (`quz`) and is licensed CC BY-NC-ND 3.0.
  - Mozilla Common Voice Ayacucho Quechua (`quy`) is CC-0 and useful for ASR/text prompts, not Spanish parallel data.
- Some public Quechua resources may have unclear licenses, domain mismatch, orthography differences, or mixed language varieties.
- Licensing risk is currently a primary reason to narrow scope to Chanka Quechua.
- The current PDF extraction uses PyMuPDF table detection, not OCR. Visual checks were done on ambiguous rendered pages. Rows with context notes, alternatives, filled merged cells, or extraction caveats still need review before use for training.
- Translation quality for low-resource languages can look deceptively good in aggregate metrics while failing cultural, dialectal, or terminology-sensitive cases.
- Do not collapse all Indigenous languages or all Quechua varieties into one undifferentiated target.

## Useful Next Steps

- Fill `README.md` with the project statement, setup instructions, and first milestone.
- Identify target translation directions and Quechua variety/varieties.
- Inventory available datasets and licenses.
- Choose an initial baseline model and evaluation strategy.
- Decide whether the hackathon deliverable is a model, dataset, demo, paper-style report, or a combination.
- Add reproducible scripts for data preparation, training/post-training, evaluation, and demo inference as the project grows.
