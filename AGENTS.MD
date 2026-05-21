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
- Current GSPO paper-profile plan: `docs/findings/gspo_paper_profile_plan.md`. Independent scripts live under `experiments/gspo/`: `run_2411_fine_grained_severity.sh`, `run_2310_severity_proxy.sh`, `run_2511_self_verifier.sh`, `run_2511_vibethinker.sh`, `run_2511_train_verifier_then_gspo.sh`, and `run_2511_learned_verifier_gspo.sh`.
- Do not present the 2310 severity-proxy run as real xCOMET. We do not have a Chanka-capable xCOMET model. Treat it only as a low-cost transparent severity ablation.
- DeepSeek-style verifier work is now the serious reward-model path. `scripts/train_verifier_chanka_unsloth.py` trains a separate JSON-scoring verifier LoRA from clean Chanka pairs plus synthetic corruptions, with validation enabled. The verifier data now includes easy corruptions plus harder fluent Chanka negatives: unrelated Chanka translations, mixed-reference candidates, unsupported Chanka additions, and repetition damage. `learned_verifier_2511` in `scripts/train_gspo_chanka_unsloth.py` can load that verifier LoRA in the reward loop and blend its JSON score with hard anti-copy/leakage/reference guards. This is expensive but now preferred over leaning on the severity proxy.
- The verifier trainer can now take real model outputs via repeated `--candidate-jsonl` flags. Use the prediction JSONL files from `scripts/evaluate_gspo_checkpoint.py` as hard negatives/near-misses for a stronger second-pass verifier if the current hard-negative verifier is not discriminative enough.
- Do not optimize serious GSPO selection against the xCOMET-inspired severity proxy. We do not expect to have a real Chanka-capable xCOMET evaluator, so `severity_proxy_2310` should stay a transparent ablation/diagnostic only. Serious reward-model work should prioritize the learned hard-negative verifier and external lexical sanity metrics.
- Current evaluation stance: cheap external metrics and transparent proxies are still useful for regression triage, but they are not the product objective and should not be treated as a real Chanka quality judge. Use chrF++/BLEU/token-F1/copy/leakage/TER to catch obvious failures and compare canaries, then use the learned hard-negative verifier plus qualitative review for serious GSPO selection. If proxies and qualitative samples disagree, trust the samples and train a stronger verifier rather than chasing the proxy.
- Current xCOMET stance: assume we will not have a usable Chanka-capable xCOMET evaluator. xCOMET-inspired/severity proxies still make sense as cheap filters and failure accounting, but not as evidence that the translator is good. For serious model selection, prioritize reload-generation quality, clean held-out references, artifact/copy/leakage metrics, qualitative samples, and a DeepSeek-style trained verifier. If cost is not the constraint, train stronger verifier passes from real candidate outputs and pairwise/listwise preference labels rather than optimizing harder against the proxy.
- Remote smoke on `root@216.81.248.197 -p 20299` passed for the renamed severity proxy, repaired VibeThinker batching, and chained verifier+GSPO script. VibeThinker needs per-device `TRAIN_BATCH_SIZE=8` when `NUM_GENERATIONS=8`; the previous `TRAIN_BATCH_SIZE=2` crashed in eval reward reshaping.
- Current GSPO canary sweep script: `experiments/gspo/run_canary_sweep.sh`. It tests individual paper profiles plus mixed profiles `mix_severity_verifier`, `mix_verifier_vibe`, `mix_all_strict`, `rosettia_guard_v1`, and `rosettia_guard_v2`, then summarizes metrics with `scripts/summarize_gspo_canaries.py`.
- Current full-contender launcher: `experiments/gspo/run_best_canary_full.sh`. Set `SWEEP_DIR` to a completed canary sweep; it selects the top external-metric `summary.jsonl` profile unless `PROFILE` is overridden, then runs a two-epoch GSPO contender over the full clean Chanka corpus with validation/checkpoints.
- Live GSPO status on `root@216.81.248.197 -p 20299` as of 2026-05-21 08:42 UTC:
  - Canary sweep `outputs/gspo_canary_sweeps/20260521-paper-mix-v1` completed. Top external-metric profile was `vibethinker_2511` with selection `21.0148`, chrF++ `37.0672`, BLEU `6.4582`, token F1 `18.2594`, source-copy `4.0551%`, exact-copy `1.5625%`, and leakage `0.3906%`.
  - First full VibeThinker contender reached `outputs/gspo_full_contenders/vibethinker_2511_20260521-073724/chanka_gspo/checkpoint-196`; eval reward there was `0.30345`. Earlier checkpoints: `checkpoint-28` eval reward `0.30127`, `checkpoint-56` eval reward `0.29794`, `checkpoint-112` eval reward `0.30599`. That original process later stopped using GPU SM while still holding GPU memory and was killed; it had been launched before completion-table logging was disabled.
  - Replacement VibeThinker contender at `outputs/gspo_full_contenders/vibethinker_2511_continued_no_tables_20260521-083900` was initialized from the old GSPO `checkpoint-196`, with `--no-log-completions` and `MAX_STEPS=1598`. It produced `checkpoint-28` eval reward `0.31191`, `checkpoint-56` eval reward `0.30850`, `checkpoint-84` eval reward `0.30904`, and `checkpoint-112` eval reward `0.31448`.
  - True `--resume-from-checkpoint` worked technically but TRL preserved the old 28-step eval/save cadence from the checkpoint despite new `EVAL_STEPS=112`/`SAVE_STEPS=112`. To actually reduce validation-generation overhead, Vibe phase 2 was launched fresh from adapter `outputs/gspo_full_contenders/vibethinker_2511_continued_no_tables_20260521-083900/chanka_gspo/checkpoint-112` into `outputs/gspo_full_contenders/vibethinker_2511_phase2_from112_eval112_20260521-0909` with `MAX_STEPS=1486`, `EVAL_STEPS=112`, `SAVE_STEPS=112`, `WARMUP_RATIO=0.01`, and `--no-log-completions`.
  - Easy learned-verifier SFT was stopped early at `outputs/chanka_translation_verifier_full_r128/checkpoint-116` because validation loss was already `0.00244` on easy synthetic labels and more epochs were wasteful. Do not confuse this with a failed run; it was intentionally stopped after a good checkpoint.
  - Attempted learned-verifier GSPO at `outputs/gspo_paper_profiles/2511_learned_verifier_full_early116`; it failed on the first reward call because the verifier processor treated prompts as images. Commit `2f0f725` fixed this by passing `text=batch_prompts` explicitly. That failed output dir has no final metrics.
  - Hard-negative verifier SFT is running in parallel at `outputs/chanka_translation_verifier_hard_r128` to use idle GPU capacity; it has 8,674 verifier examples and validation every 57 steps. Latest checked checkpoint was `checkpoint-1083`; best eval loss remained `0.002326` from `checkpoint-912`, current eval loss was `0.002389`.
  - Hard-negative learned-verifier GSPO queue is alive via `outputs/queue_hard_learned_verifier_after_fulls.sh`; it waits until no `train_gspo_chanka_unsloth.py`/`train_verifier_chanka_unsloth.py` processes remain and at least one full-contender metrics file exists.
- Live GSPO status on `root@216.81.248.197 -p 20299` as of 2026-05-21 10:17 UTC:
  - Hard-negative verifier SFT reached `outputs/chanka_translation_verifier_hard_r128/checkpoint-1383` but crashed during final save with CUDA illegal memory access. Best checkpoint is `checkpoint-1368` with eval loss `0.0022854567505419254`; `checkpoint-1383` eval loss was `0.002302516484633088`. Remote symlink `outputs/chanka_translation_verifier_hard_r128/final_verifier_lora -> checkpoint-1368` was created for scripts that expect a final adapter path.
  - Broad-SFT learned-verifier canary `outputs/gspo_paper_profiles/2511_learned_verifier_hard_canary_20260521-0936` did not beat Vibe externally: chrF++ `35.77395`, BLEU `6.19078`, token F1 `17.04413`, source-copy `4.0551%`, exact-copy `1.5625%`, leakage `0.3906%`, TER `102.89017`, trainer eval reward `0.23386`.
  - Learned-verifier-on-Vibe canary `outputs/gspo_paper_profiles/2511_learned_verifier_on_vibe112_canary_20260521-1000` is the current best canary by external metrics: chrF++ `38.20628`, BLEU `7.28880`, token F1 `18.97321`, source-copy `4.1667%`, exact-copy `1.5625%`, leakage `0.3906%`, TER `96.53179`, trainer eval reward `0.24287`. It starts from Vibe `checkpoint-112` and uses the hard verifier `checkpoint-1368`.
  - `scripts/evaluate_gspo_checkpoint.py` evaluates arbitrary saved GSPO checkpoints on the clean Chanka validation split and writes external metrics/predictions. Use it for checkpoint selection because final-step metrics may miss an earlier stronger checkpoint.
  - `learned_verifier_vibe_2511` is available as a new mix profile: it uses the hard-negative verifier reward plus the VibeThinker group diversity bonus. Use `experiments/gspo/queue_learned_verifier_vibe_canaries.sh` after the current full contenders finish; it first evaluates selected full-run checkpoints externally, then runs 4-generation and 8-generation canaries. It requires `--verifier-adapter-path`.
  - Full learned-verifier-on-Vibe contender is running at `outputs/gspo_full_contenders/learned_verifier_on_vibe112_full_20260521-1016` from Vibe `checkpoint-112`, with hard verifier `checkpoint-1368`, `MAX_STEPS=224`, `EVAL_STEPS=56`, `SAVE_STEPS=56`, `LR=7e-7`, `NUM_GENERATIONS=4`, verifier batch size `2`, and no completion-table logging.
  - Full learned-verifier-on-Vibe checkpoints: `checkpoint-56` eval reward `0.25755`, eval loss `0.02497`, eval runtime `762.2s`; `checkpoint-112` eval reward `0.26144`, eval loss `0.04412`, eval runtime `768.6s`; `checkpoint-168` eval reward `0.25736`, eval loss `0.02710`, eval runtime `717.1s`. Checkpoint `112` remains the best learned-verifier full checkpoint so far and was preserved on the remote at `outputs/gspo_selected_checkpoints/learned_verifier_on_vibe112_checkpoint_112`.
  - Full learned-verifier-on-Vibe finished at `checkpoint-224` and wrote final metrics: chrF++ `37.84528`, BLEU `5.64736`, token F1 `21.86668`, source-copy `2.4367%`, exact-source-copy `0.6329%`, leakage `0.6329%`, TER `94.0092`, trainer eval reward `0.26116`. Checkpoint `224` was also preserved on the remote at `outputs/gspo_selected_checkpoints/learned_verifier_on_vibe112_checkpoint_224` because its eval loss `0.01096` was much better than checkpoint `112` even though its reward was slightly lower.
  - Vibe phase 2 continuation was stopped after sustained decline. It peaked at `checkpoint-896` with eval reward `0.33324`, then dropped to `0.33250` at `checkpoint-1008`, `0.32695` at `checkpoint-1120`, and `0.32441` at `checkpoint-1232`. A copy of `checkpoint-896` was preserved on the remote at `outputs/gspo_selected_checkpoints/vibethinker_phase2_checkpoint_896` before save rotation could delete it. Reward scales are profile-specific; final external metrics matter most.
  - A live external evaluation attempt of preserved checkpoints was stopped because it competed with the learned-verifier full run and had not written metrics yet. Let the queued post-training checkpoint evaluation run after `train_gspo_chanka_unsloth.py` exits, when it will not compete with the full contender.
  - Ignore checkpoint eval directory `outputs/gspo_checkpoint_evals/20260521-114759`: it used a batched evaluator before decoder-only left padding was forced and produced obviously invalid low metrics for the preserved Vibe checkpoint. Commit `30d339f` fixes the evaluator by setting `tokenizer.padding_side = "left"` before batched generation. Current clean queue stamp is `20260521-115135`.
  - Saved GSPO checkpoint reloads initially appeared to show a severe chat-template artifact failure: predictions repeated `user`, `assistant`, and `<think>` after a plausible short translation. Commit `3e6e01f` still adds useful `chat_artifact_penalty` reward guards and corpus metrics, but the main reload failure was later traced to an EOS mismatch: the processor/tokenizer EOS was `<|im_end|>` id `248046`, while the loaded model generation config used EOS id `248044`. Always pass `eos_token_id=tokenizer.eos_token_id` and align `model.generation_config.eos_token_id`/`pad_token_id` after loading Unsloth Qwen3.5 adapters.
  - Focused artifact follow-up queue `outputs/queue_artifact_penalty_followup_20260521-115915b.log` supersedes the broader queue. It evaluated learned-verifier checkpoint `168`, preserved checkpoint `224`, and `final_gspo_lora` as reloads. All failed with chrF++ around `20.2`, BLEU around `0.44`, TER around `676`, and `chat_artifact_penalty` around `58`. The re-summarized checkpoint eval dir is `outputs/gspo_checkpoint_evals/20260521-115915b`; selection is around `-30.8` after artifact penalty. The same queue then starts artifact-penalized `learned_verifier_vibe_2511` 4-gen and 8-gen canaries.
  - Artifact-penalized `learned_verifier_vibe_2511` 4-gen canary `outputs/gspo_paper_profiles/2511_artifact_guard_learned_verifier_vibe_4gen_canary_20260521-115915b` finished with in-memory final metrics: chrF++ `38.19053`, BLEU `7.22303`, token F1 `19.49405`, source-copy ratio `4.1667%`, exact-source-copy `1.5625%`, Spanish leakage `0.390625%`, chat artifact penalty `0.0`, TER `97.10983`, trainer eval reward `0.24584`. After the EOS fix, raw saved-adapter reload with `--batch-size 1` produced chrF++ `37.99378`, BLEU `8.22530`, token F1 `22.55735`, source-copy ratio `2.9114%`, exact-source-copy `0.6329%`, Spanish leakage `0.7911%`, chat artifact penalty `0.0`, and TER `92.62673` over 158 eval rows. This is now a serious contender.
  - Artifact-penalized `learned_verifier_vibe_2511` 8-gen canary started after the 4-gen canary at `outputs/gspo_paper_profiles/2511_artifact_guard_learned_verifier_vibe_8gen_canary_20260521-115915b`, but was stopped early after checkpoint `16` because eval reward stayed below the 4-gen run: checkpoint `8` reward `0.23739`, checkpoint `16` reward `0.23563`. Use 4-gen for the next contender unless a later hypothesis specifically justifies paying for 8-gen again.
  - EOS-fixed checkpoint ranking dir: `outputs/gspo_checkpoint_evals/20260521-eosfix-ranking`. Current best saved adapter is `learned_verifier_vibe_on_vibe896_4gen_canary_b1`, from canary `outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146`, with selection `26.3808`, chrF++ `40.9703`, BLEU `8.1555`, token F1 `26.6169`, source-copy `2.8270%`, exact-copy `0.6329%`, leakage `0.6329%`, artifact `0.0%`, and TER `88.7097`. Second place is raw `vibethinker_phase2_checkpoint_896` with selection `25.6181`, chrF++ `40.2643`, BLEU `5.8940`, token F1 `25.8484`, source-copy `2.5105%`, exact-copy `0.6329%`, leakage `0.7911%`, artifact `0.0%`, and TER `88.7097`.
- Full serious contender now running on the remote: `outputs/gspo_full_contenders/learned_verifier_vibe_on_vibe896_full_20260521-135419`, log `outputs/learned_verifier_vibe_on_vibe896_full_20260521-135419.log`, PID observed as `122513`. It starts from `outputs/gspo_selected_checkpoints/vibethinker_phase2_checkpoint_896`, uses reward `learned_verifier_vibe_2511`, verifier `outputs/chanka_translation_verifier_hard_r128/checkpoint-1368`, `MAX_STEPS=224`, `EVAL_STEPS=56`, `SAVE_STEPS=56`, LR `7e-7`, warmup ratio `0.01`, `NUM_GENERATIONS=4`, train/eval batch `4`, GA `2`, and full clean Chanka split of 897 train / 158 validation rows.
- Same full serious contender reached and preserved `checkpoint-168` at `outputs/gspo_selected_checkpoints/learned_verifier_vibe_on_vibe896_checkpoint_168`. Trainer eval reward was `0.27686325638684667`, eval loss `0.01147084217518568`. This is lower than checkpoint `112` reward `0.2869656496221506`; wait for batch-size-1 external checkpoint eval before choosing among 56/112/168/224/final.
- The full serious contender finished on 2026-05-21 15:15 UTC and preserved `checkpoint-224` at `outputs/gspo_selected_checkpoints/learned_verifier_vibe_on_vibe896_checkpoint_224`. In-memory final metrics: chrF++ `40.12930518900551`, BLEU `6.0075693712342275`, token F1 `25.644966847498495`, source-copy ratio `2.668776371308017%`, exact-copy `0.6329113924050633%`, Spanish leakage `0.7911392405063291%`, chat artifact penalty `0.0`, TER `89.86175115207374`, trainer eval reward `0.2858995793481605`. These are not enough for selection; queued batch-size-1 reload eval `outputs/gspo_checkpoint_evals/20260521-vibe896-full-eval` is now running.
- External batch-size-1 eval `outputs/gspo_checkpoint_evals/20260521-vibe896-full-eval` completed. The full run was clean but did not beat the shorter canary. Ranking: canary selection `26.3808`, chrF++ `40.9703`, BLEU `8.1555`, token F1 `26.6169`; raw Vibe896 checkpoint selection `25.6181`; full-run checkpoint `56` selection `25.3947`; checkpoint `224`/final selection `25.3835`; checkpoint `168` selection `25.2477`; checkpoint `112` selection `25.2292`. Treat the canary as the current best adapter for the verifier-v2 follow-up.
- Important verifier status: we have tried a learned verifier, but so far it is verifier v1 trained mostly from clean pairs plus synthetic corruptions and hard negatives. The fuller DeepSeek-style next step is verifier v2 trained from real model candidate outputs. Queue script `experiments/gspo/queue_verifier_v2_from_real_candidates.sh` now waits for the current full run and its external checkpoint evals, trains `outputs/chanka_translation_verifier_real_candidates_v2_20260521-verifier-v2-real` using prediction JSONLs from `outputs/gspo_checkpoint_evals/20260521-eosfix-ranking` and `outputs/gspo_checkpoint_evals/20260521-vibe896-full-eval`, then runs a 24-step canary with `learned_verifier_vibe_2511` using the verifier-v2 LoRA. Remote queue log: `outputs/queue_verifier_v2_from_real_candidates_20260521.log`.
- Commit `5679ace` hardened `experiments/gspo/queue_verifier_v2_from_real_candidates.sh`: if verifier-v2 SFT exits nonzero after writing usable checkpoints, the queue now falls back to `final_verifier_lora` when present or otherwise the checkpoint with best logged `eval_loss`. This matters because the previous hard-negative verifier run produced good checkpoints but crashed during final save with CUDA illegal memory access.
- Verifier-v2 real-candidate training started on the remote at `outputs/chanka_translation_verifier_real_candidates_v2_20260521-verifier-v2-real`, PID observed as `127123`. It uses prediction JSONLs from the EOS-fixed ranking plus the full Vibe896 eval. After dedupe it added only 305 real candidate verifier examples, for 8,979 total verifier examples, split into 7,633 train / 1,346 validation. Schedule: 4 epochs, 1,912 total optimizer steps, eval every 59 steps, LoRA r128/a256, LR `2e-5`, batch 4, gradient accumulation 4.
- Verifier-v2 early checkpoints are improving: checkpoint `59` eval loss `0.04840836673974991`, checkpoint `118` `0.0312885046005249`, checkpoint `177` `0.020037654787302017`, checkpoint `236` `0.021803665906190872`, checkpoint `295` `0.017204049974679947`, checkpoint `354` `0.015418198890984058`, checkpoint `413` `0.01523187942802906`, checkpoint `472` `0.015611322596669197`. Preserved best-so-far copies on the remote under `outputs/verifier_selected_checkpoints/`, including current best checkpoint `413`.
- Verifier-v2 later improved to checkpoint `531` eval loss `0.013897795230150223`, checkpoint `590` `0.014394257217645645`, checkpoint `649` `0.013841227628290653`, and checkpoint `708` `0.01347315777093172`. Commit `89fa4e1` adds `experiments/gspo/watch_verifier_best_checkpoint.sh` and expands the queue fallback to search `outputs/verifier_selected_checkpoints/`. The watcher is running on the remote as `outputs/watch_verifier_v2_best_20260521.log` and maintains `outputs/verifier_selected_checkpoints/chanka_translation_verifier_real_candidates_v2_20260521-verifier-v2-real_best`.
- DeepSeek-style verifier-v2 was actually tried. The real-candidate verifier was stopped after overfit signs; best preserved checkpoint was `outputs/verifier_selected_checkpoints/chanka_translation_verifier_real_candidates_v2_20260521-verifier-v2-real_checkpoint-944` with eval loss `0.012264552526175976` (later checkpoints `1003` and `1062` worsened). A 24-step `learned_verifier_vibe_2511` GSPO canary from the current best adapter completed at `outputs/gspo_paper_profiles/2511_verifier_v2_real_candidates_on_best_20260521-verifier-v2-real`. Canonical batch-size-1 eval `outputs/gspo_checkpoint_evals/20260521-verifier-v2-real-verifier-v2-canary` did not beat the previous best: best checkpoint `24`/final selection `25.4932`, chrF++ `40.1668`, BLEU `5.9356`, token F1 `25.6148`, source-copy `2.5105%`, exact-copy `0.6329%`, leakage `0.7911%`, artifact `0.0%`, TER `88.9401`. Previous best remains `learned_verifier_vibe_on_vibe896_4gen_canary` with selection `26.3808`, chrF++ `40.9703`, BLEU `8.1555`, token F1 `26.6169`.
- Commit `382e88a` extends `scripts/evaluate_gspo_checkpoint.py` for verifier candidate mining: `--split train|eval|all`, `--max-train-samples`, `--do-sample`, and `--num-return-sequences`. Use this to build a larger verifier-v3 pool from train/all Chanka rows if verifier-v2 does not beat the current canary. Example pattern: evaluate the current best adapter with `--split train --max-train-samples 512 --do-sample --num-return-sequences 4 --temperature 0.85 --top-p 0.95 --batch-size 1 --predictions-jsonl ...`; then pass that JSONL to `scripts/train_verifier_chanka_unsloth.py --candidate-jsonl`.
- Verifier-v3 sampled-candidate queue script: `experiments/gspo/queue_verifier_v3_sampled_candidates.sh`. It mines sampled candidates from the current best adapter over `--split all` with two temperature/top-p settings by default, trains a verifier with up to 12k real candidate examples, runs a 24-step `learned_verifier_vibe_2511` canary from the current best adapter, then batch-size-1 evaluates checkpoint 8/16/24/final. Launched on the remote with `STAMP=20260521-verifier-v3-sampled`; log `outputs/queue_verifier_v3_sampled_candidates_20260521.log`, candidate dir `outputs/verifier_candidate_mining/20260521-verifier-v3-sampled`, verifier dir `outputs/chanka_translation_verifier_sampled_candidates_v3_20260521-verifier-v3-sampled`, canary dir `outputs/gspo_paper_profiles/2511_verifier_v3_sampled_candidates_on_best_20260521-verifier-v3-sampled`, eval dir `outputs/gspo_checkpoint_evals/20260521-verifier-v3-sampled-verifier-v3-canary`.
- Verifier-v3 completed and improved v2 but did not beat the standing best. Candidate mining produced two 4,220-row sampled sets: `temperature=0.75/top_p=0.90` scored chrF++ `38.5675`, BLEU `8.0288`; `temperature=0.95/top_p=0.95` scored chrF++ `34.2871`, BLEU `5.0552`. Verifier-v3 trained on 7,150 deduped real candidate examples plus synthetic/corruption rows; best verifier eval loss was checkpoint/final `1200` at `0.07788554579019547`. Its best GSPO canary checkpoint was `checkpoint-16` in `outputs/gspo_checkpoint_evals/20260521-verifier-v3-sampled-verifier-v3-canary`: selection `25.7042`, chrF++ `40.5972`, BLEU `5.8257`, token F1 `26.1197`, source-copy `2.6688%`, leakage `0.6329%`, TER `89.6313`. This beats verifier-v2 (`25.4932`) but not current best (`26.3808`).
- `learned_verifier_ensemble_vibe_2511` was added to test v1+v3 verifier ensembling with Vibe diversity. It loads a primary verifier via `--verifier-adapter-path` and a secondary verifier via `--secondary-verifier-adapter-path`. Canary `outputs/gspo_paper_profiles/2511_verifier_ensemble_v1_v3_on_best_20260521-verifier-ensemble-v1-v3` underperformed: best checkpoint `24` selection `25.2932`, chrF++ `40.1285`, BLEU `5.8648`, token F1 `25.1627`, source-copy `2.6688%`, leakage `0.7911%`, TER `89.6313`. Do not full-run this ensemble.
- Low-LR refinement canary script: `experiments/gspo/run_low_lr_refinement_canary.sh`. It continues from the current best adapter using the original `learned_verifier_vibe_2511` reward but with LR `2e-7`, warmup `0`, and `16` max steps. Remote run `outputs/gspo_paper_profiles/2511_low_lr_refine_current_best_20260521-low-lr-refine-current-best` did not beat the standing best: best checkpoint `16` selection `25.9950`, chrF++ `40.3932`, BLEU `8.4617`, token F1 `25.8886`, source-copy `2.6688%`, leakage `0.7911%`, TER `88.9401`. This has the highest BLEU seen so far, but lower chrF++/token-F1 and lower selection than current best.
- BLEU-margin reward profile `learned_verifier_bleu_margin_vibe_2511` and canary script `experiments/gspo/run_bleu_margin_canary.sh` were tested. It keeps the v1 learned verifier but changes the guard term toward chrF++/token-F1 plus a small BLEU/token-precision margin. Remote run `outputs/gspo_paper_profiles/2511_bleu_margin_current_best_20260521-bleu-margin-current-best` underperformed: best checkpoint `8` selection `25.5784`, chrF++ `40.3062`, BLEU `5.8180`, token F1 `25.7731`, source-copy `2.5105%`, leakage `0.6329%`, TER `89.4009`. It did not preserve the high BLEU from the low-LR run and should not be full-run.
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
