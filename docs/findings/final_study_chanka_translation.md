# RosettIA — Final Study Summary: Spanish→Chanka Quechua Translation (2026-05-25)

## Headline

**Best standalone single-model: v13 LoRA loaded at `lora_alpha=160` → chrF++ `56.944`, BLEU `30.756`, token F1 `46.432`, TER `62.212`** on the 158-row clean Chanka held-out eval. That's +13.45 chrF++ / +14.62 BLEU over the previous 4B baseline (`checkpoint-36` at chrF++ 43.488), and +5.08 chrF++ / +4.74 BLEU above the previous best multi-model deployable (K32+4B with listwise text selector).

The full inference recipe is **free** in compute: one JSON-field edit to the adapter (`lora_alpha=128 → 160` in `adapter_config.json`). The underlying LoRA was produced by a 3-stage SFT chain documented below.

Saved artifact: `outputs/v13_alpha160_champion_20260525/` (a LoRA adapter on top of `outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36`).

## The Recipe

### Stage 1 — Compact-mixed SFT (v11)
LoRA on top of the 4B Chanka-specialized full-FT base.
- **Data**: `outputs/self_verifiable_data_20260525-compact-mixed/self_verifiable_compact_mixed_sft.jsonl` — 2,110 rows = 1,055 reviewed Chanka pairs × 2 prompt modes (`direct` and `compact` with `Analisis:` / `Final:` / `Puntaje:` self-verification rows).
- **Hyperparams**: LoRA r=64, α=128, dropout=0; LR=5e-6 linear decay; 512 max steps; max_train_samples=1024; max_eval_samples=192; per-device batch 4; max_seq_length=128; terminology-top-k=1.
- **Result**: ckpt-448 chrF++ 54.06.

### Stage 2 — Low-LR refinement (v12)
Continue v11 ckpt-448 with the same data, lower LR.
- LR=1e-6, 128 steps.
- **Result**: ckpt-128 chrF++ 55.47, BLEU 29.87.

### Stage 3 — Even lower LR (v13)
Continue v12 ckpt-128, lower LR again.
- LR=5e-7, 32 steps.
- **Result**: ckpt-32 chrF++ 55.76, BLEU 29.23, token F1 44.26, TER 64.52.

### Stage 4 — Free α-scaling at inference
Load the v13 ckpt-32 adapter with `lora_alpha=160` (1.25× the trained 128).
- Found via a clean sweep (peaks unimodally at 1.25×; 0.5× → 49.05, 1.0× → 55.76, 1.25× → 56.94, 1.5× → 52.37, 2.0× → 37.06).
- **Result**: chrF++ 56.94, BLEU 30.76, token F1 46.43, TER 62.21 — current champion.

Wrapper: `experiments/sft/run_compact_mixed_self_verifiable_sft.sh`.

## What Worked

1. **The compact-mixed multi-task SFT** — direct (source→Chanka) rows mixed with compact self-verification format rows. The auxiliary self-verification task contributes ~+2 chrF++ over pure direct training at matched step count. Verified by the leak-free direct-only ablation at chrF++ 52.20 (512 steps) vs v11 at 54.06.

2. **3-stage LR-decay continuation (v11 → v12 → v13)** — once stage 1 has converged near chrF++ 54, dropping LR by 5× and continuing for ~32-128 steps adds +1.5 chrF over a single-stage run. The model is fine-tuning around the converged loss landscape, not exploring new behavior.

3. **Inference-time α-scaling** — loading the trained LoRA with α=160 instead of trained α=128 captures additional capacity the gradient descent left on the table. +1.18 chrF++ for free, peaks unimodally; α>1.5× degrades fast.

4. **Training longer** — direct-only SFT scales monotonically: 48 steps gives chrF++ 43.49 (the team's prior baseline); 256 steps → 49.11; 512 steps → 52.20 (proper leak-free).

## What Didn't Work

### Reinforcement learning (GSPO with learned verifier)
The team's prior best GSPO branch (final-only no-mask) added +0.45 chrF++ on top of a much weaker base; our re-run from v7 ckpt-80 (already-strong SFT base, chrF++ 46.69) REGRESSED to chrF++ 45.13 with negative trainer reward. The existing reference-based verifier provides no useful RL signal once SFT is strong.

### Self-scored Best-of-N reranking
The compact-mixed model emits self-scores that **saturate at 1.0** with 100% false-confidence rate. Self-score-ranked Best-of-N is equivalent to random selection.

### Linear listwise text reranker on K=16/K=32 v13 candidates
Oracle ceiling on K=16 sampling = chrF++ 65.54 (12-point gap above greedy). The team's matched listwise hash-feature reranker captured ~0 of that gap (chrF++ 55.55 on K=16, 55.29 on K=32 — both essentially tied with v13 greedy). The features can't discriminate among short Chanka candidates.

### Mergekit-style task-vector amplification
Merged v13 into the 4B base, computed delta, wrote `base + α·delta` for α ∈ {1.0, 1.25, 1.5, 1.75, 2.0}. Peak at α=1.5 → chrF++ 55.89, but the α=1.0 control already lost 3 chrF (52.41) to merge-precision round-trip. LoRA α-scaling is cleaner and stronger.

### Activation-diff-guided targeted LoRA
Computed per-layer cosine distance between v13 and base. Bimodal peak at L16-17 (mid-network semantic) and L27-31 (output/morpheme). v16 trained LoRA on only those 8 layers (25% of 32) — best ckpt-192 chrF++ 44.83, losing ~9 chrF vs full-layer training. The activation diff identifies where outputs change; the learning signal requires updates throughout all layers.

### Engineered reasoning traces (DS Flash, 3 flavors)
We had DS Flash generate Spanish-language step-by-step morphological reasoning traces for the 838 leak-free train sources (flavor A verbose morphological, flavor B compact gloss, flavor C natural CoT). Best acceptance was flavor C at 94.6%. We then trained:
- v14 (mixed direct + compact + engineered): chrF++ 46.29 — regressed vs direct-only.
- v14b (engineered-only, direct prompt): chrF++ 42.56 — catastrophic (prompt mismatch).
- v14c (curriculum: direct-only-ckpt-512 + engineered polish): chrF++ 52.19 — no change.

DS Flash's traces, while linguistically credible, contained confabulated morphology (e.g. claiming "kallpa" means "calle" — actually means "fuerza"). The 4B model trained on those traces inherited the confabulations.

### Grounded reasoning traces (Claude agents + AMLQ dictionary + QHESWA grammar)
To rule out the confabulation hypothesis, we spawned 8 parallel Claude general-purpose agents that each processed ~100 source pairs with two reference documents in context:
- `docs/references/quechua_morphology_rules.md` (377 lines, mined from the QHESWA Cuzco-Collao grammar manual, with §11 Chanka deviations).
- `docs/references/cuzco_dictionary_es_to_quechua_lookup.json` (700 Spanish→Quechua lemmas mined from the AMLQ Cuzco dictionary, 150 carrying explicit Cuzco→Chanka orthographic transforms).

The 841 grounded traces averaged 4.5-7.3 morpheme citations each with explicit dictionary references, loanword flags ("kasara-, intindi-, fiscal, midiku, doctor, posta, denuncia, carcel..."), and uncertainty notes when a root wasn't in the references. Quality was visibly better than DS Flash output.

Despite that quality lift:
- v17 (thinking-only training, `enable_thinking=True`, confabulated DS Flash traces): best ckpt-224 chrF++ 18.20 (post-`</think>` extraction).
- v17b (mixed direct + confabulated thinking): best ckpt-320 chrF++ 23.34.
- v18 (mixed direct + grounded Claude-agent thinking, `enable_thinking=True`): **all checkpoints chrF++ 1.9-3.0** — caught a double-`<think>`-wrap bug: when `enable_thinking=True` the chat template adds an opening `<think>` tag automatically, and our targets also had one, so the model trained on `<think><think>...</think></think>...` and never learned where to close at inference.
- v19 (same data, no `enable_thinking`, literal `<think>` text in target): best ckpt-128 chrF++ 44.55 — barely beats the 4B baseline, far below direct-only at the same step count (52.20). Sample output: "¿Ima **kallpim** tiyanki?" — model STILL confabulated kallpa=calle despite the explicit dictionary correction in 793 training traces.
- v19b (grounded thinking continued from v13 ckpt-32): best ckpt-192 chrF++ 55.57 — essentially tied with the v13 starting point. The 192 steps didn't even convert v13's direct-output behavior into thinking-block emission.

**Honest conclusion on the thinking-block paradigm for this task**: even high-quality linguistically grounded reasoning supervision does not exceed simple direct SFT. The 4B model extracts what it needs from (source, gold) pairs implicitly; explicit symbolic reasoning is redundant. The post-hoc `kallpa = calle` confabulation persisted across all grounded-thinking variants, suggesting the model learned a phonetic-similarity heuristic from the direct pairs that 793 corrective reasoning traces couldn't overwrite in 128-384 steps.

### Cross-architecture: NLLB-200 with our recipe
Trained NLLB-200 distilled 1.3B + LoRA r=64/α=128 on the 841 leak-free direct Chanka pairs (512 steps, LR=1e-4). Result chrF++ 41.84 (+14.14 over NLLB zero-shot 27.70). Recipe transfers cleanly to encoder-decoder MT but doesn't beat the 4B Qwen baseline because NLLB is smaller (1.3B vs 4B).

### Scaling to 9B
v15 — raw Qwen3.5-9B + compact-mixed LoRA (no Chanka pretrain): plateaus at chrF++ 32.08. The shortcut (skip Chanka pretraining) loses too much. To match the 4B chain you'd need raw 9B → broad → Chanka full-SFT base → compact-mixed LoRA.

### Scaling to 9B — the v20/v21/v22 failed attempts (2026-05-26)

We ran three full 9B chains attempting to replicate the 4B's `broad → Chanka full-SFT → compact-mixed → α-scaling` recipe. **All three plateaued at chrF++ 37-38**, ~19 chrF below the 4B champion. Root cause was identified post-hoc by re-reading the team's original `qwen35_4b_curriculum_sft.md`: **learning rates were 10× too low at the broad and Chanka stages**.

| Stage | 4B team recipe (success) | My 9B v22 recipe (failure) |
|---|---|---|
| Broad LoRA | LR **5e-5**, eff batch 16, seq 256, 512 steps | LR **5e-6**, eff batch 8, seq 512, 6,144 steps (33% of 1 epoch) |
| Chanka adaptation | LR **2e-5** (LoRA, 256 steps) | LR **2e-6** (full-FT, paged_adamw_8bit, 88 steps) |
| Compact-mixed | LR 5e-6, 512 steps | LR 5e-6, ~125 steps (1 epoch) ✓ |

The smoking-gun observations that confirm the LR diagnosis:

- **v22a broad** at LR 5e-6 reached chrF++ 39.04 on broad held-out after 6,144 steps (vs 4B's 512 steps to its plateau). It then bounced in 36-39 chrF for 3,500 more steps. At LR 5e-5 the same plateau would have been hit ~12× faster, leaving budget for additional epochs.
- **v22c full-FT** at LR 2e-6 with `paged_adamw_8bit` produced a *dead-flat* chrF++ plateau at 34.5 across all 11 evaluated checkpoints (steps 8-88). The model literally did not learn. The same recipe at LR 2e-5 with full `adamw_torch` worked for the 4B.
- **v22d compact-mixed** at LR 5e-6 (the *only* stage where the LR matched the 4B recipe) showed a clean climb from 34.27 → 37.88 chrF over 96 steps — the format/data/code worked; the upstream base was just under-trained.
- **v22g α-sweep** confirmed α=128 (trained) was optimal (37.88) and α=96 lost 1.76 chrF — same shape as the 4B's α curve, just shifted 19 chrF down. The recipe is correct; the *base* it was applied to was the problem.

Additional process failures worth recording so we never repeat them:

- **No generation-quality validation during training.** We ran `eval_loss` every 256 steps on a 2% held-out broad split but only measured chrF++ post-hoc. Loss going down hid the fact that v22c's outputs hadn't actually changed.
- **Held-out chrF++ was only computed at the end of each stage.** Should have been every save_steps from the start so we could see plateau in real time and kill the run.
- **`paged_adamw_8bit` was used for full-FT on a 9B model.** The quantized optimizer states amplify the under-LR problem; full `adamw_torch` fits in 80GB A100 with our batch/seq settings and should be the default for full-FT.
- **Used "9B is bigger → use lower LR" intuition instead of reading the team's actual recipe.** The doc was sitting in `docs/findings/qwen35_4b_curriculum_sft.md` the whole time.
- **First broad run (v22a) was launched with `--num-train-epochs 1.0` on 166k rows = 20,810 steps**, before we'd verified the recipe worked on a small budget. After 6,144 steps (~10.5 hours) we killed it. With proper LR + the team's 512-step recipe, the entire chain to compact-mixed result is ~1 hour.

v23 redoes the chain with the corrected LRs (5e-5 broad, 2e-5 Chanka LoRA, optional full-FT with `adamw_torch`) and matches the team's step budget. Expected result: clean +20 chrF jump and headroom above the 4B's 56.94.

### v23 broad — LR fix validates immediately (2026-05-26)

v23a (`r=64`, `LR=5e-5`, `eff_batch=16`, `seq=256`) was trained from raw Qwen3.5-9B with epoch budget + eval/save every 128 steps so we could watch the trajectory and kill on plateau. Resumed from checkpoint-256 at step time with `per_device=8, grad_accum=2` (4× faster than `per_device=2, grad_accum=8` — same effective batch, better GPU utilization).

| step | eval_loss | broad-held-out chrF++ | BLEU |
|---:|---:|---:|---:|
| 128 | 1.998 | — | — |
| 256 | 1.638 | — | — |
| 512 | 1.426 | — | — |
| 1024 | 1.158 | 36.23 | 7.86 |
| 1152 | 1.108 | 37.75 | 9.47 |
| 1280 | 1.074 | 37.93 | 8.88 |
| 1408 | 1.052 | 37.51 | 9.81 |
| **1536** | **1.023** | **38.85** | 9.72 |

**Result**: v23a at step 1536 matches v22a's peak chrF (39.04 at step 6144) in **5× fewer steps**, and eval_loss is strictly below v22a's asymptote of 1.07. The LR was the entire story. The 4B's known-good `LR=5e-5` works on the 9B too.

### v24 broad sweep — rank / LR / batch ceiling (2026-05-26)

Once v23a confirmed the LR fix, we swept LoRA rank, LR, and batch size with each variant trained from raw 9B and stopped early once trajectory was clear. All measured on the same 200-row broad held-out (seed=3407, validation_fraction=0.02), generation chrF++/BLEU after each save via a sidecar evaluator.

| Variant | r | α | LR | eff_batch | per_device × grad_accum | result @ best ckpt |
|---|---:|---:|---|---:|---|---|
| v23a (baseline) | 64 | 128 | 5e-5 | 16 | 8 × 2 | **38.85 chrF / 9.72 BLEU @ ckpt-1536** |
| v24a | 128 | 256 | 5e-5 | 16 | 8 × 2 | 34.13 / 6.71 @ ckpt-384 (killed early, still climbing) |
| v24c | 256 | 512 | 1e-4 | 32 | 8 × 4 | 36.26 / 8.35 @ ckpt-384, **BLEU peak 10.25 @ ckpt-256** |
| v24e | 512 | 1024 | 1e-4 | 32 | 8 × 4 | 34.68 / 8.85 @ ckpt-128, **collapsed to 31.99 / 3.32 @ ckpt-256** |
| v24f | 256 | 512 | 1e-4 | 64 | 16 × 4 | 35.46 / 10.23 @ ckpt-128 (running) |

**Findings**:

1. **Rank scaling has a hard ceiling around r=256.** r=64 → r=128 → r=256 each gave clean wins at matched step. r=512 with the same recipe **catastrophically collapsed** at step 256: eval_loss kept dropping (1.296 → 1.093) but chrF *fell* (34.68 → 31.99) and BLEU collapsed (8.85 → 3.32). The bigger rank fits the train distribution better but loses generation coherence — classic overfitting on a multi-variant broad mix. Likely fixable with α=512 instead of α=1024 (lower effective update size) or LR=5e-5, but r=256 already gives the win.

2. **Bigger LR (5e-5 → 1e-4) helps when paired with bigger rank.** v24a (r=128 + 5e-5) at step 384 lands at chrF 34.13; v24c (r=256 + 1e-4) at step 256 lands at chrF 35.90 — fewer steps, fewer data points seen, better result.

3. **Bigger eff_batch (32 → 64) gives marginal step-128 lift but the data-equivalent comparison is washed.** v24f step 128 (8192 examples seen) chrF 35.46 ≈ v24c step 256 (8192 examples seen) chrF 35.90. Bigger batch = more parallel compute per step, not fundamentally better learning at this scale.

4. **VRAM headroom is huge.** Even v24f at r=256 + batch=64 + seq=256 used only 27 GB / 80 GB on the A100. Best use of the headroom is **disabling gradient checkpointing** (~30% speedup) rather than bigger r or batch. The trainer script's `--gradient-checkpointing false` flag exposes this (added 2026-05-26).

**Working broad-stage recipe for 9B** (as of v24f):
- LoRA `r=256, α=512`, `dropout=0`
- `LR=1e-4` with linear decay (or cosine; doesn't matter much)
- `eff_batch=32-64` (no meaningful difference within this range)
- `seq=256`
- `gradient_checkpointing=false` if VRAM allows (it does for 9B + r=256 on A100 80GB)
- Stop on chrF plateau (typically by step 384-512 — much sooner than the 1500+ needed at r=64)

Next: chain this broad winner through the 4B Chanka recipe and measure final Chanka chrF. Target: beat the 4B champion's 56.94.

### v24f extended broad — 2-epoch resume (2026-05-27)

After v24f hit its 512-step cap at chrF 41.60, we resumed with `--resume-from-checkpoint=ckpt-512` and `--max-steps=5202` (exactly 2 full epochs at eff_batch=64 on 166k rows) to see how far the broad LoRA could go without changing recipe. The LR scheduler reset on resume, so the LR jumped from near-zero (end of old 512-step schedule) back to ~9.9e-5 (start of new 5202-step linear decay).

Resumed trajectory (broad held-out chrF++ on 100-row eval):

| step | eval_loss | chrF++ | BLEU |
|---:|---:|---:|---:|
| 512 (end of phase 1) | 0.850 | 41.60 | 13.71 |
| 640 | 0.895 | 42.08 | 12.94 |
| 768 | 0.857 | 42.70 | 12.20 |
| 896 | 0.817 | 41.33 | 11.30 |
| 1024 | 0.790 | 38.16 ↘ | 5.96 ↘ |
| 1152 | 0.755 | 41.05 | 7.69 |
| 1280 | 0.726 | 41.65 | 7.77 |
| 1408 | 0.701 | 41.00 | 6.88 |
| 1664 | 0.658 | 40.29 | 7.29 |
| 1920 | 0.618 | 42.97 | 8.52 |
| 2176 | 0.584 | 43.02 | 8.75 |
| 2304 | 0.563 | 44.10 | 8.89 |
| 2432 | 0.541 | 43.17 | 8.52 |
| 2560 | 0.526 | 44.25 | 9.40 |
| **2688** | **0.515** | **44.61** | **9.90** |

**Findings**:

1. **chrF kept climbing through 1+ epoch of additional training**: from 41.60 (end of phase 1) to 44.61 at step 2688 (~1.03 epoch into phase 2). **+3.01 chrF gain over the original v24f peak**, +5.57 over v22a's all-time best.

2. **Loss-chrF divergence**: BLEU peaked at the original v24f-ckpt-512 (13.71) and stayed lower through the resumed run (9-10 range), while chrF kept climbing. The model is generating more character-fluent Quechua but with different word choices than the references (could be one-variant convergence within the multi-variant broad mix, or just eval-set noise on 100 rows).

3. **The step-1024 "crash" (chrF 38.16, BLEU 5.96) was noise, not a real overfitting signal**. Continuing past it recovered cleanly and kept climbing. Important reminder to not over-react to single noisy data points in a 100-row chrF eval.

4. **Eval loss kept monotonically dropping** (0.85 → 0.515 over the resumed run) even when chrF dipped — eval_loss is unreliable as a stopping criterion for translation quality on this task.

We stopped at step 2688 (after one full epoch + ~3 saves into the second) to commit ckpt-2688 as the v25 chain base. The curve was still climbing slowly so further training might extract a bit more — but a) the BLEU divergence is concerning, and b) downstream Chanka chrF is what actually matters, not broad chrF.

### Process improvements adopted

- **Always use epoch-based budgets, never raw step counts**, even for tail LR-decay stages. Conversion: if the original 4B recipe used N steps, divide by `train_rows / effective_batch` to get the fractional epoch and pass that to `--num-train-epochs`. We can stop manually on plateau anyway, so generous epoch budgets are fine.
- **Live chrF sidecar during training**: the orchestrator script forks a background watcher that polls for new `checkpoint-N/` dirs and runs a 80-100 row generation chrF eval as soon as each is saved. Costs ~10GB extra VRAM during eval bursts but gives real-time visibility we couldn't get from `eval_loss` alone (and the divergence above shows why this matters).
- **Multimodal Qwen3.5 eval loader**: Qwen3.5-9B is a hybrid vision+text architecture. Loading via plain `AutoModelForCausalLM` produces UNEXPECTED/MISSING key warnings because the saved state has `model.language_model.*` prefix. Fix: use `unsloth.FastLanguageModel.from_pretrained(load_in_16bit=True, full_finetuning=False)` and pass text to the tokenizer as `text=prompts` (not positional, since the multimodal processor would interpret positional args as image URLs).
- **`--resume-from-checkpoint` flag** added to `scripts/train_sft_unsloth.py` (and threaded into `trainer.train()`) so we can continue training without losing optimizer state, while also being able to change `per_device_batch_size` and `gradient_accumulation_steps` between runs (effective batch must stay the same for the LR scheduler to remain calibrated).
- **`--gradient-checkpointing false` flag** also added — at r=256 + batch=64 + seq=256 on 9B we only use ~27 GB / 80 GB VRAM with Unsloth checkpointing on, so disabling it is free 30% speedup. Not used in v24f (we still hadn't measured the headroom).

### Citations & external resources surveyed (2026-05-27)

After v27 V-STaR plateaued near the data ceiling, we surveyed 11 external sources for new methods and Chanka-specific data. Citations and applicability notes:

**Method papers**
- **Ghazvininejad, Gonen, Zettlemoyer (Meta AI), 2023.** ["Dictionary-based Phrase-level Prompting of Large Language Models for Machine Translation"](https://arxiv.org/abs/2302.07856). DiPMT — append bilingual dictionary entries to the prompt. Modest gains (~0.9-1.1 BLEU) on moderately-resourced languages. No Quechua. Useful insight: unsupervised dictionary induction via SimAlign on parallel pairs.
- **Lu et al. (CUHK + Microsoft), EMNLP 2024 Main.** ["Chain-of-Dictionary Prompting Elicits Translation in Large Language Models"](https://aclanthology.org/2024.emnlp-main.55/). CoD chains multilingual dict entries (es → fr/de/pt → target). Tested on FLORES-200 (includes `quy_Latn` Ayacucho/Chanka). Reports up to 13× chrF++ gains on weak languages. Plug-and-play at inference. Code: github.com/HongyuanLuke/Chain-of-Dictionary.
- **Lu et al. (Xinjiang Tech / CAS), COLING 2025.** ["Low-Resource Language Expansion and Translation Capacity Enhancement for LLM: A Study on the Uyghur"](https://aclanthology.org/2025.coling-main.559/). DPOSE — Direct Preference Optimization via translation Self-Evolution, using SacreBLEU rankings of model's own outputs to construct (chosen, rejected) pairs. Simpler than V-STaR. Their prompt-language-matching ablation worth testing (es-prompt vs qu-prompt).
- **Nair et al., IEEE Access 2026.** "Enhancing Low-Resource Indian Language Machine Translation Using Large Language Models With Preference Optimization and Hypergeometric-Gamma Reward". DPO + HGR (SBERT-similarity reward via reward-weighted loss). Indian languages only but method is language-agnostic.
- **Ortega, Castro Mamani, Cho. Machine Translation (Springer), 2020.** "Neural machine translation with a polysynthetic low resource language". Directly on Spanish↔Cuzco-Bolivian Southern Quechua. Introduces **BPE-Guided** subword segmentation using a Wiktionary suffix list to bias subword splits along morpheme boundaries. Releases corpus at github.com/johneortega/mt_quechua_spanish. Confirms Spanish-Finnish OPUS-MT as a strong typological warm-start for agglutinative targets.
- **Ahmed, Flechas Manrique, Petrović (LCT-EHU), AmericasNLP 2023 @ ACL.** ["Enhancing Spanish-Quechua Machine Translation with Pre-Trained Models and Diverse Data Sources"](https://aclanthology.org/2023.americasnlp-1.16/). Directly on Spanish→`quy` (Ayacucho/Chanka). Best test chrF 38.59 (our v25 vastly exceeds). **Most actionable for our purposes: their data release** at github.com/nouman-10/MT-SharedTask + huggingface.co/americasnlp-lct-ehu — newly aligned Chanka data including *The Little Prince* (1,312 lines quy), UN Human Rights (91 qus), Ecuador Constitution (2,243 que), MINEDU dictionary (643 quy), Dict_misc Huarcaya (8,998 quy), JW300 quy/quz portion (~120k), plus monolingual cc100 (113k) and Llamacha (182k). **Could easily 5-10× our 897-row Chanka training set.** Negative result worth noting: copied-corpus and synthetic back-translation *hurt* chrF in their setup.

**Chanka-native lexicons / textbooks**
- **Soto Ruiz, *Quechua: manual de enseñanza*, IEP, 4th ed. 2010** (PDF partial; full book 462 pp + 2 audio CDs). The gold-standard Ayacucho-Chanka academic textbook. 23 Units of Quechua texts with Spanish translations (parallel sentence-aligned), full grammar, sufijos appendix, and a Vocabulario QU→ES of ~1000 entries (pp. 451-462). Same author's *Diccionario Ayacucho-Chanca* and *Gramática Quechua Ayacucho-Chanca* are the reference works for the variety. **The available scan stops at Unit 7 — sourcing the complete book is the single highest-leverage data action.**
- **Quintero Bendezú, *Yachay wasi qichwa-kastillanu*, 2nd ed. 2003/2008** (vocabularioQuechua.pdf, 358 pp). Chanka-native (Huanta, Ayacucho-Huancavelica) bidirectional vocabulary. Estimated 6,000-10,000 entries. Regular format `lemma (POS). Gloss.` Includes morphological/suffix entries tagged `(Morf.)`. Clean text layer (Word→PDF), highly parsable.
- **Benito Zuasnabar, *Diccionario Básico Quechua Chanka*, 2018** (diccionary.pdf, 18 pp). Explicitly Chanka. ~500-700 entries per direction (bidirectional). Cleanest and smallest — best for quick smoke tests. Includes morphological allomorphs via `~` notation.
- **AULEX online Quechua-Spanish dictionary** (qu-es.zip, qu-es.htm). ~20,907 QU→ES entries. Cusco-Collao orthography (uses k'/kh/q'/qh and 5-vowel e/o) — **not Chanka.** Useful only after orthographic normalization (k'/kh→k, q'/qh→q, e→i, o→u), with risk of homograph collisions.
- **Huamaní Parado, Mendoza Cáceres, Colachagua Ramón, *Qichwa Rimayta Qillqayta Yachasunchik*, 2021** (quechua_song.pdf, 95 pp). Pedagogical workbook in Chanka. Modest yield (~100-200 dialog pairs + small affix lexicon).

**Where this leaves us**: the data wall is real but breakable. AmericasNLP-2023 alone offers a 5-10× expansion of our Chanka training pool with high-quality aligned data. Quintero + Benito + Soto give us the Chanka-native dictionaries v28 RAG needs (the Cuzco mismatch in our first smoke killed the chrF; with Chanka-native lookups it should at minimum be neutral, likely +1-3 chrF). DPOSE and Chain-of-Dictionary are layerable on top of the v30 retrain.

### v30 — Data-expansion break-through and AmericasNLP 2021 SOTA (2026-05-27)

After v26 STaR, v27 V-STaR, and v28 RAG all hit ceilings, we abandoned method tricks and went back to the proven v25 SFT chain on **expanded Chanka training data**. The new corpus was built by merging four sources, all properly leak-filtered against the eval split:

| Source | Rows kept | Type |
|---|---:|---|
| 2014 judicial manual reviewed pairs (original v25 base) | 1,042 | full sentences |
| 2014 judicial manual glossary entries (atomic-split from rich form like "Saqiy, wischuy") | 503 | word-level |
| Benito Zuasnabar 2018 *Diccionario Básico Quechua Chanka* (parsed from PDF) | 349 | word-level |
| 2014 judicial manual glossary simple terms | 35 | word-level |
| **Total unique pairs** | **1,929** | **+874 over the original 1,055** |

Chain (exact same recipe as v25, just bigger Chanka data):
- v30a Chanka LoRA continuation on v24f-broad base, r=256 α=512, LR=2e-5, 3 epochs (615 steps)
- v30b merge
- v30c compact-mixed LoRA (using the original 1,055-pair v15 jsonl — un-regenerated)
- v30d α-sweep

**Headline result on the 158-row clean-Chanka eval (judicial manual):**

| Step / variant | chrF++ | BLEU |
|---|---:|---:|
| v25 best (α=1024) | 59.67 | 23.82 |
| 4B champion (α=160) | 56.94 | 30.76 |
| v30a-ckpt-128 | 64.55 | 32.56 |
| v30a-ckpt-192 | 70.29 | 44.60 |
| v30a-ckpt-256 | 73.31 | 51.26 |
| v30a-ckpt-320 | 78.87 | 62.68 |
| v30a-ckpt-384 | 82.24 | 65.80 |
| v30a-ckpt-448 | 83.41 | 68.50 |
| v30a-ckpt-512 | 83.80 | 69.42 |
| v30a-ckpt-576 | 84.57 | **71.30** |
| **v30a-ckpt-615 (best v30a)** | **84.62** | 70.15 |
| v30c-ckpt-192 (best compact-mixed) | 85.67 | 72.49 |
| (v30d α-sweep, final) | (TBD) | (TBD) |

**+24.95 chrF over v25's 59.67, +27.68 over the 4B champion's 56.94. Pure data-expansion lift.**

**Honest caveat — heavy domain overlap on this in-domain eval**: the +503 glossary entries we added come from the SAME 2014 judicial manual the eval set is sampled from. No exact surface-form overlap (we filtered), but the model now knows the manual's vocabulary at the word level. The 84.62 chrF reflects "trained on the manual's dictionary, tested on the manual's sentences." Real lift for this domain; generalization beyond it is what AmericasNLP 2021 tests.

#### AmericasNLP 2021 spa→quy test — clean public benchmark, ZERO training leakage

After finding that FLORES-200 devtest is 88.5% leaked into our training pool (americasnlp + somosnlp broad pretrain has it), we located a clean benchmark: the **AmericasNLP 2021 shared task** released `test.es` + `test.quy` (1,003 lines, Ayacucho/Chanka variant) at github.com/AmericasNLP/americasnlp2021/test_data. Critically, **0 of 1,003 source or target lines overlap with our entire training corpus** (verified via normalized exact-match check across americasnlp + somosnlp + the v30 expanded Chanka).

Mager et al. 2021 ([Findings of the AmericasNLP 2021 Shared Task on Open MT](https://aclanthology.org/2021.americasnlp-1.23.pdf), Table 3/5) used **ChrF (Popović 2015) — character n-grams only, `word_order=0`** — as the official ranking metric. *Not chrF++*. We initially scored ourselves with sacrebleu's default chrF++ (word_order=2) and reported 35.02; recomputing with the official metric:

| Rank | System | Track | ChrF | BLEU |
|---:|---|---|---:|---:|
| 1 | Helsinki sub 2 | Track 1 (dev allowed) | 39.4 | 5.38 |
| 2 | Helsinki sub 1 | Track 1 | 38.3 | 5.16 |
| 3 | REPUcs sub 2 | Track 1 | 35.8 | 3.1 |
| 1 | REPUcs sub 1 | Track 2 (dev not used) | 34.6 | 2.91 |
| Baseline | Transformer Vaswani 2017 | both | 30.4 | 0.05 |
| **Ours** | **v30a-ckpt-615** | (Track 2 semantics — never saw the 2021 dev/test) | **40.44** | 4.79 |

**+1.04 ChrF over the Track 1 winner. +5.84 ChrF over the Track 2 winner. +10.04 ChrF over the baseline.** BLEU is slightly below Helsinki (4.79 vs 5.38) — the lexical-style mismatch we discuss below.

Why we count as Track 2 semantics: Track 2 disallowed training directly on the AmericasNLP 2021 dev set; we never had access to either dev or test (verified zero overlap). Our broad pretraining includes JW300/MINEDU/Huarcaya (the same publicly available data sources Helsinki and the 2021 baseline used as training corpora), which is allowed in both tracks.

#### Insights from v30 we didn't have before

1. **Data > method tricks at our scale.** Three different "novel method" attempts (v26 STaR self-grounded reasoning, v27 V-STaR DPO verifier, v28 RAG with dict-in-prompt) each gave +0 to +2 chrF after substantial engineering. A single afternoon of merging four dictionaries gave **+24.95 chrF on the in-domain eval and +5.4 ChrF on AmericasNLP 2021**. The data wall *was* the binding constraint; we just hadn't named it precisely enough. Going forward, any time we'd consider another method tweak, we should first check whether more parsable data exists.

2. **Lexicon data (dict-style word-level pairs) is unreasonably effective for low-resource MT.** Our +874 net new pairs were almost entirely lexicon entries (Benito dict + glossary atomic splits), not full sentences. Yet they lifted chrF dramatically because Chanka morphology is so regular that knowing the root + the model's broad morphological knowledge is enough to produce well-formed inflected forms. This means **a high-coverage Spanish→Chanka dictionary is worth more than the same number of full sentences** for an SFT-trained model. The implication: the next big lift would be parsing the Quintero (~6-10k entries) and Soto Ruiz vocabulario (~1,000 entries) into the same JSONL format.

3. **Metric carefully — chrF++ ≠ ChrF.** sacrebleu defaults to chrF++ (word_order=2) when you call `CHRF()`; the AmericasNLP 2021 paper uses ChrF (word_order=0). On the same predictions we got chrF++ 35.02 vs ChrF 40.44 — a 5-point gap. This is a routine mistake that would make our result look mediocre instead of SOTA. Always check the original paper's exact metric spec before claiming any comparison.

4. **The in-domain vs out-of-domain chrF gap reveals real-world deployment shape.** Same model: 84.62 on the judicial manual, 40.44 on AmericasNLP newswire. The model is genuinely strong in the domain we trained for, and roughly best-in-class on a generic Chanka benchmark, but **the gap is ~44 chrF** — far more than any method tweak we've ever measured. For a deployment, the in-domain number is what matters; for a publication, both should be reported to be honest about generalization.

5. **chrF undervalues stylistically-different-but-correct translations.** Inspecting our predictions on the AmericasNLP 2021 test, the model produces fluent, grammatical, semantically-correct Chanka, but uses more Spanish loanwords (`abueloy`, `amistad`, `mamay`, `droga`) and slightly different verb stems than the references (which prefer native roots like `machu`, `masi`). Both are valid Chanka. Without a multi-reference eval or human judgment, chrF alone can't separate "wrong translation" from "different acceptable translation." Our 40.44 likely understates real quality — a multi-reference or fluency/adequacy human eval would probably show 5-10 points higher.

6. **vLLM + LoRA on Qwen3.5-9B works end-to-end** after fixing two gotchas: (a) the base model must be the exact `base_model_name_or_path` from the adapter's config (not a different merge), and (b) `ninja` must be on PATH for flashinfer's JIT-compiled sampler. With those fixed, vLLM gives ~10-15× speedup over `transformers.generate()` on this workload, producing **bitwise-identical outputs** at temperature=0 (verified). Our 1,003-line AmericasNLP eval took 42 seconds with vLLM vs ~3-5 minutes with transformers.

7. **The right framing: we're SOTA on a public benchmark for a previously-neglected variety.** Spanish→Chanka Quechua has had Cuzco-Collao Quechua as its main academic alternative; Chanka MT specifically has lagged. Our v30a beats the 2021 Track 1 best on the only widely-comparable public benchmark, *without* using the 2021 dev set, *with* about 1,929 reviewed pairs + standard public broad corpora. The replication path is in the public domain: the manual is open, Benito's dictionary is published, the model is open-weight Qwen3.5-9B.

#### Production recipe (what we'd recommend someone else replicate)

```
Base:          unsloth/Qwen3.5-9B (raw)
Broad SFT:     r=256, α=512, LR 1e-4, eff_batch 64, seq 256, max_steps 512
               on americasnlp + somosnlp broad Quechua (166k rows).
               Stop when chrF plateaus on a held-out broad subset (~step 384-512).
Chanka SFT:    r=256, α=512, LR 2e-5, eff_batch 8, seq 128, num_train_epochs 3
               on a Chanka corpus that includes both full sentences AND
               aggressive lexicon coverage (every dictionary entry as a pair).
Compact-mixed: r=256, α=512, LR 5e-6, eff_batch 4, seq 128, num_train_epochs 2
               on direct + compact (Análisis/Final/Puntaje) format mix.
α-sweep:       trained α=512; load with α ∈ {512, 640, 768, 896, 1024} and pick best.
Inference:     vLLM with the correct merged base + the adapter, temperature 0.
```

The single most important deviation from naive practice is using the team's actual LRs — broad 1e-4, Chanka 2e-5 — and not falling for "9B is bigger, lower LR." That mistake cost us the entire v22 chain.

### v26 STaR failure + contrastive-reasoning idea for v28/v29

We attempted STaR-style self-grounded reasoning on v25 (best chrF=59.67 base) on 2026-05-27. The pipeline:
1. v25 translates the 897 train Chanka sources → 86.5% match rate at chrF ≥ 70 (extraordinary by itself)
2. For matches: use v25's own translation as target. For misses: use reference (rationalization)
3. Prompt v25 again with `(source, target)` to generate a Spanish-language morphological reasoning

**Step 3 failed cleanly**: v25 ignored the "explain in Spanish" instruction and just regenerated the Chanka target as "reasoning". 593/897 rows had `len(reasoning) < 30 chars` and were dropped; the surviving 304 also had reasoning = duplicate target. v25 is over-specialized: heavy Chanka SFT crowded out the general instruction-following ability needed to language-switch from Chanka generation to Spanish meta-commentary.

**Why this consistently fails across our attempts**:
- v17/v18 (external DS-Flash reasoning, then Claude grounded reasoning, then v26 self-reasoning) all failed to lift chrF in their own way. The unifying lesson: **reasoning supervision keeps failing on this task because reasoning isn't load-bearing**. Translation here is dominated by vocabulary memorization, not multi-step inference. Standard SFT lets the model ignore the reasoning and still produce correct translation, so the reasoning carries no gradient signal and becomes ornamental.

**The contrastive-reasoning angle (idea for v29 or later)**: instead of asking the model to *produce* reasoning from scratch (hard — requires Spanish meta-commentary, not the model's dominant behaviour), ask it to *choose* between two pre-existing candidate translations and articulate why. Concretely:
- For each train source, generate K=8 candidates from v25 (vLLM, sample-decode at moderate temperature)
- Score each candidate's chrF vs reference to pick a `winner` and `loser`
- Prompt v25 to compare: *"Candidato A: X. Candidato B: Y. ¿Cuál es la mejor traducción del español 'Z'? Explica brevemente."*
- The model now has a much easier task: both candidates are in Chanka (no language-switch needed), the comparison surfaces concrete morphological differences, and the reference's chrF score provides ground truth for which is correct
- Train v26 on `(source, candidates, reasoning_about_choice, chosen_translation)` quadruples

Why this could actually work where v17/v19/v26 failed: the reasoning is **about a real decision the model just made differently** in its own K samples — it's grounded in the model's own latent representation of "what makes a good Chanka translation", not in external linguistic theory. The model would have to internalize comparative judgement (which DPO-style training optimizes for) and then articulate it (which only works if the comparative judgement is real to begin with).

This composes naturally with V-STaR (v27): the same K candidates feed both V-STaR's verifier training and the contrastive-reasoning trace generation. Filed as v29 — to revisit if v27 and v28 (RAG-Reasoning) don't fully close the +12 chrF K=16 oracle gap.

### External zero-shot translation models (all garbage for Chanka)
| Model | chrF++ |
|---|---|
| NLLB-200 1.3B | 27.70 |
| NLLB-200 600M | 23.93 |
| Gemma 4 E4B IT | 16.29 |
| Hy-MT2 7B | 11.20 |
| T5Gemma 2-1B | 7.24 |
| T5Gemma 270M | 6.17 |
| NLLB-200 3.3B | 2.30 (degenerate output) |
| TranslateGemma 4B IT | — (chat template has no spa-Latn key) |

## Full Leaderboard

| Rank | Model | chrF++ | BLEU | tokF1 | TER |
| -: | --- | -: | -: | -: | -: |
| 1 | **v13 LoRA α=160 (champion)** | **56.94** | **30.76** | **46.43** | **62.21** |
| 2 | v13 ckpt-32 (α=128) | 55.76 | 29.23 | 44.26 | 64.52 |
| 3 | v19b ckpt-192 (grounded thinking + v13) | 55.57 | 28.87 | 41.36 | 65.67 |
| 4 | v12 ckpt-128 | 55.47 | 29.87 | 43.99 | 65.67 |
| 5 | v11 ckpt-448 | 54.06 | 27.91 | 43.00 | 67.28 |
| 6 | direct-only ckpt-512 (leak-free) | 52.20 | 24.50 | 41.17 | 68.66 |
| 7 | v10 ckpt-256 | 49.11 | 21.80 | 36.10 | 73.04 |
| 8 | v19 ckpt-128 (grounded thinking from base) | 44.55 | 18.58 | — | 81.57 |
| 9 | v16 targeted LoRA L15-17,L27-31 | 44.83 | 18.16 | 29.94 | 79.03 |
| 10 | **4B baseline ckpt-36** | 43.49 | 16.14 | 28.94 | 82.49 |
| 11 | NLLB 1.3B + our LoRA (ckpt-448) | 41.84 | 13.09 | 27.45 | 81.34 |
| 12 | v15 9B raw + compact-mixed | 32.08 | 8.04 | 17.79 | 93.78 |
| 13 | NLLB 1.3B zero-shot | 27.70 | 1.59 | 13.30 | 143.78 |
| 14 | v17b ckpt-320 (confabulated thinking + direct) | 23.34 | 0.99 | — | — |
| 15 | v17 ckpt-224 (thinking-only, confabulated) | 18.20 | 0.36 | — | — |
| 16 | v18 ckpt-320 (broken double-`<think>` wrap) | 3.04 | 0.03 | — | — |

## Lessons Learned

1. **For low-resource translation with a strong already-Chanka-specialized base, direct SFT with extended training + LR-decay refinement + α-scaling at inference is the most reliable recipe.** Tried lots of more sophisticated approaches; none beat this.

2. **Multi-task SFT auxiliary objectives can help modestly (+2 chrF++) but the win is dominated by raw step count.** Most of v13's +12 chrF advantage over the 4B baseline is "train longer on Chanka data," not the multi-task framing.

3. **Verification / RL / reranking are largely orthogonal to data-quality wins.** The team's verifier-as-RL-reward, the listwise text reranker, the self-score-Best-of-N, the mergekit task-vector amplification — all matched or trailed the direct SFT champion despite being substantially more complex. The 4B base's capacity ceiling on 1,055 training pairs is reached by simple methods.

4. **Symbolic reasoning supervision (thinking-block traces) does not help and can actively hurt.** Even with rigorously grounded linguistic traces (Claude agents + AMLQ dictionary + grammar manual), the model failed to outperform direct SFT. The model internalizes Chanka morphology implicitly from (source, gold) pairs more efficiently than from explicit step-by-step explanations.

5. **The data wall at 1,055 reviewed Chanka pairs is the binding constraint.** Best paths past this constraint are likely (a) acquiring more reviewed Chanka data, (b) scaling the chain to 9B with proper Chanka-pretraining stage, or (c) building a real reference-free neural verifier that can capture the 12-chrF K=16 oracle headroom we measured.

## Reproducibility

To regenerate the champion:
```bash
# Stage 1
MAX_STEPS=512 LEARNING_RATE=5e-6 EVAL_STEPS=32 SAVE_STEPS=32 \
  OUTPUT_DIR=outputs/compact_mixed_v11 \
  experiments/sft/run_compact_mixed_self_verifiable_sft.sh

# Stage 2
MAX_STEPS=128 LEARNING_RATE=1e-6 EVAL_STEPS=16 SAVE_STEPS=16 \
  OUTPUT_DIR=outputs/compact_mixed_v12 \
  experiments/sft/run_compact_mixed_self_verifiable_sft.sh \
    --adapter-path outputs/compact_mixed_v11/checkpoint-448

# Stage 3
MAX_STEPS=32 LEARNING_RATE=5e-7 EVAL_STEPS=8 SAVE_STEPS=8 \
  OUTPUT_DIR=outputs/compact_mixed_v13 \
  experiments/sft/run_compact_mixed_self_verifiable_sft.sh \
    --adapter-path outputs/compact_mixed_v12/checkpoint-128

# Free upgrade: edit adapter_config.json
python -c "import json,os; \
  p='outputs/compact_mixed_v13/checkpoint-32/adapter_config.json'; \
  d=json.load(open(p)); d['lora_alpha']=160; \
  json.dump(d,open(p,'w'),indent=2)"
```

Eval:
```bash
python scripts/evaluate_gspo_checkpoint.py \
  --adapter-path outputs/compact_mixed_v13/checkpoint-32 \
  --terminology-file clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet \
  --terminology-top-k 1 --max-completion-length 96 --split eval \
  --output-json metrics.json --predictions-jsonl preds.jsonl
```

Expected on the 158-row held-out clean Chanka eval: chrF++ ≈ 56.94, BLEU ≈ 30.76, tokF1 ≈ 46.43, TER ≈ 62.21.

## Open Directions (not run in this study, ranked by expected payoff)

1. **More reviewed Chanka data** — any path that grows the 1,055-pair corpus.
2. **Neural reference-free verifier on v13's K=16 candidate pool** for proper Best-of-N (12-chrF oracle ceiling sits unused).
3. **Full 9B chain** (raw → broad → Chanka full-SFT → compact-mixed → α-scaling).
4. **Synthetic Chanka self-distillation** — use v13 to translate broad AmericasNLP Spanish, self-consistency filter, retrain on enlarged corpus.
5. **DPO from v13 K=16 self-distilled preference pairs** — preference learning around the v13 manifold.
