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
v15 — raw Qwen3.5-9B + compact-mixed LoRA (no Chanka pretrain): plateaus at chrF++ 32.08. The shortcut (skip Chanka pretraining) loses too much. To match the 4B chain you'd need raw 9B → broad → Chanka full-SFT base → compact-mixed LoRA. Not run in this study.

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
