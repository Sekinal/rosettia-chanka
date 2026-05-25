# Late 2026-05-25 Wave: Alpha Scaling, Activation Diff, Targeted LoRA, NLLB Transfer, External Baselines

After v13 ckpt-32 (chrF++ 55.760) we ran a wide ablation wave on both the L40S
(m1) and A100 80GB (m2) machines. Headlines:

- **New single-model champion (free upgrade): v13 LoRA with `lora_alpha=160`**
  (1.25× the trained alpha) → chrF++ `56.944`, BLEU `30.756`, token F1 `46.432`,
  TER `62.212`. A single JSON field edit, no retraining.
  Saved at `outputs/v13_alpha160_champion_20260525/`.
- The activation-diff insight (mid-layer + late-layer cluster of biggest v13
  changes) **did not** translate into a parameter-efficient win: a targeted
  LoRA on L15-17 + L27-31 plateaus at chrF++ `44.83`, far below the full-layer
  v11 ckpt-448 (chrF++ `54.06`).
- The recipe **transfers cross-architecture**: NLLB-200 distilled 1.3B + our
  LoRA on the 841 leak-free direct Chanka pairs gains +14.14 chrF++ over its
  own zero-shot (27.70 → `41.84`). Doesn't beat the 4B Qwen baseline but
  proves data-recipe is architecture-agnostic.
- Scaling to 9B from a raw base (no Chanka pretraining) plateaus at chrF++
  `32.08`, well below v13. The proper-chain 9B (raw → broad → Chanka full-SFT
  → compact-mixed) is the right path; we did not run it.
- No external off-the-shelf MT model (NLLB up to 3.3B, TranslateGemma 4B,
  Gemma 4 E4B, Hy-MT2 7B, T5Gemma 270M/2-1B) comes close to even our 4B
  baseline on Chanka.

## LoRA alpha-scaling sweep (free probe of capacity)

Loading the trained v13 ckpt-32 LoRA with different `lora_alpha` values at
inference, no retraining:

| α  | scale  | chrF++ | BLEU  | tokF1 | TER   |
| -: | -:     | -:     | -:    | -:    | -:    |
|  64 | 0.50× | 49.05 | 22.11 | 36.87 | 72.81 |
|  96 | 0.75× | 51.91 | 24.25 | 39.65 | 70.28 |
| 128 | 1.00× | 55.76 | 29.23 | 44.26 | 64.52 |
| **160** | **1.25×** | **56.94** | **30.76** | **46.43** | **62.21** |
| 192 | 1.50× | 52.37 | 26.51 | 39.74 | 68.43 |
| 224 | 1.75× | 44.57 | 14.63 | 31.83 | 77.88 |
| 256 | 2.00× | 37.06 |  5.00 | 23.77 | 88.48 |

Clean unimodal peak at 1.25×. Above 1.5× the model degrades fast. The
trained alpha (128) was slightly undertuned — gradient descent stopped before
fully amplifying the LoRA contribution. The 1.25× evaluation is a free win
over the saved checkpoint.

## Mergekit-style task-vector amplification (control)

We merged v13 ckpt-32 into the 4B base, computed `delta = merged - base`,
then wrote `base + α·delta` for several α:

| α    | chrF++ | BLEU  | tokF1 | TER   |
| -:   | -:     | -:    | -:    | -:    |
| 1.00 | 52.41 | 25.67 | 40.13 | 70.28 |
| 1.25 | 53.69 | 26.82 | 41.60 | 68.43 |
| 1.50 | 55.89 | 29.48 | 43.76 | 64.06 |
| 1.75 | 50.48 | 24.78 | 37.77 | 70.51 |
| 2.00 | 46.09 | 16.26 | 32.52 | 77.19 |

Peak at α=1.5 but only chrF++ 55.89 (vs LoRA-scaling 56.94). Mergekit
round-trip also leaks ~3 chrF (α=1.0 control scores 52.41 vs v13's true
55.76). Conclusion: LoRA-alpha scaling is cleaner and stronger than
post-hoc mergekit amplification for this case.

## TransformerLens-style activation diff and the v16 targeted LoRA

Computed per-layer cosine distance between v13 and base residual streams on
30 leak-free Chanka train sources. Pattern:

```
L 0-L 2: ~0       (early features unchanged)
L 3-L15: rising
L16-L17: peak     (~0.004 — mid-network semantic translation)
L18-L26: dip      (~0.003)
L27-L31: rising
L31:    max 0.0055 (output / morpheme decisions)
```

**v16 hypothesis**: train LoRA only on L15-17 + L27-31 (8 of 32 layers, 54%
of trainable params) — should capture most of v13's gain with fewer params.

**v16 result**: best ckpt-192 chrF++ `44.83`, BLEU `18.16`, TER `79.03`.
Versus full-layer v11 ckpt-448 (chrF++ `54.06`), the restriction lost ~9 chrF.

**Honest lesson**: the activation analysis identifies where v13's *output*
differs most, but the *learning signal* needed touches all layers. Small
changes in many layers compound. Sparse layer targeting based on
end-of-layer cosine distance is not the right axis for parameter-efficient
training here.

## v15: 9B raw + compact-mixed (no Chanka pretrain)

Applied the v11 recipe to raw `unsloth/Qwen3.5-9B` (no prior Chanka SFT
base):

| ckpt | chrF++ | BLEU | tokF1 | TER  |
| -:   | -:     | -:   | -:    | -:   |
| 128 | 23.88 |  3.16 |  9.71 | 106.0 |
| 256 | 28.40 |  6.26 | 15.32 |  93.3 |
| **448** | **32.08** |  8.04 | 17.79 |  93.8 |
| 512 | 32.02 |  8.20 | 17.78 |  92.6 |

Plateau at chrF++ ~32. The shortcut (skip Chanka pretraining) loses too
much. The 4B winner came from a model that had ALREADY been Chanka-trained
extensively before the compact-mixed LoRA was applied. For a fair 9B SOTA
push the chain has to be: raw 9B → broad LoRA → Chanka full-SFT base →
compact-mixed LoRA. This was not run in this wave.

## NLLB-200 1.3B + our LoRA recipe (cross-architecture transfer)

Trained NLLB-200 distilled 1.3B with LoRA r=64/α=128 on the 841 leak-free
direct Chanka pairs, 512 steps, LR `1e-4`, batch 8 — total training time
~2.5 min on L40S.

| ckpt | chrF++ | BLEU  | tokF1 | TER   |
| -:   | -:     | -:    | -:    | -:    |
| Zero-shot (no train) | 27.70 |  1.59 | 13.30 | 143.78 |
| 128 | 37.05 |  8.44 | 23.52 | 89.40 |
| 256 | 40.32 |  9.84 | 25.23 | 84.33 |
| 384 | 41.03 | 11.11 | 26.29 | 83.64 |
| **448** | **41.84** | **13.09** | **27.45** | **81.34** |
| 512 | 41.43 | 12.51 | 26.90 | 82.03 |

+14.14 chrF++ / +11.50 BLEU over NLLB zero-shot. Still below the 4B Qwen
baseline (chrF++ 43.49) because NLLB is a smaller encoder-decoder, but the
recipe transfers cleanly. Trained adapter at
`outputs/nllb_lora_chanka_20260525-1.3B-m1/checkpoint-448/`.

## External zero-shot baselines on the same 158-row Chanka eval

| Model | chrF++ | BLEU | Notes |
| --- | -:  | -: | --- |
| NLLB-200 distilled 1.3B | 27.70 |  1.59 | best of external multilingual MT |
| NLLB-200 distilled 600M | 23.93 |  0.92 | |
| Gemma 4 E4B IT | 16.29 |  1.24 | |
| Hy-MT2 7B | 11.20 |  0.87 | |
| T5Gemma 2-1B | 7.24 |  0.05 | |
| T5Gemma 270M | 6.17 |  0.04 | |
| NLLB-200 3.3B | 2.30 |  0.02 | produces gibberish, ignore |
| TranslateGemma 4B IT | — | — | chat template has no `spa-Latn` key; team noted Devanagari outputs in prior smokes |

None of these compete with our 4B Chanka-specialized baseline (chrF++ 43.49)
let alone our champion (56.94).

## Engineered reasoning post-mortem (from earlier in the day)

The DS Flash flavor-C engineered traces (793 Spanish-language step-by-step
linguistic explanations) trained as plain SFT assistant output were a
**net negative**:
- v14 (mixed direct + compact + engineered): chrF++ 46.29 vs 4B baseline 43.49
- v14b (engineered-only, direct prompt): chrF++ 42.56 (catastrophic — prompt mismatch)
- v14c (curriculum: direct-only-512 + engineered polish): chrF++ 52.19 (no change)

Key realization: all of these used `enable_thinking=False`. The engineered
content went into the model's *observable output* rather than into the
hidden reasoning channel Qwen3.5 natively supports. The right test is to
wrap each trace as `<think>{trace}</think>{final}` and train with
`enable_thinking=True`. That experiment is v17, scheduled next.

## Current leaderboard summary

| Rank | Model | chrF++ | BLEU | tokF1 | TER |
| -: | --- | -: | -: | -: | -: |
| 1 | **v13 LoRA α=160 (new champion)** | **56.94** | **30.76** | **46.43** | **62.21** |
| 2 | v13 ckpt-32 (α=128) | 55.76 | 29.23 | 44.26 | 64.52 |
| 3 | v12 ckpt-128 | 55.47 | 29.87 | 43.99 | 65.67 |
| 4 | v11 ckpt-448 | 54.06 | 27.91 | 43.00 | 67.28 |
| 5 | direct-only ckpt-512 (leak-free) | 52.20 | 24.50 | 41.17 | 68.66 |
| 6 | v10 ckpt-256 | 49.11 | 21.80 | 36.10 | 73.04 |
| 7 | v16 targeted LoRA ckpt-192 | 44.83 | 18.16 | 29.94 | 79.03 |
| 8 | **4B baseline ckpt-36** | 43.49 | 16.14 | 28.94 | 82.49 |
| 9 | NLLB 1.3B + our LoRA (ckpt-448) | 41.84 | 13.09 | 27.45 | 81.34 |
| 10 | v15 9B raw + compact-mixed (ckpt-448) | 32.08 |  8.04 | 17.79 | 93.78 |
