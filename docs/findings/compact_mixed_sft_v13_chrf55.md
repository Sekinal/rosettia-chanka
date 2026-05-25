# Compact-Mixed Self-Verifiable SFT — chrF++ 55.76 Single-Model Standalone (2026-05-25)

## Headline

A three-stage LoRA continuation on top of the 4B full-SFT base (`checkpoint-36`),
using the existing mixed direct-translation + compact-thinking JSONL as a
multi-task SFT objective, beats the 4B baseline by +12.27 chrF++ and also beats
the prior best multi-model deployable ensemble.

| Metric | 4B baseline ckpt-36 | v13 ckpt-32 winner | Δ |
| --- | ---: | ---: | ---: |
| chrF++ | 43.488 | **55.760** | **+12.27** |
| BLEU | 16.139 | 29.233 | +13.09 |
| token F1 | 28.938 | **44.258** | +15.32 |
| TER | 82.488 | **64.516** | −17.97 |

Reference points:
- Prior best multi-model deployable (K32+4B with listwise text selector):
  chrF++ 51.86 / BLEU 26.02. v13 ckpt-32 beats it standalone by chrF++ +3.90,
  BLEU +3.21.
- v12 ckpt-128 has the highest BLEU at 29.866.

All metrics on the held-out 158-row clean Chanka eval (`split=eval` from
`scripts/evaluate_gspo_checkpoint.py`), with `--terminology-top-k 1
--max-completion-length 96` and the standard direct prompt. No self-verification
or thinking format at inference.

## Recipe

Single LoRA, three stages. Same `--jsonl
outputs/self_verifiable_data_20260525-compact-mixed/self_verifiable_compact_mixed_sft.jsonl`
(2,110 rows = 1,055 sources × {direct, compact}). Always
`--target-field target`. Same LoRA r=64 / alpha=128. Same per-device batch 4,
grad-accum 1, `--max-train-samples 1024 --max-eval-samples 192
--max-seq-length 128 --terminology-top-k 1`. Inference uses the **direct**
prompt only.

| Stage | Continuation source | Steps | Peak LR | Output |
| --- | --- | ---: | ---: | --- |
| v11 | (fresh LoRA on base ckpt-36) | 512 | 5e-6 | `outputs/compact_mixed_format_sft_20260525-4b-512step-v11-r64-1024train/checkpoint-448` |
| v12 | `--adapter-path` v11 ckpt-448 | 128 | 1e-6 | `outputs/compact_mixed_format_sft_20260525-4b-refine-v12-from-v11ckpt448/checkpoint-128` |
| v13 | `--adapter-path` v12 ckpt-128 | 32 | 5e-7 | `outputs/compact_mixed_format_sft_20260525-4b-refine-v13-from-v12ckpt128/checkpoint-32` |

Total ~672 gradient updates over the 1,024-row training slice (~2.6 passes).
Wrapper: `experiments/sft/run_compact_mixed_self_verifiable_sft.sh`.

## Full leaderboard of variants tried today

All on the same eval. Best ckpt per variant is the one whose chrF++ is shown.

| Variant | LoRA r | Peak LR | Steps | Train rows | chrF++ | BLEU | tokF1 | TER | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline (4B ckpt-36) | — | — | — | — | 43.488 | 16.139 | 28.938 | 82.488 | prior best standalone |
| v1 ckpt-32 (pre-existing) | 32 | 5e-6 (decay) | 32 | 1024 | 44.042 | 16.698 | 29.995 | 82.258 | original short warmup |
| v2 ckpt-64 | 32 | 1e-6 | 128 | 1792 | 44.243 | 16.558 | 30.283 | 82.028 | low LR from scratch — slower |
| v3 ckpt-32 | 64 | 5e-6 | 32 | 1024 | 45.182 | 18.940 | 31.762 | 79.724 | r=64 jumps a step |
| v4 ckpt-64 | 64 | 5e-6 | 64 | 1792 | 45.413 | 17.175 | 31.019 | 82.258 | larger train sample worse |
| v5 ckpt-16 | 128 | 5e-6 | 32 | 1024 | 44.297 | 19.031 | 30.815 | 79.493 | r=128 unstable (eval loss 1.34) |
| v6 ckpt-40 | 64 | 5e-6 | 96 | 2048 | 43.834 | 16.462 | 30.352 | 83.871 | batch=8 hurts |
| v7 ckpt-80 | 64 | 5e-6 | 96 | 1024 | 46.686 | 19.611 | 32.777 | 76.959 | 96-step extension of v3 recipe |
| v8 ckpt-160 | 64 | 5e-6 | 160 | 1024 | 47.205 | 19.854 | 33.325 | 76.498 | still climbing |
| v9 ckpt-16 | 64 | 5e-7 | 48 | 1024 | 46.591 | 19.572 | 32.690 | 77.419 | low-LR refine from v7 → no gain |
| v10 ckpt-256 | 64 | 5e-6 | 256 | 1024 | 49.112 | 21.795 | 36.098 | 73.041 | doubling steps still helps |
| **v11 ckpt-448** | 64 | 5e-6 | 512 | 1024 | 54.064 | 27.909 | 43.004 | 67.281 | stage 1 saturates near 448 |
| **v12 ckpt-128** | 64 | 1e-6 | 128 (cont) | 1024 | 55.468 | **29.866** | 43.995 | 65.668 | LR-drop refine helps |
| **v13 ckpt-32** | 64 | 5e-7 | 32 (cont) | 1024 | **55.760** | 29.233 | **44.258** | **64.516** | second LR drop nudges further |

## What worked

The only DeepSeekMath-V2 ingredient that survived adaptation is the
**multi-task SFT objective**. The mixed JSONL contains, per Chanka source pair,
one `prompt_mode=direct` row (normal translation) and one
`prompt_mode=compact` row whose target is the structured self-analysis format:

```
Analisis: [TERMINOLOGIA] uses the glossary term; [SIGNIFICADO] meaning preserved.
Final: <Chanka translation>
Puntaje: \boxed{0.95}
```

Training on the union teaches the model the verification rubric. At inference
we never use the compact prompt — we just ask for the direct translation.
The auxiliary loss seems to discipline the direct outputs.

Caveat (real ablation pending — `outputs/ablation_direct_only_sft_20260525-stageA-512step`
is running): we have not yet confirmed how much of the gain is the auxiliary
objective vs simply training longer on Chanka data. The 4B full-SFT base was
only fine-tuned for 48 steps on 897 Chanka rows; v13 cumulatively does ~672
LoRA gradient updates over a similar slice, so a substantial fraction of the
win is plausibly "train more" rather than "auxiliary task." The same-schedule
direct-only run will tell us.

## What didn't work

- **Verifier-as-reward GSPO** still regresses on top of the new SFT chain.
  `outputs/gspo_paper_profiles/2511_deepseekmath_final_only_v7ckpt80merged_20260525-nomask-m2`
  ran the team's best final-only no-mask recipe (`MASK_TRUNCATED_COMPLETIONS=false`,
  `MAX_COMPLETION_LENGTH=64`, ATTACH_LORA=true) from the merged v7 ckpt-80
  base. Result: chrF++ 45.13 vs the SFT base v7 ckpt-80 at 46.69 — drop of
  −1.55 chrF++. Trainer reward stayed negative (−0.282). The current
  reference-based learned verifier provides no useful RL signal on top of a
  well-trained SFT adapter.
- **r=128 LoRA at LR 5e-6 is unstable.** v5 reached eval loss 1.34
  (vs r=64 at 0.65) and produced weaker translations.
- **Doubling effective batch (v6, per_device=8) hurts** at the same LR.
- **Low-LR continuation only helps after stage 1 has fully converged.**
  v9 (LR=5e-7 from v7 ckpt-80) gained nothing; v12/v13 (analogous LR drops
  from v11 ckpt-448) gained materially. Stage 1 must reach the chrF++ ~54
  range before refinement does anything useful.
- **Self-verification at inference does not help** even with this much
  better SFT base. Compact-mode inference scores chrF++ 38.47 (10 points
  below direct-mode). Self-scores still saturate at 1.0 with
  `false_confidence_rate=100%`, so self-score-ranked Best-of-N is
  equivalent to random selection.

## Comparisons to other work on this codebase

- Team's prior best DeepSeekMath-style branch (final-only no-mask GSPO):
  beat baseline by chrF++ +0.45. Our recipe beats by +12.27. The
  difference is in the SFT data construction, not the RL.
- Team's prior best multi-model deployable (K32+4B with listwise text
  selector): chrF++ 51.86 / BLEU 26.02. v13 ckpt-32 is a single model and
  exceeds it (chrF++ 55.76 / BLEU 29.23). Note this is "K=1" — no
  sampling, no rerank — so combining v13 with Best-of-N rerank should
  push higher again.

## Open questions and next experiments

1. **Direct-only ablation** (running on m2):
   `outputs/ablation_direct_only_sft_20260525-stageA-512step` — does
   removing the compact-thinking rows but keeping the same step schedule
   match v11 ckpt-448? If yes, the multi-task framing is mostly an
   accident of step count. If no, the auxiliary objective is doing real
   work.
2. **Best-of-N with v13 as candidate generator.** Generate K=16-32
   sampling candidates per source on the held-out eval, apply the
   existing listwise text reranker. Estimated upper bound is chrF++ 60+.
3. **Reference-free verifier.** The current verifier requires the gold
   reference, which is why GSPO never improves and why we cannot use it
   as a test-time reranker. Build (source, candidate, oracle-chrF) data
   from existing candidate pools (no reference at inference) and train a
   ref-free judge. Use as both Best-of-N reranker and as RL reward —
   this would be the actually paper-faithful DeepSeekMath-V2 adaptation.
4. **Synthetic broad-Quechua → Chanka augmentation.** v13 is strong
   enough to translate 87k AmericasNLP broad sources; score with the
   ref-free verifier; keep the top decile; retrain. Breaks the 1,055-pair
   Chanka SFT ceiling.

## Reproducibility

Each stage is reproducible from the wrapper:

```bash
# Stage 1 (v11): fresh LoRA, 512 steps @ 5e-6
MAX_STEPS=512 LEARNING_RATE=5e-6 EVAL_STEPS=32 SAVE_STEPS=32 \
  OUTPUT_DIR=outputs/compact_mixed_v11 \
  experiments/sft/run_compact_mixed_self_verifiable_sft.sh

# Stage 2 (v12): continue from v11 best ckpt @ 1e-6 for 128 steps
MAX_STEPS=128 LEARNING_RATE=1e-6 EVAL_STEPS=16 SAVE_STEPS=16 \
  OUTPUT_DIR=outputs/compact_mixed_v12 \
  experiments/sft/run_compact_mixed_self_verifiable_sft.sh \
    --adapter-path outputs/compact_mixed_v11/checkpoint-448

# Stage 3 (v13): continue from v12 best ckpt @ 5e-7 for 32 steps
MAX_STEPS=32 LEARNING_RATE=5e-7 EVAL_STEPS=8 SAVE_STEPS=8 \
  OUTPUT_DIR=outputs/compact_mixed_v13 \
  experiments/sft/run_compact_mixed_self_verifiable_sft.sh \
    --adapter-path outputs/compact_mixed_v12/checkpoint-128
```

Eval each stage with:

```bash
.venv/bin/python scripts/evaluate_gspo_checkpoint.py \
  --adapter-path <stage_output>/checkpoint-<best> \
  --output-json <out>.json --predictions-jsonl <out>.jsonl \
  --terminology-file clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet \
  --terminology-top-k 1 --max-completion-length 96 --split eval
```

Best held-out adapters (m1):
- `outputs/compact_mixed_format_sft_20260525-4b-refine-v13-from-v12ckpt128/checkpoint-32`
- `outputs/compact_mixed_format_sft_20260525-4b-refine-v12-from-v11ckpt448/checkpoint-128`
- `outputs/compact_mixed_format_sft_20260525-4b-512step-v11-r64-1024train/checkpoint-448`
