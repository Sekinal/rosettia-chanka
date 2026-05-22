# External Model Baselines

Date: 2026-05-22

Purpose: test whether current external translation/general models can directly beat or usefully seed the RosettIA Chanka translator before spending time on larger SFT/GSPO runs.

## Standing Internal Baseline

Current best adapter remains:

`outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora`

Canonical held-out eval:

- Selection: `26.3808`
- chrF++: `40.9703`
- BLEU: `8.1555`
- Token F1: `26.6169`
- Source-copy: `2.8270%`
- Spanish leakage: `0.6329%`
- TER: `88.7097`

## Zero-Shot Smokes

All smokes used 5 rows from the clean Chanka validation split. These are not full evaluations; they are viability checks before spending GPU on larger tests.

| Model | Backend | chrF++ | BLEU | Token F1 | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| `tencent/Hy-MT2-1.8B` | `causal-chat` | `12.6362` | `2.8117` | `0.0` | Not Chanka; mixed Spanish/other-language-looking output. |
| `google/gemma-4-E2B-it` | `causal-chat` with Gemma 4 processor | `8.2584` | `3.8436` | `0.0` | Malformed/repetitive Chanka-like output. |
| `google/gemma-4-E4B-it` | `causal-chat` with Gemma 4 processor | `13.5899` | `4.4485` | `0.0` | Cleaner Chanka-like text than E2B, but semantically poor. |
| `google/translategemma-4b-it` | `translategemma`, `es -> qu` | `2.4792` | `3.6688` | `0.0` | Invalid for our target; generated Devanagari-like text. |
| `google/t5gemma-2-270m-270m` | `seq2seq-chat` | `6.3123` | `0.1828` | `0.0` | Not instruction/translation ready for this target. |

Earlier full baseline:

- `facebook/nllb-200-distilled-600M`, `spa_Latn -> quy_Latn`: chrF++ `24.1377`, BLEU `0.9596`, token F1 `10.1348`, TER `166.1290`.

## Implementation Notes

- `scripts/evaluate_translation_baseline.py` now supports `causal-chat` models.
- Gemma 4 models require the processor/chat-template path with `enable_thinking=False`; the tokenizer-only path leaked thinking artifacts.
- `tencent/Hy-MT2-1.8B` loaded with Transformers 5.5 but its model card supports Spanish and many major languages, not Quechua/Chanka. Treat it as a possible SFT base only, not a zero-shot Chanka translator.
- TranslateGemma supports 55 languages and raises on unsupported chat-template language codes in principle, but `qu` produced non-Quechua output here. Do not full-evaluate it unless a verified Quechua/Chanka code or alternate prompt is found.

## Unsloth Compatibility

Tiny Gemma 4 E2B SFT smoke:

```bash
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage broad \
  --model-id google/gemma-4-E2B-it \
  --max-train-samples 16 \
  --max-eval-samples 4 \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 256 \
  --output-dir outputs/smoke_gemma4_e2b_broad_lora2
```

Result: passed with validation enabled and wrote `outputs/smoke_gemma4_e2b_broad_lora2/broad/final_lora`. Final smoke eval loss was `6.6835`.

Important fix: response-only SFT masking must use Gemma 4 markers `"<|turn>user\n"` and `"<|turn>model\n"`. The previous Qwen-only markers masked out every Gemma 4 training row.

## Decision

Do not spend a full eval on these zero-shot models yet. The only plausible path for Gemma 4 or Hy-MT2 is supervised adaptation. For the next serious model-family experiment, run a controlled broad-SFT smoke-to-small-run on Gemma 4 E4B or Hy-MT2 and evaluate the resulting LoRA against the same clean Chanka split.
