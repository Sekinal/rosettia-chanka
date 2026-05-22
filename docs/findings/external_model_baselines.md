# External Model Baselines

Date: 2026-05-22

Purpose: test whether current external translation/general models can directly beat or usefully seed the RosettIA Chanka translator before spending time on larger SFT/GSPO runs.

## Standing Internal Baseline

Current deployable best is:

`outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-lr3e-7-32steps/checkpoint-24`

Use it with terminology top-1 inference from `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`.

Canonical held-out eval:

- Selection: `26.8011`
- chrF++: `41.3017`
- BLEU: `9.5399`
- Token F1: `26.6291`
- Source-copy: `2.8270%`
- Spanish leakage: `0.4747%`
- TER: `87.5576`

## Zero-Shot Smokes

All smokes used 5 rows from the clean Chanka validation split. These are not full evaluations; they are viability checks before spending GPU on larger tests.

| Model | Backend | chrF++ | BLEU | Token F1 | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| `tencent/Hy-MT2-1.8B` | `causal-chat` | `12.6362` | `2.8117` | `0.0` | Not Chanka; mixed Spanish/other-language-looking output. |
| `google/gemma-4-E2B-it` | `causal-chat` with Gemma 4 processor | `8.2584` | `3.8436` | `0.0` | Malformed/repetitive Chanka-like output. |
| `google/gemma-4-E4B-it` | `causal-chat` with Gemma 4 processor | `13.5899` | `4.4485` | `0.0` | Cleaner Chanka-like text than E2B, but semantically poor. |
| `google/translategemma-4b-it` | `translategemma`, `es -> qu` | `2.4792` | `3.6688` | `0.0` | Invalid for our target; generated Devanagari-like text. |
| `google/t5gemma-2-270m-270m` | `seq2seq-chat` | `8.5113` | `0.3152` | `0.0` | Became accessible after HF auth; still not instruction/translation ready for this target. |

Earlier full baseline:

- `facebook/nllb-200-distilled-600M`, `spa_Latn -> quy_Latn`: chrF++ `24.1377`, BLEU `0.9596`, token F1 `10.1348`, TER `166.1290`.

## Implementation Notes

- `scripts/evaluate_translation_baseline.py` now supports `causal-chat` models.
- Gemma 4 models require the processor/chat-template path with `enable_thinking=False`; the tokenizer-only path leaked thinking artifacts.
- `tencent/Hy-MT2-1.8B` loaded with Transformers 5.5 but its model card supports Spanish and many major languages, not Quechua/Chanka. Treat it as a possible SFT base only, not a zero-shot Chanka translator.
- TranslateGemma supports 55 languages and raises on unsupported chat-template language codes in principle, but `qu` produced non-Quechua output here. Do not full-evaluate it unless a verified Quechua/Chanka code or alternate prompt is found.
- T5Gemma became accessible after the newer HF token was installed on the remote. The 5-row smoke wrote `outputs/external_baselines/t5gemma_2_270m_5row_after_auth_metrics.json` and remained very weak: chrF++ `8.5113`, BLEU `0.3152`, token F1 `0.0`, Spanish leakage `5.3140%`, TER `714.2857`.

## Unsloth Compatibility

`scripts/train_sft_unsloth.py` supports both `--stage broad` and `--stage chanka` for model-family canaries. The Chanka stage uses the reviewed clean Chanka corpus and keeps validation/checkpointing enabled.

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

Hy-MT2 1.8B SFT smoke:

```bash
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id tencent/Hy-MT2-1.8B \
  --max-train-samples 8 \
  --max-eval-samples 4 \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 128 \
  --output-dir outputs/smoke_hymt2_1_8b_chanka_lora2
```

Result: passed with validation enabled and wrote `outputs/smoke_hymt2_1_8b_chanka_lora2/chanka/final_lora`. Unsloth recognizes this model as `Hunyuan_V1_Dense`. Final smoke eval loss was `6.7444`.

Important fix: response-only SFT masking must also detect Hy-MT2 markers `"<｜hy_User｜>"` and `"<｜hy_Assistant｜>"`.

## Gemma 4 E4B Chanka SFT

Small clean-Chanka canary:

```bash
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id google/gemma-4-E4B-it \
  --max-train-samples 128 \
  --max-eval-samples 32 \
  --max-steps 32 \
  --eval-steps 8 \
  --save-steps 8 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 128 \
  --learning-rate 2e-5 \
  --lora-r 64 \
  --lora-alpha 128 \
  --output-dir outputs/gemma4_e4b_chanka_lora_canary_r64_a128_s128_32steps
```

This verified that Gemma 4 E4B can train on the clean Chanka corpus. Eval loss improved from `6.665` at step 8 to `5.905` after final eval. Full held-out generation from `final_lora` was still weak: chrF++ `15.1027`, BLEU `1.0070`, token F1 `4.1937`, TER `174.8848`.

Two-epoch clean-Chanka SFT:

```bash
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id google/gemma-4-E4B-it \
  --num-train-epochs 2 \
  --eval-steps 56 \
  --save-steps 56 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 128 \
  --learning-rate 2e-5 \
  --lora-r 64 \
  --lora-alpha 128 \
  --output-dir outputs/gemma4_e4b_chanka_lora_full_r64_a128_s128_2ep
```

Training loss and eval loss improved, but generated translation quality remained far below the standing Qwen adapter:

| Adapter | chrF++ | BLEU | Token F1 | TER | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `checkpoint-392` | `16.1325` | `1.4071` | `4.1030` | `142.8571` | Best external generation among tested checkpoints. |
| `checkpoint-450` | `15.4580` | `1.2037` | `4.1537` | `157.6037` | Best trainer eval loss, worse generation metrics. |

Qualitative pattern: the model stops emitting explanations and starts producing Chanka-like text, but it is often generic or semantically wrong. Examples from `checkpoint-392`: `¿En qué calle vive? -> ¿Imayna riqsi?`, `Yo vivo en Quinua -> Ñuqa Quinuawan kachkani`, `No es un buen esposo -> Mana allin waynaqmi`, and `Tengo 45 años -> 45 uraymi kani`.

## Decision

Gemma 4 E4B and Hy-MT2 can both be used with Unsloth after chat-marker fixes, but Gemma 4 E4B clean-Chanka SFT is not competitive with the current Qwen3.5 adapter. It does not justify replacing the Qwen base. If these families are revisited, they need broad Quechua SFT first, then clean Chanka SFT/GSPO, not clean Chanka alone.
