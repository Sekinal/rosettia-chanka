# External Model Baselines

Date: 2026-05-22

Purpose: test whether current external translation/general models can directly beat or usefully seed the RosettIA Chanka translator before spending time on larger SFT/GSPO runs.

## Standing Internal Baseline

Current deployable best is:

`outputs/mbr_self_training_sft/20260522-k16-fullnewbest-noterm-margin000-clean512-termtrain-lr2e-7-24steps/checkpoint-8`

Use it with terminology top-1 inference from `clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet`.

Canonical held-out eval:

- Selection: `27.0396`
- chrF++: `41.4823`
- BLEU: `9.7158`
- Token F1: `27.0736`
- Source-copy: `2.8270%`
- Spanish leakage: `0.4747%`
- TER: `87.3272`

## Zero-Shot Smokes

All smokes used 5 rows from the clean Chanka validation split. These are not full evaluations; they are viability checks before spending GPU on larger tests.

| Model | Backend | chrF++ | BLEU | Token F1 | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| `tencent/Hy-MT2-1.8B` | `causal-chat` | `12.6362` | `2.8117` | `0.0` | Not Chanka; mixed Spanish/other-language-looking output. |
| `tencent/Hy-MT2-7B` | `causal-chat`, `hymt2` prompt | `8.5036` | `2.1364` | `0.0` | Not Chanka; source-copied Spanish under both `Quechua Chanka` and `Quechua` target names. |
| `google/gemma-4-E2B-it` | `causal-chat` with Gemma 4 processor | `8.2584` | `3.8436` | `0.0` | Malformed/repetitive Chanka-like output. |
| `google/gemma-4-E4B-it` | `causal-chat` with Gemma 4 processor | `13.5899` | `4.4485` | `0.0` | Cleaner Chanka-like text than E2B, but semantically poor. |
| `google/translategemma-4b-it` | `translategemma`, `es -> qu` | `2.4792` | `3.6688` | `0.0` | Invalid for our target; generated Devanagari-like text. |
| `google/t5gemma-2-270m-270m` | `seq2seq-chat` | `8.5113` | `0.3152` | `0.0` | Became accessible after HF auth; still not instruction/translation ready for this target. |

Earlier full baseline:

- `facebook/nllb-200-distilled-600M`, `spa_Latn -> quy_Latn`: chrF++ `24.1377`, BLEU `0.9596`, token F1 `10.1348`, TER `166.1290`.

## Implementation Notes

- `scripts/evaluate_translation_baseline.py` now supports `causal-chat` models.
- `scripts/evaluate_translation_baseline.py` supports `--adapter-path`, `--prompt-style hymt2`, target-language names, and generation knobs. Use this for evaluating non-Qwen LoRA adapters with the same external metrics as base models.
- Gemma 4 models require the processor/chat-template path with `enable_thinking=False`; the tokenizer-only path leaked thinking artifacts.
- HF model search on 2026-05-22 showed current Google Gemma 4 entries including `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`, and `google/gemma-4-31B-it`. Current Hy-MT2 entries include `tencent/Hy-MT2-1.8B`, `tencent/Hy-MT2-7B`, and `tencent/Hy-MT2-30B-A3B`.
- The official Hy-MT2 documentation says the family supports 33 languages and lists Spanish, but not Quechua. It also recommends a user-only translation prompt, so zero-shot and SFT tests should use `--prompt-style hymt2` rather than the generic system prompt.
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

Hy-MT2 7B SFT smoke:

```bash
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id tencent/Hy-MT2-7B \
  --prompt-style hymt2 \
  --target-language-name "Quechua Chanka" \
  --max-train-samples 16 \
  --max-eval-samples 4 \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 128 \
  --learning-rate 2e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --output-dir outputs/smoke_hymt2_7b_chanka_hymt2prompt_lora_r32_s128_2steps
```

Result: passed with validation enabled and wrote `outputs/smoke_hymt2_7b_chanka_hymt2prompt_lora_r32_s128_2steps/chanka/final_lora`. Final smoke eval loss was `6.9954`, down from step-1 eval loss `7.230`.

5-row adapter eval with the same Hy-MT2 prompt still source-copied Spanish:

```bash
.venv/bin/python scripts/evaluate_translation_baseline.py \
  --backend causal-chat \
  --model-id tencent/Hy-MT2-7B \
  --adapter-path outputs/smoke_hymt2_7b_chanka_hymt2prompt_lora_r32_s128_2steps/chanka/final_lora \
  --prompt-style hymt2 \
  --target-language-name "Quechua Chanka" \
  --max-eval-samples 5 \
  --batch-size 1 \
  --max-new-tokens 80 \
  --torch-dtype bfloat16 \
  --do-sample \
  --temperature 0.1 \
  --top-p 0.7 \
  --top-k 20 \
  --repetition-penalty 1.05 \
  --output-json outputs/external_baselines/hymt2_7b_hymt2prompt_lora2step_5row_metrics.json
```

Metrics matched the base smoke: chrF++ `8.5036`, BLEU `2.1364`, token F1 `0.0`, source-copy `100.0%`, TER `171.4286`. This is only a wiring smoke, not evidence against a real broad-then-Chanka Hy-MT2 7B run.

Important fix: Hy-MT2 7B response-only SFT masking must detect `"<|extra_4|>"` and `"<|extra_0|>"`. The earlier 1.8B markers are different.

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

Broad-to-Chanka SFT:

```bash
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage broad \
  --model-id google/gemma-4-E4B-it \
  --max-train-samples 2048 \
  --max-eval-samples 128 \
  --max-steps 96 \
  --eval-steps 24 \
  --save-steps 24 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 256 \
  --learning-rate 1e-4 \
  --lora-r 64 \
  --lora-alpha 128 \
  --output-dir outputs/gemma4_e4b_broad_lora_r64_a128_s256_96steps
```

Broad validation improved overall but was not monotonic.

| Checkpoint | Eval loss |
| --- | ---: |
| `checkpoint-24` | `5.327` |
| `checkpoint-48` | `4.812` |
| `checkpoint-72` | `4.640` |
| `checkpoint-96` | `4.740` |
| `final_lora` | `4.5090` |

Chanka continuation from the broad adapter:

```bash
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id google/gemma-4-E4B-it \
  --adapter-path outputs/gemma4_e4b_broad_lora_r64_a128_s256_96steps/broad/final_lora \
  --max-train-samples 512 \
  --max-eval-samples 128 \
  --max-steps 96 \
  --eval-steps 24 \
  --save-steps 24 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 128 \
  --learning-rate 2e-5 \
  --lora-r 64 \
  --lora-alpha 128 \
  --output-dir outputs/gemma4_e4b_broad96_chanka_lora_s128_96steps
```

Chanka validation improved from `4.362` at step 24 to `4.1666` final. Full held-out generation was worse than the clean-only checkpoint eval:

| Adapter | chrF++ | BLEU | Token F1 | Source copy % | Leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `outputs/gemma4_e4b_broad96_chanka_lora_s128_96steps/chanka/final_lora` | `14.9705` | `0.7609` | `2.4589` | `3.6920` | `0.0` | `129.4931` |

Important evaluator fix: PEFT cannot attach Gemma 4 LoRA adapters to `Gemma4ClippableLinear`, so Gemma 4 adapter eval must use the Unsloth adapter loader. `scripts/evaluate_translation_baseline.py` now supports `--adapter-loader auto|peft|unsloth`; `auto` uses Unsloth for Gemma 4 adapters and PEFT otherwise. The Unsloth path loads the saved LoRA directory with `FastLanguageModel.from_pretrained`, calls `FastLanguageModel.for_inference`, and preserves the Gemma chat template.

Qualitative pattern: broad+Chanka SFT did not fix the core semantic issue. It avoids Spanish leakage and emits Chanka-looking fragments, but the translations are usually wrong: `¿En qué calle vive? -> Imay kancha?`, `No es un buen esposo -> Chanka uyaytaqtaq`, `En la fiesta -> Pachamamañan`, `Tengo 45 años -> 45 qillqayni`.

## Hy-MT2 7B Broad-to-Chanka SFT

Purpose: test the strongest plausible Hy-MT2 path after zero-shot source-copying: first teach broad Spanish-to-Quechua behavior, then specialize to the clean Chanka split.

Broad canary:

```bash
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage broad \
  --model-id tencent/Hy-MT2-7B \
  --prompt-style hymt2 \
  --target-language-name Quechua \
  --max-train-samples 2048 \
  --max-eval-samples 128 \
  --max-steps 96 \
  --eval-steps 24 \
  --save-steps 24 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 256 \
  --learning-rate 1e-4 \
  --lora-r 64 \
  --lora-alpha 128 \
  --output-dir outputs/hymt2_7b_broad_hymt2prompt_lora_r64_a128_s256_96steps
```

Result: validation loss improved monotonically over the 96-step run.

| Checkpoint | Eval loss |
| --- | ---: |
| `checkpoint-24` | `3.446` |
| `checkpoint-48` | `3.041` |
| `checkpoint-72` | `2.863` |
| `checkpoint-96` / `final_lora` | `2.7638` |

Chanka continuation from the broad adapter:

```bash
.venv/bin/python scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id tencent/Hy-MT2-7B \
  --adapter-path outputs/hymt2_7b_broad_hymt2prompt_lora_r64_a128_s256_96steps/broad/final_lora \
  --prompt-style hymt2 \
  --target-language-name "Quechua Chanka" \
  --max-train-samples 512 \
  --max-eval-samples 128 \
  --max-steps 96 \
  --eval-steps 24 \
  --save-steps 24 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 128 \
  --learning-rate 2e-5 \
  --lora-r 64 \
  --lora-alpha 128 \
  --output-dir outputs/hymt2_7b_broad96_chanka_hymt2prompt_lora_s128_96steps
```

Result: validation loss again improved monotonically.

| Checkpoint | Eval loss |
| --- | ---: |
| `checkpoint-24` | `3.025` |
| `checkpoint-48` | `2.808` |
| `checkpoint-72` | `2.716` |
| `checkpoint-96` / `final_lora` | `2.6787` |

Full held-out clean Chanka generation eval:

| Decoding | chrF++ | BLEU | Token F1 | Source copy % | Leakage % | TER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | `19.4632` | `3.1464` | `4.9005` | `3.5759` | `0.3165` | `101.6129` |
| `temperature=0.1`, `top_p=0.7`, `top_k=20`, `repetition_penalty=1.05` | `18.9966` | `2.7194` | `4.7212` | `3.2595` | `0.4747` | `102.9954` |

Qualitative pattern: broad+Chanka SFT fixes the zero-shot source-copy failure and produces Chanka-like text, but semantic fidelity is still poor. Examples from deterministic decoding:

| Spanish | Prediction | Reference |
| --- | --- | --- |
| `¿En qué calle vive?` | `¿Ima kallpanpiytaqmi wawaykikusqanki?` | `¿May k’ikllupin tiyanki?` |
| `Yo vivo en Quinua` | `Quinuapi kachkani` | `Quinuapim tiyani` |
| `No es un buen esposo` | `Mana allin kachkani` | `Manam allin qusachu` |
| `Tengo 45 años` | `Manam 45 wawaytaqmi` | `Tawa chunka pichqanpim kachkani` |

Decision: this is a useful negative result. Hy-MT2 7B can be adapted technically and no longer source-copies after broad+Chanka SFT, but this 96+96 step recipe is far below the standing Qwen3.5 adapter (`chrF++ 41.4823`, BLEU `9.7158`, token F1 `27.0736`). Do not replace Qwen with Hy-MT2 from these results. Scaling Hy-MT2 would require much more broad data/steps and close early evaluation, not clean-Chanka-only or tiny broad SFT.

## Decision

Gemma 4 E4B and Hy-MT2 can both be used with Unsloth after chat-marker fixes, but Gemma 4 E4B clean-only SFT, Gemma 4 E4B broad-to-Chanka SFT, and the tiny Hy-MT2 7B broad-to-Chanka run are not competitive with the current Qwen3.5 adapter. Neither family justifies replacing the Qwen base from current results. If these families are revisited, they need much larger broad Quechua SFT before clean Chanka SFT/GSPO, with generation evals at checkpoints rather than relying on trainer loss.
