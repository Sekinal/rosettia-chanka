# Unsloth SFT Workflow

This project uses Unsloth for SFT on Qwen3.5 with **16-bit LoRA-style adapters**, not QLoRA.

Official Unsloth points checked on 2026-05-20:

- The first comparison pass uses `unsloth/Qwen3.5-2B` to keep smoke tests and ablations cheaper before scaling back up.
- Unsloth says Qwen3.5 QLoRA / 4-bit training is not recommended because quantization differences are higher than normal.
- Their Qwen3.5 code recipe uses `FastLanguageModel`, `load_in_4bit=False`, `load_in_16bit=True`, `SFTTrainer`, `SFTConfig`, `optim="adamw_8bit"`, and `use_gradient_checkpointing="unsloth"`.
- Their LoRA guide recommends targeting all major linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- The training script exposes `--adapter-method lora|dora|rslora` so LoRA, DoRA, and rank-stabilized LoRA can be compared with the same data split and step schedule.
- Validation matters for this project because the Chanka clean set is small; every training command below uses an eval split and keeps the best checkpoint by `eval_loss`.
- If `--eval-steps` is not passed, the script evaluates several times inside each epoch via `--evals-per-epoch` instead of waiting for epoch boundaries. This is meant to catch Chanka overfitting early.
- Saved Unsloth LoRA adapters can be loaded with `FastLanguageModel.from_pretrained(...)` for continued finetuning; the script exposes this as `--adapter-path`.

Sources:

- https://unsloth.ai/docs/models/qwen3.5/fine-tune
- https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
- https://unsloth.ai/docs/get-started/install
- https://unsloth.ai/docs/basics/continued-pretraining

## Remote Setup

On the training server:

```bash
git clone https://github.com/Sekinal/rosettia-chanka.git
cd rosettia-chanka
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install polars pyarrow datasets huggingface_hub trl accelerate
```

## Smoke Test

Use a tiny run first to verify the model, tokenizer, dataset download, validation, and checkpoint path:

```bash
python scripts/train_sft_unsloth.py \
  --stage broad \
  --max-train-samples 32 \
  --max-eval-samples 8 \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 1 \
  --output-dir outputs/smoke
```

This smoke test passed on the RTX 3090 server on 2026-05-20. The first run downloads and caches the model, so it is much slower than later runs. Flash Attention 2 was not installed on that server, so Unsloth used the XFormers fallback.

## Broad SFT

Use this stage first for main training. It uses the broad, non-judicial SomosNLP and AmericasNLP data.

```bash
python scripts/train_sft_unsloth.py \
  --stage broad \
  --adapter-method lora \
  --output-dir outputs/qwen35_2b_broad \
  --max-seq-length 512 \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --lora-r 64 \
  --lora-alpha 128 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --eval-steps 250 \
  --save-steps 250
```

The broad stage uses:

- `broad_quechua/somosnlp_spanish_to_quechua_high_quality_sft.parquet`
- `americasnlp/americasnlp_quechua_spanish_high_quality_real_sft.parquet`

On 2026-05-20 the formatted broad corpus had p99 length 216 tokens and max length 443 tokens, so `max_seq_length=512` covers the current broad data. Short VRAM probes at seq512 fit batch 16 on the L40S; the default uses batch 8 with gradient accumulation 2 to preserve an effective batch size of 16.

## Chanka GSPO

Do not run Chanka as SFT. The reviewed judicial Chanka data is reserved for final GSPO. The SFT script rejects `--stage chanka` from the CLI so the judicial data is not accidentally used in SFT.

On 2026-05-20 the formatted Chanka data had p99 length 99 tokens and max length 115 tokens. A calibration-only LoRA r64/a128 run with batch 8 and gradient accumulation 1 peaked at about 6.3 GiB VRAM on the L40S remote host. Keep those measurements as GSPO planning context, not as an SFT training recommendation.

## Adapter Comparison Pass

Start with the 2B model and identical tiny smoke tests:

```bash
for method in lora dora rslora; do
  python scripts/train_sft_unsloth.py \
    --stage chanka \
    --model-id unsloth/Qwen3.5-2B \
    --adapter-method "$method" \
    --max-train-samples 64 \
    --max-eval-samples 16 \
    --max-steps 2 \
    --eval-steps 1 \
    --save-steps 1 \
    --output-dir "outputs/smoke_qwen35_2b_${method}"
done
```

Then compare short broad-data runs with the same seed and validation split. Keep the same `--evals-per-epoch` across methods so eval loss curves are comparable. Chanka comparisons belong in the GSPO workflow, not this SFT script.
