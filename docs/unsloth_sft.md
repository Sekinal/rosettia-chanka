# Unsloth SFT Workflow

This project uses Unsloth for SFT on Qwen3.5 with **16-bit LoRA**, not QLoRA.

Official Unsloth points checked on 2026-05-20:

- Qwen3.5 4B BF16 LoRA is listed at about 10 GB VRAM, so it should fit on the RTX 3090 with conservative batch/context settings.
- Unsloth says Qwen3.5 QLoRA / 4-bit training is not recommended because quantization differences are higher than normal.
- Their Qwen3.5 code recipe uses `FastLanguageModel`, `load_in_4bit=False`, `load_in_16bit=True`, `SFTTrainer`, `SFTConfig`, `optim="adamw_8bit"`, and `use_gradient_checkpointing="unsloth"`.
- Their LoRA guide recommends targeting all major linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Validation matters for this project because the Chanka clean set is small; every training command below uses an eval split and keeps the best checkpoint by `eval_loss`.
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

## Broad SFT

```bash
python scripts/train_sft_unsloth.py \
  --stage broad \
  --output-dir outputs/qwen35_4b_broad \
  --max-seq-length 1024 \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --lora-r 64 \
  --lora-alpha 128 \
  --gradient-accumulation-steps 16 \
  --eval-steps 250 \
  --save-steps 250
```

The broad stage uses:

- `broad_quechua/somosnlp_spanish_to_quechua_high_quality_sft.parquet`
- `americasnlp/americasnlp_quechua_spanish_high_quality_real_sft.parquet`

## Clean Chanka SFT

Run this after broad SFT and continue from the broad adapter:

```bash
python scripts/train_sft_unsloth.py \
  --stage chanka \
  --adapter-path outputs/qwen35_4b_broad/broad/final_lora \
  --output-dir outputs/qwen35_4b_chanka \
  --max-seq-length 1024 \
  --num-train-epochs 8 \
  --learning-rate 2e-5 \
  --lora-r 128 \
  --lora-alpha 256 \
  --gradient-accumulation-steps 8 \
  --eval-steps 10 \
  --save-steps 10
```

The Chanka stage uses a 15% validation split by default to reduce overfit risk.
