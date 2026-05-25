"""Mergekit-style task-vector amplification on v13.

We have:
  - base = 4B full-SFT checkpoint-36 (full model)
  - v13_merged = v13 LoRA merged into base (we'll produce it if not present)
  - delta = v13_merged - base  (the task vector learned by v11→v12→v13)

Amplification: write amplified = base + alpha * delta for several alphas,
save each as a full model, and let the caller run the standard eval against
each.

For 4B safetensors this is just per-tensor arithmetic, ~9 GB intermediate
state. We stream tensor-by-tensor to keep memory bounded.
"""
from __future__ import annotations
import argparse, json, os, sys, gc
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    sd = {}
    shards = sorted(model_dir.glob("model*.safetensors"))
    if not shards:
        shards = [model_dir / "model.safetensors"]
    for shard in shards:
        sd.update(load_file(str(shard)))
    return sd


def save_full_model(sd: dict[str, torch.Tensor], out_dir: Path, template_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out_dir / "model.safetensors"))
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "generation_config.json",
                  "processor_config.json"):
        src = template_dir / fname
        if src.exists():
            import shutil
            shutil.copy(src, out_dir / fname)


def amplify(base_sd: dict[str, torch.Tensor],
            v13_sd: dict[str, torch.Tensor],
            alpha: float) -> dict[str, torch.Tensor]:
    out = {}
    for k in base_sd:
        if k not in v13_sd:
            out[k] = base_sd[k].clone()
            continue
        b = base_sd[k]
        v = v13_sd[k]
        if b.shape != v.shape or not torch.is_floating_point(b):
            out[k] = v.clone()  # tied embeddings or non-trainable tensor
            continue
        delta = v.float() - b.float()
        out[k] = (b.float() + alpha * delta).to(b.dtype)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--v13-merged-dir", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[1.0, 1.25, 1.5, 1.75, 2.0])
    args = parser.parse_args()

    print(f"loading base {args.base_dir}", flush=True)
    base_sd = load_state_dict(args.base_dir)
    print(f"loading v13_merged {args.v13_merged_dir}", flush=True)
    v13_sd = load_state_dict(args.v13_merged_dir)

    for alpha in args.alphas:
        out_dir = args.out_root / f"alpha{alpha:.2f}"
        print(f"\namplifying alpha={alpha} -> {out_dir}", flush=True)
        amp = amplify(base_sd, v13_sd, alpha)
        save_full_model(amp, out_dir, args.v13_merged_dir)
        del amp
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
