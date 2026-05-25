"""Layer-level activation analysis comparing v13 vs base 4B on Chanka inputs.

For each transformer layer, compute the cosine distance between activations
of v13 (LoRA on base) and base alone on a batch of LEAK-FREE train Chanka
sources. Layers with largest activation diff are where v13 most differs
from base — those are 'translation-relevant' regions.

This is the raw-PyTorch substitute for TransformerLens analysis. ~30 lines
of forward hooks; no TL dependency.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, "/root/rosettia-chanka")
from scripts import train_gspo_chanka_unsloth as gspo


def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    # a, b: (B, T, D) — mean cosine distance across (B,T) positions
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    cs = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return float((1.0 - cs).mean().item())


def collect_residual_streams(model, tokenizer, prompts: list[str], max_len: int = 128
                              ) -> dict[int, torch.Tensor]:
    """Return per-layer residual-stream activations (post-attention + post-MLP block output)."""
    activations: dict[int, list[torch.Tensor]] = {}
    handles = []
    # Qwen3.5 (multimodal) layer paths vary by load wrapper:
    #   base full model loaded via Unsloth: model.model.language_model.layers
    #   PEFT-wrapped (adapter loaded):     model.base_model.model.model.language_model.layers
    import torch.nn as nn
    layers = None
    for name, mod in model.named_modules():
        if hasattr(mod, "layers"):
            l = getattr(mod, "layers")
            if isinstance(l, (list, nn.ModuleList)) and len(l) >= 16:
                layers = l
                break
    if layers is None:
        raise RuntimeError("Could not find decoder layers attribute on model")
    n_layers = len(layers)

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            # output is a tuple for decoder layers in HF; take hidden states
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            activations.setdefault(idx, []).append(hs.detach().cpu())
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        with torch.inference_mode():
            # Qwen3.5 uses a multimodal processor that mis-interprets positional
            # input as image URLs. Use text= keyword.
            inputs = tokenizer(text=prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_len).to(model.device)
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return {i: torch.cat(v, dim=0) for i, v in activations.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--v13-adapter", required=True)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    # Load leak-free train sources
    rows = gspo.load_chanka_rows(gspo.DATASET_REPO, gspo.CHANKA_FILE)
    train_rows, eval_rows = gspo.split_rows(rows, validation_fraction=0.15, seed=3407,
                                             max_train_samples=None, max_eval_samples=None)
    eval_sources = {r["source"] for r in eval_rows}
    train_safe = [r for r in train_rows if r["source"] not in eval_sources][:args.n_samples]
    prompts = [
        f"Traduce del español al quechua chanka:\nEspañol: {r['source']}\nQuechua chanka:"
        for r in train_safe
    ]
    print(f"using {len(prompts)} leak-free train sources for activation diff", flush=True)

    from unsloth import FastLanguageModel

    print(f"loading base model {args.base_model}", flush=True)
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    FastLanguageModel.for_inference(base_model)
    base_acts = collect_residual_streams(base_model, tokenizer, prompts, max_len=args.max_seq_length)
    print(f"collected base residual streams: {len(base_acts)} layers", flush=True)
    del base_model
    import gc; gc.collect(); torch.cuda.empty_cache()

    print(f"loading v13 adapter {args.v13_adapter}", flush=True)
    v13_model, _ = FastLanguageModel.from_pretrained(
        model_name=args.v13_adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    FastLanguageModel.for_inference(v13_model)
    v13_acts = collect_residual_streams(v13_model, tokenizer, prompts, max_len=args.max_seq_length)
    print(f"collected v13 residual streams: {len(v13_acts)} layers", flush=True)
    del v13_model
    gc.collect(); torch.cuda.empty_cache()

    layer_diffs = {}
    for layer in sorted(base_acts.keys()):
        if layer not in v13_acts:
            continue
        # Match shapes (in case padding differs)
        a, b = base_acts[layer], v13_acts[layer]
        min_t = min(a.shape[1], b.shape[1])
        d = cosine_dist(a[:, :min_t], b[:, :min_t])
        layer_diffs[layer] = d

    sorted_layers = sorted(layer_diffs.items(), key=lambda kv: kv[1], reverse=True)
    print("\n=== Top-10 layers by activation cosine distance (v13 vs base) ===")
    for layer, d in sorted_layers[:10]:
        print(f"  layer {layer:>3d}: cos_dist = {d:.4f}")
    print("\n=== All layers (sorted) ===")
    for layer, d in sorted(layer_diffs.items()):
        print(f"  layer {layer:>3d}: cos_dist = {d:.4f}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "base_model": args.base_model,
        "v13_adapter": args.v13_adapter,
        "n_samples": len(prompts),
        "max_seq_length": args.max_seq_length,
        "layer_cos_dist": {str(k): v for k, v in sorted(layer_diffs.items())},
        "top10_layers": [{"layer": k, "cos_dist": v} for k, v in sorted_layers[:10]],
    }, open(args.output_json, "w"), indent=2)
    print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
