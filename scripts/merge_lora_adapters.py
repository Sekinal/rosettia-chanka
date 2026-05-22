"""Create a weighted soup of compatible LoRA adapters.

This is intentionally simpler than a full model merge: it averages adapter
tensors that share the same base model, LoRA rank, alpha, and module shapes.
The resulting directory is a PEFT adapter that can be loaded anywhere the
original adapters can be loaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--name",
        default="lora_soup",
        help="Human-readable name recorded in merge_metadata.json.",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    config_path = path / ADAPTER_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    return json.loads(config_path.read_text())


def check_configs(paths: Sequence[Path]) -> dict[str, Any]:
    configs = [load_config(path) for path in paths]
    reference = configs[0]
    comparable_keys = [
        "base_model_name_or_path",
        "peft_type",
        "task_type",
        "r",
        "lora_alpha",
        "bias",
        "use_dora",
    ]
    for path, config in zip(paths[1:], configs[1:], strict=True):
        for key in comparable_keys:
            if config.get(key) != reference.get(key):
                raise ValueError(
                    f"Incompatible adapter config for {path}: {key}={config.get(key)!r} "
                    f"!= {reference.get(key)!r}"
                )
        if sorted(config.get("target_modules", [])) != sorted(reference.get("target_modules", [])):
            raise ValueError(f"Incompatible target_modules for {path}")
    return reference


def normalized_weights(count: int, weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, got {len(weights)}")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value")
    return [weight / total for weight in weights]


def merge_state_dicts(paths: Sequence[Path], weights: Sequence[float]) -> dict[str, Any]:
    from safetensors.torch import load_file

    merged: dict[str, Any] = {}
    expected_keys: set[str] | None = None
    for path, weight in zip(paths, weights, strict=True):
        weights_path = path / ADAPTER_WEIGHTS
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing {weights_path}")
        state = load_file(weights_path)
        keys = set(state)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise ValueError(f"Incompatible tensor keys for {path}: missing={missing[:5]} extra={extra[:5]}")
        for key, tensor in state.items():
            value = tensor.to(dtype=tensor.dtype) * weight
            if key in merged:
                if merged[key].shape != value.shape:
                    raise ValueError(f"Shape mismatch for {key} in {path}")
                merged[key] = merged[key] + value
            else:
                merged[key] = value.clone()
    return merged


def copy_optional_adapter_files(source: Path, output_dir: Path) -> None:
    for filename in ["README.md", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"]:
        source_path = source / filename
        if source_path.exists():
            shutil.copy2(source_path, output_dir / filename)


def main() -> None:
    args = parse_args()
    adapters = [path.resolve() for path in args.adapter]
    if not adapters:
        raise ValueError("At least one --adapter is required")
    weights = normalized_weights(len(adapters), args.weight)
    config = check_configs(adapters)
    merged = merge_state_dicts(adapters, weights)

    from safetensors.torch import save_file

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ADAPTER_CONFIG).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    save_file(merged, args.output_dir / ADAPTER_WEIGHTS)
    copy_optional_adapter_files(adapters[0], args.output_dir)
    metadata = {
        "name": args.name,
        "adapters": [str(path) for path in adapters],
        "weights": weights,
        "tensor_count": len(merged),
    }
    (args.output_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
