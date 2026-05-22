"""Export an Unsloth LoRA adapter as a merged full model.

Use this when full-finetuning should continue from an adapter-trained model
state. The generated model weights are large and must stay out of git.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True, help="Local or HF Unsloth/PEFT adapter path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for merged full model weights.")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument(
        "--save-method",
        choices=["merged_16bit", "merged_4bit"],
        default="merged_16bit",
        help="Unsloth merge format. Use merged_16bit for subsequent full finetuning.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(args.output_dir), tokenizer, save_method=args.save_method)
    print(f"Saved merged model to {args.output_dir}")


if __name__ == "__main__":
    main()
