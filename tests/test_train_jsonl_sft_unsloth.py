from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts import train_jsonl_sft_unsloth as train_jsonl_sft


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class TrainJsonlSftUnslothTests(unittest.TestCase):
    def test_load_jsonl_rows_uses_prediction_as_target_and_dedupes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mbr_predictions.jsonl"
            write_jsonl(
                path,
                [
                    {"source": "Buenos dias.", "prediction": "Allin punchaw.", "reference": "Allin punchaw"},
                    {"source": "Buenos dias.", "prediction": "Allin punchaw.", "reference": "Duplicate"},
                    {"source": "Sin salida", "prediction": "", "reference": "Mana"},
                ],
            )
            args = train_jsonl_sft.parse_args(["--jsonl", str(path), "--output-dir", str(Path(tmpdir) / "out")])

            rows = train_jsonl_sft.load_jsonl_rows(args)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "Buenos dias.")
        self.assertEqual(rows[0]["target"], "Allin punchaw.")
        self.assertEqual(rows[0]["reference"], "Allin punchaw")
        self.assertEqual(rows[0]["target_field"], "prediction")

    def test_load_jsonl_rows_filters_exact_source_copies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mbr_predictions.jsonl"
            write_jsonl(path, [{"source": "Copiame.", "prediction": "Copiame.", "reference": "Qillqay."}])
            args = train_jsonl_sft.parse_args(["--jsonl", str(path), "--output-dir", str(Path(tmpdir) / "out")])

            with self.assertRaisesRegex(RuntimeError, "No JSONL rows"):
                train_jsonl_sft.load_jsonl_rows(args)

    def test_format_example_uses_general_chanka_prompt_and_target_only(self):
        tokenizer = DummyTokenizer()
        row = {
            "source": "Buenos dias.",
            "target": "Allin punchaw.",
            "reference": "Referencia no entrenable.",
            "source_name": "mbr",
            "variant": "quy/chanka_pseudo",
            "target_field": "prediction",
        }

        formatted = train_jsonl_sft.format_example(tokenizer, row)

        self.assertIn("quechua chanka", formatted["text"])
        self.assertIn("Buenos dias.", formatted["text"])
        self.assertIn("Allin punchaw.", formatted["text"])
        self.assertNotIn("Referencia no entrenable.", formatted["text"])
        self.assertFalse(tokenizer.tokenize)
        self.assertFalse(tokenizer.add_generation_prompt)

    def test_format_example_can_include_terminology_prompt(self):
        tokenizer = DummyTokenizer()
        row = {
            "source": "¿Es usted casado?",
            "target": "¿Warmiyuqchu kanki?",
            "reference": "",
            "source_name": "mbr",
            "variant": "quy/chanka_pseudo",
            "target_field": "target",
        }

        formatted = train_jsonl_sft.format_example(
            tokenizer,
            row,
            terminology_entries=[("Casado", "Warmiyuq")],
            terminology_top_k=1,
        )

        self.assertIn("Glosario sugerido", formatted["text"])
        self.assertIn("- Casado = Warmiyuq", formatted["text"])
        self.assertIn("¿Warmiyuqchu kanki?", formatted["text"])

    def test_build_dataset_can_count_terminology_matches(self):
        row = {
            "source": "La madre abandonada llego.",
            "target": "Saqisqa mama chayamurqan.",
            "reference": "",
            "source_name": "mbr",
            "variant": "quy/chanka_pseudo",
            "target_field": "target",
        }

        terminology = train_jsonl_sft.terminology_for_row(
            row,
            [("madre abandonada", "saqisqa mama"), ("madre", "mama")],
            terminology_top_k=1,
        )

        self.assertEqual(terminology, [("madre abandonada", "saqisqa mama")])

    def test_step_schedule_uses_max_steps_when_present(self):
        args = argparse.Namespace(
            eval_steps=None,
            save_steps=None,
            max_steps=64,
            evals_per_epoch=8,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
        )

        train_jsonl_sft.configure_step_schedule(args, train_row_count=512)

        self.assertEqual(args.eval_steps, 8)
        self.assertEqual(args.save_steps, 8)


if __name__ == "__main__":
    unittest.main()
