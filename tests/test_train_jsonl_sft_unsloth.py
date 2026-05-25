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


class DummyParameter:
    def __init__(self, size: int, requires_grad: bool):
        self._size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._size


class DummyModel:
    def __init__(self, parameters):
        self._parameters = parameters

    def parameters(self):
        return iter(self._parameters)


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class TrainJsonlSftUnslothTests(unittest.TestCase):
    def test_cli_accepts_full_finetuning_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mbr_predictions.jsonl"
            write_jsonl(path, [{"source": "Buenos dias.", "prediction": "Allin punchaw."}])

            args = train_jsonl_sft.parse_args(
                ["--jsonl", str(path), "--output-dir", str(Path(tmpdir) / "out"), "--training-mode", "full"]
            )

        self.assertEqual(args.training_mode, "full")

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

    def test_load_jsonl_rows_can_keep_duplicates_for_oversampling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mbr_predictions.jsonl"
            write_jsonl(
                path,
                [
                    {"source": "Buenos dias.", "prediction": "Allin punchaw.", "reference": "A"},
                    {"source": "Buenos dias.", "prediction": "Allin punchaw.", "reference": "B"},
                ],
            )
            args = train_jsonl_sft.parse_args(
                [
                    "--jsonl",
                    str(path),
                    "--output-dir",
                    str(Path(tmpdir) / "out"),
                    "--no-dedupe-rows",
                ]
            )

            rows = train_jsonl_sft.load_jsonl_rows(args)

        self.assertEqual(len(rows), 2)

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

    def test_format_example_can_use_bounded_thinking_prompt(self):
        tokenizer = DummyTokenizer()
        row = {
            "source": "Buenos dias.",
            "target": "Analisis de traduccion: conserva el significado.\nTraduccion final: Allin punchaw.\nAutoevaluacion: no veo errores.\nPuntaje: \\boxed{0.98}",
            "reference": "",
            "source_name": "self",
            "variant": "quy/chanka_self",
            "target_field": "target",
        }

        formatted = train_jsonl_sft.format_example(
            tokenizer,
            row,
            prompt_self_verification_thinking=True,
        )

        self.assertIn("Analisis de traduccion:", formatted["text"])
        self.assertIn("maximo 35 palabras", formatted["text"])
        self.assertIn("Allin punchaw.", formatted["text"])

    def test_format_example_can_use_compact_thinking_prompt(self):
        tokenizer = DummyTokenizer()
        row = {
            "source": "Buenos dias.",
            "target": "Analisis: [SIGNIFICADO] conserva saludo.\nFinal: Allin punchaw.\nPuntaje: \\boxed{0.98}",
            "reference": "",
            "source_name": "self",
            "variant": "quy/chanka_self",
            "target_field": "target",
        }

        formatted = train_jsonl_sft.format_example(
            tokenizer,
            row,
            prompt_self_verification_compact=True,
        )

        self.assertIn("exactamente 3 lineas", formatted["text"])
        self.assertIn("Final:", formatted["text"])
        self.assertIn("No escribas Autoevaluacion", formatted["text"])
        self.assertIn("Allin punchaw.", formatted["text"])

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

    def test_step_schedule_uses_save_only_model_by_default(self):
        args = argparse.Namespace(
            training_mode="lora",
            eval_steps=None,
            save_steps=None,
            save_only_model=None,
            max_steps=64,
            evals_per_epoch=8,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
        )

        train_jsonl_sft.configure_step_schedule(args, train_row_count=512)

        self.assertEqual(args.eval_steps, 8)
        self.assertEqual(args.save_steps, 8)
        self.assertTrue(args.save_only_model)

    def test_step_schedule_saves_only_model_for_full_finetuning(self):
        args = argparse.Namespace(
            training_mode="full",
            eval_steps=8,
            save_steps=8,
            save_only_model=None,
            max_steps=64,
            evals_per_epoch=8,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
        )

        train_jsonl_sft.configure_step_schedule(args, train_row_count=512)

        self.assertTrue(args.save_only_model)

    def test_trainable_parameter_count_counts_only_trainable_weights(self):
        model = DummyModel([DummyParameter(10, True), DummyParameter(20, False), DummyParameter(3, True)])

        self.assertEqual(train_jsonl_sft.trainable_parameter_count(model), 13)


if __name__ == "__main__":
    unittest.main()
