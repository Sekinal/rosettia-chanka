from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts import train_dpo_unsloth as train_dpo


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class TrainDpoUnslothTests(unittest.TestCase):
    def test_load_preference_rows_dedupes_and_skips_empty_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pairs.jsonl"
            write_jsonl(
                path,
                [
                    {"source": "Buenos dias.", "chosen": "Allin punchaw.", "rejected": "Mana."},
                    {"source": "Buenos dias.", "chosen": "Allin punchaw.", "rejected": "Mana."},
                    {"source": "Sin salida", "chosen": "", "rejected": "Mana."},
                ],
            )
            args = train_dpo.parse_args(["--jsonl", str(path), "--output-dir", str(Path(tmpdir) / "out")])

            rows = train_dpo.load_preference_rows(args)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "Buenos dias.")
        self.assertEqual(rows[0]["chosen"], "Allin punchaw.")
        self.assertEqual(rows[0]["rejected"], "Mana.")

    def test_format_preference_row_uses_conversational_dpo_schema(self):
        row = {
            "source": "¿Es usted casado?",
            "chosen": "¿Warmiyuqchu kanki?",
            "rejected": "¿Kasaduchu kanki?",
            "source_name": "pairs",
            "variant": "quy",
        }

        formatted = train_dpo.format_preference_row(
            row,
            terminology_entries=[("casado", "warmiyuq")],
            terminology_top_k=1,
        )

        self.assertEqual([message["role"] for message in formatted["prompt"]], ["system", "user"])
        self.assertIn("Glosario sugerido", formatted["prompt"][1]["content"])
        self.assertEqual(formatted["chosen"], [{"role": "assistant", "content": "¿Warmiyuqchu kanki?"}])
        self.assertEqual(formatted["rejected"], [{"role": "assistant", "content": "¿Kasaduchu kanki?"}])

    def test_step_schedule_uses_max_steps_when_present(self):
        args = argparse.Namespace(
            eval_steps=None,
            save_steps=None,
            max_steps=16,
            evals_per_epoch=4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
        )

        train_dpo.configure_step_schedule(args, train_row_count=256)

        self.assertEqual(args.eval_steps, 4)
        self.assertEqual(args.save_steps, 4)


if __name__ == "__main__":
    unittest.main()
