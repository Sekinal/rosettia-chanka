from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts import train_meta_verifier_chanka_unsloth as meta


class TrainMetaVerifierChankaUnslothTests(unittest.TestCase):
    def test_meta_verifier_prompt_includes_analysis(self):
        messages = meta.meta_verifier_prompt_messages(
            "Hola",
            "Rimaykullayki",
            "Hola",
            "Puntaje: \\boxed{0.1}",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("meta-verificador", messages[0]["content"])
        self.assertIn("Español: Hola", messages[1]["content"])
        self.assertIn("Referencia chanka: Rimaykullayki", messages[1]["content"])
        self.assertIn("Candidata: Hola", messages[1]["content"])
        self.assertIn("Analisis: Puntaje", messages[1]["content"])

    def test_load_meta_rows_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta.jsonl"
            row = {
                "source": "No es un buen esposo",
                "reference": "Mana allin qusachu",
                "candidate": "No es un buen esposo",
                "analysis": "Afirma falsamente que esta bien.",
                "label": json.dumps({"score": 0.05, "severity": "critical", "rationale": "false_confidence"}),
            }
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n")

            rows = meta.load_meta_rows(
                Namespace(
                    meta_jsonl=path,
                    max_rows=None,
                    dataset_repo="unused",
                    dataset_file="unused",
                    seed=1,
                )
            )

        self.assertEqual(rows[0]["source"], row["source"])
        self.assertEqual(rows[0]["analysis"], row["analysis"])

    def test_configure_step_schedule_uses_effective_batch(self):
        args = Namespace(
            eval_steps=None,
            save_steps=None,
            max_steps=-1,
            evals_per_epoch=4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
        )

        meta.configure_step_schedule(args, train_row_count=80)

        self.assertEqual(args.eval_steps, 5)
        self.assertEqual(args.save_steps, 5)


if __name__ == "__main__":
    unittest.main()
