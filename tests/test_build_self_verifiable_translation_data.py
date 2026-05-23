from __future__ import annotations

import json
import unittest

from scripts import build_self_verifiable_translation_data as builder


class BuildSelfVerifiableTranslationDataTests(unittest.TestCase):
    def test_build_records_creates_three_deepseekmath_style_datasets(self):
        rows = [
            {
                "source": "No es un buen esposo",
                "target": "Mana allin qusachu",
                "source_name": "manual",
                "variant": "quy/chanka",
            }
        ]

        verifier_records, meta_records, generator_records = builder.build_records(rows, seed=1)

        self.assertGreaterEqual(len(verifier_records), 5)
        self.assertEqual(len(generator_records), 1)
        self.assertIn("Traduccion final:", generator_records[0]["target"])
        self.assertIn("Autoevaluacion:", generator_records[0]["target"])
        self.assertIn("Puntaje:", generator_records[0]["target"])
        thinking_target = builder.thinking_generator_target(rows[0]["target"])
        self.assertIn("Analisis de traduccion:", thinking_target)
        self.assertIn("Traduccion final:", thinking_target)
        self.assertIn("Puntaje:", thinking_target)
        self.assertEqual(len(meta_records), len(verifier_records) * 2)
        labels = [json.loads(record["label"]) for record in meta_records]
        self.assertGreater(max(label["score"] for label in labels), 0.9)
        self.assertLess(min(label["score"] for label in labels), 0.1)

    def test_verifier_analysis_uses_score_box(self):
        label = json.dumps({"score": 0.42, "severity": "major", "rationale": "incomplete"})

        analysis = builder.verifier_analysis_for_label(label)

        self.assertIn("Resumen de problemas", analysis)
        self.assertIn("\\boxed{0.42}", analysis)


if __name__ == "__main__":
    unittest.main()
