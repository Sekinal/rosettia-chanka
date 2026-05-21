from __future__ import annotations

import json
import random
import unittest

from scripts import train_verifier_chanka_unsloth as verifier


class TrainVerifierChankaUnslothTests(unittest.TestCase):
    def test_verifier_target_is_compact_json_score(self):
        payload = json.loads(verifier.verifier_target(1.5, "none", "ok"))

        self.assertEqual(payload["score"], 1.0)
        self.assertEqual(payload["severity"], "none")
        self.assertEqual(payload["rationale"], "ok")

    def test_verifier_prompt_includes_source_reference_and_candidate(self):
        messages = verifier.verifier_prompt_messages("Hola", "Rimaykullayki", "Hola")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("verificador", messages[0]["content"])
        self.assertIn("Español: Hola", messages[1]["content"])
        self.assertIn("Referencia chanka: Rimaykullayki", messages[1]["content"])
        self.assertIn("Candidata: Hola", messages[1]["content"])

    def test_verifier_examples_include_positive_and_source_copy_negative(self):
        examples = verifier.verifier_examples_for_row(
            {"source": "No es un buen esposo", "target": "Mana allin qusa", "source_name": "manual", "variant": "quy"},
            random.Random(7),
        )
        labels = [json.loads(example["label"]) for example in examples]

        self.assertGreaterEqual(max(label["score"] for label in labels), 0.95)
        self.assertLessEqual(min(label["score"] for label in labels), 0.05)
        self.assertTrue(any(example["candidate"] == "No es un buen esposo" for example in examples))

    def test_verifier_examples_include_hard_chanka_negatives(self):
        examples = verifier.verifier_examples_for_row(
            {"source": "No es un buen esposo", "target": "Mana allin qusa", "source_name": "manual", "variant": "quy"},
            random.Random(7),
            distractors=["Allin punchaw taytay", "Huk punchawmi hamusaq"],
        )
        labels = [json.loads(example["label"]) for example in examples]
        rationales = {label["rationale"] for label in labels}

        self.assertIn("fluent_but_semantically_unrelated_chanka", rationales)
        self.assertIn("mixed_translation_from_another_example", rationales)
        self.assertIn("unsupported_extra_chanka_content", rationales)
        self.assertTrue(any(0.1 <= label["score"] <= 0.7 for label in labels))

    def test_build_verifier_rows_expands_parallel_rows(self):
        rows = verifier.build_verifier_rows(
            [
                {"source": "A", "target": "B", "source_name": "manual", "variant": "quy"},
                {"source": "C", "target": "D", "source_name": "manual", "variant": "quy"},
            ],
            seed=1,
        )

        self.assertGreaterEqual(len(rows), 14)


if __name__ == "__main__":
    unittest.main()
