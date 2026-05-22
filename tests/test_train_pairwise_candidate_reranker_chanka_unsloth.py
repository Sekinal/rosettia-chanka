from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import train_pairwise_candidate_reranker_chanka_unsloth as train_pairwise


class TrainPairwiseCandidateRerankerTests(unittest.TestCase):
    def test_prompt_does_not_include_reference(self):
        messages = train_pairwise.pairwise_prompt_messages("Hola", "Rimaykullayki", "Allin")
        prompt = "\n".join(message["content"] for message in messages)

        self.assertIn("Hola", prompt)
        self.assertIn("Candidata A: Rimaykullayki", prompt)
        self.assertIn("Candidata B: Allin", prompt)
        self.assertNotIn("Referencia", prompt)

    def test_winner_target_is_compact_json(self):
        self.assertEqual(train_pairwise.winner_target("A"), '{"winner":"A"}')
        self.assertEqual(train_pairwise.winner_target("B"), '{"winner":"B"}')

    def test_pairwise_training_rows_create_hidden_reference_winner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"source":"s","reference":"r","prediction":"bad"}',
                        '{"source":"s","reference":"r","prediction":"good"}',
                    ]
                )
                + "\n"
            )
            with mock.patch.object(train_pairwise, "hidden_oracle_score", side_effect=[0.1, 0.8]):
                rows = train_pairwise.pairwise_training_rows(
                    [path],
                    max_examples=None,
                    min_score_margin=0.03,
                    seed=1,
                )

        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0]["label"], {'{"winner":"A"}', '{"winner":"B"}'})
        if rows[0]["candidate_a"] == "good":
            self.assertEqual(rows[0]["label"], '{"winner":"A"}')
        else:
            self.assertEqual(rows[0]["label"], '{"winner":"B"}')


if __name__ == "__main__":
    unittest.main()
