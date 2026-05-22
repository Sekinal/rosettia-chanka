from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import train_candidate_reranker_chanka_unsloth as train_reranker


class TrainCandidateRerankerTests(unittest.TestCase):
    def test_prompt_does_not_include_reference(self):
        messages = train_reranker.reranker_prompt_messages("Buenos dias.", "Allin punchaw.")
        prompt = "\n".join(message["content"] for message in messages)

        self.assertIn("Buenos dias.", prompt)
        self.assertIn("Allin punchaw.", prompt)
        self.assertNotIn("Referencia", prompt)

    def test_candidate_training_rows_blends_absolute_and_group_relative_scores(self):
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
            with mock.patch.object(train_reranker, "hidden_reference_score", side_effect=[0.2, 0.8]):
                rows = train_reranker.candidate_training_rows([path], max_examples=None, seed=1)

        labels = {row["candidate"]: row["label"] for row in rows}
        self.assertIn('"score":0.13', labels["bad"])
        self.assertIn('"score":0.87', labels["good"])


if __name__ == "__main__":
    unittest.main()
