from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import rerank_candidate_predictions as rerank


class RerankCandidatePredictionsTests(unittest.TestCase):
    def test_load_candidates_assigns_per_group_candidate_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"source":"s1","reference":"r1","prediction":"p1","source_name":"manual","variant":"chanka","pool_path":"outputs/current/preds.jsonl"}',
                        '{"source":"s1","reference":"r1","prediction":"p2","source_name":"manual","variant":"chanka"}',
                        '{"source":"s2","reference":"r2","prediction":"p3","source_name":"manual","variant":"chanka"}',
                    ]
                )
                + "\n"
            )

            candidates = rerank.load_candidates(path)
            groups = rerank.group_candidates(candidates)

        self.assertEqual([candidate.candidate_index for candidate in candidates], [0, 1, 0])
        self.assertEqual(candidates[0].pool_path, "outputs/current/preds.jsonl")
        self.assertEqual(len(groups), 2)
        self.assertEqual([candidate.prediction for candidate in groups[0]], ["p1", "p2"])

    def test_oracle_selects_best_candidate_by_reference_aware_score(self):
        candidates = [
            rerank.Candidate("Buenos dias.", "Allin punchaw.", "Buenos dias.", candidate_index=0),
            rerank.Candidate("Buenos dias.", "Allin punchaw.", "Allin punchaw.", candidate_index=1),
        ]
        with mock.patch.object(rerank.gspo, "sentence_chrfpp", side_effect=[0.2, 0.9]), mock.patch.object(
            rerank.gspo, "sentence_bleu", side_effect=[0.1, 0.8]
        ):
            selected = rerank.select_oracle([candidates])

        self.assertEqual(selected[0].prediction, "Allin punchaw.")
        self.assertEqual(selected[0].candidate_index, 1)

    def test_metrics_records_oracle_non_first_rate(self):
        selected = [
            rerank.Candidate("s1", "r1", "p1", candidate_index=0),
            rerank.Candidate("s2", "r2", "p2", candidate_index=3),
        ]
        with mock.patch.object(
            rerank.gspo,
            "corpus_metrics",
            return_value={
                "chrf++": 10.0,
                "bleu": 1.0,
                "token_f1": 5.0,
                "length_ratio_score": 90.0,
                "source_copy_ratio": 0.0,
                "exact_source_copy_rate": 0.0,
                "spanish_leakage_penalty": 0.0,
                "chat_artifact_penalty": 0.0,
                "ter": 80.0,
            },
        ):
            metrics = rerank.metrics_for_selection(selected, "oracle", Path("predictions.jsonl"), 8)

        self.assertEqual(metrics["prediction_groups"], 2)
        self.assertEqual(metrics["total_candidates"], 8)
        self.assertEqual(metrics["oracle_non_first_rate"], 50.0)
        self.assertEqual(metrics["oracle_mean_selected_index"], 1.5)
        self.assertIn("selection_score", metrics)


if __name__ == "__main__":
    unittest.main()
