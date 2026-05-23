from __future__ import annotations

import unittest
from unittest import mock

from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_mbr_consensus_selector as consensus


class TrainMbrConsensusSelectorTests(unittest.TestCase):
    def test_jaccard_similarity_handles_overlap(self):
        self.assertGreater(
            consensus.jaccard_similarity("allin punchaw taytay", "allin punchaw mamay"),
            consensus.jaccard_similarity("allin punchaw", "huk iskay"),
        )

    def test_fast_overlap_approximations_reward_similar_text(self):
        self.assertGreater(
            consensus.fast_chrf_approx("allin punchaw taytay", "allin punchaw mamay"),
            consensus.fast_chrf_approx("allin punchaw", "huk iskay"),
        )
        self.assertGreater(
            consensus.fast_bleu_approx("allin punchaw taytay", "allin punchaw mamay"),
            consensus.fast_bleu_approx("allin punchaw", "huk iskay"),
        )

    def test_signal_rows_include_reference_free_consensus_features(self):
        group = [
            oracle_rerank.Candidate("Buenos dias.", "Allin punchaw.", "Allin punchaw.", candidate_index=0),
            oracle_rerank.Candidate("Buenos dias.", "Allin punchaw.", "Buenos dias.", candidate_index=1),
        ]

        rows = consensus.signal_rows_for_group(group)

        self.assertEqual(len(rows), 2)
        self.assertIn("pairwise_chrf", rows[0].signals)
        self.assertIn("pairwise_bleu", rows[0].signals)
        self.assertIn("pairwise_jaccard", rows[0].signals)
        self.assertEqual(rows[1].signals["exact_source_copy"], 1.0)
        self.assertGreaterEqual(rows[0].oracle_score, 0.0)

    def test_peer_subset_keeps_bounded_evenly_spaced_peers(self):
        peers = [
            oracle_rerank.Candidate("src", "ref", f"candidate {index}", candidate_index=index)
            for index in range(6)
        ]

        selected = consensus.peer_subset(peers, max_peers=3)

        self.assertEqual([item.candidate_index for item in selected], [0, 2, 5])

    def test_select_consensus_uses_weighted_signals(self):
        group = [
            consensus.ConsensusRow(
                oracle_rerank.Candidate("src", "ref", "a", candidate_index=0),
                signals={"pairwise_chrf": 0.1},
                oracle_score=0.0,
            ),
            consensus.ConsensusRow(
                oracle_rerank.Candidate("src", "ref", "b", candidate_index=1),
                signals={"pairwise_chrf": 0.9},
                oracle_score=1.0,
            ),
        ]
        model = consensus.ConsensusModel(["pairwise_chrf"], {"pairwise_chrf": 1.0})

        selected = consensus.select_consensus([group], model)

        self.assertEqual(selected[0].candidate.prediction, "b")

    def test_clamp_weights_keeps_penalties_from_becoming_rewards(self):
        weights = consensus.clamp_weights(
            {
                "source_copy_ratio": 1.0,
                "candidate_index_fraction": 0.5,
                "pairwise_chrf": -1.0,
            }
        )

        self.assertEqual(weights["source_copy_ratio"], 0.0)
        self.assertEqual(weights["candidate_index_fraction"], 0.0)
        self.assertEqual(weights["pairwise_chrf"], 0.0)

    def test_train_model_can_improve_toward_oracle(self):
        group = [
            oracle_rerank.Candidate("src", "ref", "bad", candidate_index=0),
            oracle_rerank.Candidate("src", "ref", "good", candidate_index=1),
        ]
        with mock.patch.object(
            consensus.oracle_rerank,
            "candidate_oracle_score",
            side_effect=lambda candidate: 1.0 if candidate.prediction == "good" else 0.0,
        ):
            rows = consensus.signal_rows_for_group(group)
        bad_signals = {}
        good_signals = {}
        for name, weight in consensus.initial_weights().items():
            if weight >= 0.0:
                bad_signals[name] = -1.0
                good_signals[name] = 1.0
            else:
                bad_signals[name] = 1.0
                good_signals[name] = -1.0
        rows = [
            consensus.ConsensusRow(rows[0].candidate, bad_signals, rows[0].oracle_score),
            consensus.ConsensusRow(rows[1].candidate, good_signals, rows[1].oracle_score),
        ]

        model, diagnostics = consensus.train_model(
            [rows],
            seed=1,
            search_iterations=10,
            initial_noise=0.1,
            min_noise=0.01,
        )

        self.assertEqual(diagnostics["training_objective"], "mbr_consensus_hillclimb")
        self.assertEqual(consensus.select_consensus([rows], model)[0].candidate.prediction, "good")


if __name__ == "__main__":
    unittest.main()
