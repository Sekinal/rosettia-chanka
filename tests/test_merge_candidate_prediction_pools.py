from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import merge_candidate_prediction_pools as merge


class MergeCandidatePredictionPoolsTests(unittest.TestCase):
    def test_merge_records_reindexes_and_dedupes_by_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.jsonl"
            second = Path(tmp) / "second.jsonl"
            first.write_text(
                "\n".join(
                    [
                        '{"source":"s1","reference":"r1","prediction":"p1","source_name":"manual","variant":"chanka"}',
                        '{"source":"s1","reference":"r1","prediction":"p2","source_name":"manual","variant":"chanka"}',
                        '{"source":"s2","reference":"r2","prediction":"q1","source_name":"manual","variant":"chanka"}',
                    ]
                )
                + "\n"
            )
            second.write_text(
                "\n".join(
                    [
                        '{"source":"s1","reference":"r1","prediction":"p2","source_name":"manual","variant":"chanka"}',
                        '{"source":"s1","reference":"r1","prediction":"p3","source_name":"manual","variant":"chanka"}',
                    ]
                )
                + "\n"
            )

            records = merge.merge_records([first, second])

        s1 = [record for record in records if record["source"] == "s1"]
        self.assertEqual([record["prediction"] for record in s1], ["p1", "p2", "p3"])
        self.assertEqual([record["candidate_index"] for record in s1], [0, 1, 2])
        self.assertEqual(records[-1]["source"], "s2")

    def test_summarize_reports_group_sizes(self):
        records = [
            {"source": "s", "reference": "r", "prediction": "a", "candidate_index": 0},
            {"source": "s", "reference": "r", "prediction": "b", "candidate_index": 1},
        ]

        summary = merge.summarize(records)

        self.assertEqual(summary["groups"], 1)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["max_candidates_per_group"], 2)


if __name__ == "__main__":
    unittest.main()
