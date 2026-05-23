import json
import tempfile
import unittest
from pathlib import Path

from scripts import write_nested_metrics_summary as writer
from scripts.summarize_gspo_canaries import selection_score


class WriteNestedMetricsSummaryTests(unittest.TestCase):
    def test_record_computes_selection_score_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "checkpoint-8" / "metrics.json"
            metrics_path.parent.mkdir()
            metrics = {
                "chrf++": 40.0,
                "bleu": 10.0,
                "token_f1": 25.0,
                "length_ratio_score": 90.0,
                "source_copy_ratio": 2.0,
                "exact_source_copy_rate": 0.0,
                "spanish_leakage_penalty": 0.0,
                "chat_artifact_penalty": 0.0,
                "ter": 85.0,
            }
            metrics_path.write_text(json.dumps(metrics))

            record = writer.record_from_metrics(metrics_path, [])

        self.assertEqual(record["checkpoint"], "checkpoint-8")
        self.assertAlmostEqual(record["selection_score"], selection_score(metrics))
        self.assertEqual(record["bleu"], 10.0)

    def test_write_summary_sorts_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = Path(tmpdir)
            for name, chrf in [("checkpoint-8", 38.0), ("checkpoint-16", 42.0)]:
                metrics_path = eval_dir / name / "metrics.json"
                metrics_path.parent.mkdir()
                metrics_path.write_text(
                    json.dumps(
                        {
                            "chrf++": chrf,
                            "bleu": 10.0,
                            "token_f1": 25.0,
                            "length_ratio_score": 90.0,
                            "ter": 80.0,
                        }
                    )
                )

            result = writer.write_summary(eval_dir, None, [])
            summary = json.loads((eval_dir / "summary.json").read_text())

        self.assertEqual(result["records"], 2)
        self.assertEqual(summary["records"][0]["checkpoint"], "checkpoint-16")


if __name__ == "__main__":
    unittest.main()
