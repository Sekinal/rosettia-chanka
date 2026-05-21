from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_gspo_canaries as summarize


class SummarizeGspoCanariesTests(unittest.TestCase):
    def test_collect_metrics_sorts_by_reward_then_quality_and_copy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for name, reward, chrf, token_f1, copy in [
                ("weak_high_reward", 0.9, 20.0, 10.0, 1.0),
                ("best_external_metrics", 0.2, 35.0, 25.0, 0.5),
                ("middle_external_metrics", 0.4, 30.0, 20.0, 0.1),
            ]:
                metrics_path = root / name / "chanka_gspo" / "final_metrics.json"
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(
                    json.dumps(
                        {
                            "reward_profile": name,
                            "trainer_eval_reward": reward,
                            "chrf++": chrf,
                            "bleu": 5.0,
                            "token_f1": token_f1,
                            "length_ratio_score": 50.0,
                            "source_copy_ratio": copy,
                            "exact_source_copy_rate": 0.0,
                            "spanish_leakage_penalty": 0.0,
                            "ter": 100.0,
                        }
                    )
                )

            records = summarize.collect_metrics(root)

        self.assertEqual(
            [record["reward_profile"] for record in records],
            ["best_external_metrics", "middle_external_metrics", "weak_high_reward"],
        )
        self.assertIn("selection_score", records[0])

    def test_write_markdown_includes_profile_table(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "summary.md"
            summarize.write_markdown(
                [
                    {
                        "reward_profile": "rosettia_guard_v2",
                        "trainer_eval_reward": 0.4,
                        "chrf++": 50.0,
                        "bleu": 10.0,
                        "token_f1": 30.0,
                        "source_copy_ratio": 1.0,
                        "exact_source_copy_rate": 0.0,
                        "spanish_leakage_penalty": 0.0,
                    }
                ],
                path,
            )

            content = path.read_text()

        self.assertIn("rosettia_guard_v2", content)
        self.assertIn("Eval reward", content)
        self.assertIn("Selection", content)


if __name__ == "__main__":
    unittest.main()
