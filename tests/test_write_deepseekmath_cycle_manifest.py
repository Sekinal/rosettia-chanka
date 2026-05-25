from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import write_deepseekmath_cycle_manifest as manifest


def hardcase(source: str) -> dict[str, str]:
    return {
        "source": source,
        "reference": "Allin punchaw.",
        "candidate": "Buenos dias.",
        "analysis": "Puntaje: \\boxed{0.95}",
        "label": json.dumps({"score": 0.1, "severity": "critical", "rationale": "false_confidence"}),
    }


class WriteDeepSeekMathCycleManifestTests(unittest.TestCase):
    def test_manifest_collects_artifacts_metrics_promotion_and_hardcases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics = root / "metrics.json"
            promotion = root / "promotion.json"
            input_hardcases = root / "input.jsonl"
            output_hardcases = root / "output.jsonl"
            predictions = root / "predictions.jsonl"
            frontier = root / "frontier.jsonl"
            frontier_report = root / "frontier_report.json"
            frontier_summary = root / "frontier_summary.json"
            frontier_gate = root / "frontier_gate.json"
            meta_adapter = root / "meta" / "final"
            policy_adapter = root / "policy" / "final_lora"
            output = root / "manifest.json"
            metrics.write_text(json.dumps({"chrf++": 42.0, "bleu": 12.0}))
            promotion.write_text(json.dumps({"promoted": True, "reasons": []}))
            input_hardcases.write_text(json.dumps(hardcase("a")) + "\n" + json.dumps(hardcase("b")) + "\n")
            output_hardcases.write_text(json.dumps(hardcase("c")) + "\n")
            predictions.write_text('{"source":"a"}\n')
            frontier.write_text('{"source":"a"}\n{"source":"b"}\n')
            frontier_report.write_text(json.dumps({"gate_metrics": {"written_rows": 2}}))
            frontier_summary.write_text(json.dumps({"total_written_rows": 2}))
            frontier_gate.write_text(json.dumps({"passed": True}))
            meta_adapter.mkdir(parents=True)
            policy_adapter.mkdir(parents=True)

            args = manifest.parse_args(
                [
                    "--output-json",
                    str(output),
                    "--stamp",
                    "cycle",
                    "--stage",
                    "sft_seed",
                    "--base-model",
                    "base",
                    "--policy-adapter",
                    str(policy_adapter),
                    "--meta-verifier-adapter",
                    str(meta_adapter),
                    "--meta-output-dir",
                    str(root / "meta"),
                    "--followup-output-dir",
                    str(root / "followup"),
                    "--metrics-json",
                    str(metrics),
                    "--promotion-json",
                    str(promotion),
                    "--predictions-jsonl",
                    str(predictions),
                    "--input-hardcase-jsonl",
                    str(input_hardcases),
                    "--output-hardcase-jsonl",
                    str(output_hardcases),
                    "--frontier-jsonl",
                    str(frontier),
                    "--frontier-report-json",
                    str(frontier_report),
                    "--frontier-summary-json",
                    str(frontier_summary),
                    "--frontier-paid-gate-json",
                    str(frontier_gate),
                ]
            )

            payload = manifest.manifest_for(args)

        self.assertEqual(payload["stamp"], "cycle")
        self.assertEqual(payload["stage"], "sft_seed")
        self.assertEqual(payload["policy_adapter"], str(policy_adapter))
        self.assertTrue(payload["promoted"])
        self.assertEqual(payload["metrics"]["chrf++"], 42.0)
        self.assertEqual(payload["input_hardcases"]["valid_records"], 2)
        self.assertEqual(payload["output_hardcases"]["valid_records"], 1)
        self.assertTrue(payload["artifacts"]["predictions"]["exists"])
        self.assertTrue(payload["artifacts"]["policy_adapter"]["exists"])
        self.assertTrue(payload["artifacts"]["policy_adapter"]["is_dir"])
        self.assertTrue(payload["artifacts"]["meta_verifier_adapter"]["exists"])
        self.assertTrue(payload["artifacts"]["meta_verifier_adapter"]["is_dir"])
        self.assertEqual(payload["frontier_data"]["accepted_records"], 2)
        self.assertTrue(payload["frontier_data"]["accepted_jsonl"]["exists"])
        self.assertTrue(payload["frontier_data"]["paid_gate_json"]["exists"])


if __name__ == "__main__":
    unittest.main()
