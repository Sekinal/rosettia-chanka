from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_deepseekmath_staged_run as summarize


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_frontier(root: Path, gate_passed: bool = True) -> None:
    write_json(
        root / "deepseek_v4_pro_thinking_report.json",
        {
            "summary": {"api_requests_used": 16, "max_api_requests": 16},
            "gate_metrics": {
                "written_rows": 8,
                "failed_rows": 0,
                "accept_rate": 1.0,
                "distinct_primitives": 5,
                "expected_primitive_coverage": 1.0,
            },
        },
    )
    write_json(root / "deepseek_v4_pro_paid_smoke_gate.json", {"passed": gate_passed})


def write_preapi(root: Path) -> None:
    write_json(
        root / "deepseek_v4_pro_preapi_readiness.json",
        {
            "ready_for_api": True,
            "prompt_preview_rows": 8,
            "selection_report": {"selected_rows": 8},
        },
    )


def write_cycle(
    root: Path,
    stage: str,
    promoted: bool,
    policy_exists: bool = True,
    hardcases: int = 12,
) -> None:
    adapter = root / "final_lora"
    if policy_exists:
        adapter.mkdir(parents=True)
    hardcase_path = root / "meta_hardcases.jsonl"
    if hardcases:
        hardcase_path.write_text("{}\n" * hardcases)
    write_json(
        root / "cycle_manifest.json",
        {
            "stage": stage,
            "promoted": promoted,
            "policy_adapter": str(adapter),
            "metrics": {
                "chrf++": 40.0,
                "bleu": 10.0,
                "token_f1": 25.0,
                "ter": 90.0,
                "self_verification_required_format_rate": 80.0,
                "self_verification_false_confidence_rate": 20.0,
                "self_verification_missing_score_rate": 5.0,
            },
            "promotion": {"promoted": promoted, "reasons": [] if promoted else ["regressed"]},
            "artifacts": {
                "policy_adapter": {"path": str(adapter), "exists": policy_exists},
                "metrics": {"path": "metrics.json", "exists": True},
                "promotion": {"path": "promotion.json", "exists": True},
                "output_hardcases": {
                    "path": str(hardcase_path),
                    "exists": bool(hardcases),
                    "is_file": bool(hardcases),
                },
            },
            "output_hardcases": {"valid_records": hardcases},
        },
    )


class SummarizeDeepSeekMathStagedRunTests(unittest.TestCase):
    def test_recommends_paid_smoke_when_no_frontier_report_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = summarize.build_report(
                summarize.parse_args(
                    [
                        "--frontier-dir",
                        str(Path(tmpdir) / "frontier"),
                        "--output-json",
                        str(Path(tmpdir) / "status.json"),
                    ]
                )
            )

        self.assertEqual(report["next_action"]["stage"], "frontier_generation")
        self.assertTrue(report["blocked"])

    def test_recommends_sft_after_frontier_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frontier = Path(tmpdir) / "frontier"
            write_frontier(frontier)

            report = summarize.build_report(
                summarize.parse_args(
                    ["--frontier-dir", str(frontier), "--output-json", str(Path(tmpdir) / "status.json")]
                )
            )

        self.assertEqual(report["next_action"]["stage"], "sft_seed")
        self.assertIn("run_deepseek_v4_pro_sft_from_frontier_data.sh", report["next_action"]["command"])
        self.assertFalse(report["blocked"])

    def test_frontier_report_without_paid_gate_does_not_unlock_sft(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frontier = root / "frontier"
            write_json(
                frontier / "deepseek_v4_pro_thinking_report.json",
                {
                    "summary": {"api_requests_used": 16, "max_api_requests": 16},
                    "gate_metrics": {
                        "written_rows": 8,
                        "failed_rows": 0,
                        "accept_rate": 1.0,
                        "distinct_primitives": 5,
                        "expected_primitive_coverage": 1.0,
                    },
                },
            )

            report = summarize.build_report(
                summarize.parse_args(
                    ["--frontier-dir", str(frontier), "--output-json", str(root / "status.json")]
                )
            )

        self.assertFalse(report["frontier"]["ready"])
        self.assertIsNone(report["frontier"]["paid_gate_passed"])
        self.assertEqual(report["next_action"]["stage"], "frontier_generation")
        self.assertIn("DATA_DIR=", report["next_action"]["command"])
        self.assertIn("paid-smoke data gate has not passed", report["next_action"]["reason"])

    def test_frontier_report_with_failed_paid_gate_does_not_unlock_sft(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frontier = root / "frontier"
            write_frontier(frontier, gate_passed=False)

            report = summarize.build_report(
                summarize.parse_args(
                    ["--frontier-dir", str(frontier), "--output-json", str(root / "status.json")]
                )
            )

        self.assertFalse(report["frontier"]["ready"])
        self.assertFalse(report["frontier"]["paid_gate_passed"])
        self.assertEqual(report["next_action"]["stage"], "frontier_generation")
        self.assertIn("run_deepseek_v4_pro_paid_generation_smoke.sh", report["next_action"]["command"])

    def test_discovers_preapi_ready_frontier_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frontier = root / "outputs" / "frontier_thinking_data_preapi"
            write_preapi(frontier)

            report = summarize.build_report(
                summarize.parse_args(["--output-root", str(root / "outputs"), "--output-json", str(root / "status.json")])
            )

        self.assertEqual(report["discovery"]["frontier_dir"], str(frontier))
        self.assertTrue(report["frontier"]["preapi_ready"])
        self.assertTrue(report["frontier"]["preapi_only"])
        self.assertEqual(report["next_action"]["stage"], "frontier_generation")
        self.assertIn("DATA_DIR=", report["next_action"]["command"])
        self.assertIn("Pre-API readiness passed", report["next_action"]["reason"])
        self.assertFalse(report["blocked"])

    def test_discovery_prefers_paid_gate_over_newer_ungated_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "outputs"
            paid_frontier = output_root / "paid_frontier"
            ungated_frontier = output_root / "ungated_frontier"
            write_frontier(paid_frontier)
            write_json(
                ungated_frontier / "deepseek_v4_pro_thinking_report.json",
                {
                    "summary": {"api_requests_used": 16, "max_api_requests": 16},
                    "gate_metrics": {
                        "written_rows": 8,
                        "failed_rows": 0,
                        "accept_rate": 1.0,
                        "distinct_primitives": 5,
                        "expected_primitive_coverage": 1.0,
                    },
                },
            )

            report = summarize.build_report(
                summarize.parse_args(["--output-root", str(output_root), "--output-json", str(root / "status.json")])
            )

        self.assertEqual(report["discovery"]["frontier_dir"], str(paid_frontier))
        self.assertTrue(report["frontier"]["ready"])
        self.assertEqual(report["next_action"]["stage"], "sft_seed")

    def test_discovers_latest_sft_and_gspo_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "outputs"
            frontier = output_root / "frontier"
            sft = output_root / "sft"
            gspo = output_root / "gspo"
            write_frontier(frontier)
            write_cycle(sft, "sft_seed", promoted=False)
            write_cycle(gspo, "initial_gspo", promoted=False)

            report = summarize.build_report(
                summarize.parse_args(["--output-root", str(output_root), "--output-json", str(root / "status.json")])
            )

        self.assertEqual(report["discovery"]["frontier_dir"], str(frontier))
        self.assertEqual(report["discovery"]["sft_dir"], str(sft))
        self.assertEqual(report["discovery"]["gspo_dir"], str(gspo))
        self.assertEqual(report["next_action"]["stage"], "hardcase_iteration")
        self.assertIn("GSPO_META_JSONL=", report["next_action"]["command"])
        self.assertIn(str(gspo / "meta_hardcases.jsonl"), report["next_action"]["command"])

    def test_recommends_gspo_after_sft_seed_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frontier = root / "frontier"
            sft = root / "sft"
            write_frontier(frontier)
            write_cycle(sft, "sft_seed", promoted=False)

            report = summarize.build_report(
                summarize.parse_args(
                    [
                        "--frontier-dir",
                        str(frontier),
                        "--sft-dir",
                        str(sft),
                        "--output-json",
                        str(root / "status.json"),
                    ]
                )
            )

        self.assertEqual(report["next_action"]["stage"], "initial_gspo")
        self.assertIn("run_deepseek_v4_pro_gspo_from_sft_seed.sh", report["next_action"]["command"])

    def test_recommends_hardcase_iteration_after_failed_gspo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frontier = root / "frontier"
            sft = root / "sft"
            gspo = root / "gspo"
            write_frontier(frontier)
            write_cycle(sft, "sft_seed", promoted=False)
            write_cycle(gspo, "initial_gspo", promoted=False)

            report = summarize.build_report(
                summarize.parse_args(
                    [
                        "--frontier-dir",
                        str(frontier),
                        "--sft-dir",
                        str(sft),
                        "--gspo-dir",
                        str(gspo),
                        "--output-json",
                        str(root / "status.json"),
                    ]
                )
            )

        self.assertEqual(report["next_action"]["stage"], "hardcase_iteration")
        self.assertIn("GSPO_META_JSONL=", report["next_action"]["command"])

    def test_failed_gspo_without_hardcases_uses_generic_hardcase_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frontier = root / "frontier"
            sft = root / "sft"
            gspo = root / "gspo"
            write_frontier(frontier)
            write_cycle(sft, "sft_seed", promoted=False)
            write_cycle(gspo, "initial_gspo", promoted=False, hardcases=0)

            report = summarize.build_report(
                summarize.parse_args(
                    [
                        "--frontier-dir",
                        str(frontier),
                        "--sft-dir",
                        str(sft),
                        "--gspo-dir",
                        str(gspo),
                        "--output-json",
                        str(root / "status.json"),
                    ]
                )
            )

        self.assertEqual(report["next_action"]["stage"], "hardcase_iteration")
        self.assertNotIn("GSPO_META_JSONL=", report["next_action"]["command"])

    def test_writes_markdown_with_next_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frontier = root / "frontier"
            out = root / "status.md"
            write_frontier(frontier)
            report = summarize.build_report(
                summarize.parse_args(
                    ["--frontier-dir", str(frontier), "--output-json", str(root / "status.json")]
                )
            )

            summarize.write_markdown(report, out)
            content = out.read_text()

        self.assertIn("DeepSeekMath Staged Run Status", content)
        self.assertIn("next command", content)
        self.assertIn("SFT Seed", content)


if __name__ == "__main__":
    unittest.main()
