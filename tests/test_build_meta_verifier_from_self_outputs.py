from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_meta_verifier_from_self_outputs as builder


class BuildMetaVerifierFromSelfOutputsTests(unittest.TestCase):
    def test_build_records_labels_false_confidence(self):
        payload = {
            "source": "Buenos dias autoridad.",
            "reference": "Allin punchaw kamachiq.",
            "prediction": "Buenos dias autoridad.",
            "raw_prediction": (
                "Traduccion final: Buenos dias autoridad. "
                "Autoevaluacion: no veo errores importantes. "
                "Puntaje: \\boxed{0.95}"
            ),
            "self_verification": {
                "translation": "Buenos dias autoridad.",
                "analysis": "no veo errores importantes",
                "self_score": 0.95,
                "has_format": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "self.jsonl"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
            with mock.patch.object(builder.gspo, "sentence_chrfpp", return_value=0.1), mock.patch.object(
                builder.gspo, "sentence_bleu", return_value=0.0
            ):
                records, summary = builder.build_records([path], min_quality_gap=0.2)

        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["label_rationales"]["false_confidence_self_score_too_high"], 1)
        self.assertEqual(summary["label_severities"]["critical"], 1)
        self.assertGreater(summary["avg_self_score_gap"], 0.2)
        label = json.loads(records[0]["label"])
        self.assertLess(label["score"], 0.25)
        self.assertEqual(label["rationale"], "false_confidence_self_score_too_high")

    def test_build_records_accepts_good_calibration(self):
        payload = {
            "source": "Buenos dias autoridad.",
            "reference": "Allin punchaw kamachiq.",
            "prediction": "Allin punchaw kamachiq.",
            "raw_prediction": (
                "Traduccion final: Allin punchaw kamachiq. "
                "Autoevaluacion: no veo errores importantes. "
                "Puntaje: \\boxed{0.80}"
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "self.jsonl"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
            with mock.patch.object(builder.gspo, "sentence_chrfpp", return_value=0.8), mock.patch.object(
                builder.gspo, "sentence_bleu", return_value=0.4
            ):
                records, _summary = builder.build_records([path], min_quality_gap=0.2)

        label = json.loads(records[0]["label"])
        self.assertGreaterEqual(label["score"], 0.8)


if __name__ == "__main__":
    unittest.main()
