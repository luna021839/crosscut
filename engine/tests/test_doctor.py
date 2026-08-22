from __future__ import annotations

import unittest
from unittest.mock import patch

from crosscut.doctor import ToolStatus, runtime_report
from crosscut.face_identity import SFACE_EMBEDDING_SIZE, SFACE_SHA256


class DoctorTests(unittest.TestCase):
    @patch("crosscut.doctor._command_status")
    @patch("crosscut.doctor.runtime_contract_report")
    def test_report_separates_group_ready_from_member_tracker_contract(
        self,
        contract_report,
        command_status,
    ) -> None:
        contract_report.return_value = {
            "python": {"version_ready": True},
            "numpy": {"version_ready": True},
            "opencv": {"version_ready": True},
            "base_runtime_ready": True,
            "mode_readiness": {"group": True, "member": False},
            "full_ready": False,
            "member_anchor_tracking": {
                "ready": False,
                "backends": [],
                "selection_order": ["CSRT", "KCF", "MIL"],
                "verification": "factory_create_init_update_tiny_frame",
                "scope": "user_anchor_forward_continuous_track_only",
                "required_for": ["member_mode"],
                "optional_for": ["group_mode"],
                "fallback": "group_mode_available_member_render_unavailable",
                "probe_error": None,
            },
        }
        command_status.side_effect = lambda name, version_args=("-version",): ToolStatus(
            name=name,
            available=True,
            path=f"/usr/bin/{name}",
            version="test",
        )

        report = runtime_report()
        capabilities = report["mode_capabilities"]
        self.assertIsInstance(capabilities, dict)
        tracking = capabilities["member_anchor_tracking"]
        self.assertIsInstance(tracking, dict)
        optional = report["optional_features"]
        self.assertIsInstance(optional, dict)
        self.assertNotIn("member_anchor_tracking", optional)

        self.assertFalse(report["ready"])
        self.assertEqual(
            report["mode_readiness"],
            {"group": True, "member": False},
        )
        self.assertFalse(tracking["ready"])
        self.assertEqual(tracking["required_for"], ["member_mode"])
        self.assertEqual(tracking["optional_for"], ["group_mode"])
        self.assertEqual(
            tracking["scope"],
            "user_anchor_forward_continuous_track_only",
        )

    def test_report_verifies_sface_digest_and_face_recognizer_loadability(self) -> None:
        report = runtime_report()
        optional = report["optional_features"]
        self.assertIsInstance(optional, dict)
        appearance = optional["tier_a_face_appearance"]
        self.assertIsInstance(appearance, dict)
        model = appearance["model"]
        self.assertIsInstance(model, dict)

        self.assertTrue(appearance["ready"])
        self.assertEqual(appearance["scope"], "tier_a_dissolve_eligibility_only")
        self.assertEqual(
            appearance["fallback"],
            "tier_b_or_tier_c_analyzed_hard_cut",
        )
        self.assertTrue(model["present"])
        self.assertTrue(model["digest_valid"])
        self.assertTrue(model["loadable"])
        self.assertEqual(model["expected_sha256"], SFACE_SHA256)
        self.assertEqual(model["actual_sha256"], SFACE_SHA256)
        self.assertEqual(model["embedding_size"], SFACE_EMBEDDING_SIZE)


if __name__ == "__main__":
    unittest.main()
