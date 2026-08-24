from __future__ import annotations

import unittest
from unittest.mock import patch

from crosscut.runtime_contract import runtime_contract_report, version_at_least


def _module(version: str, *, ready: bool = True) -> dict[str, object]:
    return {
        "available": True,
        "version": version,
        "minimum_version": "test",
        "version_ready": ready,
        "import_error": None,
    }


class RuntimeContractTests(unittest.TestCase):
    def test_numeric_minimum_versions_are_not_lexicographic(self) -> None:
        self.assertTrue(version_at_least("2.0.0", (2, 0)))
        self.assertTrue(version_at_least("4.10.0-dev", (4, 10)))
        self.assertFalse(version_at_least("1.26.4", (2, 0)))
        self.assertFalse(version_at_least("4.9.9", (4, 10)))
        self.assertFalse(version_at_least("unknown", (4, 10)))

    @patch("crosscut.tracking.available_opencv_trackers", return_value=())
    @patch("crosscut.runtime_contract._module_status")
    def test_group_can_be_ready_while_full_member_contract_is_not(
        self,
        module_status,
        _tracker_probe,
    ) -> None:
        module_status.side_effect = (
            _module("2.2.0"),
            _module("4.10.0"),
        )

        report = runtime_contract_report()

        self.assertTrue(report["base_runtime_ready"])
        self.assertTrue(report["mode_readiness"]["group"])
        self.assertFalse(report["mode_readiness"]["member"])
        self.assertFalse(report["full_ready"])
        tracking = report["member_anchor_tracking"]
        self.assertEqual(tracking["required_for"], ["local_member"])
        self.assertEqual(
            tracking["optional_for"],
            ["local_group", "automatic_group", "automatic_member"],
        )
        self.assertEqual(
            tracking["fallback"],
            "local_group_and_ready_automatic_modes_remain_available",
        )
        self.assertEqual(
            tracking["verification"],
            "factory_create_init_update_tiny_frame",
        )

    @patch("crosscut.runtime_contract._module_status")
    def test_package_version_failure_blocks_both_modes(self, module_status) -> None:
        module_status.side_effect = (
            _module("1.26.4", ready=False),
            _module("4.10.0"),
        )

        report = runtime_contract_report()

        self.assertFalse(report["base_runtime_ready"])
        self.assertEqual(
            report["mode_readiness"],
            {"group": False, "member": False},
        )
        self.assertFalse(report["full_ready"])

    @patch(
        "crosscut.runtime_contract._stage_edit_generation_status",
        return_value={
            "architecture_available": True,
            "default_activation": "disabled_visual_baseline_unchanged",
            "requested_failure_policy": "fail_closed_no_silent_visual_fallback",
        },
    )
    @patch("crosscut.tracking.available_opencv_trackers", return_value=("CSRT",))
    @patch("crosscut.runtime_contract._module_status")
    def test_stage_edit_generation_is_optional_and_does_not_replace_mode_readiness(
        self,
        module_status,
        _tracker_probe,
        _stage_edit_probe,
    ) -> None:
        module_status.side_effect = (
            _module("2.2.0"),
            _module("4.10.0"),
        )

        report = runtime_contract_report()

        self.assertTrue(report["mode_readiness"]["group"])
        learned = report["stage_edit_generation"]
        self.assertTrue(learned["architecture_available"])
        self.assertEqual(
            "disabled_visual_baseline_unchanged",
            learned["default_activation"],
        )
        self.assertEqual(
            "fail_closed_no_silent_visual_fallback",
            learned["requested_failure_policy"],
        )


if __name__ == "__main__":
    unittest.main()
