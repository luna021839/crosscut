from __future__ import annotations

import unittest
from unittest.mock import patch

from crosscut.doctor import ToolStatus, runtime_report
from crosscut.face_identity import SFACE_EMBEDDING_SIZE, SFACE_SHA256
from crosscut.pose import (
    HUMAN_SEGMENTER_MODEL,
    PERSON_DETECTOR_MODEL,
    POSE_ESTIMATOR_MODEL,
    ModelAssetStatus,
)
from crosscut.ytdlp_runtime import YTDLPFrontend


def _ready_runtime_contract() -> dict[str, object]:
    tracking = {
        "ready": True,
        "backends": ["CSRT"],
        "selection_order": ["CSRT", "KCF", "MIL"],
        "verification": "factory_create_init_update_tiny_frame",
        "scope": "user_anchor_forward_continuous_track_only",
        "required_for": ["local_member"],
        "optional_for": [
            "local_group",
            "automatic_group",
            "automatic_member",
        ],
        "fallback": "local_group_and_ready_automatic_modes_remain_available",
        "probe_error": None,
    }
    return {
        "python": {"version_ready": True},
        "numpy": {"version_ready": True},
        "opencv": {"version_ready": True},
        "base_runtime_ready": True,
        "mode_readiness": {"group": True, "member": True},
        "full_ready": True,
        "member_anchor_tracking": tracking,
    }


def _tool_status(
    name: str,
    version_args: tuple[str, ...] = ("-version",),
) -> ToolStatus:
    del version_args
    return ToolStatus(
        name=name,
        available=True,
        path=f"/usr/bin/{name}",
        version="test",
        command_succeeded=True,
    )


def _model_status(
    name: str,
    *,
    present: bool = True,
    digest_valid: bool = True,
    loadable: bool | None = True,
) -> ModelAssetStatus:
    expected = f"expected-{name}"
    return ModelAssetStatus(
        name=name,
        path=f"/models/{name}",
        present=present,
        digest_valid=digest_valid,
        loadable=loadable,
        expected_sha256=expected,
        actual_sha256=(
            expected if digest_valid else ("invalid-digest" if present else None)
        ),
    )


def _pose_statuses(
    person_detector: ModelAssetStatus,
) -> tuple[ModelAssetStatus, ...]:
    return (
        person_detector,
        _model_status(POSE_ESTIMATOR_MODEL),
        _model_status(HUMAN_SEGMENTER_MODEL),
    )


def _yunet_status(*, ready: bool = True) -> dict[str, object]:
    return {
        "name": "face_detection_yunet_2023mar.onnx",
        "path": "/models/face_detection_yunet_2023mar.onnx",
        "present": True,
        "digest_valid": ready,
        "loadable": ready,
        "expected_sha256": "expected-yunet",
        "actual_sha256": "expected-yunet" if ready else "invalid-yunet",
    }


class DoctorTests(unittest.TestCase):
    def _report_with_pose_statuses(
        self,
        statuses: tuple[ModelAssetStatus, ...],
        *,
        runtime_contract: dict[str, object] | None = None,
        yunet_status: dict[str, object] | None = None,
    ) -> dict[str, object]:
        frontend = YTDLPFrontend(
            command=("/usr/bin/yt-dlp",),
            version="test",
            kind="path_executable",
        )
        with patch(
            "crosscut.doctor.runtime_contract_report",
            return_value=runtime_contract or _ready_runtime_contract(),
        ), patch(
            "crosscut.doctor._command_status",
            side_effect=_tool_status,
        ), patch(
            "crosscut.doctor.discover_ytdlp_frontend",
            return_value=frontend,
        ), patch(
            "crosscut.pose.model_asset_status",
            return_value=statuses,
        ), patch(
            "crosscut.visual_score.yunet_model_asset_status",
            return_value=yunet_status or _yunet_status(),
        ):
            return runtime_report()

    def _report_with_person_detector(
        self,
        person_detector: ModelAssetStatus,
    ) -> dict[str, object]:
        return self._report_with_pose_statuses(_pose_statuses(person_detector))

    def test_report_separates_group_ready_from_member_tracker_contract(
        self,
    ) -> None:
        runtime_contract = {
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
                "required_for": ["local_member"],
                "optional_for": [
                    "local_group",
                    "automatic_group",
                    "automatic_member",
                ],
                "fallback": "local_group_and_ready_automatic_modes_remain_available",
                "probe_error": None,
            },
        }
        report = self._report_with_pose_statuses(
            _pose_statuses(_model_status(PERSON_DETECTOR_MODEL)),
            runtime_contract=runtime_contract,
        )

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
            {
                "group": True,
                "member": False,
                "local_group": True,
                "local_member": False,
                "automatic_group": True,
                "automatic_member": True,
            },
        )
        self.assertFalse(tracking["ready"])
        self.assertEqual(tracking["required_for"], ["local_member"])
        self.assertEqual(
            tracking["optional_for"],
            ["local_group", "automatic_group", "automatic_member"],
        )
        self.assertEqual(
            tracking["scope"],
            "user_anchor_forward_continuous_track_only",
        )

    def test_automatic_member_requires_verified_person_detector(self) -> None:
        failures = {
            "missing": _model_status(
                PERSON_DETECTOR_MODEL,
                present=False,
                digest_valid=False,
                loadable=None,
            ),
            "digest_invalid": _model_status(
                PERSON_DETECTOR_MODEL,
                digest_valid=False,
                loadable=None,
            ),
            "load_failed": _model_status(
                PERSON_DETECTOR_MODEL,
                digest_valid=True,
                loadable=False,
            ),
        }
        for label, detector in failures.items():
            with self.subTest(label=label):
                report = self._report_with_person_detector(detector)
                readiness = report["mode_readiness"]
                self.assertIsInstance(readiness, dict)
                self.assertTrue(readiness["member"])
                self.assertTrue(readiness["local_member"])
                self.assertTrue(readiness["automatic_group"])
                self.assertFalse(readiness["automatic_member"])
                self.assertFalse(report["ready"])

                capabilities = report["mode_capabilities"]
                self.assertIsInstance(capabilities, dict)
                tracking = capabilities["member_anchor_tracking"]
                self.assertIsInstance(tracking, dict)
                self.assertTrue(tracking["ready"])
                visual_continuity = capabilities[
                    "automatic_member_visual_continuity"
                ]
                self.assertIsInstance(visual_continuity, dict)
                self.assertFalse(visual_continuity["ready"])
                self.assertEqual(
                    visual_continuity["person_detector"]["name"],
                    PERSON_DETECTOR_MODEL,
                )

    def test_automatic_member_requires_complete_visual_model_suite(self) -> None:
        for failed_name in (POSE_ESTIMATOR_MODEL, HUMAN_SEGMENTER_MODEL):
            with self.subTest(failed_name=failed_name):
                statuses = tuple(
                    _model_status(
                        name,
                        digest_valid=name != failed_name,
                        loadable=None if name == failed_name else True,
                    )
                    for name in (
                        PERSON_DETECTOR_MODEL,
                        POSE_ESTIMATOR_MODEL,
                        HUMAN_SEGMENTER_MODEL,
                    )
                )
                report = self._report_with_pose_statuses(statuses)
                readiness = report["mode_readiness"]
                self.assertIsInstance(readiness, dict)
                self.assertTrue(readiness["automatic_group"])
                self.assertFalse(readiness["automatic_member"])
                self.assertFalse(report["ready"])

                capabilities = report["mode_capabilities"]
                self.assertIsInstance(capabilities, dict)
                visual_continuity = capabilities[
                    "automatic_member_visual_continuity"
                ]
                self.assertFalse(visual_continuity["ready"])
                self.assertEqual(len(visual_continuity["models"]), 3)

    def test_automatic_member_requires_verified_yunet_face_evidence(self) -> None:
        report = self._report_with_pose_statuses(
            _pose_statuses(_model_status(PERSON_DETECTOR_MODEL)),
            yunet_status=_yunet_status(ready=False),
        )
        readiness = report["mode_readiness"]
        self.assertIsInstance(readiness, dict)
        self.assertTrue(readiness["automatic_group"])
        self.assertFalse(readiness["automatic_member"])
        self.assertFalse(report["ready"])

        capabilities = report["mode_capabilities"]
        self.assertIsInstance(capabilities, dict)
        visual_continuity = capabilities[
            "automatic_member_visual_continuity"
        ]
        self.assertFalse(visual_continuity["ready"])
        self.assertFalse(visual_continuity["face_detector"]["digest_valid"])

    def test_automatic_member_is_ready_with_verified_person_detector(self) -> None:
        report = self._report_with_person_detector(
            _model_status(PERSON_DETECTOR_MODEL)
        )

        readiness = report["mode_readiness"]
        self.assertIsInstance(readiness, dict)
        self.assertTrue(readiness["member"])
        self.assertTrue(readiness["local_member"])
        self.assertTrue(readiness["automatic_group"])
        self.assertTrue(readiness["automatic_member"])
        self.assertTrue(report["ready"])

        capabilities = report["mode_capabilities"]
        self.assertIsInstance(capabilities, dict)
        self.assertNotIn("automatic_member_anchor_inference", capabilities)
        visual_continuity = capabilities["automatic_member_visual_continuity"]
        self.assertIsInstance(visual_continuity, dict)
        self.assertTrue(visual_continuity["ready"])
        self.assertEqual(
            visual_continuity["person_detector"]["name"],
            PERSON_DETECTOR_MODEL,
        )
        self.assertTrue(visual_continuity["face_detector"]["digest_valid"])
        self.assertTrue(visual_continuity["face_detector"]["loadable"])
        self.assertEqual(
            visual_continuity["required_for"],
            ["automatic_member"],
        )
        self.assertEqual(
            visual_continuity["scope"],
            (
                "official_vertical_fancam_performance_presence_and_"
                "non_identity_full_frame_visual_continuity"
            ),
        )
        self.assertEqual(
            visual_continuity["failure_policy"],
            "fail_closed_before_render",
        )
        self.assertNotIn("fallback", visual_continuity)
        self.assertFalse(
            visual_continuity["raw_biometric_embeddings_persisted"]
        )

    def test_report_verifies_sface_digest_and_face_recognizer_loadability(self) -> None:
        report = runtime_report()
        capabilities = report["mode_capabilities"]
        self.assertIsInstance(capabilities, dict)
        visual_continuity = capabilities["automatic_member_visual_continuity"]
        self.assertIsInstance(visual_continuity, dict)
        detector = visual_continuity["person_detector"]
        self.assertIsInstance(detector, dict)

        self.assertTrue(visual_continuity["ready"])
        self.assertEqual(detector["name"], PERSON_DETECTOR_MODEL)
        self.assertTrue(detector["present"])
        self.assertTrue(detector["digest_valid"])
        self.assertTrue(detector["loadable"])

        optional = report["optional_features"]
        self.assertIsInstance(optional, dict)
        appearance = optional["tier_a_face_appearance"]
        self.assertIsInstance(appearance, dict)
        model = appearance["model"]
        self.assertIsInstance(model, dict)

        self.assertTrue(appearance["ready"])
        self.assertEqual(
            appearance["scope"],
            "closeup_source_switch_compatibility_and_tier_a_dissolve_eligibility",
        )
        self.assertEqual(
            appearance["fallback"],
            "hold_source_for_ambiguous_closeup_or_use_body_safe_hard_cut",
        )
        self.assertTrue(model["present"])
        self.assertTrue(model["digest_valid"])
        self.assertTrue(model["loadable"])
        self.assertEqual(model["expected_sha256"], SFACE_SHA256)
        self.assertEqual(model["actual_sha256"], SFACE_SHA256)
        self.assertEqual(model["embedding_size"], SFACE_EMBEDDING_SIZE)


if __name__ == "__main__":
    unittest.main()
