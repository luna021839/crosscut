from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import crosscut_cli

from crosscut.events import EventReporter
from crosscut.pipeline import _enforce_automatic_stage_edit_post_render_qa
from crosscut.post_render_qa import fingerprint_post_render_artifacts
from engine.tests.test_review_first_runtime import _group_integrity_metrics


def _disposition(status="REVIEW_READY"):
    review = status == "REVIEW_READY"
    return {"contract": "ive_review_first_runtime_v1", "status": status,
            "artifact_role": "review_ready" if review else "rejected",
            "video_output_exposed": review, "delivery_completion_message_allowed": False,
            "technical_issue_codes": [] if review else ["fixture_failure"],
            "editorial_issue_codes": ["source_switch_density_below_hard_minimum"] if review else []}


class ReviewFirstPipelineTests(unittest.TestCase):
    def test_cli_exposes_member_review_ready_without_claiming_delivery(self) -> None:
        artifact = SimpleNamespace(
            output_paths=(Path("/tmp/member-review.mp4"),),
            manifest={
                "request": {
                    "group": "IVE",
                    "edit_mode": "member",
                    "source_mode": "automatic",
                },
                "artifact_role": "review_ready",
                "render": {
                    "video_path": "/tmp/member-review.mp4",
                    "direct_full_frame": True,
                    "artifact_role": "review_ready",
                },
                "app_result_exposure": {
                    "contract": "automatic_ive_member_app_result_exposure_v1",
                    "video_output_exposed": True,
                    "delivery_completion_message_allowed": False,
                },
                "post_render_qa": {
                    "contract": "automatic_ive_member_mandatory_post_render_qa_v1",
                    "passed": False,
                    "content_binding_passed": True,
                    "review_first_disposition": _disposition(),
                },
            },
        )

        self.assertTrue(crosscut_cli._result_contains_exposable_video(artifact))

    def test_cli_exposes_group_review_ready_only_with_bound_group_contract(self) -> None:
        artifact = SimpleNamespace(
            output_paths=(Path("/tmp/group-review.mp4"),),
            manifest={
                "request": {
                    "group": "IVE",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                },
                "artifact_role": "review_ready",
                "render": {"artifact_role": "review_ready", "video_path": "/tmp/group-review.mp4"},
                "app_result_exposure": {
                    "contract": "automatic_ive_group_app_result_exposure_v1",
                    "video_output_exposed": True,
                    "delivery_completion_message_allowed": False,
                },
                "post_render_qa": {
                    "contract": (
                        "automatic_ive_group_stage_edit_mandatory_post_render_qa_v2"
                    ),
                    "passed": False,
                    "promotion_binding": {"passed": True},
                    "review_first_disposition": _disposition(),
                },
            },
        )

        self.assertTrue(crosscut_cli._result_contains_exposable_video(artifact))

        artifact.output_paths = (Path("/tmp/different-unreviewed.mp4"),)
        with self.assertRaisesRegex(RuntimeError, "unique QA-bound render path"):
            crosscut_cli._result_contains_exposable_video(artifact)
        artifact.output_paths = (Path("/tmp/group-review.mp4"), Path("/tmp/unreviewed-extra.mp4"))
        with self.assertRaisesRegex(RuntimeError, "unique QA-bound render path"):
            crosscut_cli._result_contains_exposable_video(artifact)

    def test_cli_blocks_rejected_group_mp4(self) -> None:
        artifact = SimpleNamespace(
            output_paths=(Path("/tmp/group-rejected.mp4"),),
            manifest={
                "request": {
                    "group": "IVE",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                },
                "artifact_role": "rejected",
                "render": {"artifact_role": "rejected"},
                "app_result_exposure": {
                    "contract": "automatic_ive_group_app_result_exposure_v1",
                    "video_output_exposed": False,
                    "delivery_completion_message_allowed": False,
                },
                "post_render_qa": {
                    "contract": (
                        "automatic_ive_group_stage_edit_mandatory_post_render_qa_v2"
                    ),
                    "passed": False,
                    "promotion_binding": {"passed": True},
                    "review_first_disposition": {"status": "INPUT_UNUSABLE"},
                },
            },
        )

        with self.assertRaises(RuntimeError):
            crosscut_cli._result_contains_exposable_video(artifact)

    def test_cli_blocks_mp4_without_manifest(self) -> None:
        artifact = SimpleNamespace(
            output_paths=(Path("/tmp/unbound.mp4"),),
            manifest=None,
        )

        with self.assertRaises(RuntimeError):
            crosscut_cli._result_contains_exposable_video(artifact)

    def test_cli_blocks_contradictory_group_review_status(self) -> None:
        artifact = SimpleNamespace(
            output_paths=(Path("/tmp/group-contradiction.mp4"),),
            manifest={
                "request": {
                    "group": "IVE",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                },
                "artifact_role": "review_ready",
                "render": {"artifact_role": "review_ready"},
                "app_result_exposure": {
                    "contract": "automatic_ive_group_app_result_exposure_v1",
                    "video_output_exposed": True,
                    "delivery_completion_message_allowed": False,
                },
                "post_render_qa": {
                    "contract": (
                        "automatic_ive_group_stage_edit_mandatory_post_render_qa_v2"
                    ),
                    "passed": True,
                    "promotion_binding": {"passed": True},
                    "review_first_disposition": {"status": "REVIEW_READY"},
                },
            },
        )

        with self.assertRaises(RuntimeError):
            crosscut_cli._result_contains_exposable_video(artifact)

    def test_forged_group_pass_does_not_leave_passed_delivery_provenance(self) -> None:
        self.test_content_bound_quality_failure_returns_review_ready_video(_forged_pass=True)

    def test_content_bound_quality_failure_returns_review_ready_video(self, _forged_pass=False) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = "IVE-fixture-20260920-120000"
            manifest_path = root / f"{base}-analysis.json"
            manifest: dict[str, object] = {
                "request": {
                    "group": "IVE",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                    "review_first_enabled": True,
                },
                "artifact_role": "pending_qa",
                "app_result_exposure": {
                    "contract": "automatic_ive_group_app_result_exposure_v1",
                    "video_output_exposed": False,
                    "delivery_completion_message_allowed": False,
                },
                "render": {"artifact_role": "pending_qa"},
                "review_first_runtime": {
                    "contract": "ive_review_first_runtime_v1",
                    "status": "RUNNING",
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / f"{base}-render-plan.json").write_text(
                "{}", encoding="utf-8"
            )
            (root / f"{base}-rendered-candidate-selection.json").write_text(
                "{}", encoding="utf-8"
            )
            (root / f"{base}-group-crosscut.mp4").write_bytes(b"video")

            def evaluate(artifact_set, **_kwargs):  # type: ignore[no-untyped-def]
                payload = {
                    "passed": _forged_pass,
                    "issue_count": 0 if _forged_pass else 1,
                    "metrics": {} if _forged_pass else _group_integrity_metrics(),
                    "fail_issue_count": 0 if _forged_pass else 1,
                    "conditional_issue_count": 0,
                    "issues": [] if _forged_pass else [
                        {"code": "source_switch_density_below_hard_minimum"}
                    ],
                    "input_artifact_fingerprints": (
                        fingerprint_post_render_artifacts(artifact_set)
                    ),
                }
                report = SimpleNamespace(
                    verdict=SimpleNamespace(value="PASS" if _forged_pass else "FAIL"),
                    passed=_forged_pass,
                    analysis_basis="fixture_quality_failure",
                    to_dict=lambda: dict(payload),
                )
                artifact_set.default_report_path.write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                return report, artifact_set.default_report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                if _forged_pass:
                    with self.assertRaisesRegex(RuntimeError, "integrity contract failure"):
                        _enforce_automatic_stage_edit_post_render_qa(
                            registered_stage_edit_bundle=None,
                            manifest=manifest,
                            manifest_path=manifest_path,
                            reporter=EventReporter(),
                            automatic_ive_group_delivery=True,
                        )
                    self.assertEqual(manifest["artifact_role"], "rejected")
                    self.assertFalse(manifest["post_render_qa"]["passed"])
                    self.assertEqual(manifest["post_render_qa"]["delivery_gate_verdict"], "INPUT_UNUSABLE")
                    self.assertFalse(manifest["app_result_exposure"]["video_output_exposed"])
                    return
                report_path = _enforce_automatic_stage_edit_post_render_qa(
                    registered_stage_edit_bundle=None,
                    manifest=manifest,  # type: ignore[arg-type]
                    manifest_path=manifest_path,
                    reporter=EventReporter(),
                    automatic_ive_group_delivery=True,
                )

            self.assertTrue(report_path.is_file())
            self.assertEqual(manifest["artifact_role"], "review_ready")
            exposure = manifest["app_result_exposure"]
            self.assertTrue(exposure["video_output_exposed"])
            self.assertFalse(exposure["delivery_completion_message_allowed"])
            self.assertFalse(manifest["post_render_qa"]["passed"])
            self.assertEqual(
                manifest["review_first_runtime"]["status"], "REVIEW_READY"
            )


if __name__ == "__main__":
    unittest.main()
