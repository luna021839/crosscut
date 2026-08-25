from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.post_render_qa import (
    DYNAMIC_CROP_V3_POST_RENDER_EVIDENCE_REPLAY_CONTRACT,
    DynamicCropV3PostRenderEvidenceReplayError,
    _IssueCollector,
    _dynamic_crop_v3_motion_evidence_metrics,
    replay_dynamic_crop_v3_post_render_evidence,
)
from crosscut.render import (
    AspectRatio,
    BoundaryLookKeyframe,
    BoundaryLookTrack,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderPlan,
    RenderSegment,
    SourceLookAdjustment,
)
from crosscut.stage_edit_dynamic_crop_motion_authority_v3 import (
    DYNAMIC_CROP_MOTION_THRESHOLD_FREEZE_CONTRACT,
    DynamicCropMotionAuthorityV3,
    apply_dynamic_crop_motion_authority_to_render_plan_v3,
    corroborate_dynamic_crop_motion_timeline_v3,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropTrackDecodeV3,
)
from crosscut.stage_edit_production_registry import (
    RegisteredStageEditMotionAuthorityIdentityV3,
)


class PostRenderDynamicCropV3EvidenceReplayTests(unittest.TestCase):
    @staticmethod
    def _serialized(value: object) -> object:
        return json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        )

    @staticmethod
    def _track(*, dynamic: bool) -> EditorialTransformTrack:
        keyframes = (
            (
                EditorialTransformKeyframe(0.0, 1.02),
                EditorialTransformKeyframe(0.8, 1.05),
            )
            if dynamic
            else (EditorialTransformKeyframe(0.0, 1.02),)
        )
        return EditorialTransformTrack(
            keyframes,
            reason="stage_edit_dynamic_crop_head_v3",
        )

    @staticmethod
    def _overlaid(track: EditorialTransformTrack) -> EditorialTransformTrack:
        return EditorialTransformTrack(
            track.keyframes,
            reason=f"{track.reason}_stage_edit_preserved",
        )

    def _fixture(
        self,
        root: Path,
        *,
        dynamic_segment_index: int = 0,
    ) -> dict[str, object]:
        freeze_path = (root / "neutral-release-freeze.json").resolve()
        freeze_path.write_text("{}", encoding="utf-8")
        freeze_file_sha256 = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        freeze_content_sha256 = "d" * 64
        source_commit_oid = "e" * 40
        candidate_id = "neutral-production-candidate"
        authority = DynamicCropMotionAuthorityV3.create(
            release_freeze_content_sha256=freeze_content_sha256,
            release_freeze_file_sha256=freeze_file_sha256,
            source_commit_oid=source_commit_oid,
            candidate_id=candidate_id,
        )
        identity = RegisteredStageEditMotionAuthorityIdentityV3(
            release_freeze_path=freeze_path,
            release_freeze_file_sha256=freeze_file_sha256,
            release_freeze_content_sha256=freeze_content_sha256,
            source_commit_oid=source_commit_oid,
            candidate_id=candidate_id,
            motion_logit_threshold=0.0,
            threshold_freeze_contract=(
                DYNAMIC_CROP_MOTION_THRESHOLD_FREEZE_CONTRACT
            ),
        )
        freeze_payload = {
            "freeze_content_sha256": freeze_content_sha256,
            "source_provenance": {"head_commit_oid": source_commit_oid},
            "candidate_id": candidate_id,
        }
        decodes: list[DynamicCropTrackDecodeV3] = []
        segments: list[RenderSegment] = []
        for index in range(10):
            track = self._track(dynamic=index == dynamic_segment_index)
            decodes.append(
                DynamicCropTrackDecodeV3(
                    source_id="neutral-source",
                    source_index=0,
                    eligible=True,
                    loss_masked=False,
                    track=track,
                    predicted_keyframe_count=len(track.keyframes),
                    applied_keyframe_count=len(track.keyframes),
                    minimum_safe_zoom_projection_count=0,
                    rejection_reasons=(),
                    motion_logit=0.75,
                    shot_duration_seconds=1.0,
                )
            )
            segments.append(
                RenderSegment(
                    source_index=0,
                    source_start=float(index),
                    duration=1.0,
                    master_start=float(index),
                    crop=(
                        NormalizedCrop(0.0, 0.0, 1.0, 1.0)
                        if index == 0
                        else None
                    ),
                    pan_crop=(
                        PanCropTrack(
                            (
                                PanCropKeyframe(
                                    0.0,
                                    NormalizedCrop(0.0, 0.0, 1.0, 1.0),
                                ),
                                PanCropKeyframe(
                                    0.5,
                                    NormalizedCrop(0.0, 0.0, 1.0, 1.0),
                                ),
                            )
                        )
                        if index == 1
                        else None
                    ),
                    editorial_transform=self._overlaid(track),
                    look_adjustment=(
                        SourceLookAdjustment(brightness=0.01)
                        if index == 2
                        else None
                    ),
                    boundary_look_track=(
                        BoundaryLookTrack(
                            (
                                BoundaryLookKeyframe(0.0),
                                BoundaryLookKeyframe(0.5, brightness=0.01),
                            ),
                            reason="neutral_boundary_look",
                        )
                        if index == 3
                        else None
                    ),
                )
            )
        overlaid_plan = RenderPlan(
            sources=(Path("/tmp/neutral-source.mp4"),),
            segments=tuple(segments),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("/tmp/neutral-output.mp4"),
            fps=30.0,
        )
        final_plan = apply_dynamic_crop_motion_authority_to_render_plan_v3(
            render_plan=overlaid_plan,
            decodes=tuple(decodes),
            authority=authority,
        )
        evidence = corroborate_dynamic_crop_motion_timeline_v3(
            render_plan=final_plan,
            decodes=tuple(decodes),
            authority=authority,
        )
        return {
            "freeze_path": freeze_path,
            "freeze_payload": freeze_payload,
            "identity": identity,
            "authority": authority,
            "decodes": tuple(decodes),
            "overlaid_plan": overlaid_plan,
            "final_plan": final_plan,
            "evidence": evidence,
        }

    def _replay(self, fixture: dict[str, object], **overrides):
        identity = fixture["identity"]
        decodes = fixture["decodes"]
        final_plan = fixture["final_plan"]
        evidence = fixture["evidence"]
        assert isinstance(identity, RegisteredStageEditMotionAuthorityIdentityV3)
        assert isinstance(final_plan, RenderPlan)
        return replay_dynamic_crop_v3_post_render_evidence(
            serialized_render_plan=overrides.get(
                "serialized_render_plan", self._serialized(final_plan.to_json())
            ),
            registered_motion_authority_identity=overrides.get(
                "registered_motion_authority_identity", identity.to_dict()
            ),
            serialized_decodes=overrides.get(
                "serialized_decodes", [item.to_dict() for item in decodes]
            ),
            persisted_timeline_evidence=overrides.get(
                "persisted_timeline_evidence", evidence.to_dict()
            ),
        )

    def test_independent_replay_returns_exact_typed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            with patch(
                "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                "load_dynamic_crop_v3_release_freeze",
                return_value=fixture["freeze_payload"],
            ) as strict_loader:
                reconstructed = self._replay(fixture)

        strict_loader.assert_called_once_with(fixture["freeze_path"])
        self.assertEqual(fixture["evidence"], reconstructed)
        self.assertEqual(
            "final_render_plan_registered_release_freeze_decode_and_motion_"
            "evidence_independent_replay_v1",
            DYNAMIC_CROP_V3_POST_RENDER_EVIDENCE_REPLAY_CONTRACT,
        )
        self.assertEqual(1, reconstructed.motion_event_applied_count)
        self.assertTrue(reconstructed.motion_budget_within_range)

    def test_missing_or_malformed_decode_and_registry_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            decodes = fixture["decodes"]
            identity = fixture["identity"]
            assert isinstance(identity, RegisteredStageEditMotionAuthorityIdentityV3)
            malformed = [item.to_dict() for item in decodes]
            malformed[0].pop("motion_logit")
            mismatched_identity = identity.to_dict()
            mismatched_identity["candidate_id"] = "different-candidate"
            cases = (
                (
                    {"registered_motion_authority_identity": None},
                    "authority identity is missing",
                ),
                (
                    {"serialized_decodes": None},
                    "decode list is missing",
                ),
                (
                    {"persisted_timeline_evidence": None},
                    "motion evidence is missing",
                ),
                ({"serialized_decodes": [item.to_dict() for item in decodes[:-1]]}, "one typed"),
                ({"serialized_decodes": malformed}, "decode is invalid"),
                (
                    {
                        "registered_motion_authority_identity": (
                            mismatched_identity
                        )
                    },
                    "differs from its registry identity",
                ),
            )
            for overrides, message in cases:
                with (
                    self.subTest(message=message),
                    patch(
                        "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                        "load_dynamic_crop_v3_release_freeze",
                        return_value=fixture["freeze_payload"],
                    ),
                    self.assertRaisesRegex(
                        DynamicCropV3PostRenderEvidenceReplayError,
                        message,
                    ),
                ):
                    self._replay(fixture, **overrides)

    def test_persisted_tamper_and_valid_but_different_evidence_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root, dynamic_segment_index=0)
            other = self._fixture(root, dynamic_segment_index=1)
            tampered = deepcopy(fixture["evidence"].to_dict())
            tampered["motion_event_applied_count"] = 2
            cases = (
                (tampered, "persisted Dynamic Crop v3 motion evidence is invalid"),
                (
                    other["evidence"].to_dict(),
                    "differs from reconstruction",
                ),
            )
            for persisted, message in cases:
                with (
                    self.subTest(message=message),
                    patch(
                        "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                        "load_dynamic_crop_v3_release_freeze",
                        return_value=fixture["freeze_payload"],
                    ),
                    self.assertRaisesRegex(
                        DynamicCropV3PostRenderEvidenceReplayError,
                        message,
                    ),
                ):
                    self._replay(
                        fixture,
                        persisted_timeline_evidence=persisted,
                    )

    def test_unsanitized_final_plan_and_serialized_plan_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            overlaid = fixture["overlaid_plan"]
            decodes = list(fixture["decodes"])
            assert isinstance(overlaid, RenderPlan)
            decodes[0] = replace(decodes[0], motion_logit=-0.5)
            tampered_plan = deepcopy(
                self._serialized(fixture["final_plan"].to_json())
            )
            tampered_plan["output_frame_count"] += 1
            cases = (
                (
                    {
                        "serialized_render_plan": self._serialized(
                            overlaid.to_json()
                        ),
                        "serialized_decodes": [
                            item.to_dict() for item in decodes
                        ],
                    },
                    "was not motion-authority sanitized",
                ),
                (
                    {"serialized_render_plan": tampered_plan},
                    "output_frame_count disagrees",
                ),
            )
            for overrides, message in cases:
                with (
                    self.subTest(message=message),
                    patch(
                        "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                        "load_dynamic_crop_v3_release_freeze",
                        return_value=fixture["freeze_payload"],
                    ),
                    self.assertRaisesRegex(
                        DynamicCropV3PostRenderEvidenceReplayError,
                        message,
                    ),
                ):
                    self._replay(fixture, **overrides)

    def test_qa_metrics_publish_only_independently_replayed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            identity = fixture["identity"]
            final_plan = fixture["final_plan"]
            evidence = fixture["evidence"]
            assert isinstance(identity, RegisteredStageEditMotionAuthorityIdentityV3)
            assert isinstance(final_plan, RenderPlan)
            analysis = {
                "dynamic_crop_version": 3,
                "dynamic_crop_v3_decodes": [
                    item.to_dict() for item in fixture["decodes"]
                ],
                "dynamic_crop_motion_v3": evidence.to_dict(),
                "stage_edit_generation": {
                    "selection_policy": {"dynamic_crop_version": 3},
                    "activation": {
                        "dynamic_crop_version": 3,
                        "dynamic_crop_v3_motion_authority": {
                            "qualified": True,
                            "identity": identity.to_dict(),
                        }
                    }
                },
            }
            collector = _IssueCollector()
            with patch(
                "crosscut.stage_edit_dynamic_crop_v3_final_release_launcher."
                "load_dynamic_crop_v3_release_freeze",
                return_value=fixture["freeze_payload"],
            ):
                metrics, replayed = _dynamic_crop_v3_motion_evidence_metrics(
                    analysis,
                    self._serialized(final_plan.to_json()),
                    collector,
                )

        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["motion_event_applied_count"], 1)
        self.assertEqual(replayed, evidence)
        self.assertEqual(collector.issues, [])

    def test_qa_metrics_fail_closed_when_registered_identity_is_missing(self) -> None:
        collector = _IssueCollector()
        metrics, replayed = _dynamic_crop_v3_motion_evidence_metrics(
            {
                "dynamic_crop_version": 3,
                "dynamic_crop_v3_decodes": [],
                "dynamic_crop_motion_v3": {},
                "stage_edit_generation": {
                    "selection_policy": {"dynamic_crop_version": 3},
                    "activation": {
                        "dynamic_crop_version": 3,
                        "dynamic_crop_v3_motion_authority": {
                            "qualified": False,
                            "identity": None,
                        }
                    }
                },
            },
            {},
            collector,
        )

        self.assertFalse(metrics["passed"])
        self.assertIsNone(replayed)
        self.assertEqual(
            [item.code for item in collector.issues],
            ["dynamic_crop_v3_post_render_evidence_replay_failed"],
        )

    def test_registered_v3_activation_cannot_be_downgraded_or_removed(self) -> None:
        for final_version in (2, None):
            with self.subTest(final_version=final_version):
                collector = _IssueCollector()
                metrics, replayed = _dynamic_crop_v3_motion_evidence_metrics(
                    {
                        "dynamic_crop_version": final_version,
                        "dynamic_crop_v3_decodes": None,
                        "dynamic_crop_motion_v3": None,
                        "stage_edit_generation": {
                            "selection_policy": {"dynamic_crop_version": 3},
                            "activation": {
                                "dynamic_crop_version": 3,
                                "dynamic_crop_v3_motion_authority": {
                                    "qualified": True,
                                    "identity": {"registered": "identity"},
                                },
                            }
                        },
                    },
                    {},
                    collector,
                )

                self.assertTrue(metrics["required"])
                self.assertFalse(metrics["passed"])
                self.assertIsNone(replayed)
                self.assertEqual(
                    [item.code for item in collector.issues],
                    ["dynamic_crop_v3_activation_version_mismatch"],
                )

    def test_legacy_v2_activation_does_not_require_v3_replay(self) -> None:
        collector = _IssueCollector()
        metrics, replayed = _dynamic_crop_v3_motion_evidence_metrics(
            {
                "dynamic_crop_version": 2,
                "dynamic_crop_v3_decodes": None,
                "dynamic_crop_motion_v3": None,
                "stage_edit_generation": {
                    "selection_policy": {"dynamic_crop_version": 2},
                    "activation": {
                        "dynamic_crop_v3_motion_authority": {
                            "qualified": False,
                            "legacy_without_motion_authority": True,
                            "identity": None,
                        }
                    },
                },
            },
            {},
            collector,
        )

        self.assertFalse(metrics["required"])
        self.assertTrue(metrics["passed"])
        self.assertIsNone(replayed)
        self.assertEqual(collector.issues, [])


if __name__ == "__main__":
    unittest.main()
