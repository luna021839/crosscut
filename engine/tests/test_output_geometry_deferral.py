from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from crosscut.boundary_subject_alignment import (
    BoundarySubjectAlignmentDecision,
    BoundarySubjectAlignmentPlan,
)
from crosscut.editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from crosscut.ive import AspectRatio, EditMode, SourceMode
from crosscut.pipeline import (
    _decoded_output_geometry_resolution,
    _output_geometry_deferred_boundary_indices,
    _require_reviewed_automatic_group_timeline,
    _validated_output_geometry_deferral,
)
from crosscut.render import EditorialTransformKeyframe, EditorialTransformTrack
from crosscut.rendered_proxy_review import ProxyReviewConfig
from crosscut.request import AnalysisRequest
from crosscut.timeline_review import ReviewIssue, TimelineReview, review_timeline


def _review_with_failure(code: str, boundary_index: int = 5) -> TimelineReview:
    sources = tuple(
        source
        for source, count in enumerate((3, 3, 3, 3, 2, 2, 2, 2, 2, 2))
        for _ in range(count)
    )
    passing = review_timeline(
        {
            "analysis_basis": "output_geometry_deferral_test",
            "choices": [
                {
                    "segment_index": index,
                    "source_index": source,
                    "canonical_master_start": index * 1.4,
                    "canonical_master_end": (index + 1) * 1.4,
                }
                for index, source in enumerate(sources)
            ],
        },
        available_source_count=10,
    )
    if not passing.passed:
        raise AssertionError("synthetic baseline timeline must pass")
    return replace(
        passing,
        hard_failures=(
            ReviewIssue(
                code=code,
                severity="error",
                message="synthetic raw-coordinate failure",
                occurrences=(boundary_index,),
            ),
        ),
    )


def _alignment_and_motion(
    *,
    boundary_index: int = 5,
    attach_track: bool = True,
    validated_output_frame_count: int = 8,
    subject_kind: str = "face",
) -> tuple[BoundarySubjectAlignmentPlan, EditorialMotionPlan]:
    decision = BoundarySubjectAlignmentDecision(
        boundary_index=boundary_index,
        outgoing_segment_index=boundary_index,
        incoming_segment_index=boundary_index + 1,
        outgoing_source_index=0,
        incoming_source_index=1,
        outgoing_source_seconds=10.0,
        incoming_source_seconds=10.0,
        eligible=True,
        rejection_reason=None,
        subject_kind=subject_kind,
        transformed_side="incoming",
        baseline_geometry_mismatch=0.42,
        transformed_geometry_mismatch=0.12,
        baseline_layout_mismatch=0.31,
        transformed_layout_mismatch=0.10,
        validated_output_frame_count=validated_output_frame_count,
    )
    alignment = BoundarySubjectAlignmentPlan(
        decisions=(decision,),
        segments=(),
        source_headroom=(),
        window_frames=8,
    )
    track = (
        EditorialTransformTrack(
            (EditorialTransformKeyframe(0.0, 1.05),),
            reason="synthetic_boundary_subject_alignment",
        )
        if attach_track
        else None
    )
    motion = EditorialMotionPlan(
        segments=(
            EditorialMotionSegment(
                segment_index=boundary_index + 1,
                source_index=1,
                track=track,
                reasons=("synthetic",) if track is not None else (),
            ),
        ),
        boundary_match_count=1 if track is not None else 0,
        zoom_event_count=1 if track is not None else 0,
        pan_event_count=0,
        roll_event_count=0,
        blocked_dissolve_segment_count=0,
        boundary_subject_alignment_plan=alignment,
        accepted_boundary_indices=(boundary_index,) if track is not None else (),
    )
    return alignment, motion


def _decoded_payload(
    *,
    boundary_index: int = 5,
    boundary_passed: bool = True,
) -> dict[str, object]:
    issue_reasons = [] if boundary_passed else ["face_position_discontinuity"]
    return {
        "schema_version": 3,
        "review_policy": ProxyReviewConfig().to_dict(),
        "proxy": {"content_sha256": "1" * 64},
        "segment_interior_geometry": {
            "complete": True,
            "geometry_provider_failure_count": 0,
            "render_timeline_binding_sha256": "2" * 64,
        },
        "geometry_provider_failure_count": 0,
        "boundaries": [
            {
                "boundary_index": boundary_index,
                "review_domain": "introduced_cross_source",
                "passed": boundary_passed,
                "issue_reasons": issue_reasons,
                "technical_integrity_issue_reasons": [],
                "geometry_comparability": {
                    "face": True,
                    "silhouette": True,
                },
                "discontinuities": {
                    "face_scale_discontinuity": 0.10,
                    "face_position_discontinuity": (
                        0.10 if boundary_passed else 0.50
                    ),
                    "silhouette_scale_discontinuity": 0.12,
                },
            }
        ],
        "passed": True,
    }


class OutputGeometryDeferralTests(unittest.TestCase):
    def test_only_raw_geometry_failures_are_deferred(self) -> None:
        for code in ("focus_mismatch_exceeded", "scale_mismatch_exceeded"):
            with self.subTest(code=code):
                self.assertEqual(
                    _output_geometry_deferred_boundary_indices(
                        _review_with_failure(code)
                    ),
                    (5,),
                )
        for code in (
            "formation_mismatch_exceeded",
            "layout_mismatch_exceeded",
            "silhouette_mismatch_exceeded",
            "pose_mismatch_exceeded",
        ):
            with self.subTest(code=code):
                self.assertEqual(
                    _output_geometry_deferred_boundary_indices(
                        _review_with_failure(code)
                    ),
                    (),
                )

    def test_dense_renderer_safe_transform_authorizes_proxy_only(self) -> None:
        review = _review_with_failure("focus_mismatch_exceeded")
        alignment, motion = _alignment_and_motion()

        result = _validated_output_geometry_deferral(
            timeline_review=review,
            boundary_indices=(5,),
            subject_alignment=alignment,
            editorial_motion=motion,
        )

        self.assertTrue(result["pre_proxy_eligible"])
        self.assertFalse(result["raw_timeline_review_passed"])
        self.assertTrue(result["decoded_proxy_review_remains_mandatory"])
        self.assertFalse(result["raw_failure_thresholds_lowered"])
        self.assertEqual(result["unresolved_boundary_indices"], [])

    def test_bilateral_deferral_requires_both_attached_tracks_and_side_counts(self):
        review = _review_with_failure('focus_mismatch_exceeded')
        alignment, motion = _alignment_and_motion(validated_output_frame_count=16)
        decision = replace(alignment.decisions[0], transformed_side='both',
            outgoing_validated_output_frame_count=8,incoming_validated_output_frame_count=8)
        alignment = replace(alignment, decisions=(decision,))
        def eligible(candidate, evidence=alignment):
            return _validated_output_geometry_deferral(timeline_review=review,boundary_indices=(5,),
                subject_alignment=evidence,editorial_motion=candidate)['pre_proxy_eligible']
        self.assertFalse(eligible(motion))
        complete = replace(motion, segments=(replace(motion.segments[0],segment_index=5,source_index=0),*motion.segments))
        self.assertTrue(eligible(complete))
        self.assertFalse(eligible(complete,replace(alignment,decisions=(replace(decision,
            outgoing_validated_output_frame_count=0),))))

    def test_missing_track_or_dense_frames_fails_closed(self) -> None:
        review = _review_with_failure("focus_mismatch_exceeded")
        for attach_track, frame_count in ((False, 8), (True, 7)):
            with self.subTest(
                attach_track=attach_track,
                validated_output_frame_count=frame_count,
            ):
                alignment, motion = _alignment_and_motion(
                    attach_track=attach_track,
                    validated_output_frame_count=frame_count,
                )
                result = _validated_output_geometry_deferral(
                    timeline_review=review,
                    boundary_indices=(5,),
                    subject_alignment=alignment,
                    editorial_motion=motion,
                )
                self.assertFalse(result["pre_proxy_eligible"])
                self.assertEqual(result["unresolved_boundary_indices"], [5])

    def test_target_decoded_boundary_must_independently_pass(self) -> None:
        review = _review_with_failure("focus_mismatch_exceeded")
        alignment, motion = _alignment_and_motion()
        deferral = _validated_output_geometry_deferral(
            timeline_review=review,
            boundary_indices=(5,),
            subject_alignment=alignment,
            editorial_motion=motion,
        )

        passed = _decoded_output_geometry_resolution(
            candidate_index=12,
            output_geometry_deferral=deferral,
            review_payload=_decoded_payload(),
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["unresolved_boundary_indices"], [])

        failed = _decoded_output_geometry_resolution(
            candidate_index=12,
            output_geometry_deferral=deferral,
            review_payload=_decoded_payload(boundary_passed=False),
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["unresolved_boundary_indices"], [5])

    def test_decoded_resolution_fails_closed_on_evidence_breaks(self) -> None:
        review = _review_with_failure("focus_mismatch_exceeded")
        alignment, motion = _alignment_and_motion()
        deferral = _validated_output_geometry_deferral(
            timeline_review=review,
            boundary_indices=(5,),
            subject_alignment=alignment,
            editorial_motion=motion,
        )
        mutations = {
            "wrong_domain": lambda payload: payload["boundaries"][0].update(
                {"review_domain": "native_source_cut"}
            ),
            "not_comparable": lambda payload: payload["boundaries"][0][
                "geometry_comparability"
            ].update({"face": False}),
            "provider_failure": lambda payload: payload.update(
                {"geometry_provider_failure_count": 1}
            ),
            "missing_boundary": lambda payload: payload.update(
                {"boundaries": []}
            ),
            "duplicate_boundary": lambda payload: payload["boundaries"].append(
                deepcopy(payload["boundaries"][0])
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = _decoded_payload()
                mutate(payload)
                result = _decoded_output_geometry_resolution(
                    candidate_index=12,
                    output_geometry_deferral=deferral,
                    review_payload=payload,
                )
                self.assertFalse(result["passed"])

    def test_automatic_full_render_requires_boundary_resolution_and_proxy_pass(
        self,
    ) -> None:
        request = AnalysisRequest(
            song="XOXZ",
            edit_mode=EditMode.GROUP,
            aspect_ratio=AspectRatio.LANDSCAPE,
            sources=(),
            output_directory=Path("."),
            source_mode=SourceMode.AUTOMATIC,
            desired_source_count=8,
        )
        review = _review_with_failure("focus_mismatch_exceeded")
        alignment, motion = _alignment_and_motion()
        deferral = _validated_output_geometry_deferral(
            timeline_review=review,
            boundary_indices=(5,),
            subject_alignment=alignment,
            editorial_motion=motion,
        )

        resolution = _decoded_output_geometry_resolution(
            candidate_index=12,
            output_geometry_deferral=deferral,
            review_payload=_decoded_payload(),
        )
        resolved_deferral = {
            **deferral,
            "decoded_resolution": resolution,
        }

        for candidate_deferral, proxy_passed in (
            (deferral, True),
            (resolved_deferral, False),
        ):
            with self.subTest(proxy_passed=proxy_passed):
                with self.assertRaises(RuntimeError):
                    _require_reviewed_automatic_group_timeline(
                        request,
                        visual_timeline=object(),  # type: ignore[arg-type]
                        timeline_review=review,
                        allow_failed_comparison_render=False,
                        output_geometry_deferral=candidate_deferral,
                        decoded_proxy_passed=proxy_passed,
                    )

        _require_reviewed_automatic_group_timeline(
            request,
            visual_timeline=object(),  # type: ignore[arg-type]
            timeline_review=review,
            allow_failed_comparison_render=False,
            output_geometry_deferral=resolved_deferral,
            decoded_proxy_passed=True,
        )

    def test_automatic_full_render_rejects_failed_boundary_resolution(self) -> None:
        request = AnalysisRequest(
            song="XOXZ",
            edit_mode=EditMode.GROUP,
            aspect_ratio=AspectRatio.LANDSCAPE,
            sources=(),
            output_directory=Path("."),
            source_mode=SourceMode.AUTOMATIC,
            desired_source_count=8,
        )
        review = _review_with_failure("focus_mismatch_exceeded")
        alignment, motion = _alignment_and_motion()
        deferral = _validated_output_geometry_deferral(
            timeline_review=review,
            boundary_indices=(5,),
            subject_alignment=alignment,
            editorial_motion=motion,
        )
        failed_resolution = _decoded_output_geometry_resolution(
            candidate_index=12,
            output_geometry_deferral=deferral,
            review_payload=_decoded_payload(boundary_passed=False),
        )
        with self.assertRaises(RuntimeError):
            _require_reviewed_automatic_group_timeline(
                request,
                visual_timeline=object(),  # type: ignore[arg-type]
                timeline_review=review,
                allow_failed_comparison_render=False,
                output_geometry_deferral={
                    **deferral,
                    "decoded_resolution": failed_resolution,
                },
                decoded_proxy_passed=True,
            )


if __name__ == "__main__":
    unittest.main()
