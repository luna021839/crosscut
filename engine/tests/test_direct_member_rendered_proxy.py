from __future__ import annotations

from pathlib import Path
import unittest

from crosscut.direct_member_rendered_continuity import (
    DirectMemberRenderedContinuityConfig,
    DirectMemberRenderedContinuityResult,
)
from crosscut.direct_member_rendered_proxy import (
    DirectMemberRenderedProxyPreflight,
    demote_failed_direct_member_pose_dissolves_from_rendered_proxy,
    promote_direct_member_pose_dissolves_from_rendered_proxy,
    refine_direct_member_camera_plan_from_rendered_proxy,
    render_and_review_direct_member_proxy,
)
from crosscut.ive import AspectRatio
from crosscut.pipeline import _direct_member_proxy_refinement_rank
from crosscut.render import (
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
    NormalizedCrop,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    segment_frame_counts,
)
from crosscut.rendered_proxy_review import (
    RenderedProxyReview,
    derive_proxy_render_plan,
)


def _plan() -> RenderPlan:
    return RenderPlan(
        sources=(Path("one.mp4"), Path("two.mp4")),
        segments=(
            RenderSegment(0, 0.0, 0.4, 0.0),
            RenderSegment(1, 0.4, 0.4, 0.4),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.PORTRAIT,
        output=Path("full.mp4"),
        width=1080,
        height=1920,
        fps=10.0,
        transitions=(RenderTransition(),),
        member_pose_dissolve_requested=True,
        member_pose_dissolve_effective=False,
        member_pose_dissolve_count=0,
    )


def _continuity(
    *,
    passed: bool,
    indices: tuple[int, ...] = (),
) -> DirectMemberRenderedContinuityResult:
    reason = "rendered_boundary_position_discontinuity"
    return DirectMemberRenderedContinuityResult(
        passed=passed,
        failure_reasons=() if passed else (reason,),
        failure_boundary_indices={} if passed else {reason: indices},
        metrics={},
        boundaries=(),
        config=DirectMemberRenderedContinuityConfig(),
    )


def _generic_payload(*, passed: bool, perceptual_targets=()):
    return {
        "passed": passed,
        "review_policy": {
            "minimum_overall_quality_p10": 0.36,
            "maximum_source_quality_median_span": 0.22,
        },
        "overall_quality": {"p10": 0.80},
        "source_quality": {"median_span": 0.05},
        "introduced_cross_source_boundary_issue_indices": [0],
        "perceptual_source_admission": {
            "admission": {
                "passed": not perceptual_targets,
                "repair_target_boundary_indices": list(perceptual_targets),
            }
        },
        "boundary_domains": {
            "tier_d_continuity": {"passed": True},
            "native_source_cut": {"passed": True},
        },
        "boundaries": [],
        "introduced_cross_source_technical_issue_reasons": [],
        "geometry_provider_failure_count": 0,
    }


def _camera_preflight(
    *,
    outgoing_centroid=(0.35, 0.30),
    incoming_centroid=(0.55, 0.30),
    outgoing_size=(0.10, 0.10),
    incoming_size=(0.08, 0.08),
    outgoing_scale_area_proxy=None,
    incoming_scale_area_proxy=None,
    reference_guided=True,
) -> DirectMemberRenderedProxyPreflight:
    position = (
        (
            (outgoing_centroid[0] - incoming_centroid[0]) ** 2
            + (outgoing_centroid[1] - incoming_centroid[1]) ** 2
        )
        ** 0.5
        / 0.42
    )
    import math

    scale = abs(
        math.log(
            (
                outgoing_scale_area_proxy
                if outgoing_scale_area_proxy is not None
                else outgoing_size[0] * outgoing_size[1]
            )
            / (
                incoming_scale_area_proxy
                if incoming_scale_area_proxy is not None
                else incoming_size[0] * incoming_size[1]
            )
        )
    ) / math.log(3.2)
    return DirectMemberRenderedProxyPreflight(
        {
            "passed": False,
            "rendered_continuity": {
                "thresholds": {
                    "maximum_attention_position_mismatch_p95": 0.12,
                    "maximum_attention_scale_mismatch_p95": 0.10,
                },
                "boundaries": [
                    {
                        "boundary_index": 0,
                        "transition_kind": "hard_cut",
                        "outgoing_source_index": 0,
                        "incoming_source_index": 1,
                        "focal_face_reference_guided": reference_guided,
                        "attention_position_mismatch": position,
                        "attention_scale_mismatch": scale,
                        "attention_geometry": {
                            "outgoing": {
                                "bbox_xywh": [
                                    outgoing_centroid[0]
                                    - outgoing_size[0] / 2.0,
                                    outgoing_centroid[1]
                                    - outgoing_size[1] / 2.0,
                                    *outgoing_size,
                                ],
                                "centroid_xy": list(outgoing_centroid),
                                "scale_area_proxy": outgoing_scale_area_proxy,
                            },
                            "incoming": {
                                "bbox_xywh": [
                                    incoming_centroid[0]
                                    - incoming_size[0] / 2.0,
                                    incoming_centroid[1]
                                    - incoming_size[1] / 2.0,
                                    *incoming_size,
                                ],
                                "centroid_xy": list(incoming_centroid),
                                "scale_area_proxy": incoming_scale_area_proxy,
                            },
                        },
                    }
                ],
            },
        }
    )


def _passing_dissolve_preflight(
    *,
    subject_direction_cosine: float = 0.8,
) -> DirectMemberRenderedProxyPreflight:
    return DirectMemberRenderedProxyPreflight(
        {
            "passed": True,
            "member_output_quality_passed": True,
            "direct_member_rendered_continuity_passed": True,
            "rendered_continuity": {
                "passed": True,
                "thresholds": {
                    "maximum_attention_position_mismatch_p95": 0.12,
                    "maximum_attention_scale_mismatch_p95": 0.10,
                    "maximum_pose_mismatch_p95": 0.24,
                },
                "boundaries": [
                    {
                        "boundary_index": 0,
                        "output_seconds": 0.4,
                        "transition_kind": "hard_cut",
                        "descriptors_available": True,
                        "focal_face_reference_guided": True,
                        "attention_position_mismatch": 0.05,
                        "attention_scale_mismatch": 0.03,
                        "pose_mismatch": 0.10,
                        "motion": {
                            "both_moving": True,
                            "endpoint_pose_shape_mismatch": 0.11,
                            "subject_direction_cosine": (
                                subject_direction_cosine
                            ),
                            "motion_speed_ratio_mismatch": 0.10,
                            "pose_phase_continuation_mismatch": 0.08,
                        },
                    }
                ],
            },
        }
    )


class DirectMemberRenderedProxyTests(unittest.TestCase):
    @staticmethod
    def _refinement_result(
        *,
        quality_passed: bool,
        continuity_passed: bool,
        repair_targets: tuple[int, ...],
        position_p95: float,
        weighted_score: float,
        continuity_targets: tuple[int, ...] | None = None,
        generic_targets: tuple[int, ...] | None = None,
    ) -> DirectMemberRenderedProxyPreflight:
        payload = {
                "passed": quality_passed and continuity_passed,
                "member_output_quality_passed": quality_passed,
                "member_output_quality_failure_reasons": (
                    [] if quality_passed else ["rapid_quality_alternation"]
                ),
                "direct_member_rendered_continuity_passed": (
                    continuity_passed
                ),
                "repair_target_boundary_indices": list(repair_targets),
                "rendered_proxy_review": {
                    "weighted_score": weighted_score,
                },
                "rendered_continuity": {
                    "metrics": {
                        "attention_position_mismatch_p95": position_p95,
                    },
                    "thresholds": {
                        "maximum_attention_position_mismatch_p95": 0.12,
                    },
                },
            }
        if continuity_targets is not None:
            payload["continuity_repair_target_boundary_indices"] = list(
                continuity_targets
            )
        if generic_targets is not None:
            payload["generic_repair_target_boundary_indices"] = list(
                generic_targets
            )
        return DirectMemberRenderedProxyPreflight(payload)

    def test_refinement_rank_rejects_a_later_visual_regression(self) -> None:
        base = self._refinement_result(
            quality_passed=False,
            continuity_passed=False,
            repair_targets=(0,),
            position_p95=0.14,
            weighted_score=62.0,
        )
        regressed = self._refinement_result(
            quality_passed=False,
            continuity_passed=False,
            repair_targets=(0, 1),
            position_p95=0.40,
            weighted_score=64.0,
        )

        self.assertLess(
            _direct_member_proxy_refinement_rank(base),
            _direct_member_proxy_refinement_rank(regressed),
        )

    def test_refinement_rank_prefers_a_completed_member_quality_gate(self) -> None:
        incomplete = self._refinement_result(
            quality_passed=False,
            continuity_passed=False,
            repair_targets=(0, 1),
            position_p95=0.20,
            weighted_score=61.0,
        )
        quality_fixed = self._refinement_result(
            quality_passed=True,
            continuity_passed=False,
            repair_targets=(0, 1),
            position_p95=0.20,
            weighted_score=61.0,
        )

        self.assertLess(
            _direct_member_proxy_refinement_rank(quality_fixed),
            _direct_member_proxy_refinement_rank(incomplete),
        )

    def test_refinement_rank_keeps_large_decoded_continuity_improvement(
        self,
    ) -> None:
        unrefined = self._refinement_result(
            quality_passed=False,
            continuity_passed=False,
            repair_targets=(0, 1, 4, 5, 7, 8),
            continuity_targets=(0, 1, 4, 5, 8),
            generic_targets=(1, 4, 5, 7, 8),
            position_p95=0.4545,
            weighted_score=61.0,
        )
        camera_refined = self._refinement_result(
            quality_passed=False,
            continuity_passed=False,
            repair_targets=(0, 1, 2, 4, 5, 6, 7, 8),
            continuity_targets=(1, 4, 8),
            generic_targets=(0, 1, 2, 4, 5, 6, 7, 8),
            position_p95=0.0939,
            weighted_score=58.0,
        )

        self.assertLess(
            _direct_member_proxy_refinement_rank(camera_refined),
            _direct_member_proxy_refinement_rank(unrefined),
        )

    def test_combines_generic_and_exact_member_gates(self) -> None:
        received_references = None

        def render_review(plan, output, **kwargs):
            self.assertEqual(kwargs["video_codec"], "h264_videotoolbox")
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(_generic_payload(passed=True))

        def continuity_evaluator(*_args, **kwargs):
            nonlocal received_references
            received_references = kwargs[
                "focal_reference_candidates_by_boundary_index"
            ]
            return _continuity(passed=True)

        references = {}
        proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=continuity_evaluator,
            focal_reference_candidates_by_boundary_index=references,
        )

        self.assertEqual((proxy.width, proxy.height), (480, 854))
        self.assertTrue(result.passed)
        self.assertEqual(result.repair_target_boundary_indices, ())
        self.assertIs(received_references, references)

    def test_unions_perceptual_and_continuity_repair_targets(self) -> None:
        def render_review(plan, output, **_kwargs):
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(
                _generic_payload(passed=False, perceptual_targets=(0, 2))
            )

        _proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=lambda *_args, **_kwargs: _continuity(
                passed=False,
                indices=(0, 1),
            ),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.repair_target_boundary_indices, (0, 1, 2))

    def test_generic_group_editorial_issues_do_not_replace_member_gate(self) -> None:
        def render_review(plan, output, **_kwargs):
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(_generic_payload(passed=False))

        _proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=lambda *_args, **_kwargs: _continuity(
                passed=True
            ),
        )

        self.assertTrue(result.passed)
        self.assertFalse(result.payload["generic_rendered_proxy_passed"])
        self.assertTrue(result.payload["member_output_quality_passed"])

    def test_ranks_worst_continuity_boundaries_for_bounded_reselection(self) -> None:
        def render_review(plan, output, **_kwargs):
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(_generic_payload(passed=True))

        continuity = DirectMemberRenderedContinuityResult(
            passed=False,
            failure_reasons=(
                "rendered_boundary_position_discontinuity",
                "rendered_boundary_motion_direction_opposed",
            ),
            failure_boundary_indices={
                "rendered_boundary_position_discontinuity": (2,),
                "rendered_boundary_motion_direction_opposed": (1, 2),
            },
            metrics={},
            boundaries=(
                {
                    "boundary_index": 1,
                    "attention_position_mismatch": 0.13,
                    "attention_scale_mismatch": 0.02,
                },
                {
                    "boundary_index": 2,
                    "attention_position_mismatch": 0.30,
                    "attention_scale_mismatch": 0.02,
                },
            ),
            config=DirectMemberRenderedContinuityConfig(),
        )
        _proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=lambda *_args, **_kwargs: continuity,
        )

        self.assertEqual(
            result.ranked_continuity_reselection_boundary_indices(
                maximum_count=2
            ),
            (2, 1),
        )
        penalties = result.ranked_reselection_boundary_penalties(
            maximum_count=2
        )
        self.assertEqual(tuple(item[0] for item in penalties), (2, 1))
        self.assertGreater(penalties[0][1], penalties[1][1])
        self.assertGreaterEqual(penalties[1][1], 0.75)
        self.assertLessEqual(penalties[0][1], 4.0)

    def test_quality_only_failure_still_produces_ranked_reselection(self) -> None:
        payload = _generic_payload(
            passed=False,
            perceptual_targets=(3, 4),
        )
        payload["perceptual_source_admission"][
            "rapid_quality_alternation"
        ] = {
            "boundaries": [
                {
                    "boundary_index": 3,
                    "normalized_quality_shock": 1.2,
                },
                {
                    "boundary_index": 4,
                    "normalized_quality_shock": 2.4,
                },
            ]
        }

        def render_review(plan, output, **_kwargs):
            proxy = derive_proxy_render_plan(plan, output)
            return proxy, RenderedProxyReview(payload)

        _proxy, result = render_and_review_direct_member_proxy(
            _plan(),
            Path("proxy.mp4"),
            render_review=render_review,
            continuity_evaluator=lambda *_args, **_kwargs: _continuity(
                passed=True
            ),
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            result.ranked_reselection_boundary_indices(maximum_count=2),
            (4, 3),
        )

    def test_output_face_geometry_drives_bounded_camera_inverse(self) -> None:
        original = _plan()
        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                original,
                _camera_preflight(),
                maximum_zoom_by_source_index={0: 1.50, 1: 1.50},
            )
        )

        self.assertIsNot(refined, original)
        self.assertTrue(diagnostics["applied"])
        self.assertEqual(diagnostics["refined_boundary_indices"], [0])
        boundary = diagnostics["boundaries"][0]
        self.assertLessEqual(
            boundary["predicted_attention_position_mismatch"],
            diagnostics["target_attention_position_mismatch"] + 1e-9,
        )
        self.assertLessEqual(
            boundary["predicted_attention_scale_mismatch"],
            diagnostics["target_attention_scale_mismatch"] + 1e-9,
        )
        frame_counts = segment_frame_counts(refined)
        outgoing = refined.segments[0].editorial_transform
        incoming = refined.segments[1].editorial_transform
        self.assertIsNotNone(outgoing)
        self.assertIsNotNone(incoming)
        assert outgoing is not None and incoming is not None
        outgoing.evaluate((frame_counts[0] - 1) / refined.fps)
        incoming.evaluate(0.0)

    def test_refinement_respects_per_source_1080p_zoom_cap(self) -> None:
        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                _plan(),
                _camera_preflight(
                    outgoing_centroid=(0.45, 0.30),
                    incoming_centroid=(0.54, 0.30),
                    outgoing_size=(0.10, 0.10),
                    incoming_size=(0.09, 0.09),
                ),
                maximum_zoom_by_source_index={0: 1.15, 1: 1.15},
            )
        )

        self.assertTrue(diagnostics["applied"])
        for segment in refined.segments:
            self.assertIsNotNone(segment.editorial_transform)
            assert segment.editorial_transform is not None
            self.assertLessEqual(
                max(
                    item.zoom
                    for item in segment.editorial_transform.keyframes
                ),
                1.15 + 1e-9,
            )

    def test_pose_dissolve_refinement_holds_camera_through_overlap(self) -> None:
        base = _plan()
        original = RenderPlan(
            sources=base.sources,
            segments=base.segments,
            reference_source=base.reference_source,
            aspect_ratio=base.aspect_ratio,
            output=base.output,
            width=base.width,
            height=base.height,
            fps=base.fps,
            transitions=(
                RenderTransition(
                    TransitionKind.FACE_DISSOLVE,
                    3,
                    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
                ),
            ),
            member_pose_dissolve_requested=True,
            member_pose_dissolve_effective=True,
            member_pose_dissolve_count=1,
        )
        payload = dict(_camera_preflight().payload)
        continuity = dict(payload["rendered_continuity"])
        boundary = dict(continuity["boundaries"][0])
        boundary["transition_kind"] = "face_dissolve"
        continuity["boundaries"] = [boundary]
        payload["rendered_continuity"] = continuity

        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                original,
                DirectMemberRenderedProxyPreflight(payload),
                maximum_zoom_by_source_index={0: 1.50, 1: 1.50},
            )
        )

        self.assertTrue(diagnostics["applied"], diagnostics)
        self.assertEqual(diagnostics["refined_overlap_boundary_indices"], [0])
        self.assertEqual(
            refined.transitions[0].transform_overlap_authority,
            MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
        )
        incoming = refined.segments[1].editorial_transform
        self.assertIsNotNone(incoming)
        assert incoming is not None
        overlap_end = refined.transitions[0].frame_count / refined.fps
        entry = incoming.evaluate(0.0)
        endpoint = incoming.evaluate(overlap_end)
        self.assertEqual(
            (
                entry.zoom,
                entry.center_x,
                entry.center_y,
                entry.roll_degrees,
            ),
            (
                endpoint.zoom,
                endpoint.center_x,
                endpoint.center_y,
                endpoint.roll_degrees,
            ),
        )

    def test_camera_inverse_uses_stable_landmark_scale_when_available(
        self,
    ) -> None:
        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                _plan(),
                _camera_preflight(
                    outgoing_centroid=(0.45, 0.30),
                    incoming_centroid=(0.45, 0.30),
                    outgoing_size=(0.12, 0.12),
                    incoming_size=(0.10, 0.10),
                    outgoing_scale_area_proxy=0.04**2,
                    incoming_scale_area_proxy=0.03**2,
                ),
                maximum_zoom_by_source_index={0: 1.50, 1: 1.50},
            )
        )

        self.assertTrue(diagnostics["applied"], diagnostics)
        boundary = diagnostics["boundaries"][0]
        self.assertEqual(boundary["stable_landmark_scale_used"], 1.0)
        self.assertLessEqual(
            boundary["predicted_attention_scale_mismatch"],
            diagnostics["target_attention_scale_mismatch"] + 1e-9,
        )
        self.assertTrue(
            any(
                keyframe.zoom > 1.0
                for segment in refined.segments
                for keyframe in segment.editorial_transform.keyframes
            )
        )

    def test_lossless_4k_envelope_repairs_opposite_edge_faces(self) -> None:
        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                _plan(),
                _camera_preflight(
                    outgoing_centroid=(0.18259, 0.15037),
                    incoming_centroid=(0.62276, 0.23501),
                    outgoing_size=(0.14556, 0.09777),
                    incoming_size=(0.09927, 0.07198),
                ),
                maximum_zoom_by_source_index={0: 2.0, 1: 2.0},
            )
        )

        self.assertTrue(diagnostics["applied"], diagnostics)
        boundary = diagnostics["boundaries"][0]
        self.assertLessEqual(
            boundary["predicted_attention_position_mismatch"],
            diagnostics["target_attention_position_mismatch"] + 1e-9,
        )
        self.assertLessEqual(
            boundary["predicted_attention_scale_mismatch"],
            diagnostics["target_attention_scale_mismatch"] + 1e-9,
        )
        endpoint_zooms = []
        for segment in refined.segments:
            self.assertIsNotNone(segment.editorial_transform)
            assert segment.editorial_transform is not None
            endpoint_zooms.extend(
                keyframe.zoom
                for keyframe in segment.editorial_transform.keyframes
            )
        self.assertGreater(max(endpoint_zooms), 1.50)
        self.assertLessEqual(max(endpoint_zooms), 2.0 + 1e-9)

    def test_aggregate_silhouette_never_drives_member_camera(self) -> None:
        original = _plan()
        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                original,
                _camera_preflight(reference_guided=False),
                maximum_zoom_by_source_index={0: 1.50, 1: 1.50},
            )
        )

        self.assertIs(refined, original)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(
            diagnostics["boundaries"][0]["status"],
            "unverified_or_unbound_subject_geometry_reselection_required",
        )

    def test_verified_cross_output_face_can_drive_camera_without_source_reference(
        self,
    ) -> None:
        original = _plan()
        preflight = _camera_preflight(reference_guided=False)
        payload = dict(preflight.payload)
        continuity = dict(payload["rendered_continuity"])
        record = dict(continuity["boundaries"][0])
        record["attention_basis"] = {
            "outgoing": "verified_official_fancam_focal_face_correspondence",
            "incoming": "verified_official_fancam_focal_face_correspondence",
        }
        record["focal_face_correspondence_similarity"] = 0.58
        continuity["boundaries"] = [record]
        payload["rendered_continuity"] = continuity

        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                original,
                DirectMemberRenderedProxyPreflight(payload),
                maximum_zoom_by_source_index={0: 1.50, 1: 1.50},
            )
        )

        self.assertTrue(diagnostics["applied"], diagnostics)
        self.assertEqual(
            diagnostics["boundaries"][0]["geometry_authority"],
            "verified_official_fancam_cross_output_face_correspondence",
        )
        self.assertIsNot(refined, original)

    def test_official_fancam_pose_bound_silhouette_is_explicit_opt_in(
        self,
    ) -> None:
        original = _plan()
        preflight = _camera_preflight(reference_guided=False)
        payload = dict(preflight.payload)
        continuity = dict(payload["rendered_continuity"])
        record = dict(continuity["boundaries"][0])
        record.update(
            {
                "attention_basis": {
                    "outgoing": "aggregate_human_silhouette",
                    "incoming": "aggregate_human_silhouette",
                },
                "descriptors_available": True,
                "pose_mismatch": 0.18,
            }
        )
        continuity["boundaries"] = [record]
        payload["rendered_continuity"] = continuity
        official_preflight = DirectMemberRenderedProxyPreflight(payload)

        unchanged, default_diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                original,
                official_preflight,
                maximum_zoom_by_source_index={0: 1.50, 1: 1.50},
            )
        )
        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                original,
                official_preflight,
                maximum_zoom_by_source_index={0: 1.50, 1: 1.50},
                allow_official_fancam_subject_geometry=True,
            )
        )

        self.assertIs(unchanged, original)
        self.assertFalse(default_diagnostics["applied"])
        self.assertTrue(diagnostics["applied"], diagnostics)
        self.assertEqual(
            diagnostics["boundaries"][0]["geometry_authority"],
            "official_fancam_pose_bound_full_subject_silhouette",
        )
        self.assertIsNot(refined, original)

    def test_refinement_preserves_non_camera_edit_decisions(self) -> None:
        original = _plan()
        original = RenderPlan(
            sources=original.sources,
            segments=(
                RenderSegment(
                    0,
                    0.0,
                    0.4,
                    0.0,
                    crop=NormalizedCrop(0.1, 0.0, 0.8, 1.0),
                ),
                original.segments[1],
            ),
            reference_source=original.reference_source,
            aspect_ratio=original.aspect_ratio,
            output=original.output,
            width=original.width,
            height=original.height,
            fps=original.fps,
            transitions=original.transitions,
            member_pose_dissolve_requested=True,
            member_pose_dissolve_effective=False,
            member_pose_dissolve_count=0,
        )
        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                original,
                _camera_preflight(),
                maximum_zoom_by_source_index={0: 1.50, 1: 1.50},
            )
        )

        self.assertTrue(diagnostics["applied"])
        self.assertEqual(refined.sources, original.sources)
        self.assertEqual(refined.transitions, original.transitions)
        self.assertEqual(refined.master_audio_start, original.master_audio_start)
        for before, after in zip(
            original.segments,
            refined.segments,
            strict=True,
        ):
            self.assertEqual(before.source_index, after.source_index)
            self.assertEqual(before.source_start, after.source_start)
            self.assertEqual(before.duration, after.duration)
            self.assertEqual(before.master_start, after.master_start)
            self.assertEqual(before.crop, after.crop)
            self.assertEqual(before.look_adjustment, after.look_adjustment)

    def test_unrepairable_camera_limits_leave_plan_unchanged(self) -> None:
        original = _plan()
        refined, diagnostics = (
            refine_direct_member_camera_plan_from_rendered_proxy(
                original,
                _camera_preflight(),
                maximum_zoom_by_source_index={0: 1.0, 1: 1.0},
            )
        )

        self.assertIs(refined, original)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(
            diagnostics["boundaries"][0]["status"],
            "bounded_inverse_camera_solution_unavailable",
        )

    def test_failed_effect_endpoint_is_demoted_without_losing_good_dissolve(
        self,
    ) -> None:
        transition = RenderTransition(
            kind=TransitionKind.FACE_DISSOLVE,
            frame_count=3,
            reason=DIRECT_MEMBER_POSE_DISSOLVE_REASON,
            face_mismatch=0.08,
            tier="B",
            future_pose_max_mismatch=0.10,
        )
        original = RenderPlan(
            sources=(Path("one.mp4"), Path("two.mp4"), Path("three.mp4")),
            segments=(
                RenderSegment(0, 0.0, 0.6, 0.0),
                RenderSegment(1, 0.6, 0.6, 0.6),
                RenderSegment(2, 1.2, 0.6, 1.2),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.PORTRAIT,
            output=Path("full.mp4"),
            width=1080,
            height=1920,
            fps=10.0,
            transitions=(transition, transition),
            member_pose_dissolve_requested=True,
            member_pose_dissolve_effective=True,
            member_pose_dissolve_count=2,
        )
        preflight = DirectMemberRenderedProxyPreflight(
            {
                "passed": False,
                "rendered_continuity": {
                    "failure_boundary_indices": {
                        "rendered_effect_endpoint_position_discontinuity": [0],
                        "rendered_boundary_scale_discontinuity": [1],
                    },
                    "thresholds": {
                        "maximum_attention_position_mismatch": 0.20,
                        "maximum_attention_scale_mismatch": 0.18,
                        "maximum_pose_mismatch": 0.30,
                    },
                    "boundaries": [
                        {
                            "boundary_index": 0,
                            "transition_kind": "face_dissolve",
                            "attention_position_mismatch": 0.11,
                            "attention_scale_mismatch": 0.09,
                            "pose_mismatch": 0.13,
                        },
                        {
                            "boundary_index": 1,
                            "transition_kind": "face_dissolve",
                            "attention_position_mismatch": 0.08,
                            "attention_scale_mismatch": 0.07,
                            "pose_mismatch": 0.10,
                        },
                    ],
                },
            }
        )

        repaired, diagnostics = (
            demote_failed_direct_member_pose_dissolves_from_rendered_proxy(
                original,
                preflight,
            )
        )

        self.assertTrue(diagnostics["applied"], diagnostics)
        self.assertEqual(diagnostics["demoted_boundary_indices"], [0])
        self.assertEqual(diagnostics["retained_boundary_indices"], [1])
        self.assertIs(
            repaired.transitions[0].kind,
            TransitionKind.HARD_CUT,
        )
        self.assertIs(
            repaired.transitions[1].kind,
            TransitionKind.FACE_DISSOLVE,
        )
        self.assertTrue(repaired.member_pose_dissolve_effective)
        self.assertEqual(repaired.member_pose_dissolve_count, 1)

    def test_failed_effect_is_not_demoted_to_an_already_bad_hardcut(
        self,
    ) -> None:
        transition = RenderTransition(
            kind=TransitionKind.FACE_DISSOLVE,
            frame_count=3,
            reason=DIRECT_MEMBER_POSE_DISSOLVE_REASON,
            face_mismatch=0.08,
            tier="B",
            future_pose_max_mismatch=0.10,
        )
        original = RenderPlan(
            sources=(Path("one.mp4"), Path("two.mp4"), Path("three.mp4")),
            segments=(
                RenderSegment(0, 0.0, 0.6, 0.0),
                RenderSegment(1, 0.6, 0.6, 0.6),
                RenderSegment(2, 1.2, 0.6, 1.2),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.PORTRAIT,
            output=Path("full.mp4"),
            width=1080,
            height=1920,
            fps=10.0,
            transitions=(transition, transition),
            member_pose_dissolve_requested=True,
            member_pose_dissolve_effective=True,
            member_pose_dissolve_count=2,
        )
        preflight = DirectMemberRenderedProxyPreflight(
            {
                "passed": False,
                "rendered_continuity": {
                    "failure_boundary_indices": {
                        "rendered_effect_endpoint_scale_discontinuity": [0]
                    },
                    "thresholds": {
                        "maximum_attention_position_mismatch": 0.20,
                        "maximum_attention_scale_mismatch": 0.18,
                        "maximum_pose_mismatch": 0.30,
                    },
                    "boundaries": [
                        {
                            "boundary_index": 0,
                            "transition_kind": "face_dissolve",
                            "attention_position_mismatch": 0.13,
                            "attention_scale_mismatch": 0.229,
                            "pose_mismatch": 0.12,
                        },
                        {
                            "boundary_index": 1,
                            "transition_kind": "face_dissolve",
                            "attention_position_mismatch": 0.08,
                            "attention_scale_mismatch": 0.07,
                            "pose_mismatch": 0.10,
                        },
                    ],
                },
            }
        )

        repaired, diagnostics = (
            demote_failed_direct_member_pose_dissolves_from_rendered_proxy(
                original,
                preflight,
            )
        )

        self.assertIs(repaired, original)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(
            diagnostics["status"],
            "demoted_effect_hardcut_continuity_not_viable",
        )
        self.assertEqual(diagnostics["unsafe_hardcut_boundary_indices"], [0])
        self.assertEqual(
            diagnostics["hardcut_viability_by_boundary"]["0"][
                "failed_requirements"
            ],
            ["hardcut_scale_not_viable"],
        )

    def test_all_failed_effects_require_reselection_instead_of_zero_dissolve(
        self,
    ) -> None:
        transition = RenderTransition(
            kind=TransitionKind.FACE_DISSOLVE,
            frame_count=3,
            reason=DIRECT_MEMBER_POSE_DISSOLVE_REASON,
            face_mismatch=0.08,
            tier="B",
            future_pose_max_mismatch=0.10,
        )
        original = RenderPlan(
            sources=(Path("one.mp4"), Path("two.mp4")),
            segments=(
                RenderSegment(0, 0.0, 0.6, 0.0),
                RenderSegment(1, 0.6, 0.6, 0.6),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.PORTRAIT,
            output=Path("full.mp4"),
            width=1080,
            height=1920,
            fps=10.0,
            transitions=(transition,),
            member_pose_dissolve_requested=True,
            member_pose_dissolve_effective=True,
            member_pose_dissolve_count=1,
        )
        preflight = DirectMemberRenderedProxyPreflight(
            {
                "passed": False,
                "rendered_continuity": {
                    "failure_boundary_indices": {
                        "rendered_effect_endpoint_pose_discontinuity": [0]
                    }
                },
            }
        )

        repaired, diagnostics = (
            demote_failed_direct_member_pose_dissolves_from_rendered_proxy(
                original,
                preflight,
            )
        )

        self.assertIs(repaired, original)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(
            diagnostics["status"],
            "all_pose_dissolves_failed_reselection_required",
        )

    def test_passing_output_boundary_is_promoted_to_revalidated_dissolve(
        self,
    ) -> None:
        original = _plan()

        promoted, diagnostics = (
            promote_direct_member_pose_dissolves_from_rendered_proxy(
                original,
                _passing_dissolve_preflight(),
                source_durations_by_index={0: 1.0, 1: 1.0},
                highlight_intervals=((0.35, 0.45),),
            )
        )

        self.assertTrue(diagnostics["applied"], diagnostics)
        self.assertEqual(diagnostics["selected_boundary_indices"], [0])
        self.assertEqual(diagnostics["selected_highlight_count"], 1)
        self.assertTrue(promoted.member_pose_dissolve_effective)
        self.assertEqual(promoted.member_pose_dissolve_count, 1)
        self.assertIs(
            promoted.transitions[0].kind,
            TransitionKind.FACE_DISSOLVE,
        )
        self.assertEqual(
            promoted.transitions[0].transform_overlap_authority,
            MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
        )
        self.assertIs(original.transitions[0].kind, TransitionKind.HARD_CUT)

    def test_opposed_or_failed_effect_boundary_is_never_promoted(self) -> None:
        original = _plan()
        opposed, opposed_diagnostics = (
            promote_direct_member_pose_dissolves_from_rendered_proxy(
                original,
                _passing_dissolve_preflight(
                    subject_direction_cosine=-0.1
                ),
                source_durations_by_index={0: 1.0, 1: 1.0},
            )
        )
        excluded, excluded_diagnostics = (
            promote_direct_member_pose_dissolves_from_rendered_proxy(
                original,
                _passing_dissolve_preflight(),
                source_durations_by_index={0: 1.0, 1: 1.0},
                excluded_boundary_indices=(0,),
            )
        )

        self.assertIs(opposed, original)
        self.assertFalse(opposed_diagnostics["applied"])
        self.assertEqual(
            opposed_diagnostics["rejection_reason_counts"],
            {"decoded_subject_motion_direction_opposed": 1},
        )
        self.assertIs(excluded, original)
        self.assertFalse(excluded_diagnostics["applied"])
        self.assertEqual(
            excluded_diagnostics["rejection_reason_counts"],
            {"excluded_after_failed_effect_proxy": 1},
        )


if __name__ == "__main__":
    unittest.main()
