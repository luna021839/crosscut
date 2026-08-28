from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from crosscut.direct_member_transition import (
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
    DirectMemberPoseDissolveConfig,
    build_direct_member_pose_dissolve_plan,
)
from crosscut.direct_member_visual import _AttentionGeometry, _BoundarySample
from crosscut.ive import AspectRatio
from crosscut.render import (
    FACE_DISSOLVE_ALPHA_CURVE,
    RenderPlan,
    RenderSegment,
    TransitionKind,
    build_filter_graph,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor


_FACE_BASIS = "unambiguous_dominant_face_bbox_geometry_only"


def _descriptor(value: float = 0.1) -> FrameVisualDescriptor:
    box = (0.39 + value * 0.01, 0.18, 0.22, 0.54)
    embedding = (0.10 + value * 0.01, 0.30, 0.60)
    return FrameVisualDescriptor(
        quality_score=0.92,
        composition_score=0.90,
        shot_scale=0.50,
        shot_type=ShotType.MEDIUM,
        face_count=1,
        structure_embedding=embedding,
        primary_face_bbox=box,
        face_detector_backend="yunet",
        pose_embedding=embedding,
        pose_confidence=0.90,
        pose_backend="synthetic_verified_pose",
        detected_person_count=1,
        silhouette_embedding=embedding,
        silhouette_bbox_xywh=box,
        silhouette_centroid_xy=(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0),
        silhouette_area_ratio=box[2] * box[3],
        silhouette_confidence=0.90,
        silhouette_backend="synthetic_verified_silhouette",
    )


def _sample(master: float, value: float = 0.1) -> _BoundarySample:
    descriptor = _descriptor(value)
    assert descriptor.primary_face_bbox is not None
    box = descriptor.primary_face_bbox
    return _BoundarySample(
        master_seconds=master,
        source_local_seconds=master,
        descriptor=descriptor,
        attention=_AttentionGeometry(
            bbox_xywh=box,
            centroid_xy=(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0),
            basis=_FACE_BASIS,
        ),
    )


def _motion(
    *,
    canonical_master: float,
    fps: float,
    both_moving: bool = True,
    endpoint_mismatch: float = 0.08,
):
    step = 1.0 / fps
    return SimpleNamespace(
        timestamp_bound_adjacent=True,
        outgoing_previous_master_seconds=canonical_master - 2.0 * step,
        outgoing_endpoint_master_seconds=canonical_master - step,
        incoming_start_master_seconds=canonical_master,
        incoming_next_master_seconds=canonical_master + step,
        outgoing_previous_source_local_seconds=canonical_master - 2.0 * step,
        outgoing_endpoint_source_local_seconds=canonical_master - step,
        incoming_start_source_local_seconds=canonical_master,
        incoming_next_source_local_seconds=canonical_master + step,
        outgoing_motion_magnitude=0.07 if both_moving else 0.0,
        incoming_motion_magnitude=0.065 if both_moving else 0.0,
        outgoing_motion_component_count=3 if both_moving else 2,
        incoming_motion_component_count=3 if both_moving else 2,
        both_moving=both_moving,
        endpoint_pose_shape_mismatch=endpoint_mismatch,
    )


def _selection(
    source_path: tuple[int, ...],
    *,
    fps: float = 30.0,
    duration: float = 4.0,
    warning_boundaries: frozenset[int] = frozenset(),
    severe_boundaries: frozenset[int] = frozenset(),
    moving_boundaries: frozenset[int] | None = None,
    future_divergent_boundary: int | None = None,
    native_cut_boundary: int | None = None,
):
    moving = (
        frozenset(range(len(source_path) - 1))
        if moving_boundaries is None
        else moving_boundaries
    )
    boundaries = []
    for index, (left_source, right_source) in enumerate(
        zip(source_path, source_path[1:])
    ):
        boundary = (index + 1) * duration
        boundaries.append(
            SimpleNamespace(
                boundary_index=index,
                canonical_master_seconds=boundary,
                outgoing_eligible_source_index=left_source,
                incoming_eligible_source_index=right_source,
                outgoing_sample=_sample(boundary - 1.0 / fps, index * 0.01),
                incoming_sample=_sample(boundary, index * 0.01 + 0.01),
                mismatch=SimpleNamespace(
                    warning_flags=("attention_position",)
                    if index in warning_boundaries
                    else (),
                    severe_flags=("total_mismatch",)
                    if index in severe_boundaries
                    else (),
                    evidence_component_count=5,
                    attention_position_mismatch=0.07,
                    attention_scale_mismatch=0.06,
                    dominant_pose_mismatch=0.08,
                    formation_mismatch=None,
                    silhouette_mismatch=0.08,
                    transform_repair=None,
                    cinematic_bridge=None,
                    motion_connection=_motion(
                        canonical_master=boundary,
                        fps=fps,
                        both_moving=index in moving,
                    ),
                ),
            )
        )

    source_count = max(source_path) + 1
    spans = []
    for segment_index in range(len(source_path)):
        start = segment_index * duration
        sources = []
        for source_index in range(source_count):
            value = segment_index * 0.01
            if (
                future_divergent_boundary is not None
                and segment_index == future_divergent_boundary + 1
                and source_index == source_path[future_divergent_boundary]
            ):
                value = 70.0
            samples = (
                _sample(start + 1.0 / fps, value),
                _sample(start + 0.20, value + 0.01),
            )
            native_cuts = (
                (start + 0.10,)
                if native_cut_boundary is not None
                and segment_index == native_cut_boundary + 1
                and source_index == source_path[native_cut_boundary]
                else ()
            )
            sources.append(
                SimpleNamespace(
                    eligible_source_index=source_index,
                    valid=True,
                    samples=samples,
                    native_scene_cut_master_seconds=native_cuts,
                    native_scene_cut_source_seconds=native_cuts,
                )
            )
        spans.append(SimpleNamespace(sources=tuple(sources)))
    return SimpleNamespace(
        source_indices=source_path,
        usage_policy="exact_balanced_quota_v1",
        boundaries=tuple(boundaries),
        span_validity=tuple(spans),
        config=SimpleNamespace(
            minimum_pose_confidence=0.25,
            minimum_silhouette_confidence=0.30,
        ),
    )


def _cinematic_selection(
    *,
    style: str = "zoom_in",
    cross_attention_basis: bool = False,
):
    selection = _selection((0, 1))
    selection.usage_policy = "heterogeneous_role_aware_cinematic_v1"
    boundary = selection.boundaries[0]
    if cross_attention_basis:
        incoming = boundary.incoming_sample
        descriptor = replace(
            incoming.descriptor,
            face_count=0,
            face_detector_backend="none",
            primary_face_bbox=None,
        )
        assert descriptor.silhouette_bbox_xywh is not None
        box = descriptor.silhouette_bbox_xywh
        boundary.incoming_sample = replace(
            incoming,
            descriptor=descriptor,
            attention=_AttentionGeometry(
                bbox_xywh=box,
                centroid_xy=(
                    box[0] + box[2] / 2.0,
                    box[1] + box[3] / 2.0,
                ),
                basis="aggregate_human_silhouette",
            ),
        )
    severe_flags = ("attention_position", "attention_scale")
    boundary.mismatch.severe_flags = severe_flags
    boundary.mismatch.warning_flags = (
        "cinematic_effect_bridge_required",
    )
    boundary.mismatch.cinematic_bridge = SimpleNamespace(
        style=style,
        authorized_severe_flags=severe_flags,
        pose_mismatch=0.10,
        silhouette_mismatch=0.12,
        base_mismatch=0.20,
    )
    return selection


def _segments(
    source_path: tuple[int, ...],
    *,
    duration: float = 4.0,
) -> tuple[RenderSegment, ...]:
    return tuple(
        RenderSegment(
            source_index=source_index,
            source_start=index * duration,
            duration=duration,
            master_start=index * duration,
        )
        for index, source_index in enumerate(source_path)
    )


def _plan(
    source_path: tuple[int, ...],
    *,
    selection=None,
    durations: tuple[float, ...] | None = None,
    metadata_contract: str | None = OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
    enabled: bool = True,
    highlight_intervals: tuple[tuple[float, float], ...] = (),
    config: DirectMemberPoseDissolveConfig | None = None,
):
    source_count = max(source_path) + 1
    return build_direct_member_pose_dissolve_plan(
        visual_selection=selection or _selection(source_path),
        render_segments=_segments(source_path),
        eligible_request_source_indices=tuple(range(source_count)),
        eligible_source_durations_seconds=durations or (120.0,) * source_count,
        common_master_start=0.0,
        source_metadata_contract=metadata_contract,
        enabled=enabled,
        highlight_intervals=highlight_intervals,
        fps=30.0,
        config=config,
    )


class DirectMemberTransitionTests(unittest.TestCase):
    def test_safe_motion_pose_boundaries_fill_bounded_budget_and_render_logistic(self) -> None:
        source_path = (0, 1, 2, 3, 0, 1, 2, 3)
        transition_plan = _plan(source_path)

        self.assertEqual(transition_plan.selected_count, 2)
        selected = [
            item.boundary_index
            for item in transition_plan.decisions
            if item.selected
        ]
        self.assertTrue(all(right - left > 1 for left, right in zip(selected, selected[1:])))
        for transition in transition_plan.transitions:
            if transition.kind is TransitionKind.FACE_DISSOLVE:
                self.assertEqual(transition.reason, DIRECT_MEMBER_POSE_DISSOLVE_REASON)
                self.assertGreaterEqual(transition.duration_seconds(30.0), 0.18)
                self.assertLessEqual(transition.duration_seconds(30.0), 0.30)

        render_plan = RenderPlan(
            sources=tuple(Path(f"{index}.mp4") for index in range(4)),
            segments=_segments(source_path),
            reference_source=0,
            aspect_ratio=AspectRatio.PORTRAIT,
            output=Path("out.mp4"),
            width=1080,
            height=1920,
            fps=30.0,
            transitions=transition_plan.transitions,
            member_pose_dissolve_requested=True,
            member_pose_dissolve_effective=True,
            member_pose_dissolve_count=2,
        )
        serialized = render_plan.to_json()
        self.assertEqual(serialized["member_pose_dissolve_count"], 2)
        self.assertTrue(
            all(
                item["alpha_curve"] == FACE_DISSOLVE_ALPHA_CURVE
                for item in serialized["transitions"]
                if item["kind"] == "face_dissolve"
            )
        )
        self.assertIn("xfade=transition=custom", build_filter_graph(render_plan))

    def test_warning_severe_and_insufficient_evidence_fail_closed(self) -> None:
        source_path = (0, 1, 2, 3)
        selection = _selection(
            source_path,
            warning_boundaries=frozenset({0}),
            severe_boundaries=frozenset({1}),
        )
        selection.boundaries[2].mismatch.evidence_component_count = 2

        plan = _plan(source_path, selection=selection)

        self.assertEqual(plan.selected_count, 0)
        self.assertIn("warning_endpoint_geometry", plan.decisions[0].rejection_reasons)
        self.assertIn("severe_endpoint_geometry", plan.decisions[1].rejection_reasons)
        self.assertIn(
            "insufficient_endpoint_geometry_evidence",
            plan.decisions[2].rejection_reasons,
        )
        self.assertTrue(
            all(item.kind is TransitionKind.HARD_CUT for item in plan.transitions)
        )

    def test_required_geometry_repair_cannot_overlap_a_dissolve(self) -> None:
        source_path = (0, 1)
        selection = _selection(source_path)
        selection.boundaries[0].mismatch.transform_repair = SimpleNamespace(
            required=True
        )

        plan = _plan(source_path, selection=selection)

        self.assertEqual(plan.selected_count, 0)
        self.assertEqual(
            plan.decisions[0].rejection_reasons,
            ("lossless_transform_repair_requires_hard_cut",),
        )
        self.assertIs(plan.transitions[0].kind, TransitionKind.HARD_CUT)

    def test_disabled_or_unverified_metadata_is_exact_all_hard_cut(self) -> None:
        source_path = (0, 1, 2, 3)
        disabled = _plan(source_path, enabled=False)
        unverified = _plan(source_path, metadata_contract="local_files")

        self.assertFalse(disabled.to_dict()["member_pose_dissolve_requested"])
        self.assertFalse(disabled.to_dict()["member_pose_dissolve_effective"])
        self.assertEqual(disabled.to_dict()["member_pose_dissolve_count"], 0)
        self.assertTrue(
            all(
                "effect_disabled_by_request" in item.rejection_reasons
                for item in disabled.decisions
            )
        )
        self.assertTrue(
            all(
                "source_metadata_contract_unverified" in item.rejection_reasons
                for item in unverified.decisions
            )
        )

    def test_render_plan_rejects_disabled_flag_with_injected_member_dissolve(self) -> None:
        source_path = (0, 1)
        transition_plan = _plan(source_path)
        self.assertEqual(transition_plan.selected_count, 1)

        with self.assertRaisesRegex(
            ValueError,
            "disabled member pose dissolve",
        ):
            RenderPlan(
                sources=(Path("0.mp4"), Path("1.mp4")),
                segments=_segments(source_path),
                reference_source=0,
                aspect_ratio=AspectRatio.PORTRAIT,
                output=Path("out.mp4"),
                width=1080,
                height=1920,
                fps=30.0,
                transitions=transition_plan.transitions,
                member_pose_dissolve_requested=False,
                member_pose_dissolve_effective=True,
                member_pose_dissolve_count=1,
            )
        with self.assertRaisesRegex(
            ValueError,
            "uncontracted face dissolve",
        ):
            RenderPlan(
                sources=(Path("0.mp4"), Path("1.mp4")),
                segments=_segments(source_path),
                reference_source=0,
                aspect_ratio=AspectRatio.PORTRAIT,
                output=Path("out.mp4"),
                width=1080,
                height=1920,
                fps=30.0,
                transitions=(
                    replace(
                        transition_plan.transitions[0],
                        reason="forged_dissolve_reason",
                    ),
                ),
                member_pose_dissolve_requested=False,
                member_pose_dissolve_effective=False,
                member_pose_dissolve_count=0,
            )

    def test_handle_duration_incoming_frames_and_native_cut_fail_closed(self) -> None:
        source_path = (0, 1)
        too_short = _plan(source_path, durations=(4.05, 120.0))
        native_cut = _plan(
            source_path,
            selection=_selection(source_path, native_cut_boundary=0),
        )
        short_segments_plan = build_direct_member_pose_dissolve_plan(
            visual_selection=_selection(source_path, duration=4.0),
            render_segments=(
                RenderSegment(0, 0.0, 4.0, 0.0),
                RenderSegment(1, 4.0, 0.10, 4.0),
            ),
            eligible_request_source_indices=(0, 1),
            eligible_source_durations_seconds=(120.0, 120.0),
            common_master_start=0.0,
            source_metadata_contract=OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
            fps=30.0,
        )

        self.assertIn("outgoing_source_handle_too_short", too_short.decisions[0].rejection_reasons)
        self.assertIn(
            "outgoing_handle_crosses_native_source_cut",
            native_cut.decisions[0].rejection_reasons,
        )
        self.assertIn(
            "incoming_segment_shorter_than_dissolve",
            short_segments_plan.decisions[0].rejection_reasons,
        )

    def test_future_pose_divergence_and_stale_source_binding_fail_closed(self) -> None:
        source_path = (0, 1)
        divergent = _plan(
            source_path,
            selection=_selection(source_path, future_divergent_boundary=0),
        )
        stale = _selection(source_path)
        stale.boundaries[0].incoming_eligible_source_index = 0
        stale_plan = _plan(source_path, selection=stale)

        self.assertEqual(divergent.selected_count, 0)
        self.assertTrue(
            any(
                reason.startswith("outgoing_handle_")
                for reason in divergent.decisions[0].rejection_reasons
            )
        )
        self.assertIn(
            "boundary_source_metadata_disagrees_with_render",
            stale_plan.decisions[0].rejection_reasons,
        )

    def test_highlight_and_both_moving_boundary_wins_single_effect_budget(self) -> None:
        source_path = (0, 1, 2, 3)
        selection = _selection(
            source_path,
            moving_boundaries=frozenset({0, 1, 2}),
        )
        plan = _plan(
            source_path,
            selection=selection,
            highlight_intervals=((7.5, 8.5),),
        )

        self.assertEqual(plan.selected_count, 1)
        self.assertTrue(plan.decisions[1].selected)
        self.assertTrue(plan.decisions[1].highlight_priority)
        payload = plan.to_dict()
        self.assertEqual(payload["selected_highlight_count"], 1)
        self.assertEqual(payload["selected_both_moving_count"], 1)
        self.assertGreater(payload["eligible_candidates_per_minute"], 0.0)
        self.assertEqual(
            payload["config"]["reference_calibration"]["profile_file_sha256"],
            "6f9e579a0bea0900d708e98950469faa6b712c976e66d9bca45f3b0b60db167c",
        )

    def test_forged_low_aggregate_cannot_hide_exact_endpoint_divergence(self) -> None:
        source_path = (0, 1)
        selection = _selection(source_path)
        boundary = selection.boundaries[0]
        incoming = boundary.incoming_sample
        boundary.incoming_sample = _sample(
            incoming.master_seconds,
            20.0,
        )

        plan = _plan(source_path, selection=selection)

        self.assertEqual(plan.selected_count, 0)
        self.assertIn(
            "attention_position_mismatch_exceeds_dissolve_bound",
            plan.decisions[0].rejection_reasons,
        )

    def test_nonexact_motion_and_handle_source_clock_evidence_fail_closed(self) -> None:
        source_path = (0, 1)
        stale_motion = _selection(source_path)
        stale_motion.boundaries[
            0
        ].mismatch.motion_connection.incoming_next_master_seconds += 0.2
        stale_clock = _selection(source_path)
        stale_clock.span_validity[1].sources[0].samples = (
            replace(
                stale_clock.span_validity[1].sources[0].samples[0],
                source_local_seconds=4.25,
            ),
            stale_clock.span_validity[1].sources[0].samples[1],
        )

        motion_plan = _plan(source_path, selection=stale_motion)
        clock_plan = _plan(source_path, selection=stale_clock)

        self.assertIn(
            "adjacent_motion_timestamps_not_output_frame_bound",
            motion_plan.decisions[0].rejection_reasons,
        )
        self.assertIn(
            "outgoing_handle_future_not_render_source_clock_bound",
            clock_plan.decisions[0].rejection_reasons,
        )

    def test_selected_cut_exact_handles_override_stale_coarse_span_samples(
        self,
    ) -> None:
        source_path = (0, 1)
        selection = _selection(source_path)
        boundary = selection.boundaries[0]
        boundary.outgoing_handle_samples = (
            _sample(4.0 + 1.0 / 30.0, 0.10),
            _sample(4.20, 0.11),
        )
        boundary.incoming_handle_samples = (
            _sample(4.0 + 1.0 / 30.0, 0.11),
            _sample(4.20, 0.12),
        )
        for source in selection.span_validity[1].sources:
            source.samples = (
                _sample(3.80, 20.0),
                _sample(4.00, 20.0),
            )

        plan = _plan(source_path, selection=selection)

        self.assertEqual(plan.selected_count, 1)
        self.assertNotIn(
            "outgoing_handle_future_timestamps_not_exact",
            plan.decisions[0].rejection_reasons,
        )
        self.assertNotIn(
            "outgoing_handle_future_not_render_source_clock_bound",
            plan.decisions[0].rejection_reasons,
        )

    def test_hidden_same_source_cut_at_handle_entry_fails_closed(self) -> None:
        source_path = (0, 1)
        selection = _selection(source_path)
        future_span = selection.span_validity[1]
        for source in future_span.sources:
            source.samples = (
                _sample(4.0 + 1.0 / 30.0, 20.0),
                _sample(4.20, 20.0),
            )

        plan = _plan(source_path, selection=selection)

        self.assertEqual(plan.selected_count, 0)
        self.assertIn(
            "outgoing_handle_entry_discontinuity",
            plan.decisions[0].rejection_reasons,
        )

    def test_required_cinematic_bridges_render_exact_zoom_and_blur_effects(self) -> None:
        for style, expected_kind, filter_token in (
            (
                "zoom_in",
                TransitionKind.MEMBER_ZOOM_BRIDGE,
                "xfade=transition=zoomin",
            ),
            (
                "horizontal_blur",
                TransitionKind.MEMBER_BLUR_BRIDGE,
                "xfade=transition=hblur",
            ),
        ):
            with self.subTest(style=style):
                transition_plan = _plan(
                    (0, 1),
                    selection=_cinematic_selection(style=style),
                    highlight_intervals=((3.5, 4.5),),
                )
                self.assertEqual(transition_plan.selected_count, 0)
                self.assertEqual(transition_plan.cinematic_bridge_count, 1)
                self.assertIs(
                    transition_plan.transitions[0].kind,
                    expected_kind,
                )
                self.assertTrue(
                    transition_plan.decisions[0].required_cinematic_bridge
                )
                render_plan = RenderPlan(
                    sources=(Path("0.mp4"), Path("1.mp4")),
                    segments=_segments((0, 1)),
                    reference_source=0,
                    aspect_ratio=AspectRatio.PORTRAIT,
                    output=Path("out.mp4"),
                    width=1080,
                    height=1920,
                    fps=30.0,
                    transitions=transition_plan.transitions,
                    member_pose_dissolve_requested=True,
                    member_pose_dissolve_effective=False,
                    member_pose_dissolve_count=0,
                )
                self.assertIn(filter_token, build_filter_graph(render_plan))

    def test_required_bridge_accepts_verified_cross_attention_basis(self) -> None:
        transition_plan = _plan(
            (0, 1),
            selection=_cinematic_selection(cross_attention_basis=True),
            highlight_intervals=((3.5, 4.5),),
        )

        self.assertEqual(transition_plan.cinematic_bridge_count, 1)
        self.assertTrue(
            transition_plan.decisions[0].required_cinematic_bridge
        )
        self.assertEqual(
            transition_plan.decisions[0].attention_basis,
            "verified_cross_basis_pose_silhouette_effect_bridge",
        )

    def test_nonhighlight_bridge_requires_exceptional_pose_body_envelope(self) -> None:
        accepted = _plan(
            (0, 1),
            selection=_cinematic_selection(),
            highlight_intervals=(),
        )
        rejected_selection = _cinematic_selection()
        rejected_selection.boundaries[
            0
        ].mismatch.cinematic_bridge.pose_mismatch = 0.23

        self.assertEqual(accepted.cinematic_bridge_count, 1)
        self.assertFalse(accepted.decisions[0].highlight_priority)
        self.assertEqual(accepted.transitions[0].tier, "B")
        with self.assertRaisesRegex(
            RuntimeError,
            "cinematic_bridge_nonpriority_pose_mismatch",
        ):
            _plan(
                (0, 1),
                selection=rejected_selection,
                highlight_intervals=(),
            )

    def test_required_bridge_audits_short_outgoing_handle_independently(self) -> None:
        selection = _cinematic_selection()
        outgoing_future = selection.span_validity[1].sources[0]
        outgoing_future.valid = False

        transition_plan = _plan(
            (0, 1),
            selection=selection,
            highlight_intervals=((3.5, 4.5),),
        )

        self.assertEqual(transition_plan.cinematic_bridge_count, 1)
        self.assertNotIn(
            "cinematic_bridge_outgoing_future_content_invalid",
            transition_plan.decisions[0].rejection_reasons,
        )

    def test_required_cinematic_bridge_never_degrades_to_hard_cut(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "required direct member cinematic bridge",
        ):
            _plan(
                (0, 1),
                selection=_cinematic_selection(),
                highlight_intervals=((3.5, 4.5),),
                enabled=False,
            )

    def test_malformed_source_metadata_and_low_fps_degrade_to_hard_cut(self) -> None:
        source_path = (0, 1)
        malformed = build_direct_member_pose_dissolve_plan(
            visual_selection=_selection(source_path),
            render_segments=_segments(source_path),
            eligible_request_source_indices=(0, True),
            eligible_source_durations_seconds=(120.0, "invalid"),
            common_master_start=0.0,
            source_metadata_contract=OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
            fps=30.0,
        )
        low_fps = build_direct_member_pose_dissolve_plan(
            visual_selection=_selection(source_path, fps=5.0),
            render_segments=_segments(source_path),
            eligible_request_source_indices=(0, 1),
            eligible_source_durations_seconds=(120.0, 120.0),
            common_master_start=0.0,
            source_metadata_contract=OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
            fps=5.0,
        )

        self.assertEqual(malformed.selected_count, 0)
        self.assertIn(
            "eligible_request_source_metadata_invalid",
            malformed.decisions[0].rejection_reasons,
        )
        self.assertEqual(low_fps.selected_count, 0)
        self.assertIn(
            "output_fps_cannot_represent_safe_dissolve_duration",
            low_fps.decisions[0].rejection_reasons,
        )


if __name__ == "__main__":
    unittest.main()
