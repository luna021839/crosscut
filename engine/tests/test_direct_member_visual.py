from __future__ import annotations

import math
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.direct_member_visual import (
    DirectBoundaryMismatch,
    DirectMemberVisualConfig,
    DirectMemberVisualSelectionError,
    _VideoDescriptorProvider,
    _balanced_visual_path,
    _hard_boundary_connectivity_diagnostics,
    select_balanced_direct_member_sources,
)
from crosscut.pose import FormationPersonDescriptor
from crosscut.timewarp import (
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor


def _descriptor(
    value: float,
    *,
    scale: float = 0.22,
    face_detector_backend: str = "yunet",
    quality_score: float = 1.0,
    foreground_obstruction_score: float = 0.0,
    gray_mean_score: float | None = None,
    gray_contrast_score: float | None = None,
    highlight_clipping_ratio: float = 0.0,
) -> FrameVisualDescriptor:
    x = min(0.74, max(0.0, value))
    bbox = (x, 0.20, scale, 0.44)
    centroid = (x + scale / 2.0, 0.42)
    person = FormationPersonDescriptor(
        bbox_xywh=bbox,
        centroid_xy=centroid,
        detection_confidence=1.0,
    )
    embedding = (value, value * 0.5, 1.0 - value)
    color_embedding = (
        ()
        if gray_mean_score is None or gray_contrast_score is None
        else (
            0.5,
            0.5,
            0.5,
            0.25,
            0.25,
            0.25,
            gray_mean_score,
            gray_contrast_score,
        )
    )
    return FrameVisualDescriptor(
        quality_score=quality_score,
        composition_score=1.0,
        shot_scale=scale,
        shot_type=ShotType.MEDIUM,
        face_count=1,
        face_detector_backend=face_detector_backend,
        structure_embedding=embedding,
        color_embedding=color_embedding,
        primary_face_bbox=bbox,
        pose_embedding=embedding,
        pose_confidence=1.0,
        detected_person_count=1,
        person_layout_embedding=embedding,
        formation_people=(person,),
        pose_backend="synthetic_non_identity_pose",
        silhouette_embedding=embedding,
        silhouette_bbox_xywh=bbox,
        silhouette_centroid_xy=centroid,
        silhouette_area_ratio=scale * 0.44,
        silhouette_confidence=1.0,
        silhouette_backend="synthetic_silhouette",
        foreground_obstruction_score=foreground_obstruction_score,
        highlight_clipping_ratio=highlight_clipping_ratio,
    )


def _end_card_descriptor() -> FrameVisualDescriptor:
    return FrameVisualDescriptor(
        quality_score=0.75,
        composition_score=0.5,
        shot_scale=0.12,
        shot_type=ShotType.GROUP_WIDE,
        face_count=0,
        structure_embedding=(0.2, 0.2, 0.2),
        color_embedding=(0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.5, 0.01),
    )


def _static_false_human_end_card_descriptor() -> FrameVisualDescriptor:
    """Model a logo card that fools pose/person/silhouette but not face detection."""

    bbox = (0.18, 0.16, 0.58, 0.70)
    person = FormationPersonDescriptor(
        bbox_xywh=bbox,
        centroid_xy=(0.47, 0.51),
        detection_confidence=0.91,
    )
    return FrameVisualDescriptor(
        quality_score=0.85,
        composition_score=0.8,
        shot_scale=0.40,
        shot_type=ShotType.MEDIUM,
        face_count=0,
        structure_embedding=(0.2, 0.2, 0.2),
        color_embedding=(0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.5, 0.01),
        pose_embedding=(0.2, 0.2, 0.2),
        pose_confidence=0.8,
        detected_person_count=1,
        person_layout_embedding=(0.2, 0.2, 0.2),
        formation_people=(person,),
        pose_backend="synthetic_false_positive_pose",
        silhouette_embedding=(0.2, 0.2, 0.2),
        silhouette_bbox_xywh=bbox,
        silhouette_centroid_xy=(0.47, 0.51),
        silhouette_area_ratio=0.406,
        silhouette_confidence=0.9,
        silhouette_backend="synthetic_false_positive_silhouette",
    )


def _optimizer_mismatch(
    cost: float = 0.0,
    *,
    severe: bool = False,
) -> DirectBoundaryMismatch:
    return DirectBoundaryMismatch(
        cost=cost,
        base_mismatch=cost,
        attention_position_mismatch=None,
        attention_scale_mismatch=None,
        dominant_pose_mismatch=None,
        pose_basis=None,
        formation_mismatch=None,
        silhouette_mismatch=None,
        person_layout_mismatch=None,
        person_count_mismatch=None,
        evidence_component_count=0,
        warning_flags=(),
        severe_flags=("synthetic_incompatible_edge",) if severe else (),
    )


def _beam_exhaustion_adversarial_case() -> tuple[
    tuple[tuple[tuple[DirectBoundaryMismatch, ...], ...], ...],
    tuple[tuple[bool, ...], ...],
    tuple[float, ...],
]:
    source_count = 5
    span_count = 15
    required_predecessor = 3
    final_source = 4
    matrices = tuple(
        tuple(
            tuple(
                _optimizer_mismatch(
                    0.0 if right == required_predecessor else 100.0,
                    severe=(
                        boundary_index == span_count - 2
                        and right == final_source
                        and left != required_predecessor
                    ),
                )
                for right in range(source_count)
            )
            for left in range(source_count)
        )
        for boundary_index in range(span_count - 1)
    )
    validity = (
        *((True,) * source_count for _ in range(span_count - 1)),
        tuple(index == final_source for index in range(source_count)),
    )
    return (matrices, validity, (1.0,) * span_count)


def _linear_timewarp(
    *,
    duration: float,
    global_offset: float,
    rate: float,
) -> TimeWarpMap:
    return TimeWarpMap(
        global_offset_seconds=global_offset,
        master_duration_seconds=duration,
        source_duration_seconds=global_offset + duration * rate,
        control_points=(
            TimeWarpControlPoint(0.0, global_offset),
            TimeWarpControlPoint(duration, global_offset + duration * rate),
        ),
        anchors=(),
        residuals=TimeWarpResidualDiagnostics(None, None, None, 0),
        config=TimeWarpConfig(maximum_drift_rate=0.01),
        scalar_fallback=False,
        fallback_reason=None,
    )


class DirectMemberVisualTests(unittest.TestCase):
    def test_hard_boundary_connectivity_localizes_only_the_actual_zero_edge(
        self,
    ) -> None:
        safe = _optimizer_mismatch()
        severe = _optimizer_mismatch(severe=True)
        matrices = (
            (
                (safe, severe),
                (severe, safe),
            ),
            (
                (safe, safe),
                (safe, safe),
            ),
        )
        diagnostics = _hard_boundary_connectivity_diagnostics(
            matrices,
            ((True, True), (True, True), (True, True)),
        )

        self.assertEqual(
            diagnostics["zero_local_nonsevere_transition_boundary_indices"],
            [0],
        )
        # Global forward/backward reachability collapses after boundary zero,
        # but that must not falsely identify boundary one as the repair site.
        self.assertEqual(
            diagnostics["zero_reachable_transition_boundary_indices"],
            [0, 1],
        )

    def test_video_descriptor_provider_reuses_the_same_millisecond_sample(self) -> None:
        class Capture:
            def __init__(self) -> None:
                self.read_count = 0

            def set(self, _property: int, _value: float) -> None:
                return None

            def read(self) -> tuple[bool, object]:
                self.read_count += 1
                return (True, object())

            def release(self) -> None:
                return None

        capture = Capture()
        provider = object.__new__(_VideoDescriptorProvider)
        provider._captures = [capture]
        provider._descriptor_cache = {}
        provider._request_count = 0
        provider._cache_hit_count = 0
        provider._decode_count = 0
        descriptor = _descriptor(0.2)

        with patch(
            "crosscut.direct_member_visual.describe_frame",
            return_value=descriptor,
        ) as describe:
            first = provider(0, Path("source.mp4"), 1.0001)
            second = provider(0, Path("source.mp4"), 1.0004)

        self.assertIs(first, descriptor)
        self.assertIs(second, descriptor)
        self.assertEqual(capture.read_count, 1)
        describe.assert_called_once()
        self.assertEqual(
            provider.cache_diagnostics(),
            {
                "descriptor_request_count": 2,
                "descriptor_cache_hit_count": 1,
                "descriptor_decode_count": 1,
                "descriptor_cache_entry_count": 1,
            },
        )

    def test_visual_path_beats_round_robin_while_remaining_exactly_balanced(self) -> None:
        desired_path = (0, 2, 1, 0, 2, 1)
        spans = tuple((float(index), float(index + 1)) for index in range(6))
        config = DirectMemberVisualConfig(
            sample_frame_step_seconds=0.1,
            beam_width=1_000,
        )

        def provider(
            source_index: int,
            _source: Path,
            source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            boundary = int(round(source_local_seconds))
            incoming = source_local_seconds > boundary
            desired_source = (
                desired_path[boundary]
                if incoming
                else desired_path[boundary - 1]
            )
            boundary_value = 0.12 + boundary * 0.05
            if source_index == desired_source:
                value = boundary_value
            else:
                value = min(0.74, boundary_value + 0.24 + source_index * 0.08)
            return _descriptor(value)

        selection = select_balanced_direct_member_sources(
            sources=tuple(Path(f"{index}.mp4") for index in range(3)),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.0, 0.0),
            preferred_source_order=(0, 1, 2),
            config=config,
            descriptor_provider=provider,
        )

        self.assertEqual(selection.source_indices, desired_path)
        self.assertNotEqual(selection.source_indices, (0, 1, 2, 0, 1, 2))
        self.assertEqual(selection.source_counts, (2, 2, 2))
        self.assertEqual(selection.source_quotas, (2, 2, 2))
        self.assertTrue(
            all(
                left != right
                for left, right in zip(
                    selection.source_indices,
                    selection.source_indices[1:],
                )
            )
        )
        self.assertEqual(len(selection.boundaries), 5)
        self.assertGreater(
            sum(boundary.severe_pair_count for boundary in selection.boundaries),
            0,
        )
        for boundary in selection.boundaries:
            self.assertEqual(
                boundary.candidate_pair_count,
                boundary.raw_cross_source_pair_count
                - boundary.invalid_content_pair_count
                - boundary.severe_pair_count,
            )
            self.assertLessEqual(
                boundary.selected_pair_rank,
                boundary.candidate_pair_count,
            )
            self.assertLessEqual(
                boundary.minimum_pair_cost,
                boundary.mismatch.cost,
            )
        self.assertEqual(selection.sample_count, 168)
        self.assertEqual(selection.failed_sample_count, 0)
        payload = selection.to_dict()
        self.assertEqual(payload["distinct_source_count"], 3)
        self.assertTrue(payload["counts_differ_by_at_most_one"])
        self.assertEqual(payload["same_source_adjacent_count"], 0)
        self.assertTrue(payload["aggregate"]["duration_balance"]["passed"])
        self.assertEqual(
            [item["duration_seconds"] for item in payload["source_usage"]],
            [2.0, 2.0, 2.0],
        )
        self.assertEqual(
            [item["target_duration_seconds"] for item in payload["source_usage"]],
            [2.0, 2.0, 2.0],
        )
        self.assertEqual(
            [item["duration_relative_deviation"] for item in payload["source_usage"]],
            [0.0, 0.0, 0.0],
        )
        self.assertFalse(payload["privacy"]["face_identity_inference_used"])
        self.assertFalse(
            payload["privacy"]["raw_pose_silhouette_layout_embeddings_persisted"]
        )
        self.assertEqual(
            payload["optimizer_search"]["selection_basis"],
            "adaptive_cost_beam",
        )
        self.assertFalse(payload["optimizer_search"]["search_capacity_exhausted"])

    def test_cut_adjacent_both_moving_connection_beats_static_endpoint_tie(
        self,
    ) -> None:
        def provider(
            source_index: int,
            _source: Path,
            source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            sample = round(source_local_seconds, 1)
            if source_index == 0 and sample == 0.8:
                return _descriptor(0.20)
            if source_index == 1 and sample == 1.2:
                return _descriptor(0.20)
            return _descriptor(0.40)

        selection = select_balanced_direct_member_sources(
            sources=(Path("moving-out.mp4"), Path("moving-in.mp4")),
            canonical_spans=((0.0, 1.0), (1.0, 2.0)),
            audio_offsets=(0.0, 0.0),
            preferred_source_order=(1, 0),
            config=DirectMemberVisualConfig(
                sample_frame_step_seconds=0.1,
                beam_width=200,
            ),
            descriptor_provider=provider,
        )

        # Static endpoint geometry is identical for both possible paths and
        # preferred order would otherwise choose 1->0. The actual adjacent
        # motion evidence must make the opposite, moving 0->1 connection win.
        self.assertEqual(selection.source_indices, (0, 1))
        evidence = selection.boundaries[0].mismatch.motion_connection
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertTrue(evidence.timestamp_bound_adjacent)
        self.assertTrue(evidence.both_moving)
        self.assertGreater(evidence.soft_preference_applied, 0.0)
        self.assertIsNotNone(evidence.direction_cosine_audit_only)
        self.assertLess(float(evidence.direction_cosine_audit_only), 0.0)
        payload = selection.to_dict()
        motion = payload["boundaries"][0]["mismatch"]["motion_connection"]
        self.assertFalse(
            motion["direction_cosine_influenced_gate_or_bonus"]
        )
        self.assertFalse(motion["sparse_or_nonadjacent_evidence_used"])
        self.assertFalse(motion["raw_vectors_persisted"])
        self.assertEqual(
            motion["master_timestamps_seconds"],
            {
                "outgoing_previous": 0.8,
                "outgoing_endpoint": 0.9,
                "incoming_start": 1.1,
                "incoming_next": 1.2,
            },
        )

    def test_span_content_sampling_includes_entry_early_and_quartiles(self) -> None:
        selection = select_balanced_direct_member_sources(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            canonical_spans=((0.0, 4.0), (4.0, 8.0)),
            audio_offsets=(0.0, 0.0),
            config=DirectMemberVisualConfig(beam_width=200),
            descriptor_provider=lambda source_index, _source, _time: _descriptor(
                0.20 + source_index * 0.02
            ),
        )

        samples = selection.span_validity[0].sources[0].samples
        self.assertEqual(len(samples), 6)
        self.assertEqual(
            [round(sample.master_seconds, 6) for sample in samples],
            [round(1.0 / 29.97, 6), 0.2, 1.0, 2.0, 3.0, 3.82],
        )
        content_policy = selection.to_dict()["config"]["content_validity"]
        self.assertEqual(content_policy["sample_count"], 6)
        self.assertEqual(
            content_policy["persistent_entry_overexposure_policy"][
                "required_consecutive_samples"
            ],
            2,
        )

    def test_adaptive_beam_and_exact_lifeline_recover_pruned_quota_path(self) -> None:
        matrices, validity, durations = _beam_exhaustion_adversarial_case()

        adaptive_path, _cost, adaptive_diagnostics = _balanced_visual_path(
            matrices,
            source_quotas=(3, 3, 3, 3, 3),
            source_order=(0, 1, 2, 3, 4),
            candidate_validity=validity,
            span_durations=durations,
            config=DirectMemberVisualConfig(
                beam_width=100,
                maximum_adaptive_beam_width=800,
            ),
        )
        lifeline_path, _cost, lifeline_diagnostics = _balanced_visual_path(
            matrices,
            source_quotas=(3, 3, 3, 3, 3),
            source_order=(0, 1, 2, 3, 4),
            candidate_validity=validity,
            span_durations=durations,
            config=DirectMemberVisualConfig(
                beam_width=100,
                maximum_adaptive_beam_width=100,
            ),
        )

        for path in (adaptive_path, lifeline_path):
            self.assertEqual(tuple(path.count(index) for index in range(5)), (3,) * 5)
            self.assertTrue(
                all(left != right for left, right in zip(path, path[1:]))
            )
            self.assertEqual(path[-2:], (3, 4))
        self.assertEqual(
            adaptive_diagnostics["attempted_beam_widths"],
            [100, 200, 400, 800],
        )
        self.assertEqual(
            adaptive_diagnostics["selection_basis"],
            "adaptive_cost_beam",
        )
        self.assertEqual(
            lifeline_diagnostics["selection_basis"],
            "exact_feasibility_lifeline",
        )
        self.assertTrue(lifeline_diagnostics["exact_lifeline"]["path_found"])

    def test_search_capacity_exhaustion_is_not_reported_as_infeasible(self) -> None:
        matrices, validity, durations = _beam_exhaustion_adversarial_case()

        with self.assertRaises(DirectMemberVisualSelectionError) as raised:
            _balanced_visual_path(
                matrices,
                source_quotas=(3, 3, 3, 3, 3),
                source_order=(0, 1, 2, 3, 4),
                candidate_validity=validity,
                span_durations=durations,
                config=DirectMemberVisualConfig(
                    beam_width=100,
                    maximum_adaptive_beam_width=100,
                    feasibility_lifeline_state_limit=1,
                ),
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(diagnostics["failure"], "search_capacity_exhausted")
        self.assertFalse(diagnostics["constraint_infeasibility_proven"])
        self.assertFalse(diagnostics["visual_gates_relaxed"])
        self.assertEqual(
            diagnostics["search"]["attempted_beam_widths"],
            [100],
        )
        self.assertGreater(
            diagnostics["search"]["attempts"][0]["pruned_state_count"],
            0,
        )
        self.assertTrue(
            diagnostics["search"]["exact_lifeline"][
                "state_capacity_exhausted"
            ]
        )

    def test_exact_lifeline_reports_root_prefix_valid_slot_shortfalls(self) -> None:
        source_count = 3
        span_count = 6
        safe_row = tuple(
            _optimizer_mismatch() for _ in range(source_count)
        )
        matrices = tuple(
            tuple(safe_row for _ in range(source_count))
            for _ in range(span_count - 1)
        )
        # Sources 0 and 1 each have their second and final quota slot at the
        # same last segment.  The aggregate per-source precheck passes, but
        # choosing either final candidate leaves the other source one prefix
        # slot short: a small Hall-style conflict.
        validity = (
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (False, False, True),
            (False, False, True),
            (True, True, False),
        )

        with self.assertRaises(DirectMemberVisualSelectionError) as raised:
            _balanced_visual_path(
                matrices,
                source_quotas=(2, 2, 2),
                source_order=(0, 1, 2),
                candidate_validity=validity,
                span_durations=(1.0,) * span_count,
                config=DirectMemberVisualConfig(
                    beam_width=100,
                    maximum_adaptive_beam_width=100,
                ),
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(diagnostics["failure"], "validity_quota_path_exhausted")
        lifeline = diagnostics["search"]["exact_lifeline"]
        self.assertEqual(lifeline["explored_state_count"], 2)
        roots = lifeline["root_candidate_diagnostics"]
        self.assertEqual(
            [root["final_candidate_source_index"] for root in roots],
            [0, 1],
        )
        self.assertTrue(
            all(root["search_outcome"] == "root_prefix_pruned" for root in roots)
        )
        expected_blocked_source = {0: 1, 1: 0}
        for root in roots:
            reasons = root["prune_reasons"]
            self.assertEqual(len(reasons), 1)
            reason = reasons[0]
            self.assertEqual(reason["reason"], "prefix_valid_slot_shortfall")
            self.assertEqual(
                reason["eligible_source_index"],
                expected_blocked_source[root["final_candidate_source_index"]],
            )
            self.assertEqual(reason["quota_deficit"], 2)
            self.assertEqual(reason["prefix_valid_slot_count"], 1)
            self.assertEqual(reason["shortfall"], 1)

    def test_exact_lifeline_reports_root_duration_upper_bound(self) -> None:
        source_count = 2
        span_count = 4
        safe_row = tuple(
            _optimizer_mismatch() for _ in range(source_count)
        )
        matrices = tuple(
            tuple(safe_row for _ in range(source_count))
            for _ in range(span_count - 1)
        )

        with self.assertRaises(DirectMemberVisualSelectionError) as raised:
            _balanced_visual_path(
                matrices,
                source_quotas=(2, 2),
                source_order=(0, 1),
                candidate_validity=((True, True),) * span_count,
                span_durations=(1.0, 1.0, 1.0, 10.0),
                config=DirectMemberVisualConfig(
                    beam_width=100,
                    maximum_adaptive_beam_width=100,
                    maximum_source_duration_relative_deviation=0.10,
                ),
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(
            diagnostics["failure"],
            "duration_balance_threshold_unsatisfied",
        )
        roots = diagnostics["search"]["exact_lifeline"][
            "root_candidate_diagnostics"
        ]
        self.assertEqual(len(roots), 2)
        for root in roots:
            duration_reasons = [
                reason
                for reason in root["prune_reasons"]
                if reason["reason"] == "duration_upper_bound_exceeded"
            ]
            self.assertEqual(len(duration_reasons), 1)
            reason = duration_reasons[0]
            self.assertGreater(
                reason["projected_minimum_duration_seconds"],
                reason["allowed_upper_duration_seconds"],
            )

    def test_optimizer_failure_preserves_full_scalar_validity_diagnostics(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(4))
        optimizer_error = DirectMemberVisualSelectionError(
            "synthetic optimizer failure",
            {
                "failure": "validity_quota_path_exhausted",
                "search": {"constraint_infeasibility_proven": True},
            },
        )

        with patch(
            "crosscut.direct_member_visual._balanced_visual_path",
            side_effect=optimizer_error,
        ), self.assertRaises(DirectMemberVisualSelectionError) as raised:
            select_balanced_direct_member_sources(
                sources=(Path("zero.mp4"), Path("one.mp4")),
                canonical_spans=spans,
                audio_offsets=(0.0, 0.0),
                config=DirectMemberVisualConfig(beam_width=200),
                descriptor_provider=lambda source_index, _source, _time: _descriptor(
                    0.20 + source_index * 0.004
                ),
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(len(diagnostics["span_content_validity"]), 4)
        self.assertEqual(
            diagnostics["valid_count_by_source"],
            [
                {
                    "eligible_source_index": 0,
                    "quota": 2,
                    "valid_span_count": 4,
                    "invalid_span_count": 0,
                    "valid_segment_indices": [0, 1, 2, 3],
                },
                {
                    "eligible_source_index": 1,
                    "quota": 2,
                    "valid_span_count": 4,
                    "invalid_span_count": 0,
                    "valid_segment_indices": [0, 1, 2, 3],
                },
            ],
        )
        self.assertTrue(diagnostics["privacy"]["validity_evidence_is_scalar_only"])
        self.assertTrue(
            all(
                source["raw_embeddings_persisted"] is False
                for span in diagnostics["span_content_validity"]
                for source in span["sources"]
            )
        )

    def test_reference_scale_fifty_span_path_uses_ten_sources_five_times(self) -> None:
        spans = tuple(
            (index * 3.9, (index + 1) * 3.9)
            for index in range(50)
        )

        selection = select_balanced_direct_member_sources(
            sources=tuple(Path(f"{index}.mp4") for index in range(10)),
            canonical_spans=spans,
            audio_offsets=(0.0,) * 10,
            config=DirectMemberVisualConfig(beam_width=4_096),
            descriptor_provider=lambda source_index, _source, _time: _descriptor(
                0.20 + source_index * 0.006
            ),
        )

        self.assertEqual(len(selection.source_indices), 50)
        self.assertEqual(selection.source_counts, (5,) * 10)
        self.assertEqual(len(set(selection.source_indices)), 10)
        self.assertEqual(
            sum(
                left == right
                for left, right in zip(
                    selection.source_indices,
                    selection.source_indices[1:],
                )
            ),
            0,
        )
        self.assertTrue(
            all(
                math.isclose(value, 19.5, abs_tol=1e-9)
                for value in selection.source_durations_seconds
            )
        )
        self.assertTrue(
            all(
                math.isclose(value, 19.5, abs_tol=1e-9)
                for value in selection.source_target_durations_seconds
            )
        )
        self.assertTrue(
            all(
                math.isclose(value, 0.0, abs_tol=1e-9)
                for value in selection.source_duration_relative_deviations
            )
        )

    def test_fifty_span_path_balances_eight_sources_by_floor_and_ceil_quota(
        self,
    ) -> None:
        spans = tuple(
            (index * 3.9, (index + 1) * 3.9)
            for index in range(50)
        )

        selection = select_balanced_direct_member_sources(
            sources=tuple(Path(f"{index}.mp4") for index in range(8)),
            canonical_spans=spans,
            audio_offsets=(0.0,) * 8,
            config=DirectMemberVisualConfig(beam_width=4_096),
            descriptor_provider=lambda source_index, _source, _time: _descriptor(
                0.20 + source_index * 0.006
            ),
        )

        self.assertEqual(selection.source_quotas, (7, 7, 6, 6, 6, 6, 6, 6))
        self.assertEqual(selection.source_counts, selection.source_quotas)
        self.assertEqual(len(set(selection.source_indices)), 8)
        self.assertFalse(
            any(
                left == right
                for left, right in zip(
                    selection.source_indices,
                    selection.source_indices[1:],
                )
            )
        )
        self.assertLessEqual(
            max(selection.source_duration_relative_deviations),
            0.08,
        )

    def test_eight_source_quota_shortfall_still_fails_closed(self) -> None:
        spans = tuple(
            (index * 3.9, (index + 1) * 3.9)
            for index in range(50)
        )
        scene_cuts = ((),) * 7 + (
            tuple(index * 3.9 + 1.0 for index in range(45)),
        )

        with self.assertRaises(DirectMemberVisualSelectionError) as raised:
            select_balanced_direct_member_sources(
                sources=tuple(Path(f"{index}.mp4") for index in range(8)),
                canonical_spans=spans,
                audio_offsets=(0.0,) * 8,
                source_scene_cut_times=scene_cuts,
                descriptor_provider=lambda source_index, _source, _time: _descriptor(
                    0.20 + source_index * 0.006
                ),
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(
            diagnostics["failure"],
            "content_validity_quota_precheck_failed",
        )
        self.assertEqual(
            diagnostics["quota_shortfalls"],
            [{"eligible_source_index": 7, "quota": 6, "valid_span_count": 5}],
        )
        self.assertFalse(diagnostics.get("visual_gates_relaxed", False))

    def test_variable_spans_are_balanced_by_duration_not_only_segment_count(self) -> None:
        durations = (3.0, 1.0, 3.0, 1.0, 3.0, 1.0)
        cursor = 0.0
        spans: list[tuple[float, float]] = []
        for duration in durations:
            spans.append((cursor, cursor + duration))
            cursor += duration

        selection = select_balanced_direct_member_sources(
            sources=tuple(Path(f"{index}.mp4") for index in range(3)),
            canonical_spans=tuple(spans),
            audio_offsets=(0.0, 0.0, 0.0),
            config=DirectMemberVisualConfig(
                beam_width=1_000,
                duration_balance_weight=4.0,
                maximum_source_duration_relative_deviation=0.01,
            ),
            descriptor_provider=lambda source_index, _source, _time: _descriptor(
                0.20 + source_index * 0.004
            ),
        )

        self.assertEqual(selection.source_counts, (2, 2, 2))
        self.assertEqual(selection.source_target_durations_seconds, (4.0, 4.0, 4.0))
        self.assertEqual(selection.source_durations_seconds, (4.0, 4.0, 4.0))
        self.assertEqual(selection.source_duration_relative_deviations, (0.0, 0.0, 0.0))
        self.assertTrue(selection.to_dict()["aggregate"]["duration_balance"]["passed"])

    def test_impossible_duration_balance_fails_closed_before_render(self) -> None:
        spans = ((0.0, 1.0), (1.0, 4.0), (4.0, 5.0), (5.0, 8.0))

        with self.assertRaises(DirectMemberVisualSelectionError) as raised:
            select_balanced_direct_member_sources(
                sources=(Path("zero.mp4"), Path("one.mp4")),
                canonical_spans=spans,
                audio_offsets=(0.0, 0.0),
                config=DirectMemberVisualConfig(
                    beam_width=200,
                    maximum_source_duration_relative_deviation=0.10,
                ),
                descriptor_provider=lambda source_index, _source, _time: _descriptor(
                    0.20 + source_index * 0.004
                ),
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(
            diagnostics["failure"],
            "duration_balance_threshold_unsatisfied",
        )
        self.assertAlmostEqual(
            diagnostics["best_maximum_relative_deviation"],
            0.5,
        )
        self.assertEqual(
            diagnostics["maximum_allowed_relative_deviation"],
            0.10,
        )
        self.assertEqual(
            [item["target_duration_seconds"] for item in diagnostics["source_usage"]],
            [4.0, 4.0],
        )
        self.assertTrue(diagnostics["constraint_infeasibility_proven"])
        self.assertEqual(
            sorted(item["duration_seconds"] for item in diagnostics["source_usage"]),
            [2.0, 6.0],
        )
        self.assertEqual(
            diagnostics["duration_threshold_effect"],
            "fail_closed_before_render",
        )

    def test_timewarp_maps_every_before_and_after_sample_to_source_clock(self) -> None:
        spans = ((0.0, 1.0), (1.0, 2.0))
        mapped = _linear_timewarp(
            duration=2.0,
            global_offset=0.2,
            rate=1.004,
        )
        calls: list[tuple[int, float]] = []

        def provider(
            source_index: int,
            _source: Path,
            source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            calls.append((source_index, source_local_seconds))
            return _descriptor(0.25 + source_index * 0.02)

        selection = select_balanced_direct_member_sources(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.2),
            source_timewarps=(None, mapped),
            config=DirectMemberVisualConfig(
                sample_frame_step_seconds=0.1,
                beam_width=200,
            ),
            descriptor_provider=provider,
        )

        source_one_times = [time for source, time in calls if source == 1]
        self.assertTrue(
            any(
                abs(value - mapped.master_to_source(0.9)) < 1e-9
                for value in source_one_times
            )
        )
        self.assertTrue(
            any(
                abs(value - mapped.master_to_source(1.1)) < 1e-9
                for value in source_one_times
            )
        )
        diagnostic = selection.boundaries[0]
        selected_sample = (
            diagnostic.outgoing_sample
            if diagnostic.outgoing_eligible_source_index == 1
            else diagnostic.incoming_sample
        )
        expected_master = (
            0.9 if diagnostic.outgoing_eligible_source_index == 1 else 1.1
        )
        self.assertAlmostEqual(
            selected_sample.source_local_seconds,
            mapped.master_to_source(expected_master),
        )

    def test_timewarped_native_cut_between_samples_hard_invalidates_span(self) -> None:
        # These are already-coarsened four-second spans. The native cut was a
        # boundary in the finer timing draft, but is intentionally absent from
        # the direct canonical span boundaries.
        spans = ((0.0, 4.0), (4.0, 8.0))
        mapped = _linear_timewarp(
            duration=8.0,
            global_offset=0.2,
            rate=1.004,
        )
        hidden_cut_master = 1.37
        hidden_cut_source = mapped.master_to_source(hidden_cut_master)

        selection = select_balanced_direct_member_sources(
            sources=(Path("cut.mp4"), Path("continuous.mp4")),
            canonical_spans=spans,
            audio_offsets=(0.2, 0.0),
            source_timewarps=(mapped, None),
            source_scene_cut_times=((hidden_cut_source,), ()),
            config=DirectMemberVisualConfig(beam_width=200),
            descriptor_provider=lambda source_index, _source, _time: _descriptor(
                0.20 + source_index * 0.01
            ),
        )

        # The six content descriptors alone are all valid and do not sample
        # 1.37s. The native-cut contract must still prevent source 0 from
        # rendering the first coarsened span.
        self.assertEqual(selection.source_indices, (1, 0))
        cut_candidate = selection.span_validity[0].sources[0]
        self.assertFalse(cut_candidate.valid)
        self.assertEqual(cut_candidate.face_evidence_sample_count, 6)
        self.assertEqual(cut_candidate.human_evidence_sample_count, 6)
        self.assertIn(
            "source_native_scene_cut_inside_rendered_span",
            cut_candidate.rejection_reasons,
        )
        self.assertEqual(
            cut_candidate.native_scene_cut_source_seconds,
            (hidden_cut_source,),
        )
        self.assertAlmostEqual(
            cut_candidate.native_scene_cut_master_seconds[0],
            hidden_cut_master,
        )
        aggregate = selection.aggregate_diagnostics()["content_validity"]
        self.assertEqual(aggregate["native_scene_cut_invalid_candidate_count"], 1)
        self.assertEqual(aggregate["native_scene_cut_count"], 1)

    def test_native_cut_one_frame_guard_is_exact_and_epsilon_inside_rejects(self) -> None:
        spans = ((0.0, 1.0), (1.0, 2.0))
        config = DirectMemberVisualConfig(beam_width=200)
        guard = config.native_scene_cut_boundary_tolerance_seconds
        provider = lambda source_index, _source, _time: _descriptor(  # noqa: E731
            0.20 + source_index * 0.01
        )

        aligned = select_balanced_direct_member_sources(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.0),
            source_scene_cut_times=((guard, 1.0 - guard), ()),
            config=config,
            descriptor_provider=provider,
        )
        self.assertTrue(aligned.span_validity[0].sources[0].valid)
        self.assertEqual(
            aligned.span_validity[0].sources[0].native_scene_cut_master_seconds,
            (),
        )

        inside = select_balanced_direct_member_sources(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.0),
            source_scene_cut_times=((guard + 1e-5,), ()),
            config=config,
            descriptor_provider=provider,
        )
        self.assertFalse(inside.span_validity[0].sources[0].valid)
        self.assertEqual(inside.source_indices, (1, 0))

    def test_native_cuts_that_exhaust_source_quota_fail_closed(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(4))

        with self.assertRaises(DirectMemberVisualSelectionError) as raised:
            select_balanced_direct_member_sources(
                sources=(Path("cut.mp4"), Path("continuous.mp4")),
                canonical_spans=spans,
                audio_offsets=(0.0, 0.0),
                source_scene_cut_times=((0.5, 1.5, 2.5), ()),
                config=DirectMemberVisualConfig(beam_width=200),
                descriptor_provider=lambda source_index, _source, _time: _descriptor(
                    0.20 + source_index * 0.01
                ),
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(
            diagnostics["failure"],
            "content_validity_quota_precheck_failed",
        )
        self.assertEqual(
            diagnostics["quota_shortfalls"],
            [{"eligible_source_index": 0, "quota": 2, "valid_span_count": 1}],
        )
        self.assertTrue(
            all(
                "source_native_scene_cut_inside_rendered_span"
                in diagnostics["span_content_validity"][index]["sources"][0][
                    "rejection_reasons"
                ]
                for index in range(3)
            )
        )

    def test_ordinary_warning_jump_is_soft_and_exposed_in_aggregate_maxima(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(4))

        def provider(
            source_index: int,
            _source: Path,
            _source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            return _descriptor(0.20 if source_index == 0 else 0.38)

        selection = select_balanced_direct_member_sources(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.0),
            config=DirectMemberVisualConfig(beam_width=200),
            descriptor_provider=provider,
        )

        self.assertEqual(selection.source_counts, (2, 2))
        aggregate = selection.aggregate_diagnostics()
        self.assertAlmostEqual(
            aggregate["components"]["attention_position"]["max"],
            0.18 / 0.42,
        )
        self.assertEqual(aggregate["warning_boundary_count"], 3)
        self.assertGreaterEqual(
            aggregate["warning_flag_counts"]["attention_position"],
            3,
        )
        self.assertGreater(aggregate["cost"]["max"], 0.45)
        self.assertEqual(aggregate["severe_boundary_count"], 0)

    def test_quota_path_requiring_severe_attention_jump_fails_closed(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(4))

        def provider(
            source_index: int,
            _source: Path,
            _source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            return _descriptor(0.02 if source_index == 0 else 0.74)

        with self.assertRaises(DirectMemberVisualSelectionError) as raised:
            select_balanced_direct_member_sources(
                sources=(Path("zero.mp4"), Path("one.mp4")),
                canonical_spans=spans,
                audio_offsets=(0.0, 0.0),
                config=DirectMemberVisualConfig(beam_width=200),
                descriptor_provider=provider,
            )

        self.assertEqual(
            raised.exception.diagnostics["failure"],
            "validity_quota_path_exhausted",
        )
        self.assertTrue(
            raised.exception.diagnostics["severe_edges_are_ineligible"]
        )
        self.assertEqual(
            raised.exception.diagnostics["severe_edge_pair_count"],
            2,
        )
        self.assertTrue(
            raised.exception.diagnostics["constraint_infeasibility_proven"]
        )
        roots = raised.exception.diagnostics["search"]["exact_lifeline"][
            "root_candidate_diagnostics"
        ]
        self.assertEqual(len(roots), 2)
        self.assertTrue(
            all(
                root["search_outcome"] == "root_predecessor_exhausted"
                for root in roots
            )
        )
        for root in roots:
            self.assertEqual(root["eligible_predecessor_source_indices"], [])
            cross_source = [
                item
                for item in root["predecessor_diagnostics"]
                if item["eligible_source_index"]
                != root["final_candidate_source_index"]
            ]
            self.assertEqual(len(cross_source), 1)
            self.assertIn(
                "severe_boundary_mismatch",
                cross_source[0]["rejection_reasons"],
            )

    def test_persistent_low_quality_is_rejected_but_one_transient_sample_passes(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(6))

        def provider(
            source_index: int,
            _source: Path,
            source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            within_first_span = 0.0 < source_local_seconds < 1.0
            # Source 0 is poor for four of the six first-span samples. Source
            # 1 has only a transient midpoint dip and must remain eligible.
            persistent_low = (
                source_index == 0
                and within_first_span
                and source_local_seconds < 0.70
            )
            transient_low = (
                source_index == 1
                and within_first_span
                and abs(source_local_seconds - 0.50) < 0.05
            )
            return _descriptor(
                0.20 + source_index * 0.02,
                quality_score=0.60 if persistent_low or transient_low else 0.90,
            )

        selection = select_balanced_direct_member_sources(
            sources=tuple(Path(f"{index}.mp4") for index in range(3)),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.0, 0.0),
            config=DirectMemberVisualConfig(beam_width=1_000),
            descriptor_provider=provider,
        )

        rejected = selection.span_validity[0].sources[0]
        transient = selection.span_validity[0].sources[1]
        self.assertFalse(rejected.valid)
        self.assertEqual(rejected.acceptable_quality_sample_count, 2)
        self.assertIn(
            "persistent_low_visual_quality_or_foreground_obstruction",
            rejected.rejection_reasons,
        )
        self.assertTrue(transient.valid)
        self.assertEqual(transient.acceptable_quality_sample_count, 5)
        self.assertNotEqual(selection.source_indices[0], 0)
        payload = selection.to_dict()
        source = payload["span_content_validity"][0]["sources"][0]
        self.assertEqual(source["acceptable_quality_sample_count"], 2)
        self.assertEqual(
            source["quality_sample_scores"],
            [0.6, 0.6, 0.6, 0.6, 0.9, 0.9],
        )

    def test_single_foreground_camera_obstruction_sample_is_hard_rejected(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(6))

        def provider(
            source_index: int,
            _source: Path,
            source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            obstructed_entry = (
                source_index == 0
                and 0.0 < source_local_seconds < 0.10
            )
            return _descriptor(
                0.20 + source_index * 0.02,
                foreground_obstruction_score=(
                    0.94 if obstructed_entry else 0.0
                ),
            )

        selection = select_balanced_direct_member_sources(
            sources=tuple(Path(f"{index}.mp4") for index in range(3)),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.0, 0.0),
            config=DirectMemberVisualConfig(beam_width=1_000),
            descriptor_provider=provider,
        )

        rejected = selection.span_validity[0].sources[0]
        self.assertFalse(rejected.valid)
        self.assertEqual(rejected.foreground_obstruction_sample_count, 1)
        self.assertIn(
            "foreground_camera_equipment_or_subject_obstruction",
            rejected.rejection_reasons,
        )
        self.assertNotEqual(selection.source_indices[0], 0)
        payload = selection.to_dict()["span_content_validity"][0]["sources"][0]
        self.assertEqual(payload["foreground_obstruction_sample_count"], 1)
        self.assertEqual(
            payload["foreground_obstruction_sample_scores"],
            [0.94, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_persistent_entry_overexposure_rejects_but_one_frame_flash_passes(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(6))

        def provider(
            source_index: int,
            _source: Path,
            source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            within_first_span = 0.0 < source_local_seconds < 1.0
            persistent = (
                source_index == 0
                and within_first_span
                and source_local_seconds <= 0.20
            )
            one_frame_flash = (
                source_index == 1
                and within_first_span
                and source_local_seconds < 0.10
            )
            blown_out = persistent or one_frame_flash
            return _descriptor(
                0.20 + source_index * 0.02,
                quality_score=0.73 if blown_out else 0.95,
                gray_mean_score=0.82 if blown_out else 0.48,
                gray_contrast_score=0.30 if blown_out else 0.55,
                highlight_clipping_ratio=0.23 if blown_out else 0.05,
            )

        selection = select_balanced_direct_member_sources(
            sources=tuple(Path(f"{index}.mp4") for index in range(3)),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.0, 0.0),
            config=DirectMemberVisualConfig(beam_width=1_000),
            descriptor_provider=provider,
        )

        persistent = selection.span_validity[0].sources[0]
        flash = selection.span_validity[0].sources[1]
        self.assertFalse(persistent.valid)
        self.assertTrue(persistent.persistent_entry_overexposure_candidate)
        self.assertIn(
            "persistent_entry_stage_light_overexposure",
            persistent.rejection_reasons,
        )
        self.assertEqual(persistent.acceptable_quality_sample_count, 6)
        self.assertTrue(flash.valid)
        self.assertFalse(flash.persistent_entry_overexposure_candidate)
        self.assertNotEqual(selection.source_indices[0], 0)
        payload = persistent.to_dict()
        self.assertEqual(
            payload["highlight_clipping_sample_scores"][:2],
            [0.23, 0.23],
        )
        self.assertEqual(
            selection.aggregate_diagnostics()["content_validity"][
                "rejection_reason_counts"
            ]["persistent_entry_stage_light_overexposure"],
            1,
        )

    def test_terminal_broadcast_composite_rejects_but_exit_motion_blur_passes(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(6))

        def provider(
            source_index: int,
            _source: Path,
            source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            within_first_span = 0.0 < source_local_seconds < 1.0
            local = source_local_seconds if within_first_span else 0.0
            if source_index == 0 and local >= 0.80:
                quality, contrast = 0.30, 0.12
            elif source_index == 0 and local >= 0.70:
                quality, contrast = 0.84, 0.40
            elif source_index == 0 and local >= 0.45:
                quality, contrast = 0.96, 0.50
            elif source_index == 1 and local >= 0.80:
                # A single soft frame is allowed when the preceding contrast
                # stays stable instead of progressively collapsing.
                quality, contrast = 0.30, 0.48
            else:
                quality, contrast = 0.96, 0.50
            return _descriptor(
                0.20 + source_index * 0.02,
                quality_score=quality,
                gray_mean_score=0.50,
                gray_contrast_score=contrast,
            )

        selection = select_balanced_direct_member_sources(
            sources=tuple(Path(f"{index}.mp4") for index in range(3)),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.0, 0.0),
            config=DirectMemberVisualConfig(beam_width=1_000),
            descriptor_provider=provider,
        )

        composite = selection.span_validity[0].sources[0]
        motion_blur = selection.span_validity[0].sources[1]
        self.assertFalse(composite.valid)
        self.assertTrue(composite.terminal_composite_transition_candidate)
        self.assertIn(
            "terminal_broadcast_composite_or_unreported_native_transition",
            composite.rejection_reasons,
        )
        self.assertEqual(composite.acceptable_quality_sample_count, 5)
        self.assertTrue(motion_blur.valid)
        self.assertFalse(motion_blur.terminal_composite_transition_candidate)
        self.assertNotEqual(selection.source_indices[0], 0)
        self.assertEqual(
            selection.aggregate_diagnostics()["content_validity"][
                "rejection_reason_counts"
            ]["terminal_broadcast_composite_or_unreported_native_transition"],
            1,
        )

    def test_early_end_card_source_is_scheduled_only_in_valid_early_spans(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(6))

        def provider(
            source_index: int,
            _source: Path,
            source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            if source_index == 0 and source_local_seconds >= 3.0:
                return _end_card_descriptor()
            return _descriptor(0.20 + source_index * 0.02)

        selection = select_balanced_direct_member_sources(
            sources=tuple(Path(f"{index}.mp4") for index in range(3)),
            canonical_spans=spans,
            audio_offsets=(0.0, 0.0, 0.0),
            config=DirectMemberVisualConfig(beam_width=1_000),
            descriptor_provider=provider,
        )

        source_zero_positions = tuple(
            index
            for index, source_index in enumerate(selection.source_indices)
            if source_index == 0
        )
        self.assertEqual(source_zero_positions, (0, 2))
        self.assertEqual(selection.source_counts, (2, 2, 2))
        late_source_zero = selection.span_validity[4].sources[0]
        self.assertFalse(late_source_zero.valid)
        self.assertIn(
            "static_logo_or_end_card_without_person",
            late_source_zero.rejection_reasons,
        )
        self.assertNotIn(0, selection.source_indices[3:])
        payload = selection.to_dict()
        self.assertGreater(
            payload["aggregate"]["content_validity"]["invalid_candidate_count"],
            0,
        )
        filtered_boundaries = tuple(
            boundary
            for boundary in selection.boundaries
            if boundary.invalid_content_pair_count
        )
        self.assertTrue(filtered_boundaries)
        self.assertTrue(
            all(
                boundary.candidate_pair_count
                == boundary.raw_cross_source_pair_count
                - boundary.invalid_content_pair_count
                - boundary.severe_pair_count
                for boundary in filtered_boundaries
            )
        )

    def test_static_human_performance_hold_is_not_mislabeled_as_end_card(self) -> None:
        held = FrameVisualDescriptor(
            quality_score=1.0,
            composition_score=1.0,
            shot_scale=0.22,
            shot_type=ShotType.MEDIUM,
            face_count=1,
            face_detector_backend="yunet",
            structure_embedding=(0.2, 0.1, 0.8),
            primary_face_bbox=(0.2, 0.2, 0.22, 0.44),
            color_embedding=(0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.5, 0.01),
            pose_embedding=(0.2, 0.1, 0.8),
            pose_confidence=1.0,
            detected_person_count=1,
            person_layout_embedding=(0.2, 0.1, 0.8),
            pose_backend="synthetic_non_identity_pose",
            silhouette_embedding=(0.2, 0.1, 0.8),
            silhouette_bbox_xywh=(0.2, 0.2, 0.22, 0.44),
            silhouette_centroid_xy=(0.31, 0.42),
            silhouette_area_ratio=0.0968,
            silhouette_confidence=1.0,
            silhouette_backend="synthetic_silhouette",
        )

        selection = select_balanced_direct_member_sources(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            canonical_spans=((0.0, 1.0), (1.0, 2.0)),
            audio_offsets=(0.0, 0.0),
            config=DirectMemberVisualConfig(beam_width=200),
            descriptor_provider=lambda *_args: held,
        )

        self.assertTrue(
            all(
                source.valid and source.static_visual_candidate
                for span in selection.span_validity
                for source in span.sources
            )
        )
        self.assertTrue(
            all(
                source.face_evidence_sample_count == 6
                for span in selection.span_validity
                for source in span.sources
            )
        )

    def test_haar_fallback_cannot_satisfy_verified_face_gate(self) -> None:
        with self.assertRaises(DirectMemberVisualSelectionError):
            select_balanced_direct_member_sources(
                sources=(Path("zero.mp4"), Path("one.mp4")),
                canonical_spans=((0.0, 1.0), (1.0, 2.0)),
                audio_offsets=(0.0, 0.0),
                config=DirectMemberVisualConfig(beam_width=200),
                descriptor_provider=lambda *_args: _descriptor(
                    0.2,
                    face_detector_backend="haar",
                ),
            )

    def test_static_nonface_pose_false_positive_is_hard_rejected(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(4))

        def provider(
            source_index: int,
            _source: Path,
            _source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            if source_index == 1:
                return _static_false_human_end_card_descriptor()
            return _descriptor(0.2)

        with self.assertRaises(DirectMemberVisualSelectionError) as raised:
            select_balanced_direct_member_sources(
                sources=(Path("performance.mp4"), Path("end-card.mp4")),
                canonical_spans=spans,
                audio_offsets=(0.0, 0.0),
                config=DirectMemberVisualConfig(beam_width=200),
                descriptor_provider=provider,
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(
            diagnostics["failure"],
            "content_validity_quota_precheck_failed",
        )
        rejected = diagnostics["span_content_validity"][0]["sources"][1]
        self.assertEqual(rejected["face_evidence_sample_count"], 0)
        self.assertEqual(rejected["human_evidence_sample_count"], 6)
        self.assertIn(
            "no_face_performance_evidence",
            rejected["rejection_reasons"],
        )
        self.assertIn(
            "static_nonface_graphic_or_end_card",
            rejected["rejection_reasons"],
        )

    def test_quota_with_no_valid_performance_spans_fails_closed(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(6))

        def provider(
            source_index: int,
            _source: Path,
            _source_local_seconds: float,
        ) -> FrameVisualDescriptor:
            if source_index == 2:
                return _end_card_descriptor()
            return _descriptor(0.20 + source_index * 0.02)

        with self.assertRaises(DirectMemberVisualSelectionError) as raised:
            select_balanced_direct_member_sources(
                sources=tuple(Path(f"{index}.mp4") for index in range(3)),
                canonical_spans=spans,
                audio_offsets=(0.0, 0.0, 0.0),
                config=DirectMemberVisualConfig(beam_width=1_000),
                descriptor_provider=provider,
            )

        self.assertEqual(
            raised.exception.diagnostics["failure"],
            "content_validity_quota_precheck_failed",
        )
        self.assertEqual(
            raised.exception.diagnostics["quota_shortfalls"][0][
                "eligible_source_index"
            ],
            2,
        )

    def test_invalid_configuration_and_noncontiguous_spans_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "beam_width"):
            DirectMemberVisualConfig(beam_width=99)
        with self.assertRaisesRegex(ValueError, "maximum_adaptive_beam_width"):
            DirectMemberVisualConfig(maximum_adaptive_beam_width=99)
        with self.assertRaisesRegex(
            ValueError,
            "maximum_generated_beam_state_count",
        ):
            DirectMemberVisualConfig(maximum_generated_beam_state_count=999)
        with self.assertRaisesRegex(
            ValueError,
            "feasibility_lifeline_state_limit",
        ):
            DirectMemberVisualConfig(feasibility_lifeline_state_limit=0)
        with self.assertRaisesRegex(ValueError, "minimum_face_sample_count"):
            DirectMemberVisualConfig(minimum_face_sample_count=7)
        with self.assertRaisesRegex(
            ValueError,
            "minimum_acceptable_quality_sample_count",
        ):
            DirectMemberVisualConfig(minimum_acceptable_quality_sample_count=0)
        with self.assertRaisesRegex(ValueError, "thresholds"):
            DirectMemberVisualConfig(
                maximum_source_duration_relative_deviation=1.01
            )
        with self.assertRaisesRegex(ValueError, "thresholds"):
            DirectMemberVisualConfig(minimum_acceptable_quality_score=1.01)
        with self.assertRaisesRegex(ValueError, "thresholds"):
            DirectMemberVisualConfig(foreground_obstruction_rejection_score=1.01)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            DirectMemberVisualConfig(foreground_obstruction_rejection_score=0.0)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            select_balanced_direct_member_sources(
                sources=(Path("zero.mp4"), Path("one.mp4")),
                canonical_spans=((0.0, 1.0), (1.1, 2.0)),
                audio_offsets=(0.0, 0.0),
                descriptor_provider=lambda *_args: _descriptor(0.2),
            )


if __name__ == "__main__":
    unittest.main()
