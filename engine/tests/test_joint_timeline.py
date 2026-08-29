from __future__ import annotations

from dataclasses import replace
from itertools import product
import unittest

from crosscut.joint_timeline import (
    AuthorizedBoundaryOption,
    BoundaryMeasurementKey,
    BoundaryOffsetAudit,
    BoundaryOffsetDisposition,
    BoundaryOptionSet,
    BoundaryTransformState,
    JOINT_BOUNDARY_FRAME_OFFSETS,
    SegmentBridgeProof,
    select_joint_timeline,
    select_joint_timeline_candidates,
)
from crosscut.timeline import (
    NoCompleteTimelinePath,
    TimelineCandidate,
    TimelineConfig,
    select_timeline,
    timeline_transition_hard_gates_admit,
)


def candidate(
    segment: int,
    source: str,
    quality: float,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> TimelineCandidate:
    start = float(segment) if start_seconds is None else start_seconds
    end = float(segment + 1) if end_seconds is None else end_seconds
    return TimelineCandidate(
        segment_index=segment,
        source_id=source,
        start_seconds=start,
        end_seconds=end,
        quality_score=quality,
        sync_score=0.0,
        composition_score=0.0,
        boundary_score=0.0,
        candidate_id=f"candidate:{segment}:{source}",
        shot_id=f"shot:{segment}:{source}",
    )


def isolated_score_config() -> TimelineConfig:
    return TimelineConfig(
        quality_weight=1.0,
        sync_weight=0.0,
        composition_weight=0.0,
        boundary_weight=0.0,
        continuity_bonus=0.0,
        cut_penalty=0.0,
        early_cut_penalty=0.0,
        long_hold_penalty=0.0,
        source_hold_penalty=0.0,
        source_switch_bonus=0.0,
        ping_pong_penalty=0.0,
        pose_transition_weight=0.0,
        silhouette_transition_weight=0.0,
        motion_transition_weight=0.0,
        scale_transition_weight=0.0,
        color_transition_weight=0.0,
        face_geometry_transition_weight=0.0,
        focus_position_transition_weight=0.0,
        subject_geometry_transition_weight=0.0,
        person_count_transition_weight=0.0,
        person_layout_transition_weight=0.0,
        formation_transition_weight=0.0,
        shot_type_transition_weight=0.0,
        face_identity_transition_weight=0.0,
        minimum_hold_segments=1,
    )


def transform(
    identity: str,
    *,
    track_id: str = "track:stable",
) -> BoundaryTransformState:
    return BoundaryTransformState(
        transform_identity=identity,
        crop_xywh=(0.1, 0.1, 0.8, 0.8),
        track_id=track_id,
    )


def authorized_option(
    key: BoundaryMeasurementKey,
    offset: int,
    *,
    score_adjustment: float = 0.0,
    scoring_outgoing: TimelineCandidate | None = None,
    scoring_incoming: TimelineCandidate | None = None,
) -> AuthorizedBoundaryOption:
    proof = (
        f"boundary:{key.boundary_index}:"
        f"{key.outgoing.candidate_id}:{key.incoming.candidate_id}:{offset}"
    )
    return AuthorizedBoundaryOption(
        measurement_key=key,
        frame_offset=offset,
        outgoing_transform=transform(f"{proof}:out"),
        incoming_transform=transform(f"{proof}:in"),
        proof_identity=proof,
        score_adjustment=score_adjustment,
        scoring_outgoing=scoring_outgoing,
        scoring_incoming=scoring_incoming,
    )


def option_set(
    key: BoundaryMeasurementKey,
    *,
    authorized: dict[int, float] | None = None,
    incomplete_offsets: set[int] | None = None,
    hard_failure_code: str = "nonwaivable_boundary_failure",
    scoring_candidates_by_offset: dict[
        int,
        tuple[TimelineCandidate, TimelineCandidate],
    ]
    | None = None,
) -> BoundaryOptionSet:
    authorized = dict(authorized or {})
    incomplete_offsets = set(incomplete_offsets or ())
    scoring_candidates_by_offset = dict(scoring_candidates_by_offset or {})
    audits: list[BoundaryOffsetAudit] = []
    for offset in JOINT_BOUNDARY_FRAME_OFFSETS:
        if offset in authorized:
            option = authorized_option(
                key,
                offset,
                score_adjustment=authorized[offset],
                scoring_outgoing=(
                    scoring_candidates_by_offset[offset][0]
                    if offset in scoring_candidates_by_offset
                    else None
                ),
                scoring_incoming=(
                    scoring_candidates_by_offset[offset][1]
                    if offset in scoring_candidates_by_offset
                    else None
                ),
            )
            audits.append(
                BoundaryOffsetAudit(
                    frame_offset=offset,
                    disposition=BoundaryOffsetDisposition.AUTHORIZED,
                    evidence_identity=option.proof_identity,
                    authorized_option=option,
                )
            )
        elif offset in incomplete_offsets:
            audits.append(
                BoundaryOffsetAudit(
                    frame_offset=offset,
                    disposition=BoundaryOffsetDisposition.EVIDENCE_INCOMPLETE,
                    evidence_identity=(
                        f"incomplete:{key.boundary_index}:"
                        f"{key.outgoing.candidate_id}:"
                        f"{key.incoming.candidate_id}:{offset}"
                    ),
                    failure_codes=("descriptor_unavailable",),
                )
            )
        else:
            audits.append(
                BoundaryOffsetAudit(
                    frame_offset=offset,
                    disposition=BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
                    evidence_identity=(
                        f"hard:{key.boundary_index}:"
                        f"{key.outgoing.candidate_id}:"
                        f"{key.incoming.candidate_id}:{offset}"
                    ),
                    failure_codes=(hard_failure_code,),
                )
            )
    return BoundaryOptionSet(key, tuple(audits))


def valid_bridge(
    middle: TimelineCandidate,
    entry: AuthorizedBoundaryOption,
    exit: AuthorizedBoundaryOption,
    duration_frames: float,
    *,
    score_adjustment: float = 0.0,
) -> SegmentBridgeProof:
    return SegmentBridgeProof(
        candidate=middle,
        entry_option=entry,
        exit_option=exit,
        duration_frames=duration_frames,
        proof_identity=(
            f"bridge:{middle.candidate_id}:"
            f"{entry.proof_identity}:{exit.proof_identity}"
        ),
        evidence_complete=True,
        crop_compatible=True,
        track_compatible=True,
        score_adjustment=score_adjustment,
    )


class JointTimelineTests(unittest.TestCase):
    def test_tiny_dynamic_program_matches_brute_force_optimum(self) -> None:
        segments = (
            (candidate(0, "A", 0.80), candidate(0, "B", 0.62)),
            (candidate(1, "A", 0.70), candidate(1, "B", 0.91)),
            (candidate(2, "A", 0.86), candidate(2, "B", 0.74)),
        )

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            pair_penalty = (
                -0.03
                if key.outgoing.source_id == key.incoming.source_id
                else 0.0
            )
            return option_set(
                key,
                authorized={0: pair_penalty, 2: pair_penalty - 0.04},
            )

        def provide_bridge(
            middle: TimelineCandidate,
            entry: AuthorizedBoundaryOption,
            exit: AuthorizedBoundaryOption,
            duration_frames: float,
        ) -> SegmentBridgeProof:
            return valid_bridge(
                middle,
                entry,
                exit,
                duration_frames,
                score_adjustment=(
                    -0.02 if entry.frame_offset != exit.frame_offset else 0.0
                ),
            )

        result = select_joint_timeline(
            segments,
            option_provider=provide_options,
            bridge_provider=provide_bridge,
            fps=30.0,
            config=isolated_score_config(),
        )

        brute_values: list[
            tuple[float, tuple[str, ...], tuple[int, ...]]
        ] = []
        for path in product(*segments):
            option_sets = tuple(
                provide_options(
                    BoundaryMeasurementKey.from_candidates(
                        boundary_index,
                        path[boundary_index],
                        path[boundary_index + 1],
                    )
                )
                for boundary_index in range(2)
            )
            for selected_options in product(
                *(item.authorized_options for item in option_sets)
            ):
                duration = (
                    (path[1].end_seconds - path[1].start_seconds) * 30.0
                    + selected_options[1].frame_offset
                    - selected_options[0].frame_offset
                )
                if duration < 2.0:
                    continue
                bridge = provide_bridge(
                    path[1],
                    selected_options[0],
                    selected_options[1],
                    duration,
                )
                score = (
                    sum(item.quality_score for item in path)
                    + sum(item.score_adjustment for item in selected_options)
                    + bridge.score_adjustment
                )
                brute_values.append(
                    (
                        score,
                        tuple(item.source_id for item in path),
                        tuple(item.frame_offset for item in selected_options),
                    )
                )
        brute_score, brute_sources, brute_offsets = max(brute_values)
        self.assertAlmostEqual(result.total_score, brute_score)
        self.assertEqual(
            tuple(item.source_id for item in result.candidates),
            brute_sources,
        )
        self.assertEqual(
            tuple(item.frame_offset for item in result.boundary_options),
            brute_offsets,
        )
        self.assertEqual(len(result.segment_bridge_proofs), 1)

    def test_offset_zero_can_fail_while_plus_five_is_selected(self) -> None:
        outgoing = replace(
            candidate(0, "A", 0.8),
            exit_shot_scale=0.10,
        )
        incoming = replace(
            candidate(1, "B", 0.8),
            entry_shot_scale=0.90,
        )
        segments = ((outgoing,), (incoming,))
        settings = replace(
            isolated_score_config(),
            maximum_cross_source_scale_delta=0.20,
        )
        self.assertFalse(
            timeline_transition_hard_gates_admit(
                outgoing,
                incoming,
                config=settings,
            )
        )

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            return option_set(
                key,
                authorized={5: -0.02},
                scoring_candidates_by_offset={
                    5: (
                        replace(outgoing, exit_shot_scale=0.50),
                        replace(incoming, entry_shot_scale=0.50),
                    )
                },
            )

        result = select_joint_timeline(
            segments,
            option_provider=provide_options,
            config=settings,
        )
        self.assertEqual(result.boundary_options[0].frame_offset, 5)
        selected_set = result.selected_boundary_option_sets[0]
        self.assertIs(
            selected_set.audit_for(0).disposition,
            BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
        )
        self.assertEqual(len(selected_set.offset_audits), 17)

    def test_nonwaivable_boundary_failure_rejects_higher_score_path(self) -> None:
        segments = (
            (candidate(0, "A", 0.8),),
            (candidate(1, "B", 1.0), candidate(1, "C", 0.4)),
        )

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            if key.incoming.source_id == "B":
                return option_set(
                    key,
                    hard_failure_code="nonwaivable_source_local_camera_cut",
                )
            return option_set(key, authorized={0: 0.0})

        result = select_joint_timeline(
            segments,
            option_provider=provide_options,
            config=isolated_score_config(),
        )
        self.assertEqual(result.candidates[-1].source_id, "C")
        self.assertGreater(
            dict(result.rejection_counts).get("boundary_hard_incompatible", 0),
            0,
        )

    def test_crop_track_bridge_incompatibility_rejects_path(self) -> None:
        segments = (
            (candidate(0, "A", 0.8),),
            (candidate(1, "B", 1.0), candidate(1, "C", 0.45)),
            (candidate(2, "D", 0.8),),
        )

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            return option_set(key, authorized={0: 0.0})

        def provide_bridge(
            middle: TimelineCandidate,
            entry: AuthorizedBoundaryOption,
            exit: AuthorizedBoundaryOption,
            duration_frames: float,
        ) -> SegmentBridgeProof:
            if middle.source_id == "B":
                return SegmentBridgeProof(
                    candidate=middle,
                    entry_option=entry,
                    exit_option=exit,
                    duration_frames=duration_frames,
                    proof_identity=f"bridge:invalid:{middle.candidate_id}",
                    evidence_complete=True,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("crop_bridge_incompatible",),
                )
            return valid_bridge(middle, entry, exit, duration_frames)

        result = select_joint_timeline(
            segments,
            option_provider=provide_options,
            bridge_provider=provide_bridge,
            fps=30.0,
            config=isolated_score_config(),
        )
        self.assertEqual(result.candidates[1].source_id, "C")
        self.assertGreater(
            dict(result.rejection_counts).get("bridge_hard_incompatible", 0),
            0,
        )

    def test_adjacent_offsets_cannot_collapse_middle_segment_below_two_frames(
        self,
    ) -> None:
        segments = (
            (candidate(0, "A", 0.8, start_seconds=0.0, end_seconds=1.0),),
            (candidate(1, "B", 0.8, start_seconds=1.0, end_seconds=1.1),),
            (candidate(2, "C", 0.8, start_seconds=1.1, end_seconds=2.1),),
        )

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            return option_set(key, authorized={0: 0.0})

        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_timeline(
                segments,
                option_provider=provide_options,
                bridge_provider=valid_bridge,
                fps=10.0,
                config=isolated_score_config(),
            )
        self.assertEqual(
            caught.exception.reason_code,
            "joint_timeline_hard_incompatible",
        )
        self.assertGreater(
            caught.exception.rejection_counts.get(
                "segment_duration_below_two_frames",
                0,
            ),
            0,
        )

    def test_incomplete_boundary_evidence_is_not_classified_as_hard(self) -> None:
        segments = ((candidate(0, "A", 0.8),), (candidate(1, "B", 0.8),))

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            return option_set(
                key,
                incomplete_offsets=set(JOINT_BOUNDARY_FRAME_OFFSETS),
            )

        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_timeline(
                segments,
                option_provider=provide_options,
                config=isolated_score_config(),
            )
        self.assertEqual(
            caught.exception.reason_code,
            "joint_timeline_evidence_incomplete",
        )
        self.assertEqual(
            caught.exception.rejection_counts,
            {"boundary_evidence_incomplete": 1},
        )

    def test_option_provider_is_called_once_per_exact_measurement_key(self) -> None:
        segments = (
            (candidate(0, "A", 0.8),),
            (candidate(1, "B", 0.8),),
            (candidate(2, "C", 0.8),),
        )
        call_counts: dict[BoundaryMeasurementKey, int] = {}

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            call_counts[key] = call_counts.get(key, 0) + 1
            return option_set(
                key,
                authorized=(
                    {0: 0.0, 1: -0.01}
                    if key.boundary_index == 0
                    else {0: 0.0}
                ),
            )

        result = select_joint_timeline(
            segments,
            option_provider=provide_options,
            bridge_provider=valid_bridge,
            fps=30.0,
            config=isolated_score_config(),
        )
        self.assertEqual(set(call_counts.values()), {1})
        self.assertEqual(len(call_counts), 2)
        self.assertEqual(result.option_provider_call_count, 2)
        self.assertEqual(len(result.measured_boundary_option_sets), 2)

    def test_preference_adjustment_and_switch_scoring_matches_timeline(self) -> None:
        segments = (
            (candidate(0, "A", 0.83), candidate(0, "B", 0.76)),
            (candidate(1, "A", 0.78), candidate(1, "B", 0.87)),
            (candidate(2, "A", 0.72), candidate(2, "B", 0.89)),
        )
        settings = replace(
            isolated_score_config(),
            preferred_source_bonus=0.14,
            source_switch_bonus=0.05,
            ping_pong_penalty=0.08,
        )
        preferred_sources = ("B", "A", "B")
        emission_adjustments = {
            (0, "A"): -0.03,
            (1, "B"): -0.02,
            (2, "B"): -0.01,
        }
        switch_bonuses = {1: 0.11, 2: 0.07}

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            return option_set(key, authorized={0: 0.0})

        ordinary = select_timeline(
            segments,
            config=settings,
            preferred_source_ids=preferred_sources,
            emission_adjustments=emission_adjustments,
            source_switch_bonus_by_segment=switch_bonuses,
        )
        joint = select_joint_timeline(
            segments,
            option_provider=provide_options,
            bridge_provider=valid_bridge,
            fps=30.0,
            config=settings,
            preferred_source_ids=preferred_sources,
            emission_adjustments=emission_adjustments,
            source_switch_bonus_by_segment=switch_bonuses,
        )

        self.assertEqual(
            tuple(item.candidate_id for item in joint.candidates),
            tuple(item.candidate_id for item in ordinary.candidates),
        )
        self.assertAlmostEqual(joint.total_score, ordinary.total_score)
        self.assertEqual(len(joint.choices), len(ordinary.choices))
        for joint_choice, ordinary_choice in zip(
            joint.choices,
            ordinary.choices,
            strict=True,
        ):
            self.assertAlmostEqual(
                joint_choice.emission_score,
                ordinary_choice.emission_score,
            )
            self.assertAlmostEqual(
                joint_choice.transition_score,
                ordinary_choice.transition_score,
            )
            self.assertAlmostEqual(
                joint_choice.cumulative_score,
                ordinary_choice.cumulative_score,
            )

        with self.assertRaisesRegex(ValueError, "non-positive"):
            select_joint_timeline(
                segments,
                option_provider=provide_options,
                bridge_provider=valid_bridge,
                emission_adjustments={(0, "A"): 0.01},
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            select_joint_timeline(
                segments,
                option_provider=provide_options,
                bridge_provider=valid_bridge,
                source_switch_bonus_by_segment={1: -0.01},
            )

    def test_ranked_candidates_match_brute_force_top_eight(self) -> None:
        segments = (
            (candidate(0, "A", 1.00), candidate(0, "B", 0.53)),
            (candidate(1, "C", 0.31), candidate(1, "D", 0.17)),
            (candidate(2, "E", 0.09), candidate(2, "F", 0.04)),
        )
        call_counts: dict[BoundaryMeasurementKey, int] = {}

        def measured_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            source_code = sum(
                ord(value)
                for value in (
                    key.outgoing.source_id + key.incoming.source_id
                )
            )
            pair_penalty = -(source_code % 17) * 0.0001
            shifted_penalty = pair_penalty - 0.011 * (
                key.boundary_index + 1
            )
            return option_set(
                key,
                authorized={0: pair_penalty, 3: shifted_penalty},
            )

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            call_counts[key] = call_counts.get(key, 0) + 1
            return measured_options(key)

        def provide_bridge(
            middle: TimelineCandidate,
            entry: AuthorizedBoundaryOption,
            exit: AuthorizedBoundaryOption,
            duration_frames: float,
        ) -> SegmentBridgeProof:
            return valid_bridge(
                middle,
                entry,
                exit,
                duration_frames,
                score_adjustment=(
                    -0.004
                    if entry.frame_offset != exit.frame_offset
                    else 0.0
                ),
            )

        ranked = select_joint_timeline_candidates(
            segments,
            option_provider=provide_options,
            bridge_provider=provide_bridge,
            fps=30.0,
            config=isolated_score_config(),
            maximum_results=8,
        )

        brute_values: list[
            tuple[
                float,
                tuple[str, ...],
                tuple[tuple[str, int], ...],
            ]
        ] = []
        for path in product(*segments):
            option_sets = tuple(
                measured_options(
                    BoundaryMeasurementKey.from_candidates(
                        boundary_index,
                        path[boundary_index],
                        path[boundary_index + 1],
                    )
                )
                for boundary_index in range(2)
            )
            for options in product(
                *(item.authorized_options for item in option_sets)
            ):
                duration_frames = (
                    (path[1].end_seconds - path[1].start_seconds) * 30.0
                    + options[1].frame_offset
                    - options[0].frame_offset
                )
                bridge = provide_bridge(
                    path[1],
                    options[0],
                    options[1],
                    duration_frames,
                )
                brute_values.append(
                    (
                        sum(item.quality_score for item in path)
                        + sum(item.score_adjustment for item in options)
                        + bridge.score_adjustment,
                        tuple(item.candidate_id for item in path),
                        tuple(
                            (item.proof_identity, item.frame_offset)
                            for item in options
                        ),
                    )
                )
        brute_values.sort(
            key=lambda item: (-item[0], item[1], item[2])
        )
        expected = brute_values[:8]
        actual = tuple(
            (
                item.total_score,
                tuple(candidate.candidate_id for candidate in item.candidates),
                tuple(
                    (option.proof_identity, option.frame_offset)
                    for option in item.boundary_options
                ),
            )
            for item in ranked
        )

        self.assertEqual(len(actual), 8)
        self.assertEqual(len(set((item[1], item[2]) for item in actual)), 8)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            self.assertAlmostEqual(actual_item[0], expected_item[0])
            self.assertEqual(actual_item[1:], expected_item[1:])
        self.assertEqual(set(call_counts.values()), {1})
        self.assertEqual(len(call_counts), 8)
        self.assertTrue(
            all(item.option_provider_call_count == 8 for item in ranked)
        )

    def test_ranked_candidate_limit_is_at_most_eight(self) -> None:
        segments = ((candidate(0, "A", 0.8),),)

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            return option_set(key, authorized={0: 0.0})

        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            select_joint_timeline_candidates(
                segments,
                option_provider=provide_options,
                maximum_results=9,
            )

    def test_endpoint_segments_must_remain_at_least_two_frames(self) -> None:
        segments = (
            (
                candidate(
                    0,
                    "A",
                    0.8,
                    start_seconds=0.0,
                    end_seconds=0.2,
                ),
            ),
            (
                candidate(
                    1,
                    "B",
                    0.8,
                    start_seconds=0.2,
                    end_seconds=0.4,
                ),
            ),
        )

        for unsafe_offset in (-8, 8):
            with self.subTest(unsafe_offset=unsafe_offset):
                with self.assertRaises(NoCompleteTimelinePath):
                    select_joint_timeline(
                        segments,
                        option_provider=lambda key, offset=unsafe_offset: (
                            option_set(key, authorized={offset: 0.0})
                        ),
                        fps=30.0,
                        config=isolated_score_config(),
                    )

        result = select_joint_timeline(
            segments,
            option_provider=lambda key: option_set(
                key,
                authorized={-8: 0.0, 0: 0.0, 8: 0.0},
            ),
            fps=30.0,
            config=isolated_score_config(),
        )
        self.assertEqual(result.boundary_options[0].frame_offset, 0)

    def test_terminal_reason_uses_only_the_exhausted_layer(self) -> None:
        segments = (
            (candidate(0, "A", 0.8),),
            (candidate(1, "B", 0.8), candidate(1, "C", 0.7)),
            (candidate(2, "D", 0.8),),
        )

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            if key.boundary_index == 0 and key.incoming.source_id == "B":
                return option_set(key, incomplete_offsets={0})
            if key.boundary_index == 0:
                return option_set(key, authorized={0: 0.0})
            return option_set(key)

        with self.assertRaises(NoCompleteTimelinePath) as raised:
            select_joint_timeline(
                segments,
                option_provider=provide_options,
                bridge_provider=valid_bridge,
                fps=30.0,
                config=isolated_score_config(),
            )

        self.assertEqual(
            raised.exception.reason_code,
            "joint_timeline_hard_incompatible",
        )
        self.assertEqual(
            raised.exception.rejection_counts,
            {"boundary_hard_incompatible": 1},
        )

    def test_identity_provider_exact_tie_matches_existing_selector(self) -> None:
        segments = tuple(
            (
                candidate(segment, "B", 0.8),
                candidate(segment, "A", 0.8),
            )
            for segment in range(2)
        )

        ordinary = select_timeline(
            segments,
            config=isolated_score_config(),
        )
        joint = select_joint_timeline(
            segments,
            option_provider=lambda key: option_set(
                key,
                authorized={0: 0.0},
            ),
            fps=30.0,
            config=isolated_score_config(),
        )

        self.assertEqual(
            tuple(item.candidate_id for item in joint.candidates),
            tuple(item.candidate_id for item in ordinary.candidates),
        )
        self.assertAlmostEqual(joint.total_score, ordinary.total_score)


if __name__ == "__main__":
    unittest.main()
