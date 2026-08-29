from __future__ import annotations

from dataclasses import replace
from itertools import product
import sys
import unittest

from crosscut.joint_timeline import (
    AuthorizedBoundaryOption,
    BoundaryMeasurementKey,
    BoundaryOffsetAudit,
    BoundaryOffsetDisposition,
    BoundaryOptionEvidenceScope,
    BoundaryOptionSet,
    BoundaryTransformState,
    JOINT_BOUNDARY_FRAME_OFFSETS,
    JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES,
    JointTimelineProgress,
    JointTimelineSharedBridgeEvidenceCache,
    SegmentBridgeDisposition,
    SegmentBridgeDispositionRequest,
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
from crosscut.timeline_no_good import (
    TimelineNoGoodEdgeKey,
    TimelineNoGoodSet,
    build_timeline_no_good_edge_key,
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


def selection_scalar_option_set(
    key: BoundaryMeasurementKey,
    *,
    authorized: dict[int, float],
) -> BoundaryOptionSet:
    return replace(
        option_set(key, authorized=authorized),
        evidence_scope=BoundaryOptionEvidenceScope.SELECTION_SCALAR,
    )


def exact_rehydrated_option_set(
    provisional: AuthorizedBoundaryOption,
    provisional_set: BoundaryOptionSet,
    *,
    score_adjustment: float | None = None,
) -> tuple[AuthorizedBoundaryOption, BoundaryOptionSet]:
    exact = replace(
        provisional,
        proof_identity=provisional.proof_identity + ":exact",
        score_adjustment=(
            provisional.score_adjustment
            if score_adjustment is None
            else score_adjustment
        ),
    )
    audits = tuple(
        replace(
            audit,
            evidence_identity=exact.proof_identity,
            authorized_option=exact,
        )
        if audit.frame_offset == provisional.frame_offset
        else audit
        for audit in provisional_set.offset_audits
    )
    return exact, BoundaryOptionSet(provisional.measurement_key, audits)


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


class BatchedBridgeProvider:
    def __init__(self, *, malformed: str | None = None) -> None:
        self.malformed = malformed
        self.exact_requests: list[SegmentBridgeDispositionRequest] = []
        self.batch_requests: list[
            tuple[SegmentBridgeDispositionRequest, ...]
        ] = []

    @staticmethod
    def _scalars(
        request: SegmentBridgeDispositionRequest,
    ) -> tuple[bool, bool, bool, tuple[str, ...], float]:
        rejected = request.candidate.source_id.startswith("reject")
        return (
            True,
            not rejected,
            True,
            ("synthetic_crop_incompatible",) if rejected else (),
            (
                -0.01
                if request.entry_option.frame_offset
                != request.exit_option.frame_offset
                else 0.0
            ),
        )

    def bridge_provider(
        self,
        candidate_: TimelineCandidate,
        entry: AuthorizedBoundaryOption,
        exit: AuthorizedBoundaryOption,
        duration_frames: float,
    ) -> SegmentBridgeProof:
        request = SegmentBridgeDispositionRequest(
            candidate_,
            entry,
            exit,
            duration_frames,
        )
        self.exact_requests.append(request)
        complete, crop, track, codes, adjustment = self._scalars(request)
        return SegmentBridgeProof(
            candidate=candidate_,
            entry_option=entry,
            exit_option=exit,
            duration_frames=duration_frames,
            proof_identity=(
                f"batch-exact:{candidate_.candidate_id}:"
                f"{entry.proof_identity}:{exit.proof_identity}"
            ),
            evidence_complete=complete,
            crop_compatible=crop,
            track_compatible=track,
            failure_codes=codes,
            score_adjustment=adjustment,
        )

    def bridge_disposition_batch(
        self,
        requests: tuple[SegmentBridgeDispositionRequest, ...],
    ) -> tuple[SegmentBridgeDisposition, ...]:
        self.batch_requests.append(requests)
        if self.malformed == "exception":
            raise RuntimeError("synthetic batch failure")
        dispositions = tuple(
            SegmentBridgeDisposition(
                request=request,
                evidence_complete=values[0],
                crop_compatible=values[1],
                track_compatible=values[2],
                failure_codes=values[3],
                score_adjustment=values[4],
            )
            for request in requests
            for values in (self._scalars(request),)
        )
        if self.malformed == "missing":
            return dispositions[:-1]
        if self.malformed == "extra":
            return dispositions + (dispositions[-1],)
        if self.malformed == "wrong_request":
            wrong_request = replace(
                requests[0],
                duration_frames=requests[0].duration_frames + 1.0,
            )
            return (
                replace(dispositions[0], request=wrong_request),
            ) + dispositions[1:]
        if self.malformed == "wrong_type":
            return (object(),) + dispositions[1:]  # type: ignore[return-value]
        if self.malformed == "list":
            return list(dispositions)  # type: ignore[return-value]
        return dispositions


def three_segment_fixture() -> tuple[tuple[TimelineCandidate, ...], ...]:
    return tuple((candidate(index, source, 0.8),) for index, source in enumerate("ABC"))


def zero_offset_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
    return option_set(key, authorized={0: 0.0})


class JointTimelineTests(unittest.TestCase):
    def test_selected_scalar_rehydration_rejects_once_and_reselects_exact(
        self,
    ) -> None:
        preferred = candidate(0, "preferred", 1.0)
        fallback = candidate(0, "fallback", 0.5)
        incoming = candidate(1, "incoming", 0.5)
        provisional_sets: dict[BoundaryMeasurementKey, BoundaryOptionSet] = {}
        rehydrated_keys: list[BoundaryMeasurementKey] = []

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            measured = selection_scalar_option_set(
                key,
                authorized={0: 0.0},
            )
            provisional_sets[key] = measured
            return measured

        def exact_option_set(
            provisional: AuthorizedBoundaryOption,
        ) -> tuple[AuthorizedBoundaryOption, BoundaryOptionSet]:
            return exact_rehydrated_option_set(
                provisional,
                provisional_sets[provisional.measurement_key],
            )

        def rehydrate(
            provisional: AuthorizedBoundaryOption,
        ) -> tuple[AuthorizedBoundaryOption, BoundaryOptionSet] | None:
            rehydrated_keys.append(provisional.measurement_key)
            if provisional.measurement_key.outgoing is preferred:
                return None
            return exact_option_set(provisional)

        result = select_joint_timeline(
            ((preferred, fallback), (incoming,)),
            option_provider=provide_options,
            option_rehydrator=rehydrate,
            fps=30.0,
            config=isolated_score_config(),
        )

        self.assertIs(result.candidates[0], fallback)
        self.assertTrue(result.boundary_options[0].proof_identity.endswith(":exact"))
        self.assertIn(
            result.boundary_options[0],
            result.selected_boundary_option_sets[0].authorized_options,
        )
        self.assertIs(
            result.selected_boundary_option_sets[0].evidence_scope,
            BoundaryOptionEvidenceScope.EXACT,
        )
        self.assertTrue(
            all(
                option_set_.evidence_scope
                is BoundaryOptionEvidenceScope.SELECTION_SCALAR
                for option_set_ in result.measured_boundary_option_sets
            )
        )
        self.assertEqual(
            tuple(key.outgoing for key in rehydrated_keys),
            (preferred, fallback),
        )
        self.assertEqual(
            dict(result.rejection_counts)[
                "boundary_selection_exact_rehydration_rejected"
            ],
            1,
        )

    def test_selection_scalar_options_require_exact_rehydrator(self) -> None:
        outgoing = candidate(0, "outgoing", 0.8)
        incoming = candidate(1, "incoming", 0.8)

        with self.assertRaisesRegex(
            RuntimeError,
            "requires an exact option_rehydrator",
        ):
            select_joint_timeline(
                ((outgoing,), (incoming,)),
                option_provider=lambda key: selection_scalar_option_set(
                    key,
                    authorized={0: 0.0},
                ),
                fps=30.0,
                config=isolated_score_config(),
            )

    def test_exact_rehydration_exhaustion_reports_hard_incompatibility(
        self,
    ) -> None:
        outgoing = candidate(0, "outgoing", 0.8)
        incoming = candidate(1, "incoming", 0.8)

        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_timeline(
                ((outgoing,), (incoming,)),
                option_provider=lambda key: selection_scalar_option_set(
                    key,
                    authorized={0: 0.0},
                ),
                option_rehydrator=lambda _option: None,
                fps=30.0,
                config=isolated_score_config(),
            )

        self.assertEqual(
            caught.exception.reason_code,
            "joint_timeline_hard_incompatible",
        )
        self.assertEqual(
            caught.exception.rejection_counts[
                "boundary_selection_exact_rehydration_rejected"
            ],
            1,
        )

    def test_exact_scalar_parity_divergence_becomes_a_no_good(self) -> None:
        preferred = candidate(0, "preferred", 1.0)
        fallback = candidate(0, "fallback", 0.5)
        incoming = candidate(1, "incoming", 0.5)
        measured: dict[BoundaryMeasurementKey, BoundaryOptionSet] = {}

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            result = selection_scalar_option_set(
                key,
                authorized={0: 0.0},
            )
            measured[key] = result
            return result

        def rehydrate(
            provisional: AuthorizedBoundaryOption,
        ) -> tuple[AuthorizedBoundaryOption, BoundaryOptionSet]:
            return exact_rehydrated_option_set(
                provisional,
                measured[provisional.measurement_key],
                score_adjustment=(
                    -0.25
                    if provisional.measurement_key.outgoing is preferred
                    else None
                ),
            )

        result = select_joint_timeline(
            ((preferred, fallback), (incoming,)),
            option_provider=provide_options,
            option_rehydrator=rehydrate,
            fps=30.0,
            config=isolated_score_config(),
        )

        self.assertIs(result.candidates[0], fallback)
        self.assertEqual(
            dict(result.rejection_counts)[
                "boundary_selection_exact_rehydration_rejected"
            ],
            1,
        )

    def test_retry_progress_and_bridge_batch_audit_remain_monotonic(
        self,
    ) -> None:
        preferred = candidate(0, "preferred", 1.0)
        fallback = candidate(0, "fallback", 0.5)
        middle = candidate(1, "middle", 0.5)
        last = candidate(2, "last", 0.5)
        measured: dict[BoundaryMeasurementKey, BoundaryOptionSet] = {}
        progress: list[JointTimelineProgress] = []

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            result = selection_scalar_option_set(
                key,
                authorized={0: 0.0},
            )
            measured[key] = result
            return result

        def rehydrate(
            provisional: AuthorizedBoundaryOption,
        ) -> tuple[AuthorizedBoundaryOption, BoundaryOptionSet] | None:
            if (
                provisional.measurement_key.boundary_index == 0
                and provisional.measurement_key.outgoing is preferred
            ):
                return None
            return exact_rehydrated_option_set(
                provisional,
                measured[provisional.measurement_key],
            )

        class BridgeProvider:
            def __init__(self) -> None:
                self.batch_call_count = 0
                self.exact_call_count = 0

            def bridge(
                self,
                bridge_candidate: TimelineCandidate,
                entry: AuthorizedBoundaryOption,
                exit_: AuthorizedBoundaryOption,
                duration_frames: float,
            ) -> SegmentBridgeProof:
                self.exact_call_count += 1
                return valid_bridge(
                    bridge_candidate,
                    entry,
                    exit_,
                    duration_frames,
                )

            def batch(
                self,
                requests: tuple[SegmentBridgeDispositionRequest, ...],
            ) -> tuple[SegmentBridgeDisposition, ...]:
                self.batch_call_count += 1
                return tuple(
                    SegmentBridgeDisposition(
                        request=request,
                        evidence_complete=True,
                        crop_compatible=True,
                        track_compatible=True,
                        failure_codes=(),
                    )
                    for request in requests
                )

        bridges = BridgeProvider()
        result = select_joint_timeline(
            ((preferred, fallback), (middle,), (last,)),
            option_provider=provide_options,
            option_rehydrator=rehydrate,
            bridge_provider=bridges.bridge,
            bridge_disposition_batch_provider=bridges.batch,
            fps=30.0,
            config=isolated_score_config(),
            progress_callback=progress.append,
            shared_bridge_evidence_cache=(
                JointTimelineSharedBridgeEvidenceCache(maximum_entries=128)
            ),
        )

        self.assertIs(result.candidates[0], fallback)
        self.assertEqual(bridges.batch_call_count, 1)
        self.assertEqual(bridges.exact_call_count, 1)
        self.assertEqual(result.bridge_disposition_batch_call_count, 1)
        self.assertGreater(result.bridge_disposition_batch_request_count, 0)
        self.assertEqual(
            [item.completed_boundary_layers for item in progress],
            [0, 1, 2, 2],
        )
        self.assertEqual(
            [item.phase for item in progress],
            [
                "selection_start",
                "boundary_layer_complete",
                "boundary_layer_complete",
                "selection_complete",
            ],
        )
        self.assertEqual(
            progress[-1].bridge_disposition_batch_call_count,
            1,
        )
        self.assertEqual(
            dict(result.rejection_counts)[
                "boundary_selection_exact_rehydration_rejected"
            ],
            1,
        )

    def test_exact_no_good_retry_loop_exceeds_python_recursion_depth(
        self,
    ) -> None:
        invalid_count = sys.getrecursionlimit() + 5
        outgoing = tuple(
            candidate(
                0,
                f"source-{index:04d}",
                1.0 - index * 1e-7,
            )
            for index in range(invalid_count + 1)
        )
        incoming = candidate(1, "incoming", 0.5)
        final_outgoing = outgoing[-1]
        measured: dict[BoundaryMeasurementKey, BoundaryOptionSet] = {}
        option_provider_calls = 0
        rehydration_count = 0

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            nonlocal option_provider_calls
            option_provider_calls += 1
            result = selection_scalar_option_set(
                key,
                authorized={0: 0.0},
            )
            measured[key] = result
            return result

        def rehydrate(
            provisional: AuthorizedBoundaryOption,
        ) -> tuple[AuthorizedBoundaryOption, BoundaryOptionSet] | None:
            nonlocal rehydration_count
            rehydration_count += 1
            if provisional.measurement_key.outgoing is not final_outgoing:
                return None
            return exact_rehydrated_option_set(
                provisional,
                measured[provisional.measurement_key],
            )

        result = select_joint_timeline(
            (outgoing, (incoming,)),
            option_provider=provide_options,
            option_rehydrator=rehydrate,
            fps=30.0,
            config=isolated_score_config(),
        )

        self.assertIs(result.candidates[0], final_outgoing)
        self.assertEqual(option_provider_calls, len(outgoing))
        self.assertEqual(result.option_provider_call_count, len(outgoing))
        self.assertEqual(rehydration_count, len(outgoing))
        self.assertEqual(
            dict(result.rejection_counts)[
                "boundary_selection_exact_rehydration_rejected"
            ],
            invalid_count,
        )

    def test_selection_scalar_top_k_score_tie_fails_closed(self) -> None:
        first = candidate(0, "same", 0.8)
        second = replace(
            first,
            candidate_id="candidate:0:same:duplicate",
        )
        incoming = candidate(1, "incoming", 0.8)

        with self.assertRaisesRegex(
            RuntimeError,
            "unresolved exact top-K tie",
        ):
            select_joint_timeline(
                ((first, second), (incoming,)),
                option_provider=lambda key: selection_scalar_option_set(
                    key,
                    authorized={0: 0.0},
                ),
                option_rehydrator=lambda option: (
                    option,
                    BoundaryOptionSet(
                        option.measurement_key,
                        option_set(
                            option.measurement_key,
                            authorized={0: 0.0},
                        ).offset_audits,
                    ),
                ),
                fps=30.0,
                config=isolated_score_config(),
            )

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

    def test_bridge_disposition_batch_matches_legacy_order_scores_and_rejections(
        self,
    ) -> None:
        segments = (
            (candidate(0, "A", 0.82), candidate(0, "B", 0.77)),
            (
                candidate(1, "reject-high", 0.99),
                candidate(1, "C", 0.81),
            ),
            (candidate(2, "D", 0.86), candidate(2, "E", 0.75)),
            (candidate(3, "F", 0.84), candidate(3, "G", 0.73)),
        )

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            return option_set(key, authorized={0: 0.0, 2: -0.02})

        legacy_provider = BatchedBridgeProvider()
        legacy = select_joint_timeline_candidates(
            segments,
            option_provider=provide_options,
            bridge_provider=legacy_provider.bridge_provider,
            fps=30.0,
            config=isolated_score_config(),
            maximum_results=8,
        )
        batch_provider = BatchedBridgeProvider()
        batched = select_joint_timeline_candidates(
            segments,
            option_provider=provide_options,
            bridge_provider=batch_provider.bridge_provider,
            bridge_disposition_batch_provider=(
                batch_provider.bridge_disposition_batch
            ),
            fps=30.0,
            config=isolated_score_config(),
            maximum_results=8,
        )

        self.assertEqual(len(batch_provider.batch_requests), 2)
        self.assertTrue(all(batch_provider.batch_requests))
        flattened_batch_requests = tuple(
            request
            for requests in batch_provider.batch_requests
            for request in requests
        )
        self.assertEqual(
            flattened_batch_requests,
            tuple(
                legacy_provider.exact_requests[: len(flattened_batch_requests)]
            ),
        )
        self.assertEqual(len(legacy), len(batched))
        for legacy_item, batch_item in zip(legacy, batched, strict=True):
            self.assertEqual(batch_item.timeline, legacy_item.timeline)
            self.assertEqual(
                batch_item.boundary_options,
                legacy_item.boundary_options,
            )
            self.assertEqual(
                batch_item.segment_bridge_proofs,
                legacy_item.segment_bridge_proofs,
            )
            self.assertEqual(
                batch_item.rejection_counts,
                legacy_item.rejection_counts,
            )
            self.assertEqual(batch_item.bridge_disposition_batch_call_count, 2)
            self.assertEqual(
                batch_item.bridge_disposition_batch_request_count,
                sum(len(requests) for requests in batch_provider.batch_requests),
            )
        expected_exact_requests = {
            SegmentBridgeDispositionRequest(
                proof.candidate,
                proof.entry_option,
                proof.exit_option,
                proof.duration_frames,
            )
            for item in batched
            for proof in item.segment_bridge_proofs
        }
        self.assertEqual(
            set(batch_provider.exact_requests),
            expected_exact_requests,
        )

        repeat_provider = BatchedBridgeProvider()
        select_joint_timeline_candidates(
            segments,
            option_provider=provide_options,
            bridge_provider=repeat_provider.bridge_provider,
            bridge_disposition_batch_provider=(
                repeat_provider.bridge_disposition_batch
            ),
            fps=30.0,
            config=isolated_score_config(),
            maximum_results=8,
        )
        request_signature = lambda request: (
            request.candidate.candidate_id,
            request.entry_option.proof_identity,
            request.exit_option.proof_identity,
            request.duration_frames,
        )
        self.assertEqual(
            tuple(
                tuple(request_signature(request) for request in batch)
                for batch in batch_provider.batch_requests
            ),
            tuple(
                tuple(request_signature(request) for request in batch)
                for batch in repeat_provider.batch_requests
            ),
        )

    def test_batch_dedupes_semantic_aliases_but_revalidates_exact_backtraces(
        self,
    ) -> None:
        first_segment = tuple(
            replace(
                candidate(0, "A", 0.8),
                candidate_id=f"candidate:0:A:alias:{index}",
                shot_id=f"shot:0:A:alias:{index}",
            )
            for index in range(8)
        )
        segments = (
            first_segment,
            (candidate(1, "B", 0.8),),
            (candidate(2, "C", 0.8),),
        )
        provider = BatchedBridgeProvider()
        ranked = select_joint_timeline_candidates(
            segments,
            option_provider=zero_offset_options,
            bridge_provider=provider.bridge_provider,
            bridge_disposition_batch_provider=provider.bridge_disposition_batch,
            fps=30.0,
            config=isolated_score_config(),
            maximum_results=8,
        )

        self.assertEqual(len(ranked), 8)
        self.assertEqual(len(provider.batch_requests), 1)
        self.assertEqual(len(provider.batch_requests[0]), 1)
        self.assertEqual(len(provider.exact_requests), 8)
        self.assertEqual(
            len({request.entry_option for request in provider.exact_requests}),
            8,
        )
        self.assertTrue(
            all(
                item.segment_bridge_proofs[0].entry_option
                is item.boundary_options[0]
                for item in ranked
            )
        )

    def test_batch_shared_cache_suppresses_second_and_third_profile_batches(
        self,
    ) -> None:
        segments = three_segment_fixture()
        provider = BatchedBridgeProvider()
        cache = JointTimelineSharedBridgeEvidenceCache(maximum_entries=1)

        results = tuple(
            select_joint_timeline(
                segments,
                option_provider=zero_offset_options,
                bridge_provider=provider.bridge_provider,
                bridge_disposition_batch_provider=(
                    provider.bridge_disposition_batch
                ),
                shared_bridge_evidence_cache=cache,
                fps=30.0,
                config=isolated_score_config(),
            )
            for _ in range(3)
        )

        self.assertEqual(len(provider.batch_requests), 1)
        self.assertEqual(len(provider.exact_requests), 3)
        self.assertEqual(
            [item.bridge_disposition_batch_call_count for item in results],
            [1, 0, 0],
        )
        self.assertEqual(
            [item.bridge_disposition_batch_request_count for item in results],
            [1, 0, 0],
        )
        audit = cache.to_dict()
        self.assertEqual(
            (
                audit["entry_count"],
                audit["hit_count"],
                audit["miss_count"],
                audit["admission_count"],
            ),
            (1, 2, 1, 1),
        )

    def test_batch_shared_cache_cap_preserves_first_seen_admission_order(
        self,
    ) -> None:
        segments = (
            (candidate(0, "A", 0.8),),
            (candidate(1, "B", 0.8), candidate(1, "C", 0.8)),
            (candidate(2, "D", 0.8),),
        )
        provider = BatchedBridgeProvider()
        cache = JointTimelineSharedBridgeEvidenceCache(maximum_entries=1)
        select_joint_timeline(
            segments,
            option_provider=zero_offset_options,
            bridge_provider=provider.bridge_provider,
            bridge_disposition_batch_provider=provider.bridge_disposition_batch,
            shared_bridge_evidence_cache=cache,
            fps=30.0,
            config=isolated_score_config(),
        )
        first_batch = provider.batch_requests[0]
        self.assertEqual(len(first_batch), 2)

        select_joint_timeline(
            segments,
            option_provider=zero_offset_options,
            bridge_provider=provider.bridge_provider,
            bridge_disposition_batch_provider=provider.bridge_disposition_batch,
            shared_bridge_evidence_cache=cache,
            fps=30.0,
            config=isolated_score_config(),
        )
        self.assertEqual(provider.batch_requests[1], first_batch[1:])
        audit = cache.to_dict()
        self.assertEqual(
            (
                audit["entry_count"],
                audit["hit_count"],
                audit["miss_count"],
                audit["admission_count"],
                audit["saturation_count"],
            ),
            (1, 1, 3, 1, 2),
        )

    def test_malformed_batch_is_transactional_and_admits_no_scalars(self) -> None:
        for malformed in (
            "missing",
            "extra",
            "wrong_request",
            "wrong_type",
            "list",
            "exception",
        ):
            with self.subTest(malformed=malformed):
                provider = BatchedBridgeProvider(malformed=malformed)
                cache = JointTimelineSharedBridgeEvidenceCache(maximum_entries=8)
                before = cache.to_dict()
                with self.assertRaises((RuntimeError, TypeError)):
                    select_joint_timeline(
                        three_segment_fixture(),
                        option_provider=zero_offset_options,
                        bridge_provider=provider.bridge_provider,
                        bridge_disposition_batch_provider=(
                            provider.bridge_disposition_batch
                        ),
                        shared_bridge_evidence_cache=cache,
                        fps=30.0,
                        config=isolated_score_config(),
                    )
                self.assertEqual(cache.to_dict(), before)
                self.assertEqual(provider.exact_requests, [])
                self.assertEqual(len(provider.batch_requests), 1)

    def test_batch_preflight_rejects_unrelated_callables_before_option_decode(
        self,
    ) -> None:
        exact_provider = BatchedBridgeProvider()
        other_provider = BatchedBridgeProvider()
        option_calls = 0

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            nonlocal option_calls
            option_calls += 1
            return zero_offset_options(key)

        with self.assertRaisesRegex(ValueError, "provider-bound"):
            select_joint_timeline(
                three_segment_fixture(),
                option_provider=provide_options,
                bridge_provider=exact_provider.bridge_provider,
                bridge_disposition_batch_provider=(
                    other_provider.bridge_disposition_batch
                ),
            )
        self.assertEqual(option_calls, 0)

        with self.assertRaisesRegex(ValueError, "provider-bound"):
            select_joint_timeline(
                three_segment_fixture(),
                option_provider=provide_options,
                bridge_provider=valid_bridge,
                bridge_disposition_batch_provider=lambda requests: (),
            )
        self.assertEqual(option_calls, 0)

    def test_batch_no_path_and_callback_failure_match_legacy_control_flow(
        self,
    ) -> None:
        segments = (
            (candidate(0, "A", 0.8),),
            (candidate(1, "reject-only", 0.8),),
            (candidate(2, "C", 0.8),),
        )
        legacy_provider = BatchedBridgeProvider()
        batch_provider = BatchedBridgeProvider()
        legacy_progress: list[JointTimelineProgress] = []
        batch_progress: list[JointTimelineProgress] = []
        with self.assertRaises(NoCompleteTimelinePath) as legacy_error:
            select_joint_timeline(
                segments,
                option_provider=zero_offset_options,
                bridge_provider=legacy_provider.bridge_provider,
                fps=30.0,
                config=isolated_score_config(),
                progress_callback=legacy_progress.append,
            )
        with self.assertRaises(NoCompleteTimelinePath) as batch_error:
            select_joint_timeline(
                segments,
                option_provider=zero_offset_options,
                bridge_provider=batch_provider.bridge_provider,
                bridge_disposition_batch_provider=(
                    batch_provider.bridge_disposition_batch
                ),
                fps=30.0,
                config=isolated_score_config(),
                progress_callback=batch_progress.append,
            )
        self.assertEqual(
            (
                batch_error.exception.reason_code,
                batch_error.exception.segment_index,
                batch_error.exception.rejection_counts,
            ),
            (
                legacy_error.exception.reason_code,
                legacy_error.exception.segment_index,
                legacy_error.exception.rejection_counts,
            ),
        )
        self.assertEqual(
            [
                (item.phase, item.completed_boundary_layers)
                for item in batch_progress
            ],
            [
                (item.phase, item.completed_boundary_layers)
                for item in legacy_progress
            ],
        )
        self.assertEqual(len(batch_provider.batch_requests), 1)
        self.assertEqual(batch_provider.exact_requests, [])

        callback_provider = BatchedBridgeProvider()
        callback_failure = RuntimeError("synthetic callback abort")

        def abort(progress: JointTimelineProgress) -> None:
            if progress.phase == "boundary_layer_complete":
                raise callback_failure

        with self.assertRaises(RuntimeError) as caught:
            select_joint_timeline(
                three_segment_fixture(),
                option_provider=zero_offset_options,
                bridge_provider=callback_provider.bridge_provider,
                bridge_disposition_batch_provider=(
                    callback_provider.bridge_disposition_batch
                ),
                fps=30.0,
                config=isolated_score_config(),
                progress_callback=abort,
            )
        self.assertIs(caught.exception, callback_failure)
        self.assertEqual(callback_provider.batch_requests, [])

    def test_batch_scalar_must_match_selected_exact_backtrace(self) -> None:
        class DivergingBatchProvider(BatchedBridgeProvider):
            def bridge_provider(
                self,
                candidate_: TimelineCandidate,
                entry: AuthorizedBoundaryOption,
                exit: AuthorizedBoundaryOption,
                duration_frames: float,
            ) -> SegmentBridgeProof:
                proof = super().bridge_provider(
                    candidate_,
                    entry,
                    exit,
                    duration_frames,
                )
                return replace(proof, score_adjustment=-0.25)

        provider = DivergingBatchProvider()
        with self.assertRaisesRegex(
            RuntimeError,
            "diverged within one renderer-semantic state",
        ):
            select_joint_timeline(
                three_segment_fixture(),
                option_provider=zero_offset_options,
                bridge_provider=provider.bridge_provider,
                bridge_disposition_batch_provider=(
                    provider.bridge_disposition_batch
                ),
                fps=30.0,
                config=isolated_score_config(),
            )
        self.assertEqual(len(provider.batch_requests), 1)
        self.assertEqual(len(provider.exact_requests), 1)

    def test_batch_request_and_disposition_repr_hide_exact_provenance(self) -> None:
        segments = three_segment_fixture()
        entry = zero_offset_options(
            BoundaryMeasurementKey(0, segments[0][0], segments[1][0])
        ).authorized_options[0]
        exit_ = zero_offset_options(
            BoundaryMeasurementKey(1, segments[1][0], segments[2][0])
        ).authorized_options[0]
        request = SegmentBridgeDispositionRequest(
            segments[1][0],
            entry,
            exit_,
            30.0,
        )
        disposition = SegmentBridgeDisposition(
            request,
            True,
            True,
            True,
            (),
        )
        for secret in (
            segments[1][0].candidate_id,
            entry.proof_identity,
            exit_.proof_identity,
        ):
            self.assertNotIn(secret, repr(request))
            self.assertNotIn(secret, repr(disposition))

    def test_shared_bridge_scalars_suppress_semantic_redecode_but_keep_exact_proof(
        self,
    ) -> None:
        segments = three_segment_fixture()

        class BridgeCounter:
            def __init__(self) -> None:
                self.calls = 0

            def provide(
                self,
                middle: TimelineCandidate,
                entry: AuthorizedBoundaryOption,
                exit: AuthorizedBoundaryOption,
                duration_frames: float,
            ) -> SegmentBridgeProof:
                self.calls += 1
                return valid_bridge(middle, entry, exit, duration_frames)

        baseline_provider = BridgeCounter()
        baseline = select_joint_timeline(
            segments,
            option_provider=zero_offset_options,
            bridge_provider=baseline_provider.provide,
            fps=30.0,
            config=isolated_score_config(),
        )
        self.assertEqual(baseline_provider.calls, 2)

        shared_provider = BridgeCounter()
        shared_cache = JointTimelineSharedBridgeEvidenceCache(maximum_entries=1)
        select_joint_timeline(
            segments,
            option_provider=zero_offset_options,
            bridge_provider=shared_provider.provide,
            shared_bridge_evidence_cache=shared_cache,
            fps=30.0,
            config=isolated_score_config(),
        )
        second_progress: list[JointTimelineProgress] = []
        second = select_joint_timeline(
            segments,
            option_provider=zero_offset_options,
            bridge_provider=shared_provider.provide,
            shared_bridge_evidence_cache=shared_cache,
            fps=30.0,
            config=isolated_score_config(),
            progress_callback=second_progress.append,
        )

        self.assertEqual(shared_provider.calls, 3)
        self.assertEqual(second.bridge_provider_call_count, 1)
        self.assertEqual(second.timeline, baseline.timeline)
        self.assertEqual(second.boundary_options, baseline.boundary_options)
        self.assertEqual(
            second.segment_bridge_proofs,
            baseline.segment_bridge_proofs,
        )
        audit = shared_cache.to_dict()
        self.assertEqual(
            (audit["hit_count"], audit["miss_count"], audit["admission_count"]),
            (1, 1, 1),
        )
        self.assertFalse(audit["proof_identities_retained"])
        self.assertEqual(second_progress[-1].shared_bridge_cache_hit_count, 1)
        self.assertEqual(
            [item.completed_boundary_layers for item in second_progress],
            [0, 1, 2, 2],
        )
        self.assertEqual(
            [item.option_key_count for item in second_progress],
            [0, 1, 2, 2],
        )

        other_provider = BridgeCounter()
        option_calls = 0

        def rejected_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            nonlocal option_calls
            option_calls += 1
            return zero_offset_options(key)

        with self.assertRaisesRegex(ValueError, "another provider or fps"):
            select_joint_timeline(
                segments,
                option_provider=rejected_options,
                bridge_provider=other_provider.provide,
                shared_bridge_evidence_cache=shared_cache,
                fps=30.0,
                config=isolated_score_config(),
            )
        self.assertEqual(option_calls, 0)

        preflight_cache = JointTimelineSharedBridgeEvidenceCache(maximum_entries=1)
        invalid_provider = BridgeCounter()
        valid_provider = BridgeCounter()
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            select_joint_timeline_candidates(
                segments,
                option_provider=zero_offset_options,
                bridge_provider=invalid_provider.provide,
                shared_bridge_evidence_cache=preflight_cache,
                maximum_results=9,
            )
        select_joint_timeline(
            segments,
            option_provider=zero_offset_options,
            bridge_provider=valid_provider.provide,
            shared_bridge_evidence_cache=preflight_cache,
            fps=30.0,
            config=isolated_score_config(),
        )
        self.assertEqual(invalid_provider.calls, 0)
        self.assertEqual(valid_provider.calls, 2)
        self.assertEqual(other_provider.calls, 0)
        with self.assertRaisesRegex(ValueError, "another provider or fps"):
            select_joint_timeline(
                segments,
                option_provider=rejected_options,
                bridge_provider=shared_provider.provide,
                shared_bridge_evidence_cache=shared_cache,
                fps=24.0,
                config=isolated_score_config(),
            )
        self.assertEqual(option_calls, 0)

    def test_shared_bridge_hit_still_revalidates_exact_backtrace_scalars(self) -> None:
        segments = three_segment_fixture()

        class DivergingProvider:
            diverge = False

            def provide(
                self,
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
                    score_adjustment=-0.1 if self.diverge else 0.0,
                )

        provider = DivergingProvider()
        cache = JointTimelineSharedBridgeEvidenceCache(maximum_entries=1)
        select_joint_timeline(
            segments,
            option_provider=zero_offset_options,
            bridge_provider=provider.provide,
            shared_bridge_evidence_cache=cache,
            fps=30.0,
            config=isolated_score_config(),
        )
        provider.diverge = True

        with self.assertRaisesRegex(
            RuntimeError,
            "diverged within one renderer-semantic state",
        ):
            select_joint_timeline(
                segments,
                option_provider=zero_offset_options,
                bridge_provider=provider.provide,
                shared_bridge_evidence_cache=cache,
                fps=30.0,
                config=isolated_score_config(),
            )

    def test_failed_bridge_measurement_is_never_admitted(self) -> None:
        segments = three_segment_fixture()
        for mismatch in (False, True):
            def failing_provider(
                middle: TimelineCandidate,
                entry: AuthorizedBoundaryOption,
                exit: AuthorizedBoundaryOption,
                duration_frames: float,
            ) -> SegmentBridgeProof:
                if not mismatch:
                    raise RuntimeError("synthetic decode failure")
                return valid_bridge(
                    middle,
                    entry,
                    exit,
                    duration_frames + 1.0,
                )
            with self.subTest(mismatch=mismatch):
                cache = JointTimelineSharedBridgeEvidenceCache(maximum_entries=1)
                with self.assertRaises((RuntimeError, TypeError)):
                    select_joint_timeline(
                        segments,
                        option_provider=zero_offset_options,
                        bridge_provider=failing_provider,
                        shared_bridge_evidence_cache=cache,
                        fps=30.0,
                        config=isolated_score_config(),
                    )
                self.assertEqual(cache.entry_count, 0)
                self.assertEqual(cache.admission_count, 0)

    def test_shared_bridge_cache_cap_is_admission_only_without_eviction(self) -> None:
        self.assertEqual(JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES, 300_000)
        first = three_segment_fixture()
        second = (first[0], (candidate(1, "D", 0.8),), first[2])
        cache = JointTimelineSharedBridgeEvidenceCache(maximum_entries=1)

        for segments in (first, second, first):
            select_joint_timeline(
                segments,
                option_provider=zero_offset_options,
                bridge_provider=valid_bridge,
                shared_bridge_evidence_cache=cache,
                fps=30.0,
                config=isolated_score_config(),
            )

        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.miss_count, 2)
        self.assertEqual(cache.hit_count, 1)
        self.assertEqual(cache.admission_count, 1)
        self.assertEqual(cache.saturation_count, 1)

    def test_exact_no_good_rejects_before_option_provider_decoding(self) -> None:
        segments = (
            (candidate(0, "A", 0.8),),
            (candidate(1, "B", 0.8),),
        )
        forbidden = build_timeline_no_good_edge_key(
            0,
            segments[0][0],
            segments[1][0],
        )
        provider_calls = 0

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            nonlocal provider_calls
            provider_calls += 1
            return option_set(key, authorized={0: 0.0})

        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_timeline(
                segments,
                option_provider=provide_options,
                config=isolated_score_config(),
                forbidden_boundary_edges=TimelineNoGoodSet((forbidden,)),
            )

        self.assertEqual(provider_calls, 0)
        self.assertEqual(
            caught.exception.reason_code,
            "joint_timeline_hard_incompatible",
        )
        self.assertEqual(
            caught.exception.rejection_counts,
            {"exact_boundary_no_good": 1},
        )

    def test_unknown_exact_no_good_fails_before_option_provider(self) -> None:
        segments = (
            (candidate(0, "A", 0.8),),
            (candidate(1, "B", 0.8),),
        )
        unknown = TimelineNoGoodEdgeKey(
            boundary_index=0,
            outgoing_source_id="A",
            outgoing_shot_id="shot:0:A",
            incoming_source_id="B",
            incoming_shot_id="shot:1:unknown",
        )
        provider_calls = 0

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            nonlocal provider_calls
            provider_calls += 1
            return option_set(key, authorized={0: 0.0})

        with self.assertRaisesRegex(
            ValueError,
            "forbidden boundary edge does not belong",
        ):
            select_joint_timeline_candidates(
                segments,
                option_provider=provide_options,
                config=isolated_score_config(),
                forbidden_boundary_edges=(unknown,),
            )

        self.assertEqual(provider_calls, 0)

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

    def test_duplicate_logical_path_does_not_evict_unique_runner_up(self) -> None:
        duplicate_best = candidate(0, "duplicate", 1.0)
        duplicate_lower = replace(duplicate_best, quality_score=0.99)
        unique_runner_up = candidate(0, "unique", 0.9)
        segments = (
            (duplicate_best, duplicate_lower, unique_runner_up),
            (candidate(1, "end", 0.0),),
        )

        ranked = select_joint_timeline_candidates(
            segments,
            option_provider=zero_offset_options,
            config=isolated_score_config(),
            maximum_results=2,
        )

        self.assertEqual(len(ranked), 2)
        self.assertEqual(
            tuple(
                tuple(item.source_id for item in result.candidates)
                for result in ranked
            ),
            (("duplicate", "end"), ("unique", "end")),
        )
        self.assertAlmostEqual(ranked[0].total_score, 1.0)
        self.assertAlmostEqual(ranked[1].total_score, 0.9)

    def test_three_by_eight_semantic_bridge_dominance_preserves_exact_proofs(
        self,
    ) -> None:
        source_ids = tuple(f"S{index}" for index in range(8))
        segments = tuple(
            tuple(candidate(segment, source, 0.8) for source in source_ids)
            for segment in range(3)
        )
        option_calls = 0
        bridge_calls = 0

        def provide_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
            nonlocal option_calls
            option_calls += 1
            authorized = (
                {0: 0.0}
                if key.outgoing.source_id == key.incoming.source_id
                else {offset: 0.0 for offset in JOINT_BOUNDARY_FRAME_OFFSETS}
            )
            return option_set(key, authorized=authorized)

        def provide_bridge(
            middle: TimelineCandidate,
            entry: AuthorizedBoundaryOption,
            exit: AuthorizedBoundaryOption,
            duration_frames: float,
        ) -> SegmentBridgeProof:
            nonlocal bridge_calls
            bridge_calls += 1
            return valid_bridge(middle, entry, exit, duration_frames)

        ranked = select_joint_timeline_candidates(
            segments,
            option_provider=provide_options,
            bridge_provider=provide_bridge,
            fps=30.0,
            config=isolated_score_config(),
            maximum_results=8,
        )

        old_exact_option_product = 8 * (1 + 7 * 17) ** 2
        semantic_bridge_bound = 8 * 17**2 + len(ranked)
        self.assertEqual(option_calls, 2 * 8**2)
        self.assertEqual(len(ranked), 8)
        self.assertLessEqual(bridge_calls, semantic_bridge_bound)
        self.assertLess(bridge_calls, old_exact_option_product // 40)
        self.assertTrue(
            all(
                item.bridge_provider_call_count == bridge_calls
                for item in ranked
            )
        )
        proof_variants = {
            tuple(option.proof_identity for option in item.boundary_options)
            for item in ranked
        }
        self.assertEqual(len(proof_variants), len(ranked))
        for item in ranked:
            self.assertEqual(len(item.segment_bridge_proofs), 1)
            bridge = item.segment_bridge_proofs[0]
            self.assertIs(bridge.entry_option, item.boundary_options[0])
            self.assertIs(bridge.exit_option, item.boundary_options[1])
            self.assertIn(
                item.boundary_options[0].proof_identity,
                bridge.proof_identity,
            )
            self.assertIn(
                item.boundary_options[1].proof_identity,
                bridge.proof_identity,
            )
            self.assertTrue(
                all(
                    len(option_set_.offset_audits) == 17
                    for option_set_ in item.measured_boundary_option_sets
                )
            )

    def test_semantic_state_bucket_retains_top_eight_proof_variants(self) -> None:
        first_segment = tuple(
            replace(
                candidate(0, "A", 0.8),
                candidate_id=f"candidate:0:A:{index}",
                shot_id=f"shot:0:A:{index}",
            )
            for index in range(8)
        )
        segments = (
            first_segment,
            (candidate(1, "B", 0.8),),
            (candidate(2, "C", 0.8),),
        )
        bridge_calls = 0

        def provide_bridge(
            middle: TimelineCandidate,
            entry: AuthorizedBoundaryOption,
            exit: AuthorizedBoundaryOption,
            duration_frames: float,
        ) -> SegmentBridgeProof:
            nonlocal bridge_calls
            bridge_calls += 1
            return valid_bridge(middle, entry, exit, duration_frames)

        ranked = select_joint_timeline_candidates(
            segments,
            option_provider=lambda key: option_set(
                key,
                authorized={0: 0.0},
            ),
            bridge_provider=provide_bridge,
            fps=30.0,
            config=isolated_score_config(),
            maximum_results=8,
        )

        self.assertEqual(len(ranked), 8)
        self.assertEqual(bridge_calls, 9)
        self.assertEqual(
            {item.candidates[0].candidate_id for item in ranked},
            {item.candidate_id for item in first_segment},
        )
        self.assertEqual(
            len(
                {
                    item.boundary_options[0].proof_identity
                    for item in ranked
                }
            ),
            8,
        )
        for item in ranked:
            bridge = item.segment_bridge_proofs[0]
            self.assertIs(bridge.entry_option, item.boundary_options[0])
            self.assertIs(bridge.exit_option, item.boundary_options[1])

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
