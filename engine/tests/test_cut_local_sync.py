from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from crosscut.cut_local_sync import (
    CutLocalMotionSample,
    _canonical_sha256,
    analyze_cut_local_sync,
    estimate_motion_phase,
    renderer_selected_frame_ordinals,
    validate_cut_local_sync_evidence,
)
from crosscut.ive import AspectRatio
from crosscut.reference_match import DecodedTargetFrames
from crosscut.render import (
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from crosscut.timeline import ShotType
from crosscut.timewarp import (
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)
from crosscut.visual_score import FrameVisualDescriptor


_MOTION_VALUES = (
    0.08,
    0.42,
    0.13,
    0.71,
    0.22,
    0.94,
    0.31,
    0.16,
    0.63,
    0.27,
    0.85,
    0.19,
    0.51,
    0.11,
    0.76,
    0.24,
    0.58,
    0.15,
    0.67,
    0.29,
    0.88,
    0.20,
    0.47,
    0.10,
    0.73,
    0.26,
    0.55,
    0.14,
    0.82,
    0.23,
    0.61,
    0.17,
    0.91,
    0.28,
    0.49,
    0.12,
    0.79,
    0.25,
    0.57,
    0.18,
    0.69,
    0.30,
    0.86,
    0.21,
    0.53,
    0.09,
    0.75,
    0.32,
    0.64,
    0.16,
    0.89,
    0.27,
    0.46,
    0.13,
    0.81,
    0.22,
    0.60,
    0.18,
    0.70,
    0.24,
    0.84,
    0.29,
    0.50,
    0.11,
    0.77,
    0.26,
    0.56,
    0.15,
    0.92,
    0.20,
    0.66,
    0.28,
    0.48,
    0.12,
    0.80,
    0.23,
    0.59,
    0.17,
    0.72,
    0.25,
    0.87,
    0.31,
    0.52,
    0.10,
    0.78,
    0.27,
    0.62,
    0.14,
    0.90,
    0.22,
    0.54,
)


def _motion_samples(*, phase_frames: float = 0.0) -> tuple[CutLocalMotionSample, ...]:
    return tuple(
        CutLocalMotionSample(
            master_seconds=(index + phase_frames) / 30.0,
            value=value,
            basis="actual_pose_motion",
        )
        for index, value in enumerate(_MOTION_VALUES[:15])
    )


def _identity_timewarp() -> TimeWarpMap:
    return TimeWarpMap(
        global_offset_seconds=0.0,
        master_duration_seconds=3.0,
        source_duration_seconds=3.0,
        control_points=(
            TimeWarpControlPoint(0.0, 0.0),
            TimeWarpControlPoint(3.0, 3.0),
        ),
        anchors=(),
        residuals=TimeWarpResidualDiagnostics(0.0, 0.0, 0.0, 3),
        config=TimeWarpConfig(),
        scalar_fallback=False,
        fallback_reason=None,
    )


def _plan() -> RenderPlan:
    return RenderPlan(
        sources=(Path("/tmp/cut-sync-a.mp4"), Path("/tmp/cut-sync-b.mp4")),
        segments=(
            RenderSegment(
                source_index=0,
                source_start=0.0,
                master_start=0.0,
                duration=1.0,
                source_duration=1.0,
            ),
            RenderSegment(
                source_index=1,
                source_start=1.0,
                master_start=1.0,
                duration=1.0,
                source_duration=1.0,
            ),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path("/tmp/cut-sync-output.mp4"),
        fps=30.0,
    )


def _timewarp_payloads() -> tuple[dict, dict]:
    result = []
    for index in range(2):
        payload = _identity_timewarp().to_dict()
        payload.update(
            {
                "source_index": index,
                "status": "local_monotonic_timewarp",
            }
        )
        result.append(payload)
    return (result[0], result[1])


def _descriptor_for_index(index: int) -> FrameVisualDescriptor:
    cumulative = sum(_MOTION_VALUES[:index])
    return FrameVisualDescriptor(
        quality_score=0.9,
        composition_score=0.9,
        shot_scale=0.5,
        shot_type=ShotType.MEDIUM,
        face_count=0,
        structure_embedding=(),
        pose_embedding=(cumulative,),
        pose_confidence=0.9,
        pose_backend="opencv_zoo_mediapipe_pose_fixture",
    )


def _rehash(evidence: dict) -> None:
    evidence.pop("proof_sha256", None)
    evidence["proof_sha256"] = _canonical_sha256(evidence)


def _fixture_evidence() -> tuple[RenderPlan, dict]:
    plan = _plan()
    points = tuple(index / 30.0 for index in range(91))

    def fake_decode(_path: Path, indices, _config) -> DecodedTargetFrames:
        requested = tuple(sorted(set(indices)))
        return DecodedTargetFrames(
            frame_rate=30.0,
            declared_frame_count=len(points),
            requested_indices=requested,
            frames={
                index: np.full((8, 8, 3), index, dtype=np.uint8)
                for index in requested
            },
        )

    return (
        plan,
        analyze_cut_local_sync(
            plan,
            source_timewarps=(_identity_timewarp(), _identity_timewarp()),
            timewarp_payloads=_timewarp_payloads(),
            pts_provider=lambda _path, _timeout: points,
            decode_provider=fake_decode,
            descriptor_provider=lambda frame: _descriptor_for_index(
                int(frame[0, 0, 0])
            ),
        ),
    )


class CutLocalSyncTests(unittest.TestCase):
    def test_renderer_selection_matches_cfr_half_frame_trim_ownership(self) -> None:
        selected = renderer_selected_frame_ordinals(
            (0.0, 0.1, 0.2, 0.3, 0.4),
            source_start_seconds=0.05,
            source_duration_seconds=0.30,
            timeline_duration_seconds=0.30,
            output_fps=10.0,
            frame_count=3,
        )

        self.assertEqual(selected, (1, 2, 3))

    def test_renderer_selection_matches_vfr_drop_and_duplicate_ownership(self) -> None:
        selected = renderer_selected_frame_ordinals(
            (0.0, 0.04, 0.15, 0.31, 0.50),
            source_start_seconds=0.0,
            source_duration_seconds=0.50,
            timeline_duration_seconds=0.50,
            output_fps=10.0,
            frame_count=4,
        )

        self.assertEqual(selected, (1, 1, 2, 3))

    def test_motion_phase_measures_zero_and_subframe_actual_pts_residual(self) -> None:
        exact = estimate_motion_phase(
            _motion_samples(),
            _motion_samples(),
            output_fps=30.0,
        )
        shifted = estimate_motion_phase(
            _motion_samples(),
            _motion_samples(phase_frames=0.60),
            output_fps=30.0,
        )

        self.assertEqual(exact["status"], "measured")
        self.assertAlmostEqual(exact["residual_frames"], 0.0)
        self.assertEqual(shifted["status"], "measured")
        self.assertAlmostEqual(shifted["residual_frames"], 0.60)

    def test_static_or_ambiguous_motion_is_unknown_not_fabricated_zero(self) -> None:
        static = tuple(
            CutLocalMotionSample(
                master_seconds=index / 30.0,
                value=0.2,
                basis="actual_pose_motion",
            )
            for index in range(15)
        )
        result = estimate_motion_phase(static, static, output_fps=30.0)

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(
            result["reason"], "insufficient_nonstatic_pose_body_motion"
        )
        self.assertNotIn("residual_frames", result)

    def test_duplicate_canonical_motion_grid_index_fails_closed(self) -> None:
        outgoing = tuple(
            CutLocalMotionSample(
                master_seconds=index / 30.0,
                value=value,
                basis="actual_pose_motion",
                grid_index=index,
            )
            for index, value in enumerate(_MOTION_VALUES[:10])
        )
        incoming = list(outgoing)
        incoming[-1] = CutLocalMotionSample(
            master_seconds=incoming[-1].master_seconds,
            value=incoming[-1].value,
            basis=incoming[-1].basis,
            grid_index=8,
        )

        result = estimate_motion_phase(
            outgoing,
            tuple(incoming),
            output_fps=30.0,
        )

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "duplicate_motion_grid_index")

    def test_actual_pts_analysis_is_hash_bound_and_validator_rejects_tamper(self) -> None:
        plan = _plan()
        points = tuple(index / 30.0 for index in range(91))

        def fake_pts(_path: Path, _timeout: float) -> tuple[float, ...]:
            return points

        def fake_decode(path: Path, indices, _config) -> DecodedTargetFrames:
            requested = tuple(sorted(set(indices)))
            frames = {
                index: np.full((8, 8, 3), index, dtype=np.uint8)
                for index in requested
            }
            return DecodedTargetFrames(
                frame_rate=30.0,
                declared_frame_count=len(points),
                requested_indices=requested,
                frames=frames,
            )

        evidence = analyze_cut_local_sync(
            plan,
            source_timewarps=(_identity_timewarp(), _identity_timewarp()),
            timewarp_payloads=_timewarp_payloads(),
            pts_provider=fake_pts,
            decode_provider=fake_decode,
            descriptor_provider=lambda frame: _descriptor_for_index(
                int(frame[0, 0, 0])
            ),
        )
        validated = validate_cut_local_sync_evidence(
            evidence,
            plan=plan.to_json(),
            timewarp_payloads=_timewarp_payloads(),
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["measured_boundary_count"], 1)
        self.assertEqual(evidence["unknown_boundary_count"], 0)
        self.assertTrue(validated["passed"])
        self.assertTrue(validated["hash_bindings_valid"])

        tampered_plan = plan.to_json()
        tampered_plan["segments"][0]["source_local_end"] = 1.02
        tampered = validate_cut_local_sync_evidence(
            evidence,
            plan=tampered_plan,
            timewarp_payloads=_timewarp_payloads(),
        )
        self.assertFalse(tampered["passed"])
        self.assertIn("plan_sha256_mismatch", tampered["errors"])
        self.assertIn("boundary_clock_contract_sha256_mismatch", tampered["errors"])

    def test_missing_pose_and_body_motion_remains_unknown_and_fails(self) -> None:
        plan = _plan()
        points = tuple(index / 30.0 for index in range(91))

        def fake_decode(_path: Path, indices, _config) -> DecodedTargetFrames:
            requested = tuple(sorted(set(indices)))
            return DecodedTargetFrames(
                frame_rate=30.0,
                declared_frame_count=len(points),
                requested_indices=requested,
                frames={
                    index: np.zeros((8, 8, 3), dtype=np.uint8)
                    for index in requested
                },
            )

        missing = FrameVisualDescriptor(
            quality_score=0.9,
            composition_score=0.9,
            shot_scale=0.5,
            shot_type=ShotType.MEDIUM,
            face_count=0,
            structure_embedding=(),
        )
        evidence = analyze_cut_local_sync(
            plan,
            source_timewarps=(_identity_timewarp(), _identity_timewarp()),
            timewarp_payloads=_timewarp_payloads(),
            pts_provider=lambda _path, _timeout: points,
            decode_provider=fake_decode,
            descriptor_provider=lambda _frame: missing,
        )

        self.assertFalse(evidence["passed"])
        self.assertEqual(evidence["measured_boundary_count"], 0)
        self.assertEqual(evidence["unknown_boundary_count"], 1)
        self.assertEqual(
            evidence["unknown_reason_counts"],
            {"insufficient_motion_samples": 1},
        )
        validated = validate_cut_local_sync_evidence(
            evidence,
            plan=plan.to_json(),
            timewarp_payloads=_timewarp_payloads(),
        )
        self.assertFalse(validated["passed"])
        self.assertEqual(validated["unknown_boundary_count"], 1)

    def test_runtime_timewarp_payload_must_exactly_match_used_clock(self) -> None:
        payloads = [deepcopy(item) for item in _timewarp_payloads()]
        payloads[1]["mean_slope"] = 1.0001

        with self.assertRaisesRegex(
            ValueError, "used_timewarp_payload_runtime_mismatch"
        ):
            analyze_cut_local_sync(
                _plan(),
                source_timewarps=(_identity_timewarp(), _identity_timewarp()),
                timewarp_payloads=payloads,
                pts_provider=lambda _path, _timeout: tuple(
                    index / 30.0 for index in range(91)
                ),
            )

    def test_pts_decode_ordinal_count_mismatch_is_unknown_and_path_safe(self) -> None:
        points = tuple(index / 30.0 for index in range(91))

        def mismatched_decode(
            path: Path, indices, _config
        ) -> DecodedTargetFrames:
            requested = tuple(sorted(set(indices)))
            declared = len(points) - 1 if path.name.endswith("a.mp4") else len(points)
            return DecodedTargetFrames(
                frame_rate=30.0,
                declared_frame_count=declared,
                requested_indices=requested,
                frames={
                    index: np.full((8, 8, 3), index, dtype=np.uint8)
                    for index in requested
                },
            )

        evidence = analyze_cut_local_sync(
            _plan(),
            source_timewarps=(_identity_timewarp(), _identity_timewarp()),
            timewarp_payloads=_timewarp_payloads(),
            pts_provider=lambda _path, _timeout: points,
            decode_provider=mismatched_decode,
            descriptor_provider=lambda frame: _descriptor_for_index(
                int(frame[0, 0, 0])
            ),
        )

        self.assertFalse(evidence["passed"])
        failed = evidence["source_decode_audit"][0]
        self.assertEqual(failed["source_name"], "cut-sync-a.mp4")
        self.assertEqual(failed["reason"], "pts_decode_ordinal_count_mismatch")
        self.assertNotIn("error", failed)
        self.assertNotIn("/tmp", str(failed))

    def test_face_dissolve_uses_only_exact_handle_and_incoming_overlap(self) -> None:
        plan = RenderPlan(
            sources=(Path("/tmp/dissolve-a.mp4"), Path("/tmp/dissolve-b.mp4")),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0, source_duration=1.0),
                RenderSegment(1, 1.0, 1.0, 1.0, source_duration=1.0),
            ),
            transitions=(
                RenderTransition(
                    kind=TransitionKind.FACE_DISSOLVE,
                    frame_count=10,
                    reason="fixture_face_dissolve",
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("/tmp/dissolve-output.mp4"),
            fps=30.0,
        )
        points = tuple(index / 30.0 for index in range(91))
        requested_by_source: dict[str, tuple[int, ...]] = {}
        good_motion = _MOTION_VALUES[:12]
        good_cumulative = np.cumsum((0.0, *good_motion))

        def fake_decode(path: Path, indices, _config) -> DecodedTargetFrames:
            requested = tuple(sorted(set(indices)))
            requested_by_source[path.name] = requested
            frames: dict[int, np.ndarray] = {}
            for ordinal in requested:
                position = ordinal - 30
                code = position if path.name == "dissolve-a.mp4" else 100 + position
                frames[ordinal] = np.full((8, 8, 3), code, dtype=np.uint8)
            return DecodedTargetFrames(
                frame_rate=30.0,
                declared_frame_count=len(points),
                requested_indices=requested,
                frames=frames,
            )

        def descriptor(frame: np.ndarray) -> FrameVisualDescriptor:
            code = int(frame[0, 0, 0])
            position = code - 100 if code >= 100 else code
            value = 0.2 * position if code >= 100 else float(good_cumulative[position])
            return FrameVisualDescriptor(
                quality_score=0.9,
                composition_score=0.9,
                shot_scale=0.5,
                shot_type=ShotType.MEDIUM,
                face_count=0,
                structure_embedding=(),
                pose_embedding=(value,),
                pose_confidence=0.9,
                pose_backend="opencv_zoo_mediapipe_pose_fixture",
            )

        evidence = analyze_cut_local_sync(
            plan,
            source_timewarps=(_identity_timewarp(), _identity_timewarp()),
            timewarp_payloads=_timewarp_payloads(),
            pts_provider=lambda _path, _timeout: points,
            decode_provider=fake_decode,
            descriptor_provider=descriptor,
        )

        self.assertEqual(requested_by_source["dissolve-a.mp4"], tuple(range(30, 40)))
        self.assertEqual(requested_by_source["dissolve-b.mp4"], tuple(range(30, 40)))
        self.assertFalse(evidence["passed"])
        self.assertEqual(
            evidence["per_boundary"][0]["sampling_domain"],
            "face_dissolve_exact_outgoing_handle_and_incoming_overlap",
        )
        self.assertEqual(
            evidence["unknown_reason_counts"],
            {"insufficient_nonstatic_pose_body_motion": 1},
        )

    def test_short_face_dissolve_overlap_is_unknown(self) -> None:
        plan = RenderPlan(
            sources=(Path("/tmp/short-a.mp4"), Path("/tmp/short-b.mp4")),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0, source_duration=1.0),
                RenderSegment(1, 1.0, 1.0, 1.0, source_duration=1.0),
            ),
            transitions=(
                RenderTransition(
                    kind=TransitionKind.FACE_DISSOLVE,
                    frame_count=3,
                    reason="short_fixture_face_dissolve",
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("/tmp/short-output.mp4"),
            fps=30.0,
        )
        points = tuple(index / 30.0 for index in range(91))

        def fake_decode(_path: Path, indices, _config) -> DecodedTargetFrames:
            requested = tuple(sorted(set(indices)))
            return DecodedTargetFrames(
                frame_rate=30.0,
                declared_frame_count=len(points),
                requested_indices=requested,
                frames={
                    index: np.full((8, 8, 3), index, dtype=np.uint8)
                    for index in requested
                },
            )

        evidence = analyze_cut_local_sync(
            plan,
            source_timewarps=(_identity_timewarp(), _identity_timewarp()),
            timewarp_payloads=_timewarp_payloads(),
            pts_provider=lambda _path, _timeout: points,
            decode_provider=fake_decode,
            descriptor_provider=lambda frame: _descriptor_for_index(
                int(frame[0, 0, 0])
            ),
        )

        self.assertEqual(
            evidence["unknown_reason_counts"],
            {"insufficient_dissolve_overlap_frames": 1},
        )

    def test_validator_rejects_semantic_forgery_even_with_recomputed_proof_hash(self) -> None:
        plan, evidence = _fixture_evidence()
        mutations = {
            "residual_relation": lambda item: item["per_boundary"][0].__setitem__(
                "residual_frames", 0.25
            ),
            "boundary_payload": lambda item: item["per_boundary"][0].__setitem__(
                "output_seconds", 1.1
            ),
            "trusted_policy": lambda item: item["policy"].__setitem__(
                "minimum_correlation", 0.1
            ),
            "lag_pair_count": lambda item: item["per_boundary"][0][
                "lag_candidates"
            ][2].__setitem__("pair_count", 999),
            "source_audit": lambda item: item["source_decode_audit"][0].__setitem__(
                "requested_ordinal_count", 999
            ),
            "aggregate": lambda item: item.__setitem__(
                "threshold_exceeding_boundary_count", 1
            ),
            "actual_pts_phase": lambda item: item["per_boundary"][0][
                "renderer_frame_selection"
            ]["outgoing"]["normalized_pts_seconds"].__setitem__(
                0,
                item["per_boundary"][0]["renderer_frame_selection"][
                    "outgoing"
                ]["normalized_pts_seconds"][0]
                + 0.01,
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                forged = deepcopy(evidence)
                mutate(forged)
                _rehash(forged)
                validated = validate_cut_local_sync_evidence(
                    forged,
                    plan=plan.to_json(),
                    timewarp_payloads=_timewarp_payloads(),
                )
                self.assertFalse(validated["passed"])
                self.assertTrue(validated["errors"])

    def test_validator_strict_parses_timewarp_and_rechecks_boundary_endpoint(self) -> None:
        plan, evidence = _fixture_evidence()
        timewarps = [deepcopy(item) for item in _timewarp_payloads()]
        timewarps[0]["control_points"][1]["offset_seconds"] = 0.1
        evidence = deepcopy(evidence)
        evidence["timewarp_contract_sha256"] = _canonical_sha256(timewarps)
        _rehash(evidence)

        validated = validate_cut_local_sync_evidence(
            evidence,
            plan=plan.to_json(),
            timewarp_payloads=timewarps,
        )

        self.assertFalse(validated["passed"])
        self.assertIn(
            "source_0_timewarp_timewarp_control_point_offset_mismatch",
            validated["errors"],
        )


if __name__ == "__main__":
    unittest.main()
