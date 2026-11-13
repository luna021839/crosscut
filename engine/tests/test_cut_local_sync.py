from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from crosscut.cut_local_sync import (
    CutLocalMotionSample,
    MEMBER_ACTIVATION,
    PTS_CACHE_ANALYZER,
    PTS_CACHE_NAMESPACE,
    _canonical_sha256,
    _expected_pts_cache_analyzer_sha256,
    _probe_actual_video_pts,
    _probe_actual_video_pts_cached,
    analyze_cut_local_sync,
    estimate_motion_phase,
    renderer_selected_frame_ordinals,
    validate_cut_local_sync_evidence,
)
from crosscut.ive import AspectRatio
from crosscut.reference_match import DecodedTargetFrames
from crosscut.render import (
    MEMBER_BLUR_BRIDGE_REASON,
    MEMBER_ZOOM_BRIDGE_REASON,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from crosscut.source_scalar_cache import (
    PersistentSourceScalarCacheStorageUnavailableError,
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


def _scalar_timewarp() -> TimeWarpMap:
    return TimeWarpMap(
        global_offset_seconds=0.0,
        master_duration_seconds=3.0,
        source_duration_seconds=3.0,
        control_points=(),
        anchors=(),
        residuals=TimeWarpResidualDiagnostics(None, None, None, 0),
        config=TimeWarpConfig(),
        scalar_fallback=True,
        fallback_reason="insufficient_confident_anchors",
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


def _mixed_timewarp_payloads() -> tuple[dict, dict]:
    local = _identity_timewarp().to_dict()
    local.update({"source_index": 0, "status": "local_monotonic_timewarp"})
    scalar = _scalar_timewarp().to_dict()
    scalar.update({"source_index": 1, "status": "global_scalar_offset"})
    return (local, scalar)


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
    def test_actual_pts_probe_uses_fast_packet_pts_in_presentation_order(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0.066667\n0.000000\n0.033333\n",
            stderr="",
        )
        with patch(
            "crosscut.cut_local_sync.subprocess.run",
            return_value=completed,
        ) as run:
            points = _probe_actual_video_pts(Path("/tmp/source.mp4"), 4.0)

        self.assertEqual(points, (0.0, 0.033333, 0.066667))
        command = run.call_args.args[0]
        self.assertIn("packet=pts_time", command)
        self.assertNotIn("frame=best_effort_timestamp_time", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 4.0)

    def test_actual_pts_probe_persists_and_reuses_scalar_cache(self) -> None:
        points = (0.0, 0.033, 0.067, 0.1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"stable-test-source")
            cache = root / "cache"
            with patch(
                "crosscut.cut_local_sync._probe_actual_video_pts",
                return_value=points,
            ) as probe:
                first_points, first_audit = _probe_actual_video_pts_cached(
                    source,
                    1.0,
                    cache,
                )
                second_points, second_audit = _probe_actual_video_pts_cached(
                    source,
                    1.0,
                    cache,
                )

        self.assertEqual(first_points, points)
        self.assertEqual(second_points, points)
        probe.assert_called_once_with(source, 1.0)
        self.assertFalse(first_audit["analysis_reused"])
        self.assertEqual(first_audit["miss_count"], 1)
        self.assertEqual(first_audit["write_count"], 1)
        self.assertTrue(second_audit["analysis_reused"])
        self.assertEqual(second_audit["hit_count"], 1)
        self.assertEqual(second_audit["write_count"], 0)
        self.assertFalse(second_audit["raw_media_persisted"])

    def test_actual_pts_cache_forwards_direct_derived_root(self) -> None:
        points = (0.0, 0.033, 0.067, 0.1)
        source = Path("/tmp/direct-pts-source.mp4")
        legacy_root = Path("/tmp/legacy-pts-cache")
        direct_root = Path("/tmp/direct-derived-cache")
        with patch(
            "crosscut.cut_local_sync.PersistentSourceScalarCache"
        ) as cache_type, patch(
            "crosscut.cut_local_sync._probe_actual_video_pts"
        ) as probe:
            cache = cache_type.return_value
            cache.get.return_value = {
                "analysis": "ffprobe_video_packet_pts_presentation_order_v2",
                "pts_basis": "ffprobe_packet_pts_time_sorted_presentation_order",
                "pts_seconds": list(points),
            }
            cache.diagnostics.return_value = {
                "cache_contract": "ive_source_scalar_analysis_v1",
                "hit_count": 1,
            }

            actual, audit = _probe_actual_video_pts_cached(
                source,
                1.0,
                legacy_root,
                derived_cache_directory=direct_root,
            )

        self.assertEqual(actual, points)
        self.assertTrue(audit["analysis_reused"])
        probe.assert_not_called()
        cache_type.assert_called_once_with(
            source_cache_directory=legacy_root,
            source_path=source,
            namespace=PTS_CACHE_NAMESPACE,
            analyzer=PTS_CACHE_ANALYZER,
            derived_cache_directory=direct_root,
        )

    def test_direct_pts_cache_runtime_loss_propagates_without_ffprobe_fallback(
        self,
    ) -> None:
        direct_root = Path("/tmp/direct-derived-cache")
        with patch(
            "crosscut.cut_local_sync.PersistentSourceScalarCache"
        ) as cache_type, patch(
            "crosscut.cut_local_sync.subprocess.run"
        ) as ffprobe_run:
            cache_type.return_value.get.side_effect = (
                PersistentSourceScalarCacheStorageUnavailableError(
                    "required direct source scalar cache storage is unavailable"
                )
            )

            with self.assertRaises(
                PersistentSourceScalarCacheStorageUnavailableError,
            ) as raised:
                analyze_cut_local_sync(
                    _plan(),
                    source_timewarps=(
                        _identity_timewarp(),
                        _identity_timewarp(),
                    ),
                    timewarp_payloads=_timewarp_payloads(),
                    pts_derived_cache_directory=direct_root,
                )

        ffprobe_run.assert_not_called()
        self.assertNotIn(str(direct_root), str(raised.exception))
        cache_type.assert_called_once_with(
            source_cache_directory=direct_root,
            source_path=Path("/tmp/cut-sync-a.mp4"),
            namespace=PTS_CACHE_NAMESPACE,
            analyzer=PTS_CACHE_ANALYZER,
            derived_cache_directory=direct_root,
        )

    def test_direct_pts_cache_rejects_custom_provider_before_provider_call(
        self,
    ) -> None:
        direct_root = Path("/tmp/direct-derived-cache")
        provider = Mock(return_value=(0.0, 0.033, 0.067, 0.1))

        with self.assertRaisesRegex(
            ValueError,
            "required direct PTS cache needs the built-in PTS provider",
        ) as raised:
            analyze_cut_local_sync(
                _plan(),
                source_timewarps=(
                    _identity_timewarp(),
                    _identity_timewarp(),
                ),
                timewarp_payloads=_timewarp_payloads(),
                pts_derived_cache_directory=direct_root,
                pts_provider=provider,
            )

        provider.assert_not_called()
        self.assertNotIn(str(direct_root), str(raised.exception))

    def test_validator_accepts_legacy_source_audit_without_pts_cache(self) -> None:
        plan, evidence = _fixture_evidence()
        legacy = deepcopy(evidence)
        for audit in legacy["source_decode_audit"]:
            audit.pop("persistent_pts_cache", None)
        _rehash(legacy)

        validated = validate_cut_local_sync_evidence(
            legacy,
            plan=plan.to_json(),
            timewarp_payloads=_timewarp_payloads(),
        )

        self.assertTrue(validated["passed"], validated["errors"])

    def test_validator_accepts_direct_pts_cache_diagnostics_contract(
        self,
    ) -> None:
        plan, evidence = _fixture_evidence()
        direct = deepcopy(evidence)
        direct["source_decode_audit"][0]["persistent_pts_cache"] = {
            "cache_contract": "ive_source_scalar_analysis_v1",
            "namespace": PTS_CACHE_NAMESPACE,
            "source_sha256": "0" * 64,
            "analyzer_sha256": _expected_pts_cache_analyzer_sha256(),
            "hit_count": 0,
            "miss_count": 1,
            "write_count": 1,
            "invalid_record_count": 0,
            "disabled_by_source_change": False,
            "required": True,
            "direct_derived_cache_configured": True,
            "storage_io_failure_count": 0,
            "local_paths_exposed": False,
            "raw_media_persisted": False,
            "analysis_reused": False,
        }
        _rehash(direct)

        validated = validate_cut_local_sync_evidence(
            direct,
            plan=plan.to_json(),
            timewarp_payloads=_timewarp_payloads(),
        )

        self.assertTrue(validated["passed"], validated["errors"])

        path_leak = deepcopy(direct)
        path_leak["source_decode_audit"][0]["persistent_pts_cache"][
            "local_paths_exposed"
        ] = True
        _rehash(path_leak)
        rejected = validate_cut_local_sync_evidence(
            path_leak,
            plan=plan.to_json(),
            timewarp_payloads=_timewarp_payloads(),
        )
        self.assertFalse(rejected["passed"])
        self.assertIn(
            "source_0_audit_pts_cache_contract_invalid",
            rejected["errors"],
        )

    def test_validator_rejects_forged_pts_cache_analyzer_identity(self) -> None:
        plan, evidence = _fixture_evidence()
        forged = deepcopy(evidence)
        forged["source_decode_audit"][0]["persistent_pts_cache"] = {
            "cache_contract": "ive_source_scalar_analysis_v1",
            "namespace": "cut-local-video-pts-v1",
            "source_sha256": "0" * 64,
            "analyzer_sha256": "1" * 64,
            "hit_count": 1,
            "miss_count": 0,
            "write_count": 0,
            "invalid_record_count": 0,
            "disabled_by_source_change": False,
            "raw_media_persisted": False,
            "analysis_reused": True,
        }
        _rehash(forged)

        validated = validate_cut_local_sync_evidence(
            forged,
            plan=plan.to_json(),
            timewarp_payloads=_timewarp_payloads(),
        )

        self.assertFalse(validated["passed"])
        self.assertIn(
            "source_0_audit_pts_cache_contract_invalid",
            validated["errors"],
        )

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

    def test_low_correlation_nonstatic_motion_uses_exact_zero_lag_pts_proof(
        self,
    ) -> None:
        plan = _plan()
        points = tuple(index / 30.0 for index in range(91))
        motion_by_source = (
            np.random.default_rng(0).uniform(0.01, 0.20, len(points)),
            np.random.default_rng(21).uniform(0.01, 0.20, len(points)),
        )

        def fake_decode(path: Path, indices, _config) -> DecodedTargetFrames:
            requested = tuple(sorted(set(indices)))
            source_index = 0 if path.name.endswith("a.mp4") else 1
            code_offset = source_index * 100
            return DecodedTargetFrames(
                frame_rate=30.0,
                declared_frame_count=len(points),
                requested_indices=requested,
                frames={
                    index: np.full(
                        (8, 8, 3),
                        code_offset + index,
                        dtype=np.uint8,
                    )
                    for index in requested
                },
            )

        def descriptor(frame: np.ndarray) -> FrameVisualDescriptor:
            code = int(frame[0, 0, 0])
            source_index = 1 if code >= 100 else 0
            frame_index = code - source_index * 100
            cumulative = float(
                np.sum(motion_by_source[source_index][:frame_index])
            )
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

        evidence = analyze_cut_local_sync(
            plan,
            source_timewarps=(_identity_timewarp(), _identity_timewarp()),
            timewarp_payloads=_timewarp_payloads(),
            pts_provider=lambda _path, _timeout: points,
            decode_provider=fake_decode,
            descriptor_provider=descriptor,
        )
        boundary = evidence["per_boundary"][0]
        validated = validate_cut_local_sync_evidence(
            evidence,
            plan=plan.to_json(),
            timewarp_payloads=_timewarp_payloads(),
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(
            boundary["measurement_mode"],
            "renderer_zero_lag_actual_pts_with_nonstatic_body_evidence_v1",
        )
        self.assertEqual(boundary["measurement_lag_frames"], 0)
        self.assertEqual(
            boundary["advisory_motion_phase"]["reason"],
            "motion_correlation_below_minimum",
        )
        self.assertFalse(
            boundary["advisory_motion_phase"][
                "influenced_release_measurement"
            ]
        )
        self.assertAlmostEqual(boundary["residual_frames"], 0.0)
        self.assertTrue(validated["passed"], validated["errors"])

        forged = deepcopy(evidence)
        forged["per_boundary"][0]["advisory_motion_phase"]["reason"] = (
            "ambiguous_motion_lag"
        )
        _rehash(forged)
        rejected = validate_cut_local_sync_evidence(
            forged,
            plan=plan.to_json(),
            timewarp_payloads=_timewarp_payloads(),
        )
        self.assertFalse(rejected["passed"])
        self.assertIn(
            "boundary_0_advisory_motion_phase_mismatch",
            rejected["errors"],
        )

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

    def test_scalar_alignment_fallback_retains_exact_cut_local_proof(self) -> None:
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

        evidence = analyze_cut_local_sync(
            plan,
            source_timewarps=(_identity_timewarp(), _scalar_timewarp()),
            timewarp_payloads=_mixed_timewarp_payloads(),
            pts_provider=lambda _path, _timeout: points,
            decode_provider=fake_decode,
            descriptor_provider=lambda frame: _descriptor_for_index(
                int(frame[0, 0, 0])
            ),
        )
        validated = validate_cut_local_sync_evidence(
            evidence,
            plan=plan.to_json(),
            timewarp_payloads=_mixed_timewarp_payloads(),
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["measured_boundary_count"], 1)
        self.assertTrue(validated["passed"], validated["errors"])

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

    def test_member_cinematic_bridges_bind_exact_overlap_frames(self) -> None:
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

        for kind, reason in (
            (TransitionKind.MEMBER_ZOOM_BRIDGE, MEMBER_ZOOM_BRIDGE_REASON),
            (TransitionKind.MEMBER_BLUR_BRIDGE, MEMBER_BLUR_BRIDGE_REASON),
        ):
            with self.subTest(kind=kind.value):
                plan = RenderPlan(
                    sources=(Path("/tmp/bridge-a.mp4"), Path("/tmp/bridge-b.mp4")),
                    segments=(
                        RenderSegment(0, 0.0, 1.0, 0.0, source_duration=1.0),
                        RenderSegment(1, 1.0, 1.0, 1.0, source_duration=1.0),
                    ),
                    transitions=(
                        RenderTransition(
                            kind=kind,
                            frame_count=7,
                            reason=reason,
                            tier="A",
                        ),
                    ),
                    reference_source=0,
                    aspect_ratio=AspectRatio.LANDSCAPE,
                    output=Path("/tmp/bridge-output.mp4"),
                    fps=30.0,
                    member_pose_dissolve_requested=True,
                    member_pose_dissolve_effective=False,
                    member_pose_dissolve_count=0,
                )
                evidence = analyze_cut_local_sync(
                    plan,
                    source_timewarps=(
                        _identity_timewarp(),
                        _identity_timewarp(),
                    ),
                    timewarp_payloads=_timewarp_payloads(),
                    activation=MEMBER_ACTIVATION,
                    pts_provider=lambda _path, _timeout: points,
                    decode_provider=fake_decode,
                    descriptor_provider=lambda frame: _descriptor_for_index(
                        int(frame[0, 0, 0])
                    ),
                )

                self.assertEqual(
                    evidence["per_boundary"][0]["sampling_domain"],
                    "member_cinematic_bridge_exact_outgoing_handle_and_incoming_overlap",
                )
                self.assertEqual(evidence["per_boundary"][0]["status"], "measured")
                self.assertEqual(evidence["unknown_reason_counts"], {})
                self.assertEqual(evidence["policy"]["minimum_motion_pair_count"], 5)
                self.assertEqual(evidence["policy"]["minimum_correlation"], 0.55)
                self.assertEqual(
                    evidence["policy"]["minimum_correlation_margin"],
                    0.08,
                )
                validated = validate_cut_local_sync_evidence(
                    evidence,
                    plan=plan.to_json(),
                    timewarp_payloads=_timewarp_payloads(),
                    expected_activation=MEMBER_ACTIVATION,
                )
                self.assertTrue(validated["passed"], validated["errors"])

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
