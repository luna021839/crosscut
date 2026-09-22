from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import crosscut_cli
from crosscut.cut_local_sync import (
    MEMBER_ACTIVATION as MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
    _canonical_sha256,
    analyze_cut_local_sync,
)
from crosscut.direct_member_look import (
    ANALYSIS_BASIS as MEMBER_LOOK_ANALYSIS_BASIS,
    DirectMemberLookConfig,
)
from crosscut.direct_member_rendered_continuity import (
    ANALYSIS_BASIS as RENDERED_CONTINUITY_ANALYSIS_BASIS,
)
from crosscut.events import EventReporter
from crosscut.ive import AspectRatio
from crosscut.member_post_render_qa import (
    ANALYSIS_BASIS,
    ARTIFACT_FINGERPRINT_CONTRACT,
    AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT,
    AUTOMATIC_IVE_MEMBER_POST_RENDER_QA_CONTRACT,
    AutomaticMemberPostRenderArtifactSet,
    RENDER_RECEIPT_CONTRACT,
    automatic_member_post_render_qa_required,
    evaluate_and_write_automatic_member_post_render_qa,
    evaluate_automatic_member_post_render_qa,
)
from crosscut.pipeline import _enforce_automatic_member_post_render_qa
from crosscut.post_render_perceptual_quality import (
    ANALYSIS_BASIS as PERCEPTUAL_ANALYSIS_BASIS,
)
from crosscut.reference_match import DecodedTargetFrames
from crosscut.review_first_runtime import classify_review_first_qa, ReviewFirstStatus
from crosscut.render import (
    RENDER_PLAN_SOURCE_INDEX_DOMAIN,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    SourceLookAdjustment,
)
from crosscut.timeline import ShotType
from crosscut.timewarp import (
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)
from crosscut.visual_score import FrameVisualDescriptor


def _runner(
    command: tuple[str, ...] | list[str], _timeout: float
) -> subprocess.CompletedProcess[str]:
    if "-count_frames" in command:
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1080,
                    "height": 1920,
                    "avg_frame_rate": "30/1",
                    "r_frame_rate": "30/1",
                    "nb_read_frames": "120",
                    "duration": "4.0",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "4.0"},
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        )
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def _perceptual_payload(
    eligible: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[str, object]:
    per_source = [
        {
            "source_index": source_index,
            "decoded_source_sample_count": 3,
            "structure_comparable_sample_count": 3,
            "exact_structure_similarity": {
                "count": 3,
                "minimum": 0.9,
                "p10": 0.9,
                "median": 0.95,
                "p90": 0.99,
                "p99": 0.99,
                "maximum": 0.99,
            },
        }
        for source_index in eligible
    ]
    per_segment = [
        {
            "segment_index": source_index,
            "source_index": source_index,
            "structure_comparable_sample_count": 1,
            "structure_binding_passed": True,
        }
        for source_index in eligible
    ]
    return {
        "schema_version": 2,
        "analysis": PERCEPTUAL_ANALYSIS_BASIS,
        "verdict": "PASS",
        "passed": True,
        "issues": [],
        "metrics": {
            "exact_source_retention": {
                "mapping_basis": "fixture_exact_source_clock",
                "comparison_domain": "fixture_rendered_pixel_structure",
                "segment_sampling_policy": (
                    "one_mandatory_safe_interior_midpoint_per_render_segment"
                ),
                "per_source": per_source,
                "per_segment": per_segment,
            }
        },
        "serialization_policy": {
            "raw_frames_serialized": False,
            "per_frame_descriptors_serialized": False,
        },
    }


def _rendered_continuity_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis": RENDERED_CONTINUITY_ANALYSIS_BASIS,
        "passed": True,
        "failure_reasons": [],
        "failure_boundary_indices": {},
        "metrics": {},
        "boundaries": [],
    }


def _rendered_continuity_evaluator(
    *_args: object,
    **_kwargs: object,
) -> dict[str, object]:
    return _rendered_continuity_payload()


def _identity_timewarp() -> TimeWarpMap:
    return TimeWarpMap(
        global_offset_seconds=0.0,
        master_duration_seconds=5.0,
        source_duration_seconds=5.0,
        control_points=(
            TimeWarpControlPoint(0.0, 0.0),
            TimeWarpControlPoint(5.0, 5.0),
        ),
        anchors=(),
        residuals=TimeWarpResidualDiagnostics(0.0, 0.0, 0.0, 3),
        config=TimeWarpConfig(),
        scalar_fallback=False,
        fallback_reason=None,
    )


def _timewarp_payloads(count: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for source_index in range(count):
        payload = _identity_timewarp().to_dict()
        payload.update(
            {
                "source_index": source_index,
                "status": "local_monotonic_timewarp",
            }
        )
        result.append(payload)
    return result


def _motion_descriptor(
    frame: np.ndarray,
    *,
    source_motion_shift_frames: int,
) -> FrameVisualDescriptor:
    frame_index = int(frame[0, 0, 0])
    source_index = int(frame[0, 0, 1])
    motion_index = frame_index + source_index * source_motion_shift_frames
    cumulative_motion = sum(
        (((index * 37) % 101) + 1) / 101.0
        for index in range(max(0, motion_index))
    )
    return FrameVisualDescriptor(
        quality_score=0.9,
        composition_score=0.9,
        shot_scale=0.5,
        shot_type=ShotType.MEDIUM,
        face_count=0,
        structure_embedding=(),
        pose_embedding=(cumulative_motion,),
        pose_confidence=0.9,
        pose_backend="opencv_zoo_mediapipe_pose_member_fixture",
    )


def _cut_local_sync_evidence(
    plan: RenderPlan,
    timewarp_payloads: list[dict[str, object]],
    *,
    source_motion_shift_frames: int,
) -> dict[str, object]:
    points = tuple(index / 30.0 for index in range(151))
    source_index_by_path = {
        source.resolve(): index for index, source in enumerate(plan.sources)
    }

    def fake_decode(
        path: Path,
        indices: object,
        _config: object,
    ) -> DecodedTargetFrames:
        requested = tuple(sorted(set(indices)))  # type: ignore[arg-type]
        source_index = source_index_by_path[path.resolve()]
        frames: dict[int, np.ndarray] = {}
        for frame_index in requested:
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[0, 0, 0] = frame_index
            frame[0, 0, 1] = source_index
            frames[frame_index] = frame
        return DecodedTargetFrames(
            frame_rate=30.0,
            declared_frame_count=len(points),
            requested_indices=requested,
            frames=frames,
        )

    return analyze_cut_local_sync(
        plan,
        source_timewarps=tuple(_identity_timewarp() for _ in plan.sources),
        timewarp_payloads=timewarp_payloads,
        activation=MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
        pts_provider=lambda _path, _timeout: points,
        decode_provider=fake_decode,
        descriptor_provider=lambda frame: _motion_descriptor(
            frame,
            source_motion_shift_frames=source_motion_shift_frames,
        ),
    )


def _fixture(
    root: Path,
    *,
    source_motion_shift_frames: int = 0,
) -> tuple[AutomaticMemberPostRenderArtifactSet, dict[str, object]]:
    output = root / "out"
    output.mkdir()
    sources = tuple(
        (root / f"source-{index}.mp4").resolve() for index in range(4)
    )
    for index, source in enumerate(sources):
        source.write_bytes(f"source-{index}-pixels".encode())
    base = "IVE-ATTITUDE-20260825-120000"
    analysis_path = output / f"{base}-analysis.json"
    plan_path = output / f"{base}-gaeul-render-plan.json"
    video_path = output / f"{base}-gaeul-member-crosscut.mp4"
    video_path.write_bytes(b"final-member-video")

    serialized_sources = [str(path.resolve()) for path in sources]
    identity_look = SourceLookAdjustment()
    plan_model = RenderPlan(
        sources=sources,
        segments=tuple(
            RenderSegment(
                source_index=index,
                source_start=float(index),
                duration=1.0,
                master_start=float(index),
                source_duration=1.0,
                look_adjustment=identity_look,
            )
            for index in range(4)
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.PORTRAIT,
        output=video_path,
        width=1080,
        height=1920,
        fps=30.0,
        transitions=tuple(RenderTransition(reason="fixture") for _ in range(3)),
    )
    plan = plan_model.to_json()
    segments = plan["segments"]
    usage = plan_model.source_usage_payload()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    from crosscut.render_clock_artifact import render_clock_path
    # File-binding fixture; pixel/source-clock evaluation is injected separately.
    render_clock_path(video_path).write_text('{"fixture": "member clock identity"}')

    eligible = [0, 1, 2, 3]
    mapping = {
        "request_index_domain": "request_source_index",
        "eligible_index_domain": "eligible_source_index",
        "render_plan_index_domain": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
        "domain_equivalences": {
            RENDER_PLAN_SOURCE_INDEX_DOMAIN: "request_source_index"
        },
        "eligible_to_request": eligible,
        "request_to_eligible": eligible,
        "eligible_sources": [
            {
                "eligible_source_index": index,
                "request_source_index": index,
                "source": serialized_sources[index],
            }
            for index in eligible
        ],
    }
    direct_segments = [
        {
            "segment_index": index,
            "eligible_source_index": index,
            "request_source_index": index,
            "source_local_start": float(index),
            "source_local_end": float(index + 1),
            "crop_applied": False,
            "editorial_transform_applied": False,
        }
        for index in eligible
    ]
    look_payload: dict[str, object] = {
        "analysis_basis": MEMBER_LOOK_ANALYSIS_BASIS,
        "selected_graph_regularization": 24.0,
        "rapid_boundary_count": 3,
        "predicted_high_shock_count": 0,
        "predicted_high_shock_fraction": 0.0,
        "predicted_source_spans": {
            "sharpness_log": 0.0,
            "luma": 0.0,
            "chroma": 0.0,
        },
        "source_span_policy": {
            "enforced": True,
            "repeated_source_fraction": 1.0,
            "minimum_repeated_source_fraction": 0.50,
            "underidentified_initial_cache_requires_decoded_proxy_qa": False,
        },
        "output_proxy_refinement_applied": False,
        "output_proxy_refinement_iteration": 0,
        "segment_adjustments": [
            {
                "segment_index": index,
                "source_index": index,
                "brightness": identity_look.brightness,
                "contrast": identity_look.contrast,
                "saturation": identity_look.saturation,
                "temperature_balance": identity_look.temperature_balance,
                "tint_balance": identity_look.tint_balance,
                "detail_enhancement": identity_look.detail_enhancement,
                "identity": True,
            }
            for index in eligible
        ],
        "diagnostics": [
            {
                "outgoing_run_index": index,
                "incoming_run_index": index + 1,
                "outgoing_source_index": index,
                "incoming_source_index": index + 1,
                "normalized_shock": 0.0,
                "above_threshold": False,
            }
            for index in range(3)
        ],
        "config": DirectMemberLookConfig().to_dict(),
        "calibration": {
            "color_axes": "ffmpeg_native_yuv_luma_exact_temperature_tint_geq_v1",
            "lab_response_basis": (
                "eight_real_ive_member_sources_exact_ffmpeg_central_difference_v2"
            ),
            "cached_descriptor_fields": "lab_mean_gray_mean_gray_contrast",
            "full_song_proxy_rendered": False,
            "additional_source_frames_decoded": False,
            "output_domain_residual_composed_with_cached_grade": False,
        },
        "privacy": {
            "raw_frames_persisted": False,
            "raw_descriptor_embeddings_persisted": False,
            "scalar_aggregate_only": True,
        },
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "request": {
            "group": "IVE",
            "song": "ATTITUDE",
            "edit_mode": "member",
            "aspect_ratio": "9:16",
            "source_mode": "automatic",
            "sources": serialized_sources,
            "target_member": "가을",
        },
        "source_index_mapping": mapping,
        "alignment_eligibility": [
            {
                "index_domain": "request_source_index",
                "source_index": index,
                "source": serialized_sources[index],
                "render_eligible": True,
            }
            for index in eligible
        ],
        "local_timewarps": _timewarp_payloads(len(sources)),
        "member_look_calibration": look_payload,
        "member_direct_full_frame": {
            "enabled": True,
            "eligible_request_source_indices": eligible,
            "required_used_request_source_indices": eligible,
            "minimum_required_distinct_source_count": 4,
            "final_source_usage": usage,
            "segments": direct_segments,
            "look_calibration": look_payload,
        },
        "render": {
            "video_path": str(video_path),
            "plan_path": str(plan_path),
            "direct_full_frame": True,
            "final_source_usage": usage,
            "artifact_role": "pending_qa",
            "cut_local_sync": None,
            "member_look_calibration": look_payload,
        },
        "artifact_role": "pending_qa",
        "app_result_exposure": {
            "contract": AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT,
            "video_output_exposed": False,
            "delivery_completion_message_allowed": False,
        },
    }
    render_payload = manifest["render"]
    assert isinstance(render_payload, dict)
    timewarp_payloads = manifest["local_timewarps"]
    assert isinstance(timewarp_payloads, list)
    render_payload["cut_local_sync"] = _cut_local_sync_evidence(
        plan_model,
        timewarp_payloads,
        source_motion_shift_frames=source_motion_shift_frames,
    )
    analysis_path.write_text(json.dumps(manifest), encoding="utf-8")
    return (
        AutomaticMemberPostRenderArtifactSet.from_manifest(
            analysis_path, manifest
        ),
        manifest,
    )


class AutomaticMemberPostRenderQATests(unittest.TestCase):
    def test_activation_is_required_only_for_automatic_direct_ive_member(self) -> None:
        required = {
            "request": {
                "group": "IVE",
                "source_mode": "automatic",
                "edit_mode": "member",
            },
            "render": {"direct_full_frame": True},
        }
        self.assertTrue(automatic_member_post_render_qa_required(required))
        for section, field, replacement in (
            ("request", "group", "OTHER"),
            ("request", "source_mode", "local"),
            ("request", "edit_mode", "group"),
            ("render", "direct_full_frame", False),
        ):
            with self.subTest(section=section, field=field):
                candidate = json.loads(json.dumps(required))
                candidate[section][field] = replacement
                self.assertFalse(
                    automatic_member_post_render_qa_required(candidate)
                )

    def test_pass_binds_all_artifact_shas_and_every_eligible_source_pixel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _manifest = _fixture(Path(directory))
            report = evaluate_automatic_member_post_render_qa(
                artifacts,
                runner=_runner,
                perceptual_evaluator=lambda *_args, **_kwargs: (
                    _perceptual_payload()
                ),
                rendered_continuity_evaluator=(
                    _rendered_continuity_evaluator
                ),
            )

        self.assertTrue(report.passed)
        self.assertEqual(report.verdict, "PASS")
        payload = report.to_dict()
        self.assertEqual(payload["analysis"], ANALYSIS_BASIS)
        self.assertEqual(
            payload["input_artifact_fingerprints"]["contract"],
            ARTIFACT_FINGERPRINT_CONTRACT,
        )
        self.assertEqual(
            payload["input_artifact_fingerprints"],
            payload["final_artifact_fingerprints"],
        )
        receipt = payload["render_receipt"]
        self.assertEqual(receipt["contract"], RENDER_RECEIPT_CONTRACT)
        self.assertTrue(receipt["all_eligible_sources_pixel_proven"])
        cut_local = payload["metrics"]["cut_local_sync"]
        self.assertEqual(
            cut_local["activation"], MEMBER_CUT_LOCAL_SYNC_ACTIVATION
        )
        self.assertTrue(cut_local["hash_bindings_valid"])
        self.assertTrue(cut_local["passed"])
        self.assertLessEqual(cut_local["residual_p95_frames"], 0.5)
        self.assertTrue(receipt["cut_local_sync"]["passed"])
        self.assertEqual(
            [item["request_source_index"] for item in receipt["sources"]],
            [0, 1, 2, 3],
        )
        self.assertTrue(all(item["passed"] for item in receipt["sources"]))

    def test_actual_member_look_report_separates_quality_from_invalid_measurements(self) -> None:
        for invalid_measurements in (False, True):
            with self.subTest(invalid_measurements=invalid_measurements), tempfile.TemporaryDirectory() as directory:
                artifacts, manifest = _fixture(Path(directory))
                look = manifest["member_look_calibration"]
                look["predicted_source_spans"]["luma"] = 10.0
                if invalid_measurements:
                    look["diagnostics"] = []
                artifacts.analysis_path.write_text(json.dumps(manifest), encoding="utf-8")
                report = evaluate_automatic_member_post_render_qa(
                    artifacts, runner=_runner,
                    perceptual_evaluator=lambda *_args, **_kwargs: _perceptual_payload(),
                    rendered_continuity_evaluator=_rendered_continuity_evaluator,
                )
                disposition = classify_review_first_qa(
                    report.to_dict(), content_binding_passed=True,
                    review_first_enabled=True, member=True,
                )
                self.assertFalse(report.passed)
                self.assertEqual(
                    disposition.status,
                    ReviewFirstStatus.INPUT_UNUSABLE if invalid_measurements else ReviewFirstStatus.REVIEW_READY,
                    report.to_dict(),
                )
                self.assertEqual(
                    report.metrics["member_look_calibration"]["provenance_valid"],
                    report.metrics["member_look_calibration"]["cache_only_provenance_valid"],
                )

    def test_missing_cut_local_sync_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, manifest = _fixture(Path(directory))
            render = manifest["render"]
            assert isinstance(render, dict)
            render.pop("cut_local_sync")
            artifacts.analysis_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report = evaluate_automatic_member_post_render_qa(
                artifacts,
                runner=_runner,
                perceptual_evaluator=lambda *_args, **_kwargs: (
                    _perceptual_payload()
                ),
            )

        codes = {item["code"] for item in report.issues}
        self.assertFalse(report.passed)
        self.assertIn("member_cut_local_sync_evidence_missing", codes)
        self.assertIn("member_cut_local_sync_required_gate_failed", codes)
        self.assertFalse(report.render_receipt["passed"])

    def test_tampered_cut_local_sync_proof_fails_binding_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, manifest = _fixture(Path(directory))
            render = manifest["render"]
            assert isinstance(render, dict)
            evidence = render["cut_local_sync"]
            assert isinstance(evidence, dict)
            evidence["residual_p95_frames"] = 0.25
            artifacts.analysis_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report = evaluate_automatic_member_post_render_qa(
                artifacts,
                runner=_runner,
                perceptual_evaluator=lambda *_args, **_kwargs: (
                    _perceptual_payload()
                ),
            )

        codes = {item["code"] for item in report.issues}
        self.assertFalse(report.passed)
        self.assertIn("member_cut_local_sync_binding_invalid", codes)
        self.assertIn("member_cut_local_sync_required_gate_failed", codes)
        self.assertFalse(report.metrics["cut_local_sync"]["proof_sha256_valid"])
        self.assertFalse(report.render_receipt["passed"])

    def test_group_activation_proof_cannot_authorize_member_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, manifest = _fixture(Path(directory))
            render = manifest["render"]
            assert isinstance(render, dict)
            evidence = render["cut_local_sync"]
            assert isinstance(evidence, dict)
            evidence["activation"] = "automatic_registered_stage_edit_group"
            evidence.pop("proof_sha256")
            evidence["proof_sha256"] = _canonical_sha256(evidence)
            artifacts.analysis_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report = evaluate_automatic_member_post_render_qa(
                artifacts,
                runner=_runner,
                perceptual_evaluator=lambda *_args, **_kwargs: (
                    _perceptual_payload()
                ),
            )

        cut_local = report.metrics["cut_local_sync"]
        self.assertFalse(report.passed)
        self.assertTrue(cut_local["proof_sha256_valid"])
        self.assertIn("activation_contract_invalid", cut_local["errors"])
        self.assertIn(
            "member_cut_local_sync_binding_invalid",
            {item["code"] for item in report.issues},
        )

    def test_tampered_timewarp_breaks_cut_local_plan_clock_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, manifest = _fixture(Path(directory))
            timewarps = manifest["local_timewarps"]
            assert isinstance(timewarps, list)
            first = timewarps[0]
            assert isinstance(first, dict)
            first["mean_slope"] = 1.01
            artifacts.analysis_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report = evaluate_automatic_member_post_render_qa(
                artifacts,
                runner=_runner,
                perceptual_evaluator=lambda *_args, **_kwargs: (
                    _perceptual_payload()
                ),
            )

        errors = report.metrics["cut_local_sync"]["errors"]
        codes = {item["code"] for item in report.issues}
        self.assertFalse(report.passed)
        self.assertIn("timewarp_contract_sha256_mismatch", errors)
        self.assertIn("source_0_timewarp_timewarp_mean_slope_mismatch", errors)
        self.assertIn("member_cut_local_sync_binding_invalid", codes)
        self.assertIn("member_cut_local_sync_required_gate_failed", codes)
        self.assertFalse(report.render_receipt["passed"])

    def test_cut_local_sync_p95_above_half_output_frame_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _manifest = _fixture(
                Path(directory),
                source_motion_shift_frames=1,
            )
            report = evaluate_automatic_member_post_render_qa(
                artifacts,
                runner=_runner,
                perceptual_evaluator=lambda *_args, **_kwargs: (
                    _perceptual_payload()
                ),
            )

        codes = {item["code"] for item in report.issues}
        cut_local = report.metrics["cut_local_sync"]
        self.assertFalse(report.passed)
        self.assertGreater(cut_local["residual_p95_frames"], 0.5)
        self.assertEqual(cut_local["errors"], [])
        self.assertIn(
            "member_cut_local_sync_residual_exceeds_half_frame",
            codes,
        )
        self.assertIn("member_cut_local_sync_required_gate_failed", codes)
        self.assertFalse(report.render_receipt["passed"])

    def test_missing_or_disabled_member_direct_block_fails_closed(self) -> None:
        for replacement in (None, {"enabled": False}):
            with self.subTest(replacement=replacement):
                with tempfile.TemporaryDirectory() as directory:
                    artifacts, manifest = _fixture(Path(directory))
                    manifest["member_direct_full_frame"] = replacement
                    artifacts.analysis_path.write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    report = evaluate_automatic_member_post_render_qa(
                        artifacts,
                        runner=_runner,
                        perceptual_evaluator=lambda *_args, **_kwargs: (
                            _perceptual_payload()
                        ),
                    )
            self.assertFalse(report.passed)
            self.assertIn(
                "member_contract_member_direct_present_enabled",
                {item["code"] for item in report.issues},
            )

    def test_missing_one_eligible_source_pixel_receipt_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _manifest = _fixture(Path(directory))
            report = evaluate_automatic_member_post_render_qa(
                artifacts,
                runner=_runner,
                perceptual_evaluator=lambda *_args, **_kwargs: (
                    _perceptual_payload((0, 1, 2))
                ),
            )

        self.assertFalse(report.passed)
        self.assertIn(
            "member_pixel_provenance_incomplete",
            {item["code"] for item in report.issues},
        )

    def test_clock_mutation_during_pixel_review_breaks_sha_binding(self) -> None:
        from crosscut.render_clock_artifact import render_clock_path
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _manifest = _fixture(Path(directory))
            clock = render_clock_path(artifacts.video_path)
            original = clock.read_bytes()

            def mutate(*_args: object, **_kwargs: object) -> dict[str, object]:
                clock.write_bytes(original.replace(b"identity", b"tampered"))
                self.assertEqual(clock.stat().st_size, len(original))
                return _perceptual_payload()

            report = evaluate_automatic_member_post_render_qa(
                artifacts, runner=_runner, perceptual_evaluator=mutate,
            )
        self.assertFalse(report.passed)
        self.assertIn("member_artifacts_changed_during_qa", {item["code"] for item in report.issues})

    def test_video_mutation_during_pixel_review_breaks_sha_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _manifest = _fixture(Path(directory))

            def mutate(*_args: object, **_kwargs: object) -> dict[str, object]:
                with artifacts.video_path.open("ab") as stream:
                    stream.write(b"changed")
                return _perceptual_payload()

            report = evaluate_automatic_member_post_render_qa(
                artifacts,
                runner=_runner,
                perceptual_evaluator=mutate,
            )

        self.assertFalse(report.passed)
        self.assertIn(
            "member_artifacts_changed_during_qa",
            {item["code"] for item in report.issues},
        )


class AutomaticMemberPipelineExposureTests(unittest.TestCase):
    def test_pass_promotes_delivery_and_cli_accepts_only_the_bound_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, manifest = _fixture(Path(directory))

            def evaluate(item, **kwargs):  # type: ignore[no-untyped-def]
                return evaluate_and_write_automatic_member_post_render_qa(
                    item,
                    runner=_runner,
                    perceptual_evaluator=lambda *_args, **_kwargs: (
                        _perceptual_payload()
                    ),
                    rendered_continuity_evaluator=(
                        _rendered_continuity_evaluator
                    ),
                    **kwargs,
                )

            with patch(
                "crosscut.pipeline.evaluate_and_write_automatic_member_post_render_qa",
                side_effect=evaluate,
            ):
                report_path = _enforce_automatic_member_post_render_qa(
                    manifest=manifest,
                    manifest_path=artifacts.analysis_path,
                    reporter=EventReporter(),
                )

            artifact = SimpleNamespace(
                manifest=manifest,
                output_paths=(artifacts.analysis_path, report_path, artifacts.video_path),
            )
            self.assertTrue(crosscut_cli._result_contains_exposable_video(artifact))
            self.assertEqual(manifest["artifact_role"], "delivery")
            self.assertTrue(manifest["post_render_qa"]["passed"])
            self.assertEqual(
                manifest["post_render_qa"]["contract"],
                AUTOMATIC_IVE_MEMBER_POST_RENDER_QA_CONTRACT,
            )

    def test_fail_persists_rejected_and_cli_refuses_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts, manifest = _fixture(Path(directory))
            manifest["member_direct_full_frame"] = {"enabled": False}
            artifacts.analysis_path.write_text(json.dumps(manifest), encoding="utf-8")

            def evaluate(item, **kwargs):  # type: ignore[no-untyped-def]
                return evaluate_and_write_automatic_member_post_render_qa(
                    item,
                    runner=_runner,
                    perceptual_evaluator=lambda *_args, **_kwargs: (
                        _perceptual_payload()
                    ),
                    rendered_continuity_evaluator=(
                        _rendered_continuity_evaluator
                    ),
                    **kwargs,
                )

            with patch(
                "crosscut.pipeline.evaluate_and_write_automatic_member_post_render_qa",
                side_effect=evaluate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "automatic IVE MEMBER final render blocked"
                ):
                    _enforce_automatic_member_post_render_qa(
                        manifest=manifest,
                        manifest_path=artifacts.analysis_path,
                        reporter=EventReporter(),
                    )

            persisted = json.loads(artifacts.analysis_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["artifact_role"], "rejected")
            self.assertFalse(
                persisted["app_result_exposure"]["video_output_exposed"]
            )
            leaked = SimpleNamespace(
                manifest=persisted,
                output_paths=(artifacts.analysis_path, artifacts.video_path),
            )
            with self.assertRaisesRegex(RuntimeError, "not exposed"):
                crosscut_cli._result_contains_exposable_video(leaked)


if __name__ == "__main__":
    unittest.main()
