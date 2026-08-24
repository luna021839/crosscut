from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.failed_candidate_replay import (
    CandidateReplayValidationError,
    ReplayBoundaryRepairResult,
    ReplayRepairContext,
    apply_current_boundary_repairs,
    _current_boundary_geometry_hook,
    reconstruct_failed_candidate,
    replay_failed_candidate,
)
from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
)
from crosscut.rendered_proxy_review import RenderedProxyReview, derive_proxy_render_plan


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FailedCandidateReplayTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        source0 = root / "source0.mp4"
        source1 = root / "source1.mp4"
        source0.write_bytes(b"source-zero")
        source1.write_bytes(b"source-one")
        sources = (source0.resolve(), source1.resolve())

        acquisition = {
            "content_integrity_contract_version": 1,
            "summary": {"status": "complete"},
            "selected": [
                {
                    "source_index": index,
                    "local_path": str(path),
                    "content_integrity": {
                        "algorithm": "sha256",
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    },
                }
                for index, path in enumerate(sources)
            ],
        }
        acquisition_path = root / "acquisition-manifest.json"
        acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

        source_look = {
            "brightness": 0.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "temperature_balance": 0.0,
            "identity": True,
        }
        choices = [
            {
                "segment_index": 0,
                "source_index": 0,
                "canonical_master_start": 3.5,
                "canonical_master_end": 4.5,
                "duration_seconds": 1.0,
                "source_local_start": 5.0,
                "source_local_end": 6.0,
                "source_duration_seconds": 1.0,
                "source_playback_rate": 1.0,
                "dissolve_entry_evidence": {"appearance_embedding_count": 0},
                "dissolve_exit_evidence": {"appearance_embedding_count": 0},
            },
            {
                "segment_index": 1,
                "source_index": 1,
                "canonical_master_start": 4.5,
                "canonical_master_end": 5.5,
                "duration_seconds": 1.0,
                "source_local_start": 7.0,
                "source_local_end": 8.0,
                "source_duration_seconds": 1.0,
                "source_playback_rate": 1.0,
                "dissolve_entry_evidence": {"appearance_embedding_count": 0},
                "dissolve_exit_evidence": {"appearance_embedding_count": 0},
            },
        ]
        final_segments = [
            {
                "segment_index": 0,
                "request_source_index": 0,
                "canonical_master_start": 3.5,
                "canonical_master_end": 4.5,
                "output_start_seconds": 0.0,
                "output_end_seconds": 1.0,
                "source_local_start": 5.0,
                "source_local_end": 6.0,
                "source_duration_seconds": 1.0,
                "source_playback_rate": 1.0,
                "source_clock_basis": "global_scalar_offset",
                "crop_mode": "full_frame",
                "editorial_transform_keyframe_count": 0,
                "source_look_adjustment": source_look,
            },
            {
                "segment_index": 1,
                "request_source_index": 1,
                "canonical_master_start": 4.5,
                "canonical_master_end": 5.5,
                "output_start_seconds": 1.0,
                "output_end_seconds": 2.0,
                "source_local_start": 7.0,
                "source_local_end": 8.0,
                "source_duration_seconds": 1.0,
                "source_playback_rate": 1.0,
                "source_clock_basis": "global_scalar_offset",
                "crop_mode": "editorial_camera",
                "editorial_transform_keyframe_count": 2,
                "source_look_adjustment": source_look,
            },
        ]
        face_segments = [
            {
                "segment_index": index,
                "source_index": index,
                "applied": False,
                "crop_keyframes": [],
            }
            for index in range(2)
        ]
        editorial_segments = [
            {
                "segment_index": 0,
                "source_index": 0,
                "applied": False,
                "reasons": ["not_selected"],
                "keyframes": [],
            },
            {
                "segment_index": 1,
                "source_index": 1,
                "applied": True,
                "reasons": ["test_zoom"],
                "keyframes": [
                    {
                        "time_seconds": 0.0,
                        "zoom": 1.0,
                        "center_x": 0.5,
                        "center_y": 0.5,
                        "roll_degrees": 0.0,
                    },
                    {
                        "time_seconds": 0.8,
                        "zoom": 1.05,
                        "center_x": 0.5,
                        "center_y": 0.5,
                        "roll_degrees": 0.0,
                    },
                ],
            },
        ]
        transition = {
            "kind": "hard_cut",
            "frame_count": 0,
            "duration_seconds": 0.0,
            "reason": "test_cross_source_hard_cut",
            "face_mismatch": None,
            "tier": "C",
            "appearance_min_cosine_similarity": None,
            "future_pose_max_mismatch": None,
            "future_silhouette_max_mismatch": None,
        }
        base_candidate = {
            "candidate_index": 0,
            "pre_render_score": 70.0,
            "visual_timeline": {"choices": choices},
            "face_match": {"applied_count": 0, "segments": face_segments},
            "editorial_motion": {
                "applied_segment_count": 1,
                "segments": editorial_segments,
            },
            "group_transitions": {
                "dissolve_count": 0,
                "same_source_hold_count": 0,
                "hard_cut_boundary_count": 1,
                "transitions": [transition],
            },
            "final_render_segments": final_segments,
        }
        review = {
            "analysis": "decoded_output_proxy_all_transition_quality_review_v2",
            "proxy": {
                "width": 854,
                "height": 480,
                "fps": 30.0,
                "decoded_frame_count": 60,
                "duration_seconds": 2.0,
            },
            "boundary_score": {"p10": 63.0, "issue_fraction": 1.0},
            "weighted_score": 67.0,
            "boundaries": [
                {
                    "boundary_index": 0,
                    "cut_frame_index": 30,
                    "outgoing_source_index": 0,
                    "incoming_source_index": 1,
                    "transition_kind": "hard_cut",
                    "transition_frame_count": 0,
                    "weighted_score": 63.0,
                    "passed": False,
                }
            ],
        }
        complete_candidates = []
        candidate_reviews = []
        for index in range(8):
            candidate = deepcopy(base_candidate)
            candidate["candidate_index"] = index
            complete_candidates.append(candidate)
            candidate_reviews.append(
                {
                    "candidate_index": index,
                    "error": None,
                    "review": deepcopy(review),
                }
            )
        report = {
            "analysis_basis": "sequential_decoded_proxy_candidate_rerank_v1",
            "passed": False,
            "requested_review_count": 8,
            "reviewed_candidate_count": 8,
            "minimum_required_candidate_count": 8,
            "minimum_candidate_count_satisfied": True,
            "successful_review_count": 8,
            "complete_candidate_count": 8,
            "exact_complete_candidate_count_satisfied": True,
            "exact_successful_decoded_review_count_satisfied": True,
            "source_index_mapping": {
                "request_index_domain": "request_source_index",
                "eligible_index_domain": "eligible_source_index",
                "render_plan_index_domain": "render_plan_sources_array_index",
                "domain_equivalences": {
                    "render_plan_sources_array_index": "request_source_index"
                },
                "eligible_to_request": [0, 1],
                "request_to_eligible": [0, 1],
                "eligible_sources": [
                    {
                        "eligible_source_index": index,
                        "request_source_index": index,
                        "source": str(path),
                    }
                    for index, path in enumerate(sources)
                ],
            },
            "complete_candidates": complete_candidates,
            "candidates": candidate_reviews,
        }
        report_path = root / "failed-selection.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, acquisition_path, source0

    def test_reconstructs_exact_timing_sources_transform_and_review_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, acquisition, _source = self._fixture(root)
            reconstructed = reconstruct_failed_candidate(
                report,
                acquisition,
                candidate_index=3,
                reference_source=0,
                output=root / "candidate3-qa-replay.mp4",
            )
            self.assertEqual(len(reconstructed.plan.sources), 2)
            self.assertEqual(len(reconstructed.plan.segments), 2)
            self.assertEqual(reconstructed.plan.output_frame_count, 60)
            self.assertEqual(reconstructed.plan.master_audio_start, 3.5)
            self.assertEqual(
                [item.source_index for item in reconstructed.plan.segments],
                [0, 1],
            )
            self.assertIsNone(reconstructed.plan.segments[0].editorial_transform)
            self.assertEqual(
                len(reconstructed.plan.segments[1].editorial_transform.keyframes),
                2,
            )
            self.assertTrue(
                reconstructed.reconstruction_checks[
                    "all_serialized_time_source_boundary_invariants_match"
                ]
            )

    def test_rejects_source_content_that_no_longer_matches_acquisition_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, acquisition, source = self._fixture(root)
            source.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                CandidateReplayValidationError, "size fingerprint mismatch"
            ):
                reconstruct_failed_candidate(
                    report,
                    acquisition,
                    candidate_index=0,
                    reference_source=0,
                    output=root / "candidate0-qa-replay.mp4",
                )

    def test_rejects_choice_and_final_time_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, acquisition, _source = self._fixture(root)
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["complete_candidates"][2]["final_render_segments"][0][
                "source_local_end"
            ] = 6.25
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                CandidateReplayValidationError, "source-local duration"
            ):
                reconstruct_failed_candidate(
                    report,
                    acquisition,
                    candidate_index=2,
                    reference_source=0,
                    output=root / "candidate2-qa-replay.mp4",
                )

    def test_repair_hook_may_change_pixel_tracks_but_not_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, acquisition, _source = self._fixture(root)
            reconstructed = reconstruct_failed_candidate(
                report,
                acquisition,
                candidate_index=1,
                reference_source=0,
                output=root / "candidate1-qa-replay.mp4",
            )
            context = ReplayRepairContext(
                report,
                acquisition,
                1,
                reconstructed.candidate_payload,
            )

            def pixel_hook(plan: RenderPlan, _context: ReplayRepairContext):
                track = EditorialTransformTrack(
                    (
                        EditorialTransformKeyframe(0.0, 1.0),
                        EditorialTransformKeyframe(0.8, 1.04),
                    ),
                    reason="test_replay_hook",
                )
                return ReplayBoundaryRepairResult(
                    plan=RenderPlan(
                        sources=plan.sources,
                        segments=(
                            plan.segments[0],
                            plan.segments[1].__class__(
                                source_index=plan.segments[1].source_index,
                                source_start=plan.segments[1].source_start,
                                duration=plan.segments[1].duration,
                                master_start=plan.segments[1].master_start,
                                editorial_transform=track,
                                look_adjustment=plan.segments[1].look_adjustment,
                            ),
                        ),
                        reference_source=plan.reference_source,
                        aspect_ratio=plan.aspect_ratio,
                        output=plan.output,
                        master_audio_start=plan.master_audio_start,
                        width=plan.width,
                        height=plan.height,
                        fps=plan.fps,
                        transitions=plan.transitions,
                    ),
                    audit={"applied": True},
                )

            def identity_current(plan: RenderPlan, _context: ReplayRepairContext):
                return ReplayBoundaryRepairResult(plan, {"applied": False})

            with patch(
                "crosscut.failed_candidate_replay._current_boundary_look_hook",
                identity_current,
            ), patch(
                "crosscut.failed_candidate_replay._current_boundary_geometry_hook",
                identity_current,
            ):
                repaired, audits = apply_current_boundary_repairs(
                    reconstructed.plan,
                    context,
                    extension_hooks=(("pixel", pixel_hook),),
                )
            self.assertEqual(repaired.output_frame_count, 60)
            self.assertEqual(len(audits), 3)
            self.assertAlmostEqual(
                repaired.segments[1].editorial_transform.keyframes[-1].zoom,
                1.04,
            )

            def cadence_hook(plan: RenderPlan, _context: ReplayRepairContext):
                first = plan.segments[0]
                changed = first.__class__(
                    source_index=first.source_index,
                    source_start=first.source_start,
                    duration=0.9,
                    master_start=first.master_start,
                    look_adjustment=first.look_adjustment,
                )
                return ReplayBoundaryRepairResult(
                    RenderPlan(
                        sources=plan.sources,
                        segments=(changed, plan.segments[1]),
                        reference_source=plan.reference_source,
                        aspect_ratio=plan.aspect_ratio,
                        output=plan.output,
                        master_audio_start=plan.master_audio_start,
                        width=plan.width,
                        height=plan.height,
                        fps=plan.fps,
                        transitions=plan.transitions,
                    ),
                    {},
                )

            with self.assertRaises(ValueError):
                apply_current_boundary_repairs(
                    reconstructed.plan,
                    context,
                    extension_hooks=(("cadence", cadence_hook),),
                )

    def test_builtin_geometry_overlays_only_exact_boundary_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, acquisition, _source = self._fixture(root)
            reconstructed = reconstruct_failed_candidate(
                report,
                acquisition,
                candidate_index=1,
                reference_source=0,
                output=root / "candidate1-qa-replay.mp4",
            )
            context = ReplayRepairContext(
                report,
                acquisition,
                1,
                reconstructed.candidate_payload,
            )
            choices = (
                SimpleNamespace(
                    source_index=0,
                    source_local_start=5.0,
                    source_local_end=6.0,
                    duration_seconds=1.0,
                    shot_id="0:shot",
                ),
                SimpleNamespace(
                    source_index=1,
                    source_local_start=7.0,
                    source_local_end=8.0,
                    duration_seconds=1.0,
                    shot_id="1:shot",
                ),
            )
            visual = SimpleNamespace(choices=choices)
            reason = "exact_frame_silhouette_scale_position_matched_hard_cut"
            tracks = (
                EditorialTransformTrack(
                    (
                        EditorialTransformKeyframe(0.0, 1.0),
                        EditorialTransformKeyframe(0.8, 1.04),
                    ),
                    reason=reason,
                ),
                EditorialTransformTrack(
                    (
                        EditorialTransformKeyframe(0.0, 1.04),
                        EditorialTransformKeyframe(0.8, 1.0),
                    ),
                    reason=reason,
                ),
            )
            editorial_segments = tuple(
                SimpleNamespace(reasons=(reason,), track=track)
                for track in tracks
            )
            editorial = SimpleNamespace(
                segments=editorial_segments,
                applied_count=2,
                boundary_match_count=1,
                affine_boundary_match_count=0,
                skipped_boundary_reasons=(),
                track_for=lambda index: tracks[index],
            )
            face = SimpleNamespace(
                applied_count=0,
                fallback_count=2,
                pan_crop_for=lambda _index: None,
            )
            affine = SimpleNamespace(
                decisions=(object(),),
                analyzed_count=1,
                eligible_count=0,
                exact_subject_pair_count=1,
                to_dict=lambda: {"rejection_counts": []},
            )
            with patch(
                "crosscut.failed_candidate_replay._deserialize_visual_timeline",
                return_value=visual,
            ), patch(
                "crosscut.failed_candidate_replay._geometry_source_contract",
                return_value=(
                    reconstructed.plan.sources,
                    (16 / 9, 16 / 9),
                    (0, 1),
                ),
            ), patch(
                "crosscut.face_match.build_face_match_plan", return_value=face
            ), patch(
                "crosscut.boundary_affine_plan.build_boundary_affine_plan",
                return_value=affine,
            ), patch(
                "crosscut.editorial_motion.build_editorial_motion_plan",
                return_value=editorial,
            ):
                result = _current_boundary_geometry_hook(
                    reconstructed.plan,
                    context,
                )
            self.assertEqual(result.audit["applied_boundary_indices"], [0])
            self.assertEqual(result.plan.segments[0].editorial_transform, tracks[0])
            self.assertEqual(result.plan.segments[1].editorial_transform, tracks[1])
            self.assertTrue(
                result.audit["source_cadence_audio_transitions_unchanged"]
            )

    def test_persistent_replay_is_marked_qa_only_and_writes_atomic_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, acquisition, _source = self._fixture(root)
            output = root / "candidate4-qa-replay.mp4"

            def fake_render_review(plan: RenderPlan, destination: Path):
                proxy = derive_proxy_render_plan(plan, destination)
                destination.write_bytes(b"fake-qa-proxy")
                return (
                    proxy,
                    RenderedProxyReview(
                        {
                            "analysis": "decoded_output_proxy_all_transition_quality_review_v2",
                            "proxy": {
                                "decoded_frame_count": 60,
                            },
                            "boundary_score": {
                                "p10": 64.0,
                                "issue_fraction": 0.5,
                            },
                            "weighted_score": 68.0,
                            "passed": False,
                        }
                    ),
                )

            artifact = replay_failed_candidate(
                report,
                acquisition,
                candidate_index=4,
                reference_source=0,
                output=output,
                render_review=fake_render_review,
            )
            self.assertTrue(artifact.video_path.is_file())
            self.assertTrue(artifact.audit_path.is_file())
            self.assertTrue(artifact.payload["production_use_forbidden"])
            self.assertEqual(
                artifact.payload["mode"],
                "exact_serialized_old_plan_fidelity_replay",
            )
            self.assertFalse(artifact.payload["qa_thresholds_weakened"])

            with self.assertRaisesRegex(
                CandidateReplayValidationError, "must contain 'qa-replay'"
            ):
                replay_failed_candidate(
                    report,
                    acquisition,
                    candidate_index=4,
                    reference_source=0,
                    output=root / "looks-like-final.mp4",
                    render_review=fake_render_review,
                )

    def test_professional_reference_profile_is_forwarded_and_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, acquisition, _source = self._fixture(root)
            profile = root / "five-reference-profile.json"
            profile.write_text('{"schema_version":1}', encoding="utf-8")
            received: list[Path] = []

            def fake_render_review(
                plan: RenderPlan,
                destination: Path,
                *,
                professional_reference_profile: Path,
            ):
                received.append(professional_reference_profile)
                proxy = derive_proxy_render_plan(plan, destination)
                destination.write_bytes(b"fake-relative-gated-qa-proxy")
                return (
                    proxy,
                    RenderedProxyReview(
                        {
                            "analysis": (
                                "decoded_output_proxy_domain_aware_quality_review_v3"
                            ),
                            "proxy": {"decoded_frame_count": 60},
                            "visible_boundary_score": {
                                "p10": 64.0,
                                "issue_fraction": 0.5,
                                "issue_count": 1,
                                "count_basis": "introduced_cross_source",
                            },
                            "boundary_domains": {
                                "introduced_cross_source": {"boundary_count": 2}
                            },
                            "weighted_score": 68.0,
                            "passed": False,
                            "professional_reference_gate": {"passed": False},
                        }
                    ),
                )

            artifact = replay_failed_candidate(
                report,
                acquisition,
                candidate_index=4,
                reference_source=0,
                output=root / "candidate4-relative-gate-qa-replay.mp4",
                professional_reference_profile=profile,
                render_review=fake_render_review,
            )
            self.assertEqual(received, [profile.resolve()])
            self.assertEqual(
                artifact.payload["professional_reference_profile"],
                {
                    "kind": "file",
                    "path": str(profile.resolve()),
                    "sha256": _sha256(profile),
                },
            )
            self.assertIn(
                "professional_reference_gate",
                artifact.payload["decoded_proxy_review"],
            )


if __name__ == "__main__":
    unittest.main()
