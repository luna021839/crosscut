from dataclasses import replace
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import tempfile
import shutil
import subprocess

import numpy as np

from crosscut.ive import AspectRatio
from crosscut.joint_proposal_domain import _requirement_covering_path
from crosscut.joint_switch_requirements import SourceSwitchRequirement, SourceSwitchCounter, SourceSwitchSearchBudget
from crosscut.prescribed_render_geometry import PrescribedGeometrySampler
from crosscut.render_coverage import render_coverage_payload
from crosscut.render import RenderPlan, RenderSegment, EditorialTransformKeyframe, EditorialTransformTrack
from crosscut.rendered_proxy_review import (
    _boundary_score, _boundary_evidence, _boundary_domain_summary, _issue_reasons,
    INTRODUCED_CROSS_SOURCE_DOMAIN, ProxyReviewConfig,
)
from crosscut.review_first_runtime import ReviewFirstRuntime, ReviewFirstStatus, finalize_review_first_runtime_payload, artifact_disposition_payload_valid
from crosscut.review_first_runtime import classify_review_first_qa
from crosscut.post_render_qa import _probe_and_decode, _validate_ffprobe_against_plan, _IssueCollector, PostRenderQAConfig
from crosscut.stage_edit_visual_candidate_adapter import dynamic_crop_v3_deterministic_geometry_override_indices
from crosscut.timeline import NoCompleteTimelinePath
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor, build_prescribed_visual_timeline_candidate_set, materialize_visual_timeline_candidate
from engine.tests.test_joint_timeline import candidate


class ProAuditRegressionTests(unittest.TestCase):
    def test_missing_geometry_is_unknown_not_a_perfect_continuity_match(self):
        self.assertIsNone(_boundary_score({}))
        appearance = {"sharpness_discontinuity": 0., "exposure_discontinuity": 0., "color_shock": 0., "structure_shock": 0.}
        evidence = _boundary_evidence(appearance)
        self.assertIsNone(evidence["quality_score"])
        self.assertLess(evidence["weighted_coverage"], 1.)
        self.assertEqual(evidence["status"], "UNKNOWN_SUBJECT_CONTINUITY")
        self.assertIn("subject_continuity_unobserved", _issue_reasons(appearance, _boundary_score(appearance), ProxyReviewConfig()))

    def test_empty_or_missing_boundary_measurements_cannot_be_aggregated_to_100(self):
        for boundaries, status in (
            ([], "NOT_APPLICABLE"),
            ([{"boundary_index": 0, "review_domain": INTRODUCED_CROSS_SOURCE_DOMAIN,
               "weighted_score": None, "passed": True}], "INCOMPLETE"),
        ):
            with self.subTest(status=status):
                summary = _boundary_domain_summary(
                    boundaries, domain=INTRODUCED_CROSS_SOURCE_DOMAIN, config=ProxyReviewConfig(),
                )
                self.assertIsNone(summary["score"]["p10"])
                self.assertEqual(summary["score"]["evidence_status"], status)
                self.assertEqual(summary["score"]["distribution"]["count"], 0)
                self.assertFalse(summary["passed"])

    def test_unobserved_boundary_score_survives_selection_and_partial_reedit(self):
        from crosscut.pipeline import _decoded_proxy_rank
        from crosscut.rendered_candidate_selection import RenderedCandidate, RenderedCandidateReview
        plan = RenderPlan(
            sources=(Path("source.mp4"),), segments=(RenderSegment(0, 0., 1., 0.),),
            reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"), width=1920, height=1080, fps=30.,
        )
        payload = {"passed": False, "weighted_score": 20.,
                   "visible_boundary_score": {"p10": None, "issue_fraction": 1.}}
        reviewed = RenderedCandidateReview(RenderedCandidate(0, plan, 20.), payload)
        self.assertIsNone(reviewed.to_dict(selected=True)["boundary_score_p10"])
        self.assertEqual(_decoded_proxy_rank(payload, pre_render_score=20.)[4], -float("inf"))

    def test_prescribed_geometry_changes_actual_observed_pixels_at_each_time(self):
        frame = np.repeat(np.arange(160, dtype=np.uint8)[None, :, None], 90, axis=0)
        frame = np.repeat(frame, 3, axis=2)
        track = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.), EditorialTransformKeyframe(2., 1.3)))
        segment = RenderSegment(0, 10., 2., 0., editorial_transform=track)
        sampler = PrescribedGeometrySampler(SimpleNamespace(frame_at=lambda _: frame, frame_rate=30),
            segment, master_start=0.)
        np.testing.assert_array_equal(sampler.frame_at(10.), frame)
        zoomed = sampler.frame_at(12.)
        self.assertEqual(zoomed.shape, frame.shape)
        self.assertGreater(int(zoomed[45, 0, 0]), 15)
        self.assertLess(int(zoomed[45, -1, 0]), 145)
        self.assertEqual(int(frame[45, 0, 0]), 0)

    def test_identity_placeholder_does_not_override_learned_v3_geometry(self):
        identity = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.),))
        substantive = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.2),))
        plan = RenderPlan(sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
            segments=(RenderSegment(0, 0., 2., 0., editorial_transform=identity),
                      RenderSegment(1, 2., 2., 2., editorial_transform=substantive)),
            reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("/tmp/out.mp4"), master_audio_start=0., width=160, height=90, fps=30.)
        self.assertEqual(dynamic_crop_v3_deterministic_geometry_override_indices(plan), (1,))

    def test_prescribed_builder_and_materializer_use_final_render_clock_not_alignment(self):
        descriptor = FrameVisualDescriptor(quality_score=.8, composition_score=.8,
            shot_scale=.4, shot_type=ShotType.MEDIUM, face_count=0,
            structure_embedding=(0.,) * 96, motion=(0., 0.))
        sampled = []
        class Sampler:
            frame_rate = 30.
            def __init__(self, path, **kwargs): self.path = path
            def frame_at(self, seconds):
                sampled.append((self.path, seconds))
                return np.full((90, 160, 3), 100, dtype=np.uint8)
            def close(self): pass
        segment = RenderSegment(0, 10., .1, 0., source_duration=.102)
        with patch("crosscut.visual_score._VideoSampler", Sampler), patch(
            "crosscut.visual_score.describe_frame", return_value=descriptor
        ), patch("crosscut.visual_score._selected_dissolve_evidence", side_effect=AssertionError("raw dissolve forbidden")):
            candidates = build_prescribed_visual_timeline_candidate_set(
                sources=(Path("a.mp4"), Path("b.mp4")), canonical_spans=((4., 4.1),),
                prescribed_source_indices=(0,), prescribed_render_segments=(segment,),
                audio_offsets=(0., 0.), sync_scores=(.9, .9), scene_cut_times=((), ()))
            timeline = materialize_visual_timeline_candidate(candidates)
        self.assertTrue(sampled)
        self.assertTrue(all(path == Path("a.mp4") and 10. <= seconds <= 10.102 for path, seconds in sampled))
        self.assertEqual(timeline.choices[0].source_local_start, 10.)
        self.assertAlmostEqual(timeline.choices[0].source_local_end, 10.102)
        self.assertEqual(timeline.geometry_basis, "prescribed_render_geometry_v1")
        self.assertEqual(timeline.source_clock_basis, "prescribed_render_segment_clock_v1")

    def test_complete_selected_interval_does_not_claim_complete_original_song(self):
        plan = RenderPlan(sources=(Path("a.mp4"),), segments=(RenderSegment(0, 30., 8., 0.),),
            reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE, output=Path("out.mp4"),
            master_audio_start=30., width=160, height=90, fps=30.)
        coverage = render_coverage_payload(plan, common_start=30., common_end=38.)
        self.assertTrue(coverage["covers_common_master_interval"])
        self.assertFalse(coverage["full_song_delivery_claim_allowed"])
        self.assertEqual(coverage["canonical_song_coverage_status"], "UNKNOWN")
        self.assertFalse(render_coverage_payload(plan, common_start=0., common_end=38.)["covers_common_master_interval"])

    def test_many_expired_windows_do_not_accumulate_exponential_states(self):
        rows = tuple(tuple(candidate(i, source, .8) for source in ("A", "B")) for i in range(26))
        requirements = tuple(SourceSwitchRequirement(2.*i, 2.*i+2., 1) for i in range(13))
        budget = SourceSwitchSearchBudget(maximum_states=64, maximum_expansions=10000)
        path = _requirement_covering_path(rows, requirements, None, fps=30., maximum_offset_frames=0, budget=budget)
        self.assertEqual(len(path), len(rows))
        self.assertLessEqual(budget.peak_states, 64)

    def test_descriptor_work_and_deadline_are_bounded_before_pixel_calls(self):
        rows = tuple(tuple(candidate(i, s, .8) for s in ("A", "B")) for i in range(4))
        with self.assertRaises(NoCompleteTimelinePath) as caught:
            SourceSwitchCounter((SourceSwitchRequirement(0., 4., 2),), rows, fps=30., maximum_offset_frames=8,
                                budget=SourceSwitchSearchBudget(maximum_expansions=2))
        self.assertEqual(caught.exception.reason_code, "section_source_switch_work_budget_exhausted")
        with self.assertRaises(NoCompleteTimelinePath):
            SourceSwitchSearchBudget(deadline_monotonic_seconds=monotonic() - 1)

    def test_final_runtime_clock_includes_final_qa(self):
        runtime = ReviewFirstRuntime(enabled=True)
        payload = runtime.to_dict()
        finalize_review_first_runtime_payload(payload, ReviewFirstStatus.REVIEW_READY)
        self.assertTrue(payload["elapsed_includes_final_qa"])
        self.assertEqual(payload["status"], "REVIEW_READY")
        self.assertGreaterEqual(payload["elapsed_seconds"], 0.)

    def test_status_label_alone_cannot_authorize_a_native_result(self):
        self.assertFalse(artifact_disposition_payload_valid({"status": "REVIEW_READY"}, role="review_ready"))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_silent_decodable_mp4_is_never_exposed_as_review_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "silent.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            "color=blue:s=32x32:r=30:d=0.5", "-an", "-c:v", "libx264", str(video)],
                           check=True, timeout=10)
            collector = _IssueCollector()
            config = PostRenderQAConfig()
            checks = _probe_and_decode(video, config=config,
                runner=lambda command, timeout: subprocess.run(command, timeout=timeout, capture_output=True, text=True),
                collector=collector)
            _validate_ffprobe_against_plan(checks, {"render": {"video_codec": "libx264"}},
                {"width": 32, "height": 32, "fps": 30., "output_frame_count": 15, "render_duration": .5}, config, collector)
            codes = {issue.code for issue in collector.issues}
            self.assertIn("final_media_exactly_one_audio_stream", codes)
            self.assertNotIn("full_decode_failed", codes)
            result = classify_review_first_qa({"passed": False, "issues": [{"code": code} for code in codes]},
                content_binding_passed=True, review_first_enabled=True, member=False)
            self.assertFalse(result.video_output_exposed)

    def test_every_audited_integrity_error_is_fail_closed(self):
        codes = ("final_media_exactly_one_audio_stream", "final_media_audio_duration_matches_video",
                 "final_media_decoded_frame_count_matches", "final_media_duration_matches", "final_media_fps_matches",
                 "artifact_binding_render_plan_output_matches", "artifact_binding_standalone_selection_matches_analysis",
                 "plan_invariant_final_render_segments_match_plan", "plan_invariant_proxy_boundary_schedule_matches_plan",
                 "member_contract_plan_output_path", "member_contract_request_plan_sources_exact",
                 "member_look_calibration_provenance_invalid", "member_look_calibration_not_bound_to_render_plan",
                 "full_decode_failed", "artifact_content_changed_during_qa", "production_sync_residual_exceeds_half_frame")
        for code in codes:
            for member in (False, True):
                with self.subTest(code=code, member=member):
                    result = classify_review_first_qa({"passed": False, "issues": [{"code": code}]},
                        content_binding_passed=True, review_first_enabled=True, member=member)
                    self.assertFalse(result.video_output_exposed)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_actual_corrupt_media_probe_and_decoder_failure_cannot_be_exposed(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "broken.mp4"
            video.write_bytes(b"deliberately invalid mp4 fixture")
            collector = _IssueCollector()
            _probe_and_decode(video, config=PostRenderQAConfig(),
                runner=lambda command, timeout: subprocess.run(command, timeout=timeout, capture_output=True, text=True),
                collector=collector)
            self.assertTrue(collector.issues)
            disposition = classify_review_first_qa({"passed": False,
                "issues": [{"code": issue.code} for issue in collector.issues]},
                content_binding_passed=True, review_first_enabled=True, member=False)
            self.assertFalse(disposition.video_output_exposed)
