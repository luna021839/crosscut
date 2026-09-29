"""Actual deterministic producer -> final QA, without invented learned proof."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.boundary_subject_alignment import build_boundary_subject_alignment_plan
from crosscut.editorial_motion import build_editorial_motion_plan
from crosscut.ive import AspectRatio
from crosscut.post_render_qa import (
    PostRenderQAConfig, _IssueCollector, _bound_local_alignment_tracks,
    _file_content_fingerprint, _retained_proxy_matches_render_plan, _transform_metrics,
    _rendered_montage_grammar_metrics,
)
from crosscut.render import RenderPlan, RenderSegment, RenderTransition
from crosscut.rendered_proxy_review import derive_proxy_render_plan
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_boundary_subject_alignment import _choice, _frame, _observations
from engine.tests.test_post_render_qa import _fixture_payloads, _install_valid_registered_montage_evidence


def produced_plan(root: Path):
    visual = VisualTimelinePlan((
        _choice(0, 0, start=0., end=1.),
        _choice(1, 1, start=1., end=2.)), 1.)
    sources = (root/'a.mp4', root/'b.mp4')
    transitions = (RenderTransition(tier='C'),)
    alignment = build_boundary_subject_alignment_plan(visual_timeline=visual,
        sources=sources, transitions=transitions, fps=30.,
        frame_provider=lambda index, _seconds: _frame(index),
        observation_provider=_observations(
            source0_box=(.35,.23,.225,.27), source1_box=(.40,.26,.20,.24)),
        source_resolutions=((1920,1080),(1920,1080)))
    motion = build_editorial_motion_plan(visual_timeline=visual,
        transitions=transitions, fps=30., boundary_subject_alignment_plan=alignment)
    plan = RenderPlan(sources=sources, reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE, output=root/'final.mp4', fps=30.,
        transitions=transitions, segments=tuple(
            RenderSegment(item.source_index, item.source_local_start,
                item.duration_seconds, item.canonical_master_start,
                editorial_transform=motion.track_for(item.segment_index))
            for item in visual.choices))
    return plan, motion


class CurrentGroupContractTests(unittest.TestCase):
    def test_deterministic_profile_does_not_require_fabricated_stage_edit_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, plan, selection = _fixture_payloads(Path(directory))
            analysis['request'] = {'group':'IVE', 'edit_mode':'group', 'source_mode':'automatic'}
            analysis['stage_edit_generation'] = {}
            _install_valid_registered_montage_evidence(analysis, plan, selection)
            analysis['professional_montage_grammar_profile'] = deepcopy(
                analysis['stage_edit_generation']['professional_montage_grammar_profile'])
            analysis['stage_edit_generation'] = None
            collector = _IssueCollector()
            metrics = _rendered_montage_grammar_metrics(analysis, plan, selection,
                PostRenderQAConfig(), collector, trusted_registered_stage_edit=False)
            self.assertTrue(metrics['reference_profile_identity_matches'])
            self.assertNotIn('production_rendered_montage_grammar_evidence_missing_or_invalid',
                [i.code for i in collector.issues])
            # If a learned producer DOES declare a profile, the deterministic
            # profile cannot paper over an invalid learned identity.
            analysis['stage_edit_generation'] = {'professional_montage_grammar_profile': None}
            bad = _IssueCollector()
            _rendered_montage_grammar_metrics(analysis, plan, selection,
                PostRenderQAConfig(), bad, trusted_registered_stage_edit=False)
            self.assertIn('production_rendered_montage_grammar_evidence_missing_or_invalid',
                [i.code for i in bad.issues])

    def test_incoming_settle_is_not_miscounted_as_new_pan_or_zoom(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, motion = produced_plan(Path(directory))
            serialized = json.loads(json.dumps(plan.to_json()))
            payload = motion.to_dict()
            self.assertEqual(motion.boundary_match_count, 1)
            self.assertEqual(motion.pan_event_count, 0)
            self.assertEqual(payload['authored_event_counts']['pan'], 1)
            collector = _IssueCollector()
            metrics = _transform_metrics({'editorial_motion': payload},
                serialized, PostRenderQAConfig(), collector)
            self.assertEqual(metrics['bound_local_alignment_segment_indices'], [1])
            self.assertGreater(max(abs(k.center_x-.5)
                for k in plan.segments[1].editorial_transform.keyframes), .025)
            self.assertEqual(collector.issues, [])
            forged = deepcopy(payload)
            forged['pan_event_count'] = 1
            bad = _IssueCollector()
            _transform_metrics({'editorial_motion': forged}, serialized, PostRenderQAConfig(), bad)
            self.assertIn('editorial_motion_pan_track_evidence', [i.code for i in bad.issues])

    def test_local_center_bound_requires_dense_report_matching_actual_track(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, motion = produced_plan(Path(directory))
            serialized = json.loads(json.dumps(plan.to_json()))
            for mutation in ('count', 'endpoint', 'track', 'no_report'):
                payload = deepcopy(motion.to_dict())
                if mutation == 'count':
                    payload['boundary_subject_alignment']['decisions'][0]['validated_output_frame_count'] = 0
                elif mutation == 'endpoint':
                    payload['boundary_subject_alignment']['decisions'][0]['incoming_state']['center_x'] = .5
                elif mutation == 'track':
                    payload['boundary_subject_alignment']['segments'][1]['keyframes'][0]['center_x'] = .5
                else:
                    payload['boundary_subject_alignment'] = None
                with self.subTest(mutation=mutation):
                    self.assertEqual(_bound_local_alignment_tracks(payload, serialized), (set(),set()))
                    collector = _IssueCollector()
                    _transform_metrics({'editorial_motion': payload}, serialized, PostRenderQAConfig(), collector)
                    self.assertIn('editorial_transform_safety_bound_exceeded', [i.code for i in collector.issues])

    def test_retained_proxy_requires_exact_plan_and_unchanged_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _motion = produced_plan(root)
            video = root/'run-review-proxy.mp4'
            video.write_bytes(b'explicit test artifact, not decoded video evidence')
            proxy_plan = video.with_suffix('.plan.json')
            proxy_plan.write_text(json.dumps(derive_proxy_render_plan(plan, video).to_json()))
            review = {'proxy': {'path':str(video), 'temporary_proxy_removed_after_review':False,
                'retained_artifact': {'contract':'review_first_retained_proxy_v1',
                    'video':_file_content_fingerprint(video),
                    'render_plan':_file_content_fingerprint(proxy_plan)}}}
            self.assertTrue(_retained_proxy_matches_render_plan(review, plan.to_json()))
            forged = deepcopy(review)
            forged['proxy']['retained_artifact']['video']['sha256'] = '0'*64
            self.assertFalse(_retained_proxy_matches_render_plan(forged, plan.to_json()))
            changed_plan = replace(plan, master_audio_start=.1)
            self.assertFalse(_retained_proxy_matches_render_plan(review, changed_plan.to_json()))
            video.write_bytes(b'changed')
            self.assertFalse(_retained_proxy_matches_render_plan(review, plan.to_json()))


if __name__ == '__main__':
    unittest.main()
