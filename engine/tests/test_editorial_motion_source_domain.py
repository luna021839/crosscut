"""Eligible-local production motion -> request-domain final QA receipts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from crosscut.editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from crosscut.post_render_qa import (_bound_local_alignment_tracks, _transform_metrics,
    _IssueCollector, PostRenderQAConfig)
from crosscut.shot_context_refinement import attach_shot_context_proposal, attach_coupled_shot_proposal
from engine.tests.test_shot_context_binding import ShotContextBindingTests, motion_for
from engine.tests.test_coupled_shot_camera import CoupledShotCameraTests
from engine.tests.test_post_render_qa_current_group_contract import produced_plan


DOMAIN = (0, 2, 3, 5)


def request_plan(plan):
    sources = [Path(f'/unused/excluded-{index}.mp4') for index in range(6)]
    for local, path in enumerate(plan.sources):
        sources[DOMAIN[local]] = path
    return replace(plan, sources=tuple(sources), reference_source=DOMAIN[plan.reference_source],
        segments=tuple(replace(item, source_index=DOMAIN[item.source_index]) for item in plan.segments))


def eligible_motion(motion):
    return replace(motion, eligible_request_indices=DOMAIN)


class EditorialMotionSourceDomainTests(unittest.TestCase):
    def test_single_context_survives_nonidentity_source_mapping_into_real_final_qa(self):
        before, proposal = ShotContextBindingTests().fixture()
        current, proposed = request_plan(before), request_plan(proposal.plan)
        proposal = replace(proposal, plan=proposed)
        local = motion_for(before)
        with self.assertRaisesRegex(ValueError, 'dense camera binding'):
            attach_shot_context_proposal(current, local, proposal)
        motion = attach_shot_context_proposal(current, eligible_motion(local), proposal)
        raw, plan = motion.to_dict(), json.loads(json.dumps(proposed.to_json()))
        self.assertEqual([item.source_index for item in motion.segments], [0, 1, 2])
        self.assertEqual([item['source_index'] for item in raw['segments']], [0, 2, 3])
        self.assertEqual(raw['source_index_domain'], 'request')
        self.assertEqual(raw['typed_source_index_domain'], 'eligible_local')
        self.assertEqual(raw['eligible_request_indices'], list(DOMAIN))
        self.assertEqual(raw['shot_context_cameras'][0]['source_index'], 2)
        self.assertEqual(_bound_local_alignment_tracks(raw, plan), ({1}, {0, 1}))
        collector = _IssueCollector()
        metrics = _transform_metrics({'editorial_motion': raw}, plan, PostRenderQAConfig(), collector)
        self.assertTrue(all(metrics['count_reconciliation_checks'].values()),
            [item.to_dict() for item in collector.issues])
        bad = deepcopy(raw)
        bad['segments'][1]['source_index'] = 1
        self.assertEqual(_bound_local_alignment_tracks(bad, plan), (set(), set()))

    def test_coupled_context_uses_request_receipts_and_does_not_map_them_twice(self):
        previous, _, pair = CoupledShotCameraTests().fixture()
        before = request_plan(replace(previous.plan, segments=tuple(
            replace(item, editorial_transform=None) for item in previous.plan.segments)))
        local = EditorialMotionPlan(tuple(EditorialMotionSegment(i, DOMAIN.index(item.source_index), None)
            for i, item in enumerate(before.segments)), 0, 0, 0, 0, 0,
            eligible_request_indices=DOMAIN)
        first = replace(previous, plan=request_plan(previous.plan))
        motion = attach_shot_context_proposal(before, local, first)
        pair = replace(pair, outgoing=replace(pair.outgoing, plan=request_plan(pair.plan)))
        changed = attach_coupled_shot_proposal(first.plan, motion, pair)
        raw = changed.to_dict()
        self.assertEqual(_bound_local_alignment_tracks(raw, json.loads(json.dumps(pair.plan.to_json()))),
            ({1, 2}, {0, 1}))
        self.assertEqual(raw['coupled_shot_cameras'][0]['incoming_source_index'], 3)
        self.assertEqual(raw['coupled_shot_cameras'][0]['outgoing']['source_index'], 2)

    def test_local_boundary_alignment_decisions_headroom_and_segments_map_together(self):
        plan, motion = produced_plan(Path('/unused'), paired_corrections=True)
        expected = _bound_local_alignment_tracks(motion.to_dict(), json.loads(json.dumps(plan.to_json())))
        raw = eligible_motion(motion).to_dict()
        mapped = request_plan(plan)
        self.assertEqual(_bound_local_alignment_tracks(raw, json.loads(json.dumps(mapped.to_json()))), expected)
        alignment = raw['boundary_subject_alignment']
        self.assertEqual([item['source_index'] for item in alignment['source_resolution_headroom']], [0, 2])
        self.assertEqual([item['source_index'] for item in alignment['segments']], [0, 2, 0])
        for item in alignment['decisions']:
            self.assertEqual(item['outgoing_source_index'], mapped.segments[item['outgoing_segment_index']].source_index)
            self.assertEqual(item['incoming_source_index'], mapped.segments[item['incoming_segment_index']].source_index)
        self.assertEqual(motion.segments[1].source_index, 1)
        self.assertEqual(motion.boundary_subject_alignment_plan.segments[1].source_index, 1)

    def test_other_source_alignment_cannot_authorize_the_same_numeric_camera(self):
        plan, motion = produced_plan(Path('/unused'), paired_corrections=True)
        raw, mapped = eligible_motion(motion).to_dict(), request_plan(plan)
        serialized = json.loads(json.dumps(mapped.to_json()))
        self.assertTrue(_bound_local_alignment_tracks(raw, serialized)[0])
        for kind in ('segment', 'decision'):
            bad = deepcopy(raw)
            if kind == 'segment':
                for item in bad['boundary_subject_alignment']['segments']:
                    item['source_index'] = 5
            else:
                for item in bad['boundary_subject_alignment']['decisions']:
                    item['outgoing_source_index'] = 5
                    item['incoming_source_index'] = 5
            with self.subTest(kind=kind):
                self.assertEqual(_bound_local_alignment_tracks(bad, serialized), (set(), set()))

    def test_mapping_is_immutable_unique_and_cannot_cover_a_missing_local_source(self):
        motion = motion_for(ShotContextBindingTests().fixture()[0])
        for mapping in ([0, 2, 3], (0, 2, 2), (0, True, 3), (0, 2), (0, -1, 3)):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                replace(motion, eligible_request_indices=mapping)
