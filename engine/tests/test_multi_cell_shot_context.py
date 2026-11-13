from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
import unittest
from unittest.mock import patch

from crosscut.render import RenderSegment,RenderTransition,segment_frame_counts
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.shot_context_binding import build_shot_context_binding,shot_context_binding_valid,_digest,RUN_CONTRACT
from crosscut.shot_context_refinement import (attach_shot_context_proposal,multi_cell_shot_context_shortlist,
    refine_shot_context_cameras,shot_context_shortlist)
from crosscut.post_render_qa import _bound_local_alignment_tracks,_transform_metrics,_IssueCollector,PostRenderQAConfig
from crosscut.visual_score import VisualTimelinePlan
from crosscut.timeline import ShotType
from engine.tests import test_shot_context_camera as fixtures
from engine.tests.test_shot_context_binding import motion_for
from engine.tests.test_boundary_subject_alignment import _choice


class MultiCellShotContextTests(unittest.TestCase):
    def fixture(self):
        helper=fixtures.ShotContextProposalTests();args=helper.fixture();plan=args['plan']
        plan=replace(plan,segments=(plan.segments[0],RenderSegment(1,1.,.6,1.),
            RenderSegment(1,1.6,.4,1.6),plan.segments[2]),transitions=(RenderTransition(tier='C'),)*3)
        args.update(plan=plan,last_segment_index=2)
        return helper,args

    def timeline(self,plan):
        return VisualTimelinePlan(tuple(replace(_choice(i,s.source_index,start=s.master_start,end=s.master_start+s.duration),
            shot_type=ShotType.CLOSE_UP,entry_shot_type=ShotType.CLOSE_UP,exit_shot_type=ShotType.CLOSE_UP,
            shot_scale=.8,entry_shot_scale=.8,exit_shot_scale=.8)
            for i,s in enumerate(plan.segments)),1.)

    def test_continuous_camera_does_not_reset_at_an_artificial_beat_cell(self):
        helper,args=self.fixture();before=args['plan'];proposal=helper.run_proposal(args)
        self.assertEqual(proposal.end_segment_index,2)
        self.assertEqual(proposal.native_frame_ids,tuple(range(30,60)))
        count=segment_frame_counts(before);offset=0
        for index in (1,2):
            for n in range(count[index]):
                full=proposal.camera.track.evaluate((offset+n)/before.fps)
                actual=proposal.plan.segments[index].editorial_transform.evaluate(n/before.fps)
                self.assertEqual(replace(full,time_seconds=n/before.fps),actual)
            offset+=count[index]
        self.assertGreater(proposal.plan.segments[2].editorial_transform.evaluate(0.).zoom,1.25)
        for index in (0,3):self.assertIs(proposal.plan.segments[index],before.segments[index])
        clocks={i:VerifiedRenderFrameClock(Fraction(30),3000,'a'*64) for i in range(3)}
        old=native_frame_clocks_for_output(before,clocks,range(before.output_frame_count))
        new=native_frame_clocks_for_output(proposal.plan,clocks,range(before.output_frame_count))
        self.assertEqual(old,new)

    def test_atomic_multi_cell_binding_counts_only_external_cut_matches(self):
        helper,args=self.fixture();before=args['plan'];proposal=helper.run_proposal(args)
        motion=attach_shot_context_proposal(before,motion_for(before),proposal)
        raw=motion.to_dict();plan=json.loads(json.dumps(proposal.plan.to_json()))
        self.assertEqual(raw['shot_context_cameras'][0]['contract'],RUN_CONTRACT)
        self.assertEqual(_bound_local_alignment_tracks(raw,plan),({1,2},{0,2}))
        self.assertEqual(motion.boundary_match_count,2)
        collector=_IssueCollector();metrics=_transform_metrics({'editorial_motion':raw},plan,PostRenderQAConfig(),collector)
        self.assertTrue(all(metrics['count_reconciliation_checks'].values()),[i.to_dict() for i in collector.issues])
        self.assertNotIn('editorial_motion_shot_context_bindings_complete',[i.code for i in collector.issues])
        for index in (1,2):
            broken=deepcopy(raw);broken['segments'][index]['keyframes']=[]
            self.assertEqual(_bound_local_alignment_tracks(broken,plan),(set(),set()))

    def test_shortlist_uses_a_linear_native_run_not_source_equality_only(self):
        _,args=self.fixture();plan=args['plan'];timeline=self.timeline(plan)
        self.assertEqual(shot_context_shortlist(plan,timeline),())
        self.assertEqual(multi_cell_shot_context_shortlist(plan,timeline),((1,2),))
        broken=replace(plan,segments=(*plan.segments[:2],replace(plan.segments[2],source_start=1.65),plan.segments[3]))
        self.assertEqual(multi_cell_shot_context_shortlist(broken,self.timeline(broken)),())
        native_cut=replace(timeline,choices=(*timeline.choices[:2],
            replace(timeline.choices[2],internal_scene_cut_master_seconds=(1.7,)),timeline.choices[3]))
        self.assertEqual(multi_cell_shot_context_shortlist(plan,native_cut),())

    def test_same_budget_auto_stage_covers_a_previously_omitted_run(self):
        helper,args=self.fixture();plan=args['plan'];proposal=helper.run_proposal(args)
        with patch('crosscut.shot_context_refinement.propose_shot_context_camera',return_value=proposal) as call:
            result=refine_shot_context_cameras(plan=plan,timeline=self.timeline(plan),motion=motion_for(plan),
                scene_cut_times=((),)*3,include_multi_cell=True)
        call.assert_called_once()
        self.assertEqual(call.call_args.kwargs['last_segment_index'],2)
        self.assertEqual(result.changed_segments,(1,2))
        self.assertEqual(result.plan,proposal.plan)
        self.assertFalse(result.to_dict()['quality_approved'])

    def test_unproven_multi_cell_cost_is_not_added_to_normal_app_stage(self):
        _,args=self.fixture();plan=args['plan']
        with patch('crosscut.shot_context_refinement.propose_shot_context_camera') as propose:
            result=refine_shot_context_cameras(plan=plan,timeline=self.timeline(plan),
                motion=motion_for(plan),scene_cut_times=((),)*3)
        propose.assert_not_called()
        self.assertEqual(result.changed_segments,())

    def test_native_cut_or_discontinuous_clock_cannot_hide_at_a_cell_split(self):
        helper,args=self.fixture()
        for cut in (1.6,1.7):
            with self.assertRaisesRegex(ValueError,'native camera cut'):
                helper.run_proposal(args,scene_cut_times=((),(cut,),()))
        plan=args['plan']
        changed=replace(plan,segments=(*plan.segments[:2],replace(plan.segments[2],source_start=1.65),plan.segments[3]))
        with self.assertRaisesRegex(ValueError,'source run'):helper.run_proposal(args,plan=changed)

    def test_rehashed_partial_span_and_invented_internal_cut_credit_fail(self):
        helper,args=self.fixture();proposal=helper.run_proposal(args);good=build_shot_context_binding(proposal)
        for key,value in (('last_segment_index',1),('last_segment_index',3),('boundary_indices',[0,1,2]),
                ('contract',[]),('native_frame_ids',[30]*30),('frame_count',18)):
            bad=deepcopy(good);bad[key]=value
            bad['content_sha256']=_digest({k:v for k,v in bad.items() if k!='content_sha256'})
            self.assertFalse(shot_context_binding_valid(bad,proposal.plan),key)
        changed=replace(proposal.plan,segments=(*proposal.plan.segments[:2],
            replace(proposal.plan.segments[2],editorial_transform=None),proposal.plan.segments[3]))
        self.assertFalse(shot_context_binding_valid(good,changed))

    def test_fractional_cell_lengths_keep_every_output_state_and_source_frame(self):
        helper,args=self.fixture();plan=args['plan']
        plan=replace(plan,fps=29.97,segments=(plan.segments[0],RenderSegment(1,1.,.615,1.),
            RenderSegment(1,1.615,.385,1.615),plan.segments[3]))
        proposal=helper.run_proposal(args,plan=plan)
        self.assertEqual(sum(segment_frame_counts(plan)[1:3]),proposal.camera.frame_count)
        self.assertTrue(shot_context_binding_valid(build_shot_context_binding(proposal),proposal.plan))


if __name__=='__main__':unittest.main()
