from dataclasses import replace
from pathlib import Path
import math
import shutil
import subprocess
import unittest
from unittest.mock import patch

from crosscut.shot_context_refinement import refine_shot_context_cameras,shot_context_shortlist
from crosscut.pipeline import (_review_first_group_bundle,_review_first_group_shot_context_bundle,
    _validate_review_first_group_bundle)
from crosscut.render import _piecewise_linear_expression
from crosscut.section_director import SectionPlan,SectionDirective,SectionRole
from crosscut.visual_score import VisualTimelinePlan
from crosscut.timeline import ShotType
import engine.tests.test_shot_context_camera as camera_tests
from engine.tests.test_shot_context_binding import motion_for
from engine.tests.test_boundary_subject_alignment import _choice
from types import SimpleNamespace


class ShotContextRefinementTests(unittest.TestCase):
    def fixture(self):
        f=camera_tests.ShotContextProposalTests();args=f.fixture();proposal=f.run_proposal(args)
        plan=args['plan']
        timeline=VisualTimelinePlan(tuple(replace(_choice(i,i,start=float(i),end=float(i+1)),
            shot_type=ShotType.CLOSE_UP,entry_shot_type=ShotType.CLOSE_UP,exit_shot_type=ShotType.CLOSE_UP,
            shot_scale=.8,entry_shot_scale=.8,exit_shot_scale=.8) for i in range(3)),1.)
        return plan,timeline,proposal

    def test_automatic_shortlist_promotes_real_producer_and_final_bundle_atomically(self):
        plan,timeline,proposal=self.fixture();motion=motion_for(plan)
        section=SectionPlan((SectionDirective(0,SectionRole.BUILD,0.,3.,.5,.5,1,4,15.,.4,.1,False,False),))
        request=SimpleNamespace(sources=plan.sources,reference_source=0,aspect_ratio=plan.aspect_ratio)
        bundle=_review_first_group_bundle(timeline=timeline,request=request,
            eligible_indices=(0,1,2),offsets=(0.,)*3,timewarps=(None,)*3,common_start=0.,
            section_plan=section,phrase_times=(0.,1.,2.,3.),output=plan.output,
            reporter=SimpleNamespace(log=lambda *a,**k:None),warnings=[])
        bundle=replace(bundle,render_plan=plan,editorial_motion_plan=motion,
            group_transition_plan=replace(bundle.group_transition_plan,transitions=plan.transitions))
        with patch('crosscut.shot_context_refinement.propose_shot_context_camera',return_value=proposal) as call:
            updated=_review_first_group_shot_context_bundle(bundle,scene_cut_times=((),)*3)
        call.assert_called_once()
        self.assertEqual(call.call_args.kwargs['segment_index'],1)
        self.assertEqual(updated.render_plan,proposal.plan)
        self.assertEqual(updated.visual_timeline,bundle.visual_timeline)
        self.assertEqual(updated.stage_edit_exact_shot_clock,bundle.stage_edit_exact_shot_clock)
        self.assertEqual(bundle.render_plan,plan)
        self.assertFalse(updated.candidate_lineage[-1]['quality_approved'])
        _validate_review_first_group_bundle(updated,eligible_indices=(0,1,2),offsets=(0.,)*3,
            timewarps=(None,)*3,common_start=0.)
        from crosscut.group_boundary_grammar_adapter import validate_group_boundary_grammar_plan
        from crosscut.pipeline import _source_index_mapping
        validate_group_boundary_grammar_plan(render_plan=updated.render_plan,
            source_index_mapping=_source_index_mapping(plan.sources,(0,1,2)),
            selected_timeline=updated.visual_timeline,source_scene_cut_times=((),)*3,
            timeline_review=updated.timeline_review,section_plan=updated.section_plan,
            editorial_motion=updated.editorial_motion_plan)

    def test_separated_shots_can_share_an_unchanged_neighbor_without_losing_binding(self):
        from crosscut.render import RenderSegment,RenderTransition
        from crosscut.shot_context_refinement import attach_shot_context_proposal
        from crosscut.post_render_qa import _bound_local_alignment_tracks
        import json
        f=camera_tests.ShotContextProposalTests();plan=f.fixture()['plan']
        plan=replace(plan,segments=tuple(RenderSegment(source,float(i),1.,float(i))
            for i,source in enumerate((0,1,2,1,0))),transitions=(RenderTransition(tier='C'),)*4)
        motion=motion_for(plan)
        for index in (1,3):
            proposal=f.run_proposal(dict(plan=plan,segment_index=index,scene_cut_times=((),)*3,experimental=True))
            motion=attach_shot_context_proposal(plan,motion,proposal);plan=proposal.plan
        self.assertEqual(_bound_local_alignment_tracks(motion.to_dict(),json.loads(json.dumps(plan.to_json()))),
            ({1,3},{0,1,2,3}))

    def test_no_retry_on_unavailable_or_changed_neighbor(self):
        plan,timeline,proposal=self.fixture()
        for exception in (ValueError('missing head'),OSError('source unavailable')):
            with patch('crosscut.shot_context_refinement.propose_shot_context_camera',side_effect=exception) as call:
                result=refine_shot_context_cameras(plan=plan,timeline=timeline,motion=motion_for(plan),scene_cut_times=((),)*3)
            call.assert_called_once();self.assertEqual(result.plan,plan)
            self.assertEqual(result.changed_segments,())
        changed=replace(proposal,plan=replace(proposal.plan,segments=(
            replace(proposal.plan.segments[0],source_start=.1),*proposal.plan.segments[1:])))
        with patch('crosscut.shot_context_refinement.propose_shot_context_camera',return_value=changed):
            result=refine_shot_context_cameras(plan=plan,timeline=timeline,motion=motion_for(plan),scene_cut_times=((),)*3)
        self.assertEqual(result.plan,plan)
        self.assertIn('fixed neighbor',result.attempts[0]['reason'])

    def test_shortlist_excludes_native_cuts_and_invalid_budgets_before_decode(self):
        plan,timeline,_=self.fixture()
        self.assertEqual(shot_context_shortlist(plan,timeline),(1,))
        timeline=replace(timeline,choices=(timeline.choices[0],
            replace(timeline.choices[1],internal_scene_cut_master_seconds=(1.5,)),timeline.choices[2]))
        self.assertEqual(shot_context_shortlist(plan,timeline),())
        for kwargs in ({'maximum_attempts':65},{'maximum_attempts':True},{'maximum_runtime_seconds':float('inf')}):
            with self.assertRaises(ValueError):
                refine_shot_context_cameras(plan=plan,timeline=timeline,motion=motion_for(plan),scene_cut_times=((),)*3,**kwargs)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_dense_camera_expression_decodes_and_preserves_all_keyframe_targets(self):
        positions=tuple((i/30.,.15+.12*math.sin(i*.13)) for i in range(300))
        expression=_piecewise_linear_expression(positions,1,time_variable='N/30')
        # Real expression evaluation, not a text-length or build assertion.
        result=subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
            f"nullsrc=s=16x16:r=30,geq=lum='255*({expression})':cb=128:cr=128",
            '-frames:v','300','-f','rawvideo','-pix_fmt','gray','pipe:1'],
            capture_output=True,timeout=15.)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        self.assertEqual(len(result.stdout),300*256)
        # gray conversion expands limited-range luma; compare in that domain.
        for index,(_,value) in enumerate(positions):
            expected=max(0,min(255,round((int(255*value)-16)*255/219)))
            self.assertLessEqual(abs(result.stdout[index*256]-expected),2,index)


if __name__=='__main__':unittest.main()
