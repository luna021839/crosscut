from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from crosscut.editorial_motion import EditorialMotionPlan,EditorialMotionSegment
from crosscut.post_render_qa import _bound_local_alignment_tracks,_transform_metrics,_IssueCollector,PostRenderQAConfig
from crosscut.shot_context_binding import (build_shot_context_binding,shot_context_binding_valid,
    rebind_unchanged_shot_context,_digest)
from crosscut.render import RenderSegment,RenderTransition
from crosscut.shot_context_refinement import attach_shot_context_proposal,both_boundary_framings_improve
import engine.tests.test_shot_context_camera as camera_tests


def motion_for(plan):
    return EditorialMotionPlan(tuple(EditorialMotionSegment(i,s.source_index,s.editorial_transform,())
        for i,s in enumerate(plan.segments)),0,0,0,0,0)


class ShotContextBindingTests(unittest.TestCase):
    def fixture(self):
        fixture=camera_tests.ShotContextProposalTests()
        args=fixture.fixture()
        return args['plan'],fixture.run_proposal(args)

    def test_separate_whole_shot_receipt_reaches_final_qa_without_v5_header(self):
        before,proposal=self.fixture()
        motion=attach_shot_context_proposal(before,motion_for(before),proposal)
        raw=motion.to_dict();plan=json.loads(json.dumps(proposal.plan.to_json()))
        self.assertIsNone(raw['boundary_subject_alignment'])
        self.assertEqual(_bound_local_alignment_tracks(raw,plan),({1},{0,1}))
        collector=_IssueCollector()
        metrics=_transform_metrics({'editorial_motion':raw},plan,PostRenderQAConfig(),collector)
        self.assertTrue(all(metrics['count_reconciliation_checks'].values()),metrics)
        self.assertEqual(metrics['bound_shot_context_segment_indices'],[1])
        with self.assertRaises(TypeError):motion.shot_context_cameras[0]['frame_count']=0

    def test_camera_clock_context_and_provenance_cannot_be_tampered_or_relabelled(self):
        _,proposal=self.fixture();good=build_shot_context_binding(proposal)
        for key,value in (('contract','exact_cut_frame_shot_adaptive_subject_alignment_v5'),
                ('professional_quality_approved',True),('identity_or_gesture_certified',True),
                ('native_frame_ids',[31]*30),('frame_count',29),('maximum_center_distance',.11)):
            bad=deepcopy(good);bad[key]=value
            bad['content_sha256']=_digest({k:v for k,v in bad.items() if k!='content_sha256'})
            self.assertFalse(shot_context_binding_valid(bad,proposal.plan),key)
        for mutate in (lambda p:replace(p,master_audio_start=.1),
                lambda p:replace(p,segments=(replace(p.segments[0],source_start=.1),*p.segments[1:])),
                lambda p:replace(p,width=1280,height=720)):
            self.assertFalse(shot_context_binding_valid(good,mutate(proposal.plan)))
        bad=deepcopy(good);bad['head_evidence'][10]['head_envelope_xywh']=[0.,0.,.8,.8]
        bad['content_sha256']=_digest({k:v for k,v in bad.items() if k!='content_sha256'})
        self.assertFalse(shot_context_binding_valid(bad,proposal.plan))

    def test_malformed_clock_duplicate_missing_motion_or_dropped_row_has_no_authority(self):
        before,proposal=self.fixture()
        motion=attach_shot_context_proposal(before,motion_for(before),proposal).to_dict()
        plan=json.loads(json.dumps(proposal.plan.to_json()))
        corruptions=[]
        missing=deepcopy(motion);missing['segments'][1]['keyframes']=[];corruptions.append(missing)
        duplicate=deepcopy(motion);duplicate['shot_context_cameras']*=2;corruptions.append(duplicate)
        clock=deepcopy(motion);clock['shot_context_cameras'][0]['source_clock']['frame_rate']='1/0'
        row=clock['shot_context_cameras'][0]
        row['content_sha256']=_digest({k:v for k,v in row.items() if k!='content_sha256'});corruptions.append(clock)
        for bad in corruptions:
            self.assertEqual(_bound_local_alignment_tracks(bad,plan),(set(),set()))
            collector=_IssueCollector()
            _transform_metrics({'editorial_motion':bad},plan,PostRenderQAConfig(),collector)
            self.assertIn('editorial_motion_shot_context_bindings_complete',[i.code for i in collector.issues])

    def test_improved_position_cannot_buy_worse_scale_or_other_cut(self):
        _,proposal=self.fixture()
        self.assertTrue(both_boundary_framings_improve(proposal))
        rows=deepcopy(proposal.boundary_framing)
        rows[1]['after'][0]=rows[1]['before'][0]+.03
        self.assertFalse(both_boundary_framings_improve(replace(proposal,boundary_framing=rows)))
        rows[1]['after']=[None,0.]
        self.assertFalse(both_boundary_framings_improve(replace(proposal,boundary_framing=rows)))

    def test_geometric_success_with_redetected_pixel_regression_is_not_admitted(self):
        before,proposal=self.fixture()
        rows=deepcopy(proposal.pixel_boundary_framing)
        # Actual XOXZ B53/B54 counterexample: analytic projection improved,
        # but the decoded face measurement did not. Do not fix its receipt.
        for row,old,new in zip(rows,((.030813,.008365),(.004989,.002829)),
                ((.047933,.020397),(.036784,.014873))):
            row.update(before=list(old),after=list(new))
        bad=replace(proposal,pixel_boundary_framing=rows)
        self.assertFalse(both_boundary_framings_improve(bad))
        with self.assertRaisesRegex(ValueError,'both boundary'):
            attach_shot_context_proposal(before,motion_for(before),bad)

    def test_subpixel_redetection_jitter_does_not_erase_large_other_cut_improvement(self):
        _,proposal=self.fixture()
        rows=deepcopy(proposal.pixel_boundary_framing)
        rows[1].update(before=[.0053275,.1572266],after=[.0047464,.1573536])
        self.assertTrue(both_boundary_framings_improve(replace(proposal,pixel_boundary_framing=rows)))
        rows[1]['after'][1]=rows[1]['before'][1]+2.1/proposal.plan.width
        self.assertFalse(both_boundary_framings_improve(replace(proposal,pixel_boundary_framing=rows)))
        rows[1]['after'][1]=rows[1]['before'][1]+2.1/proposal.plan.height
        self.assertFalse(both_boundary_framings_improve(replace(proposal,pixel_boundary_framing=rows)))
        rows[1]['after']=[rows[1]['before'][0]+.001,rows[1]['before'][1]]
        self.assertFalse(both_boundary_framings_improve(replace(proposal,pixel_boundary_framing=rows)))

    def test_native_budget_is_still_independently_probed(self):
        before,proposal=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            sources=tuple(Path(tmp)/s.name for s in before.sources)
            for source in sources:source.write_bytes(b'probe fixture')
            before=replace(before,sources=sources)
            proposal=replace(proposal,plan=replace(proposal.plan,sources=sources))
            motion=attach_shot_context_proposal(before,motion_for(before),proposal)
            for height in (1080,2160):
                def runner(command,timeout):
                    return subprocess.CompletedProcess(command,0,json.dumps({'streams':[
                        dict(width=height*16//9,height=height,sample_aspect_ratio='1:1')]}),'')
                collector=_IssueCollector()
                metrics=_transform_metrics({'editorial_motion':motion.to_dict()},
                    json.loads(json.dumps(proposal.plan.to_json())),PostRenderQAConfig(),collector,runner=runner)
                self.assertEqual(bool(metrics['native_resolution_zoom_budgets']),height==2160)

    def test_one_context_receipt_cannot_upgrade_unrelated_legacy_zoom_authority(self):
        from crosscut.post_render_qa import _native_local_alignment_zoom_budgets
        from engine.tests.test_post_render_qa_current_group_contract import produced_plan
        plan,motion=produced_plan(Path('/unused'),native_zoom=True,paired_corrections=True)
        raw=motion.to_dict()
        raw['boundary_subject_alignment']['analysis_basis']='exact_cut_frame_shot_adaptive_subject_alignment_v2'
        with patch('crosscut.post_render_qa._bound_context_cameras',return_value={2:{}}), \
                patch('crosscut.post_render_qa._run_command') as probe:
            self.assertEqual(_native_local_alignment_zoom_budgets(raw,
                json.loads(json.dumps(plan.to_json())),{1},PostRenderQAConfig(),runner=probe),{})
            probe.assert_not_called()


class UnchangedShotContextReuseTests(unittest.TestCase):
    def fixture(self):
        source=camera_tests.ShotContextProposalTests();args=source.fixture();plan=args['plan']
        plan=replace(plan,segments=(RenderSegment(0,0.,1.,0.),RenderSegment(0,1.,1.,1.),
            *(replace(segment,master_start=segment.master_start+2.) for segment in plan.segments)),
            transitions=(RenderTransition(),)*4)
        proposal=source.run_proposal(dict(args,plan=plan,segment_index=3))
        before=proposal.plan;parts=list(before.segments);delta=4/30.
        parts[0]=replace(parts[0],duration=parts[0].duration-delta)
        parts[1]=replace(parts[1],master_start=parts[1].master_start-delta,
            source_start=parts[1].source_start-delta,duration=parts[1].duration+delta)
        return before,replace(before,segments=tuple(parts)),build_shot_context_binding(proposal)

    def test_unrelated_retime_reuses_every_native_and_head_row_without_new_sensing(self):
        before,after,row=self.fixture();snapshot=deepcopy(row)
        self.assertFalse(shot_context_binding_valid(row,after))
        rebound=rebind_unchanged_shot_context(row,before,after)
        self.assertTrue(shot_context_binding_valid(rebound,after))
        self.assertNotEqual(rebound['geometry_identity'],row['geometry_identity'])
        for key in row:
            if key not in ('geometry_identity','content_sha256'):
                self.assertEqual(row[key],rebound[key],key)
        self.assertEqual(row,snapshot)
        self.assertFalse(rebound['professional_quality_approved'])
        self.assertFalse(rebound['identity_or_gesture_certified'])

    def test_changed_source_neighbor_camera_audio_or_geometry_cannot_reuse_receipt(self):
        before,after,row=self.fixture()
        mutations=[replace(after,master_audio_start=.1),replace(after,width=1280,height=720),
            replace(after,fps=29.97),replace(after,sources=tuple(Path('other')/p for p in after.sources))]
        for index in (2,3,4):
            parts=list(after.segments);parts[index]=replace(parts[index],source_start=parts[index].source_start+.1)
            mutations.append(replace(after,segments=tuple(parts)))
        parts=list(after.segments);parts[3]=replace(parts[3],editorial_transform=None)
        mutations.append(replace(after,segments=tuple(parts)))
        for mutated in mutations:
            with self.subTest(plan=mutated),self.assertRaisesRegex(ValueError,'context'):
                rebind_unchanged_shot_context(row,before,mutated)

    def test_invalid_prior_receipt_cannot_be_rehashed_into_valid_authority(self):
        before,after,row=self.fixture()
        for key,value in (('professional_quality_approved',True),('native_frame_ids',[30]*30),
                ('source_clock',None),('head_evidence',[]),('content_sha256','0'*64)):
            bad=deepcopy(row);bad[key]=value
            if key!='content_sha256':
                bad['content_sha256']=_digest({k:v for k,v in bad.items() if k!='content_sha256'})
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'prior'):
                rebind_unchanged_shot_context(bad,before,after)


if __name__=='__main__':unittest.main()
