from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic
import unittest

import numpy as np

from crosscut.group_joint_observations import GroupJointObservationSession
from crosscut.group_joint_source_camera import (GroupOutputRole,JointGroupWorkBudget,
    JointGroupBudgetExhausted,joint_state_from_render_plan)
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import RenderPlan,RenderSegment,EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.render_frame_schedule import probe_render_frame_clock


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
class JointObservationSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='crosscut-joint-observations-')
        root=Path(self.directory.name)
        self.video=root/'eligible-2.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=96x54:r=30',
            '-frames:v','90','-c:v','libx264','-crf','0','-pix_fmt','yuv420p',str(self.video)],
            capture_output=True,check=True,timeout=10.)
        self.clock=probe_render_frame_clock(self.video)
        self.assertIsNotNone(self.clock)
        self.sources=tuple(self.video if i==2 else root/f'never-open-{i}.mp4' for i in range(6))
        self.domain=GroupSourceDomain(self.sources,(0,2,5),(0.,)*3,(None,)*3,((),)*3,(self.clock,)*3)
        self.plan=RenderPlan(sources=self.sources,segments=(RenderSegment(2,0.,1.,0.),),
            reference_source=2,aspect_ratio='16:9',output=root/'not-rendered.mp4',fps=30.,width=96,height=54,transitions=())
        self.state=joint_state_from_render_plan(plan=self.plan,domain=self.domain,slot=1,
            first_output_frame=0,end_output_frame=30,role=GroupOutputRole.FOCAL_HEAD)
        self.model=root/'fixture-model.bin';self.model.write_bytes(b'fixture model not real weights')

    def tearDown(self):self.directory.cleanup()

    def budget(self,**limits):return JointGroupWorkBudget(absolute_deadline=monotonic()+10.,**limits)

    def test_only_actual_eligible_source_is_opened_and_raw_frame_is_reused_across_two_cameras(self):
        changed=replace(self.plan,segments=(replace(self.plan.segments[0],editorial_transform=
            EditorialTransformTrack((EditorialTransformKeyframe(0.,1.25),))),))
        other=joint_state_from_render_plan(plan=changed,domain=self.domain,slot=1,
            first_output_frame=0,end_output_frame=30,role=GroupOutputRole.FOCAL_HEAD)
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            a=session.projected_pixels(self.state,15)
            b=session.projected_pixels(other,15)
            self.assertFalse(np.array_equal(a,b))
            self.assertIs(a,session.projected_pixels(self.state,15))
            report=session.diagnostics()
            self.assertEqual(report['opened_request_sources'],[2])
            self.assertEqual(report['budget']['counts']['new_decodes'],1)
            self.assertEqual(report['source_readers']['2']['read_count'],1)
            self.assertFalse(a.flags.writeable)
            self.assertFalse(report['persistent_raw_feature_caches_modified'])

    def test_source_replacement_invalidates_both_raw_and_projected_cache_hits(self):
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            session.projected_pixels(self.state,15)
            # Only this disposable fixture is changed.
            with self.video.open('ab') as stream:stream.write(b'changed')
            for operation in (lambda:session.native_pixels(2,15),lambda:session.projected_pixels(self.state,15)):
                with self.assertRaisesRegex(ValueError,'source identity changed'):operation()

    def test_wrong_clock_fails_before_opening_any_decoder(self):
        domain=replace(self.domain,render_frame_clocks=(self.clock,replace(self.clock,packet_grid_sha256='f'*64),self.clock))
        with GroupJointObservationSession(domain=domain,budget=self.budget()) as session:
            with self.assertRaisesRegex(ValueError,'native clock/file changed'):session.native_pixels(2,15)
            self.assertEqual(session.captures,{})
            self.assertEqual(session.budget.counts['new_decodes'],0)

    def test_decode_and_named_model_budgets_are_real_work_counts(self):
        budget=self.budget(maximum_new_decodes=2,maximum_model_calls=1)
        with GroupJointObservationSession(domain=self.domain,budget=budget) as session:
            calls=[]
            session.register_model('fixture-pixel-observer',self.model,lambda p:calls.append(float(p.mean())) or p.shape)
            self.assertEqual(session.observe(self.state,15,model='fixture-pixel-observer'),(54,96,3))
            self.assertEqual(session.observe(self.state,15,model='fixture-pixel-observer'),(54,96,3))
            self.assertEqual(session.diagnostics()['reused_model_observations'],1)
            with self.assertRaisesRegex(JointGroupBudgetExhausted,'model_calls'):
                session.observe(self.state,16,model='fixture-pixel-observer')
            with self.assertRaisesRegex(JointGroupBudgetExhausted,'new_decodes'):
                session.projected_pixels(self.state,17)
            self.assertEqual(len(calls),1)
            self.assertEqual(budget.model_calls,{'fixture-pixel-observer':1})

    def test_model_change_and_excluded_source_are_not_silently_reused(self):
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            session.register_model('fixture',self.model,lambda p:p.shape)
            self.model.write_bytes(b'changed fixture weights')
            with self.assertRaisesRegex(ValueError,'model identity changed'):
                session.observe(self.state,15,model='fixture')
            with self.assertRaisesRegex(ValueError,'excluded'):session.native_pixels(1,15)
            self.assertEqual(session.budget.counts['model_calls'],0)
            self.assertEqual(session.captures,{})

    def test_same_frame_different_body_rois_do_not_borrow_cached_pose(self):
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            calls=[]
            def observer(pixels,*,bbox_xywh):
                calls.append(tuple(bbox_xywh));return {'box':bbox_xywh}
            session.register_model('roi-fixture',self.model,observer)
            a={'bbox_xywh':[.1,.2,.2,.6]};b={'bbox_xywh':[.6,.2,.2,.6]}
            first=session.observe(self.state,15,model='roi-fixture',arguments=a)
            second=session.observe(self.state,15,model='roi-fixture',arguments=b)
            first['box'][0]=99
            self.assertEqual(second['box'],b['bbox_xywh'])
            self.assertEqual(session.observe(self.state,15,model='roi-fixture',arguments=a)['box'],a['bbox_xywh'])
            self.assertEqual(len(calls),2)
            self.assertEqual(session.budget.counts['new_decodes'],1)
            self.assertEqual(session.budget.model_calls,{'roi-fixture':2})
            self.assertEqual(session.diagnostics()['reused_model_observations'],1)

    def test_invalid_model_arguments_fail_before_any_model_call(self):
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            session.register_model('fixture',self.model,lambda pixels:None)
            for value in ([],{1:'invalid'},{'bbox':float('nan')}):
                with self.assertRaises(ValueError):session.observe(self.state,15,model='fixture',arguments=value)
            self.assertEqual(session.budget.counts['model_calls'],0)

    def test_context_exit_releases_actual_decoder_and_pixel_caches_on_error(self):
        session=GroupJointObservationSession(domain=self.domain,budget=self.budget())
        with self.assertRaisesRegex(RuntimeError,'fixture failure'):
            with session:
                session.native_pixels(2,15);capture=session.captures[2]
                raise RuntimeError('fixture failure')
        self.assertFalse(capture.isOpened());self.assertTrue(session.closed)
        self.assertEqual((session.raw_bytes,session.projected.bytes),(0,0))
        with self.assertRaisesRegex(ValueError,'closed'):session.native_pixels(2,15)


if __name__=='__main__':unittest.main()
