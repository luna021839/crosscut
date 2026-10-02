from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic
import unittest
from unittest.mock import patch

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

    def test_loaded_asset_digest_must_match_before_registering_any_observer(self):
        expected=hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.model.write_bytes(b'changed between model load and session registration')
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            with self.assertRaisesRegex(ValueError,'bytes loaded'):
                session.register_model('fixture',self.model,lambda p:p.shape,
                    expected_asset_sha256=expected)
            self.assertEqual(session.models,{})
            self.assertEqual(session.model_adapters,{})
            self.assertEqual(session.model_asset_dependencies,{})
            self.assertEqual(session.model_adapter_preimages,{})
            fresh=hashlib.sha256(self.model.read_bytes()).hexdigest()
            session.register_model('fixture',self.model,lambda p:p.shape,
                expected_asset_sha256=fresh)
            self.assertEqual(session.models['fixture'][2],fresh)

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

    def test_native_model_measurement_is_shared_before_camera_search_but_not_with_projected_models(self):
        changed=replace(self.plan,segments=(replace(self.plan.segments[0],editorial_transform=
            EditorialTransformTrack((EditorialTransformKeyframe(0.,1.25),))),))
        other=joint_state_from_render_plan(plan=changed,domain=self.domain,slot=1,
            first_output_frame=0,end_output_frame=30,role=GroupOutputRole.FOCAL_HEAD)
        calls=[]
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            session.register_model('native-fixture',self.model,
                lambda p,*,bbox_xywh:calls.append((p.copy(),bbox_xywh)) or dict(mean=float(p.mean()),roi=bbox_xywh))
            roi={'bbox_xywh':[.1,.2,.2,.6]}
            a=session.observe_native(2,15,model='native-fixture',arguments=roi)
            session.projected_pixels(self.state,15);session.projected_pixels(other,15)
            b=session.observe_native(2,15,model='native-fixture',arguments=roi)
            self.assertEqual(a,b)
            self.assertEqual(len(calls),1)
            projected=session.observe(other,15,model='native-fixture',arguments=roi)
            self.assertEqual(len(calls),2)
            self.assertNotEqual(a['mean'],projected['mean'])
            self.assertEqual(session.budget.counts['new_decodes'],1)
            self.assertGreaterEqual(session.diagnostics()['native_pixel_cache_reuse'],2)
            self.assertEqual(session.diagnostics()['reused_model_observations'],1)
            self.assertEqual(session.diagnostics()['budget']['model_calls'],{'native-fixture':2})

    def test_raw_source_clock_model_roi_and_missing_values_cannot_be_borrowed(self):
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            session.register_model('missing-fixture',self.model,lambda pixels,**arguments:None)
            self.assertIsNone(session.observe_native(2,15,model='missing-fixture',arguments={'roi':[.1,.2]}))
            self.assertIsNone(session.observe_native(2,15,model='missing-fixture',arguments={'roi':[.6,.2]}))
            self.assertEqual(session.budget.model_calls,{'missing-fixture':2})
            with self.assertRaisesRegex(ValueError,'excluded'):
                session.observe_native(1,15,model='missing-fixture')
            with self.video.open('ab') as stream:stream.write(b'changed disposable fixture')
            with self.assertRaisesRegex(ValueError,'source identity changed'):
                session.observe_native(2,15,model='missing-fixture',arguments={'roi':[.1,.2]})

    def test_context_exit_releases_actual_decoder_and_pixel_caches_on_error(self):
        session=GroupJointObservationSession(domain=self.domain,budget=self.budget())
        with self.assertRaisesRegex(RuntimeError,'fixture failure'):
            with session:
                session.native_pixels(2,15);capture=session.captures[2]
                raise RuntimeError('fixture failure')
        self.assertFalse(capture.isOpened());self.assertTrue(session.closed)
        self.assertEqual((session.raw_bytes,session.projected.bytes),(0,0))
        with self.assertRaisesRegex(ValueError,'closed'):session.native_pixels(2,15)

    def test_external_weight_change_invalidates_native_and_projected_cache_hits(self):
        dependency=Path(self.directory.name)/'fixture.onnx.data'
        dependency.write_bytes(b'disposable external weights')
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            calls=[]
            session.register_model('external-fixture',self.model,
                lambda p:calls.append(p.shape) or p.shape,asset_dependencies=(dependency,))
            session.observe_native(2,15,model='external-fixture')
            session.observe(self.state,15,model='external-fixture')
            self.assertEqual(len(calls),2)
            self.assertIn('fixture.onnx.data',session.diagnostics()['model_external_asset_sha256']['external-fixture'])
            dependency.write_bytes(b'changed disposable external weights')
            for operation in (lambda:session.observe_native(2,15,model='external-fixture'),
                    lambda:session.observe(self.state,15,model='external-fixture')):
                with self.assertRaisesRegex(ValueError,'external asset identity changed'):operation()
            self.assertEqual(len(calls),2,'cached result cannot hide changed external weights')

    def test_external_asset_changed_inside_observer_is_not_cached(self):
        dependency=Path(self.directory.name)/'changing-fixture.onnx.data'
        dependency.write_bytes(b'disposable weights')
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            def observer(pixels):
                dependency.write_bytes(b'changed fixture inside model call')
                return pixels.shape
            session.register_model('changing-external-fixture',self.model,observer,asset_dependencies=(dependency,))
            with self.assertRaisesRegex(ValueError,'external asset identity changed'):
                session.observe_native(2,15,model='changing-external-fixture')
            self.assertEqual(session.model_observations,{})

    def test_raw_lru_evicts_before_third_decode_and_does_not_copy_owned_rasters(self):
        size=96*54*3
        with GroupJointObservationSession(domain=self.domain,budget=self.budget(),maximum_raw_cache_bytes=2*size) as session:
            session.native_pixels(2,15);session.native_pixels(2,16);session.native_pixels(2,17)
            report=session.diagnostics()
            self.assertEqual(report['peak_raw_pixel_cache_bytes'],2*size)
            self.assertEqual(report['peak_owned_raw_raster_bytes_during_decode'],2*size)
            self.assertEqual(report['raw_pixel_cache_eviction_count'],1)
            self.assertEqual(report['native_pixel_copy_count'],0)
            self.assertEqual(report['budget']['counts']['new_decodes'],3)
            self.assertFalse(report['process_peak_rss_measured'])

    def test_explicit_pair_retention_keeps_model_evidence_but_later_pixel_miss_is_counted(self):
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            calls=[]
            session.register_model('retention-fixture',self.model,lambda p:calls.append(p.shape) or p.shape)
            session.observe_native(2,15,model='retention-fixture')
            kept=session.native_pixels(2,16)
            session.retain_native_pixels(((2,16),))
            self.assertIs(session.native_pixels(2,16),kept)
            self.assertEqual(len(session.raw),1)
            self.assertEqual(session.observe_native(2,15,model='retention-fixture'),(54,96,3))
            self.assertEqual(len(calls),1,'raster release must not discard immutable model evidence')
            self.assertEqual(session.budget.counts['new_decodes'],2)
            session.native_pixels(2,15)
            self.assertEqual(session.budget.counts['new_decodes'],3,'re-materialization is not hidden')
            for invalid in (((1,15),),((2,True),),((2,9000),),[(2,15)]):
                with self.assertRaises(ValueError):session.retain_native_pixels(invalid)

    def test_decoder_view_is_copied_without_mutating_its_parent_buffer(self):
        with GroupJointObservationSession(domain=self.domain,budget=self.budget()) as session:
            session.native_pixels(2,15)
            parent=np.zeros((54,192,3),np.uint8);view=parent[:,:96,:]
            with patch.object(session.readers[2],'frame_at',return_value=view):
                actual=session.native_pixels(2,16)
            self.assertFalse(actual.flags.writeable)
            self.assertTrue(actual.flags.owndata)
            parent[:]=25
            self.assertEqual(int(actual.max()),0)
            self.assertEqual(session.diagnostics()['native_pixel_copy_count'],1)


if __name__=='__main__':unittest.main()
