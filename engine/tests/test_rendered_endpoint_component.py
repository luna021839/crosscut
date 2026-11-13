from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math
import shutil
import subprocess
import unittest
from unittest.mock import Mock,patch

from crosscut.rendered_endpoint_component import (EndpointComponentFrame,EndpointComponentPair,
    endpoint_component_clocks,incoming_dissolve_weight)
from crosscut.rendered_body_window import RenderedBodyWindowProvider
from crosscut.rendered_body_reference import RenderedBodyMeasurement
from crosscut.direct_member_rendered_continuity import evaluate_direct_member_rendered_continuity
from crosscut.render import (FACE_DISSOLVE_ALPHA_CURVE,FACE_DISSOLVE_SMOOTHSTEP_ALPHA_CURVE,
                            _FACE_DISSOLVE_XFADE_EXPRESSION,face_dissolve_xfade_expression)
from engine.tests import test_rendered_body_window as body_fixture
from engine.tests.test_direct_member_rendered_continuity import _effect_plan,_descriptor,_provider


class EndpointComponentTests(unittest.TestCase):
    def setUp(self):
        fixture=body_fixture.RenderedBodyIntegrationTests();fixture.setUp()
        self.observed=fixture.observed

    def pair(self,label='effect_exit',frames=(10,11),measurement=None):
        m=measurement or self.observed
        weights=(1.,1.) if label=='effect_entry' else (incoming_dissolve_weight(6,7),1.)
        return EndpointComponentPair(0,label,tuple(EndpointComponentFrame(n,int(label=='effect_exit'),
            n*3,n/10.,w,m) for n,w in zip(frames,weights)))

    def test_detector_box_padding_does_not_masquerade_as_observed_torso_scale(self):
        pair=self.pair()
        m=replace(self.observed,person=replace(self.observed.person,bbox_xywh=(0.,0.,1.,1.)),
            pose=replace(self.observed.pose,bbox_xywh=(0.,0.,1.,1.)))
        pair=replace(pair,frames=(replace(pair.frames[0],measurement=m),pair.frames[1]))
        self.assertEqual(pair.metrics()['attention_position_step'],0.)
        self.assertEqual(pair.metrics()['attention_scale_step'],0.)
        self.assertFalse(pair.audit()['whole_frame_single_source_claim'])
        self.assertNotIn('observed_output_pose_xyc',pair.audit()['frames'][0]['measurement'])

    def test_real_scale_and_position_changes_remain_large(self):
        pair=self.pair()
        pose=replace(self.observed.pose,upper_body_xyc=tuple(
            (.5+(x-.5)*1.3+.09,.5+(y-.5)*1.3,c) for x,y,c in self.observed.pose.upper_body_xyc))
        changed=replace(self.observed,pose=pose,person=pose.as_formation_person(.9))
        pair=replace(pair,frames=(pair.frames[0],replace(pair.frames[1],measurement=changed)))
        self.assertGreater(pair.metrics()['attention_scale_step'],.1)
        self.assertGreater(pair.metrics()['attention_position_step'],.12)

    def test_missing_joint_ownership_or_low_component_weight_cannot_authorize(self):
        pair=self.pair()
        for m in (replace(self.observed,reason='output_target_body_ambiguous'),
                  replace(self.observed,similarity=float('nan')),replace(self.observed,margin=.01),
                  replace(self.observed,source_output_pose_mismatch=.4),
                  replace(self.observed,pose=replace(self.observed.pose,upper_body_xyc=()))):
            self.assertIsNone(replace(pair,frames=(replace(pair.frames[0],measurement=m),pair.frames[1])).metrics())
        self.assertIsNone(replace(pair,frames=(replace(pair.frames[0],component_weight=.89),pair.frames[1])).metrics())

    def test_nonadjacent_or_different_source_frames_are_rejected(self):
        pair=self.pair()
        for changed in (replace(pair.frames[1],output_frame=12),replace(pair.frames[1],source_index=3),
                        replace(pair.frames[1],source_frame=0)):
            with self.assertRaises(ValueError):replace(pair,frames=(pair.frames[0],changed))

    def evaluate(self,endpoint_provider):
        plan=_effect_plan()
        plan['segments'][1]['duration']=1.
        plan['output_frame_count']=14
        plan['transitions'][0].update(frame_count=7,alpha_curve=FACE_DISSOLVE_ALPHA_CURVE)
        descriptors={i:_descriptor(position=.3,pose=.2,scale=.2 if i<11 else .4,
            include_face=False) for i in range(14)}
        return evaluate_direct_member_rendered_continuity(Path('unused.mp4'),plan,
            descriptor_provider=_provider(descriptors),effect_endpoint_provider=endpoint_provider)

    def test_target_evidence_replaces_group_mask_gate_not_actual_jump_threshold(self):
        def measured(index,label):return self.pair(label,(3,4) if label=='effect_entry' else (10,11))
        result=self.evaluate(measured)
        self.assertTrue(result.passed,result.failure_reasons)
        self.assertEqual(result.metrics['effect_endpoint_target_guided_count'],2)
        row=result.boundaries[0]['endpoint_steps'][1]
        self.assertTrue(row['target_reference_guided'])
        self.assertGreater(row['aggregate_attention_scale_step_diagnostic'],.10)
        self.assertEqual(row['attention_scale_step'],0.)
        self.assertTrue(row['aggregate_silhouette_diagnostic_only'])

    def test_unresolved_component_retains_failure_and_wrong_binding_raises(self):
        self.assertIn('rendered_effect_endpoint_scale_discontinuity',self.evaluate(lambda *_:None).failure_reasons)
        with self.assertRaisesRegex(ValueError,'differ from rendered boundary'):
            self.evaluate(lambda *_:self.pair())

    def test_measured_target_jump_still_fails_production_endpoint_qa(self):
        def measured(index,label):
            pair=self.pair(label,(3,4) if label=='effect_entry' else (10,11))
            pose=replace(self.observed.pose,upper_body_xyc=tuple(
                (.5+(x-.5)*1.3+.09,.5+(y-.5)*1.3,c) for x,y,c in self.observed.pose.upper_body_xyc))
            m=replace(self.observed,pose=pose,person=pose.as_formation_person(.9))
            return replace(pair,frames=(pair.frames[0],replace(pair.frames[1],measurement=m)))
        result=self.evaluate(measured)
        self.assertFalse(result.passed)
        self.assertIn('rendered_effect_endpoint_position_discontinuity',result.failure_reasons)
        self.assertIn('rendered_effect_endpoint_scale_discontinuity',result.failure_reasons)

    def test_provider_cannot_claim_blended_endpoint_is_a_pure_frame(self):
        def measured(index,label):
            pair=self.pair(label,(3,4) if label=='effect_entry' else (10,11))
            return replace(pair,frames=(replace(pair.frames[0],component_weight=1.),pair.frames[1]))
        with self.assertRaisesRegex(ValueError,'weights differ from renderer'):self.evaluate(measured)


class ComponentClockTests(unittest.TestCase):
    def provider(self):
        plan=body_fixture.BodyWindowClockTests().plan()
        plan['transitions']=[dict(kind='face_dissolve',frame_count=3,alpha_curve=FACE_DISSOLVE_ALPHA_CURVE)]
        return RenderedBodyWindowProvider(plan,Mock(),body_model_path=None,pose_model_path=None)

    def test_outgoing_handle_owns_separate_source_trim_clock(self):
        p=self.provider()
        before,handle=endpoint_component_clocks(p.spans,p.transitions,0,'effect_entry')
        self.assertEqual((before[0],handle[0]),(3,4))
        self.assertEqual(handle[1].source_start,10.401)
        self.assertEqual(handle[2],0)
        self.assertEqual(handle[3],1.)
        with patch.object(p,'_measurement_on_clock',return_value=RenderedBodyMeasurement('output_body_unobserved')) as measured:
            source=Mock();source.faces._samplers=[SimpleNamespace(frame_rate=30.)]*2
            p._source=source
            result=p.endpoint_window(0,'effect_entry')
        self.assertEqual(result.frames[1].source_frame,313)
        self.assertEqual(result.frames[1].source_seconds,313/30.)
        self.assertEqual(measured.call_count,2)

    def test_unknown_curves_low_weights_and_non_dissolves_do_not_open_models(self):
        p=self.provider()
        with patch.object(p,'_open') as opened:
            self.assertIsNone(p.endpoint_window(0,'effect_exit'))  # 3-frame endpoint <90% incoming
            p.transitions[0]['alpha_curve']='unknown'
            self.assertIsNone(p.endpoint_window(0,'effect_entry'))
            p.transitions[0]['alpha_curve']=FACE_DISSOLVE_ALPHA_CURVE
            p.transitions[0]['kind']='member_zoom_bridge'
            self.assertIsNone(p.endpoint_window(0,'effect_entry'))
        opened.assert_not_called()

    def test_component_cache_hit_still_checks_live_source_authority(self):
        p=self.provider()
        source=Mock();source.faces._samplers=[SimpleNamespace(frame_rate=30.)]*2
        p._source=source
        for output,span,relative,weight in endpoint_component_clocks(p.spans,p.transitions,0,'effect_entry'):
            from crosscut.post_render_perceptual_quality import _rendered_source_frame_index
            frame=_rendered_source_frame_index(span,relative_time=relative/p.fps,source_fps=30.)
            key=output if output==3 else ('endpoint_component',output,span.source_index,frame)
            p._cache[key]=RenderedBodyMeasurement('output_body_unobserved')
        self.assertIsNone(p.endpoint_window(0,'effect_entry').metrics())
        self.assertEqual(source.selector_geometry.call_count,2)
        source.selector_geometry.side_effect=OSError('SSD disconnected')
        with self.assertRaisesRegex(OSError,'SSD disconnected'):p.endpoint_window(0,'effect_entry')

    @unittest.skipUnless(shutil.which('ffmpeg'),'requires actual FFmpeg alpha proof')
    def test_formula_matches_production_expression_in_decoded_pixels(self):
        fps=29.97
        for curve,count in ((FACE_DISSOLVE_ALPHA_CURVE,3),(FACE_DISSOLVE_ALPHA_CURVE,7),
                            (FACE_DISSOLVE_ALPHA_CURVE,15),(FACE_DISSOLVE_SMOOTHSTEP_ALPHA_CURVE,8)):
            graph=(f"[0:v]format=gray,settb=AVTB[a];[1:v]format=gray,settb=AVTB[b];"
                f"[a][b]xfade=transition=custom:expr='{face_dissolve_xfade_expression(curve)}':"
                f"duration={count/fps:.9f}:offset={30/fps:.9f},"
                f"select='eq(n,30)+eq(n,{30+count-1})+eq(n,{30+count})'[out]")
            pixels=subprocess.run([shutil.which('ffmpeg'),'-v','error','-filter_complex_threads','1',
                '-f','lavfi','-i',f'color=black:s=16x16:r={fps}:d=2',
                '-f','lavfi','-i',f'color=white:s=16x16:r={fps}:d=2',
                '-filter_complex',graph,'-map','[out]','-fps_mode','passthrough','-frames:v','3',
                '-f','rawvideo','-pix_fmt','gray','pipe:1'],check=True,capture_output=True,timeout=30).stdout
            self.assertEqual(len(pixels),3*256)
            for index,relative in enumerate((0,count-1,count)):
                observed=sum(pixels[index*256:(index+1)*256])/256
                self.assertAlmostEqual(observed,255*incoming_dissolve_weight(relative,count,alpha_curve=curve),delta=2.,
                    msg=f'count={count}, relative={relative}')


if __name__=='__main__':unittest.main()
