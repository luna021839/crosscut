from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, DirectMemberCinematicBridge,
    FOCAL_POSE_MOTION_BRIDGE_BASIS, _boundary_mismatch,
    _boundary_sample_with_focal_geometry,
)
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.focal_selector_geometry import FocalSelectorGeometry, FOCAL_SELECTOR_POSE_BASIS
from crosscut.member_bridge_contract import RequiredMemberBridgeAuthoringError
from crosscut.render import TransitionKind, RenderPlan, render
from crosscut.ive import AspectRatio
from engine.tests.test_focal_selector_geometry import descriptor, person
from engine.tests.test_direct_member_transition import _selection, _plan, _segments


def focal_sample(source, seconds, *, missing=False, reverse=False):
    tick=(seconds-4.)*30
    motion=(-1 if reverse else 1)*tick
    x=.05+source*.25+motion*.006
    body=replace(person(motion*.06),bbox_xywh=(x,.08,.4,.8),centroid_xy=(x+.2,.48))
    focal=FocalSelectorGeometry(source,seconds,None if missing else (x+.1,.10,.1,.1),
                               None if missing else .035,None if missing else body)
    return _boundary_sample_with_focal_geometry(master_seconds=seconds,source_local_seconds=seconds,
        source_index=source,descriptor=descriptor(raw_pose=source*100.),
        provider=SimpleNamespace(focal_geometry_at=lambda *args:focal))


def selection():
    result=_selection((0,1))
    result.usage_policy='heterogeneous_role_aware_cinematic_v1'
    boundary=result.boundaries[0]
    boundary.outgoing_sample=focal_sample(0,4.-1/30)
    boundary.incoming_sample=focal_sample(1,4.)
    geometry=direct_member_source_frame_geometry((1080,1920),output_size=(1080,1920))
    boundary.mismatch=_boundary_mismatch(boundary.outgoing_sample,boundary.incoming_sample,
        DirectMemberVisualConfig(sample_frame_step_seconds=1/30),
        outgoing_previous=focal_sample(0,4.-2/30),incoming_next=focal_sample(1,4.+1/30),
        outgoing_source_index=0,incoming_source_index=1,outgoing_source_geometry=geometry,
        incoming_source_geometry=geometry,continuous_official_portrait_fancam_contract=True)
    boundary.outgoing_handle_samples=tuple(focal_sample(0,t) for t in (4.+1/30,4.2))
    boundary.incoming_handle_samples=tuple(focal_sample(1,t) for t in (4.+1/30,4.2))
    return result


class FocalRequiredBridgeTests(unittest.TestCase):
    def test_real_selector_focal_motion_reaches_required_effect_planner(self):
        selected=selection()
        mismatch=selected.boundaries[0].mismatch
        self.assertIsNone(mismatch.silhouette_mismatch)
        self.assertEqual(mismatch.pose_basis,FOCAL_SELECTOR_POSE_BASIS)
        self.assertIsNotNone(mismatch.cinematic_bridge, mismatch.to_dict())
        bridge=mismatch.cinematic_bridge
        self.assertEqual(bridge.evidence_basis,FOCAL_POSE_MOTION_BRIDGE_BASIS)
        self.assertIsNone(bridge.silhouette_mismatch)
        self.assertFalse(bridge.to_dict()['hard_cut_authorized'])
        plan=_plan((0,1),selection=selected,highlight_intervals=((3.5,4.5),))
        self.assertEqual(plan.cinematic_bridge_count,1)
        self.assertIs(plan.transitions[0].kind,TransitionKind.FACE_DISSOLVE)

    def test_missing_target_in_handle_cannot_borrow_matching_background(self):
        selected=selection()
        selected.boundaries[0].incoming_handle_samples=(focal_sample(1,4.+1/30,missing=True),focal_sample(1,4.2))
        with self.assertRaises(RequiredMemberBridgeAuthoringError):
            _plan((0,1),selection=selected,highlight_intervals=((3.5,4.5),))

    def test_stale_focal_endpoint_cannot_authorize_effect(self):
        selected=selection()
        boundary=selected.boundaries[0]
        boundary.incoming_sample=replace(boundary.incoming_sample,
            focal_geometry=replace(boundary.incoming_sample.focal_geometry,source_seconds=4.1))
        with self.assertRaises(RequiredMemberBridgeAuthoringError):
            _plan((0,1),selection=selected,highlight_intervals=((3.5,4.5),))

    def test_missing_or_opposite_motion_cannot_authorize_focal_effect(self):
        for values in ({'motion_speed_ratio_mismatch':None},
                       {'pose_phase_continuation_mismatch':None},
                       {'direction_cosine_audit_only':-.5},
                       {'both_moving':False}):
            selected=selection()
            boundary=selected.boundaries[0]
            boundary.mismatch=replace(boundary.mismatch,
                motion_connection=replace(boundary.mismatch.motion_connection,**values))
            with self.subTest(values=values),self.assertRaises(RequiredMemberBridgeAuthoringError):
                _plan((0,1),selection=selected,highlight_intervals=((3.5,4.5),))

    def test_missing_silhouette_is_not_globally_reinterpreted_as_zero(self):
        with self.assertRaises(ValueError):
            DirectMemberCinematicBridge('pose_dissolve',False,False,('attention_position',),.1,None,.2)
        with self.assertRaises(ValueError):
            DirectMemberCinematicBridge('zoom_in',False,False,('attention_position',),.1,None,.2,FOCAL_POSE_MOTION_BRIDGE_BASIS)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'requires actual FFmpeg renderer')
    def test_focal_selector_to_planner_to_actual_logistic_pixels(self):
        ffmpeg=shutil.which('ffmpeg')
        with tempfile.TemporaryDirectory(prefix='focal-effect-test-') as temp:
            root=Path(temp)
            sources=tuple(root/f'{color}.mp4' for color in ('red','blue'))
            for color,path in zip(('red','blue'),sources):
                subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i',f'color=c={color}:s=192x342:r=30:d=9',
                    '-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-shortest','-c:v','libx264','-preset','ultrafast',
                    '-pix_fmt','yuv420p','-c:a','aac',str(path)],check=True,capture_output=True)
            effects=_plan((0,1),selection=selection(),highlight_intervals=((3.5,4.5),))
            output=root/'focal-logistic.mp4'
            plan=RenderPlan(sources=sources,segments=_segments((0,1)),reference_source=0,
                aspect_ratio=AspectRatio.PORTRAIT,output=output,width=192,height=342,fps=30.,
                transitions=effects.transitions,member_pose_dissolve_requested=True,
                member_pose_dissolve_effective=True,member_pose_dissolve_count=1)
            render(plan,ffmpeg=ffmpeg,video_codec='libx264')
            probe=json.loads(subprocess.run([str(Path(ffmpeg).with_name('ffprobe')),'-v','error',
                '-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(output)],
                check=True,capture_output=True).stdout)
            self.assertEqual(int(probe['streams'][0]['nb_frames']),240)
            pixels=subprocess.run([ffmpeg,'-v','error','-i',str(output),'-vf','select=eq(n\\,123)',
                '-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],check=True,capture_output=True).stdout
            self.assertEqual(len(pixels),192*342*3)
            self.assertGreater(sum(pixels[0::3])/len(pixels[0::3]),15.)
            self.assertGreater(sum(pixels[2::3])/len(pixels[2::3]),15.)
            provenance=json.loads(output.with_suffix('.mp4.frame-clock.json').read_text())
            self.assertIn('exp(-12*(P-0.5))',provenance['filter_graph'])
            self.assertFalse(provenance['quality_approval'])


if __name__ == '__main__':
    unittest.main()
