from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch
from collections import OrderedDict

import numpy as np

from crosscut.closeup_gesture import CloseupGestureEvidence,CloseupGestureProvider,ObservedArmMotion
from crosscut.observed_head_motion import (
    HEAD_MOTION_BASIS,HEAD_DIMENSIONS,HeadMotionEvidence,ObservedHeadMotion,
    head_motion_signature,measure_head_motion,
)
from crosscut.group_motion_camera_refinement import measured_motion_improves,measured_motion_frontier
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.group_focal_attention import CENTRAL_FOCAL_BASIS
from crosscut.native_global_face import NativeGlobalFace


class ObservedHeadMotionTests(unittest.TestCase):
    def signature(self):
        return (.02,.01,.4,.8,.01)

    def evidence(self,pose=.2,velocity=.1,guard=None):
        return HeadMotionEvidence(ObservedHeadMotion((10,13,20,23),(.1,1/30.,.1),pose,velocity),guard)

    def test_known_geometry_is_its_separate_head_basis(self):
        points=((.4,.2),(.5,.21),(.45,.25),(.42,.3),(.48,.3))
        geometry=(.4,.2,.5,.21,.1,.1,.01,.4,.8,.02)
        self.assertEqual(head_motion_signature(geometry,points),(.1/math.pi,.01,.4,.8,.02))
        for changes in ({4:.001},{5:math.pi/2},{6:float('nan')},{6:True},{6:3.01}):
            bad=list(geometry)
            for i,value in changes.items():bad[i]=value
            self.assertIsNone(head_motion_signature(tuple(bad),points))
        self.assertIsNone(head_motion_signature(geometry,None))
        self.assertIsNone(head_motion_signature(geometry,points[:4]))
        self.assertIsNone(head_motion_signature(geometry,(*points[:4],(1.,.3))))

    def test_natural_progress_beats_frozen_head_and_reverse_direction(self):
        # The cut is one output tick, while each measured side spans three.
        base=np.asarray(self.signature());velocity=np.asarray((.3,0,0,0,0))
        seconds=(0,.1,.1+1/30.,.2+1/30.)
        signatures=tuple(tuple(base+velocity*t) for t in seconds)
        natural=measure_head_motion(signatures,native_frames=(10,13,20,23),
            output_intervals_seconds=(.1,1/30.,.1),same_source_face_paths_pixel_verified=True)
        self.assertAlmostEqual(natural.cost,0.)
        reverse=(*signatures[:3],tuple(np.asarray(signatures[2])-velocity*.1))
        changed=measure_head_motion(reverse,native_frames=(10,13,20,23),
            output_intervals_seconds=(.1,1/30.,.1),same_source_face_paths_pixel_verified=True)
        self.assertGreater(changed.cost,.01)
        frozen=(signatures[0],signatures[1],signatures[1],tuple(np.asarray(signatures[1])+velocity*.1))
        paused=measure_head_motion(frozen,native_frames=(10,13,20,23),
            output_intervals_seconds=(.1,1/30.,.1),same_source_face_paths_pixel_verified=True)
        self.assertGreater(paused.cost,natural.cost)

    def test_missing_points_or_pixel_path_never_earn_zero_cost(self):
        for values,verified in (((self.signature(),)*4,False),
                ((None,self.signature(),self.signature(),self.signature()),True)):
            self.assertIsNone(measure_head_motion(values,native_frames=(10,13,20,23),
                output_intervals_seconds=(.1,1/30.,.1),same_source_face_paths_pixel_verified=verified))
        with self.assertRaises(ValueError):
            measure_head_motion((None,)*4,native_frames=(10,10,20,23),
                output_intervals_seconds=(.1,1/30.,.1),same_source_face_paths_pixel_verified=False)

    def test_observed_clocks_must_be_noncloned_finite_and_positive(self):
        for frames,times in (((True,13,20,23),(.1,.03,.1)),((13,10,20,23),(.1,.03,.1)),
                ((10,13,23,20),(.1,.03,.1)),((10,13,20,23),(.1,0.,.1)),
                ((10,13,20,23),(.1,True,.1)),((10,13,20,23),(.1,float('nan'),.1))):
            with self.assertRaises(ValueError):ObservedHeadMotion(frames,times,.1,.1)
        for value in (True,-.1,float('nan')):
            with self.assertRaises(ValueError):self.evidence(pose=value)

    def test_arm_bones_are_not_relabeled_head_motion(self):
        guard=CloseupGestureEvidence(13,20,(.2,.3,None,None))
        before=self.evidence(guard=guard)
        better=self.evidence(pose=.1,velocity=.05,guard=replace(guard,bone_mismatches=(.1,.2,None,None)))
        self.assertTrue(measured_motion_improves(better,before))
        for modified in (replace(better,endpoint_arm_guard=None),
                replace(better,endpoint_arm_guard=replace(guard,bone_mismatches=(.1,None,.1,None))),
                replace(better,endpoint_arm_guard=replace(guard,bone_mismatches=(.21,.2,None,None)))):
            self.assertFalse(measured_motion_improves(modified,before))
        payload=before.to_dict()
        self.assertTrue(payload['arm_motion_unknown_not_imputed'])
        self.assertFalse(payload['motion']['arm_motion_observed'])
        self.assertEqual(payload['motion']['measured_dimensions'],list(HEAD_DIMENSIONS))
        self.assertNotIn('observed_bone_indices',payload['motion'])

    def test_head_cannot_replace_existing_four_frame_arm_basis(self):
        arm=CloseupGestureEvidence(13,20,(.2,.3,None,None),motion=
            ObservedArmMotion((10,13,20,23),(.1,1/30.,.1),(.2,.2,None,None),(.1,.1,None,None)))
        self.assertFalse(measured_motion_improves(self.evidence(pose=.01,velocity=.01),arm))
        self.assertFalse(measured_motion_improves(arm,self.evidence()))
        self.assertFalse(measured_motion_improves(self.evidence(pose=.21,velocity=0.),self.evidence()))

    def test_endpoint_arm_guard_may_not_change_to_another_focal_person(self):
        guard=CloseupGestureEvidence(13,20,(.2,.3,None,None))
        for field in ('outgoing_focal_basis','incoming_focal_basis'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.evidence(guard=replace(guard,**{field:CENTRAL_FOCAL_BASIS}))
        with self.assertRaises(ValueError):
            self.evidence(guard=replace(guard,incoming_native_frame=21))

    def test_uncalibrated_bases_get_bounded_independently_ranked_slots(self):
        arm=CloseupGestureEvidence(13,20,(.2,.3,None,None),motion=
            ObservedArmMotion((10,13,20,23),(.1,1/30.,.1),(.2,.2,None,None),(.1,.1,None,None)))
        rows=[(100-i,i,arm,None,None,None) for i in range(4)]
        rows.extend((.1-i*.01,10+i,self.evidence(),None,None,None) for i in range(4))
        self.assertEqual([row[1] for row in measured_motion_frontier(rows,4)],[0,10,1,11])
        self.assertEqual([row[1] for row in measured_motion_frontier(rows[:4],4)],[0,1,2,3])


class HeadMotionProviderTests(unittest.TestCase):
    def native(self):
        descriptor=self.requests()[0][-1]
        return NativeGlobalFace(1,descriptor.primary_face_bbox,descriptor.primary_face_landmarks_xy,
            0.,.1,descriptor.primary_face_geometry)

    def provider(self):
        provider=CloseupGestureProvider.__new__(CloseupGestureProvider)
        provider.sources=(Path('0.mp4'),Path('1.mp4'));provider.identities=('same','same')
        provider._identity=lambda p:'same';provider.head_cache=OrderedDict()
        provider.capture=SimpleNamespace(frame_at_native_index=lambda *a:np.zeros((480,854,3),np.uint8))
        provider.compare=lambda *a:None
        return provider

    def requests(self):
        clock=VerifiedRenderFrameClock(Fraction(30),3000,'a'*64)
        descriptor=SimpleNamespace(primary_face_bbox=(.35,.15,.25,.3),
            primary_face_geometry=(.4,.2,.5,.2,.1,0.,0.,.4,.8,0.),
            primary_face_landmarks_xy=((.4,.2),(.5,.2),(.45,.25),(.42,.3),(.48,.3)))
        return tuple((source,index/30.,clock,descriptor) for source,index in ((0,10),(0,13),(1,20),(1,23)))

    def test_known_five_point_geometry_is_reused_only_after_fresh_global_verification(self):
        provider=self.provider()
        with patch('crosscut.group_focal_attention.focal_landmarks_track',return_value=True), \
                patch('crosscut.native_global_face.observe_native_global_face',return_value=self.native()) as detect:
            result=provider.compare_head_motion(*self.requests(),output_fps=30.)
            again=provider.compare_head_motion(*self.requests(),output_fps=30.)
        self.assertEqual(detect.call_count,4,'run-local verified geometry is reused, not re-inferred')
        self.assertEqual(result,again)
        self.assertEqual(result.motion.native_frames,(10,13,20,23))
        self.assertEqual(len(provider.head_cache),4)
        self.assertFalse(result.to_dict()['motion']['cross_camera_identity_proven'])

    def test_failed_same_source_pixel_link_stays_unknown(self):
        with patch('crosscut.group_focal_attention.focal_landmarks_track',return_value=False), \
                patch('crosscut.boundary_subject_alignment._optical_flow_box',return_value=None), \
                patch('crosscut.native_global_face.observe_native_global_face',return_value=self.native()):
            self.assertIsNone(self.provider().compare_head_motion(*self.requests(),output_fps=30.))

    def test_central_person_arms_are_not_attached_to_the_global_head(self):
        guard=CloseupGestureEvidence(13,20,(.2,.3,None,None))
        for field in ('outgoing_focal_basis','incoming_focal_basis'):
            provider=self.provider()
            provider.compare=lambda *a:replace(guard,**{field:CENTRAL_FOCAL_BASIS})
            with self.subTest(field=field),patch('crosscut.group_focal_attention.focal_landmarks_track',return_value=True), \
                    patch('crosscut.native_global_face.observe_native_global_face',return_value=self.native()):
                result=provider.compare_head_motion(*self.requests(),output_fps=30.)
            self.assertIsNotNone(result)
            self.assertIsNone(result.endpoint_arm_guard)

    def test_missing_face_may_not_choose_a_central_or_ambiguous_face(self):
        requests=list(self.requests());descriptor=requests[0][-1]
        requests[0]=(*requests[0][:3],SimpleNamespace(primary_face_bbox=None,
            primary_face_geometry=None,primary_face_landmarks_xy=None))
        with patch('crosscut.roll_aware_face_geometry.detect_roll_aware_faces',return_value=()), \
                patch('crosscut.group_focal_attention.observe_central_focal_face') as central:
            self.assertIsNone(self.provider().compare_head_motion(*requests,output_fps=30.))
        central.assert_not_called()

    def test_cached_background_head_and_its_arms_are_not_preserved_as_foreground(self):
        provider=self.provider();requests=self.requests()
        provider.compare=lambda *a:CloseupGestureEvidence(13,20,(.1,.1,None,None))
        foreground=replace(self.native(),bbox_xywh=(.1,.15,.3,.35),
            geometry=(*self.native().geometry[:6],.25,.4,.8,0.))
        with patch('crosscut.native_global_face.observe_native_global_face',return_value=foreground), \
                patch('crosscut.group_focal_attention.focal_landmarks_track',return_value=True):
            result=provider.compare_head_motion(*requests,output_fps=30.)
        self.assertIsNotNone(result)
        self.assertIsNone(result.endpoint_arm_guard,'legacy arms belonged to a displaced cached background face')
        self.assertTrue(all(not row[3] for row in provider.head_cache.values()))

    def test_new_global_ambiguity_revokes_even_previously_known_head_geometry(self):
        with patch('crosscut.native_global_face.observe_native_global_face',return_value=NativeGlobalFace(2,None,None,None,None,None)):
            self.assertIsNone(self.provider().compare_head_motion(*self.requests(),output_fps=30.))

    def test_cached_low_resolution_derivatives_do_not_override_fresh_five_points(self):
        requests=[]
        for i,row in enumerate(self.requests()):
            old=row[-1];geometry=(*old.primary_face_geometry[:6],i*.3,.4,.8,0.)
            requests.append((*row[:3],SimpleNamespace(primary_face_bbox=old.primary_face_bbox,
                primary_face_landmarks_xy=old.primary_face_landmarks_xy,primary_face_geometry=geometry)))
        with patch('crosscut.native_global_face.observe_native_global_face',return_value=self.native()), \
                patch('crosscut.group_focal_attention.focal_landmarks_track',return_value=True):
            result=self.provider().compare_head_motion(*requests,output_fps=30.)
        self.assertAlmostEqual(result.ranking_cost,0.,msg='same fresh observed head must not inherit stale derivative jumps')

    def test_changed_source_or_wrong_source_runs_are_rejected(self):
        provider=self.provider();provider._identity=lambda p:'changed'
        with self.assertRaises(OSError):provider.compare_head_motion(*self.requests(),output_fps=30.)
        requests=list(self.requests());requests[1]=(1,*requests[1][1:])
        with self.assertRaises(ValueError):self.provider().compare_head_motion(*requests,output_fps=30.)


if __name__=='__main__':unittest.main()
