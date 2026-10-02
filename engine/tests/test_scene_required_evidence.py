"""Focused correctness regressions, no media/network/model invocation.

Motion values below are explicit function doubles to isolate the required-duty
boolean; no synthetic value is sold as observed human motion.
"""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from crosscut import group_scene_observations as scene
from crosscut import group_scene_composition as composition
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.group_scene_role_obligations import scene_role_obligations
from crosscut.group_joint_formation import ObservedFormationPerson


class RequiredMotionTests(unittest.TestCase):
    def call(self, *, role, known_parts, head, arm):
        rows=[dict(projected_landmarks_xy=None,projected_face_roi_xywh=(.4,.2,.15,.2),
            scene_people=[],selected_frame_local_head_id=0,native_frame_index=n,
            role_obligations=dict(required_parts=known_parts)) for n in (10,13,20,23)]
        state=SimpleNamespace(role=role,output_fps=30.,output_size=(320,180))
        with patch('crosscut.observed_head_motion.measure_head_motion',return_value=head), \
             patch('crosscut.closeup_gesture.measure_arm_motion',return_value=arm), \
             patch('crosscut.closeup_gesture.observed_arm_articulation',return_value=None):
            return scene.selected_focal_motion(rows,[state]*4)

    def head(self):
        return SimpleNamespace(predicted_pose_cost=.1,velocity_jump_per_output_tick=.1,
            to_dict=lambda:dict(explicit_function_double=True))

    def arm(self):
        return SimpleNamespace(observed_bones=(0,1,2,3),predicted_jump_radians=(.1,)*4,
            velocity_jump_per_output_tick_radians=(.1,)*4,
            to_dict=lambda:dict(explicit_function_double=True))

    def test_head_arms_cannot_be_complete_with_missing_head_motion(self):
        complete,_,evidence=self.call(role=GroupOutputRole.FOCAL_HEAD_ARMS,
            known_parts=[(0,'left_arm'),(0,'right_arm')],head=None,arm=self.arm())
        self.assertFalse(complete);self.assertIsNone(evidence['head_motion'])
        self.assertIsNotNone(evidence['arm_motion'])

    def test_head_role_known_arm_still_requires_motion(self):
        complete,_,_=self.call(role=GroupOutputRole.FOCAL_HEAD,
            known_parts=[(0,'left_arm')],head=self.head(),arm=None)
        self.assertFalse(complete)

    def test_head_role_without_observed_arms_does_not_invent_them(self):
        complete,_,_=self.call(role=GroupOutputRole.FOCAL_HEAD,
            known_parts=[],head=self.head(),arm=None)
        self.assertTrue(complete)

    def test_head_arms_with_both_measured_modalities_is_eligible(self):
        complete,_,_=self.call(role=GroupOutputRole.FOCAL_HEAD_ARMS,
            known_parts=[(0,'left_arm'),(0,'right_arm')],head=self.head(),arm=self.arm())
        self.assertTrue(complete)

    def test_required_endpoint_arm_link_runs_even_for_head_only_role(self):
        sensor=object.__new__(scene.SceneSensors);sensor.endpoint_cache={}
        state=SimpleNamespace(key='state',role=GroupOutputRole.FOCAL_HEAD,
            first_output_frame=0,end_output_frame=60)
        sensor.hypothesis=lambda s:dict(analysis_roll=None)
        sensor.frame=lambda s,t:dict(selected_frame_local_head_id=0,
            role_obligations=dict(required_parts=[(4,'left_arm')]))
        sensor.raw=lambda s,t,r:SimpleNamespace(native_frame_index=t)
        sensor._head_links=lambda a,b:dict(unique_links=[[0,0]])
        with patch.object(scene,'native_body_links',return_value=dict(complete_required_joint_paths=False)) as link:
            with self.assertRaisesRegex(ValueError,'joint'):
                sensor.endpoint(state,outgoing=True)
        self.assertEqual(link.call_args.kwargs,dict(selected_head_ids=(0,0),selected_part_names=('left_arm',)))


class BodySupportPixelTests(unittest.TestCase):
    def observe_pair(self, a,b, *, ids=(0,1)):
        import hashlib
        class Raw:
            request_source_index=0;source_clock_identity='clock';source_file_identity=('chart',1)
            frame_size_wh=(320,180)
            def __init__(self,n):
                self.native_frame_index=n;self.sha256=f'RAW-CHART-{n}'
                self.native_pixel_sha256=hashlib.sha256(memoryview((a,b)[n])).hexdigest()
                self.people=tuple(ObservedFormationPerson(i,box,(),.99) for i,box in zip(ids,
                    ((.1,.2,.22,.5),(.6,.2,.22,.5))))
        counts={}
        def spend(k,v=1):counts[k]=counts.get(k,0)+v
        session=SimpleNamespace(native_pixels=lambda s,n:(a,b)[n],budget=SimpleNamespace(spend=spend))
        sensor=SimpleNamespace(session=session,link_cache={})
        return composition.body_support_links(sensor,Raw(0),Raw(1)),counts

    def chart(self,textured):
        image=np.zeros((180,320,3),np.uint8)
        for x in (32,192):
            image[36:126,x:x+70]=(40,210,60)
            if textured:
                for y in range(44,120,9):
                    for xx in range(x+8,x+65,9):image[y:y+3,xx:xx+3]=(10,10,10)
        return image

    def test_texture_real_lk_complete_but_never_joint_or_identity_authority(self):
        image=self.chart(True);link,counts=self.observe_pair(image,image.copy())
        self.assertTrue(link['complete_support_paths'])
        self.assertEqual(link['assignments'],[[0,0],[1,1]])
        self.assertEqual(counts['pose_comparisons'],2)
        self.assertEqual(len(link['native_pixel_sha256']),2)
        self.assertFalse(link['articulated_motion_certified'])
        self.assertFalse(link['named_identity_certified'])

    def test_flat_or_vanished_texture_stays_unknown(self):
        for a,b in ((self.chart(False),self.chart(False)),(self.chart(True),np.zeros((180,320,3),np.uint8))):
            with self.subTest(flat=not np.any(a!=b)):
                link,_=self.observe_pair(a,b)
                self.assertFalse(link['complete_support_paths'])

    def test_body_cohort_scope_rejects_heads_and_singleton(self):
        with self.assertRaises(ValueError):
            scene_role_obligations(dict(heads=[dict(observation_id=1)],people=[]),GroupOutputRole.COMPOSITION,None)
        with self.assertRaises(ValueError):
            scene_role_obligations(dict(heads=[],people=[dict(observation_id=0,part_states=[])]),GroupOutputRole.COMPOSITION,None)
