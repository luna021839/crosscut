from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

import numpy as np

from crosscut.group_joint_formation_observer import (observe_formation_people,
    declared_formation_observation,register_formation_models,observe_temporal_formation_people)
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from engine.tests.test_group_joint_formation import person


class FormationObserverTests(unittest.TestCase):
    def session(self,rows,*,missing=False):
        calls=[]
        def observe(state,tick,*,model,arguments=None):
            calls.append((model,arguments))
            if model=='nanodet_people':return rows
            return SimpleNamespace(reason='pose_model_unavailable' if missing else 'observed',
                upper_body_xyc=((.5,.5,.9),)*8)
        return SimpleNamespace(observe=observe),calls

    def test_each_person_has_a_distinct_actual_top_down_model_roi(self):
        rows=np.asarray(((10,5,30,45,.9),(60,5,80,45,.8)),dtype=float)
        session,calls=self.session(rows)
        state=SimpleNamespace(output_size=(100,50))
        people=observe_formation_people(session,state,15)
        self.assertEqual(len(people),2)
        self.assertEqual(people[0].box_xywh,(.1,.1,.2,.8))
        self.assertEqual(people[1].box_xywh,(.6,.1,.2,.8))
        self.assertEqual([n for n,_ in calls],['nanodet_people','rtmpose_body7','rtmpose_body7'])
        self.assertNotEqual(calls[1][1],calls[2][1])

    def test_missing_body_pose_does_not_drop_a_required_occluded_person(self):
        session,calls=self.session(np.asarray(((10,5,30,45,.9),(60,5,80,45,.8))),missing=True)
        observed=declared_formation_observation(session,SimpleNamespace(output_size=(100,50)),15,
            required_observation_ids=(0,1))
        self.assertEqual(len(observed.required),2)
        self.assertTrue(all(p.upper_body_xyc==() for p in observed.required))

    def test_unresolved_detection_overflow_is_not_silently_truncated(self):
        session,calls=self.session(np.asarray(((10,5,30,45,.9),)*13))
        self.assertIsNone(observe_formation_people(session,SimpleNamespace(output_size=(100,50)),15))
        self.assertEqual([n for n,_ in calls],['nanodet_people'])

    def test_missing_required_detection_is_explicit_in_the_observer_result(self):
        session, _ = self.session(np.asarray(((10, 5, 30, 45, .9),)))
        result = declared_formation_observation(session, SimpleNamespace(output_size=(100, 50)), 15,
            required_observation_ids=(0, 1))
        self.assertIsNotNone(result)
        self.assertEqual(result.missing_required_observation_ids, (1,))
        self.assertEqual(result.required_observation_ids, (0, 1))

    def test_incorrect_model_asset_cannot_register_or_download_an_observer(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'invalid-fixture.onnx';path.write_bytes(b'not pinned model weights')
            calls=[]
            session=SimpleNamespace(budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.),
                register_model=lambda *a:calls.append(a))
            with self.assertRaisesRegex(ValueError,'pinned asset'):
                register_formation_models(session,person_model_path=path,pose_model_path=path)
            self.assertEqual(calls,[])

    def temporal_fixture(self):
        first=(person(0,.2),person(1,.8))
        points=list(first[0].upper_body_xyc);x,y,c=points[2];points[2]=(x,y+.03,c)
        second=(replace(first[0],upper_body_xyc=tuple(points)),first[1])
        state=SimpleNamespace(first_output_frame=0,end_output_frame=30,output_fps=30.,
            native_frame_indices=tuple(range(30)),key='a'*64,output_size=(100,50))
        session=SimpleNamespace(projected_pixels=lambda s,t:np.zeros((50,100,3),np.uint8))
        return state,session,first,second

    def test_temporal_measurement_owns_exact_ticks_and_actual_source_side_pixel_path(self):
        state,session,first,second=self.temporal_fixture()
        with patch('crosscut.group_joint_formation_observer.observe_formation_people',side_effect=(first,second)),\
                patch('crosscut.boundary_subject_alignment._optical_flow_box',side_effect=[p.box_xywh for p in second]):
            result=observe_temporal_formation_people(session,state,outgoing=True)
        value=result[0].temporal_motion
        self.assertEqual(value.output_frames,(26,29))
        self.assertEqual(value.native_frames,(26,29))
        self.assertEqual(value.state_key,state.key)
        self.assertTrue(value.same_source_pixel_path_verified)
        self.assertTrue(any(abs(x)+abs(y)>0 for _,x,y in value.bone_velocities))
        self.assertTrue(all(x==0 and y==0 for _,x,y in result[1].temporal_motion.bone_velocities))

    def test_missing_pixel_flow_is_no_motion_not_zero_motion(self):
        state,session,first,second=self.temporal_fixture()
        with patch('crosscut.group_joint_formation_observer.observe_formation_people',side_effect=(first,second)),\
                patch('crosscut.boundary_subject_alignment._optical_flow_box',return_value=None):
            result=observe_temporal_formation_people(session,state,outgoing=False)
        self.assertTrue(all(p.temporal_motion is None for p in result))

    def test_two_previous_people_cannot_both_lend_motion_to_one_visible_current_person(self):
        state,session,first,second=self.temporal_fixture()
        with patch('crosscut.group_joint_formation_observer.observe_formation_people',side_effect=(first,second[:1])),\
                patch('crosscut.boundary_subject_alignment._optical_flow_box',return_value=second[0].box_xywh):
            result=observe_temporal_formation_people(session,state,outgoing=True)
        self.assertIsNone(result[0].temporal_motion)
