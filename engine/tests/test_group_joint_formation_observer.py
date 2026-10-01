from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from crosscut.group_joint_formation_observer import (observe_formation_people,
    declared_formation_observation,register_formation_models)
from crosscut.group_joint_source_camera import JointGroupWorkBudget


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

    def test_incorrect_model_asset_cannot_register_or_download_an_observer(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'invalid-fixture.onnx';path.write_bytes(b'not pinned model weights')
            calls=[]
            session=SimpleNamespace(budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.),
                register_model=lambda *a:calls.append(a))
            with self.assertRaisesRegex(ValueError,'pinned asset'):
                register_formation_models(session,person_model_path=path,pose_model_path=path)
            self.assertEqual(calls,[])
