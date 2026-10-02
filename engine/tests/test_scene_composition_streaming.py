"""Bound raw raster lifetime without weakening any body-support judgement.

Small, explicitly nonhuman rasters. Synthetic flow is used only for exact
ownership/lifetime negatives; a separate test compares actual OpenCV LK to
the previous person-major loop. No models or private media are loaded.
"""
from dataclasses import replace
import hashlib
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import cv2
import numpy as np

from crosscut import group_scene_composition as composition
from crosscut.group_joint_formation import ObservedFormationPerson
from crosscut.group_joint_source_camera import JointGroupWorkBudget


class RasterSession:
    def __init__(self, factory=None, missing=None):
        self.budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30)
        self.factory=factory or (lambda tick: np.full((64,96,3),tick,np.uint8))
        self.missing=missing
        self.frames=[]
        self.peak_live_rasters=0
        self.reads=[]
        self.retention=[]

    def native_pixels(self,source,tick):
        self.budget.spend('new_decodes')
        self.reads.append((source,tick))
        if tick==self.missing:return None
        pixels=self.factory(tick)
        self.frames.append(weakref.ref(pixels))
        self.peak_live_rasters=max(self.peak_live_rasters,sum(r() is not None for r in self.frames))
        return pixels

    def retain_native_pixels(self,keys):
        self.retention.append(keys)  # No cache in this explicit tiny fixture.


def observation(tick, factory=None, *, shift=.005):
    pixels=(factory or (lambda n:np.full((64,96,3),n,np.uint8)))(tick)
    people=tuple(ObservedFormationPerson(i,(x+tick*shift,.15,.22,.65),(),.9)
        for i,x in enumerate((.15,.60)))
    return composition.NativeCompositionObservation(0,tick,(96,64),'a'*64,(1,2,3,4,5),
        hashlib.sha256(memoryview(pixels)).hexdigest(),(('nanodet_people','b'*64),),
        people,(),'nonhuman_fixture',(('nanodet_people','c'*64),))


class SceneCompositionStreamingTests(unittest.TestCase):
    def sensors(self,**kwargs):
        return SimpleNamespace(session=RasterSession(**kwargs),link_cache={})

    @staticmethod
    def shifted(a,b,box):
        return (box[0]+.005, *box[1:])

    def test_full_gap_keeps_at_most_one_native_pair_alive(self):
        sensors=self.sensors()
        first,last=observation(0),observation(composition.MAX_NATIVE_GAP)
        # A MagicMock would retain every ndarray in call_args_list and itself
        # recreate the lifetime defect being measured.
        with patch.object(composition,'_optical_flow_box',new=self.shifted):
            result=composition.body_support_links(sensors,first,last)
        self.assertTrue(result['complete_support_paths'])
        self.assertEqual(result['assignments'],[[0,0],[1,1]])
        self.assertEqual(sensors.session.reads,[(0,n) for n in range(composition.MAX_NATIVE_GAP+1)])
        self.assertEqual(result['native_pixel_sha256'],[observation(n).native_pixel_sha256
            for n in range(composition.MAX_NATIVE_GAP+1)])
        self.assertLessEqual(sensors.session.peak_live_rasters,2)
        self.assertEqual(sensors.session.budget.counts['pose_comparisons'],2*composition.MAX_NATIVE_GAP)
        composition.validate_support_link(result,first.to_dict(),last.to_dict())

    def test_cached_link_neither_redecodes_nor_retracks(self):
        sensors=self.sensors();first,last=observation(0),observation(3)
        with patch.object(composition,'_optical_flow_box',side_effect=self.shifted) as flow:
            result=composition.body_support_links(sensors,first,last)
            work=dict(sensors.session.budget.counts)
            self.assertIs(composition.body_support_links(sensors,first,last),result)
        self.assertEqual(sensors.session.budget.counts,work)
        self.assertEqual(flow.call_count,6)

    def test_missing_intermediate_frame_stays_unavailable(self):
        sensors=self.sensors(missing=2)
        with patch.object(composition,'_optical_flow_box',side_effect=self.shifted):
            result=composition.body_support_links(sensors,observation(0),observation(3))
        self.assertFalse(result['complete_support_paths'])
        self.assertEqual(result['reason'],'intermediate_native_pixels_unavailable')
        self.assertEqual(result['assignments'],[])
        self.assertEqual(result['native_pixel_sha256'],[])

    def test_lost_roi_stops_only_its_own_path(self):
        sensors=self.sensors();calls=[]
        def flow(a,b,box):
            calls.append((int(a[0,0,0]),int(b[0,0,0]),box[0]>.5))
            return None if box[0]>.5 and int(b[0,0,0])==2 else self.shifted(a,b,box)
        with patch.object(composition,'_optical_flow_box',side_effect=flow):
            result=composition.body_support_links(sensors,observation(0),observation(4))
        self.assertEqual(result['assignments'],[[0,0]])
        self.assertFalse(result['complete_support_paths'])
        self.assertEqual(result['predicted_boxes'][1],None)
        self.assertEqual(len(calls),6)
        self.assertNotIn((2,3,True),calls)
        self.assertNotIn((3,4,True),calls)

    def test_invalid_gap_and_source_fail_before_any_raster(self):
        sensors=self.sensors()
        result=composition.body_support_links(sensors,observation(0),observation(13))
        self.assertFalse(result['complete_support_paths'])
        self.assertEqual(sensors.session.reads,[])
        with self.assertRaises(ValueError):
            composition.body_support_links(sensors,observation(0),replace(observation(3),request_source_index=1))
        self.assertEqual(sensors.session.reads,[])

    def test_real_lk_matches_prior_person_major_loop(self):
        def textured(tick):
            image=np.full((64,96,3),60,np.uint8)
            for bx in (16,59):
                for y in range(12,51,5):
                    for x in range(bx,bx+19,5):
                        cv2.rectangle(image,(x,y),(x+1,y+1),(220,220,220),-1)
            return image
        first,last=observation(0,textured,shift=0),observation(6,textured,shift=0)
        frames=[textured(n) for n in range(7)]
        legacy=[]
        for person in first.people:
            box=person.box_xywh
            for a,b in zip(frames,frames[1:]):
                box=composition._optical_flow_box(a,b,box)
                if box is None:break
            legacy.append(None if box is None else list(box))
        del frames,a,b
        sensors=self.sensors(factory=textured)
        result=composition.body_support_links(sensors,first,last)
        self.assertEqual(result['predicted_boxes'],legacy)
        self.assertTrue(all(box is not None for box in legacy),'nonhuman LK positive must actually track')
        self.assertTrue(result['complete_support_paths'])
        self.assertLessEqual(sensors.session.peak_live_rasters,2)


if __name__=='__main__':unittest.main()
