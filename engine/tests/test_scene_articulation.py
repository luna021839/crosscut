"""NONHUMAN pose metadata/registered-model substitutes; real joint pixel LK.

No human inference, actor identity, normal-app admission or output quality is
claimed. The normal-path integration is tested separately on real encoded charts.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import unittest

import numpy as np

from crosscut import group_scene_articulation as art
from crosscut.body_pose_2d import pose_input_geometry
from crosscut.group_scene_composition import NativeCompositionObservation
from crosscut.group_joint_person_ownership import OwnedPersonObservation,bind_person_ownership,ownership_policy_fingerprint
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.group_performer_family_owner import plain
from engine.tests.test_group_joint_formation import person


def owned(p):
    _,_,affine,crop=pose_input_geometry((1000,800),p.box_xywh)
    raw=[(p.box_xywh[0]+p.box_xywh[2]/2,.3,.9)]*17
    for index,point in zip((5,6,7,8,9,10,11,12),p.upper_body_xyc):raw[index]=point
    return OwnedPersonObservation(p.observation_id,p.box_xywh,p.upper_body_xyc,p.confidence,
        pose_crop_xywh=crop,source_pixel_to_pose_input_affine=tuple(tuple(map(float,row)) for row in affine),
        raw_pose_xyc=tuple(raw),pose_reason='observed',face_anchor_consistent=None)


class Budget:
    def spend(self,*args):pass


class PoseProducerTests(unittest.TestCase):
    def fixture(self):
        p=person(0,.25)
        support=NativeCompositionObservation(0,10,(1000,800),'a'*64,(1,2,3,4,5),'b'*64,
            (('nanodet_people','1'*64),),(replace(p,upper_body_xyc=()),),(),
            'EXPLICIT_NONHUMAN_RECTANGLE',(('nanodet_people','3'*64),))
        subject=owned(p)
        calls=[]
        def observe(source,native,*,model,arguments):
            calls.append((source,native,model,arguments))
            return SimpleNamespace(reason='observed',upper_body_xyc=subject.upper_body_xyc,
                pose_crop_xywh=subject.pose_crop_xywh,source_pixel_to_pose_input_affine=subject.source_pixel_to_pose_input_affine,
                raw_keypoints_xyc=subject.raw_pose_xyc,face_anchor_consistent=None)
        session=SimpleNamespace(models={'rtmpose_body7':(None,None,'2'*64)},
            model_adapters={'rtmpose_body7':'4'*64},observe_native=observe)
        return support,session,calls

    def test_registered_pose_call_retains_raw_support_without_focal_identity(self):
        support,session,calls=self.fixture()
        row=art.observe_composition_pose(session,support)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][3],{'bbox_xywh':list(support.people[0].box_xywh)})
        self.assertEqual(row.people[0].body_ownership_state,'observed')
        self.assertIsNone(row.people[0].unique_head_observation_id)
        self.assertIsNone(row.people[0].face_anchor_consistent)
        self.assertFalse(row.to_dict()['semantic_cast_membership_certified'])
        self.assertEqual(support.to_dict()['articulated_pose_state'],'not_requested')

    def test_missing_registered_model_stays_unknown(self):
        support,session,_=self.fixture();session.models={}
        self.assertIsNone(art.observe_composition_pose(session,support))

    def test_duplicate_skeletons_are_preserved_as_ambiguous_not_merged(self):
        support,session,_=self.fixture()
        support=replace(support,people=(*support.people,replace(support.people[0],observation_id=1)))
        row=art.observe_composition_pose(session,support)
        self.assertEqual(len(row.people),2)
        self.assertTrue(all(p.body_ownership_state=='ambiguous' for p in row.people))


class ArticulatedEndpointTests(unittest.TestCase):
    def fixture(self):
        images={};cache={};raws={}
        for source,first,ticks in ((0,0,(56,59)),(1,60,(60,63))):
            start_native,end_native=(56,59) if source==0 else (0,3)
            rng=np.random.default_rng(14)
            people=tuple(person(i,.25+.45*i,reverse=bool(i)) for i in range(2))
            patches={(p.observation_id,j):rng.integers(10,240,(17,17,3),dtype=np.uint8)
                for p in people for j in range(8)}
            for step in range(4):
                image=np.zeros((800,1000,3),np.uint8)
                for p in people:
                    for j,(x,y,_) in enumerate(p.upper_body_xyc):
                        dx=step if j in (0,1,6,7) else step*2 if j in (2,4) else -step
                        xx,yy=round(x*1000)+dx,round(y*800)
                        image[yy-8:yy+9,xx-8:xx+9]=patches[p.observation_id,j]
                images[source,start_native+step]=image
            for tick,native,step in ((ticks[0],start_native,0),(ticks[1],end_native,3)):
                moved=tuple(replace(p,upper_body_xyc=tuple((x+(step if j in (0,1,6,7)
                    else step*2 if j in (2,4) else -step)/1000,y,c)
                    for j,(x,y,c) in enumerate(p.upper_body_xyc))) for p in people)
                actual=tuple(owned(p) for p in moved)
                # A third unresolved support is present only at the old endpoint.
                if step==0:actual=(*actual,replace(owned(person(7,.95)),upper_body_xyc=(),raw_pose_xyc=(),pose_reason='occluded'))
                actual=bind_person_ownership(actual,(),(1000,800))[0]
                bodies=tuple(replace(person(p.observation_id,p.box_xywh[0]+p.box_xywh[2]/2),
                    upper_body_xyc=(),box_xywh=p.box_xywh) for p in actual)
                args=(source,native,(1000,800),'a'*64,(1,2,3,4,5),sha256(memoryview(images[source,native])).hexdigest())
                base=NativeCompositionObservation(*args,(('nanodet_people','1'*64),),bodies,(),
                    'EXPLICIT_NONHUMAN_RECTANGLES',(('nanodet_people','3'*64),))
                pose=art.NativeCompositionPoseObservation(*args,(('nanodet_people','1'*64),('rtmpose_body7','2'*64)),
                    actual,(),'EXPLICIT_NONHUMAN_POSES',(('nanodet_people','3'*64),('rtmpose_body7','4'*64)),
                    ownership_policy_fingerprint())
                raws[source,tick]=base;cache[base.sha256]=pose
        def state(source,first):
            segment=SimpleNamespace(pan_crop=None,crop=None,editorial_transform=None)
            return SimpleNamespace(key=str(source+5)*64,role=GroupOutputRole.COMPOSITION,
                eligible_source_index=source,first_output_frame=first,end_output_frame=first+60,
                native_frame_indices=tuple(range(60)),segments=(segment,),segment_output_frame_counts=(60,),
                output_size=(1000,800),output_fps=30.)
        sensors=SimpleNamespace(articulation_raw_rows=cache,articulation_endpoint_cache={},
            session=SimpleNamespace(native_pixels=lambda s,n:images.get((s,n)),budget=Budget()),
            raw=lambda state,tick,roll:raws[state.eligible_source_index,tick])
        return sensors,state(0,0),state(1,60)

    def test_actual_joint_pixel_steps_keep_occluded_support_unmatched(self):
        sensors,left,_=self.fixture()
        view,evidence=art.endpoint_articulation(sensors,left,outgoing=True)
        self.assertIsNotNone(view)
        self.assertEqual(evidence['unmatched_observation_ids'],[[7],[]])
        self.assertEqual(evidence['native_frame_indices'],[56,57,58,59])
        self.assertTrue(evidence['native_steps_complete'])
        self.assertFalse(evidence['complete_semantic_cohort'])
        self.assertEqual(len(view.people),2)

    def test_two_sided_partial_articulation_and_consumer_binding(self):
        sensors,left,right=self.fixture()
        value=plain(art.compare_composition_articulation(sensors,left,right))
        self.assertTrue(value['available'])
        self.assertFalse(value['cross_camera_identity_certified'])
        assets={'person_model_path':{'sha256':'1'*64},'pose_model_path':{'sha256':'2'*64}}
        art.validate_articulation_evidence(value,value['metrics'],assets)
        for key in ('quality_approved','body_phase_certified','complete_semantic_cohort'):
            bad=deepcopy(value);bad[key]=True
            with self.assertRaises(ValueError):art.validate_articulation_evidence(bad,bad['metrics'],assets)
        bad=deepcopy(value);bad['endpoints'][0]['native_frame_indices'].pop(1)
        with self.assertRaises(ValueError):art.validate_articulation_evidence(bad,bad['metrics'],assets)
        bad=deepcopy(value);bad['endpoints'][0]['assignments'][1]=bad['endpoints'][0]['assignments'][0]
        with self.assertRaises(ValueError):art.validate_articulation_evidence(bad,bad['metrics'],assets)

    def test_flat_native_pixels_leave_pose_motion_unknown(self):
        sensors,left,right=self.fixture()
        sensors.session.native_pixels=lambda source,native:np.zeros((800,1000,3),np.uint8)
        with self.assertRaisesRegex(ValueError,'pixels changed'):
            art.compare_composition_articulation(sensors,left,right)
