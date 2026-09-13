from dataclasses import replace
from unittest import TestCase

from crosscut.joint_boundary_provider import (
    _exact_face_geometry, _updated_outgoing_candidate, _updated_incoming_candidate,
)
from crosscut.joint_timeline import BoundaryTransformState
from crosscut.timeline import face_geometry_minimax_mismatch, _transition_score
from crosscut.visual_boundary_repair import _visual_evidence
from engine.tests.test_joint_timeline import candidate, isolated_score_config
from engine.tests.test_visual_boundary_repair import _descriptor


GEOMETRY=(.4,.3,.6,.3,.2,0.,0.,.6,1.,0.)


class JointExactFaceGeometryTests(TestCase):
    def setUp(self):
        self.descriptor=replace(_descriptor(.5),primary_face_bbox=(.35,.2,.3,.4),
            primary_face_geometry=GEOMETRY)
        self.transform=BoundaryTransformState('identity',track_id='track')

    def test_exact_frame_replaces_stale_seed_window_on_both_sides(self):
        seed=candidate(0,'0',.8)
        stale=(tuple(v+.02 for v in GEOMETRY),)*3
        seed=replace(seed,entry_face_geometry_window=stale,exit_face_geometry_window=stale)
        evidence=_visual_evidence(self.descriptor,self.descriptor)
        for update,field,untouched in [(_updated_outgoing_candidate,'exit_face_geometry_exact','entry_face_geometry_window'),
                                        (_updated_incoming_candidate,'entry_face_geometry_exact','exit_face_geometry_window')]:
            measured=update(seed,self.descriptor,evidence,scene_cut_near=False,
                face_identity=(),transform=self.transform,transformed_shot_scale=None)
            self.assertEqual(getattr(measured,field),GEOMETRY)
            self.assertEqual(getattr(measured,field.replace('exact','window')),())
            self.assertEqual(getattr(measured,untouched),stale)

    def test_observed_roll_and_head_geometry_now_affect_existing_objective(self):
        changed=replace(self.descriptor,primary_face_geometry=(*GEOMETRY[:5],.4,.4,.6,1.,0.))
        a=_exact_face_geometry(self.descriptor,self.transform,descriptor_already_transformed=False)
        b=_exact_face_geometry(changed,self.transform,descriptor_already_transformed=False)
        self.assertEqual(face_geometry_minimax_mismatch((a,),(a,)),0.)
        self.assertGreater(face_geometry_minimax_mismatch((a,),(b,)),.05)

    def test_real_transition_score_honors_profile_weight_without_new_hard_gate(self):
        left=replace(candidate(0,'0',.8),exit_face_geometry_exact=GEOMETRY)
        right=replace(candidate(1,'1',.8),entry_face_geometry_exact=(*GEOMETRY[:5],.4,.4,.6,1.,0.))
        config=replace(isolated_score_config(),face_geometry_transition_weight=.72)
        weighted=_transition_score(left,right,1,config)
        neutral=_transition_score(left,right,1,replace(config,face_geometry_transition_weight=0.))
        self.assertIsNotNone(weighted)
        self.assertIsNotNone(neutral)
        self.assertLess(weighted[0],neutral[0])
        same=replace(right,entry_face_geometry_exact=GEOMETRY)
        self.assertAlmostEqual(_transition_score(left,same,1,config)[0],neutral[0])

    def test_exact_provider_authorized_options_keep_measured_geometry(self):
        from engine.tests.test_joint_boundary_provider import _provider, _key, _descriptor as provider_descriptor
        descriptor=replace(provider_descriptor(.5,face_bbox=(.35,.2,.3,.4),pose_value=.1),
            primary_face_geometry=GEOMETRY)
        provider=_provider(lambda _source,_seconds:descriptor)
        options=provider(_key()).authorized_options
        self.assertTrue(options)
        for option in options:
            self.assertEqual(option.scoring_outgoing.exit_face_geometry_exact,GEOMETRY)
            self.assertEqual(option.scoring_incoming.entry_face_geometry_exact,GEOMETRY)
            self.assertEqual(option.scoring_outgoing.exit_face_geometry_window,())

    def test_unremeasured_crop_does_not_claim_rendered_eye_geometry(self):
        transform=BoundaryTransformState('crop',crop_xywh=(.1,.1,.8,.8),track_id='track')
        self.assertEqual(_exact_face_geometry(self.descriptor,transform,descriptor_already_transformed=False),())
        self.assertEqual(_exact_face_geometry(self.descriptor,transform,descriptor_already_transformed=True),GEOMETRY)

    def test_unremeasured_rotation_remains_unknown(self):
        transform=replace(self.transform,rotation_degrees=3.)
        self.assertEqual(_exact_face_geometry(self.descriptor,transform,descriptor_already_transformed=False),())

    def test_missing_ambiguous_or_invalid_face_data_is_not_invented(self):
        for descriptor in [replace(self.descriptor,primary_face_bbox=None),
                           replace(self.descriptor,primary_face_geometry=None),
                           replace(self.descriptor,primary_face_geometry=(0.,)*10),
                           replace(self.descriptor,primary_face_geometry=(float('nan'),)*10)]:
            self.assertEqual(_exact_face_geometry(descriptor,self.transform,descriptor_already_transformed=False),())
