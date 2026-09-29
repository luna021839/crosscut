from dataclasses import replace
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import numpy as np

from crosscut.direct_member_target_face import FocalFaceCandidate
from crosscut.focal_body_track import FocalBodyObservation, FocalBodyTrackResult
from crosscut.focal_selector_geometry import (FocalSelectorGeometry, FOCAL_SELECTOR_BODY_BASIS,
    common_focal_pose_vectors, focal_pose_shape_mismatch)
from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
from crosscut.direct_member_visual import _boundary_sample_with_focal_geometry, _boundary_mismatch, DirectMemberVisualConfig
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from engine.tests.test_focal_selector_geometry import person, descriptor, BOX
from engine.tests.test_focal_body_track_provider import FACE, observation
from crosscut.focal_body_track import FocalBodyFrame


def geometry(source=0, *, body_only=False):
    tracked = FocalBodyObservation(source, 1., person(), (1., 0.), .8)
    track = FocalBodyTrackResult(source, 1., tracked, 'observed_target_body_track', (.8, 1.2)) if body_only else None
    return FocalSelectorGeometry(source, 1., None if body_only else BOX,
        None if body_only else .035, person(), track, 'rtmpose_body7')


class SourceFirstGeometryTests(unittest.TestCase):
    def test_actual_resolver_uses_exact_face_pose_without_clothing_veto_or_track(self):
        provider = SourceFirstFocalGeometryProvider.__new__(SourceFirstFocalGeometryProvider)
        provider.sources = (Path('source'),)
        provider._identity = lambda _: 'fixed'
        provider._identities = ('fixed',)
        provider._persistent = None
        provider._persistent_key = lambda s, k: str(k)
        provider._cache = OrderedDict()
        provider._reasons = {}
        provider.faces = SimpleNamespace(
            _samplers=(SimpleNamespace(frame_at=lambda _: np.zeros((100, 100, 3), np.uint8)),),
            _identity_key=lambda s, sampler, t: (s, t),
            _ensure_focal_consensus=lambda s: SimpleNamespace(select=lambda _: FACE),
            _focal_candidates=lambda s, t: (FACE,))
        provider.bodies = SimpleNamespace(frame_observations=lambda s, t:
            FocalBodyFrame(s, t, (observation(),)),
            track_at=Mock(side_effect=AssertionError('direct proof does not need a track')))
        provider.pose = SimpleNamespace(describe=Mock(side_effect=AssertionError('pose already measured')))
        geometry = provider.selector_geometry(0, 1.)
        self.assertEqual(geometry.face_bbox_xywh, FACE.bbox_xywh)
        self.assertEqual(len(geometry.person.upper_body_keypoints_xyc), 8)
        self.assertIsNone(geometry.body_track)
        self.assertEqual(provider._reasons, {'frozen_face_with_exact_pose_agreement': 1})

    def test_observed_body_with_hidden_face_keeps_its_own_geometry(self):
        focal=geometry(body_only=True)
        self.assertIsNone(focal.face_bbox_xywh)
        self.assertIsNotNone(focal.person)
        self.assertFalse(focal.to_dict()['body_track']['face_geometry_extrapolated'])

    def test_other_source_track_cannot_authorize_this_body(self):
        with self.assertRaises(ValueError):
            replace(geometry(body_only=True), source_index=1)

    def test_single_anchor_certificate_is_not_track_proof(self):
        focal=geometry(body_only=True)
        with self.assertRaises(ValueError):
            replace(focal, body_track=replace(focal.body_track, support_seconds=(.8,)))

    def test_pose_backends_cannot_be_mixed_to_raise_coverage(self):
        left=geometry()
        right=replace(geometry(1), pose_backend='formation_pose')
        self.assertIsNone(focal_pose_shape_mismatch(left,right))
        self.assertIsNone(common_focal_pose_vectors((left,right)))

    def test_body_face_cut_compares_one_common_geometry_basis(self):
        samples=[]
        for focal in (geometry(body_only=True),geometry(1)):
            samples.append(_boundary_sample_with_focal_geometry(master_seconds=1.,
                source_local_seconds=1.,source_index=focal.source_index,descriptor=descriptor(),
                provider=SimpleNamespace(focal_geometry_at=lambda *a,f=focal:f)))
        native=direct_member_source_frame_geometry((2160,3840))
        result=_boundary_mismatch(*samples, DirectMemberVisualConfig(),outgoing_source_index=0,
            incoming_source_index=1,outgoing_source_geometry=native,incoming_source_geometry=native,
            continuous_official_portrait_fancam_contract=True)
        self.assertNotIn('cross_source_focal_subject_unresolved',result.severe_flags)
        self.assertEqual(result.attention_position_mismatch,0.)
        self.assertEqual(result.attention_scale_mismatch,0.)
        self.assertEqual(result.dominant_pose_mismatch,0.)

    def test_dense_body_sample_never_claims_a_face(self):
        provider=SourceFirstFocalGeometryProvider.__new__(SourceFirstFocalGeometryProvider)
        provider.selector_geometry=lambda *a:geometry(body_only=True)
        sample=provider.sample(0,1.)
        self.assertEqual(sample.face_count,0)
        self.assertEqual(sample.detector_backend,'none')
        self.assertEqual(sample.alignment_geometry_basis,FOCAL_SELECTOR_BODY_BASIS)
        self.assertEqual(sample.bbox_xywh,person().bbox_xywh)

    def test_dense_pair_uses_body_on_both_sides_when_one_face_is_hidden(self):
        provider=SourceFirstFocalGeometryProvider.__new__(SourceFirstFocalGeometryProvider)
        provider.selector_geometry=lambda s,t:geometry(s,body_only=s==0)
        provider.faces=SimpleNamespace(_focal_candidates=lambda *a:(FocalFaceCandidate(BOX,(1.,0.)),))
        provider._boundary_references=[]
        left,right=provider.paired_samples(0,1.,1,1.)
        self.assertEqual((left.face_count,right.face_count),(0,0))
        self.assertEqual(left.alignment_geometry_basis,right.alignment_geometry_basis)
        capture = provider.focal_boundary_reference_capture()[0]
        self.assertEqual(capture[:2], (0, 1))
        self.assertIsNone(capture[2])
        self.assertEqual(capture[3].bbox_xywh, BOX)

    def test_body_alignment_does_not_promote_neighbor_face_on_hidden_side(self):
        provider=SourceFirstFocalGeometryProvider.__new__(SourceFirstFocalGeometryProvider)
        provider.selector_geometry=lambda s,t:geometry(s,body_only=s==1)
        target=FocalFaceCandidate(BOX,(1.,0.))
        neighbor=FocalFaceCandidate((.7,.1,.1,.1),(0.,1.))
        provider.faces=SimpleNamespace(_focal_candidates=lambda s,t:(neighbor,target))
        provider._boundary_references=[]
        provider.paired_samples(0,1.,1,1.)
        self.assertEqual(provider.focal_boundary_reference_capture(),((0,1,target,None),))

    def test_duplicate_observed_face_box_does_not_select_first_identity(self):
        provider=SourceFirstFocalGeometryProvider.__new__(SourceFirstFocalGeometryProvider)
        provider.selector_geometry=lambda s,t:geometry(s)
        provider.faces=SimpleNamespace(_focal_candidates=lambda *a:(
            FocalFaceCandidate(BOX,(1.,0.)),FocalFaceCandidate(BOX,(0.,1.))))
        provider._boundary_references=[]
        provider.paired_samples(0,1.,1,1.)
        self.assertEqual(provider.focal_boundary_reference_capture(),((0,1,None,None),))

    def test_missing_target_does_not_use_dominant_descriptor(self):
        provider=SourceFirstFocalGeometryProvider.__new__(SourceFirstFocalGeometryProvider)
        provider.selector_geometry=lambda *a:FocalSelectorGeometry(0,1.,None)
        self.assertIsNone(provider.sample(0,1.))


if __name__=='__main__':unittest.main()
