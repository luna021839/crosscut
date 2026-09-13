from dataclasses import replace
from unittest import TestCase

from crosscut.joint_boundary_provider import _crop_resolved_transforms
from crosscut.paired_face_crop_geometry import matched_face_crops
from crosscut.visual_boundary_repair import _face_resolved_scale_delta, _visual_evidence
from engine.tests.test_visual_boundary_repair import _descriptor
from engine.tests.test_joint_timeline import candidate


def face(height, *, basis='dominant_face_height'):
    return replace(_descriptor(.5),primary_face_bbox=(.4,.1,.15,height),
                   shot_scale=min(1.,height/.42),shot_scale_basis=basis,face_count=1)


class DecodedFaceScaleGeometryTests(TestCase):
    def test_saturated_closeups_keep_actual_size_difference(self):
        left,right=face(.694),face(.471)
        self.assertEqual(left.shot_scale,right.shot_scale)
        self.assertAlmostEqual(_face_resolved_scale_delta(left,right),(.694-.471)/.42)
        evidence=_visual_evidence(left,right)
        self.assertAlmostEqual(evidence.shot_scale_delta,(.694-.471)/.42)

    def test_unsaturated_face_units_are_unchanged(self):
        self.assertAlmostEqual(_face_resolved_scale_delta(face(.1),face(.3)),
                              abs(face(.1).shot_scale-face(.3).shot_scale))

    def test_equal_faces_and_reversed_comparison(self):
        self.assertEqual(_face_resolved_scale_delta(face(.6),face(.6)),0.)
        self.assertEqual(_face_resolved_scale_delta(face(.6),face(.4)),
                         _face_resolved_scale_delta(face(.4),face(.6)))

    def test_body_or_unknown_face_basis_is_not_invented(self):
        a,b=face(.6,basis='legacy'),face(.4)
        self.assertEqual(_face_resolved_scale_delta(a,b),abs(a.shot_scale-b.shot_scale))
        a=replace(a,primary_face_bbox=None)
        self.assertEqual(_face_resolved_scale_delta(a,b),abs(a.shot_scale-b.shot_scale))

    def test_large_difference_is_still_a_normalized_mismatch(self):
        self.assertEqual(_face_resolved_scale_delta(face(.8),face(.1)),1.)

    def test_actual_provider_partial_proposal_does_not_enlarge_the_larger_face(self):
        boxes=((.3133149783,.3379135132,.2100918611,.5263528329),
               (.4109475454,.2181653623,.0984589179,.2360827411))
        result=_crop_resolved_transforms(raw_context_identity='a'*64,
            outgoing_candidate=candidate(0,'0',.8), incoming_candidate=candidate(1,'1',.8),
            outgoing_descriptor=replace(face(.526),primary_face_bbox=boxes[0]),
            incoming_descriptor=replace(face(.236),primary_face_bbox=boxes[1]))
        self.assertIsNotNone(result)
        self.assertIsNone(result[0].crop_xywh)
        expected=matched_face_crops(*boxes,allow_partial_extent_match=True)[1]
        for a,b in zip(result[1].crop_xywh,expected):self.assertAlmostEqual(a,b)
