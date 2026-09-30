from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from crosscut.boundary_subject_alignment import (
    FOCAL_EXPERIMENT_BASIS,
    _local_focal_observation_provider, build_boundary_subject_alignment_plan,
)
from crosscut.group_focal_attention import (
    CENTRAL_FOCAL_BASIS, GroupFocalSubject, focal_landmarks_track, resolve_group_focal_subject,
)
from crosscut.post_render_qa import _bound_local_alignment_tracks
from crosscut.render import RenderTransition
from crosscut.visual_score import VisualTimelinePlan
from tests import test_boundary_subject_alignment as boundary_tests
from tests import test_group_focal_attention as focal_tests
from tests.test_boundary_subject_alignment import _choice, _frame
from tests.test_group_focal_attention import person
from tests.test_visual_boundary_repair import _descriptor


def local_observation(*, small=True, moved=False):
    value = boundary_tests.BoundarySubjectAlignmentTests._tight_observation(small=small)
    if moved:
        x,y,w,h = value.face_box_xywh
        value = replace(value, face_box_xywh=(x+.18,y,w,h),
            face_landmarks_xy=tuple((px+.18,py) for px,py in value.face_landmarks_xy),
            formation_boxes_xywh=((.48,.23,.4,.77),))
    owner = person(value.formation_boxes_xywh[0])
    focal = GroupFocalSubject(value.face_box_xywh, value.face_landmarks_xy,
        0., value.face_eye_scale, owner, 2, CENTRAL_FOCAL_BASIS)
    return replace(value, face_count=2, face_head_roll_radians=0., selected_focal_subject=focal)


class GroupFocalAlignmentTests(unittest.TestCase):
    def test_landmarks_require_measured_pixel_flow_not_similar_face_boxes(self):
        frame=np.random.default_rng(95).integers(0,255,(96,160,3),dtype=np.uint8)
        shifted=cv2.warpAffine(frame,np.float32(((1,0,2),(0,1,1))),(160,96))
        points=np.float32(((.40,.40),(.60,.40),(.50,.50),(.42,.62),(.58,.62)))
        observed=points+np.float32((2/160,1/96))
        self.assertTrue(focal_landmarks_track(frame,shifted,points,observed))
        self.assertFalse(focal_landmarks_track(frame,shifted,points,observed+.2))
        self.assertFalse(focal_landmarks_track(np.zeros_like(frame),np.zeros_like(frame),points,points))
        self.assertFalse(focal_landmarks_track(frame,shifted,points,None))

    def test_shared_resolver_preserves_unknown_primary_and_explicit_local_basis(self):
        setup = focal_tests.GroupFocalAttentionTests()
        setup.setUp()
        focal = setup.select((setup.peripheral,setup.center))
        original = replace(_descriptor(.5), face_count=2, primary_face_bbox=None,
            primary_face_landmarks_xy=None, primary_face_eye_scale=None,
            formation_people=(setup.body,))
        with patch('crosscut.visual_score.describe_frame', return_value=original), patch(
                'crosscut.group_focal_attention.observe_central_focal_face',return_value=focal):
            motion = resolve_group_focal_subject(_frame(0), original)
            geometry = _local_focal_observation_provider(_frame(0))
        self.assertEqual(motion, geometry.selected_focal_subject)
        self.assertEqual(motion.person.bbox_xywh, geometry.formation_boxes_xywh[0])
        self.assertEqual(CENTRAL_FOCAL_BASIS, motion.basis)
        self.assertFalse(geometry.face_is_unambiguous_dominant)
        self.assertIsNone(original.primary_face_bbox)

    def test_existing_primary_with_unowned_body_cannot_switch_to_local_face(self):
        original = replace(_descriptor(.5), primary_face_bbox=(.02,.02,.06,.08),
            formation_people=(person((.3,.2,.4,.8)),))
        with patch('crosscut.group_focal_attention.observe_central_focal_face') as central:
            self.assertIsNone(resolve_group_focal_subject(_frame(0), original))
        central.assert_not_called()

    def test_focal_observation_rejects_forged_dominance_or_changed_owner(self):
        observed = local_observation()
        for changes in (dict(face_is_unambiguous_dominant=True),dict(face_count=1),
                dict(face_box_xywh=(.1,.2,.1,.2)),dict(formation_boxes_xywh=()),
                dict(selected_focal_subject=replace(observed.selected_focal_subject,
                    landmarks_xy=None)),dict(selected_focal_subject='central')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(observed,**changes)

    def build(self, *, moved=False, flow=True, local=True):
        timeline = VisualTimelinePlan((_choice(0,0,start=0.,end=1.),
            _choice(1,1,start=1.,end=2.)),1.)
        def observe(frame):
            marker = round(float(frame.mean()))
            return local_observation(small=marker<75,moved=marker==60)
        with patch('crosscut.boundary_subject_alignment._optical_flow_box',
                side_effect=lambda _a,_b,box: box if flow else None):
            return build_boundary_subject_alignment_plan(visual_timeline=timeline,
                sources=(Path('a.mp4'),Path('b.mp4')),
                transitions=(RenderTransition(tier='C'),),fps=30.,
                frame_provider=lambda index,seconds:_frame(index,
                    marker=60 if moved and index==0 and round(seconds*30)==26 else None),
                observation_provider=observe,source_resolutions=((3840,2160),)*2,
                local_focal_attention=local)

    def test_dense_local_camera_requires_adjacent_frame_ownership(self):
        valid = self.build()
        self.assertEqual(valid.eligible_count,1)
        self.assertGreater(valid.validated_output_frame_count,0)
        self.assertEqual(valid.analysis_basis,FOCAL_EXPERIMENT_BASIS)
        for options in (dict(moved=True),dict(flow=False)):
            with self.subTest(options=options):
                rejected = self.build(**options)
                self.assertEqual(rejected.eligible_count,0)
                self.assertEqual(rejected.decisions[0].rejection_reason,
                                 'per_frame_local_focal_ownership_unverified')

    def test_local_evidence_cannot_claim_v5_or_production_binding(self):
        without_opt_in = self.build(local=False)
        self.assertEqual(without_opt_in.eligible_count,0)
        experimental = self.build()
        self.assertEqual(_bound_local_alignment_tracks(
            {'boundary_subject_alignment': experimental.to_dict()}, {}), (set(),set()))


if __name__ == '__main__':
    unittest.main()
