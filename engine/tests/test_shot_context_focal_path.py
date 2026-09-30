from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from crosscut.boundary_subject_alignment import BoundarySubjectObservation
from crosscut.group_focal_attention import GroupFocalSubject,CENTRAL_FOCAL_BASIS,GLOBAL_FOCAL_BASIS
from crosscut.pose import FormationPersonDescriptor
from crosscut.shot_context_camera import fixed_neighbor_face_state,_source_local_focal_path
from engine.tests import test_shot_context_camera as fixtures

face=fixtures.face


def focal_face(height=.2,*,central=True,x=.44):
    observed=face(height,x=x);box=(.25,.2,.5,.8)
    body=FormationPersonDescriptor(box,(.5,.6),.95)
    focal=GroupFocalSubject(observed.face_box_xywh,observed.face_landmarks_xy,
        observed.face_head_roll_radians,observed.face_eye_scale,body,2,
        CENTRAL_FOCAL_BASIS if central else GLOBAL_FOCAL_BASIS)
    return replace(observed,face_count=2,face_is_unambiguous_dominant=not central,
        formation_boxes_xywh=(box,),selected_focal_subject=focal)


class ShotContextFocalPathTests(unittest.TestCase):
    def test_central_face_needs_explicit_mode_without_forging_global_dominance(self):
        source=focal_face();target=face(.26)
        self.assertFalse(source.face_is_unambiguous_dominant)
        with self.assertRaisesRegex(ValueError,'observed unambiguous'):
            fixed_neighbor_face_state(source,target,maximum_zoom=2.)
        result=fixed_neighbor_face_state(source,target,maximum_zoom=2.,local_focal_attention=True)
        self.assertAlmostEqual(result.zoom,1.3)
        self.assertFalse(source.face_is_unambiguous_dominant)

    def test_background_face_during_occlusion_cannot_change_track_owner(self):
        frames=[np.full((16,16,3),n,np.uint8) for n in range(5)]
        front=focal_face();wrong=focal_face(x=.30,central=False)
        observations=[front,front,wrong,front,front]
        def tracks(a,b,before,after):
            return 'measured_landmarks_forward_backward_pixel_flow' if after==front else None
        with patch('crosscut.shot_context_camera._focal_pixel_path_step',side_effect=tracks) as track:
            accepted,provenance=_source_local_focal_path(frames,observations,lambda:None)
        self.assertIsNone(accepted[2].face_box_xywh)
        self.assertIsNone(provenance[2])
        self.assertEqual(provenance[3]['tracked_from_frame'],1)
        self.assertIs(track.call_args_list[2].args[0],frames[1])

    def test_three_misses_missing_owner_and_unobserved_anchor_stay_unavailable(self):
        frames=[np.zeros((16,16,3),np.uint8)]*5;front=focal_face()
        for observations in ([front,face(),face(),face(),front],
                [BoundarySubjectObservation(),front,front,front,front]):
            with patch('crosscut.shot_context_camera._focal_pixel_path_step',return_value=None), \
                    self.assertRaisesRegex(ValueError,'source-local focal path is unobserved'):
                _source_local_focal_path(frames,observations,lambda:None)

    def test_real_proposal_keeps_focal_basis_distinct_and_requires_temporal_pixels(self):
        helper=fixtures.ShotContextProposalTests();args=helper.fixture()
        with patch('crosscut.shot_context_camera._local_focal_observation_provider',return_value=focal_face()), \
                patch('crosscut.shot_context_camera._focal_pixel_path_step',return_value='measured_landmarks_forward_backward_pixel_flow'):
            proposal=helper.run_proposal(args,local_focal_attention=True)
        self.assertTrue(proposal.local_focal_attention)
        self.assertEqual(len(proposal.head_evidence),30)
        self.assertTrue(all(r['basis']=='observed_focal_face_head_envelope' for r in proposal.head_evidence))
        self.assertTrue(all(r['face_selection_basis']==CENTRAL_FOCAL_BASIS for r in proposal.head_evidence))
        self.assertIsNone(proposal.head_evidence[0]['tracked_from_frame'])
        self.assertEqual(proposal.head_evidence[-1]['tracked_from_frame'],28)
        self.assertFalse(proposal.to_dict()['source_local_performer_identity_certified'])
        from crosscut.shot_context_binding import build_shot_context_binding
        with self.assertRaisesRegex(ValueError,'unpromoted experiment'):
            build_shot_context_binding(proposal)


if __name__=='__main__':unittest.main()
