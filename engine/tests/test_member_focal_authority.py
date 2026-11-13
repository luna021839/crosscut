import unittest
from types import SimpleNamespace

from crosscut.direct_member_rendered_continuity import _focal_face_window
from crosscut.direct_member_target_face import FocalFaceCandidate


class MemberFocalAuthorityTests(unittest.TestCase):
    def test_missing_requested_output_face_does_not_promote_different_matching_performer(self):
        target=FocalFaceCandidate((.4,.2,.15,.1),(1.,0.))
        other=FocalFaceCandidate((.4,.2,.15,.1),(0.,1.))
        provider=SimpleNamespace(focal_face_candidates=lambda seconds:(other,))
        self.assertIsNone(_focal_face_window(provider,frame_indices=(0,1,2,3),fps=30.,
            outgoing_references=(target,),incoming_references=(target,)))

    def test_source_bound_focal_face_is_retained_among_neighbours(self):
        target=FocalFaceCandidate((.4,.2,.15,.1),(1.,0.))
        other=FocalFaceCandidate((.6,.2,.15,.1),(0.,1.))
        provider=SimpleNamespace(focal_face_candidates=lambda seconds:(other,target))
        result=_focal_face_window(provider,frame_indices=(0,1,2,3),fps=30.,
            outgoing_references=(target,),incoming_references=(target,))
        self.assertIsNotNone(result)
        self.assertTrue(result[2])
        self.assertEqual(result[0],(target,target,target,target))


if __name__=='__main__':
    unittest.main()
