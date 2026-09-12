from dataclasses import replace
import unittest

from crosscut.member_dissolve_camera import align_member_dissolve_cameras
from crosscut.direct_member_editorial_motion import DirectMemberFaceSample
from crosscut.direct_member_target_face import FOCAL_FACE_BASIS
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.render import RenderSegment, RenderTransition, TransitionKind, EditorialTransformKeyframe, EditorialTransformTrack


def face(source, seconds):
    box=(.42,.2,.16,.16) if source == 0 else (.44,.25,.12,.12)
    return DirectMemberFaceSample(box,1,'yunet',subject_bbox_xywh=(.25,.16,.50,.70),
        subject_confidence=.9,subject_geometry_basis='source_local_focal_body_track',
        alignment_geometry_basis_override=FOCAL_FACE_BASIS)


class MemberDissolveCameraTests(unittest.TestCase):
    def run_camera(self, provider=face, kind=TransitionKind.FACE_DISSOLVE, segments=None, **kwargs):
        geometries=(direct_member_source_frame_geometry((2160,3840),output_size=(1080,1920)),)*2
        return align_member_dissolve_cameras(render_segments=segments or (
            RenderSegment(0,0.,3.,0.),RenderSegment(1,3.,3.,3.)),
            transitions=(RenderTransition(kind=kind,frame_count=7 if kind is TransitionKind.FACE_DISSOLVE else 0,reason='test'),),
            source_indices=(0,1),eligible_request_source_indices=(0,1),source_geometries=geometries,
            common_master_start=0.,geometry_provider=provider,fps=30.,**kwargs)

    def test_aligns_incoming_and_holds_entire_overlap(self):
        segments, audit=self.run_camera()
        self.assertEqual(audit['applied_count'],1,audit)
        self.assertEqual(audit['boundaries'][0]['transformed_side'],'incoming')
        track=segments[1].editorial_transform
        self.assertIsNotNone(track)
        self.assertAlmostEqual(track.evaluate(0.).zoom,4/3)
        self.assertEqual(track.evaluate(0.).zoom,track.evaluate(7/30).zoom)
        self.assertEqual(track.evaluate(1.).zoom,1.)
        self.assertLess(audit['boundaries'][0]['residual_mean_position_plus_scale'],1e-8)
        self.assertFalse(audit['effects_changed'])

    def test_missing_overlap_frame_never_fills_with_neighbour(self):
        _,audit=self.run_camera(lambda s,t: None if s==1 and abs(t-(3+3/30))<1e-8 else face(s,t))
        self.assertEqual(audit['applied_count'],0)
        self.assertEqual(audit['boundaries'][0]['reason'],'overlap_focal_face_missing')

    def test_missing_ramp_frame_prevents_unverified_camera(self):
        _,audit=self.run_camera(lambda s,t: None if s==1 and abs(t-3.6)<1e-8 else face(s,t))
        self.assertEqual(audit['applied_count'],0)

    def test_existing_camera_is_preserved(self):
        track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.),EditorialTransformKeyframe(3.,1.1)),reason='existing')
        original=(RenderSegment(0,0.,3.,0.,editorial_transform=track),RenderSegment(1,3.,3.,3.))
        segments,audit=self.run_camera(segments=original)
        self.assertEqual(segments,original)
        self.assertEqual(audit['boundaries'][0]['reason'],'existing_camera_preserved')

    def test_hard_cut_and_source_mapping_are_not_reinterpreted(self):
        _,audit=self.run_camera(kind=TransitionKind.HARD_CUT)
        self.assertEqual(audit['measurement_count'],0)
        with self.assertRaises(ValueError):
            self.run_camera(segments=(RenderSegment(1,0.,3.,0.),RenderSegment(0,3.,3.,3.)))

    def test_unbound_face_is_not_focal_authority(self):
        _,audit=self.run_camera(lambda s,t:replace(face(s,t),alignment_geometry_basis_override=None))
        self.assertEqual(audit['applied_count'],0)


if __name__ == '__main__':unittest.main()
