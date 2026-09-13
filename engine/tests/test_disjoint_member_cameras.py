from dataclasses import replace
import unittest
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.disjoint_member_cameras import compose_disjoint_member_cameras

def track(points,reason='confirmed_member_subject_scale_position_match'):
    return EditorialTransformTrack(tuple(EditorialTransformKeyframe(t,z) for t,z in points),reason)

class DisjointMemberCameraTests(unittest.TestCase):
    def test_preserves_both_cameras_on_every_frame_and_between_frames(self):
        a=track(((0.,1.1),(.75,1.),(3.9,1.)))
        b=track(((0.,1.),(3.,1.),(4.,1.3)),'joint_dissolve_outgoing_ramp_pending_containment_review')
        c=compose_disjoint_member_cameras(a,b,duration=4.)
        for i in range(1001):
            t=4*i/1000
            actual=c.evaluate(t);expected=(a if t<=.75 else b).evaluate(t)
            for key in ('zoom','center_x','center_y','roll_degrees'):
                self.assertAlmostEqual(getattr(actual,key),getattr(expected,key),places=10)

    def test_collision_never_discards_or_retimes_either_camera(self):
        a=track(((0.,1.1),(1.5,1.),(4.,1.)))
        b=track(((0.,1.),(1.,1.),(4.,1.3)))
        with self.assertRaisesRegex(ValueError,'overlap'):compose_disjoint_member_cameras(a,b,duration=4.)

    def test_unowned_track_and_shorter_clock_rejected(self):
        a=track(((0.,1.1),(.75,1.),(4.,1.)))
        b=track(((0.,1.),(3.,1.),(4.,1.3)))
        with self.assertRaises(ValueError):compose_disjoint_member_cameras(replace(a,reason='unowned'),b,duration=4.)
        with self.assertRaises(ValueError):compose_disjoint_member_cameras(a,b,duration=3.)

if __name__=='__main__':unittest.main()
