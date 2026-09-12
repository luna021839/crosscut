from dataclasses import replace
import unittest

from crosscut.focal_pose import face_associated_upper_body_person
from crosscut.pose import FormationPersonDescriptor
from crosscut.timeline import _upper_body_shape_mismatch


POINTS = ((-.2, -.3, .9), (.2, -.3, .9), (-.4, 0, .9), (.4, 0, .9),
          (-.5, .2, .9), (.5, .2, .9), (-.15, .3, .9), (.15, .3, .9))


def body(box=(.5, .1, .4, .85), *, points=POINTS, confidence=.8):
    x, y, w, h = box
    return FormationPersonDescriptor(box, (x + w / 2, y + h / 2), confidence, points)


class FocalPoseTests(unittest.TestCase):
    def test_focal_pose_available_even_with_other_visible_performers(self):
        focal = body()
        background = body((.2, .2, .2, .7))
        self.assertIs(face_associated_upper_body_person((.65, .18, .1, .18),
                                                       (background, focal)), focal)

    def test_face_association_does_not_depend_on_person_order(self):
        focal = body()
        background = body((.2, .2, .2, .7))
        for people in ((background, focal), (focal, background)):
            self.assertIs(face_associated_upper_body_person((.65, .18, .1, .18), people), focal)

    def test_missing_focal_pose_never_borrows_background_pose(self):
        self.assertIsNone(face_associated_upper_body_person((.65, .18, .1, .18),
                          (body(points=()), body((.2, .2, .2, .7)))))

    def test_ambiguous_body_stays_ambiguous_when_one_has_no_pose(self):
        self.assertIsNone(face_associated_upper_body_person((.65, .18, .1, .18),
                          (body(), body((.48, .08, .42, .87), points=()))))

    def test_face_in_lower_body_does_not_establish_head_association(self):
        self.assertIsNone(face_associated_upper_body_person((.65, .70, .1, .1), (body(),)))

    def test_tiny_overlap_does_not_establish_face_body_association(self):
        self.assertIsNone(face_associated_upper_body_person((.85, .18, .1, .18), (body(),)))

    def test_low_detection_confidence_does_not_supply_a_pose(self):
        self.assertIsNone(face_associated_upper_body_person((.65, .18, .1, .18),
                                                           (body(confidence=.49),)))

    def test_incomparable_points_do_not_supply_pose_evidence(self):
        points = tuple((x, y, .1) for x, y, _ in POINTS)
        self.assertIsNone(face_associated_upper_body_person((.65, .18, .1, .18),
                                                           (body(points=points),)))

    def test_missing_invalid_or_unassociated_face_stays_missing(self):
        for face in (None, (.1, .1, 0, .2), (-.1, .1, .1, .2), (.9, .1, .2, .2),
                     (float('nan'), .1, .1, .2), (.05, .05, .1, .1),
                     (True, .1, .1, .2), ('invalid', .1, .1, .2)):
            with self.subTest(face=face):
                self.assertIsNone(face_associated_upper_body_person(face, (body(),)))

    def test_real_sbs_foreground_box_reuses_eight_observed_joints(self):
        focal = body((.5509165554, .1179201577, .4159296647, .8552054044))
        background = body((.3999192798, .3082206689, .1375089015, .6917793311))
        self.assertIs(face_associated_upper_body_person(
            (.6669313431, .1859561496, .0931979179, .2026860555), (background, focal)), focal)

    def test_existing_upper_body_metric_keeps_gesture_difference(self):
        focal = body()
        changed = replace(focal, upper_body_keypoints_xyc=tuple(
            (x, -y if index in (2, 3, 4, 5) else y, confidence)
            for index, (x, y, confidence) in enumerate(POINTS)
        ))
        self.assertEqual(_upper_body_shape_mismatch(focal, focal), 0)
        self.assertGreater(_upper_body_shape_mismatch(focal, changed), .1)


if __name__ == '__main__':
    unittest.main()
