from dataclasses import replace
import unittest

from crosscut.group_joint_formation import (ObservedFormationPerson, DeclaredFormationView,
    compare_projected_formation)


def person(index, x, *, height=.4, reverse=False, confidence=.9):
    sign = -1 if reverse else 1
    points = ((x-.025, .4, confidence), (x+.025, .4, confidence),
        (x-.06, .4+sign*.05, confidence), (x+.06, .4+sign*.05, confidence),
        (x-.08, .4+sign*.12, confidence), (x+.08, .4+sign*.12, confidence),
        (x-.025, .55, confidence), (x+.025, .55, confidence))
    return ObservedFormationPerson(index, (x-.05, .2, .1, height), points, .9)


def view(*people, required=None):
    return DeclaredFormationView(tuple(people), tuple(p.observation_id for p in people) if required is None else required)


class GroupJointFormationTests(unittest.TestCase):
    def test_permutation_is_not_screen_order_zip(self):
        left = view(person(0, .2), person(1, .5), person(2, .8))
        right = view(person(20, .8), person(21, .2), person(22, .5))
        result = compare_projected_formation(left, right)
        self.assertTrue(result.observed)
        self.assertEqual(set(result.pairs), {(0, 21), (1, 22), (2, 20)})
        self.assertAlmostEqual(result.framing_cost, 0.)

    def test_extra_background_detection_cannot_shift_the_declared_cohort(self):
        left = view(person(0, .2), person(1, .5), person(2, .8))
        right = view(person(99, .05, height=.08), person(20, .2), person(21, .5), person(22, .8),
            required=(20, 21, 22))
        result = compare_projected_formation(left, right)
        self.assertTrue(result.observed)
        self.assertNotIn(99, {b for _, b in result.pairs})
        self.assertAlmostEqual(result.framing_cost, 0.)

    def test_occluded_person_is_unmatched_not_missing_zero_pose(self):
        left = view(person(0, .2), person(1, .4), person(2, .6), person(3, .8))
        right = view(person(10, .2), person(11, .4), person(12, .6),
            replace(person(13, .8), upper_body_xyc=()))
        result = compare_projected_formation(left, right)
        self.assertTrue(result.observed)
        self.assertEqual(result.unmatched_incoming, (13,))
        self.assertEqual(result.unmatched_outgoing, (3,))
        self.assertEqual(result.outgoing_coverage, .75)
        self.assertEqual(result.incoming_coverage, .75)

    def test_most_people_occluded_does_not_fabricate_complete_correspondence(self):
        left = view(*(person(i, .2+.2*i) for i in range(4)))
        right = view(*(person(i+10, .2+.2*i, confidence=.1 if i>0 else .9) for i in range(4)))
        result = compare_projected_formation(left, right)
        self.assertFalse(result.observed)
        self.assertIsNone(result.framing_cost)

    def test_identical_overlapping_people_are_ambiguous(self):
        result = compare_projected_formation(view(person(0, .5), person(1, .5)),
            view(person(2, .5), person(3, .5)))
        self.assertEqual(result.status, 'ambiguous_partial_correspondence')
        self.assertIsNone(result.pose_mismatch)
        self.assertIsNone(result.framing_cost)

    def test_equal_maximum_height_does_not_hide_wrong_group_distribution(self):
        left = view(person(0, .2), person(1, .5), person(2, .8))
        right = view(person(10, .2), person(11, .5, height=.2), person(12, .8, height=.2))
        self.assertEqual(max(p.box_xywh[3] for p in left.people), max(p.box_xywh[3] for p in right.people))
        result = compare_projected_formation(left, right)
        self.assertTrue(result.observed)
        self.assertGreater(result.height_distribution_mismatch, .3)
        self.assertGreater(result.framing_cost, .1)

    def test_equal_height_wrong_span_and_location_remain_wrong(self):
        left = view(person(0, .2), person(1, .5), person(2, .8))
        right = view(person(10, .4), person(11, .5), person(12, .6))
        result = compare_projected_formation(left, right)
        self.assertTrue(result.observed)
        self.assertGreater(result.horizontal_span_mismatch, .7)
        self.assertGreater(result.paired_subject_location_mismatch, .1)

    def test_opposite_observed_arm_pose_is_not_good_geometry_evidence(self):
        left = view(person(0, .2), person(1, .5), person(2, .8))
        right = view(person(10, .2, reverse=True), person(11, .5, reverse=True), person(12, .8, reverse=True))
        result = compare_projected_formation(left, right)
        self.assertFalse(result.observed)
        self.assertIsNone(result.framing_cost)

    def test_offscreen_points_are_not_clamped_into_visibility(self):
        p = replace(person(0, .2), box_xywh=(-.1, .2, .2, .4),
            upper_body_xyc=((-2., 3., .9),)*8)
        self.assertEqual(p.box_xywh[0], -.1)
        self.assertEqual(p.upper_body_xyc[0][0], -2.)
        self.assertFalse(compare_projected_formation(view(p, person(1, .5)),
            view(person(10, .2), person(11, .5))).observed)

    def test_invalid_cohort_or_duplicate_ids_cannot_be_admitted(self):
        with self.assertRaises(ValueError): view(person(0, .2), person(0, .8))
        with self.assertRaises(ValueError): view(person(0, .2), required=(1,))
        with self.assertRaises(ValueError): view(person(0, .2), required=(True,))

    def test_no_family_threshold_parameter_and_no_quality_or_motion_claim(self):
        result = compare_projected_formation(view(person(0, .2), person(1, .8)),
            view(person(10, .2), person(11, .8)))
        payload = result.to_dict()
        for key in ('named_member_identity_certified', 'motion_certified', 'visibility_certified',
                'quality_approved', 'raw_family_gate'):
            self.assertFalse(payload[key])

    def test_different_output_aspect_cannot_compare_different_coordinate_bases(self):
        a = view(person(0, .2), person(1, .8))
        with self.assertRaises(ValueError): compare_projected_formation(a, replace(a, output_size=(1080, 1920)))
