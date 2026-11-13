from copy import deepcopy
import unittest

from crosscut.group_joint_instance_prompts import native_instance_prompt_hypotheses


class NativeInstancePromptTests(unittest.TestCase):
    def row(self):
        body = dict(observation_id=4, box_xywh=(.2, .1, .4, .9),
            raw_pose_xyc=[(.4, .4, .8)]*17,
            source_pixel_to_pose_input_affine=((1., 0., 0.), (0., 1., 0.)))
        other = dict(body, observation_id=5, box_xywh=(.4, .1, .4, .9), raw_pose_xyc=[(.7, .7, .8)]*17)
        return dict(contract='actual_source_native_owned_person_body_face_observation_rows_v3',
            native_pixel_sha256='a'*64, heads=[dict(observation_id=2,
                box_xywh=(.3, .2, .1, .2), candidate_body_observation_ids=(4, 5)),
                dict(observation_id=9, box_xywh=(.8, .1, .2, .2), candidate_body_observation_ids=())],
            people=[body, other])

    def test_ambiguous_body_owners_are_two_independent_prompts_not_a_union(self):
        row = self.row(); before = deepcopy(row)
        outputs = native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2)
        self.assertEqual(tuple(r.body_observation_id for r in outputs), (4, 5))
        for actual, expected in zip(outputs[0].box_xyxy, (.2, .1, .6, 1.)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(outputs[1].box_xyxy, (.4, .1, .8, 1.)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(outputs[0].point_labels, (1,)*9+(0,))
        self.assertEqual(outputs[0].point_coords_xy[-1], (.9, .2))
        self.assertNotIn((.7, .7), outputs[0].point_coords_xy)
        self.assertNotIn((.4, .4), outputs[1].point_coords_xy)
        self.assertEqual(row, before)
        self.assertFalse(outputs[0].to_dict()['prompt_is_surface_certificate'])

    def test_missing_explicit_seed_does_not_choose_the_larger_background_face(self):
        self.assertEqual(native_instance_prompt_hypotheses(raw_observation=self.row(), face_observation_id=0), ())

    def test_off_frame_joint_is_not_clamped_or_used_as_visible_positive_support(self):
        row = self.row(); row['people'][0]['raw_pose_xyc'][7] = (1.2, .3, .9)
        output = native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2)[0]
        self.assertNotIn(7, output.used_raw_joint_indices)
        self.assertIn((7, 'raw_joint_outside_native_frame'), output.unused_raw_joints)
        self.assertEqual(row['people'][0]['raw_pose_xyc'][7], (1.2, .3, .9))

    def test_low_confidence_joint_stays_unknown_and_floor_cannot_be_loosened(self):
        row = self.row(); row['people'][0]['raw_pose_xyc'][10] = (.4, .4, .549)
        output = native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2)[0]
        self.assertNotIn(10, output.used_raw_joint_indices)
        with self.assertRaises(ValueError):
            native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2, minimum_joint_confidence=.54)

    def test_right_bottom_edge_is_outside_the_native_raster_not_clamped_inward(self):
        for x, y in ((1., .3), (.3, 1.)):
            row = self.row(); row['people'][0]['raw_pose_xyc'][7] = (x, y, .9)
            output = native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2)[0]
            self.assertNotIn(7, output.used_raw_joint_indices)
            self.assertIn((7, 'raw_joint_outside_native_frame'), output.unused_raw_joints)
            self.assertEqual(row['people'][0]['raw_pose_xyc'][7], (x, y, .9))

    def test_invalid_raw_confidence_or_negative_inventory_id_is_not_a_prompt(self):
        for confidence in (True, float('nan'), float('inf')):
            row = self.row(); row['people'][0]['raw_pose_xyc'][7] = (.4, .4, confidence)
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2)
        row = self.row(); row['heads'][1]['observation_id'] = -1
        with self.assertRaises(ValueError):
            native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2)

    def test_simcc_raw_peak_above_one_is_preserved_and_negative_peak_is_unused(self):
        row = self.row(); row['people'][0]['raw_pose_xyc'][7] = (.4,.4,1.0234019756317139)
        original = deepcopy(row)
        output = native_instance_prompt_hypotheses(raw_observation=row,face_observation_id=2)[0]
        self.assertIn(7,output.used_raw_joint_indices)
        self.assertEqual(output.to_dict()['raw_score_basis'],'rtmpose_simcc_raw_peak_not_probability_v1')
        self.assertFalse(output.to_dict()['raw_scores_clipped'])
        self.assertEqual(row,original)
        row['people'][0]['raw_pose_xyc'][7] = (.4,.4,-.1)
        output = native_instance_prompt_hypotheses(raw_observation=row,face_observation_id=2)[0]
        self.assertNotIn(7,output.used_raw_joint_indices)
        self.assertIn((7,'raw_joint_below_unchanged_prompt_floor'),output.unused_raw_joints)

    def test_missing_raw17_or_affine_never_invents_a_body_prompt(self):
        for field, value in (('raw_pose_xyc', ()), ('source_pixel_to_pose_input_affine', ()),
                ('raw_pose_xyc', [(float('nan'), .4, .8)]*17)):
            row = self.row(); row['people'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2)

    def test_raw_coordinate_change_rebinds_hypothesis_and_policy_is_explicit(self):
        row = self.row()
        first = native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2)[0]
        row['people'][0]['raw_pose_xyc'][6] = (.45, .45, .8)
        second = native_instance_prompt_hypotheses(raw_observation=row, face_observation_id=2)[0]
        self.assertNotEqual(first.raw_observation_sha256, second.raw_observation_sha256)
        self.assertEqual(first.policy_sha256, second.policy_sha256)


if __name__ == '__main__':
    unittest.main()
