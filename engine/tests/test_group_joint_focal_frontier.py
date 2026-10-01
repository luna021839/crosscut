from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from crosscut.group_joint_focal_frontier import (OwnedFocalAppearance,
    NativeFaceAppearance, observe_native_face_appearances,
    resolve_inherited_focal_reference,
    match_focal_trajectory,
    match_focal_appearance_anchors, observe_owned_focal_appearances, _dominant_owned_face, _same_source_links)
from crosscut.group_joint_person_ownership import bind_person_ownership, ownership_policy_fingerprint
from engine.tests import test_group_joint_person_ownership as ownership_fixtures
from engine.tests import test_group_joint_native_tracks as track_fixtures


def appearance(person=0, face=7, axis=0):
    return OwnedFocalAppearance(person, face, tuple(1. if i == axis else 0. for i in range(128)))


class FocalFrontierTests(unittest.TestCase):
    def row(self):
        fixture = ownership_fixtures.PersonOwnershipTests()
        head = replace(fixture.head(), candidate_body_observation_ids=(0,))
        people, _ = bind_person_ownership((fixture.closeup(),), (head,), (1000, 800))
        _, rows = track_fixtures.NativeGroupTrackTests().fixture(((),))
        return replace(rows[0], frame_size_wh=(1000, 800), people=people, heads=(head,),
            person_ownership_policy_sha256=ownership_policy_fingerprint())

    def test_distinct_three_anchor_appearance_is_shortlist_not_identity(self):
        refs = (appearance(),) * 3
        values, reason, similarities, margins = match_focal_appearance_anchors(refs, ((appearance(4),),) * 3)
        self.assertEqual(tuple(row.person_observation_id for row in values), (4, 4, 4))
        self.assertEqual(reason, 'appearance_shortlist_only')
        self.assertEqual(similarities, (1., 1., 1.))
        self.assertEqual(margins, (None, None, None), 'single candidate has no fabricated confidence margin')

    def test_missing_or_competing_same_face_remains_unavailable(self):
        refs = (appearance(),) * 3
        for values in (((appearance(),), (), (appearance(),)),
                ((appearance(), appearance(1)),) * 3, ((appearance(axis=1),),) * 3):
            self.assertIsNone(match_focal_appearance_anchors(refs, values)[0])
        with self.assertRaises(ValueError):
            match_focal_appearance_anchors(refs[:2], ((appearance(),),) * 2)

    def test_owned_cropped_hips_can_supply_actual_face_embedding_without_claiming_torso(self):
        row = self.row()
        session = SimpleNamespace(observe_native=Mock(return_value=appearance().embedding))
        values = observe_owned_focal_appearances(session, row)
        self.assertEqual(values, (appearance(),))
        self.assertEqual(row.people[0].body_ownership_state, 'unknown')
        self.assertEqual(dict(row.people[0].part_states)['head_extent'], 'unknown')
        self.assertEqual(session.observe_native.call_args.kwargs['model'], 'sface_owned_face')
        self.assertEqual(session.observe_native.call_args.args, (row.request_source_index, row.native_frame_index))

    def test_legacy_policy_and_forged_owned_parts_cannot_supply_embeddings(self):
        row = self.row()
        session = SimpleNamespace(observe_native=Mock(return_value=appearance().embedding))
        for wrong in (replace(row, person_ownership_policy_sha256='a'*64),
                replace(row, people=(replace(row.people[0], body_ownership_state='observed'),)),
                replace(row, heads=(replace(row.heads[0], candidate_body_observation_ids=(1,)),))):
            with self.assertRaises(ValueError):
                observe_owned_focal_appearances(session, wrong)
        session.observe_native.assert_not_called()

    def test_reference_is_global_face_dominance_not_convenient_body_or_order(self):
        row = self.row()
        self.assertEqual(_dominant_owned_face(row, (appearance(),)), appearance())
        head = replace(row.heads[0], observation_id=8, box_xywh=(.65, .27, .065, .06))
        ambiguous = replace(row, heads=(*row.heads, head))
        self.assertIsNone(_dominant_owned_face(ambiguous, (appearance(),)))
        self.assertIsNone(_dominant_owned_face(row, ()))

    def test_face_roi_sensing_preserves_all_ambiguous_body_hypotheses_without_selecting_one(self):
        row = self.row()
        other = replace(row.people[0], observation_id=1)
        head = replace(row.heads[0], candidate_body_observation_ids=(0, 1))
        people, _ = bind_person_ownership((row.people[0], other), (head,), row.frame_size_wh)
        row = replace(row, people=people, heads=(head,))
        self.assertEqual(tuple(p.upper_body_ownership_state for p in row.people), ('ambiguous', 'ambiguous'))
        session = SimpleNamespace(observe_native=Mock(return_value=appearance().embedding))
        self.assertEqual(observe_owned_focal_appearances(session, row), ())
        values = observe_native_face_appearances(session, row)
        self.assertEqual(values, (NativeFaceAppearance(7, appearance().embedding, (0, 1)),))
        self.assertIsNone(values[0].person_observation_id)
        self.assertEqual(values[0].candidate_body_observation_ids, (0, 1))

    def test_landmark_pixel_thresholds_use_detector_raster_not_arbitrary_native_resolution(self):
        row = self.row()
        rows = tuple(replace(row, native_frame_index=n, frame_size_wh=(1920, 1080)) for n in (0, 3, 6))
        face = NativeFaceAppearance(7, appearance().embedding, (0, 1))
        session = SimpleNamespace(domain=SimpleNamespace(eligible_index=lambda _: 0,
            render_frame_clocks=(SimpleNamespace(frame_rate=30.),), cuts_for_request=lambda _: ()),
            native_pixels=lambda *_: np.zeros((1080, 1920, 3), np.uint8))
        with patch('crosscut.group_joint_focal_frontier.focal_landmarks_track', return_value=True) as flow:
            self.assertTrue(_same_source_links(session, rows, (face,) * 3))
        self.assertEqual(flow.call_count, 2)
        self.assertEqual(flow.call_args.args[0].shape, (540, 960, 3))
        self.assertEqual(flow.call_args.args[2], row.heads[0].landmarks_xy)
        session.domain.cuts_for_request = lambda _: (.05,)
        with patch('crosscut.group_joint_focal_frontier.focal_landmarks_track') as flow:
            self.assertFalse(_same_source_links(session, rows, (face,) * 3))
            flow.assert_not_called()

    def test_reference_survives_background_face_growth_without_switching_or_lowering_area_margin(self):
        row = self.row()
        other = replace(row.heads[0], observation_id=8, box_xywh=(.65, .27, .061, .06))
        middle = replace(row, heads=(*row.heads, other))
        self.assertIsNone(_dominant_owned_face(middle, (appearance(),)))
        values = ((appearance(),), (appearance(), appearance(1, 8, axis=1)), (appearance(),))
        with patch('crosscut.group_joint_focal_frontier._same_source_links', return_value=True):
            self.assertEqual(resolve_inherited_focal_reference(None, (row, middle, row), values), (appearance(),)*3)
            self.assertIsNone(resolve_inherited_focal_reference(None, (middle, row, row), values),
                'no independent initial motive/seed may be fabricated')
        with patch('crosscut.group_joint_focal_frontier._same_source_links', return_value=False):
            self.assertIsNone(resolve_inherited_focal_reference(None, (row, middle, row), values))

    def test_native_trajectory_prevents_per_frame_appearance_switching_people(self):
        good = appearance(1, 8)
        alternative = appearance(2, 9)
        alternatives = ((alternative, good), (good,), (good,))
        path, info = match_focal_trajectory((appearance(),)*3, alternatives,
            path_available=lambda rows: len({row.face_observation_id for row in rows}) == 1)
        self.assertEqual(path, (good,)*3)
        self.assertEqual(info['hypotheses'][0]['face_observation_ids'], [8, 8, 8])
        self.assertFalse(info['identity_certified'])
        path, info = match_focal_trajectory((appearance(),)*3, alternatives, path_available=lambda _: True)
        self.assertEqual(len(info['hypotheses']), 2, 'retain all actual competing paths, no silent pruning')
        self.assertEqual(info['competing_trajectory_margin'], 0.)

    def test_hypothesis_overflow_remains_unknown_without_truncating_to_convenient_face(self):
        available = Mock(return_value=True)
        path, info = match_focal_trajectory((appearance(),)*3,
            ((appearance(), appearance(1)),)*3, path_available=available, maximum_hypotheses=7)
        self.assertIsNone(path)
        self.assertEqual(info['hypothesis_count'], 8)
        available.assert_not_called()
