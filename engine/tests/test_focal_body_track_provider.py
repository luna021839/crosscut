from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from crosscut.body_pose_2d import BodyPose2DEvidence
from crosscut.direct_member_target_face import FocalFaceCandidate
from crosscut.focal_body_track import FocalBodyFrame, FocalBodyObservation
from crosscut.focal_body_track_provider import (SourceLocalFocalBodyProvider,
    select_direct_focal_observation)
from crosscut.pose import FormationPersonDescriptor


FACE = FocalFaceCandidate((.30, .20, .10, .10), (1., 0.))
PERSON = FormationPersonDescriptor((.20, .10, .40, .80), (.40, .50), .9, ())
POSE = ((.30, .35, .9), (.50, .35, .9), (.25, .45, .9), (.55, .45, .9),
        (.20, .50, .9), (.60, .50, .9), (.32, .65, .9), (.48, .65, .9))


def observation(*, score=.585, consistent=True, frozen=True, face=FACE):
    return FocalBodyObservation(0, 1., PERSON, (1., 0.), score, face, frozen,
                               POSE, consistent)


class DirectFocalBodyEvidenceTests(unittest.TestCase):
    def test_exact_face_pose_proof_is_not_vetoed_by_clothing(self):
        target = observation()
        competitor = replace(observation(score=.95), face=None, frozen_face_match=False,
                             face_pose_consistent=None)
        selected, reason = select_direct_focal_observation(FACE, (FACE,),
            FocalBodyFrame(0, 1., (target, competitor)))
        self.assertIs(selected, target)
        self.assertEqual(reason, 'frozen_face_with_exact_pose_agreement')

    def test_unverified_pose_or_face_does_not_bypass_body_rule(self):
        for value in (observation(consistent=None), observation(consistent=False),
                      observation(frozen=False)):
            selected, _ = select_direct_focal_observation(FACE, (FACE,),
                FocalBodyFrame(0, 1., (value,)))
            self.assertIsNone(selected)

    def test_face_pose_flag_requires_observed_face_and_joints(self):
        for values in (dict(face=None, frozen_face_match=False), dict(pose_screen_xyc=()),
                       dict(face_pose_consistent=1)):
            with self.assertRaises(ValueError): replace(observation(), **values)

    def test_ambiguous_body_binding_is_not_selected(self):
        target = observation()
        selected, _ = select_direct_focal_observation(FACE, (FACE,),
            FocalBodyFrame(0, 1., (target, replace(target, reference_similarity=.9))))
        self.assertIsNone(selected)

    def test_absent_face_does_not_gain_body_authority(self):
        selected, _ = select_direct_focal_observation(None, (),
            FocalBodyFrame(0, 1., (observation(),)))
        self.assertIsNone(selected)

    def test_existing_appearance_rule_and_solo_face_remain(self):
        target = observation(score=.8, consistent=None)
        self.assertIs(select_direct_focal_observation(FACE, (FACE,),
            FocalBodyFrame(0, 1., (target,)))[0], target)
        self.assertEqual(select_direct_focal_observation(FACE, (FACE,),
            FocalBodyFrame(0, 1., ()))[1], 'frozen_solo_face_body_unobserved')


class FocalBodyPixelProviderTests(unittest.TestCase):
    def provider(self, *, score=.585, references=3, consistency=True, pose_reason='observed'):
        pixels = np.zeros((100, 100, 3), np.uint8)
        provider = SourceLocalFocalBodyProvider.__new__(SourceLocalFocalBodyProvider)
        provider.face_provider = SimpleNamespace(
            _samplers=(SimpleNamespace(frame_at=Mock(return_value=pixels)),),
            _identity_key=lambda s, sampler, t: (s, round(t*30)),
            _ensure_focal_consensus=Mock(return_value=SimpleNamespace(select=lambda _: FACE)),
            _focal_candidates=Mock(return_value=(FACE,)))
        # Cosine similarity against (1,0) is exactly the specified score.
        features = (score, (1-score**2)**.5)
        provider._people_and_features = Mock(return_value=((PERSON,), (features,)))
        provider._ensure_references = Mock(return_value=((1., 0.),)*references)
        provider.pose_analyzer = SimpleNamespace(describe=Mock(return_value=BodyPose2DEvidence(
            pose_reason, PERSON.bbox_xywh, (100, 100), POSE if pose_reason == 'observed' else (),
            consistency if pose_reason == 'observed' else None)))
        provider._frame_cache = OrderedDict()
        provider.frame_inference_count = provider.frame_cache_hit_count = 0
        return provider

    def test_frozen_face_runs_pose_below_appearance_threshold(self):
        provider = self.provider()
        frame = provider.frame_observations(0, 1.)
        provider.pose_analyzer.describe.assert_called_once()
        self.assertAlmostEqual(frame.observations[0].reference_similarity, .585)
        self.assertTrue(frame.observations[0].face_pose_consistent)
        self.assertIsNotNone(select_direct_focal_observation(FACE, (FACE,), frame)[0])

    def test_face_pose_remains_available_without_three_body_seeds(self):
        for count in (0, 1, 2):
            provider = self.provider(references=count)
            frame = provider.frame_observations(0, 1.)
            self.assertEqual(frame.observations[0].reference_similarity, -1.)
            self.assertIsNotNone(select_direct_focal_observation(FACE, (FACE,), frame)[0])

    def test_mismatched_pose_head_erases_face_binding(self):
        provider = self.provider(consistency=False)
        frame = provider.frame_observations(0, 1.)
        self.assertIsNone(frame.observations[0].face)
        self.assertFalse(frame.observations[0].frozen_face_match)
        self.assertIsNone(select_direct_focal_observation(FACE, (FACE,), frame)[0])

    def test_unavailable_model_does_not_fabricate_pose_proof(self):
        provider = self.provider(pose_reason='pose_model_unavailable')
        frame = provider.frame_observations(0, 1.)
        self.assertIsNone(frame.observations[0].face_pose_consistent)
        self.assertIsNone(select_direct_focal_observation(FACE, (FACE,), frame)[0])

    def test_native_frame_cache_preserves_proof_and_caller_clock(self):
        provider = self.provider()
        first = provider.frame_observations(0, 1.)
        cached = provider.frame_observations(0, 1.001)
        provider.pose_analyzer.describe.assert_called_once()
        self.assertTrue(cached.observations[0].face_pose_consistent)
        self.assertEqual(cached.observations[0].source_seconds, 1.001)
        self.assertEqual(cached.observations[0].pose_screen_xyc, first.observations[0].pose_screen_xyc)


if __name__ == '__main__': unittest.main()
