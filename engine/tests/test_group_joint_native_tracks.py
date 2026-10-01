from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.group_joint_native_observer import NativeGroupObservation
from crosscut.group_joint_native_tracks import track_native_group
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from engine.tests.test_group_joint_formation import person


class NativeGroupTrackTests(unittest.TestCase):
    def fixture(self, people, *, cuts=()):
        clock = VerifiedRenderFrameClock(Fraction(30), 100, 'a'*64)
        domain = GroupSourceDomain(tuple(Path(f'{n}.mp4') for n in range(6)),
            (0, 2, 5), (0.,)*3, (None,)*3, ((), cuts, ()), (clock,)*3)
        session = SimpleNamespace(domain=domain,
            budget=JointGroupWorkBudget(absolute_deadline=10., clock=lambda:0.),
            native_pixels=lambda source, native:np.full((50, 100, 3), native, np.uint8))
        observations = tuple(NativeGroupObservation(2, index, (100, 50), 'c'*64,
            (1, 2, 3, 4, 5), 'd'*64, (('synthetic_test_only', 'e'*64),), tuple(rows), (),
            'synthetic_test_observation_not_real_model_quality') for index, rows in enumerate(people))
        return session, observations

    def run_tracks(self, people, *, cuts=(), flow=lambda a,b,box:box):
        session, observations = self.fixture(people, cuts=cuts)
        with patch('crosscut.group_joint_native_tracks.observe_native_group', side_effect=observations), \
                patch('crosscut.group_joint_native_tracks._optical_flow_box', side_effect=flow):
            return track_native_group(session, request_source_index=2, native_frame_indices=tuple(range(len(people))))

    def test_detection_order_changes_without_changing_source_native_track_references(self):
        result = self.run_tracks(((person(0,.2), person(1,.8)),
            (person(8,.8), person(9,.2)), (person(20,.2), person(21,.8))))
        self.assertEqual(tuple(point.observation_id for point in result.tracks[0].points), (0,9,20))
        self.assertEqual(tuple(point.observation_id for point in result.tracks[1].points), (1,8,21))
        self.assertTrue(result.complete_required_native_paths)
        self.assertEqual(result.required_references, tuple(item.reference for item in result.tracks))
        self.assertFalse(result.to_dict()['cross_camera_correspondence_verified'])
        self.assertFalse(result.to_dict()['quality_approved'])

    def test_dropout_does_not_remove_an_obligation_or_claim_complete_visibility_after_recovery(self):
        result = self.run_tracks(((person(0,.2), person(1,.8)), (person(10,.2),),
            (person(20,.2), person(21,.8))))
        self.assertEqual(len(result.required_references), 2)
        self.assertEqual(tuple(p.observation_id for p in result.tracks[1].points), (1,None,21))
        self.assertFalse(result.complete_required_native_paths)

    def test_ambiguous_links_keep_all_candidates_and_do_not_choose_screen_order(self):
        result = self.run_tracks(((person(0,.5), person(1,.5)), (person(10,.5), person(11,.5))))
        for track in result.tracks:
            self.assertIsNone(track.points[1].observation_id)
            self.assertEqual(track.points[1].alternative_observation_ids, (10,11))
            self.assertEqual(track.points[1].basis, 'native_pixel_link_ambiguous')
        self.assertFalse(result.complete_required_native_paths)

    def test_actual_native_camera_cut_breaks_even_identical_pose_and_boxes(self):
        result = self.run_tracks(((person(0,.2), person(1,.8)), (person(10,.2), person(11,.8))), cuts=(1/30.,))
        self.assertTrue(all(track.points[1].basis == 'native_camera_cut_breaks_person_path' for track in result.tracks))
        self.assertFalse(result.complete_required_native_paths)

    def test_unknown_pixel_link_is_not_observed_zero_velocity(self):
        result = self.run_tracks(((person(0,.2), person(1,.8)), (person(10,.2), person(11,.8))),
            flow=lambda *a:None)
        self.assertFalse(result.complete_required_native_paths)
        self.assertTrue(all(track.points[1].observed_iou is None for track in result.tracks))

    def test_changed_source_file_or_clock_cannot_join_tracklet_rows(self):
        session, observations = self.fixture(((person(0,.2),),(person(1,.2),)))
        for extra in ({'source_file_identity':(2,3,4,5,6)}, {'source_clock_identity':'f'*64},
                {'request_source_index':5}, {'frame_size_wh':(200,100)}):
            rows=(observations[0],replace(observations[1],**extra))
            with patch('crosscut.group_joint_native_tracks.observe_native_group',side_effect=rows):
                with self.subTest(extra=extra),self.assertRaisesRegex(ValueError,'source/file/clock/raster'):
                    track_native_group(session,request_source_index=2,native_frame_indices=(0,1))
