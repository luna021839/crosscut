"""Synthetic sensor-policy contracts, not actual people or editorial QA."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from crosscut.group_joint_native_observer import NativeGroupHead
from crosscut.group_joint_source_camera import JointGroupWorkBudget, JointGroupBudgetExhausted
from crosscut.group_native_head_orientation import OrientedNativeHeadAlternative, native_head_analysis_spec
from crosscut.group_native_head_window import ANALYSIS_VARIANTS, observe_native_head_window_policy
from tests import test_group_joint_temporal_pose as pose_fixtures


class NativeHeadWindowTests(unittest.TestCase):
    def head(self, *, center=.5, scale=1., id=0):
        points = ((-.025,-.025),(.025,-.025),(0.,0.),(-.025,.025),(.025,.025))
        return NativeGroupHead(id, (center-.075*scale,.35-.08*scale,.15*scale,.16*scale),
            tuple((center+x*scale,.35+y*scale) for x,y in points), .95)

    def session(self, observations, *, source=0, limit=100, cuts=()):
        domain = pose_fixtures.NativeTemporalPoseTests().domain()
        if cuts:
            domain = replace(domain, scene_cut_times=(cuts, *domain.scene_cut_times[1:]))
        budget = JointGroupWorkBudget(absolute_deadline=10., clock=lambda:0., maximum_model_calls=limit)
        pixels = np.zeros((500,1000,3),np.uint8)
        models = {native_head_analysis_spec(v)[0]: (None, None, 'b'*64, None) for v in ANALYSIS_VARIANTS}
        session = SimpleNamespace(domain=domain, budget=budget, models=models,
            model_adapters={name:'a'*64 for name in models},
            native_pixels=Mock(return_value=pixels), _assert_model_assets=Mock(),
            _assert_source_identity=Mock(return_value=(1,2,3,4,5)), retain_native_pixels=Mock())
        def observe(request, native, *, model):
            budget.spend('model_calls', model=model)
            angle = next(v for v in ANALYSIS_VARIANTS if native_head_analysis_spec(v)[0] == model)
            values = observations.get((angle,native), observations.get(angle, ()))
            if values is None or angle is None:
                return values
            return tuple(OrientedNativeHeadAlternative(model,float(angle),(1000,500),
                tuple(tuple(float(v) for v in line) for line in
                    cv2.getRotationMatrix2D((499.5,249.5),float(angle),1.)),
                h, ()) for h in values)
        session.observe_native=Mock(side_effect=observe)
        return session

    def policy(self, session, *, source=0):
        with patch('crosscut.group_native_head_window.focal_landmarks_track', return_value=True):
            return observe_native_head_window_policy(session, request_source_index=source,
                native_frame_indices=(10,11,12))

    def test_background_upright_head_cannot_hide_larger_tilted_foreground_window(self):
        session = self.session({None:(self.head(center=.25,scale=.5),), 60.:(self.head(scale=2.),)})
        result = self.policy(session)
        self.assertTrue(result.selected)
        self.assertEqual(result.selected_analysis_roll_degrees,60.)
        self.assertEqual(session.budget.counts['model_calls'],15)
        self.assertEqual(len(result.variants),5)
        self.assertTrue(all(v['heads_by_native_frame'] for v in result.variants))
        for name in ('variants_merged_as_people','body_ownership_verified','temporal_phase_certified',
                     'full_head_extent_verified','candidate_admitted','quality_approved'):
            self.assertFalse(result.to_dict()[name])

    def test_same_head_across_variants_prefers_upright_without_box_envelope_inflation(self):
        face = self.head()
        inflated = replace(face, box_xywh=(.1,.05,.8,.6))
        # Identical measured landmarks prove duplicate sensor hypotheses;
        # the wider rotated rectangle must not replace upright geometry.
        session = self.session({None:(face,), 60.:(inflated,)})
        result = self.policy(session)
        self.assertTrue(result.selected)
        self.assertIsNone(result.selected_analysis_roll_degrees)
        self.assertEqual(len(result.distinct_trajectory_classes),1)
        self.assertEqual(result.variants[0]['heads_by_native_frame'][0][0]['box_xywh'],face.box_xywh)

    def test_equally_large_distinct_trajectories_are_not_a_convenient_subject_choice(self):
        session = self.session({None:(self.head(center=.25),),60.:(self.head(center=.75),)})
        result = self.policy(session)
        self.assertFalse(result.selected)
        self.assertEqual(result.reason,'competing_native_foreground_trajectories_remain_ambiguous')
        self.assertEqual(len(result.distinct_trajectory_classes),2)

    def test_native_landmark_pixel_failure_stays_unobserved_even_with_similar_boxes(self):
        session = self.session({None:(self.head(),)})
        with patch('crosscut.group_native_head_window.focal_landmarks_track',return_value=False):
            result=observe_native_head_window_policy(session,request_source_index=0,native_frame_indices=(10,11,12))
        self.assertFalse(result.selected)
        self.assertEqual(result.variants[0]['adjacent_pixel_links'],[False,False])

    def test_existing_measured_box_flow_is_kept_distinct_from_landmark_motion(self):
        face=self.head()
        session=self.session({None:(face,)})
        with patch('crosscut.group_native_head_window.focal_landmarks_track',return_value=False), \
                patch('crosscut.group_native_head_window._optical_flow_box',return_value=face.box_xywh):
            result=observe_native_head_window_policy(session,request_source_index=0,native_frame_indices=(10,11,12))
        self.assertFalse(result.selected)
        self.assertEqual(result.variants[0]['adjacent_link_basis'],
            ['native_face_box_path_not_whole_head_or_part_ownership']*2)
        self.assertTrue(result.variants[0]['stable_foreground_available'])
        self.assertFalse(result.variants[0]['landmark_geometry_window_available'])
        self.assertTrue(result.variants[0]['box_only_geometry_is_not_eye_area_ranking_evidence'])
        self.assertFalse(result.to_dict()['temporal_phase_certified'])
        self.assertEqual(len(result.source_code_sha256),6)
        session=self.session({None:(face,)})
        with patch('crosscut.group_native_head_window.focal_landmarks_track',return_value=False), \
                patch('crosscut.group_native_head_window._optical_flow_box',return_value=(.8,.8,.1,.1)):
            result=observe_native_head_window_policy(session,request_source_index=0,native_frame_indices=(10,11,12))
        self.assertFalse(result.selected)

    def test_box_only_regressed_eyes_do_not_erase_an_observed_landmark_window(self):
        measured=self.head()
        regressed=replace(measured,box_xywh=(.2,.15,.6,.6),
            landmarks_xy=tuple((x*2.-.5,y) for x,y in measured.landmarks_xy))
        session=self.session({None:(regressed,),30.:(measured,)})
        with patch('crosscut.group_native_head_window.focal_landmarks_track',
                    side_effect=lambda _a,_b,before,_after:before==measured.landmarks_xy), \
                patch('crosscut.group_native_head_window._optical_flow_box',return_value=regressed.box_xywh):
            result=observe_native_head_window_policy(session,request_source_index=0,native_frame_indices=(10,11,12))
        self.assertTrue(result.selected)
        self.assertEqual(result.selected_analysis_roll_degrees,30.)
        self.assertTrue(result.variants[0]['box_only_geometry_is_not_eye_area_ranking_evidence'])
        self.assertEqual(len(result.variants[0]['heads_by_native_frame']),3,
            'box-path geometry was preserved, not fabricated as missing or erased')

    def test_a_favorable_different_roll_at_each_anchor_cannot_form_one_window(self):
        face=self.head()
        session=self.session({(None,10):(face,), (30.,11):(face,), (60.,12):(face,)})
        self.assertFalse(self.policy(session).selected)

    def test_missing_detector_variant_and_raw_raster_remain_missing(self):
        session=self.session({None:None})
        result=self.policy(session)
        self.assertFalse(result.selected)
        self.assertEqual(result.variants[0]['unavailable_native_frame_indices'],[10,11,12])
        session=self.session({None:(self.head(),)})
        session.native_pixels.return_value=None
        result=self.policy(session)
        self.assertFalse(result.selected)
        self.assertEqual(session.budget.counts['model_calls'],0)
        self.assertEqual(result.pixel_sha256,(None,None,None))

    def test_actual_native_cut_and_repeated_native_aliases_do_not_make_motion_evidence(self):
        session=self.session({None:(self.head(),)},cuts=(11/30.,))
        result=self.policy(session)
        self.assertFalse(result.selected)
        self.assertEqual(result.reason,'window_crosses_actual_native_camera_cut')
        self.assertEqual(session.budget.counts['model_calls'],0)
        session=self.session({None:(self.head(),)})
        result=observe_native_head_window_policy(session,request_source_index=0,native_frame_indices=(10,10,11))
        self.assertFalse(result.selected)
        self.assertEqual(result.reason,'unique_native_window_samples_unavailable')

    def test_all_calls_share_named_budget_and_no_source_specific_angles_exist(self):
        session=self.session({60.:(self.head(),)},limit=4)
        with self.assertRaises(JointGroupBudgetExhausted):
            self.policy(session)
        self.assertEqual(session.budget.counts['model_calls'],4)
        session=self.session({60.:(self.head(),)})
        result=self.policy(session,source=1)
        self.assertTrue(result.selected)
        self.assertEqual(result.request_source_index,1)
        self.assertEqual(result.selected_analysis_roll_degrees,60.)

    def test_unregistered_variant_and_source_change_reject_whole_policy(self):
        session=self.session({None:(self.head(),)})
        session.models.pop(native_head_analysis_spec(60.)[0])
        with self.assertRaisesRegex(ValueError,'registered before observation'):
            self.policy(session)
        self.assertEqual(session.budget.counts['model_calls'],0)
        session=self.session({None:(self.head(),)})
        session._assert_source_identity.side_effect=[(1,2,3,4,5),(1,2,3,4,6)]
        with self.assertRaisesRegex(ValueError,'source changed'):
            self.policy(session)

    def test_invalid_inputs_fail_before_camera_or_decoder_work(self):
        for indices in ((10,9,12),(True,11,12),(10,11),[-1,11,12]):
            session=self.session({})
            with self.subTest(indices=indices),self.assertRaises(ValueError):
                observe_native_head_window_policy(session,request_source_index=0,native_frame_indices=indices)
            session.native_pixels.assert_not_called()
        for variants in ((),(True,), (None,None),(0.,), (90.,)):
            session=self.session({})
            with self.subTest(variants=variants),self.assertRaises(ValueError):
                observe_native_head_window_policy(session,request_source_index=0,
                    native_frame_indices=(10,11,12),analysis_variants=variants)


if __name__=='__main__':
    unittest.main()
