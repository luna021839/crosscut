"""Synthetic pose/time/schema regressions; no real performer or quality proof."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from crosscut.group_joint_temporal_pose import (FrozenPoseMotionDutiesV1,
    PoseJointObservationState, TemporalPoseUnavailable, native_pose_windows_from_rows,
    render_pose_windows_from_rows, compare_pose_phase_windows, compare_pose_boundary_windows,
    independent_native_pose_hypotheses, supported_pose_spans, native_pose_association_graph)
from crosscut.group_joint_native_observer import NativeGroupObservation, NativeGroupHead
from crosscut.group_joint_person_ownership import (OwnedPersonObservation,
    bind_person_ownership, ownership_policy_fingerprint)
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.group_joint_source_camera import JointGroupWorkBudget, JointGroupBudgetExhausted
from crosscut.group_joint_focal_frontier import (NativeFaceAppearance,
    native_pose_hypotheses_for_face_paths, observe_synchronized_focal_frontier)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from tests import test_group_joint_state_factory as state_fixtures


class NativeTemporalPoseTests(unittest.TestCase):
    duties = FrozenPoseMotionDutiesV1('synthetic:predeclared-head-and-arms-sensor-not-semantic-owner',
        (0, 5, 6, 7, 8, 9, 10), .3)

    def domain(self, fps=(Fraction(30), Fraction(30)), cuts=((), ())):
        return GroupSourceDomain((Path('synthetic-unopened-0.mp4'), Path('synthetic-unopened-1.mp4')),
            (0, 1), (0., 0.), (None, None), cuts,
            tuple(VerifiedRenderFrameClock(rate, 10000, 'a'*64) for rate in fps))

    def person(self, shift=0., index=0, *, lost_joint=None, off_joint=None):
        raw = [(.5, .33, .9)]*17
        for n, point in {5:(.45,.4,.9), 6:(.55,.4,.9), 7:(.4+shift,.52,.9),
                8:(.6,.52,.9), 9:(.35+shift,.5,.9), 10:(.65,.5,.9),
                11:(.47,.75,.9), 12:(.53,.75,.9)}.items():
            raw[n] = point
        if lost_joint is not None:
            raw[lost_joint] = (*raw[lost_joint][:2], .1)
        if off_joint is not None:
            raw[off_joint] = (1.1, .5, .9)
        return OwnedPersonObservation(index, (.25,.2,.5,.65), tuple(raw[n] for n in (5,6,7,8,9,10,11,12)), .95,
            pose_crop_xywh=(0.,0.,1.,1.), source_pixel_to_pose_input_affine=((1.,0.,0.),(0.,1.,0.)),
            raw_pose_xyc=tuple(raw), pose_reason='observed', face_anchor_consistent=True)

    def row(self, domain, source, native, shift=0., *, duplicate=False, lost_joint=None, off_joint=None):
        p = self.person(shift, lost_joint=lost_joint, off_joint=off_joint)
        head = NativeGroupHead(0, (.47,.29,.06,.08), ((.5,.33),)*5, .95)
        people, owners = bind_person_ownership((p, replace(p, observation_id=1)) if duplicate else (p,),
            (head,), (1000,800))
        head = replace(head, candidate_body_observation_ids=tuple(owners[0]))
        clock = domain.clocks_by_request[source]
        digest = hashlib.sha256(json.dumps(clock.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        names = ('nanodet_people', 'rtmpose_body7', 'yunet_head_faces')
        return NativeGroupObservation(source, native, (1000,800), digest, (1,source,3,4,5), 'd'*64,
            tuple((name,'e'*64) for name in names), people, (head,),
            'synthetic-raw-pose-test-not-real-model-evidence', tuple((name,'f'*64) for name in names),
            ownership_policy_fingerprint())

    def window(self, domain=None, source=0, frames=(0,3,6), shifts=(0.,.01,.02), **kw):
        domain = domain or self.domain()
        rows = tuple(self.row(domain, source, native, shift, **kw) for native, shift in zip(frames, shifts))
        choices = ((0,1),)*len(rows) if kw.get('duplicate') else ((0,),)*len(rows)
        return native_pose_windows_from_rows(domain=domain, rows=rows, choices=choices, duties=self.duties)

    def row_people(self, domain, source, native, ids):
        row = self.row(domain, source, native)
        people, owners = bind_person_ownership(tuple(self.person(index=n) for n in ids),
            row.heads, row.frame_size_wh)
        heads = tuple(replace(h, candidate_body_observation_ids=tuple(owners[h.observation_id]))
            for h in row.heads)
        return replace(row, people=people, heads=heads)

    def joint(self, report, n=9):
        return next(j for j in report['joints'] if j['joint_index'] == n)

    def test_same_pose_at_endpoint_but_reversed_movement_is_not_a_zero_motion_match(self):
        domain = self.domain()
        a = self.window(domain, shifts=(-.02,-.01,0.))[0]
        b = self.window(domain, source=1, shifts=(.02,.01,0.))[0]
        report = compare_pose_phase_windows(a,b)
        wrist = self.joint(report)
        self.assertAlmostEqual(wrist['multi_anchor_residual'][-1], 0.)
        self.assertTrue(all(c < -.99 for c in wrist['velocity_direction_cosines']))
        self.assertGreater(wrist['maximum_velocity_difference_per_second'], 1.99)
        self.assertFalse(report['temporal_phase_certified'])
        self.assertFalse(report['quality_approved'])

    def test_native_30_and_60_fps_use_real_seconds_not_array_steps(self):
        domain = self.domain((Fraction(30), Fraction(60)))
        a = self.window(domain)[0]
        b = self.window(domain, source=1, frames=(0,6,12))[0]
        wrist = self.joint(compare_pose_phase_windows(a,b))
        self.assertLess(wrist['maximum_velocity_difference_per_second'], 1e-12)
        self.assertTrue(all(abs(c-1.) < 1e-12 for c in wrist['velocity_direction_cosines']))

    def test_2997_native_grid_has_exact_fractional_time_not_assumed_30(self):
        domain = self.domain((Fraction(30000,1001),)*2)
        window = self.window(domain)[0]
        self.assertAlmostEqual(window.anchors[1].measurement_seconds, .1001)
        self.assertNotEqual(window.anchors[1].measurement_seconds, .1)

    def test_static_joint_has_unknown_direction_not_perfect_motion_score(self):
        domain = self.domain()
        a = self.window(domain, shifts=(0.,)*3)[0]
        b = self.window(domain, source=1, shifts=(0.,)*3)[0]
        wrist = self.joint(compare_pose_phase_windows(a,b))
        self.assertEqual(wrist['velocity_direction_cosines'], [None,None])
        self.assertEqual(wrist['maximum_velocity_difference_per_second'], 0.)

    def test_reversal_inside_window_is_retained(self):
        domain = self.domain()
        a = self.window(domain, shifts=(0.,.01,0.))[0]
        b = self.window(domain, source=1, shifts=(0.,.01,.02))[0]
        wrist = self.joint(compare_pose_phase_windows(a,b))
        self.assertEqual(wrist['reference_reversals'], 1)
        self.assertEqual(wrist['candidate_reversals'], 0)
        self.assertLess(wrist['velocity_direction_cosines'][-1], -.99)

    def test_missing_wrist_keeps_its_required_duty_and_unknown_cost(self):
        domain = self.domain()
        a = self.window(domain)[0]
        b = self.window(domain, source=1, lost_joint=9)[0]
        report = compare_pose_phase_windows(a,b); wrist = self.joint(report)
        self.assertIsNone(wrist['multi_anchor_residual'])
        self.assertFalse(wrist['all_source_support_observed'])
        self.assertTrue(wrist['required'])
        self.assertFalse(report['missing_is_zero_cost'])
        self.assertIn(9, b.duties.required_joint_indices)
        self.assertEqual(b.anchors[0].joint_states[9], PoseJointObservationState.DETECTOR_UNKNOWN)

    def test_off_native_wrist_is_not_clamped_or_relabelled_source_occluded(self):
        window = self.window(off_joint=9)[0]
        self.assertEqual(window.anchors[0].joint_states[9], PoseJointObservationState.OUT_OF_SOURCE_FRAME)
        self.assertIsNone(window.anchors[0].torso_local_xy[9])
        self.assertFalse(window.to_dict()['source_occlusion_inferred_from_detector_failure'])

    def test_ambiguous_duplicate_owner_geometry_is_retained_in_all_eight_hypotheses(self):
        domain = self.domain()
        windows = self.window(domain, duplicate=True)
        self.assertEqual(len(windows), 8)
        self.assertEqual({tuple(a.person_observation_id for a in w.anchors) for w in windows},
            {(a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)})
        report = compare_pose_phase_windows(windows[0], self.window(domain, source=1)[0])
        self.assertIsNotNone(self.joint(report)['multi_anchor_residual'])
        self.assertFalse(report['all_required_source_support_observed'])
        self.assertEqual(windows[0].anchors[0].joint_states[9], PoseJointObservationState.OWNERSHIP_AMBIGUOUS)
        self.assertFalse(windows[0].to_dict()['source_local_identity_certified'])

    def test_changed_raw_binding_cannot_be_called_an_unavailable_zero_loss(self):
        domain = self.domain(); rows = tuple(self.row(domain,0,n) for n in (0,3,6))
        for change in ({'source_clock_identity':'b'*64}, {'source_file_identity':(2,3,4,5,6)},
                {'model_adapter_fingerprints':(('nanodet_people','b'*64),('rtmpose_body7','f'*64),('yunet_head_faces','f'*64))},
                {'person_ownership_policy_sha256':'b'*64}, {'frame_size_wh':(2000,800)},
                {'request_source_index':1}):
            bad = (*rows[:2],replace(rows[2],**change))
            with self.subTest(change=change), self.assertRaises(ValueError) as cm:
                native_pose_windows_from_rows(domain=domain,rows=bad,choices=((0,),)*3,duties=self.duties)
            self.assertNotIsInstance(cm.exception, TemporalPoseUnavailable)

    def test_native_camera_cut_has_distinct_unavailable_state(self):
        domain = self.domain(cuts=((.1,),()))
        with self.assertRaisesRegex(TemporalPoseUnavailable, 'broadcaster cut'):
            self.window(domain)

    def test_hypothesis_overflow_rejects_without_pruning_convenient_owner(self):
        domain = self.domain(); rows = tuple(self.row(domain,0,n,duplicate=True) for n in (0,3,6))
        with self.assertRaisesRegex(TemporalPoseUnavailable, 'not success-biased truncation'):
            native_pose_windows_from_rows(domain=domain,rows=rows,choices=((0,1),)*3,duties=self.duties,
                maximum_hypotheses=7)
        self.assertEqual(len(rows[0].people), 2)

    def test_forged_derived_geometry_or_timestamp_is_replayed_and_rejected(self):
        window = self.window()[0]
        anchor = window.anchors[1]
        xy = list(anchor.torso_local_xy); xy[9] = (0.,0.)
        for altered in (replace(anchor, torso_local_xy=tuple(xy)),
                replace(anchor, measurement_seconds=.11),
                replace(anchor, joint_states=(PoseJointObservationState.SOURCE_OCCLUDED,)*17)):
            with self.assertRaisesRegex(ValueError, 'replay'):
                replace(window, anchors=(window.anchors[0],altered,window.anchors[2]))

    def test_actual_output_window_and_cut_prediction_bind_complete_state_ticks(self):
        _,domain,states = state_fixtures.JointStateFactoryTests().fixture()
        out_ticks,in_ticks = (23,26,29),(30,33,36)
        windows = []
        for state,ticks in ((states[0],out_ticks),(states[1],in_ticks)):
            source = domain.request_index(state.eligible_source_index)
            rows = tuple(self.row(domain,source,state.native_frame_indices[t-state.first_output_frame],
                t*.001) for t in ticks)
            windows.append(render_pose_windows_from_rows(state=state,domain=domain,rows=rows,
                output_ticks=ticks,choices=((0,),)*3,duties=self.duties)[0])
        report = compare_pose_boundary_windows(*windows)
        self.assertAlmostEqual(report['boundary_gap_output_seconds'], 1/30.)
        self.assertLess(self.joint(report)['predicted_pose_jump'], 1e-12)
        self.assertLess(self.joint(report)['velocity_difference_per_output_second'], 1e-12)
        self.assertFalse(report['candidate_admitted'])
        with self.assertRaisesRegex(ValueError, 'chronological output'):
            compare_pose_boundary_windows(*reversed(windows))
        rows = tuple(a.observation for a in windows[0].anchors)
        with self.assertRaisesRegex(ValueError, 'output/native join'):
            render_pose_windows_from_rows(state=states[0],domain=domain,rows=rows,
                output_ticks=(22,25,28),choices=((0,),)*3,duties=self.duties)

    def test_repeated_render_native_ticks_are_retained_but_not_new_motion_samples(self):
        _,domain,states = state_fixtures.JointStateFactoryTests().fixture(59.94,Fraction(30000,1001))
        state = states[0]; ticks = tuple(range(6)); rows = []; by_native = {}
        for t in ticks:
            native = state.native_frame_indices[t]
            by_native.setdefault(native,self.row(domain,0,native,shift=.001*native))
            rows.append(by_native[native])
        result = render_pose_windows_from_rows(state=state,domain=domain,rows=tuple(rows),
            output_ticks=ticks,choices=((0,),)*6,duties=self.duties)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].anchors), 6)
        self.assertLess(len(by_native), len(rows))
        self.assertEqual(result[0].output_ticks, ticks)
        self.assertFalse(result[0].to_dict()['pose_or_velocity_imputed'])

    def test_boolean_invalid_duties_and_undeclared_person_are_denied(self):
        for fields in ({'required_joint_indices':(True,)}, {'minimum_joint_confidence':True},
                {'required_joint_indices':()}, {'reference':''}, {'minimum_joint_confidence':float('nan')}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(self.duties,**fields)
        domain = self.domain(); rows = tuple(self.row(domain,0,n) for n in (0,3,6))
        with self.assertRaisesRegex(ValueError, 'absent raw person'):
            native_pose_windows_from_rows(domain=domain,rows=rows,choices=((4,),)*3,duties=self.duties)

    def test_missing_native_decode_keeps_unknown_anchor_and_all_duties(self):
        domain = self.domain(); rows = tuple(self.row(domain,0,n) for n in (0,3,6))
        missing = replace(rows[1], native_pixel_sha256=None, frame_size_wh=(0,0), people=(), heads=(),
            reason='exact_native_decode_unavailable')
        windows = native_pose_windows_from_rows(domain=domain, rows=(rows[0],missing,rows[2]),
            choices=((0,),(None,),(0,)), duties=self.duties)
        self.assertEqual(windows[0].anchors[1].joint_states, (PoseJointObservationState.DETECTOR_UNKNOWN,)*17)
        self.assertEqual(windows[0].duties.required_joint_indices, self.duties.required_joint_indices)
        self.assertIsNone(self.joint(compare_pose_phase_windows(windows[0],self.window(domain,source=1)[0]))[
            'multi_anchor_residual'])

    def test_face_path_consumer_measures_all_body_alternatives_without_joint_lk_success(self):
        domain = self.domain(); rows = tuple(self.row(domain,0,n,duplicate=True) for n in (0,3,6))
        with patch('crosscut.group_joint_focal_frontier.native_joint_pixel_flows', side_effect=AssertionError('not a prerequisite')):
            windows = native_pose_hypotheses_for_face_paths(domain=domain,rows=rows,
                face_paths=((0,0,0),),duties=self.duties)
        self.assertEqual(len(windows), 8)
        self.assertTrue(all(faces == (0,0,0) for faces,_ in windows))
        self.assertTrue(all(not track.to_dict()['source_local_identity_certified'] for _,track in windows))
        with self.assertRaisesRegex(TemporalPoseUnavailable, 'not success-biased truncation'):
            native_pose_hypotheses_for_face_paths(domain=domain,rows=rows,
                face_paths=((0,0,0),),duties=self.duties,maximum_hypotheses=7)

    def test_existing_frontier_exports_all_new_temporal_owner_pairs_under_shared_budget(self):
        domain = self.domain(); pixels = np.zeros((800,1000,3),np.uint8)
        digest = hashlib.sha256(memoryview(pixels)).hexdigest()
        raw = {(s,n):replace(self.row(domain,s,n,shift=.001*n,duplicate=s==1),native_pixel_sha256=digest)
            for s in (0,1) for n in (0,3,6)}
        names = ('nanodet_people','rtmpose_body7','yunet_head_faces')
        budget = JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.)
        owner = SimpleNamespace(domain=domain,budget=budget,retain_native_pixels=Mock(),
            native_pixels=lambda *_:pixels, _assert_source_identity=lambda source:(1,source,3,4,5),
            _assert_model_assets=Mock(), models={name:(None,None,'e'*64) for name in names},
            model_adapters={name:'f'*64 for name in names})
        def appearance(_owner,row,**_kwargs):
            return (NativeFaceAppearance(0,tuple(1. if n==0 else 0. for n in range(128)),
                row.heads[0].candidate_body_observation_ids),)
        with patch('crosscut.group_joint_focal_frontier.observe_native_group',
                side_effect=lambda _owner,request_source_index,native_frame_index:raw[request_source_index,native_frame_index]), \
                patch('crosscut.group_joint_focal_frontier.observe_native_face_appearances',side_effect=appearance), \
                patch('crosscut.group_joint_focal_frontier.focal_landmarks_track',return_value=True):
            result = observe_synchronized_focal_frontier(owner,reference_source=0,canonical_times=(0.,.1,.2),face_roi_only=True)
        candidates = {c['source_index']:c for c in result['candidates']}
        self.assertEqual(len(candidates[0]['independent_native_pose_motion']), 1)
        self.assertEqual(len(candidates[1]['independent_native_pose_motion']), 8)
        for source in (0,1):
            for hypothesis in candidates[source]['independent_native_pose_motion']:
                spans = hypothesis['partial_sensor_spans']
                self.assertEqual(tuple(row['joint_index'] for row in spans),self.duties.required_joint_indices)
                self.assertTrue(all(not row['required_duty_satisfied'] for row in spans))
        comparisons = {c['source_index']:c for c in result['independent_raw17_temporal_pose_comparisons']}
        self.assertEqual(len(comparisons[1]['all_owner_pair_hypotheses']), 8)
        self.assertFalse(comparisons[1]['source_eligibility_or_appearance_ranking_changed'])
        self.assertTrue(all(not pair['evidence']['all_required_source_support_observed']
            for pair in comparisons[1]['all_owner_pair_hypotheses']))
        self.assertEqual(budget.counts['pose_comparisons'], 9)
        self.assertEqual(budget.counts['edges'], 0)
        self.assertFalse(result['production_admitted'])
        self.assertFalse(result['inherited_role_or_required_parts_downgraded'])
        encoded_size = lambda value: len(json.dumps(value,sort_keys=True,
            separators=(',', ':'),allow_nan=False).encode())
        self.assertEqual(budget.counts['pose_evidence_bytes'],
            sum(encoded_size(dict(conditional_raw_person_pose=c['independent_native_pose_motion'],
                unresolved_subject_graph=c['independent_subject_association_graph'])) for c in result['candidates'])
            + sum(encoded_size(c) for c in result['independent_raw17_temporal_pose_comparisons']))
        self.assertEqual(budget.counts['association_nodes'],15)
        self.assertEqual(budget.counts['association_edges'],26)
        self.assertTrue(all(c['independent_subject_association_graph']['resolved_subject_path'] is None
            for c in result['candidates']))
        self.assertTrue(all('partial_sensor_spans' not in pair['evidence']
            for c in result['independent_raw17_temporal_pose_comparisons']
            for pair in c['all_owner_pair_hypotheses']))
        # The unresolved graph survives even when the legacy Cartesian pose
        # inventory is too large. It does not become a chosen identity path.
        for n in (0,3,6):
            raw[1,n] = replace(self.row_people(domain,1,n,tuple(range(12))),native_pixel_sha256=digest)
        owner.budget = JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.)
        with patch('crosscut.group_joint_focal_frontier.observe_native_group',
                side_effect=lambda _owner,request_source_index,native_frame_index:raw[request_source_index,native_frame_index]), \
                patch('crosscut.group_joint_focal_frontier.observe_native_face_appearances',side_effect=appearance), \
                patch('crosscut.group_joint_focal_frontier.focal_landmarks_track',return_value=True):
            overflow_result = observe_synchronized_focal_frontier(owner,reference_source=0,
                canonical_times=(0.,.1,.2),face_roi_only=True)
        overflow = next(c for c in overflow_result['candidates'] if c['source_index'] == 1)
        self.assertEqual(overflow['independent_native_pose_motion'],[])
        graph = overflow['independent_subject_association_graph']
        self.assertEqual(len(graph['nodes']),39)
        self.assertEqual(len(graph['edges']),338)
        self.assertIsNone(graph['resolved_subject_path'])
        self.assertTrue(overflow['unresolved_subject_association_prevents_identity_or_motion_admission'])
        self.assertFalse(overflow_result['production_admitted'])
        self.assertEqual(owner.budget.counts['association_nodes'],45)
        self.assertEqual(owner.budget.counts['association_edges'],346)
        from crosscut.group_joint_source_camera import JointGroupBudgetExhausted
        owner.budget = JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,
            maximum_pose_evidence_bytes=1)
        with patch('crosscut.group_joint_focal_frontier.observe_native_group',
                side_effect=lambda _owner,request_source_index,native_frame_index:raw[request_source_index,native_frame_index]), \
                patch('crosscut.group_joint_focal_frontier.observe_native_face_appearances',side_effect=appearance), \
                patch('crosscut.group_joint_focal_frontier.focal_landmarks_track',return_value=True):
            with self.assertRaisesRegex(JointGroupBudgetExhausted,'pose_evidence_bytes'):
                observe_synchronized_focal_frontier(owner,reference_source=0,
                    canonical_times=(0.,.1,.2),face_roi_only=True)
        self.assertEqual(owner.budget.counts['pose_evidence_bytes'],0)

    def test_no_lk_face_paths_do_not_erase_actual_raw_pose_hypotheses(self):
        domain = self.domain(); rows = tuple(self.row(domain,0,n,duplicate=True) for n in (0,3,6))
        result = independent_native_pose_hypotheses(domain=domain,rows=rows,duties=self.duties,
            initial_person_ids=(0,))
        self.assertEqual(len(result), 4)
        self.assertTrue(all(w.anchors[0].person_observation_id == 0 for w in result))
        self.assertTrue(all(not w.to_dict()['lk_feature_survival_prerequisite'] for w in result))
        self.assertTrue(all(not w.to_dict()['source_local_identity_certified'] for w in result))
        with self.assertRaises(ValueError):
            independent_native_pose_hypotheses(domain=domain,rows=rows,duties=self.duties,initial_person_ids=(7,))

    def test_arcface_frontier_keeps_weak_ranked_donor_without_claiming_sface_shortlist(self):
        from crosscut.native_face_descriptor import (ARCFACE_MODEL_NAME, ARCFACE_R50_SHA256)
        domain = self.domain(); pixels = np.zeros((800,1000,3),np.uint8)
        digest = hashlib.sha256(memoryview(pixels)).hexdigest()
        raw = {(s,n):replace(self.row(domain,s,n,shift=.001*n),native_pixel_sha256=digest)
            for s in (0,1) for n in (0,3,6)}
        names = ('nanodet_people','rtmpose_body7','yunet_head_faces')
        budget = JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.)
        def actual_counted_descriptor(source,native,*,model,arguments):
            self.assertEqual(model,ARCFACE_MODEL_NAME)
            budget.spend('model_calls',model=model)
            self.assertEqual(arguments['bbox_xywh'],list(raw[source,native].heads[0].box_xywh))
            return tuple((1. if source==0 else .42) if i==0 else
                float(np.sqrt(1-.42**2)) if i==1 and source==1 else 0. for i in range(512))
        owner = SimpleNamespace(domain=domain,budget=budget,retain_native_pixels=Mock(),
            native_pixels=lambda *_:pixels,_assert_source_identity=lambda source:(1,source,3,4,5),
            _assert_model_assets=Mock(),models={**{name:(None,None,'e'*64) for name in names},
                ARCFACE_MODEL_NAME:(None,None,ARCFACE_R50_SHA256)},
            model_adapters={**{name:'f'*64 for name in names},ARCFACE_MODEL_NAME:'d'*64},
            observe_native=Mock(side_effect=actual_counted_descriptor))
        with patch('crosscut.group_joint_focal_frontier.observe_native_group',
                side_effect=lambda _owner,request_source_index,native_frame_index:raw[request_source_index,native_frame_index]), \
                patch('crosscut.group_joint_focal_frontier.focal_landmarks_track',return_value=True):
            result = observe_synchronized_focal_frontier(owner,reference_source=0,
                canonical_times=(0.,.1,.2),face_roi_only=True,descriptor_model=ARCFACE_MODEL_NAME)
        donor = next(c for c in result['candidates'] if c['source_index']==1)
        self.assertTrue(donor['ranking_only_path_available'])
        self.assertFalse(donor['shortlist_available'])
        self.assertEqual(donor['reason'],'descriptor_space_uncalibrated_ranking_only')
        self.assertEqual(tuple(donor['ranking_similarities']),(.42,)*3)
        self.assertFalse(donor['source_eligibility_changed'])
        self.assertIsNone(result['appearance_policy']['minimum_similarity'])
        self.assertEqual(result['appearance_policy']['descriptor_space']['dimension'],512)
        self.assertEqual(budget.model_calls,{ARCFACE_MODEL_NAME:6})
        self.assertFalse(result['production_admitted'])
        self.assertFalse(result['cross_source_identity_certified'])
        self.assertFalse(result['full_required_part_coverage_verified'])

    def test_remaining_person_does_not_erase_unassociated_subject_layer(self):
        domain = self.domain()
        rows = tuple(self.row_people(domain,0,n,ids) for n,ids in
            ((0,(0,1)),(3,(1,)),(6,(0,1))))
        graph = native_pose_association_graph(domain=domain,rows=rows,initial_person_ids=(0,))
        choices = tuple(tuple(graph.nodes[n][1] for n in layer) for layer in graph.layer_node_indices)
        self.assertEqual(choices, ((0,None),(1,None),(0,1,None)))
        unknown = graph.layer_node_indices[1][-1]
        self.assertTrue(any(b == unknown for _,b in graph.edges))
        self.assertTrue(any(a == unknown for a,_ in graph.edges))
        self.assertEqual(graph.to_dict()['possible_path_count'],12)
        self.assertIsNone(graph.to_dict()['resolved_subject_path'])
        self.assertFalse(graph.to_dict()['physical_subject_seed_verified'])
        self.assertFalse(graph.to_dict()['candidate_admitted'])
        self.assertTrue(graph.to_dict()['unassociated_subject_is_not_source_occlusion'])

    def test_repeated_first_native_layer_keeps_frozen_seed_and_all_raw_head_links(self):
        domain = self.domain(); first = self.row_people(domain,0,0,(0,1))
        rows = (first,first,self.row_people(domain,0,3,(0,1)),self.row_people(domain,0,6,(0,1)))
        graph = native_pose_association_graph(domain=domain,rows=rows,initial_person_ids=(0,))
        self.assertEqual(graph.native_frame_indices,(0,3,6))
        self.assertEqual(graph.anchor_layer_indices,(0,0,1,2))
        self.assertEqual(tuple(graph.nodes[n][1] for n in graph.layer_node_indices[0]),(0,None))
        self.assertEqual(graph.raw_head_body_alternatives[0],((0,(0,1)),))
        self.assertEqual(graph.to_dict()['possible_path_count'],18)

    def test_empty_detection_layer_is_unknown_not_other_person_or_occlusion(self):
        domain = self.domain()
        rows = tuple(self.row_people(domain,0,n,ids) for n,ids in
            ((0,(0,)),(3,()),(6,(0,))))
        graph = native_pose_association_graph(domain=domain,rows=rows)
        self.assertEqual(tuple(graph.nodes[n][1] for n in graph.layer_node_indices[1]),(None,))
        self.assertEqual(len(graph.edges),4)
        self.assertEqual(graph.to_dict()['materialized_subject_path_count'],0)
        self.assertTrue(graph.to_dict()['unassociated_subject_is_not_source_occlusion'])

    def test_dense_all_person_alternatives_are_compact_not_cartesian_paths(self):
        domain = self.domain()
        rows = tuple(self.row_people(domain,0,n,tuple(range(12))) for n in range(14))
        budget = JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.)
        graph = native_pose_association_graph(domain=domain,rows=rows,budget=budget)
        self.assertEqual(len(graph.nodes),182)
        self.assertEqual(len(graph.edges),2197)
        self.assertEqual(graph.to_dict()['possible_path_count'],13**14)
        self.assertEqual(graph.to_dict()['materialized_subject_path_count'],0)
        self.assertEqual(budget.counts['association_nodes'],182)
        self.assertEqual(budget.counts['association_edges'],2197)
        self.assertFalse(graph.to_dict()['geometric_duplicate_or_performer_alternatives_merged'])
        with self.assertRaisesRegex(TemporalPoseUnavailable,'not success-biased truncation'):
            independent_native_pose_hypotheses(domain=domain,rows=rows,duties=self.duties)

    def test_graph_overflow_and_shared_capacity_fail_without_pruning_or_partial_reservation(self):
        domain = self.domain(); rows = tuple(self.row_people(domain,0,n,(0,1)) for n in (0,3,6))
        for bounds in ({'maximum_nodes':8},{'maximum_edges':17}):
            budget = JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.)
            with self.subTest(bounds=bounds),self.assertRaisesRegex(TemporalPoseUnavailable,'budget_unresolved'):
                native_pose_association_graph(domain=domain,rows=rows,budget=budget,**bounds)
            self.assertEqual(budget.counts['association_nodes'],0)
            self.assertEqual(budget.counts['association_edges'],0)
        budget = JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,maximum_association_edges=18)
        budget.spend('association_edges',1)
        before = budget.to_dict()
        with self.assertRaisesRegex(JointGroupBudgetExhausted,'association_edges'):
            native_pose_association_graph(domain=domain,rows=rows,budget=budget)
        self.assertEqual(budget.to_dict(),before)
        self.assertEqual(tuple(len(row.people) for row in rows),(2,2,2))

    def test_graph_binding_cut_and_repeated_raw_corruption_are_not_zero_cost_unknowns(self):
        domain = self.domain(); rows = tuple(self.row(domain,0,n) for n in (0,3,6))
        bad = (*rows[:2],replace(rows[2],model_adapter_fingerprints=(
            ('nanodet_people','b'*64),('rtmpose_body7','f'*64),('yunet_head_faces','f'*64))))
        with self.assertRaises(ValueError) as cm:
            native_pose_association_graph(domain=domain,rows=bad)
        self.assertNotIsInstance(cm.exception,TemporalPoseUnavailable)
        with self.assertRaisesRegex(ValueError,'contradictory raw'):
            native_pose_association_graph(domain=domain,rows=(rows[0],replace(rows[0],reason='altered'),*rows[1:]))
        cut_domain = self.domain(cuts=((.1,),()))
        with self.assertRaisesRegex(TemporalPoseUnavailable,'broadcaster cut'):
            native_pose_association_graph(domain=cut_domain,
                rows=tuple(self.row(cut_domain,0,n) for n in (0,3,6)))

    def test_detection_order_does_not_choose_or_merge_a_subject(self):
        domain = self.domain()
        forward = tuple(self.row_people(domain,0,n,(0,1)) for n in (0,3,6))
        reversed_rows = tuple(self.row_people(domain,0,n,(1,0)) for n in (0,3,6))
        a = native_pose_association_graph(domain=domain,rows=forward)
        b = native_pose_association_graph(domain=domain,rows=reversed_rows)
        self.assertEqual((a.nodes,a.edges,a.layer_node_indices),(b.nodes,b.edges,b.layer_node_indices))
        # Graph topology is order-independent, but raw preimages retain their
        # original detector enumeration and are not silently canonicalized.
        self.assertEqual(tuple(tuple((head,set(ids)) for head,ids in layer)
            for layer in a.raw_head_body_alternatives),tuple(tuple((head,set(ids)) for head,ids in layer)
            for layer in b.raw_head_body_alternatives))
        self.assertNotEqual(a.raw_observation_sha256,b.raw_observation_sha256)

    def test_graph_limits_and_frozen_seed_have_strict_types(self):
        domain = self.domain(); rows = tuple(self.row(domain,0,n) for n in (0,3,6))
        for args in ({'maximum_nodes':True},{'maximum_nodes':193},{'maximum_edges':2201},
                {'maximum_edges':0},{'budget':object()},{'initial_person_ids':(True,)},
                {'initial_person_ids':(0,0)},{'initial_person_ids':(7,)},{'initial_person_ids':[]}):
            with self.subTest(args=args),self.assertRaises(ValueError):
                native_pose_association_graph(domain=domain,rows=rows,**args)
        for args in ({'maximum_association_nodes':1537},{'maximum_association_edges':18001},
                {'maximum_association_nodes':True},{'maximum_association_edges':0}):
            with self.subTest(args=args),self.assertRaises(ValueError):
                JointGroupWorkBudget(absolute_deadline=10.,**args)
        with self.assertRaisesRegex(JointGroupBudgetExhausted,'absolute work deadline'):
            native_pose_association_graph(domain=domain,rows=rows,
                budget=JointGroupWorkBudget(absolute_deadline=1.,clock=lambda:1.))

    def test_pose_comparison_work_has_distinct_shared_deadline_counter_and_hard_cap(self):
        from crosscut.group_joint_source_camera import JointGroupBudgetExhausted
        budget = JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,maximum_pose_comparisons=2)
        budget.spend('pose_comparisons',2)
        with self.assertRaisesRegex(JointGroupBudgetExhausted,'pose_comparisons'):
            budget.spend('pose_comparisons')
        self.assertEqual(budget.counts['pose_comparisons'],2)
        self.assertEqual(budget.counts['edges'],0)
        with self.assertRaises(ValueError):
            JointGroupWorkBudget(absolute_deadline=10.,maximum_pose_comparisons=4097)

    def test_actual_simcc_above_one_remains_raw_and_legacy_body8_clipping_is_replayed(self):
        domain = self.domain(); rows = []
        for n in (0,3,6):
            row = self.row(domain,0,n); p = row.people[0]; raw = list(p.raw_pose_xyc)
            raw[0] = (*raw[0][:2],1.0234019756317139)
            raw[5] = (*raw[5][:2],1.0044535398483276)
            upper = list(p.upper_body_xyc); upper[0] = (*upper[0][:2],1.)
            p = replace(p,raw_pose_xyc=tuple(raw),upper_body_xyc=tuple(upper))
            people,_ = bind_person_ownership((p,),row.heads,row.frame_size_wh)
            rows.append(replace(row,people=people))
        windows = native_pose_windows_from_rows(domain=domain,rows=tuple(rows),choices=((0,),)*3,duties=self.duties)
        self.assertEqual(windows[0].anchors[0].joint_states[0],PoseJointObservationState.VISIBLE_SUPPORTED)
        self.assertIsNotNone(windows[0].anchors[0].torso_local_xy[0])
        self.assertEqual(windows[0].anchors[0].observation.people[0].raw_pose_xyc[0][2],1.0234019756317139)
        self.assertFalse(windows[0].to_dict()['raw_scores_clipped'])
        self.assertTrue(windows[0].to_dict()['legacy_body8_score_clipping_replayed'])

    def test_pose_evidence_serialization_bytes_share_same_budget_without_reset(self):
        from crosscut.group_joint_source_camera import JointGroupBudgetExhausted
        budget = JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,maximum_pose_evidence_bytes=8)
        budget.spend('pose_evidence_bytes',5); budget.spend('pose_evidence_bytes',3)
        with self.assertRaisesRegex(JointGroupBudgetExhausted,'pose_evidence_bytes'):
            budget.spend('pose_evidence_bytes',1)
        self.assertEqual(budget.counts['pose_evidence_bytes'],8)
        with self.assertRaises(ValueError):
            JointGroupWorkBudget(absolute_deadline=10.,maximum_pose_evidence_bytes=8*1024*1024+1)


    def test_partial_spans_keep_both_sides_of_missing_wrist_without_imputation(self):
        domain = self.domain()
        rows = tuple(self.row(domain,0,n,shift=.001*n,
            lost_joint=9 if n == 6 else None) for n in (0,3,6,9,12))
        window = native_pose_windows_from_rows(domain=domain,rows=rows,
            choices=((0,),)*5,duties=self.duties)[0]
        evidence = supported_pose_spans(window,9)
        self.assertEqual([s.unique_anchor_indices for s in evidence.spans],[(0,1),(3,4)])
        self.assertEqual([len(s.secant_velocity_xy_per_second) for s in evidence.spans],[1,1])
        self.assertEqual(evidence.unavailable_anchor_reasons[0][0],2)
        self.assertIsNone(window.anchors[2].torso_local_xy[9])
        report = compare_pose_phase_windows(window,window)
        self.assertIsNone(self.joint(report)['multi_anchor_residual'])
        self.assertFalse(report['candidate_admitted'])
        self.assertFalse(evidence.to_dict()['required_duty_satisfied'])
        json.dumps(report,allow_nan=False)

    def test_ambiguous_owner_keeps_nominal_raw_cost_but_breaks_supported_span(self):
        domain = self.domain()
        rows = tuple(self.row(domain,0,n,duplicate=n == 6) for n in (0,3,6,9,12))
        windows = native_pose_windows_from_rows(domain=domain,rows=rows,
            choices=((0,),(0,),(0,1),(0,),(0,)),duties=self.duties)
        self.assertEqual(len(windows),2)
        for window in windows:
            evidence = supported_pose_spans(window,9)
            self.assertEqual([s.unique_anchor_indices for s in evidence.spans],[(0,1),(3,4)])
            self.assertTrue(any('ownership_ambiguous' in reason
                for _, reasons in evidence.unavailable_anchor_reasons for reason in reasons))
            report = compare_pose_phase_windows(window,window)
            self.assertIsNotNone(self.joint(report)['multi_anchor_residual'])
            self.assertFalse(report['all_required_source_support_observed'])
            self.assertFalse(report['candidate_admitted'])

    def test_degenerate_shoulders_do_not_certify_torso_local_kinematics(self):
        domain = self.domain(); rows = []
        for n in (0,3,6):
            row = self.row(domain,0,n); person = row.people[0]
            raw = list(person.raw_pose_xyc); raw[6] = raw[5]
            person = replace(person,raw_pose_xyc=tuple(raw),
                upper_body_xyc=tuple(raw[i] for i in (5,6,7,8,9,10,11,12)))
            people,_ = bind_person_ownership((person,),row.heads,row.frame_size_wh)
            rows.append(replace(row,people=people))
        window = native_pose_windows_from_rows(domain=domain,rows=tuple(rows),
            choices=((0,),)*3,duties=self.duties)[0]
        self.assertTrue(all(a.joint_states[9] == PoseJointObservationState.VISIBLE_SUPPORTED
            for a in window.anchors))
        evidence = supported_pose_spans(window,9)
        self.assertEqual(evidence.spans,())
        self.assertFalse(evidence.all_scheduled_kinematic_samples_observed)
        self.assertTrue(all('torso_local_coordinate_unavailable' in reasons
            for _,reasons in evidence.unavailable_anchor_reasons))
        report = compare_pose_phase_windows(window,window)
        # Preserve legacy raw detector diagnostics, but distinguish coordinate
        # readiness instead of letting that flag certify valid kinematics.
        self.assertTrue(self.joint(report)['all_source_support_observed'])
        self.assertFalse(self.joint(report)['all_torso_local_kinematic_samples_observed'])
        self.assertFalse(report['all_required_torso_local_kinematic_samples_observed'])
        self.assertFalse(report['independent_physical_source_visibility_certified'])

    def test_frozen_first_native_cohort_covers_repeated_first_raster(self):
        domain = self.domain()
        first = self.row(domain,0,0,duplicate=True)
        rows = (first,first,self.row(domain,0,3,duplicate=True),
            self.row(domain,0,6,duplicate=True))
        windows = independent_native_pose_hypotheses(domain=domain,rows=rows,
            duties=self.duties,initial_person_ids=(0,))
        self.assertEqual(len(windows),4)
        self.assertTrue(all(tuple(a.person_observation_id for a in w.anchors[:2]) == (0,0)
            for w in windows))
        with self.assertRaises(TemporalPoseUnavailable):
            independent_native_pose_hypotheses(domain=domain,rows=rows,
                duties=self.duties,initial_person_ids=(0,),maximum_hypotheses=3)

    def test_predeclared_gap_limit_splits_secants_but_not_observed_points(self):
        window = self.window()[0]
        evidence = supported_pose_spans(window,9,maximum_gap_seconds=.05)
        self.assertEqual([len(s.unique_anchor_indices) for s in evidence.spans],[1,1,1])
        self.assertTrue(evidence.all_scheduled_kinematic_samples_observed)
        self.assertTrue(all(s.secant_velocity_xy_per_second == () for s in evidence.spans))
        self.assertFalse(evidence.to_dict()['whole_interval_visibility_certified'])
        for value in (True,0.,-1.,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                supported_pose_spans(window,9,maximum_gap_seconds=value)

    def test_model_point_outside_raster_does_not_become_anatomical_absence(self):
        window = self.window(off_joint=9)[0]
        evidence = supported_pose_spans(window,9)
        self.assertEqual(evidence.spans,())
        self.assertTrue(all(any('not_anatomical_proof' in reason for reason in reasons)
            for _, reasons in evidence.unavailable_anchor_reasons))
        self.assertFalse(evidence.to_dict()['physical_source_occlusion_or_out_of_frame_certified'])

    def test_repeated_native_raster_does_not_add_zero_velocity_to_partial_spans(self):
        domain = self.domain(); first = self.row(domain,0,0)
        rows = (first,first,self.row(domain,0,3,shift=.01),self.row(domain,0,6,shift=.02))
        window = native_pose_windows_from_rows(domain=domain,rows=rows,
            choices=((0,),)*4,duties=self.duties)[0]
        evidence = supported_pose_spans(window,9)
        self.assertEqual(evidence.repeated_native_anchor_count,1)
        self.assertEqual(len(evidence.spans[0].secant_velocity_xy_per_second),2)
        self.assertTrue(all(v[0] > 0 for v in evidence.spans[0].secant_velocity_xy_per_second))

    def test_partial_spans_preserve_real_seconds_for_30_and_60_fps_sources(self):
        domain = self.domain((Fraction(30),Fraction(60)))
        a = supported_pose_spans(self.window(domain)[0],9)
        b = supported_pose_spans(self.window(domain,source=1,frames=(0,6,12))[0],9)
        self.assertEqual(a.spans[0].measurement_seconds,b.spans[0].measurement_seconds)
        self.assertEqual(a.spans[0].secant_velocity_xy_per_second,
            b.spans[0].secant_velocity_xy_per_second)
        with self.assertRaises(ValueError):
            supported_pose_spans(self.window(domain)[0],11)


    def test_supported_then_ambiguous_then_missing_wrist_keeps_entry_velocity_unknown(self):
        domain = self.domain()
        rows = (self.row(domain,0,6168),self.row(domain,0,6169,duplicate=True),
            self.row(domain,0,6170,lost_joint=10))
        windows = native_pose_windows_from_rows(domain=domain,rows=rows,
            choices=((0,),(0,1),(0,)),duties=self.duties)
        self.assertEqual(len(windows),2)
        for window in windows:
            evidence = supported_pose_spans(window,10)
            self.assertEqual(tuple(span.unique_anchor_indices for span in evidence.spans),((0,),))
            self.assertEqual(evidence.spans[0].secant_velocity_xy_per_second,())
            self.assertEqual(tuple(i for i,_ in evidence.unavailable_anchor_reasons),(1,2))
            self.assertIsNone(self.joint(compare_pose_phase_windows(window,window),10)[
                'maximum_velocity_difference_per_second'])
            self.assertFalse(evidence.to_dict()['required_duty_satisfied'])

    def test_missing_shoulder_invalidates_high_confidence_wrist_coordinate_not_wrist_duty(self):
        window = self.window(lost_joint=5)[0]
        # The actual ownership binder can also lose the body's supported
        # state when a shoulder disappears. A high raw wrist score is not
        # permission to override that independently replayed owner state.
        self.assertGreater(window.anchors[0].observation.people[0].raw_pose_xyc[9][2],.3)
        evidence = supported_pose_spans(window,9)
        self.assertEqual(evidence.spans,())
        self.assertTrue(all('joint_5:detector_unknown' in reasons
            for _,reasons in evidence.unavailable_anchor_reasons))
        self.assertIn(9,window.duties.required_joint_indices)
        self.assertFalse(self.joint(compare_pose_phase_windows(window,window))[
            'all_torso_local_kinematic_samples_observed'])

    def test_partial_spans_keep_fractional_native_seconds(self):
        domain = self.domain((Fraction(30000,1001),)*2)
        evidence = supported_pose_spans(self.window(domain)[0],9)
        self.assertAlmostEqual(evidence.spans[0].measurement_seconds[1],.1001)
        self.assertAlmostEqual(evidence.spans[0].secant_velocity_xy_per_second[0][0],.1/.1001)


if __name__ == '__main__':
    unittest.main()
