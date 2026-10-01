from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.group_joint_native_preparation import (
    prepare_joint_group_native, prepare_joint_group_native_in_session,
    _native_window_schedule,
)
from crosscut.group_joint_observations import GroupJointObservationSession
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.render import RenderTransition
from tests import test_group_joint_state_factory as fixtures


class NativePreparationOwnerTests(unittest.TestCase):
    def fixture(self, *, single=False, **limits):
        plan, domain, _ = fixtures.JointStateFactoryTests().fixture()
        bounds = ((0, 30), (30, 60), (60, 90), (90, 120))
        if single:
            plan = replace(plan, segments=(plan.segments[0], plan.segments[1],
                replace(plan.segments[2], source_index=plan.segments[1].source_index), plan.segments[3]),
                transitions=(RenderTransition(), RenderTransition(tier='D', reason='one_native_shot'),
                    RenderTransition()))
            bounds = ((0, 30), (30, 90), (90, 120))
        budget = JointGroupWorkBudget(absolute_deadline=100., clock=lambda: 0., **limits)
        session = GroupJointObservationSession(domain=domain, budget=budget)
        # No detector is loaded by these synthetic wiring tests. The real
        # session/model/reader tests separately cover registration and assets.
        session.models = dict.fromkeys(('nanodet_people', 'rtmpose_body7', 'yunet_head_faces'),
            (None, None, 'a'*64, None))
        return plan, bounds, session

    def sense(self, session, *, request_source_index, native_frame_indices):
        session.budget.spend('new_decodes', len(native_frame_indices))
        session.budget.spend('model_calls', len(native_frame_indices), model='synthetic_only')
        return SimpleNamespace(complete_required_native_paths=True, to_dict=lambda: {
            'synthetic_only': True, 'quality_approved': False})

    def run_prefix(self, plan, bounds, session):
        with patch('crosscut.group_joint_native_preparation.track_native_group', side_effect=self.sense), \
                patch.object(session, '_assert_model_assets'):
            return prepare_joint_group_native_in_session(plan=plan, window_bounds=bounds, session=session)

    def test_three_shots_preserve_every_output_tick_of_the_whole_two_cell_unit(self):
        plan, bounds, session = self.fixture(single=True)
        result = self.run_prefix(plan, bounds, session)
        self.assertEqual(len(result.tracklets), 3)
        self.assertEqual(result.output_native_join[1], tuple((tick, 2, tick) for tick in range(30, 90)))
        self.assertEqual(result.output_native_join[0], tuple((tick, 0, tick) for tick in range(26, 30)))
        self.assertEqual(result.output_native_join[2], tuple((tick, 5, tick) for tick in range(90, 94)))
        self.assertEqual(session.budget.counts['new_decodes'], 68)
        self.assertFalse(session.closed)
        self.assertFalse(result.to_dict()['production_bundle_created'])
        self.assertFalse(result.to_dict()['quality_approved'])
        self.assertEqual(result.to_dict()['actual_shot_count'], 3)

    def test_one_owner_session_keeps_cumulative_budget_and_models_for_later_work(self):
        plan, bounds, session = self.fixture(single=True, maximum_new_decodes=100)
        with patch('crosscut.group_joint_native_preparation.register_joint_native_models') as register, \
                patch('crosscut.group_joint_native_preparation.GroupJointObservationSession.close') as close:
            first = self.run_prefix(plan, bounds, session)
            second = self.run_prefix(plan, bounds, session)
        self.assertEqual(len(first.tracklets), 3)
        self.assertEqual(second.tracklets, ())
        self.assertEqual(second.status, 'dense_budget_unavailable_before_native_sensing')
        self.assertEqual(session.budget.counts['new_decodes'], 68)
        self.assertEqual(session.budget.model_calls, {'synthetic_only': 68})
        register.assert_not_called()
        close.assert_not_called()

    def test_model_budget_preflight_accounts_for_already_spent_discovery_work(self):
        plan, bounds, session = self.fixture(single=True, maximum_model_calls=1000)
        session.budget.spend('model_calls', 100, model='earlier_discovery')
        with patch('crosscut.group_joint_native_preparation.track_native_group') as tracker:
            result = prepare_joint_group_native_in_session(plan=plan, window_bounds=bounds, session=session)
        tracker.assert_not_called()
        self.assertEqual(session.budget.model_calls, {'earlier_discovery': 100})
        self.assertEqual(result.tracklets, ())

    def test_four_real_shot_legacy_window_keeps_two_distinct_dense_grids(self):
        plan, bounds, session = self.fixture()
        result = self.run_prefix(plan, bounds, session)
        self.assertEqual(len(result.output_native_join), 4)
        self.assertEqual(result.output_native_join[1], tuple((n, 2, n) for n in range(30, 60)))
        self.assertEqual(result.output_native_join[2], tuple((n, 3, n) for n in range(60, 90)))

    def test_fractional_output_native_clock_join_uses_the_full_renderer_sampler(self):
        from fractions import Fraction
        from crosscut.render import segment_frame_counts
        from crosscut.render_boundary_clock import native_frame_clocks_for_output
        plan, domain, _ = fixtures.JointStateFactoryTests().fixture(29.97, Fraction(60000, 1001))
        counts = segment_frame_counts(plan)
        ticks = [0]
        for count in counts:
            ticks.append(ticks[-1]+count)
        bounds = tuple(zip(ticks, ticks[1:]))
        _, joins, _ = _native_window_schedule(plan, domain, bounds)
        full = native_frame_clocks_for_output(plan, domain.clocks_by_request, range(ticks[-1]))
        for rows in joins:
            for output_tick, source, native in rows:
                self.assertEqual((source, native),
                    (full[output_tick].source_index, full[output_tick].native_frame_index))

    def test_model_asset_change_is_rechecked_before_reusing_the_same_owner(self):
        plan, bounds, session = self.fixture()
        with patch.object(session, '_assert_model_assets', side_effect=ValueError('changed model asset')), \
                patch('crosscut.group_joint_native_preparation.track_native_group') as tracker, \
                self.assertRaisesRegex(ValueError, 'changed model asset'):
            prepare_joint_group_native_in_session(plan=plan, window_bounds=bounds, session=session)
        tracker.assert_not_called()

    def test_expired_shared_deadline_rejects_before_computing_another_grid(self):
        plan, bounds, session = self.fixture()
        session.budget.deadline = 0.
        with patch('crosscut.group_joint_native_preparation._native_window_schedule') as schedule, \
                self.assertRaisesRegex(ValueError, 'deadline'):
            prepare_joint_group_native_in_session(plan=plan, window_bounds=bounds, session=session)
        schedule.assert_not_called()

    def test_fake_internal_cell_seam_is_not_admitted_as_two_actual_shots(self):
        plan, _, session = self.fixture(single=True)
        with self.assertRaisesRegex(ValueError, 'actual cuts'):
            prepare_joint_group_native_in_session(plan=plan,
                window_bounds=((0, 30), (30, 60), (60, 90), (90, 120)), session=session)

    def test_missing_model_and_closed_session_fail_without_sensing_or_new_owner(self):
        for fault in ('model', 'closed'):
            plan, bounds, session = self.fixture()
            if fault == 'model':
                session.models.pop('yunet_head_faces')
            else:
                session.closed = True
            with self.subTest(fault=fault), \
                    patch('crosscut.group_joint_native_preparation.track_native_group') as tracker, \
                    self.assertRaises(ValueError):
                prepare_joint_group_native_in_session(plan=plan, window_bounds=bounds, session=session)
            tracker.assert_not_called()
            self.assertEqual(session.budget.counts['new_decodes'], 0)

    def test_changed_domain_or_non_cell_window_rejects_before_sensing(self):
        plan, bounds, session = self.fixture()
        for candidate, window in ((replace(plan, sources=tuple(reversed(plan.sources))), bounds),
                (plan, ((0, 30), (30, 61), (61, 90), (90, 120)))):
            with self.subTest(window=window), self.assertRaises(ValueError):
                prepare_joint_group_native_in_session(plan=candidate, window_bounds=window, session=session)
        self.assertEqual(session.budget.counts['new_decodes'], 0)

    def test_public_prefix_declines_large_dense_window_before_loading_models(self):
        plan, _, session = self.fixture()
        plan = replace(plan, segments=tuple(replace(s, source_start=i*7., master_start=i*7., duration=7.)
            for i, s in enumerate(plan.segments)))
        with patch('crosscut.group_joint_native_preparation.register_joint_native_models') as register, \
                patch('crosscut.group_joint_native_preparation.GroupJointObservationSession') as owner:
            result = prepare_joint_group_native(plan=plan, domain=session.domain,
                window_bounds=((0, 210), (210, 420), (420, 630), (630, 840)),
                person_model_path=None, pose_model_path=None, head_model_path=None,
                absolute_deadline=10**12)
        register.assert_not_called()
        owner.assert_not_called()
        self.assertEqual(result.tracklets, ())


if __name__ == '__main__':
    unittest.main()
