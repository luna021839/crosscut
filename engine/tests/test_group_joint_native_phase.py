from dataclasses import replace
import unittest

from crosscut.group_joint_native_phase import (NativePhaseSearchPolicy,
    NativePhaseHypothesis, phase_adjusted_group_state)
from crosscut.group_joint_state_factory import candidate_joint_group_state, assemble_single_group_source_camera_state
from crosscut.group_joint_source_camera import joint_state_from_render_plan
from engine.tests import test_group_joint_state_factory as factory_fixtures


class NativePhaseProposalTests(unittest.TestCase):
    def fixture(self):
        plan, domain, states = factory_fixtures.JointStateFactoryTests().single_shot_fixture()
        return plan, domain, states

    def adjust(self, state, domain, first=2, last=3, policy=None):
        policy = policy or NativePhaseSearchPolicy()
        hypothesis = NativePhaseHypothesis(state.key, policy.sha256, first, last)
        return phase_adjusted_group_state(state=state, domain=domain, hypothesis=hypothesis, policy=policy)

    def test_5994_native_2997_output_all_ticks_rebind_across_two_cells(self):
        plan, domain, states = self.fixture()
        base = candidate_joint_group_state(plan=plan, domain=domain, incumbent_state=states[1],
            eligible_source_index=2)
        changed = self.adjust(base, domain)
        proposed, indices = assemble_single_group_source_camera_state(plan=plan, domain=domain,
            path=(states[0], changed, states[2]))
        rebound = joint_state_from_render_plan(plan=proposed, domain=domain, slot=1,
            first_output_frame=30, end_output_frame=60, role=base.role)
        self.assertEqual(indices, (1, 2))
        self.assertEqual(rebound.key, changed.key)
        self.assertEqual(changed.native_frame_indices[0], base.native_frame_indices[0]+2)
        self.assertEqual(changed.native_frame_indices[-1], base.native_frame_indices[-1]+3)
        self.assertEqual(changed.segment_output_frame_counts, (12, 18))
        self.assertEqual(proposed.segments[0], plan.segments[0])
        self.assertEqual(proposed.segments[-1], plan.segments[-1])
        self.assertNotEqual(base.key, changed.key)

    def test_zero_shift_returns_same_complete_state(self):
        _, domain, states = self.fixture()
        self.assertIs(self.adjust(states[1], domain, 0, 0), states[1])

    def test_single_endpoint_cannot_bypass_existing_rate_limit(self):
        _, domain, states = self.fixture()
        with self.assertRaisesRegex(ValueError, 'rate-change bound'):
            self.adjust(states[1], domain, 0, 3)
        # A constant native shift has zero speed change, but not lip-sync proof.
        proposed = self.adjust(states[1], domain, 3, 3)
        self.assertAlmostEqual(proposed.segments[0].source_playback_rate, 1.)

    def test_frozen_state_policy_and_integer_endpoint_bindings_are_required(self):
        _, domain, states = self.fixture()
        policy = NativePhaseSearchPolicy()
        valid = NativePhaseHypothesis(states[1].key, policy.sha256, 1, 1)
        for wrong in (replace(valid, baseline_complete_state_key='a'*64),
                replace(valid, policy_sha256='b'*64)):
            with self.assertRaisesRegex(ValueError, 'binding'):
                phase_adjusted_group_state(state=states[1], domain=domain, hypothesis=wrong, policy=policy)
        for value in (True, 2.5, None):
            with self.assertRaises(ValueError):
                replace(valid, first_sample_delta_native_frames=value)

    def test_offset_cap_and_native_cut_remain_hard_constraints(self):
        _, domain, states = self.fixture()
        with self.assertRaisesRegex(ValueError, 'absolute source-time bound'):
            self.adjust(states[1], domain, 8, 8)
        native = states[1].native_frame_indices[-1]+1
        cuts = list(domain.scene_cut_times)
        cuts[states[1].eligible_source_index] = (native/float(domain.render_frame_clocks[1].frame_rate),)
        with self.assertRaisesRegex(ValueError, 'actual native camera cut'):
            self.adjust(states[1], replace(domain, scene_cut_times=tuple(cuts)), 3, 3)

    def test_aligned_rate_does_not_buy_excess_absolute_render_speed(self):
        _, domain, states = self.fixture()
        state = states[1]
        # Construct an already valid 1.9% base state and bind its true sample grid.
        from crosscut.render_frame_schedule import sampler_frame_schedule
        parts = (replace(state.segments[0], source_duration=state.segments[0].duration*1.019),
            replace(state.segments[1], source_start=state.segments[0].source_start+.4*1.019,
                source_duration=.6*1.019))
        grid = sampler_frame_schedule(source_start=parts[0].source_start, source_playback_rate=1.019,
            native_fps=float(domain.render_frame_clocks[1].frame_rate), output_fps=state.output_fps,
            frame_count=30, source_frame_count=10000, maximum_terminal_hold_frames=0)
        state = replace(state, segments=parts, native_frame_indices=tuple(grid.source_frame_indices))
        with self.assertRaisesRegex(ValueError, 'render safety bound'):
            self.adjust(state, domain, 2, 3)

    def test_stale_native_grid_or_fixed_outer_state_cannot_be_phase_adjusted(self):
        _, domain, states = self.fixture()
        with self.assertRaisesRegex(ValueError, 'binding'):
            self.adjust(states[0], domain)
        stale = replace(states[1], native_frame_indices=tuple(n+1 for n in states[1].native_frame_indices))
        with self.assertRaisesRegex(ValueError, 'base grid'):
            self.adjust(stale, domain, 1, 1)

    def test_policy_never_waives_global_render_speed_or_excess_offset(self):
        for key, value in (('maximum_absolute_offset_seconds', .2),
                ('maximum_rate_change_from_aligned', .05),
                ('maximum_absolute_offset_seconds', float('nan'))):
            with self.assertRaises(ValueError):
                NativePhaseSearchPolicy(**{key: value})
        _, domain, states = self.fixture()
        hypothesis = NativePhaseHypothesis(states[1].key, NativePhaseSearchPolicy().sha256, 0, 0)
        self.assertFalse(hypothesis.to_dict()['candidate_admitted'])
