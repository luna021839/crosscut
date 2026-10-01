"""Numeric clock/consumer fixtures, not broadcast pose or lip-sync evidence."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import pickle
import unittest
from unittest.mock import patch

from crosscut.shot_video_clock import (FrozenShotVideoClockContext, authorize_shot_video_clock,
    attach_authorized_shot_video_clocks, shot_video_clock_from_dict,
    validate_authorized_shot_video_clocks, validate_shot_video_clock_bindings,
    resolved_shot_video_clock_slices)
from crosscut.group_joint_native_phase import (NativePhaseSearchPolicy,
    NativePhaseHypothesis, phase_adjusted_group_state)
from crosscut.group_joint_render_observations import refresh_joint_group_render_observations
from crosscut.group_joint_bundle_metadata import complete_joint_group_bundle_metadata
from crosscut.pipeline import _validate_review_first_group_bundle
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut.render import build_filter_graph
from crosscut.timeline_interchange import build_otio_payload
from tests import test_group_joint_bundle_metadata as metadata_fixture
from tests.test_visual_boundary_repair import _descriptor
from tests.test_sparse_source_lattice import Sampler
from tests import test_group_joint_state_factory as state_fixture
from crosscut.group_joint_state_factory import (candidate_joint_group_state,
    assemble_single_group_source_camera_state, assemble_joint_group_states)
from crosscut.group_joint_source_camera import joint_state_from_render_plan


class ShotVideoClockTests(unittest.TestCase):
    def prepare(self, *, cells=4, mutable_first=1):
        args = metadata_fixture.JointCompleteMetadataTests().fixture(single=True, cells=cells, mutable_first=mutable_first)
        base = args['observations'].path[1]
        context = FrozenShotVideoClockContext(args['incumbent'].render_plan,
            args['domain'], NativePhaseSearchPolicy())
        hypothesis = NativePhaseHypothesis(base.key, context.policy.sha256, 1, 1)
        phase = phase_adjusted_group_state(state=base, domain=args['domain'],
            hypothesis=hypothesis, policy=context.policy)
        clock = authorize_shot_video_clock(context=context, aligned_state=base,
            phase_state=phase, hypothesis=hypothesis)
        return args, context, base, phase, clock

    def refresh(self, args, context, phase, clock):
        path = args['observations'].path
        with patch('crosscut.visual_score._VideoSampler', metadata_fixture.GeometryJointSampler), \
                patch('crosscut.visual_score.describe_frame', return_value=_descriptor(.5)):
            rows = refresh_joint_group_render_observations(plan=args['incumbent'].render_plan,
                timeline=args['incumbent'].visual_timeline, domain=args['domain'],
                path=(path[0], phase, path[2]), sync_scores=(1.,)*4,
                shot_clock_context=context, shot_video_clocks=(clock,))
        return rows

    def test_plus_one_native_phase_refreshes_real_numeric_metadata_and_existing_consumer(self):
        args, context, base, phase, clock = self.prepare()
        rows = self.refresh(args, context, phase, clock)
        complete = complete_joint_group_bundle_metadata(**dict(args, observations=rows,
            shot_clock_context=context))
        bundle = complete.candidate_bundle
        _validate_review_first_group_bundle(bundle, eligible_indices=args['domain'].eligible_indices,
            offsets=args['domain'].audio_offsets, timewarps=args['domain'].source_timewarps,
            common_start=0., phrase_times=bundle.phrase_times, shot_clock_context=context)
        self.assertEqual(clock.aligned_native_indices[0]+1, clock.native_indices[0])
        self.assertEqual(clock.segment_indices, (1, 2))
        for index in clock.segment_indices:
            self.assertAlmostEqual(bundle.visual_timeline.choices[index].source_local_start,
                bundle.render_plan.segments[index].source_start)
        self.assertEqual(bundle.render_plan.authorized_shot_video_clocks, (clock,))
        self.assertFalse(clock.to_dict()['candidate_admitted'])
        self.assertFalse(clock.to_dict()['lip_sync_certified'])
        self.assertFalse(complete.to_dict()['raw_v2_semantics_independently_replayed'])

    def test_missing_external_context_never_implicitly_authorizes_normal_bundle(self):
        args, context, _, phase, clock = self.prepare()
        rows = self.refresh(args, context, phase, clock)
        with self.assertRaisesRegex(ValueError, 'external frozen'):
            complete_joint_group_bundle_metadata(**dict(args, observations=rows))
        bundle = complete_joint_group_bundle_metadata(**dict(args, observations=rows,
            shot_clock_context=context)).candidate_bundle
        with self.assertRaisesRegex(ValueError, 'external parent'):
            _validate_review_first_group_bundle(bundle, eligible_indices=args['domain'].eligible_indices,
                offsets=args['domain'].audio_offsets, timewarps=args['domain'].source_timewarps,
                common_start=0., phrase_times=bundle.phrase_times)

    def test_no_receipt_keeps_original_clock_rejection_before_any_decode(self):
        args, _, _, phase, _ = self.prepare()
        path = args['observations'].path
        with patch('crosscut.visual_score._VideoSampler') as sampler, self.assertRaisesRegex(
                ValueError, 'source clock mismatch'):
            refresh_joint_group_render_observations(plan=args['incumbent'].render_plan,
                timeline=args['incumbent'].visual_timeline, domain=args['domain'],
                path=(path[0], phase, path[2]), sync_scores=(1.,)*4)
        sampler.assert_not_called()

    def test_same_source_unrelated_shots_and_shared_audio_mapping_stay_exactly_unchanged(self):
        args, context, _, phase, clock = self.prepare(cells=8, mutable_first=3)
        rows = self.refresh(args, context, phase, clock)
        self.assertEqual(clock.request_source_index, rows.plan.segments[0].source_index)
        self.assertEqual(rows.plan.segments[0], context.incumbent_plan.segments[0])
        for i, old in enumerate(context.incumbent_plan.segments):
            if i not in clock.segment_indices:
                self.assertEqual(rows.plan.segments[i], old)
        self.assertEqual(context.source_domain.audio_offsets, args['domain'].audio_offsets)
        self.assertEqual(context.source_domain.source_timewarps, args['domain'].source_timewarps)
        self.assertEqual(rows.plan.reference_source, context.incumbent_plan.reference_source)
        self.assertEqual(rows.plan.master_audio_start, context.incumbent_plan.master_audio_start)
        self.assertTrue(all(slice is None for i, slice in enumerate(
            resolved_shot_video_clock_slices(plan=rows.plan, context=context)) if i not in clock.segment_indices))

    def test_nonidentity_source_map_and_full_clock_survive_json_and_pickle_roundtrip(self):
        args, context, _, phase, clock = self.prepare()
        rows = self.refresh(args, context, phase, clock)
        self.assertEqual((clock.eligible_source_index, clock.request_source_index), (2, 3))
        self.assertEqual(shot_video_clock_from_dict(clock.to_dict()), clock)
        decoded = render_plan_from_json(rows.plan.to_json())
        self.assertEqual(decoded, rows.plan)
        ctx, actual = pickle.loads(pickle.dumps((context, decoded)))
        self.assertEqual(validate_authorized_shot_video_clocks(plan=actual, context=ctx),
            {i: clock for i in clock.segment_indices})

    def test_untyped_parent_context_is_rejected_without_attribute_access_or_media_decode(self):
        args, context, _, phase, clock = self.prepare()
        rows = self.refresh(args, context, phase, clock)
        bundle = complete_joint_group_bundle_metadata(**dict(args, observations=rows,
            shot_clock_context=context)).candidate_bundle
        for bad in (None, {}, object()):
            with self.subTest(context=type(bad).__name__), self.assertRaisesRegex(ValueError, 'external frozen'):
                complete_joint_group_bundle_metadata(**dict(args, observations=rows, shot_clock_context=bad))
            with self.subTest(context=type(bad).__name__), self.assertRaisesRegex(ValueError, 'external parent'):
                _validate_review_first_group_bundle(bundle, eligible_indices=args['domain'].eligible_indices,
                    offsets=args['domain'].audio_offsets, timewarps=args['domain'].source_timewarps,
                    common_start=0., phrase_times=bundle.phrase_times, shot_clock_context=bad)

    def test_clock_schema_safety_flags_redundant_grid_and_integer_types_cannot_be_forged(self):
        _, _, _, _, clock = self.prepare()
        for key, value in (('contract', 'unknown_v2'), ('quality_approved', True),
                ('candidate_admitted', True), ('first_delta_native_frames', True),
                ('source_rate', float('nan'))):
            bad = dict(clock.to_dict(), **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                shot_video_clock_from_dict(bad)
        bad = deepcopy(clock.to_dict()); bad['native_indices'][15] += 1
        with self.assertRaisesRegex(ValueError, 'renderer sampling'):
            shot_video_clock_from_dict(bad)

    def test_external_alignment_incumbent_and_policy_are_not_taken_from_child_record(self):
        args, context, _, phase, clock = self.prepare()
        rows = self.refresh(args, context, phase, clock)
        for changes in ({'incumbent_plan_sha256': 'f'*64}, {'source_domain_sha256': 'e'*64},
                {'policy_sha256': 'd'*64}, {'maximum_absolute_offset_seconds': .15},
                {'eligible_source_index': 1}):
            forged = replace(clock, **changes)
            altered = replace(rows.plan, authorized_shot_video_clocks=(forged,))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'external'):
                validate_authorized_shot_video_clocks(plan=altered, context=context)

    def test_stale_pts_grid_overlapping_receipts_or_mutated_segment_are_rejected(self):
        args, context, _, phase, clock = self.prepare()
        rows = self.refresh(args, context, phase, clock)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            replace(rows.plan, authorized_shot_video_clocks=(clock, clock))
        parts = list(rows.plan.segments); parts[1] = replace(parts[1], source_start=parts[1].source_start+.01)
        with self.assertRaises(ValueError):
            replace(rows.plan, segments=tuple(parts))
        fake_grid = dict(args['domain'].clocks_by_request)
        fake_grid[clock.request_source_index] = replace(clock.native_clock, packet_grid_sha256='b'*64)
        with self.assertRaisesRegex(ValueError, 'native/PTS grid'):
            validate_shot_video_clock_bindings(rows.plan, frame_clocks=fake_grid)

    def test_fractional_5994_to_2997_whole_shot_phase_and_rate_have_one_exact_grid(self):
        plan, domain, states = state_fixture.JointStateFactoryTests().single_shot_fixture()
        base = candidate_joint_group_state(plan=plan, domain=domain,
            incumbent_state=states[1], eligible_source_index=2)
        context = FrozenShotVideoClockContext(plan, domain, NativePhaseSearchPolicy())
        hypothesis = NativePhaseHypothesis(base.key, context.policy.sha256, 2, 3)
        phase = phase_adjusted_group_state(state=base, domain=domain, hypothesis=hypothesis, policy=context.policy)
        clock = authorize_shot_video_clock(context=context, aligned_state=base, phase_state=phase, hypothesis=hypothesis)
        proposed, _ = assemble_single_group_source_camera_state(plan=plan, domain=domain,
            path=(states[0], phase, states[2]))
        typed = attach_authorized_shot_video_clocks(plan=proposed, context=context, clocks=(clock,))
        self.assertEqual(clock.native_clock.frame_rate, Fraction(60000, 1001))
        self.assertEqual(clock.native_indices, phase.native_frame_indices)
        self.assertEqual(clock.segment_indices, (1, 2))
        graph = build_filter_graph(typed, source_clocks=domain.clocks_by_request)
        self.assertIn('setpts=', graph)
        self.assertEqual(render_plan_from_json(typed.to_json()), typed)

    def test_already_shifted_baseline_cannot_accumulate_offset_by_reauthorizing(self):
        _, context, _, phase, _ = self.prepare()
        hypothesis = NativePhaseHypothesis(phase.key, context.policy.sha256, 1, 1)
        twice = phase_adjusted_group_state(state=phase, domain=context.source_domain,
            hypothesis=hypothesis, policy=context.policy)
        with self.assertRaisesRegex(ValueError, 'original unmodified'):
            authorize_shot_video_clock(context=context, aligned_state=phase,
                phase_state=twice, hypothesis=hypothesis)

    def test_interchange_preserves_authorized_exact_shot_clock_without_claiming_semantics(self):
        args, context, _, phase, clock = self.prepare()
        rows = self.refresh(args, context, phase, clock)
        payload = build_otio_payload(rows.plan, project_name='numeric phase fixture')
        children = payload['tracks']['children'][0]['children']
        clips = [child for child in children if child['OTIO_SCHEMA'] == 'Clip.2']
        metadata = clips[1]['metadata']['crosscut']
        self.assertEqual(metadata['authorized_shot_video_clock'], clock.to_dict())
        self.assertAlmostEqual(metadata['source_start_seconds'], phase.segments[0].source_start)
        self.assertNotIn('authorized_shot_video_clock', clips[0]['metadata']['crosscut'])

    def test_renderer_requires_the_actual_verified_grid_not_timestamp_fallback(self):
        args, context, _, phase, clock = self.prepare()
        rows = self.refresh(args, context, phase, clock)
        with self.assertRaisesRegex(ValueError, 'native/PTS grid'):
            build_filter_graph(rows.plan)
        wrong = dict(args['domain'].clocks_by_request)
        wrong[clock.request_source_index] = replace(clock.native_clock, packet_grid_sha256='d'*64)
        with self.assertRaisesRegex(ValueError, 'native/PTS grid'):
            build_filter_graph(rows.plan, source_clocks=wrong)

    def test_qa_replays_same_clock_but_does_not_promote_numeric_phase_to_semantic_approval(self):
        from crosscut.post_render_qa import _shot_video_clock_numeric_bindings
        args, context, _, phase, clock = self.prepare()
        rows = self.refresh(args, context, phase, clock)
        payload = rows.plan.to_json()
        frame_clock = dict(plan=payload, enabled=True,
            source_clocks={str(k): v.to_dict() for k, v in args['domain'].clocks_by_request.items()})
        report = _shot_video_clock_numeric_bindings(payload, frame_clock)
        self.assertEqual(report['declared_count'], 1)
        self.assertEqual(report['bound_count'], 1)
        self.assertTrue(report['numeric_clock_replayed'])
        self.assertTrue(report['semantic_audio_evidence_required'])
        self.assertFalse(report['choreography_certified'])
        self.assertFalse(report['quality_approved'])
        for changed in ({'enabled': False}, {'plan': context.incumbent_plan.to_json()}, {'source_clocks': {}}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _shot_video_clock_numeric_bindings(payload, dict(frame_clock, **changed))

    def test_non_scalar_audio_warp_is_original_baseline_not_a_new_shared_offset(self):
        from crosscut.timewarp import TimeWarpMap, TimeWarpControlPoint, TimeWarpConfig, TimeWarpResidualDiagnostics
        from crosscut.render import aligned_render_segment
        plan, domain, _ = state_fixture.JointStateFactoryTests().fixture()
        warp = TimeWarpMap(0., 10., 10.1,
            (TimeWarpControlPoint(0., .02), TimeWarpControlPoint(10., 10.04)), (),
            TimeWarpResidualDiagnostics(None, None, None, 0), TimeWarpConfig(), False, None)
        domain = replace(domain, source_timewarps=(None, None, warp, None))
        plan = replace(plan, segments=tuple(aligned_render_segment(source_index=s.source_index,
            reference_start=s.master_start, common_master_start=0., source_offset=0., duration=s.duration,
            timewarp=domain.source_timewarps[domain.eligible_index(s.source_index)]) for s in plan.segments))
        states = tuple(joint_state_from_render_plan(plan=plan, domain=domain, slot=i,
            first_output_frame=30*i, end_output_frame=30*(i+1),
            role=state_fixture.JointStateFactoryTests().fixture()[2][i].role) for i in range(4))
        base = candidate_joint_group_state(plan=plan, domain=domain, incumbent_state=states[1], eligible_source_index=2)
        context = FrozenShotVideoClockContext(plan, domain, NativePhaseSearchPolicy())
        hypothesis = NativePhaseHypothesis(base.key, context.policy.sha256, 1, 1)
        phase = phase_adjusted_group_state(state=base, domain=domain, hypothesis=hypothesis, policy=context.policy)
        clock = authorize_shot_video_clock(context=context, aligned_state=base, phase_state=phase, hypothesis=hypothesis)
        c = candidate_joint_group_state(plan=plan, domain=domain, incumbent_state=states[2], eligible_source_index=1)
        proposed, _ = assemble_joint_group_states(plan=plan, domain=domain, path=(states[0], phase, c, states[3]))
        actual = attach_authorized_shot_video_clocks(plan=proposed, context=context, clocks=(clock,))
        validate_authorized_shot_video_clocks(plan=actual, context=context)
        self.assertAlmostEqual(clock.aligned_source_start, warp.master_to_source(1.))
        self.assertAlmostEqual(clock.aligned_source_rate, 1.002)
        self.assertIs(domain.source_timewarps[2], warp)


if __name__ == '__main__':
    unittest.main()
