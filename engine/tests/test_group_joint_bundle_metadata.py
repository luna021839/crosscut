from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from crosscut.group_joint_bundle_metadata import complete_joint_group_bundle_metadata
from crosscut.group_joint_render_observations import refresh_joint_group_render_observations
from crosscut.group_joint_state_factory import candidate_joint_group_state
from crosscut.pipeline import _review_first_group_bundle, _validate_review_first_group_bundle
from crosscut.editorial_transform_events import rendered_transform_event_records
from crosscut.section_director import SectionPlan, SectionDirective, SectionRole
from crosscut.visual_score import VisualTimelinePlan
from crosscut.render import EditorialTransformTrack, EditorialTransformKeyframe
from tests import test_group_joint_state_factory as factory_fixture
from tests.test_group_joint_render_observations import JointSampler
from tests.test_sparse_source_lattice import Sampler
from tests.test_visual_boundary_repair import _choice


class GeometryJointSampler(JointSampler):
    def frame_at(self, seconds):
        return np.full((90, 160, 3), self.source, np.uint8)


class JointCompleteMetadataTests(unittest.TestCase):
    def fixture(self, *, camera=False, no_op=False, single=False, cells=4, mutable_first=1):
        plan, domain, states = factory_fixture.JointStateFactoryTests().fixture()
        timeline = VisualTimelinePlan(tuple(_choice(i, i % 4) for i in range(cells)), 1.)
        if single:
            timeline = replace(timeline, choices=tuple(replace(c, source_index=timeline.choices[mutable_first].source_index)
                if i == mutable_first+1 else c for i, c in enumerate(timeline.choices)))
        section = SectionPlan((SectionDirective(0, SectionRole.BUILD, 0., float(cells), .5, .5, 1,
            4, 15., .4, .1, False, False),))
        request = SimpleNamespace(sources=plan.sources, reference_source=0, aspect_ratio=plan.aspect_ratio)
        baseline = _review_first_group_bundle(timeline=timeline, request=request,
            eligible_indices=domain.eligible_indices, offsets=domain.audio_offsets,
            timewarps=domain.source_timewarps, common_start=0., section_plan=section,
            phrase_times=tuple(float(i) for i in range(cells+1)), output=Path('/synthetic/not-rendered.mp4'),
            reporter=SimpleNamespace(log=lambda *a, **k: None), warnings=[])
        track = (EditorialTransformTrack((EditorialTransformKeyframe(0., 1.1),
            EditorialTransformKeyframe(1., 1.2))) if camera else None)
        # Bind states to the ACTUAL complete bundle's plan, including its
        # primitive transition reasons and output path; not another fixture.
        from crosscut.group_joint_source_camera import joint_state_from_render_plan
        first = mutable_first-1
        bounds = (((first*30, (first+1)*30), ((first+1)*30, (first+3)*30), ((first+3)*30, (first+4)*30))
            if single else tuple((i*30, (i+1)*30) for i in range(first, first+4)))
        role = states[0].role
        states = tuple(joint_state_from_render_plan(plan=baseline.render_plan, domain=domain,
            slot=i, first_output_frame=a, end_output_frame=b, role=role)
            for i, (a, b) in enumerate(bounds))
        if no_op:
            path = states
        elif single:
            path = (states[0], candidate_joint_group_state(plan=baseline.render_plan, domain=domain,
                incumbent_state=states[1], eligible_source_index=(states[1].eligible_source_index+1) % 4,
                camera_track=track), states[2])
        else:
            path = (states[0], candidate_joint_group_state(plan=baseline.render_plan, domain=domain,
                incumbent_state=states[1], eligible_source_index=states[2].eligible_source_index, camera_track=track),
                candidate_joint_group_state(plan=baseline.render_plan, domain=domain,
                    incumbent_state=states[2], eligible_source_index=states[1].eligible_source_index), states[3])
        Sampler.instances = []; Sampler.unreadable_source = None
        from tests.test_visual_boundary_repair import _descriptor
        with patch('crosscut.visual_score._VideoSampler', GeometryJointSampler), \
                patch('crosscut.visual_score.describe_frame', return_value=_descriptor(.5)):
            observations = refresh_joint_group_render_observations(plan=baseline.render_plan,
                timeline=baseline.visual_timeline, domain=domain, path=path, sync_scores=(1.,)*4)
        events = rendered_transform_event_records(observations.plan.segments, observations.plan.fps)
        motion = replace(baseline.editorial_motion_plan, segments=tuple(replace(row,
            source_index=choice.source_index, track=segment.editorial_transform)
            for row, choice, segment in zip(baseline.editorial_motion_plan.segments,
                observations.timeline.choices, observations.plan.segments, strict=True)),
            zoom_event_count=len(events['zoom']), pan_event_count=len(events['pan']),
            roll_event_count=len(events['roll']))
        return dict(incumbent=baseline, observations=observations, domain=domain,
            fresh_motion=motion, fresh_face_match=None, phrase_times=baseline.phrase_times)

    def validate(self, bundle, domain):
        _validate_review_first_group_bundle(bundle, eligible_indices=domain.eligible_indices,
            offsets=domain.audio_offsets, timewarps=domain.source_timewarps, common_start=0.,
            phrase_times=bundle.phrase_times)

    def test_complete_nonidentity_source_and_camera_metadata_survives_owned_consumer(self):
        args = self.fixture(camera=True)
        result = complete_joint_group_bundle_metadata(**args)
        candidate = result.candidate_bundle
        self.validate(candidate, args['domain'])
        self.assertEqual(tuple(s.source_index for s in candidate.render_plan.segments), (0, 3, 2, 5))
        self.assertEqual(tuple(s.source_index for s in candidate.editorial_motion_plan.segments), (0, 2, 1, 3))
        self.assertEqual(tuple(s['source_index'] for s in candidate.stage_edit_exact_shot_clock['shots']), (0, 2, 1, 3))
        self.assertIs(candidate.section_plan, args['incumbent'].section_plan)
        self.assertEqual(candidate.editorial_motion_plan.segments[1].track,
            candidate.render_plan.segments[1].editorial_transform)
        self.assertNotEqual(candidate.timeline_review, args['incumbent'].timeline_review)
        for flag in ('candidate_admitted', 'quality_approved', 'raw_v2_semantics_independently_replayed'):
            self.assertFalse(result.to_dict()[flag])
        self.assertEqual(args['incumbent'].candidate_lineage, ())

    def test_old_source_motion_cannot_be_installed_next_to_the_new_plan(self):
        args = self.fixture()
        with self.assertRaisesRegex(ValueError, 'motion source/camera'):
            complete_joint_group_bundle_metadata(**dict(args, fresh_motion=args['incumbent'].editorial_motion_plan))

    def test_one_actual_mutable_shot_is_not_split_into_two_semantic_source_choices(self):
        args = self.fixture(single=True, camera=True)
        result = complete_joint_group_bundle_metadata(**args)
        self.validate(result.candidate_bundle, args['domain'])
        self.assertEqual(len(args['observations'].path), 3)
        self.assertEqual(result.changed_segment_indices, (1, 2))
        self.assertEqual(tuple(s.source_index for s in result.candidate_bundle.render_plan.segments), (0, 3, 3, 5))

    def test_far_boundary_history_is_retained_and_cannot_be_globally_erased(self):
        args = self.fixture(cells=8, mutable_first=3)
        for field, value in (('accepted_boundary_indices', (0,)), ('natural_match_boundary_indices', (0,)),
                ('required_boundary_failures', ((0, ('historical_reason',), None),)),
                ('boundary_residual_attention_mismatches', ((0, .15, .2),))):
            baseline = replace(args['incumbent'], editorial_motion_plan=replace(
                args['incumbent'].editorial_motion_plan, **{field: value}))
            fresh = replace(args['fresh_motion'], **{field: value})
            current = dict(args, incumbent=baseline, fresh_motion=fresh)
            kept = complete_joint_group_bundle_metadata(**current).candidate_bundle
            self.assertEqual(getattr(kept.editorial_motion_plan, field), value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'unrelated'):
                complete_joint_group_bundle_metadata(**dict(current, fresh_motion=replace(fresh, **{field: ()})))

    def test_stale_incumbent_and_forged_changed_plan_inventory_are_rejected(self):
        args = self.fixture(); rows = args['observations']
        for changes in ({'incumbent_plan_identity': 'f'*64}, {'changed_segment_indices': (1,)},
                {'plan': replace(rows.plan, segments=(replace(rows.plan.segments[0], source_start=.01),
                    *rows.plan.segments[1:]))}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                complete_joint_group_bundle_metadata(**dict(args, observations=replace(rows, **changes)))

    def test_missing_descriptor_edges_and_unrefreshed_scaffold_cannot_be_complete(self):
        args = self.fixture(); rows = args['observations']
        for changes in ({'refreshed_segment_indices': (1, 2)},
                {'timeline': replace(rows.timeline, analysis_basis='UNREFRESHED_numeric_scaffold')},
                {'timeline': replace(rows.timeline, candidate_review_summaries=({'passed': True},))}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                complete_joint_group_bundle_metadata(**dict(args, observations=replace(rows, **changes)))

    def test_stale_source_local_endpoint_does_not_hide_behind_correct_source_index(self):
        args = self.fixture(); rows = args['observations']
        choices = list(rows.timeline.choices)
        choices[1] = replace(choices[1], source_local_start=1.01)
        bad = replace(rows, timeline=replace(rows.timeline, choices=tuple(choices)))
        with self.assertRaisesRegex(ValueError, 'source-local endpoints'):
            complete_joint_group_bundle_metadata(**dict(args, observations=bad))

    def test_unrelated_motion_history_cannot_be_erased_to_make_source_change_pass(self):
        args = self.fixture(); motion = args['fresh_motion']
        bad = replace(motion, segments=(replace(motion.segments[0], reasons=('erased',)), *motion.segments[1:]))
        with self.assertRaisesRegex(ValueError, 'unrelated motion'):
            complete_joint_group_bundle_metadata(**dict(args, fresh_motion=bad))

    def test_unknown_learned_or_final_authority_is_not_globally_reset(self):
        for field in ('dynamic_crop_application', 'cut_local_sync', 'splice_graph_authorization',
                'output_geometry_deferral', 'stage_edit_dynamic_crop_version'):
            args = self.fixture(); marker = 3 if field == 'stage_edit_dynamic_crop_version' else {'unknown': 'retain'}
            baseline = replace(args['incumbent'], **{field: marker})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'learned/clock/look authority'):
                complete_joint_group_bundle_metadata(**dict(args, incumbent=baseline))
            self.assertEqual(getattr(baseline, field), marker)

    def test_noop_returns_the_exact_incumbent_without_new_review_or_clock(self):
        args = self.fixture(no_op=True)
        result = complete_joint_group_bundle_metadata(**dict(args, fresh_motion=None))
        self.assertIs(result.candidate_bundle, args['incumbent'])
        self.assertEqual(result.changed_segment_indices, ())

    def test_noop_cannot_launder_a_different_descriptor_timeline(self):
        args = self.fixture(no_op=True); rows = args['observations']
        rows = replace(rows, timeline=replace(rows.timeline, analysis_basis='forged_new_descriptors'))
        with self.assertRaisesRegex(ValueError, 'no-op'):
            complete_joint_group_bundle_metadata(**dict(args, observations=rows))


if __name__ == '__main__':
    unittest.main()
