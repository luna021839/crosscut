from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from crosscut.automatic_joint_dissolve import register_automatic_member_dissolves
from crosscut.direct_member_editorial_motion import DirectMemberFaceSample
from crosscut.direct_member_target_face import FOCAL_FACE_BASIS
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.joint_dissolve_registration import fit_joint_dissolve_registration
from crosscut.post_render_qa import _boundary_render_plan_from_serialized
from crosscut.ive import AspectRatio
from crosscut.render import (
    RenderSegment, RenderTransition, RenderPlan, TransitionKind,
    EditorialTransformKeyframe, EditorialTransformTrack,
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
)


class AutomaticJointDissolveTests(unittest.TestCase):
    def inputs(self):
        geometry = direct_member_source_frame_geometry((2160, 3840), output_size=(1080, 1920))
        self.queries = []

        def sample(source, local):
            self.queries.append((source, local))
            box = (.42, .2, .16, .16) if source == 0 else (.43, .22, .14, .14)
            return DirectMemberFaceSample(
                box, 1, 'yunet', subject_bbox_xywh=(.30, .16, .40, .60),
                subject_confidence=.9, subject_geometry_basis='source_local_focal_body_track',
                alignment_geometry_basis_override=FOCAL_FACE_BASIS,
            )

        sources = tuple(SimpleNamespace(eligible_source_index=i,
            native_scene_cut_source_seconds=()) for i in range(2))
        return dict(
            render_segments=(RenderSegment(3, 10., 3., 0., source_duration=3.03),
                             RenderSegment(5, 20., 3., 3., source_duration=2.97)),
            transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE, 7,
                                          DIRECT_MEMBER_POSE_DISSOLVE_REASON),),
            visual_selection=SimpleNamespace(source_indices=(0, 1),
                source_frame_geometries=(geometry, geometry),
                span_validity=(SimpleNamespace(sources=sources),)*2),
            eligible_request_source_indices=(3, 5), common_master_start=100.,
            focal_provider=SimpleNamespace(sample=sample), fps=30.,
        )

    def plan(self, inputs, segments, transitions):
        return RenderPlan(sources=tuple(Path(f'/tmp/{n}.mp4') for n in range(6)),
            segments=segments, transitions=transitions, reference_source=0,
            aspect_ratio=AspectRatio.PORTRAIT, output=Path('/tmp/registered.mp4'),
            master_audio_start=inputs['common_master_start'], width=720, height=1280,
            fps=inputs['fps'], member_pose_dissolve_requested=True,
            member_pose_dissolve_effective=True, member_pose_dissolve_count=1)

    def test_automatic_registration_uses_compact_provider_and_exact_source_clocks(self):
        args = self.inputs()
        segments, effects, audit = register_automatic_member_dissolves(**args)
        self.assertEqual(audit['applied_count'], 1, audit)
        self.assertEqual({i for i, t in self.queries}, {0, 1})
        self.assertIn((0, 13.03), self.queries)
        self.assertIn((1, 20. + 6/30 * .99), self.queries)
        self.assertEqual(len(self.queries), len(set(self.queries)))
        self.assertEqual(audit['additional_render_count'], 0)
        self.assertFalse(audit['quality_approval'])
        self.assertFalse(audit['source_selection_changed'])
        for before, after in zip(args['render_segments'], segments):
            self.assertEqual((before.source_start, before.source_duration, before.master_start),
                             (after.source_start, after.source_duration, after.master_start))
        self.plan(args, segments, effects)

    def test_unchanged_proposal_reuses_solver_but_changed_evidence_does_not(self):
        args = self.inputs()
        cache = {}
        with patch('crosscut.automatic_joint_dissolve.fit_joint_dissolve_registration',
                   wraps=fit_joint_dissolve_registration) as solver:
            for _ in range(2):
                _, _, audit = register_automatic_member_dissolves(**args, registration_cache=cache)
            self.assertEqual(solver.call_count, 1)
            self.assertTrue(audit['boundaries'][0]['registration_cache_hit'])
            sample = args['focal_provider'].sample
            args['focal_provider'] = SimpleNamespace(sample=lambda s, t:
                replace(sample(s, t), bbox_xywh=(.41, .2, .16, .16)))
            register_automatic_member_dissolves(**args, registration_cache=cache)
            self.assertEqual(solver.call_count, 2)

    def test_native_cut_rejects_only_registration_without_changing_effect_or_path(self):
        args = self.inputs()
        args['visual_selection'].span_validity[0].sources[0].native_scene_cut_source_seconds=(13.1,)
        segments, effects, audit = register_automatic_member_dissolves(**args)
        self.assertEqual(segments, args['render_segments'])
        self.assertEqual(effects, args['transitions'])
        self.assertEqual(audit['boundaries'][0]['reason'], 'native_cut_in_overlap')
        self.assertEqual(self.queries, [])

    def test_missing_face_does_not_exclude_source_or_authorize_camera(self):
        args = self.inputs()
        args['focal_provider'] = SimpleNamespace(sample=lambda s, t: None)
        segments, effects, audit = register_automatic_member_dissolves(**args)
        self.assertEqual(segments, args['render_segments'])
        self.assertEqual(effects, args['transitions'])
        self.assertEqual(audit['boundaries'][0]['reason'], 'focal_overlap_not_measured')

    def test_existing_camera_is_not_overwritten(self):
        args = self.inputs()
        original = args['render_segments']
        args['render_segments'] = (replace(original[0], editorial_transform=EditorialTransformTrack(
            (EditorialTransformKeyframe(0., 1.), EditorialTransformKeyframe(3., 1.1)),
            reason='existing')), original[1])
        segments, _, audit = register_automatic_member_dissolves(**args)
        self.assertEqual(segments, args['render_segments'])
        self.assertEqual(audit['boundaries'][0]['reason'], 'existing_camera_ownership_preserved')
        self.assertEqual(self.queries, [])

    def test_full_plan_frame_allocation_is_passed_to_containment(self):
        args = self.inputs()
        args['render_segments'] = (RenderSegment(3, 10., 3.011, 0.),
                                  RenderSegment(5, 20., 3.011, 3.011))
        with patch('crosscut.automatic_joint_dissolve.review_joint_dissolve_containment',
                   return_value={'passed': True}) as reviewer:
            register_automatic_member_dissolves(**args)
        self.assertEqual(reviewer.call_args.kwargs['frame_counts'], (90, 91))

    def test_serialized_qa_preserves_both_cameras_and_pending_authority(self):
        args = self.inputs()
        segments, effects, audit = register_automatic_member_dissolves(**args)
        self.assertEqual(audit['applied_count'], 1)
        plan = self.plan(args, segments, effects)
        saved = json.loads(json.dumps(plan.to_json()))
        rebuilt = _boundary_render_plan_from_serialized(saved)
        self.assertEqual(rebuilt.segments, plan.segments)
        self.assertEqual(rebuilt.transitions, plan.transitions)
        for key, value in (('zoom', True), ('center_x', float('nan')),
                           ('time_seconds', '0'), ('zoom', None)):
            payload = deepcopy(saved)
            payload['segments'][0]['outgoing_overlap_transform']['keyframes'][0][key] = value
            with self.assertRaises(ValueError):
                _boundary_render_plan_from_serialized(payload)
        payload = deepcopy(saved)
        del payload['segments'][0]['outgoing_overlap_transform']
        with self.assertRaises(ValueError):
            _boundary_render_plan_from_serialized(payload)

    def test_domain_mismatch_is_rejected_before_observation(self):
        args = self.inputs()
        args['eligible_request_source_indices'] = (5, 3)
        with self.assertRaises(ValueError):
            register_automatic_member_dissolves(**args)
        self.assertEqual(self.queries, [])

    def test_disabled_effect_path_does_no_measurement(self):
        args = self.inputs()
        args['transitions'] = (RenderTransition(),)
        segments, effects, audit = register_automatic_member_dissolves(**args)
        self.assertEqual(segments, args['render_segments'])
        self.assertEqual(effects, args['transitions'])
        self.assertEqual(audit['applied_count'], 0)
        self.assertEqual(self.queries, [])

    def test_production_portrait_attachment_preserves_registered_tracks(self):
        from crosscut.pipeline import _attach_direct_member_portrait_fill_geometry
        args = self.inputs()
        segments, effects, audit = register_automatic_member_dissolves(**args)
        self.assertEqual(audit['applied_count'], 1)
        attached = _attach_direct_member_portrait_fill_geometry(
            visual_selection=args['visual_selection'],
            eligible_indices=args['eligible_request_source_indices'],
            eligible_source_aspect_ratios=(9/16, 9/16), render_segments=segments)
        self.assertEqual(attached, segments)
        self.plan(args, attached, effects)

    def test_endpoint_refinement_preserves_moving_overlap_ownership(self):
        from crosscut.direct_member_rendered_proxy import (
            DirectMemberRenderedProxyPreflight, refine_direct_member_camera_plan_from_rendered_proxy,
        )
        args = self.inputs()
        segments, effects, _ = register_automatic_member_dissolves(**args)
        plan = self.plan(args, segments, effects)
        preflight = DirectMemberRenderedProxyPreflight({'rendered_continuity': {
            'boundaries': [{'boundary_index': 0, 'transition_kind': 'face_dissolve'}]}})
        result, audit = refine_direct_member_camera_plan_from_rendered_proxy(
            plan, preflight, maximum_zoom_by_source_index={3: 1.5, 5: 1.5})
        self.assertIs(result, plan)
        self.assertFalse(audit['applied'])
        self.assertEqual(audit['boundaries'][0]['status'], 'joint_moving_camera_ownership_preserved')

    def test_authorized_demotion_removes_only_unused_overlap_handle(self):
        from crosscut.direct_member_rendered_proxy import (
            DirectMemberRenderedProxyPreflight, demote_failed_direct_member_pose_dissolves_from_rendered_proxy,
        )
        args = self.inputs()
        segments, effects, _ = register_automatic_member_dissolves(**args)
        plan = self.plan(args, segments, effects)
        plan = replace(plan, segments=(*segments, *(replace(s, master_start=s.master_start+6.) for s in segments)),
            transitions=(effects[0], RenderTransition(), effects[0]), member_pose_dissolve_count=2)
        preflight = DirectMemberRenderedProxyPreflight({'rendered_continuity': {
            'failure_boundary_indices': {'rendered_effect_endpoint_scale_discontinuity': [0]},
            'boundaries': [{'boundary_index': 0, 'transition_kind': 'face_dissolve',
                'attention_position_mismatch': .01, 'attention_scale_mismatch': .01, 'pose_mismatch': .01}]}})
        result, audit = demote_failed_direct_member_pose_dissolves_from_rendered_proxy(plan, preflight)
        self.assertTrue(audit['applied'], audit)
        self.assertTrue(audit['effect_proxy_revalidation_required'])
        self.assertIsNone(result.segments[0].outgoing_overlap_transform)
        self.assertEqual(result.segments[0].editorial_transform, plan.segments[0].editorial_transform)
        self.assertEqual(result.segments[2:], plan.segments[2:])
        self.assertEqual(result.member_pose_dissolve_count, 1)


if __name__ == '__main__':
    unittest.main()
