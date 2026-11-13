from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.member_camera_worker import MemberCameraProviderSpec, validate_member_camera_clock, _identity
from crosscut.ive import AspectRatio
from crosscut.render import (
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)


def _synthetic_member_camera_ipc_bundle():
    """Pickle a production-shaped local->global MEMBER camera proposal."""
    sources = tuple(Path(f'/synthetic/source-{index}.mp4') for index in range(3))
    baseline_segments = (
        RenderSegment(2, 0., 1., 0.),
        RenderSegment(0, 1., 1., 1.),
    )
    baseline = RenderPlan(
        sources,
        baseline_segments,
        0,
        AspectRatio.LANDSCAPE,
        Path('/synthetic/output.mp4'),
        member_pose_dissolve_requested=True,
        member_pose_dissolve_effective=False,
        member_pose_dissolve_count=0,
    )
    camera = EditorialTransformTrack(
        (
            EditorialTransformKeyframe(0., 1.10, .48, .50),
            EditorialTransformKeyframe(1., 1.10, .48, .50),
        ),
        reason='synthetic_verified_member_camera',
    )
    transition = RenderTransition(
        kind=TransitionKind.FACE_DISSOLVE,
        frame_count=3,
        reason=DIRECT_MEMBER_POSE_DISSOLVE_REASON,
        transform_overlap_authority=MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
    )
    proposal = replace(
        baseline,
        segments=tuple(replace(segment, editorial_transform=camera) for segment in baseline_segments),
        transitions=(transition,),
        member_pose_dissolve_effective=True,
        member_pose_dissolve_count=1,
    )
    bundle = (
        proposal,
        MappingProxyType({'motion': 'camera'}),
        MappingProxyType({'look': 'bounded'}),
        MappingProxyType({'applied_count': 1}),
        MappingProxyType({0: ('left-reference', 'right-reference')}),
    )
    return baseline, bundle


class MemberCameraWorkerContractTests(unittest.TestCase):
    def test_nontrivial_camera_dissolve_bundle_survives_ipc_and_controller_promotion(self):
        from crosscut.first_result_controller import FirstResultController
        from crosscut.optional_native_worker import run_isolated_python_call
        from crosscut.review_first_runtime import ReviewFirstRuntime

        baseline, proposal = run_isolated_python_call(
            'tests.test_member_camera_worker',
            '_synthetic_member_camera_ipc_bundle',
            {},
            deadline=monotonic() + 15.,
        )
        self.assertIsInstance(proposal[3], MappingProxyType)
        self.assertEqual(tuple(segment.source_index for segment in proposal[0].segments), (2, 0))
        self.assertEqual(proposal[0].transitions[0].kind, TransitionKind.FACE_DISSOLVE)
        self.assertAlmostEqual(proposal[0].segments[0].editorial_transform.keyframes[0].zoom, 1.10)
        validate_member_camera_clock(proposal, baseline)

        runtime = ReviewFirstRuntime(enabled=True)
        controller = FirstResultController(
            runtime,
            sources=baseline.sources,
            source_durations=(10., 10., 10.),
            master_start=0.,
            duration=baseline.duration,
            allowed_source_indices=(0, 2),
        )
        primitive = (baseline, None, None, {}, {})
        controller.install_baseline(primitive, plan=baseline, authority='verified_member_source_primitive')
        accepted = controller.run_optional(
            'member_camera_and_dissolve',
            lambda: proposal,
            validate=lambda value: (
                validate_member_camera_clock(value, baseline),
                controller.validate_plan(value[0]),
            ),
        )
        self.assertIsNotNone(accepted)
        controller.promote(accepted, plan=accepted[0], authority='registered_member_camera_proposal')
        self.assertIs(controller.incumbent, accepted)
        self.assertEqual(controller.incumbent[0].transitions[0].kind, TransitionKind.FACE_DISSOLVE)
        self.assertEqual(controller.to_dict()['attempts'][0]['status'], 'validated_proposal')

    def test_rejected_camera_proposal_keeps_complete_incumbent(self):
        from crosscut.first_result_controller import FirstResultController
        from crosscut.review_first_runtime import ReviewFirstRuntime

        baseline, proposal = _synthetic_member_camera_ipc_bundle()
        runtime = ReviewFirstRuntime(enabled=True)
        controller = FirstResultController(
            runtime,
            sources=baseline.sources,
            source_durations=(10., 10., 10.),
            master_start=0.,
            duration=baseline.duration,
            allowed_source_indices=(0, 2),
        )
        primitive = (baseline, 'primitive-motion', 'primitive-look', {}, {})
        controller.install_baseline(primitive, plan=baseline, authority='verified_member_source_primitive')
        changed = replace(
            proposal[0],
            segments=(replace(proposal[0].segments[0], source_index=1), proposal[0].segments[1]),
        )
        rejected = (changed, *proposal[1:])
        result = controller.run_optional(
            'member_camera_and_dissolve',
            lambda: rejected,
            validate=lambda value: validate_member_camera_clock(value, baseline),
        )
        self.assertIsNone(result)
        self.assertIs(controller.incumbent, primitive)
        self.assertEqual(controller.to_dict()['attempts'][0]['status'], 'rejected')

    def test_proposal_keeps_source_path_but_can_change_camera(self):
        plan = RenderPlan((Path('/synthetic/a'), Path('/synthetic/b')),
            (RenderSegment(0, 0., 1., 0.), RenderSegment(1, 1., 1., 1.)),
            0, AspectRatio.LANDSCAPE, Path('/synthetic/output'))
        camera = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.1), EditorialTransformKeyframe(1., 1.2)))
        adjusted = replace(plan, segments=(replace(plan.segments[0], editorial_transform=camera), plan.segments[1]))
        validate_member_camera_clock((adjusted, None, None, {}, {}), plan)
        for changed in (replace(plan, reference_source=1), replace(plan, fps=25.),
                        replace(plan, segments=(replace(plan.segments[0], source_index=1), plan.segments[1])),
                        replace(plan, segments=(replace(plan.segments[0], source_start=.1), plan.segments[1])),
                        replace(plan, segments=(replace(plan.segments[0], source_duration=1.01), plan.segments[1]))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_member_camera_clock((changed, None, None, {}, {}), plan)
        with self.assertRaises(ValueError):
            validate_member_camera_clock((plan,), plan)

    def test_provider_spec_preserves_frozen_authority_and_rejects_file_change(self):
        from crosscut.source_first_focal_geometry import SourceFirstFocalGeometryProvider
        with TemporaryDirectory() as directory:
            files = tuple(Path(directory) / name for name in ('source', 'body', 'pose'))
            for path in files:
                path.touch()
            provider = object.__new__(SourceFirstFocalGeometryProvider)
            provider.sources = (files[0],)
            provider._identities = (_identity(files[0]),)
            provider.encoder = SimpleNamespace(model_path=files[1])
            provider.pose = SimpleNamespace(model_path=files[2])
            provider.body_execution_device = 'cpu'
            provider.faces = SimpleNamespace(_frozen_focal_banks={0: None})
            provider.bodies = SimpleNamespace(_references={0: ((.1, .2),)})
            provider.observation_namespace = 'source-model-bound-test'
            spec = MemberCameraProviderSpec.from_provider(provider, {'feature_cache_directory': None})
            spec.validate_files()
            provider.faces._frozen_focal_banks.clear()
            provider.bodies._references.clear()
            self.assertEqual(spec.face_banks, ((0, None),))
            self.assertEqual(spec.body_references, ((0, ((.1, .2),)),))
            child = SimpleNamespace(observation_namespace=spec.observation_namespace,
                faces=SimpleNamespace(_focal_references={}), bodies=SimpleNamespace(), close=lambda: None)
            with patch('crosscut.source_first_focal_geometry.SourceFirstFocalGeometryProvider', return_value=child) as create, patch(
                    'crosscut.source_first_unresolved_memo.install_source_first_unresolved_memo'):
                self.assertIs(spec.open(), child)
                self.assertEqual(child.faces._frozen_focal_banks, {0: None})
                self.assertEqual(child.bodies._references, {0: ((.1, .2),)})
                self.assertEqual(create.call_args.kwargs['body_model_path'], files[1])
            # A new inode must not be mistaken for the already bound source/model.
            files[2].unlink()
            files[2].mkdir()
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                spec.validate_files()
