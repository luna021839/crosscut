from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.member_camera_worker import MemberCameraProviderSpec, validate_member_camera_clock, _identity
from crosscut.ive import AspectRatio
from crosscut.render import RenderPlan, RenderSegment, EditorialTransformTrack, EditorialTransformKeyframe


class MemberCameraWorkerContractTests(unittest.TestCase):
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
