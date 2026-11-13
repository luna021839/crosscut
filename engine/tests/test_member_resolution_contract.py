"""Selector proposals must survive the same dense camera resolution policy."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from crosscut.direct_member_editorial_motion import (
    DirectMemberFaceSample, build_direct_member_editorial_motion_plan,
)
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, _boundary_mismatch, _boundary_sample_with_focal_geometry,
)
from crosscut.pipeline import _direct_member_editorial_motion_config_from_visual_config
from crosscut.render import RenderSegment, RenderTransition
from tests.test_focal_selector_geometry import observation, descriptor, BODY


class MemberResolutionContractTests(unittest.TestCase):
    def test_hd_endpoint_proposal_reaches_dense_output_frame_validation(self):
        boxes = ((.4, .1, .1, .1), (.394, .094, .112, .112))
        geometry = direct_member_source_frame_geometry((1080, 1920))
        samples = []
        for source, box in enumerate(boxes):
            focal = replace(observation(source=source), face_bbox_xywh=box, face_scale=.035*box[3]/.1)
            samples.append(_boundary_sample_with_focal_geometry(
                master_seconds=1., source_local_seconds=1., source_index=source,
                descriptor=descriptor(), provider=SimpleNamespace(focal_geometry_at=lambda *a, f=focal: f)))
        visual_config = DirectMemberVisualConfig()
        mismatch = _boundary_mismatch(*samples, visual_config,
            outgoing_source_index=0, incoming_source_index=1,
            outgoing_source_geometry=geometry, incoming_source_geometry=geometry,
            continuous_official_portrait_fancam_contract=True)
        self.assertIsNotNone(mismatch.transform_repair)
        self.assertGreater(mismatch.transform_repair.upscale_ratio, 1.0)
        selection = SimpleNamespace(source_indices=(0, 1), source_frame_geometries=(geometry, geometry),
            boundaries=(SimpleNamespace(boundary_index=0, outgoing_eligible_source_index=0,
                incoming_eligible_source_index=1, outgoing_sample=samples[0], incoming_sample=samples[1],
                mismatch=mismatch),))
        calls = []
        def provider(source, seconds):
            calls.append((source, seconds))
            return DirectMemberFaceSample(boxes[source], 1, 'yunet', BODY, .95,
                'face_associated_formation_person_box', mismatch.transform_repair.alignment_geometry_basis)
        plan = build_direct_member_editorial_motion_plan(visual_selection=selection,
            render_segments=(RenderSegment(0, 0., 2., 0.), RenderSegment(1, 2., 2., 2.)),
            transitions=(RenderTransition(),), common_master_start=0., geometry_provider=provider,
            fps=30., config=_direct_member_editorial_motion_config_from_visual_config(visual_config))
        self.assertEqual(plan.accepted_boundary_indices, (0,))
        tracks = [plan.track_for(source) for source in (0, 1) if plan.track_for(source) is not None]
        self.assertTrue(tracks)
        self.assertGreater(len(calls), 20, 'endpoint proposal cannot replace dense observations')
        self.assertTrue(all(point.zoom <= 1.15 + 1e-9 for track in tracks for point in track.keyframes))


if __name__ == '__main__':
    unittest.main()
