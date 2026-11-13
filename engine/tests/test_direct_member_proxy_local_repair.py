from __future__ import annotations

from pathlib import Path
import unittest

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig,
    select_balanced_direct_member_sources,
)
from crosscut.pose import FormationPersonDescriptor
from crosscut.pipeline import (
    _direct_member_proxy_reserved_render_camera_refinement_is_authorized,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor


def _descriptor(value: float) -> FrameVisualDescriptor:
    bbox = (value, 0.20, 0.22, 0.44)
    centroid = (value + 0.11, 0.42)
    embedding = (value, value * 0.5, 1.0 - value)
    return FrameVisualDescriptor(
        quality_score=1.0,
        composition_score=1.0,
        shot_scale=0.22,
        shot_type=ShotType.MEDIUM,
        face_count=1,
        face_detector_backend="yunet",
        structure_embedding=embedding,
        primary_face_bbox=bbox,
        pose_embedding=embedding,
        pose_confidence=1.0,
        detected_person_count=1,
        person_layout_embedding=embedding,
        formation_people=(
            FormationPersonDescriptor(
                bbox_xywh=bbox,
                centroid_xy=centroid,
                detection_confidence=1.0,
            ),
        ),
        pose_backend="synthetic_non_identity_pose",
        silhouette_embedding=embedding,
        silhouette_bbox_xywh=bbox,
        silhouette_centroid_xy=centroid,
        silhouette_area_ratio=0.22 * 0.44,
        silhouette_confidence=1.0,
        silhouette_backend="synthetic_silhouette",
    )


class DirectMemberProxyLocalRepairTests(unittest.TestCase):
    @staticmethod
    def _common() -> dict[str, object]:
        return {
            "sources": tuple(Path(f"{index}.mp4") for index in range(3)),
            "canonical_spans": tuple(
                (float(index), float(index + 1)) for index in range(6)
            ),
            "audio_offsets": (0.0, 0.0, 0.0),
            "config": DirectMemberVisualConfig(beam_width=4_096),
            "descriptor_provider": (
                lambda source_index, _source, _time: _descriptor(
                    0.20 + source_index * 0.006
                )
            ),
        }

    def test_rejected_path_changes_locally_instead_of_wholesale(self) -> None:
        common = self._common()
        initial = select_balanced_direct_member_sources(**common)
        measured_failure = (
            initial.selected_boundary_master_seconds[0],
            initial.source_indices[0],
            initial.source_indices[1],
            8.0,
        )
        plain_retry = select_balanced_direct_member_sources(
            **common,
            render_penalized_transition_edges=(measured_failure,),
            forbidden_source_paths=(initial.source_indices,),
        )
        local_retry = select_balanced_direct_member_sources(
            **common,
            render_penalized_transition_edges=(measured_failure,),
            forbidden_source_paths=(initial.source_indices,),
            decoded_proxy_preferred_source_path=initial.source_indices,
            decoded_proxy_repair_target_boundary_indices=(0,),
        )

        plain_unchanged = sum(
            left == right
            for left, right in zip(
                initial.source_indices,
                plain_retry.source_indices,
                strict=True,
            )
        )
        local_unchanged = sum(
            left == right
            for left, right in zip(
                initial.source_indices,
                local_retry.source_indices,
                strict=True,
            )
        )
        self.assertNotEqual(local_retry.source_indices, initial.source_indices)
        self.assertGreater(local_unchanged, plain_unchanged)
        diagnostics = local_retry.optimizer_search_diagnostics[
            "decoded_proxy_local_path_repair"
        ]
        self.assertTrue(diagnostics["enabled"])
        self.assertEqual(diagnostics["repair_window_span_indices"], [0, 1])
        self.assertEqual(diagnostics["protected_span_indices"], [2, 3, 4, 5])
        self.assertFalse(diagnostics["hard_locked"])
        self.assertFalse(diagnostics["visual_thresholds_relaxed"])

    def test_path_and_targets_must_be_supplied_together(self) -> None:
        common = self._common()
        with self.assertRaisesRegex(ValueError, "supplied together"):
            select_balanced_direct_member_sources(
                **common,
                decoded_proxy_repair_target_boundary_indices=(0,),
            )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            select_balanced_direct_member_sources(
                **common,
                decoded_proxy_preferred_source_path=(0, 1, 2, 0, 1, 2),
            )

    def test_final_reserved_render_prefers_geometry_only_near_pass(self) -> None:
        payload = {
            "direct_member_rendered_continuity_passed": False,
            "continuity_repair_target_boundary_indices": [2, 5, 8],
            "rendered_continuity": {
                "failure_boundary_indices": {
                    "rendered_boundary_scale_discontinuity": [2],
                    "rendered_effect_endpoint_position_discontinuity": [5, 8],
                    "rendered_effect_endpoint_scale_discontinuity": [8],
                }
            },
        }

        self.assertTrue(
            _direct_member_proxy_reserved_render_camera_refinement_is_authorized(
                payload,
                decoded_proxy_path_audition_count=2,
                decoded_proxy_render_count=3,
            )
        )
        self.assertFalse(
            _direct_member_proxy_reserved_render_camera_refinement_is_authorized(
                payload,
                decoded_proxy_path_audition_count=1,
                decoded_proxy_render_count=3,
            )
        )
        payload["rendered_continuity"]["failure_boundary_indices"][
            "rendered_effect_endpoint_pose_discontinuity"
        ] = [8]
        self.assertFalse(
            _direct_member_proxy_reserved_render_camera_refinement_is_authorized(
                payload,
                decoded_proxy_path_audition_count=2,
                decoded_proxy_render_count=3,
            )
        )


if __name__ == "__main__":
    unittest.main()
