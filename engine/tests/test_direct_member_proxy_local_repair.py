from __future__ import annotations

import unittest
from pathlib import Path

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig,
    select_balanced_direct_member_sources,
)
from crosscut.pipeline import (
    _accumulate_direct_member_proxy_feedback_penalties,
    _direct_member_proxy_local_repair_anchor_penalties,
)
from crosscut.pose import FormationPersonDescriptor
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
    def test_rejected_path_changes_locally_instead_of_wholesale(self) -> None:
        spans = tuple((float(index), float(index + 1)) for index in range(6))
        common = {
            "sources": tuple(Path(f"{index}.mp4") for index in range(3)),
            "canonical_spans": spans,
            "audio_offsets": (0.0, 0.0, 0.0),
            "config": DirectMemberVisualConfig(beam_width=4_096),
            "descriptor_provider": (
                lambda source_index, _source, _time: _descriptor(
                    0.20 + source_index * 0.006
                )
            ),
        }
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
        anchors = _direct_member_proxy_local_repair_anchor_penalties(
            initial.selected_boundary_master_seconds,
            initial.source_indices,
            (0,),
            source_count=3,
        )
        local_retry_penalties = (
            _accumulate_direct_member_proxy_feedback_penalties(
                (measured_failure,),
                anchors,
                source_count=3,
                range_start=0.0,
                range_end=6.0,
                maximum_count=512,
            )
        )
        local_retry = select_balanced_direct_member_sources(
            **common,
            render_penalized_transition_edges=local_retry_penalties,
            forbidden_source_paths=(initial.source_indices,),
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
        self.assertEqual(local_unchanged, 4)

    def test_successful_boundaries_penalize_every_alternative_pair(self) -> None:
        boundaries = (1.0, 2.0, 3.0, 4.0)
        path = (0, 1, 2, 0, 1)

        anchors = _direct_member_proxy_local_repair_anchor_penalties(
            boundaries,
            path,
            (1,),
            source_count=3,
        )

        self.assertEqual(len(anchors), 15)
        self.assertNotIn(2.0, {item[0] for item in anchors})
        for boundary_index in (0, 2, 3):
            boundary_anchors = {
                (left, right)
                for boundary, left, right, _penalty in anchors
                if boundary == boundaries[boundary_index]
            }
            self.assertEqual(len(boundary_anchors), 5)
            self.assertNotIn(
                (path[boundary_index], path[boundary_index + 1]),
                boundary_anchors,
            )
            self.assertTrue(all(left != right for left, right in boundary_anchors))

    def test_capacity_never_emits_a_partial_boundary_domain(self) -> None:
        anchors = _direct_member_proxy_local_repair_anchor_penalties(
            (1.0, 2.0, 3.0, 4.0),
            (0, 1, 2, 0, 1),
            (1,),
            source_count=3,
            maximum_count=7,
        )

        self.assertEqual(len(anchors), 5)
        self.assertEqual({item[0] for item in anchors}, {4.0})

    def test_transient_anchor_budget_can_exceed_persistent_limit(self) -> None:
        boundaries = tuple(float(index) for index in range(1, 10))
        path = (0, 1, 2, 3, 4, 5, 6, 7, 0, 1)
        anchors = _direct_member_proxy_local_repair_anchor_penalties(
            boundaries,
            path,
            (2, 5, 8),
            source_count=8,
        )

        self.assertEqual(len(anchors), 330)
        merged = _accumulate_direct_member_proxy_feedback_penalties(
            (),
            anchors,
            source_count=8,
            range_start=0.0,
            range_end=10.0,
            maximum_count=512,
        )
        self.assertEqual(len(merged), 330)
        with self.assertRaisesRegex(ValueError, "penalty limit exceeded"):
            _accumulate_direct_member_proxy_feedback_penalties(
                (),
                anchors,
                source_count=8,
                range_start=0.0,
                range_end=10.0,
            )

    def test_zero_or_broad_failure_sets_disable_local_bias(self) -> None:
        common = {
            "boundary_master_seconds": (1.0, 2.0, 3.0, 4.0),
            "source_path": (0, 1, 2, 0, 1),
            "source_count": 3,
        }
        self.assertEqual(
            _direct_member_proxy_local_repair_anchor_penalties(
                **common,
                continuity_target_boundary_indices=(),
            ),
            (),
        )
        self.assertEqual(
            _direct_member_proxy_local_repair_anchor_penalties(
                **common,
                continuity_target_boundary_indices=(0, 1, 2, 3),
            ),
            (),
        )


if __name__ == "__main__":
    unittest.main()
