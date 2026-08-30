from __future__ import annotations

import unittest
from pathlib import Path

from crosscut.ive import AspectRatio
from crosscut.joint_decoded_montage_feedback import (
    joint_decoded_montage_observations_from_bound_review,
    merge_joint_decoded_montage_observations,
)
from crosscut.render import RenderPlan, RenderSegment
from crosscut.timeline import TimelineCandidate, TimelineChoice


def _choice(index: int, source_index: int) -> TimelineChoice:
    candidate = TimelineCandidate(
        segment_index=index,
        source_id=str(source_index),
        start_seconds=5.0 + index,
        end_seconds=6.0 + index,
        quality_score=0.9,
        sync_score=0.9,
        composition_score=0.8,
        boundary_score=0.8,
        candidate_id=f"candidate:{index}:{source_index}",
        shot_id=f"shot:{index}:{source_index}",
    )
    return TimelineChoice(
        candidate=candidate,
        emission_score=1.0,
        transition_score=0.0,
        cumulative_score=float(index + 1),
        used_group_wide_fallback=False,
    )


def _plan() -> RenderPlan:
    return RenderPlan(
        sources=(Path("a.mp4"), Path("b.mp4")),
        segments=(
            RenderSegment(0, 5.0, 1.0, 0.0),
            RenderSegment(1, 6.0, 1.0, 1.0),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path("out.mp4"),
        master_audio_start=5.0,
        width=640,
        height=360,
        fps=30.0,
        transitions=(),
    )


def _review(signature: str) -> dict[str, object]:
    return {
        "segment_interior_geometry": {
            "render_timeline_binding_sha256": signature,
        },
        "rendered_montage_grammar_review": {
            "render_plan_binding": {
                "passed": True,
                "bound_segment_count": 2,
                "fps": 30.0,
            },
            "rendered_segment_interior_geometry": {
                "available": True,
                "source_render_timeline_binding_sha256": signature,
                "segment_records": [
                    {
                        "segment_index": 0,
                        "source_index": 0,
                        "sample_count": 3,
                        "sampled_proxy_shot_classes": [
                            "close",
                            "wide",
                            "close",
                        ],
                    },
                    {
                        "segment_index": 1,
                        "source_index": 1,
                        "sample_count": 3,
                        "sampled_proxy_shot_classes": [
                            "medium",
                            "unknown",
                            "medium",
                        ],
                    },
                ],
            },
        },
    }


class JointDecodedMontageFeedbackBridgeTests(unittest.TestCase):
    def test_extracts_complete_bound_unshifted_observations(self) -> None:
        signature = "a" * 64
        observations = joint_decoded_montage_observations_from_bound_review(
            timeline_choices=(_choice(0, 0), _choice(1, 1)),
            render_plan=_plan(),
            review_payload=_review(signature),
            render_variant_signature=signature,
        )

        self.assertEqual(len(observations), 2)
        self.assertEqual(
            observations[0].sampled_proxy_shot_classes,
            ("close", "wide", "close"),
        )
        self.assertEqual(observations[0].candidate_clock.start_seconds, 5.0)
        self.assertEqual(observations[1].candidate_clock.source_id, "1")

    def test_rejects_montage_sha_or_source_binding_drift(self) -> None:
        signature = "b" * 64
        review = _review(signature)
        geometry = review["rendered_montage_grammar_review"]
        assert isinstance(geometry, dict)
        interior = geometry["rendered_segment_interior_geometry"]
        assert isinstance(interior, dict)
        interior["source_render_timeline_binding_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "render binding changed"):
            joint_decoded_montage_observations_from_bound_review(
                timeline_choices=(_choice(0, 0), _choice(1, 1)),
                render_plan=_plan(),
                review_payload=review,
                render_variant_signature=signature,
            )

        review = _review(signature)
        montage = review["rendered_montage_grammar_review"]
        assert isinstance(montage, dict)
        interior = montage["rendered_segment_interior_geometry"]
        assert isinstance(interior, dict)
        records = interior["segment_records"]
        assert isinstance(records, list)
        assert isinstance(records[1], dict)
        records[1]["source_index"] = 0
        with self.assertRaisesRegex(ValueError, "source binding changed"):
            joint_decoded_montage_observations_from_bound_review(
                timeline_choices=(_choice(0, 0), _choice(1, 1)),
                render_plan=_plan(),
                review_payload=review,
                render_variant_signature=signature,
            )

    def test_merge_deduplicates_exact_replay_and_rejects_conflict(self) -> None:
        first = joint_decoded_montage_observations_from_bound_review(
            timeline_choices=(_choice(0, 0), _choice(1, 1)),
            render_plan=_plan(),
            review_payload=_review("d" * 64),
            render_variant_signature="d" * 64,
        )
        merged = merge_joint_decoded_montage_observations(first, first)
        self.assertEqual(merged, first)

        conflicting_review = _review("d" * 64)
        montage = conflicting_review["rendered_montage_grammar_review"]
        assert isinstance(montage, dict)
        interior = montage["rendered_segment_interior_geometry"]
        assert isinstance(interior, dict)
        records = interior["segment_records"]
        assert isinstance(records, list)
        assert isinstance(records[0], dict)
        records[0]["sampled_proxy_shot_classes"] = [
            "wide",
            "wide",
            "wide",
        ]
        conflicting = joint_decoded_montage_observations_from_bound_review(
            timeline_choices=(_choice(0, 0), _choice(1, 1)),
            render_plan=_plan(),
            review_payload=conflicting_review,
            render_variant_signature="d" * 64,
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            merge_joint_decoded_montage_observations(first, conflicting)


if __name__ == "__main__":
    unittest.main()
