from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut import visual_score
from crosscut.timeline import BoundarySideStability, ShotType, TimelineCandidate
from crosscut.timeline_no_good import (
    TimelineNoGoodSet,
    build_timeline_no_good_edge_key,
)
from crosscut.visual_score import (
    VisualDiversityConfig,
    VisualTimelineCandidateSet,
    VisualTimelineOptimizerWorkspace,
    regenerate_visual_timeline_candidate_set,
)


def _candidate(segment_index: int, source_index: int) -> TimelineCandidate:
    start = segment_index * 0.5
    embedding = (0.20, 0.40, 0.60, 0.80)
    stability = BoundarySideStability(
        sample_span_seconds=0.10,
        scale_delta=0.0,
        structure_mismatch=0.0,
        pose_mismatch=0.0,
        shot_type_mismatch=0.0,
    )
    bbox = (0.35, 0.18, 0.30, 0.70)
    return TimelineCandidate(
        segment_index=segment_index,
        source_id=str(source_index),
        start_seconds=start,
        end_seconds=start + 0.5,
        quality_score=0.96 - source_index * 0.01,
        sync_score=0.98,
        composition_score=0.90,
        boundary_score=0.92,
        shot_scale=0.50,
        entry_shot_scale=0.50,
        exit_shot_scale=0.50,
        shot_type=ShotType.MEDIUM,
        entry_shot_type=ShotType.MEDIUM,
        exit_shot_type=ShotType.MEDIUM,
        entry_boundary_side_stability=stability,
        exit_boundary_side_stability=stability,
        pose_embedding=embedding,
        entry_pose_embedding=embedding,
        exit_pose_embedding=embedding,
        structure_embedding=embedding,
        entry_structure_embedding=embedding,
        exit_structure_embedding=embedding,
        silhouette_embedding=embedding,
        entry_silhouette_embedding=embedding,
        exit_silhouette_embedding=embedding,
        person_layout_embedding=embedding,
        entry_person_layout_embedding=embedding,
        exit_person_layout_embedding=embedding,
        face_identity_embedding=(1.0, 0.0),
        entry_face_identity_embedding=(1.0, 0.0),
        exit_face_identity_embedding=(1.0, 0.0),
        color_embedding=embedding,
        entry_color_embedding=embedding,
        exit_color_embedding=embedding,
        focus_bbox_xywh=bbox,
        entry_focus_bbox_xywh=bbox,
        exit_focus_bbox_xywh=bbox,
        subject_bbox_xywh=bbox,
        entry_subject_bbox_xywh=bbox,
        exit_subject_bbox_xywh=bbox,
        subject_centroid_xy=(0.50, 0.53),
        entry_subject_centroid_xy=(0.50, 0.53),
        exit_subject_centroid_xy=(0.50, 0.53),
        detected_person_count=1,
        entry_detected_person_count=1,
        exit_detected_person_count=1,
        motion=(0.0, 0.0),
        entry_motion=(0.0, 0.0),
        exit_motion=(0.0, 0.0),
        candidate_id=f"candidate-{segment_index}-{source_index}",
        shot_id=f"{source_index}:shot-{segment_index}",
    )


def _lattice() -> tuple[tuple[TimelineCandidate, ...], ...]:
    return tuple(
        tuple(_candidate(segment_index, source_index) for source_index in range(2))
        for segment_index in range(8)
    )


def _candidate_set() -> VisualTimelineCandidateSet:
    lattice = _lattice()
    sources = (Path("source-0.mp4"), Path("source-1.mp4"))
    audio_offsets = (0.0, 0.0)
    timewarps = (None, None)
    scene_cut_times = ((), ())
    source_clock_basis = "synthetic_master_clock"
    input_identity = visual_score._visual_optimizer_input_identity(
        lattice,
        sources=sources,
        audio_offsets=audio_offsets,
        source_timewarps=timewarps,
        source_clock_basis=source_clock_basis,
        scene_cut_times=scene_cut_times,
        maximum_results=8,
    )
    workspace = VisualTimelineOptimizerWorkspace(
        candidates_by_segment=lattice,
        input_identity=input_identity,
        maximum_results=8,
    )
    results, profiles, _audit = visual_score._select_optimizer_workspace_candidates(
        workspace,
        forbidden_boundary_edges=TimelineNoGoodSet(),
    )
    records, selected_index = visual_score._review_visual_timeline_candidates(
        results,
        profiles,
        available_source_count=2,
    )
    return VisualTimelineCandidateSet(
        candidates=records,
        selected_candidate_index=selected_index,
        sources=sources,
        audio_offsets=audio_offsets,
        source_timewarps=timewarps,
        source_clock_basis=source_clock_basis,
        requested_maximum_results=8,
        scene_cut_times=scene_cut_times,
        optimizer_workspace=workspace,
        optimizer_input_identity=input_identity,
    )


class VisualTimelineOptimizerWorkspaceTests(unittest.TestCase):
    def test_regeneration_is_deterministic_and_excludes_no_good_edge(self) -> None:
        candidate_set = _candidate_set()
        selected = candidate_set.selected.timeline_result.candidates
        forbidden = TimelineNoGoodSet(
            (build_timeline_no_good_edge_key(2, selected[2], selected[3]),)
        )

        first = regenerate_visual_timeline_candidate_set(
            candidate_set,
            forbidden_boundary_edges=forbidden,
        )
        second = regenerate_visual_timeline_candidate_set(
            candidate_set,
            forbidden_boundary_edges=forbidden,
        )

        self.assertEqual(
            tuple(record.path_signature for record in first.candidates),
            tuple(record.path_signature for record in second.candidates),
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        for record in first.candidates:
            actual_edges = TimelineNoGoodSet.from_plan_boundaries(
                record.timeline_result
            )
            self.assertNotIn(forbidden.edges[0], actual_edges)

    def test_regeneration_never_opens_sampler_or_calls_descriptor(self) -> None:
        candidate_set = _candidate_set()
        selected = candidate_set.selected.timeline_result.candidates
        forbidden = TimelineNoGoodSet(
            (build_timeline_no_good_edge_key(1, selected[1], selected[2]),)
        )

        with (
            patch.object(
                visual_score,
                "_VideoSampler",
                side_effect=AssertionError("sampler opened during DP regeneration"),
            ) as sampler,
            patch.object(
                visual_score,
                "_describe_sampler_frame",
                side_effect=AssertionError("descriptor requested during regeneration"),
            ) as sampler_descriptor,
            patch.object(
                visual_score,
                "describe_frame",
                side_effect=AssertionError("frame described during regeneration"),
            ) as frame_descriptor,
        ):
            regenerated = regenerate_visual_timeline_candidate_set(
                candidate_set,
                forbidden_boundary_edges=forbidden,
            )

        self.assertTrue(regenerated.candidates)
        sampler.assert_not_called()
        sampler_descriptor.assert_not_called()
        frame_descriptor.assert_not_called()

    def test_workspace_is_transient_in_repr_serialization_and_equality(self) -> None:
        candidate_set = _candidate_set()
        self.assertNotIn("optimizer_workspace", repr(candidate_set))
        self.assertNotIn("optimizer_workspace", candidate_set.to_dict())
        self.assertEqual(
            candidate_set.to_dict()["optimizer_input_identity"],
            candidate_set.optimizer_input_identity,
        )
        assert candidate_set.optimizer_workspace is not None
        cloned_workspace = replace(candidate_set.optimizer_workspace)
        self.assertEqual(
            candidate_set,
            replace(candidate_set, optimizer_workspace=cloned_workspace),
        )

    def test_foreign_source_order_workspace_is_rejected(self) -> None:
        candidate_set = _candidate_set()
        assert candidate_set.optimizer_workspace is not None
        foreign_sources = tuple(reversed(candidate_set.sources))
        foreign_identity = visual_score._visual_optimizer_input_identity(
            candidate_set.optimizer_workspace.candidates_by_segment,
            sources=foreign_sources,
            audio_offsets=candidate_set.audio_offsets,
            source_timewarps=candidate_set.source_timewarps,
            source_clock_basis=candidate_set.source_clock_basis,
            scene_cut_times=candidate_set.scene_cut_times,
            maximum_results=8,
        )
        foreign_workspace = replace(
            candidate_set.optimizer_workspace,
            input_identity=foreign_identity,
        )

        with self.assertRaisesRegex(ValueError, "workspace identity"):
            replace(candidate_set, optimizer_workspace=foreign_workspace)

    def test_foreign_descriptor_lattice_is_rejected_even_with_same_sources(
        self,
    ) -> None:
        candidate_set = _candidate_set()
        assert candidate_set.optimizer_workspace is not None
        original_workspace = candidate_set.optimizer_workspace
        first = original_workspace.candidates_by_segment[0][0]
        mutations = (
            replace(first, shot_id="0:shot-foreign"),
            replace(first, candidate_id="candidate-foreign"),
            replace(first, source_id="7"),
            replace(first, quality_score=0.25),
            replace(first, pose_embedding=(0.99, 0.01)),
            replace(first, entry_shot_scale=0.33),
            replace(
                first,
                entry_boundary_side_stability=replace(
                    first.entry_boundary_side_stability,
                    structure_mismatch=0.25,
                ),
            ),
        )
        for mutated in mutations:
            with self.subTest(mutated=mutated):
                foreign_lattice = (
                    (mutated, *original_workspace.candidates_by_segment[0][1:]),
                    *original_workspace.candidates_by_segment[1:],
                )
                foreign_identity = visual_score._visual_optimizer_input_identity(
                    foreign_lattice,
                    sources=candidate_set.sources,
                    audio_offsets=candidate_set.audio_offsets,
                    source_timewarps=candidate_set.source_timewarps,
                    source_clock_basis=candidate_set.source_clock_basis,
                    scene_cut_times=candidate_set.scene_cut_times,
                    maximum_results=8,
                )
                self.assertNotEqual(
                    foreign_identity,
                    original_workspace.input_identity,
                )
                stale_workspace = replace(
                    original_workspace,
                    candidates_by_segment=foreign_lattice,
                )
                with self.assertRaisesRegex(ValueError, "workspace identity"):
                    replace(candidate_set, optimizer_workspace=stale_workspace)
                foreign_workspace = replace(
                    original_workspace,
                    candidates_by_segment=foreign_lattice,
                    input_identity=foreign_identity,
                )
                with self.assertRaisesRegex(ValueError, "workspace identity"):
                    replace(candidate_set, optimizer_workspace=foreign_workspace)

        timing_lattice = (
            tuple(
                replace(item, start_seconds=0.125, end_seconds=0.625)
                for item in original_workspace.candidates_by_segment[0]
            ),
            *original_workspace.candidates_by_segment[1:],
        )
        timing_identity = visual_score._visual_optimizer_input_identity(
            timing_lattice,
            sources=candidate_set.sources,
            audio_offsets=candidate_set.audio_offsets,
            source_timewarps=candidate_set.source_timewarps,
            source_clock_basis=candidate_set.source_clock_basis,
            scene_cut_times=candidate_set.scene_cut_times,
            maximum_results=8,
        )
        self.assertNotEqual(
            timing_identity,
            original_workspace.input_identity,
        )
        with self.assertRaisesRegex(ValueError, "workspace identity"):
            replace(
                candidate_set,
                optimizer_workspace=replace(
                    original_workspace,
                    candidates_by_segment=timing_lattice,
                ),
            )

    def test_foreign_clock_and_scene_metadata_reject_stale_workspace(self) -> None:
        candidate_set = _candidate_set()
        mutations = (
            {"audio_offsets": (0.125, 0.0)},
            {"scene_cut_times": ((0.25,), ())},
            {"source_clock_basis": "foreign_master_clock"},
        )

        for changes in mutations:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "workspace identity"):
                    replace(candidate_set, **changes)

    def test_reachability_excludes_candidates_with_only_forbidden_incoming_edges(
        self,
    ) -> None:
        lattice = _lattice()[:3]
        forbidden = TimelineNoGoodSet(
            tuple(
                build_timeline_no_good_edge_key(
                    0,
                    lattice[0][source_index],
                    lattice[1][0],
                )
                for source_index in range(2)
            )
        )
        config = visual_score._safe_cadence_profiles(
            VisualDiversityConfig()
        )[0].config

        reachable = visual_score._complete_path_reachable_candidate_keys(
            lattice,
            timeline_config=config,
            forbidden_boundary_edges=forbidden,
        )

        self.assertNotIn((1, "0"), reachable)
        self.assertIn((1, "1"), reachable)


if __name__ == "__main__":
    unittest.main()
