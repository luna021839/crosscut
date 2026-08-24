from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from crosscut.render import RenderSegment, RenderTransition
from crosscut.request import AspectRatio
from crosscut.rhythm import Beat, OnsetEnvelope, RhythmAnalysis, TempoEstimate
from crosscut.section_director import build_section_plan
from crosscut.stage_edit import StageBoundary, StageEdit, StageShot


def _sections():
    rhythm = RhythmAnalysis(
        100,
        9.0,
        OnsetEnvelope((0.2,) * 90, 100, 20, 10),
        TempoEstimate(120.0, 0.5, 0.0, 0.8),
        tuple(Beat(i, i * 0.5, 0.2) for i in range(19)),
        (),
    )
    return build_section_plan(rhythm, start_seconds=10.0, end_seconds=19.0)


def _edit() -> StageEdit:
    sources = (Path("/tmp/source-a.mp4"), Path("/tmp/source-b.mp4"))
    segments = (
        RenderSegment(0, 10.0, 3.0, 0.0),
        RenderSegment(1, 13.0, 3.0, 3.0),
        RenderSegment(0, 16.0, 3.0, 6.0),
    )
    shots = tuple(
        StageShot(i, 0, segment, f"decision-{i}", 0.8)
        for i, segment in enumerate(segments)
    )
    boundaries = tuple(
        StageBoundary(i, RenderTransition(reason="reviewed_hard_cut"), "passed")
        for i in range(2)
    )
    return StageEdit(
        sources=sources,
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        width=1920,
        height=1080,
        fps=30.0,
        master_audio_start=10.0,
        section_plan=_sections(),
        shots=shots,
        boundaries=boundaries,
        output=Path("/tmp/stage-edit-output.mp4"),
    )


class StageEditTests(unittest.TestCase):
    def test_to_render_plan_is_frame_and_clock_exact(self) -> None:
        edit = _edit()
        plan = edit.to_render_plan()
        self.assertEqual(plan.output_frame_count, 270)
        self.assertEqual(plan.master_audio_start, 10.0)
        self.assertEqual(plan.segments, tuple(item.segment for item in edit.shots))
        self.assertEqual(len(plan.transitions), 2)

    def test_partial_patch_preserves_outside_shot_objects(self) -> None:
        edit = _edit()
        replacement = replace(
            edit.shots[1],
            segment=RenderSegment(0, 13.0, 3.0, 3.0),
            decision_id="repaired-decision",
        )
        patched = edit.replace_window(
            start_shot=1,
            end_shot_exclusive=2,
            replacement_shots=(replacement,),
            replacement_boundaries=(
                StageBoundary(0, RenderTransition(reason="new-left"), "passed"),
                StageBoundary(1, RenderTransition(reason="new-right"), "passed"),
            ),
            reason="proxy_boundary_jump",
            score_before=71.0,
            score_after=83.0,
        )
        self.assertIs(patched.shots[0], edit.shots[0])
        self.assertIs(patched.shots[2], edit.shots[2])
        self.assertEqual(patched.shots[1].decision_id, "repaired-decision")
        self.assertEqual(len(patched.lineage), 1)
        self.assertEqual(patched.to_render_plan().output_frame_count, 270)

    def test_partial_patch_rejects_non_improvement(self) -> None:
        edit = _edit()
        with self.assertRaisesRegex(ValueError, "improve"):
            edit.replace_window(
                start_shot=1,
                end_shot_exclusive=2,
                replacement_shots=(edit.shots[1],),
                replacement_boundaries=(edit.boundaries[0], edit.boundaries[1]),
                reason="no improvement",
                score_before=80.0,
                score_after=79.0,
            )

    def test_reviewer_history_is_append_only(self) -> None:
        edit = _edit()
        reviewed = edit.with_reviewer_result({"score": 91.2, "passed": True})
        self.assertEqual(len(edit.reviewer_history), 0)
        self.assertEqual(reviewed.reviewer_history[0]["score"], 91.2)
        self.assertEqual(reviewed.to_dict()["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
