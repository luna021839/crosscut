from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.ive import AspectRatio
from crosscut.render import RenderSegment
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit import StageEdit, StageShot
from crosscut.stage_edit_pipeline_bridge import RetainedStageEditCandidate
from crosscut.stage_edit_visual_candidate_adapter import (
    materialize_stage_edit_visual_candidate,
)
from crosscut.visual_score import (
    PersistentVisualFeatureCacheInitializationError,
    PersistentVisualFeatureCacheRuntimeError,
)


class StageEditVisualCacheContractTests(unittest.TestCase):
    def test_required_direct_cache_failure_is_not_classified_as_a_candidate(
        self,
    ) -> None:
        sources = (
            Path("/tmp/stage-edit-cache-a.mp4").resolve(),
            Path("/tmp/stage-edit-cache-b.mp4").resolve(),
        )
        section_plan = SectionPlan(
            (
                SectionDirective(
                    section_index=0,
                    role=SectionRole.CHORUS_LIKE,
                    start_seconds=0.0,
                    end_seconds=2.0,
                    energy_score=0.8,
                    confidence=0.9,
                    phrase_count=1,
                    preferred_source_hold_beats=2,
                    target_source_switches_per_minute=20.0,
                    camera_motion_budget=0.6,
                    target_roll_events_per_minute=0.0,
                    establish_with_wide_shot=False,
                    protect_last_beat_from_extra_cut=False,
                ),
            )
        )
        edit = StageEdit(
            sources=sources,
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            width=1920,
            height=1080,
            fps=30.0,
            master_audio_start=0.0,
            section_plan=section_plan,
            shots=(
                StageShot(
                    shot_index=0,
                    section_index=0,
                    segment=RenderSegment(0, 0.0, 2.0, 0.0),
                    decision_id="strict-cache-test",
                ),
            ),
            boundaries=(),
            output=Path("/tmp/stage-edit-cache-output.mp4"),
        )
        retained = RetainedStageEditCandidate(
            rank=1,
            score=1.0,
            edit=edit,
            exact_clock_reprojected=True,
            timewarped_segment_count=0,
        )
        direct_root = Path("/mounted/direct-visual-cache")
        for error_type in (
            PersistentVisualFeatureCacheInitializationError,
            PersistentVisualFeatureCacheRuntimeError,
        ):
            with self.subTest(error_type=error_type.__name__):
                error = error_type("required persistent visual feature cache failed")
                with patch(
                    "crosscut.stage_edit_visual_candidate_adapter."
                    "build_prescribed_visual_timeline_candidate_set",
                    side_effect=error,
                ) as prescribed:
                    with self.assertRaises(error_type):
                        materialize_stage_edit_visual_candidate(
                            retained,
                            candidate_index=0,
                            review_rank=0,
                            sources=sources,
                            audio_offsets=(0.0, 0.0),
                            sync_scores=(0.9, 0.8),
                            scene_cut_times=((), ()),
                            source_timewarps=(None, None),
                            feature_cache_directory=Path(
                                "/legacy/source-cache"
                            ),
                            feature_derived_cache_directory=direct_root,
                            require_persistent_feature_cache=True,
                        )

                prescribed.assert_called_once()
                self.assertEqual(
                    prescribed.call_args.kwargs[
                        "feature_derived_cache_directory"
                    ],
                    direct_root,
                )
                self.assertIs(
                    prescribed.call_args.kwargs[
                        "require_persistent_feature_cache"
                    ],
                    True,
                )


if __name__ == "__main__":
    unittest.main()
