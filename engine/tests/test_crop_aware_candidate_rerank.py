from __future__ import annotations

from types import SimpleNamespace
import unittest

from crosscut.crop_aware_candidate_rerank import (
    evaluate_crop_aware_visual_plan,
    rerank_materialized_visual_candidates_for_safe_crops,
)
from crosscut.rendered_montage_grammar_review import (
    RenderedMontageGrammarConfig,
)
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.timeline import ShotType
from crosscut.visual_score import (
    ProfessionalMontageSelectionPolicy,
    VisualTimelineChoice,
    VisualTimelinePlan,
)


def _policy() -> ProfessionalMontageSelectionPolicy:
    return ProfessionalMontageSelectionPolicy(
        highlight_minimum_closeup_fraction=0.25,
        highlight_preferred_closeup_fraction=0.34,
        highlight_minimum_nonwide_fraction=0.49,
        highlight_preferred_nonwide_fraction=0.66,
        build_minimum_closeup_fraction=0.12,
        build_preferred_closeup_fraction=0.25,
        build_minimum_nonwide_fraction=0.15,
        build_preferred_nonwide_fraction=0.31,
        bridge_minimum_closeup_fraction=0.27,
        bridge_preferred_closeup_fraction=0.43,
        bridge_minimum_nonwide_fraction=0.42,
        bridge_preferred_nonwide_fraction=0.58,
        highlight_minimum_source_switches_per_minute=18.6,
        highlight_preferred_source_switches_per_minute=23.5,
        minimum_early_wide_anchor_fraction=1.0 / 3.0,
        minimum_phrase_entry_wide_anchor_fraction=0.28,
    )


def _section_plan() -> SectionPlan:
    return SectionPlan(
        sections=(
            SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=0.0,
                end_seconds=4.0,
                energy_score=0.9,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=20.0,
                camera_motion_budget=0.8,
                target_roll_events_per_minute=0.0,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _visual_plan(*, crop_safe: bool) -> VisualTimelinePlan:
    face_height = 0.214
    face_width = 0.16
    face = (
        0.5 - face_width / 2.0,
        0.42 - face_height / 2.0,
        face_width,
        face_height,
    )
    return VisualTimelinePlan(
        choices=tuple(
            VisualTimelineChoice(
                segment_index=index,
                source_index=index % 2,
                canonical_master_start=float(index),
                canonical_master_end=float(index + 1),
                source_local_start=float(index),
                source_local_end=float(index + 1),
                quality_score=0.9,
                composition_score=0.9,
                boundary_score=0.9,
                shot_scale=0.2,
                shot_type=ShotType.WIDE,
                shot_id=f"source-{index % 2}:shot-{index}",
                entry_shot_type=ShotType.WIDE,
                exit_shot_type=ShotType.WIDE,
                face_samples_xywh=(face, face, face),
                face_detector_backends=("yunet", "yunet", "yunet"),
                face_match_candidate_failure=(
                    None if crop_safe else "ambiguous_multiple_faces"
                ),
                detected_person_counts=(1, 1, 1),
                formation_person_counts=(1, 1, 1),
            )
            for index in range(4)
        ),
        optimizer_score=1.0,
        selected_candidate_index=0,
    )


class CropAwareCandidateRerankTests(unittest.TestCase):
    def test_safe_crop_projection_upgrades_original_wide_highlight(self) -> None:
        evaluation = evaluate_crop_aware_visual_plan(
            _visual_plan(crop_safe=True),
            candidate_index=7,
            cadence_profile="reference",
            input_rank=0,
            source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
            section_plan=_section_plan(),
            policy=_policy(),
            rendered_config=RenderedMontageGrammarConfig(),
        )

        self.assertTrue(evaluation.all_highlight_metadata_floors_passed)
        self.assertEqual(evaluation.safe_crop_applied_segment_count, 4)
        self.assertEqual(evaluation.crop_upgraded_sample_count, 12)
        section = evaluation.sections[0]
        self.assertEqual(section["pre_crop_nonwide_duration_fraction"], 0.0)
        self.assertEqual(section["projected_nonwide_duration_fraction"], 1.0)
        self.assertFalse(evaluation.to_dict()["decoded_output_authority"])

    def test_crop_qualified_candidate_precedes_input_metadata_rank(self) -> None:
        raw = SimpleNamespace(
            candidate_index=0,
            cadence_profile="balanced",
            visual_plan=_visual_plan(crop_safe=False),
        )
        cropped = SimpleNamespace(
            candidate_index=1,
            cadence_profile="reference",
            visual_plan=_visual_plan(crop_safe=True),
        )

        result = rerank_materialized_visual_candidates_for_safe_crops(
            (raw, cropped),
            source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
            section_plan=_section_plan(),
            policy=_policy(),
            rendered_config=RenderedMontageGrammarConfig(),
            maximum_candidates=1,
        )

        self.assertEqual(result.selected_candidate_indices, (1,))
        self.assertEqual(result.ordered_candidates[0].candidate_index, 1)
        self.assertTrue(result.evaluations[0].all_highlight_metadata_floors_passed)
        self.assertFalse(result.to_dict()["decoded_output_authority"])


if __name__ == "__main__":
    unittest.main()
