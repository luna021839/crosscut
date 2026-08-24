from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter import (
    DynamicCropHeadPredictionV2,
)
from crosscut.stage_edit_learned_section_director import (
    CONTRACT,
    LearnedSectionDirectorError,
    adapt_section_plan_from_stage_edit_heads,
    build_learned_section_plan_from_checkpoint,
)
from crosscut.stage_edit_semimarkov_beam import StageEditHeadFrame


def _section_plan() -> SectionPlan:
    return SectionPlan(
        (
            SectionDirective(
                section_index=0,
                role=SectionRole.VERSE_LIKE,
                start_seconds=0.0,
                end_seconds=4.0,
                energy_score=0.4,
                confidence=0.8,
                phrase_count=1,
                preferred_source_hold_beats=4,
                target_source_switches_per_minute=14.0,
                camera_motion_budget=0.3,
                target_roll_events_per_minute=0.1,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
            SectionDirective(
                section_index=1,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=4.0,
                end_seconds=8.0,
                energy_score=0.9,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=4,
                target_source_switches_per_minute=14.0,
                camera_motion_budget=0.3,
                target_roll_events_per_minute=0.1,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _multi_section_plan() -> SectionPlan:
    specs = (
        (SectionRole.INTRO, 2.0, 0.3, 1, True),
        (SectionRole.CHORUS_LIKE, 3.0, 1.0, 1, True),
        (SectionRole.VERSE_LIKE, 2.0, 0.4, 1, False),
        (SectionRole.CHORUS_LIKE, 4.0, 0.9, 2, True),
        (SectionRole.BUILD, 2.0, 0.7, 1, False),
        (SectionRole.CHORUS_LIKE, 6.0, 0.7, 2, True),
        (SectionRole.BRIDGE_LIKE, 2.0, 0.2, 1, False),
    )
    directives = []
    cursor = 0.0
    for index, (role, duration, energy, phrase_count, establish) in enumerate(specs):
        directives.append(
            SectionDirective(
                section_index=index,
                role=role,
                start_seconds=cursor,
                end_seconds=cursor + duration,
                energy_score=energy,
                confidence=0.9,
                phrase_count=phrase_count,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=17.0,
                camera_motion_budget=0.5,
                target_roll_events_per_minute=0.1,
                establish_with_wide_shot=establish,
                protect_last_beat_from_extra_cut=False,
            )
        )
        cursor += duration
    return SectionPlan(tuple(directives))


def _crop(*, motion_logit: float, roll: float) -> DynamicCropHeadPredictionV2:
    return DynamicCropHeadPredictionV2(
        normalized_times=tuple(index / 7.0 for index in range(8)),
        values=tuple((1.02, 0.5, 0.5, roll) for _ in range(8)),
        presence_logits=(4.0,) * 8,
        motion_logit=motion_logit,
    )


def _frame(
    timestamp: float,
    *,
    cut_logit: float,
    motion_logit: float,
    roll: float,
    shot_sizes: tuple[str, str],
) -> StageEditHeadFrame:
    return StageEditHeadFrame(
        master_time_seconds=timestamp,
        cut_logit=cut_logit,
        source_logits=(3.0, 1.0),
        duration_logits=(0.0, 0.0, 0.0, 0.0),
        transition_logits=(1.0, 0.0, 0.0, 0.0),
        crop=(1.0, 0.5, 0.5, 0.0),
        available_source_mask=(True, True),
        hard_source_gate_mask=(True, True),
        source_shot_sizes=shot_sizes,
        dynamic_crop_v2=_crop(motion_logit=motion_logit, roll=roll),
    )


class LearnedSectionDirectorTests(unittest.TestCase):
    def test_learned_heads_change_section_budgets_before_semimarkov(self) -> None:
        frames = (
            _frame(
                0.0,
                cut_logit=-3.0,
                motion_logit=-3.0,
                roll=0.0,
                shot_sizes=("medium", "wide"),
            ),
            _frame(
                2.0,
                cut_logit=-2.5,
                motion_logit=-2.5,
                roll=0.0,
                shot_sizes=("medium", "wide"),
            ),
            _frame(
                4.0,
                cut_logit=3.0,
                motion_logit=3.0,
                roll=1.4,
                shot_sizes=("wide", "medium"),
            ),
            _frame(
                6.0,
                cut_logit=2.5,
                motion_logit=2.5,
                roll=1.2,
                shot_sizes=("wide", "medium"),
            ),
        )
        result = adapt_section_plan_from_stage_edit_heads(
            _section_plan(),
            frames,
            chorus_wide_anchor_fraction=1.0 / 3.0,
            cut_probability_threshold=0.5,
            cut_threshold_freeze_sha256="a" * 64,
            checkpoint_id="checkpoint-v3",
            training_report_path="/checkpoint/training-report.json",
            training_report_sha256="b" * 64,
        )

        calm, active = result.section_plan.sections
        self.assertLess(
            calm.target_source_switches_per_minute,
            active.target_source_switches_per_minute,
        )
        self.assertGreater(
            calm.preferred_source_hold_beats,
            active.preferred_source_hold_beats,
        )
        self.assertLess(calm.camera_motion_budget, active.camera_motion_budget)
        self.assertLess(
            calm.target_roll_events_per_minute,
            active.target_roll_events_per_minute,
        )
        self.assertFalse(calm.establish_with_wide_shot)
        self.assertTrue(active.establish_with_wide_shot)
        self.assertTrue(calm.protect_last_beat_from_extra_cut)
        self.assertFalse(active.protect_last_beat_from_extra_cut)
        self.assertIn(CONTRACT, result.section_plan.analysis_basis)
        self.assertEqual(len(result.evidence), 2)
        self.assertTrue(
            all(math.isfinite(item.learned_source_switches_per_minute) for item in result.evidence)
        )

    def test_one_third_chorus_allocation_selects_structurally_major_section(
        self,
    ) -> None:
        plan = _multi_section_plan()
        frames = tuple(
            _frame(
                section.start_seconds + 0.1,
                cut_logit=0.0,
                motion_logit=0.0,
                roll=0.0,
                # The structurally major chorus (index 5) deliberately lacks
                # wide candidate support. Verse/build/bridge deliberately have
                # it, proving support cannot promote those roles or inflate the
                # exact chorus quota.
                shot_sizes=(
                    ("medium", "wide")
                    if section.section_index == 5
                    else ("wide", "medium")
                ),
            )
            for section in plan.sections
        )

        result = adapt_section_plan_from_stage_edit_heads(
            plan,
            frames,
            chorus_wide_anchor_fraction=1.0 / 3.0,
            cut_probability_threshold=0.5,
            cut_threshold_freeze_sha256="a" * 64,
            checkpoint_id="checkpoint-v3",
            training_report_path="/checkpoint/training-report.json",
            training_report_sha256="b" * 64,
        )
        reordered = adapt_section_plan_from_stage_edit_heads(
            plan,
            tuple(reversed(frames)),
            chorus_wide_anchor_fraction=1.0 / 3.0,
            cut_probability_threshold=0.5,
            cut_threshold_freeze_sha256="a" * 64,
            checkpoint_id="checkpoint-v3",
            training_report_path="/checkpoint/training-report.json",
            training_report_sha256="b" * 64,
        )

        self.assertEqual(result.schema_version, 2)
        self.assertTrue(result.contract.endswith("_v2"))
        self.assertEqual(result.chorus_section_count, 3)
        self.assertEqual(result.chorus_wide_anchor_allocated_count, 1)
        self.assertEqual(
            result.chorus_wide_anchor_allocated_section_indices,
            (5,),
        )
        self.assertEqual(
            [item.establish_with_wide_shot for item in result.section_plan.sections],
            [True, False, False, False, False, True, False],
        )
        self.assertFalse(result.section_plan.sections[2].establish_with_wide_shot)
        self.assertFalse(result.section_plan.sections[4].establish_with_wide_shot)
        self.assertFalse(result.section_plan.sections[6].establish_with_wide_shot)
        self.assertEqual(
            result.chorus_wide_anchor_allocated_section_indices,
            reordered.chorus_wide_anchor_allocated_section_indices,
        )
        self.assertEqual(
            result.section_plan.to_dict(),
            reordered.section_plan.to_dict(),
        )
        self.assertEqual(
            [item.to_dict() for item in result.evidence],
            [item.to_dict() for item in reordered.evidence],
        )
        self.assertEqual(result.evidence[2].learned_top_source_wide_support, 1.0)
        self.assertFalse(result.evidence[2].chorus_wide_anchor_selected)
        self.assertEqual(result.evidence[5].learned_top_source_wide_support, 0.0)
        self.assertTrue(result.evidence[5].chorus_wide_anchor_selected)
        serialized = result.to_dict()["chorus_wide_anchor_allocation"]
        self.assertEqual(serialized["requested_fraction"], 1.0 / 3.0)
        self.assertEqual(serialized["allocated_count"], 1)
        self.assertEqual(serialized["allocated_section_indices"], [5])

    def test_chorus_wide_anchor_fraction_must_be_finite_and_in_unit_range(
        self,
    ) -> None:
        frames = (
            _frame(
                0.0,
                cut_logit=0.0,
                motion_logit=0.0,
                roll=0.0,
                shot_sizes=("medium", "wide"),
            ),
            _frame(
                4.0,
                cut_logit=0.0,
                motion_logit=0.0,
                roll=0.0,
                shot_sizes=("wide", "medium"),
            ),
        )
        for fraction in (0.0, -0.1, 1.1, math.nan, math.inf, True, None):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(
                    LearnedSectionDirectorError,
                    "chorus wide-anchor fraction",
                ):
                    adapt_section_plan_from_stage_edit_heads(
                        _section_plan(),
                        frames,
                        chorus_wide_anchor_fraction=fraction,
                        cut_probability_threshold=0.5,
                        cut_threshold_freeze_sha256="a" * 64,
                        checkpoint_id="checkpoint-v3",
                        training_report_path="/checkpoint/training-report.json",
                        training_report_sha256="b" * 64,
                    )

    def test_missing_section_head_coverage_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            LearnedSectionDirectorError,
            "does not cover every acoustic section",
        ):
            adapt_section_plan_from_stage_edit_heads(
                _section_plan(),
                (
                    _frame(
                        0.0,
                        cut_logit=0.0,
                        motion_logit=0.0,
                        roll=0.0,
                        shot_sizes=("medium", "wide"),
                    ),
                ),
                chorus_wide_anchor_fraction=1.0 / 3.0,
                cut_probability_threshold=0.5,
                cut_threshold_freeze_sha256="a" * 64,
                checkpoint_id="checkpoint-v3",
                training_report_path="/checkpoint/training-report.json",
                training_report_sha256="b" * 64,
            )

    def test_checkpoint_adapter_binds_validation_only_threshold_freeze(self) -> None:
        freeze = {
            "selected_probability_threshold": 0.41,
            "selected_checkpoint_id": "checkpoint-v3",
            "heldout_labels_consumed": False,
            "heldout_probabilities_consumed": False,
            "selected_before_heldout_forward": True,
        }
        freeze["freeze_sha256"] = hashlib.sha256(
            json.dumps(
                freeze,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        frames = (
            _frame(
                0.0,
                cut_logit=-1.0,
                motion_logit=-1.0,
                roll=0.0,
                shot_sizes=("medium", "wide"),
            ),
            _frame(
                4.0,
                cut_logit=1.0,
                motion_logit=1.0,
                roll=0.5,
                shot_sizes=("wide", "medium"),
            ),
        )
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = Path(raw)
            report = checkpoint / "training-report.json"
            report.write_text(
                json.dumps(
                    {
                        "checkpoint": {"checkpoint_id": "checkpoint-v3"},
                        "cut_threshold_freeze": freeze,
                    }
                ),
                encoding="utf-8",
            )
            result = build_learned_section_plan_from_checkpoint(
                _section_plan(),
                frames,
                checkpoint_directory=checkpoint,
                chorus_wide_anchor_fraction=1.0 / 3.0,
            )
            self.assertEqual(result.cut_probability_threshold, 0.41)
            self.assertEqual(result.checkpoint_id, "checkpoint-v3")

            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["cut_threshold_freeze"]["selected_probability_threshold"] = 0.9
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                LearnedSectionDirectorError,
                "digest is invalid",
            ):
                build_learned_section_plan_from_checkpoint(
                    _section_plan(),
                    frames,
                    checkpoint_directory=checkpoint,
                    chorus_wide_anchor_fraction=1.0 / 3.0,
                )


if __name__ == "__main__":
    unittest.main()
