from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.ive import AspectRatio
from crosscut.section_director import (
    SectionDirective,
    SectionPlan,
    SectionRole,
)
from crosscut.source_derivative_diversity import SourceDerivativeFamilyMetadata
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter import (
    DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT,
    DynamicCropHeadPredictionV2,
)
from crosscut.stage_edit_semimarkov_beam import (
    StageEditCheckpointNotReadyError,
    StageEditDecodeInputError,
    StageEditHeadFrame,
    StageEditNoValidTimelineError,
    StageEditSectionConstraint,
    StageEditSemiMarkovConfig,
    StageEditSourceContinuity,
    StageEditSourceMetadata,
    decode_stage_edit_semimarkov_beam,
    stage_edit_boundary_continuity_score,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_DECISION_HISTORY_OWNER,
    StageEditReadinessReport,
)


def _ready_report() -> StageEditReadinessReport:
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=True,
        status="production_checkpoint_ready",
        reason="test production evidence",
        mlx_version="test",
        checkpoint_path="/tmp/stageedit-production",
        training_steps=100,
    )


def _not_ready_report() -> StageEditReadinessReport:
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=False,
        status="production_training_report_missing",
        reason="no production report",
        mlx_version="test",
        checkpoint_path="/tmp/stageedit-random",
        training_steps=1,
    )


def _section(
    index: int,
    start: float,
    end: float,
    *,
    rate: float = 20.0,
) -> SectionDirective:
    return SectionDirective(
        section_index=index,
        role=SectionRole.CHORUS_LIKE,
        start_seconds=start,
        end_seconds=end,
        energy_score=0.8,
        confidence=0.9,
        phrase_count=1,
        preferred_source_hold_beats=2,
        target_source_switches_per_minute=rate,
        camera_motion_budget=0.7,
        target_roll_events_per_minute=0.2,
        establish_with_wide_shot=False,
        protect_last_beat_from_extra_cut=False,
    )


def _plan(duration: int = 6) -> SectionPlan:
    return SectionPlan((_section(0, 0.0, float(duration)),))


def _sources(count: int, duration: float) -> tuple[Path, ...]:
    return tuple(Path(f"/tmp/stageedit-source-{index}.mp4") for index in range(count))


def _metadata(
    count: int,
    duration: float,
    *,
    internal_cuts: dict[int, tuple[float, ...]] | None = None,
) -> tuple[StageEditSourceMetadata, ...]:
    sizes = ("wide", "medium", "closeup", "full", "extreme_closeup")
    internal_cuts = internal_cuts or {}
    return tuple(
        StageEditSourceMetadata(
            source_index=index,
            source_id=f"source-{index}",
            source_duration_seconds=duration,
            internal_cut_times_seconds=internal_cuts.get(index, ()),
            maximum_occupancy_fraction=0.60,
            quality_log_prior=-0.02 * index,
            default_shot_size=sizes[index % len(sizes)],
        )
        for index in range(count)
    )


def _frames(
    duration: int,
    source_count: int,
    *,
    gate_overrides: dict[int, tuple[bool, ...]] | None = None,
) -> tuple[StageEditHeadFrame, ...]:
    gate_overrides = gate_overrides or {}
    duration_bucket = DEFAULT_STAGE_EDIT_SCHEMA.duration_bucket(1.0)
    sizes = ("wide", "medium", "closeup", "full", "extreme_closeup")
    result = []
    for time_index in range(duration):
        preferred = time_index % source_count
        source_logits = tuple(
            2.5 if index == preferred else 0.4 - 0.05 * index
            for index in range(source_count)
        )
        duration_logits = tuple(
            2.2 if index == duration_bucket else -0.2
            for index in range(DEFAULT_STAGE_EDIT_SCHEMA.duration_bucket_count)
        )
        result.append(
            StageEditHeadFrame(
                master_time_seconds=float(time_index),
                cut_logit=2.0 if time_index else 0.0,
                source_logits=source_logits,
                duration_logits=duration_logits,
                transition_logits=(2.0, 0.8, 0.2, -0.2),
                crop=(1.04 + 0.005 * (time_index % 2), 0.5, 0.5, 0.0),
                available_source_mask=(True,) * source_count,
                hard_source_gate_mask=gate_overrides.get(
                    time_index, (True,) * source_count
                ),
                source_shot_sizes=tuple(
                    sizes[index % len(sizes)] for index in range(source_count)
                ),
                candidate_id=f"beat-{time_index}",
            )
        )
    return tuple(result)


def _decode(
    *,
    duration: int = 6,
    source_count: int = 4,
    frames: tuple[StageEditHeadFrame, ...] | None = None,
    metadata: tuple[StageEditSourceMetadata, ...] | None = None,
    plan: SectionPlan | None = None,
    config: StageEditSemiMarkovConfig | None = None,
    constraints: tuple[StageEditSectionConstraint, ...] = (),
    derivative_metadata: SourceDerivativeFamilyMetadata | None = None,
):
    selected_plan = plan or _plan(duration)
    return decode_stage_edit_semimarkov_beam(
        checkpoint_directory=Path("/tmp/stageedit-production"),
        sources=_sources(source_count, float(duration)),
        source_metadata=metadata or _metadata(source_count, float(duration)),
        head_frames=frames or _frames(duration, source_count),
        section_plan=selected_plan,
        output=Path("/tmp/stageedit-result.mp4"),
        section_constraints=constraints,
        config=config or StageEditSemiMarkovConfig(
            beam_width=16,
            timeline_count=8,
            minimum_unique_sources=3,
        ),
        aspect_ratio=AspectRatio.LANDSCAPE,
        width=1920,
        height=1080,
        fps=30.0,
        source_derivative_metadata=derivative_metadata,
    )


def _assert_finite_tree(test: unittest.TestCase, value: object) -> None:
    if isinstance(value, float):
        test.assertTrue(math.isfinite(value))
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_tree(test, item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_finite_tree(test, item)


class StageEditSemiMarkovBeamTests(unittest.TestCase):
    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_verified_derivative_families_own_diversity_and_occupancy(
        self, _mocked
    ) -> None:
        metadata = SourceDerivativeFamilyMetadata(
            source_count=4,
            family_ids=(0, 0, 1, 2),
            audit_sha256="a" * 64,
            production_ready=True,
            uncertain_pair_count=0,
            clock_contract_complete=True,
        )
        result = _decode(
            derivative_metadata=metadata,
            config=StageEditSemiMarkovConfig(
                beam_width=32,
                timeline_count=8,
                minimum_unique_sources=3,
                maximum_source_occupancy_fraction=0.60,
            ),
        )
        audit = result.audit.input_summary["source_derivative_diversity"]
        self.assertEqual([0, 0, 1, 2], audit["family_ids"])
        self.assertEqual(
            "aggregate_all_sources_in_derivative_family",
            result.audit.input_summary["occupancy_semantics"],
        )
        self.assertGreater(
            result.audit.constraint_rejection_counts.get(
                "derivative_family_occupancy_cap", 0
            ),
            0,
        )
        for timeline in result.timelines:
            family_path = tuple(
                metadata.family_ids[decision.source_index]
                for decision in timeline.decisions
            )
            self.assertTrue(
                all(left != right for left, right in zip(family_path, family_path[1:]))
            )
            self.assertGreaterEqual(len(set(family_path)), 3)
            self.assertTrue(
                all(value <= 3.6 + 1e-6 for value in timeline.derivative_family_occupancy_seconds)
            )
            self.assertAlmostEqual(
                timeline.source_occupancy_seconds[0]
                + timeline.source_occupancy_seconds[1],
                timeline.derivative_family_occupancy_seconds[0],
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_uncertain_or_insufficient_family_metadata_fails_before_search(
        self, _mocked
    ) -> None:
        uncertain = SourceDerivativeFamilyMetadata(
            source_count=4,
            family_ids=(0, 1, 2, 3),
            audit_sha256="b" * 64,
            production_ready=False,
            uncertain_pair_count=1,
            clock_contract_complete=True,
        )
        with self.assertRaisesRegex(StageEditDecodeInputError, "uncertain"):
            _decode(derivative_metadata=uncertain)

        only_three_families = SourceDerivativeFamilyMetadata(
            source_count=4,
            family_ids=(0, 0, 1, 2),
            audit_sha256="c" * 64,
            production_ready=True,
            uncertain_pair_count=0,
            clock_contract_complete=True,
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError, "derivative-family count"
        ):
            _decode(
                derivative_metadata=only_three_families,
                config=StageEditSemiMarkovConfig(minimum_unique_sources=4),
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_beam_state_owns_and_applies_all_editorial_history(self, _mocked) -> None:
        result = _decode()
        self.assertEqual(
            STAGE_EDIT_DECISION_HISTORY_OWNER,
            result.audit.input_summary["decision_history_owner"],
        )
        self.assertEqual(
            {
                "current_source",
                "recent_sources",
                "recent_shot_sizes",
                "recent_shot_durations",
                "recent_cut_times",
                "source_occupancy_seconds",
                "section_switch_counts",
            },
            set(result.audit.input_summary["owned_decision_history_state"]),
        )
        self.assertGreater(
            result.audit.constraint_rejection_counts.get("aba_source_return", 0),
            0,
        )
        self.assertGreater(
            result.audit.constraint_rejection_counts.get("source_occupancy_cap", 0),
            0,
        )
        for timeline in result.timelines:
            for decision in timeline.decisions:
                self.assertEqual(
                    decision.source_index, decision.recent_sources_after[-1]
                )
                self.assertAlmostEqual(
                    decision.duration_seconds,
                    decision.recent_shot_durations_after[-1],
                )
                self.assertAlmostEqual(
                    decision.master_end_seconds,
                    sum(decision.source_occupancy_after_seconds),
                )
                self.assertEqual(
                    decision.shot_index,
                    sum(decision.section_switch_counts_after),
                )
            self.assertEqual(
                timeline.source_occupancy_seconds,
                timeline.decisions[-1].source_occupancy_after_seconds,
            )
            self.assertEqual(
                timeline.recent_cut_density_per_minute,
                timeline.decisions[-1].recent_cut_density_per_minute_after,
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_not_ready_report(),
    )
    def test_requires_production_ready_checkpoint_before_input_validation(self, mocked) -> None:
        with self.assertRaisesRegex(
            StageEditCheckpointNotReadyError,
            "production-ready checkpoint",
        ):
            decode_stage_edit_semimarkov_beam(
                checkpoint_directory=Path("/tmp/random"),
                sources=(),
                source_metadata=(),
                head_frames=(),
                section_plan=_plan(),
                output=Path("/tmp/out.mp4"),
            )
        mocked.assert_called_once_with(Path("/tmp/random"))

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_all_hard_gated_frame_fails_closed(self, _mocked) -> None:
        frames = list(_frames(6, 4))
        item = frames[2]
        frames[2] = StageEditHeadFrame(
            master_time_seconds=item.master_time_seconds,
            cut_logit=item.cut_logit,
            source_logits=item.source_logits,
            duration_logits=item.duration_logits,
            transition_logits=item.transition_logits,
            crop=item.crop,
            available_source_mask=item.available_source_mask,
            hard_source_gate_mask=(False,) * 4,
            source_shot_sizes=item.source_shot_sizes,
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "no source surviving availability and hard gates",
        ):
            _decode(frames=tuple(frames))

    def test_nonfinite_head_value_is_rejected_without_nan_propagation(self) -> None:
        with self.assertRaisesRegex(StageEditDecodeInputError, "must be finite"):
            StageEditHeadFrame(
                master_time_seconds=0.0,
                cut_logit=0.0,
                source_logits=(0.0, float("nan")),
                duration_logits=(0.0,) * 7,
                transition_logits=(0.0,) * 4,
                crop=(1.0, 0.5, 0.5, 0.0),
                available_source_mask=(True, True),
                hard_source_gate_mask=(True, True),
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_deterministic_joint_decode_returns_valid_audited_stage_edits(
        self, _mocked
    ) -> None:
        metadata = _metadata(4, 6.0, internal_cuts={0: (1.5,)})
        gates = {2: (True, False, True, True)}
        frames = _frames(6, 4, gate_overrides=gates)
        first = _decode(frames=frames, metadata=metadata)
        second = _decode(frames=frames, metadata=metadata)

        self.assertEqual(8, len(first.timelines))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(first.audit.request_fulfilled)
        self.assertEqual(8, first.audit.returned_timeline_count)
        self.assertGreater(first.audit.transition_hypothesis_count, 0)
        self.assertGreater(first.audit.accepted_hypothesis_count, 0)
        self.assertIn(
            "source_internal_cut_inside_shot",
            first.audit.constraint_rejection_counts,
        )
        self.assertIn(
            "source_unavailable_or_hard_gated_inside_span",
            first.audit.constraint_rejection_counts,
        )

        for timeline in first.timelines:
            edit = timeline.edit
            self.assertAlmostEqual(6.0, edit.duration_seconds)
            self.assertEqual(len(edit.shots) - 1, len(edit.boundaries))
            self.assertTrue(
                str(edit.output).endswith(f"stageedit-{timeline.rank:02d}.mp4")
            )
            sources = tuple(item.source_index for item in timeline.decisions)
            self.assertGreaterEqual(len(set(sources)), 3)
            self.assertTrue(all(left != right for left, right in zip(sources, sources[1:])))
            self.assertTrue(
                all(
                    sources[index] != sources[index - 2]
                    for index in range(2, len(sources))
                )
            )
            self.assertTrue(all(value <= 3.6 + 1e-6 for value in timeline.source_occupancy_seconds))
            for decision in timeline.decisions:
                self.assertGreaterEqual(decision.duration_seconds, 0.5)
                self.assertLessEqual(decision.duration_seconds, 3.0)
                self.assertEqual(10, len(decision.score_components))
                self.assertIn(
                    "boundary_continuity",
                    {item.name for item in decision.score_components},
                )
            self.assertTrue(
                all(
                    boundary.transition.kind.value == "hard_cut"
                    and boundary.transition.reason.startswith(
                        "stage_edit_semimarkov_requested_"
                    )
                    for boundary in edit.boundaries
                )
            )
        _assert_finite_tree(self, first.to_dict())

    def test_same_time_continuity_score_prefers_geometry_and_look_match(self) -> None:
        outgoing = StageEditSourceContinuity(
            face_scale=0.30,
            face_center_x=0.48,
            face_center_y=0.43,
            camera_motion_x=0.15,
            camera_motion_y=-0.10,
            exposure=0.52,
            color_temperature_normalized=0.47,
            sharpness=0.61,
        )
        matched = StageEditSourceContinuity(
            face_scale=0.31,
            face_center_x=0.49,
            face_center_y=0.44,
            camera_motion_x=0.14,
            camera_motion_y=-0.09,
            exposure=0.50,
            color_temperature_normalized=0.49,
            sharpness=0.59,
        )
        mismatched = StageEditSourceContinuity(
            face_scale=0.12,
            face_center_x=0.76,
            face_center_y=0.21,
            camera_motion_x=-0.60,
            camera_motion_y=0.55,
            exposure=0.88,
            color_temperature_normalized=0.10,
            sharpness=0.20,
        )

        matched_score = stage_edit_boundary_continuity_score(outgoing, matched)
        mismatched_score = stage_edit_boundary_continuity_score(
            outgoing, mismatched
        )
        self.assertGreater(matched_score, mismatched_score)
        self.assertLessEqual(matched_score, 0.0)
        self.assertGreaterEqual(mismatched_score, -1.0)
        self.assertAlmostEqual(
            -0.12,
            stage_edit_boundary_continuity_score(
                StageEditSourceContinuity(), StageEditSourceContinuity()
            ),
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_decoder_scores_the_beam_specific_outgoing_incoming_pair(
        self, _mocked
    ) -> None:
        continuity = tuple(
            StageEditSourceContinuity(
                face_scale=0.18 + 0.08 * source_index,
                face_center_x=0.20 + 0.18 * source_index,
                face_center_y=0.42,
                exposure=0.35 + 0.12 * source_index,
            )
            for source_index in range(4)
        )
        frames = tuple(
            replace(frame, source_continuity=continuity)
            for frame in _frames(6, 4)
        )
        result = _decode(frames=frames)
        pair_components = [
            component
            for decision in result.timelines[0].decisions[1:]
            for component in decision.score_components
            if component.name == "boundary_continuity"
        ]
        self.assertTrue(pair_components)
        self.assertTrue(any(component.raw_value < 0.0 for component in pair_components))
        self.assertTrue(
            all(
                component.contribution <= 0.0
                and component.weight
                == StageEditSemiMarkovConfig().boundary_continuity_weight
                for component in pair_components
            )
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_rejected_source_logits_cannot_change_valid_timeline_scores(
        self, _mocked
    ) -> None:
        masks = {index: (True, True, True, False) for index in range(6)}
        baseline = _frames(6, 4, gate_overrides=masks)
        changed = tuple(
            replace(
                frame,
                source_logits=frame.source_logits[:3] + (1_000_000.0,),
            )
            for frame in baseline
        )
        first = _decode(frames=baseline)
        second = _decode(frames=changed)

        self.assertNotEqual(first.audit.input_sha256, second.audit.input_sha256)
        self.assertEqual(
            [item.to_dict() for item in first.timelines],
            [item.to_dict() for item in second.timelines],
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_supports_32_wide_beam_and_32_ranked_timelines(self, _mocked) -> None:
        result = _decode(
            duration=8,
            source_count=5,
            config=StageEditSemiMarkovConfig(
                beam_width=32,
                timeline_count=32,
                minimum_unique_sources=3,
                maximum_global_switches_per_minute=30.0,
            ),
        )
        self.assertEqual(32, len(result.timelines))
        self.assertTrue(result.audit.request_fulfilled)
        self.assertEqual(list(range(1, 33)), [item.rank for item in result.timelines])
        self.assertEqual(
            sorted((item.score for item in result.timelines), reverse=True),
            [item.score for item in result.timelines],
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_section_boundaries_and_overrides_are_hard_constraints(self, _mocked) -> None:
        plan = SectionPlan(
            (
                _section(0, 0.0, 3.0, rate=18.0),
                _section(1, 3.0, 6.0, rate=20.0),
            )
        )
        constraints = (
            StageEditSectionConstraint(
                section_index=0,
                minimum_shot_seconds=1.0,
                maximum_shot_seconds=2.0,
                allowed_transition_labels=("hard_cut", "zoom_match"),
            ),
            StageEditSectionConstraint(
                section_index=1,
                minimum_shot_seconds=1.0,
                maximum_shot_seconds=2.0,
                allowed_transition_labels=("hard_cut",),
            ),
        )
        result = _decode(plan=plan, constraints=constraints)
        for timeline in result.timelines:
            for decision in timeline.decisions:
                section = plan.directive_at(
                    decision.master_start_seconds + decision.duration_seconds / 2.0
                )
                self.assertGreaterEqual(decision.master_start_seconds, section.start_seconds)
                self.assertLessEqual(decision.master_end_seconds, section.end_seconds)
                self.assertGreaterEqual(decision.duration_seconds, 1.0)
                self.assertLessEqual(decision.duration_seconds, 2.0)
                if decision.master_start_seconds >= 3.0 and decision.transition_label != "initial":
                    self.assertEqual("hard_cut", decision.transition_label)
        self.assertIn(
            "transition_disallowed_by_section",
            result.audit.constraint_rejection_counts,
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_no_complete_path_raises_instead_of_hard_cut_fallback(self, _mocked) -> None:
        cuts = tuple(0.25 + 0.5 * index for index in range(12))
        metadata = _metadata(2, 6.0, internal_cuts={0: cuts, 1: cuts})
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "eliminated every complete",
        ):
            _decode(
                source_count=2,
                metadata=metadata,
                config=StageEditSemiMarkovConfig(
                    beam_width=8,
                    timeline_count=8,
                    minimum_unique_sources=2,
                ),
            )

    def test_beam_and_timeline_widths_are_bounded_to_eight_through_thirty_two(self) -> None:
        for kwargs in (
            {"beam_width": 7, "timeline_count": 7},
            {"beam_width": 33, "timeline_count": 8},
            {"beam_width": 16, "timeline_count": 17},
        ):
            with self.assertRaises(StageEditDecodeInputError):
                StageEditSemiMarkovConfig(**kwargs)
        with self.assertRaisesRegex(StageEditDecodeInputError, "A-B-A"):
            StageEditSemiMarkovConfig(recent_source_window=1)
        with self.assertRaisesRegex(StageEditDecodeInputError, "shot_size_window"):
            StageEditSemiMarkovConfig(
                recent_shot_size_window=2,
                maximum_same_shot_size_run=3,
            )

    @staticmethod
    def _dynamic_prediction(*, first_time: float = 0.0) -> DynamicCropHeadPredictionV2:
        return DynamicCropHeadPredictionV2(
            normalized_times=(first_time, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            values=(
                (1.04, 0.500, 0.500, 0.0),
                (1.08, 0.505, 0.500, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
            presence_logits=(20.0, 20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0),
            motion_logit=20.0,
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_dynamic_crop_v2_reaches_renderer_as_duration_scaled_k8_track(
        self, _mocked
    ) -> None:
        prediction = self._dynamic_prediction()
        frames = tuple(
            replace(
                frame,
                # Deliberately unsafe legacy values prove that the v1 clamp is
                # not consulted once the v2 contract is present.
                crop=(99.0, 99.0, -99.0, 99.0),
                dynamic_crop_v2=prediction,
            )
            for frame in _frames(6, 4)
        )

        result = _decode(frames=frames)

        self.assertEqual(4, result.audit.schema_version)
        self.assertEqual(
            "verified_derivative_family_not_file_source",
            result.audit.input_summary["diversity_semantics"],
        )
        self.assertEqual(
            2,
            result.audit.input_summary["decoder_schema_migration"]["supersedes"],
        )
        self.assertEqual(
            DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT,
            result.audit.input_summary["crop_contract"],
        )
        self.assertGreater(
            result.audit.input_summary["selected_dynamic_crop_applied_keyframe_count"],
            0,
        )
        for timeline in result.timelines:
            for decision, shot in zip(
                timeline.decisions, timeline.edit.shots, strict=True
            ):
                self.assertEqual(
                    DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT,
                    decision.crop_contract,
                )
                self.assertIsNotNone(decision.dynamic_crop_decode)
                self.assertTrue(decision.dynamic_crop_decode.eligible)
                self.assertIs(decision.editorial_transform, shot.segment.editorial_transform)
                assert shot.segment.editorial_transform is not None
                self.assertEqual(2, len(shot.segment.editorial_transform.keyframes))
                self.assertAlmostEqual(
                    decision.duration_seconds * 0.5,
                    shot.segment.editorial_transform.keyframes[1].time_seconds,
                )
                self.assertEqual((1.04, 0.5, 0.5, 0.0), decision.crop_applied)

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_rejected_dynamic_track_never_uses_legacy_or_identity_track_fallback(
        self, _mocked
    ) -> None:
        prediction = self._dynamic_prediction(first_time=0.1)
        frames = tuple(
            replace(
                frame,
                crop=(1.15, 0.50, 0.50, 0.0),
                dynamic_crop_v2=prediction,
            )
            for frame in _frames(6, 4)
        )

        result = _decode(frames=frames)

        self.assertEqual(
            0,
            result.audit.input_summary["selected_dynamic_crop_eligible_count"],
        )
        self.assertTrue(
            result.audit.input_summary["selected_dynamic_crop_rejection_reasons"]
        )
        for timeline in result.timelines:
            for decision, shot in zip(
                timeline.decisions, timeline.edit.shots, strict=True
            ):
                self.assertIsNone(decision.editorial_transform)
                self.assertIsNone(shot.segment.editorial_transform)
                self.assertEqual((1.0, 0.5, 0.5, 0.0), decision.crop_applied)
                self.assertFalse(decision.crop_was_corrected)
                assert decision.dynamic_crop_decode is not None
                self.assertFalse(decision.dynamic_crop_decode.eligible)
                self.assertIn(
                    "does not start",
                    decision.dynamic_crop_decode.rejection_reasons[0],
                )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_mixed_legacy_and_dynamic_crop_frames_fail_closed(self, _mocked) -> None:
        frames = list(_frames(6, 4))
        frames[0] = replace(
            frames[0], dynamic_crop_v2=self._dynamic_prediction()
        )
        with self.assertRaisesRegex(StageEditDecodeInputError, "cannot mix"):
            _decode(frames=tuple(frames))


if __name__ == "__main__":
    unittest.main()
