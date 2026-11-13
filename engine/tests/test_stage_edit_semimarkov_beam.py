from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.ive import AspectRatio
from crosscut.professional_montage_policy import (
    close_burst_policy_binding_sha256,
)
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
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT,
    DynamicCropHeadPredictionV3,
)
from crosscut.stage_edit_semimarkov_beam import (
    CONTINUITY_AUTHORITY_CONTRACT,
    DECODER_CONTRACT,
    DECODER_SCHEMA_VERSION,
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
    StageEditCheckpointNotReadyError,
    StageEditDecodeInputError,
    StageEditHeadFrame,
    StageEditHighlightMotionCalibration,
    StageEditNoValidTimelineError,
    StageEditSectionConstraint,
    StageEditSemiMarkovConfig,
    StageEditSourceContinuity,
    StageEditSourceMetadata,
    StageEditTemporalPoseMotionCalibration,
    _formation_wide_evidence,
    _boundary_motion_amount_plus_continuity_evidence,
    _highlight_motion_calibration_audit,
    _highlight_policy_provenance,
    _close_only_multi_source_burst_evidence,
    _montage_grammar_score_raw,
    _wide_switch_preferred_target_score_raw,
    decode_stage_edit_semimarkov_beam,
    stage_edit_boundary_continuity_score,
)
from crosscut.stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES,
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
    StageEditSourcePairScorer,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_DECISION_HISTORY_OWNER,
    STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT,
    StageEditReadinessReport,
    StageEditTemporalPoseSidecar,
)
from crosscut.stage_edit_temporal_pose_motion import (
    TemporalPoseMotionThresholds,
)
from crosscut.stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
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
                source_continuity=tuple(
                    StageEditSourceContinuity(
                        detected_person_count_normalized=(
                            1.0
                            if sizes[index % len(sizes)] in {"wide", "full"}
                            else 2.0 / 6.0
                        ),
                        formation_person_count_normalized=(
                            1.0
                            if sizes[index % len(sizes)] in {"wide", "full"}
                            else 0.0
                        ),
                        formation_pose_coverage=(
                            5.0 / 6.0
                            if sizes[index % len(sizes)] in {"wide", "full"}
                            else 0.0
                        ),
                    )
                    for index in range(source_count)
                ),
                candidate_id=f"beat-{time_index}",
                section_progress=(time_index / max(1, duration - 1)),
                # Exact serialized-token resets at indices 2 and 4.  This
                # gives the professional phrase-entry gate real evidence in
                # decoder tests instead of a wall-clock approximation.
                phrase_progress=(time_index % 2) / 2.0,
                section_role="chorus_like",
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
    source_pair_scorer: StageEditSourcePairScorer | None = None,
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
            enforce_montage_grammar=False,
        ),
        aspect_ratio=AspectRatio.LANDSCAPE,
        width=1920,
        height=1080,
        fps=30.0,
        source_derivative_metadata=derivative_metadata,
        source_pair_scorer=source_pair_scorer,
    )


def _single_feature_scorer(feature_name: str) -> StageEditSourcePairScorer:
    weights = [0.0] * len(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES)
    weights[STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES.index(feature_name)] = 1.0
    return StageEditSourcePairScorer(tuple(weights))


def _actual_highlight_motion_calibration() -> StageEditHighlightMotionCalibration:
    return StageEditHighlightMotionCalibration(
        analysis_basis=(
            "ive_20_reference_snapshot_source_pair_supervision_v3_calibration"
        ),
        human_reference_count=20,
        source_pair_negative_samples=2,
        maximum_sources=32,
        confirmed_source_switch_count=170,
        highlight_role_close_involved_switch_count=54,
        build_close_involved_switch_count=10,
        chorus_close_involved_switch_count=29,
        bridge_close_involved_switch_count=15,
        pose_pair_observed_switch_count=23,
        continuity_observed_switch_count=54,
        pose_embedding_cosine_observed_switch_count=14,
        sharpness_pair_observed_switch_count=0,
        pose_motion_minimum_quantiles=(
            0.0002705025925969218,
            0.0013651752724601618,
            0.0024240018497856952,
            0.010089000247239965,
        ),
        continuity_score_quantiles=(
            -0.49213925079872173,
            -0.39991898058929115,
            -0.32434112477150095,
            -0.2522282468388802,
        ),
        pose_embedding_cosine_quantiles=(
            0.7354177153190032,
            0.7831558357163766,
            0.8486102174679631,
            0.8940526060955091,
        ),
        sharpness_minimum_quantiles=None,
    )


def _pose60(angle_radians: float) -> tuple[float, ...]:
    return (
        math.cos(angle_radians),
        math.sin(angle_radians),
        *(0.0 for _ in range(58)),
    )


def _temporal_sidecar(
    first_angle: float,
    second_angle: float,
) -> StageEditTemporalPoseSidecar:
    return StageEditTemporalPoseSidecar(
        embeddings=(_pose60(first_angle), _pose60(second_angle)),
        interval_seconds=0.1,
        sampling_contract=(
            STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT
        ),
    )


def _assert_finite_tree(test: unittest.TestCase, value: object) -> None:
    if isinstance(value, float):
        test.assertTrue(math.isfinite(value))
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_tree(test, item)


def _assert_no_raw_pose_vectors(
    test: unittest.TestCase,
    value: object,
) -> None:
    forbidden_keys = {
        "pose_model_embedding",
        "normalized_velocity",
        "pose_embedding",
        "embeddings",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key)
            test.assertFalse(
                normalized_key in forbidden_keys
                or normalized_key.endswith("_pose_embedding")
                or normalized_key.endswith("_normalized_velocity")
                or normalized_key.endswith("_embeddings"),
                key,
            )
            _assert_no_raw_pose_vectors(test, item)
    elif isinstance(value, (tuple, list)):
        if value and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            test.assertNotIn(len(value), {60, 120})
        for item in value:
            _assert_no_raw_pose_vectors(test, item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_finite_tree(test, item)


class StageEditSemiMarkovBeamTests(unittest.TestCase):
    @staticmethod
    def _forced_frames(
        source_sequence: tuple[int, ...],
        *,
        source_count: int = 6,
    ) -> tuple[StageEditHeadFrame, ...]:
        return tuple(
            replace(
                frame,
                source_logits=tuple(
                    20.0 if source_index == source_sequence[time_index] else -20.0
                    for source_index in range(source_count)
                ),
            )
            for time_index, frame in enumerate(
                _frames(len(source_sequence), source_count)
            )
        )

    def test_chorus_wide_involved_family_switch_has_small_bounded_prior(self) -> None:
        directive = _section(0, 0.0, 6.0)
        config = StageEditSemiMarkovConfig()

        def score(
            *,
            switched: bool,
            formation: bool,
            prior_switches: int = 4,
            prior_wide_switches: int = 3,
        ) -> float:
            return _montage_grammar_score_raw(
                directive=directive,
                shot_size="medium",
                start_seconds=3.0,
                duration_seconds=1.0,
                previous_shot_size="wide",
                previous_duration_seconds=1.0,
                derivative_family_source_switched=switched,
                previous_formation_wide_evidenced=formation,
                formation_person_count_normalized=0.0,
                formation_pose_coverage=0.0,
                config=config,
                prior_source_switch_boundaries=prior_switches,
                prior_wide_involved_source_switch_boundaries=(
                    prior_wide_switches
                ),
            )

        same_family = score(switched=False, formation=True)
        generic_wide_switch = score(switched=True, formation=False)
        formation_wide_switch = score(switched=True, formation=True)
        repeated_wide_only_switch = score(
            switched=True,
            formation=True,
            prior_switches=2,
            prior_wide_switches=2,
        )
        self.assertAlmostEqual(0.10, generic_wide_switch - same_family)
        self.assertAlmostEqual(0.06, formation_wide_switch - generic_wide_switch)
        self.assertAlmostEqual(same_family, repeated_wide_only_switch)
        self.assertLess(
            config.montage_grammar_score_weight * formation_wide_switch,
            config.source_score_weight,
        )

    def test_strong_multi_person_wide_survives_tiny_pose_failure(self) -> None:
        config = StageEditSemiMarkovConfig()
        self.assertTrue(
            _formation_wide_evidence(
                shot_size="wide",
                formation_person_count_normalized=4.0 / 6.0,
                formation_pose_coverage=0.0,
                config=config,
            )
        )
        self.assertFalse(
            _formation_wide_evidence(
                shot_size="wide",
                formation_person_count_normalized=3.0 / 6.0,
                formation_pose_coverage=0.0,
                config=config,
            )
        )
        self.assertTrue(
            _formation_wide_evidence(
                shot_size="wide",
                formation_person_count_normalized=1.0 / 6.0,
                formation_pose_coverage=1.0 / 6.0,
                config=config,
            )
        )

    def test_strong_formation_threshold_cannot_weaken_reference_floor(self) -> None:
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "strong formation threshold",
        ):
            StageEditSemiMarkovConfig(
                minimum_formation_anchor_person_count_normalized=0.5,
                strong_formation_anchor_person_count_normalized=0.4,
            )

    def test_nonwide_burst_prior_breaks_wide_only_run_during_soft_target_dip(
        self,
    ) -> None:
        directive = _section(0, 0.0, 6.0)
        config = StageEditSemiMarkovConfig()
        temporary_dip_bonus = _wide_switch_preferred_target_score_raw(
            prior_source_switch_boundaries=1,
            prior_wide_involved_source_switch_boundaries=1,
            derivative_family_source_switched=True,
            wide_involved_switch=False,
            preferred_fraction=0.8,
        )
        self.assertEqual(temporary_dip_bonus, 0.0)

        common = {
            "directive": directive,
            "start_seconds": 3.0,
            "duration_seconds": 1.0,
            "previous_shot_size": "wide",
            "previous_duration_seconds": 1.0,
            "derivative_family_source_switched": True,
            "previous_formation_wide_evidenced": True,
            "config": config,
            "prior_source_switch_boundaries": 1,
            "prior_wide_involved_source_switch_boundaries": 1,
        }
        stay_wide = _montage_grammar_score_raw(
            shot_size="wide",
            formation_person_count_normalized=1.0,
            formation_pose_coverage=1.0,
            **common,
        )
        enter_nonwide = _montage_grammar_score_raw(
            shot_size="medium",
            formation_person_count_normalized=0.0,
            formation_pose_coverage=0.0,
            **common,
        )
        self.assertGreater(enter_nonwide, stay_wide)

        burst = _montage_grammar_score_raw(
            directive=directive,
            shot_size="closeup",
            start_seconds=4.0,
            duration_seconds=1.0,
            previous_shot_size="medium",
            previous_duration_seconds=1.0,
            derivative_family_source_switched=True,
            previous_formation_wide_evidenced=False,
            formation_person_count_normalized=0.0,
            formation_pose_coverage=0.0,
            config=config,
            prior_source_switch_boundaries=2,
            prior_wide_involved_source_switch_boundaries=1,
        )
        self.assertGreater(burst, enter_nonwide)

    def test_per_shot_prior_stops_rewarding_closeups_past_p50_target(
        self,
    ) -> None:
        directive = _section(0, 0.0, 10.0)
        config = StageEditSemiMarkovConfig()
        common = {
            "directive": directive,
            "shot_size": "closeup",
            "start_seconds": 5.0,
            "duration_seconds": 1.0,
            "derivative_family_source_switched": True,
            "previous_formation_wide_evidenced": False,
            "formation_person_count_normalized": 0.0,
            "formation_pose_coverage": 0.0,
            "config": config,
        }
        toward_target = _montage_grammar_score_raw(
            previous_shot_size="medium",
            previous_duration_seconds=1.0,
            prior_section_known_duration_seconds=4.0,
            prior_section_closeup_duration_seconds=1.0,
            prior_section_nonwide_duration_seconds=2.0,
            **common,
        )
        past_target = _montage_grammar_score_raw(
            previous_shot_size="closeup",
            previous_duration_seconds=1.0,
            prior_section_known_duration_seconds=5.0,
            prior_section_closeup_duration_seconds=2.0,
            prior_section_nonwide_duration_seconds=3.0,
            **common,
        )
        self.assertGreater(toward_target, past_target)
        self.assertEqual(0.0, past_target)

    def test_profile_v3_motion_distribution_is_audited_without_a_hard_default(
        self,
    ) -> None:
        calibration = _actual_highlight_motion_calibration()
        config = StageEditSemiMarkovConfig(
            highlight_motion_calibration=calibration,
            highlight_motion_boundary_minimum_pose_motion=(
                calibration.pose_motion_minimum_quantiles[0]
            ),
            highlight_motion_boundary_minimum_continuity_score=(
                calibration.continuity_score_quantiles[0]
            ),
            highlight_motion_boundary_minimum_pose_phase_cosine=(
                calibration.pose_embedding_cosine_quantiles[0]
            ),
        )

        audit = _highlight_motion_calibration_audit(config)

        self.assertTrue(audit["profile_v3_calibration_input_bound"])
        self.assertTrue(audit["soft_bonus_active"])
        self.assertTrue(
            audit["motion_continuity_qualifier"]
            ["independent_from_primary_boundary_continuity"]
        )
        self.assertFalse(
            audit["motion_continuity_qualifier"]
            ["primary_boundary_score_reused"]
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            audit["motion_continuity_qualifier"]["scorer_fingerprint"],
        )
        self.assertEqual(
            audit["motion_continuity_qualifier"]["scorer_fingerprint"],
            audit["calibration"]["continuity_qualifier"]
            ["scorer_fingerprint"],
        )
        self.assertEqual(0, audit["hard_minimum_supported_boundary_count"])
        self.assertEqual(
            170,
            audit["calibration"]["counts"][
                "confirmed_source_switch_count"
            ],
        )
        self.assertEqual(
            54,
            audit["calibration"]["counts"][
                "highlight_role_close_involved_switch_count"
            ],
        )
        self.assertEqual(
            23,
            audit["calibration"]["distributions"][
                "pose_motion_minimum"
            ]["observed_switch_count"],
        )
        self.assertEqual(
            {"build": 10, "chorus_like": 29, "bridge": 15},
            audit["calibration"]["counts"][
                "highlight_role_close_involved_switch_count_by_role"
            ],
        )
        self.assertEqual(
            14,
            audit["calibration"]["distributions"][
                "same_time_pose_embedding_cosine_proxy"
            ]["observed_switch_count"],
        )
        self.assertTrue(
            audit["calibration"]["distributions"][
                "same_time_pose_embedding_cosine_proxy"
            ]["runtime_pair_feature_available"]
        )
        self.assertEqual(
            0,
            audit["calibration"]["distributions"]["sharpness_minimum"][
                "observed_switch_count"
            ],
        )
        self.assertIsNone(
            audit["calibration"]["distributions"]["sharpness_minimum"][
                "values"
            ]
        )
        self.assertEqual("p10", audit["selected_thresholds"]["pose_motion_quantile"])
        self.assertEqual(
            "p10",
            audit["selected_thresholds"]["continuity_score_quantile"],
        )
        self.assertEqual(
            "p10",
            audit["selected_thresholds"]
            ["handcrafted_continuity_qualifier_score_quantile"],
        )
        self.assertEqual(
            "p10",
            audit["selected_thresholds"]["pose_phase_cosine_quantile"],
        )
        self.assertTrue(audit["pose_phase_threshold_bound"])
        self.assertTrue(
            audit["soft_bonus_requires_pose_phase_when_threshold_bound"]
        )
        self.assertFalse(audit["pose_phase_direction_or_velocity_proven"])
        self.assertEqual(
            64,
            len(audit["calibration"]["calibration_fingerprint_sha256"]),
        )

    def test_motion_closeup_bonus_needs_profile_authority_and_stops_at_p50(
        self,
    ) -> None:
        calibration = _actual_highlight_motion_calibration()
        calibrated = StageEditSemiMarkovConfig(
            highlight_motion_calibration=calibration,
            highlight_motion_boundary_minimum_pose_motion=(
                calibration.pose_motion_minimum_quantiles[0]
            ),
            highlight_motion_boundary_minimum_continuity_score=(
                calibration.continuity_score_quantiles[0]
            ),
            highlight_motion_boundary_minimum_pose_phase_cosine=(
                calibration.pose_embedding_cosine_quantiles[0]
            ),
        )
        unbound = StageEditSemiMarkovConfig(
            highlight_motion_boundary_minimum_pose_motion=(
                calibration.pose_motion_minimum_quantiles[0]
            ),
            highlight_motion_boundary_minimum_continuity_score=(
                calibration.continuity_score_quantiles[0]
            ),
        )
        common = {
            "directive": _section(0, 0.0, 10.0),
            "shot_size": "closeup",
            "start_seconds": 5.0,
            "duration_seconds": 1.0,
            "previous_shot_size": "wide",
            "previous_duration_seconds": 1.0,
            "derivative_family_source_switched": True,
            "previous_formation_wide_evidenced": False,
            "formation_person_count_normalized": 0.0,
            "formation_pose_coverage": 0.0,
            "prior_source_switch_boundaries": 0,
            "prior_wide_involved_source_switch_boundaries": 0,
            "prior_section_known_duration_seconds": 4.0,
            "prior_section_closeup_duration_seconds": 1.0,
            "prior_section_nonwide_duration_seconds": 2.0,
        }
        static_score = _montage_grammar_score_raw(
            config=calibrated,
            boundary_motion_accent_supported=False,
            **common,
        )
        moving_score = _montage_grammar_score_raw(
            config=calibrated,
            boundary_motion_accent_supported=True,
            **common,
        )
        authority_missing_score = _montage_grammar_score_raw(
            config=unbound,
            boundary_motion_accent_supported=True,
            **common,
        )
        self.assertAlmostEqual(0.12, moving_score - static_score)
        self.assertAlmostEqual(static_score, authority_missing_score)

        past_target = dict(common)
        past_target.update(
            prior_section_known_duration_seconds=5.0,
            prior_section_closeup_duration_seconds=2.0,
            prior_section_nonwide_duration_seconds=3.0,
        )
        past_static = _montage_grammar_score_raw(
            config=calibrated,
            boundary_motion_accent_supported=False,
            **past_target,
        )
        past_moving = _montage_grammar_score_raw(
            config=calibrated,
            boundary_motion_accent_supported=True,
            **past_target,
        )
        self.assertAlmostEqual(past_static, past_moving)

    def test_learned_primary_score_is_not_reused_for_handcrafted_motion_quantile(
        self,
    ) -> None:
        calibration = _actual_highlight_motion_calibration()
        config = StageEditSemiMarkovConfig(
            continuity_authority=PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            continuity_scorer_contract=STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            continuity_scorer_fingerprint="a" * 64,
            highlight_motion_calibration=calibration,
            highlight_motion_boundary_minimum_pose_motion=(
                calibration.pose_motion_minimum_quantiles[0]
            ),
            highlight_motion_boundary_minimum_continuity_score=(
                calibration.continuity_score_quantiles[0]
            ),
        )
        outgoing = StageEditSourceContinuity(
            face_scale=0.30,
            face_center_x=0.10,
            face_center_y=0.10,
            camera_motion_x=1.0,
            camera_motion_y=1.0,
            pose_motion=0.9,
            camera_zoom_velocity=1.0,
            camera_roll_normalized=1.0,
            exposure=0.0,
            color_temperature_normalized=0.0,
            sharpness=1.0,
        )
        incompatible = StageEditSourceContinuity(
            face_scale=0.10,
            face_center_x=0.90,
            face_center_y=0.90,
            camera_motion_x=-1.0,
            camera_motion_y=-1.0,
            pose_motion=0.9,
            camera_zoom_velocity=-1.0,
            camera_roll_normalized=-1.0,
            exposure=1.0,
            color_temperature_normalized=1.0,
            sharpness=1.0,
        )

        learned_passes_handcrafted_fails = (
            _boundary_motion_amount_plus_continuity_evidence(
                frame=replace(
                    _frames(1, 2)[0],
                    source_continuity=(outgoing, incompatible),
                ),
                outgoing_source_index=0,
                incoming_source_index=1,
                derivative_family_ids=(0, 1),
                continuity_score=0.0,
                config=config,
            )
        )
        self.assertEqual(
            0.0,
            learned_passes_handcrafted_fails[
                "primary_boundary_continuity_score"
            ],
        )
        self.assertLess(
            learned_passes_handcrafted_fails[
                "handcrafted_continuity_qualifier_score"
            ],
            calibration.continuity_score_quantiles[0],
        )
        self.assertFalse(
            learned_passes_handcrafted_fails["accent_supported"]
        )

        learned_fails_handcrafted_passes = (
            _boundary_motion_amount_plus_continuity_evidence(
                frame=replace(
                    _frames(1, 2)[0],
                    source_continuity=(outgoing, outgoing),
                ),
                outgoing_source_index=0,
                incoming_source_index=1,
                derivative_family_ids=(0, 1),
                continuity_score=-1.0,
                config=config,
            )
        )
        self.assertEqual(
            -1.0,
            learned_fails_handcrafted_passes[
                "primary_boundary_continuity_score"
            ],
        )
        self.assertGreaterEqual(
            learned_fails_handcrafted_passes[
                "handcrafted_continuity_qualifier_score"
            ],
            calibration.continuity_score_quantiles[0],
        )
        self.assertTrue(learned_fails_handcrafted_passes["accent_supported"])

        audit = _highlight_motion_calibration_audit(config)
        self.assertEqual(
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            audit["primary_boundary_continuity"]["authority"],
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_AUTHORITY,
            audit["motion_continuity_qualifier"]["authority"],
        )
        self.assertTrue(audit["soft_bonus_active"])

    def test_zero_observed_sharpness_cannot_activate_a_profile_threshold(self) -> None:
        calibration = _actual_highlight_motion_calibration()
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "no calibrated observations",
        ):
            StageEditSemiMarkovConfig(
                highlight_motion_calibration=calibration,
                highlight_motion_boundary_minimum_pose_motion=(
                    calibration.pose_motion_minimum_quantiles[0]
                ),
                highlight_motion_boundary_minimum_continuity_score=(
                    calibration.continuity_score_quantiles[0]
                ),
                highlight_motion_boundary_minimum_sharpness=0.5,
            )

    def test_semimarkov_highlight_policy_identity_is_exact_and_immutable(self) -> None:
        identity = {
            "analysis_basis": "ive_human_reference_visual_montage_grammar_v2",
            "schema_version": 2,
            "path": "/tmp/ive-professional-montage-v2.json",
            "file_sha256": "a" * 64,
            "content_sha256": "b" * 64,
        }
        binding = close_burst_policy_binding_sha256(
            profile_identity=identity,
            policy_fingerprint_sha256="c" * 64,
            minimum_shot_count=2.0,
            minimum_duration_seconds=2.1503940144767864,
            minimum_source_switch_fraction=2.0 / 3.0,
            event_reference_count=6,
        )
        config = StageEditSemiMarkovConfig(
            highlight_policy_profile_analysis_basis=identity["analysis_basis"],
            highlight_policy_profile_schema_version=identity["schema_version"],
            highlight_policy_profile_path=identity["path"],
            highlight_policy_profile_file_sha256=identity["file_sha256"],
            highlight_policy_profile_content_sha256=identity["content_sha256"],
            highlight_policy_fingerprint_sha256="c" * 64,
            highlight_minimum_close_burst_shot_count=2.0,
            highlight_minimum_close_burst_duration_seconds=2.1503940144767864,
            highlight_minimum_close_burst_source_switch_fraction=2.0 / 3.0,
            highlight_close_burst_event_reference_count=6,
            highlight_close_burst_policy_binding_sha256=binding,
        )

        provenance = _highlight_policy_provenance(config)

        self.assertTrue(provenance["profile_identity_bound"])
        self.assertEqual("c" * 64, provenance["policy_fingerprint_sha256"])
        self.assertEqual(
            "/tmp/ive-professional-montage-v2.json",
            provenance["profile_identity"]["path"],
        )
        self.assertEqual(
            "minimum_close_duration_fraction",
            provenance["metrics"]["local_closeup_duration_minimum"][
                "professional_montage_policy_member"
            ],
        )
        self.assertEqual(
            binding,
            provenance["close_burst_soft_policy_binding"]["binding_sha256"],
        )

        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "missing its close-burst",
        ):
            replace(
                config,
                highlight_close_burst_policy_binding_sha256=None,
            )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "binding SHA-256 mismatch",
        ):
            replace(
                config,
                highlight_minimum_close_burst_duration_seconds=2.2,
            )

    def test_close_only_burst_is_conditional_soft_evidence_not_a_quota(self) -> None:
        identity = {
            "analysis_basis": "ive_human_reference_visual_montage_grammar_v2",
            "schema_version": 2,
            "path": "/tmp/ive-professional-montage-v2.json",
            "file_sha256": "a" * 64,
            "content_sha256": "b" * 64,
        }
        values = {
            "minimum_shot_count": 2.0,
            "minimum_duration_seconds": 2.1503940144767864,
            "minimum_source_switch_fraction": 2.0 / 3.0,
            "event_reference_count": 6,
        }
        binding = close_burst_policy_binding_sha256(
            profile_identity=identity,
            policy_fingerprint_sha256="c" * 64,
            **values,
        )
        config = StageEditSemiMarkovConfig(
            highlight_policy_profile_analysis_basis=identity["analysis_basis"],
            highlight_policy_profile_schema_version=identity["schema_version"],
            highlight_policy_profile_path=identity["path"],
            highlight_policy_profile_file_sha256=identity["file_sha256"],
            highlight_policy_profile_content_sha256=identity["content_sha256"],
            highlight_policy_fingerprint_sha256="c" * 64,
            highlight_minimum_close_burst_shot_count=values["minimum_shot_count"],
            highlight_minimum_close_burst_duration_seconds=values[
                "minimum_duration_seconds"
            ],
            highlight_minimum_close_burst_source_switch_fraction=values[
                "minimum_source_switch_fraction"
            ],
            highlight_close_burst_event_reference_count=6,
            highlight_close_burst_policy_binding_sha256=binding,
        )

        def decision(index: int, shot_size: str, family: int) -> SimpleNamespace:
            return SimpleNamespace(
                shot_index=index,
                shot_size=shot_size,
                derivative_family_id=family,
                duration_seconds=1.1,
                master_start_seconds=index * 1.1,
                master_end_seconds=(index + 1) * 1.1,
                boundary_motion_amount_plus_continuity_supported=True,
                boundary_pose_phase_match_proven=True,
                boundary_pose_direction_or_velocity_proven=False,
            )

        absent = _close_only_multi_source_burst_evidence(
            (
                decision(0, "medium", 0),
                decision(1, "closeup", 1),
                decision(2, "medium", 2),
            ),
            config,
        )
        self.assertEqual(0, absent["candidate_event_count"])
        self.assertEqual(0.0, absent["saturated_soft_preference_score"])
        self.assertFalse(absent["absence_is_hard_failure"])

        qualified = _close_only_multi_source_burst_evidence(
            (
                decision(0, "medium", 0),
                decision(1, "closeup", 1),
                decision(2, "extreme_closeup", 2),
                decision(3, "closeup", 3),
                decision(4, "full", 0),
            ),
            config,
        )
        self.assertEqual(1, qualified["qualified_event_count"])
        self.assertEqual(1.0, qualified["saturated_soft_preference_score"])
        event = qualified["events"][0]
        self.assertEqual(3, event["shot_count"])
        self.assertEqual(2, event["source_switch_count"])
        self.assertEqual(2, event["same_time_pose_phase_matched_switch_count"])
        self.assertEqual(0, event["temporal_direction_proven_switch_count"])

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
                enforce_montage_grammar=False,
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

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_renderer_owned_logits_cannot_change_path_or_fabricate_effects(
        self, _mocked
    ) -> None:
        config = StageEditSemiMarkovConfig(
            beam_width=16,
            timeline_count=8,
            minimum_unique_sources=3,
            enforce_montage_grammar=False,
            transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
            transition_score_weight=0.0,
        )
        original = _frames(6, 4)
        adversarial = tuple(
            replace(
                frame,
                transition_logits=(-1000.0, 1000.0, 900.0, 800.0),
            )
            for frame in original
        )
        first = _decode(frames=original, config=config)
        second = _decode(frames=adversarial, config=config)
        self.assertEqual(
            tuple(tuple(item.signature() for item in timeline.decisions) for timeline in first.timelines),
            tuple(tuple(item.signature() for item in timeline.decisions) for timeline in second.timelines),
        )
        self.assertEqual(
            tuple(timeline.score for timeline in first.timelines),
            tuple(timeline.score for timeline in second.timelines),
        )
        self.assertTrue(
            all(
                decision.transition_label in {"initial", "hard_cut"}
                for timeline in first.timelines
                for decision in timeline.decisions
            )
        )
        self.assertEqual(
            RENDERER_OWNED_TRANSITION_AUTHORITY,
            first.audit.input_summary["transition_authority"]["mode"],
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError, "only hard_cut metadata"
        ):
            _decode(
                frames=original,
                config=config,
                constraints=(
                    StageEditSectionConstraint(
                        section_index=0,
                        allowed_transition_labels=("zoom_match",),
                    ),
                ),
            )
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
                self.assertEqual(11, len(decision.score_components))
                self.assertIn(
                    "boundary_continuity",
                    {item.name for item in decision.score_components},
                )
                self.assertIn(
                    "montage_grammar",
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

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_all_wide_highlight_is_rejected_when_closeups_are_available(
        self, _mocked
    ) -> None:
        plan = SectionPlan(
            (
                replace(
                    _section(0, 0.0, 6.0),
                    establish_with_wide_shot=True,
                ),
            )
        )
        # 0=wide, 3=full, 5=wide.  Medium/close/extreme-close candidates are
        # available at every beat, but a deliberately collapsed policy ranks
        # only wides.  The professional grammar must fail instead of blessing
        # the visually flat result.
        frames = self._forced_frames((0, 3, 5, 0, 3, 5))
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "highlight_(closeup|nonwide)_fraction_below_floor",
        ):
            _decode(
                duration=6,
                source_count=6,
                frames=frames,
                metadata=_metadata(6, 6.0),
                plan=plan,
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=3,
                    enforce_montage_grammar=True,
                ),
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_formation_anchor_then_short_closeup_burst_passes_and_is_audited(
        self, _mocked
    ) -> None:
        plan = SectionPlan(
            (
                replace(
                    _section(0, 0.0, 6.0),
                    establish_with_wide_shot=True,
                ),
            )
        )
        # wide anchor -> medium -> close, wide reset -> medium -> close.
        frames = self._forced_frames((0, 1, 2, 3, 1, 2))
        result = _decode(
            duration=6,
            source_count=6,
            frames=frames,
            metadata=_metadata(6, 6.0),
            plan=plan,
            config=StageEditSemiMarkovConfig(
                beam_width=32,
                timeline_count=8,
                minimum_unique_sources=3,
                enforce_montage_grammar=True,
                maximum_global_switches_per_minute=60.0,
                transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
                transition_score_weight=0.0,
            ),
            constraints=(
                StageEditSectionConstraint(
                    section_index=0,
                    minimum_shot_seconds=1.0,
                    maximum_shot_seconds=1.0,
                    maximum_switches_per_minute=60.0,
                ),
            ),
        )

        self.assertTrue(result.timelines)
        grammar = result.timelines[0].montage_grammar
        self.assertTrue(grammar["passed"])
        self.assertEqual(
            "ive_human_reference_wide_transition_floor_plus_"
            "exact_section_and_phrase_wide_anchor_plus_formation_evidenced_"
            "highlight_burst_reset_reference_cadence_and_close_burst_"
            "soft_target_v7",
            grammar["contract"],
        )
        section = grammar["sections"][0]
        motion_audit = section["motion_amount_plus_continuity_boundaries"]
        self.assertEqual(
            "exact_source_pair_pose_motion_plus_independent_handcrafted_"
            "continuity_qualifier_plus_optional_same_time_pose_phase_cosine_"
            "and_sharpness_v3",
            motion_audit["evidence_contract"],
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_AUTHORITY,
            motion_audit["motion_continuity_qualifier"]["authority"],
        )
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            motion_audit["motion_continuity_qualifier"]["scorer_fingerprint"],
        )
        self.assertFalse(
            motion_audit["motion_continuity_qualifier"]
            ["primary_boundary_score_reused"]
        )
        self.assertEqual(
            motion_audit["minimum_handcrafted_continuity_qualifier_score"],
            motion_audit["minimum_continuity_score"],
        )
        self.assertEqual(
            "minimum_handcrafted_continuity_qualifier_score",
            motion_audit["minimum_continuity_score_legacy_alias_semantics"],
        )
        self.assertTrue(section["formation_wide_anchor_present"])
        self.assertGreaterEqual(section["closeup_shot_fraction"], 0.18)
        self.assertGreaterEqual(section["nonwide_shot_fraction"], 0.35)
        self.assertGreaterEqual(section["nonwide_burst_boundary_count"], 2)
        self.assertGreaterEqual(
            section["source_switch_rate_per_minute"],
            18.642084765823732,
        )
        self.assertGreaterEqual(
            section["wide_involved_source_switch_fraction"],
            0.37333333333333335,
        )
        self.assertIn("montage_grammar", result.timelines[0].score_components)
        first_two = result.timelines[0].decisions[:2]
        self.assertEqual("wide", first_two[0].shot_size)
        self.assertGreater(first_two[0].formation_pose_coverage, 0.0)
        self.assertNotEqual(
            first_two[0].derivative_family_id,
            first_two[1].derivative_family_id,
        )
        self.assertIn(first_two[1].shot_size, {"medium", "closeup"})
        self.assertEqual(
            "source_switch_entry_then_nonwide_close_or_medium_burst_not_hold",
            result.timelines[0].edit.candidate_history[0][
                "transition_authority"
            ]["wide_anchor_editorial_role"],
        )
        self.assertEqual(
            0.8,
            grammar["policy"]["chorus_wide_involved_family_switch_prior"][
                "reference_median"
            ],
        )
        self.assertEqual(
            17,
            grammar["policy"]["chorus_wide_involved_family_switch_prior"][
                "eligible_human_reference_count"
            ],
        )
        self.assertEqual(
            0.8,
            section["wide_involved_source_switch_preferred_target"],
        )
        self.assertFalse(
            section[
                "wide_involved_source_switch_preferred_target_is_hard_gate"
            ]
        )
        self.assertGreaterEqual(
            section["wide_involved_source_switch_preferred_target_deviation"],
            0.0,
        )
        cadence = grammar["policy"]["chorus_source_switch_cadence"]
        self.assertEqual(20, cadence["eligible_human_reference_count"])
        self.assertEqual(
            18.642084765823732,
            cadence["product_minimum_per_minute"],
        )
        self.assertEqual(
            23.505554096533405,
            cadence["soft_preferred_target_per_minute"],
        )
        self.assertFalse(cadence["product_minimum_is_reference_floor"])
        aggregate = grammar["highlight_aggregate"]
        self.assertTrue(aggregate["passed"])
        self.assertGreaterEqual(
            aggregate["observed"]["closeup_duration_fraction"],
            aggregate["hard_targets"][
                "pooled_product_minimum_closeup_duration_fraction"
            ],
        )
        self.assertGreaterEqual(
            aggregate["observed"]["nonwide_duration_fraction"],
            aggregate["hard_targets"][
                "pooled_product_minimum_nonwide_duration_fraction"
            ],
        )
        self.assertIn(
            "highlight_aggregate_soft_target",
            result.timelines[0].score_components,
        )
        self.assertFalse(
            aggregate["soft_targets"]["over_close_past_target_is_rewarded"]
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_candidate_support_below_local_and_pooled_need_fails_before_render(
        self, _mocked
    ) -> None:
        frames = tuple(
            replace(
                frame,
                hard_source_gate_mask=tuple(
                    source_index in {0, 3, 5} for source_index in range(6)
                ),
            )
            for frame in self._forced_frames((0, 3, 5, 0, 3, 5))
        )
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "highlight_pooled_closeup_candidate_support_insufficient",
        ):
            _decode(
                duration=6,
                source_count=6,
                frames=frames,
                metadata=_metadata(6, 6.0),
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=3,
                    enforce_montage_grammar=True,
                    maximum_same_shot_size_run=6,
                    recent_shot_size_window=6,
                    maximum_global_switches_per_minute=60.0,
                ),
                constraints=(
                    StageEditSectionConstraint(
                        section_index=0,
                        minimum_shot_seconds=1.0,
                        maximum_shot_seconds=1.0,
                        maximum_switches_per_minute=60.0,
                    ),
                ),
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_pooled_highlight_cannot_rescue_local_p10_failure(
        self, _mocked
    ) -> None:
        sequence = (0, 1, 2, 3, 5, 0, 3, 2, 4, 8, 7, 9)
        frames = tuple(
            replace(
                frame,
                hard_source_gate_mask=tuple(
                    source_index in {sequence[index], 11, 12}
                    for source_index in range(13)
                ),
                section_progress=(index % 6) / 5.0,
            )
            for index, frame in enumerate(
                self._forced_frames(sequence, source_count=13)
            )
        )
        plan = SectionPlan(
            (_section(0, 0.0, 6.0), _section(1, 6.0, 12.0))
        )
        constraints = tuple(
            StageEditSectionConstraint(
                section_index=index,
                minimum_shot_seconds=1.0,
                maximum_shot_seconds=1.0,
                maximum_switches_per_minute=60.0,
            )
            for index in range(2)
        )
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "highlight_closeup_fraction_below_floor",
        ):
            _decode(
                duration=12,
                source_count=13,
                frames=frames,
                metadata=_metadata(
                    13,
                    12.0,
                    internal_cuts={
                        11: tuple(0.5 + index for index in range(12)),
                        12: tuple(0.5 + index for index in range(12)),
                    },
                ),
                plan=plan,
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=3,
                    enforce_montage_grammar=True,
                    maximum_global_switches_per_minute=60.0,
                ),
                constraints=constraints,
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_human_maximum_consecutive_wide_run_is_a_hard_gate(
        self, _mocked
    ) -> None:
        sequence = (
            0,
            3,
            5,
            8,
            10,
            0,
            3,
            5,
            8,
            1,
            2,
            6,
            4,
            10,
            12,
            7,
            1,
            10,
            6,
            2,
        )
        frames = tuple(
            replace(
                frame,
                hard_source_gate_mask=tuple(
                    source_index == sequence[index]
                    for source_index in range(13)
                ),
            )
            for index, frame in enumerate(
                self._forced_frames(sequence, source_count=13)
            )
        )
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "highlight_maximum_consecutive_wide_run_exceeded",
        ):
            _decode(
                duration=20,
                source_count=13,
                frames=frames,
                metadata=_metadata(13, 20.0),
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=3,
                    enforce_montage_grammar=True,
                    maximum_same_shot_size_run=12,
                    recent_shot_size_window=12,
                    maximum_global_switches_per_minute=60.0,
                ),
                constraints=(
                    StageEditSectionConstraint(
                        section_index=0,
                        minimum_shot_seconds=1.0,
                        maximum_shot_seconds=1.0,
                        maximum_switches_per_minute=60.0,
                    ),
                ),
            )

    def test_high_pose_motion_alone_cannot_claim_compatible_closeup_cut(
        self,
    ) -> None:
        config = StageEditSemiMarkovConfig(
            highlight_motion_boundary_minimum_pose_motion=0.5,
            highlight_motion_boundary_minimum_continuity_score=-0.10,
            highlight_motion_boundary_minimum_sharpness=0.5,
            highlight_minimum_motion_closeup_switch_boundaries=1,
        )
        frame = replace(
            _frames(1, 2)[0],
            source_continuity=(
                StageEditSourceContinuity(
                    pose_motion=0.9,
                    camera_motion_x=0.8,
                    camera_motion_y=0.0,
                    sharpness=0.9,
                ),
                StageEditSourceContinuity(
                    pose_motion=0.9,
                    camera_motion_x=-0.8,
                    camera_motion_y=0.0,
                    sharpness=0.9,
                ),
            ),
        )
        incompatible = _boundary_motion_amount_plus_continuity_evidence(
            frame=frame,
            outgoing_source_index=0,
            incoming_source_index=1,
            derivative_family_ids=(0, 1),
            continuity_score=-0.8,
            config=config,
        )
        self.assertTrue(incompatible["evidence_complete"])
        self.assertFalse(incompatible["accent_supported"])

        blurred = replace(
            frame,
            source_continuity=(
                frame.source_continuity[0],
                replace(frame.source_continuity[1], sharpness=0.1),
            ),
        )
        blur_rejected = _boundary_motion_amount_plus_continuity_evidence(
            frame=blurred,
            outgoing_source_index=0,
            incoming_source_index=1,
            derivative_family_ids=(0, 1),
            continuity_score=-0.01,
            config=config,
        )
        self.assertTrue(blur_rejected["evidence_complete"])
        self.assertFalse(blur_rejected["accent_supported"])

    def test_pose_phase_soft_qualification_needs_complete_same_time_pair(
        self,
    ) -> None:
        calibration = _actual_highlight_motion_calibration()
        config = StageEditSemiMarkovConfig(
            highlight_motion_calibration=calibration,
            highlight_motion_boundary_minimum_pose_motion=(
                calibration.pose_motion_minimum_quantiles[0]
            ),
            highlight_motion_boundary_minimum_continuity_score=(
                calibration.continuity_score_quantiles[0]
            ),
            highlight_motion_boundary_minimum_pose_phase_cosine=(
                calibration.pose_embedding_cosine_quantiles[0]
            ),
        )
        same_phase = (1.0,) + (0.0,) * 59
        opposite_phase = (-1.0,) + (0.0,) * 59

        def evidence(
            incoming_pose: tuple[float, ...] | None,
            *,
            outgoing_motion: float = 0.8,
            incoming_motion: float = 0.8,
            families: tuple[int, int] = (0, 1),
        ) -> dict[str, object]:
            frame = replace(
                _frames(1, 2)[0],
                source_continuity=(
                    StageEditSourceContinuity(
                        pose_motion=outgoing_motion,
                        pose_model_embedding=same_phase,
                    ),
                    StageEditSourceContinuity(
                        pose_motion=incoming_motion,
                        pose_model_embedding=incoming_pose,
                    ),
                ),
            )
            return _boundary_motion_amount_plus_continuity_evidence(
                frame=frame,
                outgoing_source_index=0,
                incoming_source_index=1,
                derivative_family_ids=families,
                continuity_score=-0.01,
                config=config,
            )

        missing = evidence(None)
        self.assertFalse(missing["pose_phase_evidence_observed"])
        self.assertFalse(missing["evidence_complete"])
        self.assertFalse(missing["pose_phase_match_proven"])
        self.assertFalse(missing["accent_supported"])

        opposite = evidence(opposite_phase)
        self.assertAlmostEqual(-1.0, opposite["pose_phase_cosine"])
        self.assertTrue(opposite["pose_phase_evidence_observed"])
        self.assertTrue(opposite["evidence_complete"])
        self.assertFalse(opposite["pose_phase_match_proven"])
        self.assertFalse(opposite["accent_supported"])

        matched = evidence(same_phase)
        self.assertAlmostEqual(1.0, matched["pose_phase_cosine"])
        self.assertTrue(matched["pose_phase_match_proven"])
        self.assertTrue(matched["accent_supported"])
        self.assertFalse(matched["pose_direction_or_velocity_proven"])

        phase_without_motion = evidence(
            same_phase,
            outgoing_motion=0.0,
            incoming_motion=0.0,
        )
        self.assertTrue(phase_without_motion["pose_phase_match_proven"])
        self.assertFalse(phase_without_motion["accent_supported"])

        same_family = evidence(same_phase, families=(3, 3))
        self.assertIsNone(same_family["pose_phase_cosine"])
        self.assertFalse(same_family["pose_phase_match_proven"])
        self.assertFalse(same_family["accent_supported"])

    def test_sparse_temporal_sidecars_are_audit_only_without_cut_alignment(
        self,
    ) -> None:
        calibration = _actual_highlight_motion_calibration()
        config = StageEditSemiMarkovConfig(
            highlight_motion_calibration=calibration,
            highlight_motion_boundary_minimum_pose_motion=(
                calibration.pose_motion_minimum_quantiles[0]
            ),
            highlight_motion_boundary_minimum_continuity_score=(
                calibration.continuity_score_quantiles[0]
            ),
        )
        outgoing_frame = replace(
            _frames(1, 2)[0],
            source_continuity=(
                StageEditSourceContinuity(
                    pose_motion=0.8,
                    pose_model_embedding=_pose60(0.1),
                    temporal_pose_sidecar=_temporal_sidecar(0.9, 0.0),
                ),
                StageEditSourceContinuity(pose_motion=0.8),
            ),
        )

        def evidence(
            outgoing_sidecar: StageEditTemporalPoseSidecar | None,
            incoming_sidecar: StageEditTemporalPoseSidecar | None,
        ) -> dict[str, object]:
            incoming_frame = replace(
                _frames(1, 2)[0],
                source_continuity=(
                    StageEditSourceContinuity(
                        pose_motion=0.8,
                        pose_model_embedding=_pose60(0.1),
                        temporal_pose_sidecar=outgoing_sidecar,
                    ),
                    StageEditSourceContinuity(
                        pose_motion=0.8,
                        pose_model_embedding=_pose60(0.1),
                        temporal_pose_sidecar=incoming_sidecar,
                    ),
                ),
            )
            return _boundary_motion_amount_plus_continuity_evidence(
                frame=incoming_frame,
                outgoing_frame=outgoing_frame,
                outgoing_source_index=0,
                incoming_source_index=1,
                derivative_family_ids=(0, 1),
                continuity_score=-0.01,
                config=config,
            )

        moving = evidence(
            _temporal_sidecar(0.0, 0.1),
            _temporal_sidecar(0.1, 0.2),
        )
        self.assertFalse(moving["temporal_pose_motion_thresholds_bound"])
        self.assertFalse(moving["temporal_pose_motion_input_valid"])
        self.assertEqual(
            moving["temporal_classification"], "boundary_alignment_unproven"
        )
        self.assertIn(
            "boundary_alignment_unproven",
            moving["temporal_failure_reasons"],
        )
        self.assertFalse(moving["temporal_dynamic_pose_phase_match"])
        self.assertFalse(moving["pose_direction_or_velocity_proven"])
        # Exact same-time endpoint pose and independently calibrated
        # both-moving/continuity evidence remain a safe non-directional soft
        # preference even though sparse sidecars prove no phase direction.
        self.assertTrue(moving["accent_supported"])
        self.assertEqual(0.0, moving["temporal_confidence"])
        self.assertIsNone(moving["temporal_endpoint_pose_cosine"])
        self.assertIsNone(moving["temporal_velocity_direction_cosine"])
        self.assertIsNone(moving["temporal_phase_compatibility"])
        self.assertEqual(60, moving["pose_phase_feature_dimension"])
        self.assertEqual(60, moving["temporal_feature_dimension"])
        self.assertTrue(moving["temporal_outgoing_sidecar_observed"])
        self.assertTrue(moving["temporal_incoming_sidecar_observed"])
        for key in (
            "pose_phase_outgoing_content_sha256",
            "pose_phase_incoming_content_sha256",
            "pose_phase_pair_content_sha256",
            "temporal_outgoing_content_sha256",
            "temporal_incoming_content_sha256",
            "temporal_pair_content_sha256",
        ):
            self.assertEqual(64, len(moving[key]))

        static = evidence(
            _temporal_sidecar(0.0, 0.0),
            _temporal_sidecar(0.1, 0.1),
        )
        opposite = evidence(
            _temporal_sidecar(0.0, 0.1),
            _temporal_sidecar(0.1, 0.0),
        )
        for unqualified in (static, opposite):
            self.assertEqual(
                "boundary_alignment_unproven",
                unqualified["temporal_classification"],
            )
            self.assertFalse(unqualified["temporal_static_pose_match"])
            self.assertFalse(unqualified["pose_direction_or_velocity_proven"])
            self.assertIsNone(
                unqualified["temporal_velocity_direction_cosine"]
            )

        missing = evidence(_temporal_sidecar(0.0, 0.1), None)
        self.assertFalse(missing["temporal_pose_motion_input_valid"])
        self.assertFalse(missing["pose_direction_or_velocity_proven"])
        self.assertEqual(
            "temporal_pose_sidecar_missing",
            missing["temporal_classification"],
        )
        self.assertIn(
            "boundary_alignment_unproven",
            missing["temporal_failure_reasons"],
        )
        self.assertIn(
            "temporal_incoming_sidecar_missing",
            missing["temporal_failure_reasons"],
        )

        with patch(
            "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
            return_value=_ready_report(),
        ):
            decoded = _decode()
        decision = replace(
            decoded.timelines[0].decisions[0],
            boundary_pose_phase_feature_dimension=(
                moving["pose_phase_feature_dimension"]
            ),
            boundary_pose_phase_outgoing_observed=True,
            boundary_pose_phase_incoming_observed=True,
            boundary_pose_phase_trusted_producer_contract=(
                moving["pose_phase_trusted_producer_contract"]
            ),
            boundary_pose_phase_outgoing_content_sha256=(
                moving["pose_phase_outgoing_content_sha256"]
            ),
            boundary_pose_phase_incoming_content_sha256=(
                moving["pose_phase_incoming_content_sha256"]
            ),
            boundary_pose_phase_pair_content_sha256=(
                moving["pose_phase_pair_content_sha256"]
            ),
            boundary_temporal_feature_dimension=(
                moving["temporal_feature_dimension"]
            ),
            boundary_temporal_outgoing_sidecar_observed=True,
            boundary_temporal_incoming_sidecar_observed=True,
            boundary_temporal_outgoing_trusted_producer_contract=(
                moving["temporal_outgoing_trusted_producer_contract"]
            ),
            boundary_temporal_incoming_trusted_producer_contract=(
                moving["temporal_incoming_trusted_producer_contract"]
            ),
            boundary_temporal_outgoing_content_sha256=(
                moving["temporal_outgoing_content_sha256"]
            ),
            boundary_temporal_incoming_content_sha256=(
                moving["temporal_incoming_content_sha256"]
            ),
            boundary_temporal_pair_content_sha256=(
                moving["temporal_pair_content_sha256"]
            ),
        )
        decision_payload = json.loads(json.dumps(decision.to_dict()))
        _assert_no_raw_pose_vectors(self, decision_payload)
        temporal_payload = decision_payload[
            "boundary_motion_amount_plus_continuity"
        ]["temporal_pose_motion"]
        self.assertEqual(
            moving["temporal_pair_content_sha256"],
            temporal_payload["pair_content_sha256"],
        )
        poisoned_audit = replace(
            decoded.audit,
            input_summary={
                **decoded.audit.input_summary,
                "pose_embedding": [0.0] * 60,
            },
        )
        with self.assertRaisesRegex(ValueError, "forbidden raw pose vector"):
            poisoned_audit.to_dict()

        audit = _highlight_motion_calibration_audit(config)
        self.assertFalse(audit["temporal_pose_motion"]["calibration_bound"])
        self.assertFalse(audit["temporal_pose_motion"]["activation_available"])
        self.assertFalse(
            audit["temporal_pose_motion"]["boundary_alignment_proven"]
        )
        self.assertTrue(
            audit["temporal_pose_motion"]
            ["sparse_sidecars_are_scalar_hash_audit_only"]
        )
        self.assertEqual(audit["temporal_pose_motion"]["new_hard_boundary_quota"], 0)
        self.assertFalse(
            audit["temporal_pose_motion"]["static_pose_match_can_receive_dynamic_bonus"]
        )
        self.assertEqual(
            StageEditSemiMarkovConfig(**asdict(config)),
            config,
        )

    def test_temporal_direction_is_unbound_by_default_and_needs_separate_calibration(
        self,
    ) -> None:
        default = StageEditSemiMarkovConfig()
        self.assertIsNone(default.highlight_temporal_pose_motion_calibration)
        audit = _highlight_motion_calibration_audit(default)
        self.assertFalse(audit["temporal_pose_motion"]["calibration_bound"])
        self.assertFalse(audit["temporal_direction_proof_capability_bound"])
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "immutable evidence loader/registry",
        ):
            StageEditSemiMarkovConfig(
                highlight_temporal_pose_motion_calibration=(
                    StageEditTemporalPoseMotionCalibration(
                        analysis_basis="separate_temporal_evidence",
                        source_evidence_sha256="e" * 64,
                        observed_boundary_count=10,
                        thresholds=TemporalPoseMotionThresholds(
                            minimum_boundary_motion_magnitude=0.1,
                            minimum_endpoint_pose_cosine=0.8,
                            minimum_velocity_direction_cosine=0.5,
                            minimum_phase_compatibility=0.7,
                            pose_norm_epsilon=1.0e-9,
                        ),
                    )
                )
            )

    def test_pose_phase_threshold_is_calibrated_optional_and_not_a_hard_default(
        self,
    ) -> None:
        calibration = _actual_highlight_motion_calibration()
        self.assertIsNone(
            StageEditSemiMarkovConfig()
            .highlight_motion_boundary_minimum_pose_phase_cosine
        )
        self.assertEqual(
            0,
            StageEditSemiMarkovConfig()
            .highlight_minimum_motion_closeup_switch_boundaries,
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "immutable motion calibration",
        ):
            StageEditSemiMarkovConfig(
                highlight_motion_boundary_minimum_pose_motion=0.5,
                highlight_motion_boundary_minimum_continuity_score=-0.1,
                highlight_motion_boundary_minimum_pose_phase_cosine=0.8,
            )

        configured = StageEditSemiMarkovConfig(
            highlight_motion_calibration=calibration,
            highlight_motion_boundary_minimum_pose_motion=(
                calibration.pose_motion_minimum_quantiles[0]
            ),
            highlight_motion_boundary_minimum_continuity_score=(
                calibration.continuity_score_quantiles[0]
            ),
            highlight_motion_boundary_minimum_pose_phase_cosine=(
                calibration.pose_embedding_cosine_quantiles[0]
            ),
        )
        round_tripped = StageEditSemiMarkovConfig(**asdict(configured))
        self.assertEqual(configured, round_tripped)
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "not a calibrated quantile",
        ):
            StageEditSemiMarkovConfig(
                highlight_motion_calibration=calibration,
                highlight_motion_boundary_minimum_pose_motion=(
                    calibration.pose_motion_minimum_quantiles[0]
                ),
                highlight_motion_boundary_minimum_continuity_score=(
                    calibration.continuity_score_quantiles[0]
                ),
                highlight_motion_boundary_minimum_pose_phase_cosine=0.8,
            )

    def test_pose_serialization_is_scalar_hash_only_and_bounds_are_explicit(
        self,
    ) -> None:
        pose = (1.0,) + (0.0,) * 59
        continuity = StageEditSourceContinuity(
            pose_motion=0.5,
            pose_model_embedding=pose,
        )
        payload = continuity.to_dict()
        evidence = payload["same_time_pose_evidence"]
        self.assertTrue(evidence["observed"])
        self.assertEqual(60, evidence["feature_dimension"])
        self.assertEqual(64, len(evidence["content_sha256"]))
        self.assertNotIn("pose_model_embedding", payload)
        missing = StageEditSourceContinuity().to_dict()[
            "same_time_pose_evidence"
        ]
        self.assertFalse(missing["observed"])
        self.assertIsNone(missing["content_sha256"])
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "exactly 60",
        ):
            StageEditSourceContinuity(pose_model_embedding=(1.0,))
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            r"lie in \[-1, 1\]",
        ):
            StageEditSourceContinuity(
                pose_model_embedding=(1.01,) + (0.0,) * 59
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_two_shot_wide_to_close_highlight_cannot_bypass_burst_gate(
        self, _mocked
    ) -> None:
        plan = SectionPlan(
            (
                replace(
                    _section(0, 0.0, 6.0),
                    establish_with_wide_shot=True,
                ),
            )
        )
        frames = self._forced_frames((0, 1, 1, 2, 1, 1))
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "highlight_nonwide_burst_missing",
        ):
            _decode(
                duration=6,
                source_count=6,
                frames=frames,
                metadata=_metadata(6, 6.0),
                plan=plan,
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=2,
                    enforce_montage_grammar=True,
                ),
                constraints=(
                    StageEditSectionConstraint(
                        section_index=0,
                        minimum_shot_seconds=3.0,
                        maximum_shot_seconds=3.0,
                        maximum_switches_per_minute=20.0,
                    ),
                ),
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_flat_highlight_cannot_pass_with_only_fractional_shot_quotas(
        self, _mocked
    ) -> None:
        plan = SectionPlan(
            (
                replace(
                    _section(0, 0.0, 12.0),
                    establish_with_wide_shot=True,
                ),
            )
        )
        # The selected 3-second shots are wide -> medium -> close -> wide.
        # This satisfies close/non-wide fractions, a multi-source non-wide
        # burst, a formation reset, and wide-switch structure, but only three
        # boundaries in twelve seconds (15/min).  It must no longer pass as a
        # lively choreography highlight merely by meeting composition quotas.
        frames = self._forced_frames(
            (0, 4, 5, 1, 4, 5, 2, 4, 5, 3, 4, 5)
        )
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "highlight_source_switch_rate_below_product_minimum",
        ):
            _decode(
                duration=12,
                source_count=6,
                frames=frames,
                metadata=_metadata(6, 12.0),
                plan=plan,
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=3,
                    enforce_montage_grammar=True,
                    maximum_global_switches_per_minute=60.0,
                    highlight_burst_maximum_shot_seconds=3.0,
                    highlight_minimum_nonwide_burst_boundaries=1,
                    transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
                    transition_score_weight=0.0,
                ),
                constraints=(
                    StageEditSectionConstraint(
                        section_index=0,
                        minimum_shot_seconds=3.0,
                        maximum_shot_seconds=3.0,
                        maximum_switches_per_minute=60.0,
                    ),
                ),
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_soft_target_does_not_replace_p10_wide_switch_hard_floor(
        self, _mocked
    ) -> None:
        source_sequence = (0, 1, 2, 6, 7, 4, 3)
        frames = tuple(
            replace(
                frame,
                hard_source_gate_mask=tuple(
                    source_index == source_sequence[time_index]
                    for source_index in range(8)
                ),
            )
            for time_index, frame in enumerate(
                self._forced_frames(source_sequence, source_count=8)
            )
        )
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "highlight_wide_involved_source_switch_fraction_below_reference_floor",
        ):
            _decode(
                duration=7,
                source_count=8,
                frames=frames,
                metadata=_metadata(8, 7.0),
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=3,
                    enforce_montage_grammar=True,
                    maximum_global_switches_per_minute=60.0,
                ),
                constraints=(
                    StageEditSectionConstraint(
                        section_index=0,
                        minimum_shot_seconds=1.0,
                        maximum_shot_seconds=1.0,
                        maximum_switches_per_minute=60.0,
                    ),
                ),
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_unknown_in_first_three_consumes_exact_early_anchor_slot(
        self, _mocked
    ) -> None:
        source_sequence = (1, 2, 4, 0, 3, 5)
        frames = [
            replace(
                frame,
                hard_source_gate_mask=tuple(
                    source_index == source_sequence[time_index]
                    for source_index in range(6)
                ),
            )
            for time_index, frame in enumerate(
                self._forced_frames(source_sequence)
            )
        ]
        third = frames[2]
        sizes = list(third.source_shot_sizes)
        sizes[4] = "unknown"
        frames[2] = replace(third, source_shot_sizes=tuple(sizes))

        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "highlight_early_wide_anchor_fraction_below_reference_floor",
        ):
            _decode(
                duration=6,
                source_count=6,
                frames=tuple(frames),
                metadata=_metadata(6, 6.0),
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=3,
                    enforce_montage_grammar=True,
                    maximum_global_switches_per_minute=60.0,
                ),
                constraints=(
                    StageEditSectionConstraint(
                        section_index=0,
                        minimum_shot_seconds=1.0,
                        maximum_shot_seconds=1.0,
                        maximum_switches_per_minute=60.0,
                    ),
                ),
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_generic_wide_without_people_evidence_is_not_a_formation_anchor(
        self, _mocked
    ) -> None:
        plan = SectionPlan(
            (
                replace(
                    _section(0, 0.0, 6.0),
                    establish_with_wide_shot=True,
                ),
            )
        )
        frames = tuple(
            replace(
                frame,
                source_continuity=tuple(
                    StageEditSourceContinuity(
                        detected_person_count_normalized=0.0,
                        formation_person_count_normalized=0.0,
                        formation_pose_coverage=0.0,
                    )
                    for _ in frame.source_continuity
                ),
            )
            for frame in self._forced_frames((0, 1, 2, 3, 1, 2))
        )
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "formation_wide_(candidate_evidence_insufficient|anchor_evidence_missing)",
        ):
            _decode(
                duration=6,
                source_count=6,
                frames=frames,
                metadata=_metadata(6, 6.0),
                plan=plan,
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=3,
                    enforce_montage_grammar=True,
                    maximum_global_switches_per_minute=60.0,
                ),
                constraints=(
                    StageEditSectionConstraint(
                        section_index=0,
                        minimum_shot_seconds=1.0,
                        maximum_shot_seconds=1.0,
                        maximum_switches_per_minute=60.0,
                    ),
                ),
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_detected_person_without_formation_pose_is_not_an_anchor(
        self, _mocked
    ) -> None:
        plan = SectionPlan(
            (
                replace(
                    _section(0, 0.0, 6.0),
                    establish_with_wide_shot=True,
                ),
            )
        )
        frames = tuple(
            replace(
                frame,
                source_continuity=tuple(
                    StageEditSourceContinuity(
                        detected_person_count_normalized=1.0,
                        formation_person_count_normalized=None,
                        formation_pose_coverage=None,
                    )
                    for _ in frame.source_continuity
                ),
            )
            for frame in self._forced_frames((0, 1, 2, 3, 1, 2))
        )
        with self.assertRaisesRegex(
            StageEditNoValidTimelineError,
            "formation_wide_(candidate_evidence_insufficient|anchor_evidence_missing)",
        ):
            _decode(
                duration=6,
                source_count=6,
                frames=frames,
                metadata=_metadata(6, 6.0),
                plan=plan,
                config=StageEditSemiMarkovConfig(
                    beam_width=32,
                    timeline_count=8,
                    minimum_unique_sources=3,
                    enforce_montage_grammar=True,
                    maximum_global_switches_per_minute=60.0,
                ),
                constraints=(
                    StageEditSectionConstraint(
                        section_index=0,
                        minimum_shot_seconds=1.0,
                        maximum_shot_seconds=1.0,
                        maximum_switches_per_minute=60.0,
                    ),
                ),
            )

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

    @staticmethod
    def _continuity_authority_frames() -> tuple[StageEditHeadFrame, ...]:
        # Handcrafted geometry prefers source 1 after the forced source-0
        # opening. A camera-motion-only learned scorer prefers source 2.
        continuity = (
            StageEditSourceContinuity(
                face_scale=0.20,
                face_center_x=0.50,
                face_center_y=0.50,
                camera_motion_x=0.0,
                camera_motion_y=0.0,
            ),
            StageEditSourceContinuity(
                face_scale=0.20,
                face_center_x=0.50,
                face_center_y=0.50,
                camera_motion_x=1.0,
                camera_motion_y=1.0,
            ),
            StageEditSourceContinuity(
                face_scale=0.80,
                face_center_x=0.90,
                face_center_y=0.10,
                camera_motion_x=0.0,
                camera_motion_y=0.0,
            ),
            StageEditSourceContinuity(
                face_scale=0.35,
                face_center_x=0.65,
                face_center_y=0.65,
                camera_motion_x=0.70,
                camera_motion_y=0.70,
            ),
        )
        frames = []
        for index, frame in enumerate(_frames(6, 4)):
            frames.append(
                replace(
                    frame,
                    source_logits=(0.0,) * 4,
                    available_source_mask=(True,) * 4,
                    hard_source_gate_mask=(
                        (True, False, False, False)
                        if index == 0
                        else (True,) * 4
                    ),
                    source_continuity=continuity,
                )
            )
        return tuple(frames)

    @staticmethod
    def _continuity_authority_config(authority: str) -> StageEditSemiMarkovConfig:
        return StageEditSemiMarkovConfig(
            beam_width=32,
            timeline_count=8,
            minimum_shot_seconds=1.0,
            maximum_shot_seconds=1.0,
            minimum_unique_sources=3,
            enforce_aba_exclusion=False,
            enforce_montage_grammar=False,
            maximum_global_switches_per_minute=120.0,
            section_cut_density_multiplier=4.0,
            cut_score_weight=0.0,
            source_score_weight=0.0,
            duration_score_weight=0.0,
            transition_score_weight=0.0,
            crop_correction_weight=0.0,
            source_quality_weight=0.0,
            boundary_continuity_weight=1.0,
            cadence_score_weight=0.0,
            diversity_score_weight=0.0,
            occupancy_balance_weight=0.0,
            montage_grammar_score_weight=0.0,
            continuity_authority=authority,
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_learned_source_pair_scorer_changes_candidate_order_and_exact_component(
        self, _mocked
    ) -> None:
        frames = self._continuity_authority_frames()
        learned_scorer = _single_feature_scorer("camera_motion_distance")
        handcrafted = _decode(
            frames=frames,
            config=self._continuity_authority_config(
                HANDCRAFTED_CONTINUITY_AUTHORITY
            ),
        )
        learned = _decode(
            frames=frames,
            config=self._continuity_authority_config(
                PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
            ),
            source_pair_scorer=learned_scorer,
        )

        handcrafted_path = tuple(
            decision.source_index for decision in handcrafted.timelines[0].decisions
        )
        learned_path = tuple(
            decision.source_index for decision in learned.timelines[0].decisions
        )
        self.assertEqual(0, handcrafted_path[0])
        self.assertEqual(0, learned_path[0])
        self.assertEqual(1, handcrafted_path[1])
        self.assertEqual(2, learned_path[1])
        self.assertNotEqual(handcrafted_path, learned_path)

        decisions = learned.timelines[0].decisions
        for decision_index, decision in enumerate(decisions):
            component = next(
                value
                for value in decision.score_components
                if value.name == "boundary_continuity"
            )
            if decision_index == 0:
                expected = 0.0
            else:
                outgoing = decisions[decision_index - 1].source_index
                incoming = decision.source_index
                exact_frame = frames[decision.start_candidate_index]
                expected = learned_scorer.score(
                    exact_frame.source_continuity[outgoing],
                    exact_frame.source_continuity[incoming],
                )
            self.assertAlmostEqual(expected, component.raw_value)
            self.assertAlmostEqual(
                expected * component.weight,
                component.contribution,
            )
            self.assertGreaterEqual(component.raw_value, -1.0)
            self.assertLessEqual(component.raw_value, 0.0)

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_continuity_authority_rejects_missing_or_unexpected_scorer(
        self, _mocked
    ) -> None:
        scorer = StageEditSourcePairScorer.uniform()
        learned_config = self._continuity_authority_config(
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "requires a typed StageEditSourcePairScorer",
        ):
            _decode(config=learned_config)
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "rejects an unexpected learned source-pair scorer",
        ):
            _decode(source_pair_scorer=scorer)
        missing_observations = tuple(
            replace(frame, source_continuity=()) for frame in _frames(6, 4)
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "requires exact canonical source observations",
        ):
            _decode(
                frames=missing_observations,
                config=learned_config,
                source_pair_scorer=scorer,
            )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "configured continuity scorer fingerprint",
        ):
            _decode(
                config=replace(
                    learned_config,
                    continuity_scorer_fingerprint="0" * 64,
                ),
                source_pair_scorer=scorer,
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_learned_continuity_is_neutral_on_same_derivative_family(
        self, _mocked
    ) -> None:
        scorer = _single_feature_scorer("camera_motion_distance")
        frames = list(self._continuity_authority_frames())
        frames[1] = replace(
            frames[1],
            hard_source_gate_mask=(False, True, False, False),
        )
        derivative_metadata = SourceDerivativeFamilyMetadata(
            source_count=4,
            family_ids=(0, 0, 1, 2),
            audit_sha256="d" * 64,
            production_ready=True,
            uncertain_pair_count=0,
            clock_contract_complete=True,
        )
        config = replace(
            self._continuity_authority_config(
                PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
            ),
            allow_consecutive_same_source=True,
        )
        result = _decode(
            frames=tuple(frames),
            config=config,
            derivative_metadata=derivative_metadata,
            source_pair_scorer=scorer,
        )
        opening, same_family = result.timelines[0].decisions[:2]
        self.assertEqual((0, 1), (opening.source_index, same_family.source_index))
        component = next(
            value
            for value in same_family.score_components
            if value.name == "boundary_continuity"
        )
        exact_frame = frames[same_family.start_candidate_index]
        self.assertLess(
            scorer.score(
                exact_frame.source_continuity[0],
                exact_frame.source_continuity[1],
            ),
            0.0,
        )
        self.assertEqual(0.0, component.raw_value)
        self.assertEqual(0.0, component.contribution)

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_learned_continuity_fingerprint_is_deterministic_and_audited(
        self, _mocked
    ) -> None:
        scorer = _single_feature_scorer("camera_motion_distance")
        config = self._continuity_authority_config(
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
        )
        frames = self._continuity_authority_frames()
        first = _decode(
            frames=frames, config=config, source_pair_scorer=scorer
        )
        second = _decode(
            frames=frames, config=config, source_pair_scorer=scorer
        )
        provenance = first.audit.input_summary[
            "boundary_continuity_score_provenance"
        ]
        self.assertEqual(
            CONTINUITY_AUTHORITY_CONTRACT,
            provenance["authority_contract"],
        )
        self.assertEqual(
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            provenance["authority"],
        )
        self.assertEqual(
            STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            provenance["scorer_contract"],
        )
        self.assertEqual(scorer.fingerprint, provenance["scorer_fingerprint"])
        self.assertFalse(provenance["scorer_weights_embedded"])
        self.assertNotIn("weights", provenance)
        self.assertEqual(first.audit.input_sha256, second.audit.input_sha256)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            scorer.fingerprint,
            first.audit.score_provenance["boundary_continuity"][
                "scorer_fingerprint"
            ],
        )
        self.assertEqual(
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            first.audit.config["continuity_authority"],
        )
        self.assertEqual(
            CONTINUITY_AUTHORITY_CONTRACT,
            first.audit.config["continuity_authority_contract"],
        )
        self.assertEqual(
            STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            first.audit.config["continuity_scorer_contract"],
        )
        self.assertEqual(
            scorer.fingerprint,
            first.audit.config["continuity_scorer_fingerprint"],
        )
        for timeline in first.timelines:
            self.assertEqual(
                scorer.fingerprint,
                timeline.boundary_continuity_score_provenance[
                    "scorer_fingerprint"
                ],
            )
            self.assertEqual(
                provenance,
                timeline.to_dict()["score_provenance"]["boundary_continuity"],
            )
            self.assertEqual(
                provenance,
                timeline.edit.candidate_history[0][
                    "boundary_continuity_score_provenance"
                ],
            )

        different = _decode(
            frames=frames,
            config=config,
            source_pair_scorer=StageEditSourcePairScorer.uniform(),
        )
        self.assertNotEqual(
            first.audit.input_sha256, different.audit.input_sha256
        )
        self.assertNotEqual(
            scorer.fingerprint,
            different.audit.score_provenance["boundary_continuity"][
                "scorer_fingerprint"
            ],
        )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_default_handcrafted_authority_is_backward_compatible(
        self, _mocked
    ) -> None:
        common = {
            "beam_width": 16,
            "timeline_count": 8,
            "minimum_unique_sources": 3,
            "enforce_montage_grammar": False,
        }
        implicit = _decode(config=StageEditSemiMarkovConfig(**common))
        explicit = _decode(
            config=StageEditSemiMarkovConfig(
                **common,
                continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            )
        )
        self.assertEqual(implicit.to_dict(), explicit.to_dict())
        provenance = implicit.audit.input_summary[
            "boundary_continuity_score_provenance"
        ]
        self.assertEqual(
            HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            provenance["scorer_contract"],
        )
        self.assertEqual(HANDCRAFTED_CONTINUITY_AUTHORITY, provenance["authority"])
        self.assertEqual(DECODER_CONTRACT, implicit.audit.decoder)

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
                enforce_montage_grammar=False,
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
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "preferred wide-switch target",
        ):
            StageEditSemiMarkovConfig(
                highlight_minimum_wide_involved_source_switch_fraction=0.9,
                highlight_preferred_wide_involved_source_switch_fraction=0.8,
            )
        with self.assertRaisesRegex(
            StageEditDecodeInputError,
            "preferred cadence exceeds",
        ):
            StageEditSemiMarkovConfig(
                maximum_global_switches_per_minute=20.0,
                enforce_montage_grammar=True,
            )
        # Legacy/non-professional callers may deliberately impose a tighter
        # global cap; the IVE professional soft target is inactive there.
        StageEditSemiMarkovConfig(
            maximum_global_switches_per_minute=20.0,
            enforce_montage_grammar=False,
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

    @staticmethod
    def _dynamic_prediction_v3(
        source_index: int,
        *,
        source_id: str | None = None,
        zoom: float | None = None,
    ) -> DynamicCropHeadPredictionV3:
        selected_zoom = zoom if zoom is not None else 1.02 + 0.04 * source_index
        return DynamicCropHeadPredictionV3(
            source_id=source_id or f"source-{source_index}",
            source_index=source_index,
            values=((selected_zoom, 0.5, 0.5, 0.0),) * 8,
            count_logits=(20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0),
            time_interval_logits=(0.0,) * 35,
            motion_logit=0.0,
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

        self.assertEqual(DECODER_SCHEMA_VERSION, result.audit.schema_version)
        self.assertEqual(
            "verified_derivative_family_not_file_source",
            result.audit.input_summary["diversity_semantics"],
        )
        self.assertEqual(
            13,
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

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_dynamic_crop_v3_decodes_only_selected_canonical_source_slot(
        self, _mocked
    ) -> None:
        frames = tuple(
            replace(
                frame,
                # Unsafe legacy values prove v3 does not consult that head.
                crop=(99.0, 99.0, -99.0, 99.0),
                dynamic_crop_v3_by_source=tuple(
                    self._dynamic_prediction_v3(source_index)
                    for source_index in range(4)
                ),
            )
            for frame in _frames(6, 4)
        )

        result = _decode(frames=frames)

        self.assertEqual(
            DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT,
            result.audit.input_summary["crop_contract"],
        )
        self.assertEqual("v3", result.audit.input_summary["dynamic_crop_mode"])
        self.assertEqual(
            result.audit.input_summary["selected_dynamic_crop_attempt_count"],
            result.audit.input_summary["selected_dynamic_crop_v3_attempt_count"],
        )
        for timeline in result.timelines:
            for decision, shot in zip(
                timeline.decisions, timeline.edit.shots, strict=True
            ):
                expected_zoom = 1.02 + 0.04 * decision.source_index
                self.assertAlmostEqual(expected_zoom, decision.crop_requested[0])
                self.assertAlmostEqual(expected_zoom, decision.crop_applied[0])
                self.assertIsNone(decision.dynamic_crop_decode)
                decoded = decision.dynamic_crop_decode_v3
                assert decoded is not None
                self.assertIs(decoded, shot.dynamic_crop_decode_v3)
                self.assertEqual(
                    decoded.to_dict(),
                    shot.to_dict()["dynamic_crop_v3"],
                )
                self.assertEqual(decision.source_index, decoded.source_index)
                self.assertEqual(decision.source_id, decoded.source_id)
                self.assertEqual(
                    DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT,
                    decision.crop_contract,
                )
                diagnostic = decision.to_dict()["dynamic_crop_v3"]
                assert diagnostic is not None
                self.assertEqual(decision.source_id, diagnostic["source_id"])
                self.assertEqual(decision.source_index, diagnostic["source_index"])

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_dynamic_crop_v3_source_identity_mismatch_fails_closed(
        self, _mocked
    ) -> None:
        predictions = tuple(
            self._dynamic_prediction_v3(
                source_index,
                source_id=("wrong-source" if source_index == 2 else None),
            )
            for source_index in range(4)
        )
        frames = tuple(
            replace(frame, dynamic_crop_v3_by_source=predictions)
            for frame in _frames(6, 4)
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError, "v3 source identity mismatch"
        ):
            _decode(frames=frames)

    def test_dynamic_crop_v3_effective_slot_cannot_omit_prediction(self) -> None:
        predictions = tuple(
            None if source_index == 1 else self._dynamic_prediction_v3(source_index)
            for source_index in range(4)
        )
        with self.assertRaisesRegex(
            StageEditDecodeInputError, "prediction presence differs from source gates"
        ):
            replace(
                _frames(6, 4)[0],
                dynamic_crop_v3_by_source=predictions,
            )

    @patch(
        "crosscut.stage_edit_semimarkov_beam.stage_edit_readiness_report",
        return_value=_ready_report(),
    )
    def test_mixed_v2_v3_and_legacy_frame_sets_fail_closed(self, _mocked) -> None:
        v3_predictions = tuple(
            self._dynamic_prediction_v3(source_index)
            for source_index in range(4)
        )
        v3_frames = tuple(
            replace(frame, dynamic_crop_v3_by_source=v3_predictions)
            for frame in _frames(6, 4)
        )
        legacy_and_v3 = (replace(v3_frames[0], dynamic_crop_v3_by_source=()),) + v3_frames[1:]
        with self.assertRaisesRegex(StageEditDecodeInputError, "cannot mix"):
            _decode(frames=legacy_and_v3)

        v2_and_v3 = (
            replace(
                v3_frames[0],
                dynamic_crop_v3_by_source=(),
                dynamic_crop_v2=self._dynamic_prediction(),
            ),
        ) + v3_frames[1:]
        with self.assertRaisesRegex(StageEditDecodeInputError, "cannot mix"):
            _decode(frames=v2_and_v3)


if __name__ == "__main__":
    unittest.main()
