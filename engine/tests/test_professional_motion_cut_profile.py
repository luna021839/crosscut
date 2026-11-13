from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from crosscut import professional_motion_cut_profile as motion
from crosscut.professional_motion_cut_profile import (
    EmpiricalMotionDistribution,
    MotionCutSoftPreferenceRecommendation,
    ProfessionalMotionCutProfile,
    ProfessionalMotionCutProfileError,
    load_professional_motion_cut_profile,
)
from crosscut.stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
)
from crosscut.stage_edit_source_pair_training import (
    materialize_stage_edit_source_pair_training_corpus,
)
from crosscut.stage_edit_semimarkov_beam import StageEditSemiMarkovConfig
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES,
    SourceCandidateToken,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditToken,
)


def _distribution(
    values: tuple[float, ...],
    *,
    minimum: float,
    maximum: float,
) -> dict[str, object]:
    return EmpiricalMotionDistribution.from_values(
        values,
        minimum_allowed=minimum,
        maximum_allowed=maximum,
    ).to_dict()


def _valid_payload() -> dict[str, object]:
    pose = _distribution(
        (
            0.00010383278019288067,
            0.0002705025925969218,
            0.0013651752724601618,
            0.0024240018497856952,
            0.010089000247239965,
            0.042721026258255856,
        ),
        minimum=0.0,
        maximum=1.0,
    )
    continuity = _distribution(
        (
            -0.6951708362954652,
            -0.49213925079872173,
            -0.39991898058929115,
            -0.32434112477150095,
            -0.2522282468388802,
            -0.1097543777159719,
        ),
        minimum=-1.0,
        maximum=0.0,
    )
    cosine = _distribution(
        (
            0.674734094136797,
            0.7354177153190036,
            0.7831558357163764,
            0.8486102174679628,
            0.8940526060955092,
            0.9637479799345603,
        ),
        minimum=-1.0,
        maximum=1.0,
    )
    # The compact fixture retains production support counts while reusing six
    # representative values.  Distribution support is an independently bound
    # count, not an assertion that raw per-event values are persisted.
    pose["observed_switch_count"] = 23
    continuity["observed_switch_count"] = 54
    cosine["observed_switch_count"] = 14
    recommendation = MotionCutSoftPreferenceRecommendation(
        both_moving_minimum_pose_motion=float(pose["quantiles"]["p10"]),
        continuity_compatible_minimum_score=float(
            continuity["quantiles"]["p10"]
        ),
        pose_phase_compatible_preferred_cosine=float(
            cosine["quantiles"]["p50"]
        ),
    ).to_dict()
    payload: dict[str, object] = {
        "schema_version": motion.PROFILE_SCHEMA_VERSION,
        "contract": motion.PROFILE_CONTRACT,
        "analysis_basis": motion.PROFILE_ANALYSIS_BASIS,
        "scope": {
            "artist": "IVE",
            "song_count": 4,
            "human_reference_count": 20,
            "source_pair_contributing_reference_count": 17,
            "highlight_roles": ["build", "chorus_like", "bridge_like"],
            "close_involved_definition": {
                "source_feature": "shot_scale",
                "outgoing_or_incoming_minimum_inclusive": 0.52,
                "missing_is_close": False,
                "both_sides_must_be_observed": True,
            },
        },
        "sampling": {
            "boundary_guard_seconds": 0.18,
            "max_sources": 32,
            "negative_samples_per_hold": 2,
            "visual_working_width": 480,
            "parameters_match_every_source_snapshot_manifest": True,
        },
        "input_policy": {
            "confirmed_nonopening_actual_source_switch_positives_only": True,
            "outgoing_and_incoming_observed_at_same_exact_snapshot_token": True,
            "reference_match_evidence_used_as_model_input": False,
            "selected_or_rejected_source_targets_used_as_model_input": False,
            "raw_frames_or_identity_embeddings_persisted": False,
            "song_specific_runtime_threshold_lookup_present": False,
            "runtime_requires_reference_video": False,
        },
        "counts": {
            "source_pair_positive_count": 170,
            "per_song_source_pair_positive_count": {
                "IVE-BANG-BANG": 32,
                "IVE-BLACKHOLE": 80,
                "IVE-REBEL-HEART": 37,
                "IVE-XOXZ": 21,
            },
            "build_close_involved_switch_count": 10,
            "chorus_close_involved_switch_count": 29,
            "bridge_close_involved_switch_count": 15,
            "highlight_close_involved_switch_count": 54,
            "pose_motion_pair_observed_switch_count": 23,
            "continuity_observed_switch_count": 54,
            "pose_embedding_cosine_observed_switch_count": 14,
            "sharpness_pair_observed_switch_count": 0,
            "per_role_observed_support": {
                "build": {
                    "close_involved_switch_count": 10,
                    "pose_motion_pair_observed_switch_count": 4,
                    "continuity_observed_switch_count": 10,
                    "pose_embedding_cosine_observed_switch_count": 2,
                    "sharpness_pair_observed_switch_count": 0,
                },
                "chorus_like": {
                    "close_involved_switch_count": 29,
                    "pose_motion_pair_observed_switch_count": 13,
                    "continuity_observed_switch_count": 29,
                    "pose_embedding_cosine_observed_switch_count": 10,
                    "sharpness_pair_observed_switch_count": 0,
                },
                "bridge_like": {
                    "close_involved_switch_count": 15,
                    "pose_motion_pair_observed_switch_count": 6,
                    "continuity_observed_switch_count": 15,
                    "pose_embedding_cosine_observed_switch_count": 2,
                    "sharpness_pair_observed_switch_count": 0,
                },
            },
            "source_pair_exclusions": {
                "opening_selection_excluded_count": 18,
                "unconfirmed_selection_excluded_count": 0,
                "same_source_selection_excluded_count": 12,
                "priorless_selection_excluded_count": 2,
            },
        },
        "distributions": {
            "pose_motion_minimum": pose,
            "handcrafted_continuity_score": continuity,
            "pose_embedding_cosine_similarity": {
                **cosine,
                "semantics": motion.SAME_TOKEN_POSE_SIMILARITY_SEMANTICS,
                "runtime_pair_feature_available": True,
                "default_runtime_threshold_bound": False,
                "used_for_hard_gate": False,
            },
            "sharpness_minimum": {
                "observed_switch_count": 0,
                "quantile_method": None,
                "quantiles": None,
                "minimum": None,
                "maximum": None,
                "unavailable": True,
            },
        },
        "recommendation": recommendation,
        "evidence_limitations": {
            "motion_direction_observed": False,
            "motion_direction_claimed": False,
            "pose_embedding_cosine_is_temporal_phase_ground_truth": False,
            "pose_embedding_cosine_is_same_token_similarity_proxy_only": True,
            "pose_embedding_cosine_runtime_coverage_fraction": 14 / 54,
            "pose_motion_pair_coverage_fraction": 23 / 54,
            "sharpness_pair_coverage_fraction": 0.0,
            "strict_song_disjoint_evaluation_of_a_source_song_requires_loso_rebuild": True,
            "unseen_ive_song_inference_requires_target_reference_labels": False,
            "held_out_or_human_parity_claimed": False,
        },
        "provenance": {
            "content_hash_algorithm": motion.PROFILE_CONTENT_HASH_ALGORITHM,
            "profile_content_sha256": None,
            "source_montage_profile": {
                "path": "/synthetic/ive-professional-montage-v2.json",
                "file_sha256": "a" * 64,
                "content_sha256": "b" * 64,
                "schema_version": 2,
                "analysis_basis": "ive_human_reference_visual_montage_grammar_v2",
            },
            "token_schema_fingerprint": DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
            "reference_corpus_data_id": "stageedit-ref-synthetic",
            "source_pair_training_schema_version": 3,
            "source_pair_training_contract": (
                "confirmed_nonopening_actual_switch_same_token_source_pair_"
                "reference_training_with_priorless_bootstrap_v3"
            ),
            "source_pair_corpus_sha256": "c" * 64,
            "handcrafted_continuity_scorer_contract": (
                "semimarkov_boundary_continuity_handcrafted_v1"
            ),
            "handcrafted_continuity_scorer_fingerprint": hashlib.sha256(
                b"semimarkov_boundary_continuity_handcrafted_v1"
            ).hexdigest(),
        },
    }
    payload["provenance"]["profile_content_sha256"] = (
        motion._profile_content_sha256(payload)
    )
    return payload


def _reseal(payload: dict[str, object]) -> dict[str, object]:
    payload["provenance"]["profile_content_sha256"] = None
    payload["provenance"]["profile_content_sha256"] = (
        motion._profile_content_sha256(payload)
    )
    return payload


def _source_candidate(
    source_id: str,
    *,
    shot_scale: float,
    pose_motion: float,
    pose_axis: int,
    face_center_x: float,
) -> SourceCandidateToken:
    names = DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names
    features = [0.0] * len(names)
    missing = [True] * len(names)

    def observed(name: str, value: float) -> None:
        index = names.index(name)
        features[index] = value
        missing[index] = False

    observed("shot_scale", shot_scale)
    observed("pose_motion", pose_motion)
    observed("face_scale", 0.30)
    observed("face_center_x", face_center_x)
    observed("face_center_y", 0.50)
    observed("camera_motion_x", 0.10)
    observed("camera_motion_y", 0.10)
    for index, name in enumerate(STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES):
        observed(name, 1.0 if index == pose_axis else 0.0)
    return SourceCandidateToken(
        source_id=source_id,
        features=tuple(features),
        missing_mask=tuple(missing),
        available=True,
        hard_gate_passed=True,
    )


def _token(
    time_seconds: float,
    role_index: int,
    candidates: tuple[SourceCandidateToken, ...],
) -> StageEditToken:
    section = [0.0] * len(DEFAULT_STAGE_EDIT_SCHEMA.section_feature_names)
    section[role_index] = 1.0
    return StageEditToken(
        master_time_seconds=time_seconds,
        music_features=(0.0,) * len(DEFAULT_STAGE_EDIT_SCHEMA.music_feature_names),
        section_features=tuple(section),
        source_candidates=candidates,
        recent_edit_features=(0.0,)
        * len(DEFAULT_STAGE_EDIT_SCHEMA.recent_edit_feature_names),
        style_features=(0.0,) * len(DEFAULT_STAGE_EDIT_SCHEMA.style_feature_names),
        missing_mask=(False,) * DEFAULT_STAGE_EDIT_SCHEMA.raw_context_dimension,
    )


def _synthetic_corpus() -> ReferenceTrainingCorpus:
    candidates = (
        _source_candidate(
            "A", shot_scale=0.70, pose_motion=0.20, pose_axis=0, face_center_x=0.48
        ),
        _source_candidate(
            "B", shot_scale=0.80, pose_motion=0.40, pose_axis=0, face_center_x=0.50
        ),
        _source_candidate(
            "C", shot_scale=0.90, pose_motion=0.60, pose_axis=1, face_center_x=0.52
        ),
    )
    sequence_id = "synthetic-motion-cut"
    sequence = StageEditSequence(
        sequence_id,
        (
            _token(0.0, 2, candidates),
            _token(1.0, 2, candidates),
            _token(2.0, 3, candidates),
            _token(3.0, 4, candidates),
        ),
    )
    targets = StageEditTargetSequence(
        sequence_id,
        (
            StageEditTarget(cut=1.0, selected_source_id="A"),
            StageEditTarget(cut=1.0, selected_source_id="B"),
            StageEditTarget(cut=1.0, selected_source_id="C"),
            StageEditTarget(cut=1.0, selected_source_id="A"),
        ),
    )
    example = ReferenceTrainingExample(
        reference_id="REF",
        song_id="IVE-SYNTHETIC",
        style_id="IVE",
        sequence=sequence,
        targets=targets,
        statistics={},
        feature_snapshot_sha256="d" * 64,
    )
    return ReferenceTrainingCorpus(
        examples=(example,),
        data_id="stageedit-ref-synthetic",
        source_paths=("/synthetic/ref.json",),
        statistics={"reference_count": 1},
    )


class ProfessionalMotionCutProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_nearest_rank_distribution_is_not_interpolated(self) -> None:
        distribution = EmpiricalMotionDistribution.from_values(
            (0.1, 0.2, 0.3, 0.4),
            minimum_allowed=0.0,
            maximum_allowed=1.0,
        )
        self.assertEqual((0.1, 0.1, 0.2, 0.3), distribution.quantiles)
        self.assertEqual("nearest_rank_empirical", distribution.to_dict()["quantile_method"])

    def test_recommendation_is_soft_joint_and_has_no_section_quota(self) -> None:
        recommendation = MotionCutSoftPreferenceRecommendation(
            both_moving_minimum_pose_motion=0.1,
            continuity_compatible_minimum_score=-0.4,
            pose_phase_compatible_preferred_cosine=0.8,
        )
        payload = recommendation.to_dict()
        self.assertFalse(payload["hard_gate"])
        self.assertIsNone(payload["minimum_required_switches_per_section"])
        self.assertTrue(
            payload["activation"]["pose_phase_compatible"][
                "runtime_pair_feature_available"
            ]
        )
        self.assertFalse(
            payload["activation"]["pose_phase_compatible"][
                "default_runtime_threshold_bound"
            ]
        )
        self.assertIsNone(
            payload["activation"]["pose_phase_compatible"]["hard_threshold"]
        )
        self.assertFalse(payload["activation"]["high_motion_alone_is_sufficient"])

    def test_exact_pair_measurement_classifies_three_highlight_roles(self) -> None:
        corpus = _synthetic_corpus()
        pairs = materialize_stage_edit_source_pair_training_corpus(corpus)
        measured = motion._measure_motion_cut_pairs(corpus, pairs)
        self.assertEqual(3, measured.source_pair_positive_count)
        self.assertEqual(1, measured.build_close_involved_switch_count)
        self.assertEqual(1, measured.chorus_close_involved_switch_count)
        self.assertEqual(1, measured.bridge_close_involved_switch_count)
        self.assertEqual((0.2, 0.4, 0.2), measured.pose_motion_minimum_values)
        self.assertEqual(3, len(measured.continuity_score_values))
        self.assertEqual(3, len(measured.pose_embedding_cosine_values))
        self.assertEqual((), measured.sharpness_minimum_values)

    def test_profile_round_trip_and_stage_adapter_preserve_exact_metrics(self) -> None:
        profile = ProfessionalMotionCutProfile.from_mapping(_valid_payload())
        self.assertEqual(54, profile.to_dict()["counts"]["highlight_close_involved_switch_count"])
        calibration = profile.to_stage_edit_calibration()
        self.assertEqual(170, calibration.confirmed_source_switch_count)
        self.assertEqual(10, calibration.build_close_involved_switch_count)
        self.assertEqual(29, calibration.chorus_close_involved_switch_count)
        self.assertEqual(15, calibration.bridge_close_involved_switch_count)
        self.assertEqual(23, calibration.pose_pair_observed_switch_count)
        self.assertEqual(14, calibration.pose_embedding_cosine_observed_switch_count)
        self.assertEqual(0, calibration.sharpness_pair_observed_switch_count)
        self.assertIsNone(calibration.sharpness_minimum_quantiles)
        self.assertTrue(
            calibration.to_dict()["continuity_qualifier"]
            ["independent_from_primary_boundary_continuity"]
        )
        self.assertFalse(
            calibration.to_dict()["continuity_qualifier"]
            ["primary_boundary_score_reused"]
        )
        recommendation = profile.recommendation
        config = StageEditSemiMarkovConfig(
            highlight_motion_calibration=calibration,
            highlight_motion_boundary_minimum_pose_motion=(
                recommendation.both_moving_minimum_pose_motion
            ),
            highlight_motion_boundary_minimum_continuity_score=(
                recommendation.continuity_compatible_minimum_score
            ),
            highlight_motion_boundary_minimum_pose_phase_cosine=(
                recommendation.pose_phase_compatible_preferred_cosine
            ),
            highlight_minimum_motion_closeup_switch_boundaries=0,
        )
        self.assertEqual(
            recommendation.pose_phase_compatible_preferred_cosine,
            config.highlight_motion_boundary_minimum_pose_phase_cosine,
        )
        self.assertEqual(0, config.highlight_minimum_motion_closeup_switch_boundaries)

    def test_profile_hash_rejects_metric_tampering(self) -> None:
        payload = _valid_payload()
        payload["counts"]["chorus_close_involved_switch_count"] += 1
        with self.assertRaisesRegex(
            ProfessionalMotionCutProfileError,
            "highlight close-involved counts mismatch|content SHA-256",
        ):
            ProfessionalMotionCutProfile.from_mapping(payload)

    def test_rehashed_motion_direction_claim_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["evidence_limitations"]["motion_direction_claimed"] = True
        _reseal(payload)
        with self.assertRaisesRegex(
            ProfessionalMotionCutProfileError, "overclaim motion"
        ):
            ProfessionalMotionCutProfile.from_mapping(payload)

    def test_rehashed_pose_phase_hard_threshold_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["recommendation"]["activation"]["pose_phase_compatible"][
            "hard_threshold"
        ] = 0.8
        _reseal(payload)
        with self.assertRaisesRegex(
            ProfessionalMotionCutProfileError, "bounded soft policy"
        ):
            ProfessionalMotionCutProfile.from_mapping(payload)

    def test_rehashed_zero_support_sharpness_quantiles_are_rejected(self) -> None:
        payload = _valid_payload()
        payload["distributions"]["sharpness_minimum"]["quantiles"] = {
            "p10": 0.1,
            "p25": 0.2,
            "p50": 0.3,
            "p75": 0.4,
        }
        _reseal(payload)
        with self.assertRaisesRegex(
            ProfessionalMotionCutProfileError, "sharpness must remain unavailable"
        ):
            ProfessionalMotionCutProfile.from_mapping(payload)

    def test_rehashed_distribution_support_mismatch_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["distributions"]["pose_motion_minimum"][
            "observed_switch_count"
        ] = 22
        _reseal(payload)
        with self.assertRaisesRegex(
            ProfessionalMotionCutProfileError, "support does not match"
        ):
            ProfessionalMotionCutProfile.from_mapping(payload)

    def test_file_loader_binds_file_hash_and_rejects_symlinks(self) -> None:
        profile = ProfessionalMotionCutProfile.from_mapping(_valid_payload())
        path = profile.write_json(self.root / "profile.json")
        raw = path.read_bytes()
        loaded = load_professional_motion_cut_profile(
            path,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            verify_source_inputs=False,
        )
        self.assertEqual(profile.profile_content_sha256, loaded.profile_content_sha256)
        with self.assertRaisesRegex(
            ProfessionalMotionCutProfileError, "file SHA-256 mismatch"
        ):
            load_professional_motion_cut_profile(
                path,
                expected_file_sha256="0" * 64,
                verify_source_inputs=False,
            )
        link = self.root / "link.json"
        link.symlink_to(path)
        with self.assertRaisesRegex(
            ProfessionalMotionCutProfileError, "non-symlink"
        ):
            load_professional_motion_cut_profile(
                link,
                verify_source_inputs=False,
            )

    def test_duplicate_json_key_is_rejected(self) -> None:
        path = self.root / "duplicate.json"
        path.write_text('{"schema_version":1,"schema_version":1}\n')
        with self.assertRaisesRegex(
            ProfessionalMotionCutProfileError, "duplicate JSON key"
        ):
            load_professional_motion_cut_profile(
                path,
                verify_source_inputs=False,
            )

    def test_nonfinite_values_are_rejected_before_hash_acceptance(self) -> None:
        payload = _valid_payload()
        payload["distributions"]["pose_motion_minimum"]["quantiles"]["p10"] = math.nan
        with self.assertRaisesRegex(
            ProfessionalMotionCutProfileError, "finite"
        ):
            ProfessionalMotionCutProfile.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
