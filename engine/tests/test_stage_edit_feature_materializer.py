from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from types import SimpleNamespace
import unittest

from crosscut.stage_edit_feature_materializer import (
    CausalEditDecision,
    FEATURE_MATERIALIZER_LEGACY_CONTRACT_V2,
    FEATURE_SNAPSHOT_LEGACY_SCHEMA_VERSION,
    SourceFeatureObservation,
    StageEditFeatureFrame,
    StageEditFeatureProvenance,
    StageEditFeatureSnapshot,
    materialize_stage_edit_feature_snapshot,
    materialize_stage_edit_source_candidate,
    materialize_stage_edit_token,
    source_observation_from_timeline_candidate,
    source_observation_from_visual_cache_payload,
)
from crosscut.stage_edit_reference_adapter import build_reference_training_example
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES,
    STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES,
    STAGE_EDIT_TEMPORAL_POSE_PRODUCER_BACKEND,
    STAGE_EDIT_TEMPORAL_POSE_TIMELINE_ENTRY_EXIT_CONTRACT,
    STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT,
)
from crosscut.rhythm import (
    Beat,
    OnsetEnvelope,
    PhraseCandidate,
    RhythmAnalysis,
    TempoEstimate,
)
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.pose import FormationPersonDescriptor, POSE_BACKEND
from crosscut.timeline import TimelineCandidate
from tests.test_stage_edit_reference_adapter import (
    PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS,
    _canonical_payload,
    _payload,
)


SCHEMA = DEFAULT_STAGE_EDIT_SCHEMA


def _snapshot_payload_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sources() -> tuple[SourceFeatureObservation, ...]:
    return (
        SourceFeatureObservation(
            "B",
            sync_confidence=0.74,
            shot_scale=0.63,
            body_center_x=0.52,
            body_center_y=0.61,
            camera_motion_x=-0.2,
            sharpness=0.79,
            available=False,
            hard_gate_passed=False,
        ),
        SourceFeatureObservation(
            "A",
            sync_confidence=0.88,
            sync_residual_normalized=0.08,
            shot_scale=0.38,
            face_scale=0.31,
            face_center_x=0.48,
            face_center_y=0.39,
            pose_motion=0.22,
            camera_motion_x=0.13,
            camera_motion_y=-0.04,
            camera_zoom_velocity=0.05,
            exposure=0.67,
            color_temperature_normalized=-0.12,
            occlusion_probability=0.04,
            automatic_visual_confidence=0.81,
        ),
    )


class StageEditFeatureMaterializerTests(unittest.TestCase):
    def test_source_input_contract_cannot_name_reference_labels_or_match_scores(self) -> None:
        input_names = {field.name for field in fields(SourceFeatureObservation)}
        self.assertNotIn("selected_source_id", input_names)
        self.assertNotIn("observed_match_rank", input_names)
        self.assertNotIn("score", input_names)
        self.assertNotIn("visual_match_confidence", input_names)

    def test_current_or_future_editorial_decision_cannot_enter_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "current or future"):
            materialize_stage_edit_token(
                master_time_seconds=1.0,
                source_observations=_sources(),
                causal_history=(CausalEditDecision(1.0, "A"),),
                performance_end_seconds=2.0,
            )

    def test_rhythm_section_and_causal_context_have_schema_aligned_masks(self) -> None:
        rhythm = RhythmAnalysis(
            sample_rate=100,
            duration_seconds=2.0,
            onset_envelope=OnsetEnvelope(
                values=(0.1, 0.4, 0.8, 0.2, 0.1),
                sample_rate=4,
                frame_length=2,
                hop_length=1,
            ),
            tempo=TempoEstimate(120.0, 0.5, 0.0, 0.9),
            beats=tuple(
                Beat(index, index * 0.5, 0.9)
                for index in range(4)
            ),
            phrase_candidates=(
                PhraseCandidate(4, 0, 4, 0.0, 2.0, 0.9),
            ),
        )
        sections = SectionPlan(
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
                    target_source_switches_per_minute=17.0,
                    camera_motion_budget=0.82,
                    target_roll_events_per_minute=0.35,
                    establish_with_wide_shot=True,
                    protect_last_beat_from_extra_cut=False,
                ),
            )
        )
        token = materialize_stage_edit_token(
            master_time_seconds=0.75,
            source_observations=_sources(),
            rhythm=rhythm,
            section_plan=sections,
            causal_history=(CausalEditDecision(0.5, "A"),),
            style_features={"target_cut_density": 0.7},
            performance_end_seconds=2.0,
        )
        music_index = {
            name: index for index, name in enumerate(SCHEMA.music_feature_names)
        }
        section_index = {
            name: index for index, name in enumerate(SCHEMA.section_feature_names)
        }
        style_offset = (
            len(SCHEMA.music_feature_names)
            + len(SCHEMA.section_feature_names)
            + len(SCHEMA.recent_edit_feature_names)
        )
        self.assertAlmostEqual(
            token.music_features[music_index["beat_phase_cos"]], -1.0
        )
        self.assertFalse(token.missing_mask[music_index["spectral_flux"]])
        self.assertTrue(token.missing_mask[music_index["rms_energy"]])
        self.assertTrue(token.missing_mask[music_index["vocal_probability"]])
        self.assertEqual(
            token.section_features[section_index["chorus_probability"]], 0.9
        )
        self.assertFalse(
            token.missing_mask[
                len(SCHEMA.music_feature_names)
                + section_index["chorus_probability"]
            ]
        )
        self.assertFalse(token.missing_mask[style_offset])

    def test_source_values_missing_bits_and_hard_gate_are_exact(self) -> None:
        token = materialize_stage_edit_token(
            master_time_seconds=1.0,
            source_observations=_sources(),
            performance_end_seconds=4.0,
        )
        self.assertEqual(
            tuple(item.source_id for item in token.source_candidates),
            ("A", "B"),
        )
        by_name = {
            name: index for index, name in enumerate(SCHEMA.source_feature_names)
        }
        first, second = token.source_candidates
        self.assertEqual(first.features[by_name["sync_confidence"]], 0.88)
        self.assertFalse(first.missing_mask[by_name["sync_confidence"]])
        self.assertTrue(first.missing_mask[by_name["sharpness"]])
        self.assertEqual(first.features[by_name["sharpness"]], 0.0)
        self.assertTrue(first.available)
        self.assertTrue(first.hard_gate_passed)
        self.assertFalse(second.available)
        self.assertFalse(second.hard_gate_passed)
        self.assertFalse(second.missing_mask[by_name["sharpness"]])
        self.assertTrue(second.missing_mask[by_name["face_scale"]])

    def test_timeline_visual_adapter_uses_only_pre_choice_evidence(self) -> None:
        candidate = TimelineCandidate(
            segment_index=0,
            source_id="source-a",
            start_seconds=1.0,
            end_seconds=2.0,
            sync_score=0.91,
            composition_score=0.72,
            boundary_score=0.84,
            shot_scale=0.40,
            entry_shot_scale=0.36,
            exit_shot_scale=0.48,
            entry_focus_bbox_xywh=(0.30, 0.20, 0.20, 0.32),
            entry_subject_centroid_xy=(0.44, 0.58),
            entry_pose_embedding=(1.0, 0.0),
            exit_pose_embedding=(0.0, 1.0),
            entry_motion=(0.20, -0.10),
        )
        observation = source_observation_from_timeline_candidate(candidate)
        self.assertEqual(observation.face_scale, 0.32)
        self.assertAlmostEqual(observation.face_center_x or 0.0, 0.40)
        self.assertAlmostEqual(observation.face_center_y or 0.0, 0.36)
        self.assertEqual(observation.body_center_x, 0.44)
        self.assertEqual(observation.body_center_y, 0.58)
        self.assertAlmostEqual(observation.pose_motion or 0.0, 0.5)
        self.assertEqual(observation.camera_motion_x, 0.20)
        self.assertAlmostEqual(observation.camera_zoom_velocity or 0.0, 0.12)
        self.assertAlmostEqual(
            observation.automatic_visual_confidence or 0.0, 0.78
        )

    def test_visual_cache_payload_maps_geometry_pose_motion_and_color(self) -> None:
        observation = source_observation_from_visual_cache_payload(
            "source-a",
            {
                "quality_score": 0.75,
                "composition_score": 0.85,
                "shot_scale": 0.40,
                "primary_face_bbox": [0.30, 0.20, 0.20, 0.30],
                "silhouette_centroid_xy": [0.44, 0.61],
                "pose_embedding": [1.0, 0.0],
                "motion": [0.12, -0.08],
                "color_embedding": [0.5, 0.5, 0.60, 0.1, 0.1, 0.1, 0.70, 0.2],
                "foreground_obstruction_score": 0.09,
                # A poisoned unrelated field is deliberately not consumed.
                "observed_match_rank": -999,
            },
            following_numeric_features={
                "shot_scale": 0.50,
                "pose_embedding": [0.0, 1.0],
            },
            interval_seconds=0.5,
            sync_confidence=0.92,
        )
        self.assertEqual(observation.face_scale, 0.30)
        self.assertEqual(observation.body_center_x, 0.44)
        self.assertAlmostEqual(observation.pose_motion or 0.0, 0.5)
        self.assertAlmostEqual(observation.camera_zoom_velocity or 0.0, 0.2)
        self.assertAlmostEqual(observation.exposure or 0.0, 0.7)
        self.assertAlmostEqual(
            observation.color_temperature_normalized or 0.0, 0.2
        )
        self.assertAlmostEqual(
            observation.automatic_visual_confidence or 0.0, 0.81
        )

    def test_real_pose_adapters_preserve_source_exact_two_sample_sidecars(self) -> None:
        current = tuple(0.10 + index * 0.001 for index in range(60))
        following = tuple(0.12 + index * 0.001 for index in range(60))
        visual = source_observation_from_visual_cache_payload(
            "source-temporal",
            {
                "pose_backend": POSE_BACKEND,
                "pose_embedding": current,
            },
            following_numeric_features={
                "pose_backend": POSE_BACKEND,
                "pose_embedding": following,
            },
            interval_seconds=1.0 / 30.0,
        )
        assert visual.temporal_pose_sidecar is not None
        self.assertEqual(
            visual.temporal_pose_sidecar.sampling_contract,
            STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT,
        )
        self.assertEqual(visual.temporal_pose_sidecar.embeddings, (current, following))
        self.assertEqual(
            visual.temporal_pose_sidecar.producer_backend,
            POSE_BACKEND,
        )
        self.assertEqual(
            STAGE_EDIT_TEMPORAL_POSE_PRODUCER_BACKEND,
            POSE_BACKEND,
        )
        self.assertIs(
            materialize_stage_edit_source_candidate(visual).temporal_pose_sidecar,
            visual.temporal_pose_sidecar,
        )

        timeline = source_observation_from_timeline_candidate(
            TimelineCandidate(
                segment_index=0,
                source_id="source-temporal",
                start_seconds=1.0,
                end_seconds=2.0,
                entry_pose_embedding=current,
                exit_pose_embedding=following,
            )
        )
        self.assertIsNone(timeline.temporal_pose_sidecar)

        trusted_timeline = source_observation_from_timeline_candidate(
            SimpleNamespace(
                source_id="source-temporal",
                duration_seconds=1.0,
                entry_pose_embedding=current,
                exit_pose_embedding=following,
                pose_backends=(POSE_BACKEND, POSE_BACKEND, POSE_BACKEND),
            )
        )
        assert trusted_timeline.temporal_pose_sidecar is not None
        self.assertEqual(
            trusted_timeline.temporal_pose_sidecar.sampling_contract,
            STAGE_EDIT_TEMPORAL_POSE_TIMELINE_ENTRY_EXIT_CONTRACT,
        )
        self.assertEqual(
            trusted_timeline.temporal_pose_sidecar.interval_seconds,
            1.0,
        )

        forged_timeline = source_observation_from_timeline_candidate(
            SimpleNamespace(
                source_id="source-temporal",
                duration_seconds=1.0,
                entry_pose_embedding=current,
                exit_pose_embedding=following,
                pose_backends=(
                    f"{POSE_BACKEND}_forged",
                    POSE_BACKEND,
                    POSE_BACKEND,
                ),
            )
        )
        self.assertIsNone(forged_timeline.temporal_pose_sidecar)

        for forged_backend in (
            "forged_backend",
            "none",
            "opencv_zoo_mediapipe_pose",
            f"{POSE_BACKEND}_forged",
        ):
            with self.subTest(forged_backend=forged_backend):
                forged = source_observation_from_visual_cache_payload(
                    "source-temporal",
                    {
                        "pose_backend": forged_backend,
                        "pose_embedding": current,
                    },
                    following_numeric_features={
                        "pose_backend": forged_backend,
                        "pose_embedding": following,
                    },
                    interval_seconds=1.0 / 30.0,
                )
                self.assertIsNone(forged.temporal_pose_sidecar)

        fallback = source_observation_from_visual_cache_payload(
            "source-temporal",
            {
                "pose_backend": POSE_BACKEND,
                "pose_embedding": current,
            },
            following_numeric_features={
                "pose_backend": "structure_hog_fallback",
                "pose_embedding": following,
            },
            interval_seconds=1.0 / 30.0,
        )
        self.assertIsNone(fallback.temporal_pose_sidecar)

    def test_real_pose_and_formation_features_have_fixed_masks_without_padding(self) -> None:
        pose = [(-0.5 if index % 2 == 0 else 0.5) for index in range(60)]
        layout = [0.25 for _ in range(134)]
        points = [
            [-0.4 + index * 0.1, -0.3 + index * 0.05, 0.8]
            for index in range(8)
        ]
        observation = source_observation_from_visual_cache_payload(
            "source-pose",
            {
                "pose_backend": "opencv_zoo_mediapipe_pose_2023mar",
                "pose_embedding": pose,
                "pose_confidence": 0.77,
                "detected_person_count": 2,
                "person_layout_embedding": layout,
                "formation_people": [
                    {
                        "centroid_xy": [0.75, 0.5],
                        "upper_body_keypoints_xyc": [],
                    },
                    {
                        "centroid_xy": [0.25, 0.5],
                        "upper_body_keypoints_xyc": points,
                    },
                ],
            },
        )
        self.assertEqual(len(observation.pose_model_embedding), 60)
        self.assertEqual(len(observation.formation_layout_embedding), 134)
        self.assertEqual(len(observation.formation_pose_embedding), 144)
        self.assertEqual(observation.detected_person_count_normalized, 2.0 / 6.0)
        self.assertEqual(observation.formation_person_count_normalized, 2.0 / 6.0)
        self.assertEqual(observation.formation_pose_coverage, 0.5)

        candidate = materialize_stage_edit_source_candidate(observation)
        by_name = {
            name: index for index, name in enumerate(SCHEMA.source_feature_names)
        }
        self.assertFalse(
            candidate.missing_mask[
                by_name[STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES[0]]
            ]
        )
        self.assertFalse(
            candidate.missing_mask[
                by_name[STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES[0]]
            ]
        )
        # The second detected person had no valid upper-body pose.  Its slot is
        # missing rather than a fabricated zero skeleton.
        self.assertTrue(
            candidate.missing_mask[
                by_name[STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES[8 * 3]]
            ]
        )
        self.assertTrue(
            candidate.missing_mask[
                by_name[STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES[2 * 8 * 3]]
            ]
        )

        fallback = source_observation_from_visual_cache_payload(
            "source-fallback",
            {
                "pose_backend": "structure_hog_fallback",
                "pose_embedding": [0.1] * 96,
                "pose_confidence": 0.9,
                "detected_person_count": 0,
                "person_layout_embedding": [],
                "formation_people": [],
            },
        )
        fallback_candidate = materialize_stage_edit_source_candidate(fallback)
        self.assertTrue(
            fallback_candidate.missing_mask[
                by_name[STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES[0]]
            ]
        )
        self.assertTrue(fallback_candidate.missing_mask[by_name["pose_model_confidence"]])
        with self.assertRaisesRegex(ValueError, "exactly 60"):
            source_observation_from_visual_cache_payload(
                "source-invalid",
                {
                    "pose_backend": "opencv_zoo_mediapipe_pose_2023mar",
                    "pose_embedding": [0.1] * 59,
                    "pose_confidence": 0.8,
                },
            )

    def test_timeline_and_visual_adapters_share_action_formation_columns(self) -> None:
        pose = tuple(-0.4 + 0.8 * index / 59.0 for index in range(60))
        layout = tuple(0.2 + 0.4 * index / 133.0 for index in range(134))
        points = tuple(
            (-0.3 + 0.08 * index, -0.2 + 0.05 * index, 0.75)
            for index in range(8)
        )
        person = FormationPersonDescriptor(
            bbox_xywh=(0.2, 0.1, 0.3, 0.7),
            centroid_xy=(0.35, 0.45),
            detection_confidence=0.9,
            upper_body_keypoints_xyc=points,
        )
        timeline = source_observation_from_timeline_candidate(
            TimelineCandidate(
                segment_index=0,
                source_id="source-shared",
                start_seconds=1.0,
                end_seconds=2.0,
                entry_pose_embedding=pose,
                entry_person_layout_embedding=layout,
                entry_formation_people=(person,),
                entry_detected_person_count=1,
            )
        )
        visual = source_observation_from_visual_cache_payload(
            "source-shared",
            {
                "pose_backend": "opencv_zoo_mediapipe_pose_2023mar",
                "pose_embedding": list(pose),
                "pose_confidence": None,
                "detected_person_count": 1,
                "person_layout_embedding": list(layout),
                "formation_people": [
                    {
                        "centroid_xy": list(person.centroid_xy),
                        "upper_body_keypoints_xyc": [list(point) for point in points],
                    }
                ],
            },
        )
        self.assertEqual(timeline.pose_model_embedding, visual.pose_model_embedding)
        self.assertEqual(
            timeline.formation_layout_embedding,
            visual.formation_layout_embedding,
        )
        self.assertEqual(
            timeline.formation_pose_embedding,
            visual.formation_pose_embedding,
        )
        self.assertEqual(
            timeline.detected_person_count_normalized,
            visual.detected_person_count_normalized,
        )
        self.assertEqual(
            timeline.formation_person_count_normalized,
            visual.formation_person_count_normalized,
        )

    def test_snapshot_round_trip_is_schema_bound_and_tamper_evident(self) -> None:
        snapshot = materialize_stage_edit_feature_snapshot(
            "ive:test",
            (
                StageEditFeatureFrame(0.5, _sources()),
                StageEditFeatureFrame(
                    1.0,
                    _sources(),
                    (CausalEditDecision(0.75, "A", transition_strength=1.0),),
                ),
            ),
            performance_end_seconds=2.0,
            provenance=StageEditFeatureProvenance(
                upstream_sha256=(("timeline_candidate", "a" * 64),),
                contracts=("ive_source_local_visual_features_v1",),
            ),
        )
        restored = StageEditFeatureSnapshot.from_dict(snapshot.to_dict())
        self.assertEqual(restored, snapshot)
        tampered = deepcopy(snapshot.to_dict())
        tampered["tokens"][0]["source_candidates"][0]["features"][0] = 0.01
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            StageEditFeatureSnapshot.from_dict(tampered)
        legacy = deepcopy(snapshot.to_dict())
        legacy["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "unsupported.*version"):
            StageEditFeatureSnapshot.from_dict(legacy)

    def test_v3_sidecar_is_hash_bound_and_v2_loads_as_explicit_missing(self) -> None:
        first = tuple(0.10 + index * 0.001 for index in range(60))
        second = tuple(0.12 + index * 0.001 for index in range(60))
        observation = source_observation_from_visual_cache_payload(
            "A",
            {
                "pose_backend": POSE_BACKEND,
                "pose_embedding": first,
            },
            following_numeric_features={
                "pose_backend": POSE_BACKEND,
                "pose_embedding": second,
            },
            interval_seconds=0.04,
        )
        snapshot = materialize_stage_edit_feature_snapshot(
            "ive:temporal-snapshot",
            (StageEditFeatureFrame(0.5, (observation,)),),
            performance_end_seconds=1.0,
        )
        payload = snapshot.to_dict()
        audit = snapshot.temporal_pose_sidecar_audit()
        self.assertEqual(audit["status"], "v3_hash_bound_temporal_sidecar")
        self.assertEqual(audit["observed_sidecar_count"], 1)
        self.assertTrue(audit["sidecar_hash_bound"])
        self.assertFalse(audit["enters_transformer_tensors"])
        self.assertEqual(
            payload["tokens"][0]["source_candidates"][0][
                "temporal_pose_sidecar"
            ]["producer_backend"],
            POSE_BACKEND,
        )

        tampered = deepcopy(payload)
        tampered["tokens"][0]["source_candidates"][0][
            "temporal_pose_sidecar"
        ]["embeddings"][1][0] += 0.01
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            StageEditFeatureSnapshot.from_dict(tampered)

        forged_producer = deepcopy(payload)
        forged_producer["tokens"][0]["source_candidates"][0][
            "temporal_pose_sidecar"
        ]["producer_backend"] = f"{POSE_BACKEND}_forged"
        unsigned_forged = deepcopy(forged_producer)
        unsigned_forged.pop("snapshot_sha256")
        forged_producer["snapshot_sha256"] = _snapshot_payload_sha256(
            unsigned_forged
        )
        with self.assertRaisesRegex(ValueError, "producer backend is not trusted"):
            StageEditFeatureSnapshot.from_dict(forged_producer)

        legacy = deepcopy(payload)
        legacy["schema_version"] = FEATURE_SNAPSHOT_LEGACY_SCHEMA_VERSION
        legacy["provenance"]["producer"] = (
            FEATURE_MATERIALIZER_LEGACY_CONTRACT_V2
        )
        for token in legacy["tokens"]:
            for candidate in token["source_candidates"]:
                candidate.pop("temporal_pose_sidecar")
        legacy["input_policy"] = {
            "training_inference_materializer_shared": True,
            "reference_attribution_match_inputs_present": False,
            "selected_labels_present": False,
            "missing_values_are_explicitly_masked": True,
            "source_availability_and_hard_gate_are_distinct": True,
        }
        unsigned = deepcopy(legacy)
        unsigned.pop("snapshot_sha256")
        legacy["snapshot_sha256"] = _snapshot_payload_sha256(unsigned)
        restored = StageEditFeatureSnapshot.from_dict(legacy)
        legacy_audit = restored.temporal_pose_sidecar_audit()
        self.assertEqual(
            restored.schema_version,
            FEATURE_SNAPSHOT_LEGACY_SCHEMA_VERSION,
        )
        self.assertEqual(
            legacy_audit["status"],
            "legacy_v2_explicit_temporal_sidecar_missing",
        )
        self.assertEqual(legacy_audit["observed_sidecar_count"], 0)
        self.assertFalse(legacy_audit["sidecar_hash_bound"])
        self.assertFalse(
            legacy_audit["pose_direction_or_velocity_proven_by_snapshot_alone"]
        )
        with self.assertRaisesRegex(
            ValueError,
            "snapshot v2 cannot carry a temporal pose sidecar",
        ):
            StageEditFeatureSnapshot(
                sequence=snapshot.sequence,
                token_schema_fingerprint=restored.token_schema_fingerprint,
                provenance=restored.provenance,
                snapshot_sha256=restored.snapshot_sha256,
                schema_version=FEATURE_SNAPSHOT_LEGACY_SCHEMA_VERSION,
            )

    def test_reference_training_and_inference_consume_identical_snapshot(self) -> None:
        times = (0.0, 0.5, 1.0, 1.75, 2.5, 3.25)
        frames = tuple(
            StageEditFeatureFrame(
                timestamp,
                _sources(),
                (
                    (CausalEditDecision(2.5, "B", transition_strength=1.0),)
                    if timestamp > 2.5
                    else ()
                ),
            )
            for timestamp in times
        )
        snapshot = materialize_stage_edit_feature_snapshot(
            "stageedit-reference:test:ref-a",
            frames,
            performance_end_seconds=4.0,
        )
        payload = _payload("ref-a")
        example = build_reference_training_example(
            payload,
            feature_snapshot=snapshot,
        )
        self.assertEqual(example.sequence.tokens, snapshot.sequence.tokens)
        self.assertEqual(example.feature_snapshot_sha256, snapshot.snapshot_sha256)
        self.assertTrue(example.statistics["shared_feature_materializer_used"])
        self.assertTrue(example.statistics["materialized_feature_snapshot_used"])

    def test_canonical_boundary_adapter_preserves_snapshot_train_serve_parity(self) -> None:
        frames = tuple(
            StageEditFeatureFrame(
                timestamp,
                _sources(),
                tuple(
                    decision
                    for decision in (
                        (
                            CausalEditDecision(
                                0.0,
                                "A",
                                transition_strength=1.0,
                            )
                            if timestamp > 0.0
                            else None
                        ),
                        (
                            CausalEditDecision(
                                2.5,
                                "B",
                                transition_strength=1.0,
                            )
                            if timestamp > 2.5
                            else None
                        ),
                    )
                    if decision is not None
                ),
            )
            for timestamp in (0.0, 0.5, 1.0, 1.75, 2.5, 3.25)
        )
        snapshot = materialize_stage_edit_feature_snapshot(
            "stageedit-reference:test:ref-a:canonical",
            frames,
            performance_end_seconds=4.0,
        )
        example = build_reference_training_example(
            _canonical_payload(),
            feature_snapshot=snapshot,
        )
        self.assertEqual(example.sequence.tokens, snapshot.sequence.tokens)
        self.assertEqual(example.statistics["canonical_detected_boundary_count"], 2)
        self.assertEqual(example.statistics["boundary_token_count"], 3)
        gated_target = next(
            target
            for token, target in zip(
                example.sequence.tokens, example.targets.targets, strict=True
            )
            if token.master_time_seconds == 2.5
        )
        self.assertIsNone(gated_target.selected_source_id)
        self.assertEqual(gated_target.rejected_source_ids, ())

    def test_reference_match_poison_cannot_change_materialized_inputs(self) -> None:
        times = (0.0, 0.5, 1.0, 1.75, 2.5, 3.25)
        snapshot = materialize_stage_edit_feature_snapshot(
            "stageedit-reference:test:ref-a",
            tuple(StageEditFeatureFrame(timestamp, _sources()) for timestamp in times),
            performance_end_seconds=4.0,
        )
        original = _payload("ref-a")
        poisoned = deepcopy(original)
        for record in poisoned["listwise_records"]:
            record["observed_score_margin"] = -999999.0
            record["candidates"].reverse()
            for rank, candidate in enumerate(record["candidates"]):
                candidate["observed_match_rank"] = 1000 + rank
                candidate["evidence"] = {
                    name: 999999.0
                    for name in PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS
                }
        first = build_reference_training_example(
            original,
            feature_snapshot=snapshot,
        )
        second = build_reference_training_example(
            poisoned,
            feature_snapshot=snapshot,
        )
        self.assertEqual(first.sequence, second.sequence)
        self.assertEqual(first.targets, second.targets)
        serialized = json.dumps(snapshot.to_dict(), sort_keys=True)
        self.assertNotIn("observed_match_rank", serialized)
        self.assertNotIn("selected_source_id", serialized)


if __name__ == "__main__":
    unittest.main()
