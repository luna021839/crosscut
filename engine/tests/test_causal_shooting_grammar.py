from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from crosscut.causal_shooting_grammar import (
    CausalShootingGrammarArtifact,
    CausalShootingGrammarSmoothing,
    STAGE_EDIT_SHOOTING_GRAMMAR_ADAPTER,
    STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES,
    ShootingGrammarObservation,
    ShootingGrammarProvenance,
    adapt_causal_shooting_grammar_for_stage_edit,
    build_causal_shooting_grammar_artifact,
    load_causal_shooting_grammar_artifact,
    write_causal_shooting_grammar_artifact,
)


def _provenance() -> ShootingGrammarProvenance:
    return ShootingGrammarProvenance(
        upstream_sha256=(
            ("camera_motion_analyzer", "a" * 64),
            ("source_catalog", "b" * 64),
        ),
        analyzer_fingerprint_sha256="c" * 64,
        contracts=("native_cut_interval_detector_v1", "shot_classifier_v1"),
    )


def _observation(
    timestamp: float,
    *,
    source_id: str = "A",
    native_cut: bool | None = False,
    closeup: float = 0.2,
    medium: float = 0.3,
    wide: float = 0.5,
    pan: float = -0.4,
    tilt: float = 0.4,
    zoom_in: float = 0.1,
    zoom_out: float = 0.2,
    static: float = 0.7,
    roll: float = 0.1,
    headroom: float = 0.2,
    lead_room: float = 0.3,
    detected_count: int = 6,
    formation_count: int = 4,
    confidence: float = 0.8,
) -> ShootingGrammarObservation:
    return ShootingGrammarObservation(
        source_id=source_id,
        master_time_seconds=timestamp,
        shot_closeup_probability=closeup,
        shot_medium_probability=medium,
        shot_wide_probability=wide,
        camera_pan_velocity=pan,
        camera_tilt_velocity=tilt,
        camera_zoom_in_probability=zoom_in,
        camera_zoom_out_probability=zoom_out,
        camera_static_probability=static,
        camera_roll_degrees_normalized=roll,
        subject_headroom_normalized=headroom,
        subject_lead_room_normalized=lead_room,
        detected_person_count=detected_count,
        formation_person_count=formation_count,
        internal_native_cut_since_previous=native_cut,
        shooting_grammar_confidence=confidence,
    )


class CausalShootingGrammarTests(unittest.TestCase):
    def test_causal_smoothing_state_and_named_adapter(self) -> None:
        artifact = build_causal_shooting_grammar_artifact(
            (
                _observation(0.0, native_cut=True),
                _observation(
                    0.25,
                    closeup=0.6,
                    medium=0.2,
                    wide=0.2,
                    pan=0.4,
                    tilt=-0.2,
                    zoom_in=0.7,
                    zoom_out=0.1,
                    static=0.2,
                    roll=-0.1,
                    headroom=0.4,
                    lead_room=0.5,
                    detected_count=5,
                    formation_count=5,
                    confidence=0.6,
                ),
            ),
            provenance=_provenance(),
        )
        first, current = artifact.records
        self.assertEqual("wide", first.shot_scale_state)
        self.assertEqual("left", first.camera_pan_direction)
        self.assertEqual("down", first.camera_tilt_direction)
        self.assertEqual("static", first.camera_zoom_state)
        self.assertEqual(1.0, first.native_cut_proximity)

        self.assertAlmostEqual(0.4, current.shot_closeup_probability or 0.0)
        self.assertAlmostEqual(0.25, current.shot_medium_probability or 0.0)
        self.assertAlmostEqual(0.35, current.shot_wide_probability or 0.0)
        self.assertEqual("closeup", current.shot_scale_state)
        self.assertEqual(0.0, current.camera_pan_velocity)
        self.assertEqual("static", current.camera_pan_direction)
        self.assertAlmostEqual(0.1, current.camera_tilt_velocity or 0.0)
        self.assertEqual("down", current.camera_tilt_direction)
        self.assertEqual("static", current.camera_zoom_state)
        self.assertAlmostEqual(0.5, current.native_cut_proximity or 0.0)

        adapted = adapt_causal_shooting_grammar_for_stage_edit(current)
        self.assertEqual(STAGE_EDIT_SHOOTING_GRAMMAR_ADAPTER, adapted.adapter)
        self.assertEqual(
            STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES,
            adapted.feature_names,
        )
        self.assertEqual((False,) * len(adapted.feature_names), adapted.missing_mask)
        named = adapted.named_values()
        self.assertAlmostEqual(5.0 / 6.0, named["detected_person_count_normalized"][0])
        self.assertEqual((1.0, False), named["formation_pose_coverage"])
        self.assertEqual((0.5, False), named["native_cut_proximity"])

    def test_future_append_cannot_change_prefix_records(self) -> None:
        prefix_observations = (
            _observation(0.0, source_id="A", native_cut=True),
            _observation(0.0, source_id="B", native_cut=False, pan=0.2),
            _observation(0.25, source_id="A", pan=-0.2),
            _observation(0.25, source_id="B", pan=0.6),
        )
        future = (
            _observation(
                0.5,
                source_id="A",
                native_cut=True,
                closeup=0.9,
                medium=0.05,
                wide=0.05,
                pan=1.0,
            ),
            _observation(0.5, source_id="B", native_cut=False, pan=-1.0),
        )
        prefix = build_causal_shooting_grammar_artifact(
            prefix_observations,
            provenance=_provenance(),
        )
        full = build_causal_shooting_grammar_artifact(
            prefix_observations + future,
            provenance=_provenance(),
        )
        self.assertEqual(
            [record.to_dict() for record in prefix.records],
            [record.to_dict() for record in full.records[: len(prefix.records)]],
        )
        self.assertNotEqual(
            prefix.artifact_content_sha256,
            full.artifact_content_sha256,
        )
        self.assertAlmostEqual(-0.3, prefix.records[2].camera_pan_velocity or 0.0)
        self.assertAlmostEqual(0.4, prefix.records[3].camera_pan_velocity or 0.0)
        self.assertFalse(full.to_dict()["temporal_policy"]["future_frames_used"])
        self.assertFalse(
            full.to_dict()["temporal_policy"]["future_native_cuts_used"]
        )

    def test_native_cut_and_unknown_detector_reset_causal_smoothing(self) -> None:
        artifact = build_causal_shooting_grammar_artifact(
            (
                _observation(0.0, native_cut=True, pan=-1.0),
                _observation(0.2, native_cut=True, pan=1.0),
                _observation(0.3, native_cut=None, pan=-0.5),
                _observation(0.6, native_cut=False, pan=-0.5),
                _observation(0.9, native_cut=False, pan=-0.5),
                _observation(2.1, native_cut=False, pan=0.5),
            ),
            provenance=_provenance(),
        )
        records = artifact.records
        self.assertEqual(1.0, records[1].camera_pan_velocity)
        self.assertEqual("right", records[1].camera_pan_direction)
        self.assertEqual(-0.5, records[2].camera_pan_velocity)
        self.assertIsNone(records[2].native_cut_proximity)
        self.assertIsNone(records[3].native_cut_proximity)
        self.assertEqual(0.0, records[4].native_cut_proximity)
        self.assertEqual(0.5, records[5].camera_pan_velocity)

    def test_unknown_probability_state_is_not_missing(self) -> None:
        unknown = ShootingGrammarObservation(
            source_id="A",
            master_time_seconds=0.0,
            shot_closeup_probability=0.4,
            shot_medium_probability=0.4,
            shot_wide_probability=0.2,
            camera_zoom_in_probability=0.5,
            camera_zoom_out_probability=0.5,
            camera_static_probability=0.0,
            internal_native_cut_since_previous=False,
        )
        missing = ShootingGrammarObservation(
            source_id="A",
            master_time_seconds=0.25,
            internal_native_cut_since_previous=None,
        )
        artifact = build_causal_shooting_grammar_artifact(
            (unknown, missing),
            provenance=_provenance(),
        )
        unknown_record, missing_record = artifact.records
        self.assertEqual("unknown", unknown_record.shot_scale_state)
        self.assertEqual("unknown", unknown_record.camera_zoom_state)
        unknown_adapter = adapt_causal_shooting_grammar_for_stage_edit(
            unknown_record
        )
        self.assertFalse(unknown_adapter.missing_mask[0])
        self.assertFalse(unknown_adapter.missing_mask[5])
        self.assertFalse(
            artifact.to_dict()["feature_adapter"][
                "unknown_classification_sets_missing_mask"
            ]
        )

        self.assertEqual("missing", missing_record.shot_scale_state)
        self.assertEqual("missing", missing_record.camera_zoom_state)
        missing_adapter = adapt_causal_shooting_grammar_for_stage_edit(
            missing_record
        )
        self.assertEqual(
            (True,) * len(STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES),
            missing_adapter.missing_mask,
        )
        self.assertEqual(
            (0.0,) * len(STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES),
            missing_adapter.values,
        )

    def test_unordered_nonfinite_partial_and_out_of_range_inputs_fail_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "strict master-time/source ordered"):
            build_causal_shooting_grammar_artifact(
                (_observation(0.5), _observation(0.25)),
                provenance=_provenance(),
            )
        with self.assertRaisesRegex(ValueError, "strict master-time/source ordered"):
            build_causal_shooting_grammar_artifact(
                (
                    _observation(0.0, source_id="B"),
                    _observation(0.0, source_id="A"),
                ),
                provenance=_provenance(),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            ShootingGrammarObservation("A", math.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            ShootingGrammarObservation("A", 0.0, camera_tilt_velocity=math.inf)
        with self.assertRaisesRegex(ValueError, "all present or all missing"):
            ShootingGrammarObservation(
                "A",
                0.0,
                shot_closeup_probability=0.5,
            )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            ShootingGrammarObservation(
                "A",
                0.0,
                shot_closeup_probability=0.5,
                shot_medium_probability=0.5,
                shot_wide_probability=0.5,
            )
        with self.assertRaisesRegex(ValueError, r"\[-1.0, 1.0\]"):
            ShootingGrammarObservation("A", 0.0, camera_pan_velocity=1.01)
        with self.assertRaisesRegex(ValueError, r"\[0.0, 1.0\]"):
            ShootingGrammarObservation(
                "A",
                0.0,
                shooting_grammar_confidence=-0.01,
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            ShootingGrammarObservation(
                "A",
                0.0,
                detected_person_count=2,
                formation_person_count=3,
            )

    def test_classification_uses_reviewed_thresholds_without_clamping(self) -> None:
        config = CausalShootingGrammarSmoothing(
            ema_alpha=1.0,
            direction_static_threshold=0.1,
        )
        artifact = build_causal_shooting_grammar_artifact(
            (
                _observation(0.0, pan=-0.1, tilt=0.1),
                _observation(0.25, pan=-0.100001, tilt=0.100001),
            ),
            provenance=_provenance(),
            smoothing=config,
        )
        self.assertEqual("static", artifact.records[0].camera_pan_direction)
        self.assertEqual("static", artifact.records[0].camera_tilt_direction)
        self.assertEqual("left", artifact.records[1].camera_pan_direction)
        self.assertEqual("down", artifact.records[1].camera_tilt_direction)

    def test_content_addressed_round_trip_is_deterministic_and_no_replace(self) -> None:
        observations = (
            _observation(0.0, source_id="A", native_cut=True),
            _observation(0.0, source_id="B", native_cut=False),
        )
        first = build_causal_shooting_grammar_artifact(
            observations,
            provenance=_provenance(),
        )
        second = build_causal_shooting_grammar_artifact(
            observations,
            provenance=_provenance(),
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            first.artifact_content_sha256,
            second.artifact_content_sha256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_causal_shooting_grammar_artifact(first, root)
            self.assertEqual(
                f"{first.artifact_content_sha256}.json",
                path.name,
            )
            original_bytes = path.read_bytes()
            self.assertEqual(path, write_causal_shooting_grammar_artifact(first, root))
            self.assertEqual(original_bytes, path.read_bytes())
            loaded = load_causal_shooting_grammar_artifact(
                path,
                expected_content_sha256=first.artifact_content_sha256,
            )
            self.assertEqual(first.to_dict(), loaded.to_dict())
            replay_path = loaded.write_content_addressed(root / "replay")
            self.assertEqual(path.name, replay_path.name)
            self.assertEqual(original_bytes, replay_path.read_bytes())

            symlink = root / "artifact-link.json"
            symlink.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "cannot be a symlink"):
                load_causal_shooting_grammar_artifact(symlink)

            bad_root = root / "bad"
            bad_path = (
                bad_root
                / first.artifact_content_sha256[:2]
                / f"{first.artifact_content_sha256}.json"
            )
            bad_path.parent.mkdir(parents=True)
            bad_path.write_text('{"invalid":true}\n', encoding="utf-8")
            before = bad_path.read_bytes()
            with self.assertRaises(ValueError):
                write_causal_shooting_grammar_artifact(first, bad_root)
            self.assertEqual(before, bad_path.read_bytes())

    def test_load_rejects_hash_mask_policy_duplicate_and_nonfinite_tampering(
        self,
    ) -> None:
        artifact = build_causal_shooting_grammar_artifact(
            (_observation(0.0, native_cut=True),),
            provenance=_provenance(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = artifact.write_content_addressed(root)

            def tampered_path(name: str, payload: object) -> Path:
                directory = root / name
                directory.mkdir()
                path = directory / f"{artifact.artifact_content_sha256}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                return path

            changed = deepcopy(artifact.to_dict())
            changed["records"][0]["subject_headroom_normalized"] = 0.9
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                load_causal_shooting_grammar_artifact(
                    tampered_path("changed", changed)
                )

            mask = deepcopy(artifact.to_dict())
            mask["records"][0]["missing_mask"]["camera_pan_velocity"] = True
            with self.assertRaisesRegex(ValueError, "missing mask disagrees"):
                load_causal_shooting_grammar_artifact(tampered_path("mask", mask))

            policy = deepcopy(artifact.to_dict())
            policy["temporal_policy"]["future_frames_used"] = True
            with self.assertRaisesRegex(ValueError, "temporal policy is unsafe"):
                load_causal_shooting_grammar_artifact(
                    tampered_path("policy", policy)
                )

            bool_schema = deepcopy(artifact.to_dict())
            bool_schema["schema_version"] = True
            with self.assertRaisesRegex(ValueError, "schema version"):
                CausalShootingGrammarArtifact.from_dict(bool_schema)

            duplicate_dir = root / "duplicate"
            duplicate_dir.mkdir()
            duplicate = duplicate_dir / original.name
            duplicate.write_text(
                '{"schema_version":1,"schema_version":1}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_causal_shooting_grammar_artifact(duplicate)

            nonfinite_dir = root / "nonfinite"
            nonfinite_dir.mkdir()
            nonfinite = nonfinite_dir / original.name
            nonfinite.write_text('{"schema_version":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                load_causal_shooting_grammar_artifact(nonfinite)

    def test_artifact_persists_no_raw_frames_or_embeddings(self) -> None:
        artifact = build_causal_shooting_grammar_artifact(
            (_observation(0.0, native_cut=True),),
            provenance=_provenance(),
        )
        payload = artifact.to_dict()
        self.assertEqual(
            {
                "raw_frames_persisted": False,
                "pixels_persisted": False,
                "visual_descriptors_persisted": False,
                "identity_or_appearance_embeddings_persisted": False,
                "scalar_features_and_state_only": True,
            },
            payload["storage_policy"],
        )
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw_frame_path", encoded)
        self.assertNotIn("pose_embedding", encoded)
        self.assertNotIn("identity_embedding", encoded)
        self.assertEqual(
            len(STAGE_EDIT_SHOOTING_GRAMMAR_FEATURE_NAMES),
            len(payload["records"][0]["missing_mask"]),
        )

    def test_from_dict_rejects_forged_classification_even_with_rehashed_payload(
        self,
    ) -> None:
        artifact = build_causal_shooting_grammar_artifact(
            (_observation(0.0, native_cut=True),),
            provenance=_provenance(),
        )
        payload = deepcopy(artifact.to_dict())
        payload["records"][0]["camera_pan_direction"] = "right"
        payload["provenance"]["artifact_content_sha256"] = None
        forged_digest = hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        payload["provenance"]["artifact_content_sha256"] = forged_digest
        with self.assertRaisesRegex(ValueError, "pan state is inconsistent"):
            CausalShootingGrammarArtifact.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
