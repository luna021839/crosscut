from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from crosscut.stage_edit_semimarkov_beam import StageEditSourceContinuity
from crosscut.stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME,
    STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES,
    StageEditSourcePairCandidate,
    StageEditSourcePairCheckpointError,
    StageEditSourcePairInputError,
    StageEditSourcePairScorer,
    StageEditSourcePairTrainingGroup,
    load_stage_edit_source_pair_checkpoint,
    materialize_stage_edit_source_pair_features,
    new_stage_edit_source_pair_checkpoint,
    save_stage_edit_source_pair_checkpoint,
    stage_edit_source_pair_reference_balanced_loss,
    train_stage_edit_source_pair_scorer,
)


def _continuity(**values: float) -> StageEditSourceContinuity:
    return StageEditSourceContinuity(**values)


def _candidate(
    source_id: str, continuity: StageEditSourceContinuity
) -> StageEditSourcePairCandidate:
    return StageEditSourcePairCandidate(source_id, continuity)


def _face_group(
    group_id: str,
    *,
    reference_id: str = "reference-face",
    reverse_candidates: bool = False,
) -> StageEditSourcePairTrainingGroup:
    candidates = (
        _candidate("match-low", _continuity(face_scale=0.20)),
        _candidate("far-high", _continuity(face_scale=0.80)),
    )
    if reverse_candidates:
        candidates = tuple(reversed(candidates))
    return StageEditSourcePairTrainingGroup(
        group_id=group_id,
        reference_id=reference_id,
        outgoing_source_id="previous-low",
        outgoing=_continuity(face_scale=0.20),
        candidates=candidates,
        chosen_incoming_source_id="match-low",
    )


def _camera_group(group_id: str) -> StageEditSourcePairTrainingGroup:
    return StageEditSourcePairTrainingGroup(
        group_id=group_id,
        reference_id="reference-camera",
        outgoing_source_id="previous-static",
        outgoing=_continuity(camera_motion_x=0.0, camera_motion_y=0.0),
        candidates=(
            _candidate(
                "match-static",
                _continuity(camera_motion_x=0.0, camera_motion_y=0.0),
            ),
            _candidate(
                "far-pan",
                _continuity(camera_motion_x=0.75, camera_motion_y=0.0),
            ),
        ),
        chosen_incoming_source_id="match-static",
    )


class StageEditSourcePairScorerTests(unittest.TestCase):
    def test_materializer_maps_every_current_continuity_signal(self) -> None:
        features = materialize_stage_edit_source_pair_features(
            _continuity(
                face_scale=0.20,
                face_center_x=0.30,
                face_center_y=0.40,
                body_center_x=0.30,
                body_center_y=0.40,
                pose_motion=0.10,
                camera_motion_x=0.10,
                camera_motion_y=-0.10,
                camera_zoom_velocity=0.10,
                camera_roll_normalized=-0.10,
                sharpness=0.80,
                exposure=0.40,
                color_temperature_normalized=-0.10,
                detected_person_count_normalized=0.20,
                formation_person_count_normalized=0.20,
                formation_pose_coverage=0.80,
            ),
            _continuity(
                face_scale=0.30,
                face_center_x=0.40,
                face_center_y=0.50,
                body_center_x=0.40,
                body_center_y=0.50,
                pose_motion=0.30,
                camera_motion_x=0.30,
                camera_motion_y=0.10,
                camera_zoom_velocity=0.30,
                camera_roll_normalized=0.10,
                sharpness=0.60,
                exposure=0.60,
                color_temperature_normalized=0.10,
                detected_person_count_normalized=0.40,
                formation_person_count_normalized=0.40,
                formation_pose_coverage=0.60,
            ),
        ).by_name
        expected_observed_deltas = set(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES) - {
            "body_center_distance_fallback",
            "missing_evidence_fraction",
        }
        self.assertTrue(
            all(features[name] > 0.0 for name in expected_observed_deltas)
        )
        self.assertEqual(0.0, features["body_center_distance_fallback"])
        self.assertEqual(0.0, features["missing_evidence_fraction"])

    def test_materializer_uses_body_fallback_and_explicit_missing_coverage(
        self,
    ) -> None:
        features = materialize_stage_edit_source_pair_features(
            _continuity(body_center_x=0.20, body_center_y=0.40),
            _continuity(body_center_x=0.34, body_center_y=0.40),
        ).by_name
        self.assertEqual(0.0, features["face_center_distance"])
        self.assertGreater(features["body_center_distance_fallback"], 0.0)
        self.assertAlmostEqual(
            11.0 / 12.0, features["missing_evidence_fraction"], places=12
        )
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features.values()))

    def test_missing_coverage_counts_each_named_evidence_family_once(self) -> None:
        absent = materialize_stage_edit_source_pair_features(
            _continuity(), _continuity()
        ).by_name
        roll_only = materialize_stage_edit_source_pair_features(
            _continuity(camera_roll_normalized=-0.10),
            _continuity(camera_roll_normalized=0.10),
        ).by_name
        face_and_body_centers = materialize_stage_edit_source_pair_features(
            _continuity(
                face_center_x=0.20,
                face_center_y=0.30,
                body_center_x=0.20,
                body_center_y=0.30,
            ),
            _continuity(
                face_center_x=0.30,
                face_center_y=0.40,
                body_center_x=0.30,
                body_center_y=0.40,
            ),
        ).by_name

        self.assertEqual(1.0, absent["missing_evidence_fraction"])
        self.assertAlmostEqual(
            11.0 / 12.0,
            roll_only["missing_evidence_fraction"],
            places=12,
        )
        self.assertAlmostEqual(
            11.0 / 12.0,
            face_and_body_centers["missing_evidence_fraction"],
            places=12,
        )
        self.assertGreater(roll_only["camera_roll_delta"], 0.0)
        self.assertEqual(
            0.0, face_and_body_centers["body_center_distance_fallback"]
        )

    def test_candidate_permutation_is_invariant_and_previous_source_flips_rank(
        self,
    ) -> None:
        weights = [0.0] * len(STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES)
        weights[
            STAGE_EDIT_SOURCE_PAIR_FEATURE_NAMES.index("face_scale_log_delta")
        ] = 1.0
        scorer = StageEditSourcePairScorer(tuple(weights))
        candidates = (
            _candidate("low", _continuity(face_scale=0.20)),
            _candidate("high", _continuity(face_scale=0.80)),
        )

        low_previous = _continuity(face_scale=0.20)
        first = scorer.rank_candidates(low_previous, candidates)
        permuted = scorer.rank_candidates(low_previous, tuple(reversed(candidates)))
        self.assertEqual(
            [(item.source_id, item.score) for item in first],
            [(item.source_id, item.score) for item in permuted],
        )
        self.assertEqual("low", first[0].source_id)
        self.assertTrue(all(-1.0 <= item.score <= 0.0 for item in first))

        high_previous = _continuity(face_scale=0.80)
        changed = scorer.rank_candidates(high_previous, candidates)
        self.assertEqual("high", changed[0].source_id)
        self.assertNotEqual(first[0].source_id, changed[0].source_id)

    def test_reference_balanced_loss_and_training_ignore_dominant_duplication(
        self,
    ) -> None:
        base_groups = (_face_group("face-0"), _camera_group("camera-0"))
        duplicated_groups = tuple(
            _face_group(f"face-{index}") for index in range(9)
        ) + (_camera_group("camera-0"),)
        uniform = StageEditSourcePairScorer.uniform()

        base_loss = stage_edit_source_pair_reference_balanced_loss(
            uniform, base_groups
        )
        duplicated_loss = stage_edit_source_pair_reference_balanced_loss(
            uniform, duplicated_groups
        )
        self.assertAlmostEqual(base_loss, duplicated_loss, places=12)

        base = train_stage_edit_source_pair_scorer(
            base_groups, steps=120, learning_rate=0.30
        )
        duplicated = train_stage_edit_source_pair_scorer(
            duplicated_groups, steps=120, learning_rate=0.30
        )
        np.testing.assert_allclose(
            base.scorer.weights,
            duplicated.scorer.weights,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertLess(base.final_loss, base.initial_loss)
        self.assertTrue(all(weight >= 0.0 for weight in base.scorer.weights))
        self.assertAlmostEqual(1.0, sum(base.scorer.weights), places=12)

        for group in base_groups:
            ranked = base.scorer.rank_candidates(group.outgoing, group.candidates)
            self.assertEqual(group.chosen_incoming_source_id, ranked[0].source_id)

    def test_training_and_corpus_hash_are_candidate_order_deterministic(self) -> None:
        ordered = (_face_group("face-0"), _camera_group("camera-0"))
        permuted = (
            _face_group("face-0", reverse_candidates=True),
            StageEditSourcePairTrainingGroup(
                group_id="camera-0",
                reference_id="reference-camera",
                outgoing_source_id="previous-static",
                outgoing=_continuity(camera_motion_x=0.0, camera_motion_y=0.0),
                candidates=tuple(reversed(_camera_group("unused").candidates)),
                chosen_incoming_source_id="match-static",
            ),
        )
        first = train_stage_edit_source_pair_scorer(ordered, steps=80)
        second = train_stage_edit_source_pair_scorer(permuted, steps=80)
        self.assertEqual(first.corpus_sha256, second.corpus_sha256)
        np.testing.assert_allclose(
            first.scorer.weights, second.scorer.weights, rtol=0.0, atol=0.0
        )

    def test_checkpoint_round_trip_rejects_nonfinite_negative_and_tampering(
        self,
    ) -> None:
        result = train_stage_edit_source_pair_scorer(
            (_face_group("face-0"), _camera_group("camera-0")), steps=60
        )
        checkpoint = new_stage_edit_source_pair_checkpoint(
            "pair-scorer-test",
            result,
            created_at="2026-08-24T12:00:00+00:00",
        )
        self.assertEqual(64, len(checkpoint.checkpoint_sha256))

        with tempfile.TemporaryDirectory() as directory:
            path = save_stage_edit_source_pair_checkpoint(
                Path(directory), checkpoint
            )
            loaded_scorer, loaded = load_stage_edit_source_pair_checkpoint(
                Path(directory),
                expected_corpus_sha256=result.corpus_sha256,
                expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
            )
            self.assertEqual(checkpoint, loaded)
            self.assertEqual(result.scorer, loaded_scorer)
            self.assertEqual(
                STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME, path.name
            )
            original = path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                save_stage_edit_source_pair_checkpoint(Path(directory), checkpoint)
            self.assertEqual(original, path.read_bytes())

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["weights"] = list(reversed(payload["weights"]))
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                StageEditSourcePairCheckpointError, "digest"
            ):
                load_stage_edit_source_pair_checkpoint(Path(directory))

            unsigned = dict(payload)
            unsigned.pop("checkpoint_sha256")
            payload["checkpoint_sha256"] = hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                StageEditSourcePairCheckpointError, "expected digest"
            ):
                load_stage_edit_source_pair_checkpoint(
                    Path(directory),
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                )

        with self.assertRaisesRegex(StageEditSourcePairInputError, "cannot be blank"):
            new_stage_edit_source_pair_checkpoint(
                "pair-scorer-test", result, created_at=""
            )

        valid = list(StageEditSourcePairScorer.uniform().weights)
        negative = list(valid)
        negative[0] = -0.1
        negative[1] += 0.1
        with self.assertRaisesRegex(StageEditSourcePairInputError, "non-negative"):
            StageEditSourcePairScorer(tuple(negative))
        nonfinite = list(valid)
        nonfinite[0] = float("nan")
        with self.assertRaisesRegex(StageEditSourcePairInputError, "finite"):
            StageEditSourcePairScorer(tuple(nonfinite))

    def test_checkpoint_save_and_load_reject_symlink_targets(self) -> None:
        result = train_stage_edit_source_pair_scorer(
            (_face_group("face-0"), _camera_group("camera-0")), steps=10
        )
        checkpoint = new_stage_edit_source_pair_checkpoint(
            "pair-scorer-symlink-test",
            result,
            created_at="2026-08-24T12:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            checkpoint_path = root / STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME
            checkpoint_path.symlink_to(target)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                save_stage_edit_source_pair_checkpoint(root, checkpoint)
            with self.assertRaisesRegex(
                StageEditSourcePairCheckpointError, "non-symlink"
            ):
                load_stage_edit_source_pair_checkpoint(root)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
