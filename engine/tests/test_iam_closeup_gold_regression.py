from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut import iam_closeup_gold_regression as gold
from crosscut.iam_closeup_gold_regression import (
    IAmCloseupGoldRegressionError,
    load_iam_closeup_gold_regression,
)
from crosscut.professional_motion_cut_profile import (
    MotionCutSoftPreferenceRecommendation,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reseal(payload: dict[str, object]) -> bytes:
    normalized = deepcopy(payload)
    normalized["provenance"]["content_sha256"] = None
    payload["provenance"]["content_sha256"] = _canonical_sha256(normalized)
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")


class IAmCloseupGoldRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_checked_in_asset_is_exact_scalar_only_regression_exemplar(self) -> None:
        exemplar = load_iam_closeup_gold_regression()
        payload = exemplar.to_dict()
        self.assertEqual(exemplar.file_sha256, gold.ASSET_FILE_SHA256)
        self.assertEqual(payload["artifact_policy"]["artifact_role"], "regression_exemplar")
        self.assertFalse(payload["artifact_policy"]["calibration_authority"])
        self.assertFalse(payload["artifact_policy"]["hard_gate"])
        self.assertFalse(payload["artifact_policy"]["production_selection_authority"])
        self.assertTrue(payload["privacy"]["derived_scalars_only"])
        self.assertFalse(payload["privacy"]["raw_vectors_stored"])
        self.assertFalse(payload["privacy"]["raw_pixels_stored"])
        self.assertFalse(payload["privacy"]["face_identity_claimed"])
        self.assertEqual(
            payload["scope"]["video_sha256"],
            gold.EXPECTED_VIDEO_SHA256,
        )
        self.assertEqual(
            payload["scope"]["render_plan_sha256"],
            gold.EXPECTED_RENDER_PLAN_SHA256,
        )
        self.assertEqual(
            payload["scope"]["analysis_sha256"],
            gold.EXPECTED_ANALYSIS_SHA256,
        )
        self.assertEqual(
            [item.frame_index for item in exemplar.boundaries],
            [1607, 1640],
        )
        self.assertTrue(
            all(item.motion_direction_cosine < 0.0 for item in exemplar.boundaries)
        )

    def test_gold_examples_pass_bound_profile_soft_components_without_direction(self) -> None:
        exemplar = load_iam_closeup_gold_regression()
        recommendation = MotionCutSoftPreferenceRecommendation(
            both_moving_minimum_pose_motion=(
                gold.PROFESSIONAL_BOTH_MOVING_MINIMUM
            ),
            continuity_compatible_minimum_score=(
                gold.PROFESSIONAL_CONTINUITY_MINIMUM
            ),
            pose_phase_compatible_preferred_cosine=(
                gold.PROFESSIONAL_SAME_TIME_POSE_SOFT_QUALIFIER
            ),
        )
        results = exemplar.evaluate_same_time_pose_and_both_moving_soft_qualifiers(
            recommendation
        )
        self.assertEqual([item.frame_index for item in results], [1607, 1640])
        for result in results:
            self.assertTrue(result.both_moving_passed)
            self.assertTrue(result.same_time_pose_shape_passed)
            self.assertTrue(result.combined_soft_components_passed)
            self.assertLess(result.direction_cosine_audit_value, 0.0)
            self.assertFalse(result.direction_threshold_bound)
            self.assertFalse(result.direction_used_for_activation)
        self.assertIsNone(recommendation.pose_phase_hard_threshold)
        self.assertFalse(recommendation.hard_gate)

    def test_file_and_content_hashes_are_independently_verified(self) -> None:
        raw = gold.ASSET_PATH.read_bytes()
        path = self.root / "copy.json"
        path.write_bytes(raw + b" ")
        with self.assertRaisesRegex(
            IAmCloseupGoldRegressionError, "file SHA-256 mismatch"
        ):
            load_iam_closeup_gold_regression(path)

        payload = json.loads(raw)
        payload["boundaries"][0]["metrics"]["center_delta"] = 0.08
        tampered = (
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
            + "\n"
        ).encode("utf-8")
        path.write_bytes(tampered)
        with self.assertRaisesRegex(
            IAmCloseupGoldRegressionError, "content SHA-256 mismatch"
        ):
            load_iam_closeup_gold_regression(
                path,
                expected_file_sha256=hashlib.sha256(tampered).hexdigest(),
            )

    def test_rehashed_calibration_or_direction_authority_is_rejected(self) -> None:
        for field in ("calibration_authority", "hard_gate", "may_activate_direction_threshold"):
            with self.subTest(field=field):
                payload = json.loads(gold.ASSET_PATH.read_bytes())
                payload["artifact_policy"][field] = True
                raw = _reseal(payload)
                path = self.root / f"{field}.json"
                path.write_bytes(raw)
                with self.assertRaisesRegex(
                    IAmCloseupGoldRegressionError,
                    "regression-exemplar policy changed",
                ):
                    load_iam_closeup_gold_regression(
                        path,
                        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
                    )

    def test_professional_threshold_mismatch_is_rejected(self) -> None:
        exemplar = load_iam_closeup_gold_regression()
        changed = MotionCutSoftPreferenceRecommendation(
            both_moving_minimum_pose_motion=(
                gold.PROFESSIONAL_BOTH_MOVING_MINIMUM + 0.0001
            ),
            continuity_compatible_minimum_score=(
                gold.PROFESSIONAL_CONTINUITY_MINIMUM
            ),
            pose_phase_compatible_preferred_cosine=(
                gold.PROFESSIONAL_SAME_TIME_POSE_SOFT_QUALIFIER
            ),
        )
        with self.assertRaisesRegex(
            IAmCloseupGoldRegressionError,
            "bound immutable professional motion profile",
        ):
            exemplar.evaluate_same_time_pose_and_both_moving_soft_qualifiers(
                changed
            )


if __name__ == "__main__":
    unittest.main()
