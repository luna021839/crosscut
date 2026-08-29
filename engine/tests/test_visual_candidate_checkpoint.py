from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from crosscut.timeline import ShotType
from crosscut.visual_candidate_checkpoint import (
    VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS,
    VISUAL_CANDIDATE_CHECKPOINT_CONTRACT,
    canonical_json_sha256,
    find_latest_matching_visual_candidate_checkpoint,
    load_visual_candidate_checkpoint,
    materialized_candidate_checkpoint_payload,
)
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def _candidate(index: int) -> SimpleNamespace:
    summaries = tuple(
        {"candidate_index": value, "hard_failure_codes": ["focus"]}
        for value in range(8)
    )
    plan = VisualTimelinePlan(
        choices=(
            VisualTimelineChoice(
                segment_index=0,
                source_index=0,
                canonical_master_start=0.0,
                canonical_master_end=1.0,
                source_local_start=0.0,
                source_local_end=1.0,
                quality_score=0.8,
                composition_score=0.7,
                boundary_score=0.6,
                shot_scale=0.3,
                shot_type=ShotType.MEDIUM,
                shot_id="0:0",
            ),
            VisualTimelineChoice(
                segment_index=1,
                source_index=1,
                canonical_master_start=1.0,
                canonical_master_end=2.0,
                source_local_start=1.0,
                source_local_end=2.0,
                quality_score=0.8,
                composition_score=0.7,
                boundary_score=0.6,
                shot_scale=0.3,
                shot_type=ShotType.MEDIUM,
                shot_id="1:1",
            ),
        ),
        optimizer_score=90.0 - index,
        selected_candidate_index=index,
        candidate_review_summaries=summaries,
    )
    return SimpleNamespace(
        candidate_index=index,
        review_rank=index,
        cadence_profile="balanced",
        visual_plan=plan,
        candidate_record=SimpleNamespace(
            review=SimpleNamespace(
                hard_failures=(SimpleNamespace(code="focus_mismatch_exceeded"),)
            )
        ),
    )


class VisualCandidateCheckpointTests(unittest.TestCase):
    def test_round_trip_reconstructs_full_plans_and_exact_initial_eight(self) -> None:
        identity = {"sources": ["a", "b"], "policy": "v1"}
        digest = canonical_json_sha256(identity)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "IVE-XOXZ-pre-derivation-candidate-selection.json"
            payload = {
                "checkpoint_contract": VISUAL_CANDIDATE_CHECKPOINT_CONTRACT,
                "analysis_basis": VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS,
                "input_identity": identity,
                "input_identity_sha256": digest,
                "materialized_candidate_plans": [
                    materialized_candidate_checkpoint_payload(
                        _candidate(index),
                        selected_for_initial_stream=True,
                    )
                    for index in range(8)
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_visual_candidate_checkpoint(
                path,
                expected_input_identity_sha256=digest,
                expected_source_count=2,
            )
            latest = find_latest_matching_visual_candidate_checkpoint(
                root,
                expected_input_identity_sha256=digest,
                expected_source_count=2,
            )

        self.assertEqual(len(loaded.candidates), 8)
        self.assertEqual(len(loaded.initial_candidates), 8)
        self.assertEqual(loaded.candidates[3].candidate_index, 3)
        self.assertEqual(
            loaded.candidates[3].pre_review_hard_failure_codes,
            ("focus_mismatch_exceeded",),
        )
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.path, loaded.path)

    def test_input_identity_mismatch_fails_closed(self) -> None:
        identity = {"sources": ["a", "b"]}
        digest = canonical_json_sha256(identity)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "checkpoint_contract": (
                            VISUAL_CANDIDATE_CHECKPOINT_CONTRACT
                        ),
                        "analysis_basis": (
                            VISUAL_CANDIDATE_CHECKPOINT_ANALYSIS_BASIS
                        ),
                        "input_identity": identity,
                        "input_identity_sha256": digest,
                        "materialized_candidate_plans": [
                            materialized_candidate_checkpoint_payload(
                                _candidate(index),
                                selected_for_initial_stream=True,
                            )
                            for index in range(8)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input identity mismatch"):
                load_visual_candidate_checkpoint(
                    path,
                    expected_input_identity_sha256="f" * 64,
                    expected_source_count=2,
                )


if __name__ == "__main__":
    unittest.main()
