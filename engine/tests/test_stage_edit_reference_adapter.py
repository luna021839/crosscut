from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.reference_attribution import REFERENCE_ATTRIBUTION_BASIS
from crosscut.stage_edit_reference_adapter import (
    PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS,
    build_reference_training_example,
    load_reference_training_corpus,
)
from crosscut.stage_edit_training import (
    deterministic_reference_split,
    require_stage_edit_production_supervision,
    validate_stage_edit_production_supervision,
)
from tests.stage_edit_causal_sensing_test_support import (
    production_gate_statistics,
    synthetic_production_snapshot,
)


def _candidate(source_id: str, selected: bool, score: float) -> dict:
    return {
        "source_id": source_id,
        "selected": selected,
        "observed_match_rank": 0 if selected else 1,
        "evidence": {
            "score": score,
            "orb_good_matches": int(score * 100),
            "orb_inlier_ratio": score,
            "structure_ncc": score,
            "color_similarity": score,
        },
    }


def _payload(
    reference_id: str,
    *,
    selected_first: str = "A",
    include_editorial_switch: bool = True,
) -> dict:
    other = "B" if selected_first == "A" else "A"
    spans = (
        (0.0, 1.0, "reference_start", None, False),
        (1.0, 2.5, "same_source_internal_camera_cut", None, False),
        (
            2.5,
            4.0,
            (
                "cross_source_hard_cut"
                if include_editorial_switch
                else "same_source_internal_camera_cut"
            ),
            "hard_cut" if include_editorial_switch else None,
            include_editorial_switch,
        ),
    )
    selected_values = (
        selected_first,
        selected_first,
        other if include_editorial_switch else selected_first,
    )
    shot_labels = []
    listwise = []
    for index, ((start, end, boundary, transition, eligible), selected) in enumerate(
        zip(spans, selected_values, strict=True)
    ):
        rejected = "B" if selected == "A" else "A"
        previous = selected_values[index - 1] if index else None
        event_type = {
            "reference_start": "reference_start",
            "same_source_internal_camera_cut": "native_source_camera_cut",
            "cross_source_hard_cut": "editorial_source_switch_hard_cut",
        }[boundary]
        is_native = event_type == "native_source_camera_cut"
        is_editorial = event_type == "editorial_source_switch_hard_cut"
        labels = {
            "cut": {
                "boundary_label": boundary,
                "cut_master_time": start,
                "end_master_time": end,
                "duration_seconds": end - start,
            },
            "source_transition": {
                "previous_source_id": previous,
                "selected_source_id": selected,
                "is_source_switch": is_editorial,
                "supervision_eligible": eligible,
                "transition": transition,
            },
            "boundary_event": {
                "schema_version": 1,
                "event_type": event_type,
                "event_scope": (
                    "editorial"
                    if is_editorial
                    else "native_source" if is_native else "reference"
                ),
                "previous_source_id": previous,
                "selected_source_id": selected,
                "is_native_source_camera_cut": is_native,
                "is_editorial_source_switch": is_editorial,
                "editorial_timing_supervision_eligible": is_editorial,
                "editorial_source_selection_supervision_eligible": is_editorial,
                "editorial_duration_supervision_eligible": is_editorial,
                "editorial_transition_supervision_eligible": is_editorial,
                "dense_attribution_ready": True,
                "label_basis": "test_fixture",
            },
            "dense_attribution": {
                "schema_version": 1,
                "ready": True,
                "reason": "validated_synchronized_sample_path",
                "first_sample_index": index,
                "last_sample_index": index,
                "sample_count": 1,
            },
            "crop": {"supervision_eligible": False, "transform": None},
        }
        shot_labels.append(
            {
                "shot_index": index,
                "selected_source_id": selected,
                "edit_labels": labels,
            }
        )
        listwise.append(
            {
                "shot_index": index,
                "chosen_source_id": selected,
                "observed_score_margin": 0.8,
                "candidates": [
                    _candidate(selected, True, 0.95),
                    _candidate(rejected, False, 0.15),
                ],
                "edit_labels": labels,
            }
        )
    return {
        "schema_version": 2,
        "dataset_kind": "reference_stage_mix_pairwise_listwise_supervision",
        "style_id": "test-style",
        "reference_id": reference_id,
        "attribution_analysis": REFERENCE_ATTRIBUTION_BASIS,
        "dense_attribution_schema": {
            "schema_version": 1,
            "temporal_granularity": "synchronized_sample_path",
            "sample_level_source_path_available": True,
            "shot_spans_reference_contiguous_source_path_samples": True,
            "frame_dense_boundary_proof_required_for_hard_cut": True,
            "native_source_camera_cut_is_not_editorial_source_switch": True,
        },
        "source_catalog": [
            {"video_id": "A", "alignment_score": 0.8},
            {"video_id": "B", "alignment_score": 0.7},
        ],
        "shot_labels": shot_labels,
        "listwise_records": listwise,
        "pairwise_records": [],
    }


def _canonical_payload(reference_id: str = "ref-a") -> dict:
    payload = _payload(reference_id)
    partition_sha256 = "a" * 64
    attribution_sha256 = "b" * 64
    payload["canonical_shot_schema"] = {
        "schema_version": 1,
        "partition_basis": "reference_hard_cut_detector",
        "reference_evidence_analysis": (
            "reference_hard_cut_reverse_match_evidence_v1"
        ),
        "detected_cut_count": 2,
        "canonical_shot_count": 3,
        "source_runs_are_attribution_only_not_shot_partitions": True,
        "same_source_adjacent_detector_shots_are_never_merged": True,
        "unknown_or_failed_boundary_gate_is_loss_masked": True,
        "partition_content_sha256": partition_sha256,
        "hash_basis": (
            "sha256_canonical_json_of_master_bounds_detected_cuts_and_"
            "sample_assignment"
        ),
        "attribution_content_sha256": attribution_sha256,
        "attribution_hash_basis": (
            "sha256_canonical_json_of_partition_path_shot_aggregates_and_"
            "boundary_labels"
        ),
    }
    payload["reference_evidence_provenance"] = {
        "schema_version": 1,
        "analysis": "reference_hard_cut_reverse_match_evidence_v1",
        "partition_content_sha256": partition_sha256,
        "attribution_content_sha256": attribution_sha256,
    }

    def provenance() -> dict:
        return {
            "schema_version": 1,
            "partition_content_sha256": partition_sha256,
            "attribution_content_sha256": attribution_sha256,
            "reference_hard_cut_detector_owns_partition": True,
            "sample_run_merge_cannot_remove_boundary": True,
        }

    payload["detected_boundary_records"] = [
        {
            "record_id": f"test-style:{reference_id}:detected-boundary:0",
            "style_id": "test-style",
            "reference_id": reference_id,
            "boundary_index": 0,
            "outgoing_shot_index": 0,
            "incoming_shot_index": 1,
            "detector_frame_index": 30,
            "nominal_cut_master_time": 1.0,
            "cut_master_time": 1.0,
            "old_source_id": "A",
            "new_source_id": "A",
            "classification": "native_source_camera_cut_hard_negative",
            "classification_reason": "same_known_source_across_complete_boundary_window",
            "cut_target": 0.0,
            "loss_masked": False,
            "target_masks": {
                "cut": True,
                "source_selection": False,
                "duration": False,
                "transition": False,
                "crop": False,
            },
            "is_editorial_source_switch_positive": False,
            "is_native_source_camera_cut_hard_negative": True,
            "dense_attribution_ready": True,
            "boundary_refinement": None,
            "boundary_path_source_ids": ["A"] * 5,
            "boundary_sample_indices_by_offset": {
                "-2": 0,
                "-1": 1,
                "0": 2,
                "1": 3,
                "2": 4,
            },
            "provenance": provenance(),
        },
        {
            "record_id": f"test-style:{reference_id}:detected-boundary:1",
            "style_id": "test-style",
            "reference_id": reference_id,
            "boundary_index": 1,
            "outgoing_shot_index": 1,
            "incoming_shot_index": 2,
            "detector_frame_index": 75,
            "nominal_cut_master_time": 2.5,
            "cut_master_time": 2.5,
            "old_source_id": "A",
            "new_source_id": "B",
            "classification": "proven_cross_source_editorial_switch_positive",
            "classification_reason": "stable_cross_source_window_and_resolved_frame_boundary",
            "cut_target": 1.0,
            "loss_masked": False,
            "target_masks": {
                "cut": True,
                "source_selection": True,
                "duration": True,
                "transition": True,
                "crop": False,
            },
            "is_editorial_source_switch_positive": True,
            "is_native_source_camera_cut_hard_negative": False,
            "dense_attribution_ready": True,
            "boundary_refinement": {
                "status": "resolved_hard_cut",
                "old_source_id": "A",
                "new_source_id": "B",
                "selected_boundary_seconds": 2.5,
            },
            "boundary_path_source_ids": ["A", "A", "B", "B", "B"],
            "boundary_sample_indices_by_offset": {
                "-2": 5,
                "-1": 6,
                "0": 7,
                "1": 8,
                "2": 9,
            },
            "provenance": provenance(),
        },
    ]
    payload["detected_boundary_statistics"] = {
        "detected_boundary_count": 2,
        "editorial_switch_positive_count": 1,
        "native_source_camera_cut_hard_negative_count": 1,
        "loss_masked_count": 0,
        "all_detector_boundaries_preserved": True,
    }
    return payload


class StageEditReferenceAdapterTests(unittest.TestCase):
    def test_leading_unknown_prefix_starts_at_first_observed_known_shot(self) -> None:
        payload = _canonical_payload()
        payload["shot_labels"] = payload["shot_labels"][1:]
        payload["listwise_records"] = payload["listwise_records"][1:]
        first = payload["shot_labels"][0]
        labels = first["edit_labels"]
        labels["cut"]["boundary_label"] = "reference_start"
        labels["source_transition"].update(
            {
                "previous_source_id": None,
                "is_source_switch": False,
                "supervision_eligible": False,
                "transition": None,
            }
        )
        labels["boundary_event"].update(
            {
                "event_type": "reference_start",
                "event_scope": "reference",
                "previous_source_id": None,
                "is_native_source_camera_cut": False,
                "is_editorial_source_switch": False,
                "editorial_timing_supervision_eligible": False,
                "editorial_source_selection_supervision_eligible": False,
                "editorial_duration_supervision_eligible": False,
                "editorial_transition_supervision_eligible": False,
            }
        )

        example = build_reference_training_example(payload)

        self.assertEqual(1.0, example.sequence.tokens[0].master_time_seconds)
        self.assertIsNone(example.targets.targets[0].cut)
        self.assertEqual("A", example.targets.targets[0].selected_source_id)
        self.assertEqual(
            1, example.statistics["canonical_prefix_boundary_excluded_count"]
        )
        self.assertEqual(2, example.statistics["boundary_token_count"])
        self.assertEqual(1, example.statistics["editorial_source_switch_positive_count"])

    def test_canonical_detector_partition_replaces_collapsed_shot_boundaries(self) -> None:
        example = build_reference_training_example(_canonical_payload())
        self.assertTrue(
            example.statistics["canonical_detected_boundary_partition_used"]
        )
        self.assertEqual(example.statistics["canonical_detected_boundary_count"], 2)
        # Detector boundaries own the partition, while the explicit reference
        # start remains one source-selection/duration token with cut loss
        # masked.  Dynamic crop trajectories therefore have a token for shot 0.
        self.assertEqual(example.statistics["boundary_token_count"], 3)
        self.assertEqual(example.statistics["explicit_boundary_event_count"], 3)
        self.assertEqual(example.statistics["legacy_boundary_event_count"], 0)
        self.assertEqual(
            example.statistics["editorial_source_switch_positive_count"], 1
        )
        self.assertEqual(
            example.statistics["dense_editorial_source_switch_positive_count"],
            1,
        )
        self.assertEqual(example.statistics["native_source_camera_cut_count"], 1)
        boundary_targets = {
            token.master_time_seconds: target
            for token, target in zip(
                example.sequence.tokens, example.targets.targets, strict=True
            )
            if token.master_time_seconds in {1.0, 2.5}
        }
        initial_target = example.targets.targets[0]
        self.assertIsNone(initial_target.cut)
        self.assertEqual(initial_target.selected_source_id, "A")
        self.assertEqual(initial_target.rejected_source_ids, ("B",))
        self.assertEqual(boundary_targets[1.0].cut, 0.0)
        self.assertEqual(boundary_targets[2.5].cut, 1.0)
        self.assertEqual(boundary_targets[2.5].selected_source_id, "B")
        self.assertEqual(boundary_targets[2.5].rejected_source_ids, ("A",))
        self.assertEqual(boundary_targets[2.5].transition_label, "hard_cut")

    def test_canonical_loss_masked_boundary_is_retained_without_any_target(self) -> None:
        payload = _canonical_payload()
        masked = payload["detected_boundary_records"][1]
        masked.update(
            {
                "classification": "loss_masked",
                "classification_reason": "unknown_or_failed_candidate_gate",
                "cut_target": None,
                "loss_masked": True,
                "target_masks": {
                    "cut": False,
                    "source_selection": False,
                    "duration": False,
                    "transition": False,
                    "crop": False,
                },
                "is_editorial_source_switch_positive": False,
                "dense_attribution_ready": False,
                "boundary_refinement": None,
            }
        )
        payload["detected_boundary_statistics"].update(
            {
                "editorial_switch_positive_count": 0,
                "loss_masked_count": 1,
            }
        )
        example = build_reference_training_example(payload)
        self.assertEqual(example.statistics["canonical_loss_masked_boundary_count"], 1)
        target = next(
            target
            for token, target in zip(
                example.sequence.tokens, example.targets.targets, strict=True
            )
            if token.master_time_seconds == 2.5
        )
        self.assertEqual(target, type(target)())

    def test_missing_five_sample_window_is_allowed_only_as_loss_masked(self) -> None:
        payload = _canonical_payload()
        masked = payload["detected_boundary_records"][1]
        masked.update(
            {
                "classification": "loss_masked",
                "classification_reason": "missing_boundary_window_samples",
                "cut_target": None,
                "loss_masked": True,
                "target_masks": {
                    "cut": False,
                    "source_selection": False,
                    "duration": False,
                    "transition": False,
                    "crop": False,
                },
                "is_editorial_source_switch_positive": False,
                "dense_attribution_ready": False,
                "boundary_refinement": None,
                "boundary_path_source_ids": [],
                "boundary_sample_indices_by_offset": {},
            }
        )
        payload["detected_boundary_statistics"].update(
            {
                "editorial_switch_positive_count": 0,
                "loss_masked_count": 1,
            }
        )

        example = build_reference_training_example(payload)
        target = next(
            target
            for token, target in zip(
                example.sequence.tokens, example.targets.targets, strict=True
            )
            if token.master_time_seconds == 2.5
        )
        self.assertEqual(target, type(target)())

        tampered = deepcopy(payload)
        tampered["detected_boundary_records"][1][
            "classification_reason"
        ] = "unknown_or_failed_candidate_gate"
        with self.assertRaisesRegex(ValueError, "complete five-sample path"):
            build_reference_training_example(tampered)

    def test_canonical_dense_or_hash_claim_tamper_fails_closed(self) -> None:
        payload = _canonical_payload()
        payload["detected_boundary_records"][1]["dense_attribution_ready"] = False
        with self.assertRaisesRegex(ValueError, "editorial-switch positive"):
            build_reference_training_example(payload)

        payload = _canonical_payload()
        payload["detected_boundary_records"][1]["provenance"][
            "attribution_content_sha256"
        ] = "c" * 64
        with self.assertRaisesRegex(ValueError, "provenance is inconsistent"):
            build_reference_training_example(payload)

    def test_match_evidence_and_observed_rank_never_change_model_inputs(self) -> None:
        original = _payload("ref-a")
        poisoned = deepcopy(original)
        for record in poisoned["listwise_records"]:
            record["observed_score_margin"] = -999999.0
            record["candidates"].reverse()
            for rank, candidate in enumerate(record["candidates"]):
                candidate["observed_match_rank"] = 999 - rank
                candidate["evidence"] = {
                    name: 999999.0
                    for name in PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS
                }
        first = build_reference_training_example(original)
        second = build_reference_training_example(poisoned)
        self.assertEqual(first.sequence, second.sequence)
        self.assertEqual(first.targets, second.targets)
        self.assertEqual(
            first.statistics["reference_match_features_in_model_input"], 0
        )
        for token in first.sequence.tokens:
            self.assertEqual(
                tuple(candidate.source_id for candidate in token.source_candidates),
                ("A", "B"),
            )
            for candidate in token.source_candidates:
                # Only sync confidence is available in the label-only adapter;
                # every visual/action/formation input remains explicitly
                # missing, independent of the current token schema width.
                self.assertEqual(
                    sum(candidate.missing_mask),
                    len(candidate.missing_mask) - 1,
                )

    def test_cut_negatives_are_strictly_inside_holds_and_unknown_labels_mask(self) -> None:
        example = build_reference_training_example(
            _payload("ref-a"),
            negative_samples_per_hold=1,
            boundary_guard_seconds=0.2,
        )
        self.assertEqual(example.statistics["boundary_token_count"], 3)
        self.assertEqual(example.statistics["cut_positive_count"], 1)
        self.assertEqual(example.statistics["cut_negative_count"], 4)
        self.assertEqual(example.statistics["cut_masked_count"], 1)
        self.assertEqual(
            example.statistics["editorial_source_switch_positive_count"], 1
        )
        self.assertEqual(example.statistics["native_source_camera_cut_count"], 1)
        self.assertEqual(
            example.statistics[
                "native_source_camera_cut_editorial_positive_count"
            ],
            0,
        )
        self.assertEqual(example.statistics["crop_coordinate_supervision_count"], 0)
        negative_times = [
            token.master_time_seconds
            for token, target in zip(
                example.sequence.tokens, example.targets.targets, strict=True
            )
            if target.cut == 0.0
            and token.master_time_seconds not in {0.0, 1.0, 2.5}
        ]
        self.assertEqual(len(negative_times), 3)
        for value, (start, end) in zip(
            negative_times, ((0.0, 1.0), (1.0, 2.5), (2.5, 4.0)), strict=True
        ):
            self.assertGreater(value, start + 0.19)
            self.assertLess(value, end - 0.19)
        boundary_targets = [
            target
            for token, target in zip(
                example.sequence.tokens, example.targets.targets, strict=True
            )
            if token.master_time_seconds in {0.0, 1.0, 2.5}
        ]
        self.assertIsNone(boundary_targets[0].cut)
        self.assertEqual(boundary_targets[1].cut, 0.0)
        # A native broadcast cut is not silently relabelled as an editor cut.
        self.assertIsNone(boundary_targets[1].transition_label)
        self.assertEqual(boundary_targets[2].transition_label, "hard_cut")

    def test_production_gate_requires_dense_switches_across_songs_and_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for reference_id in ("ref-a", "ref-b"):
                path = root / f"{reference_id}.json"
                path.write_text(json.dumps(_payload(reference_id)), encoding="utf-8")
                paths.append(path)
            corpus = load_reference_training_corpus(
                paths,
                song_id_by_reference={"ref-a": "song-a", "ref-b": "song-b"},
            )
            gate = validate_stage_edit_production_supervision(corpus)
            self.assertTrue(gate.passed)
            self.assertEqual(gate.dense_editorial_switch_count, 2)
            self.assertEqual(gate.reference_count_with_dense_editorial_switches, 2)
            self.assertEqual(gate.song_count_with_dense_editorial_switches, 2)
            self.assertEqual(require_stage_edit_production_supervision(corpus), gate)

    def test_production_gate_rejects_internal_camera_cuts_as_only_positives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for reference_id in ("ref-a", "ref-b"):
                path = root / f"{reference_id}.json"
                path.write_text(
                    json.dumps(
                        _payload(reference_id, include_editorial_switch=False)
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            corpus = load_reference_training_corpus(
                paths,
                song_id_by_reference={"ref-a": "song-a", "ref-b": "song-b"},
            )
            gate = validate_stage_edit_production_supervision(corpus)
            self.assertFalse(gate.passed)
            self.assertEqual(gate.dense_editorial_switch_count, 0)
            self.assertIn("no_dense_cross_source_editorial_switches", gate.reasons)
            self.assertEqual(
                corpus.statistics[
                    "native_source_camera_cut_editorial_positive_count"
                ],
                0,
            )
            with self.assertRaisesRegex(
                ValueError, "production supervision gate failed"
            ):
                require_stage_edit_production_supervision(corpus)

    def test_claimed_dense_event_without_payload_provenance_is_rejected(self) -> None:
        payload = _payload("ref-a")
        payload.pop("dense_attribution_schema")
        with self.assertRaisesRegex(
            ValueError, "trusted payload-level provenance"
        ):
            build_reference_training_example(payload)

    def test_corpus_writes_only_to_explicit_manifest_and_split_has_no_group_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for reference_id in ("ref-a", "ref-b", "ref-c"):
                path = root / f"{reference_id}.json"
                path.write_text(json.dumps(_payload(reference_id)), encoding="utf-8")
                paths.append(path)
            corpus = load_reference_training_corpus(
                paths,
                song_id_by_reference={
                    "ref-a": "song-1",
                    "ref-b": "song-1",
                    "ref-c": "song-2",
                },
            )
            reversed_corpus = load_reference_training_corpus(
                list(reversed(paths)),
                song_id_by_reference={
                    "ref-a": "song-1",
                    "ref-b": "song-1",
                    "ref-c": "song-2",
                },
            )
            self.assertEqual(corpus.data_id, reversed_corpus.data_id)
            self.assertEqual(corpus.statistics["reference_count"], 3)
            self.assertEqual(list(root.glob("*manifest*")), [])
            destination = root / "out" / "manifest.json"
            corpus.write_manifest(destination)
            self.assertTrue(destination.is_file())
            split = deterministic_reference_split(corpus, seed=7)
            self.assertEqual(split.split_level, "song")
            train_songs = {item.song_id for item in split.train}
            validation_songs = {item.song_id for item in split.validation}
            self.assertTrue(train_songs.isdisjoint(validation_songs))
            self.assertEqual(split, deterministic_reference_split(corpus, seed=7))

    def test_production_causal_gate_is_immutable_and_data_id_manifest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ref-a.json"
            payload = _payload("ref-a")
            path.write_text(json.dumps(payload), encoding="utf-8")
            base = build_reference_training_example(payload)
            group_snapshot = synthetic_production_snapshot(base.sequence)
            member_snapshot = synthetic_production_snapshot(
                base.sequence,
                edit_mode="member",
                target_member="가을",
            )
            group = load_reference_training_corpus(
                (path,),
                feature_snapshot_by_reference={"ref-a": group_snapshot},
                production_edit_mode="group",
            )
            member = load_reference_training_corpus(
                (path,),
                feature_snapshot_by_reference={"ref-a": member_snapshot},
                production_edit_mode="member",
                production_target_member="가을",
            )
            expected_gate = production_gate_statistics(group_snapshot)
            stored_gate = group.examples[0].statistics[
                "production_causal_sensing_gate"
            ]
            self.assertEqual(expected_gate, dict(stored_gate))
            with self.assertRaises(TypeError):
                stored_gate["edit_mode"] = "member"
            self.assertNotEqual(group.data_id, member.data_id)

            manifest = group.manifest()
            self.assertEqual(
                expected_gate,
                manifest["production_causal_sensing_gate_by_reference"][
                    "ref-a"
                ],
            )
            self.assertEqual(
                expected_gate,
                manifest["examples"][0]["statistics"][
                    "production_causal_sensing_gate"
                ],
            )
            manifest["production_causal_sensing_gate_by_reference"]["ref-a"][
                "edit_mode"
            ] = "member"
            self.assertEqual("group", stored_gate["edit_mode"])
            destination = group.write_manifest(root / "production-manifest.json")
            restored = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(group.data_id, restored["data_id"])
            self.assertEqual(
                expected_gate,
                restored["production_causal_sensing_gate_by_reference"][
                    "ref-a"
                ],
            )

    def test_single_song_corpus_falls_back_to_whole_reference_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for reference_id in ("ref-a", "ref-b"):
                path = root / f"{reference_id}.json"
                path.write_text(json.dumps(_payload(reference_id)), encoding="utf-8")
                paths.append(path)
            corpus = load_reference_training_corpus(
                paths,
                song_id_by_reference={"ref-a": "same-song", "ref-b": "same-song"},
            )
            split = deterministic_reference_split(corpus, seed=11)
            self.assertEqual(split.split_level, "reference")
            self.assertIsNotNone(split.fallback_reason)
            self.assertEqual(len(split.train), 1)
            self.assertEqual(len(split.validation), 1)


if __name__ == "__main__":
    unittest.main()
