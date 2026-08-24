from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.audio import DEFAULT_ANALYSIS_SAMPLE_RATE
from crosscut.reference_match import DecodedTargetFrames
from crosscut.rhythm import (
    Beat,
    OnsetEnvelope,
    PhraseCandidate,
    RhythmAnalysis,
    TempoEstimate,
)
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_feature_snapshot_job import (
    ACTION_FORMATION_EXTRACTION_CONTRACT,
    _SAFE_ARTIFACT_ID,
    _reference_scaffold,
    _write_content_addressed_json,
    build_stage_edit_feature_snapshots,
    map_reference_master_to_source_seconds,
)
from crosscut.stage_edit_reference_adapter import (
    PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS,
    build_reference_training_example,
)
from crosscut.stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA
from crosscut.timeline import ShotType
from crosscut.visual_score import (
    FrameVisualDescriptor,
    frame_visual_numeric_features,
    visual_analyzer_fingerprint_sha256,
)
from tests.test_stage_edit_reference_adapter import _canonical_payload, _payload


def _clock(offset_seconds: float) -> dict:
    return {
        "basis": "test_scalar_clock_v1",
        "mapping_mode": "scalar_offset",
        "global_offset_seconds": offset_seconds,
        "global_offset_application": "added_to_master_seconds",
        "control_points": [],
        "provenance": {},
    }


def _rhythm() -> RhythmAnalysis:
    return RhythmAnalysis(
        sample_rate=100,
        duration_seconds=14.0,
        onset_envelope=OnsetEnvelope(
            values=(0.1, 0.8, 0.2, 0.5),
            sample_rate=4,
            frame_length=2,
            hop_length=1,
        ),
        tempo=TempoEstimate(120.0, 0.5, 0.0, 0.9),
        beats=(
            Beat(0, 10.0, 0.8),
            Beat(1, 10.5, 0.9),
            Beat(2, 11.0, 0.8),
            Beat(3, 11.5, 0.9),
            Beat(4, 12.0, 0.8),
            Beat(5, 12.5, 0.9),
            Beat(6, 13.0, 0.8),
            Beat(7, 13.5, 0.9),
        ),
        phrase_candidates=(PhraseCandidate(8, 0, 8, 10.0, 14.0, 0.9),),
    )


def _sections() -> SectionPlan:
    return SectionPlan(
        (
            SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=10.0,
                end_seconds=14.0,
                energy_score=0.8,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=18.0,
                camera_motion_budget=0.8,
                target_roll_events_per_minute=0.3,
                establish_with_wide_shot=True,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _shifted_media_payload(reference_id: str, source_paths: tuple[Path, Path]) -> dict:
    payload = _payload(reference_id)
    # The listwise fixture shares these label dictionaries in memory, so one
    # shift updates both serialized locations without introducing drift.
    for shot in payload["shot_labels"]:
        cut = shot["edit_labels"]["cut"]
        cut["cut_master_time"] += 10.0
        cut["end_master_time"] += 10.0
    payload["song_id"] = "IVE-TEST"
    payload["source_catalog"] = [
        {
            "video_id": "A",
            "path": str(source_paths[0]),
            "alignment_score": 0.9,
            "clock": _clock(0.0),
            "media_probe": {
                "frame_rate": 10.0,
                "declared_frame_count": 300,
                "size_bytes": source_paths[0].stat().st_size,
            },
            "provenance": {"coarse_score": 0.9, "coarse_margin": 0.2},
        },
        {
            "video_id": "B",
            "path": str(source_paths[1]),
            "alignment_score": 0.8,
            "clock": _clock(0.25),
            "media_probe": {
                "frame_rate": 10.0,
                "declared_frame_count": 300,
                "size_bytes": source_paths[1].stat().st_size,
            },
            "provenance": {"coarse_score": 0.8, "coarse_margin": 0.1},
        },
    ]
    return payload


class StageEditFeatureSnapshotJobTests(unittest.TestCase):
    def test_real_youtube_reference_ids_can_begin_with_underscore_or_hyphen(
        self,
    ) -> None:
        self.assertIsNotNone(_SAFE_ARTIFACT_ID.fullmatch("_IuHLlXCQpk"))
        self.assertIsNotNone(_SAFE_ARTIFACT_ID.fullmatch("-leading-id"))
        self.assertIsNone(_SAFE_ARTIFACT_ID.fullmatch("../escape"))

    def test_scalar_and_piecewise_clocks_match_reference_clock_contract(self) -> None:
        self.assertAlmostEqual(
            map_reference_master_to_source_seconds(_clock(0.375), 5.0),
            5.375,
        )
        piecewise = {
            "mapping_mode": "piecewise_linear",
            # This scalar is diagnostic only and must not be added again.
            "global_offset_seconds": 99.0,
            "global_offset_application": (
                "diagnostic_scalar_baseline_control_points_are_absolute"
            ),
            "control_points": [
                {"master_seconds": 2.0, "source_seconds": 3.0},
                {"master_seconds": 10.0, "source_seconds": 13.0},
            ],
        }
        self.assertAlmostEqual(
            map_reference_master_to_source_seconds(piecewise, 6.0), 8.0
        )
        self.assertAlmostEqual(
            map_reference_master_to_source_seconds(piecewise, 14.0), 18.0
        )
        self.assertAlmostEqual(
            map_reference_master_to_source_seconds(piecewise, 0.0), 0.5
        )
        broken = deepcopy(piecewise)
        broken["control_points"][1]["source_seconds"] = 2.0
        with self.assertRaisesRegex(ValueError, "monotonic"):
            map_reference_master_to_source_seconds(broken, 5.0)

    def test_content_addressed_writer_is_idempotent_and_collision_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact-deadbeef.json"
            first = _write_content_addressed_json(path, {"b": 2, "a": 1})
            before = first.read_bytes()
            second = _write_content_addressed_json(path, {"a": 1, "b": 2})
            self.assertEqual(second, first)
            self.assertEqual(second.read_bytes(), before)
            with self.assertRaisesRegex(FileExistsError, "collision"):
                _write_content_addressed_json(path, {"a": 999, "b": 2})

            redirected = Path(temporary) / "redirected.json"
            dangling = Path(temporary) / "dangling.json"
            dangling.symlink_to(redirected)
            with self.assertRaisesRegex(FileExistsError, "collision"):
                _write_content_addressed_json(dangling, {"safe": True})
            self.assertFalse(redirected.exists())

            guarded = Path(temporary) / "guarded.json"
            temporary_link = guarded.with_name(f".{guarded.name}.tmp")
            temporary_link.symlink_to(redirected)
            with self.assertRaisesRegex(FileExistsError, "stale"):
                _write_content_addressed_json(guarded, {"safe": True})
            self.assertFalse(redirected.exists())

    def test_reference_scaffold_times_and_histories_are_strictly_causal(self) -> None:
        scaffold = _reference_scaffold(
            _payload("ref-causal"),
            song_id="IVE-TEST",
            negative_samples_per_hold=1,
            boundary_guard_seconds=0.2,
            max_sources=8,
        )
        self.assertEqual(
            scaffold.times,
            (0.0, 0.5, 1.0, 1.75, 2.5, 3.25),
        )
        self.assertEqual(scaffold.histories[:-1], ((), (), (), (), ()))
        self.assertEqual(len(scaffold.histories[-1]), 1)
        previous = scaffold.histories[-1][0]
        self.assertEqual(previous.cut_time_seconds, 2.5)
        self.assertEqual(previous.source_id, "B")
        self.assertEqual(previous.transition_strength, 1.0)
        self.assertLess(previous.cut_time_seconds, scaffold.times[-1])

        canonical_actual = _reference_scaffold(
            _canonical_payload("ref-canonical-exact"),
            song_id="IVE-TEST",
            negative_samples_per_hold=1,
            boundary_guard_seconds=0.2,
            max_sources=8,
        )
        self.assertEqual(
            canonical_actual.times,
            (0.0, 0.5, 1.0, 1.75, 2.5, 3.25),
        )
        self.assertEqual(canonical_actual.histories[0], ())
        self.assertEqual(
            tuple(
                (decision.cut_time_seconds, decision.source_id)
                for decision in canonical_actual.histories[1]
            ),
            ((0.0, "A"),),
        )
        # The cut at 2.5 is a target at that instant, never current context.
        self.assertEqual(
            tuple(
                (decision.cut_time_seconds, decision.source_id)
                for decision in canonical_actual.histories[4]
            ),
            ((0.0, "A"),),
        )
        self.assertEqual(
            tuple(
                (decision.cut_time_seconds, decision.source_id)
                for decision in canonical_actual.histories[5]
            ),
            ((0.0, "A"), (2.5, "B")),
        )

        # Canonical detector evidence, rather than listwise target
        # availability, owns the already-issued source history.
        canonical = _canonical_payload("ref-canonical-causal")
        masked_example = SimpleNamespace(
            reference_id="ref-canonical-causal",
            sequence=SimpleNamespace(
                tokens=(
                    SimpleNamespace(master_time_seconds=2.5),
                    SimpleNamespace(master_time_seconds=3.25),
                )
            ),
            targets=SimpleNamespace(
                targets=(
                    SimpleNamespace(cut=1.0, selected_source_id=None),
                    SimpleNamespace(cut=0.0, selected_source_id=None),
                )
            ),
        )
        with patch(
            "crosscut.stage_edit_feature_snapshot_job.build_reference_training_example",
            return_value=masked_example,
        ):
            canonical_scaffold = _reference_scaffold(
                canonical,
                song_id="IVE-TEST",
                negative_samples_per_hold=1,
                boundary_guard_seconds=0.2,
                max_sources=8,
            )
        self.assertEqual(
            canonical_scaffold.histories[-1][0].source_id,
            "B",
        )

    def test_visual_score_public_wrappers_emit_json_safe_feature_only_payload(self) -> None:
        descriptor = FrameVisualDescriptor(
            quality_score=0.75,
            composition_score=0.65,
            shot_scale=0.4,
            shot_type=ShotType.MEDIUM,
            face_count=1,
            structure_embedding=(0.1, 0.2),
            motion=(0.03, -0.02),
            primary_face_bbox=(0.3, 0.2, 0.2, 0.3),
            color_embedding=(0.5, 0.4, 0.6),
        )
        payload = frame_visual_numeric_features(descriptor)
        self.assertEqual(payload["primary_face_bbox"], [0.3, 0.2, 0.2, 0.3])
        self.assertEqual(payload["motion"], [0.03, -0.02])
        self.assertNotIn("frame", payload)
        self.assertNotIn("identity_embedding", payload)
        json.dumps(payload, allow_nan=False)
        payload["primary_face_bbox"][0] = 999.0
        self.assertEqual(descriptor.primary_face_bbox[0], 0.3)
        with patch(
            "crosscut.visual_score._visual_analyzer_fingerprint",
            return_value=SimpleNamespace(digest="f" * 64),
        ):
            self.assertEqual(visual_analyzer_fingerprint_sha256(), "f" * 64)
        with self.assertRaises(TypeError):
            frame_visual_numeric_features(object())  # type: ignore[arg-type]

    def test_mocked_two_source_job_roundtrips_without_reference_match_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = root / "A.mp4"
            source_b = root / "B.mp4"
            source_a.write_bytes(b"source-a-content")
            source_b.write_bytes(b"source-b-content")
            payload = _shifted_media_payload("ref-e2e", (source_a, source_b))
            supervision = root / "supervision.json"
            supervision.write_text(json.dumps(payload), encoding="utf-8")
            poisoned = deepcopy(payload)
            for record in poisoned["listwise_records"]:
                record["observed_score_margin"] = -999999.0
                record["candidates"].reverse()
                for rank, candidate in enumerate(record["candidates"]):
                    candidate["observed_match_rank"] = 999 - rank
                    candidate["evidence"] = {
                        name: 999999.0
                        for name in PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS
                    }
            poisoned_supervision = root / "supervision-poisoned.json"
            poisoned_supervision.write_text(
                json.dumps(poisoned), encoding="utf-8"
            )
            decode_calls: list[tuple[str, tuple[int, ...], int]] = []
            describe_calls: list[dict[str, object]] = []

            def decode(path: Path, indices: tuple[int, ...], *, config):
                requested = tuple(indices)
                decode_calls.append((Path(path).name, requested, config.working_width))
                frames = {
                    index: np.full((4, 6, 3), index % 251, dtype=np.uint8)
                    for index in requested
                }
                return DecodedTargetFrames(
                    frame_rate=10.0,
                    declared_frame_count=300,
                    requested_indices=requested,
                    frames=frames,
                )

            def describe(frame: np.ndarray, **kwargs) -> FrameVisualDescriptor:
                describe_calls.append(dict(kwargs))
                scalar = float(frame[0, 0, 0]) / 250.0
                return FrameVisualDescriptor(
                    quality_score=0.8,
                    composition_score=0.7,
                    shot_scale=min(1.0, 0.3 + scalar * 0.1),
                    shot_type=ShotType.MEDIUM,
                    face_count=1,
                    structure_embedding=(scalar, 1.0 - scalar),
                    motion=(0.01, -0.02),
                    primary_face_bbox=(0.3, 0.2, 0.2, 0.3),
                    color_embedding=(0.5, 0.4, 0.6, 0.1, 0.1, 0.1, 0.7, 0.2),
                )

            with patch(
                "crosscut.stage_edit_feature_snapshot_job.extract_mono_pcm",
                return_value=np.zeros(1400, dtype=np.float64),
            ) as extract, patch(
                "crosscut.stage_edit_feature_snapshot_job.analyze_rhythm",
                return_value=_rhythm(),
            ), patch(
                "crosscut.stage_edit_feature_snapshot_job.build_section_plan",
                return_value=_sections(),
            ) as build_sections, patch(
                "crosscut.stage_edit_feature_snapshot_job.decode_target_frames_sequential",
                side_effect=decode,
            ), patch(
                "crosscut.stage_edit_feature_snapshot_job.describe_frame",
                side_effect=describe,
            ), patch(
                "crosscut.stage_edit_feature_snapshot_job.visual_analyzer_fingerprint_sha256",
                return_value="a" * 64,
            ):
                result = build_stage_edit_feature_snapshots(
                    (supervision,),
                    song_id="IVE-TEST",
                    output_directory=root / "out",
                    expected_source_count=2,
                    negative_samples_per_hold=1,
                    boundary_guard_seconds=0.2,
                    max_sources=8,
                    working_width=320,
                )
                poisoned_result = build_stage_edit_feature_snapshots(
                    (poisoned_supervision,),
                    song_id="IVE-TEST",
                    output_directory=root / "out-poisoned",
                    expected_source_count=2,
                    negative_samples_per_hold=1,
                    boundary_guard_seconds=0.2,
                    max_sources=8,
                    working_width=320,
                )

            self.assertEqual(extract.call_count, 2)
            for actual in extract.call_args_list:
                self.assertEqual(actual.args, (source_a.resolve(),))
                self.assertEqual(
                    actual.kwargs,
                    {
                        "sample_rate": DEFAULT_ANALYSIS_SAMPLE_RATE,
                        "start_seconds": None,
                        "duration_seconds": 14.0,
                    },
                )
            self.assertEqual(build_sections.call_count, 2)
            self.assertEqual(build_sections.call_args.kwargs["start_seconds"], 10.0)
            self.assertEqual(build_sections.call_args.kwargs["end_seconds"], 14.0)
            self.assertEqual(
                [call[0] for call in decode_calls],
                ["A.mp4", "B.mp4", "A.mp4", "B.mp4"],
            )
            self.assertTrue(all(call[2] == 320 for call in decode_calls))
            self.assertTrue(describe_calls)
            self.assertTrue(
                all(call.get("include_subject_geometry") is True for call in describe_calls)
            )
            self.assertTrue(
                all(call.get("include_formation_pose") is True for call in describe_calls)
            )
            self.assertEqual(
                set(decode_calls[0][1]),
                {100, 101, 105, 106, 110, 111, 118, 119, 125, 126, 132, 133},
            )
            self.assertEqual(
                set(decode_calls[1][1]),
                {102, 103, 108, 109, 112, 113, 120, 121, 128, 129, 135, 136},
            )

            snapshot_payload = json.loads(
                result.snapshots["ref-e2e"].read_text(encoding="utf-8")
            )
            manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["schema_version"], 2)
            self.assertIn(
                ACTION_FORMATION_EXTRACTION_CONTRACT,
                manifest_payload["feature_provenance"]["contracts"],
            )
            self.assertIn(
                "model_pose60_observed_time_count",
                manifest_payload["statistics"]["per_source"]["A"],
            )
            snapshot = StageEditFeatureSnapshot.from_dict(snapshot_payload)
            self.assertEqual(
                poisoned_result.snapshot_sha256["ref-e2e"],
                snapshot.snapshot_sha256,
            )
            self.assertEqual(
                tuple(token.master_time_seconds for token in snapshot.sequence.tokens),
                (10.0, 10.5, 11.0, 11.75, 12.5, 13.25),
            )
            self.assertEqual(snapshot.snapshot_sha256, result.snapshot_sha256["ref-e2e"])
            self.assertIn(snapshot.snapshot_sha256[:16], result.snapshots["ref-e2e"].name)
            self.assertEqual(
                StageEditFeatureSnapshot.from_dict(snapshot.to_dict()), snapshot
            )
            training = build_reference_training_example(
                payload,
                feature_snapshot=snapshot,
                song_id="IVE-TEST",
                negative_samples_per_hold=1,
                boundary_guard_seconds=0.2,
                max_sources=8,
            )
            self.assertEqual(training.sequence.tokens, snapshot.sequence.tokens)
            self.assertEqual(training.feature_snapshot_sha256, snapshot.snapshot_sha256)

            serialized = json.dumps(snapshot_payload, sort_keys=True)
            for prohibited in PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS:
                self.assertNotIn(f'"{prohibited}"', serialized)
            recent_index = {
                name: index
                for index, name in enumerate(
                    DEFAULT_STAGE_EDIT_SCHEMA.recent_edit_feature_names
                )
            }
            before_cut = snapshot.token_at(12.5)
            after_cut = snapshot.token_at(13.25)
            recent_offset = (
                len(DEFAULT_STAGE_EDIT_SCHEMA.music_feature_names)
                + len(DEFAULT_STAGE_EDIT_SCHEMA.section_feature_names)
            )
            self.assertTrue(
                before_cut.missing_mask[
                    recent_offset + recent_index["seconds_since_cut_normalized"]
                ]
            )
            self.assertAlmostEqual(
                after_cut.recent_edit_features[
                    recent_index["seconds_since_cut_normalized"]
                ],
                0.75 / 8.0,
            )
            self.assertAlmostEqual(
                after_cut.recent_edit_features[recent_index["recent_cut_density"]],
                1.0 / 8.0,
            )
            self.assertAlmostEqual(
                after_cut.recent_edit_features[
                    recent_index["previous_transition_strength"]
                ],
                1.0,
            )

            unsafe = deepcopy(payload)
            unsafe["reference_id"] = "../escape"
            unsafe_path = root / "unsafe-supervision.json"
            unsafe_path.write_text(json.dumps(unsafe), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safe artifact ids"):
                build_stage_edit_feature_snapshots(
                    (unsafe_path,),
                    song_id="IVE-TEST",
                    output_directory=root / "unsafe-out",
                )
            self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
