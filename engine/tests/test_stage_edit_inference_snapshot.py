from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut.audio import AlignmentResult, SourceAlignment
from crosscut.causal_shooting_grammar import (
    ShootingGrammarObservation,
    ShootingGrammarProvenance,
    build_causal_shooting_grammar_artifact,
    write_causal_shooting_grammar_artifact,
)
from crosscut.draft import DraftSegment, GroupDraftTimeline
from crosscut.rhythm import (
    Beat,
    OnsetEnvelope,
    PhraseCandidate,
    RhythmAnalysis,
    TempoEstimate,
)
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.ive import IVEMember
from crosscut.ive_performer_state import (
    IVEPerformerStateProvenance,
    IVEPerformerStateSourceProvenance,
    build_ive_performer_state_artifact,
    write_ive_performer_state_artifact,
)
from crosscut.member_state_graph import (
    CausalMemberObservation,
    MemberProbabilityMass,
    MemberStateArtifactProvenance,
    MemberStateSourceProvenance,
    build_member_state_graph,
    write_member_state_graph_artifact,
)
from crosscut.stage_edit_feature_materializer import StageEditCausalSensingArtifacts
from crosscut.stage_edit_inference_snapshot import (
    StageEditInferenceSnapshotError,
    build_stage_edit_inference_decision_times,
    materialize_stage_edit_inference_snapshot,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    stage_edit_recent_history_mask_contract,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor


def _rhythm() -> RhythmAnalysis:
    return RhythmAnalysis(
        sample_rate=100,
        duration_seconds=4.0,
        onset_envelope=OnsetEnvelope(
            values=(0.1, 0.8, 0.2, 0.6),
            sample_rate=4,
            frame_length=2,
            hop_length=1,
        ),
        tempo=TempoEstimate(120.0, 0.5, 0.0, 0.9),
        beats=tuple(
            Beat(index, index * 0.5, 0.9)
            for index in range(8)
        ),
        phrase_candidates=(PhraseCandidate(8, 0, 8, 0.0, 4.0, 0.9),),
    )


def _sections() -> SectionPlan:
    return SectionPlan(
        (
            SectionDirective(
                section_index=0,
                role=SectionRole.VERSE_LIKE,
                start_seconds=0.0,
                end_seconds=2.0,
                energy_score=0.4,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=18.0,
                camera_motion_budget=0.4,
                target_roll_events_per_minute=0.0,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
            SectionDirective(
                section_index=1,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=2.0,
                end_seconds=4.0,
                energy_score=0.9,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=20.0,
                camera_motion_budget=0.8,
                target_roll_events_per_minute=0.4,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _draft() -> GroupDraftTimeline:
    segments = tuple(
        DraftSegment(
            segment_index=index,
            source_index=index % 2,
            canonical_master_start=float(index),
            canonical_master_end=float(index + 1),
            output_start_seconds=float(index),
            source_local_start=float(index),
            source_local_end=float(index + 1),
            audio_offset_seconds=0.0,
            sync_score=1.0,
            estimated_beat_count=2.0,
            start_boundary_reasons=("beat",),
            end_boundary_reasons=("beat",),
        )
        for index in range(4)
    )
    return GroupDraftTimeline(
        source_count=2,
        common_master_start=0.0,
        common_master_end=4.0,
        beat_period_seconds=0.5,
        audio_offsets=(0.0, 0.0),
        sync_scores=(1.0, 0.9),
        segments=segments,
    )


def _alignment(sources: tuple[Path, Path]) -> AlignmentResult:
    return AlignmentResult(
        reference=str(sources[0]),
        sample_rate=8_000,
        alignments=(
            SourceAlignment(
                source=str(sources[0]),
                is_reference=True,
                offset_samples=0,
                offset_seconds=0.0,
                score=1.0,
                overlap_samples=32_000,
                overlap_seconds=4.0,
                score_margin=1.0,
                overlap_ratio=1.0,
                backend="reference",
            ),
            SourceAlignment(
                source=str(sources[1]),
                is_reference=False,
                offset_samples=0,
                offset_seconds=0.0,
                score=0.9,
                overlap_samples=32_000,
                overlap_seconds=4.0,
                second_best_score=0.2,
                score_margin=0.7,
                overlap_ratio=1.0,
                backend="spectral_flux_ncc",
            ),
        ),
    )


def _causal_sensing(
    root: Path,
    *,
    source_ids: tuple[str, ...],
    decision_times: tuple[float, ...],
) -> StageEditCausalSensingArtifacts:
    source_hashes = tuple(str(index + 1) * 64 for index in range(len(source_ids)))
    member_mass = MemberProbabilityMass(
        {member: float(member is IVEMember.GAEUL) * 0.9 for member in IVEMember},
        unknown_mass=0.1,
    )
    member_graph = build_member_state_graph(
        tuple(
            CausalMemberObservation(
                observation_id=f"member-{source_id}",
                source_id=source_id,
                shot_id=f"shot-{source_id}",
                timestamp_seconds=0.0,
                frame_index=0,
                causal_order=index,
                member_probabilities=member_mass,
                face_bbox=(0.42, 0.20, 0.58, 0.38),
                body_bbox=(0.30, 0.10, 0.70, 0.95),
                face_identity_id=f"face-{source_id}",
                tracker_tracklet_id=f"track-{source_id}",
            )
            for index, source_id in enumerate(source_ids)
        )
    )
    member_artifact = write_member_state_graph_artifact(
        root / "member",
        member_graph,
        MemberStateArtifactProvenance(
            producer_id="stageedit-inference-unit-test",
            producer_version="1",
            created_at_utc="2026-08-25T00:00:00Z",
            sources=tuple(
                MemberStateSourceProvenance(
                    source_id=source_id,
                    source_content_sha256=source_hash,
                    observation_manifest_sha256=hashlib.sha256(
                        f"observations:{source_id}".encode()
                    ).hexdigest(),
                )
                for source_id, source_hash in zip(
                    source_ids, source_hashes, strict=True
                )
            ),
        ),
    )
    grammar = build_causal_shooting_grammar_artifact(
        tuple(
            ShootingGrammarObservation(
                source_id=source_id,
                master_time_seconds=timestamp,
                shot_closeup_probability=0.6,
                shot_medium_probability=0.3,
                shot_wide_probability=0.1,
                camera_pan_velocity=0.1,
                camera_tilt_velocity=0.0,
                camera_zoom_in_probability=0.2,
                camera_zoom_out_probability=0.1,
                camera_static_probability=0.7,
                camera_roll_degrees_normalized=0.0,
                subject_headroom_normalized=0.2,
                subject_lead_room_normalized=0.1,
                detected_person_count=0,
                formation_person_count=0,
                internal_native_cut_since_previous=False,
                shooting_grammar_confidence=0.9,
            )
            for timestamp in decision_times
            for source_id in source_ids
        ),
        provenance=ShootingGrammarProvenance(
            upstream_sha256=(("source_catalog", "c" * 64),),
            analyzer_fingerprint_sha256="d" * 64,
            contracts=("inference_exact_timestamp_unit_test_v1",),
        ),
    )
    grammar_path = write_causal_shooting_grammar_artifact(
        grammar,
        root / "grammar",
    )
    performer = build_ive_performer_state_artifact(
        member_graph,
        grammar.records,
        provenance=IVEPerformerStateProvenance(
            analyzer_fingerprint_sha256="e" * 64,
            sources=tuple(
                IVEPerformerStateSourceProvenance(
                    source_id=source_id,
                    source_content_sha256=source_hash,
                )
                for source_id, source_hash in zip(
                    source_ids, source_hashes, strict=True
                )
            ),
            contracts=("inference_member_grammar_unit_test_v1",),
        ),
    )
    performer_path = write_ive_performer_state_artifact(
        root / "performer",
        performer,
    )
    return StageEditCausalSensingArtifacts(
        shooting_grammar_path=grammar_path,
        shooting_grammar_content_sha256=grammar.artifact_content_sha256,
        member_state_graph_path=member_artifact.path,
        member_state_graph_content_sha256=member_artifact.content_sha256,
        target_member=None,
        ive_performer_state_path=performer_path,
        ive_performer_state_content_sha256=performer.content_sha256,
    )


def _descriptor() -> FrameVisualDescriptor:
    return FrameVisualDescriptor(
        quality_score=0.9,
        composition_score=0.8,
        shot_scale=0.5,
        shot_type=ShotType.MEDIUM,
        face_count=0,
        structure_embedding=(0.1, 0.2),
    )


class _FakeSampler:
    next_index = 0

    def __init__(self, _path: Path, *, feature_cache_directory=None) -> None:
        del feature_cache_directory
        index = _FakeSampler.next_index
        _FakeSampler.next_index += 1
        self.frame_rate = 30.0
        self.persistent_feature_cache = SimpleNamespace(
            source_sha256=(str(index + 1) * 64)[:64]
        )

    def feature_descriptor_at(self, *_args, **_kwargs):
        return _descriptor()

    def close(self) -> None:
        return None


class StageEditInferenceSnapshotTests(unittest.TestCase):
    def test_decision_grid_is_dense_and_keeps_exact_section_boundary(self) -> None:
        times = build_stage_edit_inference_decision_times(
            timing_draft=_draft(),
            rhythm=_rhythm(),
            section_plan=_sections(),
            maximum_sequence_length=128,
        )
        self.assertEqual(times[0], 0.0)
        self.assertIn(2.0, times)
        self.assertLess(times[-1], 4.0)
        self.assertLessEqual(
            max(right - left for left, right in zip(times, times[1:])),
            0.5,
        )
        with self.assertRaisesRegex(
            StageEditInferenceSnapshotError,
            "exceeds",
        ):
            build_stage_edit_inference_decision_times(
                timing_draft=_draft(),
                rhythm=_rhythm(),
                section_plan=_sections(),
                maximum_sequence_length=2,
            )

    def test_direct_cache_contract_closes_partial_sampler_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = (root / "a.mp4", root / "b.mp4")
            for source in sources:
                source.write_bytes(b"fixture")
            decision_times = build_stage_edit_inference_decision_times(
                timing_draft=_draft(),
                rhythm=_rhythm(),
                section_plan=_sections(),
                maximum_sequence_length=128,
            )
            causal_sensing = _causal_sensing(
                root / "sensing",
                source_ids=(
                    "source-00-1111111111111111",
                    "source-01-2222222222222222",
                ),
                decision_times=decision_times,
            )
            direct_root = root / "direct-derived"
            direct_root.mkdir()
            calls: list[dict[str, object]] = []
            instances: list[object] = []

            class FailingSecondSampler:
                def __init__(self, _path: Path, **kwargs: object) -> None:
                    self.closed = False
                    calls.append(dict(kwargs))
                    instances.append(self)
                    if len(instances) == 2:
                        raise RuntimeError("required cache failed")

                def close(self) -> None:
                    self.closed = True

            with patch(
                "crosscut.stage_edit_inference_snapshot._VideoSampler",
                FailingSecondSampler,
            ):
                with self.assertRaisesRegex(RuntimeError, "required cache failed"):
                    materialize_stage_edit_inference_snapshot(
                        song_id="NEW IVE SONG",
                        sources=sources,
                        alignment=_alignment(sources),
                        alignment_hard_gates=(True, True),
                        source_timewarps=(None, None),
                        timing_draft=_draft(),
                        rhythm=_rhythm(),
                        section_plan=_sections(),
                        output_directory=root / "snapshots",
                        feature_cache_directory=root / "legacy-cache",
                        feature_derived_cache_directory=direct_root,
                        require_persistent_feature_cache=True,
                        maximum_sequence_length=128,
                        maximum_sources=8,
                        causal_sensing=causal_sensing,
                    )

            self.assertEqual(len(calls), 2)
            for kwargs in calls:
                self.assertEqual(
                    kwargs["feature_cache_directory"],
                    (root / "legacy-cache").resolve(),
                )
                self.assertEqual(
                    kwargs["feature_derived_cache_directory"],
                    direct_root,
                )
                self.assertIs(
                    kwargs["require_persistent_feature_cache"],
                    True,
                )
            self.assertTrue(instances[0].closed)

    def test_actual_media_snapshot_is_content_addressed_and_source_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = (root / "a.mp4", root / "b.mp4")
            for source in sources:
                source.write_bytes(b"fixture")
            decision_times = build_stage_edit_inference_decision_times(
                timing_draft=_draft(),
                rhythm=_rhythm(),
                section_plan=_sections(),
                maximum_sequence_length=128,
            )
            causal_sensing = _causal_sensing(
                root / "sensing",
                source_ids=(
                    "source-00-1111111111111111",
                    "source-01-2222222222222222",
                ),
                decision_times=decision_times,
            )
            with self.assertRaisesRegex(
                StageEditInferenceSnapshotError,
                "complete shooting-grammar/member-state/performer-state",
            ):
                materialize_stage_edit_inference_snapshot(
                    song_id="NEW IVE SONG",
                    sources=sources,
                    alignment=_alignment(sources),
                    alignment_hard_gates=(True, True),
                    source_timewarps=(None, None),
                    timing_draft=_draft(),
                    rhythm=_rhythm(),
                    section_plan=_sections(),
                    output_directory=root / "missing-sensing",
                    feature_cache_directory=root / "cache",
                    maximum_sequence_length=128,
                    maximum_sources=8,
                )
            with self.assertRaisesRegex(
                StageEditInferenceSnapshotError,
                "does not match MEMBER",
            ):
                materialize_stage_edit_inference_snapshot(
                    song_id="NEW IVE SONG",
                    sources=sources,
                    alignment=_alignment(sources),
                    alignment_hard_gates=(True, True),
                    source_timewarps=(None, None),
                    timing_draft=_draft(),
                    rhythm=_rhythm(),
                    section_plan=_sections(),
                    output_directory=root / "wrong-mode",
                    feature_cache_directory=root / "cache",
                    maximum_sequence_length=128,
                    maximum_sources=8,
                    causal_sensing=causal_sensing,
                    edit_mode="member",
                    target_member="가을",
                )
            _FakeSampler.next_index = 0
            with (
                patch(
                    "crosscut.stage_edit_inference_snapshot._VideoSampler",
                    _FakeSampler,
                ),
                patch(
                    "crosscut.stage_edit_inference_snapshot.visual_analyzer_fingerprint_sha256",
                    return_value="a" * 64,
                ),
            ):
                result = materialize_stage_edit_inference_snapshot(
                    song_id="NEW IVE SONG",
                    sources=sources,
                    alignment=_alignment(sources),
                    alignment_hard_gates=(True, True),
                    source_timewarps=(None, None),
                    timing_draft=_draft(),
                    rhythm=_rhythm(),
                    section_plan=_sections(),
                    output_directory=root / "snapshots",
                    feature_cache_directory=root / "cache",
                    maximum_sequence_length=128,
                    maximum_sources=8,
                    causal_sensing=causal_sensing,
                )

            self.assertTrue(result.snapshot_path.is_file())
            self.assertTrue(result.manifest_path.is_file())
            self.assertEqual(len(result.source_ids), 2)
            self.assertEqual(
                tuple(
                    candidate.source_id
                    for candidate in result.snapshot.sequence.tokens[0].source_candidates
                ),
                tuple(sorted(result.source_ids)),
            )
            self.assertEqual(
                result.snapshot.token_schema_fingerprint,
                DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
            )
            self.assertTrue(
                all(
                    item.hard_gate_passed
                    for token in result.snapshot.sequence.tokens
                    for item in token.source_candidates
                )
            )
            recent_start = len(DEFAULT_STAGE_EDIT_SCHEMA.music_feature_names) + len(
                DEFAULT_STAGE_EDIT_SCHEMA.section_feature_names
            )
            recent_stop = recent_start + len(
                DEFAULT_STAGE_EDIT_SCHEMA.recent_edit_feature_names
            )
            for token in result.snapshot.sequence.tokens:
                self.assertEqual(
                    (0.0,) * len(
                        DEFAULT_STAGE_EDIT_SCHEMA.recent_edit_feature_names
                    ),
                    token.recent_edit_features,
                )
                self.assertEqual(
                    (True,) * len(
                        DEFAULT_STAGE_EDIT_SCHEMA.recent_edit_feature_names
                    ),
                    token.missing_mask[recent_start:recent_stop],
                )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                stage_edit_recent_history_mask_contract(
                    DEFAULT_STAGE_EDIT_SCHEMA
                ),
                manifest["decision_history_input_contract"],
            )
            self.assertTrue(
                manifest["input_policy"][
                    "transformer_recent_edit_slice_fully_missing_masked"
                ]
            )
            self.assertFalse(
                manifest["input_policy"][
                    "reference_or_baseline_decision_history_present"
                ]
            )

            with self.assertRaisesRegex(
                StageEditInferenceSnapshotError,
                "every current source",
            ):
                materialize_stage_edit_inference_snapshot(
                    song_id="NEW IVE SONG",
                    sources=sources,
                    alignment=_alignment(sources),
                    alignment_hard_gates=(True, False),
                    source_timewarps=(None, None),
                    timing_draft=_draft(),
                    rhythm=_rhythm(),
                    section_plan=_sections(),
                    output_directory=root / "snapshots",
                    feature_cache_directory=root / "cache",
                    maximum_sequence_length=128,
                    maximum_sources=8,
                    causal_sensing=causal_sensing,
                )


if __name__ == "__main__":
    unittest.main()
