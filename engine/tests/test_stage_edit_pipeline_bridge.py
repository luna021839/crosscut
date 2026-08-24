from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut.ive import AspectRatio, EditMode, SourceMode
from crosscut.pipeline import (
    _discover_automatic_stage_edit_production_bundle,
    _load_stage_edit_generation_preflight,
    _materialize_registered_stage_edit_preflight,
    _stage_edit_continuity_config_identity,
    _stage_edit_semimarkov_config_from_montage_policy,
    analyze,
)
from crosscut.render import RenderSegment
from crosscut.request import AnalysisRequest
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit import StageEdit, StageShot
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_head_materializer import (
    StageEditHeadMaterializationResult,
)
from crosscut.stage_edit_learned_section_director import (
    LearnedSectionBudgetEvidence,
    LearnedSectionDirectorResult,
)
from crosscut.stage_edit_pipeline_bridge import (
    STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
    STAGE_EDIT_PIPELINE_BRIDGE_SCHEMA_VERSION,
    RetainedStageEditCandidate,
    StageEditPipelineBridgeError,
    StageEditPipelineBridgeInputError,
    StageEditPipelineBridgeNotReadyError,
    StageEditPipelineBridgePreflight,
    StageEditPipelineBridgeResult,
    preflight_stage_edit_pipeline_bridge,
    run_stage_edit_pipeline_bridge,
)
from crosscut.stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    StageEditBeamTimeline,
    StageEditDecodeAudit,
    StageEditHeadFrame,
    StageEditSemiMarkovConfig,
    StageEditSemiMarkovResult,
)
from crosscut.stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME,
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
    StageEditSourcePairScorer,
    StageEditSourcePairTrainingResult,
    new_stage_edit_source_pair_checkpoint,
    save_stage_edit_source_pair_checkpoint,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditReadinessReport,
    StageEditSequence,
    StageEditToken,
)


def _candidate(source_id: str, value: float) -> SourceCandidateToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    return SourceCandidateToken(
        source_id=source_id,
        features=(value,) * schema.raw_source_dimension,
        missing_mask=(False,) * schema.raw_source_dimension,
        available=True,
        hard_gate_passed=True,
    )


def _snapshot() -> StageEditFeatureSnapshot:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    sequence = StageEditSequence(
        "ive:bridge",
        (
            StageEditToken(
                master_time_seconds=0.0,
                music_features=(0.1,) * len(schema.music_feature_names),
                section_features=(0.2,) * len(schema.section_feature_names),
                source_candidates=(
                    _candidate("source-a", 0.3),
                    _candidate("source-b", 0.4),
                ),
                recent_edit_features=(0.0,)
                * len(schema.recent_edit_feature_names),
                style_features=(0.0,) * len(schema.style_feature_names),
                missing_mask=(False,) * schema.raw_context_dimension,
            ),
        ),
    )
    return StageEditFeatureSnapshot.build(sequence)


def _ready(checkpoint: Path, *, dynamic_crop_version: int = 2) -> StageEditReadinessReport:
    checkpoint_format = f"dynamic_crop_v{dynamic_crop_version}"
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=True,
        status="production_checkpoint_ready",
        reason="test production evidence",
        mlx_version="test",
        checkpoint_path=str(checkpoint.resolve()),
        training_steps=10,
        checkpoint_format=checkpoint_format,
        dynamic_crop_version=dynamic_crop_version,
        checkpoint_metadata_filename=(
            f"dynamic-crop-v{dynamic_crop_version}-metadata.json"
        ),
    )


def _not_ready(checkpoint: Path, *, dynamic_crop_version: int = 2) -> StageEditReadinessReport:
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=False,
        status="production_training_report_missing",
        reason="test report missing",
        mlx_version="test",
        checkpoint_path=str(checkpoint.resolve()),
        training_steps=10,
        checkpoint_format=f"dynamic_crop_v{dynamic_crop_version}",
        dynamic_crop_version=dynamic_crop_version,
        checkpoint_metadata_filename=(
            f"dynamic-crop-v{dynamic_crop_version}-metadata.json"
        ),
    )


def _saved_uniform_source_pair_scorer(
    root: Path,
) -> tuple[StageEditSourcePairScorer, Path, str, str, str]:
    scorer = StageEditSourcePairScorer.uniform()
    corpus_sha256 = "c" * 64
    result = StageEditSourcePairTrainingResult(
        scorer=scorer,
        training_steps=1,
        reference_count=1,
        group_count=1,
        initial_loss=0.0,
        final_loss=0.0,
        margin=0.0,
        corpus_sha256=corpus_sha256,
    )
    checkpoint = new_stage_edit_source_pair_checkpoint(
        "test-source-pair",
        result,
        created_at="2026-08-24T12:00:00+09:00",
    )
    path = save_stage_edit_source_pair_checkpoint(root, checkpoint)
    return (
        scorer,
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint.checkpoint_sha256,
        corpus_sha256,
    )


def _section_plan() -> SectionPlan:
    return SectionPlan(
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
                target_source_switches_per_minute=20.0,
                camera_motion_budget=0.5,
                target_roll_events_per_minute=0.0,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _edit(sources: tuple[Path, ...]) -> StageEdit:
    section_plan = _section_plan()
    return StageEdit(
        sources=sources,
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        width=1920,
        height=1080,
        fps=30.0,
        master_audio_start=0.0,
        section_plan=section_plan,
        shots=(
            StageShot(
                shot_index=0,
                section_index=0,
                segment=RenderSegment(0, 0.0, 2.0, 0.0),
                decision_id="test-decision",
                decision_confidence=0.8,
            ),
        ),
        boundaries=(),
        output=Path("/tmp/stage-edit-bridge.mp4"),
    )


@dataclass(frozen=True)
class _FakeHeadResult:
    frames: tuple[object, ...] = (object(),)
    dynamic_crop_version: int = 2

    def to_dict(self):
        return {"frame_count": len(self.frames)}


@dataclass(frozen=True)
class _FakeDecodeResult:
    timelines: tuple[object, ...]
    audit: object

    def to_dict(self):
        return {
            "audit": {"request_fulfilled": self.audit.request_fulfilled},
            "timeline_count": len(self.timelines),
        }


def _typed_head_result(
    checkpoint: Path,
    *,
    dynamic_crop_version: int = 2,
) -> StageEditHeadMaterializationResult:
    frame = StageEditHeadFrame(
        master_time_seconds=0.0,
        cut_logit=0.0,
        source_logits=(1.0, 0.0),
        duration_logits=(1.0,),
        transition_logits=(1.0,),
        crop=(1.0, 0.5, 0.5, 0.0),
        available_source_mask=(True, True),
        hard_source_gate_mask=(True, True),
    )
    digest = "a" * 64
    return StageEditHeadMaterializationResult(
        frames=(frame,),
        sequence_id="ive:bridge",
        canonical_source_ids=("source-a", "source-b"),
        checkpoint_directory=str(checkpoint.resolve()),
        checkpoint_id="test-checkpoint",
        training_steps=10,
        metadata_sha256=digest,
        weights_sha256=digest,
        supplied_sequence_sha256=digest,
        input_sequence_sha256=digest,
        output_frames_sha256=digest,
        token_schema_fingerprint=DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
        dynamic_crop_config_fingerprint=digest,
        readiness=_ready(
            checkpoint,
            dynamic_crop_version=dynamic_crop_version,
        ),
        decision_history_input_contract={"contract_sha256": digest},
        checkpoint_format=f"dynamic_crop_v{dynamic_crop_version}",
        dynamic_crop_version=dynamic_crop_version,
        checkpoint_metadata_filename=(
            f"dynamic-crop-v{dynamic_crop_version}-metadata.json"
        ),
    )


def _typed_section_director(
    *,
    dynamic_crop_version: int = 2,
) -> LearnedSectionDirectorResult:
    source = _section_plan().sections[0]
    directed_plan = SectionPlan(
        (replace(source, establish_with_wide_shot=True),)
    )
    evidence = LearnedSectionBudgetEvidence(
        section_index=0,
        frame_count=1,
        mean_threshold_centered_cut_probability=0.0,
        learned_switch_rate_multiplier=1.0,
        base_source_switches_per_minute=20.0,
        learned_source_switches_per_minute=20.0,
        mean_dynamic_motion_probability=None,
        base_camera_motion_budget=0.5,
        learned_camera_motion_budget=0.5,
        mean_dynamic_roll_support=None,
        base_roll_events_per_minute=0.0,
        learned_roll_events_per_minute=0.0,
        base_preferred_source_hold_beats=2,
        learned_preferred_source_hold_beats=2,
        learned_top_source_wide_support=1.0,
        chorus_wide_anchor_selected=True,
        dynamic_crop_version=dynamic_crop_version,
    )
    return LearnedSectionDirectorResult(
        section_plan=directed_plan,
        evidence=(evidence,),
        cut_probability_threshold=0.5,
        cut_threshold_freeze_sha256="b" * 64,
        checkpoint_id="test-checkpoint",
        training_report_path="/tmp/test-training-report.json",
        training_report_sha256="c" * 64,
        chorus_wide_anchor_requested_fraction=0.42,
        chorus_section_count=1,
        chorus_wide_anchor_allocated_count=1,
        chorus_wide_anchor_allocated_section_indices=(0,),
        dynamic_crop_version=dynamic_crop_version,
    )


def _continuity_provenance(
    *,
    authority: str,
    scorer_contract: str,
    scorer_fingerprint: str,
) -> dict[str, str]:
    return {
        "authority": authority,
        "scorer_contract": scorer_contract,
        "scorer_fingerprint": scorer_fingerprint,
    }


def _typed_timelines(
    sources: tuple[Path, ...],
    *,
    count: int,
    authority: str,
    scorer_contract: str,
    scorer_fingerprint: str,
) -> tuple[StageEditBeamTimeline, ...]:
    provenance = _continuity_provenance(
        authority=authority,
        scorer_contract=scorer_contract,
        scorer_fingerprint=scorer_fingerprint,
    )
    decision = SimpleNamespace(
        source_index=0,
        master_start_seconds=0.0,
        duration_seconds=2.0,
        editorial_transform=None,
    )
    return tuple(
        StageEditBeamTimeline(
            rank=rank,
            score=4.2 - rank * 0.01,
            score_components={},
            decisions=(decision,),
            edit=_edit(sources),
            source_occupancy_seconds=(2.0, 0.0),
            recent_cut_density_per_minute=0.0,
            boundary_continuity_score_provenance=provenance,
        )
        for rank in range(1, count + 1)
    )


def _typed_decode_result(
    timelines: tuple[StageEditBeamTimeline, ...],
    *,
    config: StageEditSemiMarkovConfig,
    scorer_contract: str,
    scorer_fingerprint: str,
    request_fulfilled: bool,
) -> StageEditSemiMarkovResult:
    provenance = _continuity_provenance(
        authority=config.continuity_authority,
        scorer_contract=scorer_contract,
        scorer_fingerprint=scorer_fingerprint,
    )
    audit = StageEditDecodeAudit(
        schema_version=1,
        decoder="typed-test-semi-markov",
        input_sha256="d" * 64,
        readiness={},
        config={
            **asdict(config),
            "continuity_scorer_contract": scorer_contract,
            "continuity_scorer_fingerprint": scorer_fingerprint,
        },
        input_summary={
            "boundary_continuity_score_provenance": provenance,
        },
        endpoint_audit=(),
        constraint_rejection_counts={},
        transition_hypothesis_count=0,
        accepted_hypothesis_count=0,
        beam_pruned_state_count=0,
        complete_state_count=len(timelines),
        duplicate_completion_count=0,
        requested_timeline_count=config.timeline_count,
        returned_timeline_count=len(timelines),
        request_fulfilled=request_fulfilled,
        score_provenance={"boundary_continuity": provenance},
    )
    return StageEditSemiMarkovResult(timelines=timelines, audit=audit)


def _typed_direct_bridge_result(
    preflight: StageEditPipelineBridgePreflight,
    *,
    decode_authority: str | None = None,
    decode_scorer_contract: str | None = None,
    decode_scorer_fingerprint: str | None = None,
) -> StageEditPipelineBridgeResult:
    expected_contract = (
        preflight.source_pair_scorer.contract
        if preflight.source_pair_scorer is not None
        else HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
    )
    expected_fingerprint = (
        preflight.source_pair_scorer_fingerprint
        if preflight.source_pair_scorer_fingerprint is not None
        else HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
    )
    authority = decode_authority or preflight.continuity_authority
    scorer_contract = decode_scorer_contract or expected_contract
    scorer_fingerprint = decode_scorer_fingerprint or expected_fingerprint
    config = StageEditSemiMarkovConfig(
        minimum_unique_sources=1,
        continuity_authority=authority,
        continuity_scorer_contract=scorer_contract,
        continuity_scorer_fingerprint=scorer_fingerprint,
    )
    sources = (
        Path("/tmp/stageedit-direct-a.mp4"),
        Path("/tmp/stageedit-direct-b.mp4"),
    )
    timelines = _typed_timelines(
        sources,
        count=config.timeline_count,
        authority=authority,
        scorer_contract=scorer_contract,
        scorer_fingerprint=scorer_fingerprint,
    )
    decoded = _typed_decode_result(
        timelines,
        config=config,
        scorer_contract=scorer_contract,
        scorer_fingerprint=scorer_fingerprint,
        request_fulfilled=True,
    )
    retained = tuple(
        RetainedStageEditCandidate(
            rank=timeline.rank,
            score=timeline.score,
            edit=replace(
                timeline.edit,
                candidate_history=(
                    *timeline.edit.candidate_history,
                    {
                        "bridge": STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT,
                        "dynamic_crop_version": (
                            preflight.dynamic_crop_version
                        ),
                        "continuity_authority": (
                            preflight.continuity_authority
                        ),
                    },
                ),
            ),
            exact_clock_reprojected=True,
            timewarped_segment_count=0,
        )
        for timeline in timelines
    )
    return StageEditPipelineBridgeResult(
        preflight=preflight,
        head_materialization=_typed_head_result(
            preflight.checkpoint_directory,
            dynamic_crop_version=preflight.dynamic_crop_version,
        ),
        semimarkov=decoded,
        retained_candidates=retained,
        selection_policy={
            "visual_baseline_replaced": False,
            "selected_for_render": False,
            "dynamic_crop_version": preflight.dynamic_crop_version,
            "continuity_authority": preflight.continuity_authority,
            "continuity_scorer_contract": expected_contract,
            "continuity_scorer_fingerprint": expected_fingerprint,
        },
    )


class StageEditPipelineBridgeTests(unittest.TestCase):
    def test_bridge_result_rejects_untyped_preflight_before_cross_object_use(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "requires a typed production preflight",
        ):
            StageEditPipelineBridgeResult(
                preflight=SimpleNamespace(
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                    ),
                ),
                head_materialization=SimpleNamespace(dynamic_crop_version=2),
                semimarkov=SimpleNamespace(),
                retained_candidates=(),
                selection_policy={
                    "visual_baseline_replaced": False,
                    "selected_for_render": False,
                },
            )

    def test_bridge_result_rejects_untyped_nested_production_results(
        self,
    ) -> None:
        checkpoint = Path("/tmp/stageedit-typed-result-boundary").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=_snapshot(),
            readiness=_ready(checkpoint),
        )
        valid = _typed_direct_bridge_result(preflight)
        cases = (
            (
                {"head_materialization": _FakeHeadResult()},
                "typed head materialization",
            ),
            (
                {
                    "semimarkov": _FakeDecodeResult(
                        timelines=(),
                        audit=SimpleNamespace(request_fulfilled=True),
                    )
                },
                "typed semi-Markov result",
            ),
            (
                {
                    "semimarkov": StageEditSemiMarkovResult(
                        timelines=valid.semimarkov.timelines,
                        audit=SimpleNamespace(),
                    )
                },
                "typed semi-Markov audit",
            ),
            (
                {
                    "semimarkov": StageEditSemiMarkovResult(
                        timelines=(SimpleNamespace(),),
                        audit=valid.semimarkov.audit,
                    )
                },
                "typed semi-Markov timelines",
            ),
            (
                {
                    "learned_section_director": SimpleNamespace(
                        dynamic_crop_version=2
                    )
                },
                "typed learned Section Director",
            ),
        )
        for replacements, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(TypeError, reason):
                    replace(valid, **replacements)

    def test_bridge_result_cross_binds_all_semimarkov_continuity_provenance(
        self,
    ) -> None:
        checkpoint = Path("/tmp/stageedit-provenance-boundary").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=_snapshot(),
            readiness=_ready(checkpoint),
        )
        valid = _typed_direct_bridge_result(preflight)
        wrong = {
            "authority": PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            "scorer_contract": STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            "scorer_fingerprint": "e" * 64,
        }
        audit = valid.semimarkov.audit
        mutations = (
            (
                replace(
                    audit,
                    config={
                        **audit.config,
                        "continuity_authority": wrong["authority"],
                    },
                ),
                valid.semimarkov.timelines,
                "audit config",
            ),
            (
                replace(
                    audit,
                    score_provenance={"boundary_continuity": wrong},
                ),
                valid.semimarkov.timelines,
                "identity in audit",
            ),
            (
                replace(
                    audit,
                    input_summary={
                        **audit.input_summary,
                        "boundary_continuity_score_provenance": wrong,
                    },
                ),
                valid.semimarkov.timelines,
                "identity in input summary",
            ),
            (
                audit,
                (
                    replace(
                        valid.semimarkov.timelines[0],
                        boundary_continuity_score_provenance=wrong,
                    ),
                    *valid.semimarkov.timelines[1:],
                ),
                "identity in timeline rank 1",
            ),
        )
        for mutated_audit, timelines, reason in mutations:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    replace(
                        valid,
                        semimarkov=StageEditSemiMarkovResult(
                            timelines=timelines,
                            audit=mutated_audit,
                        ),
                    )

    def test_bridge_result_cross_binds_typed_head_identity_to_preflight(
        self,
    ) -> None:
        checkpoint = Path("/tmp/stageedit-head-identity-boundary").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=_snapshot(),
            readiness=_ready(checkpoint),
        )
        valid = _typed_direct_bridge_result(preflight)
        mismatched_head = replace(
            valid.head_materialization,
            canonical_source_ids=("source-b", "source-a"),
        )
        with self.assertRaisesRegex(ValueError, "source identity differs"):
            replace(valid, head_materialization=mismatched_head)

    def test_learned_preflight_rejects_handcrafted_decode_provenance(
        self,
    ) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        checkpoint = root / "stageedit-learned-result-boundary"
        (
            scorer,
            scorer_path,
            scorer_file_sha256,
            scorer_checkpoint_sha256,
            scorer_corpus_sha256,
        ) = _saved_uniform_source_pair_scorer(root / "source-pair")
        readiness = _ready(checkpoint, dynamic_crop_version=3)
        with patch(
            "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
            return_value=readiness,
        ):
            preflight = StageEditPipelineBridgePreflight(
                snapshot_path=root / "stageedit-feature.json",
                checkpoint_directory=checkpoint,
                source_ids=("source-a", "source-b"),
                snapshot=_snapshot(),
                readiness=readiness,
                dynamic_crop_version=3,
                continuity_authority=(
                    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                ),
                source_pair_scorer=scorer,
                source_pair_scorer_path=scorer_path,
                source_pair_scorer_file_sha256=scorer_file_sha256,
                source_pair_scorer_fingerprint=scorer.fingerprint,
                source_pair_checkpoint_sha256=scorer_checkpoint_sha256,
                source_pair_corpus_sha256=scorer_corpus_sha256,
            )
        with self.assertRaisesRegex(ValueError, "audit config"):
            _typed_direct_bridge_result(
                preflight,
                decode_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
                decode_scorer_contract=(
                    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
                ),
                decode_scorer_fingerprint=(
                    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
                ),
            )

    def test_automatic_group_strict_boundary_rejects_legacy_identity_default(self):
        snapshot = _snapshot()
        checkpoint = Path("/tmp/stageedit-bridge-ready").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=snapshot,
            readiness=_ready(checkpoint),
        )
        with self.assertRaisesRegex(
            StageEditPipelineBridgeInputError,
            "automatic GROUP.*verified source derivative metadata",
        ):
            run_stage_edit_pipeline_bridge(
                preflight=preflight,
                sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
                source_durations_seconds=(10.0, 10.0),
                source_start_offsets_seconds=(0.0, 0.0),
                internal_cut_times_seconds=((), ()),
                source_timewarps=(None, None),
                source_quality_log_priors=(0.0, 0.0),
                require_verified_source_derivative_metadata=True,
                section_plan=_section_plan(),
                output=Path("/tmp/out.mp4"),
                reference_source=0,
                config=StageEditSemiMarkovConfig(minimum_unique_sources=1),
            )

    def test_preflight_requires_exact_snapshot_sources_and_ready_v2_checkpoint(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshot_path = snapshot.write_json(root / "snapshot.json")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "dynamic-crop-v2-metadata.json").write_text(
                "{}", encoding="utf-8"
            )
            with patch(
                "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                return_value=_ready(checkpoint),
            ):
                preflight = preflight_stage_edit_pipeline_bridge(
                    feature_snapshot_path=snapshot_path,
                    checkpoint_directory=checkpoint,
                    source_ids=("source-a", "source-b"),
                )
                self.assertEqual(snapshot.snapshot_sha256, preflight.snapshot.snapshot_sha256)

                with self.assertRaisesRegex(
                    StageEditPipelineBridgeInputError, "exactly match"
                ):
                    preflight_stage_edit_pipeline_bridge(
                        feature_snapshot_path=snapshot_path,
                        checkpoint_directory=checkpoint,
                        source_ids=("source-a", "source-c"),
                    )
            with patch(
                "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                return_value=_not_ready(checkpoint),
            ):
                with self.assertRaisesRegex(
                    StageEditPipelineBridgeNotReadyError, "not production-ready"
                ):
                    preflight_stage_edit_pipeline_bridge(
                        feature_snapshot_path=snapshot_path,
                        checkpoint_directory=checkpoint,
                        source_ids=("source-a", "source-b"),
                    )

    def test_preflight_routes_v3_and_rejects_mixed_checkpoint_namespaces(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshot_path = snapshot.write_json(root / "snapshot.json")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "dynamic-crop-v3-metadata.json").write_text(
                "{}", encoding="utf-8"
            )
            with patch(
                "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                return_value=_ready(checkpoint, dynamic_crop_version=3),
            ):
                preflight = preflight_stage_edit_pipeline_bridge(
                    feature_snapshot_path=snapshot_path,
                    checkpoint_directory=checkpoint,
                    source_ids=("source-a", "source-b"),
                    dynamic_crop_version=3,
                    continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
                )
            self.assertEqual(preflight.dynamic_crop_version, 3)
            self.assertEqual(
                preflight.continuity_authority,
                HANDCRAFTED_CONTINUITY_AUTHORITY,
            )
            self.assertEqual(
                preflight.to_dict()["checkpoint_format"], "dynamic_crop_v3"
            )

            (checkpoint / "dynamic-crop-v2-metadata.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                StageEditPipelineBridgeNotReadyError, "unambiguous"
            ):
                preflight_stage_edit_pipeline_bridge(
                    feature_snapshot_path=snapshot_path,
                    checkpoint_directory=checkpoint,
                    source_ids=("source-a", "source-b"),
                    dynamic_crop_version=3,
                )

    def test_learned_v3_preflight_binds_exact_typed_source_pair_scorer(self):
        snapshot = _snapshot()
        scorer = StageEditSourcePairScorer.uniform()
        checkpoint_digest = "a" * 64
        corpus_digest = "b" * 64
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshot_path = snapshot.write_json(root / "snapshot.json")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "dynamic-crop-v3-metadata.json").write_text(
                "{}", encoding="utf-8"
            )
            pair_directory = root / "pair"
            pair_directory.mkdir()
            pair_path = pair_directory / STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME
            pair_path.write_text('{"test":true}', encoding="utf-8")
            pair_file_digest = hashlib.sha256(pair_path.read_bytes()).hexdigest()
            loaded_checkpoint = SimpleNamespace(
                checkpoint_sha256=checkpoint_digest,
                corpus_sha256=corpus_digest,
            )
            with (
                patch(
                    "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                    return_value=_ready(checkpoint, dynamic_crop_version=3),
                ),
                patch(
                    "crosscut.stage_edit_pipeline_bridge.load_stage_edit_source_pair_checkpoint",
                    return_value=(scorer, loaded_checkpoint),
                ) as load_pair,
            ):
                preflight = preflight_stage_edit_pipeline_bridge(
                    feature_snapshot_path=snapshot_path,
                    checkpoint_directory=checkpoint,
                    source_ids=("source-a", "source-b"),
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                    ),
                    source_pair_scorer_path=pair_path,
                    source_pair_scorer_file_sha256=pair_file_digest,
                    source_pair_scorer_fingerprint=scorer.fingerprint,
                    source_pair_checkpoint_sha256=checkpoint_digest,
                    source_pair_corpus_sha256=corpus_digest,
                )

    def test_direct_learned_preflight_cannot_forge_missing_scorer_evidence(self):
        checkpoint = Path("/tmp/stageedit-direct-v3-checkpoint").resolve()
        scorer = StageEditSourcePairScorer.uniform()
        with (
            patch(
                "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                return_value=_ready(checkpoint, dynamic_crop_version=3),
            ),
            self.assertRaisesRegex(
                StageEditPipelineBridgeNotReadyError,
                "canonical regular checkpoint",
            ),
        ):
            StageEditPipelineBridgePreflight(
                snapshot_path=Path("/tmp/stageedit-direct-v3.json").resolve(),
                checkpoint_directory=checkpoint,
                source_ids=("source-a", "source-b"),
                snapshot=_snapshot(),
                readiness=_ready(checkpoint, dynamic_crop_version=3),
                dynamic_crop_version=3,
                continuity_authority=(
                    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                ),
                source_pair_scorer=scorer,
                source_pair_scorer_path=Path(
                    f"/tmp/{STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME}"
                ).resolve(),
                source_pair_scorer_file_sha256="a" * 64,
                source_pair_scorer_fingerprint=scorer.fingerprint,
                source_pair_checkpoint_sha256="b" * 64,
                source_pair_corpus_sha256="c" * 64,
            )

            with (
                patch(
                    "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                    return_value=_ready(checkpoint, dynamic_crop_version=3),
                ),
                patch(
                    "crosscut.stage_edit_pipeline_bridge.load_stage_edit_source_pair_checkpoint",
                    return_value=(scorer, loaded_checkpoint),
                ),
                self.assertRaisesRegex(
                    StageEditPipelineBridgeNotReadyError, "identity changed"
                ),
            ):
                preflight_stage_edit_pipeline_bridge(
                    feature_snapshot_path=snapshot_path,
                    checkpoint_directory=checkpoint,
                    source_ids=("source-a", "source-b"),
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                    ),
                    source_pair_scorer_path=pair_path,
                    source_pair_scorer_file_sha256=pair_file_digest,
                    source_pair_scorer_fingerprint="d" * 64,
                    source_pair_checkpoint_sha256=checkpoint_digest,
                    source_pair_corpus_sha256=corpus_digest,
                )
            self.assertIs(preflight.source_pair_scorer, scorer)
            load_pair.assert_called_once_with(
                pair_directory.resolve(),
                expected_corpus_sha256=corpus_digest,
                expected_checkpoint_sha256=checkpoint_digest,
            )
            self.assertEqual(
                preflight.to_dict()["continuity"]["scorer_fingerprint"],
                scorer.fingerprint,
            )

            with self.assertRaisesRegex(
                StageEditPipelineBridgeInputError, "exact registered scorer"
            ):
                preflight_stage_edit_pipeline_bridge(
                    feature_snapshot_path=snapshot_path,
                    checkpoint_directory=checkpoint,
                    source_ids=("source-a", "source-b"),
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                    ),
                    source_pair_scorer_path=pair_path,
                )

            with (
                patch(
                    "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                    return_value=_ready(checkpoint, dynamic_crop_version=3),
                ),
                self.assertRaisesRegex(
                    StageEditPipelineBridgeNotReadyError, "file hash changed"
                ),
            ):
                preflight_stage_edit_pipeline_bridge(
                    feature_snapshot_path=snapshot_path,
                    checkpoint_directory=checkpoint,
                    source_ids=("source-a", "source-b"),
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                    ),
                    source_pair_scorer_path=pair_path,
                    source_pair_scorer_file_sha256="c" * 64,
                    source_pair_scorer_fingerprint=scorer.fingerprint,
                    source_pair_checkpoint_sha256=checkpoint_digest,
                    source_pair_corpus_sha256=corpus_digest,
                )

    def test_generation_retains_parallel_candidates_and_never_selects_baseline(self):
        snapshot = _snapshot()
        checkpoint = Path("/tmp/stageedit-bridge-ready").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=snapshot,
            readiness=_ready(checkpoint),
        )
        sources = (
            Path("/tmp/stageedit-a.mp4"),
            Path("/tmp/stageedit-b.mp4"),
        )
        config = StageEditSemiMarkovConfig(
            minimum_unique_sources=1,
            highlight_minimum_early_wide_anchor_fraction=0.42,
        )
        timelines = _typed_timelines(
            sources,
            count=8,
            authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            scorer_fingerprint=HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
        )
        decoded = _typed_decode_result(
            timelines,
            config=config,
            scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            scorer_fingerprint=HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            request_fulfilled=True,
        )
        learned_sections = _typed_section_director()
        with (
            patch(
                "crosscut.stage_edit_pipeline_bridge.materialize_stage_edit_head_frames_v2",
                return_value=_typed_head_result(checkpoint),
            ) as materialize,
            patch(
                "crosscut.stage_edit_pipeline_bridge.decode_stage_edit_semimarkov_beam",
                return_value=decoded,
            ) as decode,
            patch(
                "crosscut.stage_edit_pipeline_bridge.build_learned_section_plan_from_checkpoint",
                return_value=learned_sections,
            ) as direct_sections,
        ):
            result = run_stage_edit_pipeline_bridge(
                preflight=preflight,
                sources=sources,
                source_durations_seconds=(10.0, 10.0),
                source_start_offsets_seconds=(0.0, 0.0),
                internal_cut_times_seconds=((), ()),
                source_timewarps=(None, None),
                source_quality_log_priors=(0.0, 0.0),
                section_plan=_section_plan(),
                output=Path("/tmp/stageedit-retained.mp4"),
                reference_source=0,
                fps=30.0,
                config=config,
            )

        materialize.assert_called_once()
        direct_sections.assert_called_once()
        decode.assert_called_once()
        self.assertEqual(
            direct_sections.call_args.kwargs["chorus_wide_anchor_fraction"],
            config.highlight_minimum_early_wide_anchor_fraction,
        )
        self.assertEqual(
            direct_sections.call_args.kwargs["checkpoint_directory"],
            preflight.checkpoint_directory,
        )
        self.assertEqual(STAGE_EDIT_PIPELINE_BRIDGE_SCHEMA_VERSION, 3)
        self.assertTrue(STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT.endswith("_v3"))
        self.assertEqual(result.schema_version, 3)
        self.assertEqual(result.contract, STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT)
        self.assertEqual(8, len(result.retained_candidates))
        self.assertFalse(result.selection_policy["selected_for_render"])
        self.assertFalse(result.selection_policy["visual_baseline_replaced"])
        self.assertEqual(
            "legacy_identity_default_not_automatic_group_safe",
            result.selection_policy["source_derivative_policy"],
        )
        history = result.retained_candidates[0].edit.candidate_history[-1]
        self.assertFalse(history["visual_baseline_replaced"])
        self.assertFalse(history["selected_for_render"])

    def test_v3_learned_generation_routes_v3_heads_and_exact_typed_scorer(self):
        snapshot = _snapshot()
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        checkpoint = root / "stageedit-bridge-v3-ready"
        (
            scorer,
            scorer_path,
            scorer_file_sha256,
            scorer_checkpoint_sha256,
            scorer_corpus_sha256,
        ) = _saved_uniform_source_pair_scorer(root / "source-pair")
        with patch(
            "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
            return_value=_ready(checkpoint, dynamic_crop_version=3),
        ):
            preflight = StageEditPipelineBridgePreflight(
                snapshot_path=(root / "stageedit-v3-feature.json"),
                checkpoint_directory=checkpoint,
                source_ids=("source-a", "source-b"),
                snapshot=snapshot,
                readiness=_ready(checkpoint, dynamic_crop_version=3),
                dynamic_crop_version=3,
                continuity_authority=PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                source_pair_scorer=scorer,
                source_pair_scorer_path=scorer_path,
                source_pair_scorer_file_sha256=scorer_file_sha256,
                source_pair_scorer_fingerprint=scorer.fingerprint,
                source_pair_checkpoint_sha256=scorer_checkpoint_sha256,
                source_pair_corpus_sha256=scorer_corpus_sha256,
            )
        sources = (Path("/tmp/stageedit-a.mp4"), Path("/tmp/stageedit-b.mp4"))
        config = StageEditSemiMarkovConfig(
            minimum_unique_sources=1,
            continuity_authority=PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            continuity_scorer_contract=STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            continuity_scorer_fingerprint=scorer.fingerprint,
        )
        timelines = _typed_timelines(
            sources,
            count=8,
            authority=PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            scorer_contract=STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            scorer_fingerprint=scorer.fingerprint,
        )
        decoded = _typed_decode_result(
            timelines,
            config=config,
            scorer_contract=STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            scorer_fingerprint=scorer.fingerprint,
            request_fulfilled=True,
        )
        learned_sections = _typed_section_director(dynamic_crop_version=3)
        with (
            patch(
                "crosscut.stage_edit_pipeline_bridge.materialize_stage_edit_head_frames_v3",
                return_value=_typed_head_result(
                    checkpoint,
                    dynamic_crop_version=3,
                ),
            ) as materialize_v3,
            patch(
                "crosscut.stage_edit_pipeline_bridge.materialize_stage_edit_head_frames_v2"
            ) as materialize_v2,
            patch(
                "crosscut.stage_edit_pipeline_bridge.decode_stage_edit_semimarkov_beam",
                return_value=decoded,
            ) as decode,
            patch(
                "crosscut.stage_edit_pipeline_bridge.build_learned_section_plan_from_checkpoint",
                return_value=learned_sections,
            ),
        ):
            result = run_stage_edit_pipeline_bridge(
                preflight=preflight,
                sources=sources,
                source_durations_seconds=(10.0, 10.0),
                source_start_offsets_seconds=(0.0, 0.0),
                internal_cut_times_seconds=((), ()),
                source_timewarps=(None, None),
                source_quality_log_priors=(0.0, 0.0),
                section_plan=_section_plan(),
                output=Path("/tmp/stageedit-v3-retained.mp4"),
                reference_source=0,
                fps=30.0,
                config=config,
            )
        materialize_v3.assert_called_once()
        materialize_v2.assert_not_called()
        self.assertIs(decode.call_args.kwargs["source_pair_scorer"], scorer)
        self.assertIs(decode.call_args.kwargs["config"], config)
        self.assertEqual(result.preflight.dynamic_crop_version, 3)
        self.assertEqual(
            result.selection_policy["continuity_authority"],
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
        )

        wrong_version_director = _typed_section_director(
            dynamic_crop_version=2
        )
        with (
            patch(
                "crosscut.stage_edit_pipeline_bridge."
                "materialize_stage_edit_head_frames_v3",
                return_value=_typed_head_result(
                    checkpoint,
                    dynamic_crop_version=3,
                ),
            ),
            patch(
                "crosscut.stage_edit_pipeline_bridge."
                "build_learned_section_plan_from_checkpoint",
                return_value=wrong_version_director,
            ),
            self.assertRaisesRegex(
                StageEditPipelineBridgeError,
                "Section Director Dynamic Crop version differs",
            ),
        ):
            run_stage_edit_pipeline_bridge(
                preflight=preflight,
                sources=sources,
                source_durations_seconds=(10.0, 10.0),
                source_start_offsets_seconds=(0.0, 0.0),
                internal_cut_times_seconds=((), ()),
                source_timewarps=(None, None),
                source_quality_log_priors=(0.0, 0.0),
                section_plan=_section_plan(),
                output=Path("/tmp/stageedit-v3-version-mismatch.mp4"),
                reference_source=0,
                fps=30.0,
                config=config,
            )

        with self.assertRaisesRegex(
            StageEditPipelineBridgeInputError, "authority differs"
        ):
            run_stage_edit_pipeline_bridge(
                preflight=preflight,
                sources=sources,
                source_durations_seconds=(10.0, 10.0),
                source_start_offsets_seconds=(0.0, 0.0),
                internal_cut_times_seconds=((), ()),
                source_timewarps=(None, None),
                source_quality_log_priors=(0.0, 0.0),
                section_plan=_section_plan(),
                output=Path("/tmp/stageedit-v3-retained.mp4"),
                reference_source=0,
                fps=30.0,
                config=StageEditSemiMarkovConfig(minimum_unique_sources=1),
            )

    def test_v3_handcrafted_generation_routes_v3_heads_without_scorer(self):
        snapshot = _snapshot()
        checkpoint = Path("/tmp/stageedit-bridge-v3-handcrafted").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-v3-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=snapshot,
            readiness=_ready(checkpoint, dynamic_crop_version=3),
            dynamic_crop_version=3,
            continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
        )
        config = StageEditSemiMarkovConfig(
            minimum_unique_sources=1,
            continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            continuity_scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            continuity_scorer_fingerprint=(
                HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
            ),
        )
        decoded = _typed_decode_result(
            (),
            config=config,
            scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            scorer_fingerprint=HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            request_fulfilled=False,
        )
        learned_sections = _typed_section_director(dynamic_crop_version=3)
        with (
            patch(
                "crosscut.stage_edit_pipeline_bridge.materialize_stage_edit_head_frames_v3",
                return_value=_typed_head_result(
                    checkpoint,
                    dynamic_crop_version=3,
                ),
            ) as materialize_v3,
            patch(
                "crosscut.stage_edit_pipeline_bridge.materialize_stage_edit_head_frames_v2"
            ) as materialize_v2,
            patch(
                "crosscut.stage_edit_pipeline_bridge.decode_stage_edit_semimarkov_beam",
                return_value=decoded,
            ) as decode,
            patch(
                "crosscut.stage_edit_pipeline_bridge.build_learned_section_plan_from_checkpoint",
                return_value=learned_sections,
            ),
        ):
            with self.assertRaisesRegex(StageEditPipelineBridgeError, "silent"):
                run_stage_edit_pipeline_bridge(
                    preflight=preflight,
                    sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
                    source_durations_seconds=(10.0, 10.0),
                    source_start_offsets_seconds=(0.0, 0.0),
                    internal_cut_times_seconds=((), ()),
                    source_timewarps=(None, None),
                    source_quality_log_priors=(0.0, 0.0),
                    section_plan=_section_plan(),
                    output=Path("/tmp/out.mp4"),
                    reference_source=0,
                    config=config,
                )
        materialize_v3.assert_called_once()
        materialize_v2.assert_not_called()
        self.assertIsNone(decode.call_args.kwargs["source_pair_scorer"])

    def test_requested_decode_shortfall_fails_without_visual_fallback(self):
        snapshot = _snapshot()
        checkpoint = Path("/tmp/stageedit-bridge-ready").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=snapshot,
            readiness=_ready(checkpoint),
        )
        config = StageEditSemiMarkovConfig(minimum_unique_sources=1)
        decoded = _typed_decode_result(
            (),
            config=config,
            scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            scorer_fingerprint=HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            request_fulfilled=False,
        )
        learned_sections = _typed_section_director()
        with (
            patch(
                "crosscut.stage_edit_pipeline_bridge.materialize_stage_edit_head_frames_v2",
                return_value=_typed_head_result(checkpoint),
            ),
            patch(
                "crosscut.stage_edit_pipeline_bridge.decode_stage_edit_semimarkov_beam",
                return_value=decoded,
            ),
            patch(
                "crosscut.stage_edit_pipeline_bridge.build_learned_section_plan_from_checkpoint",
                return_value=learned_sections,
            ),
        ):
            with self.assertRaisesRegex(
                StageEditPipelineBridgeError, "silent"
            ):
                run_stage_edit_pipeline_bridge(
                    preflight=preflight,
                    sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
                    source_durations_seconds=(10.0, 10.0),
                    source_start_offsets_seconds=(0.0, 0.0),
                    internal_cut_times_seconds=((), ()),
                    source_timewarps=(None, None),
                    source_quality_log_priors=(0.0, 0.0),
                    section_plan=_section_plan(),
                    output=Path("/tmp/out.mp4"),
                    reference_source=0,
                    config=config,
                )


class StageEditGenerationRequestTests(unittest.TestCase):
    def _request_payload(self, root: Path) -> dict[str, object]:
        sources = []
        for index in range(8):
            path = root / f"source-{index}.mp4"
            path.touch()
            sources.append(str(path))
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        snapshot = root / "snapshot.json"
        snapshot.write_text("{}", encoding="utf-8")
        return {
            "song": "BANG BANG",
            "edit_mode": "group",
            "aspect_ratio": "16:9",
            "source_mode": "automatic",
            "desired_source_count": 8,
            "sources": sources,
            "output_directory": str(root / "output"),
            "stage_edit_generation_checkpoint_directory": str(checkpoint),
            "stage_edit_generation_feature_snapshot": str(snapshot),
            "stage_edit_generation_source_ids": [
                f"video-{index}" for index in range(8)
            ],
        }

    def test_request_activation_is_explicit_three_part_and_round_trips(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            payload = self._request_payload(root)
            checkpoint = Path(
                str(payload["stage_edit_generation_checkpoint_directory"])
            )
            request = AnalysisRequest.from_dict(payload)
            restored = AnalysisRequest.from_dict(request.as_dict())
            self.assertEqual(
                request.stage_edit_generation_source_ids,
                restored.stage_edit_generation_source_ids,
            )
            self.assertEqual(
                checkpoint.resolve(),
                restored.stage_edit_generation_checkpoint_directory,
            )

            incomplete = dict(payload)
            incomplete.pop("stage_edit_generation_feature_snapshot")
            with self.assertRaisesRegex(ValueError, "supplied together"):
                AnalysisRequest.from_dict(incomplete)

    def test_pipeline_preflights_exact_activation_and_requires_proxy_review(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = AnalysisRequest.from_dict(self._request_payload(root))
            marker = object()
            with (
                patch(
                    "crosscut.pipeline.stage_edit_readiness_report",
                    return_value=_ready(
                        request.stage_edit_generation_checkpoint_directory,
                    ),
                ),
                patch(
                    "crosscut.pipeline.preflight_stage_edit_pipeline_bridge",
                    return_value=marker,
                ) as preflight,
            ):
                self.assertIs(
                    marker,
                    _load_stage_edit_generation_preflight(request),
                )
            preflight.assert_called_once_with(
                feature_snapshot_path=request.stage_edit_generation_feature_snapshot,
                checkpoint_directory=(
                    request.stage_edit_generation_checkpoint_directory
                ),
                source_ids=request.stage_edit_generation_source_ids,
                dynamic_crop_version=2,
                continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            )
            with self.assertRaisesRegex(
                RuntimeError, "requires decoded proxy review"
            ):
                analyze(
                    request,
                    SimpleNamespace(),
                    review_rendered_proxies=False,
                )

    def test_explicit_activation_routes_v3_handcrafted_and_rejects_unknown_version(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = AnalysisRequest.from_dict(self._request_payload(root))
            checkpoint = request.stage_edit_generation_checkpoint_directory
            marker = object()
            with (
                patch(
                    "crosscut.pipeline.stage_edit_readiness_report",
                    return_value=_ready(checkpoint, dynamic_crop_version=3),
                ),
                patch(
                    "crosscut.pipeline.preflight_stage_edit_pipeline_bridge",
                    return_value=marker,
                ) as preflight,
            ):
                self.assertIs(
                    marker,
                    _load_stage_edit_generation_preflight(request),
                )
            preflight.assert_called_once_with(
                feature_snapshot_path=request.stage_edit_generation_feature_snapshot,
                checkpoint_directory=checkpoint,
                source_ids=request.stage_edit_generation_source_ids,
                dynamic_crop_version=3,
                continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            )

            with (
                patch(
                    "crosscut.pipeline.stage_edit_readiness_report",
                    return_value=SimpleNamespace(dynamic_crop_version=4),
                ),
                patch(
                    "crosscut.pipeline.preflight_stage_edit_pipeline_bridge"
                ) as invalid_preflight,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "unsupported Dynamic Crop version",
                ):
                    _load_stage_edit_generation_preflight(request)
            invalid_preflight.assert_not_called()

    def test_default_request_does_not_persist_or_activate_generation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources = []
            for index in range(8):
                path = root / f"source-{index}.mp4"
                path.touch()
                sources.append(str(path))
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 8,
                    "sources": sources,
                    "output_directory": str(root / "output"),
                }
            )
            self.assertIsNone(request.stage_edit_generation_checkpoint_directory)
            self.assertNotIn(
                "stage_edit_generation_checkpoint_directory", request.as_dict()
            )

    def test_simple_automatic_group_discovers_registered_bundle_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "sources": [],
                    "output_directory": str(root / "output"),
                }
            )
            marker = object()
            with patch(
                "crosscut.pipeline.discover_registered_stage_edit_production_bundle",
                return_value=marker,
            ) as discover:
                self.assertIs(
                    marker,
                    _discover_automatic_stage_edit_production_bundle(request),
                )
            discover.assert_called_once_with()

            local_sources = (root / "one.mp4", root / "two.mp4")
            for source in local_sources:
                source.touch()
            local = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "local",
                    "sources": [str(source) for source in local_sources],
                    "output_directory": str(root / "local-output"),
                }
            )
            with patch(
                "crosscut.pipeline.discover_registered_stage_edit_production_bundle"
            ) as discover:
                self.assertIsNone(
                    _discover_automatic_stage_edit_production_bundle(local)
                )
            discover.assert_not_called()

    def test_automatic_group_refuses_unregistered_visual_baseline(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 8,
                    "sources": [],
                    "output_directory": str(root / "output"),
                }
            )
            with patch(
                "crosscut.pipeline.discover_registered_stage_edit_production_bundle",
                return_value=None,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "requires a verified registered StageEdit production bundle",
                ):
                    analyze(
                        request,
                        SimpleNamespace(),
                        review_rendered_proxies=True,
                    )
            self.assertFalse((root / "output").exists())

    def test_registered_bundle_materializes_current_media_before_preflight(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources = []
            for index in range(8):
                source = root / f"source-{index}.mp4"
                source.touch()
                sources.append(str(source))
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 8,
                    "sources": sources,
                    "output_directory": str(root / "output"),
                    "source_cache_directory": str(root / "cache"),
                }
            )
            checkpoint = (root / "checkpoint").resolve()
            bundle = SimpleNamespace(
                checkpoint_directory=checkpoint,
                maximum_sequence_length=2048,
                maximum_sources=32,
            )
            snapshot = SimpleNamespace(
                snapshot_path=(root / "snapshot.json").resolve(),
                source_ids=tuple(f"source-{index}" for index in range(8)),
            )
            preflight = SimpleNamespace(
                readiness=SimpleNamespace(checkpoint_path=str(checkpoint))
            )
            alignment = SimpleNamespace(alignments=())
            eligibility = tuple(
                {"render_eligible": True} for _ in range(8)
            )
            timing = object()
            rhythm = object()
            sections = object()
            with (
                patch(
                    "crosscut.pipeline.materialize_stage_edit_inference_snapshot",
                    return_value=snapshot,
                ) as materialize,
                patch(
                    "crosscut.pipeline.preflight_stage_edit_pipeline_bridge",
                    return_value=preflight,
                ) as preflight_call,
            ):
                observed_snapshot, observed_preflight = (
                    _materialize_registered_stage_edit_preflight(
                        bundle=bundle,
                        request=request,
                        alignment=alignment,
                        alignment_eligibility=eligibility,
                        source_timewarps=(None,) * 8,
                        timing_draft=timing,
                        rhythm=rhythm,
                        section_plan=sections,
                    )
                )

            self.assertIs(observed_snapshot, snapshot)
            self.assertIs(observed_preflight, preflight)
            materialize.assert_called_once_with(
                song_id=request.song,
                sources=request.sources,
                alignment=alignment,
                alignment_hard_gates=(True,) * 8,
                source_timewarps=(None,) * 8,
                timing_draft=timing,
                rhythm=rhythm,
                section_plan=sections,
                output_directory=(
                    request.source_cache_directory
                    / "derived"
                    / "stage-edit-inference-v1"
                ),
                feature_cache_directory=request.source_cache_directory,
                maximum_sequence_length=2048,
                maximum_sources=32,
            )
            preflight_call.assert_called_once_with(
                feature_snapshot_path=snapshot.snapshot_path,
                checkpoint_directory=checkpoint,
                source_ids=snapshot.source_ids,
                dynamic_crop_version=2,
                continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            )

    def test_registered_v3_learned_bundle_forwards_exact_scorer_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = AnalysisRequest.from_dict(self._request_payload(root))
            checkpoint = request.stage_edit_generation_checkpoint_directory
            assert checkpoint is not None
            pair_path = (root / STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME).resolve()
            bundle = SimpleNamespace(
                checkpoint_directory=checkpoint,
                maximum_sequence_length=2048,
                maximum_sources=32,
                dynamic_crop_version=3,
                continuity_authority=(
                    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                ),
                source_pair_scorer_path=pair_path,
                source_pair_scorer_file_sha256="a" * 64,
                source_pair_scorer_fingerprint="b" * 64,
                source_pair_checkpoint_sha256="c" * 64,
                source_pair_corpus_sha256="d" * 64,
            )
            self.assertEqual(
                _stage_edit_continuity_config_identity(
                    registered_bundle=bundle,
                    explicit_preflight=None,
                ),
                (
                    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
                    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
                    "b" * 64,
                ),
            )
            snapshot = SimpleNamespace(
                snapshot_path=(root / "current-snapshot.json").resolve(),
                source_ids=tuple(f"source-{index}" for index in range(8)),
            )
            preflight = SimpleNamespace(
                readiness=SimpleNamespace(checkpoint_path=str(checkpoint))
            )
            with (
                patch(
                    "crosscut.pipeline.materialize_stage_edit_inference_snapshot",
                    return_value=snapshot,
                ),
                patch(
                    "crosscut.pipeline.preflight_stage_edit_pipeline_bridge",
                    return_value=preflight,
                ) as preflight_call,
            ):
                _materialize_registered_stage_edit_preflight(
                    bundle=bundle,
                    request=request,
                    alignment=SimpleNamespace(alignments=()),
                    alignment_eligibility=tuple(
                        {"render_eligible": True} for _ in range(8)
                    ),
                    source_timewarps=(None,) * 8,
                    timing_draft=object(),
                    rhythm=object(),
                    section_plan=object(),
                )
            preflight_call.assert_called_once_with(
                feature_snapshot_path=snapshot.snapshot_path,
                checkpoint_directory=checkpoint,
                source_ids=snapshot.source_ids,
                dynamic_crop_version=3,
                continuity_authority=(
                    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                ),
                source_pair_scorer_path=pair_path,
                source_pair_scorer_file_sha256="a" * 64,
                source_pair_scorer_fingerprint="b" * 64,
                source_pair_checkpoint_sha256="c" * 64,
                source_pair_corpus_sha256="d" * 64,
            )


if __name__ == "__main__":
    unittest.main()
