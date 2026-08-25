from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import create_autospec, patch

from crosscut.ive import AspectRatio, EditMode, SourceMode
from crosscut.pipeline import (
    _discover_automatic_stage_edit_production_bundle,
    _load_stage_edit_generation_preflight,
    _materialize_registered_stage_edit_preflight,
    _stage_edit_continuity_config_identity,
    _stage_edit_semimarkov_config_from_montage_policy,
    analyze,
)
from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderSegment,
)
from crosscut.request import AnalysisRequest
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.source_derivative_diversity import SourceDerivativeFamilyMetadata
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
    StageEditContextualSourcePairBridgeIdentityV2,
    _exact_clock_candidate,
    load_stage_edit_production_snapshot_identity,
    preflight_stage_edit_pipeline_bridge,
    run_stage_edit_pipeline_bridge,
)
from crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropTrackDecodeV3,
)
from crosscut.stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_contextual_beam_qa_adapter_v2 import (
    VerifiedStageEditContextualBeamQACaseV2,
    derive_stage_edit_contextual_beam_qa_case_v2,
    is_verified_stage_edit_contextual_beam_qa_case_v2,
)
from crosscut.stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    StageEditBeamTimeline,
    StageEditDecodeAudit,
    StageEditHeadFrame,
    StageEditSemiMarkovConfig,
    StageEditSemiMarkovResult,
    StageEditSourceContinuity,
)
from crosscut.stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_CHECKPOINT_FILENAME,
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
    StageEditSourcePairScorer,
    StageEditSourcePairTrainingResult,
    new_stage_edit_source_pair_checkpoint,
    save_stage_edit_source_pair_checkpoint,
)
from crosscut.stage_edit_source_pair_contextual_v2 import (
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT,
    StageEditContextualSourcePairScorerV2,
)
from crosscut.stage_edit_source_pair_contextual_v2_production import (
    StageEditContextualBeamQACaseV2,
    StageEditContextualSourcePairProductionBundleV2,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditReadinessReport,
    StageEditSequence,
    StageEditTemporalPoseSidecar,
    StageEditToken,
    STAGE_EDIT_CAUSAL_SENSING_SOURCE_FEATURE_NAMES,
    STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT,
)
from engine.tests.stage_edit_causal_sensing_test_support import (
    synthetic_production_snapshot,
)


def _assert_no_raw_pose_vectors(
    test: unittest.TestCase,
    value: object,
) -> None:
    forbidden_keys = {
        "pose_model_embedding",
        "normalized_velocity",
        "pose_embedding",
        "embeddings",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key)
            test.assertFalse(
                normalized_key in forbidden_keys
                or normalized_key.endswith("_pose_embedding")
                or normalized_key.endswith("_normalized_velocity")
                or normalized_key.endswith("_embeddings"),
                key,
            )
            _assert_no_raw_pose_vectors(test, nested)
    elif isinstance(value, list):
        if value and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            test.assertNotIn(len(value), {60, 120})
        for nested in value:
            _assert_no_raw_pose_vectors(test, nested)


def _candidate(source_id: str, value: float) -> SourceCandidateToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    causal_names = set(STAGE_EDIT_CAUSAL_SENSING_SOURCE_FEATURE_NAMES)
    missing = tuple(name in causal_names for name in schema.source_feature_names)
    return SourceCandidateToken(
        source_id=source_id,
        features=tuple(
            0.0 if is_missing else value for is_missing in missing
        ),
        missing_mask=missing,
        available=True,
        hard_gate_passed=True,
    )


def _snapshot_sequence() -> StageEditSequence:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    return StageEditSequence(
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


def _snapshot() -> StageEditFeatureSnapshot:
    return synthetic_production_snapshot(_snapshot_sequence())


def _inspection_snapshot() -> StageEditFeatureSnapshot:
    return StageEditFeatureSnapshot.build(
        _snapshot_sequence(),
        inspection_only_without_causal_sensing=True,
    )


def _snapshot_identity(
    snapshot: StageEditFeatureSnapshot,
    path: Path,
):
    canonical = Path(path).resolve()
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    return load_stage_edit_production_snapshot_identity(
        canonical,
        edit_mode="group",
        target_member=None,
    )


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


def _real_verified_contextual_beam_qa_envelope(
    root: Path,
) -> VerifiedStageEditContextualBeamQACaseV2:
    # Reuse the adapter's filesystem-complete fixture so this bridge test gets
    # a genuinely adapter-sealed envelope, not a hand-built DTO or mutated
    # dataclass.  The production bundle loader remains the mocked unit boundary.
    from engine.tests.test_stage_edit_contextual_beam_qa_adapter_v2 import (
        _Fixture,
    )

    fixture = _Fixture(root)
    with patch(
        "crosscut.stage_edit_contextual_beam_qa_adapter_v2.evaluate_post_render_qa",
        return_value=fixture.report,
    ):
        verified = derive_stage_edit_contextual_beam_qa_case_v2(
            semimarkov=fixture.semimarkov,
            selected_timeline_rank=1,
            post_render_report=fixture.report,
            decode_report_path=fixture.decode_path,
            post_render_report_path=fixture.report_path,
        )
    if not is_verified_stage_edit_contextual_beam_qa_case_v2(verified):
        raise AssertionError("adapter did not issue a verified Beam-QA envelope")
    return verified


def _typed_contextual_bundle_v2(
    directory: Path,
    verified: VerifiedStageEditContextualBeamQACaseV2,
) -> tuple[
    StageEditContextualSourcePairProductionBundleV2,
    StageEditContextualSourcePairScorerV2,
    dict[str, str],
]:
    scorer = StageEditContextualSourcePairScorerV2(
        token_schema_fingerprint=DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
        weights=(0.0,) * len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES),
    )
    pins = {
        "checkpoint_sha256": "1" * 64,
        "checkpoint_file_sha256": "2" * 64,
        "report_sha256": "3" * 64,
        "report_file_sha256": "4" * 64,
        "beam_qa_report_sha256": "5" * 64,
        "corpus_sha256": "6" * 64,
        "corpus_file_sha256": "7" * 64,
        "token_schema_fingerprint": DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
    }
    bundle = create_autospec(
        StageEditContextualSourcePairProductionBundleV2,
        instance=True,
    )
    bundle.directory = directory
    bundle.scorer = scorer
    bundle.checkpoint = SimpleNamespace(
        checkpoint_sha256=pins["checkpoint_sha256"],
        token_schema_fingerprint=pins["token_schema_fingerprint"],
    )
    bundle.checkpoint_file_sha256 = pins["checkpoint_file_sha256"]
    bundle.report = SimpleNamespace(
        report_sha256=pins["report_sha256"],
        beam_level_qa=SimpleNamespace(
            qa_report_sha256=pins["beam_qa_report_sha256"],
            cases=(verified.case,),
        ),
    )
    bundle.report_file_sha256 = pins["report_file_sha256"]
    bundle.corpus = SimpleNamespace(content_sha256=pins["corpus_sha256"])
    bundle.corpus_file_sha256 = pins["corpus_file_sha256"]
    return bundle, scorer, pins


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
        to_dict=lambda: {
            "source_index": 0,
            "master_start_seconds": 0.0,
            "duration_seconds": 2.0,
        },
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
    expected_contract = preflight.continuity_scorer_contract
    expected_fingerprint = preflight.continuity_scorer_fingerprint
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
            "contextual_source_pair_identity_v2": (
                preflight.contextual_source_pair_identity_v2.to_dict()
                if preflight.contextual_source_pair_identity_v2 is not None
                else None
            ),
            "temporal_pose_sidecar": (
                preflight.snapshot.temporal_pose_sidecar_audit()
            ),
        },
    )


class StageEditPipelineBridgeTests(unittest.TestCase):
    def test_exact_clock_projection_preserves_typed_v3_decode_or_fails(self) -> None:
        sources = (
            Path("/tmp/stageedit-v3-provenance-a.mp4"),
            Path("/tmp/stageedit-v3-provenance-b.mp4"),
        )
        timeline = _typed_timelines(
            sources,
            count=1,
            authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
            scorer_contract=HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            scorer_fingerprint=HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
        )[0]
        track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.02),
                EditorialTransformKeyframe(1.5, 1.05),
            ),
            reason="stage_edit_dynamic_crop_head_v3",
        )
        decoded = DynamicCropTrackDecodeV3(
            source_id="source-0",
            source_index=0,
            eligible=True,
            loss_masked=False,
            track=track,
            predicted_keyframe_count=2,
            applied_keyframe_count=2,
            minimum_safe_zoom_projection_count=0,
            rejection_reasons=(),
            motion_logit=0.5,
            shot_duration_seconds=2.0,
        )
        decision = SimpleNamespace(
            **{
                **vars(timeline.decisions[0]),
                "editorial_transform": track,
                "dynamic_crop_decode_v3": decoded,
            }
        )
        shot = replace(
            timeline.edit.shots[0],
            segment=replace(
                timeline.edit.shots[0].segment,
                editorial_transform=track,
            ),
            dynamic_crop_decode_v3=decoded,
        )
        exact_timeline = replace(
            timeline,
            decisions=(decision,),
            edit=replace(timeline.edit, shots=(shot,)),
        )
        common = {
            "sources": sources,
            "source_offsets": (0.0, 0.0),
            "source_durations": (10.0, 10.0),
            "internal_cuts": ((), ()),
            "timewarps": (None, None),
            "section_plan": _section_plan(),
            "config": StageEditSemiMarkovConfig(minimum_unique_sources=1),
            "dynamic_crop_version": 3,
            "continuity_authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
        }

        retained = _exact_clock_candidate(timeline=exact_timeline, **common)
        self.assertIs(
            decoded,
            retained.edit.shots[0].dynamic_crop_decode_v3,
        )
        self.assertEqual(
            decoded.to_dict(),
            retained.edit.shots[0].to_dict()["dynamic_crop_v3"],
        )

        lost = replace(
            exact_timeline,
            edit=replace(
                exact_timeline.edit,
                shots=(replace(shot, dynamic_crop_decode_v3=None),),
            ),
        )
        with self.assertRaisesRegex(
            StageEditPipelineBridgeInputError,
            "lost Dynamic Crop v3 decode provenance",
        ):
            _exact_clock_candidate(timeline=lost, **common)

    def test_direct_preflight_replace_scope_and_disk_swap_fail_closed(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        snapshot_path = root / "feature-snapshot.json"
        checkpoint = root / "checkpoint"
        snapshot = _snapshot()
        identity = _snapshot_identity(snapshot, snapshot_path)
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=snapshot_path,
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=snapshot,
            readiness=_ready(checkpoint),
            snapshot_identity=identity,
            edit_mode="group",
            target_member=None,
        )

        with self.assertRaisesRegex(
            StageEditPipelineBridgeNotReadyError,
            "identity or causal scope",
        ):
            StageEditPipelineBridgePreflight(
                snapshot_path=snapshot_path,
                checkpoint_directory=checkpoint,
                source_ids=("source-a", "source-b"),
                snapshot=_inspection_snapshot(),
                readiness=_ready(checkpoint),
                snapshot_identity=identity,
                edit_mode="group",
                target_member=None,
            )
        with self.assertRaisesRegex(
            StageEditPipelineBridgeNotReadyError,
            "identity or causal scope",
        ):
            replace(
                preflight,
                edit_mode="member",
                target_member="가을",
            )
        with self.assertRaisesRegex(
            StageEditPipelineBridgeNotReadyError,
            "identity or causal scope",
        ):
            replace(preflight, snapshot=_inspection_snapshot())

        snapshot_path.write_text(
            json.dumps(_inspection_snapshot().to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
        with (
            patch(
                "crosscut.stage_edit_pipeline_bridge."
                "materialize_stage_edit_head_frames_v2"
            ) as materialize_heads,
            self.assertRaisesRegex(
                StageEditPipelineBridgeNotReadyError,
                "disk identity",
            ),
        ):
            run_stage_edit_pipeline_bridge(
                preflight=preflight,
                sources=(root / "a.mp4", root / "b.mp4"),
                source_durations_seconds=(10.0, 10.0),
                source_start_offsets_seconds=(0.0, 0.0),
                internal_cut_times_seconds=((), ()),
                source_timewarps=(None, None),
                source_quality_log_priors=(0.0, 0.0),
                section_plan=_section_plan(),
                output=root / "out.mp4",
                reference_source=0,
                config=StageEditSemiMarkovConfig(minimum_unique_sources=1),
            )
        materialize_heads.assert_not_called()

    def test_bridge_result_recursively_redacts_all_raw_pose_vectors(self) -> None:
        checkpoint = Path("/tmp/stageedit-vector-redaction").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=_snapshot(),
            readiness=_ready(checkpoint),
            snapshot_identity=_snapshot_identity(
                _snapshot(), Path("/tmp/stageedit-feature.json")
            ),
            edit_mode="group",
            target_member=None,
        )
        result = _typed_direct_bridge_result(preflight)
        sidecar = StageEditTemporalPoseSidecar(
            embeddings=((0.1,) * 60, (0.2,) * 60),
            interval_seconds=0.04,
            sampling_contract=(
                STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT
            ),
        )
        continuity = StageEditSourceContinuity(
            pose_motion=0.5,
            pose_model_embedding=(0.25,) * 60,
            temporal_pose_sidecar=sidecar,
        )
        frame = replace(
            result.head_materialization.frames[0],
            source_continuity=(continuity, StageEditSourceContinuity()),
        )
        result = replace(
            result,
            head_materialization=replace(
                result.head_materialization,
                frames=(frame,),
            ),
        )

        self.assertEqual(60, len(continuity.pose_model_embedding or ()))
        self.assertEqual((2, 60), (len(sidecar.embeddings), len(sidecar.embeddings[0])))
        payload = json.loads(json.dumps(result.to_dict()))
        _assert_no_raw_pose_vectors(self, payload)
        _assert_no_raw_pose_vectors(self, result.semimarkov.audit.to_dict())
        with tempfile.TemporaryDirectory() as raw:
            destination = result.write_json(Path(raw) / "bridge-result.json")
            persisted = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(payload, persisted)
        _assert_no_raw_pose_vectors(self, persisted)

        evidence = payload["head_materialization"]["frames"][0][
            "source_continuity"
        ][0]
        self.assertEqual(
            64,
            len(evidence["same_time_pose_evidence"]["content_sha256"]),
        )
        self.assertEqual(
            64,
            len(evidence["temporal_pose_evidence"]["content_sha256"]),
        )
        poisoned = replace(
            result,
            selection_policy={
                **result.selection_policy,
                "embeddings": [[0.0] * 60, [0.0] * 60],
            },
        )
        with self.assertRaisesRegex(ValueError, "forbidden raw pose vector"):
            poisoned.to_dict()

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
            snapshot_identity=_snapshot_identity(
                _snapshot(), Path("/tmp/stageedit-feature.json")
            ),
            edit_mode="group",
            target_member=None,
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
            snapshot_identity=_snapshot_identity(
                _snapshot(), Path("/tmp/stageedit-feature.json")
            ),
            edit_mode="group",
            target_member=None,
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
            snapshot_identity=_snapshot_identity(
                _snapshot(), Path("/tmp/stageedit-feature.json")
            ),
            edit_mode="group",
            target_member=None,
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
                snapshot_identity=_snapshot_identity(
                    _snapshot(), root / "stageedit-feature.json"
                ),
                edit_mode="group",
                target_member=None,
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
            snapshot_identity=_snapshot_identity(
                snapshot, Path("/tmp/stageedit-feature.json")
            ),
            edit_mode="group",
            target_member=None,
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
                self.assertEqual(
                    preflight.to_dict()["temporal_pose_sidecar"],
                    snapshot.temporal_pose_sidecar_audit(),
                )

                with self.assertRaisesRegex(
                    StageEditPipelineBridgeInputError, "exactly match"
                ):
                    preflight_stage_edit_pipeline_bridge(
                        feature_snapshot_path=snapshot_path,
                        checkpoint_directory=checkpoint,
                        source_ids=("source-a", "source-c"),
                    )
                with self.assertRaisesRegex(
                    StageEditPipelineBridgeNotReadyError,
                    "does not match MEMBER",
                ):
                    preflight_stage_edit_pipeline_bridge(
                        feature_snapshot_path=snapshot_path,
                        checkpoint_directory=checkpoint,
                        source_ids=("source-a", "source-b"),
                        edit_mode=EditMode.MEMBER,
                        target_member="가을",
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
                snapshot_identity=_snapshot_identity(
                    _snapshot(), Path("/tmp/stageedit-direct-v3.json")
                ),
                edit_mode="group",
                target_member=None,
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

    def test_contextual_v2_preflight_and_generation_use_verified_bundle_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw:
            root = Path(raw)
            envelope_root = root / "verified-beam-qa"
            envelope_root.mkdir()
            verified = _real_verified_contextual_beam_qa_envelope(envelope_root)
            bundle_directory = root / "contextual-bundle"
            bundle_directory.mkdir()
            bundle, scorer, pins = _typed_contextual_bundle_v2(
                bundle_directory,
                verified,
            )
            snapshot = _snapshot()
            snapshot_path = snapshot.write_json(root / "snapshot.json")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "dynamic-crop-v3-metadata.json").write_text(
                "{}",
                encoding="utf-8",
            )
            config = StageEditSemiMarkovConfig(
                minimum_unique_sources=1,
                continuity_authority=(
                    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
                ),
                continuity_scorer_contract=(
                    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT
                ),
                continuity_scorer_fingerprint=scorer.fingerprint,
            )
            sources = (
                Path("/tmp/contextual-stageedit-a.mp4"),
                Path("/tmp/contextual-stageedit-b.mp4"),
            )
            timelines = _typed_timelines(
                sources,
                count=config.timeline_count,
                authority=PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
                scorer_contract=(
                    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT
                ),
                scorer_fingerprint=scorer.fingerprint,
            )
            decoded = _typed_decode_result(
                timelines,
                config=config,
                scorer_contract=(
                    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT
                ),
                scorer_fingerprint=scorer.fingerprint,
                request_fulfilled=True,
            )
            derivative_metadata = SourceDerivativeFamilyMetadata(
                source_count=2,
                family_ids=(0, 1),
                audit_sha256="8" * 64,
                production_ready=True,
                uncertain_pair_count=0,
                clock_contract_complete=True,
            )
            with (
                patch(
                    "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                    return_value=_ready(checkpoint, dynamic_crop_version=3),
                ),
                patch(
                    "crosscut.stage_edit_pipeline_bridge."
                    "load_stage_edit_contextual_source_pair_production_bundle_v2",
                    return_value=bundle,
                ) as load_bundle,
                patch(
                    "crosscut.stage_edit_pipeline_bridge."
                    "materialize_stage_edit_head_frames_v3",
                    return_value=_typed_head_result(
                        checkpoint,
                        dynamic_crop_version=3,
                    ),
                ) as materialize_v3,
                patch(
                    "crosscut.stage_edit_pipeline_bridge."
                    "materialize_stage_edit_head_frames_v2"
                ) as materialize_v2,
                patch(
                    "crosscut.stage_edit_pipeline_bridge."
                    "decode_stage_edit_semimarkov_beam",
                    return_value=decoded,
                ) as decode,
                patch(
                    "crosscut.stage_edit_pipeline_bridge."
                    "build_learned_section_plan_from_checkpoint",
                    return_value=_typed_section_director(
                        dynamic_crop_version=3
                    ),
                ),
            ):
                preflight = preflight_stage_edit_pipeline_bridge(
                    feature_snapshot_path=snapshot_path,
                    checkpoint_directory=checkpoint,
                    source_ids=("source-a", "source-b"),
                    dynamic_crop_version=3,
                    continuity_authority=(
                        PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
                    ),
                    contextual_source_pair_bundle_directory=bundle_directory,
                    contextual_source_pair_checkpoint_sha256=(
                        pins["checkpoint_sha256"]
                    ),
                    contextual_source_pair_checkpoint_file_sha256=(
                        pins["checkpoint_file_sha256"]
                    ),
                    contextual_source_pair_report_sha256=pins["report_sha256"],
                    contextual_source_pair_report_file_sha256=(
                        pins["report_file_sha256"]
                    ),
                    contextual_source_pair_beam_qa_report_sha256=(
                        pins["beam_qa_report_sha256"]
                    ),
                    contextual_source_pair_corpus_sha256=pins["corpus_sha256"],
                    contextual_source_pair_corpus_file_sha256=(
                        pins["corpus_file_sha256"]
                    ),
                    contextual_source_pair_token_schema_fingerprint=(
                        pins["token_schema_fingerprint"]
                    ),
                    contextual_source_pair_verified_beam_qa_cases=(verified,),
                )
                result = run_stage_edit_pipeline_bridge(
                    preflight=preflight,
                    sources=sources,
                    source_durations_seconds=(10.0, 10.0),
                    source_start_offsets_seconds=(0.0, 0.0),
                    internal_cut_times_seconds=((), ()),
                    source_timewarps=(None, None),
                    source_quality_log_priors=(0.0, 0.0),
                    source_derivative_metadata=derivative_metadata,
                    section_plan=_section_plan(),
                    output=Path("/tmp/contextual-stageedit-result.mp4"),
                    reference_source=0,
                    fps=30.0,
                    config=config,
                )

            self.assertIsInstance(
                preflight.contextual_source_pair_identity_v2,
                StageEditContextualSourcePairBridgeIdentityV2,
            )
            self.assertIs(preflight.contextual_source_pair_scorer_v2, scorer)
            self.assertEqual(
                preflight.contextual_source_pair_scorer_fingerprint_v2,
                scorer.fingerprint,
            )
            self.assertEqual(load_bundle.call_count, 2)
            expected_loader_kwargs = {
                "expected_checkpoint_sha256": pins["checkpoint_sha256"],
                "expected_checkpoint_file_sha256": pins[
                    "checkpoint_file_sha256"
                ],
                "expected_report_sha256": pins["report_sha256"],
                "expected_report_file_sha256": pins["report_file_sha256"],
                "expected_beam_qa_report_sha256": pins[
                    "beam_qa_report_sha256"
                ],
                "expected_corpus_sha256": pins["corpus_sha256"],
                "expected_corpus_file_sha256": pins["corpus_file_sha256"],
                "expected_token_schema_fingerprint": pins[
                    "token_schema_fingerprint"
                ],
                "verified_beam_qa_cases": (verified,),
            }
            for call in load_bundle.call_args_list:
                self.assertEqual(call.args, (bundle_directory,))
                self.assertEqual(call.kwargs, expected_loader_kwargs)
            materialize_v3.assert_called_once()
            materialize_v2.assert_not_called()
            self.assertEqual(
                materialize_v3.call_args.kwargs[
                    "derivative_family_id_by_source"
                ],
                {"source-a": 0, "source-b": 1},
            )
            self.assertIsNone(decode.call_args.kwargs["source_pair_scorer"])
            self.assertIs(
                decode.call_args.kwargs["contextual_source_pair_scorer_v2"],
                scorer,
            )
            self.assertIs(
                decode.call_args.kwargs["source_derivative_metadata"],
                derivative_metadata,
            )
            contextual_identity = preflight.contextual_source_pair_identity_v2
            assert contextual_identity is not None
            expected_identity_payload = contextual_identity.to_dict()
            self.assertEqual(
                result.selection_policy[
                    "contextual_source_pair_identity_v2"
                ],
                expected_identity_payload,
            )
            self.assertEqual(
                result.semimarkov.audit.config["continuity_authority"],
                PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
            )
            self.assertEqual(
                result.semimarkov.audit.score_provenance[
                    "boundary_continuity"
                ]["scorer_fingerprint"],
                scorer.fingerprint,
            )
            self.assertEqual(
                result.selection_policy["source_derivative_policy"],
                "verified_actual_pts_families_required",
            )
            serialized = result.to_dict()
            self.assertEqual(
                serialized["activation"]["continuity"]
                ["contextual_source_pair_identity_v2"],
                expected_identity_payload,
            )
            with self.assertRaisesRegex(
                ValueError,
                "contextual-v2 selection identity differs",
            ):
                replace(
                    result,
                    selection_policy={
                        **result.selection_policy,
                        "contextual_source_pair_identity_v2": {},
                    },
                )
            tampered_audit = replace(
                result.semimarkov.audit,
                config={
                    **result.semimarkov.audit.config,
                    "continuity_scorer_fingerprint": "f" * 64,
                },
            )
            with self.assertRaisesRegex(
                ValueError,
                "audit config continuity identity differs",
            ):
                replace(
                    result,
                    semimarkov=replace(
                        result.semimarkov,
                        audit=tampered_audit,
                    ),
                )

    def test_contextual_v2_raw_or_partial_activation_evidence_fails_closed(
        self,
    ) -> None:
        from engine.tests.test_stage_edit_contextual_beam_qa_adapter_v2 import (
            _raw_passing_case,
        )

        raw_case: StageEditContextualBeamQACaseV2 = _raw_passing_case()
        common = {
            "feature_snapshot_path": Path("/tmp/missing-contextual-snapshot.json"),
            "checkpoint_directory": Path("/tmp/missing-contextual-checkpoint"),
            "source_ids": ("source-a", "source-b"),
            "dynamic_crop_version": 3,
            "continuity_authority": (
                PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
            ),
            "contextual_source_pair_bundle_directory": Path(
                "/tmp/missing-contextual-bundle"
            ),
            "contextual_source_pair_checkpoint_sha256": "1" * 64,
            "contextual_source_pair_checkpoint_file_sha256": "2" * 64,
            "contextual_source_pair_report_sha256": "3" * 64,
            "contextual_source_pair_report_file_sha256": "4" * 64,
            "contextual_source_pair_beam_qa_report_sha256": "5" * 64,
            "contextual_source_pair_corpus_sha256": "6" * 64,
            "contextual_source_pair_corpus_file_sha256": "7" * 64,
            "contextual_source_pair_token_schema_fingerprint": (
                DEFAULT_STAGE_EDIT_SCHEMA.fingerprint
            ),
            "contextual_source_pair_verified_beam_qa_cases": (raw_case,),
        }
        with self.assertRaisesRegex(
            StageEditPipelineBridgeInputError,
            "raw case DTOs are inspection-only",
        ):
            preflight_stage_edit_pipeline_bridge(**common)

        partial = dict(common)
        partial["contextual_source_pair_verified_beam_qa_cases"] = (object(),)
        partial.pop("contextual_source_pair_report_file_sha256")
        with self.assertRaisesRegex(
            StageEditPipelineBridgeInputError,
            "requires complete corpus/checkpoint/report",
        ):
            preflight_stage_edit_pipeline_bridge(**partial)

        fake_bundle = create_autospec(
            StageEditContextualSourcePairProductionBundleV2,
            instance=True,
        )
        with self.assertRaisesRegex(ValueError, "issued by production preflight"):
            StageEditContextualSourcePairBridgeIdentityV2(
                directory=Path("/tmp/missing-contextual-bundle"),
                checkpoint_sha256="1" * 64,
                checkpoint_file_sha256="2" * 64,
                report_sha256="3" * 64,
                report_file_sha256="4" * 64,
                beam_qa_report_sha256="5" * 64,
                corpus_sha256="6" * 64,
                corpus_file_sha256="7" * 64,
                token_schema_fingerprint=DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
                scorer_fingerprint="8" * 64,
                verified_beam_qa_cases=(),
                _bundle=fake_bundle,
                _seal=object(),
            )
        contextual_scorer = StageEditContextualSourcePairScorerV2(
            token_schema_fingerprint=DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
            weights=(0.0,)
            * len(STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_FEATURE_NAMES),
        )
        with self.assertRaisesRegex(ValueError, "complete sealed identity"):
            StageEditPipelineBridgePreflight(
                snapshot_path=Path(
                    "/tmp/contextual-partial-snapshot.json"
                ).resolve(),
                checkpoint_directory=Path("/tmp/contextual-partial-checkpoint"),
                source_ids=("source-a", "source-b"),
                snapshot=_snapshot(),
                readiness=_ready(
                    Path("/tmp/contextual-partial-checkpoint"),
                    dynamic_crop_version=3,
                ),
                snapshot_identity=_snapshot_identity(
                    _snapshot(), Path("/tmp/contextual-partial-snapshot.json")
                ),
                edit_mode="group",
                target_member=None,
                dynamic_crop_version=3,
                continuity_authority=(
                    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
                ),
                contextual_source_pair_scorer_v2=contextual_scorer,
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
            snapshot_identity=_snapshot_identity(
                snapshot, Path("/tmp/stageedit-feature.json")
            ),
            edit_mode="group",
            target_member=None,
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
        self.assertEqual(STAGE_EDIT_PIPELINE_BRIDGE_SCHEMA_VERSION, 5)
        self.assertTrue(STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT.endswith("_v5"))
        self.assertEqual(result.schema_version, 5)
        self.assertEqual(result.contract, STAGE_EDIT_PIPELINE_BRIDGE_CONTRACT)
        self.assertEqual(8, len(result.retained_candidates))
        self.assertFalse(result.selection_policy["selected_for_render"])
        self.assertFalse(result.selection_policy["visual_baseline_replaced"])
        self.assertEqual(
            result.selection_policy["temporal_pose_sidecar"],
            preflight.snapshot.temporal_pose_sidecar_audit(),
        )
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
                snapshot_identity=_snapshot_identity(
                    snapshot, root / "stageedit-v3-feature.json"
                ),
                edit_mode="group",
                target_member=None,
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
        self.assertIsNone(
            decode.call_args.kwargs["contextual_source_pair_scorer_v2"]
        )
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
            snapshot_identity=_snapshot_identity(
                snapshot, Path("/tmp/stageedit-v3-feature.json")
            ),
            edit_mode="group",
            target_member=None,
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
        self.assertIsNone(
            decode.call_args.kwargs["contextual_source_pair_scorer_v2"]
        )

    def test_requested_decode_shortfall_fails_without_visual_fallback(self):
        snapshot = _snapshot()
        checkpoint = Path("/tmp/stageedit-bridge-ready").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=snapshot,
            readiness=_ready(checkpoint),
            snapshot_identity=_snapshot_identity(
                snapshot, Path("/tmp/stageedit-feature.json")
            ),
            edit_mode="group",
            target_member=None,
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
                edit_mode=EditMode.GROUP,
                target_member=None,
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
                edit_mode=EditMode.GROUP,
                target_member=None,
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
                statistics={},
            )
            causal_artifacts = object()
            causal = SimpleNamespace(
                stage_edit_artifacts=causal_artifacts,
                require_exact_inference_snapshot_binding=(
                    lambda observed: {
                        "exact_inference_snapshot_binding_verified": (
                            observed is snapshot
                        )
                    }
                ),
            )
            persisted_audit = SimpleNamespace(
                path=(root / "causal-sensing-audit.json").resolve(),
                to_dict=lambda: {"content_sha256": "f" * 64},
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
                    "crosscut.pipeline."
                    "materialize_ive_causal_sensing_for_stage_edit_pipeline",
                    return_value=causal,
                ) as causal_materialize,
                patch(
                    "crosscut.pipeline.write_ive_causal_sensing_pipeline_audit",
                    return_value=persisted_audit,
                ) as write_causal_audit,
                patch(
                    "crosscut.pipeline.preflight_stage_edit_pipeline_bridge",
                    return_value=preflight,
                ) as preflight_call,
            ):
                observed_snapshot, observed_preflight, observed_audit = (
                    _materialize_registered_stage_edit_preflight(
                        bundle=bundle,
                        request=request,
                        alignment=alignment,
                        alignment_eligibility=eligibility,
                        source_timewarps=(None,) * 8,
                        source_scene_cut_times=((),) * 8,
                        timing_draft=timing,
                        rhythm=rhythm,
                        section_plan=sections,
                    )
                )

            self.assertIs(observed_snapshot, snapshot)
            self.assertIs(observed_preflight, preflight)
            self.assertIs(observed_audit, persisted_audit)
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
                causal_sensing=causal_artifacts,
                edit_mode=EditMode.GROUP,
                target_member=None,
            )
            causal_materialize.assert_called_once()
            write_causal_audit.assert_called_once_with(
                pipeline_result=causal,
                inference_snapshot=snapshot,
                artifact_directory=(
                    request.source_cache_directory
                    / "derived"
                    / "stage-edit-inference-v1"
                    / "causal-sensing-audit"
                ),
            )
            preflight_call.assert_called_once_with(
                feature_snapshot_path=snapshot.snapshot_path,
                checkpoint_directory=checkpoint,
                source_ids=snapshot.source_ids,
                edit_mode=EditMode.GROUP,
                target_member=None,
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
                statistics={},
            )
            causal_artifacts = object()
            causal = SimpleNamespace(
                stage_edit_artifacts=causal_artifacts,
                require_exact_inference_snapshot_binding=(
                    lambda observed: {
                        "exact_inference_snapshot_binding_verified": (
                            observed is snapshot
                        )
                    }
                ),
            )
            persisted_audit = SimpleNamespace(
                path=(root / "causal-sensing-audit.json").resolve(),
                to_dict=lambda: {"content_sha256": "f" * 64},
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
                    "crosscut.pipeline."
                    "materialize_ive_causal_sensing_for_stage_edit_pipeline",
                    return_value=causal,
                ),
                patch(
                    "crosscut.pipeline.write_ive_causal_sensing_pipeline_audit",
                    return_value=persisted_audit,
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
                    source_scene_cut_times=((),) * 8,
                    timing_draft=object(),
                    rhythm=object(),
                    section_plan=object(),
                )
            preflight_call.assert_called_once_with(
                feature_snapshot_path=snapshot.snapshot_path,
                checkpoint_directory=checkpoint,
                source_ids=snapshot.source_ids,
                edit_mode=EditMode.GROUP,
                target_member=None,
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
