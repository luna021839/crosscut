from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile
import unittest
from unittest import mock

from crosscut.ive import AspectRatio
from crosscut.post_render_qa import (
    PostRenderArtifactSet,
    PostRenderQAConfig,
    PostRenderQAReport,
    PostRenderQAVerdict,
    fingerprint_post_render_artifacts,
)
from crosscut.render import RenderSegment, RenderTransition
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit import StageBoundary, StageEdit, StageShot
from crosscut.stage_edit_continuity_authority import (
    CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
)
from crosscut.stage_edit_contextual_beam_qa_adapter_v2 import (
    STAGE_EDIT_CONTEXTUAL_BEAM_QA_MOTION_EVIDENCE_V2_CONTRACT,
    STAGE_EDIT_CONTEXTUAL_BEAM_QA_POST_RENDER_METRICS_V2_CONTRACT,
    STAGE_EDIT_CONTEXTUAL_BEAM_QA_SOURCE_DIVERSITY_V2_CONTRACT,
    StageEditContextualBeamQAAdapterV2Error,
    VerifiedStageEditContextualBeamQACaseV2,
    _canonical_sha256,
    _derived_boundary_indices,
    _read_stable_json_artifact,
    derive_stage_edit_contextual_beam_qa_case_v2,
    is_verified_stage_edit_contextual_beam_qa_case_v2,
    rehydrate_stage_edit_contextual_beam_qa_case_v2,
    require_verified_stage_edit_contextual_beam_qa_case_v2,
)
from crosscut.stage_edit_semimarkov_beam import (
    DECODER_CONTRACT,
    DECODER_SCHEMA_VERSION,
    StageEditBeamDecision,
    StageEditBeamTimeline,
    StageEditDecodeAudit,
    StageEditSemiMarkovResult,
)
from crosscut.stage_edit_source_pair_contextual_v2_production import (
    StageEditContextualBeamQACaseV2,
    load_stage_edit_contextual_source_pair_evaluation_bundle_v2,
    new_stage_edit_contextual_source_pair_checkpoint_v2,
    save_stage_edit_contextual_source_pair_checkpoint_v2,
    save_stage_edit_contextual_training_corpus_v2,
    train_stage_edit_contextual_source_pair_v2,
)


def _raw_passing_case() -> StageEditContextualBeamQACaseV2:
    return StageEditContextualBeamQACaseV2(
        case_id="forged-full-decode",
        song_id="FORGED",
        request_sha256="1" * 64,
        decode_report_sha256="2" * 64,
        full_decode=True,
        duration_seconds=180.0,
        switch_count=45,
        eligible_switch_opportunity_count=100,
        visual_segment_median_seconds=1.8,
        source_run_median_seconds=4.0,
        longest_source_run_seconds=6.5,
        distinct_source_count=9,
        effective_source_count=8.4,
        dominant_source_ratio=0.25,
        quality_score_p10=0.42,
        quality_score_median_span=0.18,
        boundary_match_score_p10=62.0,
        weighted_proxy_score=65.0,
        highlight_window_count=3,
        highlight_switch_opportunity_count=6,
        highlight_switch_count=3,
        verified_highlight_switch_count=3,
        closeup_motion_opportunity_count=4,
        closeup_motion_switch_count=2,
        verified_closeup_motion_switch_count=2,
        geometry_issue_count=0,
        sync_issue_count=0,
    )


def _section_plan() -> SectionPlan:
    return SectionPlan(
        (
            SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=0.0,
                end_seconds=6.0,
                energy_score=0.9,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=20.0,
                camera_motion_budget=0.8,
                target_roll_events_per_minute=0.2,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _decision(index: int, *, motion: bool) -> StageEditBeamDecision:
    start = float(index * 2)
    source_occupancy = tuple(2.0 if value <= index else 0.0 for value in range(3))
    return StageEditBeamDecision(
        shot_index=index,
        start_candidate_index=index,
        end_candidate_index=index + 1,
        master_start_seconds=start,
        master_end_seconds=start + 2.0,
        source_index=index,
        source_id=f"source-{index}",
        derivative_family_id=index,
        shot_size="closeup",
        detected_person_count_normalized=0.2,
        formation_person_count_normalized=0.2,
        formation_pose_coverage=0.8,
        duration_bucket=0,
        transition_label="initial" if index == 0 else "hard_cut",
        crop_requested=(1.0, 0.5, 0.5, 0.0),
        crop_applied=(1.0, 0.5, 0.5, 0.0),
        crop_was_corrected=False,
        decision_confidence=0.9,
        score_components=(),
        recent_sources_after=tuple(range(index + 1)),
        recent_shot_sizes_after=("closeup",) * (index + 1),
        recent_shot_durations_after=(2.0,) * (index + 1),
        recent_cut_density_per_minute_after=20.0,
        source_occupancy_after_seconds=source_occupancy,
        derivative_family_occupancy_after_seconds=source_occupancy,
        section_switch_counts_after=(max(0, index),),
        section_progress=index / 3.0,
        phrase_progress=index / 3.0,
        section_role="chorus_like",
        boundary_motion_amount_plus_continuity_evidence_complete=motion,
        boundary_motion_amount_plus_continuity_supported=motion,
        boundary_temporal_pose_motion_input_valid=motion,
        boundary_temporal_sampling_contract_match=motion,
        boundary_pose_direction_or_velocity_proven=motion,
    )


def _semimarkov(request_sha256: str, output: Path) -> StageEditSemiMarkovResult:
    decisions = tuple(_decision(index, motion=index > 0) for index in range(3))
    plan = _section_plan()
    sources = tuple(Path(f"/tmp/source-{index}.mp4") for index in range(3))
    shots = tuple(
        StageShot(
            shot_index=index,
            section_index=0,
            segment=RenderSegment(
                source_index=index,
                source_start=0.0,
                duration=2.0,
                master_start=float(index * 2),
            ),
            decision_id=f"decision-{index}",
            decision_confidence=0.9,
            section_role="chorus_like",
            section_progress=index / 3.0,
            phrase_progress=index / 3.0,
        )
        for index in range(3)
    )
    edit = StageEdit(
        sources=sources,
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        width=1920,
        height=1080,
        fps=30.0,
        master_audio_start=0.0,
        section_plan=plan,
        shots=shots,
        boundaries=tuple(
            StageBoundary(index, RenderTransition()) for index in range(2)
        ),
        output=output,
    )
    timeline = StageEditBeamTimeline(
        rank=1,
        score=1.0,
        score_components={},
        decisions=decisions,
        edit=edit,
        source_occupancy_seconds=(2.0, 2.0, 2.0),
        derivative_family_occupancy_seconds=(2.0, 2.0, 2.0),
        recent_cut_density_per_minute=20.0,
        boundary_continuity_score_provenance={
            "authority": PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
        },
    )
    audit = StageEditDecodeAudit(
        schema_version=DECODER_SCHEMA_VERSION,
        decoder=DECODER_CONTRACT,
        input_sha256="d" * 64,
        readiness={},
        config={
            "continuity_authority": PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
        },
        input_summary={
            "master_start_seconds": 0.0,
            "master_end_seconds": 6.0,
            "production_request_sha256": request_sha256,
        },
        endpoint_audit=(),
        constraint_rejection_counts={},
        transition_hypothesis_count=3,
        accepted_hypothesis_count=3,
        beam_pruned_state_count=0,
        complete_state_count=1,
        duplicate_completion_count=0,
        requested_timeline_count=1,
        returned_timeline_count=1,
        request_fulfilled=True,
        score_provenance={
            "boundary_continuity": {
                "authority": PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
            }
        },
    )
    return StageEditSemiMarkovResult((timeline,), audit)


class _Fixture:
    def __init__(self, root: Path, *, omit_metric: str | None = None) -> None:
        self.base = "IVE-TEST-20260825-120000"
        self.analysis_path = root / f"{self.base}-analysis.json"
        self.plan_path = root / f"{self.base}-render-plan.json"
        self.selection_path = root / f"{self.base}-rendered-candidate-selection.json"
        self.video_path = root / f"{self.base}-group-crosscut.mp4"
        self.decode_path = root / "stage-edit-decode.json"
        self.report_path = root / f"{self.base}-post-render-qa.json"
        self.request = {
            "group": "IVE",
            "song": "TEST",
            "edit_mode": "group",
            "source_mode": "automatic",
        }
        request_sha256 = _canonical_sha256(self.request)
        self.semimarkov = _semimarkov(request_sha256, self.video_path)
        semimarkov_payload = self.semimarkov.to_dict()
        self.decode_path.write_text(
            json.dumps({"semimarkov": semimarkov_payload}, indent=2),
            encoding="utf-8",
        )
        analysis = {
            "request": self.request,
            "source_index_mapping": {
                "eligible_to_request": [0, 1, 2],
            },
            "source_derivative_diversity": {
                "family_ids": [0, 1, 2],
            },
            "stage_edit_generation": {
                "audit_path": str(self.decode_path),
                "semimarkov": semimarkov_payload,
            },
        }
        self.analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        self.plan_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {"source_index": 0, "duration": 2.0},
                        {"source_index": 1, "duration": 2.0},
                        {"source_index": 2, "duration": 2.0},
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.selection_path.write_text(
            json.dumps(
                {
                    "final_selected_version": "original",
                    "final_selected_review": {
                        "boundaries": [
                            {"boundary_index": 0, "passed": True},
                            {"boundary_index": 1, "passed": True},
                        ]
                    },
                    "stage_edit_render_promotion": {
                        "selected_candidate_lineage": [
                            {"stage_edit_candidate_rank": 1}
                        ]
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.video_path.write_bytes(b"not-a-real-video-but-content-bound")
        self.artifacts = PostRenderArtifactSet(
            base_name=self.base,
            timestamp="20260825-120000",
            analysis_path=self.analysis_path,
            render_plan_path=self.plan_path,
            selection_path=self.selection_path,
            video_path=self.video_path,
        )
        fingerprints = fingerprint_post_render_artifacts(self.artifacts)
        timeline = self.semimarkov.timelines[0]
        cadence = {
            "duration_seconds": 6.0,
            "source_switch_count": 2,
            "comparable_visual_boundary_count": 2,
            "visual_shot_median_seconds": 2.0,
            "source_run_median_seconds": 2.0,
            "longest_source_run_seconds": 2.0,
            "distinct_derivative_family_count": 3,
            "effective_derivative_family_count": 3.0,
            "dominant_derivative_family_share": 1.0 / 3.0,
        }
        if omit_metric is not None:
            cadence.pop(omit_metric)
        metrics = {
            "artifact_content_stability": {
                "stable_across_evaluation": True,
                "final_fingerprints": fingerprints,
            },
            "tool_checks": {"full_decode": {"passed": True}},
            "cadence_and_repetition": cadence,
            "proxy_quality_and_geometry": {
                "overall_quality_p10": 0.42,
                "source_quality_median_span": 0.18,
                "boundary_score_p10": 62.0,
                "weighted_score": 65.0,
                "actual_boundary_issue_count": 0,
                "geometry_provider_failure_count": 0,
                "insufficient_transition_handle_count": 0,
                "boundary_sample_contract_valid": True,
            },
            "production_sync": {
                "per_source": [{"passed": True} for _ in range(3)],
                "cut_local_sync": {
                    "unknown_boundary_count": 0,
                    "errors": [],
                },
                "passed": True,
            },
            "rendered_montage_grammar": {
                "recomputed_review": {
                    "highlight_aggregate": {
                        "strict_highlight_section_indices": [0],
                        "strict_highlight_section_count": 1,
                        "observed": {"source_switch_count": 2},
                    }
                }
            },
            "stage_edit_contextual_beam_qa_v2": {
                "contract": (
                    STAGE_EDIT_CONTEXTUAL_BEAM_QA_POST_RENDER_METRICS_V2_CONTRACT
                ),
                "source_diversity_contract": (
                    STAGE_EDIT_CONTEXTUAL_BEAM_QA_SOURCE_DIVERSITY_V2_CONTRACT
                ),
                "motion_evidence_contract": (
                    STAGE_EDIT_CONTEXTUAL_BEAM_QA_MOTION_EVIDENCE_V2_CONTRACT
                ),
                "decode_result_content_sha256": _canonical_sha256(
                    semimarkov_payload
                ),
                "decoder_input_sha256": self.semimarkov.audit.input_sha256,
                "render_plan_file_sha256": fingerprints["render_plan"]["sha256"],
                "selected_timeline_rank": 1,
                "selected_timeline_content_sha256": _canonical_sha256(
                    timeline.to_dict()
                ),
                "highlight_switch_opportunity_boundary_indices": [0, 1],
                "highlight_switch_boundary_indices": [0, 1],
                "proxy_verified_highlight_switch_boundary_indices": [0, 1],
                "closeup_motion_opportunity_boundary_indices": [0, 1],
                "closeup_motion_switch_boundary_indices": [0, 1],
                "proxy_verified_closeup_motion_switch_boundary_indices": [0, 1],
            },
        }
        self.report = PostRenderQAReport(
            artifact_set=self.artifacts,
            verdict=PostRenderQAVerdict.PASS,
            metrics=metrics,
            issues=(),
            policy=PostRenderQAConfig(),
            input_artifact_fingerprints=fingerprints,
            created_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        )
        self.report.write_atomic(self.report_path)


def _production_semimarkov(
    request_sha256: str,
    output: Path,
    *,
    scorer_contract: str,
    scorer_fingerprint: str,
) -> StageEditSemiMarkovResult:
    source_path = tuple(
        value
        for family_id in (*range(9), *range(5))
        for value in (family_id, family_id)
    )
    source_count = 9
    occupancy = [0.0] * source_count
    decisions: list[StageEditBeamDecision] = []
    for index, source_index in enumerate(source_path):
        occupancy[source_index] += 2.0
        decisions.append(
            StageEditBeamDecision(
                shot_index=index,
                start_candidate_index=index,
                end_candidate_index=index + 1,
                master_start_seconds=float(index * 2),
                master_end_seconds=float(index * 2 + 2),
                source_index=source_index,
                source_id=f"source-{source_index}",
                derivative_family_id=source_index,
                shot_size="closeup",
                detected_person_count_normalized=0.2,
                formation_person_count_normalized=0.2,
                formation_pose_coverage=0.8,
                duration_bucket=0,
                transition_label="initial" if index == 0 else "hard_cut",
                crop_requested=(1.0, 0.5, 0.5, 0.0),
                crop_applied=(1.0, 0.5, 0.5, 0.0),
                crop_was_corrected=False,
                decision_confidence=0.9,
                score_components=(),
                recent_sources_after=tuple(source_path[max(0, index - 3): index + 1]),
                recent_shot_sizes_after=("closeup",) * min(index + 1, 4),
                recent_shot_durations_after=(2.0,) * min(index + 1, 4),
                recent_cut_density_per_minute_after=14.0,
                source_occupancy_after_seconds=tuple(occupancy),
                derivative_family_occupancy_after_seconds=tuple(occupancy),
                section_switch_counts_after=(
                    sum(
                        left != right
                        for left, right in zip(
                            source_path[:index],
                            source_path[1:index + 1],
                            strict=True,
                        )
                    ),
                ),
                section_progress=index / (len(source_path) - 1),
                phrase_progress=index / (len(source_path) - 1),
                section_role="chorus_like",
                boundary_motion_amount_plus_continuity_evidence_complete=(
                    index > 0
                ),
                boundary_motion_amount_plus_continuity_supported=index > 0,
                boundary_temporal_pose_motion_input_valid=index > 0,
                boundary_temporal_sampling_contract_match=index > 0,
                boundary_pose_direction_or_velocity_proven=index > 0,
            )
        )
    plan = SectionPlan(
        (
            SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=0.0,
                end_seconds=float(len(source_path) * 2),
                energy_score=0.9,
                confidence=0.9,
                phrase_count=4,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=14.0,
                camera_motion_budget=0.8,
                target_roll_events_per_minute=0.2,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )
    sources = tuple(Path(f"/tmp/source-{index}.mp4") for index in range(source_count))
    shots = tuple(
        StageShot(
            shot_index=index,
            section_index=0,
            segment=RenderSegment(
                source_index=source_index,
                source_start=0.0,
                duration=2.0,
                master_start=float(index * 2),
            ),
            decision_id=f"decision-{index}",
            decision_confidence=0.9,
            section_role="chorus_like",
            section_progress=index / (len(source_path) - 1),
            phrase_progress=index / (len(source_path) - 1),
        )
        for index, source_index in enumerate(source_path)
    )
    edit = StageEdit(
        sources=sources,
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        width=1920,
        height=1080,
        fps=30.0,
        master_audio_start=0.0,
        section_plan=plan,
        shots=shots,
        boundaries=tuple(
            StageBoundary(index, RenderTransition())
            for index in range(len(shots) - 1)
        ),
        output=output,
    )
    provenance = {
        "authority": CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
        "scorer_contract": scorer_contract,
        "scorer_fingerprint": scorer_fingerprint,
    }
    timeline = StageEditBeamTimeline(
        rank=1,
        score=1.0,
        score_components={},
        decisions=tuple(decisions),
        edit=edit,
        source_occupancy_seconds=tuple(occupancy),
        derivative_family_occupancy_seconds=tuple(occupancy),
        recent_cut_density_per_minute=14.0,
        boundary_continuity_score_provenance=provenance,
    )
    audit = StageEditDecodeAudit(
        schema_version=DECODER_SCHEMA_VERSION,
        decoder=DECODER_CONTRACT,
        input_sha256="d" * 64,
        readiness={},
        config={
            "continuity_authority": (
                CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
            ),
            "continuity_scorer_contract": scorer_contract,
            "continuity_scorer_fingerprint": scorer_fingerprint,
        },
        input_summary={
            "master_start_seconds": 0.0,
            "master_end_seconds": float(len(source_path) * 2),
            "production_request_sha256": request_sha256,
        },
        endpoint_audit=(),
        constraint_rejection_counts={},
        transition_hypothesis_count=len(decisions),
        accepted_hypothesis_count=len(decisions),
        beam_pruned_state_count=0,
        complete_state_count=1,
        duplicate_completion_count=0,
        requested_timeline_count=1,
        returned_timeline_count=1,
        request_fulfilled=True,
        score_provenance={"boundary_continuity": provenance},
    )
    return StageEditSemiMarkovResult((timeline,), audit)


class _ProductionFixture:
    """Synthetic full-decode artifacts that meet every production Beam-QA gate."""

    def __init__(self, root: Path, *, song: str, evaluation_bundle) -> None:
        root.mkdir()
        safe_song = song.lower().replace(" ", "-")
        self.base = f"IVE-{safe_song}-20260825-120000"
        self.analysis_path = root / f"{self.base}-analysis.json"
        self.plan_path = root / f"{self.base}-render-plan.json"
        self.selection_path = root / f"{self.base}-rendered-candidate-selection.json"
        self.video_path = root / f"{self.base}-group-crosscut.mp4"
        self.decode_path = root / f"{safe_song}-stage-edit-decode.json"
        self.report_path = root / f"{self.base}-post-render-qa.json"
        self.request = {
            "group": "IVE",
            "song": song,
            "edit_mode": "group",
            "source_mode": "automatic",
        }
        request_sha256 = _canonical_sha256(self.request)
        self.semimarkov = _production_semimarkov(
            request_sha256,
            self.video_path,
            scorer_contract=evaluation_bundle.scorer.contract,
            scorer_fingerprint=evaluation_bundle.scorer.fingerprint,
        )
        semimarkov_payload = self.semimarkov.to_dict()
        self.decode_path.write_text(
            json.dumps({"semimarkov": semimarkov_payload}, indent=2),
            encoding="utf-8",
        )
        source_count = len(self.semimarkov.timelines[0].edit.sources)
        analysis = {
            "request": self.request,
            "source_index_mapping": {
                "eligible_to_request": list(range(source_count)),
            },
            "source_derivative_diversity": {
                "family_ids": list(range(source_count)),
            },
            "stage_edit_generation": {
                "audit_path": str(self.decode_path),
                "semimarkov": semimarkov_payload,
            },
        }
        self.analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        decisions = self.semimarkov.timelines[0].decisions
        self.plan_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "source_index": decision.source_index,
                            "duration": decision.duration_seconds,
                        }
                        for decision in decisions
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        boundary_count = len(decisions) - 1
        self.selection_path.write_text(
            json.dumps(
                {
                    "final_selected_version": "original",
                    "final_selected_review": {
                        "boundaries": [
                            {"boundary_index": index, "passed": True}
                            for index in range(boundary_count)
                        ]
                    },
                    "stage_edit_render_promotion": {
                        "selected_candidate_lineage": [
                            {"stage_edit_candidate_rank": 1}
                        ]
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.video_path.write_bytes(
            f"synthetic-production-video:{song}".encode("utf-8")
        )
        self.artifacts = PostRenderArtifactSet(
            base_name=self.base,
            timestamp="20260825-120000",
            analysis_path=self.analysis_path,
            render_plan_path=self.plan_path,
            selection_path=self.selection_path,
            video_path=self.video_path,
        )
        fingerprints = fingerprint_post_render_artifacts(self.artifacts)
        timeline = self.semimarkov.timelines[0]
        runs: list[float] = []
        run_sources: list[int] = []
        for decision in decisions:
            if run_sources and run_sources[-1] == decision.source_index:
                runs[-1] += decision.duration_seconds
            else:
                run_sources.append(decision.source_index)
                runs.append(decision.duration_seconds)
        durations_by_family: dict[int, float] = {}
        for decision in decisions:
            durations_by_family[decision.derivative_family_id] = (
                durations_by_family.get(decision.derivative_family_id, 0.0)
                + decision.duration_seconds
            )
        duration = sum(item.duration_seconds for item in decisions)
        shares = tuple(value / duration for value in durations_by_family.values())
        switch_indices = tuple(
            index
            for index, (left, right) in enumerate(
                zip(decisions[:-1], decisions[1:], strict=True)
            )
            if left.derivative_family_id != right.derivative_family_id
        )
        derived = _derived_boundary_indices(
            timeline,
            strict_highlight_section_indices=frozenset({0}),
        )
        cadence = {
            "duration_seconds": duration,
            "source_switch_count": len(switch_indices),
            "comparable_visual_boundary_count": boundary_count,
            "visual_shot_median_seconds": float(
                median(item.duration_seconds for item in decisions)
            ),
            "source_run_median_seconds": float(median(runs)),
            "longest_source_run_seconds": max(runs),
            "distinct_derivative_family_count": len(durations_by_family),
            "effective_derivative_family_count": math.exp(
                -sum(value * math.log(value) for value in shares)
            ),
            "dominant_derivative_family_share": max(shares),
        }
        metrics = {
            "artifact_content_stability": {
                "stable_across_evaluation": True,
                "final_fingerprints": fingerprints,
            },
            "tool_checks": {"full_decode": {"passed": True}},
            "cadence_and_repetition": cadence,
            "proxy_quality_and_geometry": {
                "overall_quality_p10": 0.42,
                "source_quality_median_span": 0.18,
                "boundary_score_p10": 62.0,
                "weighted_score": 65.0,
                "actual_boundary_issue_count": 0,
                "geometry_provider_failure_count": 0,
                "insufficient_transition_handle_count": 0,
                "boundary_sample_contract_valid": True,
            },
            "production_sync": {
                "per_source": [{"passed": True} for _ in range(source_count)],
                "cut_local_sync": {"unknown_boundary_count": 0, "errors": []},
                "passed": True,
            },
            "rendered_montage_grammar": {
                "recomputed_review": {
                    "highlight_aggregate": {
                        "strict_highlight_section_indices": [0],
                        "strict_highlight_section_count": 1,
                        "observed": {"source_switch_count": len(switch_indices)},
                    }
                }
            },
            "stage_edit_contextual_beam_qa_v2": {
                "contract": (
                    STAGE_EDIT_CONTEXTUAL_BEAM_QA_POST_RENDER_METRICS_V2_CONTRACT
                ),
                "source_diversity_contract": (
                    STAGE_EDIT_CONTEXTUAL_BEAM_QA_SOURCE_DIVERSITY_V2_CONTRACT
                ),
                "motion_evidence_contract": (
                    STAGE_EDIT_CONTEXTUAL_BEAM_QA_MOTION_EVIDENCE_V2_CONTRACT
                ),
                "decode_result_content_sha256": _canonical_sha256(
                    semimarkov_payload
                ),
                "decoder_input_sha256": self.semimarkov.audit.input_sha256,
                "render_plan_file_sha256": fingerprints["render_plan"]["sha256"],
                "selected_timeline_rank": 1,
                "selected_timeline_content_sha256": _canonical_sha256(
                    timeline.to_dict()
                ),
                **{key: list(value) for key, value in derived.items()},
                "proxy_verified_highlight_switch_boundary_indices": list(
                    derived["highlight_switch_boundary_indices"]
                ),
                "proxy_verified_closeup_motion_switch_boundary_indices": list(
                    derived["closeup_motion_switch_boundary_indices"]
                ),
            },
        }
        self.report = PostRenderQAReport(
            artifact_set=self.artifacts,
            verdict=PostRenderQAVerdict.PASS,
            metrics=metrics,
            issues=(),
            policy=PostRenderQAConfig(),
            input_artifact_fingerprints=fingerprints,
            created_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        )
        self.report.write_atomic(self.report_path)


class StageEditContextualBeamQAAdapterV2Test(unittest.TestCase):
    def test_closeup_motion_evidence_is_not_restricted_to_highlights(self) -> None:
        timeline = _semimarkov("1" * 64, Path("/tmp/result.mp4")).timelines[0]
        decisions = (
            timeline.decisions[0],
            replace(timeline.decisions[1], section_role="verse_like"),
            timeline.decisions[2],
        )
        observed = _derived_boundary_indices(
            replace(timeline, decisions=decisions),
            strict_highlight_section_indices=frozenset(),
        )
        self.assertEqual(
            observed["highlight_switch_opportunity_boundary_indices"],
            (),
        )
        self.assertEqual(
            observed["closeup_motion_opportunity_boundary_indices"],
            (0, 1),
        )
        self.assertEqual(
            observed["closeup_motion_switch_boundary_indices"],
            (0, 1),
        )

    def test_artifact_reader_rejects_symlinked_parent_component(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            root = Path(raw_root)
            actual = root / "actual"
            actual.mkdir()
            (actual / "evidence.json").write_text("{}", encoding="utf-8")
            alias = root / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(
                StageEditContextualBeamQAAdapterV2Error,
                "symlink or non-directory component",
            ):
                _read_stable_json_artifact(alias / "evidence.json", "evidence")

    def test_artifact_reader_rejects_hard_linked_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            root = Path(raw_root)
            original = root / "original.json"
            linked = root / "linked.json"
            original.write_text("{}", encoding="utf-8")
            os.link(original, linked)
            with self.assertRaisesRegex(
                StageEditContextualBeamQAAdapterV2Error,
                "regular single-link",
            ):
                _read_stable_json_artifact(linked, "evidence")

    def test_handcrafted_raw_passing_case_is_inspection_only(self) -> None:
        raw = _raw_passing_case()
        self.assertTrue(raw.authorized)
        self.assertFalse(is_verified_stage_edit_contextual_beam_qa_case_v2(raw))
        with self.assertRaisesRegex(
            StageEditContextualBeamQAAdapterV2Error, "adapter-issued"
        ):
            require_verified_stage_edit_contextual_beam_qa_case_v2(raw)
        with self.assertRaisesRegex(
            StageEditContextualBeamQAAdapterV2Error, "inspection-only"
        ):
            VerifiedStageEditContextualBeamQACaseV2(
                case=raw,
                derivation=object(),  # type: ignore[arg-type]
                _seal=object(),
            )

    def test_factory_binds_typed_decode_and_byte_plus_content_identities(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            with mock.patch(
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
            self.assertTrue(is_verified_stage_edit_contextual_beam_qa_case_v2(verified))
            self.assertIs(
                verified.case,
                require_verified_stage_edit_contextual_beam_qa_case_v2(verified),
            )
            self.assertEqual(
                fixture.semimarkov.audit.input_summary["production_request_sha256"],
                verified.case.request_sha256,
            )
            self.assertNotEqual(
                verified.derivation.decode_report.file_sha256,
                verified.derivation.decode_report.canonical_content_sha256,
            )
            self.assertNotEqual(
                verified.derivation.post_render_report.file_sha256,
                verified.derivation.post_render_report.canonical_content_sha256,
            )

    def test_factory_rejects_decode_report_that_differs_from_typed_result(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            payload = json.loads(fixture.decode_path.read_text(encoding="utf-8"))
            payload["semimarkov"]["audit"]["input_sha256"] = "e" * 64
            fixture.decode_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                StageEditContextualBeamQAAdapterV2Error,
                "typed semi-Markov result differs",
            ):
                derive_stage_edit_contextual_beam_qa_case_v2(
                    semimarkov=fixture.semimarkov,
                    selected_timeline_rank=1,
                    post_render_report=fixture.report,
                    decode_report_path=fixture.decode_path,
                    post_render_report_path=fixture.report_path,
                )

    def test_activation_reopens_and_rejects_post_factory_artifact_tamper(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            with mock.patch(
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
            fixture.plan_path.write_text(
                json.dumps({"plan": "tampered-after-factory"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                StageEditContextualBeamQAAdapterV2Error,
                "render plan identity changed",
            ):
                require_verified_stage_edit_contextual_beam_qa_case_v2(verified)

    def test_activation_rejects_post_factory_case_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            with mock.patch(
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
            object.__setattr__(verified.case, "duration_seconds", 600.0)
            with self.assertRaisesRegex(
                StageEditContextualBeamQAAdapterV2Error,
                "case digest changed",
            ):
                require_verified_stage_edit_contextual_beam_qa_case_v2(verified)

    def test_missing_post_render_metric_fails_closed_without_substitute(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root), omit_metric="source_switch_count")
            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2.evaluate_post_render_qa",
                return_value=fixture.report,
            ):
                with self.assertRaisesRegex(
                    StageEditContextualBeamQAAdapterV2Error,
                    "source_switch_count must be an integer",
                ):
                    derive_stage_edit_contextual_beam_qa_case_v2(
                        semimarkov=fixture.semimarkov,
                        selected_timeline_rank=1,
                        post_render_report=fixture.report,
                        decode_report_path=fixture.decode_path,
                        post_render_report_path=fixture.report_path,
                    )

    def test_forged_cadence_metric_cannot_override_bound_render_plan(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            cadence = fixture.report.metrics["cadence_and_repetition"]
            self.assertIsInstance(cadence, dict)
            cadence["duration_seconds"] = 600.0  # type: ignore[index]
            fixture.report.write_atomic(fixture.report_path)
            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2.evaluate_post_render_qa",
                return_value=fixture.report,
            ):
                with self.assertRaisesRegex(
                    StageEditContextualBeamQAAdapterV2Error,
                    "duration_seconds differs from the bound render plan",
                ):
                    derive_stage_edit_contextual_beam_qa_case_v2(
                        semimarkov=fixture.semimarkov,
                        selected_timeline_rank=1,
                        post_render_report=fixture.report,
                        decode_report_path=fixture.decode_path,
                        post_render_report_path=fixture.report_path,
                    )

    def test_forged_motion_counts_cannot_override_typed_decisions(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            hook = fixture.report.metrics["stage_edit_contextual_beam_qa_v2"]
            self.assertIsInstance(hook, dict)
            hook["closeup_motion_switch_boundary_indices"] = [0]  # type: ignore[index]
            fixture.report.write_atomic(fixture.report_path)
            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2.evaluate_post_render_qa",
                return_value=fixture.report,
            ):
                with self.assertRaisesRegex(
                    StageEditContextualBeamQAAdapterV2Error,
                    "closeup_motion_switch_boundary_indices differs",
                ):
                    derive_stage_edit_contextual_beam_qa_case_v2(
                        semimarkov=fixture.semimarkov,
                        selected_timeline_rank=1,
                        post_render_report=fixture.report,
                        decode_report_path=fixture.decode_path,
                        post_render_report_path=fixture.report_path,
                    )

    def test_post_render_report_must_reverify_exactly(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            different = PostRenderQAReport(
                artifact_set=fixture.artifacts,
                verdict=PostRenderQAVerdict.FAIL,
                metrics=fixture.report.metrics,
                issues=(),
                policy=fixture.report.policy,
                input_artifact_fingerprints=fixture.report.input_artifact_fingerprints,
                created_at=fixture.report.created_at,
            )
            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2.evaluate_post_render_qa",
                return_value=different,
            ):
                with self.assertRaisesRegex(
                    StageEditContextualBeamQAAdapterV2Error,
                    "independent re-evaluation",
                ):
                    derive_stage_edit_contextual_beam_qa_case_v2(
                        semimarkov=fixture.semimarkov,
                        selected_timeline_rank=1,
                        post_render_report=fixture.report,
                        decode_report_path=fixture.decode_path,
                        post_render_report_path=fixture.report_path,
                    )

    def test_revalidation_allows_only_pipeline_snapshot_video_path_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            snapshot_video = (
                Path(raw_root)
                / ".qa-snapshot"
                / fixture.video_path.name
            )
            reported_metrics = deepcopy(dict(fixture.report.metrics))
            reported_metrics["tool_checks"]["full_decode"]["command"] = [
                "ffmpeg",
                "-i",
                str(snapshot_video),
            ]
            reported_metrics["perceptual_quality"] = {
                "metrics": {"video": {"path": str(snapshot_video)}}
            }
            reported = replace(fixture.report, metrics=reported_metrics)
            reported.write_atomic(fixture.report_path)

            recomputed_metrics = deepcopy(reported_metrics)
            recomputed_metrics["tool_checks"]["full_decode"]["command"][2] = (
                str(fixture.video_path)
            )
            recomputed_metrics["perceptual_quality"]["metrics"]["video"][
                "path"
            ] = str(fixture.video_path)
            recomputed = replace(reported, metrics=recomputed_metrics)
            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2.evaluate_post_render_qa",
                return_value=recomputed,
            ):
                verified = derive_stage_edit_contextual_beam_qa_case_v2(
                    semimarkov=fixture.semimarkov,
                    selected_timeline_rank=1,
                    post_render_report=reported,
                    decode_report_path=fixture.decode_path,
                    post_render_report_path=fixture.report_path,
                )
            self.assertTrue(is_verified_stage_edit_contextual_beam_qa_case_v2(verified))

    def test_registry_manifest_rehydration_replays_and_issues_fresh_seal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
                "evaluate_post_render_qa",
                return_value=fixture.report,
            ):
                original = derive_stage_edit_contextual_beam_qa_case_v2(
                    semimarkov=fixture.semimarkov,
                    selected_timeline_rank=1,
                    post_render_report=fixture.report,
                    decode_report_path=fixture.decode_path,
                    post_render_report_path=fixture.report_path,
                )
                rehydrated = rehydrate_stage_edit_contextual_beam_qa_case_v2(
                    case_payload=original.case.to_dict(),
                    derivation_payload=original.derivation.to_dict(),
                )
            self.assertIsNot(original, rehydrated)
            self.assertTrue(
                is_verified_stage_edit_contextual_beam_qa_case_v2(rehydrated)
            )
            self.assertEqual(original.case, rehydrated.case)
            self.assertEqual(original.derivation, rehydrated.derivation)
            self.assertEqual(
                original.case,
                require_verified_stage_edit_contextual_beam_qa_case_v2(
                    rehydrated
                ),
            )

    def test_evaluation_case_binds_checkpoint_scorer_and_runtime_replay(
        self,
    ) -> None:
        from tests.test_stage_edit_source_pair_contextual_v2 import _corpus

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            root = Path(raw_root).resolve()
            contextual_directory = root / "contextual"
            contextual_directory.mkdir()
            corpus = _corpus()
            training = train_stage_edit_contextual_source_pair_v2(corpus)
            checkpoint = new_stage_edit_contextual_source_pair_checkpoint_v2(
                "beam-qa-evaluation-binding",
                training,
                created_at="2026-08-25T12:00:00+09:00",
            )
            corpus_path = save_stage_edit_contextual_training_corpus_v2(
                contextual_directory, corpus
            )
            checkpoint_path = save_stage_edit_contextual_source_pair_checkpoint_v2(
                contextual_directory, checkpoint
            )
            evaluation_bundle = (
                load_stage_edit_contextual_source_pair_evaluation_bundle_v2(
                    contextual_directory,
                    expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
                    expected_checkpoint_file_sha256=hashlib.sha256(
                        checkpoint_path.read_bytes()
                    ).hexdigest(),
                    expected_corpus_sha256=corpus.content_sha256,
                    expected_corpus_file_sha256=hashlib.sha256(
                        corpus_path.read_bytes()
                    ).hexdigest(),
                    expected_token_schema_fingerprint=(
                        corpus.token_schema_fingerprint
                    ),
                    expected_scorer_fingerprint=checkpoint.scorer_fingerprint,
                    expected_runtime_equivalence_sha256=(
                        checkpoint.runtime_equivalence_sha256
                    ),
                )
            )

            qa_directory = root / "qa"
            qa_directory.mkdir()
            fixture = _Fixture(qa_directory)
            scorer_provenance = {
                "authority": (
                    CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
                ),
                "scorer_contract": evaluation_bundle.scorer.contract,
                "scorer_fingerprint": evaluation_bundle.scorer.fingerprint,
            }
            audit = replace(
                fixture.semimarkov.audit,
                config={
                    **fixture.semimarkov.audit.config,
                    "continuity_authority": (
                        CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
                    ),
                    "continuity_scorer_contract": (
                        evaluation_bundle.scorer.contract
                    ),
                    "continuity_scorer_fingerprint": (
                        evaluation_bundle.scorer.fingerprint
                    ),
                },
                score_provenance={"boundary_continuity": scorer_provenance},
            )
            fixture.semimarkov = replace(fixture.semimarkov, audit=audit)
            semimarkov_payload = fixture.semimarkov.to_dict()
            fixture.decode_path.write_text(
                json.dumps({"semimarkov": semimarkov_payload}, indent=2),
                encoding="utf-8",
            )
            analysis = json.loads(fixture.analysis_path.read_text(encoding="utf-8"))
            analysis["stage_edit_generation"]["semimarkov"] = semimarkov_payload
            fixture.analysis_path.write_text(
                json.dumps(analysis, indent=2), encoding="utf-8"
            )
            fingerprints = fingerprint_post_render_artifacts(fixture.artifacts)
            metrics = deepcopy(dict(fixture.report.metrics))
            metrics["artifact_content_stability"]["final_fingerprints"] = (
                fingerprints
            )
            hook = metrics["stage_edit_contextual_beam_qa_v2"]
            hook["decode_result_content_sha256"] = _canonical_sha256(
                semimarkov_payload
            )
            hook["decoder_input_sha256"] = audit.input_sha256
            report = replace(
                fixture.report,
                metrics=metrics,
                input_artifact_fingerprints=fingerprints,
            )
            report.write_atomic(fixture.report_path)

            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
                "evaluate_post_render_qa",
                return_value=report,
            ):
                verified = derive_stage_edit_contextual_beam_qa_case_v2(
                    semimarkov=fixture.semimarkov,
                    selected_timeline_rank=1,
                    post_render_report=report,
                    decode_report_path=fixture.decode_path,
                    post_render_report_path=fixture.report_path,
                    contextual_evaluation_bundle_v2=evaluation_bundle,
                )
                rehydrated = rehydrate_stage_edit_contextual_beam_qa_case_v2(
                    case_payload=verified.case.to_dict(),
                    derivation_payload=verified.derivation.to_dict(),
                )

            expected_kwargs = {
                "expected_checkpoint_sha256": checkpoint.checkpoint_sha256,
                "expected_corpus_sha256": corpus.content_sha256,
                "expected_scorer_fingerprint": checkpoint.scorer_fingerprint,
                "expected_runtime_equivalence_sha256": (
                    checkpoint.runtime_equivalence_sha256
                ),
                "expected_token_schema_fingerprint": (
                    checkpoint.token_schema_fingerprint
                ),
            }
            self.assertEqual(verified.derivation, rehydrated.derivation)
            self.assertEqual(
                checkpoint.scorer_fingerprint,
                rehydrated.derivation.decode_scorer_fingerprint,
            )
            self.assertEqual(
                verified.case,
                require_verified_stage_edit_contextual_beam_qa_case_v2(
                    rehydrated, **expected_kwargs
                ),
            )
            with self.assertRaisesRegex(
                StageEditContextualBeamQAAdapterV2Error,
                "decode scorer/runtime identity differs",
            ):
                require_verified_stage_edit_contextual_beam_qa_case_v2(
                    rehydrated,
                    **{
                        **expected_kwargs,
                        "expected_checkpoint_sha256": "f" * 64,
                    },
                )

    def test_registry_manifest_rehydration_rejects_self_consistent_raw_case(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
                "evaluate_post_render_qa",
                return_value=fixture.report,
            ):
                original = derive_stage_edit_contextual_beam_qa_case_v2(
                    semimarkov=fixture.semimarkov,
                    selected_timeline_rank=1,
                    post_render_report=fixture.report,
                    decode_report_path=fixture.decode_path,
                    post_render_report_path=fixture.report_path,
                )
                forged = replace(
                    original.case,
                    quality_score_p10=0.99,
                    case_sha256=None,
                )
                with self.assertRaisesRegex(
                    StageEditContextualBeamQAAdapterV2Error,
                    "full artifact replay",
                ):
                    rehydrate_stage_edit_contextual_beam_qa_case_v2(
                        case_payload=forged.to_dict(),
                        derivation_payload=original.derivation.to_dict(),
                    )

    def test_registry_manifest_rehydration_rejects_bound_artifact_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            fixture = _Fixture(Path(raw_root))
            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
                "evaluate_post_render_qa",
                return_value=fixture.report,
            ):
                original = derive_stage_edit_contextual_beam_qa_case_v2(
                    semimarkov=fixture.semimarkov,
                    selected_timeline_rank=1,
                    post_render_report=fixture.report,
                    decode_report_path=fixture.decode_path,
                    post_render_report_path=fixture.report_path,
                )
            fixture.plan_path.write_bytes(fixture.plan_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                StageEditContextualBeamQAAdapterV2Error,
                "identity changed",
            ):
                rehydrate_stage_edit_contextual_beam_qa_case_v2(
                    case_payload=original.case.to_dict(),
                    derivation_payload=original.derivation.to_dict(),
                )

    def test_registry_manifest_rehydration_rejects_hardlinked_video_input(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            root = Path(raw_root)
            fixture = _Fixture(root)
            with mock.patch(
                "crosscut.stage_edit_contextual_beam_qa_adapter_v2."
                "evaluate_post_render_qa",
                return_value=fixture.report,
            ):
                original = derive_stage_edit_contextual_beam_qa_case_v2(
                    semimarkov=fixture.semimarkov,
                    selected_timeline_rank=1,
                    post_render_report=fixture.report,
                    decode_report_path=fixture.decode_path,
                    post_render_report_path=fixture.report_path,
                )
            linked = root / "linked-video.mp4"
            os.link(fixture.video_path, linked)
            with self.assertRaisesRegex(
                StageEditContextualBeamQAAdapterV2Error,
                "regular single-link",
            ):
                rehydrate_stage_edit_contextual_beam_qa_case_v2(
                    case_payload=original.case.to_dict(),
                    derivation_payload=original.derivation.to_dict(),
                )


if __name__ == "__main__":
    unittest.main()
