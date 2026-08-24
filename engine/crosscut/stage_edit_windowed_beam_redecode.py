"""Production adapter for reviewer-window StageEdit semi-Markov re-decode.

The generic repair loop deliberately accepts a callback.  This module is the
concrete callback used by the product pipeline: it slices the already verified
Transformer head frames to one reviewer issue window, invokes the public
semi-Markov decoder again, reprojects every proposal through the exact source
clock, and rejects proposals that would break full-edit source/cadence policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .render import RenderTransition, aligned_render_segment
from .section_director import SectionPlan
from .stage_edit import StageBoundary, StageShot
from .stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from .stage_edit_pipeline_bridge import StageEditPipelineBridgeResult
from .stage_edit_repair_loop import (
    StageEditRepairProposal,
    StageEditRepairRequest,
)
from .stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    StageEditHeadFrame,
    StageEditNoValidTimelineError,
    StageEditSemiMarkovConfig,
    StageEditSourceMetadata,
    decode_stage_edit_semimarkov_beam,
    stage_edit_boundary_continuity_score,
)
from .stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
    StageEditSourcePairScorer,
)
from .timewarp import TimeWarpMap


STAGE_EDIT_WINDOWED_BEAM_CONTRACT = (
    "reviewer_issue_window_actual_semimarkov_redecode_v1"
)
_TIME_EPSILON = 1e-6


class StageEditWindowedBeamError(RuntimeError):
    """A configured local learned re-decode could not be honored safely."""


def _clipped_section_plan(
    section_plan: SectionPlan,
    *,
    start_seconds: float,
    end_seconds: float,
) -> SectionPlan:
    sections = tuple(
        replace(
            section,
            section_index=index,
            start_seconds=max(start_seconds, section.start_seconds),
            end_seconds=min(end_seconds, section.end_seconds),
        )
        for index, section in enumerate(
            section
            for section in section_plan.sections
            if section.end_seconds > start_seconds + _TIME_EPSILON
            and section.start_seconds < end_seconds - _TIME_EPSILON
        )
    )
    if not sections:
        raise StageEditWindowedBeamError(
            "reviewer issue window does not overlap the section plan"
        )
    return SectionPlan(
        sections,
        analysis_basis=(
            f"{section_plan.analysis_basis}+{STAGE_EDIT_WINDOWED_BEAM_CONTRACT}"
        ),
        role_semantics=section_plan.role_semantics,
    )


def _maximum_switch_count(rate: float, duration: float) -> int:
    return max(1, int(math.ceil(rate * duration / 60.0 - 1e-12)))


def _resolved_continuity_identity(
    config: StageEditSemiMarkovConfig,
    source_pair_scorer: StageEditSourcePairScorer | None,
) -> dict[str, Any]:
    """Resolve the exact scorer used by both local and exterior boundaries."""

    authority = config.continuity_authority
    if authority == HANDCRAFTED_CONTINUITY_AUTHORITY:
        if source_pair_scorer is not None:
            raise StageEditWindowedBeamError(
                "handcrafted exterior continuity rejects a learned scorer"
            )
        contract = HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
        fingerprint = HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
    elif authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
        if not isinstance(source_pair_scorer, StageEditSourcePairScorer):
            raise StageEditWindowedBeamError(
                "learned exterior continuity requires the production source-pair scorer"
            )
        contract = source_pair_scorer.contract
        fingerprint = source_pair_scorer.fingerprint
        if contract != STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT:
            raise StageEditWindowedBeamError(
                "learned exterior continuity scorer contract changed"
            )
    else:
        raise StageEditWindowedBeamError(
            "windowed continuity authority is unsupported"
        )
    if (
        config.continuity_scorer_contract is not None
        and config.continuity_scorer_contract != contract
    ):
        raise StageEditWindowedBeamError(
            "windowed continuity scorer contract differs from production"
        )
    if (
        config.continuity_scorer_fingerprint is not None
        and config.continuity_scorer_fingerprint != fingerprint
    ):
        raise StageEditWindowedBeamError(
            "windowed continuity scorer fingerprint differs from production"
        )
    return {
        "continuity_authority": authority,
        "continuity_scorer_contract": contract,
        "continuity_scorer_fingerprint": fingerprint,
        "learned_source_pair_scorer_bound": source_pair_scorer is not None,
    }


def _validate_decoder_continuity_provenance(
    provenance: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    expected = {
        "authority": identity["continuity_authority"],
        "scorer_contract": identity["continuity_scorer_contract"],
        "scorer_fingerprint": identity["continuity_scorer_fingerprint"],
    }
    if any(provenance.get(name) != value for name, value in expected.items()):
        raise StageEditWindowedBeamError(
            "local decoder continuity provenance differs from its resolved scorer"
        )


def _exact_head_frame(
    frames: Sequence[StageEditHeadFrame],
    boundary_time_seconds: float,
) -> StageEditHeadFrame:
    matches = tuple(
        frame
        for frame in frames
        if abs(frame.master_time_seconds - boundary_time_seconds) <= _TIME_EPSILON
    )
    if len(matches) != 1:
        raise StageEditWindowedBeamError(
            "exterior splice boundary lacks one exact production head frame"
        )
    return matches[0]


def _exterior_splice_continuity_evidence(
    request: StageEditRepairRequest,
    replacement_shots: Sequence[StageShot],
    *,
    config: StageEditSemiMarkovConfig,
    derivative_family_ids: Sequence[int],
    all_frames: Sequence[StageEditHeadFrame],
    source_pair_scorer: StageEditSourcePairScorer | None,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Score every newly created fixed-context edge on the exact master clock."""

    source_count = len(request.base_edit.sources)
    if len(derivative_family_ids) != source_count:
        raise StageEditWindowedBeamError(
            "exterior continuity derivative families do not align with sources"
        )
    if not replacement_shots:
        raise StageEditWindowedBeamError(
            "exterior continuity needs at least one replacement shot"
        )
    edge_inputs: list[tuple[str, float, int, int]] = []
    if request.window.start_shot > 0:
        edge_inputs.append(
            (
                "left",
                request.window.actual_start_seconds,
                request.base_edit.shots[
                    request.window.start_shot - 1
                ].segment.source_index,
                replacement_shots[0].segment.source_index,
            )
        )
    if request.window.end_shot_exclusive < len(request.base_edit.shots):
        edge_inputs.append(
            (
                "right",
                request.window.actual_end_seconds,
                replacement_shots[-1].segment.source_index,
                request.base_edit.shots[
                    request.window.end_shot_exclusive
                ].segment.source_index,
            )
        )

    edges: list[dict[str, Any]] = []
    for side, boundary_time, outgoing_source, incoming_source in edge_inputs:
        frame = _exact_head_frame(all_frames, boundary_time)
        outgoing_family = derivative_family_ids[outgoing_source]
        incoming_family = derivative_family_ids[incoming_source]
        cross_family = outgoing_family != incoming_family
        scorer_invoked = False
        neutral_reason: str | None = None
        if not cross_family:
            raw_score = 0.0
            neutral_reason = "same_derivative_family_neutral"
        elif not frame.source_continuity:
            if (
                identity["continuity_authority"]
                == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
            ):
                raise StageEditWindowedBeamError(
                    "learned exterior continuity is missing exact observations"
                )
            raw_score = 0.0
            neutral_reason = "handcrafted_missing_observation_neutral"
        else:
            if len(frame.source_continuity) != source_count:
                raise StageEditWindowedBeamError(
                    "exterior continuity observations do not align with sources"
                )
            outgoing = frame.source_continuity[outgoing_source]
            incoming = frame.source_continuity[incoming_source]
            if (
                identity["continuity_authority"]
                == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
            ):
                assert isinstance(source_pair_scorer, StageEditSourcePairScorer)
                raw_score = source_pair_scorer.score(outgoing, incoming)
            else:
                raw_score = stage_edit_boundary_continuity_score(
                    outgoing,
                    incoming,
                )
            scorer_invoked = True
        raw_score = float(raw_score)
        if not math.isfinite(raw_score) or not -1.0 <= raw_score <= 0.0:
            raise StageEditWindowedBeamError(
                "exterior continuity scorer violated its bounded [-1, 0] contract"
            )
        weight = float(config.boundary_continuity_weight)
        weighted_score = raw_score * weight
        if (
            not math.isfinite(weighted_score)
            or weighted_score < -weight - _TIME_EPSILON
            or weighted_score > _TIME_EPSILON
        ):
            raise StageEditWindowedBeamError(
                "weighted exterior continuity score is outside its bounded range"
            )
        edges.append(
            {
                "side": side,
                "boundary_time_seconds": boundary_time,
                "head_frame_time_seconds": frame.master_time_seconds,
                "outgoing_source_index": outgoing_source,
                "incoming_source_index": incoming_source,
                "outgoing_derivative_family_id": outgoing_family,
                "incoming_derivative_family_id": incoming_family,
                "cross_derivative_family": cross_family,
                "authority_rule_evaluated": True,
                "scorer_evaluated": scorer_invoked,
                "scorer_invoked": scorer_invoked,
                "neutral_reason": neutral_reason,
                "raw_score": raw_score,
                "weight": weight,
                "weighted_score": weighted_score,
                **dict(identity),
            }
        )
    expected_edge_count = len(edge_inputs)
    return {
        "expected_edge_count": expected_edge_count,
        "evaluated_edge_count": len(edges),
        "scorer_invocation_count": sum(
            bool(edge["scorer_invoked"]) for edge in edges
        ),
        "raw_score_total": sum(float(edge["raw_score"]) for edge in edges),
        "weighted_score_total": sum(
            float(edge["weighted_score"]) for edge in edges
        ),
        "raw_score_range": [-1.0, 0.0],
        "full_edit_validation_passed": len(edges) == expected_edge_count,
        "edges": edges,
        **dict(identity),
    }


def _full_edit_constraint_rejections(
    request: StageEditRepairRequest,
    replacement_shots: Sequence[StageShot],
    *,
    config: StageEditSemiMarkovConfig,
    derivative_family_ids: Sequence[int],
) -> tuple[str, ...]:
    edit = request.base_edit
    shots = (
        edit.shots[: request.window.start_shot]
        + tuple(replacement_shots)
        + edit.shots[request.window.end_shot_exclusive :]
    )
    reasons: set[str] = set()
    source_path = tuple(shot.segment.source_index for shot in shots)
    if len(derivative_family_ids) != len(edit.sources):
        raise StageEditWindowedBeamError(
            "full-edit derivative family metadata does not match sources"
        )
    family_path = tuple(derivative_family_ids[index] for index in source_path)
    if any(left == right for left, right in zip(family_path, family_path[1:])):
        reasons.add("full_edit_consecutive_same_derivative_family")
    if config.enforce_aba_exclusion and any(
        left == right and left != middle
        for left, middle, right in zip(
            family_path,
            family_path[1:],
            family_path[2:],
        )
    ):
        reasons.add("full_edit_aba_derivative_family_return")
    if len(set(family_path)) < config.minimum_unique_sources:
        reasons.add("full_edit_derivative_family_diversity")

    duration = edit.duration_seconds
    occupancy = [0.0] * len(edit.sources)
    for shot in shots:
        occupancy[shot.segment.source_index] += shot.segment.duration
    family_occupancy = [0.0] * len(set(derivative_family_ids))
    for source_index, value in enumerate(occupancy):
        family_occupancy[derivative_family_ids[source_index]] += value
    if any(
        value > config.maximum_source_occupancy_fraction * duration + _TIME_EPSILON
        for value in family_occupancy
    ):
        reasons.add("full_edit_derivative_family_occupancy_cap")

    boundary_times = tuple(
        edit.master_audio_start + shot.segment.master_start
        for shot in shots[1:]
    )
    if len(boundary_times) > _maximum_switch_count(
        config.maximum_global_switches_per_minute,
        duration,
    ):
        reasons.add("full_edit_global_cut_density")
    for section in edit.section_plan.sections:
        section_count = sum(
            section.start_seconds - _TIME_EPSILON
            <= boundary
            < section.end_seconds - _TIME_EPSILON
            for boundary in boundary_times
        )
        section_rate = min(
            config.maximum_global_switches_per_minute,
            section.target_source_switches_per_minute
            * config.section_cut_density_multiplier,
        )
        if section_count > _maximum_switch_count(
            section_rate,
            section.duration_seconds,
        ):
            reasons.add(f"full_edit_section_{section.section_index}_cut_density")
    rolling_limit = _maximum_switch_count(
        config.maximum_global_switches_per_minute,
        config.cut_density_window_seconds,
    )
    if any(
        sum(
            left - config.cut_density_window_seconds <= right <= left
            for right in boundary_times
        )
        > rolling_limit
        for left in boundary_times
    ):
        reasons.add("full_edit_rolling_cut_density")
    return tuple(sorted(reasons))


@dataclass(slots=True)
class StageEditWindowedBeamRedecoder:
    """Callable actual-beam adapter consumed by ``repair_stage_edit_candidates``."""

    bridge_result: StageEditPipelineBridgeResult
    sources: tuple[Path, ...]
    source_durations_seconds: tuple[float, ...]
    source_start_offsets_seconds: tuple[float, ...]
    internal_cut_times_seconds: tuple[tuple[float, ...], ...]
    source_timewarps: tuple[TimeWarpMap | None, ...]
    source_quality_log_priors: tuple[float, ...]
    output: Path
    call_audits: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bridge_result, StageEditPipelineBridgeResult):
            raise TypeError("bridge_result must be a StageEditPipelineBridgeResult")
        source_count = len(self.bridge_result.preflight.source_ids)
        self.sources = tuple(Path(value).expanduser().resolve() for value in self.sources)
        self.source_durations_seconds = tuple(
            float(value) for value in self.source_durations_seconds
        )
        self.source_start_offsets_seconds = tuple(
            float(value) for value in self.source_start_offsets_seconds
        )
        self.internal_cut_times_seconds = tuple(
            tuple(sorted(float(item) for item in values))
            for values in self.internal_cut_times_seconds
        )
        self.source_timewarps = tuple(self.source_timewarps)
        self.source_quality_log_priors = tuple(
            float(value) for value in self.source_quality_log_priors
        )
        arrays = (
            self.sources,
            self.source_durations_seconds,
            self.source_start_offsets_seconds,
            self.internal_cut_times_seconds,
            self.source_timewarps,
            self.source_quality_log_priors,
        )
        if any(len(values) != source_count for values in arrays):
            raise StageEditWindowedBeamError(
                "windowed beam source inputs must align with the production snapshot"
            )
        retained_sources = tuple(
            Path(value).expanduser().resolve()
            for value in self.bridge_result.retained_candidates[0].edit.sources
        )
        if tuple(self.sources) != retained_sources:
            raise StageEditWindowedBeamError(
                "windowed beam media identity differs from retained StageEdit"
            )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.source_durations_seconds
        ):
            raise StageEditWindowedBeamError(
                "windowed beam source durations must be positive and finite"
            )
        if any(
            not math.isfinite(value)
            for value in self.source_start_offsets_seconds
            + self.source_quality_log_priors
        ):
            raise StageEditWindowedBeamError(
                "windowed beam source scalars must be finite"
            )
        if any(
            value is not None and not isinstance(value, TimeWarpMap)
            for value in self.source_timewarps
        ):
            raise StageEditWindowedBeamError(
                "windowed beam clocks must be checked TimeWarpMap values"
            )
        self.output = Path(self.output).expanduser().resolve()

    @property
    def canonical_boundaries_seconds(self) -> tuple[float, ...]:
        frames = self.bridge_result.head_materialization.frames
        return tuple(frame.master_time_seconds for frame in frames) + (
            self.bridge_result.semimarkov.timelines[0].edit.section_plan.end_seconds,
        )

    @property
    def production_config(self) -> StageEditSemiMarkovConfig:
        return StageEditSemiMarkovConfig(
            **dict(self.bridge_result.semimarkov.audit.config)
        )

    def _metadata(self) -> tuple[StageEditSourceMetadata, ...]:
        return tuple(
            StageEditSourceMetadata(
                source_index=index,
                source_id=source_id,
                source_duration_seconds=self.source_durations_seconds[index],
                source_start_offset_seconds=(
                    self.source_start_offsets_seconds[index]
                ),
                internal_cut_times_seconds=(
                    self.internal_cut_times_seconds[index]
                ),
                # Local occupancy is context-free.  The exact production cap
                # is rechecked on the spliced full edit below.
                maximum_occupancy_fraction=1.0,
                quality_log_prior=self.source_quality_log_priors[index],
            )
            for index, source_id in enumerate(
                self.bridge_result.preflight.source_ids
            )
        )

    def _exact_shots(
        self,
        request: StageEditRepairRequest,
        timeline: Any,
    ) -> tuple[StageShot, ...]:
        base = request.base_edit
        shots: list[StageShot] = []
        for local_index, decision in enumerate(timeline.decisions):
            source_index = decision.source_index
            segment = aligned_render_segment(
                source_index=source_index,
                reference_start=decision.master_start_seconds,
                common_master_start=base.master_audio_start,
                source_offset=self.source_start_offsets_seconds[source_index],
                duration=decision.duration_seconds,
                editorial_transform=decision.editorial_transform,
                timewarp=self.source_timewarps[source_index],
            )
            if (
                segment.source_start < -_TIME_EPSILON
                or segment.source_end
                > self.source_durations_seconds[source_index] + _TIME_EPSILON
            ):
                raise StageEditWindowedBeamError(
                    "windowed beam proposal leaves exact source-clock bounds"
                )
            margin = self.production_config.internal_cut_exclusion_margin_seconds
            if any(
                segment.source_start + margin + _TIME_EPSILON
                < cut
                < segment.source_end - margin - _TIME_EPSILON
                for cut in self.internal_cut_times_seconds[source_index]
            ):
                raise StageEditWindowedBeamError(
                    "windowed beam proposal crosses an exact native camera cut"
                )
            midpoint = decision.master_start_seconds + decision.duration_seconds / 2.0
            section_index = base.section_plan.directive_at(midpoint).section_index
            shots.append(
                StageShot(
                    shot_index=request.window.start_shot + local_index,
                    section_index=section_index,
                    segment=segment,
                    decision_id=(
                        f"stageedit-window-beam-i{request.iteration:02d}-"
                        f"r{timeline.rank:02d}-s{local_index:03d}"
                    ),
                    decision_confidence=decision.decision_confidence,
                    section_role=decision.section_role,
                    section_progress=decision.section_progress,
                    phrase_progress=decision.phrase_progress,
                )
            )
        return tuple(shots)

    def _boundary_patch(
        self,
        request: StageEditRepairRequest,
        timeline: Any,
    ) -> tuple[StageBoundary, ...]:
        transitions: list[RenderTransition] = []
        if request.window.start_shot > 0:
            transitions.append(
                RenderTransition(
                    reason="stage_edit_windowed_beam_left_context_hard_cut",
                    tier="C",
                )
            )
        transitions.extend(
            boundary.transition for boundary in timeline.edit.boundaries
        )
        if request.window.end_shot_exclusive < len(request.base_edit.shots):
            transitions.append(
                RenderTransition(
                    reason="stage_edit_windowed_beam_right_context_hard_cut",
                    tier="C",
                )
            )
        return tuple(
            StageBoundary(
                boundary_index=index,
                transition=transition,
                reviewer_status="unreviewed",
            )
            for index, transition in enumerate(transitions)
        )

    def __call__(
        self,
        request: StageEditRepairRequest,
    ) -> tuple[StageEditRepairProposal, ...]:
        if not isinstance(request, StageEditRepairRequest):
            raise TypeError("windowed beam callback needs StageEditRepairRequest")
        start = request.window.actual_start_seconds
        end = request.window.actual_end_seconds
        all_frames = self.bridge_result.head_materialization.frames
        frames = tuple(
            frame
            for frame in all_frames
            if start - _TIME_EPSILON
            <= frame.master_time_seconds
            < end - _TIME_EPSILON
        )
        if not frames or abs(frames[0].master_time_seconds - start) > _TIME_EPSILON:
            raise StageEditWindowedBeamError(
                "repair window start is not an exact production head-frame time"
            )
        section_plan = _clipped_section_plan(
            request.base_edit.section_plan,
            start_seconds=start,
            end_seconds=end,
        )
        production_config = self.production_config
        local_config = replace(
            production_config,
            beam_width=32,
            timeline_count=32,
            minimum_unique_sources=1,
            maximum_source_occupancy_fraction=1.0,
            montage_grammar_score_weight=0.0,
            enforce_montage_grammar=False,
        )
        source_pair_scorer = getattr(
            self.bridge_result.preflight, "source_pair_scorer", None
        )
        continuity_identity = _resolved_continuity_identity(
            local_config,
            source_pair_scorer,
        )
        declared_authority = getattr(
            self.bridge_result.preflight,
            "continuity_authority",
            continuity_identity["continuity_authority"],
        )
        if declared_authority != continuity_identity["continuity_authority"]:
            raise StageEditWindowedBeamError(
                "windowed continuity authority differs from production preflight"
            )
        declared_fingerprint = getattr(
            self.bridge_result.preflight,
            "source_pair_scorer_fingerprint",
            None,
        )
        if declared_fingerprint is None and (
            continuity_identity["continuity_authority"]
            == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
        ):
            raise StageEditWindowedBeamError(
                "learned windowed continuity preflight lacks its scorer fingerprint"
            )
        if declared_fingerprint is not None and declared_fingerprint != (
            continuity_identity["continuity_scorer_fingerprint"]
        ):
            raise StageEditWindowedBeamError(
                "windowed continuity fingerprint differs from production preflight"
            )
        call_audit: dict[str, Any] = {
            "contract": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
            "iteration": request.iteration,
            "window": request.window.to_dict(),
            "checkpoint_directory": str(
                self.bridge_result.preflight.checkpoint_directory
            ),
            "head_frame_count": len(frames),
            "head_frame_times_seconds": [
                frame.master_time_seconds for frame in frames
            ],
            "local_config": asdict(local_config),
            "dynamic_crop_version": getattr(
                self.bridge_result.preflight, "dynamic_crop_version", 2
            ),
            **continuity_identity,
            "local_only_relaxations": [
                "minimum_unique_sources_deferred_to_full_edit_recheck",
                "source_occupancy_deferred_to_full_edit_recheck",
                "section_montage_grammar_deferred_to_full_render_selection",
            ],
            "actual_public_decoder_invoked": True,
            "returned_timeline_count": 0,
            "accepted_proposal_count": 0,
            "rejections": [],
        }
        try:
            decoded = decode_stage_edit_semimarkov_beam(
                checkpoint_directory=(
                    self.bridge_result.preflight.checkpoint_directory
                ),
                sources=self.sources,
                source_metadata=self._metadata(),
                head_frames=frames,
                section_plan=section_plan,
                output=self.output.with_name(
                    f".{self.output.name}.window-{request.iteration:02d}.mp4"
                ),
                config=local_config,
                reference_source=request.base_edit.reference_source,
                aspect_ratio=request.base_edit.aspect_ratio,
                width=request.base_edit.width,
                height=request.base_edit.height,
                fps=request.base_edit.fps,
                source_derivative_metadata=(
                    self.bridge_result.source_derivative_diversity
                ),
                source_pair_scorer=source_pair_scorer,
            )
        except StageEditNoValidTimelineError as exc:
            call_audit["decoder_no_valid_timeline"] = str(exc)
            self.call_audits.append(call_audit)
            return ()
        call_audit["decoder_audit"] = decoded.audit.to_dict()
        decoder_score_provenance = getattr(decoded.audit, "score_provenance", None)
        decoder_continuity_provenance = (
            decoder_score_provenance.get("boundary_continuity")
            if isinstance(decoder_score_provenance, Mapping)
            else None
        )
        if not isinstance(decoder_continuity_provenance, Mapping):
            raise StageEditWindowedBeamError(
                "local decoder omitted continuity score provenance"
            )
        _validate_decoder_continuity_provenance(
            decoder_continuity_provenance,
            continuity_identity,
        )
        call_audit["returned_timeline_count"] = len(decoded.timelines)
        derivative_family_ids = (
            self.bridge_result.source_derivative_diversity.family_ids
            if self.bridge_result.source_derivative_diversity is not None
            else tuple(range(len(self.sources)))
        )
        eligible: list[tuple[float, int, Any, tuple[StageShot, ...], dict[str, Any]]] = []
        for timeline in decoded.timelines:
            try:
                timeline_provenance = getattr(
                    timeline,
                    "boundary_continuity_score_provenance",
                    None,
                )
                if not isinstance(timeline_provenance, Mapping):
                    raise StageEditWindowedBeamError(
                        "local timeline omitted continuity score provenance"
                    )
                _validate_decoder_continuity_provenance(
                    timeline_provenance,
                    continuity_identity,
                )
                shots = self._exact_shots(request, timeline)
                rejection_reasons = _full_edit_constraint_rejections(
                    request,
                    shots,
                    config=production_config,
                    derivative_family_ids=derivative_family_ids,
                )
                if rejection_reasons:
                    call_audit["rejections"].append(
                        {
                            "timeline_rank": timeline.rank,
                            "reasons": list(rejection_reasons),
                        }
                    )
                    continue
                exterior_evidence = _exterior_splice_continuity_evidence(
                    request,
                    shots,
                    config=production_config,
                    derivative_family_ids=derivative_family_ids,
                    all_frames=all_frames,
                    source_pair_scorer=source_pair_scorer,
                    identity=continuity_identity,
                )
                if not exterior_evidence["full_edit_validation_passed"]:
                    raise StageEditWindowedBeamError(
                        "exterior splice continuity full-edit validation failed"
                    )
                decoder_score = float(timeline.score)
                candidate_selection_score = decoder_score + float(
                    exterior_evidence["weighted_score_total"]
                )
                if not math.isfinite(candidate_selection_score):
                    raise StageEditWindowedBeamError(
                        "exterior-adjusted candidate selection score is not finite"
                    )
                eligible.append(
                    (
                        candidate_selection_score,
                        timeline.rank,
                        timeline,
                        shots,
                        exterior_evidence,
                    )
                )
            except (TypeError, ValueError, StageEditWindowedBeamError) as exc:
                call_audit["rejections"].append(
                    {
                        "timeline_rank": timeline.rank,
                        "reasons": [f"{type(exc).__name__}: {exc}"],
                    }
                )

        ordered = tuple(
            sorted(
                eligible,
                key=lambda value: (-value[0], value[1]),
            )[:8]
        )
        proposals: list[StageEditRepairProposal] = []
        for selection_rank, (
            candidate_selection_score,
            decoder_rank,
            timeline,
            shots,
            exterior_evidence,
        ) in enumerate(ordered, start=1):
            proposals.append(
                StageEditRepairProposal(
                    proposal_id=(
                        f"actual-windowed-semimarkov-i{request.iteration:02d}-"
                        f"r{decoder_rank:02d}"
                    ),
                    rank=selection_rank,
                    replacement_shots=shots,
                    replacement_boundaries=self._boundary_patch(
                        request,
                        timeline,
                    ),
                    audit={
                        "contract": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
                        "actual_public_decoder_invoked": True,
                        "decoder_input_sha256": decoded.audit.input_sha256,
                        "decoder_timeline_rank": decoder_rank,
                        "decoder_timeline_score": timeline.score,
                        "candidate_selection_rank": selection_rank,
                        "candidate_selection_score": candidate_selection_score,
                        "exterior_score_applied_to_candidate_selection": True,
                        "exterior_splice_continuity": exterior_evidence,
                        "exact_source_clock_reprojected": True,
                        "full_edit_constraints_rechecked": True,
                        "full_edit_exterior_continuity_rechecked": True,
                        **continuity_identity,
                    },
                )
            )
        call_audit["eligible_timeline_count"] = len(eligible)
        call_audit["candidate_selection_order"] = [
            {
                "candidate_selection_rank": proposal.rank,
                "proposal_id": proposal.proposal_id,
                "decoder_timeline_rank": proposal.audit["decoder_timeline_rank"],
                "candidate_selection_score": proposal.audit[
                    "candidate_selection_score"
                ],
                "exterior_weighted_score_total": proposal.audit[
                    "exterior_splice_continuity"
                ]["weighted_score_total"],
            }
            for proposal in proposals
        ]
        call_audit["accepted_proposal_count"] = len(proposals)
        self.call_audits.append(call_audit)
        return tuple(proposals)

    def audit_payload(self) -> dict[str, Any]:
        return {
            "contract": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
            "actual_public_decoder": (
                "crosscut.stage_edit_semimarkov_beam."
                "decode_stage_edit_semimarkov_beam"
            ),
            "call_count": len(self.call_audits),
            "calls": list(self.call_audits),
        }


__all__ = [
    "STAGE_EDIT_WINDOWED_BEAM_CONTRACT",
    "StageEditWindowedBeamError",
    "StageEditWindowedBeamRedecoder",
]
