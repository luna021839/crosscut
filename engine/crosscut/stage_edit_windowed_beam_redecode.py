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
from typing import Any, Sequence

from .render import RenderTransition, aligned_render_segment
from .section_director import SectionPlan
from .stage_edit import StageBoundary, StageShot
from .stage_edit_pipeline_bridge import StageEditPipelineBridgeResult
from .stage_edit_repair_loop import (
    StageEditRepairProposal,
    StageEditRepairRequest,
)
from .stage_edit_semimarkov_beam import (
    StageEditNoValidTimelineError,
    StageEditSemiMarkovConfig,
    StageEditSourceMetadata,
    decode_stage_edit_semimarkov_beam,
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
            "local_only_relaxations": [
                "minimum_unique_sources_deferred_to_full_edit_recheck",
                "source_occupancy_deferred_to_full_edit_recheck",
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
            )
        except StageEditNoValidTimelineError as exc:
            call_audit["decoder_no_valid_timeline"] = str(exc)
            self.call_audits.append(call_audit)
            return ()
        call_audit["decoder_audit"] = decoded.audit.to_dict()
        call_audit["returned_timeline_count"] = len(decoded.timelines)
        proposals: list[StageEditRepairProposal] = []
        for timeline in decoded.timelines:
            try:
                shots = self._exact_shots(request, timeline)
                rejection_reasons = _full_edit_constraint_rejections(
                    request,
                    shots,
                    config=production_config,
                    derivative_family_ids=(
                        self.bridge_result.source_derivative_diversity.family_ids
                        if self.bridge_result.source_derivative_diversity is not None
                        else tuple(range(len(self.sources)))
                    ),
                )
                if rejection_reasons:
                    call_audit["rejections"].append(
                        {
                            "timeline_rank": timeline.rank,
                            "reasons": list(rejection_reasons),
                        }
                    )
                    continue
                proposals.append(
                    StageEditRepairProposal(
                        proposal_id=(
                            f"actual-windowed-semimarkov-i{request.iteration:02d}-"
                            f"r{timeline.rank:02d}"
                        ),
                        rank=timeline.rank,
                        replacement_shots=shots,
                        replacement_boundaries=self._boundary_patch(
                            request,
                            timeline,
                        ),
                        audit={
                            "contract": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
                            "actual_public_decoder_invoked": True,
                            "decoder_input_sha256": decoded.audit.input_sha256,
                            "decoder_timeline_rank": timeline.rank,
                            "decoder_timeline_score": timeline.score,
                            "exact_source_clock_reprojected": True,
                            "full_edit_constraints_rechecked": True,
                        },
                    )
                )
                if len(proposals) == 8:
                    break
            except (TypeError, ValueError, StageEditWindowedBeamError) as exc:
                call_audit["rejections"].append(
                    {
                        "timeline_rank": timeline.rank,
                        "reasons": [f"{type(exc).__name__}: {exc}"],
                    }
                )
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
