"""A retained complete plan and one admission policy for optional editing work.

This controller is not a quality approver. It protects a validated render plan
from failed proposals; decoded technical and editorial QA still run afterward.
"""
from copy import deepcopy
from dataclasses import replace
from enum import Enum
from typing import Callable

from .render import RenderPlan, segment_frame_counts
from .review_first_runtime import (
    ReviewFirstDeadlineExceeded, ReviewFirstSearchCutoff, ReviewFirstRuntime,
    enforce_review_first_deadline,
)


class FirstResultStage(str, Enum):
    INPUT_VERIFIED = "INPUT_VERIFIED"
    BASELINE_PLAN_READY = "BASELINE_PLAN_READY"
    RENDERED = "RENDERED"
    TECHNICALLY_VERIFIED = "TECHNICALLY_VERIFIED"
    REVIEW_READY = "REVIEW_READY"
    APPROVED = "APPROVED"


class FirstResultController:
    def __init__(self, runtime: ReviewFirstRuntime, *, sources, source_durations,
                 master_start: float, duration: float, allowed_source_indices=None):
        self.runtime = runtime
        self.sources = tuple(sources)
        self.source_durations = tuple(source_durations)
        if len(self.sources) != len(self.source_durations):
            raise ValueError("verified source duration domain changed")
        self.master_start = master_start
        self.duration = duration
        self.allowed_source_indices = frozenset(range(len(self.sources)) if allowed_source_indices is None
                                                else allowed_source_indices)
        if not self.allowed_source_indices or not self.allowed_source_indices <= set(range(len(self.sources))):
            raise ValueError("first result source eligibility domain is invalid")
        self._render_configuration = None
        self.stage = FirstResultStage.INPUT_VERIFIED
        self.baseline = None
        self.incumbent = None
        self.attempts = []
        self.skipped_count = 0
        self._selected_authority = None
        self._history = [self.stage.value]

    def validate_plan(self, plan: RenderPlan) -> None:
        if not isinstance(plan, RenderPlan):
            raise TypeError("first result requires a complete RenderPlan")
        # Re-run geometry, transition and output-clock validation, not just a
        # serializable timeline or an optimistic quality flag.
        replace(plan)
        configuration = (plan.reference_source, plan.aspect_ratio, plan.width, plan.height, plan.fps)
        if self._render_configuration is not None and configuration != self._render_configuration:
            raise ValueError("candidate changed reference audio or output geometry/frame clock")
        if (tuple(plan.sources) != self.sources or abs(plan.master_audio_start-self.master_start) > 1e-6
                or abs(plan.duration-self.duration) > 1 / plan.fps
                or any(count <= 0 for count in segment_frame_counts(plan))):
            raise ValueError("candidate changed the verified source/interval/frame domain")
        for segment in plan.segments:
            if segment.source_index not in self.allowed_source_indices:
                raise ValueError("candidate uses a source outside verified eligibility")
            if segment.source_end > self.source_durations[segment.source_index] + 1e-6:
                raise ValueError("candidate source clock exceeds verified media duration")

    def install_baseline(self, value, *, plan: RenderPlan, authority: str):
        if self.baseline is not None:
            raise RuntimeError("baseline may only be installed once")
        self.validate_plan(plan)
        self._render_configuration = (plan.reference_source, plan.aspect_ratio, plan.width, plan.height, plan.fps)
        self.baseline = value
        self.incumbent = value
        self._selected_authority = authority
        self.stage = FirstResultStage.BASELINE_PLAN_READY
        self._history.append(self.stage.value)
        return value

    def run_optional(self, stage: str, operation: Callable, *, validate: Callable | None = None):
        if self.baseline is None:
            raise RuntimeError("optional work cannot precede the complete baseline")
        self.runtime.require_total_budget(stage)
        if not self.runtime.optional_refinement_allowed or len(self.attempts) >= 64:
            self.skipped_count += 1
            if len(self.attempts) < 64:
                self.attempts.append(dict(stage=stage, status="skipped_search_cutoff", elapsed_seconds=0.))
            return None
        started = self.runtime.elapsed_seconds
        row = dict(stage=stage, status="running")
        self.attempts.append(row)
        cutoff = min(self.runtime.deadline_monotonic_seconds,
            self.runtime.deadline_monotonic_seconds - self.runtime.total_budget_seconds
            + min(self.runtime.first_render_budget_seconds, self.runtime.refinement_stop_seconds))
        try:
            from .optional_native_worker import optional_native_scope
            with optional_native_scope(cutoff), enforce_review_first_deadline(cutoff, stage=stage):
                value = operation()
                self.runtime.require_first_render_budget(stage)
                if validate is not None:
                    validate(value)
            row["status"] = "validated_proposal" if value is not None else "unavailable"
            return value
        except ReviewFirstDeadlineExceeded:
            row["status"] = "deadline_exceeded"
            self.runtime.require_total_budget(stage)
            row["status"] = "search_cutoff"
        except ReviewFirstSearchCutoff:
            row["status"] = "search_cutoff"
        except Exception as exc:
            row.update(status="rejected", error_type=type(exc).__name__, error=str(exc)[:2000])
        finally:
            row["elapsed_seconds"] = max(0., self.runtime.elapsed_seconds-started)
        return None

    def promote(self, value, *, plan: RenderPlan, authority: str):
        self.validate_plan(plan)
        self.incumbent = value
        self._selected_authority = authority
        return value

    def render_complete(self):
        if self.stage is not FirstResultStage.BASELINE_PLAN_READY:
            raise RuntimeError("first result render completion is out of order")
        self.stage = FirstResultStage.RENDERED
        self._history.append(self.stage.value)

    def to_dict(self):
        return dict(contract="retained_complete_first_result_plan_v1", stage=self.stage.value,
            stage_history=list(self._history),
            baseline_retained=self.baseline is not None, selected_authority=self._selected_authority,
            optional_attempt_count=sum(r["status"] != "skipped_search_cutoff" for r in self.attempts),
            rejected_attempt_count=sum(r["status"] == "rejected" for r in self.attempts),
            skipped_count=self.skipped_count, attempts=deepcopy(self.attempts),
            quality_approved=False, decoded_qa_required=True)


def finalize_first_result_manifest(manifest, disposition):
    """Persist only the already content-bound final QA disposition, never infer it.

    Kept outside proposal promotion: render completion is not technical or
    editorial approval. The final GROUP/MEMBER QA owner calls this after all
    content binding and integrity classification checks.
    """
    from .review_first_runtime import ReviewFirstStatus
    locations = []
    group = manifest.get("review_first_visual_selection")
    if isinstance(group, dict):
        locations.append(group)
    render = manifest.get("render")
    if isinstance(render, dict) and isinstance(render.get("cadence"), dict):
        locations.append(render["cadence"])
    member = manifest.get("member_direct_full_frame")
    if isinstance(member, dict) and isinstance(member.get("cadence"), dict):
        locations.append(member["cadence"])
    for location in locations:
        payload = location.get("first_result_controller")
        if not isinstance(payload, dict) or payload.get("stage") != FirstResultStage.RENDERED.value:
            continue
        history = list(payload.get("stage_history", []))
        if disposition.status in (ReviewFirstStatus.REVIEW_READY, ReviewFirstStatus.APPROVED):
            history.append(FirstResultStage.TECHNICALLY_VERIFIED.value)
        history.append(disposition.status.value)
        payload.update(stage=disposition.status.value, stage_history=history,
            quality_approved=disposition.status is ReviewFirstStatus.APPROVED,
            decoded_qa_completed=True, video_output_exposed=disposition.video_output_exposed)
