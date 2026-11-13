"""One bounded, pixel-verified look pass on the retained GROUP render plan.

This changes no cuts, source clocks, geometry or quality thresholds. A missing
or rejected correction remains a no-op, not a reason to restart selection.
"""
from dataclasses import replace

from .boundary_look_match import (
    BoundaryLookMatchResult,
    analyze_boundary_look_match,
    verify_boundary_look_match_render_domain,
)
from .optional_native_worker import isolated_optional
from .render import RenderPlan


@isolated_optional()
def refine_review_first_boundary_look(
    plan: RenderPlan,
    *,
    group_16x9_safe: bool,
    maximum_verified_boundaries: int = 6,
    ffmpeg: str = "ffmpeg",
) -> tuple[RenderPlan, BoundaryLookMatchResult]:
    # Never silently erase a previous owner's admitted tracks. The ordinary
    # review-first path calls this exactly once, after its final geometry plan.
    if any(segment.boundary_look_track is not None for segment in plan.segments):
        raise ValueError("boundary look refinement requires an ungraded boundary plan")
    proposed = analyze_boundary_look_match(plan, group_16x9_safe=group_16x9_safe)
    admitted = verify_boundary_look_match_render_domain(
        plan, proposed, maximum_verified_boundaries=maximum_verified_boundaries,
        ffmpeg=ffmpeg, clipping_backoff_enabled=True,
    )
    if any(decision.applied and (
            decision.render_verification is None
            or not decision.render_verification.passed)
            for decision in admitted.decisions):
        raise ValueError("boundary look correction lacks matched decoded proof")
    adjusted = admitted.apply(plan)
    restored = replace(adjusted, segments=tuple(
        replace(segment, boundary_look_track=original.boundary_look_track)
        for segment, original in zip(adjusted.segments, plan.segments, strict=True)))
    if restored != plan:
        raise ValueError("boundary look refinement changed non-look render authority")
    return adjusted, admitted
