"""Bounded look proposals are not decoded-output quality certificates."""
from .direct_member_look import DirectMemberLookPlan, build_direct_member_look_plan


class MemberLookProposalError(RuntimeError):
    """Color evidence errors must not be mislabeled as camera geometry errors."""


def predicted_member_look_failure_reasons(plan: DirectMemberLookPlan) -> tuple[str, ...]:
    reasons = []
    if (plan.rapid_boundary_count >= 4 and plan.predicted_high_shock_fraction is not None
            and plan.predicted_high_shock_fraction > plan.config.maximum_rapid_high_shock_fraction+1e-12):
        reasons.append('rapid_shock')
    if plan.source_span_policy_enforced:
        spans = dict(plan.predicted_source_spans)
        for axis, limit in (('sharpness_log', plan.config.source_sharpness_log_span),
                            ('luma', plan.config.source_luma_span),
                            ('chroma', plan.config.source_chroma_span)):
            if spans.get(axis) is not None and spans[axis] > limit+1e-12:
                reasons.append('source_'+axis+'_span')
    return tuple(reasons)


def build_member_look_proxy_proposal(*, visual_selection, render_segments,
                                   mandatory_decoded_proxy: bool) -> DirectMemberLookPlan:
    """Use the existing bounded solver, then require the actual proxy gate.

    Cropped output is not the original full-frame cache. A cached prediction
    may warn but cannot veto the mandatory encoded test or certify a pass.
    This call does not increase the existing full-proxy retry budget.
    """
    if mandatory_decoded_proxy is not True:
        raise ValueError('a look proposal requires mandatory decoded proxy QA')
    try:
        return build_direct_member_look_plan(
            visual_selection=visual_selection, render_segments=render_segments,
            require_predicted_pass=False,
        )
    except (ValueError, TypeError, FloatingPointError) as exc:
        raise MemberLookProposalError(f'멤버 색 보정 입력 검증 실패: {exc}') from exc
