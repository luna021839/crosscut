"""Keep disjoint legacy boundary payloads across an optional GROUP edit.

This is a dependency-preservation check, not media sensing or a new quality
receipt. Declared output support expands to whole renderer units, including
shared clock/canvas geometry. Overlapping or unbound support rolls back the
whole optional edit; it never deletes an old family to make the edit pass.
"""
from dataclasses import replace
import math

from .boundary_affine_plan import BoundaryAffinePlan, ANALYSIS_BASIS as AFFINE_BASIS
from .render import RenderPlan, _video_render_units, segment_frame_counts
from .visual_boundary_repair import VisualBoundaryRepairPlan
from .visual_splice_failure_feedback import splice_feedback_result_context_identity


class BoundaryEvidencePreservationUnavailable(ValueError):
    def __init__(self, family, boundary_index, reason):
        self.family, self.boundary_index, self.reason = family, boundary_index, reason
        super().__init__(f'{family} boundary {boundary_index}: {reason}')

    def to_dict(self):
        return dict(family=self.family, boundary_index=self.boundary_index, reason=self.reason,
            disposition='complete_optional_edit_rollback_keep_incumbent_payloads')


def _units_intersecting(plan, intervals):
    counts = segment_frame_counts(plan); cursor = 0; result = []
    for unit in _video_render_units(plan, counts):
        end = cursor+unit.frame_count
        if any(a < end and b > cursor for a,b in intervals):
            first,last = unit.first_segment_index,unit.last_segment_index
            # Edge affine/overlap is a renderer dependency too, not only the
            # per-cell camera. Include both incident transitions and the run.
            result.append((cursor,end,unit,tuple(plan.transitions[max(0,first-1):last+1])))
        cursor = end
    return tuple(result)


def preserve_boundary_evidence(*, old_plan, new_plan, old_timeline, new_timeline,
        boundary_repairs, boundary_affine_plan):
    """Return payloads only when their actual support preimages are unchanged.

    Historical attempt contexts/audits remain exactly historical; only the
    output timeline wrapper is rebound. Missing legacy support is explicitly
    unavailable, not a guessed +/-1-cell footprint or erased authority.
    """
    if not isinstance(old_plan,RenderPlan) or not isinstance(new_plan,RenderPlan):
        raise TypeError('boundary evidence preservation requires complete render plans')
    if boundary_repairs is None and boundary_affine_plan is None:
        return None,None,dict(retained_repair_count=0,retained_affine_decision_count=0)
    if old_plan==new_plan and old_timeline==new_timeline:
        # A no-op does not issue a new binding and needs no invented legacy
        # support. Keep the exact incumbent, including unknown old payloads.
        return boundary_repairs,boundary_affine_plan,dict(
            unchanged_complete_incumbent=True,new_media_or_part_certificate_issued=False,
            retained_repair_count=len(boundary_repairs.repairs) if isinstance(boundary_repairs,VisualBoundaryRepairPlan) else None,
            retained_affine_decision_count=len(boundary_affine_plan.decisions) if isinstance(boundary_affine_plan,BoundaryAffinePlan) else None)
    if any(getattr(old_plan,n)!=getattr(new_plan,n) for n in
            ('sources','fps','width','height','aspect_ratio','master_audio_start')):
        raise BoundaryEvidencePreservationUnavailable('all_boundary_evidence',None,'renderer_domain_header_changed')
    counts = segment_frame_counts(old_plan); total = sum(counts)
    ticks = [0]
    for count in counts: ticks.append(ticks[-1]+count)

    def check_support(family,index,intervals):
        if (type(index) is not int or not 0 <= index < len(counts)-1
                or any(type(a) is not int or type(b) is not int or not 0 <= a < b <= total
                    for a,b in intervals)):
            raise BoundaryEvidencePreservationUnavailable(family,index,'declared_support_outside_current_output')
        if _units_intersecting(old_plan,intervals)!=_units_intersecting(new_plan,intervals):
            raise BoundaryEvidencePreservationUnavailable(family,index,'declared_support_or_shared_renderer_unit_changed')

    retained_repairs = 0
    if boundary_repairs is not None:
        if not isinstance(boundary_repairs,VisualBoundaryRepairPlan):
            raise BoundaryEvidencePreservationUnavailable('boundary_repairs',None,'unknown_payload_type')
        # The QA consumer selects the last record at each boundary. Earlier
        # attempts stay verbatim, not relabelled as measurements of a new plan.
        active = {r.boundary_index:r for r in boundary_repairs.repairs}
        for index,repair in active.items():
            support = repair.measurement_support
            if (support is None or support.get('renderer_bound') is not True
                    or support.get('output_fps')!=old_plan.fps
                    or support.get('output_timeline_origin_seconds')!=old_plan.master_audio_start):
                raise BoundaryEvidencePreservationUnavailable('boundary_repairs',index,'complete_bound_sample_support_unavailable')
            before = splice_feedback_result_context_identity(old_timeline,index)
            after = splice_feedback_result_context_identity(new_timeline,index)
            if before!=repair.result_edge_context_identity or before!=after:
                raise BoundaryEvidencePreservationUnavailable('boundary_repairs',index,'selected_cut_context_changed_or_stale')
            # Selected endpoints plus ALL measured query/guard brackets; a
            # long bracket is not shortened to neighboring cell indices.
            intervals = (*support['output_frame_intervals'],(ticks[index+1]-1,ticks[index+1]+1))
            check_support('boundary_repairs',index,intervals)
        retained_repairs = len(boundary_repairs.repairs)
        if boundary_repairs.timeline!=new_timeline:
            boundary_repairs = replace(boundary_repairs,timeline=new_timeline)

    if boundary_affine_plan is not None:
        if not isinstance(boundary_affine_plan,BoundaryAffinePlan) or boundary_affine_plan.analysis_basis!=AFFINE_BASIS:
            raise BoundaryEvidencePreservationUnavailable('boundary_affine_plan',None,'unknown_affine_contract')
        for decision in boundary_affine_plan.decisions:
            index = decision.boundary_index
            if type(index) is not int or not 0 <= index < len(counts)-1:
                raise BoundaryEvidencePreservationUnavailable('boundary_affine_plan',index,'invalid_boundary_index')
            left,right = old_timeline.choices[index:index+2]
            expected = min(left.source_local_end,left.source_local_start+
                (counts[index]-1)*left.source_playback_rate/old_plan.fps)
            if (decision.outgoing_source_index!=left.source_index or decision.incoming_source_index!=right.source_index
                    or not math.isclose(decision.outgoing_source_seconds,expected,rel_tol=0.,abs_tol=1e-9)
                    or not math.isclose(decision.incoming_source_seconds,right.source_local_start,rel_tol=0.,abs_tol=1e-9)):
                raise BoundaryEvidencePreservationUnavailable('boundary_affine_plan',index,'affine_endpoint_preimage_stale')
            check_support('boundary_affine_plan',index,((ticks[index+1]-1,ticks[index+1]+1),))
    return boundary_repairs,boundary_affine_plan,dict(retained_repair_count=retained_repairs,
        retained_affine_decision_count=len(boundary_affine_plan.decisions) if boundary_affine_plan else 0,
        historical_attempts_not_relabelled=True,shared_renderer_unit_support_checked=True,
        new_media_or_part_certificate_issued=False)
