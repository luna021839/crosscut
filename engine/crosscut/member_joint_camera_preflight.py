"""Run-local feedback before encoding a required, unregistered pose bridge.

Missing camera evidence is NOT source infeasibility. It may change proposal
ranking, never persistent source exclusion or authorization of a hard cut.
"""
import hashlib
import math
from .render import TransitionKind, MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY


class MemberJointCameraPreflight:
    def __init__(self, *, maximum_reselections=6, penalty=2.75):
        if (isinstance(maximum_reselections, bool) or not isinstance(maximum_reselections, int)
                or not 0 <= maximum_reselections <= 6
                or isinstance(penalty, bool) or not math.isfinite(penalty) or penalty <= 0):
            raise ValueError('joint camera feedback requires bounded reselection and positive cost')
        self.maximum_reselections = maximum_reselections
        self.penalty = penalty
        self.failed_plans = set()
        self.penalties = {}
        self.reselection_count = 0

    def review(self, *, visual_selection, render_segments, transitions, camera_audit):
        segments = tuple(render_segments)
        selected = tuple(visual_selection.source_indices)
        if len(segments) != len(selected) or len(transitions) != len(segments)-1:
            raise ValueError('joint camera feedback source path is not plan-bound')
        required = tuple(b for b in visual_selection.boundaries
                         if b.mismatch.cinematic_bridge is not None
                         and b.mismatch.cinematic_bridge.style == 'pose_dissolve')
        rows = camera_audit['boundaries']
        by_index = {row['boundary_index']: row for row in rows}
        if len(by_index) != len(rows):
            raise ValueError('joint camera audit repeats a boundary')
        failures = []
        for boundary in required:
            index = boundary.boundary_index
            if not 0 <= index < len(segments)-1 or selected[index] == selected[index+1]:
                raise ValueError('required camera boundary is outside the selected source path')
            row = by_index.get(index)
            applied = (row is not None and row.get('applied') is True
                and row.get('containment', {}).get('passed') is True
                and row.get('registration', {}).get('source_indices') ==
                    [segments[index].source_index, segments[index+1].source_index]
                and segments[index].outgoing_overlap_transform is not None
                and segments[index+1].editorial_transform is not None
                and transitions[index].kind is TransitionKind.FACE_DISSOLVE
                and transitions[index].transform_overlap_authority == MEMBER_SOURCE_MEASURED_JOINT_OVERLAP_AUTHORITY)
            if applied:
                continue
            cut = visual_selection.selected_boundary_master_seconds[index]
            if isinstance(cut, bool) or not math.isfinite(cut):
                raise ValueError('joint camera feedback needs a finite exact cut clock')
            failures.append(dict(boundary_index=index, cut_master_seconds=cut,
                outgoing_eligible_source_index=selected[index], incoming_eligible_source_index=selected[index+1],
                reason=row.get('reason', 'camera_application_unverified') if row else 'required_camera_audit_missing',
                source_infeasibility_claimed=False))
        # The whole source clock/transform context, not just source IDs, detects
        # an unchanged failed proposal. Fresh success is checked before this key.
        key = hashlib.sha256(repr((segments, tuple(transitions), tuple(visual_selection.selected_boundary_master_seconds))).encode()).hexdigest()
        report = dict(contract='run_local_required_joint_camera_preflight_v1',
            passed=not failures, failures=failures, failed_plan_sha256=key,
            reselection_requested=False, reselection_count=self.reselection_count,
            maximum_reselections=self.maximum_reselections, hard_cut_fallback_authorized=False,
            source_exclusion_authorized=False, evidence_persisted_as_infeasible=False,
            quality_approval=False, proxy_encoding_skipped=bool(failures))
        if not failures:
            return dict(report, reason='required_cameras_applied', ranking_penalties=tuple(self.penalties.values()))
        if key in self.failed_plans:
            return dict(report, reason='unchanged_failed_camera_plan', ranking_penalties=tuple(self.penalties.values()))
        self.failed_plans.add(key)
        if self.reselection_count >= self.maximum_reselections:
            return dict(report, reason='joint_camera_reselection_budget_exhausted', ranking_penalties=tuple(self.penalties.values()))
        for failure in failures:
            edge = (failure['cut_master_seconds'], failure['outgoing_eligible_source_index'], failure['incoming_eligible_source_index'])
            self.penalties[edge] = (*edge, self.penalty)
        self.reselection_count += 1
        return dict(report, reason='unapplied_required_camera_rank_feedback',
            reselection_requested=True, reselection_count=self.reselection_count,
            ranking_penalties=tuple(self.penalties.values()))
