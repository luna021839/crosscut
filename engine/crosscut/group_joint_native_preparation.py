"""One bounded owned native preparation for a GROUP joint-edit transaction.

This is the native sensing prefix, not a complete source-changing bundle.
Incomplete tracks/modality declarations cannot be promoted or relabelled as
an approved edit. The future joint owner consumes these same typed rows.
"""
from dataclasses import dataclass
import hashlib
import json
from time import monotonic

from .group_actual_shots import visible_group_shots
from .group_joint_native_observer import register_joint_native_models
from .group_joint_native_tracks import track_native_group
from .group_joint_observations import GroupJointObservationSession
from .group_joint_source_camera import JointGroupWorkBudget
from .group_source_domain import GroupSourceDomain
from .optional_native_worker import isolated_optional
from .render import RenderPlan, segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output


def _plan_identity(plan):
    return hashlib.sha256(json.dumps(plan.to_json(), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class GroupJointNativePreparation:
    plan_identity: str
    window_bounds: tuple
    tracklets: tuple
    session_diagnostics: dict
    elapsed_seconds: float
    status: str

    def to_dict(self):
        return dict(contract='owned_joint_group_source_native_preparation_v1', status=self.status,
            plan_identity=self.plan_identity, window_bounds=self.window_bounds,
            tracklets=[item.to_dict() for item in self.tracklets],
            session_diagnostics=self.session_diagnostics, elapsed_seconds=self.elapsed_seconds,
            camera_search_executed=False, production_bundle_created=False,
            normal_app_source_replacement_executed=False, quality_approved=False)


@isolated_optional()
def prepare_joint_group_native(*, plan, domain, window_bounds, person_model_path,
        pose_model_path, head_model_path, absolute_deadline):
    """One reader/model owner; exact B/C grids plus fixed outer boundary rows.

    Dense worst-case model work is admitted before any model loads. There is
    no repeated whole-song analysis, callback-based proof or encode here.
    """
    started = monotonic()
    if (not isinstance(plan, RenderPlan) or not isinstance(domain, GroupSourceDomain)
            or plan.sources != domain.request_sources or plan.aspect_ratio.value != '16:9'
            or any(transition.kind.uses_overlap for transition in plan.transitions)
            or not isinstance(window_bounds, tuple) or len(window_bounds) != 4
            or any(not isinstance(bounds, tuple) or len(bounds) != 2
                or any(type(value) is not int for value in bounds) for bounds in window_bounds)
            or any(a[1] != b[0] for a, b in zip(window_bounds, window_bounds[1:]))):
        raise ValueError('joint native preparation needs one complete bound four-actual-shot window')
    budget = JointGroupWorkBudget(absolute_deadline=absolute_deadline, maximum_new_decodes=200,
        maximum_model_calls=2400)
    budget.check()
    counts = segment_frame_counts(plan)
    cell_ticks = {0}
    cursor = 0
    for count in counts:
        cursor += count
        cell_ticks.add(cursor)
    actual = {(shot.first_output_frame, shot.end_output_frame) for shot in visible_group_shots(plan, domain)}
    if any(not 0 <= first < end <= plan.output_frame_count or end-first < 4
            or first not in cell_ticks or end not in cell_ticks or (first, end) not in actual
            for first, end in window_bounds):
        raise ValueError('joint native preparation changed actual cuts or needs owned cell splitting')
    mutable_count = sum(end-first for first, end in window_bounds[1:3])
    # At most twelve body ROI calls plus person/head models per raster.
    # Counting all declared dense rows here prevents shrinking the scope after
    # a sparse search has exhausted the worker's named model budget.
    reserve_count = mutable_count+8
    if mutable_count > 300 or reserve_count > 200 or reserve_count*14 > 2400:
        return GroupJointNativePreparation(_plan_identity(plan), window_bounds, (),
            budget.to_dict(), monotonic()-started, 'dense_budget_unavailable_before_native_sensing')
    samples = native_frame_clocks_for_output(plan, domain.clocks_by_request,
        range(window_bounds[0][0], window_bounds[-1][1]))
    grids = []
    for slot, (first, end) in enumerate(window_bounds):
        ticks = range(end-4, end) if slot == 0 else range(first, first+4) if slot == 3 else range(first, end)
        rows = tuple(samples[tick] for tick in ticks)
        if (any(row.terminal_clone for row in rows)
                or len({row.source_index for row in rows}) != 1):
            raise ValueError('native preparation cannot borrow another source or terminal-cloned evidence')
        grids.append((rows[0].source_index, tuple(row.native_frame_index for row in rows)))
    tracklets = []
    with GroupJointObservationSession(domain=domain, budget=budget) as session:
        register_joint_native_models(session, person_model_path=person_model_path,
            pose_model_path=pose_model_path, head_model_path=head_model_path)
        for source, frames in grids:
            tracklets.append(track_native_group(session, request_source_index=source,
                native_frame_indices=frames))
        diagnostics = session.diagnostics()
    status = ('native_required_person_paths_observed_not_an_edit_approval'
        if all(item.complete_required_native_paths for item in tracklets) else
        'native_required_person_path_unknowns_retained')
    return GroupJointNativePreparation(_plan_identity(plan), window_bounds, tuple(tracklets),
        diagnostics, monotonic()-started, status)
