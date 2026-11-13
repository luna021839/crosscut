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
    output_native_join: tuple = ()

    def to_dict(self):
        return dict(contract='owned_joint_group_source_native_preparation_v1', status=self.status,
            plan_identity=self.plan_identity, window_bounds=self.window_bounds,
            tracklets=[item.to_dict() for item in self.tracklets],
            session_diagnostics=self.session_diagnostics, elapsed_seconds=self.elapsed_seconds,
            output_native_join=[list(rows) for rows in self.output_native_join],
            actual_shot_count=len(self.window_bounds),
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
    budget = JointGroupWorkBudget(absolute_deadline=absolute_deadline, maximum_new_decodes=200,
        maximum_model_calls=2400)
    budget.check()
    grids, join, mutable_count = _native_window_schedule(plan, domain, window_bounds)
    if not _native_prefix_fits(budget, grids, mutable_count):
        return _unavailable_preparation(plan, window_bounds, budget, started, join)
    with GroupJointObservationSession(domain=domain, budget=budget) as session:
        register_joint_native_models(session, person_model_path=person_model_path,
            pose_model_path=pose_model_path, head_model_path=head_model_path)
        return _sense_native_window(session, plan, window_bounds, grids, join, started)


def _native_window_schedule(plan, domain, window_bounds):
    """Bind one/two REAL mutable shots to the full incumbent sampler once.

    There is no artificial middle seam when one shot spans several cells.
    Repeated native frames retain their individual output joins; they are not
    replaced with a neighboring native frame to make a tracking test pass.
    """
    if (not isinstance(plan, RenderPlan) or not isinstance(domain, GroupSourceDomain)
            or plan.sources != domain.request_sources or plan.aspect_ratio.value != '16:9'
            or any(transition.kind.uses_overlap for transition in plan.transitions)
            or not isinstance(window_bounds, tuple) or len(window_bounds) not in (3, 4)
            or any(not isinstance(bounds, tuple) or len(bounds) != 2
                or any(type(value) is not int for value in bounds) for bounds in window_bounds)
            or any(a[1] != b[0] for a, b in zip(window_bounds, window_bounds[1:]))):
        raise ValueError('joint native preparation needs one complete bound three/four-actual-shot window')
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
    mutable_count = sum(end-first for first, end in window_bounds[1:-1])
    samples = native_frame_clocks_for_output(plan, domain.clocks_by_request,
        range(window_bounds[0][0], window_bounds[-1][1]))
    grids, join = [], []
    for slot, (first, end) in enumerate(window_bounds):
        ticks = range(end-4, end) if slot == 0 else (
            range(first, first+4) if slot == len(window_bounds)-1 else range(first, end))
        rows = tuple(samples[tick] for tick in ticks)
        if (any(row.terminal_clone for row in rows)
                or len({row.source_index for row in rows}) != 1):
            raise ValueError('native preparation cannot borrow another source or terminal-cloned evidence')
        grids.append((rows[0].source_index, tuple(row.native_frame_index for row in rows)))
        join.append(tuple((tick, row.source_index, row.native_frame_index)
            for tick, row in zip(ticks, rows, strict=True)))
    return tuple(grids), tuple(join), mutable_count


def _native_prefix_fits(budget, grids, mutable_count):
    """Conservative incremental preflight on the OWNER'S remaining budget.

    This does not reset counts or claim free cache hits. At most twelve body
    ROI calls plus person/head calls are needed for each requested native row.
    The live session still charges every actual call and cache miss, including
    eviction/redecode. Previously spent work remains spent across prefixes.
    """
    budget.check()
    rows = sum(len(frames) for _, frames in grids)
    return (mutable_count <= 300
        and budget.counts['new_decodes']+rows <= budget.limits['new_decodes']
        and budget.counts['model_calls']+rows*14 <= budget.limits['model_calls'])


def _unavailable_preparation(plan, window_bounds, budget, started, join):
    return GroupJointNativePreparation(_plan_identity(plan), window_bounds, (),
        budget.to_dict(), monotonic()-started, 'dense_budget_unavailable_before_native_sensing', join)


def _sense_native_window(session, plan, window_bounds, grids, join, started):
    tracklets = []
    for source, frames in grids:
        tracklets.append(track_native_group(session, request_source_index=source,
            native_frame_indices=frames))
    diagnostics = session.diagnostics()
    status = ('native_required_person_paths_observed_not_an_edit_approval'
        if all(item.complete_required_native_paths for item in tracklets) else
        'native_required_person_path_unknowns_retained')
    return GroupJointNativePreparation(_plan_identity(plan), window_bounds, tuple(tracklets),
        diagnostics, monotonic()-started, status, join)


def prepare_joint_group_native_in_session(*, plan, window_bounds, session):
    """Native prefix for the static COMPLETE owner, without another worker.

    The owner creates/registers/closes one session and owns its absolute budget.
    This prefix never closes that session or re-registers detectors. It remains
    source-native path evidence, not semantic identity/part admission.
    """
    started = monotonic()
    if not isinstance(session, GroupJointObservationSession) or session.closed:
        raise ValueError('native preparation requires its live owned observation session')
    session.budget.check()
    grids, join, mutable_count = _native_window_schedule(plan, session.domain, window_bounds)
    if not _native_prefix_fits(session.budget, grids, mutable_count):
        return _unavailable_preparation(plan, window_bounds, session.budget, started, join)
    required = ('nanodet_people', 'rtmpose_body7', 'yunet_head_faces')
    if any(name not in session.models for name in required):
        raise ValueError('native owner must register all required models once before its prefix')
    for name in required:
        session._assert_model_assets(name)
    return _sense_native_window(session, plan, window_bounds, grids, join, started)
