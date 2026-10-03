"""Explicit global shot ownership and atomic numeric camera union.

Actual shots, not beat cells or role labels, own camera/source trajectories.
There are no positional L/B/C/R slots here. Roles are original observation
epochs within those shots. This module does NO subject sensing, donor
admission, bundle completion or media QA. In particular a numeric ownership
allocation is not permission to weaken original part duties. Subcell mutation
still requires a topology-aware completion owner; it is not promoted to the
unchanged-grid assembler implemented here.
"""
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
import json
import math

from .group_actual_shots import (assert_native_cut_events_preserved,
    visible_group_cut_inventory, visible_group_shots)
from .group_scene_dependency_support import derive_support
from .group_scene_epoch_program import freeze_original_role_epochs
from .group_source_domain import GroupSourceDomain
from .group_cut_camera_proposal import UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON
from .group_joint_native_preparation import _plan_identity
from .render import RenderPlan, RenderSegment, RenderTransition, segment_frame_counts, _can_share_source_clock
from .render_boundary_clock import native_frame_clocks_for_output

CONTRACT = 'global_original_actual_shot_ownership_program_v1'


def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return sha256(_dump(value).encode()).hexdigest()


def _header(plan):
    # Serialized source_usage is DERIVED from the segments, not an immutable
    # input header. Treating it as a header would reject every source change.
    # Enumerate the actual dataclass inputs so future inputs cannot disappear
    # merely because a diagnostic key list was copied from an old serializer.
    payload = plan.to_json()
    return {field.name: (payload.get(field.name, []) if field.name == 'authorized_shot_video_clocks'
        else payload[field.name]) for field in fields(plan)
        if field.name not in ('segments', 'transitions', 'output')}


def _offsets(counts):
    result = [0]
    for count in counts: result.append(result[-1]+count)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class OwnedSceneUnit:
    unit_id: str
    first_output_frame: int
    end_output_frame: int
    original_role_program_json: str

    @property
    def role_program(self): return json.loads(self.original_role_program_json)

    @property
    def frame_count(self): return self.end_output_frame-self.first_output_frame

    def to_dict(self):
        return dict(unit_id=self.unit_id, first_output_frame=self.first_output_frame,
            end_output_frame=self.end_output_frame, original_role_program=self.role_program)


@dataclass(frozen=True, slots=True)
class OwnedSceneProgram:
    original_plan_identity: str
    original_header_json: str
    source_domain_sha256: str
    global_cell_frame_counts: tuple[int, ...]
    units: tuple[OwnedSceneUnit, ...]

    @property
    def program_sha256(self): return _digest(self.to_dict())

    def to_dict(self):
        return dict(contract=CONTRACT, original_plan_identity=self.original_plan_identity,
            original_plan_header=json.loads(self.original_header_json),
            source_domain_sha256=self.source_domain_sha256,
            global_cell_frame_counts=list(self.global_cell_frame_counts),
            units=[unit.to_dict() for unit in self.units],
            original_output_role_epochs_frozen=True,
            original_native_part_lifecycle_frozen=False,
            role_epochs_are_not_camera_or_source_cuts=True,
            render_topology_materialized=False, candidate_admitted=False, quality_approved=False)


def freeze_owned_scene_program(context, domain):
    """All original actual shots/roles BEFORE any donor observation.

    Native cut and one-tick original fragments are retained exactly. This
    describes a bounded full output programme; it does not allocate NN work,
    infer original anatomy or decide which shots are mutable.
    """
    plan = context.incumbent.render_plan
    if (not isinstance(plan, RenderPlan) or not isinstance(domain, GroupSourceDomain)
            or plan.sources != domain.request_sources or plan.aspect_ratio.value != '16:9'
            or not 1 <= plan.output_frame_count <= 36000):
        raise ValueError('global scene programme needs a bounded complete GROUP source/output domain')
    identity = _plan_identity(plan)
    units = []
    for shot in visible_group_shots(plan, domain):
        original = freeze_original_role_epochs(context, shot)
        unit_id = _digest(dict(original_plan_identity=identity, original_role_program=original))
        units.append(OwnedSceneUnit(unit_id, shot.first_output_frame, shot.end_output_frame, _dump(original)))
    return OwnedSceneProgram(identity, _dump(_header(plan)), _digest(domain.to_dict()),
        tuple(segment_frame_counts(plan)), tuple(units))


def validate_owned_scene_program(value, context, domain):
    """External original rederivation, not a self-hash authority claim."""
    if value != freeze_owned_scene_program(context, domain).to_dict():
        raise ValueError('owned scene programme differs from its external original roles/clock')


@dataclass(frozen=True, slots=True)
class OwnedSceneIntervals:
    program: OwnedSceneProgram
    mutable_unit_ids: tuple[str, ...]
    maximum_mutable_units: int
    maximum_mutable_frames: int

    @property
    def mutable_output_intervals(self):
        return tuple((u.first_output_frame, u.end_output_frame) for u in self.program.units
            if u.unit_id in self.mutable_unit_ids)

    @property
    def mutable_frame_count(self):
        return sum(end-first for first, end in self.mutable_output_intervals)

    def to_dict(self):
        return dict(contract='explicit_global_owned_mutable_intervals_v1',
            original_program_sha256=self.program.program_sha256,
            mutable_unit_ids=list(self.mutable_unit_ids),
            mutable_output_intervals=[list(interval) for interval in self.mutable_output_intervals],
            maximum_mutable_units=self.maximum_mutable_units,
            maximum_mutable_frames=self.maximum_mutable_frames,
            mutable_frame_count=self.mutable_frame_count,
            positional_fixed_endpoints_assumed=False,
            original_duties_need_freezing_before_donor_sensing=True,
            native_work_or_completion_budget_reserved=False,
            candidate_admitted=False, quality_approved=False)


def own_scene_intervals(program, *, mutable_unit_ids, maximum_mutable_units, maximum_mutable_frames):
    if (not isinstance(program, OwnedSceneProgram)
            or not isinstance(mutable_unit_ids, tuple) or not mutable_unit_ids
            or any(not isinstance(value, str) for value in mutable_unit_ids)
            or len(set(mutable_unit_ids)) != len(mutable_unit_ids)
            or type(maximum_mutable_units) is not int or not 1 <= maximum_mutable_units <= 256
            or type(maximum_mutable_frames) is not int or not 1 <= maximum_mutable_frames <= 36000):
        raise ValueError('global mutable ownership requires explicit unique IDs and finite caps')
    canonical = tuple(u.unit_id for u in program.units if u.unit_id in mutable_unit_ids)
    if canonical != mutable_unit_ids:
        raise ValueError('global mutable IDs must exist once in original programme order')
    result = OwnedSceneIntervals(program, canonical, maximum_mutable_units, maximum_mutable_frames)
    if len(canonical) > maximum_mutable_units or result.mutable_frame_count > maximum_mutable_frames:
        raise ValueError('global mutable ownership exceeds the same declared interval budget')
    return result


def _validate_allocation(allocation):
    if not isinstance(allocation, OwnedSceneIntervals):
        raise ValueError('global camera union needs an explicit owned interval allocation')
    if own_scene_intervals(allocation.program, mutable_unit_ids=allocation.mutable_unit_ids,
            maximum_mutable_units=allocation.maximum_mutable_units,
            maximum_mutable_frames=allocation.maximum_mutable_frames) != allocation:
        raise ValueError('global mutable allocation changed its preimage')


def expand_owned_scene_intervals(allocation, *, blocked_fixed_unit_ids):
    """Promote incident fixed neighbors atomically without renewing any caps.

    This is numeric allocation only. Every newly mutable original interval
    must acquire its own frozen original native duties before donor sensing;
    no old failed fixed endpoint can be deleted from a sensing receipt.
    """
    _validate_allocation(allocation)
    units = allocation.program.units
    mutable = {i for i, u in enumerate(units) if u.unit_id in allocation.mutable_unit_ids}
    fixed = {j for i in mutable for j in (i-1, i+1) if 0 <= j < len(units) and j not in mutable}
    if (not isinstance(blocked_fixed_unit_ids, tuple) or not blocked_fixed_unit_ids
            or any(not isinstance(value, str) for value in blocked_fixed_unit_ids)
            or len(set(blocked_fixed_unit_ids)) != len(blocked_fixed_unit_ids)
            or not set(blocked_fixed_unit_ids).issubset({units[i].unit_id for i in fixed})):
        raise ValueError('expansion can own only explicitly blocked incident fixed neighbors')
    added = set(allocation.mutable_unit_ids)|set(blocked_fixed_unit_ids)
    return own_scene_intervals(allocation.program,
        mutable_unit_ids=tuple(u.unit_id for u in units if u.unit_id in added),
        maximum_mutable_units=allocation.maximum_mutable_units,
        maximum_mutable_frames=allocation.maximum_mutable_frames)


@dataclass(frozen=True, slots=True)
class OwnedSceneCamera:
    original_program_sha256: str
    unit_id: str
    segment_indices: tuple[int, ...]
    segments: tuple
    segment_output_frame_counts: tuple[int, ...]
    request_source_index: int
    native_frame_indices: tuple[int, ...]
    source_request_seconds: tuple[float, ...]
    role_program_sha256: str
    role_epoch_intervals: tuple[tuple[int, int], ...]

    def __post_init__(self):
        hashes = (self.original_program_sha256, self.unit_id, self.role_program_sha256)
        if (any(not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value) for value in hashes)
                or not isinstance(self.segment_indices, tuple) or not self.segment_indices
                or any(type(index) is not int or index < 0 for index in self.segment_indices)
                or tuple(sorted(set(self.segment_indices))) != self.segment_indices
                or not isinstance(self.segments, tuple) or len(self.segments) != len(self.segment_indices)
                or any(not isinstance(segment, RenderSegment) for segment in self.segments)
                or type(self.request_source_index) is not int or self.request_source_index < 0
                or any(segment.source_index != self.request_source_index for segment in self.segments)
                or not isinstance(self.segment_output_frame_counts, tuple)
                or len(self.segment_output_frame_counts) != len(self.segments)
                or any(type(count) is not int or count < 1 for count in self.segment_output_frame_counts)
                or not isinstance(self.native_frame_indices, tuple)
                or len(self.native_frame_indices) != sum(self.segment_output_frame_counts)
                or any(type(frame) is not int or frame < 0 for frame in self.native_frame_indices)
                or any(a > b for a, b in zip(self.native_frame_indices, self.native_frame_indices[1:]))
                or not isinstance(self.source_request_seconds, tuple)
                or len(self.source_request_seconds) != len(self.native_frame_indices)
                or any(type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                    for seconds in self.source_request_seconds)
                or not isinstance(self.role_epoch_intervals, tuple) or not self.role_epoch_intervals
                or any(not isinstance(span, tuple) or len(span) != 2
                    or any(type(tick) is not int for tick in span) or not 0 <= span[0] < span[1]
                    for span in self.role_epoch_intervals)):
            raise ValueError('global camera needs one explicit complete typed numeric preimage')

    def to_dict(self):
        return dict(contract='global_owned_whole_camera_numeric_preimage_v1',
            original_program_sha256=self.original_program_sha256, unit_id=self.unit_id,
            segment_indices=list(self.segment_indices), segments=[asdict(s) for s in self.segments],
            segment_output_frame_counts=list(self.segment_output_frame_counts),
            request_source_index=self.request_source_index,
            native_frame_indices=list(self.native_frame_indices),
            source_request_seconds=list(self.source_request_seconds),
            role_program_sha256=self.role_program_sha256,
            role_epoch_intervals=[list(span) for span in self.role_epoch_intervals],
            role_epoch_boundaries_restart_source_or_camera=False,
            subject_duties_measured=False, candidate_admitted=False, quality_approved=False)

    @property
    def camera_sha256(self): return _digest(self.to_dict())


def camera_unit_from_plan(*, program, unit_id, plan, domain):
    """Bind one WHOLE camera regardless of the number of original role epochs.

    A partial candidate may be invalid at its neighbors until other whole
    cameras are installed in the same union. That is not a prerequisite solo
    success. Only the final assembler validates both outside sampler phases
    and all actual cut events. Internal original role boundaries add no cuts.
    """
    if (not isinstance(program, OwnedSceneProgram) or not isinstance(plan, RenderPlan)
            or not isinstance(domain, GroupSourceDomain)
            or plan.sources != domain.request_sources or _digest(domain.to_dict()) != program.source_domain_sha256
            or _dump(_header(plan)) != program.original_header_json):
        raise ValueError('whole camera changed its complete source/domain/header preimage')
    if (tuple(segment_frame_counts(plan)) != program.global_cell_frame_counts
            or plan.authorized_shot_video_clocks
            or any(t.kind.uses_overlap for t in plan.transitions)
            or any(s.outgoing_overlap_transform is not None for s in plan.segments)):
        raise ValueError('whole camera needs topology-aware authority for changed grid/clock/overlap')
    matches = [unit for unit in program.units if unit.unit_id == unit_id]
    if len(matches) != 1:
        raise ValueError('whole camera has no unique original actual-shot unit')
    unit = matches[0]; offsets = _offsets(program.global_cell_frame_counts)
    first, end = unit.first_output_frame, unit.end_output_frame
    if first not in offsets or end not in offsets:
        raise ValueError('subcell mutable camera requires a topology-aware owner, not positional fragments')
    indices = tuple(range(offsets.index(first), offsets.index(end)))
    if (unit.frame_count < 2 or not indices
            or not _can_share_source_clock(plan, first_index=indices[0], candidate_last_index=indices[-1])):
        raise ValueError('global camera must own one complete continuous actual source clock')
    source = plan.segments[indices[0]].source_index
    domain.eligible_index(source)
    if any(first < e.first_incoming_output_frame < end for e in visible_group_cut_inventory(plan, domain)):
        raise ValueError('global whole camera crossed an actual native/source cut')
    samples = native_frame_clocks_for_output(plan, domain.clocks_by_request, range(first, end))
    if any(row.source_index != source or row.terminal_clone for row in samples.values()):
        raise ValueError('global camera cannot borrow a different source or terminal-cloned native occurrence')
    role_program = unit.role_program
    return OwnedSceneCamera(program.program_sha256, unit_id, indices,
        tuple(plan.segments[i] for i in indices), tuple(program.global_cell_frame_counts[i] for i in indices),
        source, tuple(row.native_frame_index for row in samples.values()),
        tuple(row.source_request_seconds for row in samples.values()), _digest(role_program),
        tuple((epoch['first_output_frame'], epoch['end_output_frame']) for epoch in role_program['epochs']))


@dataclass(frozen=True, slots=True)
class NumericOwnedSceneUnion:
    plan: RenderPlan
    allocation: OwnedSceneIntervals
    cameras: tuple[OwnedSceneCamera, ...]
    changed_segment_indices: tuple[int, ...]
    dependency_support_json: str

    @property
    def dependency_support(self): return json.loads(self.dependency_support_json)

    def to_dict(self):
        return dict(contract='atomic_global_owned_scene_camera_numeric_union_v1',
            allocation=self.allocation.to_dict(), cameras=[camera.to_dict() for camera in self.cameras],
            numeric_plan=self.plan.to_json(), changed_segment_indices=list(self.changed_segment_indices),
            dependency_support=self.dependency_support,
            one_numeric_union_before_completion=True,
            full_bundle_created=False, selected_parent_or_final_replay_performed=False,
            native_work_or_completion_budget_reserved=False, candidate_admitted=False, quality_approved=False)


def assemble_owned_scene_union(*, context, domain, allocation, cameras):
    """Install the explicit mutable union BEFORE one dependency closure.

    Neither local completed bundles nor their admission receipts are glued
    together. Every untouched RenderSegment stays byte-for-value original;
    all whole-camera source occurrences are rebound against the final union.
    Completion, original duty programme, distinct-camera semantics, protected
    budgets and independent parent/final consumers are still required.
    """
    _validate_allocation(allocation)
    program = allocation.program
    validate_owned_scene_program(program.to_dict(), context, domain)
    if (not isinstance(cameras, tuple) or any(not isinstance(camera, OwnedSceneCamera) for camera in cameras)
            or tuple(camera.unit_id for camera in cameras) != allocation.mutable_unit_ids):
        raise ValueError('global union cameras must cover every explicit mutable ID exactly once')
    original = context.incumbent.render_plan
    segments = list(original.segments)
    offsets = _offsets(program.global_cell_frame_counts)
    units = {unit.unit_id: unit for unit in program.units}
    for camera in cameras:
        if camera.original_program_sha256 != program.program_sha256:
            raise ValueError('global camera belongs to another original programme')
        # Structural ownership BEFORE assembly; never demand a successful solo
        # full-plan replay. Adjacent candidates can temporarily alias each
        # other's old source and change provisional sampler rounding. ALL
        # numerical native/camera preimages are rebound only on the final union.
        unit = units[camera.unit_id]
        if unit.first_output_frame not in offsets or unit.end_output_frame not in offsets:
            raise ValueError('subcell mutable union requires a topology-aware owner')
        expected_indices = tuple(range(offsets.index(unit.first_output_frame), offsets.index(unit.end_output_frame)))
        expected_counts = tuple(program.global_cell_frame_counts[i] for i in expected_indices)
        epochs = tuple((epoch['first_output_frame'], epoch['end_output_frame']) for epoch in unit.role_program['epochs'])
        if (camera.segment_indices != expected_indices or camera.segment_output_frame_counts != expected_counts
                or camera.role_program_sha256 != _digest(unit.role_program) or camera.role_epoch_intervals != epochs):
            raise ValueError('global camera changed its original unit/cell/role domain')
        domain.eligible_index(camera.request_source_index)
        for index, segment in zip(camera.segment_indices, camera.segments, strict=True):
            if (segment.master_start, segment.duration) != (
                    original.segments[index].master_start, original.segments[index].duration):
                raise ValueError('global union camera changed an original owned output cell')
            segments[index] = segment
    changed = tuple(i for i, (old, new) in enumerate(zip(original.segments, segments, strict=True)) if old != new)
    transitions = tuple(RenderTransition(tier='C', reason=UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON)
        if edge in changed or edge+1 in changed else old for edge, old in enumerate(original.transitions))
    proposed = replace(original, segments=tuple(segments), transitions=transitions) if changed else original
    if segment_frame_counts(proposed) != segment_frame_counts(original):
        raise ValueError('global union changed the full renderer frame clock')
    assert_native_cut_events_preserved(original, proposed, domain)
    old_cuts, new_cuts = visible_group_cut_inventory(original, domain), visible_group_cut_inventory(proposed, domain)
    if tuple((e.first_incoming_output_frame, e.kind) for e in old_cuts) != tuple(
            (e.first_incoming_output_frame, e.kind) for e in new_cuts):
        raise ValueError('global union changed the externally owned actual visible cut cadence')
    # Rebind EVERY original actual-shot clock, not merely mutable endpoints.
    before_samples = native_frame_clocks_for_output(original, domain.clocks_by_request,
        range(original.output_frame_count))
    after_samples = native_frame_clocks_for_output(proposed, domain.clocks_by_request,
        range(proposed.output_frame_count))
    mutable_ticks = {tick for first, end in allocation.mutable_output_intervals for tick in range(first, end)}
    if any(before_samples[tick] != after_samples[tick] for tick in before_samples if tick not in mutable_ticks):
        raise ValueError('global union changed an unowned native source occurrence/sampler phase')
    for camera in cameras:
        if camera_unit_from_plan(program=program, unit_id=camera.unit_id,
                plan=proposed, domain=domain) != camera:
            raise ValueError('global union camera/native phase differs from the observed whole candidate')
    from .group_performer_family_owner import retained_applied_look_edges
    look_edges = retained_applied_look_edges(context.incumbent)
    support = derive_support(original, proposed, retained_applied_look_edges=look_edges)
    return NumericOwnedSceneUnion(proposed, allocation, cameras, changed, _dump(support))
