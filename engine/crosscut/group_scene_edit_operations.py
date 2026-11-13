"""Numeric KEEP/CAMERA/SOURCE/SOURCE_AND_CAMERA ownership for actual shots.

This classifies an externally owned original and one assembled numeric plan.
It does not select subjects, grant original duties, score diversity, complete
a bundle or certify quality. In particular a new camera on the SAME source
occurrence is not a new broadcaster cut. The current completion slice remains
local even though this inventory describes all original actual shots.
"""
from dataclasses import asdict, fields
from enum import Enum
from fractions import Fraction

from .group_actual_shots import visible_group_shots, visible_group_cut_inventory
from .group_joint_native_preparation import _plan_identity
from .group_source_domain import GroupSourceDomain
from .render import RenderPlan, RenderSegment, segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import VerifiedRenderFrameClock

CONTRACT = 'original_owned_actual_shot_numeric_edit_operations_v1'
SOURCE_FIELDS = ('source_index', 'source_start', 'source_duration')
CAMERA_FIELDS = ('crop', 'pan_crop', 'editorial_transform', 'outgoing_overlap_transform')
COLOR_FIELDS = ('look_adjustment', 'boundary_look_track')
OUTPUT_FIELDS = ('master_start', 'duration')


class SceneEditOperation(str, Enum):
    KEEP = 'KEEP'
    CAMERA = 'CAMERA'
    SOURCE = 'SOURCE'
    SOURCE_AND_CAMERA = 'SOURCE_AND_CAMERA'


def segment_edit_operation(original, proposed):
    """Ignore hypothesis IDs and color, never an unclassified render input."""
    if not isinstance(original, RenderSegment) or not isinstance(proposed, RenderSegment):
        raise ValueError('scene operation requires actual render segments')
    if {f.name for f in fields(original)} != set(SOURCE_FIELDS+CAMERA_FIELDS+COLOR_FIELDS+OUTPUT_FIELDS):
        raise ValueError('new render inputs need an explicit scene operation classification')
    if any(getattr(original, key) != getattr(proposed, key) for key in OUTPUT_FIELDS):
        raise ValueError('scene edit changed an original output cell')
    source = any(getattr(original, key) != getattr(proposed, key) for key in SOURCE_FIELDS)
    camera = any(getattr(original, key) != getattr(proposed, key) for key in CAMERA_FIELDS)
    return (SceneEditOperation.SOURCE_AND_CAMERA if source and camera else
        SceneEditOperation.SOURCE if source else SceneEditOperation.CAMERA if camera else SceneEditOperation.KEEP)


def path_has_numeric_edit(original_states, proposed_states):
    """Eligibility before edge/dense work; a different face binding is no edit."""
    if len(original_states) != len(proposed_states):
        raise ValueError('scene path changed its owned unit inventory')
    result = False
    for old, new in zip(original_states, proposed_states, strict=True):
        if (old.first_output_frame, old.end_output_frame) != (new.first_output_frame, new.end_output_frame):
            raise ValueError('scene path changed an original output interval')
        if len(old.segments) != len(new.segments):
            raise ValueError('scene path changed an original cell inventory')
        for a, b in zip(old.segments, new.segments, strict=True):
            result |= segment_edit_operation(a, b) != SceneEditOperation.KEEP
    return result


def numeric_replay_domain(plan, payload):
    """Materialized plan clock domain, NOT a replacement alignment authority.

    Native sampler reconstruction only needs the eligible order, verified PTS
    grids and native cuts. The ordinary parent still binds the complete original
    alignment domain separately; no invented zero offsets are alignment proof.
    """
    clocks = tuple(VerifiedRenderFrameClock(Fraction(row['frame_rate']), row['frame_count'],
        row['packet_grid_sha256'], row['grid_basis']) for row in payload['render_frame_clocks'])
    indices = tuple(payload['eligible_request_indices'])
    domain = GroupSourceDomain(tuple(plan.sources), indices, (0.,)*len(indices), (None,)*len(indices),
        tuple(tuple(cuts) for cuts in payload['scene_cut_times']), clocks)
    if [str(path) for path in domain.eligible_sources] != payload['eligible_sources']:
        raise ValueError('scene numeric replay changed the external eligible source order')
    return domain


def derive_scene_edit_operations(*, original, proposed, domain, mutable_intervals):
    """Full numeric inventory with explicit mutable intervals, not middle slots."""
    from .group_performer_family_owner import digest
    if (not isinstance(original, RenderPlan) or not isinstance(proposed, RenderPlan)
            or not isinstance(domain, GroupSourceDomain) or original.sources != domain.request_sources
            or proposed.sources != original.sources or not 1 <= original.output_frame_count <= 36000
            or segment_frame_counts(original) != segment_frame_counts(proposed)
            or len(original.segments) != len(proposed.segments)):
        raise ValueError('scene operations need one complete original output/source grid')
    # Sources/cells/transitions/output are dealt with separately. Keep every
    # other actual dataclass header input, including learned clock/overlap flags.
    if any(getattr(original, f.name) != getattr(proposed, f.name) for f in fields(original)
            if f.name not in ('segments', 'transitions', 'output')):
        raise ValueError('scene operation changed an original renderer header')
    shots = visible_group_shots(original, domain)
    bounds = tuple((s.first_output_frame, s.end_output_frame) for s in shots)
    if (not isinstance(mutable_intervals, tuple) or len(mutable_intervals) > 256
            or any(not isinstance(span, tuple) or len(span) != 2
                or any(type(tick) is not int for tick in span) for span in mutable_intervals)
            or len(set(mutable_intervals)) != len(mutable_intervals)
            or tuple(span for span in bounds if span in mutable_intervals) != mutable_intervals):
        raise ValueError('scene operations require explicit original actual-shot mutable intervals')
    old_cuts, new_cuts = visible_group_cut_inventory(original, domain), visible_group_cut_inventory(proposed, domain)
    if tuple((e.first_incoming_output_frame, e.kind) for e in old_cuts) != tuple(
            (e.first_incoming_output_frame, e.kind) for e in new_cuts):
        raise ValueError('scene operations changed actual visible cut cadence')
    before = native_frame_clocks_for_output(original, domain.clocks_by_request, range(original.output_frame_count))
    after = native_frame_clocks_for_output(proposed, domain.clocks_by_request, range(proposed.output_frame_count))
    offsets = [0]
    for count in segment_frame_counts(original): offsets.append(offsets[-1]+count)
    operations = tuple(segment_edit_operation(a, b) for a, b in zip(original.segments, proposed.segments, strict=True))
    mutable_ticks = {tick for first, end in mutable_intervals for tick in range(first, end)}
    for index, operation in enumerate(operations):
        if operation != SceneEditOperation.KEEP and any(tick not in mutable_ticks
                for tick in range(offsets[index], offsets[index+1])):
            raise ValueError('scene operation changed unowned output pixels/source occurrence')
    if any(before[tick] != after[tick] for tick in before if tick not in mutable_ticks):
        raise ValueError('scene operation changed an unowned native sampler phase')
    rows = []
    original_identity = _plan_identity(original)
    for first, end in bounds:
        cells = tuple(i for i in range(len(operations)) if offsets[i] < end and offsets[i+1] > first)
        source_changed = any(operations[i] in (SceneEditOperation.SOURCE, SceneEditOperation.SOURCE_AND_CAMERA)
            for i in cells) or any(before[tick] != after[tick] for tick in range(first, end))
        camera_changed = any(operations[i] in (SceneEditOperation.CAMERA, SceneEditOperation.SOURCE_AND_CAMERA)
            for i in cells)
        operation = (SceneEditOperation.SOURCE_AND_CAMERA if source_changed and camera_changed else
            SceneEditOperation.SOURCE if source_changed else SceneEditOperation.CAMERA if camera_changed else SceneEditOperation.KEEP)
        rows.append(dict(actual_shot_occurrence_id=digest(dict(original_plan_identity=original_identity,
            first_output_frame=first, end_output_frame=end)), first_output_frame=first, end_output_frame=end,
            mutable=(first, end) in mutable_intervals, operation=operation.value,
            original_request_source_index=before[first].source_index,
            selected_request_source_index=after[first].source_index,
            request_source_index_changed=before[first].source_index != after[first].source_index,
            source_occurrence_changed=source_changed, camera_changed=camera_changed))
    payload = dict(contract=CONTRACT, original_plan_identity=original_identity,
        proposed_geometry_identity=digest([{k:v for k,v in asdict(s).items()
            if k in SOURCE_FIELDS+CAMERA_FIELDS} for s in proposed.segments]),
        mutable_intervals=[list(span) for span in mutable_intervals], actual_shot_operations=rows,
        camera_only_program=any(row['camera_changed'] for row in rows) and not any(row['source_occurrence_changed'] for row in rows),
        has_numeric_edit=any(row['operation'] != SceneEditOperation.KEEP.value for row in rows),
        distinct_camera_transition_count_not_inferred=True,
        original_duties_redefined=False, request_budget_restarted=False, quality_approved=False)
    return dict(payload, operations_sha256=digest(payload))


def validate_scene_edit_operations(value, *, original, proposed, source_domain, mutable_intervals):
    expected = derive_scene_edit_operations(original=original, proposed=proposed,
        domain=numeric_replay_domain(original, source_domain), mutable_intervals=mutable_intervals)
    if value != expected:
        raise ValueError('scene edit operations differ from the externally owned source/camera programme')
    if not expected['has_numeric_edit']:
        raise ValueError('scene completion requires an actual numeric source/camera edit, not KEEP')
