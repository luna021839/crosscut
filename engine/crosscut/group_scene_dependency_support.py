"""Numeric read/write support from the actual ordinary GROUP renderer.

This is a dependency contract, not a multi-repair selector or media approval.
Disjoint writes may share read-only evidence. Write/read conflicts require
joint recompletion, not concatenated local receipts. Global owned mutable
epochs and authorized per-shot video clocks still need their own compiler.
"""
from dataclasses import replace
from hashlib import sha256
import json

from .render import RenderPlan, segment_frame_counts, _video_render_units

CONTRACT = 'renderer_derived_scene_dependency_support_v1'


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def _plan_digest(plan):
    value = dict(plan.to_json())
    value.pop('output',None)  # Immutable final replay can relocate output only.
    return _digest(value)


def cell_intervals(cells):
    """Canonical sorted half-open cell intervals, not a min/max hull."""
    cells = tuple(cells)
    if (any(type(i) is not int or i < 0 for i in cells)
            or tuple(sorted(set(cells))) != cells):
        raise ValueError('dependency cells must be sorted distinct nonnegative integers')
    result = []
    for cell in cells:
        if result and cell == result[-1][1]:
            result[-1][1] += 1
        else:
            result.append([cell,cell+1])
    return result


def _clock_units(plan,counts):
    offsets = [0]
    for count in counts: offsets.append(offsets[-1]+count)
    return [dict(cells=list(range(unit.first_segment_index,unit.last_segment_index+1)),
        first_output_frame=offsets[unit.first_segment_index],
        end_output_frame=offsets[unit.last_segment_index+1],frame_count=unit.frame_count,
        request_source_index=unit.segments[0].source_index,
        source_start=unit.segments[0].source_start,source_end=unit.segments[-1].source_end,
        source_playback_rate=unit.segments[0].source_playback_rate,
        pixel_identity_coalesced=unit.coalesced_segment is not None)
        for unit in _video_render_units(plan,counts)]


def derive_support(before,after,*,retained_applied_look_edges):
    """Derive BEFORE descriptor/look work; no decoding or model inference.

    Descriptor and incident look writes remain distinct from the original
    applied-look proof reads. Closing over BOTH renderer clock partitions can
    reveal a conflict even when two nominal changed intervals do not overlap.
    Original/candidate frame counts are global renderer values, never per-cell
    round(duration*fps). This ordinary linear-clock slice declines authored
    shot-video clocks rather than silently granting them a dependency proof.
    """
    if (not isinstance(before,RenderPlan) or not isinstance(after,RenderPlan)
            or len(before.segments) != len(after.segments) or before.aspect_ratio.value != '16:9'
            or before.authorized_shot_video_clocks or after.authorized_shot_video_clocks
            or any(t.kind.uses_overlap for plan in (before,after) for t in plan.transitions)
            or replace(after,segments=before.segments,transitions=before.transitions,
                output=before.output) != before):
        raise ValueError('dependency support needs unchanged ordinary GROUP renderer/grid topology')
    counts = segment_frame_counts(before)
    if segment_frame_counts(after) != counts:
        raise ValueError('dependency support changed the global renderer frame counts')
    size = len(before.segments)
    if (not isinstance(retained_applied_look_edges,tuple)
            or any(type(i) is not int or not 0 <= i < size-1 for i in retained_applied_look_edges)
            or tuple(sorted(set(retained_applied_look_edges))) != retained_applied_look_edges):
        raise ValueError('retained look dependency edges must be an explicit canonical domain')
    changed = tuple(i for i,(a,b) in enumerate(zip(before.segments,after.segments,strict=True)) if a != b)
    edges = tuple(sorted({edge for cell in changed for edge in (cell-1,cell) if 0 <= edge < size-1}))
    if any(a != b and edge not in edges for edge,(a,b) in enumerate(
            zip(before.transitions,after.transitions,strict=True))):
        raise ValueError('dependency support changed an unrelated clock/camera/transition outside its incident scope')
    descriptor = tuple(sorted({i for cell in changed for i in (cell-1,cell,cell+1) if 0 <= i < size}))
    look_writes = tuple(sorted({cell for edge in edges for cell in (edge,edge+1)}))
    retained_edges = tuple(edge for edge in retained_applied_look_edges if edge not in edges)
    look_reads = tuple(sorted({cell for edge in retained_edges for cell in (edge,edge+1)}))
    writes = tuple(sorted(set(descriptor)|set(look_writes)))
    read = set(writes)|set(look_reads)
    original_units,candidate_units = _clock_units(before,counts),_clock_units(after,counts)
    # Partitions can overlap differently. One pass is insufficient when a unit
    # added by one plan touches a further unit in the other plan.
    while True:
        expanded = read|{cell for unit in (*original_units,*candidate_units)
            if read.intersection(unit['cells']) for cell in unit['cells']}
        if expanded == read: break
        read = expanded
    result = dict(contract=CONTRACT,incumbent_plan_sha256=_plan_digest(before),
        candidate_plan_sha256=_plan_digest(after),global_output_frame_counts=list(counts),
        numeric_write_cells=list(changed),descriptor_refresh_cells=list(descriptor),
        descriptor_refresh_intervals=cell_intervals(descriptor),incident_look_edges=list(edges),
        incident_look_write_cells=list(look_writes),retained_applied_look_edges=list(retained_applied_look_edges),
        retained_look_proof_edges=list(retained_edges),retained_look_read_cells=list(look_reads),
        original_clock_units=original_units,candidate_clock_units=candidate_units,
        read_cells=sorted(read),write_cells=list(writes),read_cell_intervals=cell_intervals(sorted(read)),
        renderer_clock_closure_from_both_plans=True,dependency_cells_are_not_minmax_hull=True,
        read_read_overlap_is_not_write_conflict=True,source_clock_topology_verified_by_assembler=False,
        physical_boundary_semantics_verified=False,multiple_repairs_selected_or_completed=False,
        process_peak_memory_or_completion_reserve_verified=False,quality_approved=False)
    result['support_sha256'] = _digest(result)
    return result


def validate_support(value,before,after,*,retained_applied_look_edges):
    if value != derive_support(before,after,retained_applied_look_edges=retained_applied_look_edges):
        raise ValueError('scene dependency support differs from independent renderer derivation')


def dependency_conflicts(a,b):
    """Pure planning policy only; an owner must validate both preimages first."""
    for value in (a,b):
        for key in ('read_cells','write_cells'): cell_intervals(value[key])
    return sorted((set(a['write_cells']) & (set(b['read_cells'])|set(b['write_cells'])))
        | (set(b['write_cells']) & set(a['read_cells'])))
