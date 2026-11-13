"""Split actual GROUP epochs without resetting the global source sampler.

This is a sibling of the fixed-cell joint materializer, not a relaxation of
its contract. A split is numeric preparation: it neither selects a donor nor
publishes an incomplete bundle. Changed intervals and their incident descriptor
edges must be freshly observed before the returned timeline is useful.
"""
from dataclasses import dataclass, fields, replace

from .group_actual_shots import visible_group_cut_inventory
from .group_cut_camera_proposal import retime_base_pan
from .group_joint_native_preparation import _plan_identity
from .group_source_domain import GroupSourceDomain
from .group_source_span_proposal import refresh_fixed_group_render_observations
from .optional_native_worker import isolated_optional
from .render import (BoundaryLookTrack, EditorialTransformTrack, RenderPlan,
    RenderTransition, _base_crop_values, _boundary_look_values, segment_frame_counts)
from .render_boundary_clock import native_frame_clocks_for_output
from .visual_score import VisualTimelinePlan


@dataclass(frozen=True, slots=True)
class GroupIntervalCellMap:
    old_segment_index: int
    new_segment_index: int
    first_output_frame: int
    end_output_frame: int
    old_local_frame_offset: int


@dataclass(frozen=True, slots=True)
class GroupIntervalRenderScaffold:
    incumbent: RenderPlan
    plan: RenderPlan
    cell_map: tuple[GroupIntervalCellMap, ...]
    split_old_segment_indices: tuple[int, ...]
    refreshed_segment_indices: tuple[int, ...]

    def to_dict(self):
        return dict(contract='group_exact_native_epoch_split_scaffold_v1',
            incumbent_plan_identity=_plan_identity(self.incumbent),
            plan_identity=_plan_identity(self.plan),
            cell_map=[{f.name: getattr(row, f.name) for f in fields(row)} for row in self.cell_map],
            split_old_segment_indices=self.split_old_segment_indices,
            refreshed_segment_indices=self.refreshed_segment_indices,
            global_native_schedule_preserved=True,
            numeric_camera_and_look_samples_preserved=True,
            encoded_pixel_identity_verified=False,
            full_bundle_created=False, candidate_admitted=False, quality_approved=False)


@dataclass(frozen=True, slots=True)
class GroupIntervalRenderedObservations:
    scaffold: GroupIntervalRenderScaffold
    timeline: VisualTimelinePlan
    incumbent_timeline: VisualTimelinePlan
    unrebound_boundary_authority_old_indices: tuple[int, ...] = ()

    def to_dict(self):
        return dict(contract='group_split_epochs_fresh_rendered_observations_v1',
            scaffold=self.scaffold.to_dict(), visual_timeline=self.timeline.to_dict(),
            unrebound_boundary_authority_old_indices=self.unrebound_boundary_authority_old_indices,
            incumbent_boundary_support_preserved_in_incumbent_timeline=True,
            descriptors_refreshed=True, semantic_motion_evidence_verified=False,
            full_bundle_created=False, candidate_admitted=False, quality_approved=False)


def _slice_camera(track, *, offset_seconds, duration):
    if track is None:
        return None
    # Preserve the piecewise-linear function, including the mathematical cell
    # endpoint. A per-frame-only slice would hold its last sample too early and
    # could create a false Tier-D discontinuity at the next cell.
    times = sorted({0., duration, *(key.time_seconds-offset_seconds
        for key in track.keyframes if 0. < key.time_seconds-offset_seconds < duration)})
    canvas = track.effective_sampling_canvas_zoom
    keys = tuple(replace(track.evaluate(offset_seconds+t), time_seconds=t) for t in times)
    return EditorialTransformTrack(keys, reason=track.reason,
        sampling_canvas_zoom=(canvas if track.sampling_canvas_zoom is not None
            or max(key.zoom for key in keys) < canvas else None))


def _slice_look(track, *, offset_seconds, duration):
    if track is None:
        return None
    times = sorted({0., duration, *(key.time_seconds-offset_seconds
        for key in track.keyframes if 0. < key.time_seconds-offset_seconds < duration)})
    return BoundaryLookTrack(tuple(replace(track.evaluate(offset_seconds+t), time_seconds=t)
        for t in times), reason=track.reason)


def split_group_render_intervals(*, plan, domain, split_output_frames,
        maximum_additional_cells=16, maximum_replay_frames=36000):
    """Retain every old clock/pixel parameter while inserting epoch boundaries.

    Exact output ticks, NOT native PTS rounded to a new trim origin, define the
    split. Both sides still share the original linear renderer unit. The caller
    owns the external source/model freshness and later whole-bundle metadata.
    Authorized learned/local-clock receipts require a topology-aware owner and
    are explicitly unsupported here; they are never erased to pass validation.
    """
    if (not isinstance(plan, RenderPlan) or not isinstance(domain, GroupSourceDomain)
            or plan.sources != domain.request_sources
            or any(s.source_index not in domain.eligible_indices for s in plan.segments)
            or any(t.kind.uses_overlap for t in plan.transitions)):
        raise ValueError('GROUP epoch splitting needs its complete hard-cut source domain')
    if (type(maximum_additional_cells) is not int or not 1 <= maximum_additional_cells <= 64
            or type(maximum_replay_frames) is not int or not 1 <= maximum_replay_frames <= 36000
            or plan.output_frame_count > maximum_replay_frames):
        raise ValueError('GROUP epoch splitting exceeds its explicit topology/replay budget')
    if plan.authorized_shot_video_clocks:
        raise ValueError('GROUP epoch splitting needs refreshed authorized shot-clock topology')
    if any(s.outgoing_overlap_transform is not None for s in plan.segments):
        raise ValueError('GROUP epoch splitting cannot discard outgoing overlap cameras')
    if (not isinstance(split_output_frames, tuple)
            or any(type(t) is not int or not 0 < t < plan.output_frame_count for t in split_output_frames)
            or tuple(sorted(set(split_output_frames))) != split_output_frames):
        raise ValueError('GROUP epoch split ticks must be unique ordered interior output frames')
    counts = segment_frame_counts(plan)
    starts = [0]
    for count in counts:
        starts.append(starts[-1]+count)
    inserted = tuple(t for t in split_output_frames if t not in starts)
    if len(inserted) > maximum_additional_cells:
        raise ValueError('GROUP epoch splitting exceeds its additional-cell budget')
    if not inserted:
        rows = tuple(GroupIntervalCellMap(i, i, starts[i], starts[i+1], 0)
            for i in range(len(counts)))
        return GroupIntervalRenderScaffold(plan, plan, rows, (), ())
    before_samples = native_frame_clocks_for_output(plan, domain.clocks_by_request, range(starts[-1]))
    if any(sample.terminal_clone for sample in before_samples.values()):
        raise ValueError('terminal clones cannot become newly observed native epochs')
    new_segments, transitions, rows, split_old = [], [], [], []
    for old_index, (segment, count) in enumerate(zip(plan.segments, counts, strict=True)):
        first, end = starts[old_index:old_index+2]
        boundaries = (first, *(t for t in inserted if first < t < end), end)
        if len(boundaries) > 2:
            split_old.append(old_index)
        for a, b in zip(boundaries, boundaries[1:]):
            if b-a < 2:
                raise ValueError('GROUP epoch splitting cannot create a one-frame evidence cell')
            local_first, local_end = a-first, b-first
            # Keep the original quantized-cell duration/endpoints on the tail.
            # Only new interior boundaries are expressed in exact output ticks.
            offset = local_first/plan.fps
            stop = segment.duration if b == end else local_end/plan.fps
            duration = stop-offset
            if duration <= 0:
                raise ValueError('GROUP epoch split has a non-positive canonical interval')
            if a == first and b == end:
                part = segment
            else:
                part = replace(segment, source_start=segment.source_start+offset*segment.source_playback_rate,
                    master_start=segment.master_start+offset, duration=duration,
                    source_duration=(duration*segment.source_playback_rate
                        if segment.source_duration is not None else None),
                    pan_crop=retime_base_pan(segment.pan_crop, relative_shift=offset, duration=duration),
                    editorial_transform=_slice_camera(segment.editorial_transform,
                        offset_seconds=offset, duration=duration),
                    boundary_look_track=_slice_look(segment.boundary_look_track,
                        offset_seconds=offset, duration=duration))
            if new_segments:
                transitions.append(plan.transitions[old_index-1] if a == first else
                    RenderTransition(tier='C', reason='unapproved_numeric_epoch_split_continuation'))
            rows.append(GroupIntervalCellMap(old_index, len(new_segments), a, b, local_first))
            new_segments.append(part)
    candidate = replace(plan, segments=tuple(new_segments), transitions=tuple(transitions))
    if (candidate.output_frame_count != plan.output_frame_count
            or segment_frame_counts(candidate) != tuple(row.end_output_frame-row.first_output_frame for row in rows)
            or abs(candidate.duration-plan.duration) > 1e-9):
        raise ValueError('GROUP epoch splitting changed the global quantized output timeline')
    after_samples = native_frame_clocks_for_output(candidate, domain.clocks_by_request, range(starts[-1]))
    for tick, before in before_samples.items():
        after = after_samples[tick]
        if (before.source_index != after.source_index or before.native_frame_index != after.native_frame_index
                or before.terminal_clone != after.terminal_clone
                or abs(before.source_request_seconds-after.source_request_seconds) > 1e-9):
            raise ValueError('GROUP epoch split reset or changed a global native sampler tick')
    if visible_group_cut_inventory(candidate, domain) != visible_group_cut_inventory(plan, domain):
        raise ValueError('GROUP epoch split changed an actual native/source cut event')
    for row in rows:
        old, new = plan.segments[row.old_segment_index], candidate.segments[row.new_segment_index]
        for tick in range(row.end_output_frame-row.first_output_frame):
            old_time, new_time = (row.old_local_frame_offset+tick)/plan.fps, tick/plan.fps
            left = (*_base_crop_values(old, old_time), *_boundary_look_values(old.boundary_look_track, old_time))
            right = (*_base_crop_values(new, new_time), *_boundary_look_values(new.boundary_look_track, new_time))
            if any(abs(a-b) > 1e-9 for a, b in zip(left, right, strict=True)):
                raise ValueError('GROUP epoch split changed a numeric crop/look sample')
            if old.editorial_transform is not None:
                left, right = old.editorial_transform.evaluate(old_time), new.editorial_transform.evaluate(new_time)
                if any(abs(getattr(left, name)-getattr(right, name)) > 1e-9
                        for name in ('zoom', 'center_x', 'center_y', 'roll_degrees')) or (
                        old.editorial_transform.effective_sampling_canvas_zoom !=
                        new.editorial_transform.effective_sampling_canvas_zoom):
                    raise ValueError('GROUP epoch split changed a camera sample or sampling canvas')
    changed = {row.new_segment_index for row in rows if row.old_segment_index in split_old}
    refreshed = tuple(sorted({j for i in changed for j in (i-1, i, i+1) if 0 <= j < len(rows)}))
    return GroupIntervalRenderScaffold(plan, candidate, tuple(rows), tuple(split_old), refreshed)


@isolated_optional()
def refresh_split_group_render_observations(*, scaffold, timeline, domain, sync_scores,
        feature_cache_directory=None, feature_derived_cache_directory=None,
        require_persistent_feature_cache=False):
    """Consume the split topology and measure new descriptor anchors/edges.

    Unchanged canonical intervals are re-indexed without rewriting their
    support. Split cells are merely a clock scaffold until the existing fixed
    renderer-aware observer replaces their source-dependent descriptors.
    """
    if (not isinstance(scaffold, GroupIntervalRenderScaffold) or not isinstance(timeline, VisualTimelinePlan)
            or not isinstance(domain, GroupSourceDomain) or scaffold.incumbent.sources != domain.request_sources
            or len(timeline.choices) != len(scaffold.incumbent.segments)):
        raise ValueError('split GROUP observations require the complete incumbent timeline/domain')
    for i, (choice, segment) in enumerate(zip(timeline.choices, scaffold.incumbent.segments, strict=True)):
        if (choice.segment_index != i or domain.request_index(choice.source_index) != segment.source_index
                or any(abs(a-b) > 1e-6 for a, b in (
                    (choice.canonical_master_start, scaffold.incumbent.master_audio_start+segment.master_start),
                    (choice.canonical_master_end, scaffold.incumbent.master_audio_start+segment.master_start+segment.duration),
                    (choice.source_local_start, segment.source_start), (choice.source_local_end, segment.source_end)))):
            raise ValueError('split GROUP observations have a stale incumbent source/clock timeline')
    # A caller cannot supply arbitrary cells and rely on the DTO's flags.
    reconstructed = split_group_render_intervals(plan=scaffold.incumbent, domain=domain,
        split_output_frames=tuple(row.first_output_frame for row in scaffold.cell_map
            if row.old_local_frame_offset), maximum_additional_cells=64)
    if reconstructed != scaffold:
        raise ValueError('split GROUP observation scaffold differs from independently rebuilt topology')
    if not scaffold.split_old_segment_indices:
        return GroupIntervalRenderedObservations(scaffold, timeline, timeline)
    choices = []
    unrebound = set()
    incident = set(scaffold.refreshed_segment_indices)
    for row, segment in zip(scaffold.cell_map, scaffold.plan.segments, strict=True):
        old = timeline.choices[row.old_segment_index]
        updates = dict(segment_index=row.new_segment_index,
            canonical_master_start=scaffold.plan.master_audio_start+segment.master_start,
            canonical_master_end=scaffold.plan.master_audio_start+segment.master_start+segment.duration,
            source_local_start=segment.source_start, source_local_end=segment.source_end)
        if row.new_segment_index in incident:
            # Incoming boundary authority no longer describes this interval.
            # The observer intentionally preserves its outer-edge fields, so
            # clear those claims explicitly before asking for fresh pixels.
            updates.update({f.name: None for f in fields(old) if f.name.endswith('_from_previous')})
        if row.new_segment_index != row.old_segment_index and (
                old.transition_evidence_authority_from_previous is not None
                or old.transition_evidence_contract_from_previous is not None):
            # Canonical descriptors may be re-indexed, but an old index-bound
            # authority must not be relabelled as fresh. Keep the untouched
            # original timeline on the result for the complete owner to rebind.
            unrebound.add(row.old_segment_index)
            updates.update(transition_evidence_authority_from_previous=None,
                transition_evidence_contract_from_previous=None)
        choices.append(replace(old, **updates))
    clock_timeline = replace(timeline, choices=tuple(choices), candidate_review_summaries=(),
        analysis_basis='UNREFRESHED_split_native_epoch_scaffold_not_a_selected_timeline')
    # Disjoint epoch changes must not silently broaden into a whole-song
    # refresh. The declared inventory is a union of local neighborhoods, not
    # its convex hull. Keep the unchanged gap rows and their history intact.
    windows = []
    for index in sorted(incident):
        if windows and index == windows[-1][1]+1:
            windows[-1] = (windows[-1][0], index)
        else:
            windows.append((index, index))
    fresh = clock_timeline
    for first, last in windows:
        previous = fresh
        fresh = refresh_fixed_group_render_observations(plan=scaffold.plan, clock_timeline=previous,
            first_segment_index=first, last_segment_index=last, source_domain=domain,
            sync_scores=sync_scores, feature_cache_directory=feature_cache_directory,
            feature_derived_cache_directory=feature_derived_cache_directory,
            require_persistent_feature_cache=require_persistent_feature_cache)
        if (len(fresh.choices) != len(previous.choices)
                or any(new != old for index, (old, new) in enumerate(zip(previous.choices, fresh.choices))
                    if not first <= index <= last)):
            raise ValueError('split GROUP refresh changed observations outside its declared local window')
    if 'UNREFRESHED' in fresh.analysis_basis or len(fresh.choices) != len(scaffold.plan.segments):
        raise ValueError('split GROUP descriptor observer returned an unfinished timeline')
    return GroupIntervalRenderedObservations(scaffold, fresh, timeline, tuple(sorted(unrebound)))
