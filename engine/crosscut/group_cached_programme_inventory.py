"""Whole-song physical runs from existing, exact source-local observations.

This establishes a numeric composition baseline, not anonymous actor or pose
continuity. In particular a legacy dominant-body descriptor is never promoted
to the actor behind a dominant face. Its selected runs still need the normal
actor/camera producer and final pixel validation before delivery.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from statistics import mean
from time import monotonic

from .group_cached_native_observations import CachedNativeInventory, CachedNativeObservation
from .group_native_motion_programme import ProgrammeNativeMotionSide, _aligned_parts, _immutable_parts
from .group_review_interval_compile import output_offsets
from .group_source_domain import GroupSourceDomain
from .group_whole_programme_search import ProgrammeOpportunity, ProgrammeRun
from .render import RenderPlan, RenderSegment, RenderTransition, _video_render_units
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import sampler_frame_schedule


@dataclass(frozen=True, slots=True)
class CachedProgrammeRecipe:
    run: ProgrammeRun
    side: ProgrammeNativeMotionSide | None
    native_frame_indices: tuple[int, ...]
    sampled_observations: tuple
    protected_reference_segment_index: int | None = None
    aligned_segment: RenderSegment | None = None

    @property
    def observed_fraction(self):
        return sum(row is not None for row in self.sampled_observations) / len(self.sampled_observations)


@dataclass(frozen=True, slots=True)
class CachedProgrammeInventory:
    opportunities: tuple[ProgrammeOpportunity, ...]
    recipes: tuple[CachedProgrammeRecipe, ...]
    attempted_runs: int
    physical_cut_crossings: int
    elapsed_seconds: float

    @property
    def runs(self):
        return tuple(recipe.run for recipe in self.recipes)

    def evidence(self):
        return dict(contract='cached_source_native_variable_programme_inventory_v1',
            opportunity_count=len(self.opportunities), candidate_run_count=len(self.recipes),
            attempted_runs=self.attempted_runs, physical_cut_crossings=self.physical_cut_crossings,
            elapsed_seconds=self.elapsed_seconds, new_model_calls=0, new_pixel_decodes=0,
            original_cells_are_not_the_free_edit_grid=True, pose_is_face_owned=False,
            actor_correspondence_established=False, camera_recipes_complete=False,
            missing_observations_are_not_source_vetoes=True, professional_quality_approved=False)


@dataclass(frozen=True, slots=True)
class _NumericSourceUnitView:
    """Only renderer unit arithmetic; not a partial RenderPlan or authority."""
    segments: tuple[RenderSegment, ...]
    transitions: tuple[RenderTransition, ...]
    fps: float


def _unary(rows, duration_seconds):
    known = tuple(row for row in rows if row is not None)
    # An unknown shot is neither a proven wide nor a proven close-up. Missing
    # samples carry a finite uncertainty cost and do not erase this source.
    per_second = .65 if not known else mean(
        .55*(1-row.quality) + .20*(1-row.composition)
        + (.30 if row.shot_scale_basis == 'unknown_no_subject_geometry' or row.shot_type == 'unknown' else
            .65 if row.shot_type in ('group_wide', 'wide') else
            .28 if row.shot_type == 'medium' else .02)
        + .35*row.obstruction + .20*row.clipping for row in known)
    unknown_fraction = 1-len(known)/len(rows)
    return duration_seconds*(per_second+.25*unknown_fraction) + (
        5.*max(0., .55-duration_seconds)**2 + .18*max(0., duration_seconds-2.6)**2)


def build_cached_programme_inventory(*, original, domain, inventories, opportunities,
        protected_spans=(), duration_targets_seconds=(.75, 1.15, 1.6, 2.2, 2.8, 3.4, 4.8),
        maximum_runs=32768, maximum_seconds=45., cancelled=None):
    """Freeze variable spans and their exact production native-frame grids.

    A candidate may span several music/opportunity nodes but not a measured
    broadcaster cut. All missing measurements stay inside its finite score.
    Only protected pieces retain the predecessor's recipes; every other span
    is authored from the global audio/time-warp domain, not old cell choices.
    This cache-only baseline intentionally has no fabricated appearance path
    or phase benefit. The joint producer can replace it with measured recipes.
    """
    if (not isinstance(original, RenderPlan) or not isinstance(domain, GroupSourceDomain)
            or original.sources != domain.request_sources or original.authorized_shot_video_clocks
            or any(t.kind.uses_overlap for t in original.transitions)
            or not isinstance(inventories, dict) or set(inventories) != set(domain.eligible_indices)
            or any(not isinstance(i, CachedNativeInventory) for i in inventories.values())
            or not isinstance(opportunities, tuple) or not 2 <= len(opportunities) <= 4096
            or any(not isinstance(o, ProgrammeOpportunity) for o in opportunities)
            or tuple(o.output_tick for o in opportunities) != tuple(sorted({o.output_tick for o in opportunities}))
            or opportunities[0].output_tick != 0 or opportunities[-1].output_tick != original.output_frame_count
            or not isinstance(duration_targets_seconds, tuple) or not 1 <= len(duration_targets_seconds) <= 12
            or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 < t <= 30
                for t in duration_targets_seconds)
            or type(maximum_runs) is not int or not 1 <= maximum_runs <= 32768
            or type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds)
            or not 0 < maximum_seconds <= 60 or cancelled is not None and not callable(cancelled)
            or not isinstance(protected_spans, tuple)):
        raise ValueError('cached programme needs its exact finite original/source/opportunity inventory')
    started = monotonic()

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError('whole cached programme cancelled')
        if monotonic()-started > maximum_seconds:
            raise TimeoutError('whole cached programme original inventory budget exhausted; no retry')

    check()
    ticks = tuple(o.output_tick for o in opportunities)
    previous = 0
    for span in protected_spans:
        if (not isinstance(span, tuple) or len(span) != 2 or any(type(t) is not int for t in span)
                or not previous <= span[0] < span[1] <= original.output_frame_count
                or not set(span) <= set(ticks)):
            raise ValueError('cached programme protection escaped its declared output nodes')
        previous = span[1]
    rows_by_source = {}
    cuts_by_source = {}
    for source, inventory in inventories.items():
        clock = domain.clocks_by_request[source]
        rows = {}
        content = None
        for row in inventory.observations:
            check()
            if (not isinstance(row, CachedNativeObservation)
                    or row.source_index != source or row.native_frame_rate != clock.frame_rate
                    or not 0 <= row.native_frame_index < clock.frame_count
                    or row.native_frame_index in rows
                    or content is not None and row.source_sha256 != content
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in
                        (row.quality, row.composition, row.obstruction, row.clipping))
                    or row.shot_type not in ('unknown', 'group_wide', 'wide', 'medium', 'close_up')):
                raise ValueError('cached programme observation changed its exact source/native binding')
            content = row.source_sha256
            rows[row.native_frame_index] = row
        rows_by_source[source] = rows
        cuts_by_source[source] = tuple(math.ceil(t*float(clock.frame_rate)-1e-9)
            for t in domain.cuts_for_request(source))
    offsets = output_offsets(original)
    recipes = []
    attempted = crossings = 0

    def append(recipe):
        if len(recipes) >= maximum_runs:
            raise ValueError('whole cached programme run inventory exceeds its original bound')
        recipes.append(recipe)

    # Frozen protected cells are explicit graph edges, never self-signed pixel
    # certificates. Their renderer realization must verify the external root.
    for a, b in protected_spans:
        boundaries = (a, *(t for t in offsets if a < t < b), b)
        for first, end in zip(boundaries, boundaries[1:]):
            if first not in ticks or end not in ticks or end-first < 2:
                raise ValueError('protected source recipe has no exact realizable graph edge')
            segment = bisect_right(offsets, first)-1
            source = original.segments[segment].source_index
            actual = native_frame_clocks_for_output(original, domain.clocks_by_request, range(first, end))
            grid = tuple(actual[t].native_frame_index for t in range(first, end))
            if any(actual[t].terminal_clone for t in range(first, end)):
                raise ValueError('protected original contains a terminal clone, not observed media')
            rows = tuple(rows_by_source[source].get(n) for n in grid)
            run = ProgrammeRun(f'protected-{first}-{end}-{segment}', first, end, source,
                bisect_right(cuts_by_source[source], grid[0]), None,
                f'original-segment-{segment}', 0, _unary(rows, (end-first)/original.fps), True)
            parts = _immutable_parts(original, first, end)
            if len(parts) != 1:
                raise ValueError('protected graph edge changed its frozen renderer cell')
            append(CachedProgrammeRecipe(run, None, grid, rows, segment, parts[0]))
    for position, first in enumerate(ticks[:-1]):
        check()
        if any(a <= first < b for a, b in protected_spans):
            continue
        available = tuple(t for t in ticks[position+1:]
            if 2 <= t-first <= max(duration_targets_seconds)*original.fps
            and not any(first < b and t > a for a, b in protected_spans))
        if not available:
            continue
        # Include the earliest native/music opportunities for short physical
        # source shots. Duration is a soft cost, not a global eligibility gate.
        ends = set(available[:2])
        ends.update(min(available, key=lambda t: (abs((t-first)/original.fps-target), t))
            for target in duration_targets_seconds)
        for end in sorted(ends):
            for source in sorted(domain.eligible_indices):
                check()
                attempted += 1
                side = ProgrammeNativeMotionSide(first, end, source)
                segment = _aligned_parts(original, domain, side)[0]
                clock = domain.clocks_by_request[source]
                try:
                    schedule = sampler_frame_schedule(source_start=segment.source_start,
                        source_playback_rate=segment.source_playback_rate,
                        native_fps=float(clock.frame_rate), output_fps=original.fps,
                        frame_count=end-first, source_frame_count=clock.frame_count)
                except ValueError:
                    continue  # Physical coverage only, never missing face/pose.
                grid = schedule.source_frame_indices
                native_run = bisect_right(cuts_by_source[source], grid[0])
                if bisect_right(cuts_by_source[source], grid[-1]) != native_run:
                    crossings += 1
                    continue
                rows = tuple(rows_by_source[source].get(n) for n in grid)
                run = ProgrammeRun(f'cached-{source}-{first}-{end}', first, end, source,
                    native_run, None, 'neutral-source-native-camera', 0,
                    _unary(rows, (end-first)/original.fps))
                append(CachedProgrammeRecipe(run, side, grid, rows, aligned_segment=segment))
    return CachedProgrammeInventory(opportunities, tuple(recipes), attempted, crossings, monotonic()-started)


def cached_composition_edge_cost(recipes):
    """Finite baseline seam cost; no unowned body or invented actor match."""
    by_identity = {r.run.identity: r for r in recipes}

    def edge(history, incoming):
        if not history:
            return 0.
        previous = history[-1]
        a, b = by_identity[previous.identity], by_identity[incoming.identity]
        same = previous.request_source_index == incoming.request_source_index
        cost = 1.4 if not same else .8  # Unknown cross-source identity is not a match.
        left, right = a.sampled_observations[-1], b.sampled_observations[0]
        if left is not None and right is not None and left.face_bbox and right.face_bbox:
            ax, ay, aw, ah = left.face_bbox
            bx, by, bw, bh = right.face_bbox
            if min(ah, bh) > 0:
                cost += 1.3*abs(math.log(bh/ah)) + 2.*math.hypot(
                    bx+bw/2-ax-aw/2, (by+bh/2-ay-ah/2)*9/16)
        elif (left is None or right is None):
            cost += .35
        if same:
            gap = b.native_frame_indices[0]-a.native_frame_indices[-1]
            if gap < 0:
                cost += 3.  # Measured video-time reversal, never silently merged.
            if previous.native_camera_run_index == incoming.native_camera_run_index:
                cost += 1.2  # No new shot is created by splitting a continuous hold.
        elif len(history) > 1 and history[-2].request_source_index == incoming.request_source_index:
            cost += .45
        return cost

    return edge


def cached_programme_physical_compatibility(*, original, domain, recipes):
    """Check shared decoder units BEFORE scoring a complete source-run history.

    Independently authored warp spans can be individually valid yet change
    native pixels when the renderer joins them into a linear same-source unit.
    This is a numeric clock conflict, not an identity/anatomy quality gate.
    Never select those separate grids then silently fit a new combined clock.
    """
    from functools import lru_cache
    by_identity = {r.run.identity: r for r in recipes}

    @lru_cache(maxsize=8192)
    def compatible_suffix(identities):
        frozen = tuple(by_identity[key] for key in identities)
        if any(r.aligned_segment is None for r in frozen):
            raise ValueError('physical programme recipe lost its aligned renderer segment')
        segments = tuple(recipe.aligned_segment for recipe in frozen)
        plan = _NumericSourceUnitView(segments,
            tuple(RenderTransition(tier='C', reason='whole-programme-numeric-clock')
                for _ in range(len(segments)-1)), original.fps)
        counts = tuple(r.run.end_output_frame-r.run.first_output_frame for r in frozen)
        # Keep the GLOBAL fractional cut phase. Rebasing a two-frame suffix
        # onto zero can turn it into one frame although its full-song interval
        # is valid. This is the same cumulative rounding as the real renderer.
        cursor = segments[0].master_start
        previous = round(cursor*original.fps)
        if previous != frozen[0].run.first_output_frame:
            return False
        for recipe in frozen:
            cursor += recipe.aligned_segment.duration
            boundary = round(cursor*original.fps)
            if boundary-previous != recipe.run.end_output_frame-recipe.run.first_output_frame:
                return False
            previous = boundary
        for unit in _video_render_units(plan, counts):
            first, last = unit.segments[0], unit.segments[-1]
            clock = domain.clocks_by_request[first.source_index]
            rate = (unit.coalesced_segment.source_playback_rate if unit.coalesced_segment is not None else
                (last.source_end-first.source_start)/sum(s.duration for s in unit.segments))
            schedule = sampler_frame_schedule(source_start=first.source_start, source_playback_rate=rate,
                native_fps=float(clock.frame_rate), output_fps=plan.fps, frame_count=unit.frame_count,
                source_frame_count=clock.frame_count)
            expected = tuple(n for r in frozen[unit.first_segment_index:unit.last_segment_index+1]
                for n in r.native_frame_indices)
            if schedule.source_frame_indices != expected:
                return False
        return True

    def compatible(history, incoming):
        suffix = [incoming]
        for previous in reversed(history):
            if previous.request_source_index != incoming.request_source_index:
                break
            suffix.append(previous)
        return compatible_suffix(tuple(r.identity for r in reversed(suffix)))

    return compatible
