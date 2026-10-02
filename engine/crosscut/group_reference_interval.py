"""Compare liked GROUP intervals on the renderer's exact output grid.

This is a bounded regression diagnostic, not a selection gate or a quality
score. Segment indices/reason strings may change after native-epoch splitting;
the source sampler, per-tick camera and look must be compared instead. No file
reads, models or encoding occur here. Clocks are verified by the caller.
Equal planned parameters do NOT prove equal decoded/encoded pixels, source
content freshness, moving-pose identity or professional editing quality.
"""
from bisect import bisect_right
from dataclasses import asdict, dataclass
import math

from .render import RenderPlan, SourceLookAdjustment, segment_frame_counts
from .render import _base_crop_values, _boundary_look_values
from .render_boundary_clock import native_frame_clocks_for_output


@dataclass(frozen=True, slots=True)
class GroupReferenceFrameChange:
    output_frame_index: int
    source_path_changed: bool
    native_sample_changed: bool
    source_clock_evidence_changed: bool
    framing_parameters_changed: bool
    look_parameters_changed: bool


@dataclass(frozen=True, slots=True)
class GroupReferenceIntervalComparison:
    output_intervals: tuple[tuple[int, int], ...]
    compared_output_frame_count: int
    audio_reference_parameters_equal: bool
    changes: tuple[GroupReferenceFrameChange, ...]
    terminal_clone_output_frames: tuple[int, ...]

    @property
    def planned_parameters_preserved(self):
        return not self.changes and self.audio_reference_parameters_equal

    def to_dict(self):
        return dict(asdict(self), contract='exact_group_reference_interval_parameter_comparison_v1',
            planned_parameters_preserved=self.planned_parameters_preserved,
            changed_output_frame_count=len(self.changes),
            exact_renderer_native_schedule_used=True,
            raw_source_content_freshness_verified=False,
            encoded_pixel_identity_verified=False, quality_approved=False,
            candidate_admitted=False, automatic_selection_gate=False)


def _starts(plan):
    result = [0]
    for count in segment_frame_counts(plan):
        result.append(result[-1]+count)
    return tuple(result)


def _frame_parameters(plan, starts, tick):
    index = bisect_right(starts, tick)-1
    segment = plan.segments[index]
    time = (tick-starts[index])/plan.fps
    camera = segment.editorial_transform
    if camera is None:
        transform = (1., .5, .5, 0., 1.)
    else:
        value = camera.evaluate(time)
        transform = (value.zoom, value.center_x, value.center_y, value.roll_degrees,
            camera.effective_sampling_canvas_zoom)
    static = segment.look_adjustment or SourceLookAdjustment()
    look = (static.brightness, static.contrast, static.saturation,
        static.temperature_balance, static.tint_balance, static.detail_enhancement,
        *_boundary_look_values(segment.boundary_look_track, time))
    return (*_base_crop_values(segment, time), *transform), look


def _different(left, right, tolerance):
    return any(abs(a-b) > tolerance for a, b in zip(left, right, strict=True))


def compare_group_reference_interval(*, reference, candidate, reference_source_clocks,
        candidate_source_clocks, output_intervals, maximum_output_frames=16384,
        maximum_complete_plan_frames=36000, numeric_tolerance=1e-9):
    """Report numeric regressions, without banning a tested better alternative.

    Ranges are unique, ordered, disjoint [first, end) OUTPUT ticks. A source
    index alone is not a file identity. A canonical-cell split does not become
    a change if the renderer preserves its global sampler and numeric tracks.
    A cloned terminal raster stays explicit, never new motion evidence.
    """
    if not isinstance(reference, RenderPlan) or not isinstance(candidate, RenderPlan):
        raise TypeError('GROUP interval comparison needs two complete RenderPlans')
    if (reference.fps != candidate.fps or reference.width != candidate.width
            or reference.height != candidate.height or reference.aspect_ratio != candidate.aspect_ratio
            or reference.output_frame_count != candidate.output_frame_count):
        raise ValueError('GROUP reference comparison needs the same complete output grid')
    if (any(t.kind.uses_overlap for p in (reference, candidate) for t in p.transitions)
            or reference.authorized_shot_video_clocks or candidate.authorized_shot_video_clocks):
        raise ValueError('GROUP reference comparison needs the supported pure hard-cut sampler')
    if (type(maximum_output_frames) is not int or not 1 <= maximum_output_frames <= 16384
            or type(maximum_complete_plan_frames) is not int or not 1 <= maximum_complete_plan_frames <= 36000
            or type(numeric_tolerance) not in (int, float) or not math.isfinite(numeric_tolerance)
            or not 0 <= numeric_tolerance <= 1e-6 or not isinstance(output_intervals, tuple)
            or not output_intervals):
        raise ValueError('GROUP reference interval policy must be explicit and bounded')
    # The renderer scheduler materializes each selected render unit's complete
    # sampler. A tiny liked range must not hide an arbitrarily long unit.
    if reference.output_frame_count > maximum_complete_plan_frames:
        raise ValueError('GROUP reference complete-plan sampler exceeds its explicit frame budget')
    previous_end, ticks = -1, []
    for interval in output_intervals:
        if (not isinstance(interval, tuple) or len(interval) != 2
                or any(type(t) is not int for t in interval)):
            raise ValueError('GROUP reference intervals must contain exact integer output ticks')
        first, end = interval
        if not 0 <= first < end <= reference.output_frame_count or first < previous_end:
            raise ValueError('GROUP reference intervals must be ordered disjoint interior output ranges')
        if len(ticks)+end-first > maximum_output_frames:
            raise ValueError('GROUP reference interval exceeds its explicit frame budget')
        ticks.extend(range(first, end))
        previous_end = end
    old_clocks = native_frame_clocks_for_output(reference, reference_source_clocks, ticks)
    new_clocks = native_frame_clocks_for_output(candidate, candidate_source_clocks, ticks)
    old_starts, new_starts = _starts(reference), _starts(candidate)
    changes, clones = [], []
    for tick in ticks:
        old, new = old_clocks[tick], new_clocks[tick]
        old_clock, new_clock = reference_source_clocks[old.source_index], candidate_source_clocks[new.source_index]
        framing_a, look_a = _frame_parameters(reference, old_starts, tick)
        framing_b, look_b = _frame_parameters(candidate, new_starts, tick)
        row = GroupReferenceFrameChange(tick,
            reference.sources[old.source_index] != candidate.sources[new.source_index],
            (old.native_frame_index, old.terminal_clone) != (new.native_frame_index, new.terminal_clone),
            old_clock != new_clock, _different(framing_a, framing_b, numeric_tolerance),
            _different(look_a, look_b, numeric_tolerance))
        if any((row.source_path_changed, row.native_sample_changed, row.source_clock_evidence_changed,
                row.framing_parameters_changed, row.look_parameters_changed)):
            changes.append(row)
        if old.terminal_clone or new.terminal_clone:
            clones.append(tick)
    audio_equal = (reference.sources[reference.reference_source] == candidate.sources[candidate.reference_source]
        and abs(reference.master_audio_start-candidate.master_audio_start) <= numeric_tolerance)
    return GroupReferenceIntervalComparison(output_intervals, len(ticks), audio_equal,
        tuple(changes), tuple(clones))
