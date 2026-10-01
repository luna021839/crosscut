"""Bounded whole-shot video clocks, separate from shared audio alignment.

This is a numeric execution contract, NOT proof of choreography or lip sync.
The parent freezes context before search; a child cannot authorize itself by
putting a larger limit or a different alignment in its serialized receipt.
Descriptor, bundle, render and export consumers use the same actual clock.
"""
from dataclasses import dataclass, fields, replace
from fractions import Fraction
import hashlib
import json
import math

from .render_frame_schedule import VerifiedRenderFrameClock, sampler_frame_schedule


SHOT_VIDEO_CLOCK_CONTRACT = 'bounded_authorized_whole_shot_video_clock_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _sources_identity(plan):
    return _digest([str(path) for path in plan.sources])


def _incumbent_identity(plan):
    return _digest(plan.to_json())


def _unit(parts):
    from .render import _combined_identity_segment
    return parts[0] if len(parts) == 1 else _combined_identity_segment(parts)


@dataclass(frozen=True, slots=True)
class FrozenShotVideoClockContext:
    """Caller/parent-owned pre-search context; never reconstructed from child JSON."""
    incumbent_plan: object
    source_domain: object
    policy: object

    def __post_init__(self):
        from .group_source_domain import bind_group_source_domain
        from .group_joint_native_phase import NativePhaseSearchPolicy
        from .render import RenderPlan
        if (not isinstance(self.incumbent_plan, RenderPlan)
                or self.incumbent_plan.authorized_shot_video_clocks
                or not isinstance(self.policy, NativePhaseSearchPolicy)):
            raise ValueError('shot-clock context needs an unretimed incumbent and frozen phase policy')
        bind_group_source_domain(plan=self.incumbent_plan, source_domain=self.source_domain)


@dataclass(frozen=True, slots=True)
class AuthorizedShotVideoClockV1:
    incumbent_plan_sha256: str
    source_domain_sha256: str
    sources_sha256: str
    policy_sha256: str
    request_source_index: int
    eligible_source_index: int
    segment_indices: tuple[int, ...]
    master_audio_start: float
    reference_start: float
    reference_end: float
    first_output_frame: int
    end_output_frame: int
    output_fps: float
    aligned_source_start: float
    aligned_source_rate: float
    source_start: float
    source_rate: float
    first_delta_native_frames: int
    last_delta_native_frames: int
    maximum_absolute_offset_seconds: float
    maximum_rate_change_from_aligned: float
    native_clock: VerifiedRenderFrameClock
    aligned_native_indices: tuple[int, ...]
    native_indices: tuple[int, ...]

    def __post_init__(self):
        digests = (self.incumbent_plan_sha256, self.source_domain_sha256,
            self.sources_sha256, self.policy_sha256)
        numbers = (self.master_audio_start, self.reference_start, self.reference_end,
            self.output_fps, self.aligned_source_start, self.aligned_source_rate,
            self.source_start, self.source_rate, self.maximum_absolute_offset_seconds,
            self.maximum_rate_change_from_aligned)
        if (any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                for v in digests)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)
                or any(type(v) is not int for v in (self.request_source_index, self.eligible_source_index,
                    self.first_output_frame, self.end_output_frame, self.first_delta_native_frames,
                    self.last_delta_native_frames))
                or min(self.request_source_index, self.eligible_source_index, self.first_output_frame) < 0
                or not isinstance(self.segment_indices, tuple) or not self.segment_indices
                or any(type(v) is not int or v < 0 for v in self.segment_indices)
                or self.segment_indices != tuple(range(self.segment_indices[0], self.segment_indices[-1]+1))
                or self.output_fps <= 0 or self.master_audio_start < 0
                or self.reference_start < self.master_audio_start or self.reference_end <= self.reference_start
                or min(self.aligned_source_start, self.source_start) < 0
                or not 2 <= self.end_output_frame-self.first_output_frame <= 300
                or not 0 < self.maximum_absolute_offset_seconds <= .15
                or not 0 < self.maximum_rate_change_from_aligned <= .02
                or max(abs(self.aligned_source_rate-1), abs(self.source_rate-1)) > .02+1e-12
                or not isinstance(self.native_clock, VerifiedRenderFrameClock)
                or self.native_clock.grid_basis not in ('exact_cfr_presentation_grid',
                    'bijective_nearest_index_presentation_grid')):
            raise ValueError('shot video clock needs a complete bounded numeric preimage')
        count = self.end_output_frame-self.first_output_frame
        if any(not isinstance(grid, tuple) or len(grid) != count
                or any(type(n) is not int or not 0 <= n < self.native_clock.frame_count for n in grid)
                or any(a > b for a, b in zip(grid, grid[1:]))
                for grid in (self.aligned_native_indices, self.native_indices)):
            raise ValueError('shot video clock needs every actual native/output tick, repeats included')
        if (self.first_delta_native_frames == self.last_delta_native_frames == 0
                or self.native_indices[0]-self.aligned_native_indices[0] != self.first_delta_native_frames
                or self.native_indices[-1]-self.aligned_native_indices[-1] != self.last_delta_native_frames):
            raise ValueError('shot video clock endpoints differ from the declared native phase')
        native_fps = float(self.native_clock.frame_rate)
        first_offset = self.first_delta_native_frames/native_fps
        rate_delta = (self.last_delta_native_frames-self.first_delta_native_frames)/native_fps/((count-1)/self.output_fps)
        if (abs(self.source_start-self.aligned_source_start-first_offset) > 1e-9
                or abs(self.source_rate-self.aligned_source_rate-rate_delta) > 1e-9
                or abs(rate_delta) > self.maximum_rate_change_from_aligned+1e-12
                or max(abs(a-b)/native_fps for a, b in zip(self.native_indices,
                    self.aligned_native_indices, strict=True)) > self.maximum_absolute_offset_seconds+1e-12):
            raise ValueError('shot video clock exceeds or changes its exact offset/rate policy')
        for start, rate, expected in ((self.aligned_source_start, self.aligned_source_rate,
                self.aligned_native_indices), (self.source_start, self.source_rate, self.native_indices)):
            grid = sampler_frame_schedule(source_start=start, source_playback_rate=rate,
                native_fps=native_fps, output_fps=self.output_fps, frame_count=count,
                source_frame_count=self.native_clock.frame_count, maximum_terminal_hold_frames=0)
            if grid.terminal_hold_output_indices or tuple(grid.source_frame_indices) != expected:
                raise ValueError('shot video clock differs from exact renderer sampling')

    @property
    def sha256(self):
        return _digest(self.to_dict())

    def to_dict(self):
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        for name in ('segment_indices', 'aligned_native_indices', 'native_indices'):
            result[name] = list(result[name])
        result['native_clock'] = self.native_clock.to_dict()
        return dict(contract=SHOT_VIDEO_CLOCK_CONTRACT, **result,
            authority_scope='bounded_numeric_video_execution_only',
            shared_audio_alignment_changed=False, choreography_certified=False,
            lip_sync_certified=False, candidate_admitted=False, quality_approved=False)


def shot_video_clock_from_dict(payload):
    if not isinstance(payload, dict) or payload.get('contract') != SHOT_VIDEO_CLOCK_CONTRACT:
        raise ValueError('unknown shot video clock contract')
    try:
        raw = payload['native_clock']
        clock = VerifiedRenderFrameClock(Fraction(raw['frame_rate']), raw['frame_count'],
            raw['packet_grid_sha256'], raw['grid_basis'])
        args = {field.name: payload[field.name] for field in fields(AuthorizedShotVideoClockV1)}
        args['native_clock'] = clock
        for name in ('segment_indices', 'aligned_native_indices', 'native_indices'):
            args[name] = tuple(args[name])
        result = AuthorizedShotVideoClockV1(**args)
        if result.to_dict() != payload:
            raise ValueError('shot video clock roundtrip changed authority or redundant fields')
        return result
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise ValueError('shot video clock is incomplete or malformed') from exc


def authorize_shot_video_clock(*, context, aligned_state, phase_state, hypothesis):
    """Bind a proposed full-shot phase to an externally frozen aligned state.

    No semantic/audio-quality permission is created. Normal automatic adoption
    still requires the separate raw-v2 parent and calibrated correspondence.
    """
    from .group_joint_native_phase import phase_adjusted_group_state
    from .group_joint_state_factory import _cell_interval
    if not isinstance(context, FrozenShotVideoClockContext):
        raise ValueError('shot phase authorization requires external frozen context')
    domain, plan, policy = context.source_domain, context.incumbent_plan, context.policy
    expected = phase_adjusted_group_state(state=aligned_state, domain=domain,
        hypothesis=hypothesis, policy=policy)
    if expected == aligned_state or expected != phase_state:
        raise ValueError('shot phase authorization differs from the complete proposed state')
    first, end, _ = _cell_interval(plan, aligned_state.first_output_frame, aligned_state.end_output_frame)
    baseline, proposed = _unit(aligned_state.segments), _unit(phase_state.segments)
    record = AuthorizedShotVideoClockV1(_incumbent_identity(plan), _digest(domain.to_dict()),
        _sources_identity(plan), policy.sha256, domain.request_index(aligned_state.eligible_source_index),
        aligned_state.eligible_source_index, tuple(range(first, end)), plan.master_audio_start,
        plan.master_audio_start+baseline.master_start,
        plan.master_audio_start+baseline.master_start+baseline.duration,
        aligned_state.first_output_frame, aligned_state.end_output_frame, plan.fps,
        baseline.source_start, baseline.source_playback_rate, proposed.source_start,
        proposed.source_playback_rate, hypothesis.first_sample_delta_native_frames,
        hypothesis.last_sample_delta_native_frames, policy.maximum_absolute_offset_seconds,
        policy.maximum_rate_change_from_aligned, domain.render_frame_clocks[aligned_state.eligible_source_index],
        aligned_state.native_frame_indices, phase_state.native_frame_indices)
    # Reconstruct baseline from ORIGINAL alignment rather than trusting the
    # caller's already-retimed 'aligned' state or a receipt's copied digest.
    _validate_context_record(record, context)
    return record


def _validate_context_record(record, context):
    from .render import aligned_render_segment
    plan, domain, policy = context.incumbent_plan, context.source_domain, context.policy
    if (record.incumbent_plan_sha256 != _incumbent_identity(plan)
            or record.source_domain_sha256 != _digest(domain.to_dict())
            or record.policy_sha256 != policy.sha256
            or record.maximum_absolute_offset_seconds != policy.maximum_absolute_offset_seconds
            or record.maximum_rate_change_from_aligned != policy.maximum_rate_change_from_aligned
            or domain.request_index(record.eligible_source_index) != record.request_source_index
            or record.native_clock != domain.render_frame_clocks[record.eligible_source_index]
            or record.sources_sha256 != _sources_identity(plan)
            or record.segment_indices[-1] >= len(plan.segments)):
        raise ValueError('shot-clock authority differs from external incumbent/source/alignment/policy')
    parts = tuple(aligned_render_segment(source_index=record.request_source_index,
        reference_start=plan.master_audio_start+plan.segments[i].master_start,
        common_master_start=plan.master_audio_start, source_offset=domain.audio_offsets[record.eligible_source_index],
        duration=plan.segments[i].duration, timewarp=domain.source_timewarps[record.eligible_source_index])
        for i in record.segment_indices)
    base = _unit(parts)
    if (abs(base.source_start-record.aligned_source_start) > 1e-9
            or abs(base.source_playback_rate-record.aligned_source_rate) > 1e-9):
        raise ValueError('shot-clock baseline is not the original unmodified source alignment')
    native_fps = float(record.native_clock.frame_rate)
    if any(record.native_indices[0] < math.ceil(cut*native_fps-1e-9) <= record.native_indices[-1]
            for cut in domain.scene_cut_times[record.eligible_source_index]):
        raise ValueError('shot-clock authority crosses an actual native broadcaster cut')


def validate_shot_video_clock_bindings(plan, *, frame_clocks=None):
    """Independent structural/grid replay used by renderer, JSON and exports.

    A serialized preimage is not external authorization. Parent validation below
    is additionally mandatory at numeric bundle admission; semantic raw-v2 and
    lip/beat checks are a distinct still-required family.
    """
    from .render import segment_frame_counts, _can_share_source_clock
    from .render_boundary_clock import native_frame_clocks_for_output
    rows = plan.authorized_shot_video_clocks
    if not isinstance(rows, tuple) or len(rows) > 2:
        raise ValueError('shot clocks require a bounded immutable inventory')
    if not rows:
        return {}
    counts = segment_frame_counts(plan); ticks = [0]
    for count in counts:
        ticks.append(ticks[-1]+count)
    claimed = set()
    for record in rows:
        if not isinstance(record, AuthorizedShotVideoClockV1):
            raise ValueError('unknown typed shot video clock')
        first, last = record.segment_indices[0], record.segment_indices[-1]
        if (last >= len(plan.segments) or claimed.intersection(record.segment_indices)
                or record.sources_sha256 != _sources_identity(plan)
                or record.master_audio_start != plan.master_audio_start or record.output_fps != plan.fps
                or record.first_output_frame != ticks[first] or record.end_output_frame != ticks[last+1]
                or any(t.kind.uses_overlap for t in plan.transitions)
                or not _can_share_source_clock(plan, first_index=first, candidate_last_index=last)):
            raise ValueError('shot-clock inventory is stale, overlapping or outside the exact output span')
        claimed.update(record.segment_indices)
        parts = plan.segments[first:last+1]; unit = _unit(parts)
        if (any(p.source_index != record.request_source_index for p in parts)
                or abs(plan.master_audio_start+unit.master_start-record.reference_start) > 1e-9
                or abs(plan.master_audio_start+unit.master_start+unit.duration-record.reference_end) > 1e-9
                or abs(unit.source_start-record.source_start) > 1e-9
                or abs(unit.source_playback_rate-record.source_rate) > 1e-9):
            raise ValueError('shot-clock authority differs from final source/segment clocks')
        if frame_clocks is not None and frame_clocks.get(record.request_source_index) != record.native_clock:
            raise ValueError('shot-clock authority differs from independently verified native/PTS grid')
        # Replay the ACTUAL renderer units, not a freshly rounded standalone
        # shot. Geometry may stop coalescing at a cell seam; that must not
        # quietly substitute another native frame at a half-frame boundary.
        samples = native_frame_clocks_for_output(plan,
            {record.request_source_index: record.native_clock},
            range(record.first_output_frame, record.end_output_frame))
        if (tuple(s.native_frame_index for s in samples.values()) != record.native_indices
                or any(s.source_index != record.request_source_index or s.terminal_clone
                    for s in samples.values())):
            raise ValueError('shot-clock authority differs from actual global renderer units/native grid')
    return {i: record for record in rows for i in record.segment_indices}


def validate_authorized_shot_video_clocks(*, plan, context):
    if not isinstance(context, FrozenShotVideoClockContext):
        raise ValueError('shot-clock consumer lacks an external frozen parent context')
    rows = validate_shot_video_clock_bindings(plan, frame_clocks=context.source_domain.clocks_by_request)
    old = context.incumbent_plan
    if (plan.sources != old.sources or plan.reference_source != old.reference_source
            or plan.master_audio_start != old.master_audio_start or plan.fps != old.fps
            or tuple((p.master_start, p.duration) for p in plan.segments)
                != tuple((p.master_start, p.duration) for p in old.segments)):
        raise ValueError('shot clocks cannot change reference audio or canonical cadence')
    for record in plan.authorized_shot_video_clocks:
        _validate_context_record(record, context)
    return rows


def attach_authorized_shot_video_clocks(*, plan, context, clocks):
    candidate = replace(plan, authorized_shot_video_clocks=tuple(clocks))
    validate_authorized_shot_video_clocks(plan=candidate, context=context)
    return candidate


@dataclass(frozen=True, slots=True)
class ResolvedShotVideoClockSlice:
    """One descriptor cell from the SAME whole-shot clock, not a new offset."""
    authorization_sha256: str
    eligible_source_index: int
    reference_start: float
    reference_end: float
    source_start: float
    source_end: float

    def __post_init__(self):
        if (not isinstance(self.authorization_sha256, str) or len(self.authorization_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.authorization_sha256)
                or type(self.eligible_source_index) is not int or self.eligible_source_index < 0
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in
                    (self.reference_start, self.reference_end, self.source_start, self.source_end))
                or self.reference_start < 0 or self.source_start < 0
                or self.reference_end <= self.reference_start or self.source_end <= self.source_start):
            raise ValueError('resolved shot-clock slice needs finite bound source/reference endpoints')


def resolved_shot_video_clock_slices(*, plan, context):
    inventory = validate_authorized_shot_video_clocks(plan=plan, context=context)
    result = []
    for i, segment in enumerate(plan.segments):
        record = inventory.get(i)
        result.append(None if record is None else ResolvedShotVideoClockSlice(record.sha256,
            record.eligible_source_index, plan.master_audio_start+segment.master_start,
            plan.master_audio_start+segment.master_start+segment.duration, segment.source_start, segment.source_end))
    return tuple(result)
