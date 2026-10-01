"""Explicit small native-phase hypotheses for a complete GROUP shot.

No nearest-frame probe silently becomes a render clock. The proposal changes
one entire linear source clock, retains every camera/cell/output sample and
is rebound by the normal numeric assembly. Audio, pose, lip-sync, semantic
identity and production adoption still require the independent owning gate.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

from .group_joint_source_camera import GroupSourceCameraState
from .group_source_domain import GroupSourceDomain
from .render import _combined_identity_segment, MAXIMUM_RENDER_TIMEWARP_RATE_DELTA
from .render_frame_schedule import sampler_frame_schedule


@dataclass(frozen=True, slots=True)
class NativePhaseSearchPolicy:
    maximum_absolute_offset_seconds: float = .12
    maximum_rate_change_from_aligned: float = .02

    def __post_init__(self):
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in
                (self.maximum_absolute_offset_seconds, self.maximum_rate_change_from_aligned))
                or not 0 < self.maximum_absolute_offset_seconds <= .15
                or not 0 < self.maximum_rate_change_from_aligned <= MAXIMUM_RENDER_TIMEWARP_RATE_DELTA):
            raise ValueError('phase policy needs bounded source offsets and existing render rate limits')

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class NativePhaseHypothesis:
    baseline_complete_state_key: str
    policy_sha256: str
    first_sample_delta_native_frames: int
    last_sample_delta_native_frames: int

    def __post_init__(self):
        if (any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                for v in (self.baseline_complete_state_key, self.policy_sha256))
                or any(type(v) is not int for v in
                    (self.first_sample_delta_native_frames, self.last_sample_delta_native_frames))):
            raise ValueError('native phase hypothesis must bind a complete state and frozen policy')

    def to_dict(self):
        return dict(contract='bounded_complete_source_native_phase_hypothesis_not_admission_v1',
            **asdict(self), pose_or_identity_certified=False, lip_sync_certified=False,
            candidate_admitted=False, professional_quality_approved=False)


def phase_adjusted_group_state(*, state, domain, hypothesis, policy):
    """Propose one exact linear native grid, never change only a sampled edge.

    Phase deltas name first and last ACTUAL output samples, not the unshown
    source_end at N/fps. One speed changes every cell consistently. No rate
    limit is weakened for a short shot: 0 -> +3 frames in 70 ticks exceeds
    the existing 2% limit; a different two-endpoint hypothesis is required.
    This is numeric construction, not permission to retime normal app video.
    """
    if (not isinstance(state, GroupSourceCameraState) or state.slot not in (1, 2)
            or not isinstance(domain, GroupSourceDomain)
            or not isinstance(hypothesis, NativePhaseHypothesis)
            or not isinstance(policy, NativePhaseSearchPolicy)
            or hypothesis.baseline_complete_state_key != state.key
            or hypothesis.policy_sha256 != policy.sha256):
        raise ValueError('phase proposal differs from the frozen whole-state/policy binding')
    state.bind_source(domain)
    count = state.end_output_frame-state.first_output_frame
    if not 2 <= count <= 300:
        raise ValueError('phase proposal exceeds the complete mutable shot frame budget')
    first_delta, last_delta = (hypothesis.first_sample_delta_native_frames,
        hypothesis.last_sample_delta_native_frames)
    if first_delta == last_delta == 0:
        return state
    clock = domain.render_frame_clocks[state.eligible_source_index]
    native_fps = float(clock.frame_rate)
    if max(abs(first_delta), abs(last_delta))/native_fps > policy.maximum_absolute_offset_seconds+1e-12:
        raise ValueError('native phase hypothesis exceeds the frozen absolute source-time bound')
    original_unit = (state.segments[0] if len(state.segments) == 1
        else _combined_identity_segment(state.segments))
    original_schedule = sampler_frame_schedule(source_start=original_unit.source_start,
        source_playback_rate=original_unit.source_playback_rate, native_fps=native_fps,
        output_fps=state.output_fps, frame_count=count, source_frame_count=clock.frame_count,
        maximum_terminal_hold_frames=0)
    if tuple(original_schedule.source_frame_indices) != state.native_frame_indices:
        raise ValueError('phase base grid differs from the complete actual renderer clock')
    last_sample_time = (count-1)/state.output_fps
    rate_change = (last_delta-first_delta)/native_fps/last_sample_time
    if abs(rate_change) > policy.maximum_rate_change_from_aligned+1e-12:
        raise ValueError('native phase hypothesis exceeds the frozen whole-shot rate-change bound')
    rate = original_unit.source_playback_rate+rate_change
    source_start = original_unit.source_start+first_delta/native_fps
    cursor, parts = 0., []
    for segment in state.segments:
        # RenderSegment also independently enforces its absolute 2% rate bound.
        parts.append(replace(segment, source_start=source_start+cursor*rate,
            source_duration=segment.duration*rate))
        cursor += segment.duration
    parts = tuple(parts)
    unit = parts[0] if len(parts) == 1 else _combined_identity_segment(parts)
    schedule = sampler_frame_schedule(source_start=unit.source_start,
        source_playback_rate=unit.source_playback_rate, native_fps=native_fps,
        output_fps=state.output_fps, frame_count=count, source_frame_count=clock.frame_count,
        maximum_terminal_hold_frames=0)
    native = tuple(schedule.source_frame_indices)
    if (schedule.terminal_hold_output_indices or native[0] != state.native_frame_indices[0]+first_delta
            or native[-1] != state.native_frame_indices[-1]+last_delta
            or any(abs(a-b)/native_fps > policy.maximum_absolute_offset_seconds+1e-12
                for a,b in zip(native, state.native_frame_indices, strict=True))):
        raise ValueError('complete phase sampling changed requested endpoints or exceeded a bound')
    return replace(state, segments=parts, native_frame_indices=native).bind_source(domain)
