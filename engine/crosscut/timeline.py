"""Global source selection for natural cross-cut timelines.

The optimiser works on short, already synchronized decision segments (usually
beats, phrases, or safe camera boundaries).  It uses dynamic programming rather
than greedily picking the prettiest frame at every segment, which lets a slightly
weaker current shot win when it avoids an awkward cut on the next beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log, pi, sqrt
from typing import Mapping, Sequence

from .ive import EditMode, IVEMember


def _score(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    return numeric


class TimelineMode(str, Enum):
    GENERAL = "general"
    MEMBER = "member"


class ShotType(str, Enum):
    GROUP_WIDE = "group_wide"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"


@dataclass(frozen=True, slots=True)
class TimelineCandidate:
    """One source option for one synchronized timeline segment.

    ``start_seconds`` / ``end_seconds`` are canonical reference-song time, not
    source-local decode time.  ``pose_embedding`` can be any fixed-length,
    normalized landmark vector. ``structure_embedding`` is its deterministic
    model-free fallback, while ``silhouette_embedding`` represents aggregate
    on-screen human layout.
    ``motion`` is an image-plane direction vector.  They are optional so the
    engine remains usable before pose estimation is wired in.

    For member mode, a close/medium candidate is eligible only when
    ``target_member`` matches the requested IVE member, confidence passes the
    configured gate, and ``target_crop_available`` is true.  A candidate marked
    ``is_group_wide`` remains eligible as a safety fallback.
    """

    segment_index: int
    source_id: str
    start_seconds: float
    end_seconds: float
    quality_score: float = 1.0
    sync_score: float = 1.0
    composition_score: float = 0.5
    boundary_score: float = 0.5
    shot_scale: float = 0.5
    shot_type: ShotType | str = ShotType.MEDIUM
    target_member: IVEMember | str | None = None
    target_confidence: float = 0.0
    target_crop_available: bool = False
    is_group_wide: bool = False
    crop_stability_score: float = 0.5
    pose_embedding: tuple[float, ...] = ()
    entry_pose_embedding: tuple[float, ...] = ()
    exit_pose_embedding: tuple[float, ...] = ()
    structure_embedding: tuple[float, ...] = ()
    entry_structure_embedding: tuple[float, ...] = ()
    exit_structure_embedding: tuple[float, ...] = ()
    silhouette_embedding: tuple[float, ...] = ()
    entry_silhouette_embedding: tuple[float, ...] = ()
    exit_silhouette_embedding: tuple[float, ...] = ()
    color_embedding: tuple[float, ...] = ()
    entry_color_embedding: tuple[float, ...] = ()
    exit_color_embedding: tuple[float, ...] = ()
    entry_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    exit_face_geometry_window: tuple[tuple[float, ...], ...] = ()
    motion: tuple[float, float] | None = None
    candidate_id: str = ""
    shot_id: str = ""

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("candidate time range must have positive duration")
        for field_name in (
            "quality_score",
            "sync_score",
            "composition_score",
            "boundary_score",
            "shot_scale",
            "target_confidence",
            "crop_stability_score",
        ):
            object.__setattr__(self, field_name, _score(getattr(self, field_name), field_name))
        if not isinstance(self.shot_type, ShotType):
            object.__setattr__(self, "shot_type", ShotType(self.shot_type))
        if self.target_member is not None:
            object.__setattr__(self, "target_member", IVEMember.parse(self.target_member))
        if self.is_group_wide and self.shot_type not in (
            ShotType.GROUP_WIDE,
            ShotType.WIDE,
        ):
            raise ValueError("is_group_wide candidates must use a wide shot type")
        if self.motion is not None and len(self.motion) != 2:
            raise ValueError("motion must contain x/y components")
        for embedding in (
            self.pose_embedding,
            self.entry_pose_embedding,
            self.exit_pose_embedding,
            self.structure_embedding,
            self.entry_structure_embedding,
            self.exit_structure_embedding,
            self.silhouette_embedding,
            self.entry_silhouette_embedding,
            self.exit_silhouette_embedding,
            self.color_embedding,
            self.entry_color_embedding,
            self.exit_color_embedding,
        ):
            if any(not isinstance(value, (int, float)) for value in embedding):
                raise ValueError("pose embeddings must be numeric")
        for window in (
            self.entry_face_geometry_window,
            self.exit_face_geometry_window,
        ):
            if window and len(window) < 2:
                raise ValueError("face geometry windows need at least two frames")
            for geometry in window:
                if len(geometry) != 10:
                    raise ValueError("face geometry vectors must contain ten values")
                if any(
                    not isinstance(value, (int, float))
                    or not isfinite(float(value))
                    for value in geometry
                ):
                    raise ValueError("face geometry values must be finite numbers")
                if geometry[4] <= 0.0:
                    raise ValueError("face eye scale must be positive")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def is_safe_for_member(
        self, target: IVEMember, minimum_confidence: float
    ) -> bool:
        return (
            self.target_member is target
            and self.target_confidence >= minimum_confidence
            and self.target_crop_available
        )


@dataclass(frozen=True, slots=True)
class TimelineConfig:
    """Weights and editing constraints for Viterbi source selection."""

    quality_weight: float = 1.25
    sync_weight: float = 1.7
    composition_weight: float = 0.65
    boundary_weight: float = 0.35
    target_weight: float = 2.4
    crop_stability_weight: float = 0.7
    continuity_bonus: float = 0.28
    cut_penalty: float = 0.20
    early_cut_penalty: float = 0.65
    long_hold_penalty: float = 0.14
    source_hold_penalty: float = 0.0
    preferred_source_bonus: float = 0.0
    preferred_source_max_score_sacrifice: float = 0.0
    pose_transition_weight: float = 0.85
    silhouette_transition_weight: float = 0.62
    motion_transition_weight: float = 0.48
    scale_transition_weight: float = 0.40
    color_transition_weight: float = 0.24
    face_geometry_transition_weight: float = 0.28
    group_wide_fallback_penalty: float = 0.55
    minimum_member_confidence: float = 0.72
    minimum_hold_segments: int = 2
    preferred_max_hold_segments: int = 6
    preferred_max_source_hold_segments: int | None = None
    maximum_same_source_hold_segments: int | None = None
    maximum_switch_score_sacrifice: float = 0.0

    def __post_init__(self) -> None:
        weight_names = (
            "quality_weight",
            "sync_weight",
            "composition_weight",
            "boundary_weight",
            "target_weight",
            "crop_stability_weight",
            "continuity_bonus",
            "cut_penalty",
            "early_cut_penalty",
            "long_hold_penalty",
            "source_hold_penalty",
            "preferred_source_bonus",
            "preferred_source_max_score_sacrifice",
            "pose_transition_weight",
            "silhouette_transition_weight",
            "motion_transition_weight",
            "scale_transition_weight",
            "color_transition_weight",
            "face_geometry_transition_weight",
            "group_wide_fallback_penalty",
            "maximum_switch_score_sacrifice",
        )
        if any(getattr(self, name) < 0 for name in weight_names):
            raise ValueError("timeline weights and penalties cannot be negative")
        _score(self.minimum_member_confidence, "minimum_member_confidence")
        if self.minimum_hold_segments < 1:
            raise ValueError("minimum_hold_segments must be at least 1")
        if self.preferred_max_hold_segments < self.minimum_hold_segments:
            raise ValueError(
                "preferred_max_hold_segments cannot be below minimum_hold_segments"
            )
        if (
            self.preferred_max_source_hold_segments is not None
            and self.preferred_max_source_hold_segments < 1
        ):
            raise ValueError("preferred source hold must be at least 1")
        if (
            self.maximum_same_source_hold_segments is not None
            and self.maximum_same_source_hold_segments < 1
        ):
            raise ValueError("maximum same-source hold must be at least 1")
        if (
            self.preferred_max_source_hold_segments is not None
            and self.maximum_same_source_hold_segments is not None
            and self.maximum_same_source_hold_segments
            < self.preferred_max_source_hold_segments
        ):
            raise ValueError("maximum same-source hold cannot be below preferred source hold")


@dataclass(frozen=True, slots=True)
class TimelineChoice:
    candidate: TimelineCandidate
    emission_score: float
    transition_score: float
    cumulative_score: float
    used_group_wide_fallback: bool


@dataclass(frozen=True, slots=True)
class TimelineResult:
    mode: TimelineMode
    target_member: IVEMember | None
    choices: tuple[TimelineChoice, ...]
    total_score: float

    @property
    def candidates(self) -> tuple[TimelineCandidate, ...]:
        return tuple(choice.candidate for choice in self.choices)

    @property
    def fallback_segment_indices(self) -> tuple[int, ...]:
        return tuple(
            choice.candidate.segment_index
            for choice in self.choices
            if choice.used_group_wide_fallback
        )


@dataclass(frozen=True, slots=True)
class _Node:
    cumulative_score: float
    previous_state: tuple[int, int, int] | None
    emission_score: float
    transition_score: float
    used_fallback: bool


def _normalised_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    squared = sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    # RMS is less sensitive to embedding dimension. Clamp because landmark
    # vectors from third-party models are not guaranteed to be normalized.
    return min(1.0, sqrt(squared / len(left)))


def _motion_mismatch(
    left: tuple[float, float] | None, right: tuple[float, float] | None
) -> float:
    if left is None or right is None:
        return 0.0
    left_norm = sqrt(left[0] ** 2 + left[1] ** 2)
    right_norm = sqrt(right[0] ** 2 + right[1] ** 2)
    if left_norm < 1e-8 or right_norm < 1e-8:
        return 0.0
    cosine = (left[0] * right[0] + left[1] * right[1]) / (left_norm * right_norm)
    return min(1.0, max(0.0, (1.0 - cosine) / 2.0))


def _angle_distance(left: float, right: float) -> float:
    return abs((left - right + pi) % (2.0 * pi) - pi)


def _face_geometry_mismatch(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    """Compare normalized eye endpoints, scale, roll, and coarse head pose."""

    left_eye = sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)
    right_eye = sqrt((left[2] - right[2]) ** 2 + (left[3] - right[3]) ** 2)
    # The original Cross-Cutting idea compares both eye endpoints. Normalize
    # that screen-space quantity, then supplement it with modern YuNet pose
    # geometry rather than copying the obsolete dlib/MoviePy implementation.
    endpoint_mismatch = min(1.0, (left_eye + right_eye) / (2.0 * 0.30))
    scale_mismatch = min(
        1.0,
        abs(log(float(left[4]) / float(right[4]))) / log(1.7),
    )
    roll_mismatch = min(1.0, _angle_distance(left[5], right[5]) / (pi / 4.0))
    pose_squared = sum((left[index] - right[index]) ** 2 for index in range(6, 10))
    pose_mismatch = min(1.0, sqrt(pose_squared / 4.0) / 0.75)
    return min(
        1.0,
        0.50 * endpoint_mismatch
        + 0.20 * scale_mismatch
        + 0.12 * roll_mismatch
        + 0.18 * pose_mismatch,
    )


def face_geometry_minimax_mismatch(
    outgoing_window: tuple[tuple[float, ...], ...],
    incoming_window: tuple[tuple[float, ...], ...],
) -> float:
    """Worst consecutive-frame face mismatch around a proposed hard cut.

    Audio-synchronized windows are paired as outgoing frame ``i`` and incoming
    frame ``i+1``. The maximum pair mismatch is the local minimax objective.
    Missing/ambiguous detections are represented by an empty window and remain
    neutral; this term can reject no candidate and awards no face bonus.
    """

    if (
        not outgoing_window
        or not incoming_window
        or len(outgoing_window) != len(incoming_window)
    ):
        return 0.0
    if any(
        len(geometry) != 10
        or geometry[4] <= 0.0
        or any(not isfinite(float(value)) for value in geometry)
        for window in (outgoing_window, incoming_window)
        for geometry in window
    ):
        return 0.0
    return max(
        _face_geometry_mismatch(outgoing, incoming)
        for outgoing, incoming in zip(
            outgoing_window,
            incoming_window,
            strict=True,
        )
    )


def _candidate_emission(
    candidate: TimelineCandidate,
    *,
    mode: TimelineMode,
    target: IVEMember | None,
    config: TimelineConfig,
) -> tuple[float, bool] | None:
    base = (
        config.quality_weight * candidate.quality_score
        + config.sync_weight * candidate.sync_score
        + config.composition_weight * candidate.composition_score
        + config.boundary_weight * candidate.boundary_score
    )
    if mode is TimelineMode.GENERAL:
        return (base, False)

    assert target is not None
    if candidate.is_safe_for_member(target, config.minimum_member_confidence):
        return (
            base
            + config.target_weight * candidate.target_confidence
            + config.crop_stability_weight * candidate.crop_stability_score,
            False,
        )
    if candidate.is_group_wide:
        # A verified group-wide shot is the only low-confidence fallback.  An
        # unrelated member close-up is intentionally ineligible even if its
        # generic visual score is excellent.
        return (base - config.group_wide_fallback_penalty, True)
    return None


def _transition_score(
    previous: TimelineCandidate,
    current: TimelineCandidate,
    previous_run_length: int,
    config: TimelineConfig,
) -> tuple[float, int]:
    same_source = previous.source_id == current.source_id
    same_continuous_shot = same_source and (
        (not previous.shot_id and not current.shot_id)
        or (previous.shot_id and previous.shot_id == current.shot_id)
    )
    if same_continuous_shot:
        run_length = previous_run_length + 1
        score = config.continuity_bonus
        if run_length > config.preferred_max_hold_segments:
            score -= config.long_hold_penalty * (
                run_length - config.preferred_max_hold_segments
            )
    else:
        run_length = 1
        score = -config.cut_penalty
        if previous_run_length < config.minimum_hold_segments:
            score -= config.early_cut_penalty * (
                config.minimum_hold_segments - previous_run_length
            )
        previous_pose = previous.exit_pose_embedding or previous.pose_embedding
        current_pose = current.entry_pose_embedding or current.pose_embedding
        # Never compare an actual landmark vector against the unrelated
        # gradient-HOG space. If either pose model missed, compare the same
        # deterministic structure descriptor used by the previous engine.
        if not previous_pose or not current_pose or len(previous_pose) != len(current_pose):
            previous_structure = (
                previous.exit_structure_embedding or previous.structure_embedding
            )
            current_structure = (
                current.entry_structure_embedding or current.structure_embedding
            )
            if previous_structure and current_structure:
                previous_pose = previous_structure
                current_pose = current_structure
        score -= config.pose_transition_weight * _normalised_distance(
            previous_pose, current_pose
        )
        previous_silhouette = (
            previous.exit_silhouette_embedding or previous.silhouette_embedding
        )
        current_silhouette = (
            current.entry_silhouette_embedding or current.silhouette_embedding
        )
        score -= config.silhouette_transition_weight * _normalised_distance(
            previous_silhouette,
            current_silhouette,
        )
        score -= config.motion_transition_weight * _motion_mismatch(
            previous.motion, current.motion
        )
        score -= config.scale_transition_weight * abs(
            previous.shot_scale - current.shot_scale
        )
        previous_color = previous.exit_color_embedding or previous.color_embedding
        current_color = current.entry_color_embedding or current.color_embedding
        score -= config.color_transition_weight * _normalised_distance(
            previous_color,
            current_color,
        )
        score -= config.face_geometry_transition_weight * face_geometry_minimax_mismatch(
            previous.exit_face_geometry_window,
            current.entry_face_geometry_window,
        )
    return (score, run_length)


def _normalise_segments(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
) -> tuple[tuple[TimelineCandidate, ...], ...]:
    if isinstance(candidates_by_segment, Mapping):
        ordered_keys = sorted(candidates_by_segment)
        if ordered_keys and ordered_keys != list(
            range(ordered_keys[0], ordered_keys[0] + len(ordered_keys))
        ):
            raise ValueError("timeline segment indices must be contiguous")
        groups = tuple(tuple(candidates_by_segment[key]) for key in ordered_keys)
    else:
        groups = tuple(tuple(group) for group in candidates_by_segment)
    if not groups:
        raise ValueError("at least one timeline segment is required")
    for group in groups:
        if not group:
            raise ValueError("each timeline segment needs at least one candidate")
        expected = group[0].segment_index
        if any(candidate.segment_index != expected for candidate in group):
            raise ValueError("all candidates in a group must share segment_index")
        canonical_span = (group[0].start_seconds, group[0].end_seconds)
        if any(
            abs(candidate.start_seconds - canonical_span[0]) > 1e-7
            or abs(candidate.end_seconds - canonical_span[1]) > 1e-7
            for candidate in group
        ):
            raise ValueError("all candidates in a group must share canonical span")
    indices = [group[0].segment_index for group in groups]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("candidate groups must be ordered by contiguous segment_index")
    return groups


def _resolved_preferred_sources(
    segments: tuple[tuple[TimelineCandidate, ...], ...],
    emissions: Sequence[Mapping[int, tuple[float, bool]]],
    requested: tuple[str | None, ...],
    config: TimelineConfig,
) -> tuple[str | None, ...]:
    """Rematch safe metadata preferences when their original slot is unavailable.

    The metadata draft orders sources in full first-use cycles.  Frame decoding
    or local visual quality can make one preferred source unsuitable at its
    original segment, so each complete cycle is treated as a deterministic
    bipartite matching problem.  Only source/segment edges within the configured
    score sacrifice are considered.  A genuinely weak source therefore has no
    edge and is omitted instead of being forced into the edit.
    """

    source_order = tuple(
        dict.fromkeys(source for source in requested if source is not None)
    )
    if not source_order:
        return requested
    cycle_size = len(source_order)
    resolved: list[str | None] = [None] * len(segments)

    def source_score(position: int, source_id: str) -> float | None:
        scores = [
            emissions[position][candidate_index][0]
            for candidate_index in emissions[position]
            if segments[position][candidate_index].source_id == source_id
        ]
        return max(scores) if scores else None

    def is_comparable(position: int, source_id: str) -> bool:
        score = source_score(position, source_id)
        if score is None:
            return False
        best = max(value for value, _fallback in emissions[position].values())
        return score >= best - config.preferred_source_max_score_sacrifice

    for cycle_start in range(0, len(segments), cycle_size):
        positions = tuple(
            range(cycle_start, min(len(segments), cycle_start + cycle_size))
        )
        requested_cycle = tuple(requested[position] for position in positions)
        is_complete_cycle = (
            len(positions) == cycle_size
            and set(requested_cycle) == set(source_order)
        )
        if not is_complete_cycle:
            for position in positions:
                source_id = requested[position]
                if source_id is not None and is_comparable(position, source_id):
                    resolved[position] = source_id
            continue

        original_position = {
            source_id: position
            for position, source_id in zip(positions, requested_cycle)
            if source_id is not None
        }
        options: dict[str, tuple[int, ...]] = {}
        for source_id in source_order:
            ranked: list[tuple[int, float, int, int]] = []
            for position in positions:
                score = source_score(position, source_id)
                if score is None or not is_comparable(position, source_id):
                    continue
                best = max(
                    value for value, _fallback in emissions[position].values()
                )
                ranked.append(
                    (
                        0 if original_position.get(source_id) == position else 1,
                        best - score,
                        abs(position - original_position[source_id]),
                        position,
                    )
                )
            options[source_id] = tuple(item[-1] for item in sorted(ranked))

        owner_by_position: dict[int, str] = {}

        def assign(source_id: str, visited: set[int]) -> bool:
            for position in options[source_id]:
                if position in visited:
                    continue
                visited.add(position)
                owner = owner_by_position.get(position)
                if owner is None or assign(owner, visited):
                    owner_by_position[position] = source_id
                    return True
            return False

        source_rank = {source_id: rank for rank, source_id in enumerate(source_order)}
        for source_id in sorted(
            source_order,
            key=lambda value: (len(options[value]), source_rank[value]),
        ):
            if options[source_id]:
                assign(source_id, set())
        for position, source_id in owner_by_position.items():
            resolved[position] = source_id

    return tuple(resolved)


def select_timeline(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
    *,
    mode: TimelineMode | EditMode | str = TimelineMode.GENERAL,
    target_member: IVEMember | str | None = None,
    config: TimelineConfig | None = None,
    preferred_source_ids: Sequence[str | None] | None = None,
) -> TimelineResult:
    """Select one candidate per segment with global DP/Viterbi optimisation.

    Raises ``ValueError`` when a member segment contains neither a confident
    target crop nor an explicit group-wide fallback.  This is preferable to
    emitting a plausible-looking crop of the wrong IVE member.
    """

    if isinstance(mode, TimelineMode):
        timeline_mode = mode
    elif isinstance(mode, EditMode):
        timeline_mode = (
            TimelineMode.MEMBER if mode is EditMode.MEMBER else TimelineMode.GENERAL
        )
    else:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode == EditMode.GROUP.value:
            normalized_mode = TimelineMode.GENERAL.value
        timeline_mode = TimelineMode(normalized_mode)
    target = IVEMember.parse(target_member) if target_member is not None else None
    if timeline_mode is TimelineMode.MEMBER and target is None:
        raise ValueError("target_member is required in member mode")
    settings = config or TimelineConfig()
    segments = _normalise_segments(candidates_by_segment)
    preferred_sources = (
        tuple(preferred_source_ids) if preferred_source_ids is not None else None
    )
    if preferred_sources is not None:
        if timeline_mode is not TimelineMode.GENERAL:
            raise ValueError("preferred source assignments are only supported in general mode")
        if len(preferred_sources) != len(segments):
            raise ValueError("preferred source assignments must match timeline segments")

    emissions: list[dict[int, tuple[float, bool]]] = []
    for segment in segments:
        eligible: dict[int, tuple[float, bool]] = {}
        for candidate_index, candidate in enumerate(segment):
            emission = _candidate_emission(
                candidate,
                mode=timeline_mode,
                target=target,
                config=settings,
            )
            if emission is not None:
                eligible[candidate_index] = emission
        if not eligible:
            raise ValueError(
                f"segment {segment[0].segment_index} has no safe candidate for "
                f"{target.value if target else timeline_mode.value}"
            )
        emissions.append(eligible)

    if preferred_sources is not None:
        resolved_preferences = _resolved_preferred_sources(
            segments,
            emissions,
            preferred_sources,
            settings,
        )
        for position, preferred_source in enumerate(resolved_preferences):
            if preferred_source is None:
                continue
            for candidate_index, (score, used_fallback) in tuple(
                emissions[position].items()
            ):
                if segments[position][candidate_index].source_id == preferred_source:
                    emissions[position][candidate_index] = (
                        score + settings.preferred_source_bonus,
                        used_fallback,
                    )

    # State carries both the continuous-shot run and the same-source run.
    # A camera cut inside one source resets the former (so normal cut safety
    # scoring still applies) but not the latter (so it cannot bypass an opted-in
    # source-rotation policy).
    layers: list[dict[tuple[int, int, int], _Node]] = []
    first_layer: dict[tuple[int, int, int], _Node] = {}
    for candidate_index, (emission_score, used_fallback) in emissions[0].items():
        first_layer[(candidate_index, 1, 1)] = _Node(
            cumulative_score=emission_score,
            previous_state=None,
            emission_score=emission_score,
            transition_score=0.0,
            used_fallback=used_fallback,
        )
    layers.append(first_layer)

    for segment_position in range(1, len(segments)):
        previous_segment = segments[segment_position - 1]
        current_segment = segments[segment_position]
        previous_layer = layers[-1]
        current_layer: dict[tuple[int, int, int], _Node] = {}

        for previous_state, previous_node in previous_layer.items():
            previous_index, previous_shot_run, previous_source_run = previous_state
            previous_candidate = previous_segment[previous_index]
            local_options: list[
                tuple[int, float, bool, float, int, int]
            ] = []
            for current_index, (emission_score, used_fallback) in emissions[
                segment_position
            ].items():
                current_candidate = current_segment[current_index]
                transition_score, shot_run = _transition_score(
                    previous_candidate,
                    current_candidate,
                    previous_shot_run,
                    settings,
                )
                source_run = (
                    previous_source_run + 1
                    if previous_candidate.source_id == current_candidate.source_id
                    else 1
                )
                local_options.append(
                    (
                        current_index,
                        emission_score,
                        used_fallback,
                        transition_score,
                        shot_run,
                        source_run,
                    )
                )

            for (
                current_index,
                emission_score,
                used_fallback,
                transition_score,
                shot_run,
                source_run,
            ) in local_options:
                current_candidate = current_segment[current_index]
                same_source = previous_candidate.source_id == current_candidate.source_id
                local_score = emission_score + transition_score
                comparable_alternate_exists = any(
                    previous_candidate.source_id
                    != current_segment[alternate_index].source_id
                    and alternate_emission + alternate_transition
                    >= local_score - settings.maximum_switch_score_sacrifice
                    for (
                        alternate_index,
                        alternate_emission,
                        _alternate_fallback,
                        alternate_transition,
                        _alternate_shot_run,
                        _alternate_source_run,
                    ) in local_options
                )
                if (
                    same_source
                    and settings.maximum_same_source_hold_segments is not None
                    and source_run > settings.maximum_same_source_hold_segments
                    and comparable_alternate_exists
                ):
                    continue

                cadence_score = 0.0
                if (
                    same_source
                    and settings.preferred_max_source_hold_segments is not None
                    and source_run > settings.preferred_max_source_hold_segments
                    and comparable_alternate_exists
                ):
                    cadence_score = -settings.source_hold_penalty * (
                        source_run - settings.preferred_max_source_hold_segments
                    )
                transition_with_cadence = transition_score + cadence_score
                state = (current_index, shot_run, source_run)
                cumulative = (
                    previous_node.cumulative_score
                    + transition_with_cadence
                    + emission_score
                )
                existing = current_layer.get(state)
                # Deterministic tie-break: prefer the lexicographically earlier
                # predecessor, making repeated synthetic runs byte-for-byte stable.
                if existing is None or cumulative > existing.cumulative_score + 1e-12:
                    current_layer[state] = _Node(
                        cumulative_score=cumulative,
                        previous_state=previous_state,
                        emission_score=emission_score,
                        transition_score=transition_with_cadence,
                        used_fallback=used_fallback,
                    )
        if not current_layer:
            raise ValueError(
                f"segment {current_segment[0].segment_index} has no cadence-safe candidate"
            )
        layers.append(current_layer)

    final_state, final_node = max(
        layers[-1].items(),
        key=lambda item: (
            item[1].cumulative_score,
            -item[0][0],
            -item[0][1],
            -item[0][2],
        ),
    )
    states: list[tuple[int, int, int]] = [final_state]
    for layer_position in range(len(layers) - 1, 0, -1):
        previous_state = layers[layer_position][states[-1]].previous_state
        assert previous_state is not None
        states.append(previous_state)
    states.reverse()

    choices: list[TimelineChoice] = []
    for segment_position, state in enumerate(states):
        node = layers[segment_position][state]
        choices.append(
            TimelineChoice(
                candidate=segments[segment_position][state[0]],
                emission_score=node.emission_score,
                transition_score=node.transition_score,
                cumulative_score=node.cumulative_score,
                used_group_wide_fallback=node.used_fallback,
            )
        )
    return TimelineResult(
        mode=timeline_mode,
        target_member=target,
        choices=tuple(choices),
        total_score=final_node.cumulative_score,
    )


# A name that reads naturally at call sites which think in terms of Viterbi.
optimize_timeline = select_timeline


__all__ = [
    "ShotType",
    "TimelineCandidate",
    "TimelineChoice",
    "TimelineConfig",
    "TimelineMode",
    "TimelineResult",
    "face_geometry_minimax_mismatch",
    "optimize_timeline",
    "select_timeline",
]
