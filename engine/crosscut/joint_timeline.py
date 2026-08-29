"""Joint source, boundary-offset, and endpoint-transform timeline selection.

The ordinary timeline selector chooses a source/shot path before decoded
boundary repair.  This module is a deliberately small, provider-driven core
that can instead optimize one *authorized* boundary measurement together with
each source transition.  It never decodes media and never weakens the existing
timeline score or hard gates: callers provide the complete ``-8..+8`` audit,
and only exact, complete production proofs enter the dynamic program.

For a middle segment, the option on its left supplies the incoming endpoint
transform and the option on its right supplies the outgoing endpoint
transform.  The DP therefore retains the previous boundary offset and
endpoint transform in its state until a ``SegmentBridgeProof`` certifies that
the crop/track path across the segment is compatible and at least two frames
long.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from math import isfinite
from typing import TypeAlias

from .ive import EditMode
from .timeline import (
    NoCompleteTimelinePath,
    TimelineCandidate,
    TimelineChoice,
    TimelineConfig,
    TimelineMode,
    TimelineResult,
    _candidate_emission,
    _normalise_segments,
    _resolved_preferred_sources,
    _transition_score,
    compile_timeline_no_good_constraint_index,
)
from .timeline_no_good import (
    TimelineNoGoodEdgeKey,
    TimelineNoGoodSet,
    build_timeline_no_good_edge_key,
)
from .vision import IVEMember


JOINT_BOUNDARY_FRAME_OFFSETS = tuple(range(-8, 9))
MINIMUM_JOINT_SEGMENT_DURATION_FRAMES = 2.0
JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES = 300_000
BOUNDARY_TRANSFORM_SUBJECT_BASES = frozenset(
    {"dominant_face", "subject_silhouette", "full_frame_identity"}
)


def _exact_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be an exact non-blank identifier")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_failure_codes(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("failure_codes must be a sequence")
    codes = tuple(sorted(set(values)))
    if any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        for value in codes
    ):
        raise ValueError("failure codes must be exact non-blank identifiers")
    return codes


@dataclass(frozen=True, slots=True)
class BoundaryMeasurementKey:
    """Exact candidate-pair identity for one canonical timeline boundary."""

    boundary_index: int
    outgoing: TimelineCandidate
    incoming: TimelineCandidate

    def __post_init__(self) -> None:
        if (
            isinstance(self.boundary_index, bool)
            or not isinstance(self.boundary_index, int)
            or self.boundary_index < 0
        ):
            raise ValueError("boundary_index must be a non-negative integer")
        if not isinstance(self.outgoing, TimelineCandidate) or not isinstance(
            self.incoming, TimelineCandidate
        ):
            raise TypeError("boundary measurement candidates are invalid")
        if self.incoming.segment_index != self.outgoing.segment_index + 1:
            raise ValueError(
                "boundary measurement candidates must occupy adjacent segments"
            )

    @classmethod
    def from_candidates(
        cls,
        boundary_index: int,
        outgoing: TimelineCandidate,
        incoming: TimelineCandidate,
    ) -> "BoundaryMeasurementKey":
        return cls(boundary_index, outgoing, incoming)


@dataclass(frozen=True, slots=True)
class BoundaryTransformState:
    """Exact crop/track state at one side of an authorized boundary."""

    transform_identity: str
    crop_xywh: tuple[float, float, float, float] | None = None
    rotation_degrees: float = 0.0
    track_id: str | None = None
    subject_basis: str = "full_frame_identity"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_identity",
            _exact_identifier(self.transform_identity, "transform_identity"),
        )
        if self.crop_xywh is not None:
            if isinstance(self.crop_xywh, (str, bytes)) or len(self.crop_xywh) != 4:
                raise ValueError("crop_xywh must contain x, y, width, and height")
            crop = tuple(
                _finite_number(value, "crop_xywh value") for value in self.crop_xywh
            )
            if crop[2] <= 0.0 or crop[3] <= 0.0:
                raise ValueError("crop width and height must be positive")
            object.__setattr__(self, "crop_xywh", crop)
        object.__setattr__(
            self,
            "rotation_degrees",
            _finite_number(self.rotation_degrees, "rotation_degrees"),
        )
        if self.track_id is not None:
            object.__setattr__(
                self,
                "track_id",
                _exact_identifier(self.track_id, "track_id"),
            )
        subject_basis = _exact_identifier(self.subject_basis, "subject_basis")
        if subject_basis not in BOUNDARY_TRANSFORM_SUBJECT_BASES:
            raise ValueError("subject_basis is not a supported tracking basis")
        object.__setattr__(self, "subject_basis", subject_basis)


@dataclass(frozen=True, slots=True)
class AuthorizedBoundaryOption:
    """One complete, production-authorized option from the 17-offset audit."""

    measurement_key: BoundaryMeasurementKey
    frame_offset: int
    outgoing_transform: BoundaryTransformState
    incoming_transform: BoundaryTransformState
    proof_identity: str
    score_adjustment: float = 0.0
    scoring_outgoing: TimelineCandidate | None = None
    scoring_incoming: TimelineCandidate | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.measurement_key, BoundaryMeasurementKey):
            raise TypeError("authorized boundary measurement key is invalid")
        if (
            isinstance(self.frame_offset, bool)
            or not isinstance(self.frame_offset, int)
            or self.frame_offset not in JOINT_BOUNDARY_FRAME_OFFSETS
        ):
            raise ValueError("authorized boundary offset must be within -8..+8")
        if not isinstance(
            self.outgoing_transform, BoundaryTransformState
        ) or not isinstance(self.incoming_transform, BoundaryTransformState):
            raise TypeError("authorized boundary transforms are invalid")
        object.__setattr__(
            self,
            "proof_identity",
            _exact_identifier(self.proof_identity, "boundary proof_identity"),
        )
        scoring_outgoing = self.scoring_outgoing or self.measurement_key.outgoing
        scoring_incoming = self.scoring_incoming or self.measurement_key.incoming
        if not isinstance(scoring_outgoing, TimelineCandidate) or not isinstance(
            scoring_incoming, TimelineCandidate
        ):
            raise TypeError("authorized offset scoring candidates are invalid")
        _validate_offset_scoring_candidate(
            self.measurement_key.outgoing,
            scoring_outgoing,
            side="outgoing",
        )
        _validate_offset_scoring_candidate(
            self.measurement_key.incoming,
            scoring_incoming,
            side="incoming",
        )
        object.__setattr__(self, "scoring_outgoing", scoring_outgoing)
        object.__setattr__(self, "scoring_incoming", scoring_incoming)
        adjustment = _finite_number(
            self.score_adjustment,
            "boundary score_adjustment",
        )
        if adjustment > 0.0:
            raise ValueError(
                "boundary score_adjustment cannot boost or relax the timeline score"
            )
        object.__setattr__(self, "score_adjustment", adjustment)


def _validate_offset_scoring_candidate(
    canonical: TimelineCandidate,
    measured: TimelineCandidate,
    *,
    side: str,
) -> None:
    """Allow only the endpoint-specific descriptor fields to differ."""

    if side not in {"outgoing", "incoming"}:
        raise ValueError("offset scoring side is invalid")
    allowed_prefix = "exit_" if side == "outgoing" else "entry_"
    differing = tuple(
        item.name
        for item in fields(TimelineCandidate)
        if getattr(canonical, item.name) != getattr(measured, item.name)
        and not item.name.startswith(allowed_prefix)
    )
    if differing:
        raise ValueError(
            f"{side} offset scoring candidate changed non-endpoint fields: "
            + ",".join(differing)
        )


class BoundaryOffsetDisposition(str, Enum):
    AUTHORIZED = "authorized"
    HARD_INCOMPATIBLE = "hard_incompatible"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"


@dataclass(frozen=True, slots=True)
class BoundaryOffsetAudit:
    """Immutable evidence outcome for exactly one offset in ``-8..+8``."""

    frame_offset: int
    disposition: BoundaryOffsetDisposition | str
    evidence_identity: str
    failure_codes: tuple[str, ...] = ()
    authorized_option: AuthorizedBoundaryOption | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_offset, bool)
            or not isinstance(self.frame_offset, int)
            or self.frame_offset not in JOINT_BOUNDARY_FRAME_OFFSETS
        ):
            raise ValueError("boundary audit offset must be within -8..+8")
        disposition = (
            self.disposition
            if isinstance(self.disposition, BoundaryOffsetDisposition)
            else BoundaryOffsetDisposition(self.disposition)
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "evidence_identity",
            _exact_identifier(self.evidence_identity, "evidence_identity"),
        )
        codes = _canonical_failure_codes(self.failure_codes)
        object.__setattr__(self, "failure_codes", codes)
        if disposition is BoundaryOffsetDisposition.AUTHORIZED:
            if not isinstance(self.authorized_option, AuthorizedBoundaryOption):
                raise ValueError("authorized audit requires an authorized option")
            if self.authorized_option.frame_offset != self.frame_offset:
                raise ValueError("authorized audit offset and option disagree")
            if self.authorized_option.proof_identity != self.evidence_identity:
                raise ValueError("authorized audit evidence identity is not exact")
            if codes:
                raise ValueError("authorized audit cannot carry failure codes")
        else:
            if self.authorized_option is not None:
                raise ValueError("rejected audit cannot carry an authorized option")
            if not codes:
                raise ValueError("rejected audit requires explicit failure codes")


@dataclass(frozen=True, slots=True)
class BoundaryOptionSet:
    """Complete immutable ``-8..+8`` audit for one exact boundary key."""

    measurement_key: BoundaryMeasurementKey
    offset_audits: tuple[BoundaryOffsetAudit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.measurement_key, BoundaryMeasurementKey):
            raise TypeError("boundary option-set measurement key is invalid")
        raw_audits = tuple(self.offset_audits)
        if any(not isinstance(item, BoundaryOffsetAudit) for item in raw_audits):
            raise TypeError("boundary option-set audits are invalid")
        audits = tuple(sorted(raw_audits, key=lambda item: item.frame_offset))
        if tuple(item.frame_offset for item in audits) != (
            JOINT_BOUNDARY_FRAME_OFFSETS
        ):
            raise ValueError(
                "boundary option set must preserve exactly one audit for every "
                "offset in -8..+8"
            )
        for audit in audits:
            option = audit.authorized_option
            if option is not None and option.measurement_key != self.measurement_key:
                raise ValueError("authorized option belongs to another boundary key")
        object.__setattr__(self, "offset_audits", audits)

    @property
    def authorized_options(self) -> tuple[AuthorizedBoundaryOption, ...]:
        return tuple(
            audit.authorized_option
            for audit in self.offset_audits
            if audit.authorized_option is not None
        )

    @property
    def evidence_incomplete(self) -> bool:
        return any(
            item.disposition is BoundaryOffsetDisposition.EVIDENCE_INCOMPLETE
            for item in self.offset_audits
        )

    @property
    def hard_incompatible(self) -> bool:
        return all(
            item.disposition is BoundaryOffsetDisposition.HARD_INCOMPATIBLE
            for item in self.offset_audits
        )

    def audit_for(self, frame_offset: int) -> BoundaryOffsetAudit:
        if (
            isinstance(frame_offset, bool)
            or not isinstance(frame_offset, int)
            or frame_offset not in JOINT_BOUNDARY_FRAME_OFFSETS
        ):
            raise ValueError("boundary audit lookup offset must be within -8..+8")
        return self.offset_audits[frame_offset - JOINT_BOUNDARY_FRAME_OFFSETS[0]]


@dataclass(frozen=True, slots=True)
class SegmentBridgeProof:
    """Exact crop/track compatibility evidence between adjacent boundaries."""

    candidate: TimelineCandidate
    entry_option: AuthorizedBoundaryOption
    exit_option: AuthorizedBoundaryOption
    duration_frames: float
    proof_identity: str
    evidence_complete: bool
    crop_compatible: bool
    track_compatible: bool
    failure_codes: tuple[str, ...] = ()
    score_adjustment: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, TimelineCandidate):
            raise TypeError("bridge candidate is invalid")
        if not isinstance(
            self.entry_option, AuthorizedBoundaryOption
        ) or not isinstance(self.exit_option, AuthorizedBoundaryOption):
            raise TypeError("bridge boundary options are invalid")
        if self.entry_option.measurement_key.incoming != self.candidate:
            raise ValueError("bridge entry option does not enter its candidate")
        if self.exit_option.measurement_key.outgoing != self.candidate:
            raise ValueError("bridge exit option does not leave its candidate")
        if (
            self.exit_option.measurement_key.boundary_index
            != self.entry_option.measurement_key.boundary_index + 1
        ):
            raise ValueError("bridge options must be adjacent boundaries")
        duration = _finite_number(self.duration_frames, "bridge duration_frames")
        if duration < 0.0:
            raise ValueError("bridge duration_frames cannot be negative")
        object.__setattr__(self, "duration_frames", duration)
        object.__setattr__(
            self,
            "proof_identity",
            _exact_identifier(self.proof_identity, "bridge proof_identity"),
        )
        for name in (
            "evidence_complete",
            "crop_compatible",
            "track_compatible",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"bridge {name} must be a bool")
        codes = _canonical_failure_codes(self.failure_codes)
        object.__setattr__(self, "failure_codes", codes)
        if self.authorized and codes:
            raise ValueError("authorized bridge cannot carry failure codes")
        if not self.authorized and not codes:
            raise ValueError("unauthorized bridge requires explicit failure codes")
        adjustment = _finite_number(
            self.score_adjustment,
            "bridge score_adjustment",
        )
        if adjustment > 0.0:
            raise ValueError(
                "bridge score_adjustment cannot boost or relax the timeline score"
            )
        object.__setattr__(self, "score_adjustment", adjustment)

    @property
    def authorized(self) -> bool:
        return bool(
            self.evidence_complete
            and self.crop_compatible
            and self.track_compatible
        )

    @property
    def evidence_incomplete(self) -> bool:
        return not self.evidence_complete

    @property
    def hard_incompatible(self) -> bool:
        return bool(self.evidence_complete and not self.authorized)


BoundaryOptionProvider: TypeAlias = Callable[
    [BoundaryMeasurementKey], BoundaryOptionSet
]
SegmentBridgeProvider: TypeAlias = Callable[
    [
        TimelineCandidate,
        AuthorizedBoundaryOption,
        AuthorizedBoundaryOption,
        float,
    ],
    SegmentBridgeProof,
]


@dataclass(frozen=True, slots=True)
class JointTimelineProgress:
    """Immutable, privacy-safe progress for a core or profiled joint search.

    Boundary measurements and proofs stay process-local.  Shared-cache counts
    are globally monotonic; layer, option, and bridge counts are monotonic
    within ``profile_index`` and reset only when that index advances.  Callback
    failures propagate, so no partially observed selection is returned.
    """

    phase: str
    completed_boundary_layers: int
    total_boundary_layers: int
    option_key_count: int
    bridge_provider_call_count: int
    shared_bridge_cache_hit_count: int = 0
    shared_bridge_cache_miss_count: int = 0
    shared_bridge_cache_admission_count: int = 0
    shared_bridge_cache_saturation_count: int = 0
    shared_bridge_cache_entry_count: int = 0
    cadence_profile: str | None = None
    profile_index: int = 0
    total_profiles: int = 0
    shared_option_cache_hit_count: int = 0
    shared_option_cache_miss_count: int = 0
    shared_option_cache_entry_count: int = 0
    profile_status: str = "in_progress"
    contract: str = "joint_selection_progress_v1"
    version: int = 1

    def __post_init__(self) -> None:
        if (
            self.contract != "joint_selection_progress_v1"
            or isinstance(self.version, bool)
            or self.version != 1
        ):
            raise ValueError("joint selection progress contract is invalid")
        if self.phase not in {
            "selection_start",
            "boundary_layer_complete",
            "selection_failed",
            "selection_complete",
            "profile_start",
            "profile_complete",
        }:
            raise ValueError("joint selection progress phase is invalid")
        integer_fields = (
            item
            for item in fields(self)
            if item.name.endswith("_count")
            or item.name
            in {
                "completed_boundary_layers",
                "total_boundary_layers",
                "profile_index",
                "total_profiles",
            }
        )
        if any(
            isinstance(value := getattr(self, item.name), bool)
            or not isinstance(value, int)
            or value < 0
            for item in integer_fields
        ):
            raise ValueError("joint selection progress counts must be non-negative")
        if self.completed_boundary_layers > self.total_boundary_layers:
            raise ValueError("completed boundary layers exceed the total")
        if self.shared_bridge_cache_entry_count > (
            self.shared_bridge_cache_admission_count
        ):
            raise ValueError("shared bridge cache entries exceed admissions")
        if self.shared_option_cache_entry_count > self.shared_option_cache_miss_count:
            raise ValueError("shared option cache entries exceed misses")
        if self.cadence_profile is None:
            if self.profile_index != 0 or self.total_profiles != 0:
                raise ValueError("unprofiled progress cannot carry a profile index")
        elif (
            not isinstance(self.cadence_profile, str)
            or not self.cadence_profile
            or self.cadence_profile != self.cadence_profile.strip()
            or not 1 <= self.profile_index <= self.total_profiles
        ):
            raise ValueError("profiled progress identity is invalid")
        if self.profile_status not in {
            "in_progress",
            "completed",
            "no_complete_path",
        }:
            raise ValueError("joint selection progress status is invalid")

    def to_dict(self) -> dict[str, object]:
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        payload["current_profile_option_key_count"] = payload.pop(
            "option_key_count"
        )
        payload["current_profile_bridge_provider_call_count"] = payload.pop(
            "bridge_provider_call_count"
        )
        return payload


JointTimelineProgressCallback: TypeAlias = Callable[[JointTimelineProgress], None]


@dataclass(frozen=True, slots=True)
class JointTimelineResult:
    """Selected timeline plus every exact proof used by the joint DP."""

    timeline: TimelineResult
    boundary_options: tuple[AuthorizedBoundaryOption, ...]
    selected_boundary_option_sets: tuple[BoundaryOptionSet, ...]
    segment_bridge_proofs: tuple[SegmentBridgeProof, ...]
    measured_boundary_option_sets: tuple[BoundaryOptionSet, ...]
    rejection_counts: tuple[tuple[str, int], ...]
    option_provider_call_count: int
    bridge_provider_call_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.timeline, TimelineResult):
            raise TypeError("joint timeline result requires a TimelineResult")
        boundary_count = max(0, len(self.timeline.choices) - 1)
        if len(self.boundary_options) != boundary_count or len(
            self.selected_boundary_option_sets
        ) != boundary_count:
            raise ValueError("joint selected boundary evidence count is invalid")
        if len(self.segment_bridge_proofs) != max(
            0, len(self.timeline.choices) - 2
        ):
            raise ValueError("joint segment bridge proof count is invalid")
        for option, option_set in zip(
            self.boundary_options,
            self.selected_boundary_option_sets,
            strict=True,
        ):
            if option.measurement_key != option_set.measurement_key:
                raise ValueError("selected boundary option-set identity diverged")
            if option not in option_set.authorized_options:
                raise ValueError("selected boundary option is not audited")
        if self.option_provider_call_count != len(
            self.measured_boundary_option_sets
        ):
            raise ValueError("joint option-provider call audit is inconsistent")
        if self.bridge_provider_call_count < len(self.segment_bridge_proofs):
            raise ValueError("joint bridge-provider call audit is inconsistent")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in self.rejection_counts
        ):
            raise ValueError("joint rejection counts are invalid")

    @property
    def choices(self) -> tuple[TimelineChoice, ...]:
        return self.timeline.choices

    @property
    def candidates(self) -> tuple[TimelineCandidate, ...]:
        return self.timeline.candidates

    @property
    def total_score(self) -> float:
        return self.timeline.total_score


@dataclass(frozen=True, slots=True)
class _RendererSemanticState:
    """Future-relevant renderer state without boundary-proof provenance."""

    crop_xywh: tuple[float, float, float, float] | None
    rotation_degrees: float
    track_id: str | None
    subject_basis: str


@dataclass(frozen=True, slots=True)
class _JointState:
    candidate_index: int
    source_id: str
    shot_id: str
    shot_run: int
    source_run: int
    source_before_run: str | None
    previous_boundary_offset: int | None
    endpoint_renderer_state: _RendererSemanticState | None


@dataclass(frozen=True, slots=True)
class _JointNode:
    state: _JointState
    candidate: TimelineCandidate
    cumulative_score: float
    previous_node: _JointNode | None
    emission_score: float
    transition_score: float
    used_fallback: bool
    entering_boundary_option: AuthorizedBoundaryOption | None
    bridge_evidence_for_previous_segment: _SharedBridgeEvidence | None
    path_signature: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class _BridgeSemanticKey:
    candidate: TimelineCandidate
    start_master_seconds: float
    end_master_seconds: float
    duration_frames: float
    entry_renderer_state: _RendererSemanticState
    exit_renderer_state: _RendererSemanticState


@dataclass(frozen=True, slots=True)
class _ExactBridgeRequestKey:
    candidate: TimelineCandidate
    entry_option: AuthorizedBoundaryOption
    exit_option: AuthorizedBoundaryOption
    duration_frames: float


@dataclass(frozen=True, slots=True)
class _SharedBridgeEvidence:
    semantic_key: _BridgeSemanticKey
    evidence_complete: bool
    crop_compatible: bool
    track_compatible: bool
    failure_codes: tuple[str, ...]
    score_adjustment: float

    @property
    def evidence_incomplete(self) -> bool:
        return not self.evidence_complete

    @property
    def hard_incompatible(self) -> bool:
        return bool(
            self.evidence_complete
            and not (self.crop_compatible and self.track_compatible)
        )


class JointTimelineSharedBridgeEvidenceCache:
    """Adapter-lifetime admission-only cache of scalar bridge dispositions.

    The caller bounds admission under the 300,000-entry hard cap; entries are
    never evicted or promoted.  Values retain only scalar compatibility, never
    proof identities, boundary options, materializations, or audit payloads.
    """

    __slots__ = (
        "_admission_count",
        "_binding",
        "_entries",
        "_hit_count",
        "_maximum_entries",
        "_miss_count",
        "_saturation_count",
    )

    def __init__(
        self,
        *,
        maximum_entries: int = JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES,
    ) -> None:
        if (
            isinstance(maximum_entries, bool)
            or not isinstance(maximum_entries, int)
            or not 0
            <= maximum_entries
            <= JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES
        ):
            raise ValueError("shared bridge cache maximum_entries is invalid")
        self._entries: dict[_BridgeSemanticKey, _SharedBridgeEvidence] = {}
        self._maximum_entries = maximum_entries
        self._binding: tuple[tuple[object, ...], float] | None = None
        self._hit_count = 0
        self._miss_count = 0
        self._admission_count = 0
        self._saturation_count = 0

    def _bind(self, bridge_provider: SegmentBridgeProvider, *, fps: float) -> None:
        if not callable(bridge_provider):
            raise TypeError("shared bridge cache provider binding is invalid")
        resolved_fps = _finite_number(fps, "shared bridge cache fps")
        if resolved_fps <= 0.0:
            raise ValueError("shared bridge cache fps must be positive")
        bound_self = getattr(bridge_provider, "__self__", None)
        bound_function = getattr(bridge_provider, "__func__", None)
        callable_identity = (
            (bound_self, bound_function)
            if bound_self is not None and bound_function is not None
            else (bridge_provider,)
        )
        binding = (callable_identity, resolved_fps)
        if self._binding is None:
            self._binding = binding
            return
        previous_identity, previous_fps = self._binding
        same_callable = len(previous_identity) == len(callable_identity) and all(
            left is right
            for left, right in zip(
                previous_identity,
                callable_identity,
                strict=True,
            )
        )
        if not same_callable or previous_fps != resolved_fps:
            raise ValueError(
                "shared bridge cache is bound to another provider or fps"
            )

    def _locate(
        self,
        key: _BridgeSemanticKey,
    ) -> _SharedBridgeEvidence | None:
        if self._binding is None:
            raise RuntimeError("shared bridge cache must be bound before lookup")
        if not isinstance(key, _BridgeSemanticKey):
            raise TypeError("shared bridge cache key is invalid")
        cached = self._entries.get(key)
        if cached is None:
            self._miss_count += 1
            return None
        self._hit_count += 1
        return cached

    def _admit(self, evidence: _SharedBridgeEvidence) -> bool:
        if self._binding is None:
            raise RuntimeError("shared bridge cache must be bound before admission")
        if not isinstance(evidence, _SharedBridgeEvidence):
            raise TypeError("shared bridge cache evidence is invalid")
        key = evidence.semantic_key
        if len(self._entries) >= self._maximum_entries:
            self._saturation_count += 1
            return False
        self._entries[key] = evidence
        self._admission_count += 1
        return True

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def admission_count(self) -> int:
        return self._admission_count

    @property
    def saturation_count(self) -> int:
        return self._saturation_count

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _progress_counts(self) -> dict[str, int]:
        return {
            "shared_bridge_cache_hit_count": self.hit_count,
            "shared_bridge_cache_miss_count": self.miss_count,
            "shared_bridge_cache_admission_count": self.admission_count,
            "shared_bridge_cache_saturation_count": self.saturation_count,
            "shared_bridge_cache_entry_count": self.entry_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": "joint_timeline_shared_bridge_evidence_cache_v1",
            "admission_policy": "fixed_cap_no_eviction_v1",
            "maximum_entry_count": self._maximum_entries,
            "hard_maximum_entry_count": (
                JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES
            ),
            "entry_count": self.entry_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "admission_count": self.admission_count,
            "saturation_count": self.saturation_count,
            "scalar_evidence_only": True,
            "proof_identities_retained": False,
            "boundary_options_retained": False,
            "materializations_retained": False,
        }


@dataclass(frozen=True, slots=True)
class _Extension:
    current_index: int
    emission_score: float
    used_fallback: bool
    base_transition_score: float
    shot_run: int
    source_run: int
    source_before_run: str | None
    boundary_option: AuthorizedBoundaryOption
    bridge_evidence: _SharedBridgeEvidence | None


def _renderer_semantic_state(
    transform: BoundaryTransformState,
) -> _RendererSemanticState:
    """Strip proof-only identity while preserving every rendered endpoint value."""

    return _RendererSemanticState(
        crop_xywh=transform.crop_xywh,
        rotation_degrees=transform.rotation_degrees,
        track_id=transform.track_id,
        subject_basis=transform.subject_basis,
    )


def _renderer_semantic_sort_key(
    state: _RendererSemanticState | None,
) -> tuple[object, ...]:
    if state is None:
        return (0, (), 0.0, "", "")
    return (
        1,
        state.crop_xywh or (),
        state.rotation_degrees,
        state.track_id or "",
        state.subject_basis,
    )


def _state_sort_key(state: _JointState) -> tuple[object, ...]:
    return (
        state.candidate_index,
        state.source_id,
        state.shot_id,
        state.shot_run,
        state.source_run,
        state.source_before_run or "",
        state.previous_boundary_offset
        if state.previous_boundary_offset is not None
        else -10_000,
        _renderer_semantic_sort_key(state.endpoint_renderer_state),
    )


def _measurement_sort_key(key: BoundaryMeasurementKey) -> tuple[object, ...]:
    return (
        key.boundary_index,
        key.outgoing.segment_index,
        key.outgoing.source_id,
        key.outgoing.shot_id,
        key.outgoing.candidate_id,
        key.incoming.source_id,
        key.incoming.shot_id,
        key.incoming.candidate_id,
    )


def _candidate_signature(candidate: TimelineCandidate) -> tuple[str, ...]:
    return (
        "candidate",
        str(candidate.segment_index),
        candidate.candidate_id,
        candidate.source_id,
        candidate.shot_id,
        format(candidate.start_seconds, ".17g"),
        format(candidate.end_seconds, ".17g"),
    )


def _extended_path_signature(
    previous: tuple[tuple[str, ...], ...],
    option: AuthorizedBoundaryOption,
    candidate: TimelineCandidate,
) -> tuple[tuple[str, ...], ...]:
    return previous + (
        (
            "boundary",
            str(option.measurement_key.boundary_index),
            option.proof_identity,
            str(option.frame_offset),
        ),
        _candidate_signature(candidate),
    )


def _node_rank_key(node: _JointNode) -> tuple[object, ...]:
    return (
        -node.cumulative_score,
        _state_sort_key(node.state),
        node.path_signature,
    )


def _retain_ranked_node(
    bucket: list[_JointNode],
    node: _JointNode,
    *,
    maximum_results: int,
) -> None:
    for index, existing in enumerate(bucket):
        if existing.path_signature != node.path_signature:
            continue
        if node.cumulative_score > existing.cumulative_score + 1e-12:
            bucket[index] = node
        break
    else:
        bucket.append(node)
    bucket.sort(key=_node_rank_key)
    del bucket[maximum_results:]


def _resolve_mode_and_target(
    mode: TimelineMode | EditMode | str,
    target_member: IVEMember | str | None,
) -> tuple[TimelineMode, IVEMember | None]:
    if isinstance(mode, TimelineMode):
        resolved_mode = mode
    elif isinstance(mode, EditMode):
        resolved_mode = (
            TimelineMode.MEMBER
            if mode is EditMode.MEMBER
            else TimelineMode.GENERAL
        )
    else:
        normalized = str(mode).strip().lower()
        if normalized == EditMode.GROUP.value:
            normalized = TimelineMode.GENERAL.value
        resolved_mode = TimelineMode(normalized)
    target = IVEMember.parse(target_member) if target_member is not None else None
    if resolved_mode is TimelineMode.MEMBER and target is None:
        raise ValueError("target_member is required in member mode")
    return resolved_mode, target


def _no_path_reason(rejection_counts: Mapping[str, int]) -> str:
    incomplete = sum(
        count
        for reason, count in rejection_counts.items()
        if "evidence_incomplete" in reason
    )
    hard = sum(
        count
        for reason, count in rejection_counts.items()
        if "hard_incompatible" in reason
        or reason
        in {
            "boundary_option_timeline_hard_gate",
            "exact_boundary_no_good",
            "segment_duration_below_two_frames",
        }
    )
    if incomplete and not hard:
        return "joint_timeline_evidence_incomplete"
    if hard and not incomplete:
        return "joint_timeline_hard_incompatible"
    return "joint_timeline_state_graph_exhausted"


def _select_joint_timeline_candidates_core(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
    *,
    option_provider: BoundaryOptionProvider,
    bridge_provider: SegmentBridgeProvider | None = None,
    fps: float = 29.97,
    mode: TimelineMode | EditMode | str = TimelineMode.GENERAL,
    target_member: IVEMember | str | None = None,
    config: TimelineConfig | None = None,
    preferred_source_ids: Sequence[str | None] | None = None,
    maximum_results: int,
    emission_adjustments: Mapping[tuple[int, str], float] | None = None,
    source_switch_bonus_by_segment: Mapping[int, float] | None = None,
    forbidden_boundary_edges: (
        TimelineNoGoodSet | Sequence[TimelineNoGoodEdgeKey]
    ) = (),
    progress_callback: JointTimelineProgressCallback | None = None,
    shared_bridge_evidence_cache: (
        JointTimelineSharedBridgeEvidenceCache | None
    ) = None,
) -> tuple[JointTimelineResult, ...]:
    """Select ranked candidates and exact boundary options in one finite DP.

    ``option_provider`` is memoized by ``BoundaryMeasurementKey``.  It must
    return all 17 offset audits even when only one offset is usable.  A bridge
    provider is required when the timeline has a segment between two editable
    boundaries.  Its disposition and score must be invariant for equal
    candidate clocks and numeric crop/rotation/track endpoint states.  That
    semantic evidence is shared during optimization, while each selected
    backtrace is re-requested with its exact boundary options so proof
    provenance is never collapsed.  ``forbidden_boundary_edges`` is reserved
    for exact edges already proven hard-incompatible; every key is validated
    against this lattice and rejected before the option provider can decode it.
    """

    if not callable(option_provider):
        raise TypeError("option_provider must be callable")
    if bridge_provider is not None and not callable(bridge_provider):
        raise TypeError("bridge_provider must be callable or None")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if shared_bridge_evidence_cache is not None and not isinstance(
        shared_bridge_evidence_cache,
        JointTimelineSharedBridgeEvidenceCache,
    ):
        raise TypeError(
            "shared_bridge_evidence_cache must be a "
            "JointTimelineSharedBridgeEvidenceCache or None"
        )
    resolved_fps = _finite_number(fps, "fps")
    if resolved_fps <= 0.0:
        raise ValueError("fps must be positive")
    if (
        isinstance(maximum_results, bool)
        or not isinstance(maximum_results, int)
        or not 1 <= maximum_results <= 8
    ):
        raise ValueError("maximum_results must be between 1 and 8")
    settings = config or TimelineConfig()
    if not isinstance(settings, TimelineConfig):
        raise TypeError("config must be a TimelineConfig")
    timeline_mode, target = _resolve_mode_and_target(mode, target_member)
    segments = _normalise_segments(candidates_by_segment)
    no_goods = compile_timeline_no_good_constraint_index(
        segments,
        forbidden_boundary_edges,
    ).no_goods
    if len(segments) >= 3 and bridge_provider is None:
        raise ValueError(
            "bridge_provider is required for segments between adjacent boundaries"
        )
    if settings.maximum_same_source_hold_seconds is not None:
        oversized = next(
            (
                segment[0]
                for segment in segments
                if segment[0].end_seconds - segment[0].start_seconds
                > settings.maximum_same_source_hold_seconds
            ),
            None,
        )
        if oversized is not None:
            raise ValueError(
                f"segment {oversized.segment_index} is longer than the maximum "
                "same-source hold seconds"
            )

    adjustments = dict(emission_adjustments or {})
    for key, value in adjustments.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or isinstance(key[0], bool)
            or not isinstance(key[0], int)
            or not isinstance(key[1], str)
        ):
            raise ValueError(
                "emission adjustment keys must be (segment_index, source_id)"
            )
        if not isfinite(float(value)) or float(value) > 0.0:
            raise ValueError(
                "emission adjustments must be finite non-positive values"
            )
    switch_bonus_overrides = dict(source_switch_bonus_by_segment or {})
    valid_segment_indices = {
        segment[0].segment_index for segment in segments
    }
    for segment_index, value in switch_bonus_overrides.items():
        if (
            isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or segment_index not in valid_segment_indices
        ):
            raise ValueError(
                "source-switch bonus override keys must be timeline segment indices"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                "source-switch bonus overrides must be finite and non-negative"
            )

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
                score, used_fallback = emission
                eligible[candidate_index] = (
                    score
                    + float(
                        adjustments.get(
                            (candidate.segment_index, candidate.source_id),
                            0.0,
                        )
                    ),
                    used_fallback,
                )
        if not eligible:
            raise ValueError(
                f"segment {segment[0].segment_index} has no safe candidate for "
                f"{target.value if target else timeline_mode.value}"
            )
        emissions.append(eligible)

    preferred_sources = (
        tuple(preferred_source_ids)
        if preferred_source_ids is not None
        else None
    )
    if preferred_sources is not None:
        if timeline_mode is not TimelineMode.GENERAL:
            raise ValueError(
                "preferred source assignments are only supported in general mode"
            )
        if len(preferred_sources) != len(segments):
            raise ValueError(
                "preferred source assignments must match timeline segments"
            )
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

    if shared_bridge_evidence_cache is not None and bridge_provider is not None:
        shared_bridge_evidence_cache._bind(
            bridge_provider,
            fps=resolved_fps,
        )
    option_cache: dict[BoundaryMeasurementKey, BoundaryOptionSet] = {}
    semantic_bridge_cache: dict[
        _BridgeSemanticKey,
        _SharedBridgeEvidence,
    ] = {}
    selected_exact_bridge_cache: dict[
        _ExactBridgeRequestKey,
        SegmentBridgeProof,
    ] = {}
    bridge_provider_call_count = 0
    rejection_counts: dict[str, int] = {}
    layer_rejection_counts: dict[str, int] = {}
    total_boundary_layers = max(0, len(segments) - 1)

    def report_progress(phase: str, completed_boundary_layers: int) -> None:
        if progress_callback is None:
            return
        progress_callback(
            JointTimelineProgress(
                phase=phase,
                completed_boundary_layers=completed_boundary_layers,
                total_boundary_layers=total_boundary_layers,
                option_key_count=len(option_cache),
                bridge_provider_call_count=bridge_provider_call_count,
                **(
                    shared_bridge_evidence_cache._progress_counts()
                    if shared_bridge_evidence_cache is not None
                    else {}
                ),
            )
        )

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        layer_rejection_counts[reason] = (
            layer_rejection_counts.get(reason, 0) + 1
        )

    def locate_options(key: BoundaryMeasurementKey) -> BoundaryOptionSet:
        cached = option_cache.get(key)
        if cached is not None:
            return cached
        measured = option_provider(key)
        if not isinstance(measured, BoundaryOptionSet):
            raise TypeError("option_provider returned an invalid option set")
        if measured.measurement_key != key:
            raise RuntimeError("option_provider returned evidence for another key")
        option_cache[key] = measured
        return measured

    def call_bridge_provider(
        candidate: TimelineCandidate,
        entry_option: AuthorizedBoundaryOption,
        exit_option: AuthorizedBoundaryOption,
        duration_frames: float,
    ) -> SegmentBridgeProof:
        nonlocal bridge_provider_call_count
        assert bridge_provider is not None
        proof = bridge_provider(
            candidate,
            entry_option,
            exit_option,
            duration_frames,
        )
        if not isinstance(proof, SegmentBridgeProof):
            raise TypeError("bridge_provider returned an invalid proof")
        if (
            proof.candidate != candidate
            or proof.entry_option != entry_option
            or proof.exit_option != exit_option
            or abs(proof.duration_frames - duration_frames) > 1e-9
        ):
            raise RuntimeError("bridge_provider returned proof for another request")
        bridge_provider_call_count += 1
        return proof

    def bridge_semantic_key(
        candidate: TimelineCandidate,
        entry_option: AuthorizedBoundaryOption,
        exit_option: AuthorizedBoundaryOption,
        duration_frames: float,
    ) -> _BridgeSemanticKey:
        return _BridgeSemanticKey(
            candidate=candidate,
            start_master_seconds=(
                candidate.start_seconds
                + entry_option.frame_offset / resolved_fps
            ),
            end_master_seconds=(
                candidate.end_seconds
                + exit_option.frame_offset / resolved_fps
            ),
            duration_frames=duration_frames,
            entry_renderer_state=_renderer_semantic_state(
                entry_option.incoming_transform
            ),
            exit_renderer_state=_renderer_semantic_state(
                exit_option.outgoing_transform
            ),
        )

    def locate_bridge_evidence(
        candidate: TimelineCandidate,
        entry_option: AuthorizedBoundaryOption,
        exit_option: AuthorizedBoundaryOption,
        duration_frames: float,
    ) -> _SharedBridgeEvidence:
        semantic_key = bridge_semantic_key(
            candidate,
            entry_option,
            exit_option,
            duration_frames,
        )
        cached = semantic_bridge_cache.get(semantic_key)
        if cached is not None:
            return cached
        if shared_bridge_evidence_cache is not None:
            shared = shared_bridge_evidence_cache._locate(semantic_key)
            if shared is not None:
                semantic_bridge_cache[semantic_key] = shared
                return shared
        proof = call_bridge_provider(
            candidate,
            entry_option,
            exit_option,
            duration_frames,
        )
        evidence = _SharedBridgeEvidence(
            semantic_key=semantic_key,
            evidence_complete=proof.evidence_complete,
            crop_compatible=proof.crop_compatible,
            track_compatible=proof.track_compatible,
            failure_codes=proof.failure_codes,
            score_adjustment=proof.score_adjustment,
        )
        semantic_bridge_cache[semantic_key] = evidence
        if shared_bridge_evidence_cache is not None:
            shared_bridge_evidence_cache._admit(evidence)
        return evidence

    def locate_selected_exact_bridge(
        evidence: _SharedBridgeEvidence,
        candidate: TimelineCandidate,
        entry_option: AuthorizedBoundaryOption,
        exit_option: AuthorizedBoundaryOption,
        duration_frames: float,
    ) -> SegmentBridgeProof:
        semantic_key = bridge_semantic_key(
            candidate,
            entry_option,
            exit_option,
            duration_frames,
        )
        if semantic_key != evidence.semantic_key:
            raise RuntimeError(
                "joint DP backtrace changed renderer-semantic bridge state"
            )
        exact_key = _ExactBridgeRequestKey(
            candidate,
            entry_option,
            exit_option,
            duration_frames,
        )
        cached = selected_exact_bridge_cache.get(exact_key)
        if cached is not None:
            return cached
        proof = call_bridge_provider(
            candidate,
            entry_option,
            exit_option,
            duration_frames,
        )
        if (
            proof.evidence_complete != evidence.evidence_complete
            or proof.crop_compatible != evidence.crop_compatible
            or proof.track_compatible != evidence.track_compatible
            or proof.failure_codes != evidence.failure_codes
            or proof.score_adjustment != evidence.score_adjustment
        ):
            raise RuntimeError(
                "bridge_provider diverged within one renderer-semantic state"
            )
        selected_exact_bridge_cache[exact_key] = proof
        return proof

    first_layer: dict[_JointState, list[_JointNode]] = {}
    for candidate_index, (emission_score, used_fallback) in emissions[0].items():
        candidate = segments[0][candidate_index]
        state = _JointState(
            candidate_index=candidate_index,
            source_id=candidate.source_id,
            shot_id=candidate.shot_id,
            shot_run=1,
            source_run=1,
            source_before_run=None,
            previous_boundary_offset=None,
            endpoint_renderer_state=None,
        )
        first_layer[state] = [
            _JointNode(
                state=state,
                candidate=candidate,
                cumulative_score=emission_score,
                previous_node=None,
                emission_score=emission_score,
                transition_score=0.0,
                used_fallback=used_fallback,
                entering_boundary_option=None,
                bridge_evidence_for_previous_segment=None,
                path_signature=(_candidate_signature(candidate),),
            )
        ]
    frontier = first_layer
    report_progress("selection_start", 0)

    for segment_position in range(1, len(segments)):
        layer_rejection_counts = {}
        previous_segment = segments[segment_position - 1]
        current_segment = segments[segment_position]
        current_layer: dict[_JointState, list[_JointNode]] = {}
        for previous_state, previous_nodes in sorted(
            frontier.items(),
            key=lambda item: _state_sort_key(item[0]),
        ):
            previous_candidate = previous_segment[
                previous_state.candidate_index
            ]
            ranked_previous_nodes = tuple(
                sorted(previous_nodes, key=_node_rank_key)
            )
            representative_entry_option: AuthorizedBoundaryOption | None = None
            if segment_position >= 2:
                expected_entry_state = previous_state.endpoint_renderer_state
                expected_entry_offset = previous_state.previous_boundary_offset
                if expected_entry_state is None or expected_entry_offset is None:
                    raise RuntimeError(
                        "joint DP lost its previous renderer endpoint state"
                    )
                for previous_node in ranked_previous_nodes:
                    entry_option = previous_node.entering_boundary_option
                    if entry_option is None:
                        raise RuntimeError(
                            "joint DP lost an exact previous boundary option"
                        )
                    if (
                        entry_option.frame_offset != expected_entry_offset
                        or _renderer_semantic_state(
                            entry_option.incoming_transform
                        )
                        != expected_entry_state
                    ):
                        raise RuntimeError(
                            "joint DP renderer-semantic state is not exact"
                        )
                representative_entry_option = (
                    ranked_previous_nodes[0].entering_boundary_option
                )
                assert representative_entry_option is not None
            extensions: list[_Extension] = []
            for current_index, (emission_score, used_fallback) in emissions[
                segment_position
            ].items():
                current_candidate = current_segment[current_index]
                if no_goods and build_timeline_no_good_edge_key(
                    segment_position - 1,
                    previous_candidate,
                    current_candidate,
                ) in no_goods:
                    reject("exact_boundary_no_good")
                    continue
                measurement_key = BoundaryMeasurementKey.from_candidates(
                    segment_position - 1,
                    previous_candidate,
                    current_candidate,
                )
                option_set = locate_options(measurement_key)
                options = option_set.authorized_options
                if not options:
                    reject(
                        "boundary_evidence_incomplete"
                        if option_set.evidence_incomplete
                        else "boundary_hard_incompatible"
                    )
                    continue
                source_run = (
                    previous_state.source_run + 1
                    if previous_candidate.source_id == current_candidate.source_id
                    else 1
                )
                source_before_run = (
                    previous_state.source_before_run
                    if previous_candidate.source_id == current_candidate.source_id
                    else previous_candidate.source_id
                )
                for option in options:
                    first_duration_frames = (
                        (
                            previous_candidate.end_seconds
                            - previous_candidate.start_seconds
                        )
                        * resolved_fps
                        + option.frame_offset
                    )
                    if (
                        segment_position == 1
                        and first_duration_frames
                        < MINIMUM_JOINT_SEGMENT_DURATION_FRAMES - 1e-9
                    ):
                        reject("segment_duration_below_two_frames")
                        continue
                    last_duration_frames = (
                        (
                            current_candidate.end_seconds
                            - current_candidate.start_seconds
                        )
                        * resolved_fps
                        - option.frame_offset
                    )
                    if (
                        segment_position == len(segments) - 1
                        and last_duration_frames
                        < MINIMUM_JOINT_SEGMENT_DURATION_FRAMES - 1e-9
                    ):
                        reject("segment_duration_below_two_frames")
                        continue
                    assert option.scoring_outgoing is not None
                    assert option.scoring_incoming is not None
                    transition = _transition_score(
                        option.scoring_outgoing,
                        option.scoring_incoming,
                        previous_state.shot_run,
                        settings,
                    )
                    if transition is None:
                        reject("boundary_option_timeline_hard_gate")
                        continue
                    base_transition_score, shot_run = transition
                    bridge_evidence: _SharedBridgeEvidence | None = None
                    if segment_position >= 2:
                        entry_option = representative_entry_option
                        assert entry_option is not None
                        duration_frames = (
                            (
                                previous_candidate.end_seconds
                                - previous_candidate.start_seconds
                            )
                            * resolved_fps
                            + option.frame_offset
                            - entry_option.frame_offset
                        )
                        if (
                            duration_frames
                            < MINIMUM_JOINT_SEGMENT_DURATION_FRAMES - 1e-9
                        ):
                            reject("segment_duration_below_two_frames")
                            continue
                        bridge_evidence = locate_bridge_evidence(
                            previous_candidate,
                            entry_option,
                            option,
                            duration_frames,
                        )
                        if bridge_evidence.evidence_incomplete:
                            reject("bridge_evidence_incomplete")
                            continue
                        if bridge_evidence.hard_incompatible:
                            reject("bridge_hard_incompatible")
                            continue
                    extensions.append(
                        _Extension(
                            current_index=current_index,
                            emission_score=emission_score,
                            used_fallback=used_fallback,
                            base_transition_score=base_transition_score,
                            shot_run=shot_run,
                            source_run=source_run,
                            source_before_run=source_before_run,
                            boundary_option=option,
                            bridge_evidence=bridge_evidence,
                        )
                    )

            for extension in extensions:
                current_candidate = current_segment[extension.current_index]
                same_source = (
                    previous_candidate.source_id == current_candidate.source_id
                )
                comparable_alternate_exists = any(
                    previous_candidate.source_id
                    != current_segment[alternate.current_index].source_id
                    and alternate.emission_score
                    >= extension.emission_score
                    - settings.maximum_switch_score_sacrifice
                    for alternate in extensions
                )
                source_run_start_position = (
                    segment_position - extension.source_run + 1
                )
                source_run_seconds = (
                    current_candidate.end_seconds
                    - segments[source_run_start_position][0].start_seconds
                )
                if (
                    same_source
                    and settings.maximum_same_source_hold_segments is not None
                    and extension.source_run
                    > settings.maximum_same_source_hold_segments
                    and comparable_alternate_exists
                ):
                    reject("same_source_segment_hold_cap")
                    continue
                if (
                    same_source
                    and settings.maximum_same_source_hold_seconds is not None
                    and source_run_seconds
                    > settings.maximum_same_source_hold_seconds
                ):
                    reject("same_source_duration_hold_cap")
                    continue
                cadence_score = (
                    float(
                        switch_bonus_overrides.get(
                            current_candidate.segment_index,
                            settings.source_switch_bonus,
                        )
                    )
                    if not same_source
                    else 0.0
                )
                if (
                    same_source
                    and settings.preferred_max_source_hold_segments is not None
                    and extension.source_run
                    > settings.preferred_max_source_hold_segments
                    and comparable_alternate_exists
                ):
                    cadence_score = -settings.source_hold_penalty * (
                        extension.source_run
                        - settings.preferred_max_source_hold_segments
                    )
                if (
                    same_source
                    and settings.preferred_max_source_hold_seconds is not None
                    and source_run_seconds
                    > settings.preferred_max_source_hold_seconds
                    and comparable_alternate_exists
                ):
                    cadence_score -= settings.source_hold_penalty * (
                        source_run_seconds
                        - settings.preferred_max_source_hold_seconds
                    )
                if (
                    not same_source
                    and previous_state.source_before_run is not None
                    and current_candidate.source_id
                    == previous_state.source_before_run
                ):
                    cadence_score -= settings.ping_pong_penalty
                transition_score = (
                    extension.base_transition_score
                    + cadence_score
                    + extension.boundary_option.score_adjustment
                    + (
                        extension.bridge_evidence.score_adjustment
                        if extension.bridge_evidence is not None
                        else 0.0
                    )
                )
                option = extension.boundary_option
                state = _JointState(
                    candidate_index=extension.current_index,
                    source_id=current_candidate.source_id,
                    shot_id=current_candidate.shot_id,
                    shot_run=extension.shot_run,
                    source_run=extension.source_run,
                    source_before_run=extension.source_before_run,
                    previous_boundary_offset=option.frame_offset,
                    endpoint_renderer_state=_renderer_semantic_state(
                        option.incoming_transform
                    ),
                )
                bucket = current_layer.setdefault(state, [])
                for previous_node in ranked_previous_nodes:
                    cumulative = (
                        previous_node.cumulative_score
                        + transition_score
                        + extension.emission_score
                    )
                    node = _JointNode(
                        state=state,
                        candidate=current_candidate,
                        cumulative_score=cumulative,
                        previous_node=previous_node,
                        emission_score=extension.emission_score,
                        transition_score=transition_score,
                        used_fallback=extension.used_fallback,
                        entering_boundary_option=option,
                        bridge_evidence_for_previous_segment=(
                            extension.bridge_evidence
                        ),
                        path_signature=_extended_path_signature(
                            previous_node.path_signature,
                            option,
                            current_candidate,
                        ),
                    )
                    _retain_ranked_node(
                        bucket,
                        node,
                        maximum_results=maximum_results,
                    )
        if not current_layer:
            reason_code = _no_path_reason(layer_rejection_counts)
            summary = ",".join(
                f"{reason}:{count}"
                for reason, count in sorted(layer_rejection_counts.items())
            ) or "unknown:0"
            report_progress("selection_failed", segment_position - 1)
            raise NoCompleteTimelinePath(
                f"segment {current_segment[0].segment_index} has no joint-safe "
                f"candidate; rejection_counts={summary}",
                reason_code=reason_code,
                segment_index=current_segment[0].segment_index,
                rejection_counts=layer_rejection_counts,
            )
        frontier = current_layer
        report_progress("boundary_layer_complete", segment_position)

    final_by_signature: dict[
        tuple[tuple[str, ...], ...],
        _JointNode,
    ] = {}
    for bucket in frontier.values():
        for node in bucket:
            existing = final_by_signature.get(node.path_signature)
            if (
                existing is None
                or node.cumulative_score > existing.cumulative_score + 1e-12
            ):
                final_by_signature[node.path_signature] = node
    ranked_final_nodes = tuple(
        sorted(final_by_signature.values(), key=_node_rank_key)[:maximum_results]
    )
    measured_option_sets = tuple(
        option_cache[key]
        for key in sorted(option_cache, key=_measurement_sort_key)
    )

    pending_results: list[
        tuple[
            TimelineResult,
            tuple[AuthorizedBoundaryOption, ...],
            tuple[BoundaryOptionSet, ...],
            tuple[SegmentBridgeProof, ...],
        ]
    ] = []
    for final_node in ranked_final_nodes:
        selected_nodes: list[_JointNode] = []
        node: _JointNode | None = final_node
        while node is not None:
            selected_nodes.append(node)
            node = node.previous_node
        selected_nodes.reverse()
        if len(selected_nodes) != len(segments):
            raise RuntimeError("joint timeline backtrace ended prematurely")

        choices: list[TimelineChoice] = []
        boundary_options: list[AuthorizedBoundaryOption] = []
        bridge_proofs: list[SegmentBridgeProof] = []
        for selected_position, selected_node in enumerate(selected_nodes):
            choices.append(
                TimelineChoice(
                    candidate=selected_node.candidate,
                    emission_score=selected_node.emission_score,
                    transition_score=selected_node.transition_score,
                    cumulative_score=selected_node.cumulative_score,
                    used_group_wide_fallback=selected_node.used_fallback,
                )
            )
            if selected_node.entering_boundary_option is not None:
                boundary_options.append(
                    selected_node.entering_boundary_option
                )
            bridge_evidence = (
                selected_node.bridge_evidence_for_previous_segment
            )
            if bridge_evidence is not None:
                if selected_position < 2:
                    raise RuntimeError(
                        "joint DP attached bridge evidence before a middle segment"
                    )
                entry_option = selected_nodes[
                    selected_position - 1
                ].entering_boundary_option
                exit_option = selected_node.entering_boundary_option
                if entry_option is None or exit_option is None:
                    raise RuntimeError(
                        "joint DP backtrace lost exact bridge boundary options"
                    )
                middle_candidate = selected_nodes[
                    selected_position - 1
                ].candidate
                duration_frames = (
                    middle_candidate.duration_seconds * resolved_fps
                    + exit_option.frame_offset
                    - entry_option.frame_offset
                )
                exact_bridge = locate_selected_exact_bridge(
                    bridge_evidence,
                    middle_candidate,
                    entry_option,
                    exit_option,
                    duration_frames,
                )
                if not exact_bridge.authorized:
                    raise RuntimeError(
                        "selected exact bridge lost semantic authorization"
                    )
                bridge_proofs.append(exact_bridge)
            elif selected_position >= 2:
                raise RuntimeError(
                    "joint DP backtrace lost middle-segment bridge evidence"
                )

        timeline = TimelineResult(
            mode=timeline_mode,
            target_member=target,
            choices=tuple(choices),
            total_score=choices[-1].cumulative_score,
        )
        selected_option_sets = tuple(
            option_cache[option.measurement_key]
            for option in boundary_options
        )
        pending_results.append(
            (
                timeline,
                tuple(boundary_options),
                selected_option_sets,
                tuple(bridge_proofs),
            )
        )
    results = tuple(
        JointTimelineResult(
            timeline=timeline,
            boundary_options=boundary_options,
            selected_boundary_option_sets=selected_option_sets,
            segment_bridge_proofs=bridge_proofs,
            measured_boundary_option_sets=measured_option_sets,
            rejection_counts=tuple(sorted(rejection_counts.items())),
            option_provider_call_count=len(option_cache),
            bridge_provider_call_count=bridge_provider_call_count,
        )
        for (
            timeline,
            boundary_options,
            selected_option_sets,
            bridge_proofs,
        ) in pending_results
    )
    report_progress("selection_complete", total_boundary_layers)
    return results


def select_joint_timeline(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
    *,
    option_provider: BoundaryOptionProvider,
    bridge_provider: SegmentBridgeProvider | None = None,
    fps: float = 29.97,
    mode: TimelineMode | EditMode | str = TimelineMode.GENERAL,
    target_member: IVEMember | str | None = None,
    config: TimelineConfig | None = None,
    preferred_source_ids: Sequence[str | None] | None = None,
    emission_adjustments: Mapping[tuple[int, str], float] | None = None,
    source_switch_bonus_by_segment: Mapping[int, float] | None = None,
    forbidden_boundary_edges: (
        TimelineNoGoodSet | Sequence[TimelineNoGoodEdgeKey]
    ) = (),
    progress_callback: JointTimelineProgressCallback | None = None,
    shared_bridge_evidence_cache: (
        JointTimelineSharedBridgeEvidenceCache | None
    ) = None,
) -> JointTimelineResult:
    """Return the single best fully authorized joint timeline."""

    return select_joint_timeline_candidates(
        candidates_by_segment,
        option_provider=option_provider,
        bridge_provider=bridge_provider,
        fps=fps,
        mode=mode,
        target_member=target_member,
        config=config,
        preferred_source_ids=preferred_source_ids,
        maximum_results=1,
        emission_adjustments=emission_adjustments,
        source_switch_bonus_by_segment=source_switch_bonus_by_segment,
        forbidden_boundary_edges=forbidden_boundary_edges,
        progress_callback=progress_callback,
        shared_bridge_evidence_cache=shared_bridge_evidence_cache,
    )[0]


def select_joint_timeline_candidates(
    candidates_by_segment: Mapping[int, Sequence[TimelineCandidate]]
    | Sequence[Sequence[TimelineCandidate]],
    *,
    option_provider: BoundaryOptionProvider,
    bridge_provider: SegmentBridgeProvider | None = None,
    fps: float = 29.97,
    mode: TimelineMode | EditMode | str = TimelineMode.GENERAL,
    target_member: IVEMember | str | None = None,
    config: TimelineConfig | None = None,
    preferred_source_ids: Sequence[str | None] | None = None,
    maximum_results: int = 8,
    emission_adjustments: Mapping[tuple[int, str], float] | None = None,
    source_switch_bonus_by_segment: Mapping[int, float] | None = None,
    forbidden_boundary_edges: (
        TimelineNoGoodSet | Sequence[TimelineNoGoodEdgeKey]
    ) = (),
    progress_callback: JointTimelineProgressCallback | None = None,
    shared_bridge_evidence_cache: (
        JointTimelineSharedBridgeEvidenceCache | None
    ) = None,
) -> tuple[JointTimelineResult, ...]:
    """Return up to eight distinct source/shot/boundary-proof paths.

    The k-best Viterbi keeps at most ``maximum_results`` unique path
    signatures per future-equivalent state.  Boundary and bridge providers are
    memoized once for the whole search, so increasing K never remeasures an
    exact request.
    """

    return _select_joint_timeline_candidates_core(
        candidates_by_segment,
        option_provider=option_provider,
        bridge_provider=bridge_provider,
        fps=fps,
        mode=mode,
        target_member=target_member,
        config=config,
        preferred_source_ids=preferred_source_ids,
        maximum_results=maximum_results,
        emission_adjustments=emission_adjustments,
        source_switch_bonus_by_segment=source_switch_bonus_by_segment,
        forbidden_boundary_edges=forbidden_boundary_edges,
        progress_callback=progress_callback,
        shared_bridge_evidence_cache=shared_bridge_evidence_cache,
    )


__all__ = [
    "AuthorizedBoundaryOption",
    "BoundaryMeasurementKey",
    "BoundaryOffsetAudit",
    "BoundaryOffsetDisposition",
    "BoundaryOptionProvider",
    "BoundaryOptionSet",
    "BoundaryTransformState",
    "JOINT_BOUNDARY_FRAME_OFFSETS",
    "JOINT_SHARED_BRIDGE_CACHE_MAXIMUM_ENTRIES",
    "JointTimelineResult",
    "JointTimelineProgress",
    "JointTimelineProgressCallback",
    "JointTimelineSharedBridgeEvidenceCache",
    "MINIMUM_JOINT_SEGMENT_DURATION_FRAMES",
    "SegmentBridgeProof",
    "SegmentBridgeProvider",
    "select_joint_timeline",
    "select_joint_timeline_candidates",
]
