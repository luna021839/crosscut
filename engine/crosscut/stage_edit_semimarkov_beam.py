"""Fail-closed semi-Markov beam decoding for learned StageEdit heads.

The transformer predicts evidence at candidate master times; it does not by
itself produce a legal edit.  This module performs a causal, variable-duration
search over those predictions while keeping every renderer and editorial hard
constraint explicit.  It is deliberately not imported by the production
pipeline yet.  A caller must provide a checkpoint directory that passes the
production readiness contract before any decoding work is attempted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ive import AspectRatio
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    minimum_roll_safe_zoom,
)
from .section_director import SectionPlan, SectionRole
from .source_derivative_diversity import SourceDerivativeFamilyMetadata
from .stage_edit import StageBoundary, StageEdit, StageShot
from .stage_edit_dynamic_crop_semimarkov_adapter import (
    DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT,
    DynamicCropHeadPredictionV2,
    DynamicCropTrackDecodeV2,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_DECISION_HISTORY_OWNER,
    StageEditTokenSchema,
    stage_edit_readiness_report,
)
from .stage_edit_transition_authority import (
    LEARNED_TRANSITION_AUTHORITY,
    RENDERER_OWNED_ALLOWED_MODEL_LABELS,
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)


DECODER_SCHEMA_VERSION = 8
_TIME_EPSILON = 1e-6
_SHOT_SIZES = frozenset(
    {"unknown", "wide", "full", "medium", "closeup", "extreme_closeup"}
)
_WIDE_SHOT_SIZES = frozenset({"wide", "full"})
_CLOSEUP_SHOT_SIZES = frozenset({"closeup", "extreme_closeup"})
_NONWIDE_SHOT_SIZES = frozenset({"medium", "closeup", "extreme_closeup"})
MONTAGE_GRAMMAR_CONTRACT = (
    "ive_human_reference_wide_transition_floor_plus_"
    "exact_section_and_phrase_wide_anchor_plus_formation_evidenced_"
    "highlight_burst_and_reset_v4"
)


class StageEditSemiMarkovError(RuntimeError):
    """Base class for decoder failures that must not silently fall back."""


class StageEditCheckpointNotReadyError(StageEditSemiMarkovError):
    """Raised before search when learned inference is not production-ready."""


class StageEditDecodeInputError(StageEditSemiMarkovError, ValueError):
    """Raised for malformed or unsafe model-head input."""


class StageEditNoValidTimelineError(StageEditSemiMarkovError):
    """Raised when hard constraints eliminate every complete timeline."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StageEditDecodeInputError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise StageEditDecodeInputError(f"{name} must be finite")
    return number


def _strict_bool_tuple(values: Sequence[bool], name: str) -> tuple[bool, ...]:
    result = tuple(values)
    if any(not isinstance(value, bool) for value in result):
        raise StageEditDecodeInputError(f"{name} must contain booleans")
    return result


def _float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    return tuple(_finite(value, f"{name}[{index}]") for index, value in enumerate(values))


def _normalized_shot_size(value: str, name: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "long": "wide",
        "long_shot": "wide",
        "full_shot": "full",
        "medium_shot": "medium",
        "close_up": "closeup",
        "close": "closeup",
        "extreme_close_up": "extreme_closeup",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _SHOT_SIZES:
        raise StageEditDecodeInputError(
            f"{name} must be one of {', '.join(sorted(_SHOT_SIZES))}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class StageEditSourceMetadata:
    """Static source-clock and editorial constraints for one source."""

    source_index: int
    source_id: str
    source_duration_seconds: float
    source_start_offset_seconds: float = 0.0
    internal_cut_times_seconds: tuple[float, ...] = ()
    maximum_occupancy_fraction: float = 0.60
    quality_log_prior: float = 0.0
    default_shot_size: str = "unknown"

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise StageEditDecodeInputError("source_index must be non-negative")
        source_id = str(self.source_id).strip()
        if not source_id:
            raise StageEditDecodeInputError("source_id cannot be blank")
        object.__setattr__(self, "source_id", source_id)
        duration = _finite(self.source_duration_seconds, "source_duration_seconds")
        if duration <= 0.0:
            raise StageEditDecodeInputError("source_duration_seconds must be positive")
        object.__setattr__(self, "source_duration_seconds", duration)
        object.__setattr__(
            self,
            "source_start_offset_seconds",
            _finite(self.source_start_offset_seconds, "source_start_offset_seconds"),
        )
        cuts = tuple(
            sorted(
                _finite(value, f"internal_cut_times_seconds[{index}]")
                for index, value in enumerate(self.internal_cut_times_seconds)
            )
        )
        if len(set(cuts)) != len(cuts):
            raise StageEditDecodeInputError("internal source cuts must be unique")
        if any(value < 0.0 or value > duration for value in cuts):
            raise StageEditDecodeInputError("internal source cut is outside the source")
        object.__setattr__(self, "internal_cut_times_seconds", cuts)
        occupancy = _finite(
            self.maximum_occupancy_fraction, "maximum_occupancy_fraction"
        )
        if not 0.0 < occupancy <= 1.0:
            raise StageEditDecodeInputError(
                "maximum_occupancy_fraction must lie in (0, 1]"
            )
        object.__setattr__(self, "maximum_occupancy_fraction", occupancy)
        object.__setattr__(
            self,
            "quality_log_prior",
            _finite(self.quality_log_prior, "quality_log_prior"),
        )
        object.__setattr__(
            self,
            "default_shot_size",
            _normalized_shot_size(self.default_shot_size, "default_shot_size"),
        )

    def source_start_at(self, master_seconds: float) -> float:
        return float(master_seconds) + self.source_start_offset_seconds


@dataclass(frozen=True, slots=True)
class StageEditSourceContinuity:
    """Same-canonical-time geometry and look evidence for one source.

    These values are label-free source observations already present in the
    Transformer token.  Keeping them on the head frame lets each semi-Markov
    beam score the *actual* outgoing/incoming pair at a proposed cut without
    leaking human/reference edit history into the global Transformer forward.
    Missing observations remain ``None`` and receive only a small uncertainty
    penalty; they can never masquerade as a detected perfect match.
    """

    face_scale: float | None = None
    face_center_x: float | None = None
    face_center_y: float | None = None
    body_center_x: float | None = None
    body_center_y: float | None = None
    pose_motion: float | None = None
    camera_motion_x: float | None = None
    camera_motion_y: float | None = None
    camera_zoom_velocity: float | None = None
    camera_roll_normalized: float | None = None
    sharpness: float | None = None
    exposure: float | None = None
    color_temperature_normalized: float | None = None
    detected_person_count_normalized: float | None = None
    formation_person_count_normalized: float | None = None
    formation_pose_coverage: float | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, _finite(value, name))
        for name in (
            "detected_person_count_normalized",
            "formation_person_count_normalized",
            "formation_pose_coverage",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise StageEditDecodeInputError(f"{name} must lie in [0, 1]")

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StageEditHeadFrame:
    """All StageEdit head outputs and hard masks at one candidate time.

    ``hard_source_gate_mask`` follows the transformer convention: ``True``
    means the source passed the hard gate.  It is combined with
    ``available_source_mask`` by the decoder and is never treated as a soft
    score.
    """

    master_time_seconds: float
    cut_logit: float
    source_logits: tuple[float, ...]
    duration_logits: tuple[float, ...]
    transition_logits: tuple[float, ...]
    crop: tuple[float, float, float, float]
    available_source_mask: tuple[bool, ...]
    hard_source_gate_mask: tuple[bool, ...]
    source_shot_sizes: tuple[str, ...] = ()
    source_continuity: tuple[StageEditSourceContinuity, ...] = ()
    candidate_id: str = ""
    dynamic_crop_v2: DynamicCropHeadPredictionV2 | None = None
    # These are copied from the exact serialized inference token.  They are
    # deliberately not reconstructed from wall-clock time in the decoder.
    section_progress: float | None = None
    phrase_progress: float | None = None
    section_role: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "master_time_seconds",
            _finite(self.master_time_seconds, "master_time_seconds"),
        )
        object.__setattr__(self, "cut_logit", _finite(self.cut_logit, "cut_logit"))
        object.__setattr__(
            self, "source_logits", _float_tuple(self.source_logits, "source_logits")
        )
        object.__setattr__(
            self,
            "duration_logits",
            _float_tuple(self.duration_logits, "duration_logits"),
        )
        object.__setattr__(
            self,
            "transition_logits",
            _float_tuple(self.transition_logits, "transition_logits"),
        )
        crop = _float_tuple(self.crop, "crop")
        if len(crop) != 4:
            raise StageEditDecodeInputError("crop must have zoom/x/y/roll values")
        object.__setattr__(self, "crop", crop)
        object.__setattr__(
            self,
            "available_source_mask",
            _strict_bool_tuple(
                self.available_source_mask, "available_source_mask"
            ),
        )
        object.__setattr__(
            self,
            "hard_source_gate_mask",
            _strict_bool_tuple(
                self.hard_source_gate_mask, "hard_source_gate_mask"
            ),
        )
        sizes = tuple(
            _normalized_shot_size(value, f"source_shot_sizes[{index}]")
            for index, value in enumerate(self.source_shot_sizes)
        )
        object.__setattr__(self, "source_shot_sizes", sizes)
        continuity = tuple(self.source_continuity)
        if any(not isinstance(value, StageEditSourceContinuity) for value in continuity):
            raise StageEditDecodeInputError(
                "source_continuity must contain StageEditSourceContinuity values"
            )
        object.__setattr__(self, "source_continuity", continuity)
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            candidate_id = f"candidate-{self.master_time_seconds:.6f}"
        object.__setattr__(self, "candidate_id", candidate_id)
        if self.dynamic_crop_v2 is not None and not isinstance(
            self.dynamic_crop_v2, DynamicCropHeadPredictionV2
        ):
            raise StageEditDecodeInputError(
                "dynamic_crop_v2 must be a DynamicCropHeadPredictionV2"
            )
        for name in ("section_progress", "phrase_progress"):
            value = getattr(self, name)
            if value is None:
                continue
            number = _finite(value, name)
            if not 0.0 <= number <= 1.0:
                raise StageEditDecodeInputError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, number)
        if self.section_role is not None:
            role = str(self.section_role).strip()
            allowed_roles = {item.value for item in SectionRole}
            if role not in allowed_roles:
                raise StageEditDecodeInputError("section_role is outside SectionRole")
            object.__setattr__(self, "section_role", role)

    @property
    def effective_source_mask(self) -> tuple[bool, ...]:
        return tuple(
            available and gate
            for available, gate in zip(
                self.available_source_mask,
                self.hard_source_gate_mask,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "master_time_seconds": self.master_time_seconds,
            "cut_logit": self.cut_logit,
            "source_logits": list(self.source_logits),
            "duration_logits": list(self.duration_logits),
            "transition_logits": list(self.transition_logits),
            "crop": list(self.crop),
            "available_source_mask": list(self.available_source_mask),
            "hard_source_gate_mask": list(self.hard_source_gate_mask),
            "source_shot_sizes": list(self.source_shot_sizes),
            "source_continuity": [
                value.to_dict() for value in self.source_continuity
            ],
            "candidate_id": self.candidate_id,
            "dynamic_crop_v2": (
                self.dynamic_crop_v2.to_dict()
                if self.dynamic_crop_v2 is not None
                else None
            ),
            "section_progress": self.section_progress,
            "phrase_progress": self.phrase_progress,
            "section_role": self.section_role,
        }


@dataclass(frozen=True, slots=True)
class StageEditSectionConstraint:
    """Optional hard overrides for one acoustic section."""

    section_index: int
    minimum_shot_seconds: float | None = None
    maximum_shot_seconds: float | None = None
    maximum_switches_per_minute: float | None = None
    allowed_transition_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.section_index, bool)
            or not isinstance(self.section_index, int)
            or self.section_index < 0
        ):
            raise StageEditDecodeInputError("section_index must be non-negative")
        for name in ("minimum_shot_seconds", "maximum_shot_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            number = _finite(value, name)
            if number <= 0.0:
                raise StageEditDecodeInputError(f"{name} must be positive")
            object.__setattr__(self, name, number)
        if (
            self.minimum_shot_seconds is not None
            and self.maximum_shot_seconds is not None
            and self.maximum_shot_seconds < self.minimum_shot_seconds
        ):
            raise StageEditDecodeInputError("section shot duration bounds are inverted")
        if self.maximum_switches_per_minute is not None:
            rate = _finite(
                self.maximum_switches_per_minute,
                "maximum_switches_per_minute",
            )
            if rate <= 0.0:
                raise StageEditDecodeInputError(
                    "maximum_switches_per_minute must be positive"
                )
            object.__setattr__(self, "maximum_switches_per_minute", rate)
        labels = tuple(str(value).strip() for value in self.allowed_transition_labels)
        if any(not value for value in labels) or len(set(labels)) != len(labels):
            raise StageEditDecodeInputError(
                "allowed transition labels must be non-empty and unique"
            )
        object.__setattr__(self, "allowed_transition_labels", labels)


@dataclass(frozen=True, slots=True)
class StageEditSemiMarkovConfig:
    """Bounded search and editorial policy for the causal decoder."""

    beam_width: int = 16
    timeline_count: int = 8
    minimum_shot_seconds: float = 0.50
    maximum_shot_seconds: float = 3.00
    minimum_unique_sources: int = 3
    maximum_source_occupancy_fraction: float = 0.60
    enforce_aba_exclusion: bool = True
    allow_consecutive_same_source: bool = False
    recent_source_window: int = 4
    recent_shot_size_window: int = 4
    maximum_same_shot_size_run: int = 3
    maximum_global_switches_per_minute: float = 30.0
    section_cut_density_multiplier: float = 1.35
    cut_density_window_seconds: float = 30.0
    internal_cut_exclusion_margin_seconds: float = 0.0
    cut_score_weight: float = 1.0
    source_score_weight: float = 1.0
    duration_score_weight: float = 0.65
    transition_score_weight: float = 0.30
    transition_authority: str = LEARNED_TRANSITION_AUTHORITY
    crop_correction_weight: float = 0.20
    source_quality_weight: float = 0.20
    boundary_continuity_weight: float = 0.35
    cadence_score_weight: float = 0.15
    diversity_score_weight: float = 0.12
    occupancy_balance_weight: float = 0.08
    montage_grammar_score_weight: float = 0.24
    enforce_montage_grammar: bool = True
    highlight_minimum_duration_seconds: float = 5.0
    highlight_minimum_known_shot_fraction: float = 0.80
    # Exact chorus-like P10 from the immutable four-song / twenty-reference
    # IVE professional-montage profile.  The non-wide/burst/reset requirements
    # below are deliberately labelled as product editorial requirements rather
    # than being misrepresented as lower-quantile reference floors.
    highlight_minimum_closeup_fraction: float = 0.15096153846153848
    highlight_minimum_nonwide_fraction: float = 0.35
    highlight_minimum_wide_involved_source_switch_fraction: float = (
        0.37333333333333335
    )
    highlight_minimum_early_wide_anchor_fraction: float = 1.0 / 3.0
    highlight_minimum_phrase_entry_wide_anchor_fraction: float = (
        0.2785714285714286
    )
    highlight_burst_maximum_shot_seconds: float = 1.50
    highlight_minimum_nonwide_burst_boundaries: int = 1
    formation_anchor_decision_window: int = 3
    # The reference profile only proves identity-free observed-person evidence;
    # it explicitly cannot establish a six-member formation threshold.  One
    # observed IVE-roster unit plus non-zero pose coverage distinguishes a
    # formation-evidenced wide from a generic/unknown wide without claiming
    # that it proves the complete group is visible.
    minimum_formation_anchor_person_count_normalized: float = 1.0 / 6.0
    highlight_minimum_post_burst_formation_resets: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enforce_montage_grammar, bool):
            raise StageEditDecodeInputError(
                "enforce_montage_grammar must be boolean"
            )
        for name in ("beam_width", "timeline_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 8 <= value <= 32:
                raise StageEditDecodeInputError(f"{name} must be an integer in 8...32")
        if self.timeline_count > self.beam_width:
            raise StageEditDecodeInputError("timeline_count cannot exceed beam_width")
        for name in (
            "recent_source_window",
            "recent_shot_size_window",
            "maximum_same_shot_size_run",
            "minimum_unique_sources",
            "highlight_minimum_nonwide_burst_boundaries",
            "formation_anchor_decision_window",
            "highlight_minimum_post_burst_formation_resets",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise StageEditDecodeInputError(f"{name} must be a positive integer")
        if self.enforce_aba_exclusion and self.recent_source_window < 2:
            raise StageEditDecodeInputError(
                "recent_source_window must be at least two for A-B-A exclusion"
            )
        if self.recent_shot_size_window < self.maximum_same_shot_size_run:
            raise StageEditDecodeInputError(
                "recent_shot_size_window must cover maximum_same_shot_size_run"
            )
        for name in (
            "minimum_shot_seconds",
            "maximum_shot_seconds",
            "maximum_source_occupancy_fraction",
            "maximum_global_switches_per_minute",
            "section_cut_density_multiplier",
            "cut_density_window_seconds",
            "highlight_minimum_duration_seconds",
            "highlight_burst_maximum_shot_seconds",
        ):
            number = _finite(getattr(self, name), name)
            if number <= 0.0:
                raise StageEditDecodeInputError(f"{name} must be positive")
            object.__setattr__(self, name, number)
        authority = StageEditTransitionAuthority(mode=self.transition_authority)
        if authority.renderer_owned and self.transition_score_weight != 0.0:
            raise StageEditDecodeInputError(
                "renderer-owned transition authority requires transition_score_weight=0"
            )
        if self.maximum_shot_seconds < self.minimum_shot_seconds:
            raise StageEditDecodeInputError("global shot duration bounds are inverted")
        if self.maximum_source_occupancy_fraction > 1.0:
            raise StageEditDecodeInputError(
                "maximum_source_occupancy_fraction cannot exceed one"
            )
        margin = _finite(
            self.internal_cut_exclusion_margin_seconds,
            "internal_cut_exclusion_margin_seconds",
        )
        if margin < 0.0:
            raise StageEditDecodeInputError(
                "internal_cut_exclusion_margin_seconds cannot be negative"
            )
        object.__setattr__(self, "internal_cut_exclusion_margin_seconds", margin)
        for name in (
            "cut_score_weight",
            "source_score_weight",
            "duration_score_weight",
            "transition_score_weight",
            "crop_correction_weight",
            "source_quality_weight",
            "boundary_continuity_weight",
            "cadence_score_weight",
            "diversity_score_weight",
            "occupancy_balance_weight",
            "montage_grammar_score_weight",
        ):
            number = _finite(getattr(self, name), name)
            if number < 0.0:
                raise StageEditDecodeInputError(f"{name} cannot be negative")
            object.__setattr__(self, name, number)
        for name in (
            "highlight_minimum_known_shot_fraction",
            "highlight_minimum_closeup_fraction",
            "highlight_minimum_nonwide_fraction",
            "highlight_minimum_wide_involved_source_switch_fraction",
            "highlight_minimum_early_wide_anchor_fraction",
            "highlight_minimum_phrase_entry_wide_anchor_fraction",
            "minimum_formation_anchor_person_count_normalized",
        ):
            number = _finite(getattr(self, name), name)
            if not 0.0 <= number <= 1.0:
                raise StageEditDecodeInputError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, number)
        if (
            self.highlight_minimum_closeup_fraction
            > self.highlight_minimum_nonwide_fraction
        ):
            raise StageEditDecodeInputError(
                "highlight close-up floor cannot exceed the non-wide floor"
            )


@dataclass(frozen=True, slots=True)
class StageEditScoreComponent:
    name: str
    raw_value: float
    weight: float
    contribution: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("score component name cannot be blank")
        for value in (self.raw_value, self.weight, self.contribution):
            if not math.isfinite(value):
                raise ValueError("score components must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StageEditBeamDecision:
    shot_index: int
    start_candidate_index: int
    end_candidate_index: int
    master_start_seconds: float
    master_end_seconds: float
    source_index: int
    source_id: str
    shot_size: str
    detected_person_count_normalized: float | None
    formation_person_count_normalized: float | None
    formation_pose_coverage: float | None
    duration_bucket: int
    transition_label: str
    crop_requested: tuple[float, float, float, float]
    crop_applied: tuple[float, float, float, float]
    crop_was_corrected: bool
    decision_confidence: float
    score_components: tuple[StageEditScoreComponent, ...]
    recent_sources_after: tuple[int, ...]
    recent_shot_sizes_after: tuple[str, ...]
    recent_shot_durations_after: tuple[float, ...]
    recent_cut_density_per_minute_after: float
    source_occupancy_after_seconds: tuple[float, ...]
    section_switch_counts_after: tuple[int, ...]
    derivative_family_id: int = 0
    derivative_family_occupancy_after_seconds: tuple[float, ...] = ()
    editorial_transform: EditorialTransformTrack | None = None
    dynamic_crop_decode: DynamicCropTrackDecodeV2 | None = None
    crop_contract: str = "legacy_static_crop_head_v1"
    section_progress: float | None = None
    phrase_progress: float | None = None
    section_role: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.master_end_seconds - self.master_start_seconds

    @property
    def score(self) -> float:
        return sum(item.contribution for item in self.score_components)

    def signature(self) -> tuple[Any, ...]:
        transform_signature = (
            ()
            if self.editorial_transform is None
            else tuple(
                (
                    round(item.time_seconds, 9),
                    round(item.zoom, 9),
                    round(item.center_x, 9),
                    round(item.center_y, 9),
                    round(item.roll_degrees, 9),
                )
                for item in self.editorial_transform.keyframes
            )
        )
        return (
            self.end_candidate_index,
            # Alternate files from one verified camera family must not fill
            # the retained candidate set as supposedly distinct edits.
            self.derivative_family_id,
            self.duration_bucket,
            self.transition_label,
            tuple(round(value, 9) for value in self.crop_applied),
            self.crop_contract,
            transform_signature,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_index": self.shot_index,
            "start_candidate_index": self.start_candidate_index,
            "end_candidate_index": self.end_candidate_index,
            "master_start_seconds": self.master_start_seconds,
            "master_end_seconds": self.master_end_seconds,
            "duration_seconds": self.duration_seconds,
            "source_index": self.source_index,
            "source_id": self.source_id,
            "derivative_family_id": self.derivative_family_id,
            "shot_size": self.shot_size,
            "detected_person_count_normalized": (
                self.detected_person_count_normalized
            ),
            "formation_person_count_normalized": (
                self.formation_person_count_normalized
            ),
            "formation_pose_coverage": self.formation_pose_coverage,
            "section_progress": self.section_progress,
            "phrase_progress": self.phrase_progress,
            "section_role": self.section_role,
            "duration_bucket": self.duration_bucket,
            "transition_label": self.transition_label,
            "render_transition_kind": "hard_cut",
            "crop_requested": list(self.crop_requested),
            "crop_applied": list(self.crop_applied),
            "crop_was_corrected": self.crop_was_corrected,
            "crop_contract": self.crop_contract,
            "editorial_transform": (
                None
                if self.editorial_transform is None
                else {
                    "reason": self.editorial_transform.reason,
                    "keyframes": [
                        {
                            "time_seconds": item.time_seconds,
                            "zoom": item.zoom,
                            "center_x": item.center_x,
                            "center_y": item.center_y,
                            "roll_degrees": item.roll_degrees,
                        }
                        for item in self.editorial_transform.keyframes
                    ],
                }
            ),
            "dynamic_crop_v2": (
                None
                if self.dynamic_crop_decode is None
                else {
                    "adapter_contract": self.dynamic_crop_decode.adapter_contract,
                    "eligible": self.dynamic_crop_decode.eligible,
                    "loss_masked": self.dynamic_crop_decode.loss_masked,
                    "requested_keyframe_count": (
                        self.dynamic_crop_decode.requested_keyframe_count
                    ),
                    "applied_keyframe_count": (
                        self.dynamic_crop_decode.applied_keyframe_count
                    ),
                    "minimum_safe_zoom_projection_count": (
                        self.dynamic_crop_decode.minimum_safe_zoom_projection_count
                    ),
                    "rejection_reasons": list(
                        self.dynamic_crop_decode.rejection_reasons
                    ),
                }
            ),
            "decision_confidence": self.decision_confidence,
            "score": self.score,
            "score_components": [item.to_dict() for item in self.score_components],
            "state_after": {
                "current_time_seconds": self.master_end_seconds,
                "current_source": self.source_index,
                "recent_sources": list(self.recent_sources_after),
                "recent_shot_sizes": list(self.recent_shot_sizes_after),
                "recent_shot_durations": list(self.recent_shot_durations_after),
                "recent_cut_density_per_minute": (
                    self.recent_cut_density_per_minute_after
                ),
                "source_occupancy_seconds": list(
                    self.source_occupancy_after_seconds
                ),
                "derivative_family_occupancy_seconds": list(
                    self.derivative_family_occupancy_after_seconds
                ),
                "section_switch_counts": list(self.section_switch_counts_after),
            },
        }


@dataclass(frozen=True, slots=True)
class StageEditBeamTimeline:
    rank: int
    score: float
    score_components: Mapping[str, float]
    decisions: tuple[StageEditBeamDecision, ...]
    edit: StageEdit
    source_occupancy_seconds: tuple[float, ...]
    recent_cut_density_per_minute: float
    montage_grammar: Mapping[str, Any] = field(default_factory=dict)
    derivative_family_occupancy_seconds: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "score_components": dict(self.score_components),
            "source_occupancy_seconds": list(self.source_occupancy_seconds),
            "derivative_family_occupancy_seconds": list(
                self.derivative_family_occupancy_seconds
            ),
            "recent_cut_density_per_minute": self.recent_cut_density_per_minute,
            "montage_grammar": dict(self.montage_grammar),
            "decisions": [item.to_dict() for item in self.decisions],
            "stage_edit": self.edit.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StageEditDecodeAudit:
    schema_version: int
    decoder: str
    input_sha256: str
    readiness: Mapping[str, Any]
    config: Mapping[str, Any]
    input_summary: Mapping[str, Any]
    endpoint_audit: tuple[Mapping[str, Any], ...]
    constraint_rejection_counts: Mapping[str, int]
    transition_hypothesis_count: int
    accepted_hypothesis_count: int
    beam_pruned_state_count: int
    complete_state_count: int
    duplicate_completion_count: int
    requested_timeline_count: int
    returned_timeline_count: int
    request_fulfilled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decoder": self.decoder,
            "input_sha256": self.input_sha256,
            "readiness": dict(self.readiness),
            "config": dict(self.config),
            "input_summary": dict(self.input_summary),
            "endpoint_audit": [dict(item) for item in self.endpoint_audit],
            "constraint_rejection_counts": dict(self.constraint_rejection_counts),
            "transition_hypothesis_count": self.transition_hypothesis_count,
            "accepted_hypothesis_count": self.accepted_hypothesis_count,
            "beam_pruned_state_count": self.beam_pruned_state_count,
            "complete_state_count": self.complete_state_count,
            "duplicate_completion_count": self.duplicate_completion_count,
            "requested_timeline_count": self.requested_timeline_count,
            "returned_timeline_count": self.returned_timeline_count,
            "request_fulfilled": self.request_fulfilled,
        }


@dataclass(frozen=True, slots=True)
class StageEditSemiMarkovResult:
    timelines: tuple[StageEditBeamTimeline, ...]
    audit: StageEditDecodeAudit

    @property
    def edits(self) -> tuple[StageEdit, ...]:
        return tuple(item.edit for item in self.timelines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timelines": [item.to_dict() for item in self.timelines],
            "audit": self.audit.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _ResolvedSectionConstraint:
    minimum_shot_seconds: float
    maximum_shot_seconds: float
    maximum_switches_per_minute: float
    allowed_transition_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BeamState:
    current_index: int
    current_source: int | None
    recent_sources: tuple[int, ...]
    recent_shot_sizes: tuple[str, ...]
    recent_shot_durations: tuple[float, ...]
    recent_cut_times: tuple[float, ...]
    source_occupancy_seconds: tuple[float, ...]
    section_switch_counts: tuple[int, ...]
    decisions: tuple[StageEditBeamDecision, ...]
    score: float
    component_totals: tuple[tuple[str, float], ...]

    def signature(self) -> tuple[Any, ...]:
        return tuple(item.signature() for item in self.decisions)


def _log_sigmoid(value: float) -> float:
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _log_softmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise StageEditDecodeInputError("softmax logits cannot be empty")
    maximum = max(values)
    total = sum(math.exp(value - maximum) for value in values)
    if not math.isfinite(total) or total <= 0.0:
        raise StageEditDecodeInputError("softmax normalization is not finite")
    normalizer = maximum + math.log(total)
    result = tuple(value - normalizer for value in values)
    if any(not math.isfinite(value) for value in result):
        raise StageEditDecodeInputError("softmax produced a non-finite score")
    return result


def _masked_log_softmax(
    values: Sequence[float],
    mask: Sequence[bool],
) -> tuple[float, ...]:
    if len(values) != len(mask):
        raise StageEditDecodeInputError("masked softmax values/mask length mismatch")
    selected_indices = tuple(index for index, allowed in enumerate(mask) if allowed)
    if not selected_indices:
        raise StageEditDecodeInputError("masked softmax has no allowed value")
    selected = _log_softmax(tuple(values[index] for index in selected_indices))
    # Rejected entries are deliberately finite because the decoder's
    # no-nonfinite contract covers internal audit state too.  They are never
    # read as candidate scores.
    result = [0.0] * len(values)
    for index, score in zip(selected_indices, selected, strict=True):
        result[index] = score
    return tuple(result)


def _component(name: str, raw: float, weight: float) -> StageEditScoreComponent:
    contribution = raw * weight
    if not math.isfinite(contribution):
        raise StageEditDecodeInputError(f"{name} score is not finite")
    return StageEditScoreComponent(name, raw, weight, contribution)


def _add_component_totals(
    current: tuple[tuple[str, float], ...],
    additions: Sequence[StageEditScoreComponent],
) -> tuple[tuple[str, float], ...]:
    totals = dict(current)
    for item in additions:
        totals[item.name] = totals.get(item.name, 0.0) + item.contribution
    if any(not math.isfinite(value) for value in totals.values()):
        raise StageEditDecodeInputError("cumulative beam score is not finite")
    return tuple(sorted(totals.items()))


def _state_sort_key(state: _BeamState) -> tuple[Any, ...]:
    return (-state.score, state.signature())


def _state_structure_signature(state: _BeamState) -> tuple[tuple[int, int], ...]:
    """The timing/source skeleton, excluding cosmetic transition variants."""

    return tuple(
        (item.end_candidate_index, item.derivative_family_id)
        for item in state.decisions
    )


def _prune_states(
    states: Sequence[_BeamState],
    width: int,
) -> tuple[list[_BeamState], int, int]:
    """Keep structural edit diversity before alternate transition labels.

    A plain top-k can spend the entire beam on the four transition labels for
    one timing/source skeleton.  That destroys the variable-duration part of
    the search and can make a later density constraint appear impossible.
    The first pass retains the best state for each structural skeleton; the
    second pass uses remaining capacity for genuine transition alternatives.
    """

    ordered = sorted(states, key=_state_sort_key)
    unique_exact: list[_BeamState] = []
    exact_signatures: set[tuple[Any, ...]] = set()
    duplicate_count = 0
    for state in ordered:
        signature = state.signature()
        if signature in exact_signatures:
            duplicate_count += 1
            continue
        exact_signatures.add(signature)
        unique_exact.append(state)

    kept: list[_BeamState] = []
    kept_exact: set[tuple[Any, ...]] = set()
    structures: set[tuple[tuple[int, int], ...]] = set()
    for state in unique_exact:
        structure = _state_structure_signature(state)
        if structure in structures:
            continue
        structures.add(structure)
        kept.append(state)
        kept_exact.add(state.signature())
        if len(kept) == width:
            break
    if len(kept) < width:
        for state in unique_exact:
            signature = state.signature()
            if signature in kept_exact:
                continue
            kept.append(state)
            kept_exact.add(signature)
            if len(kept) == width:
                break
    pruned_count = len(states) - len(kept)
    return kept, pruned_count, duplicate_count


def _crop_transform(
    raw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], bool, float]:
    """Project a regression-head value into the renderer's safety envelope."""

    raw_zoom, raw_x, raw_y, raw_roll = raw
    zoom = min(1.18, max(1.0, raw_zoom))
    center_x = min(0.54, max(0.46, raw_x))
    center_y = min(0.54, max(0.46, raw_y))
    dx = center_x - 0.5
    dy = center_y - 0.5
    radius = math.hypot(dx, dy)
    if radius > 0.038:
        scale = 0.038 / radius
        center_x = 0.5 + dx * scale
        center_y = 0.5 + dy * scale
    roll = min(1.60, max(-1.60, raw_roll))
    zoom = min(1.18, max(zoom, minimum_roll_safe_zoom(roll)))

    attempts = (
        (zoom, center_x, center_y, roll),
        (zoom, 0.5, 0.5, roll),
        (zoom, 0.5, 0.5, 0.0),
        (1.0, 0.5, 0.5, 0.0),
    )
    applied = attempts[-1]
    for candidate in attempts:
        try:
            EditorialTransformKeyframe(
                time_seconds=0.0,
                zoom=candidate[0],
                center_x=candidate[1],
                center_y=candidate[2],
                roll_degrees=candidate[3],
            )
        except ValueError:
            continue
        applied = candidate
        break

    correction = (
        abs(applied[0] - raw_zoom) / 0.18
        + abs(applied[1] - raw_x) / 0.08
        + abs(applied[2] - raw_y) / 0.08
        + abs(applied[3] - raw_roll) / 1.60
    ) / 4.0
    if not math.isfinite(correction):
        raise StageEditDecodeInputError("crop correction score is not finite")
    corrected = any(abs(left - right) > 1e-9 for left, right in zip(raw, applied))
    return applied, corrected, correction


def _editorial_track(
    values: tuple[float, float, float, float],
) -> EditorialTransformTrack | None:
    if all(
        abs(left - right) <= 1e-9
        for left, right in zip(values, (1.0, 0.5, 0.5, 0.0), strict=True)
    ):
        return None
    return EditorialTransformTrack(
        keyframes=(
            EditorialTransformKeyframe(
                time_seconds=0.0,
                zoom=values[0],
                center_x=values[1],
                center_y=values[2],
                roll_degrees=values[3],
            ),
        ),
        reason="stage_edit_semimarkov_crop_head",
    )


def _resolved_constraints(
    section_plan: SectionPlan,
    overrides: Sequence[StageEditSectionConstraint],
    schema: StageEditTokenSchema,
    config: StageEditSemiMarkovConfig,
) -> tuple[_ResolvedSectionConstraint, ...]:
    authority = StageEditTransitionAuthority(mode=config.transition_authority)
    if authority.renderer_owned and "hard_cut" not in schema.transition_labels:
        raise StageEditDecodeInputError(
            "renderer-owned transition authority requires hard_cut metadata"
        )
    by_index: dict[int, StageEditSectionConstraint] = {}
    for item in overrides:
        if item.section_index >= len(section_plan.sections):
            raise StageEditDecodeInputError("section constraint references unknown section")
        if item.section_index in by_index:
            raise StageEditDecodeInputError("section constraint indices must be unique")
        unknown = set(item.allowed_transition_labels) - set(schema.transition_labels)
        if unknown:
            raise StageEditDecodeInputError(
                f"section constraint has unknown transitions: {sorted(unknown)}"
            )
        if authority.renderer_owned and item.allowed_transition_labels not in {
            (),
            RENDERER_OWNED_ALLOWED_MODEL_LABELS,
        }:
            raise StageEditDecodeInputError(
                "renderer-owned section constraints allow only hard_cut metadata"
            )
        by_index[item.section_index] = item

    resolved: list[_ResolvedSectionConstraint] = []
    for directive in section_plan.sections:
        override = by_index.get(directive.section_index)
        minimum = (
            override.minimum_shot_seconds
            if override is not None and override.minimum_shot_seconds is not None
            else config.minimum_shot_seconds
        )
        maximum = (
            override.maximum_shot_seconds
            if override is not None and override.maximum_shot_seconds is not None
            else config.maximum_shot_seconds
        )
        if maximum < minimum:
            raise StageEditDecodeInputError(
                f"section {directive.section_index} shot bounds are inverted"
            )
        rate = (
            override.maximum_switches_per_minute
            if override is not None
            and override.maximum_switches_per_minute is not None
            else min(
                config.maximum_global_switches_per_minute,
                directive.target_source_switches_per_minute
                * config.section_cut_density_multiplier,
            )
        )
        labels = (
            RENDERER_OWNED_ALLOWED_MODEL_LABELS
            if authority.renderer_owned
            else override.allowed_transition_labels
            if override is not None and override.allowed_transition_labels
            else schema.transition_labels
        )
        resolved.append(
            _ResolvedSectionConstraint(minimum, maximum, rate, tuple(labels))
        )
    return tuple(resolved)


def _section_index_at_start(section_plan: SectionPlan, value: float) -> int:
    adjusted = min(section_plan.end_seconds, value + _TIME_EPSILON)
    return section_plan.directive_at(adjusted).section_index


def _maximum_switch_count(rate: float, duration: float) -> int:
    return max(1, int(math.ceil(rate * duration / 60.0 - 1e-12)))


def _shot_size(
    frame: StageEditHeadFrame,
    source: StageEditSourceMetadata,
) -> str:
    if frame.source_shot_sizes:
        return frame.source_shot_sizes[source.source_index]
    return source.default_shot_size


def _formation_wide_evidence(
    *,
    shot_size: str,
    formation_person_count_normalized: float | None,
    formation_pose_coverage: float | None,
    config: StageEditSemiMarkovConfig,
) -> bool:
    """Return only weak, identity-free evidence for a formation-aware wide.

    This must never be described as proof that all six IVE members are visible.
    The immutable human-reference profile only supports distinguishing missing
    or zero evidence from a detector/pose-backed wide observation.
    """

    return bool(
        shot_size in _WIDE_SHOT_SIZES
        and formation_person_count_normalized is not None
        and formation_person_count_normalized
        >= config.minimum_formation_anchor_person_count_normalized
        and formation_pose_coverage is not None
        and formation_pose_coverage > 0.0
    )


def _montage_grammar_score_raw(
    *,
    directive: Any,
    shot_size: str,
    start_seconds: float,
    duration_seconds: float,
    previous_shot_size: str | None,
    previous_duration_seconds: float | None,
    derivative_family_source_switched: bool,
    previous_formation_wide_evidenced: bool,
    formation_person_count_normalized: float | None,
    formation_pose_coverage: float | None,
    config: StageEditSemiMarkovConfig,
) -> float:
    """Bounded structural prior; learned logits still own the edit decision.

    A formation-wide is useful as a phrase anchor, but it must not occupy an
    entire high-energy passage.  After that anchor, medium/close shots and one
    short non-wide chain create the visual lift that a uniform sequence of
    wide-to-wide cuts cannot provide.
    """

    if shot_size == "unknown":
        return -0.12
    progress = min(
        1.0,
        max(
            0.0,
            (start_seconds - directive.start_seconds)
            / max(directive.duration_seconds, _TIME_EPSILON),
        ),
    )
    anchor_zone = progress <= 0.20
    raw = 0.0
    if directive.establish_with_wide_shot and anchor_zone:
        if shot_size in _WIDE_SHOT_SIZES:
            if _formation_wide_evidence(
                shot_size=shot_size,
                formation_person_count_normalized=(
                    formation_person_count_normalized
                ),
                formation_pose_coverage=formation_pose_coverage,
                config=config,
            ):
                raw += 0.48
            else:
                raw += 0.04
        elif shot_size in _CLOSEUP_SHOT_SIZES:
            raw -= 0.12

    if directive.role is SectionRole.CHORUS_LIKE:
        if (
            derivative_family_source_switched
            and previous_shot_size is not None
            and (
                previous_shot_size in _WIDE_SHOT_SIZES
                or shot_size in _WIDE_SHOT_SIZES
            )
        ):
            # Human references put a wide on at least one side of 80% of the
            # median chorus camera-family switches (P10 hard floor 37.33%).
            # Keep this a small bounded prior: learned source logits still
            # contribute far more than this term after its 0.24 weight.
            raw += 0.10
            if previous_formation_wide_evidenced or _formation_wide_evidence(
                shot_size=shot_size,
                formation_person_count_normalized=(
                    formation_person_count_normalized
                ),
                formation_pose_coverage=formation_pose_coverage,
                config=config,
            ):
                raw += 0.06
        if not anchor_zone:
            if shot_size in _CLOSEUP_SHOT_SIZES:
                raw += 0.48
            elif shot_size == "medium":
                raw += 0.24
            elif shot_size in _WIDE_SHOT_SIZES:
                raw -= 0.12
        if previous_shot_size is not None:
            if (
                previous_shot_size in _NONWIDE_SHOT_SIZES
                and shot_size in _NONWIDE_SHOT_SIZES
            ):
                raw += 0.22
                if (
                    previous_duration_seconds is not None
                    and previous_duration_seconds
                    <= config.highlight_burst_maximum_shot_seconds
                    and duration_seconds
                    <= config.highlight_burst_maximum_shot_seconds
                    and (
                        previous_shot_size in _CLOSEUP_SHOT_SIZES
                        or shot_size in _CLOSEUP_SHOT_SIZES
                    )
                ):
                    raw += 0.30
            elif (
                previous_shot_size in _NONWIDE_SHOT_SIZES
                and shot_size in _WIDE_SHOT_SIZES
            ):
                # A wide reset after a face-scale burst exposes the formation
                # without allowing the entire highlight to collapse to wides.
                raw += (
                    0.18
                    if _formation_wide_evidence(
                        shot_size=shot_size,
                        formation_person_count_normalized=(
                            formation_person_count_normalized
                        ),
                        formation_pose_coverage=formation_pose_coverage,
                        config=config,
                    )
                    else 0.02
                )
    elif directive.role is SectionRole.BUILD:
        if shot_size in _CLOSEUP_SHOT_SIZES:
            raw += 0.10 + 0.24 * progress
        elif shot_size == "medium":
            raw += 0.08 + 0.12 * progress
        elif shot_size in _WIDE_SHOT_SIZES:
            raw += 0.12 * (1.0 - progress)
    elif directive.role in {SectionRole.INTRO, SectionRole.OUTRO}:
        if shot_size in _WIDE_SHOT_SIZES:
            raw += 0.18
    else:
        # Verse/bridge passages retain breathing room; close-ups are legal and
        # learned, but the structural prior does not force a burst everywhere.
        if shot_size in {"wide", "full", "medium"}:
            raw += 0.06
    return min(1.0, max(-1.0, raw))


def _section_candidate_shot_support(
    *,
    frames: Sequence[StageEditHeadFrame],
    source_metadata: Sequence[StageEditSourceMetadata],
    section_plan: SectionPlan,
    derivative_family_ids: Sequence[int],
    config: StageEditSemiMarkovConfig,
) -> tuple[dict[str, Any], ...]:
    records: list[list[dict[str, Any]]] = [
        [] for _section in section_plan.sections
    ]
    for frame in frames:
        section_index = _section_index_at_start(
            section_plan, frame.master_time_seconds
        )
        known_sizes: list[tuple[int, str, bool]] = []
        for source_index, metadata in enumerate(source_metadata):
            if not frame.effective_source_mask[source_index]:
                continue
            size = _shot_size(frame, metadata)
            if size != "unknown":
                continuity = (
                    frame.source_continuity[source_index]
                    if frame.source_continuity
                    else StageEditSourceContinuity()
                )
                known_sizes.append(
                    (
                        derivative_family_ids[source_index],
                        size,
                        _formation_wide_evidence(
                            shot_size=size,
                            formation_person_count_normalized=(
                                continuity.formation_person_count_normalized
                            ),
                            formation_pose_coverage=(
                                continuity.formation_pose_coverage
                            ),
                            config=config,
                        ),
                    )
                )
        nonwide_families = {
            family
            for family, size, _formation in known_sizes
            if size in _NONWIDE_SHOT_SIZES
        }
        records[section_index].append(
            {
                "known": bool(known_sizes),
                "wide": any(
                    size in _WIDE_SHOT_SIZES
                    for _family, size, _formation in known_sizes
                ),
                "formation_wide": any(
                    formation for _family, _size, formation in known_sizes
                ),
                "nonwide": any(
                    size in _NONWIDE_SHOT_SIZES
                    for _family, size, _formation in known_sizes
                ),
                "closeup": any(
                    size in _CLOSEUP_SHOT_SIZES
                    for _family, size, _formation in known_sizes
                ),
                "multi_family_nonwide": len(nonwide_families) >= 2,
            }
        )

    result: list[dict[str, Any]] = []
    for directive, section_records in zip(
        section_plan.sections, records, strict=True
    ):
        count = len(section_records)
        known_count = sum(bool(item["known"]) for item in section_records)
        closeup_count = sum(bool(item["closeup"]) for item in section_records)
        nonwide_count = sum(bool(item["nonwide"]) for item in section_records)
        wide_count = sum(bool(item["wide"]) for item in section_records)
        multi_count = sum(
            bool(item["multi_family_nonwide"]) for item in section_records
        )
        early = section_records[: config.formation_anchor_decision_window]
        result.append(
            {
                "section_index": directive.section_index,
                "candidate_frame_count": count,
                "known_shot_size_frame_count": known_count,
                "known_shot_size_frame_fraction": (
                    known_count / count if count else 0.0
                ),
                "wide_supported_frame_count": wide_count,
                "wide_supported_frame_fraction": (
                    wide_count / count if count else 0.0
                ),
                "nonwide_supported_frame_count": nonwide_count,
                "nonwide_supported_frame_fraction": (
                    nonwide_count / count if count else 0.0
                ),
                "closeup_supported_frame_count": closeup_count,
                "closeup_supported_frame_fraction": (
                    closeup_count / count if count else 0.0
                ),
                "multi_family_nonwide_supported_frame_count": multi_count,
                "early_wide_supported": any(
                    bool(item["wide"]) for item in early
                ),
                "early_formation_wide_supported": any(
                    bool(item["formation_wide"]) for item in early
                ),
            }
        )
    return tuple(result)


def _exact_chorus_anchor_metrics(
    *,
    decisions: Sequence[StageEditBeamDecision],
    section_plan: SectionPlan,
) -> dict[str, Any]:
    """Mirror the immutable reference profile's exact token-reset semantics."""

    missing_section_progress = sum(
        item.section_progress is None for item in decisions
    )
    missing_phrase_progress = sum(item.phrase_progress is None for item in decisions)
    missing_section_role = sum(item.section_role is None for item in decisions)
    section_role_plan_mismatch = sum(
        item.section_role is not None
        and item.section_role
        != section_plan.directive_at(item.master_start_seconds).role.value
        for item in decisions
    )

    exact_sections: list[list[StageEditBeamDecision]] = []
    if missing_section_progress == 0 and missing_section_role == 0:
        for decision in decisions:
            if not exact_sections:
                exact_sections.append([decision])
                continue
            previous = exact_sections[-1][-1]
            assert previous.section_progress is not None
            assert decision.section_progress is not None
            if (
                decision.section_role != previous.section_role
                or decision.section_progress + _TIME_EPSILON
                < previous.section_progress
            ):
                exact_sections.append([decision])
            else:
                exact_sections[-1].append(decision)

    observed_early = 0
    early_wide = 0
    for selected in exact_sections:
        if selected[0].section_role != SectionRole.CHORUS_LIKE.value:
            continue
        # Match the profile exactly: take the first three labelled shot
        # positions, then exclude unknown shot-size evidence.  An unknown
        # decision therefore still consumes an early-anchor slot.
        early_known = tuple(
            item for item in selected[:3] if item.shot_size != "unknown"
        )
        if early_known:
            observed_early += 1
            if any(item.shot_size in _WIDE_SHOT_SIZES for item in early_known):
                early_wide += 1

    phrases: list[list[StageEditBeamDecision]] = []
    if missing_phrase_progress == 0:
        for decision in decisions:
            if not phrases:
                phrases.append([decision])
                continue
            previous = phrases[-1][-1]
            assert previous.phrase_progress is not None
            assert decision.phrase_progress is not None
            if (
                decision.phrase_progress + _TIME_EPSILON
                < previous.phrase_progress
            ):
                phrases.append([decision])
            else:
                phrases[-1].append(decision)

    observed_phrase_entries = 0
    phrase_entry_wide = 0
    for phrase in phrases:
        entry = phrase[0]
        if entry.section_role != SectionRole.CHORUS_LIKE.value:
            continue
        if entry.shot_size == "unknown":
            continue
        observed_phrase_entries += 1
        if entry.shot_size in _WIDE_SHOT_SIZES:
            phrase_entry_wide += 1

    return {
        "evidence_contract": (
            "serialized_inference_token_section_and_phrase_progress_reset_v1"
        ),
        "missing_section_progress_decision_count": missing_section_progress,
        "missing_section_role_decision_count": missing_section_role,
        "section_role_plan_mismatch_count": section_role_plan_mismatch,
        "missing_phrase_progress_decision_count": missing_phrase_progress,
        "observed_chorus_section_early_anchor_count": observed_early,
        "chorus_section_early_wide_anchor_count": early_wide,
        "chorus_section_early_wide_anchor_fraction": (
            early_wide / observed_early if observed_early else None
        ),
        "observed_chorus_phrase_entry_anchor_count": observed_phrase_entries,
        "chorus_phrase_entry_wide_anchor_count": phrase_entry_wide,
        "chorus_phrase_entry_wide_anchor_fraction": (
            phrase_entry_wide / observed_phrase_entries
            if observed_phrase_entries
            else None
        ),
    }


def _evaluate_montage_grammar(
    *,
    decisions: Sequence[StageEditBeamDecision],
    section_plan: SectionPlan,
    support_by_section: Sequence[Mapping[str, Any]],
    config: StageEditSemiMarkovConfig,
) -> dict[str, Any]:
    exact_anchor_metrics = _exact_chorus_anchor_metrics(
        decisions=decisions,
        section_plan=section_plan,
    )
    by_section: list[list[StageEditBeamDecision]] = [
        [] for _section in section_plan.sections
    ]
    for decision in decisions:
        section_index = _section_index_at_start(
            section_plan, decision.master_start_seconds
        )
        by_section[section_index].append(decision)

    all_failures: list[str] = []
    section_metrics: list[dict[str, Any]] = []
    for directive, selected, support in zip(
        section_plan.sections,
        by_section,
        support_by_section,
        strict=True,
    ):
        known = [item for item in selected if item.shot_size != "unknown"]
        closeups = [
            item for item in known if item.shot_size in _CLOSEUP_SHOT_SIZES
        ]
        nonwides = [
            item for item in known if item.shot_size in _NONWIDE_SHOT_SIZES
        ]
        wides = [item for item in known if item.shot_size in _WIDE_SHOT_SIZES]
        selected_duration = sum(item.duration_seconds for item in selected)
        known_duration = sum(item.duration_seconds for item in known)
        closeup_duration = sum(item.duration_seconds for item in closeups)
        nonwide_duration = sum(item.duration_seconds for item in nonwides)
        wide_duration = sum(item.duration_seconds for item in wides)
        known_fraction = (
            known_duration / selected_duration if selected_duration else 0.0
        )
        closeup_fraction = closeup_duration / known_duration if known_duration else 0.0
        nonwide_fraction = nonwide_duration / known_duration if known_duration else 0.0
        wide_fraction = wide_duration / known_duration if known_duration else 0.0
        burst_boundaries = 0
        burst_right_positions: list[int] = []
        closeup_chain_boundaries = 0
        source_switch_boundaries = 0
        wide_involved_source_switch_boundaries = 0
        for position, (left, right) in enumerate(zip(selected, selected[1:])):
            if left.derivative_family_id == right.derivative_family_id:
                continue
            source_switch_boundaries += 1
            if (
                left.shot_size in _WIDE_SHOT_SIZES
                or right.shot_size in _WIDE_SHOT_SIZES
            ):
                wide_involved_source_switch_boundaries += 1
            if (
                left.shot_size in _CLOSEUP_SHOT_SIZES
                and right.shot_size in _CLOSEUP_SHOT_SIZES
            ):
                closeup_chain_boundaries += 1
            if (
                left.shot_size in _NONWIDE_SHOT_SIZES
                and right.shot_size in _NONWIDE_SHOT_SIZES
                and (
                    left.shot_size in _CLOSEUP_SHOT_SIZES
                    or right.shot_size in _CLOSEUP_SHOT_SIZES
                )
                and left.duration_seconds
                <= config.highlight_burst_maximum_shot_seconds
                and right.duration_seconds
                <= config.highlight_burst_maximum_shot_seconds
            ):
                burst_boundaries += 1
                burst_right_positions.append(position + 1)
        wide_involved_source_switch_fraction = (
            wide_involved_source_switch_boundaries / source_switch_boundaries
            if source_switch_boundaries
            else 0.0
        )
        anchor_window = selected[: config.formation_anchor_decision_window]
        generic_wide_anchor_present = any(
            item.shot_size in _WIDE_SHOT_SIZES for item in anchor_window
        )
        formation_anchor_present = any(
            _formation_wide_evidence(
                shot_size=item.shot_size,
                formation_person_count_normalized=(
                    item.formation_person_count_normalized
                ),
                formation_pose_coverage=item.formation_pose_coverage,
                config=config,
            )
            for item in anchor_window
        )
        post_burst_formation_resets: set[int] = set()
        for burst_right in burst_right_positions:
            for candidate_position in range(burst_right + 1, len(selected)):
                candidate = selected[candidate_position]
                previous = selected[candidate_position - 1]
                if (
                    _formation_wide_evidence(
                        shot_size=candidate.shot_size,
                        formation_person_count_normalized=(
                            candidate.formation_person_count_normalized
                        ),
                        formation_pose_coverage=(
                            candidate.formation_pose_coverage
                        ),
                        config=config,
                    )
                    and previous.shot_size in _NONWIDE_SHOT_SIZES
                    and previous.derivative_family_id
                    != candidate.derivative_family_id
                ):
                    post_burst_formation_resets.add(candidate_position)
                    break

        strict_highlight = (
            config.enforce_montage_grammar
            and directive.role is SectionRole.CHORUS_LIKE
            and directive.duration_seconds
            >= config.highlight_minimum_duration_seconds
        )
        failures: list[str] = []
        if strict_highlight:
            if (
                known_fraction + _TIME_EPSILON
                < config.highlight_minimum_known_shot_fraction
            ):
                failures.append("highlight_shot_size_evidence_incomplete")
            if (
                float(support["closeup_supported_frame_fraction"])
                + _TIME_EPSILON
                < config.highlight_minimum_closeup_fraction
            ):
                failures.append("highlight_closeup_candidate_support_insufficient")
            elif (
                closeup_fraction + _TIME_EPSILON
                < config.highlight_minimum_closeup_fraction
            ):
                failures.append("highlight_closeup_fraction_below_floor")
            if (
                float(support["nonwide_supported_frame_fraction"])
                + _TIME_EPSILON
                < config.highlight_minimum_nonwide_fraction
            ):
                failures.append("highlight_nonwide_candidate_support_insufficient")
            elif nonwide_fraction + _TIME_EPSILON < (
                config.highlight_minimum_nonwide_fraction
            ):
                failures.append("highlight_nonwide_fraction_below_floor")
            if (
                source_switch_boundaries > 0
                and wide_involved_source_switch_fraction + _TIME_EPSILON
                < config.highlight_minimum_wide_involved_source_switch_fraction
            ):
                failures.append(
                    "highlight_wide_involved_source_switch_fraction_below_"
                    "reference_floor"
                )
            if int(
                support["multi_family_nonwide_supported_frame_count"]
            ) < 2:
                failures.append(
                    "highlight_multi_source_nonwide_support_insufficient"
                )
            elif (
                int(support["multi_family_nonwide_supported_frame_count"])
                >= 2
                and burst_boundaries
                < config.highlight_minimum_nonwide_burst_boundaries
            ):
                failures.append("highlight_nonwide_burst_missing")
            if (
                burst_boundaries
                >= config.highlight_minimum_nonwide_burst_boundaries
                and len(post_burst_formation_resets)
                < config.highlight_minimum_post_burst_formation_resets
            ):
                failures.append("post_burst_formation_wide_reset_missing")
            if (
                directive.establish_with_wide_shot
                and not bool(support["early_wide_supported"])
            ):
                failures.append("formation_wide_candidate_support_insufficient")
            elif (
                directive.establish_with_wide_shot
                and not generic_wide_anchor_present
            ):
                failures.append("formation_wide_anchor_missing")
            elif (
                directive.establish_with_wide_shot
                and not bool(support["early_formation_wide_supported"])
            ):
                failures.append(
                    "formation_wide_candidate_evidence_insufficient"
                )
            elif directive.establish_with_wide_shot and not formation_anchor_present:
                failures.append("formation_wide_anchor_evidence_missing")
        all_failures.extend(
            f"section_{directive.section_index}:{value}" for value in failures
        )
        section_metrics.append(
            {
                "section_index": directive.section_index,
                "role": directive.role.value,
                "duration_seconds": directive.duration_seconds,
                "strict_highlight_gate_applied": strict_highlight,
                "selected_shot_count": len(selected),
                "known_shot_size_count": len(known),
                "known_shot_size_fraction": known_fraction,
                "wide_shot_count": len(wides),
                "wide_shot_fraction": wide_fraction,
                "nonwide_shot_count": len(nonwides),
                "nonwide_shot_fraction": nonwide_fraction,
                "closeup_shot_count": len(closeups),
                "closeup_shot_fraction": closeup_fraction,
                "nonwide_burst_boundary_count": burst_boundaries,
                "closeup_chain_boundary_count": closeup_chain_boundaries,
                "source_switch_boundary_count": source_switch_boundaries,
                "wide_involved_source_switch_boundary_count": (
                    wide_involved_source_switch_boundaries
                ),
                "wide_involved_source_switch_fraction": (
                    wide_involved_source_switch_fraction
                ),
                "formation_anchor_decision_window": (
                    config.formation_anchor_decision_window
                ),
                "generic_wide_anchor_present": generic_wide_anchor_present,
                "formation_wide_anchor_present": formation_anchor_present,
                "post_burst_formation_wide_reset_count": len(
                    post_burst_formation_resets
                ),
                "candidate_support": dict(support),
                "failure_codes": failures,
                "passed": not failures,
            }
        )
    strict_chorus_present = any(
        directive.role is SectionRole.CHORUS_LIKE
        and directive.duration_seconds >= config.highlight_minimum_duration_seconds
        for directive in section_plan.sections
    )
    if config.enforce_montage_grammar and strict_chorus_present:
        if (
            int(exact_anchor_metrics["missing_section_progress_decision_count"])
            or int(exact_anchor_metrics["missing_section_role_decision_count"])
        ):
            all_failures.append("highlight_exact_section_progress_evidence_missing")
        elif int(exact_anchor_metrics["section_role_plan_mismatch_count"]):
            all_failures.append("highlight_exact_section_role_plan_mismatch")
        elif not int(
            exact_anchor_metrics["observed_chorus_section_early_anchor_count"]
        ):
            all_failures.append("highlight_early_wide_anchor_evidence_missing")
        elif float(
            exact_anchor_metrics["chorus_section_early_wide_anchor_fraction"]
        ) + _TIME_EPSILON < config.highlight_minimum_early_wide_anchor_fraction:
            all_failures.append(
                "highlight_early_wide_anchor_fraction_below_reference_floor"
            )
        if int(exact_anchor_metrics["missing_phrase_progress_decision_count"]):
            all_failures.append("highlight_exact_phrase_progress_evidence_missing")
        elif not int(
            exact_anchor_metrics["observed_chorus_phrase_entry_anchor_count"]
        ):
            all_failures.append("highlight_phrase_entry_wide_anchor_evidence_missing")
        elif float(
            exact_anchor_metrics["chorus_phrase_entry_wide_anchor_fraction"]
        ) + _TIME_EPSILON < (
            config.highlight_minimum_phrase_entry_wide_anchor_fraction
        ):
            all_failures.append(
                "highlight_phrase_entry_wide_anchor_fraction_below_reference_floor"
            )
    return {
        "schema_version": 1,
        "contract": MONTAGE_GRAMMAR_CONTRACT,
        "passed": not all_failures,
        "failure_codes": all_failures,
        "exact_anchor_metrics": exact_anchor_metrics,
        "policy": {
            "wide_shots": (
                "formation_evidenced_anchor_and_post_burst_reset_not_full_"
                "highlight_occupancy"
            ),
            "highlight_shots": (
                "reference_closeup_and_wide_transition_floors_plus_explicit_"
                "ive_product_nonwide_burst_requirement"
            ),
            "formation_evidence_semantics": (
                "identity_free_nonzero_observed_person_and_pose_evidence_"
                "not_proof_of_complete_group_formation"
            ),
            "insufficient_closeup_or_nonwide_candidates": (
                "fail_before_render_instead_of_fabricating_a_flat_highlight"
            ),
            "learned_logits_remain_primary": True,
            "chorus_wide_involved_family_switch_prior": {
                "reference_p10_hard_floor": 0.37333333333333335,
                "reference_median": 0.8,
                "generic_wide_involved_switch_raw_bonus": 0.10,
                "formation_evidenced_wide_additional_raw_bonus": 0.06,
                "same_source_or_derivative_family_bonus": 0.0,
                "applied_through_montage_grammar_weight": (
                    config.montage_grammar_score_weight
                ),
            },
        },
        "thresholds": {
            "highlight_minimum_duration_seconds": (
                config.highlight_minimum_duration_seconds
            ),
            "highlight_minimum_known_shot_fraction": (
                config.highlight_minimum_known_shot_fraction
            ),
            "highlight_minimum_closeup_fraction": (
                config.highlight_minimum_closeup_fraction
            ),
            "highlight_minimum_nonwide_fraction": (
                config.highlight_minimum_nonwide_fraction
            ),
            "highlight_minimum_wide_involved_source_switch_fraction": (
                config.highlight_minimum_wide_involved_source_switch_fraction
            ),
            "highlight_minimum_early_wide_anchor_fraction": (
                config.highlight_minimum_early_wide_anchor_fraction
            ),
            "highlight_minimum_phrase_entry_wide_anchor_fraction": (
                config.highlight_minimum_phrase_entry_wide_anchor_fraction
            ),
            "highlight_burst_maximum_shot_seconds": (
                config.highlight_burst_maximum_shot_seconds
            ),
            "highlight_minimum_nonwide_burst_boundaries": (
                config.highlight_minimum_nonwide_burst_boundaries
            ),
            "formation_anchor_decision_window": (
                config.formation_anchor_decision_window
            ),
            "minimum_formation_anchor_person_count_normalized": (
                config.minimum_formation_anchor_person_count_normalized
            ),
            "highlight_minimum_post_burst_formation_resets": (
                config.highlight_minimum_post_burst_formation_resets
            ),
        },
        "sections": section_metrics,
    }


def _bounded_continuity_delta(left: float, right: float, scale: float) -> float:
    return min(1.0, abs(left - right) / scale)


def stage_edit_boundary_continuity_score(
    outgoing: StageEditSourceContinuity,
    incoming: StageEditSourceContinuity,
) -> float:
    """Return a neutral-or-negative same-time boundary continuity score.

    The decoder should prefer a naturally matchable pair, but absent evidence
    must never look like a perfect detected match. Consequently this score is
    a bounded penalty in ``[-1, 0]``. Comparable observations determine the
    main penalty and uncovered evidence receives a small fixed uncertainty
    penalty. Exact-frame render-domain geometry and look gates remain
    authoritative.
    """

    weighted_penalties: list[tuple[float, float]] = []

    if outgoing.face_scale is not None and incoming.face_scale is not None:
        left = max(abs(outgoing.face_scale), 1.0e-6)
        right = max(abs(incoming.face_scale), 1.0e-6)
        scale_penalty = min(1.0, abs(math.log(right / left)) / math.log(1.50))
        weighted_penalties.append((0.28, scale_penalty))
    if all(
        value is not None
        for value in (
            outgoing.face_center_x,
            outgoing.face_center_y,
            incoming.face_center_x,
            incoming.face_center_y,
        )
    ):
        face_distance = math.hypot(
            float(incoming.face_center_x) - float(outgoing.face_center_x),
            float(incoming.face_center_y) - float(outgoing.face_center_y),
        )
        weighted_penalties.append((0.28, min(1.0, face_distance / 0.22)))
    elif all(
        value is not None
        for value in (
            outgoing.body_center_x,
            outgoing.body_center_y,
            incoming.body_center_x,
            incoming.body_center_y,
        )
    ):
        body_distance = math.hypot(
            float(incoming.body_center_x) - float(outgoing.body_center_x),
            float(incoming.body_center_y) - float(outgoing.body_center_y),
        )
        weighted_penalties.append((0.18, min(1.0, body_distance / 0.28)))

    if all(
        value is not None
        for value in (
            outgoing.camera_motion_x,
            outgoing.camera_motion_y,
            incoming.camera_motion_x,
            incoming.camera_motion_y,
        )
    ):
        motion_delta = math.hypot(
            float(incoming.camera_motion_x) - float(outgoing.camera_motion_x),
            float(incoming.camera_motion_y) - float(outgoing.camera_motion_y),
        )
        weighted_penalties.append((0.12, min(1.0, motion_delta / 0.75)))
    if outgoing.pose_motion is not None and incoming.pose_motion is not None:
        weighted_penalties.append(
            (
                0.06,
                _bounded_continuity_delta(
                    outgoing.pose_motion, incoming.pose_motion, 0.40
                ),
            )
        )
    if (
        outgoing.camera_zoom_velocity is not None
        and incoming.camera_zoom_velocity is not None
    ):
        weighted_penalties.append(
            (
                0.05,
                _bounded_continuity_delta(
                    outgoing.camera_zoom_velocity,
                    incoming.camera_zoom_velocity,
                    0.35,
                ),
            )
        )
    if (
        outgoing.camera_roll_normalized is not None
        and incoming.camera_roll_normalized is not None
    ):
        weighted_penalties.append(
            (
                0.05,
                _bounded_continuity_delta(
                    outgoing.camera_roll_normalized,
                    incoming.camera_roll_normalized,
                    0.25,
                ),
            )
        )

    for weight, name, scale in (
        (0.06, "exposure", 0.25),
        (0.04, "color_temperature_normalized", 0.30),
        (0.03, "sharpness", 0.35),
    ):
        left = getattr(outgoing, name)
        right = getattr(incoming, name)
        if left is not None and right is not None:
            weighted_penalties.append(
                (weight, _bounded_continuity_delta(left, right, scale))
            )

    observed_weight = sum(weight for weight, _penalty in weighted_penalties)
    if observed_weight:
        observed_penalty = sum(
            weight * value for weight, value in weighted_penalties
        ) / observed_weight
    else:
        observed_penalty = 0.0
    # The face-based full evidence path totals 0.97. Using its weight as the
    # coverage denominator keeps body-fallback evidence slightly conservative.
    evidence_coverage = min(1.0, observed_weight / 0.97)
    penalty = (
        evidence_coverage * observed_penalty
        + (1.0 - evidence_coverage) * 0.12
    )
    return -min(1.0, max(0.0, penalty))


def _boundary_continuity_raw(
    frame: StageEditHeadFrame,
    outgoing_source_index: int | None,
    incoming_source_index: int,
) -> float:
    if outgoing_source_index is None or not frame.source_continuity:
        return 0.0
    return stage_edit_boundary_continuity_score(
        frame.source_continuity[outgoing_source_index],
        frame.source_continuity[incoming_source_index],
    )


def _source_span_rejection(
    *,
    source: StageEditSourceMetadata,
    frames: Sequence[StageEditHeadFrame],
    start_index: int,
    end_index: int,
    master_end_seconds: float,
    margin: float,
) -> str | None:
    for frame in frames[start_index:end_index]:
        if not frame.effective_source_mask[source.source_index]:
            return "source_unavailable_or_hard_gated_inside_span"
    master_start = frames[start_index].master_time_seconds
    source_start = source.source_start_at(master_start)
    source_end = source.source_start_at(master_end_seconds)
    if source_start < -_TIME_EPSILON or source_end > source.source_duration_seconds + _TIME_EPSILON:
        return "source_clock_outside_available_media"
    for cut_time in source.internal_cut_times_seconds:
        if source_start + margin + _TIME_EPSILON < cut_time < source_end - margin - _TIME_EPSILON:
            return "source_internal_cut_inside_shot"
    return None


def _input_digest(
    *,
    sources: Sequence[Path],
    source_metadata: Sequence[StageEditSourceMetadata],
    frames: Sequence[StageEditHeadFrame],
    section_plan: SectionPlan,
    constraints: Sequence[StageEditSectionConstraint],
    config: StageEditSemiMarkovConfig,
    schema: StageEditTokenSchema,
    source_derivative_metadata: SourceDerivativeFamilyMetadata | None,
) -> str:
    payload = {
        "sources": [str(Path(value)) for value in sources],
        "source_metadata": [asdict(value) for value in source_metadata],
        "frames": [value.to_dict() for value in frames],
        "section_plan": section_plan.to_dict(),
        "constraints": [asdict(value) for value in constraints],
        "config": asdict(config),
        "token_schema_fingerprint": schema.fingerprint,
        "source_derivative_metadata": (
            source_derivative_metadata.to_dict()
            if source_derivative_metadata is not None
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _family_occupancy(
    source_occupancy: Sequence[float],
    family_ids: Sequence[int],
) -> tuple[float, ...]:
    if len(source_occupancy) != len(family_ids):
        raise StageEditDecodeInputError(
            "source occupancy and derivative-family metadata do not align"
        )
    family_count = len(set(family_ids))
    totals = [0.0] * family_count
    for source_index, value in enumerate(source_occupancy):
        totals[family_ids[source_index]] += float(value)
    return tuple(totals)


def _family_occupancy_caps(
    source_metadata: Sequence[StageEditSourceMetadata],
    family_ids: Sequence[int],
    config: StageEditSemiMarkovConfig,
) -> tuple[float, ...]:
    family_count = len(set(family_ids))
    caps = [config.maximum_source_occupancy_fraction] * family_count
    for source_index, metadata in enumerate(source_metadata):
        family = family_ids[source_index]
        caps[family] = min(caps[family], metadata.maximum_occupancy_fraction)
    return tuple(caps)


def _timeline_output_path(output: Path, rank: int) -> Path:
    suffix = output.suffix or ".mp4"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}.stageedit-{rank:02d}{suffix}")


def _build_stage_edit(
    *,
    rank: int,
    state: _BeamState,
    sources: tuple[Path, ...],
    source_metadata: tuple[StageEditSourceMetadata, ...],
    section_plan: SectionPlan,
    aspect_ratio: AspectRatio,
    width: int,
    height: int,
    fps: float,
    reference_source: int,
    output: Path,
    checkpoint_path: str,
    source_derivative_metadata: SourceDerivativeFamilyMetadata | None,
    config: StageEditSemiMarkovConfig,
) -> StageEdit:
    transition_authority = StageEditTransitionAuthority(
        mode=config.transition_authority
    )
    shots: list[StageShot] = []
    for decision in state.decisions:
        metadata = source_metadata[decision.source_index]
        duration = decision.duration_seconds
        midpoint = decision.master_start_seconds + duration / 2.0
        section_index = section_plan.directive_at(midpoint).section_index
        segment = RenderSegment(
            source_index=decision.source_index,
            source_start=max(0.0, metadata.source_start_at(decision.master_start_seconds)),
            duration=duration,
            master_start=decision.master_start_seconds - section_plan.start_seconds,
            editorial_transform=decision.editorial_transform,
        )
        shots.append(
            StageShot(
                shot_index=decision.shot_index,
                section_index=section_index,
                segment=segment,
                decision_id=(
                    f"stageedit-semimarkov-r{rank:02d}-s{decision.shot_index:04d}"
                ),
                decision_confidence=decision.decision_confidence,
                section_role=decision.section_role,
                section_progress=decision.section_progress,
                phrase_progress=decision.phrase_progress,
            )
        )
    boundaries = tuple(
        StageBoundary(
            boundary_index=index,
            transition=RenderTransition(
                kind=TransitionKind.HARD_CUT,
                frame_count=0,
                reason=(
                    "stage_edit_semimarkov_requested_"
                    f"{state.decisions[index + 1].transition_label}"
                ),
                tier="C",
            ),
        )
        for index in range(len(shots) - 1)
    )
    return StageEdit(
        sources=sources,
        reference_source=reference_source,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
        fps=fps,
        master_audio_start=section_plan.start_seconds,
        section_plan=section_plan,
        shots=tuple(shots),
        boundaries=boundaries,
        output=_timeline_output_path(output, rank),
        candidate_history=(
            {
                "decoder": "stage_edit_semimarkov_beam_v5",
                "timeline_rank": rank,
                "checkpoint_path": checkpoint_path,
                "score": state.score,
                "decision_count": len(state.decisions),
                "source_derivative_diversity": (
                    source_derivative_metadata.to_dict()
                    if source_derivative_metadata is not None
                    else {
                        "production_ready": False,
                        "family_ids": list(range(len(sources))),
                        "status": "decoder_unit_identity_default",
                    }
                ),
                "transition_translation": (
                    "model metadata is duration-neutral hard_cut; renderer-owned "
                    "group_transition alone selects hard_cut or evidence-gated "
                    "face_dissolve"
                    if transition_authority.renderer_owned
                    else "model transition labels retained in audit; renderer IR "
                    "uses duration-neutral hard cuts"
                ),
                "transition_authority": transition_authority.to_dict(),
            },
        ),
    )


def _validate_decode_inputs(
    *,
    sources: tuple[Path, ...],
    source_metadata: tuple[StageEditSourceMetadata, ...],
    frames: tuple[StageEditHeadFrame, ...],
    section_plan: SectionPlan,
    schema: StageEditTokenSchema,
    config: StageEditSemiMarkovConfig,
    reference_source: int,
    aspect_ratio: AspectRatio,
    width: int,
    height: int,
    fps: float,
    source_derivative_metadata: SourceDerivativeFamilyMetadata | None,
) -> None:
    source_count = len(sources)
    if source_count < 2:
        raise StageEditDecodeInputError("semi-Markov decoding needs at least two sources")
    if len(source_metadata) != source_count:
        raise StageEditDecodeInputError("source metadata count does not match sources")
    if tuple(value.source_index for value in source_metadata) != tuple(range(source_count)):
        raise StageEditDecodeInputError("source metadata indices must be contiguous")
    source_ids = tuple(value.source_id for value in source_metadata)
    if len(set(source_ids)) != len(source_ids):
        raise StageEditDecodeInputError("source ids must be unique")
    if config.minimum_unique_sources > source_count:
        raise StageEditDecodeInputError("minimum_unique_sources exceeds source count")
    if source_derivative_metadata is not None:
        if not isinstance(source_derivative_metadata, SourceDerivativeFamilyMetadata):
            raise StageEditDecodeInputError(
                "source derivative metadata has an invalid type"
            )
        if source_derivative_metadata.source_count != source_count:
            raise StageEditDecodeInputError(
                "source derivative metadata count does not match sources"
            )
        if not source_derivative_metadata.production_ready:
            raise StageEditDecodeInputError(
                "source derivative metadata is uncertain or clock-incomplete"
            )
        if source_derivative_metadata.family_count < config.minimum_unique_sources:
            raise StageEditDecodeInputError(
                "minimum_unique_sources exceeds verified derivative-family count"
            )
    if (
        isinstance(reference_source, bool)
        or not isinstance(reference_source, int)
        or not 0 <= reference_source < source_count
    ):
        raise StageEditDecodeInputError("reference_source is outside the source list")
    if not frames:
        raise StageEditDecodeInputError("candidate head frames cannot be empty")
    dynamic_crop_modes = tuple(frame.dynamic_crop_v2 is not None for frame in frames)
    if any(dynamic_crop_modes) and not all(dynamic_crop_modes):
        raise StageEditDecodeInputError(
            "candidate head frames cannot mix legacy and Dynamic Crop v2 contracts"
        )
    if abs(frames[0].master_time_seconds - section_plan.start_seconds) > _TIME_EPSILON:
        raise StageEditDecodeInputError(
            "first candidate time must equal the section-plan start"
        )
    times = tuple(value.master_time_seconds for value in frames)
    if any(right <= left for left, right in zip(times, times[1:])):
        raise StageEditDecodeInputError("candidate times must be strictly increasing")
    if frames[-1].master_time_seconds >= section_plan.end_seconds - _TIME_EPSILON:
        raise StageEditDecodeInputError(
            "candidate frames must be before the terminal section-plan time"
        )
    required_boundaries = tuple(
        section.end_seconds for section in section_plan.sections[:-1]
    )
    for boundary in required_boundaries:
        if not any(abs(value - boundary) <= _TIME_EPSILON for value in times):
            raise StageEditDecodeInputError(
                f"section boundary {boundary:.6f} is missing from candidate times"
            )
    for index, frame in enumerate(frames):
        if len(frame.source_logits) != source_count:
            raise StageEditDecodeInputError(
                f"frame {index} source logit count does not match sources"
            )
        if len(frame.available_source_mask) != source_count or len(
            frame.hard_source_gate_mask
        ) != source_count:
            raise StageEditDecodeInputError(
                f"frame {index} source mask count does not match sources"
            )
        if frame.source_shot_sizes and len(frame.source_shot_sizes) != source_count:
            raise StageEditDecodeInputError(
                f"frame {index} shot-size count does not match sources"
            )
        if frame.source_continuity and len(frame.source_continuity) != source_count:
            raise StageEditDecodeInputError(
                f"frame {index} continuity count does not match sources"
            )
        if len(frame.duration_logits) != schema.duration_bucket_count:
            raise StageEditDecodeInputError(
                f"frame {index} duration logit count does not match schema"
            )
        if len(frame.transition_logits) != len(schema.transition_labels):
            raise StageEditDecodeInputError(
                f"frame {index} transition logit count does not match schema"
            )
        if not any(frame.effective_source_mask):
            raise StageEditDecodeInputError(
                f"frame {index} has no source surviving availability and hard gates"
            )
    if not isinstance(aspect_ratio, AspectRatio):
        try:
            aspect_ratio = AspectRatio(aspect_ratio)
        except ValueError as exc:
            raise StageEditDecodeInputError("aspect_ratio is invalid") from exc
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or width % 2
        or height % 2
    ):
        raise StageEditDecodeInputError("output dimensions must be positive even integers")
    expected_ratio = 16.0 / 9.0 if aspect_ratio is AspectRatio.LANDSCAPE else 9.0 / 16.0
    if abs(width / height - expected_ratio) > 0.002:
        raise StageEditDecodeInputError("output dimensions do not match aspect_ratio")
    if _finite(fps, "fps") <= 0.0:
        raise StageEditDecodeInputError("fps must be positive")


def decode_stage_edit_semimarkov_beam(
    *,
    checkpoint_directory: Path,
    sources: Sequence[Path],
    source_metadata: Sequence[StageEditSourceMetadata],
    head_frames: Sequence[StageEditHeadFrame],
    section_plan: SectionPlan,
    output: Path,
    section_constraints: Sequence[StageEditSectionConstraint] = (),
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    config: StageEditSemiMarkovConfig = StageEditSemiMarkovConfig(),
    reference_source: int = 0,
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
    width: int = 1920,
    height: int = 1080,
    fps: float = 29.97,
    source_derivative_metadata: SourceDerivativeFamilyMetadata | None = None,
) -> StageEditSemiMarkovResult:
    """Decode StageEdit heads into ranked, renderer-valid StageEdit timelines.

    The readiness check intentionally occurs before input validation/search so
    this function cannot be repurposed as an implicit random-checkpoint
    decoder.  It performs no rendering, media probing, or pipeline mutation.
    """

    if checkpoint_directory is None:
        raise StageEditCheckpointNotReadyError(
            "StageEdit semi-Markov decoding requires a production-ready checkpoint"
        )
    readiness = stage_edit_readiness_report(Path(checkpoint_directory))
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path
        and readiness.training_steps > 0
    ):
        raise StageEditCheckpointNotReadyError(
            "StageEdit semi-Markov decoding requires a production-ready checkpoint: "
            f"{readiness.status}: {readiness.reason}"
        )

    normalized_sources = tuple(Path(value) for value in sources)
    normalized_metadata = tuple(source_metadata)
    frames = tuple(head_frames)
    constraints = tuple(section_constraints)
    if not isinstance(aspect_ratio, AspectRatio):
        try:
            aspect_ratio = AspectRatio(aspect_ratio)
        except ValueError as exc:
            raise StageEditDecodeInputError("aspect_ratio is invalid") from exc
    _validate_decode_inputs(
        sources=normalized_sources,
        source_metadata=normalized_metadata,
        frames=frames,
        section_plan=section_plan,
        schema=schema,
        config=config,
        reference_source=reference_source,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
        fps=fps,
        source_derivative_metadata=source_derivative_metadata,
    )
    resolved = _resolved_constraints(
        section_plan, constraints, schema, config
    )

    source_count = len(normalized_sources)
    derivative_family_ids = (
        source_derivative_metadata.family_ids
        if source_derivative_metadata is not None
        else tuple(range(source_count))
    )
    derivative_family_count = len(set(derivative_family_ids))
    derivative_family_caps = _family_occupancy_caps(
        normalized_metadata,
        derivative_family_ids,
        config,
    )
    montage_support_by_section = _section_candidate_shot_support(
        frames=frames,
        source_metadata=normalized_metadata,
        section_plan=section_plan,
        derivative_family_ids=derivative_family_ids,
        config=config,
    )
    section_count = len(section_plan.sections)
    master_duration = section_plan.end_seconds - section_plan.start_seconds
    terminal_index = len(frames)
    all_times = tuple(value.master_time_seconds for value in frames) + (
        section_plan.end_seconds,
    )
    source_log_probabilities = tuple(
        _masked_log_softmax(frame.source_logits, frame.effective_source_mask)
        for frame in frames
    )
    duration_log_probabilities = tuple(
        _log_softmax(frame.duration_logits) for frame in frames
    )
    transition_authority = StageEditTransitionAuthority(
        mode=config.transition_authority
    )
    if transition_authority.renderer_owned:
        transition_log_probabilities = tuple(
            tuple(
                0.0 if label == "hard_cut" else -math.inf
                for label in schema.transition_labels
            )
            for _ in frames
        )
    else:
        transition_log_probabilities = tuple(
            _masked_log_softmax(
                frame.transition_logits,
                tuple(
                    label
                    in resolved[
                        _section_index_at_start(
                            section_plan, frame.master_time_seconds
                        )
                    ].allowed_transition_labels
                    for label in schema.transition_labels
                ),
            )
            for frame in frames
        )
    # Legacy four-value crops remain available for old checkpoints.  Dynamic
    # Crop v2 predictions are intentionally *not* projected here: the model's
    # normalized K=8 times can only be interpreted after a concrete duration
    # hypothesis exists inside the semi-Markov expansion.
    crop_values = tuple(
        None if frame.dynamic_crop_v2 is not None else _crop_transform(frame.crop)
        for frame in frames
    )

    empty_state = _BeamState(
        current_index=0,
        current_source=None,
        recent_sources=(),
        recent_shot_sizes=(),
        recent_shot_durations=(),
        recent_cut_times=(),
        source_occupancy_seconds=(0.0,) * source_count,
        section_switch_counts=(0,) * section_count,
        decisions=(),
        score=0.0,
        component_totals=(),
    )
    beams: list[list[_BeamState]] = [[] for _ in range(terminal_index + 1)]
    beams[0].append(empty_state)
    rejection_counts: dict[str, int] = {}
    endpoint_records: list[dict[str, Any]] = [
        {
            "candidate_index": index,
            "master_time_seconds": all_times[index],
            "incoming_state_count": 0,
            "span_candidate_count": 0,
            "source_candidate_count": 0,
            "transition_hypothesis_count": 0,
            "accepted_hypothesis_count": 0,
            "beam_pruned_state_count": 0,
        }
        for index in range(terminal_index + 1)
    ]
    transition_hypothesis_count = 0
    accepted_hypothesis_count = 0
    pruned_count = 0

    def reject(name: str) -> None:
        rejection_counts[name] = rejection_counts.get(name, 0) + 1

    for start_index in range(terminal_index):
        states = sorted(beams[start_index], key=_state_sort_key)
        if len(states) > config.beam_width:
            states, removed, duplicates = _prune_states(
                states, config.beam_width
            )
            if duplicates:
                rejection_counts["duplicate_partial_timeline"] = (
                    rejection_counts.get("duplicate_partial_timeline", 0)
                    + duplicates
                )
            pruned_count += removed
            endpoint_records[start_index]["beam_pruned_state_count"] += removed
        beams[start_index] = states
        endpoint_records[start_index]["incoming_state_count"] = len(states)
        if not states:
            continue
        frame = frames[start_index]
        start_time = frame.master_time_seconds
        section_index = _section_index_at_start(section_plan, start_time)
        directive = section_plan.sections[section_index]
        constraint = resolved[section_index]
        section_remaining = directive.end_seconds - start_time
        maximum_duration = min(constraint.maximum_shot_seconds, section_remaining)
        minimum_duration = constraint.minimum_shot_seconds

        for state in states:
            for end_index in range(start_index + 1, terminal_index + 1):
                endpoint_records[start_index]["span_candidate_count"] += 1
                end_time = all_times[end_index]
                duration = end_time - start_time
                if duration < minimum_duration - _TIME_EPSILON:
                    reject("shot_below_minimum_duration")
                    continue
                if duration > maximum_duration + _TIME_EPSILON:
                    reject("shot_above_maximum_or_crosses_section")
                    break

                skipped_cut_score = sum(
                    _log_sigmoid(-frames[index].cut_logit)
                    for index in range(start_index + 1, end_index)
                )
                boundary_cut_score = (
                    0.0 if not state.decisions else _log_sigmoid(frame.cut_logit)
                )
                cut_raw = boundary_cut_score + skipped_cut_score
                duration_bucket = schema.duration_bucket(duration)
                duration_raw = duration_log_probabilities[start_index][duration_bucket]
                ideal_duration = min(
                    maximum_duration,
                    max(
                        minimum_duration,
                        60.0
                        / max(1e-9, directive.target_source_switches_per_minute),
                    ),
                )
                cadence_raw = -abs(duration - ideal_duration) / max(
                    ideal_duration, minimum_duration
                )

                for source_index, metadata in enumerate(normalized_metadata):
                    endpoint_records[start_index]["source_candidate_count"] += 1
                    if not frame.effective_source_mask[source_index]:
                        reject("source_unavailable_or_hard_gated_at_start")
                        continue
                    selected_continuity = (
                        frame.source_continuity[source_index]
                        if frame.source_continuity
                        else StageEditSourceContinuity()
                    )
                    span_rejection = _source_span_rejection(
                        source=metadata,
                        frames=frames,
                        start_index=start_index,
                        end_index=end_index,
                        master_end_seconds=end_time,
                        margin=config.internal_cut_exclusion_margin_seconds,
                    )
                    if span_rejection is not None:
                        reject(span_rejection)
                        continue
                    candidate_family = derivative_family_ids[source_index]
                    current_family = (
                        derivative_family_ids[state.current_source]
                        if state.current_source is not None
                        else None
                    )
                    if (
                        current_family == candidate_family
                        and not config.allow_consecutive_same_source
                    ):
                        reject("consecutive_same_derivative_family")
                        continue
                    if (
                        config.enforce_aba_exclusion
                        and len(state.recent_sources) >= 2
                        and derivative_family_ids[state.recent_sources[-2]]
                        == candidate_family
                        and derivative_family_ids[state.recent_sources[-1]]
                        != candidate_family
                    ):
                        reject("aba_derivative_family_return")
                        # Compatibility alias: the v1-v3 counter name remains
                        # readable, but its v4 semantics are family-aware.
                        reject("aba_source_return")
                        continue

                    shot_size = _shot_size(frame, metadata)
                    if shot_size != "unknown":
                        same_size_run = 0
                        for prior in reversed(state.recent_shot_sizes):
                            if prior != shot_size:
                                break
                            same_size_run += 1
                        if same_size_run >= config.maximum_same_shot_size_run:
                            reject("maximum_same_shot_size_run")
                            continue

                    prospective_occupancy = list(state.source_occupancy_seconds)
                    prospective_occupancy[source_index] += duration
                    prospective_family_occupancy = _family_occupancy(
                        prospective_occupancy,
                        derivative_family_ids,
                    )
                    if prospective_family_occupancy[candidate_family] > (
                        derivative_family_caps[candidate_family]
                        * master_duration
                        + _TIME_EPSILON
                    ):
                        reject("derivative_family_occupancy_cap")
                        reject("source_occupancy_cap")
                        continue
                    capacity = sum(
                        max(0.0, cap * master_duration - value)
                        for cap, value in zip(
                            # All derivatives of one camera share one cap.
                            derivative_family_caps,
                            prospective_family_occupancy,
                            strict=True,
                        )
                    )
                    remaining_duration = section_plan.end_seconds - end_time
                    if capacity + _TIME_EPSILON < remaining_duration:
                        reject("remaining_derivative_family_occupancy_capacity")
                        reject("remaining_source_occupancy_capacity")
                        continue

                    section_counts = list(state.section_switch_counts)
                    is_boundary = bool(state.decisions)
                    if is_boundary:
                        section_counts[section_index] += 1
                        section_limit = _maximum_switch_count(
                            constraint.maximum_switches_per_minute,
                            directive.duration_seconds,
                        )
                        if section_counts[section_index] > section_limit:
                            reject("section_cut_density")
                            continue
                        global_limit = _maximum_switch_count(
                            config.maximum_global_switches_per_minute,
                            master_duration,
                        )
                        if len(state.decisions) > global_limit:
                            reject("global_cut_density")
                            continue
                        if (
                            start_time - section_plan.start_seconds
                            >= config.cut_density_window_seconds
                        ):
                            recent_count = sum(
                                value
                                >= start_time - config.cut_density_window_seconds
                                for value in state.recent_cut_times
                            ) + 1
                            recent_limit = _maximum_switch_count(
                                config.maximum_global_switches_per_minute,
                                config.cut_density_window_seconds,
                            )
                            if recent_count > recent_limit:
                                reject("rolling_cut_density")
                                continue

                    used_families = {
                        derivative_family_ids[index]
                        for index in state.recent_sources
                    }
                    used_families.update(
                        derivative_family_ids[item.source_index]
                        for item in state.decisions
                    )
                    used_families.add(candidate_family)
                    if end_index == terminal_index:
                        if len(used_families) < config.minimum_unique_sources:
                            reject("final_derivative_family_diversity")
                            continue
                    else:
                        possible_families = set(used_families)
                        for future_frame in frames[end_index:]:
                            possible_families.update(
                                derivative_family_ids[index]
                                for index, allowed in enumerate(
                                    future_frame.effective_source_mask
                                )
                                if allowed
                            )
                        if len(possible_families) < config.minimum_unique_sources:
                            reject("unreachable_derivative_family_diversity")
                            continue

                    transition_indices = (
                        (None,)
                        if not state.decisions
                        else (
                            (schema.transition_labels.index("hard_cut"),)
                            if transition_authority.renderer_owned
                            else tuple(range(len(schema.transition_labels)))
                        )
                    )
                    for transition_index in transition_indices:
                        transition_hypothesis_count += 1
                        endpoint_records[start_index]["transition_hypothesis_count"] += 1
                        if transition_index is None:
                            transition_label = "initial"
                            transition_raw = 0.0
                        else:
                            transition_label = schema.transition_labels[transition_index]
                            if transition_label not in constraint.allowed_transition_labels:
                                reject("transition_disallowed_by_section")
                                continue
                            transition_raw = transition_log_probabilities[start_index][
                                transition_index
                            ]

                        dynamic_crop_decode: DynamicCropTrackDecodeV2 | None = None
                        editorial_transform: EditorialTransformTrack | None
                        if frame.dynamic_crop_v2 is not None:
                            dynamic_crop_decode = frame.dynamic_crop_v2.decode(duration)
                            requested_crop = frame.dynamic_crop_v2.values[0]
                            crop_contract = (
                                DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT
                            )
                            editorial_transform = dynamic_crop_decode.track
                            if dynamic_crop_decode.eligible:
                                assert editorial_transform is not None
                                first_keyframe = editorial_transform.keyframes[0]
                                applied_crop = (
                                    first_keyframe.zoom,
                                    first_keyframe.center_x,
                                    first_keyframe.center_y,
                                    first_keyframe.roll_degrees,
                                )
                                crop_corrected = bool(
                                    dynamic_crop_decode.minimum_safe_zoom_projection_count
                                )
                                crop_correction = (
                                    dynamic_crop_decode.minimum_safe_zoom_projection_count
                                    / max(
                                        1,
                                        dynamic_crop_decode.requested_keyframe_count,
                                    )
                                )
                            else:
                                # A rejected v2 trajectory means exactly "no
                                # transform" for this cut.  Never substitute
                                # the legacy crop head, a centered zoom, or a
                                # synthesized t=0 keyframe.
                                applied_crop = (1.0, 0.5, 0.5, 0.0)
                                crop_corrected = False
                                crop_correction = 1.0
                        else:
                            legacy_crop = crop_values[start_index]
                            assert legacy_crop is not None
                            applied_crop, crop_corrected, crop_correction = legacy_crop
                            requested_crop = frame.crop
                            crop_contract = "legacy_static_crop_head_v1"
                            editorial_transform = _editorial_track(applied_crop)
                        source_raw = source_log_probabilities[start_index][source_index]
                        boundary_continuity_raw = _boundary_continuity_raw(
                            frame,
                            state.current_source,
                            source_index,
                        )
                        quality_raw = metadata.quality_log_prior
                        novelty_raw = (
                            1.0
                            if candidate_family
                            not in {
                                derivative_family_ids[item.source_index]
                                for item in state.decisions
                            }
                            else -0.25
                        )
                        if shot_size != "unknown" and shot_size not in state.recent_shot_sizes:
                            novelty_raw += 0.35
                        used_occupancy = [
                            value / master_duration
                            for value in prospective_family_occupancy
                        ]
                        occupancy_mean = (
                            sum(used_occupancy) / derivative_family_count
                        )
                        occupancy_raw = -sum(
                            abs(value - occupancy_mean) for value in used_occupancy
                        ) / derivative_family_count
                        previous_in_section = (
                            state.decisions[-1]
                            if state.decisions
                            and _section_index_at_start(
                                section_plan,
                                state.decisions[-1].master_start_seconds,
                            )
                            == section_index
                            else None
                        )
                        montage_grammar_raw = _montage_grammar_score_raw(
                            directive=directive,
                            shot_size=shot_size,
                            start_seconds=start_time,
                            duration_seconds=duration,
                            previous_shot_size=(
                                previous_in_section.shot_size
                                if previous_in_section is not None
                                else None
                            ),
                            previous_duration_seconds=(
                                previous_in_section.duration_seconds
                                if previous_in_section is not None
                                else None
                            ),
                            derivative_family_source_switched=(
                                previous_in_section is not None
                                and previous_in_section.derivative_family_id
                                != candidate_family
                            ),
                            previous_formation_wide_evidenced=(
                                previous_in_section is not None
                                and _formation_wide_evidence(
                                    shot_size=previous_in_section.shot_size,
                                    formation_person_count_normalized=(
                                        previous_in_section
                                        .formation_person_count_normalized
                                    ),
                                    formation_pose_coverage=(
                                        previous_in_section
                                        .formation_pose_coverage
                                    ),
                                    config=config,
                                )
                            ),
                            formation_person_count_normalized=(
                                selected_continuity.formation_person_count_normalized
                            ),
                            formation_pose_coverage=(
                                selected_continuity.formation_pose_coverage
                            ),
                            config=config,
                        )
                        components = (
                            _component("cut", cut_raw, config.cut_score_weight),
                            _component("source", source_raw, config.source_score_weight),
                            _component(
                                "duration", duration_raw, config.duration_score_weight
                            ),
                            _component(
                                "transition",
                                transition_raw,
                                config.transition_score_weight,
                            ),
                            _component(
                                "crop_correction",
                                -crop_correction,
                                config.crop_correction_weight,
                            ),
                            _component(
                                "source_quality",
                                quality_raw,
                                config.source_quality_weight,
                            ),
                            _component(
                                "boundary_continuity",
                                boundary_continuity_raw,
                                config.boundary_continuity_weight,
                            ),
                            _component(
                                "cadence", cadence_raw, config.cadence_score_weight
                            ),
                            _component(
                                "diversity", novelty_raw, config.diversity_score_weight
                            ),
                            _component(
                                "occupancy_balance",
                                occupancy_raw,
                                config.occupancy_balance_weight,
                            ),
                            _component(
                                "montage_grammar",
                                montage_grammar_raw,
                                config.montage_grammar_score_weight,
                            ),
                        )
                        decision_score = sum(item.contribution for item in components)
                        total_score = state.score + decision_score
                        if not math.isfinite(total_score):
                            raise StageEditDecodeInputError(
                                "beam expansion produced a non-finite score"
                            )
                        probability_terms = [source_raw, duration_raw]
                        if state.decisions:
                            probability_terms.extend((boundary_cut_score, transition_raw))
                        confidence = math.exp(
                            min(0.0, sum(probability_terms) / len(probability_terms))
                        )
                        recent_sources = (
                            state.recent_sources + (source_index,)
                        )[-config.recent_source_window :]
                        recent_sizes = (
                            state.recent_shot_sizes + (shot_size,)
                        )[-config.recent_shot_size_window :]
                        recent_durations = (
                            state.recent_shot_durations + (duration,)
                        )[-config.recent_shot_size_window :]
                        recent_cuts = tuple(
                            value
                            for value in (
                                state.recent_cut_times
                                + ((start_time,) if state.decisions else ())
                            )
                            if value
                            >= end_time - config.cut_density_window_seconds
                        )
                        recent_density = (
                            len(recent_cuts)
                            * 60.0
                            / config.cut_density_window_seconds
                        )
                        decision = StageEditBeamDecision(
                            shot_index=len(state.decisions),
                            start_candidate_index=start_index,
                            end_candidate_index=end_index,
                            master_start_seconds=start_time,
                            master_end_seconds=end_time,
                            source_index=source_index,
                            source_id=metadata.source_id,
                            shot_size=shot_size,
                            detected_person_count_normalized=(
                                selected_continuity.detected_person_count_normalized
                            ),
                            formation_person_count_normalized=(
                                selected_continuity.formation_person_count_normalized
                            ),
                            formation_pose_coverage=(
                                selected_continuity.formation_pose_coverage
                            ),
                            duration_bucket=duration_bucket,
                            transition_label=transition_label,
                            crop_requested=requested_crop,
                            crop_applied=applied_crop,
                            crop_was_corrected=crop_corrected,
                            decision_confidence=confidence,
                            score_components=components,
                            recent_sources_after=recent_sources,
                            recent_shot_sizes_after=recent_sizes,
                            recent_shot_durations_after=recent_durations,
                            recent_cut_density_per_minute_after=recent_density,
                            source_occupancy_after_seconds=tuple(
                                prospective_occupancy
                            ),
                            section_switch_counts_after=tuple(section_counts),
                            derivative_family_id=candidate_family,
                            derivative_family_occupancy_after_seconds=(
                                prospective_family_occupancy
                            ),
                            editorial_transform=editorial_transform,
                            dynamic_crop_decode=dynamic_crop_decode,
                            crop_contract=crop_contract,
                            section_progress=frame.section_progress,
                            phrase_progress=frame.phrase_progress,
                            section_role=frame.section_role,
                        )
                        new_state = _BeamState(
                            current_index=end_index,
                            current_source=source_index,
                            recent_sources=recent_sources,
                            recent_shot_sizes=recent_sizes,
                            recent_shot_durations=recent_durations,
                            recent_cut_times=recent_cuts,
                            source_occupancy_seconds=tuple(prospective_occupancy),
                            section_switch_counts=tuple(section_counts),
                            decisions=state.decisions + (decision,),
                            score=total_score,
                            component_totals=_add_component_totals(
                                state.component_totals, components
                            ),
                        )
                        beams[end_index].append(new_state)
                        accepted_hypothesis_count += 1
                        endpoint_records[start_index]["accepted_hypothesis_count"] += 1

                        if len(beams[end_index]) > config.beam_width * 2:
                            unique, removed, duplicates = _prune_states(
                                beams[end_index], config.beam_width
                            )
                            if duplicates:
                                rejection_counts["duplicate_partial_timeline"] = (
                                    rejection_counts.get(
                                        "duplicate_partial_timeline", 0
                                    )
                                    + duplicates
                                )
                            if removed > 0:
                                pruned_count += removed
                                endpoint_records[end_index][
                                    "beam_pruned_state_count"
                                ] += removed
                            beams[end_index] = unique

    raw_complete = sorted(beams[terminal_index], key=_state_sort_key)
    endpoint_records[terminal_index]["incoming_state_count"] = len(raw_complete)
    if not raw_complete:
        summary = ", ".join(
            f"{name}={count}" for name, count in sorted(rejection_counts.items())
        )
        raise StageEditNoValidTimelineError(
            "hard constraints eliminated every complete StageEdit timeline"
            + (f": {summary}" if summary else "")
        )

    complete: list[_BeamState] = []
    montage_rejected_count = 0
    for state in raw_complete:
        montage = _evaluate_montage_grammar(
            decisions=state.decisions,
            section_plan=section_plan,
            support_by_section=montage_support_by_section,
            config=config,
        )
        if not bool(montage["passed"]):
            montage_rejected_count += 1
            for failure in montage["failure_codes"]:
                reason = str(failure).partition(":")[2] or str(failure)
                reject(reason)
            continue
        complete.append(state)
    if not complete:
        summary = ", ".join(
            f"{name}={count}" for name, count in sorted(rejection_counts.items())
        )
        raise StageEditNoValidTimelineError(
            "montage grammar eliminated every complete StageEdit timeline"
            + (f": {summary}" if summary else "")
        )

    selected_states: list[_BeamState] = []
    completion_signatures: set[tuple[Any, ...]] = set()
    duplicate_completions = 0
    for state in complete:
        signature = state.signature()
        if signature in completion_signatures:
            duplicate_completions += 1
            continue
        completion_signatures.add(signature)
        selected_states.append(state)
        if len(selected_states) == config.timeline_count:
            break

    checkpoint_path = str(readiness.checkpoint_path)
    timelines: list[StageEditBeamTimeline] = []
    for rank, state in enumerate(selected_states, start=1):
        edit = _build_stage_edit(
            rank=rank,
            state=state,
            sources=normalized_sources,
            source_metadata=normalized_metadata,
            section_plan=section_plan,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            fps=fps,
            reference_source=reference_source,
            output=Path(output),
            checkpoint_path=checkpoint_path,
            source_derivative_metadata=source_derivative_metadata,
            config=config,
        )
        recent_density = (
            len(state.recent_cut_times)
            * 60.0
            / config.cut_density_window_seconds
        )
        timelines.append(
            StageEditBeamTimeline(
                rank=rank,
                score=state.score,
                score_components=dict(state.component_totals),
                decisions=state.decisions,
                edit=edit,
                source_occupancy_seconds=state.source_occupancy_seconds,
                recent_cut_density_per_minute=recent_density,
                montage_grammar=_evaluate_montage_grammar(
                    decisions=state.decisions,
                    section_plan=section_plan,
                    support_by_section=montage_support_by_section,
                    config=config,
                ),
                derivative_family_occupancy_seconds=_family_occupancy(
                    state.source_occupancy_seconds,
                    derivative_family_ids,
                ),
            )
        )

    digest = _input_digest(
        sources=normalized_sources,
        source_metadata=normalized_metadata,
        frames=frames,
        section_plan=section_plan,
        constraints=constraints,
        config=config,
        schema=schema,
        source_derivative_metadata=source_derivative_metadata,
    )
    selected_dynamic_decodes = tuple(
        decision.dynamic_crop_decode
        for timeline in timelines
        for decision in timeline.decisions
        if decision.dynamic_crop_decode is not None
    )
    dynamic_rejection_counts: dict[str, int] = {}
    for decoded in selected_dynamic_decodes:
        for reason in decoded.rejection_reasons:
            dynamic_rejection_counts[reason] = (
                dynamic_rejection_counts.get(reason, 0) + 1
            )
    audit = StageEditDecodeAudit(
        schema_version=DECODER_SCHEMA_VERSION,
        decoder="stage_edit_semimarkov_beam_v5",
        input_sha256=digest,
        readiness=readiness.to_dict(),
        config=asdict(config),
        input_summary={
            "source_count": source_count,
            "transition_authority": transition_authority.to_dict(),
            "decoder_schema_migration": {
                "current": DECODER_SCHEMA_VERSION,
                "supersedes": 7,
                "reason": (
                    "renderer-owned IVE transition mode restricts learned "
                    "metadata to duration-neutral hard_cut with zero transition "
                    "score; renderer group_transition retains effect authority"
                ),
                "legacy_source_counter_aliases_retained": True,
                "automatic_group_requires_verified_family_metadata": True,
            },
            "source_derivative_diversity": (
                source_derivative_metadata.to_dict()
                if source_derivative_metadata is not None
                else {
                    "production_ready": False,
                    "family_ids": list(derivative_family_ids),
                    "family_count": derivative_family_count,
                    "status": "decoder_unit_identity_default",
                }
            ),
            "diversity_semantics": "verified_derivative_family_not_file_source",
            "occupancy_semantics": "aggregate_all_sources_in_derivative_family",
            "candidate_frame_count": len(frames),
            "section_count": section_count,
            "master_start_seconds": section_plan.start_seconds,
            "master_end_seconds": section_plan.end_seconds,
            "master_duration_seconds": master_duration,
            "montage_grammar_contract": MONTAGE_GRAMMAR_CONTRACT,
            "montage_grammar_candidate_support_by_section": [
                dict(value) for value in montage_support_by_section
            ],
            "raw_complete_state_count_before_montage_grammar": len(raw_complete),
            "montage_grammar_rejected_complete_state_count": (
                montage_rejected_count
            ),
            "token_schema_fingerprint": schema.fingerprint,
            "decision_history_owner": STAGE_EDIT_DECISION_HISTORY_OWNER,
            "owned_decision_history_state": [
                "current_source",
                "recent_sources",
                "recent_shot_sizes",
                "recent_shot_durations",
                "recent_cut_times",
                "source_occupancy_seconds",
                "section_switch_counts",
            ],
            "all_frames_have_effective_source": all(
                any(frame.effective_source_mask) for frame in frames
            ),
            "crop_contract": (
                DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT
                if frames[0].dynamic_crop_v2 is not None
                else "legacy_static_crop_head_v1"
            ),
            "selected_dynamic_crop_attempt_count": len(selected_dynamic_decodes),
            "selected_dynamic_crop_eligible_count": sum(
                value.eligible for value in selected_dynamic_decodes
            ),
            "selected_dynamic_crop_applied_keyframe_count": sum(
                value.applied_keyframe_count for value in selected_dynamic_decodes
            ),
            "selected_dynamic_crop_rejection_reasons": dict(
                sorted(dynamic_rejection_counts.items())
            ),
            "transition_render_policy": "metadata_preserved_rendered_as_hard_cut",
        },
        endpoint_audit=tuple(endpoint_records),
        constraint_rejection_counts=dict(sorted(rejection_counts.items())),
        transition_hypothesis_count=transition_hypothesis_count,
        accepted_hypothesis_count=accepted_hypothesis_count,
        beam_pruned_state_count=pruned_count,
        complete_state_count=len(complete),
        duplicate_completion_count=duplicate_completions,
        requested_timeline_count=config.timeline_count,
        returned_timeline_count=len(timelines),
        request_fulfilled=len(timelines) == config.timeline_count,
    )
    return StageEditSemiMarkovResult(tuple(timelines), audit)


__all__ = [
    "DECODER_SCHEMA_VERSION",
    "MONTAGE_GRAMMAR_CONTRACT",
    "StageEditBeamDecision",
    "StageEditBeamTimeline",
    "StageEditCheckpointNotReadyError",
    "StageEditDecodeAudit",
    "StageEditDecodeInputError",
    "StageEditHeadFrame",
    "StageEditNoValidTimelineError",
    "StageEditScoreComponent",
    "StageEditSectionConstraint",
    "StageEditSemiMarkovConfig",
    "StageEditSemiMarkovError",
    "StageEditSemiMarkovResult",
    "StageEditSourceContinuity",
    "StageEditSourceMetadata",
    "decode_stage_edit_semimarkov_beam",
    "stage_edit_boundary_continuity_score",
]
