"""Deterministic song-section directives for stage-mix editing.

This module is intentionally an *acoustic-role estimator*, not a claim that a
spectral-flux envelope alone can name a song's lyrical structure.  It turns the
rhythm analysis already produced by the engine into honest long-context tokens
and bounded editing budgets.  A learned StageEdit policy can later replace the
role estimates while preserving the same serialized contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from .rhythm import PhraseCandidate, RhythmAnalysis


ANALYSIS_BASIS = "spectral_flux_phrase_role_section_director_v1"


class SectionRole(str, Enum):
    """Editing roles inferred from energy and position, not semantic labels."""

    INTRO = "intro"
    VERSE_LIKE = "verse_like"
    BUILD = "build"
    CHORUS_LIKE = "chorus_like"
    BRIDGE_LIKE = "bridge_like"
    OUTRO = "outro"


_DEFAULT_BUDGETS: dict[SectionRole, tuple[float, float, float, int]] = {
    # source switches/minute, virtual-camera budget, roll events/minute,
    # preferred source-hold beats.  These are initial deterministic priors;
    # reference or learned values may override them within the hard clamps.
    SectionRole.INTRO: (12.5, 0.24, 0.00, 4),
    SectionRole.VERSE_LIKE: (13.5, 0.34, 0.00, 4),
    SectionRole.BUILD: (15.5, 0.54, 0.12, 3),
    SectionRole.CHORUS_LIKE: (17.0, 0.82, 0.35, 2),
    SectionRole.BRIDGE_LIKE: (12.8, 0.30, 0.00, 4),
    SectionRole.OUTRO: (10.5, 0.18, 0.00, 6),
}


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True, slots=True)
class SectionDirective:
    section_index: int
    role: SectionRole
    start_seconds: float
    end_seconds: float
    energy_score: float
    confidence: float
    phrase_count: int
    preferred_source_hold_beats: int
    target_source_switches_per_minute: float
    camera_motion_budget: float
    target_roll_events_per_minute: float
    establish_with_wide_shot: bool
    protect_last_beat_from_extra_cut: bool

    def __post_init__(self) -> None:
        if self.section_index < 0:
            raise ValueError("section_index must be non-negative")
        if (
            not math.isfinite(self.start_seconds)
            or not math.isfinite(self.end_seconds)
            or self.start_seconds < 0.0
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("section span must be positive and finite")
        for name in ("energy_score", "confidence", "camera_motion_budget"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in 0...1")
        if self.phrase_count < 1 or self.preferred_source_hold_beats < 1:
            raise ValueError("section phrase/hold counts must be positive")
        if not 8.0 <= self.target_source_switches_per_minute <= 22.0:
            raise ValueError("source-switch budget is outside its safety clamp")
        if not 0.0 <= self.target_roll_events_per_minute <= 0.75:
            raise ValueError("roll budget is outside its subtle safety clamp")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_index": self.section_index,
            "role": self.role.value,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "energy_score": self.energy_score,
            "confidence": self.confidence,
            "phrase_count": self.phrase_count,
            "preferred_source_hold_beats": self.preferred_source_hold_beats,
            "target_source_switches_per_minute": (
                self.target_source_switches_per_minute
            ),
            "camera_motion_budget": self.camera_motion_budget,
            "target_roll_events_per_minute": self.target_roll_events_per_minute,
            "establish_with_wide_shot": self.establish_with_wide_shot,
            "protect_last_beat_from_extra_cut": (
                self.protect_last_beat_from_extra_cut
            ),
        }


@dataclass(frozen=True, slots=True)
class SectionPlan:
    sections: tuple[SectionDirective, ...]
    analysis_basis: str = ANALYSIS_BASIS
    role_semantics: str = (
        "acoustic_editing_roles_not_ground_truth_verse_or_chorus_labels"
    )

    def __post_init__(self) -> None:
        if not self.sections:
            raise ValueError("section plan needs at least one section")
        for index, section in enumerate(self.sections):
            if section.section_index != index:
                raise ValueError("section indices must be contiguous")
            if index and not math.isclose(
                self.sections[index - 1].end_seconds,
                section.start_seconds,
                abs_tol=1e-6,
            ):
                raise ValueError("section plan must be contiguous")

    @property
    def start_seconds(self) -> float:
        return self.sections[0].start_seconds

    @property
    def end_seconds(self) -> float:
        return self.sections[-1].end_seconds

    def directive_at(self, master_seconds: float) -> SectionDirective:
        value = _finite(master_seconds, "master_seconds")
        if value < self.start_seconds - 1e-6 or value > self.end_seconds + 1e-6:
            raise ValueError("master time is outside the section plan")
        for section in self.sections:
            if value < section.end_seconds - 1e-9:
                return section
        return self.sections[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": self.analysis_basis,
            "role_semantics": self.role_semantics,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "section_count": len(self.sections),
            "sections": [section.to_dict() for section in self.sections],
        }


@dataclass(frozen=True, slots=True)
class _PhraseRole:
    start: float
    end: float
    energy: float
    confidence: float
    role: SectionRole


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def _energy_for_span(
    rhythm: RhythmAnalysis,
    start_seconds: float,
    end_seconds: float,
) -> float:
    envelope = np.asarray(rhythm.onset_envelope.values, dtype=np.float64)
    if envelope.size == 0:
        return 0.0
    frame_rate = rhythm.onset_envelope.frame_rate
    start = max(0, min(envelope.size - 1, int(math.floor(start_seconds * frame_rate))))
    end = max(start + 1, min(envelope.size, int(math.ceil(end_seconds * frame_rate))))
    values = envelope[start:end]
    global_p95 = max(float(np.quantile(envelope, 0.95)), 1e-9)
    local_mean = float(np.mean(values)) / global_p95
    local_p85 = float(np.quantile(values, 0.85)) / global_p95
    return _unit(0.58 * local_mean + 0.42 * local_p85)


def _primary_phrases(
    rhythm: RhythmAnalysis,
    start_seconds: float,
    end_seconds: float,
) -> tuple[PhraseCandidate, ...]:
    contained = tuple(
        phrase
        for phrase in rhythm.phrase_candidates
        if phrase.end_seconds > start_seconds + 1e-6
        and phrase.start_seconds < end_seconds - 1e-6
    )
    if not contained:
        return ()
    largest = max(item.beat_count for item in contained)
    return tuple(item for item in contained if item.beat_count == largest)


def _profile_value(
    profile: Mapping[str, Any] | None,
    key: str,
    role: SectionRole,
    fallback: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if profile is None:
        return fallback
    values = profile.get(key)
    if not isinstance(values, Mapping) or role.value not in values:
        return fallback
    return min(maximum, max(minimum, _finite(values[role.value], key)))


def _directive_budget(
    role: SectionRole,
    reference_profile: Mapping[str, Any] | None,
) -> tuple[float, float, float, int]:
    switches, motion, roll, hold = _DEFAULT_BUDGETS[role]
    switches = _profile_value(
        reference_profile,
        "target_source_switches_per_minute_by_role",
        role,
        switches,
        minimum=8.0,
        maximum=22.0,
    )
    motion = _profile_value(
        reference_profile,
        "camera_motion_budget_by_role",
        role,
        motion,
        minimum=0.0,
        maximum=1.0,
    )
    roll = _profile_value(
        reference_profile,
        "target_roll_events_per_minute_by_role",
        role,
        roll,
        minimum=0.0,
        maximum=0.75,
    )
    return switches, motion, roll, hold


def build_section_plan(
    rhythm: RhythmAnalysis,
    *,
    start_seconds: float,
    end_seconds: float,
    reference_profile: Mapping[str, Any] | None = None,
) -> SectionPlan:
    """Infer deterministic acoustic editing roles and section-local budgets."""

    start = _finite(start_seconds, "start_seconds")
    end = _finite(end_seconds, "end_seconds")
    if start < 0.0 or end <= start:
        raise ValueError("section analysis range must be positive")
    phrases = _primary_phrases(rhythm, start, end)
    if not phrases:
        switches, motion, roll, hold = _directive_budget(
            SectionRole.VERSE_LIKE,
            reference_profile,
        )
        return SectionPlan(
            (
                SectionDirective(
                    section_index=0,
                    role=SectionRole.VERSE_LIKE,
                    start_seconds=start,
                    end_seconds=end,
                    energy_score=_energy_for_span(rhythm, start, end),
                    confidence=0.20,
                    phrase_count=1,
                    preferred_source_hold_beats=hold,
                    target_source_switches_per_minute=switches,
                    camera_motion_budget=motion,
                    target_roll_events_per_minute=roll,
                    establish_with_wide_shot=True,
                    protect_last_beat_from_extra_cut=True,
                ),
            )
        )

    spans: list[tuple[float, float, float, float]] = []
    cursor = start
    for phrase in phrases:
        phrase_start = min(end, max(start, phrase.start_seconds))
        phrase_end = min(end, max(start, phrase.end_seconds))
        if phrase_end <= cursor + 1e-6:
            continue
        if phrase_start > cursor + 1e-6:
            spans.append(
                (
                    cursor,
                    phrase_start,
                    _energy_for_span(rhythm, cursor, phrase_start),
                    0.30,
                )
            )
        effective_start = max(cursor, phrase_start)
        spans.append(
            (
                effective_start,
                phrase_end,
                _energy_for_span(rhythm, effective_start, phrase_end),
                _unit(0.35 + 0.65 * phrase.confidence),
            )
        )
        cursor = phrase_end
    if cursor < end - 1e-6:
        spans.append((cursor, end, _energy_for_span(rhythm, cursor, end), 0.30))
    if not spans:
        raise ValueError("rhythm phrases do not overlap the requested range")

    energies = [item[2] for item in spans]
    low = _quantile(energies, 0.35)
    high = _quantile(energies, 0.70)
    roles = [SectionRole.VERSE_LIKE] * len(spans)
    for index, (_left, _right, energy, _confidence) in enumerate(spans):
        position = index / max(1, len(spans) - 1)
        if index == 0:
            roles[index] = SectionRole.INTRO
        elif index == len(spans) - 1:
            roles[index] = SectionRole.OUTRO
        elif energy >= high - 1e-9:
            roles[index] = SectionRole.CHORUS_LIKE
        elif (
            position >= 0.52
            and energy <= low + 1e-9
            and any(role is SectionRole.CHORUS_LIKE for role in roles[:index])
        ):
            roles[index] = SectionRole.BRIDGE_LIKE
    for index in range(1, len(spans) - 1):
        if roles[index] is not SectionRole.VERSE_LIKE:
            continue
        current = spans[index][2]
        previous = spans[index - 1][2]
        next_role = roles[index + 1]
        if next_role is SectionRole.CHORUS_LIKE and current >= previous - 0.03:
            roles[index] = SectionRole.BUILD

    raw = [
        _PhraseRole(left, right, energy, confidence, roles[index])
        for index, (left, right, energy, confidence) in enumerate(spans)
    ]
    merged: list[list[_PhraseRole]] = []
    for item in raw:
        if merged and merged[-1][-1].role is item.role:
            merged[-1].append(item)
        else:
            merged.append([item])

    directives: list[SectionDirective] = []
    for index, group in enumerate(merged):
        role = group[0].role
        switches, motion, roll, hold = _directive_budget(role, reference_profile)
        directives.append(
            SectionDirective(
                section_index=index,
                role=role,
                start_seconds=group[0].start,
                end_seconds=group[-1].end,
                energy_score=_unit(fmean(item.energy for item in group)),
                confidence=_unit(fmean(item.confidence for item in group)),
                phrase_count=len(group),
                preferred_source_hold_beats=hold,
                target_source_switches_per_minute=switches,
                camera_motion_budget=motion,
                target_roll_events_per_minute=roll,
                establish_with_wide_shot=role
                in {SectionRole.INTRO, SectionRole.CHORUS_LIKE},
                protect_last_beat_from_extra_cut=role
                in {
                    SectionRole.INTRO,
                    SectionRole.BRIDGE_LIKE,
                    SectionRole.OUTRO,
                },
            )
        )
    # Floating phrase endpoints are rewritten to one exact contiguous chain.
    normalized: list[SectionDirective] = []
    cursor = start
    for index, item in enumerate(directives):
        right = end if index == len(directives) - 1 else item.end_seconds
        normalized.append(
            SectionDirective(
                section_index=index,
                role=item.role,
                start_seconds=cursor,
                end_seconds=right,
                energy_score=item.energy_score,
                confidence=item.confidence,
                phrase_count=item.phrase_count,
                preferred_source_hold_beats=item.preferred_source_hold_beats,
                target_source_switches_per_minute=(
                    item.target_source_switches_per_minute
                ),
                camera_motion_budget=item.camera_motion_budget,
                target_roll_events_per_minute=item.target_roll_events_per_minute,
                establish_with_wide_shot=item.establish_with_wide_shot,
                protect_last_beat_from_extra_cut=(
                    item.protect_last_beat_from_extra_cut
                ),
            )
        )
        cursor = right
    return SectionPlan(tuple(normalized))


__all__ = [
    "ANALYSIS_BASIS",
    "SectionDirective",
    "SectionPlan",
    "SectionRole",
    "build_section_plan",
]
