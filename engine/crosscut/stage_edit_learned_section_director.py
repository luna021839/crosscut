"""Section-local directing budgets derived from verified StageEdit heads.

The acoustic section plan remains the source of contiguous song boundaries.
This adapter replaces its hand-authored *editing budgets* with bounded evidence
from the production Transformer's cut, source-selection, and Dynamic Crop v2
heads.  It therefore cannot invent semantic verse/chorus labels, and it cannot
run from random weights or a held-out-tuned threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .section_director import SectionDirective, SectionPlan, SectionRole
from .stage_edit_semimarkov_beam import StageEditHeadFrame
from .stage_edit_transformer import PRODUCTION_TRAINING_REPORT_FILENAME


SCHEMA_VERSION = 2
CONTRACT = (
    "validation_frozen_stageedit_heads_section_budget_"
    "profile_chorus_wide_anchor_allocation_v2"
)
CHORUS_WIDE_ANCHOR_ALLOCATION_BASIS = (
    "phrase_count_then_duration_then_energy_then_learned_wide_support_"
    "then_stable_section_index_v1"
)


class LearnedSectionDirectorError(RuntimeError):
    """A production Section Director contract could not be satisfied."""


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    positive = math.exp(value)
    return positive / (1.0 + positive)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _threshold_logit(probability: float) -> float:
    bounded = _clamp(float(probability), 1.0e-6, 1.0 - 1.0e-6)
    return math.log(bounded / (1.0 - bounded))


def _top_source_shot_size(frame: StageEditHeadFrame) -> str | None:
    if not frame.source_shot_sizes:
        return None
    eligible = [
        index for index, allowed in enumerate(frame.effective_source_mask) if allowed
    ]
    if not eligible:
        return None
    selected = max(eligible, key=lambda index: frame.source_logits[index])
    value = frame.source_shot_sizes[selected].strip().lower()
    return value or None


def _dynamic_motion_probability(frame: StageEditHeadFrame) -> float | None:
    prediction = frame.dynamic_crop_v2
    return _sigmoid(prediction.motion_logit) if prediction is not None else None


def _dynamic_roll_support(frame: StageEditHeadFrame) -> float | None:
    prediction = frame.dynamic_crop_v2
    if prediction is None:
        return None
    supports = [
        _sigmoid(presence) * _clamp(abs(values[3]) / 1.60, 0.0, 1.0)
        for presence, values in zip(
            prediction.presence_logits,
            prediction.values,
            strict=True,
        )
    ]
    return max(supports, default=0.0)


@dataclass(frozen=True, slots=True)
class LearnedSectionBudgetEvidence:
    section_index: int
    frame_count: int
    mean_threshold_centered_cut_probability: float
    learned_switch_rate_multiplier: float
    base_source_switches_per_minute: float
    learned_source_switches_per_minute: float
    mean_dynamic_motion_probability: float | None
    base_camera_motion_budget: float
    learned_camera_motion_budget: float
    mean_dynamic_roll_support: float | None
    base_roll_events_per_minute: float
    learned_roll_events_per_minute: float
    base_preferred_source_hold_beats: int
    learned_preferred_source_hold_beats: int
    learned_top_source_wide_support: float | None
    chorus_wide_anchor_selected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class LearnedSectionDirectorResult:
    section_plan: SectionPlan
    evidence: tuple[LearnedSectionBudgetEvidence, ...]
    cut_probability_threshold: float
    cut_threshold_freeze_sha256: str
    checkpoint_id: str
    training_report_path: str
    training_report_sha256: str
    chorus_wide_anchor_requested_fraction: float
    chorus_section_count: int
    chorus_wide_anchor_allocated_count: int
    chorus_wide_anchor_allocated_section_indices: tuple[int, ...]
    chorus_wide_anchor_allocation_basis: str = (
        CHORUS_WIDE_ANCHOR_ALLOCATION_BASIS
    )
    contract: str = CONTRACT
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.evidence) != len(self.section_plan.sections):
            raise ValueError("learned Section Director evidence is incomplete")
        if tuple(item.section_index for item in self.evidence) != tuple(
            item.section_index for item in self.section_plan.sections
        ):
            raise ValueError("learned Section Director evidence is misaligned")
        if not 0.0 <= self.cut_probability_threshold <= 1.0:
            raise ValueError("learned Section Director threshold is invalid")
        for digest in (
            self.cut_threshold_freeze_sha256,
            self.training_report_sha256,
        ):
            if len(digest) != 64:
                raise ValueError("learned Section Director identity is not SHA-256")
        fraction = self.chorus_wide_anchor_requested_fraction
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("chorus wide-anchor fraction is invalid")
        chorus_indices = tuple(
            item.section_index
            for item in self.section_plan.sections
            if item.role is SectionRole.CHORUS_LIKE
        )
        if self.chorus_section_count != len(chorus_indices):
            raise ValueError("chorus wide-anchor section count is inconsistent")
        expected_count = math.ceil(fraction * len(chorus_indices))
        allocated = self.chorus_wide_anchor_allocated_section_indices
        if (
            self.chorus_wide_anchor_allocated_count != expected_count
            or len(allocated) != expected_count
            or tuple(sorted(set(allocated))) != allocated
            or not set(allocated).issubset(chorus_indices)
        ):
            raise ValueError("chorus wide-anchor allocation is inconsistent")
        selected_from_evidence = tuple(
            item.section_index
            for item in self.evidence
            if item.chorus_wide_anchor_selected
        )
        if selected_from_evidence != allocated:
            raise ValueError("chorus wide-anchor evidence is inconsistent")
        selected_from_plan = tuple(
            item.section_index
            for item in self.section_plan.sections
            if item.role is SectionRole.CHORUS_LIKE
            and item.establish_with_wide_shot
        )
        if selected_from_plan != allocated:
            raise ValueError("chorus wide-anchor section plan is inconsistent")
        if any(
            item.establish_with_wide_shot
            for item in self.section_plan.sections
            if item.role not in {SectionRole.INTRO, SectionRole.CHORUS_LIKE}
        ):
            raise ValueError("non-chorus section was promoted to a wide anchor")
        if self.chorus_wide_anchor_allocation_basis != (
            CHORUS_WIDE_ANCHOR_ALLOCATION_BASIS
        ):
            raise ValueError("chorus wide-anchor allocation basis is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "checkpoint_id": self.checkpoint_id,
            "training_report_path": self.training_report_path,
            "training_report_sha256": self.training_report_sha256,
            "cut_probability_threshold": self.cut_probability_threshold,
            "cut_threshold_freeze_sha256": self.cut_threshold_freeze_sha256,
            "chorus_wide_anchor_allocation": {
                "requested_fraction": (
                    self.chorus_wide_anchor_requested_fraction
                ),
                "chorus_section_count": self.chorus_section_count,
                "allocated_count": self.chorus_wide_anchor_allocated_count,
                "allocated_section_indices": list(
                    self.chorus_wide_anchor_allocated_section_indices
                ),
                "allocation_basis": self.chorus_wide_anchor_allocation_basis,
            },
            "section_plan": self.section_plan.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
        }


def adapt_section_plan_from_stage_edit_heads(
    section_plan: SectionPlan,
    head_frames: Sequence[StageEditHeadFrame],
    *,
    chorus_wide_anchor_fraction: float,
    cut_probability_threshold: float,
    cut_threshold_freeze_sha256: str,
    checkpoint_id: str,
    training_report_path: str,
    training_report_sha256: str,
) -> LearnedSectionDirectorResult:
    """Bound section budgets using only validation-frozen learned outputs."""

    frames = tuple(head_frames)
    if not frames:
        raise LearnedSectionDirectorError("learned Section Director needs head frames")
    if isinstance(chorus_wide_anchor_fraction, bool):
        raise LearnedSectionDirectorError("chorus wide-anchor fraction is invalid")
    try:
        chorus_fraction = float(chorus_wide_anchor_fraction)
    except (TypeError, ValueError) as exc:
        raise LearnedSectionDirectorError(
            "chorus wide-anchor fraction is invalid"
        ) from exc
    if not math.isfinite(chorus_fraction) or not 0.0 < chorus_fraction <= 1.0:
        raise LearnedSectionDirectorError(
            "chorus wide-anchor fraction must be finite and lie in (0, 1]"
        )
    threshold = float(cut_probability_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise LearnedSectionDirectorError("cut probability threshold is invalid")
    threshold_logit = _threshold_logit(threshold)
    by_section: list[list[StageEditHeadFrame]] = [
        [] for _ in section_plan.sections
    ]
    for frame in frames:
        bounded = min(
            section_plan.end_seconds,
            max(section_plan.start_seconds, frame.master_time_seconds),
        )
        if math.isclose(bounded, section_plan.end_seconds, abs_tol=1.0e-9):
            bounded = math.nextafter(section_plan.end_seconds, -math.inf)
        directive = section_plan.directive_at(bounded)
        by_section[directive.section_index].append(frame)
    if any(not values for values in by_section):
        missing = [index for index, values in enumerate(by_section) if not values]
        raise LearnedSectionDirectorError(
            "head-frame lattice does not cover every acoustic section: "
            + ", ".join(str(value) for value in missing)
        )

    wide_support_by_section: list[float | None] = []
    for section_frames in by_section:
        shot_sizes = [
            value
            for frame in section_frames
            if (value := _top_source_shot_size(frame)) is not None
        ]
        wide_support_by_section.append(
            (
                sum(
                    value in {"wide", "long", "full", "full_body", "group"}
                    for value in shot_sizes
                )
                / len(shot_sizes)
                if shot_sizes
                else None
            )
        )

    chorus_sections = tuple(
        section
        for section in section_plan.sections
        if section.role is SectionRole.CHORUS_LIKE
    )
    chorus_allocation_count = math.ceil(
        chorus_fraction * len(chorus_sections)
    )
    ranked_chorus_sections = sorted(
        chorus_sections,
        key=lambda section: (
            -section.phrase_count,
            -section.duration_seconds,
            -section.energy_score,
            -(
                wide_support_by_section[section.section_index]
                if wide_support_by_section[section.section_index] is not None
                else -1.0
            ),
            section.section_index,
        ),
    )
    allocated_chorus_indices = tuple(
        sorted(
            section.section_index
            for section in ranked_chorus_sections[:chorus_allocation_count]
        )
    )
    allocated_chorus_index_set = frozenset(allocated_chorus_indices)

    learned: list[SectionDirective] = []
    evidence: list[LearnedSectionBudgetEvidence] = []
    for base, section_frames, wide_support in zip(
        section_plan.sections,
        by_section,
        wide_support_by_section,
        strict=True,
    ):
        cut_strength = fmean(
            _sigmoid(frame.cut_logit - threshold_logit)
            for frame in section_frames
        )
        rate_multiplier = _clamp(0.72 + 0.56 * cut_strength, 0.72, 1.28)
        switch_rate = _clamp(
            base.target_source_switches_per_minute * rate_multiplier,
            8.0,
            22.0,
        )

        motion_values = [
            value
            for frame in section_frames
            if (value := _dynamic_motion_probability(frame)) is not None
        ]
        mean_motion = fmean(motion_values) if motion_values else None
        motion_budget = (
            base.camera_motion_budget
            if mean_motion is None
            else _clamp(0.55 * base.camera_motion_budget + 0.45 * mean_motion, 0.0, 1.0)
        )

        roll_values = [
            value
            for frame in section_frames
            if (value := _dynamic_roll_support(frame)) is not None
        ]
        mean_roll = fmean(roll_values) if roll_values else None
        roll_rate = (
            base.target_roll_events_per_minute
            if mean_roll is None
            else _clamp(
                0.65 * base.target_roll_events_per_minute + 0.35 * 0.75 * mean_roll,
                0.0,
                0.75,
            )
        )

        hold = max(
            1,
            min(
                8,
                int(
                    round(
                        base.preferred_source_hold_beats
                        * (1.25 - 0.50 * cut_strength)
                    )
                ),
            ),
        )
        chorus_wide_anchor_selected = (
            base.section_index in allocated_chorus_index_set
        )
        establish_with_wide_shot = (
            base.establish_with_wide_shot
            if base.role is SectionRole.INTRO
            else chorus_wide_anchor_selected
            if base.role is SectionRole.CHORUS_LIKE
            else False
        )

        learned.append(
            SectionDirective(
                section_index=base.section_index,
                role=base.role,
                start_seconds=base.start_seconds,
                end_seconds=base.end_seconds,
                energy_score=base.energy_score,
                confidence=base.confidence,
                phrase_count=base.phrase_count,
                preferred_source_hold_beats=hold,
                target_source_switches_per_minute=switch_rate,
                camera_motion_budget=motion_budget,
                target_roll_events_per_minute=roll_rate,
                establish_with_wide_shot=establish_with_wide_shot,
                protect_last_beat_from_extra_cut=(
                    base.protect_last_beat_from_extra_cut or cut_strength < 0.35
                ),
            )
        )
        evidence.append(
            LearnedSectionBudgetEvidence(
                section_index=base.section_index,
                frame_count=len(section_frames),
                mean_threshold_centered_cut_probability=cut_strength,
                learned_switch_rate_multiplier=rate_multiplier,
                base_source_switches_per_minute=base.target_source_switches_per_minute,
                learned_source_switches_per_minute=switch_rate,
                mean_dynamic_motion_probability=mean_motion,
                base_camera_motion_budget=base.camera_motion_budget,
                learned_camera_motion_budget=motion_budget,
                mean_dynamic_roll_support=mean_roll,
                base_roll_events_per_minute=base.target_roll_events_per_minute,
                learned_roll_events_per_minute=roll_rate,
                base_preferred_source_hold_beats=base.preferred_source_hold_beats,
                learned_preferred_source_hold_beats=hold,
                learned_top_source_wide_support=wide_support,
                chorus_wide_anchor_selected=chorus_wide_anchor_selected,
            )
        )

    adapted = SectionPlan(
        tuple(learned),
        analysis_basis=f"{section_plan.analysis_basis}+{CONTRACT}",
        role_semantics=(
            section_plan.role_semantics
            + "; boundaries acoustic; editing budgets validation-frozen learned "
            "heads; chorus wide anchors profile-fraction allocated"
        ),
    )
    return LearnedSectionDirectorResult(
        section_plan=adapted,
        evidence=tuple(evidence),
        cut_probability_threshold=threshold,
        cut_threshold_freeze_sha256=cut_threshold_freeze_sha256,
        checkpoint_id=str(checkpoint_id),
        training_report_path=str(training_report_path),
        training_report_sha256=training_report_sha256,
        chorus_wide_anchor_requested_fraction=chorus_fraction,
        chorus_section_count=len(chorus_sections),
        chorus_wide_anchor_allocated_count=chorus_allocation_count,
        chorus_wide_anchor_allocated_section_indices=(
            allocated_chorus_indices
        ),
    )


def build_learned_section_plan_from_checkpoint(
    section_plan: SectionPlan,
    head_frames: Sequence[StageEditHeadFrame],
    *,
    checkpoint_directory: Path,
    chorus_wide_anchor_fraction: float,
) -> LearnedSectionDirectorResult:
    """Load the readiness-validated report and apply its frozen cut threshold."""

    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    report_path = checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearnedSectionDirectorError(
            f"production training report cannot be read: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LearnedSectionDirectorError("production training report is not an object")
    freeze = payload.get("cut_threshold_freeze")
    checkpoint_evidence = payload.get("checkpoint")
    if not isinstance(freeze, Mapping) or not isinstance(checkpoint_evidence, Mapping):
        raise LearnedSectionDirectorError(
            "production report lacks checkpoint or cut-threshold freeze evidence"
        )
    unsigned = dict(freeze)
    observed_freeze_sha256 = str(unsigned.pop("freeze_sha256", ""))
    if observed_freeze_sha256 != _canonical_sha256(unsigned):
        raise LearnedSectionDirectorError("cut-threshold freeze digest is invalid")
    threshold = freeze.get("selected_probability_threshold")
    checkpoint_id = str(checkpoint_evidence.get("checkpoint_id") or "").strip()
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
        or not checkpoint_id
        or freeze.get("selected_checkpoint_id") != checkpoint_id
        or freeze.get("heldout_labels_consumed") is not False
        or freeze.get("heldout_probabilities_consumed") is not False
        or freeze.get("selected_before_heldout_forward") is not True
    ):
        raise LearnedSectionDirectorError(
            "cut-threshold freeze is not validation-only production evidence"
        )
    return adapt_section_plan_from_stage_edit_heads(
        section_plan,
        head_frames,
        chorus_wide_anchor_fraction=chorus_wide_anchor_fraction,
        cut_probability_threshold=float(threshold),
        cut_threshold_freeze_sha256=observed_freeze_sha256,
        checkpoint_id=checkpoint_id,
        training_report_path=str(report_path),
        training_report_sha256=_file_sha256(report_path),
    )


__all__ = [
    "CONTRACT",
    "SCHEMA_VERSION",
    "LearnedSectionBudgetEvidence",
    "LearnedSectionDirectorError",
    "LearnedSectionDirectorResult",
    "adapt_section_plan_from_stage_edit_heads",
    "build_learned_section_plan_from_checkpoint",
]
