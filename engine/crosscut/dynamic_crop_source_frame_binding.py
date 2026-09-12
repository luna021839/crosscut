"""Preserve the exact source frames on which a crop teacher was measured.

This is target-side provenance, not a model feature or a retiming prediction.
Geometry simplification must not erase it: even a static crop can accompany
a varying source phase, including a native camera cut one frame away.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .reference_evidence_builder import ReferenceSourceClock


SOURCE_FRAME_BINDING_CONTRACT = "dynamic_crop_exact_source_frame_teacher_binding_v1"


@dataclass(frozen=True, slots=True)
class CropSourceFrameObservation:
    time_within_shot_seconds: float
    master_seconds: float
    canonical_source_seconds: float
    source_frame_index: int

    def __post_init__(self) -> None:
        for name in ("time_within_shot_seconds", "master_seconds", "canonical_source_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.time_within_shot_seconds < 0:
            raise ValueError("crop observation cannot precede its shot")
        if isinstance(self.source_frame_index, bool) or not isinstance(self.source_frame_index, int) or self.source_frame_index < 0:
            raise ValueError("crop teacher source frame must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class DynamicCropSourceFrameBinding:
    source_id: str
    source_media_sha256: str
    source_frame_rate: float
    observations: tuple[CropSourceFrameObservation, ...]
    contract: str = SOURCE_FRAME_BINDING_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("crop binding needs a source identity")
        if not isinstance(self.source_media_sha256, str) or len(self.source_media_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_media_sha256):
            raise ValueError("crop binding needs a source media SHA-256")
        rate = self.source_frame_rate
        if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("crop binding needs a positive source frame rate")
        observations = tuple(self.observations)
        if not observations or any(not isinstance(o, CropSourceFrameObservation) for o in observations):
            raise ValueError("crop binding needs exact source observations")
        if observations[0].time_within_shot_seconds != 0:
            raise ValueError("crop binding must retain its real time-zero observation")
        start = observations[0].master_seconds
        for observation in observations:
            if not math.isclose(observation.master_seconds - start, observation.time_within_shot_seconds, abs_tol=1e-6, rel_tol=0):
                raise ValueError("crop binding shot-local and master clocks differ")
        if any(b.time_within_shot_seconds <= a.time_within_shot_seconds for a, b in zip(observations, observations[1:])):
            raise ValueError("crop binding observations must have increasing master time")
        # Do not require increasing selected frames: reverse-match labels may
        # repeat or move backward. Preserve that evidence instead of silently
        # repairing it into an apparently renderable source-time trajectory.
        if self.contract != SOURCE_FRAME_BINDING_CONTRACT:
            raise ValueError("crop source binding contract differs")
        object.__setattr__(self, "observations", observations)

    @property
    def canonical_frame_indices(self) -> tuple[int, ...]:
        # Same ties-to-even rule as production feature snapshot extraction.
        return tuple(int(round(o.canonical_source_seconds * self.source_frame_rate)) for o in self.observations)

    @property
    def frame_deltas(self) -> tuple[int, ...]:
        return tuple(o.source_frame_index - index for o, index in zip(self.observations, self.canonical_frame_indices))

    @property
    def requires_source_frame_reconciliation(self) -> bool:
        return any(self.frame_deltas)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "canonical_frame_indices": list(self.canonical_frame_indices),
                "frame_deltas": list(self.frame_deltas), "model_input": False,
                "source_frame_reconciliation_required": self.requires_source_frame_reconciliation}


def source_frame_binding_from_validated_crop_payload(payload: Mapping[str, Any]) -> DynamicCropSourceFrameBinding:
    """Called only after the immutable crop artifact validator has succeeded."""
    source = payload["provenance"]["source_media"]
    clock = ReferenceSourceClock.from_mapping(source["clock"])
    rate = source["frame_rate"]
    observations = []
    for frame in payload["supervision"]["renderer_keyframes"]:
        if not math.isclose(frame["source_frame_seconds"], frame["source_frame_index"] / rate, abs_tol=1e-9, rel_tol=0):
            raise ValueError("crop source frame index and timestamp differ")
        observations.append(CropSourceFrameObservation(
            time_within_shot_seconds=frame["time_within_shot_seconds"],
            master_seconds=frame["master_seconds"],
            canonical_source_seconds=clock.master_to_source(frame["master_seconds"]),
            source_frame_index=frame["source_frame_index"],
        ))
    return DynamicCropSourceFrameBinding(payload["shot"]["source_id"], source["media_sha256"], rate, tuple(observations))
