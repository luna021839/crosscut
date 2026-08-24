"""Target-only adapter for crop-v2 variable renderer keyframes.

The existing StageEdit crop head predicts one static four-vector.  Feeding a
dynamic track through that head would silently erase the human editor's move.
This adapter therefore exposes explicit keyframes plus start/end views for a
future sequence/trajectory head and always marks the legacy static head target
as unavailable.  Match scores, local-search ranks, clock offsets, homographies,
and pixel-review diagnostics never become model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from .reference_dynamic_crop_supervision import (
    DYNAMIC_CROP_TARGET_CONTRACT,
    load_reference_dynamic_crop_supervision,
    validate_reference_dynamic_crop_supervision_payload,
)


ADAPTER_CONTRACT = "stage_edit_dynamic_crop_target_only_adapter_v1"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class DynamicCropKeyframeTarget:
    time_within_shot_seconds: float
    zoom: float
    center_x: float
    center_y: float
    roll_degrees: float

    def to_dict(self) -> dict[str, float]:
        return {
            "time_within_shot_seconds": self.time_within_shot_seconds,
            "zoom": self.zoom,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "roll_degrees": self.roll_degrees,
        }


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropTarget:
    shot_index: int
    source_id: str
    shot_duration_seconds: float
    motion_label: str
    motion_label_loss_masked: bool
    keyframes: tuple[DynamicCropKeyframeTarget, ...]
    artifact_content_sha256: str
    build_fingerprint_sha256: str

    @property
    def start_keyframe(self) -> DynamicCropKeyframeTarget:
        return self.keyframes[0]

    @property
    def end_keyframe(self) -> DynamicCropKeyframeTarget:
        return self.keyframes[-1]

    @property
    def legacy_static_crop_head_target(self) -> None:
        """crop-v2 is deliberately not coerced into the old four-vector head."""

        return None

    def to_target_dict(self) -> dict[str, Any]:
        """Serialize labels only; no reverse-match evidence is included."""

        return {
            "adapter_contract": ADAPTER_CONTRACT,
            "source_target_contract": DYNAMIC_CROP_TARGET_CONTRACT,
            "shot_index": self.shot_index,
            "source_id": self.source_id,
            "shot_duration_seconds": self.shot_duration_seconds,
            "motion_label": self.motion_label,
            "motion_label_loss_masked": self.motion_label_loss_masked,
            "keyframe_count": len(self.keyframes),
            "keyframes": [item.to_dict() for item in self.keyframes],
            "start_keyframe": self.start_keyframe.to_dict(),
            "end_keyframe": self.end_keyframe.to_dict(),
            "legacy_static_crop_head_compatible": False,
            "legacy_static_crop_head_target": None,
            "artifact_content_sha256": self.artifact_content_sha256,
            "build_fingerprint_sha256": self.build_fingerprint_sha256,
        }


def build_stage_edit_dynamic_crop_target(
    value: Mapping[str, Any],
) -> StageEditDynamicCropTarget | None:
    """Return an eligible target, or ``None`` for a loss-masked crop-v2 shot."""

    validate_reference_dynamic_crop_supervision_payload(value)
    supervision = value["supervision"]
    if supervision["eligible"] is not True:
        return None
    shot = value["shot"]
    provenance = value["provenance"]
    keyframes = tuple(
        DynamicCropKeyframeTarget(
            time_within_shot_seconds=_finite(
                raw.get("time_within_shot_seconds"), "dynamic crop keyframe time"
            ),
            zoom=_finite(raw.get("zoom"), "dynamic crop zoom"),
            center_x=_finite(raw.get("center_x"), "dynamic crop center_x"),
            center_y=_finite(raw.get("center_y"), "dynamic crop center_y"),
            roll_degrees=_finite(
                raw.get("roll_degrees"), "dynamic crop roll"
            ),
        )
        for raw in supervision["renderer_keyframes"]
    )
    return StageEditDynamicCropTarget(
        shot_index=int(shot["shot_index"]),
        source_id=str(shot["source_id"]),
        shot_duration_seconds=_finite(
            shot.get("duration_seconds"), "dynamic crop shot duration"
        ),
        motion_label=str(supervision["motion_label"]),
        motion_label_loss_masked=bool(
            supervision["motion_label_loss_mask"]["masked"]
        ),
        keyframes=keyframes,
        artifact_content_sha256=str(provenance["artifact_content_sha256"]),
        build_fingerprint_sha256=str(provenance["build_fingerprint_sha256"]),
    )


def load_stage_edit_dynamic_crop_target(
    path: Path,
    *,
    verify_input_files: bool = True,
) -> StageEditDynamicCropTarget | None:
    artifact = load_reference_dynamic_crop_supervision(
        path, verify_input_files=verify_input_files
    )
    return build_stage_edit_dynamic_crop_target(artifact.payload)


__all__ = [
    "ADAPTER_CONTRACT",
    "DynamicCropKeyframeTarget",
    "StageEditDynamicCropTarget",
    "build_stage_edit_dynamic_crop_target",
    "load_stage_edit_dynamic_crop_target",
]
