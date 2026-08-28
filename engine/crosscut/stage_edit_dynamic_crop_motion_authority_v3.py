"""Typed runtime authority for Dynamic Crop v3 framing and motion.

Dynamic Crop v3 deliberately emits a renderer-safe framing track for every
eligible source/shot.  Track presence therefore means only that framing was
applied; it is not evidence that an editorial motion event was rendered.

This module keeps those meanings separate.  A motion event is counted only
when all three independent facts agree:

* the exact v3 framing track reached the renderer, unless independently
  verified source geometry retained explicit precedence;
* the model motion logit passes a leakage-safe, release-freeze-bound threshold;
* samples at the renderer's actual output-frame timestamps cross one of the
  physical motion-label thresholds used by reference supervision.

Changes between two static tracks at a hard cut are intentionally never
combined.  They remain cut-time reframing and do not consume the professional
montage motion budget.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    TransitionKind,
    segment_frame_counts,
)
from .stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT,
    DynamicCropTrackDecodeV3,
)


DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION = 1
DYNAMIC_CROP_MOTION_AUTHORITY_CONTRACT = (
    "dynamic_crop_v3_loso_release_bound_motion_authority_v1"
)
DYNAMIC_CROP_MOTION_THRESHOLD_FREEZE_CONTRACT = (
    "zero_logit_predeclared_before_loso_heldout_evaluation_v1"
)
DYNAMIC_CROP_MOTION_PHYSICAL_THRESHOLD_CONTRACT = (
    "reference_dynamic_crop_supervision_physical_ranges_v1"
)
DYNAMIC_CROP_MOTION_RENDERER_SAMPLING_CONTRACT = (
    "exact_render_plan_segment_frame_count_local_frame_timestamps_v1"
)
DYNAMIC_CROP_MOTION_OVERLAY_BINDING_CONTRACT = (
    "exact_decode_overlay_or_verified_geometry_or_face_dissolve_precedence_v2"
)
DYNAMIC_CROP_RENDERER_FIRST_MOTION_CLOCK_CONTRACT = (
    "first_10_decimal_renderer_expression_state_change_per_corroborated_component_v1"
)
DYNAMIC_CROP_MOTION_SEGMENT_EVIDENCE_CONTRACT = (
    "hybrid_geometry_precedence_and_loso_renderer_motion_intersection_v2"
)
DYNAMIC_CROP_MOTION_TIMELINE_EVIDENCE_CONTRACT = (
    "model_motion_budget_excludes_verified_geometry_and_cut_reframes_v2"
)
DETERMINISTIC_GEOMETRY_OVERRIDE_DISPOSITION = (
    "deterministic_geometry_override_suppressed_model_track"
)

# The binary head is trained with BCE logits and evaluated at zero.  This
# threshold is predeclared, not tuned on any held-out song.  The release-freeze
# identity and its clean source commit bind the decision contract used later.
DEFAULT_MOTION_LOGIT_THRESHOLD = 0.0
DEFAULT_DYNAMIC_ZOOM_RANGE_THRESHOLD = 0.008
DEFAULT_DYNAMIC_CENTER_RANGE_THRESHOLD = 0.005
DEFAULT_DYNAMIC_ROLL_RANGE_THRESHOLD_DEGREES = 0.20
DEFAULT_MINIMUM_MOTION_APPLIED_FRACTION = 0.08
DEFAULT_MAXIMUM_MOTION_APPLIED_FRACTION = 0.35


class DynamicCropMotionAuthorityV3Error(ValueError):
    """Motion authority or renderer evidence is malformed or inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DynamicCropMotionAuthorityV3Error(
            "dynamic crop motion evidence is not canonical JSON"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicCropMotionAuthorityV3Error(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicCropMotionAuthorityV3Error(f"{label} must be finite")
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicCropMotionAuthorityV3Error(
            f"{label} must be a non-negative integer"
        )
    return value


def _deterministic_geometry_override_indices(
    values: Sequence[int],
    *,
    segment_count: int,
) -> tuple[int, ...]:
    selected = tuple(values)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < segment_count
        for value in selected
    ):
        raise DynamicCropMotionAuthorityV3Error(
            "deterministic geometry override index is invalid"
        )
    if tuple(sorted(set(selected))) != selected:
        raise DynamicCropMotionAuthorityV3Error(
            "deterministic geometry override indices must be unique and ordered"
        )
    return selected


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise DynamicCropMotionAuthorityV3Error(f"{label} keys are not exact")


def _content_addressed_payload(
    payload: Mapping[str, Any], *, hash_field: str, label: str
) -> None:
    digest = payload.get(hash_field)
    unsigned = dict(payload)
    unsigned[hash_field] = None
    if not _is_sha256(digest) or _canonical_sha256(unsigned) != digest:
        raise DynamicCropMotionAuthorityV3Error(
            f"{label} content hash does not match its payload"
        )


@dataclass(frozen=True, slots=True)
class DynamicCropMotionPhysicalThresholdsV3:
    zoom_range: float = DEFAULT_DYNAMIC_ZOOM_RANGE_THRESHOLD
    maximum_center_distance: float = DEFAULT_DYNAMIC_CENTER_RANGE_THRESHOLD
    roll_range_degrees: float = DEFAULT_DYNAMIC_ROLL_RANGE_THRESHOLD_DEGREES
    contract: str = DYNAMIC_CROP_MOTION_PHYSICAL_THRESHOLD_CONTRACT

    def __post_init__(self) -> None:
        for field_name in (
            "zoom_range",
            "maximum_center_distance",
            "roll_range_degrees",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if value <= 0.0:
                raise DynamicCropMotionAuthorityV3Error(
                    f"{field_name} must be positive"
                )
            object.__setattr__(self, field_name, value)
        if self.contract != DYNAMIC_CROP_MOTION_PHYSICAL_THRESHOLD_CONTRACT:
            raise DynamicCropMotionAuthorityV3Error(
                "dynamic crop physical threshold contract differs"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "zoom_range": self.zoom_range,
            "maximum_center_distance": self.maximum_center_distance,
            "roll_range_degrees": self.roll_range_degrees,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DynamicCropMotionPhysicalThresholdsV3":
        material = dict(payload)
        _exact_keys(
            material,
            {
                "contract",
                "zoom_range",
                "maximum_center_distance",
                "roll_range_degrees",
            },
            "dynamic crop physical thresholds",
        )
        return cls(
            zoom_range=_finite(material["zoom_range"], "zoom range threshold"),
            maximum_center_distance=_finite(
                material["maximum_center_distance"], "center range threshold"
            ),
            roll_range_degrees=_finite(
                material["roll_range_degrees"], "roll range threshold"
            ),
            contract=str(material["contract"]),
        )


_AUTHORITY_KEYS = {
    "schema_version",
    "contract",
    "release_freeze_content_sha256",
    "release_freeze_file_sha256",
    "source_commit_oid",
    "candidate_id",
    "motion_logit_threshold",
    "threshold_freeze_contract",
    "physical_thresholds",
    "renderer_sampling_contract",
    "authority_content_sha256",
}


@dataclass(frozen=True, slots=True)
class DynamicCropMotionAuthorityV3:
    release_freeze_content_sha256: str
    release_freeze_file_sha256: str
    source_commit_oid: str
    candidate_id: str
    motion_logit_threshold: float
    physical_thresholds: DynamicCropMotionPhysicalThresholdsV3
    authority_content_sha256: str
    schema_version: int = DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION
    contract: str = DYNAMIC_CROP_MOTION_AUTHORITY_CONTRACT
    threshold_freeze_contract: str = (
        DYNAMIC_CROP_MOTION_THRESHOLD_FREEZE_CONTRACT
    )
    renderer_sampling_contract: str = (
        DYNAMIC_CROP_MOTION_RENDERER_SAMPLING_CONTRACT
    )

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION:
            raise DynamicCropMotionAuthorityV3Error(
                "dynamic crop motion authority schema differs"
            )
        if self.contract != DYNAMIC_CROP_MOTION_AUTHORITY_CONTRACT:
            raise DynamicCropMotionAuthorityV3Error(
                "dynamic crop motion authority contract differs"
            )
        for value, label in (
            (self.release_freeze_content_sha256, "release freeze content hash"),
            (self.release_freeze_file_sha256, "release freeze file hash"),
            (self.authority_content_sha256, "motion authority content hash"),
        ):
            if not _is_sha256(value):
                raise DynamicCropMotionAuthorityV3Error(f"{label} is invalid")
        commit = self.source_commit_oid.strip().lower()
        if (
            len(commit) != 40
            or commit != self.source_commit_oid
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "motion authority source commit is invalid"
            )
        if not self.candidate_id.strip():
            raise DynamicCropMotionAuthorityV3Error(
                "motion authority candidate id cannot be blank"
            )
        threshold = _finite(self.motion_logit_threshold, "motion logit threshold")
        if threshold != DEFAULT_MOTION_LOGIT_THRESHOLD:
            raise DynamicCropMotionAuthorityV3Error(
                "motion threshold was not the predeclared zero-logit threshold"
            )
        object.__setattr__(self, "motion_logit_threshold", threshold)
        if not isinstance(
            self.physical_thresholds, DynamicCropMotionPhysicalThresholdsV3
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "motion physical thresholds need their typed contract"
            )
        if self.physical_thresholds != DynamicCropMotionPhysicalThresholdsV3():
            raise DynamicCropMotionAuthorityV3Error(
                "motion physical thresholds differ from reference supervision"
            )
        if (
            self.threshold_freeze_contract
            != DYNAMIC_CROP_MOTION_THRESHOLD_FREEZE_CONTRACT
            or self.renderer_sampling_contract
            != DYNAMIC_CROP_MOTION_RENDERER_SAMPLING_CONTRACT
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "motion authority provenance contract differs"
            )
        _content_addressed_payload(
            self.to_dict(),
            hash_field="authority_content_sha256",
            label="dynamic crop motion authority",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "release_freeze_content_sha256": (
                self.release_freeze_content_sha256
            ),
            "release_freeze_file_sha256": self.release_freeze_file_sha256,
            "source_commit_oid": self.source_commit_oid,
            "candidate_id": self.candidate_id,
            "motion_logit_threshold": self.motion_logit_threshold,
            "threshold_freeze_contract": self.threshold_freeze_contract,
            "physical_thresholds": self.physical_thresholds.to_dict(),
            "renderer_sampling_contract": self.renderer_sampling_contract,
            "authority_content_sha256": self.authority_content_sha256,
        }

    @classmethod
    def create(
        cls,
        *,
        release_freeze_content_sha256: str,
        release_freeze_file_sha256: str,
        source_commit_oid: str,
        candidate_id: str,
        motion_logit_threshold: float = DEFAULT_MOTION_LOGIT_THRESHOLD,
        physical_thresholds: DynamicCropMotionPhysicalThresholdsV3 = (
            DynamicCropMotionPhysicalThresholdsV3()
        ),
    ) -> "DynamicCropMotionAuthorityV3":
        unsigned = {
            "schema_version": DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION,
            "contract": DYNAMIC_CROP_MOTION_AUTHORITY_CONTRACT,
            "release_freeze_content_sha256": release_freeze_content_sha256,
            "release_freeze_file_sha256": release_freeze_file_sha256,
            "source_commit_oid": source_commit_oid,
            "candidate_id": candidate_id,
            "motion_logit_threshold": float(motion_logit_threshold),
            "threshold_freeze_contract": (
                DYNAMIC_CROP_MOTION_THRESHOLD_FREEZE_CONTRACT
            ),
            "physical_thresholds": physical_thresholds.to_dict(),
            "renderer_sampling_contract": (
                DYNAMIC_CROP_MOTION_RENDERER_SAMPLING_CONTRACT
            ),
            "authority_content_sha256": None,
        }
        return cls.from_dict(
            {
                **unsigned,
                "authority_content_sha256": _canonical_sha256(unsigned),
            }
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DynamicCropMotionAuthorityV3":
        material = dict(payload)
        _exact_keys(material, _AUTHORITY_KEYS, "dynamic crop motion authority")
        _content_addressed_payload(
            material,
            hash_field="authority_content_sha256",
            label="dynamic crop motion authority",
        )
        thresholds = material["physical_thresholds"]
        if not isinstance(thresholds, Mapping):
            raise DynamicCropMotionAuthorityV3Error(
                "motion physical thresholds must be an object"
            )
        return cls(
            release_freeze_content_sha256=str(
                material["release_freeze_content_sha256"]
            ),
            release_freeze_file_sha256=str(
                material["release_freeze_file_sha256"]
            ),
            source_commit_oid=str(material["source_commit_oid"]),
            candidate_id=str(material["candidate_id"]),
            motion_logit_threshold=_finite(
                material["motion_logit_threshold"], "motion logit threshold"
            ),
            physical_thresholds=(
                DynamicCropMotionPhysicalThresholdsV3.from_dict(thresholds)
            ),
            authority_content_sha256=str(material["authority_content_sha256"]),
            schema_version=material["schema_version"],
            contract=str(material["contract"]),
            threshold_freeze_contract=str(material["threshold_freeze_contract"]),
            renderer_sampling_contract=str(material["renderer_sampling_contract"]),
        )


def load_dynamic_crop_motion_authority_v3(
    release_freeze_path: Path,
) -> DynamicCropMotionAuthorityV3:
    """Build authority only from a strictly replayed v3 LOSO release freeze."""

    # Imported lazily to avoid making renderer primitives depend on the MLX
    # final-fit launcher at module import time.
    from .stage_edit_dynamic_crop_v3_final_release_launcher import (
        load_dynamic_crop_v3_release_freeze,
    )

    if not isinstance(release_freeze_path, Path):
        raise DynamicCropMotionAuthorityV3Error(
            "motion release freeze path must be a typed Path"
        )
    raw = release_freeze_path
    if not raw.is_absolute() or raw.is_symlink():
        raise DynamicCropMotionAuthorityV3Error(
            "motion release freeze must be an absolute normalized non-symlink file"
        )
    try:
        before_stat = raw.lstat()
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise DynamicCropMotionAuthorityV3Error(
            "motion release freeze is missing or unsafe"
        ) from exc
    if resolved != raw or not stat.S_ISREG(before_stat.st_mode):
        raise DynamicCropMotionAuthorityV3Error(
            "motion release freeze must be an absolute normalized non-symlink file"
        )
    before_identity = (
        before_stat.st_dev,
        before_stat.st_ino,
        before_stat.st_mode,
        before_stat.st_size,
        before_stat.st_mtime_ns,
        before_stat.st_ctime_ns,
    )
    before_sha256 = _file_sha256(raw)
    freeze = load_dynamic_crop_v3_release_freeze(raw)
    try:
        after_stat = raw.lstat()
        after_resolved = raw.resolve(strict=True)
        after_sha256 = _file_sha256(raw)
    except OSError as exc:
        raise DynamicCropMotionAuthorityV3Error(
            "motion release freeze changed during strict replay"
        ) from exc
    after_identity = (
        after_stat.st_dev,
        after_stat.st_ino,
        after_stat.st_mode,
        after_stat.st_size,
        after_stat.st_mtime_ns,
        after_stat.st_ctime_ns,
    )
    if (
        after_resolved != raw
        or not stat.S_ISREG(after_stat.st_mode)
        or after_identity != before_identity
        or after_sha256 != before_sha256
    ):
        raise DynamicCropMotionAuthorityV3Error(
            "motion release freeze changed during strict replay"
        )
    if not isinstance(freeze, Mapping):
        raise DynamicCropMotionAuthorityV3Error(
            "strict motion release-freeze replay returned a malformed payload"
        )
    source = freeze.get("source_provenance")
    if not isinstance(source, Mapping):
        raise DynamicCropMotionAuthorityV3Error(
            "release freeze source provenance is missing"
        )
    return DynamicCropMotionAuthorityV3.create(
        release_freeze_content_sha256=str(freeze["freeze_content_sha256"]),
        release_freeze_file_sha256=after_sha256,
        source_commit_oid=str(source.get("head_commit_oid") or ""),
        candidate_id=str(freeze.get("candidate_id") or ""),
    )


def load_dynamic_crop_motion_authority_from_registered_identity_v3(
    identity: object,
) -> DynamicCropMotionAuthorityV3:
    """Replay one registry-issued v3 release-freeze identity exactly.

    A registered checkpoint name is intentionally insufficient.  Runtime must
    receive the typed identity issued by the production registry, replay its
    pointed release freeze, and corroborate every identity field before motion
    may be authorized.
    """

    # Lazy import keeps this renderer-facing contract independent from registry
    # activation at module import time.
    from .stage_edit_production_registry import (
        RegisteredStageEditMotionAuthorityIdentityV3,
    )

    if not isinstance(identity, RegisteredStageEditMotionAuthorityIdentityV3):
        raise DynamicCropMotionAuthorityV3Error(
            "motion authority needs a typed registered v3 identity"
        )
    path = identity.release_freeze_path
    before_sha256 = _file_sha256(path)
    if before_sha256 != identity.release_freeze_file_sha256:
        raise DynamicCropMotionAuthorityV3Error(
            "registered motion release-freeze file identity changed"
        )
    authority = load_dynamic_crop_motion_authority_v3(path)
    after_sha256 = _file_sha256(path)
    if after_sha256 != before_sha256:
        raise DynamicCropMotionAuthorityV3Error(
            "motion release freeze changed during strict replay"
        )
    expected = {
        "release_freeze_file_sha256": identity.release_freeze_file_sha256,
        "release_freeze_content_sha256": identity.release_freeze_content_sha256,
        "source_commit_oid": identity.source_commit_oid,
        "candidate_id": identity.candidate_id,
        "motion_logit_threshold": identity.motion_logit_threshold,
        "threshold_freeze_contract": identity.threshold_freeze_contract,
    }
    observed = {
        "release_freeze_file_sha256": authority.release_freeze_file_sha256,
        "release_freeze_content_sha256": authority.release_freeze_content_sha256,
        "source_commit_oid": authority.source_commit_oid,
        "candidate_id": authority.candidate_id,
        "motion_logit_threshold": authority.motion_logit_threshold,
        "threshold_freeze_contract": authority.threshold_freeze_contract,
    }
    if observed != expected:
        raise DynamicCropMotionAuthorityV3Error(
            "strictly replayed motion authority differs from its registry identity"
        )
    return authority


def _track_payload(track: EditorialTransformTrack | None) -> object:
    if track is None:
        return None
    return {
        "reason": track.reason,
        "keyframes": [
            {
                "time_seconds": item.time_seconds,
                "zoom": item.zoom,
                "center_x": item.center_x,
                "center_y": item.center_y,
                "roll_degrees": item.roll_degrees,
            }
            for item in track.keyframes
        ],
    }


def _decode_payload(decode: DynamicCropTrackDecodeV3) -> dict[str, Any]:
    return decode.to_dict()


@dataclass(frozen=True, slots=True)
class _ResolvedOverlayBindingV3:
    disposition: str
    framing_track_applied: bool
    face_dissolve_suppressed: bool
    motion_suppressed_to_static_framing: bool
    time_reparameterization_scale: float | None
    selected_decode_content_sha256: str
    original_decode_track_sha256: str
    provisional_overlay_track_sha256: str | None
    rendered_track_sha256: str | None


def _expected_provisional_overlay_track(
    original: EditorialTransformTrack,
    *,
    scale: float,
) -> EditorialTransformTrack:
    return EditorialTransformTrack(
        tuple(
            EditorialTransformKeyframe(
                time_seconds=item.time_seconds * scale,
                zoom=item.zoom,
                center_x=item.center_x,
                center_y=item.center_y,
                roll_degrees=item.roll_degrees,
            )
            for item in original.keyframes
        ),
        reason=(
            f"{original.reason}_stage_edit_preserved"
            if math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-9)
            else f"{original.reason}_stage_edit_targeted_repair_reparameterized"
        ),
    )


def _static_framing_track_from_provisional(
    provisional: EditorialTransformTrack,
) -> EditorialTransformTrack:
    first = provisional.evaluate(0.0)
    return EditorialTransformTrack(
        (
            EditorialTransformKeyframe(
                0.0,
                first.zoom,
                first.center_x,
                first.center_y,
                first.roll_degrees,
            ),
        ),
        reason=(
            f"{provisional.reason}_"
            "loso_or_physical_motion_suppressed_to_static_framing"
        ),
    )


def _tracks_match_under_exact_overlay(
    *,
    original: EditorialTransformTrack,
    rendered: EditorialTransformTrack,
    expected_scale: float,
) -> bool:
    expected = _expected_provisional_overlay_track(
        original,
        scale=expected_scale,
    )
    if len(original.keyframes) != len(rendered.keyframes):
        return False
    if rendered.reason != expected.reason:
        return False
    for source, final in zip(original.keyframes, rendered.keyframes, strict=True):
        if not math.isclose(
            final.time_seconds,
            source.time_seconds * expected_scale,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or (
            final.zoom,
            final.center_x,
            final.center_y,
            final.roll_degrees,
        ) != (
            source.zoom,
            source.center_x,
            source.center_y,
            source.roll_degrees,
        ):
            return False
    return True


def _resolve_overlay_binding(
    *,
    segment_index: int,
    plan: RenderPlan,
    decode: DynamicCropTrackDecodeV3,
    deterministic_geometry_override: bool = False,
) -> _ResolvedOverlayBindingV3:
    segment = plan.segments[segment_index]
    selected_sha = _canonical_sha256(_decode_payload(decode))
    original_sha = _canonical_sha256(_track_payload(decode.track))
    rendered_sha = (
        _canonical_sha256(_track_payload(segment.editorial_transform))
        if segment.editorial_transform is not None
        else None
    )
    adjacent_face_dissolve = bool(
        (
            segment_index > 0
            and plan.transitions[segment_index - 1].kind
            is TransitionKind.FACE_DISSOLVE
        )
        or (
            segment_index < len(plan.transitions)
            and plan.transitions[segment_index].kind
            is TransitionKind.FACE_DISSOLVE
        )
    )
    if not decode.eligible:
        if segment.editorial_transform is not None:
            raise DynamicCropMotionAuthorityV3Error(
                "rejected v3 decode cannot reach the final renderer"
            )
        return _ResolvedOverlayBindingV3(
            disposition="decode_rejected_no_track",
            framing_track_applied=False,
            face_dissolve_suppressed=False,
            time_reparameterization_scale=None,
            selected_decode_content_sha256=selected_sha,
            original_decode_track_sha256=original_sha,
            provisional_overlay_track_sha256=None,
            rendered_track_sha256=None,
            motion_suppressed_to_static_framing=False,
        )
    if decode.track is None or decode.shot_duration_seconds is None:
        raise DynamicCropMotionAuthorityV3Error(
            "eligible v3 decode lacks track/duration overlay provenance"
        )
    if deterministic_geometry_override:
        if adjacent_face_dissolve:
            raise DynamicCropMotionAuthorityV3Error(
                "deterministic geometry override cannot alias face-dissolve suppression"
            )
        rendered = segment.editorial_transform
        non_model_transform = bool(
            rendered is not None
            and not rendered.reason.startswith("stage_edit_dynamic_crop_head_v3")
        )
        if not (
            segment.crop is not None
            or segment.pan_crop is not None
            or non_model_transform
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "deterministic geometry override lacks verified non-model geometry"
            )
        return _ResolvedOverlayBindingV3(
            disposition=DETERMINISTIC_GEOMETRY_OVERRIDE_DISPOSITION,
            framing_track_applied=False,
            face_dissolve_suppressed=False,
            motion_suppressed_to_static_framing=False,
            time_reparameterization_scale=None,
            selected_decode_content_sha256=selected_sha,
            original_decode_track_sha256=original_sha,
            provisional_overlay_track_sha256=None,
            rendered_track_sha256=None,
        )
    if segment.editorial_transform is None:
        if not adjacent_face_dissolve:
            raise DynamicCropMotionAuthorityV3Error(
                "eligible v3 track was suppressed without a verified face dissolve"
            )
        return _ResolvedOverlayBindingV3(
            disposition="verified_face_dissolve_suppressed",
            framing_track_applied=False,
            face_dissolve_suppressed=True,
            time_reparameterization_scale=None,
            selected_decode_content_sha256=selected_sha,
            original_decode_track_sha256=original_sha,
            provisional_overlay_track_sha256=None,
            rendered_track_sha256=None,
            motion_suppressed_to_static_framing=False,
        )
    expected_scale = segment.duration / decode.shot_duration_seconds
    provisional = _expected_provisional_overlay_track(
        decode.track,
        scale=expected_scale,
    )
    static = _static_framing_track_from_provisional(provisional)
    exact_provisional = _tracks_match_under_exact_overlay(
        original=decode.track,
        rendered=segment.editorial_transform,
        expected_scale=expected_scale,
    )
    exact_static_suppression = segment.editorial_transform == static
    if not exact_provisional and not exact_static_suppression:
        raise DynamicCropMotionAuthorityV3Error(
            "final renderer track is not the exact StageEdit overlay of its decode"
        )
    disposition = (
        "loso_or_physical_motion_suppressed_to_static_framing"
        if exact_static_suppression
        else "preserved_same_duration_renamed"
        if math.isclose(expected_scale, 1.0, rel_tol=0.0, abs_tol=1e-9)
        else "targeted_repair_duration_reparameterized"
    )
    return _ResolvedOverlayBindingV3(
        disposition=disposition,
        framing_track_applied=True,
        face_dissolve_suppressed=False,
        motion_suppressed_to_static_framing=exact_static_suppression,
        time_reparameterization_scale=expected_scale,
        selected_decode_content_sha256=selected_sha,
        original_decode_track_sha256=original_sha,
        provisional_overlay_track_sha256=_canonical_sha256(
            _track_payload(provisional)
        ),
        rendered_track_sha256=rendered_sha,
    )


@dataclass(frozen=True, slots=True)
class DynamicCropRendererFirstMotionClockV3:
    component: str
    frame_index: int
    segment_time_seconds: float
    master_time_seconds: float
    contract: str = DYNAMIC_CROP_RENDERER_FIRST_MOTION_CLOCK_CONTRACT

    def __post_init__(self) -> None:
        if self.component not in {"zoom", "center", "roll_degrees"}:
            raise DynamicCropMotionAuthorityV3Error(
                "renderer first-motion component is invalid"
            )
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 1
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "renderer first-motion frame must follow frame zero"
            )
        segment_time = _finite(
            self.segment_time_seconds, "renderer first-motion segment time"
        )
        master_time = _finite(
            self.master_time_seconds, "renderer first-motion master time"
        )
        if segment_time <= 0.0 or master_time < segment_time:
            raise DynamicCropMotionAuthorityV3Error(
                "renderer first-motion clocks are invalid"
            )
        object.__setattr__(self, "segment_time_seconds", segment_time)
        object.__setattr__(self, "master_time_seconds", master_time)
        if self.contract != DYNAMIC_CROP_RENDERER_FIRST_MOTION_CLOCK_CONTRACT:
            raise DynamicCropMotionAuthorityV3Error(
                "renderer first-motion clock contract differs"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "component": self.component,
            "frame_index": self.frame_index,
            "segment_time_seconds": self.segment_time_seconds,
            "master_time_seconds": self.master_time_seconds,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DynamicCropRendererFirstMotionClockV3":
        material = dict(payload)
        _exact_keys(
            material,
            {
                "contract",
                "component",
                "frame_index",
                "segment_time_seconds",
                "master_time_seconds",
            },
            "renderer first-motion clock",
        )
        return cls(
            component=str(material["component"]),
            frame_index=material["frame_index"],
            segment_time_seconds=_finite(
                material["segment_time_seconds"],
                "renderer first-motion segment time",
            ),
            master_time_seconds=_finite(
                material["master_time_seconds"],
                "renderer first-motion master time",
            ),
            contract=str(material["contract"]),
        )


_SEGMENT_KEYS = {
    "schema_version",
    "contract",
    "segment_index",
    "segment_master_start_seconds",
    "segment_master_end_seconds",
    "source_index",
    "source_id",
    "render_source_path_sha256",
    "source_start_seconds",
    "source_duration_seconds",
    "adapter_contract",
    "overlay_binding_contract",
    "overlay_disposition",
    "face_dissolve_suppressed",
    "decoded_shot_duration_seconds",
    "final_segment_duration_seconds",
    "time_reparameterization_scale",
    "selected_decode_content_sha256",
    "original_decode_track_sha256",
    "provisional_overlay_track_sha256",
    "rendered_track_sha256",
    "motion_suppressed_to_static_framing",
    "motion_suppression_reasons",
    "provisional_renderer_sample_states_sha256",
    "provisional_zoom_range",
    "provisional_maximum_center_distance",
    "provisional_roll_range_degrees",
    "provisional_physical_motion_corroborated",
    "framing_track_applied",
    "motion_logit",
    "motion_logit_threshold",
    "motion_logit_passed",
    "renderer_sampling_contract",
    "renderer_fps",
    "renderer_frame_count",
    "renderer_sample_times_sha256",
    "renderer_sample_states_sha256",
    "zoom_range",
    "maximum_center_distance",
    "roll_range_degrees",
    "zoom_range_corroborated",
    "center_range_corroborated",
    "roll_range_corroborated",
    "physical_motion_corroborated",
    "renderer_visible_first_motion_clocks",
    "renderer_motion_without_loso_authority",
    "motion_event_applied",
    "motion_budget_classification",
    "failure_reasons",
    "authority_content_sha256",
    "evidence_content_sha256",
}


@dataclass(frozen=True, slots=True)
class DynamicCropMotionSegmentEvidenceV3:
    segment_index: int
    segment_master_start_seconds: float
    segment_master_end_seconds: float
    source_index: int
    source_id: str | None
    render_source_path_sha256: str
    source_start_seconds: float
    source_duration_seconds: float
    overlay_disposition: str
    face_dissolve_suppressed: bool
    decoded_shot_duration_seconds: float | None
    final_segment_duration_seconds: float
    time_reparameterization_scale: float | None
    selected_decode_content_sha256: str
    original_decode_track_sha256: str
    provisional_overlay_track_sha256: str | None
    rendered_track_sha256: str | None
    motion_suppressed_to_static_framing: bool
    motion_suppression_reasons: tuple[str, ...]
    provisional_renderer_sample_states_sha256: str | None
    provisional_zoom_range: float
    provisional_maximum_center_distance: float
    provisional_roll_range_degrees: float
    provisional_physical_motion_corroborated: bool
    framing_track_applied: bool
    motion_logit: float
    motion_logit_threshold: float
    motion_logit_passed: bool
    renderer_fps: float
    renderer_frame_count: int
    renderer_sample_times_sha256: str
    renderer_sample_states_sha256: str
    zoom_range: float
    maximum_center_distance: float
    roll_range_degrees: float
    zoom_range_corroborated: bool
    center_range_corroborated: bool
    roll_range_corroborated: bool
    physical_motion_corroborated: bool
    renderer_visible_first_motion_clocks: tuple[
        DynamicCropRendererFirstMotionClockV3, ...
    ]
    renderer_motion_without_loso_authority: bool
    motion_event_applied: bool
    motion_budget_classification: str
    failure_reasons: tuple[str, ...]
    authority_content_sha256: str
    evidence_content_sha256: str
    schema_version: int = DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION
    contract: str = DYNAMIC_CROP_MOTION_SEGMENT_EVIDENCE_CONTRACT
    adapter_contract: str = DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT
    overlay_binding_contract: str = DYNAMIC_CROP_MOTION_OVERLAY_BINDING_CONTRACT
    renderer_sampling_contract: str = (
        DYNAMIC_CROP_MOTION_RENDERER_SAMPLING_CONTRACT
    )

    def __post_init__(self) -> None:
        for field_name in ("segment_index", "source_index", "renderer_frame_count"):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(getattr(self, field_name), field_name),
            )
        if self.renderer_frame_count < 2:
            raise DynamicCropMotionAuthorityV3Error(
                "motion evidence needs at least two renderer samples"
            )
        master_start = _finite(
            self.segment_master_start_seconds, "segment master start"
        )
        master_end = _finite(
            self.segment_master_end_seconds, "segment master end"
        )
        source_start = _finite(self.source_start_seconds, "source start")
        source_duration = _finite(self.source_duration_seconds, "source duration")
        if (
            master_start < 0.0
            or master_end <= master_start
            or source_start < 0.0
            or source_duration <= 0.0
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "motion segment master/source clocks are invalid"
            )
        object.__setattr__(self, "segment_master_start_seconds", master_start)
        object.__setattr__(self, "segment_master_end_seconds", master_end)
        object.__setattr__(self, "source_start_seconds", source_start)
        object.__setattr__(self, "source_duration_seconds", source_duration)
        for field_name in (
            "face_dissolve_suppressed",
            "motion_suppressed_to_static_framing",
            "provisional_physical_motion_corroborated",
            "framing_track_applied",
            "motion_logit_passed",
            "zoom_range_corroborated",
            "center_range_corroborated",
            "roll_range_corroborated",
            "physical_motion_corroborated",
            "renderer_motion_without_loso_authority",
            "motion_event_applied",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise DynamicCropMotionAuthorityV3Error(
                    f"{field_name} must be boolean"
                )
        for field_name in (
            "motion_logit",
            "motion_logit_threshold",
            "renderer_fps",
            "zoom_range",
            "maximum_center_distance",
            "roll_range_degrees",
            "provisional_zoom_range",
            "provisional_maximum_center_distance",
            "provisional_roll_range_degrees",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if field_name in {
                "renderer_fps",
                "zoom_range",
                "maximum_center_distance",
                "roll_range_degrees",
                "provisional_zoom_range",
                "provisional_maximum_center_distance",
                "provisional_roll_range_degrees",
            } and value < 0.0:
                raise DynamicCropMotionAuthorityV3Error(
                    f"{field_name} cannot be negative"
                )
            object.__setattr__(self, field_name, value)
        final_duration = _finite(
            self.final_segment_duration_seconds, "final segment duration"
        )
        if final_duration <= 0.0:
            raise DynamicCropMotionAuthorityV3Error(
                "final segment duration must be positive"
            )
        object.__setattr__(self, "final_segment_duration_seconds", final_duration)
        if not math.isclose(
            self.segment_master_end_seconds,
            self.segment_master_start_seconds + final_duration,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "segment master clock differs from final duration"
            )
        if self.decoded_shot_duration_seconds is not None:
            decoded_duration = _finite(
                self.decoded_shot_duration_seconds, "decoded shot duration"
            )
            if decoded_duration <= 0.0:
                raise DynamicCropMotionAuthorityV3Error(
                    "decoded shot duration must be positive"
                )
            object.__setattr__(
                self, "decoded_shot_duration_seconds", decoded_duration
            )
        if self.time_reparameterization_scale is not None:
            scale = _finite(
                self.time_reparameterization_scale,
                "time reparameterization scale",
            )
            if scale <= 0.0:
                raise DynamicCropMotionAuthorityV3Error(
                    "time reparameterization scale must be positive"
                )
            object.__setattr__(self, "time_reparameterization_scale", scale)
        if self.renderer_fps <= 0.0:
            raise DynamicCropMotionAuthorityV3Error(
                "renderer fps must be positive"
            )
        for value, label in (
            (self.renderer_sample_times_sha256, "sample-time hash"),
            (self.renderer_sample_states_sha256, "sample-state hash"),
            (self.render_source_path_sha256, "render-source path hash"),
            (self.selected_decode_content_sha256, "selected-decode hash"),
            (self.original_decode_track_sha256, "original-track hash"),
            (self.authority_content_sha256, "authority hash"),
            (self.evidence_content_sha256, "segment evidence hash"),
        ):
            if not _is_sha256(value):
                raise DynamicCropMotionAuthorityV3Error(f"{label} is invalid")
        if self.source_id is not None and not self.source_id.strip():
            raise DynamicCropMotionAuthorityV3Error(
                "motion evidence source id cannot be blank"
            )
        if self.rendered_track_sha256 is not None and not _is_sha256(
            self.rendered_track_sha256
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "rendered-track hash is invalid"
            )
        for value, label in (
            (
                self.provisional_overlay_track_sha256,
                "provisional-overlay track hash",
            ),
            (
                self.provisional_renderer_sample_states_sha256,
                "provisional sample-state hash",
            ),
        ):
            if value is not None and not _is_sha256(value):
                raise DynamicCropMotionAuthorityV3Error(f"{label} is invalid")
        allowed_suppression_reasons = {
            "loso_motion_logit_below_threshold",
            "renderer_physical_range_below_threshold",
        }
        if (
            any(
                reason not in allowed_suppression_reasons
                for reason in self.motion_suppression_reasons
            )
            or len(set(self.motion_suppression_reasons))
            != len(self.motion_suppression_reasons)
            or tuple(sorted(self.motion_suppression_reasons))
            != self.motion_suppression_reasons
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "motion suppression reasons are invalid"
            )
        if (
            self.contract != DYNAMIC_CROP_MOTION_SEGMENT_EVIDENCE_CONTRACT
            or self.adapter_contract
            != DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT
            or self.overlay_binding_contract
            != DYNAMIC_CROP_MOTION_OVERLAY_BINDING_CONTRACT
            or self.renderer_sampling_contract
            != DYNAMIC_CROP_MOTION_RENDERER_SAMPLING_CONTRACT
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "motion segment evidence contract differs"
            )
        overlay_dispositions = {
            "decode_rejected_no_track",
            "preserved_same_duration_renamed",
            "targeted_repair_duration_reparameterized",
            "verified_face_dissolve_suppressed",
            "loso_or_physical_motion_suppressed_to_static_framing",
            DETERMINISTIC_GEOMETRY_OVERRIDE_DISPOSITION,
        }
        if self.overlay_disposition not in overlay_dispositions:
            raise DynamicCropMotionAuthorityV3Error(
                "motion overlay disposition is invalid"
            )
        overlay_applied = self.overlay_disposition in {
            "preserved_same_duration_renamed",
            "targeted_repair_duration_reparameterized",
            "loso_or_physical_motion_suppressed_to_static_framing",
        }
        if self.framing_track_applied != overlay_applied:
            raise DynamicCropMotionAuthorityV3Error(
                "framing-track flag disagrees with exact overlay disposition"
            )
        expected_suppression = (
            self.overlay_disposition == "verified_face_dissolve_suppressed"
        )
        if self.face_dissolve_suppressed != expected_suppression:
            raise DynamicCropMotionAuthorityV3Error(
                "face-dissolve suppression flag disagrees with overlay disposition"
            )
        expected_motion_suppression = (
            self.overlay_disposition
            == "loso_or_physical_motion_suppressed_to_static_framing"
        )
        if (
            self.motion_suppressed_to_static_framing
            != expected_motion_suppression
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "static-framing suppression flag disagrees with overlay disposition"
            )
        if overlay_applied:
            if (
                self.rendered_track_sha256 is None
                or self.provisional_overlay_track_sha256 is None
                or self.provisional_renderer_sample_states_sha256 is None
                or self.decoded_shot_duration_seconds is None
                or self.time_reparameterization_scale is None
                or not math.isclose(
                    self.time_reparameterization_scale,
                    self.final_segment_duration_seconds
                    / self.decoded_shot_duration_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise DynamicCropMotionAuthorityV3Error(
                    "applied overlay lacks exact track/duration reparameterization"
                )
            scale_is_one = math.isclose(
                self.time_reparameterization_scale,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            if (
                not expected_motion_suppression
                and scale_is_one
                != (self.overlay_disposition == "preserved_same_duration_renamed")
            ):
                raise DynamicCropMotionAuthorityV3Error(
                    "overlay disposition disagrees with duration scale"
                )
        elif (
            self.rendered_track_sha256 is not None
            or self.provisional_overlay_track_sha256 is not None
            or self.provisional_renderer_sample_states_sha256 is not None
            or self.time_reparameterization_scale is not None
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "suppressed/rejected overlay cannot claim a rendered track"
            )
        provisional_physical = bool(
            self.provisional_zoom_range
            >= DEFAULT_DYNAMIC_ZOOM_RANGE_THRESHOLD
            or self.provisional_maximum_center_distance
            >= DEFAULT_DYNAMIC_CENTER_RANGE_THRESHOLD
            or self.provisional_roll_range_degrees
            >= DEFAULT_DYNAMIC_ROLL_RANGE_THRESHOLD_DEGREES
        )
        if (
            self.provisional_physical_motion_corroborated
            != provisional_physical
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "provisional physical-motion gate differs from its ranges"
            )
        expected_reasons = tuple(
            sorted(
                (
                    *(
                        ("loso_motion_logit_below_threshold",)
                        if not self.motion_logit_passed
                        else ()
                    ),
                    *(
                        ("renderer_physical_range_below_threshold",)
                        if not provisional_physical
                        else ()
                    ),
                )
            )
        )
        if expected_motion_suppression:
            if not expected_reasons or self.motion_suppression_reasons != expected_reasons:
                raise DynamicCropMotionAuthorityV3Error(
                    "static-framing suppression reasons differ from its gates"
                )
        elif overlay_applied and (
            expected_reasons or self.motion_suppression_reasons
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "unqualified provisional motion reached the final renderer"
            )
        elif not overlay_applied and self.motion_suppression_reasons:
            raise DynamicCropMotionAuthorityV3Error(
                "non-overlay segment cannot carry motion suppression reasons"
            )
        if self.motion_logit_passed != (
            self.motion_logit >= self.motion_logit_threshold
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "motion logit gate disagrees with its frozen threshold"
            )
        physical = bool(
            self.zoom_range_corroborated
            or self.center_range_corroborated
            or self.roll_range_corroborated
        )
        if self.physical_motion_corroborated != physical:
            raise DynamicCropMotionAuthorityV3Error(
                "physical motion gate disagrees with its range gates"
            )
        clocks = tuple(self.renderer_visible_first_motion_clocks)
        if any(
            not isinstance(item, DynamicCropRendererFirstMotionClockV3)
            for item in clocks
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "renderer first-motion clocks need their typed contract"
            )
        expected_components = {
            component
            for component, passed in (
                ("zoom", self.zoom_range_corroborated),
                ("center", self.center_range_corroborated),
                ("roll_degrees", self.roll_range_corroborated),
            )
            if passed
        }
        if (
            {item.component for item in clocks} != expected_components
            or len(clocks) != len(expected_components)
            or tuple(item.component for item in clocks)
            != tuple(sorted(expected_components))
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "renderer first-motion clocks differ from corroborated components"
            )
        for clock in clocks:
            if (
                clock.frame_index >= self.renderer_frame_count
                or not math.isclose(
                    clock.segment_time_seconds,
                    clock.frame_index / self.renderer_fps,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    clock.master_time_seconds,
                    self.segment_master_start_seconds
                    + clock.segment_time_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise DynamicCropMotionAuthorityV3Error(
                    "renderer first-motion clock differs from the output frame clock"
                )
        expected_motion = bool(
            self.framing_track_applied
            and self.motion_logit_passed
            and self.physical_motion_corroborated
        )
        if self.physical_motion_corroborated and not expected_motion:
            raise DynamicCropMotionAuthorityV3Error(
                "final renderer retained physical motion without motion authority"
            )
        expected_unauthorized_motion = bool(
            self.framing_track_applied
            and self.physical_motion_corroborated
            and not self.motion_logit_passed
        )
        if (
            self.renderer_motion_without_loso_authority
            != expected_unauthorized_motion
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "unauthorized renderer-motion flag differs from its authorities"
            )
        if self.motion_event_applied != expected_motion:
            raise DynamicCropMotionAuthorityV3Error(
                "motion event is not the intersection of its three authorities"
            )
        expected_classification = (
            "within_segment_motion_event"
            if expected_motion
            else "static_or_cut_time_reframe_excluded"
            if self.framing_track_applied
            else "no_framing_track"
        )
        if self.motion_budget_classification != expected_classification:
            raise DynamicCropMotionAuthorityV3Error(
                "motion budget classification differs"
            )
        _content_addressed_payload(
            self.to_dict(),
            hash_field="evidence_content_sha256",
            label="dynamic crop motion segment evidence",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "segment_index": self.segment_index,
            "segment_master_start_seconds": self.segment_master_start_seconds,
            "segment_master_end_seconds": self.segment_master_end_seconds,
            "source_index": self.source_index,
            "source_id": self.source_id,
            "render_source_path_sha256": self.render_source_path_sha256,
            "source_start_seconds": self.source_start_seconds,
            "source_duration_seconds": self.source_duration_seconds,
            "adapter_contract": self.adapter_contract,
            "overlay_binding_contract": self.overlay_binding_contract,
            "overlay_disposition": self.overlay_disposition,
            "face_dissolve_suppressed": self.face_dissolve_suppressed,
            "decoded_shot_duration_seconds": self.decoded_shot_duration_seconds,
            "final_segment_duration_seconds": self.final_segment_duration_seconds,
            "time_reparameterization_scale": (
                self.time_reparameterization_scale
            ),
            "selected_decode_content_sha256": (
                self.selected_decode_content_sha256
            ),
            "original_decode_track_sha256": self.original_decode_track_sha256,
            "provisional_overlay_track_sha256": (
                self.provisional_overlay_track_sha256
            ),
            "rendered_track_sha256": self.rendered_track_sha256,
            "motion_suppressed_to_static_framing": (
                self.motion_suppressed_to_static_framing
            ),
            "motion_suppression_reasons": list(self.motion_suppression_reasons),
            "provisional_renderer_sample_states_sha256": (
                self.provisional_renderer_sample_states_sha256
            ),
            "provisional_zoom_range": self.provisional_zoom_range,
            "provisional_maximum_center_distance": (
                self.provisional_maximum_center_distance
            ),
            "provisional_roll_range_degrees": (
                self.provisional_roll_range_degrees
            ),
            "provisional_physical_motion_corroborated": (
                self.provisional_physical_motion_corroborated
            ),
            "framing_track_applied": self.framing_track_applied,
            "motion_logit": self.motion_logit,
            "motion_logit_threshold": self.motion_logit_threshold,
            "motion_logit_passed": self.motion_logit_passed,
            "renderer_sampling_contract": self.renderer_sampling_contract,
            "renderer_fps": self.renderer_fps,
            "renderer_frame_count": self.renderer_frame_count,
            "renderer_sample_times_sha256": self.renderer_sample_times_sha256,
            "renderer_sample_states_sha256": self.renderer_sample_states_sha256,
            "zoom_range": self.zoom_range,
            "maximum_center_distance": self.maximum_center_distance,
            "roll_range_degrees": self.roll_range_degrees,
            "zoom_range_corroborated": self.zoom_range_corroborated,
            "center_range_corroborated": self.center_range_corroborated,
            "roll_range_corroborated": self.roll_range_corroborated,
            "physical_motion_corroborated": self.physical_motion_corroborated,
            "renderer_visible_first_motion_clocks": [
                item.to_dict()
                for item in self.renderer_visible_first_motion_clocks
            ],
            "renderer_motion_without_loso_authority": (
                self.renderer_motion_without_loso_authority
            ),
            "motion_event_applied": self.motion_event_applied,
            "motion_budget_classification": self.motion_budget_classification,
            "failure_reasons": list(self.failure_reasons),
            "authority_content_sha256": self.authority_content_sha256,
            "evidence_content_sha256": self.evidence_content_sha256,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DynamicCropMotionSegmentEvidenceV3":
        material = dict(payload)
        _exact_keys(
            material, _SEGMENT_KEYS, "dynamic crop motion segment evidence"
        )
        _content_addressed_payload(
            material,
            hash_field="evidence_content_sha256",
            label="dynamic crop motion segment evidence",
        )
        reasons = material["failure_reasons"]
        if not isinstance(reasons, list) or any(
            not isinstance(item, str) or not item for item in reasons
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "motion segment failure reasons are invalid"
            )
        raw_clocks = material["renderer_visible_first_motion_clocks"]
        if not isinstance(raw_clocks, list) or any(
            not isinstance(item, Mapping) for item in raw_clocks
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "renderer first-motion clocks must be objects"
            )
        suppression_reasons = material["motion_suppression_reasons"]
        if not isinstance(suppression_reasons, list) or any(
            not isinstance(item, str) for item in suppression_reasons
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "motion suppression reasons must be strings"
            )
        return cls(
            segment_index=material["segment_index"],
            segment_master_start_seconds=_finite(
                material["segment_master_start_seconds"],
                "segment master start",
            ),
            segment_master_end_seconds=_finite(
                material["segment_master_end_seconds"],
                "segment master end",
            ),
            source_index=material["source_index"],
            source_id=material["source_id"],
            render_source_path_sha256=str(
                material["render_source_path_sha256"]
            ),
            source_start_seconds=_finite(
                material["source_start_seconds"], "source start"
            ),
            source_duration_seconds=_finite(
                material["source_duration_seconds"], "source duration"
            ),
            overlay_disposition=str(material["overlay_disposition"]),
            face_dissolve_suppressed=material["face_dissolve_suppressed"],
            decoded_shot_duration_seconds=(
                _finite(
                    material["decoded_shot_duration_seconds"],
                    "decoded shot duration",
                )
                if material["decoded_shot_duration_seconds"] is not None
                else None
            ),
            final_segment_duration_seconds=_finite(
                material["final_segment_duration_seconds"],
                "final segment duration",
            ),
            time_reparameterization_scale=(
                _finite(
                    material["time_reparameterization_scale"],
                    "time reparameterization scale",
                )
                if material["time_reparameterization_scale"] is not None
                else None
            ),
            selected_decode_content_sha256=str(
                material["selected_decode_content_sha256"]
            ),
            original_decode_track_sha256=str(
                material["original_decode_track_sha256"]
            ),
            provisional_overlay_track_sha256=(
                str(material["provisional_overlay_track_sha256"])
                if material["provisional_overlay_track_sha256"] is not None
                else None
            ),
            rendered_track_sha256=(
                str(material["rendered_track_sha256"])
                if material["rendered_track_sha256"] is not None
                else None
            ),
            motion_suppressed_to_static_framing=material[
                "motion_suppressed_to_static_framing"
            ],
            motion_suppression_reasons=tuple(suppression_reasons),
            provisional_renderer_sample_states_sha256=(
                str(material["provisional_renderer_sample_states_sha256"])
                if material["provisional_renderer_sample_states_sha256"]
                is not None
                else None
            ),
            provisional_zoom_range=_finite(
                material["provisional_zoom_range"], "provisional zoom range"
            ),
            provisional_maximum_center_distance=_finite(
                material["provisional_maximum_center_distance"],
                "provisional center range",
            ),
            provisional_roll_range_degrees=_finite(
                material["provisional_roll_range_degrees"],
                "provisional roll range",
            ),
            provisional_physical_motion_corroborated=material[
                "provisional_physical_motion_corroborated"
            ],
            framing_track_applied=material["framing_track_applied"],
            motion_logit=_finite(material["motion_logit"], "motion logit"),
            motion_logit_threshold=_finite(
                material["motion_logit_threshold"], "motion logit threshold"
            ),
            motion_logit_passed=material["motion_logit_passed"],
            renderer_fps=_finite(material["renderer_fps"], "renderer fps"),
            renderer_frame_count=material["renderer_frame_count"],
            renderer_sample_times_sha256=str(
                material["renderer_sample_times_sha256"]
            ),
            renderer_sample_states_sha256=str(
                material["renderer_sample_states_sha256"]
            ),
            zoom_range=_finite(material["zoom_range"], "zoom range"),
            maximum_center_distance=_finite(
                material["maximum_center_distance"], "center range"
            ),
            roll_range_degrees=_finite(
                material["roll_range_degrees"], "roll range"
            ),
            zoom_range_corroborated=material["zoom_range_corroborated"],
            center_range_corroborated=material["center_range_corroborated"],
            roll_range_corroborated=material["roll_range_corroborated"],
            physical_motion_corroborated=material[
                "physical_motion_corroborated"
            ],
            renderer_visible_first_motion_clocks=tuple(
                DynamicCropRendererFirstMotionClockV3.from_dict(item)
                for item in raw_clocks
            ),
            renderer_motion_without_loso_authority=material[
                "renderer_motion_without_loso_authority"
            ],
            motion_event_applied=material["motion_event_applied"],
            motion_budget_classification=str(
                material["motion_budget_classification"]
            ),
            failure_reasons=tuple(reasons),
            authority_content_sha256=str(material["authority_content_sha256"]),
            evidence_content_sha256=str(material["evidence_content_sha256"]),
            schema_version=material["schema_version"],
            contract=str(material["contract"]),
            adapter_contract=str(material["adapter_contract"]),
            overlay_binding_contract=str(material["overlay_binding_contract"]),
            renderer_sampling_contract=str(
                material["renderer_sampling_contract"]
            ),
        )


def _maximum_center_distance(
    centers: Sequence[tuple[float, float]],
) -> float:
    return max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in centers
        for right in centers
    )


def _sampled_track_ranges(
    track: EditorialTransformTrack,
    *,
    sample_times: Sequence[float],
) -> tuple[
    tuple[tuple[float, float, float, float], ...],
    float,
    float,
    float,
]:
    sampled = tuple(track.evaluate(value) for value in sample_times)
    states = tuple(
        (item.zoom, item.center_x, item.center_y, item.roll_degrees)
        for item in sampled
    )
    zooms = tuple(item[0] for item in states)
    centers = tuple((item[1], item[2]) for item in states)
    rolls = tuple(item[3] for item in states)
    return (
        states,
        max(zooms) - min(zooms),
        _maximum_center_distance(centers),
        max(rolls) - min(rolls),
    )


def apply_dynamic_crop_motion_authority_to_render_plan_v3(
    *,
    render_plan: RenderPlan,
    decodes: Sequence[DynamicCropTrackDecodeV3],
    authority: DynamicCropMotionAuthorityV3,
    deterministic_geometry_override_segment_indices: Sequence[int] = (),
) -> RenderPlan:
    """Collapse every unqualified provisional motion to exact static framing.

    This second pass runs after the ordinary StageEdit visual overlay.  It does
    not invent motion and never removes the frame-zero crop.  Only a trajectory
    that passes both the frozen logit gate and the exact renderer-sampled
    physical gate remains dynamic.
    """

    if not isinstance(render_plan, RenderPlan):
        raise DynamicCropMotionAuthorityV3Error("render plan must be typed")
    if not isinstance(authority, DynamicCropMotionAuthorityV3):
        raise DynamicCropMotionAuthorityV3Error("motion authority must be typed")
    selected = tuple(decodes)
    if len(selected) != len(render_plan.segments) or any(
        not isinstance(item, DynamicCropTrackDecodeV3) for item in selected
    ):
        raise DynamicCropMotionAuthorityV3Error(
            "one typed Dynamic Crop v3 decode is required per render segment"
        )
    deterministic_geometry_overrides = set(
        _deterministic_geometry_override_indices(
            deterministic_geometry_override_segment_indices,
            segment_count=len(render_plan.segments),
        )
    )
    frame_counts = segment_frame_counts(render_plan)
    updated = []
    thresholds = authority.physical_thresholds
    for index, (segment, decode, frame_count) in enumerate(
        zip(render_plan.segments, selected, frame_counts, strict=True)
    ):
        if decode.source_index != segment.source_index:
            raise DynamicCropMotionAuthorityV3Error(
                "dynamic crop decode source differs from rendered segment"
            )
        if decode.motion_logit is None:
            raise DynamicCropMotionAuthorityV3Error(
                "dynamic crop v3 decode lost its motion logit provenance"
            )
        overlay = _resolve_overlay_binding(
            segment_index=index,
            plan=render_plan,
            decode=decode,
            deterministic_geometry_override=(
                index in deterministic_geometry_overrides
            ),
        )
        if not overlay.framing_track_applied:
            updated.append(segment)
            continue
        assert decode.track is not None
        assert decode.shot_duration_seconds is not None
        scale = segment.duration / decode.shot_duration_seconds
        provisional = _expected_provisional_overlay_track(
            decode.track,
            scale=scale,
        )
        sample_times = tuple(frame / render_plan.fps for frame in range(frame_count))
        _, zoom_range, center_range, roll_range = _sampled_track_ranges(
            provisional,
            sample_times=sample_times,
        )
        physical_passed = bool(
            zoom_range >= thresholds.zoom_range
            or center_range >= thresholds.maximum_center_distance
            or roll_range >= thresholds.roll_range_degrees
        )
        logit_passed = float(decode.motion_logit) >= authority.motion_logit_threshold
        qualified = logit_passed and physical_passed
        if overlay.motion_suppressed_to_static_framing:
            if qualified:
                raise DynamicCropMotionAuthorityV3Error(
                    "authorized motion was suppressed to static framing"
                )
            updated.append(segment)
            continue
        if qualified:
            updated.append(segment)
            continue
        static = _static_framing_track_from_provisional(provisional)
        updated.append(replace(segment, editorial_transform=static))
    return replace(render_plan, segments=tuple(updated))


def _first_renderer_motion_clocks(
    *,
    states: Sequence[tuple[float, float, float, float]],
    fps: float,
    master_start_seconds: float,
    zoom_corroborated: bool,
    center_corroborated: bool,
    roll_corroborated: bool,
) -> tuple[DynamicCropRendererFirstMotionClockV3, ...]:
    checks = {
        "zoom": lambda left, right: round(left[0], 10) != round(right[0], 10),
        "center": lambda left, right: (
            round(left[1], 10), round(left[2], 10)
        )
        != (round(right[1], 10), round(right[2], 10)),
        "roll_degrees": lambda left, right: round(left[3], 10)
        != round(right[3], 10),
    }
    enabled = {
        "zoom": zoom_corroborated,
        "center": center_corroborated,
        "roll_degrees": roll_corroborated,
    }
    clocks: list[DynamicCropRendererFirstMotionClockV3] = []
    for component in sorted(name for name, passed in enabled.items() if passed):
        first_index = next(
            (
                index
                for index in range(1, len(states))
                if checks[component](states[index - 1], states[index])
            ),
            None,
        )
        if first_index is None:
            raise DynamicCropMotionAuthorityV3Error(
                "corroborated renderer component has no visible frame change"
            )
        clocks.append(
            DynamicCropRendererFirstMotionClockV3(
                component=component,
                frame_index=first_index,
                segment_time_seconds=first_index / fps,
                master_time_seconds=master_start_seconds + first_index / fps,
            )
        )
    return tuple(clocks)


def _segment_evidence(
    *,
    segment_index: int,
    frame_count: int,
    plan: RenderPlan,
    decode: DynamicCropTrackDecodeV3,
    authority: DynamicCropMotionAuthorityV3,
    deterministic_geometry_override: bool = False,
) -> DynamicCropMotionSegmentEvidenceV3:
    segment = plan.segments[segment_index]
    if decode.source_index != segment.source_index:
        raise DynamicCropMotionAuthorityV3Error(
            "dynamic crop decode source differs from rendered segment"
        )
    if decode.motion_logit is None:
        raise DynamicCropMotionAuthorityV3Error(
            "dynamic crop v3 decode lost its motion logit provenance"
        )
    if decode.adapter_contract != DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT:
        raise DynamicCropMotionAuthorityV3Error(
            "dynamic crop decode adapter contract differs"
        )
    overlay = _resolve_overlay_binding(
        segment_index=segment_index,
        plan=plan,
        decode=decode,
        deterministic_geometry_override=deterministic_geometry_override,
    )

    sample_times = tuple(index / plan.fps for index in range(frame_count))
    provisional_states: tuple[tuple[float, float, float, float], ...] = ()
    provisional_zoom_range = 0.0
    provisional_center_range = 0.0
    provisional_roll_range = 0.0
    provisional_states_sha256: str | None = None
    if overlay.provisional_overlay_track_sha256 is not None:
        assert decode.track is not None
        assert decode.shot_duration_seconds is not None
        provisional_track = _expected_provisional_overlay_track(
            decode.track,
            scale=segment.duration / decode.shot_duration_seconds,
        )
        (
            provisional_states,
            provisional_zoom_range,
            provisional_center_range,
            provisional_roll_range,
        ) = _sampled_track_ranges(
            provisional_track,
            sample_times=sample_times,
        )
        provisional_states_sha256 = _canonical_sha256(
            [list(item) for item in provisional_states]
        )
    states: tuple[tuple[float, float, float, float], ...]
    # A deterministic face/pan/boundary curve is part of the final RenderPlan
    # but is deliberately outside the learned Dynamic Crop motion budget.  Its
    # exact pixels are governed by the upstream geometry validator and the
    # full-plan content hash, not falsely attributed to the model logit here.
    rendered_track = (
        segment.editorial_transform if overlay.framing_track_applied else None
    )
    if rendered_track is None:
        states = ()
        zoom_range = 0.0
        center_range = 0.0
        roll_range = 0.0
    else:
        sampled = tuple(rendered_track.evaluate(value) for value in sample_times)
        states = tuple(
            (item.zoom, item.center_x, item.center_y, item.roll_degrees)
            for item in sampled
        )
        zooms = tuple(item[0] for item in states)
        centers = tuple((item[1], item[2]) for item in states)
        rolls = tuple(item[3] for item in states)
        zoom_range = max(zooms) - min(zooms)
        center_range = _maximum_center_distance(centers)
        roll_range = max(rolls) - min(rolls)

    thresholds = authority.physical_thresholds
    provisional_physical = bool(
        provisional_zoom_range >= thresholds.zoom_range
        or provisional_center_range >= thresholds.maximum_center_distance
        or provisional_roll_range >= thresholds.roll_range_degrees
    )
    zoom_corroborated = zoom_range >= thresholds.zoom_range
    center_corroborated = center_range >= thresholds.maximum_center_distance
    roll_corroborated = roll_range >= thresholds.roll_range_degrees
    physical = bool(
        zoom_corroborated or center_corroborated or roll_corroborated
    )
    first_motion_clocks = _first_renderer_motion_clocks(
        states=states,
        fps=plan.fps,
        master_start_seconds=segment.master_start,
        zoom_corroborated=zoom_corroborated,
        center_corroborated=center_corroborated,
        roll_corroborated=roll_corroborated,
    )
    logit = float(decode.motion_logit)
    logit_passed = logit >= authority.motion_logit_threshold
    suppression_reasons = (
        tuple(
            sorted(
                (
                    *(
                        ("loso_motion_logit_below_threshold",)
                        if not logit_passed
                        else ()
                    ),
                    *(
                        ("renderer_physical_range_below_threshold",)
                        if not provisional_physical
                        else ()
                    ),
                )
            )
        )
        if overlay.motion_suppressed_to_static_framing
        else ()
    )
    framing = overlay.framing_track_applied
    motion = bool(framing and logit_passed and physical)
    reasons: list[str] = []
    if not framing:
        reasons.append("no_framing_track")
    if overlay.disposition == DETERMINISTIC_GEOMETRY_OVERRIDE_DISPOSITION:
        reasons.append(DETERMINISTIC_GEOMETRY_OVERRIDE_DISPOSITION)
    if not logit_passed:
        reasons.append("motion_logit_below_loso_frozen_threshold")
    if not physical:
        reasons.append("renderer_sampled_physical_range_below_threshold")
    if overlay.motion_suppressed_to_static_framing:
        reasons.append("loso_or_physical_motion_suppressed_to_static_framing")
    classification = (
        "within_segment_motion_event"
        if motion
        else "static_or_cut_time_reframe_excluded"
        if framing
        else "no_framing_track"
    )
    unsigned = {
        "schema_version": DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION,
        "contract": DYNAMIC_CROP_MOTION_SEGMENT_EVIDENCE_CONTRACT,
        "segment_index": segment_index,
        "segment_master_start_seconds": segment.master_start,
        "segment_master_end_seconds": segment.master_start + segment.duration,
        "source_index": decode.source_index,
        "source_id": decode.source_id,
        "render_source_path_sha256": _canonical_sha256(
            str(plan.sources[segment.source_index])
        ),
        "source_start_seconds": segment.source_start,
        "source_duration_seconds": segment.effective_source_duration,
        "adapter_contract": decode.adapter_contract,
        "overlay_binding_contract": DYNAMIC_CROP_MOTION_OVERLAY_BINDING_CONTRACT,
        "overlay_disposition": overlay.disposition,
        "face_dissolve_suppressed": overlay.face_dissolve_suppressed,
        "decoded_shot_duration_seconds": decode.shot_duration_seconds,
        "final_segment_duration_seconds": segment.duration,
        "time_reparameterization_scale": (
            overlay.time_reparameterization_scale
        ),
        "selected_decode_content_sha256": (
            overlay.selected_decode_content_sha256
        ),
        "original_decode_track_sha256": overlay.original_decode_track_sha256,
        "provisional_overlay_track_sha256": (
            overlay.provisional_overlay_track_sha256
        ),
        "rendered_track_sha256": overlay.rendered_track_sha256,
        "motion_suppressed_to_static_framing": (
            overlay.motion_suppressed_to_static_framing
        ),
        "motion_suppression_reasons": list(suppression_reasons),
        "provisional_renderer_sample_states_sha256": (
            provisional_states_sha256
        ),
        "provisional_zoom_range": provisional_zoom_range,
        "provisional_maximum_center_distance": provisional_center_range,
        "provisional_roll_range_degrees": provisional_roll_range,
        "provisional_physical_motion_corroborated": provisional_physical,
        "framing_track_applied": framing,
        "motion_logit": logit,
        "motion_logit_threshold": authority.motion_logit_threshold,
        "motion_logit_passed": logit_passed,
        "renderer_sampling_contract": (
            DYNAMIC_CROP_MOTION_RENDERER_SAMPLING_CONTRACT
        ),
        "renderer_fps": plan.fps,
        "renderer_frame_count": frame_count,
        "renderer_sample_times_sha256": _canonical_sha256(list(sample_times)),
        "renderer_sample_states_sha256": _canonical_sha256(
            [list(item) for item in states]
        ),
        "zoom_range": zoom_range,
        "maximum_center_distance": center_range,
        "roll_range_degrees": roll_range,
        "zoom_range_corroborated": zoom_corroborated,
        "center_range_corroborated": center_corroborated,
        "roll_range_corroborated": roll_corroborated,
        "physical_motion_corroborated": physical,
        "renderer_visible_first_motion_clocks": [
            item.to_dict() for item in first_motion_clocks
        ],
        "renderer_motion_without_loso_authority": bool(
            framing and physical and not logit_passed
        ),
        "motion_event_applied": motion,
        "motion_budget_classification": classification,
        "failure_reasons": reasons,
        "authority_content_sha256": authority.authority_content_sha256,
        "evidence_content_sha256": None,
    }
    return DynamicCropMotionSegmentEvidenceV3.from_dict(
        {
            **unsigned,
            "evidence_content_sha256": _canonical_sha256(unsigned),
        }
    )


_TIMELINE_KEYS = {
    "schema_version",
    "contract",
    "render_plan_content_sha256",
    "authority",
    "segment_evidence",
    "segment_count",
    "framing_track_applied_count",
    "motion_event_applied_count",
    "renderer_motion_without_loso_authority_count",
    "motion_authority_consistent",
    "static_or_cut_time_reframe_excluded_count",
    "motion_event_applied_fraction",
    "minimum_motion_applied_fraction",
    "maximum_motion_applied_fraction",
    "motion_budget_within_range",
    "timeline_evidence_content_sha256",
}


@dataclass(frozen=True, slots=True)
class DynamicCropMotionTimelineEvidenceV3:
    render_plan_content_sha256: str
    authority: DynamicCropMotionAuthorityV3
    segment_evidence: tuple[DynamicCropMotionSegmentEvidenceV3, ...]
    segment_count: int
    framing_track_applied_count: int
    motion_event_applied_count: int
    renderer_motion_without_loso_authority_count: int
    motion_authority_consistent: bool
    static_or_cut_time_reframe_excluded_count: int
    motion_event_applied_fraction: float
    minimum_motion_applied_fraction: float
    maximum_motion_applied_fraction: float
    motion_budget_within_range: bool
    timeline_evidence_content_sha256: str
    schema_version: int = DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION
    contract: str = DYNAMIC_CROP_MOTION_TIMELINE_EVIDENCE_CONTRACT

    def __post_init__(self) -> None:
        if (
            self.schema_version != DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION
            or self.contract != DYNAMIC_CROP_MOTION_TIMELINE_EVIDENCE_CONTRACT
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "dynamic crop motion timeline contract differs"
            )
        if not isinstance(self.authority, DynamicCropMotionAuthorityV3):
            raise DynamicCropMotionAuthorityV3Error(
                "timeline motion authority must be typed"
            )
        for value, label in (
            (self.render_plan_content_sha256, "render plan hash"),
            (self.timeline_evidence_content_sha256, "timeline evidence hash"),
        ):
            if not _is_sha256(value):
                raise DynamicCropMotionAuthorityV3Error(f"{label} is invalid")
        for field_name in (
            "segment_count",
            "framing_track_applied_count",
            "motion_event_applied_count",
            "renderer_motion_without_loso_authority_count",
            "static_or_cut_time_reframe_excluded_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(getattr(self, field_name), field_name),
            )
        if self.segment_count != len(self.segment_evidence):
            raise DynamicCropMotionAuthorityV3Error(
                "timeline segment count differs from evidence"
            )
        if tuple(item.segment_index for item in self.segment_evidence) != tuple(
            range(self.segment_count)
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "timeline segment evidence is not contiguous"
            )
        if any(
            item.authority_content_sha256
            != self.authority.authority_content_sha256
            for item in self.segment_evidence
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "timeline segment evidence uses another motion authority"
            )
        framing_count = sum(
            item.framing_track_applied for item in self.segment_evidence
        )
        motion_count = sum(
            item.motion_event_applied for item in self.segment_evidence
        )
        unauthorized_count = sum(
            item.renderer_motion_without_loso_authority
            for item in self.segment_evidence
        )
        excluded_count = sum(
            item.motion_budget_classification
            == "static_or_cut_time_reframe_excluded"
            for item in self.segment_evidence
        )
        if (
            self.framing_track_applied_count != framing_count
            or self.motion_event_applied_count != motion_count
            or self.renderer_motion_without_loso_authority_count
            != unauthorized_count
            or self.static_or_cut_time_reframe_excluded_count != excluded_count
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "timeline motion summary differs from segment evidence"
            )
        if (
            not isinstance(self.motion_authority_consistent, bool)
            or self.motion_authority_consistent != (unauthorized_count == 0)
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "timeline motion-authority consistency verdict differs"
            )
        fraction = motion_count / self.segment_count if self.segment_count else 0.0
        if not math.isclose(
            self.motion_event_applied_fraction, fraction, abs_tol=1e-12
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "timeline motion fraction differs from segment evidence"
            )
        minimum = _finite(
            self.minimum_motion_applied_fraction, "minimum motion fraction"
        )
        maximum = _finite(
            self.maximum_motion_applied_fraction, "maximum motion fraction"
        )
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise DynamicCropMotionAuthorityV3Error(
                "timeline motion budget range is invalid"
            )
        if not isinstance(self.motion_budget_within_range, bool) or (
            self.motion_budget_within_range
            != (minimum <= fraction <= maximum)
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "timeline motion budget verdict differs"
            )
        if not self.motion_budget_within_range:
            raise DynamicCropMotionAuthorityV3Error(
                "timeline motion fraction is outside the frozen production budget"
            )
        _content_addressed_payload(
            self.to_dict(),
            hash_field="timeline_evidence_content_sha256",
            label="dynamic crop motion timeline evidence",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "render_plan_content_sha256": self.render_plan_content_sha256,
            "authority": self.authority.to_dict(),
            "segment_evidence": [item.to_dict() for item in self.segment_evidence],
            "segment_count": self.segment_count,
            "framing_track_applied_count": self.framing_track_applied_count,
            "motion_event_applied_count": self.motion_event_applied_count,
            "renderer_motion_without_loso_authority_count": (
                self.renderer_motion_without_loso_authority_count
            ),
            "motion_authority_consistent": self.motion_authority_consistent,
            "static_or_cut_time_reframe_excluded_count": (
                self.static_or_cut_time_reframe_excluded_count
            ),
            "motion_event_applied_fraction": self.motion_event_applied_fraction,
            "minimum_motion_applied_fraction": (
                self.minimum_motion_applied_fraction
            ),
            "maximum_motion_applied_fraction": (
                self.maximum_motion_applied_fraction
            ),
            "motion_budget_within_range": self.motion_budget_within_range,
            "timeline_evidence_content_sha256": (
                self.timeline_evidence_content_sha256
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DynamicCropMotionTimelineEvidenceV3":
        material = dict(payload)
        _exact_keys(
            material, _TIMELINE_KEYS, "dynamic crop motion timeline evidence"
        )
        _content_addressed_payload(
            material,
            hash_field="timeline_evidence_content_sha256",
            label="dynamic crop motion timeline evidence",
        )
        raw_authority = material["authority"]
        raw_segments = material["segment_evidence"]
        if not isinstance(raw_authority, Mapping) or not isinstance(
            raw_segments, list
        ):
            raise DynamicCropMotionAuthorityV3Error(
                "timeline motion authority or segments are malformed"
            )
        return cls(
            render_plan_content_sha256=str(material["render_plan_content_sha256"]),
            authority=DynamicCropMotionAuthorityV3.from_dict(raw_authority),
            segment_evidence=tuple(
                DynamicCropMotionSegmentEvidenceV3.from_dict(item)
                if isinstance(item, Mapping)
                else (_ for _ in ()).throw(
                    DynamicCropMotionAuthorityV3Error(
                        "timeline motion segment must be an object"
                    )
                )
                for item in raw_segments
            ),
            segment_count=material["segment_count"],
            framing_track_applied_count=material[
                "framing_track_applied_count"
            ],
            motion_event_applied_count=material["motion_event_applied_count"],
            renderer_motion_without_loso_authority_count=material[
                "renderer_motion_without_loso_authority_count"
            ],
            motion_authority_consistent=material["motion_authority_consistent"],
            static_or_cut_time_reframe_excluded_count=material[
                "static_or_cut_time_reframe_excluded_count"
            ],
            motion_event_applied_fraction=_finite(
                material["motion_event_applied_fraction"],
                "motion event fraction",
            ),
            minimum_motion_applied_fraction=_finite(
                material["minimum_motion_applied_fraction"],
                "minimum motion fraction",
            ),
            maximum_motion_applied_fraction=_finite(
                material["maximum_motion_applied_fraction"],
                "maximum motion fraction",
            ),
            motion_budget_within_range=material["motion_budget_within_range"],
            timeline_evidence_content_sha256=str(
                material["timeline_evidence_content_sha256"]
            ),
            schema_version=material["schema_version"],
            contract=str(material["contract"]),
        )


def corroborate_dynamic_crop_motion_timeline_v3(
    *,
    render_plan: RenderPlan,
    decodes: Sequence[DynamicCropTrackDecodeV3],
    authority: DynamicCropMotionAuthorityV3,
    deterministic_geometry_override_segment_indices: Sequence[int] = (),
    minimum_motion_applied_fraction: float = (
        DEFAULT_MINIMUM_MOTION_APPLIED_FRACTION
    ),
    maximum_motion_applied_fraction: float = (
        DEFAULT_MAXIMUM_MOTION_APPLIED_FRACTION
    ),
) -> DynamicCropMotionTimelineEvidenceV3:
    """Corroborate v3 motion using the exact frames the renderer will emit."""

    if not isinstance(render_plan, RenderPlan):
        raise DynamicCropMotionAuthorityV3Error("render plan must be typed")
    if not isinstance(authority, DynamicCropMotionAuthorityV3):
        raise DynamicCropMotionAuthorityV3Error("motion authority must be typed")
    selected = tuple(decodes)
    if len(selected) != len(render_plan.segments) or any(
        not isinstance(item, DynamicCropTrackDecodeV3) for item in selected
    ):
        raise DynamicCropMotionAuthorityV3Error(
            "one typed Dynamic Crop v3 decode is required per render segment"
        )
    deterministic_geometry_overrides = set(
        _deterministic_geometry_override_indices(
            deterministic_geometry_override_segment_indices,
            segment_count=len(render_plan.segments),
        )
    )
    minimum = _finite(
        minimum_motion_applied_fraction, "minimum motion applied fraction"
    )
    maximum = _finite(
        maximum_motion_applied_fraction, "maximum motion applied fraction"
    )
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise DynamicCropMotionAuthorityV3Error(
            "motion applied fraction budget must lie in [0, 1]"
        )
    frame_counts = segment_frame_counts(render_plan)
    evidence = tuple(
        _segment_evidence(
            segment_index=index,
            frame_count=frame_count,
            plan=render_plan,
            decode=decode,
            authority=authority,
            deterministic_geometry_override=(
                index in deterministic_geometry_overrides
            ),
        )
        for index, (frame_count, decode) in enumerate(
            zip(frame_counts, selected, strict=True)
        )
    )
    framing_count = sum(item.framing_track_applied for item in evidence)
    motion_count = sum(item.motion_event_applied for item in evidence)
    unauthorized_count = sum(
        item.renderer_motion_without_loso_authority for item in evidence
    )
    excluded_count = sum(
        item.motion_budget_classification
        == "static_or_cut_time_reframe_excluded"
        for item in evidence
    )
    fraction = motion_count / len(evidence) if evidence else 0.0
    unsigned = {
        "schema_version": DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION,
        "contract": DYNAMIC_CROP_MOTION_TIMELINE_EVIDENCE_CONTRACT,
        "render_plan_content_sha256": _canonical_sha256(render_plan.to_json()),
        "authority": authority.to_dict(),
        "segment_evidence": [item.to_dict() for item in evidence],
        "segment_count": len(evidence),
        "framing_track_applied_count": framing_count,
        "motion_event_applied_count": motion_count,
        "renderer_motion_without_loso_authority_count": unauthorized_count,
        "motion_authority_consistent": unauthorized_count == 0,
        "static_or_cut_time_reframe_excluded_count": excluded_count,
        "motion_event_applied_fraction": fraction,
        "minimum_motion_applied_fraction": minimum,
        "maximum_motion_applied_fraction": maximum,
        "motion_budget_within_range": minimum <= fraction <= maximum,
        "timeline_evidence_content_sha256": None,
    }
    return DynamicCropMotionTimelineEvidenceV3.from_dict(
        {
            **unsigned,
            "timeline_evidence_content_sha256": _canonical_sha256(unsigned),
        }
    )


__all__ = [
    "DEFAULT_DYNAMIC_CENTER_RANGE_THRESHOLD",
    "DEFAULT_DYNAMIC_ROLL_RANGE_THRESHOLD_DEGREES",
    "DEFAULT_DYNAMIC_ZOOM_RANGE_THRESHOLD",
    "DEFAULT_MAXIMUM_MOTION_APPLIED_FRACTION",
    "DEFAULT_MINIMUM_MOTION_APPLIED_FRACTION",
    "DEFAULT_MOTION_LOGIT_THRESHOLD",
    "DETERMINISTIC_GEOMETRY_OVERRIDE_DISPOSITION",
    "DYNAMIC_CROP_MOTION_AUTHORITY_CONTRACT",
    "DYNAMIC_CROP_MOTION_AUTHORITY_SCHEMA_VERSION",
    "DYNAMIC_CROP_MOTION_PHYSICAL_THRESHOLD_CONTRACT",
    "DYNAMIC_CROP_MOTION_OVERLAY_BINDING_CONTRACT",
    "DYNAMIC_CROP_MOTION_RENDERER_SAMPLING_CONTRACT",
    "DYNAMIC_CROP_MOTION_SEGMENT_EVIDENCE_CONTRACT",
    "DYNAMIC_CROP_MOTION_THRESHOLD_FREEZE_CONTRACT",
    "DYNAMIC_CROP_MOTION_TIMELINE_EVIDENCE_CONTRACT",
    "DynamicCropMotionAuthorityV3",
    "DynamicCropMotionAuthorityV3Error",
    "DynamicCropMotionPhysicalThresholdsV3",
    "DynamicCropRendererFirstMotionClockV3",
    "DynamicCropMotionSegmentEvidenceV3",
    "DynamicCropMotionTimelineEvidenceV3",
    "apply_dynamic_crop_motion_authority_to_render_plan_v3",
    "corroborate_dynamic_crop_motion_timeline_v3",
    "load_dynamic_crop_motion_authority_from_registered_identity_v3",
    "load_dynamic_crop_motion_authority_v3",
]
