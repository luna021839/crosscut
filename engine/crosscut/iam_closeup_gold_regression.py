"""Scalar-only I AM closeup-cut gold regression evidence.

This artifact preserves two user-approved boundaries from the local I AM
51--57 second review window.  It is deliberately an immutable regression
exemplar, not training data, calibration authority, a production-selection
input, or a hard gate.  Only derived scalars and public source attribution are
stored; raw pixels, pose vectors, appearance embeddings, local paths, and face
identity claims are forbidden.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pose import POSE_BACKEND
from .professional_motion_cut_profile import (
    MOTION_RECOMMENDATION_CONTRACT,
    PROFILE_ANALYSIS_BASIS as PROFESSIONAL_MOTION_PROFILE_ANALYSIS_BASIS,
    MotionCutSoftPreferenceRecommendation,
)
from .visual_score import (
    YUNET_EXPECTED_SHA256,
    YUNET_MODEL_NAME,
)


SCHEMA_VERSION = 1
CONTRACT = "user_approved_scalar_only_closeup_boundary_regression_exemplar_v1"
ANALYSIS_BASIS = "iam_local_51_57s_exact_endpoint_scalar_regression_v1"
CONTENT_HASH_ALGORITHM = (
    "sha256_of_canonical_json_with_provenance_content_sha256_null"
)
ASSET_PATH = Path(__file__).with_name("models") / (
    "iam_closeup_gold_regression_v1.json"
)
ASSET_FILE_SHA256 = (
    "28d80e5dc09faae6274d7242cac41e5bd06d6bbd1bd52d5f8b7345aa6e26f8a7"
)

EXPECTED_VIDEO_SHA256 = (
    "cfe23a5f4053484f379559c3f6bcb754593bdbd5c9dd3581b33e270ec190c2c5"
)
EXPECTED_RENDER_PLAN_SHA256 = (
    "69240e1b5431ca6f80febe424587ceabd0a60d40b1e548f40945c01526b1a6d8"
)
EXPECTED_ANALYSIS_SHA256 = (
    "bd41d295704fcaee449d12371c093b72b08a7c7455342d1234f909c8384826e7"
)
EXPECTED_FPS = 29.97
PROFESSIONAL_MOTION_PROFILE_FILE_SHA256 = (
    "005a21f79ab1e0b5de76d3299ca2f61b54eb460b88cfb2f88dd392d93e9419b6"
)
PROFESSIONAL_MOTION_PROFILE_CONTENT_SHA256 = (
    "9f70c549acbda2e80f3497b57c189ddbb5df4e4bbd4f5eda9095c5af699ae680"
)
PROFESSIONAL_BOTH_MOVING_MINIMUM = 0.0002705025925969218
PROFESSIONAL_SAME_TIME_POSE_SOFT_QUALIFIER = 0.8486102174679628
PROFESSIONAL_CONTINUITY_MINIMUM = -0.49213925079872173

_SHA256_LENGTH = 64
_EXPECTED_SCOPE = {
    "artist": "IVE",
    "song": "I AM",
    "review_window_start_seconds": 51.0,
    "review_window_end_seconds": 57.0,
    "fps": EXPECTED_FPS,
    "video_sha256": EXPECTED_VIDEO_SHA256,
    "render_plan_sha256": EXPECTED_RENDER_PLAN_SHA256,
    "analysis_sha256": EXPECTED_ANALYSIS_SHA256,
}
_EXPECTED_POLICY = {
    "artifact_role": "regression_exemplar",
    "user_approved_positive": True,
    "calibration_authority": False,
    "hard_gate": False,
    "production_selection_authority": False,
    "may_activate_direction_threshold": False,
}
_EXPECTED_PRIVACY = {
    "derived_scalars_only": True,
    "raw_vectors_stored": False,
    "raw_pixels_stored": False,
    "local_absolute_paths_stored": False,
    "appearance_embeddings_stored": False,
    "face_identity_claimed": False,
}
_EXPECTED_MEASUREMENT_CONTRACT = {
    "shot_type": "closeup",
    "face_detector_backend": "yunet",
    "face_detector_model_name": YUNET_MODEL_NAME,
    "face_detector_model_sha256": YUNET_EXPECTED_SHA256,
    "pose_endpoint_backend": POSE_BACKEND,
    "both_endpoints_require_exact_backends": True,
    "face_height_ratio_semantics": "incoming_face_height_over_outgoing_face_height",
    "center_delta_semantics": "normalized_output_frame_euclidean_distance",
    "endpoint_pose_cosine_semantics": (
        "identity_free_same_time_endpoint_pose_shape_cosine_not_motion_direction"
    ),
    "motion_direction_cosine_semantics": "audit_only_unbound",
    "motion_phase_semantics": "derived_scalar_audit_only",
}
_EXPECTED_PROFILE_REFERENCE = {
    "schema_version": 1,
    "analysis_basis": PROFESSIONAL_MOTION_PROFILE_ANALYSIS_BASIS,
    "file_sha256": PROFESSIONAL_MOTION_PROFILE_FILE_SHA256,
    "content_sha256": PROFESSIONAL_MOTION_PROFILE_CONTENT_SHA256,
    "recommendation_contract": MOTION_RECOMMENDATION_CONTRACT,
    "both_moving_minimum_pose_motion": PROFESSIONAL_BOTH_MOVING_MINIMUM,
    "continuity_compatible_minimum_score": PROFESSIONAL_CONTINUITY_MINIMUM,
    "same_time_pose_soft_qualifier_cosine": (
        PROFESSIONAL_SAME_TIME_POSE_SOFT_QUALIFIER
    ),
    "same_time_pose_is_hard_threshold": False,
    "direction_threshold_bound": False,
    "direction_used_for_activation": False,
}
_EXPECTED_BOUNDARIES = (
    {
        "frame_index": 1607,
        "master_time_seconds": 53.6202869536,
        "classification": "user_approved_positive_cross_broadcaster_closeup",
        "outgoing": {
            "source_index": 8,
            "youtube_video_id": "5MGr54kHBQk",
            "broadcaster": "Mnet",
        },
        "incoming": {
            "source_index": 6,
            "youtube_video_id": "vMXGX4Dp3iM",
            "broadcaster": "MBC",
        },
        "endpoint_backends": {
            "outgoing_face_detector": "yunet",
            "incoming_face_detector": "yunet",
            "outgoing_pose_backend": POSE_BACKEND,
            "incoming_pose_backend": POSE_BACKEND,
        },
        "metrics": {
            "face_height_ratio": 0.9582126995,
            "center_delta": 0.0734785017,
            "endpoint_pose_cosine": 0.9519863859,
            "outgoing_pose_motion": 3.8699323369,
            "incoming_pose_motion": 4.7729826637,
            "motion_direction_cosine": -0.0451399044,
            "motion_phase": 0.4774300478,
        },
    },
    {
        "frame_index": 1640,
        "master_time_seconds": 54.7213880547,
        "classification": "user_approved_positive_cross_broadcaster_closeup",
        "outgoing": {
            "source_index": 6,
            "youtube_video_id": "vMXGX4Dp3iM",
            "broadcaster": "MBC",
        },
        "incoming": {
            "source_index": 1,
            "youtube_video_id": "dus7C4CTWUk",
            "broadcaster": "Mnet",
        },
        "endpoint_backends": {
            "outgoing_face_detector": "yunet",
            "incoming_face_detector": "yunet",
            "outgoing_pose_backend": POSE_BACKEND,
            "incoming_pose_backend": POSE_BACKEND,
        },
        "metrics": {
            "face_height_ratio": 1.0021320687,
            "center_delta": 0.0601054621,
            "endpoint_pose_cosine": 0.8722635848,
            "outgoing_pose_motion": 7.3771830875,
            "incoming_pose_motion": 4.3790995944,
            "motion_direction_cosine": -0.2934741370,
            "motion_phase": 0.3532629315,
        },
    },
)


class IAmCloseupGoldRegressionError(ValueError):
    """Raised when the gold exemplar or its intended use is not exact."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_sha256(payload: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(payload))
    provenance = normalized.get("provenance")
    if not isinstance(provenance, dict):
        raise IAmCloseupGoldRegressionError("gold provenance must be an object")
    provenance["content_sha256"] = None
    return _canonical_sha256(normalized)


def _sha256(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise IAmCloseupGoldRegressionError(f"{name} must be lowercase SHA-256")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IAmCloseupGoldRegressionError(f"{name} must be an object")
    return value


def _exact_mapping(
    value: object,
    expected: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, name)
    if dict(mapping) != dict(expected):
        raise IAmCloseupGoldRegressionError(f"{name} changed")
    return mapping


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise IAmCloseupGoldRegressionError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _assert_scalar_only(value: object, *, path: str = "gold") -> None:
    """Reject fields that could smuggle raw media, vectors, or local paths."""

    forbidden_key_fragments = (
        "raw_vector",
        "raw_pixel",
        "embedding_vector",
        "appearance_embedding",
        "local_path",
        "absolute_path",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in forbidden_key_fragments):
                # Explicit false-valued privacy declarations are the only
                # permitted mentions of storage that the schema binds.
                if not (path == "gold.privacy" and item is False):
                    raise IAmCloseupGoldRegressionError(
                        f"{path}.{key} is forbidden in scalar-only evidence"
                    )
            _assert_scalar_only(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_scalar_only(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and (
        value.startswith(("/", "~/", "file://"))
        or ":\\" in value
    ):
        raise IAmCloseupGoldRegressionError(
            f"{path} must not contain a local absolute path"
        )


@dataclass(frozen=True, slots=True)
class GoldCloseupBoundary:
    frame_index: int
    master_time_seconds: float
    outgoing_source_index: int
    incoming_source_index: int
    face_height_ratio: float
    center_delta: float
    endpoint_pose_cosine: float
    outgoing_pose_motion: float
    incoming_pose_motion: float
    motion_direction_cosine: float
    motion_phase: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GoldCloseupBoundary":
        outgoing = _mapping(value["outgoing"], "gold outgoing source")
        incoming = _mapping(value["incoming"], "gold incoming source")
        metrics = _mapping(value["metrics"], "gold boundary metrics")
        return cls(
            frame_index=int(value["frame_index"]),
            master_time_seconds=float(value["master_time_seconds"]),
            outgoing_source_index=int(outgoing["source_index"]),
            incoming_source_index=int(incoming["source_index"]),
            face_height_ratio=float(metrics["face_height_ratio"]),
            center_delta=float(metrics["center_delta"]),
            endpoint_pose_cosine=float(metrics["endpoint_pose_cosine"]),
            outgoing_pose_motion=float(metrics["outgoing_pose_motion"]),
            incoming_pose_motion=float(metrics["incoming_pose_motion"]),
            motion_direction_cosine=float(metrics["motion_direction_cosine"]),
            motion_phase=float(metrics["motion_phase"]),
        )


@dataclass(frozen=True, slots=True)
class GoldSoftQualifierResult:
    frame_index: int
    both_moving_passed: bool
    same_time_pose_shape_passed: bool
    combined_soft_components_passed: bool
    direction_cosine_audit_value: float
    direction_threshold_bound: bool = False
    direction_used_for_activation: bool = False


class IAmCloseupGoldRegression:
    """Strict view of the immutable, non-authoritative gold exemplar."""

    def __init__(self, payload: Mapping[str, Any], *, file_sha256: str) -> None:
        self._payload = deepcopy(dict(payload))
        self._file_sha256 = file_sha256
        self._validate()

    def _validate(self) -> None:
        payload = self._payload
        if set(payload) != {
            "schema_version",
            "contract",
            "analysis_basis",
            "scope",
            "artifact_policy",
            "privacy",
            "measurement_contract",
            "professional_motion_profile_reference",
            "boundaries",
            "provenance",
        }:
            raise IAmCloseupGoldRegressionError("gold exemplar schema changed")
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("contract") != CONTRACT
            or payload.get("analysis_basis") != ANALYSIS_BASIS
        ):
            raise IAmCloseupGoldRegressionError("gold exemplar identity changed")
        # Authenticate the complete payload before interpreting any measured
        # scalar.  A caller that supplies an independently verified file hash
        # therefore still receives a distinct content-hash failure when the
        # JSON was modified without resealing its self-binding.
        provenance = _mapping(payload.get("provenance"), "gold provenance")
        if set(provenance) != {
            "content_hash_algorithm",
            "content_sha256",
            "source_hashes_bound",
            "user_approval_bound",
        }:
            raise IAmCloseupGoldRegressionError("gold provenance schema changed")
        if (
            provenance.get("content_hash_algorithm") != CONTENT_HASH_ALGORITHM
            or provenance.get("source_hashes_bound") is not True
            or provenance.get("user_approval_bound") is not True
        ):
            raise IAmCloseupGoldRegressionError("gold provenance changed")
        stored_content_sha = _sha256(
            provenance.get("content_sha256"), "gold content SHA-256"
        )
        if stored_content_sha != _content_sha256(payload):
            raise IAmCloseupGoldRegressionError("gold content SHA-256 mismatch")
        _exact_mapping(payload.get("scope"), _EXPECTED_SCOPE, "gold scope")
        _exact_mapping(
            payload.get("artifact_policy"),
            _EXPECTED_POLICY,
            "gold regression-exemplar policy",
        )
        _exact_mapping(payload.get("privacy"), _EXPECTED_PRIVACY, "gold privacy")
        _exact_mapping(
            payload.get("measurement_contract"),
            _EXPECTED_MEASUREMENT_CONTRACT,
            "gold measurement contract",
        )
        _exact_mapping(
            payload.get("professional_motion_profile_reference"),
            _EXPECTED_PROFILE_REFERENCE,
            "gold professional motion-profile reference",
        )
        boundaries = payload.get("boundaries")
        if not isinstance(boundaries, list) or boundaries != list(
            _EXPECTED_BOUNDARIES
        ):
            raise IAmCloseupGoldRegressionError("gold boundary scalars changed")
        for boundary in boundaries:
            frame = int(boundary["frame_index"])
            timestamp = float(boundary["master_time_seconds"])
            if not math.isclose(
                timestamp,
                frame / EXPECTED_FPS,
                rel_tol=0.0,
                abs_tol=5.0e-11,
            ):
                raise IAmCloseupGoldRegressionError(
                    "gold frame and master time disagree"
                )
            if boundary["outgoing"]["broadcaster"] == boundary["incoming"][
                "broadcaster"
            ]:
                raise IAmCloseupGoldRegressionError(
                    "gold boundary must remain cross-broadcaster"
                )
            if float(boundary["metrics"]["motion_direction_cosine"]) >= 0.0:
                raise IAmCloseupGoldRegressionError(
                    "gold negative direction audit evidence changed"
                )
        _assert_scalar_only(payload)

    @property
    def file_sha256(self) -> str:
        return self._file_sha256

    @property
    def content_sha256(self) -> str:
        return str(self._payload["provenance"]["content_sha256"])

    @property
    def boundaries(self) -> tuple[GoldCloseupBoundary, ...]:
        return tuple(
            GoldCloseupBoundary.from_mapping(item)
            for item in self._payload["boundaries"]
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def evaluate_same_time_pose_and_both_moving_soft_qualifiers(
        self,
        recommendation: MotionCutSoftPreferenceRecommendation,
    ) -> tuple[GoldSoftQualifierResult, ...]:
        """Check only the profile's same-time-pose and both-moving soft terms.

        Continuity is not present in this bounded exemplar and is therefore
        neither invented nor evaluated.  Direction remains audit-only even
        though both approved examples have negative direction cosine.
        """

        if (
            recommendation.contract != MOTION_RECOMMENDATION_CONTRACT
            or recommendation.hard_gate is not False
            or recommendation.pose_phase_hard_threshold is not None
            or not math.isclose(
                recommendation.both_moving_minimum_pose_motion,
                PROFESSIONAL_BOTH_MOVING_MINIMUM,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                recommendation.continuity_compatible_minimum_score,
                PROFESSIONAL_CONTINUITY_MINIMUM,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                recommendation.pose_phase_compatible_preferred_cosine,
                PROFESSIONAL_SAME_TIME_POSE_SOFT_QUALIFIER,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise IAmCloseupGoldRegressionError(
                "recommendation differs from the bound immutable professional "
                "motion profile"
            )
        results: list[GoldSoftQualifierResult] = []
        for boundary in self.boundaries:
            both_moving = (
                boundary.outgoing_pose_motion
                >= recommendation.both_moving_minimum_pose_motion
                and boundary.incoming_pose_motion
                >= recommendation.both_moving_minimum_pose_motion
            )
            pose_shape = (
                boundary.endpoint_pose_cosine
                >= recommendation.pose_phase_compatible_preferred_cosine
            )
            results.append(
                GoldSoftQualifierResult(
                    frame_index=boundary.frame_index,
                    both_moving_passed=both_moving,
                    same_time_pose_shape_passed=pose_shape,
                    combined_soft_components_passed=both_moving and pose_shape,
                    direction_cosine_audit_value=(
                        boundary.motion_direction_cosine
                    ),
                )
            )
        return tuple(results)


def load_iam_closeup_gold_regression(
    path: Path = ASSET_PATH,
    *,
    expected_file_sha256: str = ASSET_FILE_SHA256,
) -> IAmCloseupGoldRegression:
    """Load the exemplar only after exact file and content verification."""

    source = Path(path).expanduser()
    if not source.is_absolute():
        raise IAmCloseupGoldRegressionError(
            "gold exemplar path must be absolute"
        )
    if source.is_symlink() or not source.is_file():
        raise IAmCloseupGoldRegressionError(
            "gold exemplar must be an existing non-symlink regular file"
        )
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise IAmCloseupGoldRegressionError(
            f"could not read gold exemplar: {exc}"
        ) from exc
    expected = _sha256(expected_file_sha256, "expected gold file SHA-256")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise IAmCloseupGoldRegressionError("gold exemplar file SHA-256 mismatch")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                IAmCloseupGoldRegressionError(
                    f"gold exemplar contains non-finite JSON number {token}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IAmCloseupGoldRegressionError(
            f"could not decode gold exemplar: {exc}"
        ) from exc
    return IAmCloseupGoldRegression(
        _mapping(payload, "gold exemplar"),
        file_sha256=actual,
    )


__all__ = [
    "ANALYSIS_BASIS",
    "ASSET_FILE_SHA256",
    "ASSET_PATH",
    "CONTRACT",
    "GoldCloseupBoundary",
    "GoldSoftQualifierResult",
    "IAmCloseupGoldRegression",
    "IAmCloseupGoldRegressionError",
    "PROFESSIONAL_BOTH_MOVING_MINIMUM",
    "PROFESSIONAL_CONTINUITY_MINIMUM",
    "PROFESSIONAL_SAME_TIME_POSE_SOFT_QUALIFIER",
    "SCHEMA_VERSION",
    "load_iam_closeup_gold_regression",
]
