"""One authoritative Python/OpenCV runtime contract for the app and doctor."""

from __future__ import annotations

import importlib
import platform
import re
import sys
from typing import Any


MINIMUM_PYTHON_VERSION = (3, 11)
MINIMUM_NUMPY_VERSION = (2, 0)
MINIMUM_OPENCV_VERSION = (4, 10)
TRACKER_SELECTION_ORDER = ("CSRT", "KCF", "MIL")


def _version_text(value: tuple[int, ...]) -> str:
    return ".".join(str(component) for component in value)


def _numeric_version(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    components = re.findall(r"\d+", value)
    if len(components) < 2:
        return None
    return tuple(int(component) for component in components)


def version_at_least(value: object, minimum: tuple[int, ...]) -> bool:
    """Compare dotted runtime versions without adding a packaging dependency."""

    parsed = _numeric_version(value)
    if parsed is None:
        return False
    width = max(len(parsed), len(minimum))
    padded_value = parsed + (0,) * (width - len(parsed))
    padded_minimum = minimum + (0,) * (width - len(minimum))
    return padded_value >= padded_minimum


def _module_status(name: str, minimum: tuple[int, ...]) -> dict[str, Any]:
    version: str | None = None
    import_error: str | None = None
    try:
        module = importlib.import_module(name)
        raw_version = getattr(module, "__version__", None)
        version = str(raw_version) if raw_version is not None else None
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"
    ready = import_error is None and version_at_least(version, minimum)
    return {
        "available": import_error is None,
        "version": version,
        "minimum_version": _version_text(minimum),
        "version_ready": ready,
        "import_error": import_error,
    }


def _stage_edit_generation_status() -> dict[str, Any]:
    """Report architecture and the verified machine-installed activation."""

    import_error: str | None = None
    mlx_version: str | None = None
    installed_bundle: dict[str, Any] | None = None
    registry_error: str | None = None
    try:
        from .stage_edit_head_materializer import (
            materialize_stage_edit_head_frames_v2 as _materialize,
        )
        from .stage_edit_semimarkov_beam import (
            decode_stage_edit_semimarkov_beam as _decode,
        )
        from .stage_edit_transformer import mlx_runtime_version
        from .stage_edit_production_registry import (
            discover_registered_stage_edit_production_bundle,
        )

        # Keep the public bridge imports live in this contract probe.  Merely
        # importing them never loads a checkpoint or changes pipeline policy.
        del _materialize, _decode
        mlx_version = mlx_runtime_version()
        try:
            bundle = discover_registered_stage_edit_production_bundle()
            installed_bundle = bundle.to_dict() if bundle is not None else None
        except Exception as exc:
            registry_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"
    architecture_available = import_error is None and mlx_version is not None
    return {
        "architecture_available": architecture_available,
        "mlx_version": mlx_version,
        "import_error": import_error,
        "installed_production_bundle": installed_bundle,
        "production_registry_error": registry_error,
        "automatic_group_generation_ready": bool(
            architecture_available
            and installed_bundle is not None
            and registry_error is None
        ),
        "default_activation": (
            "automatic_when_registered_production_bundle_is_verified"
        ),
        "activation_requires": [
            "verified_machine_local_production_registry",
            "feature_snapshot_materialized_from_current_synchronized_media",
            "content_bound_source_ids_aligned_with_request_sources",
            "schema_bound_recent_edit_slice_fully_missing_masked_train_and_serve",
            "duration_aware_semimarkov_owns_decision_history",
        ],
        "requested_failure_policy": "fail_closed_no_silent_visual_fallback",
        "candidate_policy": (
            "retain_parallel_unrendered_until_complete_proxy_review"
        ),
    }


def runtime_contract_report() -> dict[str, Any]:
    """Probe package versions and real member-tracker execution locally."""

    python_ready = sys.version_info[:2] >= MINIMUM_PYTHON_VERSION
    python_status = {
        "available": True,
        "version": platform.python_version(),
        "minimum_version": _version_text(MINIMUM_PYTHON_VERSION),
        "version_ready": python_ready,
    }
    numpy_status = _module_status("numpy", MINIMUM_NUMPY_VERSION)
    opencv_status = _module_status("cv2", MINIMUM_OPENCV_VERSION)
    base_ready = bool(
        python_ready
        and numpy_status["version_ready"]
        and opencv_status["version_ready"]
    )

    tracker_backends: tuple[str, ...] = ()
    tracker_error: str | None = None
    if base_ready:
        try:
            from .tracking import available_opencv_trackers

            tracker_backends = available_opencv_trackers()
        except Exception as exc:
            tracker_error = f"{type(exc).__name__}: {exc}"
    tracker_ready = bool(tracker_backends)
    member_ready = base_ready and tracker_ready
    tracker_status = {
        "ready": tracker_ready,
        "backends": list(tracker_backends),
        "selection_order": list(TRACKER_SELECTION_ORDER),
        "verification": "factory_create_init_update_tiny_frame",
        "scope": "user_anchor_forward_continuous_track_only",
        "required_for": ["local_member"],
        "optional_for": [
            "local_group",
            "automatic_group",
            "automatic_member",
        ],
        "fallback": "local_group_and_ready_automatic_modes_remain_available",
        "probe_error": tracker_error,
    }
    return {
        "python": python_status,
        "numpy": numpy_status,
        "opencv": opencv_status,
        "base_runtime_ready": base_ready,
        "member_anchor_tracking": tracker_status,
        "stage_edit_generation": _stage_edit_generation_status(),
        "mode_readiness": {
            "group": base_ready,
            "member": member_ready,
        },
        # This low-level composite means the base runtime and confirmed-anchor
        # local member tracker are both ready. The native locator intentionally
        # requires only base_runtime_ready; doctor.py composes product-specific
        # local/automatic readiness with media tools and visual model suites.
        "full_ready": member_ready,
    }


__all__ = [
    "MINIMUM_NUMPY_VERSION",
    "MINIMUM_OPENCV_VERSION",
    "MINIMUM_PYTHON_VERSION",
    "TRACKER_SELECTION_ORDER",
    "runtime_contract_report",
    "version_at_least",
]
