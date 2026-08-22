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
        "required_for": ["member_mode"],
        "optional_for": ["group_mode"],
        "fallback": "group_mode_available_member_render_unavailable",
        "probe_error": tracker_error,
    }
    return {
        "python": python_status,
        "numpy": numpy_status,
        "opencv": opencv_status,
        "base_runtime_ready": base_ready,
        "member_anchor_tracking": tracker_status,
        "mode_readiness": {
            "group": base_ready,
            "member": member_ready,
        },
        # The native app exposes both modes behind one engine-ready state, so
        # its strict locator and the doctor's top-level ready require both.
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
