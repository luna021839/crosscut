"""Local runtime diagnostics for required media tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import shutil
import subprocess

from .runtime_contract import runtime_contract_report


@dataclass(frozen=True, slots=True)
class ToolStatus:
    name: str
    available: bool
    path: str | None
    version: str | None


def _command_status(name: str, version_args: tuple[str, ...] = ("-version",)) -> ToolStatus:
    path = shutil.which(name)
    if path is None:
        return ToolStatus(name=name, available=False, path=None, version=None)
    try:
        completed = subprocess.run(
            (path, *version_args),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = (completed.stdout or completed.stderr).splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        first_line = None
    return ToolStatus(name=name, available=True, path=path, version=first_line)


def runtime_report() -> dict[str, object]:
    commands = [
        _command_status("ffmpeg"),
        _command_status("ffprobe"),
        _command_status("yt-dlp", ("--version",)),
    ]
    runtime_contract = runtime_contract_report()
    modules = {
        "numpy": bool(runtime_contract["numpy"]["version_ready"]),
        "cv2": bool(runtime_contract["opencv"]["version_ready"]),
    }
    pose_assets: list[dict[str, object]] = []
    pose_silhouette_ready = False
    face_asset: dict[str, object] = {}
    face_match_ready = False
    face_identity_asset: dict[str, object] = {}
    tier_a_face_appearance_ready = False
    if runtime_contract["base_runtime_ready"]:
        try:
            from .pose import model_asset_status

            statuses = model_asset_status(verify_load=True)
            pose_assets = [item.to_dict() for item in statuses]
            pose_silhouette_ready = all(
                item.digest_valid and item.loadable is True for item in statuses
            )
        except (ImportError, OSError, RuntimeError):
            # Pose/silhouette improves group selection but is deliberately not a
            # hard runtime requirement; the HOG structure fallback remains valid.
            pass
        try:
            from .visual_score import yunet_model_asset_status

            face_asset = yunet_model_asset_status(verify_load=True)
            face_match_ready = bool(
                face_asset.get("digest_valid")
                and face_asset.get("loadable") is True
            )
        except (ImportError, OSError, RuntimeError):
            # Face alignment is also optional. Haar can still describe shot scale,
            # while ambiguous/missing geometry stays an explicit raw hard cut.
            pass
        try:
            from .face_identity import sface_model_asset_status

            identity_status = sface_model_asset_status(verify_load=True)
            face_identity_asset = identity_status.to_dict()
            tier_a_face_appearance_ready = bool(
                identity_status.digest_valid
                and identity_status.loadable is True
                and identity_status.embedding_size == 128
            )
        except (AttributeError, ImportError, OSError, RuntimeError, ValueError):
            # SFace is a Tier-A-only anti-ghosting appearance-consistency gate.
            # If it is unavailable, synchronized B/C hard cuts remain valid.
            pass
    media_tools_ready = all(item.available for item in commands[:2])
    runtime_modes = runtime_contract["mode_readiness"]
    group_mode_ready = bool(media_tools_ready and runtime_modes["group"])
    member_mode_ready = bool(media_tools_ready and runtime_modes["member"])
    # The native app exposes group and member modes behind one engine-ready
    # state. Keep top-level ready strict while still reporting that a partial
    # runtime can remain usable for group edits.
    required_ready = member_mode_ready
    return {
        "ready": required_ready,
        "mode_readiness": {
            "group": group_mode_ready,
            "member": member_mode_ready,
        },
        "commands": [asdict(item) for item in commands],
        "python_modules": modules,
        "runtime_contract": runtime_contract,
        "mode_capabilities": {
            # This is optional only for group mode and is a hard requirement
            # for member mode, so it must not be labeled globally optional.
            "member_anchor_tracking": runtime_contract[
                "member_anchor_tracking"
            ],
        },
        "optional_features": {
            "group_pose_silhouette": {
                "ready": pose_silhouette_ready,
                "models": pose_assets,
                "fallback": "structure_hog",
            },
            "group_face_match": {
                "ready": face_match_ready,
                "model": face_asset,
                "fallback": "haar_bbox_only_or_raw_hard_cut",
            },
            "tier_a_face_appearance": {
                "ready": tier_a_face_appearance_ready,
                "model": face_identity_asset,
                "scope": "tier_a_dissolve_eligibility_only",
                "fallback": "tier_b_or_tier_c_analyzed_hard_cut",
            },
        },
    }
