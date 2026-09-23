"""Local runtime diagnostics for required media tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import shutil
import subprocess

from .runtime_contract import runtime_contract_report
from .ytdlp_runtime import discover_ytdlp_frontend


@dataclass(frozen=True, slots=True)
class ToolStatus:
    name: str
    available: bool
    path: str | None
    version: str | None
    command_succeeded: bool = False


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
        command_succeeded = completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired, IndexError):
        first_line = None
        command_succeeded = False
    return ToolStatus(
        name=name,
        available=True,
        path=path,
        version=first_line,
        command_succeeded=command_succeeded,
    )


def runtime_report() -> dict[str, object]:
    commands = [
        _command_status("ffmpeg"),
        _command_status("ffprobe"),
        _command_status("yt-dlp", ("--version",)),
    ]
    selected_ytdlp = discover_ytdlp_frontend()
    runtime_contract = runtime_contract_report()
    modules = {
        "numpy": bool(runtime_contract["numpy"]["version_ready"]),
        "cv2": bool(runtime_contract["opencv"]["version_ready"]),
    }
    pose_assets: list[dict[str, object]] = []
    pose_silhouette_ready = False
    automatic_member_person_detector_asset: dict[str, object] = {}
    automatic_member_person_detector_ready = False
    automatic_member_visual_continuity_ready = False
    face_asset: dict[str, object] = {}
    face_match_ready = False
    face_identity_asset: dict[str, object] = {}
    tier_a_face_appearance_ready = False
    if runtime_contract["base_runtime_ready"]:
        try:
            from .pose import PERSON_DETECTOR_MODEL, model_asset_status

            statuses = model_asset_status(verify_load=True)
            pose_assets = [item.to_dict() for item in statuses]
            pose_silhouette_ready = bool(statuses) and all(
                item.digest_valid and item.loadable is True for item in statuses
            )
            person_detector_status = next(
                (
                    item
                    for item in statuses
                    if item.name == PERSON_DETECTOR_MODEL
                ),
                None,
            )
            if person_detector_status is not None:
                automatic_member_person_detector_asset = (
                    person_detector_status.to_dict()
                )
                automatic_member_person_detector_ready = bool(
                    person_detector_status.present
                    and person_detector_status.digest_valid
                    and person_detector_status.loadable is True
                )
        except (AttributeError, ImportError, OSError, RuntimeError, ValueError):
            # Pose/silhouette is optional for group selection, but automatic
            # official member fancams fail closed before render when this
            # verified visual suite is unavailable.
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
            # SFace protects both close-up source switches and Tier-A blends.
            # If unavailable, the selector holds the current source across an
            # ambiguous close-up while body-safe B/C hard cuts remain valid.
            pass
    # The automatic official-fancam path analyzes full frames and never creates
    # an identity anchor or crop.  Its content gate requires real face evidence
    # and its continuity score requires the complete person/pose/silhouette
    # suite. Haar remains a group fallback, not product-ready evidence for this
    # fail-closed automatic member path. SFace identity matching is not used.
    automatic_member_visual_continuity_ready = bool(
        automatic_member_person_detector_ready
        and pose_silhouette_ready
        and face_match_ready
    )
    media_tools_ready = all(
        item.available and item.command_succeeded for item in commands[:2]
    )
    acquisition_ready = selected_ytdlp is not None
    runtime_modes = runtime_contract["mode_readiness"]
    group_mode_ready = bool(media_tools_ready and runtime_modes["group"])
    member_mode_ready = bool(media_tools_ready and runtime_modes["member"])
    stage_edit_generation = runtime_contract.get("stage_edit_generation")
    if not isinstance(stage_edit_generation, dict):
        stage_edit_generation = {}
    professional_group_generation_ready = bool(
        stage_edit_generation.get("automatic_group_generation_ready") is True
    )
    automatic_group_ready = bool(
        group_mode_ready
        and acquisition_ready
    )
    automatic_member_ready = bool(
        group_mode_ready
        and acquisition_ready
        and automatic_member_visual_continuity_ready
    )
    # The native app exposes group and member modes behind one engine-ready
    # state. Keep top-level ready strict while still reporting that a partial
    # runtime can remain usable for group edits.
    required_ready = bool(
        member_mode_ready
        and automatic_group_ready
        and automatic_member_ready
    )

    member_anchor_tracking = dict(runtime_contract["member_anchor_tracking"])
    member_anchor_tracking.update(
        {
            "required_for": ["local_member"],
            "optional_for": [
                "local_group",
                "automatic_group",
                "automatic_member",
            ],
            "fallback": "local_group_and_ready_automatic_modes_remain_available",
        }
    )
    return {
        "ready": required_ready,
        "mode_readiness": {
            # Legacy aliases remain for older local-only clients.
            "group": group_mode_ready,
            "member": member_mode_ready,
            "local_group": group_mode_ready,
            "local_member": member_mode_ready,
            "automatic_group": automatic_group_ready,
            "automatic_member": automatic_member_ready,
        },
        "commands": [asdict(item) for item in commands],
        "python_modules": modules,
        "runtime_contract": runtime_contract,
        "automatic_acquisition": {
            "ready": acquisition_ready,
            "yt_dlp": (
                selected_ytdlp.to_dict()
                if selected_ytdlp is not None
                else {
                    "available": False,
                    "command": [],
                    "display": None,
                    "version": None,
                    "kind": None,
                    "command_succeeded": False,
                }
            ),
            "path_executable": asdict(commands[2]),
            "verification": "newest_working_module_or_path_version_command_no_cookies_at_runtime",
            "network_policy": {
                "cookies": "disabled",
                "drm_bypass": False,
                "uploads": False,
            },
        },
        "mode_capabilities": {
            "review_first_group": {
                "ready": automatic_group_ready,
                "professional_model_available": professional_group_generation_ready,
                "warning": None if professional_group_generation_ready else
                    "Optional professional model unavailable; first review uses validated baseline",
                "quality_approved": False,
            },
            "professional_group_generation": {
                "ready": professional_group_generation_ready,
                "architecture_available": bool(
                    stage_edit_generation.get("architecture_available") is True
                ),
                "installed_production_bundle": stage_edit_generation.get(
                    "installed_production_bundle"
                ),
                "production_registry_error": stage_edit_generation.get(
                    "production_registry_error"
                ),
                "required_for": ["explicit_required_model"],
                "optional_for": ["automatic_group_review_first"],
                "failure_policy": "optional_review_first_with_provenance_explicit_required_fail_closed",
            },
            # Confirmed-anchor tracking belongs only to the local member path;
            # automatic portrait edits use verified full-frame visual evidence.
            "member_anchor_tracking": member_anchor_tracking,
            "automatic_member_visual_continuity": {
                "ready": automatic_member_visual_continuity_ready,
                "person_detector": automatic_member_person_detector_asset,
                "face_detector": face_asset,
                "models": pose_assets,
                "verification": "bundled_digest_and_opencv_dnn_load",
                "scope": (
                    "official_vertical_fancam_performance_presence_and_"
                    "non_identity_full_frame_visual_continuity"
                ),
                "required_for": ["automatic_member"],
                "optional_for": [
                    "local_member",
                    "local_group",
                    "automatic_group",
                ],
                "render_policy": (
                    "official_fancam_full_frame_hard_cut_without_"
                    "identity_anchor_or_crop"
                ),
                "failure_policy": "fail_closed_before_render",
                "raw_biometric_embeddings_persisted": False,
            },
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
                "scope": (
                    "closeup_source_switch_compatibility_and_"
                    "tier_a_dissolve_eligibility"
                ),
                "fallback": (
                    "hold_source_for_ambiguous_closeup_or_use_body_safe_hard_cut"
                ),
            },
        },
    }
