"""Resolve an explicitly migrated cache without silently recreating it.

This is storage routing only. It never mounts disks, creates directories,
changes analyzer fingerprints, or launches analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
import plistlib
import subprocess
import uuid

LOCATION_FILE = "derived-cache-location.json"
LOCATION_CONTRACT = "ive_apfs_derived_cache_location_v1"


class DerivedCacheLocationError(ValueError):
    pass


def _plist_command(args: list[str]) -> dict:
    try:
        result = subprocess.run(args, capture_output=True, check=True, timeout=10)
        value = plistlib.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError("invalid storage information")
        return value
    except (OSError, ValueError, plistlib.InvalidFileException,
            subprocess.SubprocessError) as exc:
        raise DerivedCacheLocationError("APFS cache mount identity could not be verified") from exc


def resolve_migrated_derived_cache(
    source_cache_directory: Path | None,
    explicit_directory: Path | None = None,
) -> Path | None:
    if source_cache_directory is None:
        return explicit_directory
    marker = Path(source_cache_directory).expanduser() / LOCATION_FILE
    if marker.is_symlink():
        raise DerivedCacheLocationError("derived cache location marker cannot be a symlink")
    if not marker.exists():
        return explicit_directory
    try:
        if not marker.is_file() or marker.stat().st_size > 16384:
            raise ValueError("invalid marker size or type")
        value = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {
            "contract", "image_path", "mount_path", "volume_uuid"
        } or value["contract"] != LOCATION_CONTRACT:
            raise ValueError("invalid location contract")
        if any(not isinstance(value[k], str) or not value[k].strip()
               for k in ("image_path", "mount_path", "volume_uuid")):
            raise ValueError("invalid location field")
        expected_uuid = str(uuid.UUID(value["volume_uuid"])).upper()
        image = Path(value["image_path"])
        mount = Path(value["mount_path"])
        if (not image.is_absolute() or not mount.is_absolute()
                or image.suffix != ".sparsebundle" or mount in (Path("/"), Path("/Volumes"))
                or image.is_symlink() or mount.is_symlink()):
            raise ValueError("invalid cache image or mount path")
        if not image.is_dir() or not mount.is_dir():
            raise DerivedCacheLocationError(
                "APFS analysis cache is not mounted; connect the SSD and run "
                "node scripts/analysis_cache_mounts.cjs mount. "
                "The exFAT cache will not be recreated."
            )
        image = image.resolve(strict=True)
        mount = mount.resolve(strict=True)
        if explicit_directory is not None and Path(explicit_directory).expanduser().resolve() != mount:
            raise ValueError("explicit derived-cache path conflicts with migrated cache location")
    except DerivedCacheLocationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise DerivedCacheLocationError(f"Invalid derived cache location: {exc}") from exc

    info = _plist_command(["/usr/sbin/diskutil", "info", "-plist", str(mount)])
    if (info.get("FilesystemType") != "apfs"
            or str(info.get("VolumeUUID", "")).upper() != expected_uuid
            or info.get("MountPoint") != str(mount)):
        raise DerivedCacheLocationError("APFS cache volume identity mismatch; refusing fallback")
    images = _plist_command(["/usr/bin/hdiutil", "info", "-plist"])
    backing_matches = any(
        item.get("image-path") == str(image)
        and any(entity.get("mount-point") == str(mount)
                for entity in item.get("system-entities", []))
        for item in images.get("images", [])
    )
    if not backing_matches:
        raise DerivedCacheLocationError("APFS cache backing image mismatch; refusing fallback")
    visual = mount / "visual-v1"
    if visual.is_symlink() or not visual.is_dir():
        raise DerivedCacheLocationError("APFS visual cache data is missing")
    return mount
