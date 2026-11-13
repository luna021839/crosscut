"""Non-destructive GROUP source-format upgrades with fresh media bindings.

A revision is a complete acquisition pool, not a replacement of active files.
Only its small pointer is published, after the normal acquisition validation
and relative-quality selection have run. Rendering/pose proofs never transfer.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import errno
import os
from pathlib import Path
import re
import shutil
from typing import Mapping
from uuid import uuid4

from .ive import EditMode

_CONTRACT = "group_source_format_revision_v1"
_POINTER = "active-format-revision.json"


def _copy_or_link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.ENOTSUP}:
            raise
        # ExFAT does not support hardlinks. Exclusive creation retains the
        # same no-overwrite contract on the user's external SSD.
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024*1024)


def resolve_group_format_cache(request) -> Path:
    """Resolve one scoped revision; actual content is revalidated by acquisition."""
    from .acquisition import (_manifest_matches_request_contract, _read_manifest,
                              _validated_manifest_selection, _content_integrity_matches)

    base = request.source_cache_directory
    if base is None or request.edit_mode is not EditMode.GROUP:
        return base
    pointer = base / _POINTER
    if base.is_symlink() or pointer.is_symlink() or not pointer.is_file():
        return base
    try:
        if pointer.stat().st_size > 4096:
            return base
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("contract") != _CONTRACT:
            return base
        revision = payload.get("revision")
        if (payload.get("song") != request.song or not isinstance(revision, str)
                or re.fullmatch(r"[0-9a-f]{32}", revision) is None):
            return base
        parent = base / "format-revisions"
        directory = parent / revision
        manifest_path = directory / "acquisition-manifest.json"
        if any(p.is_symlink() for p in (parent, directory, manifest_path)):
            return base
        manifest = _read_manifest(manifest_path)
        if manifest is None or not _manifest_matches_request_contract(manifest, request):
            return base
        entries = _validated_manifest_selection(manifest=manifest,
            manifest_path=manifest_path, request=request)
        if entries is None or any(path.is_symlink() or not path.is_file()
                or not _content_integrity_matches(path, entry.get("content_integrity"))
                for entry, path, _ in entries):
            return base
        return directory
    except (OSError, ValueError, TypeError):
        return base


def prepare_group_format_revision(*, request, artifact, replacements: Mapping[str, Path],
                                  reporter, media_validator=None, media_probe=None,
                                  group_quality_analyzer=None):
    """Prepare but do not activate a validated revision from same-ID downloads.

    The acquisition client supplies replacements after a format metadata check.
    Keeping preparation separate from activation also allows a caller to wait
    for a current render without mutating any of its input files.
    """
    from . import acquisition as a

    if request.edit_mode is not EditMode.GROUP or not replacements:
        raise ValueError("format revision requires GROUP and explicit replacements")
    base = request.source_cache_directory
    source_directory = artifact.manifest_path.parent
    scoped_revision = (base is not None and source_directory.parent == base / "format-revisions"
        and re.fullmatch(r"[0-9a-f]{32}", source_directory.name) is not None
        and not source_directory.parent.is_symlink() and not source_directory.is_symlink())
    if base is None or base.is_symlink() or (source_directory != base and not scoped_revision):
        raise ValueError("revision must start from the request's exact cache")
    probe = media_probe or a.probe_media
    validator = media_validator or a._validate_downloaded_media
    analyzer = group_quality_analyzer or a.analyze_group_source_quality
    manifest = deepcopy(artifact.manifest)
    if not a._manifest_matches_request_contract(manifest, request):
        raise ValueError("base acquisition contract changed")
    selected = a._validated_manifest_selection(manifest=manifest,
        manifest_path=artifact.manifest_path, request=request)
    if selected is None:
        raise ValueError("base source selection is not authenticated")
    excluded = a._validated_group_quality_excluded_bindings(manifest=manifest,
        manifest_path=artifact.manifest_path, request=request)
    ranking_indices = {record["video_id"]:record["acquisition_source_index"]
        for record in (manifest.get("group_relative_quality_gate") or {}).get("ranking", [])}
    entries = sorted((*selected, *excluded), key=lambda entry: ranking_indices.get(entry[2],
        entry[0].get("acquisition_source_index", entry[0].get("source_index", 0))))
    if len(entries) != manifest["summary"].get("materialized_source_count"):
        raise ValueError("cannot silently lose quality-excluded original sources")
    ids = {video_id for _, _, video_id in entries}
    if not set(replacements) <= ids:
        raise ValueError("replacement ID is outside the verified acquisition pool")
    resolved = {}
    upgrade_audit = []
    for entry, old_path, video_id in entries:
        if old_path.is_symlink() or not a._content_integrity_matches(old_path, entry.get("content_integrity")):
            raise ValueError("base source content changed before revision")
        if video_id not in replacements:
            continue
        path = Path(replacements[video_id])
        if (path.is_symlink() or not path.is_file() or path.stem != video_id
                or path.suffix.lower() not in a._VIDEO_SUFFIXES):
            raise ValueError("replacement is not a regular same-ID video")
        info = probe(path)
        old_info = probe(old_path)
        a._validate_media_for_request(path, request=request,
            media_validator=validator, media_probe=probe)
        before, after = old_info.video_streams[0], info.video_streams[0]
        if (not all((before.width, before.height, after.width, after.height))
                or after.width <= before.width or after.height <= before.height):
            raise ValueError("format revision must increase actual decoded dimensions")
        if (before.frame_rate is None or after.frame_rate is None
                or abs(before.frame_rate-after.frame_rate) > .05
                or old_info.duration_seconds is None or info.duration_seconds is None
                or abs(old_info.duration_seconds-info.duration_seconds) > .25):
            raise ValueError("replacement duration/frame rate differs from the same video")
        binding = a._content_integrity(path)
        resolved[video_id] = (path, binding)
        upgrade_audit.append({"video_id": video_id,
            "previous_content_integrity": entry["content_integrity"],
            "new_content_integrity": binding,
            "previous_dimensions": [before.width, before.height],
            "new_dimensions": [after.width, after.height]})

    parent = base / "format-revisions"
    if parent.is_symlink():
        raise ValueError("format revision directory may not be a symlink")
    copy_bytes = sum(resolved.get(video_id, (path, entry["content_integrity"]))[1]["size_bytes"]
        for entry,path,video_id in entries)
    if shutil.disk_usage(base).free < copy_bytes + 256*1024*1024:
        raise ValueError("insufficient space for a complete non-destructive format revision")
    revision = uuid4().hex
    directory = parent / revision
    videos = directory / "videos"
    videos.mkdir(parents=True, exist_ok=False)
    revision_request = replace(request, source_cache_directory=directory)
    records = {str(record["video_id"]): record for record in manifest["candidates"]}
    new_entries, paths = [], []
    for entry, old_path, video_id in entries:
        source, expected_binding = resolved.get(video_id, (old_path, entry["content_integrity"]))
        destination = videos / (video_id + source.suffix.lower())
        # All revision files are immutable shared content objects. Future
        # downloads use no-overwrite, and ordinary acquisition rehashes them.
        _copy_or_link(source, destination)
        if not a._content_integrity_matches(destination, expected_binding):
            raise ValueError("source changed while preparing format revision")
        media = a._validate_media_for_request(destination, request=revision_request,
            media_validator=validator, media_probe=probe)
        refreshed = deepcopy(entry)
        refreshed.update(local_path=str(destination), content_integrity=expected_binding,
            media_validation=media, aspect_validation=media, download_status="cached")
        records[video_id].update(local_path=str(destination), content_integrity=expected_binding,
            media_validation=media, aspect_validation=media, download_status="cached")
        new_entries.append(refreshed)
        paths.append(destination)
    eligible, chosen, gate = a._apply_group_relative_quality_gate(request=revision_request,
        paths=paths, selected_entries=new_entries, records_by_id=records,
        media_probe=probe, analyzer=analyzer)
    manifest["selected"] = chosen
    manifest["group_relative_quality_gate"] = gate
    manifest["summary"].update(status="complete", materialized_source_count=len(paths),
        eligible_source_count=len(eligible), downloaded_source_count=len(eligible),
        group_quality_excluded_source_count=len(paths)-len(eligible),
        source_count_fallback_applied=len(eligible)<request.desired_source_count)
    manifest["format_revision"] = {"contract": _CONTRACT, "revision": revision,
        "previous_manifest": str(artifact.manifest_path), "upgrades": upgrade_audit,
        "original_files_preserved": True, "quality_selection_recomputed": True,
        "render_or_pose_proofs_restored": False, "quality_approval": False}
    manifest_path = directory / "acquisition-manifest.json"
    a._write_manifest(manifest_path, manifest)
    verified = a._reusable_completed_manifest(request=revision_request,
        manifest_path=manifest_path, media_validator=validator, media_probe=probe)
    if verified is None:
        raise ValueError("prepared source revision failed normal acquisition validation")
    reporter.emit("source_format_revision_prepared", revision=revision,
        upgraded_source_count=len(resolved), source_count=len(verified.source_paths),
        original_files_preserved=True, quality_approval=False)
    return verified


def activate_group_format_revision(*, request, revision_artifact):
    """Publish one already prepared cache revision without replacing any video."""
    from .acquisition import (_canonical_json_bytes, _atomic_catalog_write, _read_manifest,
        _manifest_matches_request_contract, _validated_manifest_selection, _content_integrity_matches)

    base = request.source_cache_directory
    directory = revision_artifact.manifest_path.parent
    if (base is None or request.edit_mode is not EditMode.GROUP
            or directory.parent != base / "format-revisions"
            or re.fullmatch(r"[0-9a-f]{32}", directory.name) is None
            or any(p.is_symlink() for p in (base, directory.parent, directory))
            or revision_artifact.manifest.get("format_revision", {}).get("revision") != directory.name):
        raise ValueError("revision is outside the request cache")
    manifest = _read_manifest(revision_artifact.manifest_path)
    if manifest is None or not _manifest_matches_request_contract(manifest, request):
        raise ValueError("prepared revision contract changed before activation")
    entries = _validated_manifest_selection(manifest=manifest,
        manifest_path=revision_artifact.manifest_path, request=request)
    if entries is None or any(path.is_symlink() or not path.is_file()
            or not _content_integrity_matches(path,entry.get("content_integrity"))
            for entry,path,_ in entries):
        raise ValueError("prepared revision content changed before activation")
    pointer = {"contract": _CONTRACT, "song": request.song,
               "revision": directory.name, "quality_approval": False}
    _atomic_catalog_write(base / _POINTER, _canonical_json_bytes(pointer))
