"""Bounded cache format refresh; unavailable upgrades never destroy a source."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import time
from uuid import uuid4

from .ive import EditMode
from .source_format_revision import prepare_group_format_revision, activate_group_format_revision

_CONTRACT = "content_bound_actual_format_availability_v1"
_BUDGET_SECONDS = 480.0


def _fresh(record, *, binding, profile, now):
    if (not isinstance(record, dict) or record.get("content_integrity") != binding
            or record.get("profile") != profile):
        return False
    try:
        stamp = datetime.fromisoformat(record["checked_at"])
        age = (now-stamp).total_seconds()
    except (KeyError, TypeError, ValueError):
        return False
    ttl = 7*86400 if record.get("decision") == "best_available_dimensions_already_cached" else 86400
    return 0 <= age < ttl


def refresh_group_cached_formats(*, request, artifact, reporter, client,
                                 media_validator, media_probe, group_quality_analyzer):
    from . import acquisition as a

    if request.edit_mode is not EditMode.GROUP:
        return artifact
    # Legacy injected clients retain their historical offline cache behavior.
    if client is not None and not callable(getattr(client, "best_video_dimensions", None)):
        return artifact
    now = datetime.now(UTC)
    started = time.monotonic()
    profile = a._PROFESSIONAL_DOWNLOAD_PROFILE
    audit = artifact.manifest.get("format_availability_audit", {})
    records = (dict(audit.get("sources", {})) if isinstance(audit, dict)
               and audit.get("contract") == _CONTRACT else {})
    candidates = {c.video_id:c for c in a._manifest_refresh_candidates(artifact=artifact,request=request)}
    selected = artifact.manifest["selected"]
    excluded = a._validated_group_quality_excluded_bindings(manifest=artifact.manifest,
        manifest_path=artifact.manifest_path,request=request)
    entries = [*selected, *(entry for entry,_,_ in excluded)]
    pending = [entry for entry in entries if not _fresh(records.get(entry["video_id"]),
        binding=entry.get("content_integrity"),profile=profile,now=now)]
    if not pending:
        return artifact
    if client is None:
        try:
            client = a.YTDLPClient()
        except a.AcquisitionError:
            # A previously verified cache remains usable without a downloader.
            return artifact
    if getattr(client,"download_profile",None) != profile:
        return artifact
    staging_root = request.source_cache_directory / "format-downloads"
    if staging_root.is_symlink():
        reporter.log("포맷 업그레이드 경로가 링크여서 기존 검증 소스를 유지합니다.")
        return artifact
    replacements = {}
    for entry in pending:
        if time.monotonic()-started >= _BUDGET_SECONDS-40:
            break
        video_id = entry["video_id"]
        if video_id not in candidates:
            continue
        record = {"content_integrity":entry["content_integrity"], "profile":profile,
                  "checked_at":now.isoformat(), "decision":"check_failed_old_source_retained"}
        records[video_id] = record
        try:
            info = media_probe(Path(entry["local_path"]))
            stream = info.video_streams[0]
            current = (stream.width,stream.height)
            best = ((3840,2160) if current == (3840,2160)
                    else client.best_video_dimensions(video_id))
            record.update(cached_dimensions=list(current),available_dimensions=list(best))
            if not (best[0]>current[0] and best[1]>current[1]):
                record["decision"] = "best_available_dimensions_already_cached"
                continue
            reporter.status(f"{video_id}: 실제 제공되는 {best[0]}×{best[1]} 소스로 확대 여유를 확보하는 중…")
            staging = staging_root / uuid4().hex
            staging.mkdir(parents=True,exist_ok=False)
            remaining = _BUDGET_SECONDS-(time.monotonic()-started)
            if remaining < 1:
                break
            downloaded = client.download(candidates[video_id],staging,
                timeout_seconds=min(remaining,600.0))
            replacements[video_id] = downloaded
            record["decision"] = "download_staged_pending_normal_media_validation"
        except (a.AcquisitionError,a.MediaProbeError,OSError,ValueError,TypeError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            reporter.log(f"{video_id}: 고화질 업그레이드 미완료 · 기존 검증 소스 유지")
    artifact.manifest["format_availability_audit"] = {"contract":_CONTRACT,
        "sources":records, "configured_format_preference_is_not_download_evidence":True,
        "network_download_budget_seconds":_BUDGET_SECONDS, "quality_approval":False}
    a._write_manifest(artifact.manifest_path,artifact.manifest)
    if not replacements:
        return artifact
    try:
        revision = prepare_group_format_revision(request=request,artifact=artifact,
            replacements=replacements,reporter=reporter,media_validator=media_validator,
            media_probe=media_probe,group_quality_analyzer=group_quality_analyzer)
        activate_group_format_revision(request=request,revision_artifact=revision)
        reporter.log(f"고화질 소스 {len(replacements)}개를 새 캐시에 적용 · 기존 파일 보존 · 분석/편집 검증 새로 수행")
        return revision
    except (a.AcquisitionError,a.MediaProbeError,OSError,ValueError,TypeError) as exc:
        reporter.log(f"고화질 캐시 재검증 미완료 · 기존 소스 유지: {type(exc).__name__}: {exc}")
        return artifact
