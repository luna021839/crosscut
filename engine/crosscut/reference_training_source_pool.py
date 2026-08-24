"""Strict, provenance-bound source pools for human-reference attribution.

Production acquisition admits only the top quality-ranked sources to a render.
Reverse-attributing a human StageMix has a different requirement: every local
broadcast feed that may occur in the reference should remain searchable as long
as it passed the hard media contract.  This module derives an immutable
training-only manifest without changing the production acquisition manifest or
pretending that a relative-quality exclusion was a production admission.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "strict_reference_training_source_pool_v1"
CONTRACT_VERSION = 1
MINIMUM_SOURCE_COUNT = 8
PURPOSE = "reference_reverse_attribution_only_not_production_render_admission"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(payload: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(payload))
    contract = normalized.get("reference_training_source_pool")
    if not isinstance(contract, Mapping):
        raise ValueError("reference training source-pool contract is missing")
    mutable = deepcopy(dict(contract))
    mutable["manifest_content_sha256"] = None
    normalized["reference_training_source_pool"] = mutable
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def _load_json(path: Path, name: str) -> tuple[dict[str, Any], str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} is not a regular file: {source}")
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _entry_by_id(values: object, name: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} item {index} must be an object")
        video_id = str(value.get("video_id") or "")
        if not video_id or video_id in result:
            raise ValueError(f"{name} has an invalid/duplicate video id")
        result[video_id] = value
    return result


def _validate_media_admission(
    entry: Mapping[str, Any],
    *,
    allow_relative_quality_exclusion: bool,
    production_selected: bool = False,
) -> None:
    path = Path(str(entry.get("local_path") or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"training source media is missing: {path}")
    validation = entry.get("media_validation")
    aspect = entry.get("aspect_validation")
    integrity = entry.get("content_integrity")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise ValueError("training source did not pass the hard media contract")
    if not isinstance(aspect, Mapping) or aspect.get("passed") is not True:
        raise ValueError("training source did not pass the aspect contract")
    if not isinstance(integrity, Mapping) or (
        integrity.get("contract_version") != 1
        or integrity.get("algorithm") != "sha256"
    ):
        raise ValueError("training source content-integrity contract is invalid")
    digest = integrity.get("sha256")
    size = integrity.get("size_bytes")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("training source SHA-256 is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("training source byte size is invalid")
    if path.stat().st_size != size:
        raise ValueError(f"training source byte size changed: {path}")
    hard_rejections = entry.get("download_hard_rejection_reasons", [])
    if hard_rejections not in (None, []) or entry.get("rejection_reasons", []) not in (
        None,
        [],
    ):
        raise ValueError("hard-rejected media cannot enter a training pool")
    # Older acquisition manifests encoded production admission solely by
    # membership in ``selected`` and did not serialize group_quality_eligible.
    # Preserve that legacy fact without writing a fabricated field back into
    # the copied acquisition entry.  An explicit false still fails closed.
    production_eligible = entry.get("group_quality_eligible") is True or (
        production_selected and "group_quality_eligible" not in entry
    )
    if production_eligible:
        return
    if not allow_relative_quality_exclusion:
        raise ValueError("a production-selected source lost its quality admission")
    reasons = entry.get("group_quality_exclusion_reasons")
    if not isinstance(reasons, list) or not reasons:
        raise ValueError("training-only admission needs explicit quality reasons")
    allowed = {
        "relative_quality_below_top_eight",
        "duplicate_broadcast_feed_deprioritized",
    }
    if any(str(reason) not in allowed for reason in reasons):
        raise ValueError("training-only source has a non-relative exclusion reason")


def build_reference_training_source_manifest(
    acquisition_manifest_path: Path,
    *,
    additional_source_ids: Sequence[str],
    expected_artist: str,
    expected_song: str,
    expected_source_count: int,
) -> dict[str, Any]:
    """Derive a strict training-only superset from one immutable acquisition."""

    if expected_source_count < MINIMUM_SOURCE_COUNT:
        raise ValueError(f"expected_source_count must be at least {MINIMUM_SOURCE_COUNT}")
    source_path = Path(acquisition_manifest_path).expanduser().resolve()
    acquisition, acquisition_hash = _load_json(
        source_path, "acquisition manifest"
    )
    if (
        acquisition.get("schema_version") != 1
        or acquisition.get("group") != expected_artist
        or acquisition.get("song") != expected_song
    ):
        raise ValueError("acquisition manifest identity/schema mismatch")
    production = _entry_by_id(acquisition.get("selected"), "selected")
    candidates = _entry_by_id(acquisition.get("candidates"), "candidates")
    extras = tuple(str(value).strip() for value in additional_source_ids)
    if any(not value for value in extras) or len(extras) != len(set(extras)):
        raise ValueError("additional source ids must be non-empty and unique")
    if set(extras) & set(production):
        raise ValueError("additional source ids must not repeat production sources")
    if len(production) + len(extras) != expected_source_count:
        raise ValueError("production plus additional sources do not match exact count")
    missing = set(extras) - set(candidates)
    if missing:
        raise ValueError(f"additional sources are absent from acquisition: {sorted(missing)}")

    ordered_entries: list[dict[str, Any]] = []
    production_ids = tuple(production)
    for source_index, video_id in enumerate((*production_ids, *extras)):
        original = production.get(video_id) or candidates[video_id]
        training_only = video_id in extras
        _validate_media_admission(
            original,
            allow_relative_quality_exclusion=training_only,
            production_selected=not training_only,
        )
        entry = deepcopy(dict(original))
        entry["source_index"] = source_index
        entry["training_attribution_eligible"] = True
        entry["training_admission"] = {
            "contract_version": CONTRACT_VERSION,
            "purpose": PURPOSE,
            "production_group_quality_eligible": (
                original.get("group_quality_eligible") is True
                or (
                    not training_only
                    and "group_quality_eligible" not in original
                )
            ),
            "production_group_quality_eligibility_explicit": (
                "group_quality_eligible" in original
            ),
            "production_group_quality_exclusion_reasons": deepcopy(
                list(original.get("group_quality_exclusion_reasons") or [])
            ),
            "hard_media_contract_passed": True,
            "production_render_admission_modified": False,
        }
        ordered_entries.append(entry)

    derived = deepcopy(acquisition)
    derived["selected"] = ordered_entries
    derived["reference_training_source_pool"] = {
        "contract_version": CONTRACT_VERSION,
        "analysis": ANALYSIS_BASIS,
        "purpose": PURPOSE,
        "minimum_source_count": MINIMUM_SOURCE_COUNT,
        "source_count": expected_source_count,
        "source_ids": [entry["video_id"] for entry in ordered_entries],
        "production_selected_source_ids": list(production_ids),
        "training_only_additional_source_ids": list(extras),
        "production_quality_gate_preserved": True,
        "acquisition_manifest": {
            "path": str(source_path),
            "file_sha256": acquisition_hash,
        },
        "manifest_content_sha256": None,
    }
    derived["reference_training_source_pool"]["manifest_content_sha256"] = (
        _content_sha256(derived)
    )
    validate_reference_training_source_manifest(
        derived,
        expected_artist=expected_artist,
        expected_song=expected_song,
        verify_acquisition_binding=True,
    )
    return derived


def validate_reference_training_source_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_artist: str,
    expected_song: str,
    verify_acquisition_binding: bool,
) -> tuple[str, ...]:
    """Validate and return the exact source order authorized for training."""

    contract = manifest.get("reference_training_source_pool")
    if not isinstance(contract, Mapping):
        return ()
    if (
        contract.get("contract_version") != CONTRACT_VERSION
        or contract.get("analysis") != ANALYSIS_BASIS
        or contract.get("purpose") != PURPOSE
        or contract.get("minimum_source_count") != MINIMUM_SOURCE_COUNT
        or contract.get("production_quality_gate_preserved") is not True
    ):
        raise ValueError("reference training source-pool contract is invalid")
    if manifest.get("group") != expected_artist or manifest.get("song") != expected_song:
        raise ValueError("reference training source-pool song identity mismatch")
    source_ids = contract.get("source_ids")
    if not isinstance(source_ids, list) or len(source_ids) < MINIMUM_SOURCE_COUNT:
        raise ValueError("reference training source-pool source_ids are invalid")
    ordered = tuple(str(value) for value in source_ids)
    if len(set(ordered)) != len(ordered) or contract.get("source_count") != len(ordered):
        raise ValueError("reference training source-pool count/order is invalid")
    selected = manifest.get("selected")
    if not isinstance(selected, list) or tuple(
        str(item.get("video_id")) if isinstance(item, Mapping) else ""
        for item in selected
    ) != ordered:
        raise ValueError("training selected order differs from its contract")
    expected_hash = contract.get("manifest_content_sha256")
    if not isinstance(expected_hash, str) or expected_hash != _content_sha256(manifest):
        raise ValueError("reference training source-pool content hash mismatch")
    production_ids = contract.get("production_selected_source_ids")
    additional_ids = contract.get("training_only_additional_source_ids")
    if not isinstance(production_ids, list) or not isinstance(additional_ids, list):
        raise ValueError("reference training source-pool membership is invalid")
    if tuple(str(value) for value in (*production_ids, *additional_ids)) != ordered:
        raise ValueError("reference training source-pool membership/order mismatch")
    production_set = {str(value) for value in production_ids}
    additional_set = {str(value) for value in additional_ids}
    for entry in selected:
        assert isinstance(entry, Mapping)
        video_id = str(entry.get("video_id"))
        if entry.get("training_attribution_eligible") is not True:
            raise ValueError("training source is not attribution-eligible")
        admission = entry.get("training_admission")
        if not isinstance(admission, Mapping) or (
            admission.get("contract_version") != CONTRACT_VERSION
            or admission.get("purpose") != PURPOSE
            or admission.get("hard_media_contract_passed") is not True
            or admission.get("production_render_admission_modified") is not False
        ):
            raise ValueError("training source admission provenance is invalid")
        production_flag = entry.get("group_quality_eligible") is True or (
            video_id in production_set and "group_quality_eligible" not in entry
        )
        if admission.get("production_group_quality_eligible") is not production_flag:
            raise ValueError("production quality eligibility was not preserved")
        explicit_presence = admission.get(
            "production_group_quality_eligibility_explicit"
        )
        if explicit_presence is not None and explicit_presence is not (
            "group_quality_eligible" in entry
        ):
            raise ValueError("production quality eligibility presence was not preserved")
        if video_id in production_set and not production_flag:
            raise ValueError("production source lost its production eligibility")
        if video_id in additional_set and production_flag:
            raise ValueError("training-only source unexpectedly has production admission")
        _validate_media_admission(
            entry,
            allow_relative_quality_exclusion=video_id in additional_set,
            production_selected=video_id in production_set,
        )
    binding = contract.get("acquisition_manifest")
    if not isinstance(binding, Mapping):
        raise ValueError("acquisition manifest binding is missing")
    bound_path = Path(str(binding.get("path") or "")).expanduser().resolve()
    bound_hash = binding.get("file_sha256")
    if not isinstance(bound_hash, str) or len(bound_hash) != 64:
        raise ValueError("acquisition manifest binding hash is invalid")
    if verify_acquisition_binding:
        original, observed_hash = _load_json(bound_path, "bound acquisition manifest")
        if observed_hash != bound_hash:
            raise ValueError("bound acquisition manifest SHA-256 mismatch")
        original_ids = tuple(_entry_by_id(original.get("selected"), "selected"))
        if original_ids != tuple(str(value) for value in production_ids):
            raise ValueError("bound production selection changed")
        original_candidates = _entry_by_id(original.get("candidates"), "candidates")
        if not additional_set <= set(original_candidates):
            raise ValueError("bound acquisition no longer contains additional sources")
    return ordered


def build_synchronized_training_source_catalog(
    seed_evidence: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    *,
    expected_artist: str,
    expected_song: str,
    sample_rate: int = 8_000,
) -> tuple[dict[str, Any], ...]:
    """Extend a trusted production seed catalog with strictly synchronized feeds."""

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    source_ids = validate_reference_training_source_manifest(
        training_manifest,
        expected_artist=expected_artist,
        expected_song=expected_song,
        verify_acquisition_binding=True,
    )
    selected = training_manifest.get("selected")
    assert isinstance(selected, list)
    selected_by_id = {str(item["video_id"]): item for item in selected}
    seed_sources = seed_evidence.get("sources")
    if not isinstance(seed_sources, list) or len(seed_sources) < MINIMUM_SOURCE_COUNT:
        raise ValueError("seed evidence source catalog is invalid")
    seed_by_id = {
        str(item.get("video_id")): item
        for item in seed_sources
        if isinstance(item, Mapping)
    }
    production_ids = tuple(
        str(value)
        for value in training_manifest["reference_training_source_pool"][
            "production_selected_source_ids"
        ]
    )
    if tuple(seed_by_id) != production_ids:
        raise ValueError("seed evidence does not bind the production source order")

    for video_id in source_ids:
        entry = selected_by_id[video_id]
        path = Path(str(entry.get("local_path") or "")).expanduser().resolve()
        expected = str(entry["content_integrity"]["sha256"])
        if _sha256_file(path) != expected:
            raise ValueError(f"training source media SHA-256 mismatch: {video_id}")

    from .audio import (
        extract_mono_pcm,
        normalized_cross_correlation_offset,
        spectral_flux_alignment_offset,
    )
    from .reference_evidence_builder import ReferenceSourceClock
    from .timewarp import estimate_local_timewarp
    import cv2

    master_id = production_ids[0]
    master_path = Path(str(selected_by_id[master_id]["local_path"])).resolve()
    master_pcm = extract_mono_pcm(master_path, sample_rate=sample_rate)
    result: list[dict[str, Any]] = []
    for video_id in source_ids:
        if video_id in seed_by_id:
            source = deepcopy(dict(seed_by_id[video_id]))
            if Path(str(source.get("path"))).resolve() != Path(
                str(selected_by_id[video_id]["local_path"])
            ).resolve():
                raise ValueError("seed and training manifest media paths differ")
            result.append(source)
            continue
        entry = selected_by_id[video_id]
        path = Path(str(entry["local_path"])).resolve()
        source_pcm = extract_mono_pcm(path, sample_rate=sample_rate)
        minimum_overlap = min(
            10.0,
            max(2.0, min(master_pcm.size, source_pcm.size) / sample_rate / 3.0),
        )
        try:
            coarse = spectral_flux_alignment_offset(
                master_pcm,
                source_pcm,
                sample_rate=sample_rate,
                max_offset_seconds=120.0,
                min_overlap_seconds=minimum_overlap,
            )
        except ValueError:
            coarse = normalized_cross_correlation_offset(
                master_pcm,
                source_pcm,
                sample_rate=sample_rate,
                max_offset_seconds=120.0,
                min_overlap_seconds=minimum_overlap,
            )
        if coarse.score < 0.32 or coarse.score_margin < 0.025:
            raise ValueError(
                f"training source synchronization is not confident: {video_id}"
            )
        warp = estimate_local_timewarp(
            master_pcm,
            source_pcm,
            sample_rate=sample_rate,
            global_offset_seconds=coarse.offset_seconds,
        )
        if warp.scalar_fallback:
            raise ValueError(
                f"training source used forbidden scalar timewarp: {video_id}: "
                f"{warp.fallback_reason}"
            )
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"cannot probe training source: {video_id}")
            frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
            width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
            height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        finally:
            capture.release()
        if (
            not math.isfinite(frame_rate)
            or frame_rate <= 0.0
            or frame_count <= 0
            or width <= 1
            or height <= 1
        ):
            raise ValueError(f"invalid training source probe: {video_id}")
        clock = ReferenceSourceClock.from_mapping(warp.to_dict()).to_dict()
        clock["provenance"] = {
            "training_source_pool_analysis": ANALYSIS_BASIS,
            "coarse_alignment": coarse.to_dict(),
            "local_timewarp": warp.to_dict(),
        }
        result.append(
            {
                "video_id": video_id,
                "path": str(path),
                "clock": clock,
                "provenance": {
                    "coarse_score": coarse.score,
                    "coarse_margin": coarse.score_margin,
                    "training_source_pool_analysis": ANALYSIS_BASIS,
                    "production_group_quality_eligible": False,
                },
                "media_probe": {
                    "frame_rate": frame_rate,
                    "declared_frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                },
            }
        )
    if tuple(item["video_id"] for item in result) != source_ids:
        raise ValueError("synchronized training source order changed")
    return tuple(result)


def build_synchronized_training_source_catalog_from_manifest(
    training_manifest: Mapping[str, Any],
    *,
    expected_artist: str,
    expected_song: str,
    sample_rate: int = 8_000,
) -> tuple[dict[str, Any], ...]:
    """Synchronize an exact training pool without an older seed catalog.

    The first selected source is the canonical master and receives the only
    permitted scalar identity clock.  Every other source must pass the same
    confident coarse alignment and non-scalar local-timewarp contract used
    when extending an existing seed.  This is the reusable path for songs
    whose acquisition already selected the complete (for example, ten-source)
    training pool.
    """

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    source_ids = validate_reference_training_source_manifest(
        training_manifest,
        expected_artist=expected_artist,
        expected_song=expected_song,
        verify_acquisition_binding=True,
    )
    selected = training_manifest.get("selected")
    assert isinstance(selected, list)
    selected_by_id = {str(item["video_id"]): item for item in selected}
    for video_id in source_ids:
        entry = selected_by_id[video_id]
        path = Path(str(entry.get("local_path") or "")).expanduser().resolve()
        expected = str(entry["content_integrity"]["sha256"])
        if _sha256_file(path) != expected:
            raise ValueError(f"training source media SHA-256 mismatch: {video_id}")

    from .audio import (
        extract_mono_pcm,
        normalized_cross_correlation_offset,
        spectral_flux_alignment_offset,
    )
    from .reference_evidence_builder import ReferenceSourceClock
    from .timewarp import estimate_local_timewarp
    import cv2

    master_id = source_ids[0]
    master_path = Path(str(selected_by_id[master_id]["local_path"])).resolve()
    master_pcm = extract_mono_pcm(master_path, sample_rate=sample_rate)
    result: list[dict[str, Any]] = []
    for video_id in source_ids:
        entry = selected_by_id[video_id]
        path = Path(str(entry["local_path"])).resolve()
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"cannot probe training source: {video_id}")
            frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
            width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
            height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        finally:
            capture.release()
        if (
            not math.isfinite(frame_rate)
            or frame_rate <= 0.0
            or frame_count <= 0
            or width <= 1
            or height <= 1
        ):
            raise ValueError(f"invalid training source probe: {video_id}")

        if video_id == master_id:
            clock = ReferenceSourceClock.scalar(
                0.0,
                basis="strict_training_source_master_identity_v1",
                provenance={
                    "training_source_pool_analysis": ANALYSIS_BASIS,
                    "identity_master": True,
                },
            ).to_dict()
            coarse_payload: Mapping[str, Any] = {
                "offset_samples": 0,
                "offset_seconds": 0.0,
                "score": 1.0,
                "second_best_score": 0.0,
                "score_margin": 1.0,
                "sample_rate": sample_rate,
                "backend": "identity_master",
            }
            local_payload: Mapping[str, Any] | None = None
        else:
            source_pcm = extract_mono_pcm(path, sample_rate=sample_rate)
            minimum_overlap = min(
                10.0,
                max(2.0, min(master_pcm.size, source_pcm.size) / sample_rate / 3.0),
            )
            try:
                coarse = spectral_flux_alignment_offset(
                    master_pcm,
                    source_pcm,
                    sample_rate=sample_rate,
                    max_offset_seconds=120.0,
                    min_overlap_seconds=minimum_overlap,
                )
            except ValueError:
                coarse = normalized_cross_correlation_offset(
                    master_pcm,
                    source_pcm,
                    sample_rate=sample_rate,
                    max_offset_seconds=120.0,
                    min_overlap_seconds=minimum_overlap,
                )
            if coarse.score < 0.32 or coarse.score_margin < 0.025:
                raise ValueError(
                    f"training source synchronization is not confident: {video_id}"
                )
            warp = estimate_local_timewarp(
                master_pcm,
                source_pcm,
                sample_rate=sample_rate,
                global_offset_seconds=coarse.offset_seconds,
            )
            if warp.scalar_fallback:
                raise ValueError(
                    f"training source used forbidden scalar timewarp: {video_id}: "
                    f"{warp.fallback_reason}"
                )
            coarse_payload = coarse.to_dict()
            local_payload = warp.to_dict()
            clock = ReferenceSourceClock.from_mapping(local_payload).to_dict()
            clock["provenance"] = {
                "training_source_pool_analysis": ANALYSIS_BASIS,
                "coarse_alignment": coarse_payload,
                "local_timewarp": local_payload,
            }

        result.append(
            {
                "video_id": video_id,
                "path": str(path),
                "clock": clock,
                "provenance": {
                    "coarse_score": float(coarse_payload["score"]),
                    "coarse_margin": float(coarse_payload["score_margin"]),
                    "training_source_pool_analysis": ANALYSIS_BASIS,
                    "production_group_quality_eligible": (
                        entry.get("group_quality_eligible") is True
                        or (
                            video_id
                            in training_manifest["reference_training_source_pool"][
                                "production_selected_source_ids"
                            ]
                            and "group_quality_eligible" not in entry
                        )
                    ),
                    "identity_master": video_id == master_id,
                    "coarse_alignment": deepcopy(dict(coarse_payload)),
                    "local_timewarp": (
                        deepcopy(dict(local_payload))
                        if isinstance(local_payload, Mapping)
                        else None
                    ),
                },
                "media_probe": {
                    "frame_rate": frame_rate,
                    "declared_frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                },
            }
        )
    if tuple(item["video_id"] for item in result) != source_ids:
        raise ValueError("synchronized training source order changed")
    return tuple(result)


__all__ = [
    "ANALYSIS_BASIS",
    "CONTRACT_VERSION",
    "MINIMUM_SOURCE_COUNT",
    "PURPOSE",
    "build_reference_training_source_manifest",
    "build_synchronized_training_source_catalog",
    "build_synchronized_training_source_catalog_from_manifest",
    "validate_reference_training_source_manifest",
]
