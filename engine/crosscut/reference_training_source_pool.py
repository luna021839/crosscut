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
SUPPLEMENTAL_LINEAGE_CONTRACT_VERSION = 1
SUPPLEMENTAL_LINEAGE = "verified_historical_derived_training_source_manifest"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _regular_file(path: Path, name: str) -> Path:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{name} is not a regular non-symlink file: {source}")
    return source.resolve()


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


def _normalized_source_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of source ids")
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def _entry_content_sha256(entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(entry))).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _historical_production_flag(
    entry: Mapping[str, Any],
    *,
    production_selected: bool,
) -> bool:
    return entry.get("group_quality_eligible") is True or (
        production_selected and "group_quality_eligible" not in entry
    )


def _validate_media_admission(
    entry: Mapping[str, Any],
    *,
    allow_relative_quality_exclusion: bool,
    production_selected: bool = False,
    verify_live_file: bool = True,
    verify_full_sha256: bool = False,
) -> None:
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
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError("training source SHA-256 is invalid") from exc
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("training source byte size is invalid")
    if verify_full_sha256 and not verify_live_file:
        raise ValueError("full SHA-256 verification requires live media verification")
    if verify_live_file:
        declared_path = Path(str(entry.get("local_path") or "")).expanduser()
        path = (
            _regular_file(declared_path, "training source media")
            if verify_full_sha256
            else declared_path.resolve()
        )
        if not path.is_file():
            raise FileNotFoundError(f"training source media is missing: {path}")
        if path.stat().st_size != size:
            raise ValueError(f"training source byte size changed: {path}")
        if verify_full_sha256 and _sha256_file(path) != digest.lower():
            raise ValueError(f"training source media SHA-256 mismatch: {path}")
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


def _load_supplemental_training_manifest(
    path: Path,
    source_ids: tuple[str, ...],
    *,
    expected_artist: str,
    expected_song: str,
) -> tuple[Path, dict[str, Any], str, dict[str, Mapping[str, Any]]]:
    source_path = _regular_file(path, "supplemental training manifest")
    payload, file_hash = _load_json(source_path, "supplemental training manifest")
    _validate_reference_training_source_manifest(
        payload,
        expected_artist=expected_artist,
        expected_song=expected_song,
        verify_acquisition_binding=False,
        verify_media_files=False,
    )
    selected = _entry_by_id(payload.get("selected"), "supplemental selected")
    missing = set(source_ids) - set(selected)
    if missing:
        raise ValueError(f"supplemental sources are absent: {sorted(missing)}")
    return (
        source_path,
        payload,
        file_hash,
        {video_id: selected[video_id] for video_id in source_ids},
    )


def build_reference_training_source_manifest(
    acquisition_manifest_path: Path,
    *,
    additional_source_ids: Sequence[str],
    expected_artist: str,
    expected_song: str,
    expected_source_count: int,
    supplemental_training_manifest_path: Path | None = None,
    supplemental_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Derive a strict training-only superset from one immutable acquisition."""

    if expected_source_count < MINIMUM_SOURCE_COUNT:
        raise ValueError(f"expected_source_count must be at least {MINIMUM_SOURCE_COUNT}")
    source_path = Path(acquisition_manifest_path).expanduser().resolve()
    acquisition, acquisition_hash = _load_json(source_path, "acquisition manifest")
    if (
        acquisition.get("schema_version") != 1
        or acquisition.get("group") != expected_artist
        or acquisition.get("song") != expected_song
    ):
        raise ValueError("acquisition manifest identity/schema mismatch")
    production = _entry_by_id(acquisition.get("selected"), "selected")
    candidates = _entry_by_id(acquisition.get("candidates"), "candidates")
    extras = _normalized_source_ids(additional_source_ids, "additional source ids")
    supplemental_ids = _normalized_source_ids(
        supplemental_source_ids, "supplemental source ids"
    )
    if (supplemental_training_manifest_path is None) != (not supplemental_ids):
        raise ValueError("supplemental manifest path and source ids must be paired")
    if set(extras) & set(production):
        raise ValueError("additional source ids must not repeat production sources")
    if set(supplemental_ids) & (set(production) | set(extras)):
        raise ValueError("supplemental source ids overlap current source membership")
    if len(production) + len(extras) + len(supplemental_ids) != expected_source_count:
        raise ValueError("production plus additional sources do not match exact count")
    missing = set(extras) - set(candidates)
    if missing:
        raise ValueError(f"additional sources are absent from acquisition: {sorted(missing)}")

    supplemental_path: Path | None = None
    supplemental_manifest: dict[str, Any] | None = None
    supplemental_file_hash: str | None = None
    supplemental_selected: dict[str, Mapping[str, Any]] = {}
    historical_production: set[str] = set()
    if supplemental_ids:
        assert supplemental_training_manifest_path is not None
        (
            supplemental_path,
            supplemental_manifest,
            supplemental_file_hash,
            supplemental_selected,
        ) = _load_supplemental_training_manifest(
            supplemental_training_manifest_path,
            supplemental_ids,
            expected_artist=expected_artist,
            expected_song=expected_song,
        )
        historical_contract = supplemental_manifest["reference_training_source_pool"]
        historical_production = {
            str(value) for value in historical_contract["production_selected_source_ids"]
        }

    ordered_entries: list[dict[str, Any]] = []
    production_ids = tuple(production)
    for source_index, video_id in enumerate(
        (*production_ids, *extras, *supplemental_ids)
    ):
        supplemental = video_id in supplemental_selected
        original = (
            supplemental_selected[video_id]
            if supplemental
            else production.get(video_id) or candidates[video_id]
        )
        prior_selected = video_id in historical_production if supplemental else False
        training_only = video_id in extras
        _validate_media_admission(
            original,
            allow_relative_quality_exclusion=training_only or supplemental,
            production_selected=prior_selected if supplemental else not training_only,
            verify_full_sha256=supplemental,
        )
        prior_quality = _historical_production_flag(
            original,
            production_selected=prior_selected if supplemental else not training_only,
        )
        prior_reasons = deepcopy(
            list(original.get("group_quality_exclusion_reasons") or [])
        )
        entry = deepcopy(dict(original))
        entry.update(
            source_index=source_index,
            training_attribution_eligible=True,
            training_admission={
                "contract_version": CONTRACT_VERSION,
                "purpose": PURPOSE,
                "production_group_quality_eligible": prior_quality,
                "production_group_quality_eligibility_explicit": (
                    "group_quality_eligible" in original
                ),
                "production_group_quality_exclusion_reasons": prior_reasons,
                "hard_media_contract_passed": True,
                "production_render_admission_modified": False,
            },
        )
        if supplemental:
            entry["training_admission"].update(
                source_lineage=SUPPLEMENTAL_LINEAGE,
                supplemental_source_entry_content_sha256=(
                    _entry_content_sha256(original)
                ),
                prior_manifest_production_selected=prior_selected,
                prior_manifest_production_group_quality_eligible=prior_quality,
                prior_manifest_production_group_quality_eligibility_explicit=(
                    "group_quality_eligible" in original
                ),
                prior_manifest_production_group_quality_exclusion_reasons=prior_reasons,
                current_production_render_admission=False,
                live_media_full_sha256_revalidated=True,
            )
        ordered_entries.append(entry)

    derived = deepcopy(acquisition)
    derived["selected"] = ordered_entries
    contract: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "analysis": ANALYSIS_BASIS,
        "purpose": PURPOSE,
        "minimum_source_count": MINIMUM_SOURCE_COUNT,
        "source_count": expected_source_count,
        "source_ids": [entry["video_id"] for entry in ordered_entries],
        "production_selected_source_ids": list(production_ids),
        "training_only_additional_source_ids": list((*extras, *supplemental_ids)),
        "production_quality_gate_preserved": True,
        "acquisition_manifest": {
            "path": str(source_path),
            "file_sha256": acquisition_hash,
        },
        "manifest_content_sha256": None,
    }
    if supplemental_manifest is not None:
        assert supplemental_path is not None and supplemental_file_hash is not None
        historical_contract = supplemental_manifest["reference_training_source_pool"]
        historical_binding = deepcopy(historical_contract["acquisition_manifest"])
        path_match = Path(historical_binding["path"]).expanduser().resolve() == source_path
        hash_match = historical_binding["file_sha256"] == acquisition_hash
        contract.update(
            current_acquisition_additional_source_ids=list(extras),
            supplemental_source_ids=list(supplemental_ids),
            supplemental_training_source_manifest={
                "contract_version": SUPPLEMENTAL_LINEAGE_CONTRACT_VERSION,
                "lineage": SUPPLEMENTAL_LINEAGE,
                "path": str(supplemental_path),
                "file_sha256": supplemental_file_hash,
                "manifest_content_sha256": historical_contract[
                    "manifest_content_sha256"
                ],
                "source_ids": list(supplemental_ids),
                "source_entry_content_sha256_by_id": {
                    video_id: _entry_content_sha256(supplemental_selected[video_id])
                    for video_id in supplemental_ids
                },
                "historical_acquisition_manifest": historical_binding,
                "historical_acquisition_path_matches_current": path_match,
                "historical_acquisition_file_sha256_matches_current": hash_match,
                "historical_acquisition_binding_revalidated": False,
                "historical_acquisition_binding_drifted_from_current": not (
                    path_match and hash_match
                ),
                "current_acquisition_manifest_authoritative": True,
                "live_media_revalidation": (
                    "regular_non_symlink_size_full_sha256_hard_media_aspect"
                ),
                "historical_manifest_validation_scope": (
                    "sealed_content_requested_sources_live_only"
                ),
                "unrequested_historical_sources_live_revalidated": False,
                "unrequested_historical_sources_live_revalidation_reason": (
                    "historical_cached_paths_may_have_drifted_and_are_not_requested"
                ),
            },
        )
    derived["reference_training_source_pool"] = contract
    contract["manifest_content_sha256"] = _content_sha256(derived)
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

    return _validate_reference_training_source_manifest(
        manifest,
        expected_artist=expected_artist,
        expected_song=expected_song,
        verify_acquisition_binding=verify_acquisition_binding,
        verify_media_files=True,
    )


def _validate_reference_training_source_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_artist: str,
    expected_song: str,
    verify_acquisition_binding: bool,
    verify_media_files: bool,
) -> tuple[str, ...]:
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
    if contract.get("manifest_content_sha256") != _content_sha256(manifest):
        raise ValueError("reference training source-pool content hash mismatch")
    production_ids = contract.get("production_selected_source_ids")
    additional_ids = contract.get("training_only_additional_source_ids")
    if not isinstance(production_ids, list) or not isinstance(additional_ids, list):
        raise ValueError("reference training source-pool membership is invalid")
    if tuple(str(value) for value in (*production_ids, *additional_ids)) != ordered:
        raise ValueError("reference training source-pool membership/order mismatch")
    production_set = {str(value) for value in production_ids}
    additional_set = {str(value) for value in additional_ids}

    lineage = contract.get("supplemental_training_source_manifest")
    supplemental_ids = tuple(str(value) for value in contract.get("supplemental_source_ids", []))
    supplemental_set = set(supplemental_ids)
    current_additional = set(additional_set)
    if lineage is None:
        if supplemental_ids or "current_acquisition_additional_source_ids" in contract:
            raise ValueError("supplemental training lineage is incomplete")
    else:
        current_ids = contract.get("current_acquisition_additional_source_ids")
        hashes = lineage.get("source_entry_content_sha256_by_id") if isinstance(lineage, Mapping) else None
        historical_binding = lineage.get("historical_acquisition_manifest") if isinstance(lineage, Mapping) else None
        required = {
            "contract_version": SUPPLEMENTAL_LINEAGE_CONTRACT_VERSION,
            "lineage": SUPPLEMENTAL_LINEAGE,
            "source_ids": list(supplemental_ids),
            "historical_acquisition_binding_revalidated": False,
            "current_acquisition_manifest_authoritative": True,
            "live_media_revalidation": "regular_non_symlink_size_full_sha256_hard_media_aspect",
            "historical_manifest_validation_scope": "sealed_content_requested_sources_live_only",
            "unrequested_historical_sources_live_revalidated": False,
            "unrequested_historical_sources_live_revalidation_reason": (
                "historical_cached_paths_may_have_drifted_and_are_not_requested"
            ),
        }
        if (
            not supplemental_ids
            or not isinstance(current_ids, list)
            or tuple(str(value) for value in (*current_ids, *supplemental_ids))
            != tuple(str(value) for value in additional_ids)
            or supplemental_set & production_set
            or not isinstance(lineage, Mapping)
            or any(lineage.get(key) != value for key, value in required.items())
            or not isinstance(lineage.get("path"), str)
            or not _is_sha256(lineage.get("file_sha256"))
            or not _is_sha256(lineage.get("manifest_content_sha256"))
            or not isinstance(hashes, Mapping)
            or set(hashes) != supplemental_set
            or any(not _is_sha256(hashes.get(value)) for value in supplemental_ids)
            or not isinstance(historical_binding, Mapping)
            or not isinstance(historical_binding.get("path"), str)
            or not _is_sha256(historical_binding.get("file_sha256"))
        ):
            raise ValueError("supplemental training lineage contract is invalid")
        current_additional = {str(value) for value in current_ids}

    for entry in selected:
        assert isinstance(entry, Mapping)
        video_id = str(entry.get("video_id"))
        admission = entry.get("training_admission")
        if entry.get("training_attribution_eligible") is not True or not isinstance(
            admission, Mapping
        ) or (
            admission.get("contract_version") != CONTRACT_VERSION
            or admission.get("purpose") != PURPOSE
            or admission.get("hard_media_contract_passed") is not True
            or admission.get("production_render_admission_modified") is not False
        ):
            raise ValueError("training source admission provenance is invalid")
        supplemental = video_id in supplemental_set
        prior_selected = admission.get("prior_manifest_production_selected")
        if supplemental and not isinstance(prior_selected, bool):
            raise ValueError("supplemental prior production membership is invalid")
        production_flag = _historical_production_flag(
            entry,
            production_selected=bool(prior_selected) if supplemental else video_id in production_set,
        )
        if admission.get("production_group_quality_eligible") is not production_flag:
            raise ValueError("production quality eligibility was not preserved")
        explicit = admission.get("production_group_quality_eligibility_explicit")
        if explicit is not None and explicit is not ("group_quality_eligible" in entry):
            raise ValueError("production quality eligibility presence was not preserved")
        if video_id in production_set and not production_flag:
            raise ValueError("production source lost its production eligibility")
        if video_id in additional_set and not supplemental and production_flag:
            raise ValueError("training-only source unexpectedly has production admission")
        if supplemental and (
            admission.get("source_lineage") != SUPPLEMENTAL_LINEAGE
            or admission.get("supplemental_source_entry_content_sha256")
            != lineage["source_entry_content_sha256_by_id"][video_id]
            or admission.get("current_production_render_admission") is not False
            or admission.get("prior_manifest_production_group_quality_eligible")
            is not production_flag
            or admission.get("live_media_full_sha256_revalidated") is not True
        ):
            raise ValueError("supplemental training source admission is invalid")
        _validate_media_admission(
            entry,
            allow_relative_quality_exclusion=video_id in additional_set,
            production_selected=(
                bool(prior_selected) if supplemental else video_id in production_set
            ),
            verify_live_file=verify_media_files,
            verify_full_sha256=supplemental and verify_media_files,
        )
    binding = contract.get("acquisition_manifest")
    if not isinstance(binding, Mapping) or not _is_sha256(binding.get("file_sha256")):
        raise ValueError("acquisition manifest binding is invalid")
    bound_path = Path(str(binding.get("path") or "")).expanduser().resolve()
    if lineage is not None:
        historical_binding = lineage["historical_acquisition_manifest"]
        path_match = Path(historical_binding["path"]).expanduser().resolve() == bound_path
        hash_match = historical_binding["file_sha256"] == binding["file_sha256"]
        if (
            lineage.get("historical_acquisition_path_matches_current") is not path_match
            or lineage.get("historical_acquisition_file_sha256_matches_current") is not hash_match
            or lineage.get("historical_acquisition_binding_drifted_from_current")
            is not (not (path_match and hash_match))
        ):
            raise ValueError("supplemental acquisition drift provenance is invalid")
    if verify_acquisition_binding:
        original, observed_hash = _load_json(bound_path, "bound acquisition manifest")
        if observed_hash != binding["file_sha256"]:
            raise ValueError("bound acquisition manifest SHA-256 mismatch")
        original_ids = tuple(_entry_by_id(original.get("selected"), "selected"))
        if original_ids != tuple(str(value) for value in production_ids):
            raise ValueError("bound production selection changed")
        original_candidates = _entry_by_id(original.get("candidates"), "candidates")
        if not current_additional <= set(original_candidates):
            raise ValueError("bound acquisition no longer contains additional sources")
        if lineage is not None:
            path, historical, file_hash, historical_selected = (
                _load_supplemental_training_manifest(
                    Path(lineage["path"]),
                    supplemental_ids,
                    expected_artist=expected_artist,
                    expected_song=expected_song,
                )
            )
            historical_contract = historical["reference_training_source_pool"]
            if (
                str(path) != lineage["path"]
                or file_hash != lineage["file_sha256"]
                or historical_contract["manifest_content_sha256"]
                != lineage["manifest_content_sha256"]
                or historical_contract["acquisition_manifest"]
                != lineage["historical_acquisition_manifest"]
            ):
                raise ValueError("supplemental training manifest lineage changed")
            historical_production = {
                str(value)
                for value in historical_contract["production_selected_source_ids"]
            }
            output = _entry_by_id(selected, "training selected")
            safety_fields = (
                "local_path", "media_validation", "aspect_validation",
                "content_integrity", "download_hard_rejection_reasons",
                "rejection_reasons", "group_quality_eligible",
                "group_quality_exclusion_reasons",
            )
            for video_id in supplemental_ids:
                historical_entry = historical_selected[video_id]
                admission = output[video_id]["training_admission"]
                prior_flag = _historical_production_flag(
                    historical_entry,
                    production_selected=video_id in historical_production,
                )
                if (
                    _entry_content_sha256(historical_entry)
                    != lineage["source_entry_content_sha256_by_id"][video_id]
                    or any(
                        output[video_id].get(key) != historical_entry.get(key)
                        for key in safety_fields
                    )
                    or admission.get("prior_manifest_production_selected")
                    is not (video_id in historical_production)
                    or admission.get(
                        "prior_manifest_production_group_quality_eligible"
                    )
                    is not prior_flag
                    or admission.get(
                        "prior_manifest_production_group_quality_eligibility_explicit"
                    )
                    is not ("group_quality_eligible" in historical_entry)
                    or admission.get(
                        "prior_manifest_production_group_quality_exclusion_reasons"
                    )
                    != list(
                        historical_entry.get("group_quality_exclusion_reasons") or []
                    )
                ):
                    raise ValueError("supplemental source lineage changed")
    return ordered


def _supplemental_catalog_provenance(
    training_manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    contract = training_manifest.get("reference_training_source_pool")
    if not isinstance(contract, Mapping) or not isinstance(
        contract.get("supplemental_training_source_manifest"), Mapping
    ):
        return {}
    admission = entry.get("training_admission")
    assert isinstance(admission, Mapping)
    video_id = str(entry["video_id"])
    facts: dict[str, Any] = {
        "production_group_quality_eligible": admission[
            "production_group_quality_eligible"
        ],
        "current_production_render_admission": video_id
        in {str(value) for value in contract["production_selected_source_ids"]},
        "training_attribution_eligible": entry["training_attribution_eligible"],
    }
    for key in ("source_lineage", "prior_manifest_production_selected"):
        if key in admission:
            facts[key] = deepcopy(admission[key])
    return facts


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
            catalog_facts = _supplemental_catalog_provenance(
                training_manifest, selected_by_id[video_id]
            )
            if catalog_facts:
                existing = source.get("provenance")
                source["provenance"] = (
                    deepcopy(dict(existing)) if isinstance(existing, Mapping) else {}
                )
                for optional_key in (
                    "source_lineage",
                    "prior_manifest_production_selected",
                ):
                    if optional_key not in catalog_facts:
                        source["provenance"].pop(optional_key, None)
                source["provenance"].update(catalog_facts)
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
        provenance = {
            "coarse_score": coarse.score,
            "coarse_margin": coarse.score_margin,
            "training_source_pool_analysis": ANALYSIS_BASIS,
            "production_group_quality_eligible": False,
        }
        provenance.update(_supplemental_catalog_provenance(training_manifest, entry))
        result.append(
            {
                "video_id": video_id,
                "path": str(path),
                "clock": clock,
                "provenance": provenance,
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

        provenance = {
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
        }
        provenance.update(_supplemental_catalog_provenance(training_manifest, entry))
        result.append(
            {
                "video_id": video_id,
                "path": str(path),
                "clock": clock,
                "provenance": provenance,
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
