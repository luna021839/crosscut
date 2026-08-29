"""Fail-closed persistent cache for source-local numeric visual features.

The cache deliberately stores no decoded image, compressed frame, SFace
embedding, or other appearance-identity vector.  One record is bound to the
SHA-256 of the complete source file, an exact canonical frame identity, the
descriptor request options, and a fingerprint of the analyzer schema, model
assets, runtime, and implementation files.

Records are independent and atomically replaced.  An interrupted analysis can
therefore reuse every descriptor completed before the interruption without
making the timeline optimizer, reviewer, exact-eight gate, or decoded-proxy
gate persistent.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Mapping, Sequence
import uuid


VISUAL_FEATURE_CACHE_CONTRACT = "ive_source_local_visual_features_v1"
VISUAL_FEATURE_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXIMUM_RECORD_BYTES = 16 * 1024 * 1024
_FORBIDDEN_PAYLOAD_KEY_PARTS = (
    "raw_frame",
    "pixels",
    "image_bytes",
    "compressed_frame",
    "sface",
    "face_identity_embedding",
    "appearance_embedding",
    "orb_descriptor",
)
_STABLE_FILE_DIGEST_PROCESS_CACHE: dict[
    tuple[str, tuple[int, int, int, int, int]], str
] = {}
_STABLE_FILE_DIGEST_PROCESS_CACHE_LIMIT = 64
_MAXIMUM_DECODE_FREE_LOCATOR_RECORDS = 4_096
_LOCATOR_INDEX_OVER_BUDGET_PROCESS_ROOTS: set[str] = set()
# These exact visual-score changes were audited as descriptor-equivalent: one
# moves an already-persisted lookup ahead of frame decoding and the other adds
# section-role candidate-selection controls.  Neither alters numeric feature
# extraction.  Bind each migration to the exact post-change analyzer digest so
# any future visual-score edit automatically disables legacy reuse.
_DECODE_EQUIVALENT_ANALYZER_MIGRATIONS: Mapping[str, tuple[str, ...]] = {
    # First descriptor-AST fingerprint.  The exact legacy visual_score digest
    # was audited before switching from whole-module to dependency-only code
    # identity; future selector edits retain this analyzer digest automatically.
    "ac7de83aac79715f25f4a7c93f2d861b93047e5a92e89694496c2d7b3d3fb47d": (
        "e09b787bf407c2eacc09f1d1f52a0318dc8b1e981b70eaf199c863cf67af0829",
        "b7e9ad896ed5403819b8fa31772a97b4f82365ba5f65a032c22d024233f83d4a",
        "ba991e083dd4e94ef7d4a251bc05538d1bb2feaf0582408249090d2908e0b23b",
        "19740a80bc18933032e9265a747a74678a4ff7cfa22e26ddb0ea14528705d6d1",
        "c44b51f780f0d2b6e37751c2053f8630b63d3e799cd617936333ecace435da8b",
        "5d32ea38abfbad7fac015c91f00e8957ac2db95a460778e3a207c16ae3ca9fe2",
        "976479db2b02b46830aa53e55c20b03aabd5bdb660aee9795d81e2587b2cab93",
        "b7d906f1578b0bb967ad9c331a1c6ad3c7147cb596e0ba155af41c80f667a7a4",
        "48f2ed8ef454539ac3e8502c905c325b1113158c7bd840de6aefac83105ce3d1",
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
    "e3196dd457bfcf0d3227475e21083d04b808fd331e1c2a1dfe7890510ee2e8e4": (
        "dd214dcfe13d3af7c5119a5ea3067fe7ec703fd8ff864b7fa34c11f95622f433",
    ),
    "48f2ed8ef454539ac3e8502c905c325b1113158c7bd840de6aefac83105ce3d1": (
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
    "b7d906f1578b0bb967ad9c331a1c6ad3c7147cb596e0ba155af41c80f667a7a4": (
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
    "976479db2b02b46830aa53e55c20b03aabd5bdb660aee9795d81e2587b2cab93": (
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
    "5d32ea38abfbad7fac015c91f00e8957ac2db95a460778e3a207c16ae3ca9fe2": (
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
    "c44b51f780f0d2b6e37751c2053f8630b63d3e799cd617936333ecace435da8b": (
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
    "19740a80bc18933032e9265a747a74678a4ff7cfa22e26ddb0ea14528705d6d1": (
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
    "ba991e083dd4e94ef7d4a251bc05538d1bb2feaf0582408249090d2908e0b23b": (
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
    "b7e9ad896ed5403819b8fa31772a97b4f82365ba5f65a032c22d024233f83d4a": (
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
    "e09b787bf407c2eacc09f1d1f52a0318dc8b1e981b70eaf199c863cf67af0829": (
        "809393a519d0699ec8aaed96e161cc275c823bd3164a180343cfd7f67ccfc4ee",
    ),
}


def _stable_sha256(path: Path) -> tuple[str, tuple[int, int, int, int, int]]:
    """Hash one unchanged regular file and return its stable stat identity."""

    resolved = Path(path).expanduser().resolve(strict=True)
    before = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"visual feature source is not a regular file: {resolved}")
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    cache_key = (str(resolved), before_identity)
    cached_digest = _STABLE_FILE_DIGEST_PROCESS_CACHE.get(cache_key)
    if cached_digest is not None:
        after = resolved.stat()
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise OSError("visual feature source changed while SHA-256 was reused")
        return (cached_digest, after_identity)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = resolved.stat()
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise OSError("visual feature source changed while SHA-256 was computed")
    resolved_digest = digest.hexdigest()
    for stale_key in tuple(_STABLE_FILE_DIGEST_PROCESS_CACHE):
        if stale_key[0] == str(resolved) and stale_key != cache_key:
            _STABLE_FILE_DIGEST_PROCESS_CACHE.pop(stale_key, None)
    if len(_STABLE_FILE_DIGEST_PROCESS_CACHE) >= (
        _STABLE_FILE_DIGEST_PROCESS_CACHE_LIMIT
    ):
        _STABLE_FILE_DIGEST_PROCESS_CACHE.pop(
            next(iter(_STABLE_FILE_DIGEST_PROCESS_CACHE))
        )
    _STABLE_FILE_DIGEST_PROCESS_CACHE[cache_key] = resolved_digest
    return (resolved_digest, after_identity)


def _file_digest(path: Path) -> str:
    digest, _identity = _stable_sha256(path)
    return digest


@dataclass(frozen=True, slots=True)
class AnalyzerFingerprint:
    """Stable analyzer identity included in every cache record and path."""

    digest: str
    components: Mapping[str, Any]

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.digest) is None:
            raise ValueError("analyzer fingerprint must be a lowercase SHA-256")


def build_analyzer_fingerprint(
    *,
    descriptor_schema: str,
    code_paths: Sequence[Path],
    code_digests: Mapping[str, str] | None = None,
    model_assets: Mapping[str, str],
    runtime_components: Mapping[str, str],
) -> AnalyzerFingerprint:
    """Fingerprint every input that can alter a numeric frame descriptor."""

    schema = descriptor_schema.strip()
    if not schema:
        raise ValueError("descriptor_schema cannot be blank")
    code: dict[str, str] = {}
    for path in sorted(
        (Path(item).expanduser().resolve(strict=True) for item in code_paths),
        key=lambda item: item.name,
    ):
        if path.name in code:
            raise ValueError("analyzer code path names must be unique")
        code[path.name] = _file_digest(path)
    for raw_name, raw_digest in sorted((code_digests or {}).items()):
        name = str(raw_name).strip()
        digest = str(raw_digest).strip()
        if not name:
            raise ValueError("analyzer code digest name cannot be blank")
        if name in code:
            raise ValueError("analyzer code digest names must be unique")
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("analyzer code digest must be a lowercase SHA-256")
        code[name] = digest
    models = {str(key): str(value) for key, value in sorted(model_assets.items())}
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        **{
            str(key): str(value)
            for key, value in sorted(runtime_components.items())
        },
    }
    components: dict[str, Any] = {
        "cache_contract": VISUAL_FEATURE_CACHE_CONTRACT,
        "feature_schema_version": VISUAL_FEATURE_SCHEMA_VERSION,
        "descriptor_schema": schema,
        "code_sha256": code,
        "model_asset_sha256_or_state": models,
        "runtime": runtime,
    }
    encoded = json.dumps(
        components,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AnalyzerFingerprint(
        digest=hashlib.sha256(encoded).hexdigest(),
        components=components,
    )


def python_symbol_dependency_sha256(
    path: Path,
    *,
    root_symbols: Sequence[str],
) -> str:
    """Hash only the module AST that can affect named analyzer roots.

    A feature extractor and a timeline selector can live in the same Python
    module without sharing cache invalidation.  Starting at the supplied
    classes/functions/constants, this follows module-local loaded names through
    definitions and imports, then hashes canonical AST dumps without source
    locations or comments.  Changing an analyzer helper therefore invalidates
    descriptors, while changing an unrelated editor policy does not.
    """

    resolved = Path(path).expanduser().resolve(strict=True)
    source = resolved.read_text(encoding="utf-8")
    module = ast.parse(source, filename=resolved.name)
    definitions: dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                definitions[local_name] = node
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions[node.target.id] = node

    roots = tuple(str(value).strip() for value in root_symbols)
    if not roots or any(not value for value in roots):
        raise ValueError("root_symbols must contain non-blank names")
    missing = sorted(set(roots) - set(definitions))
    if missing:
        raise ValueError(
            "analyzer root symbols are missing: " + ", ".join(missing)
        )

    pending = list(dict.fromkeys(roots))
    included: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if name in included:
            continue
        node = definitions[name]
        included[name] = ast.dump(
            node,
            annotate_fields=True,
            include_attributes=False,
        )
        referenced = sorted(
            {
                item.id
                for item in ast.walk(node)
                if isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Load)
                and item.id in definitions
            }
        )
        pending.extend(
            value for value in reversed(referenced) if value not in included
        )

    encoded = json.dumps(
        {
            "contract": "python_module_symbol_dependency_ast_v1",
            "module_name": resolved.name,
            "root_symbols": sorted(set(roots)),
            "symbol_ast": {key: included[key] for key in sorted(included)},
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalFrameIdentity:
    """Exact decoded-frame identity; never a rounded millisecond timestamp."""

    frame_index: int
    fps_numerator: int
    fps_denominator: int
    source_time_numerator: int | None = None
    source_time_denominator: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if (
            isinstance(self.fps_numerator, bool)
            or not isinstance(self.fps_numerator, int)
            or self.fps_numerator <= 0
            or isinstance(self.fps_denominator, bool)
            or not isinstance(self.fps_denominator, int)
            or self.fps_denominator <= 0
        ):
            raise ValueError("canonical FPS ratio must contain positive integers")
        if self.source_time_numerator is None and self.source_time_denominator is None:
            object.__setattr__(
                self,
                "source_time_numerator",
                self.frame_index * self.fps_denominator,
            )
            object.__setattr__(self, "source_time_denominator", self.fps_numerator)
        if (
            isinstance(self.source_time_numerator, bool)
            or not isinstance(self.source_time_numerator, int)
            or self.source_time_numerator < 0
            or isinstance(self.source_time_denominator, bool)
            or not isinstance(self.source_time_denominator, int)
            or self.source_time_denominator <= 0
        ):
            raise ValueError("canonical source time ratio must be non-negative")

    @classmethod
    def from_seconds(
        cls,
        seconds: float,
        *,
        frame_rate: float,
    ) -> "CanonicalFrameIdentity":
        if not math.isfinite(seconds):
            raise ValueError("frame timestamp must be finite")
        if not math.isfinite(frame_rate) or frame_rate <= 0.0:
            raise ValueError("frame rate must be positive and finite")
        fps = Fraction(str(float(frame_rate))).limit_denominator(1_000_000)
        timestamp = Fraction(str(max(0.0, float(seconds))))
        scaled = timestamp * fps
        # Match the historical nearest-frame intent without reducing the key
        # to milliseconds.  Half frames round away from zero (all inputs are
        # non-negative), so the result is stable across Python versions.
        frame_index = (2 * scaled.numerator + scaled.denominator) // (
            2 * scaled.denominator
        )
        return cls(
            frame_index=frame_index,
            fps_numerator=fps.numerator,
            fps_denominator=fps.denominator,
            source_time_numerator=timestamp.numerator,
            source_time_denominator=timestamp.denominator,
        )

    @property
    def canonical_seconds(self) -> float:
        assert self.source_time_numerator is not None
        assert self.source_time_denominator is not None
        return float(self.source_time_numerator) / float(self.source_time_denominator)

    def to_dict(self) -> dict[str, int]:
        assert self.source_time_numerator is not None
        assert self.source_time_denominator is not None
        return {
            "frame_index": self.frame_index,
            "fps_numerator": self.fps_numerator,
            "fps_denominator": self.fps_denominator,
            "source_time_numerator": self.source_time_numerator,
            "source_time_denominator": self.source_time_denominator,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalFrameIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "frame_index",
            "fps_numerator",
            "fps_denominator",
            "source_time_numerator",
            "source_time_denominator",
        }:
            raise ValueError("canonical frame cache identity schema is invalid")
        return cls(
            frame_index=value["frame_index"],
            fps_numerator=value["fps_numerator"],
            fps_denominator=value["fps_denominator"],
            source_time_numerator=value["source_time_numerator"],
            source_time_denominator=value["source_time_denominator"],
        )


@dataclass(frozen=True, slots=True)
class DescriptorRequestLocator:
    """Decode-free address for one deterministic descriptor request.

    The complete source SHA-256 and analyzer fingerprint live in the cache
    directory and record identity.  This locator adds the exact rational frame
    requests and feature flags.  Working-frame hashes remain in the full record
    for audit and exact fallback, but are deliberately not required to discover
    an existing record: requiring them would decode every frame before a warm
    cache lookup and defeat interruption-safe reuse.
    """

    primary: CanonicalFrameIdentity
    following: CanonicalFrameIdentity | None
    include_subject_geometry: bool
    include_formation_pose: bool

    def __post_init__(self) -> None:
        if not isinstance(self.include_subject_geometry, bool) or not isinstance(
            self.include_formation_pose, bool
        ):
            raise TypeError("descriptor feature flags must be booleans")
        if self.include_formation_pose and not self.include_subject_geometry:
            raise ValueError("formation pose requires subject geometry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "following": (
                self.following.to_dict() if self.following is not None else None
            ),
            "include_subject_geometry": self.include_subject_geometry,
            "include_formation_pose": self.include_formation_pose,
        }


@dataclass(frozen=True, slots=True)
class DescriptorRequestIdentity:
    primary: CanonicalFrameIdentity
    following: CanonicalFrameIdentity | None
    primary_working_frame_sha256: str
    following_working_frame_sha256: str | None
    include_subject_geometry: bool
    include_formation_pose: bool

    def __post_init__(self) -> None:
        if not isinstance(self.include_subject_geometry, bool) or not isinstance(
            self.include_formation_pose, bool
        ):
            raise TypeError("descriptor feature flags must be booleans")
        if self.include_formation_pose and not self.include_subject_geometry:
            raise ValueError("formation pose requires subject geometry")
        if _SHA256.fullmatch(self.primary_working_frame_sha256) is None:
            raise ValueError("primary working frame must have a SHA-256 identity")
        if (self.following is None) != (
            self.following_working_frame_sha256 is None
        ):
            raise ValueError("following frame identity and SHA-256 must agree")
        if (
            self.following_working_frame_sha256 is not None
            and _SHA256.fullmatch(self.following_working_frame_sha256) is None
        ):
            raise ValueError("following working frame must have a SHA-256 identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "following": (
                self.following.to_dict() if self.following is not None else None
            ),
            "primary_working_frame_sha256": self.primary_working_frame_sha256,
            "following_working_frame_sha256": (
                self.following_working_frame_sha256
            ),
            "include_subject_geometry": self.include_subject_geometry,
            "include_formation_pose": self.include_formation_pose,
        }

    @property
    def locator(self) -> DescriptorRequestLocator:
        return DescriptorRequestLocator(
            primary=self.primary,
            following=self.following,
            include_subject_geometry=self.include_subject_geometry,
            include_formation_pose=self.include_formation_pose,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DescriptorRequestIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "primary",
            "following",
            "primary_working_frame_sha256",
            "following_working_frame_sha256",
            "include_subject_geometry",
            "include_formation_pose",
        }:
            raise ValueError("descriptor request cache identity schema is invalid")
        raw_following = value["following"]
        return cls(
            primary=CanonicalFrameIdentity.from_dict(value["primary"]),
            following=(
                CanonicalFrameIdentity.from_dict(raw_following)
                if raw_following is not None
                else None
            ),
            primary_working_frame_sha256=value[
                "primary_working_frame_sha256"
            ],
            following_working_frame_sha256=value[
                "following_working_frame_sha256"
            ],
            include_subject_geometry=value["include_subject_geometry"],
            include_formation_pose=value["include_formation_pose"],
        )


def _validate_derived_payload(value: Any, *, key_path: str = "payload") -> None:
    """Reject non-JSON, non-finite, or explicitly forbidden media payloads."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{key_path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_derived_payload(item, key_path=f"{key_path}[{index}]")
        return
    if isinstance(value, dict):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{key_path} contains a non-string key")
            folded = raw_key.casefold()
            if any(part in folded for part in _FORBIDDEN_PAYLOAD_KEY_PARTS):
                raise ValueError(
                    f"{key_path}.{raw_key} is forbidden in the visual feature cache"
                )
            _validate_derived_payload(item, key_path=f"{key_path}.{raw_key}")
        return
    raise TypeError(f"{key_path} contains unsupported type {type(value).__name__}")


class PersistentVisualFeatureCache:
    """Per-source content-addressed descriptor records with atomic writes."""

    def __init__(
        self,
        *,
        source_cache_directory: Path,
        source_path: Path,
        analyzer: AnalyzerFingerprint,
    ) -> None:
        cache_root = Path(source_cache_directory).expanduser()
        if cache_root.is_symlink():
            raise ValueError("source cache directory cannot be a symbolic link")
        cache_root.mkdir(parents=True, exist_ok=True)
        if not cache_root.is_dir():
            raise ValueError("source cache directory must be a directory")
        self.source_path = Path(source_path).expanduser().resolve(strict=True)
        self.source_sha256, self._source_stat_identity = _stable_sha256(
            self.source_path
        )
        self.analyzer = analyzer
        source_root = self._safe_directory(cache_root, "derived")
        source_root = self._safe_directory(source_root, "visual-v1")
        source_root = self._safe_directory(
            source_root,
            self.source_sha256[:2],
        )
        source_root = self._safe_directory(source_root, self.source_sha256)
        self._root = self._safe_directory(source_root, analyzer.digest)
        self._compatible_legacy_analyzer_digests = tuple(
            _DECODE_EQUIVALENT_ANALYZER_MIGRATIONS.get(analyzer.digest, ())
        )
        lookup_roots = [self._root]
        for legacy_digest in self._compatible_legacy_analyzer_digests:
            if _SHA256.fullmatch(legacy_digest) is None:
                raise ValueError("legacy analyzer migration digest is invalid")
            legacy_root = source_root / legacy_digest
            if legacy_root.is_symlink():
                raise ValueError(
                    f"visual feature legacy cache path is a symlink: {legacy_root}"
                )
            if legacy_root.is_dir():
                lookup_roots.append(legacy_root)
        self._lookup_roots = tuple(lookup_roots)
        self._hit_count = 0
        self._miss_count = 0
        self._write_count = 0
        self._invalid_count = 0
        self._disabled_by_source_change = False
        self._locator_index: dict[
            DescriptorRequestLocator, list[Path]
        ] | None = None
        self._locator_index_build_count = 0
        self._decode_free_hit_count = 0
        self._ambiguous_locator_count = 0
        self._decode_free_legacy_hit_count = 0
        self._locator_index_status = "not_built"
        self._locator_index_preflight_record_count = 0
        self._locator_index_over_budget_root: str | None = None
        self._locator_index_budget_fallback_count = 0
        self._locator_index_process_memo_hit_count = 0

    @staticmethod
    def _safe_directory(parent: Path, name: str) -> Path:
        child = parent / name
        if child.is_symlink():
            raise ValueError(f"visual feature cache path is a symlink: {child}")
        child.mkdir(exist_ok=True)
        if not child.is_dir():
            raise ValueError(f"visual feature cache path is not a directory: {child}")
        return child

    def _source_is_unchanged(self) -> bool:
        if self._disabled_by_source_change:
            return False
        try:
            stat = self.source_path.stat()
        except OSError:
            self._disabled_by_source_change = True
            return False
        identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if identity != self._source_stat_identity:
            self._disabled_by_source_change = True
            return False
        return True

    def _record_identity(
        self,
        request: DescriptorRequestIdentity,
    ) -> dict[str, Any]:
        return {
            "cache_contract": VISUAL_FEATURE_CACHE_CONTRACT,
            "feature_schema_version": VISUAL_FEATURE_SCHEMA_VERSION,
            "source_content": {
                "algorithm": "sha256",
                "sha256": self.source_sha256,
                "size_bytes": self._source_stat_identity[2],
            },
            "analyzer": {
                "fingerprint_sha256": self.analyzer.digest,
                "components": dict(self.analyzer.components),
            },
            "descriptor_request": request.to_dict(),
        }

    def _record_identity_matches_analyzer_contract(
        self,
        actual_identity: Mapping[str, Any],
        request: DescriptorRequestIdentity,
    ) -> bool:
        expected = self._record_identity(request)
        if actual_identity == expected:
            return True
        if set(actual_identity) != set(expected):
            return False
        if any(
            actual_identity.get(key) != expected.get(key)
            for key in expected
            if key != "analyzer"
        ):
            return False
        actual_analyzer = actual_identity.get("analyzer")
        expected_analyzer = expected["analyzer"]
        if not isinstance(actual_analyzer, Mapping) or set(actual_analyzer) != {
            "fingerprint_sha256",
            "components",
        }:
            return False
        legacy_digest = actual_analyzer.get("fingerprint_sha256")
        if legacy_digest not in self._compatible_legacy_analyzer_digests:
            return False
        actual_components = actual_analyzer.get("components")
        expected_components = expected_analyzer["components"]
        if not isinstance(actual_components, Mapping) or not isinstance(
            expected_components,
            Mapping,
        ):
            return False
        if set(actual_components) != set(expected_components):
            return False
        if any(
            actual_components.get(key) != expected_components.get(key)
            for key in expected_components
            if key != "code_sha256"
        ):
            return False
        actual_code = actual_components.get("code_sha256")
        expected_code = expected_components.get("code_sha256")
        if not isinstance(actual_code, Mapping) or not isinstance(
            expected_code,
            Mapping,
        ):
            return False
        actual_names = set(actual_code)
        expected_names = set(expected_code)
        if actual_names == expected_names:
            return all(
                actual_code.get(name) == expected_code.get(name)
                for name in expected_code
                if name != "visual_score.py"
            )
        if (
            actual_names - expected_names == {"visual_score.py"}
            and expected_names - actual_names
            == {"visual_score_descriptor_ast_v1"}
        ):
            # The exact analyzer-digest migration above is the explicit audit
            # authority for replacing the old whole-file hash.  Every shared
            # dependency (currently pose.py) must still match byte-for-byte.
            return all(
                actual_code.get(name) == expected_code.get(name)
                for name in actual_names & expected_names
            )
        return False

    def _record_path(self, request: DescriptorRequestIdentity) -> Path:
        identity = self._record_identity(request)
        encoded = json.dumps(
            identity,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        shard = self._safe_directory(self._root, key[:2])
        return shard / f"{key}.json"

    def _read_record(
        self,
        path: Path,
        *,
        expected_request: DescriptorRequestIdentity | None = None,
        expected_locator: DescriptorRequestLocator | None = None,
        validate_payload: bool = True,
    ) -> tuple[DescriptorRequestIdentity, dict[str, Any] | None]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("visual feature cache record path is invalid")
        size = path.stat().st_size
        if not 1 <= size <= _MAXIMUM_RECORD_BYTES:
            raise ValueError("visual feature cache record size is invalid")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("visual feature cache record is not an object")
        actual_identity = record.get("identity")
        if not isinstance(actual_identity, Mapping):
            raise ValueError("visual feature cache identity is not an object")
        request = DescriptorRequestIdentity.from_dict(
            actual_identity.get("descriptor_request")
        )
        if not self._record_identity_matches_analyzer_contract(
            actual_identity,
            request,
        ):
            raise ValueError("visual feature cache identity mismatch")
        if expected_request is not None and request != expected_request:
            raise ValueError("visual feature cache request mismatch")
        if expected_locator is not None and request.locator != expected_locator:
            raise ValueError("visual feature cache locator mismatch")
        if not validate_payload:
            return (request, None)
        payload = record.get("numeric_features")
        if not isinstance(payload, dict):
            raise ValueError("visual feature cache payload is not an object")
        _validate_derived_payload(payload)
        expected_payload_sha256 = record.get("numeric_features_sha256")
        actual_payload_sha256 = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            not isinstance(expected_payload_sha256, str)
            or not hmac.compare_digest(
                expected_payload_sha256,
                actual_payload_sha256,
            )
        ):
            raise ValueError("visual feature cache payload checksum mismatch")
        return (request, payload)

    @staticmethod
    def _locator_root_memo_key(root: Path) -> str:
        return str(root.absolute())

    @staticmethod
    def _preflight_locator_record_paths(
        lookup_root: Path,
        *,
        maximum_records: int,
    ) -> tuple[tuple[Path, ...], bool]:
        """List bounded record paths without opening or parsing any JSON."""

        records: list[Path] = []
        if lookup_root.is_symlink() or not lookup_root.is_dir():
            return (), False
        try:
            shards = sorted(
                lookup_root.iterdir(),
                key=lambda item: item.name,
            )
        except OSError:
            return (), False
        for shard in shards:
            if (
                shard.is_symlink()
                or not shard.is_dir()
                or re.fullmatch(r"[0-9a-f]{2}", shard.name) is None
            ):
                continue
            try:
                shard_records = sorted(
                    shard.iterdir(),
                    key=lambda item: item.name,
                )
            except OSError:
                continue
            for path in shard_records:
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.suffix != ".json"
                    or _SHA256.fullmatch(path.stem) is None
                ):
                    continue
                records.append(path)
                if len(records) > maximum_records:
                    return tuple(records), True
        return tuple(records), False

    def _ensure_locator_index(self) -> bool:
        if self._locator_index is not None:
            return True
        if self._locator_index_status.startswith("over_budget"):
            return False

        memoized_root = next(
            (
                root
                for root in self._lookup_roots
                if self._locator_root_memo_key(root)
                in _LOCATOR_INDEX_OVER_BUDGET_PROCESS_ROOTS
            ),
            None,
        )
        if memoized_root is not None:
            self._locator_index_status = (
                "over_budget_process_memo_exact_decode_fallback"
            )
            self._locator_index_over_budget_root = str(memoized_root)
            self._locator_index_process_memo_hit_count += 1
            return False

        # Preflight every lookup root before parsing the first record.  If the
        # aggregate would exceed the bounded decode-free budget, the exact
        # post-decode request lookup remains available and no cache JSON has
        # been opened merely to build this optional index.
        preflight_paths: list[Path] = []
        remaining = _MAXIMUM_DECODE_FREE_LOCATOR_RECORDS
        for lookup_root in self._lookup_roots:
            paths, exceeded = self._preflight_locator_record_paths(
                lookup_root,
                maximum_records=remaining,
            )
            self._locator_index_preflight_record_count += len(paths)
            if exceeded:
                memo_key = self._locator_root_memo_key(lookup_root)
                _LOCATOR_INDEX_OVER_BUDGET_PROCESS_ROOTS.add(memo_key)
                self._locator_index_status = (
                    "over_budget_exact_decode_fallback"
                )
                self._locator_index_over_budget_root = str(lookup_root)
                return False
            preflight_paths.extend(paths)
            remaining -= len(paths)

        index: dict[DescriptorRequestLocator, list[Path]] = {}
        for path in preflight_paths:
            try:
                request, _payload = self._read_record(
                    path,
                    validate_payload=False,
                )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                self._invalid_count += 1
                continue
            index.setdefault(request.locator, []).append(path)
        self._locator_index = index
        self._locator_index_build_count += 1
        self._locator_index_status = "built"
        return True

    def get_by_locator(
        self,
        locator: DescriptorRequestLocator,
    ) -> dict[str, Any] | None:
        """Reuse one uniquely bound descriptor before decoding its frame.

        A full source hash, analyzer/runtime fingerprint, exact rational frame
        request, and feature flags already bind the deterministic decode.  The
        stored working-frame SHA-256 is still validated as record metadata and
        remains available to the exact post-decode lookup.  If more than one
        full record exists for the same locator, reuse fails closed so a fresh
        decode can resolve the exact working-frame identity.
        """

        if not self._source_is_unchanged():
            self._miss_count += 1
            return None
        if not self._ensure_locator_index():
            self._locator_index_budget_fallback_count += 1
            self._miss_count += 1
            return None
        assert self._locator_index is not None
        paths = self._locator_index.get(locator, ())
        candidates: tuple[Path, ...] = ()
        # The same exact locator can legitimately exist in several audited
        # analyzer generations. Prefer the newest configured root with any
        # records; ambiguity inside that one root still fails closed.
        for lookup_root in self._lookup_roots:
            root_candidates = tuple(
                path for path in paths if path.is_relative_to(lookup_root)
            )
            if root_candidates:
                candidates = root_candidates
                break
        if len(candidates) != 1:
            if len(candidates) > 1:
                self._ambiguous_locator_count += 1
            self._miss_count += 1
            return None
        try:
            _request, payload = self._read_record(
                candidates[0],
                expected_locator=locator,
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            self._invalid_count += 1
            self._miss_count += 1
            return None
        assert payload is not None
        self._hit_count += 1
        self._decode_free_hit_count += 1
        if not candidates[0].is_relative_to(self._root):
            self._decode_free_legacy_hit_count += 1
        return payload

    def get(self, request: DescriptorRequestIdentity) -> dict[str, Any] | None:
        if not self._source_is_unchanged():
            self._miss_count += 1
            return None
        path = self._record_path(request)
        if path.is_symlink() or not path.is_file():
            self._miss_count += 1
            return None
        try:
            _actual_request, payload = self._read_record(
                path,
                expected_request=request,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            self._invalid_count += 1
            self._miss_count += 1
            return None
        assert payload is not None
        self._hit_count += 1
        return payload

    def put(
        self,
        request: DescriptorRequestIdentity,
        payload: Mapping[str, Any],
    ) -> bool:
        if not self._source_is_unchanged():
            return False
        safe_payload = dict(payload)
        _validate_derived_payload(safe_payload)
        payload_encoding = json.dumps(
            safe_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record = {
            "identity": self._record_identity(request),
            "numeric_features": safe_payload,
            "numeric_features_sha256": hashlib.sha256(
                payload_encoding
            ).hexdigest(),
            "privacy": {
                "raw_frames_persisted": False,
                "compressed_frames_persisted": False,
                "face_identity_embeddings_persisted": False,
                "appearance_embeddings_persisted": False,
            },
        }
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if not 1 <= len(encoded) <= _MAXIMUM_RECORD_BYTES:
            raise ValueError("visual feature cache record is too large")
        path = self._record_path(request)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except OSError:
            # A partial temporary is deliberately ignored by readers.  Do not
            # delete it here: interruption evidence remains recoverable and no
            # cleanup operation is needed for correctness.
            return False
        if self._locator_index is not None:
            paths = self._locator_index.setdefault(request.locator, [])
            if path not in paths:
                paths.append(path)
        self._write_count += 1
        return True

    def diagnostics(self) -> dict[str, Any]:
        return {
            "cache_contract": VISUAL_FEATURE_CACHE_CONTRACT,
            "source_sha256": self.source_sha256,
            "analyzer_fingerprint_sha256": self.analyzer.digest,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "write_count": self._write_count,
            "invalid_record_count": self._invalid_count,
            "decode_free_hit_count": self._decode_free_hit_count,
            "decode_free_legacy_hit_count": (
                self._decode_free_legacy_hit_count
            ),
            "locator_index_build_count": self._locator_index_build_count,
            "locator_index_status": self._locator_index_status,
            "locator_index_record_budget": (
                _MAXIMUM_DECODE_FREE_LOCATOR_RECORDS
            ),
            "locator_index_preflight_record_count": (
                self._locator_index_preflight_record_count
            ),
            "locator_index_over_budget_root": (
                self._locator_index_over_budget_root
            ),
            "locator_index_budget_fallback_count": (
                self._locator_index_budget_fallback_count
            ),
            "locator_index_process_memo_hit_count": (
                self._locator_index_process_memo_hit_count
            ),
            "ambiguous_locator_count": self._ambiguous_locator_count,
            "disabled_by_source_change": self._disabled_by_source_change,
            "raw_frames_persisted": False,
            "face_identity_embeddings_persisted": False,
        }


__all__ = [
    "AnalyzerFingerprint",
    "CanonicalFrameIdentity",
    "DescriptorRequestIdentity",
    "DescriptorRequestLocator",
    "PersistentVisualFeatureCache",
    "VISUAL_FEATURE_CACHE_CONTRACT",
    "VISUAL_FEATURE_SCHEMA_VERSION",
    "build_analyzer_fingerprint",
    "python_symbol_dependency_sha256",
]
