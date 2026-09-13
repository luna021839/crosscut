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
import stat
from typing import Any, Mapping, Sequence
import uuid


VISUAL_FEATURE_CACHE_CONTRACT = "ive_source_local_visual_features_v1"
VISUAL_FEATURE_SCHEMA_VERSION = 1
PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT = (
    "python_module_symbol_dependency_ast_v2"
)
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
# One 3-minute stage across exact-offset requests and compatible generations
# already exceeds 4,096 records. This is a metadata-only index (not pixels or
# descriptors); retain a bounded full-song budget instead of disabling all
# decode-free reuse halfway through ordinary production source caches.
_MAXIMUM_DECODE_FREE_LOCATOR_RECORDS = 32_768
_LOCATOR_INDEX_OVER_BUDGET_PROCESS_ROOTS: set[str] = set()


class PersistentVisualFeatureCacheIOError(RuntimeError):
    """Base class for cache-storage failures that are not record corruption."""


class PersistentVisualFeatureCacheStorageUnavailableError(
    PersistentVisualFeatureCacheIOError
):
    """The initialized cache root disappeared or changed identity."""


class PersistentVisualFeatureCacheWriteError(PersistentVisualFeatureCacheIOError):
    """An atomic persistent-cache record write could not be completed."""


def assert_visual_feature_derived_cache_root_available(
    directory: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[Path, tuple[int, int]]:
    """Resolve and attest one direct derived-cache root without opening media."""

    raw_directory = Path(directory).expanduser()
    try:
        if raw_directory.is_symlink():
            raise PersistentVisualFeatureCacheStorageUnavailableError(
                "persistent visual feature cache storage is unavailable"
            )
        resolved_directory = raw_directory.resolve(strict=True)
        root_stat = resolved_directory.lstat()
    except PersistentVisualFeatureCacheStorageUnavailableError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise PersistentVisualFeatureCacheStorageUnavailableError(
            "persistent visual feature cache storage is unavailable"
        ) from None
    identity = (root_stat.st_dev, root_stat.st_ino)
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or (
            expected_identity is not None
            and identity != expected_identity
        )
    ):
        raise PersistentVisualFeatureCacheStorageUnavailableError(
            "persistent visual feature cache storage identity changed"
        )
    return resolved_directory, identity


@dataclass(frozen=True, slots=True)
class _DecodeEquivalentAnalyzerMigration:
    """One audited, one-way analyzer migration with exact code identities."""

    legacy_analyzer_digest: str
    descriptor_component_name: str
    legacy_descriptor_digest: str
    current_descriptor_digest: str

    def __post_init__(self) -> None:
        for digest in (
            self.legacy_analyzer_digest,
            self.legacy_descriptor_digest,
            self.current_descriptor_digest,
        ):
            if _SHA256.fullmatch(digest) is None:
                raise ValueError("analyzer migration digest is invalid")
        if not self.descriptor_component_name.strip():
            raise ValueError("analyzer migration component name is blank")
        if self.legacy_descriptor_digest == self.current_descriptor_digest:
            raise ValueError("analyzer migration must change descriptor code")


@dataclass(frozen=True, slots=True)
class _DecodeEquivalentAnalyzerMigrationPath:
    """One legacy generation and every audited edge to the active analyzer."""

    legacy_analyzer_digest: str
    edges: tuple[tuple[str, _DecodeEquivalentAnalyzerMigration], ...]


# Alias-level AST v2, direct-derived-cache sampler plumbing, and lazy SFace
# eligibility/midpoint removal change only dependency code outside persisted
# descriptor numerics.  Descriptor computation and every decoded numeric
# feature are unchanged.  Every permitted forward edge is enumerated with both
# complete analyzer identities below.  Reverse directions and every unknown
# intermediate generation remain incompatible.
_DECODE_EQUIVALENT_ANALYZER_MIGRATIONS: Mapping[
    str,
    tuple[_DecodeEquivalentAnalyzerMigration, ...],
] = {
    # Only the 480px process-local pixel LRU grew from 32 to 128 frames.
    # Source clocks, decoded pixels, feature computation and model/runtime
    # bindings are unchanged; real UHD repeated-window pixels match exactly.
    "9e5522429b560ee3494f932dc0fa24ec9381f54349e9c6ddd37771fec501589b": (
        _DecodeEquivalentAnalyzerMigration(
            legacy_analyzer_digest=(
                "5a8c720efe6d42869df46685bb2aec9650308a1c42b455e0c1101349e072b048"
            ),
            descriptor_component_name="visual_score_descriptor_ast_v1",
            legacy_descriptor_digest=(
                "d19d38dd50805b619e036a4faeeeb44e0190c2543ed09565c79bb684913011a3"
            ),
            current_descriptor_digest=(
                "619a7e5f5ca525a2f95833e8bf648af3a9012474dc44cef34d502fd5af532b82"
            ),
        ),
    ),
    # H264 locality has identical decoded pixels on all 1,728 actual XOXZ
    # requests, plus sequential/B-frame/open-GOP regressions. Only this exact
    # reader dependency changed; model/schema/runtime mismatches still fail.
    "0d665ac9a9b9ef183e0c666196d63cfd84c77f35fbe3d4c59fac1350c4fa1e8d": (
        _DecodeEquivalentAnalyzerMigration(
            legacy_analyzer_digest=(
                "5e230050f8b6b83a4f208475ebbe0931e8078845cb830eaebd1333bd7f9b9509"
            ),
            descriptor_component_name="visual_score_descriptor_ast_v1",
            legacy_descriptor_digest=(
                "9f3590797903f8bbb52fd57d992b7d7f93b5992081acb4b2c659c3948dee8834"
            ),
            current_descriptor_digest=(
                "14ec585b0d692c05498594a529632dca6c89be35fcba77ac9b67999c64da1d5f"
            ),
        ),
    ),
    # Verified AV1 decoder locality: 170 real frame payloads and real full
    # descriptors are byte-identical. The endpoint includes the entire reader
    # dependency hash; unknown reader/model/runtime changes never migrate.
    "5e230050f8b6b83a4f208475ebbe0931e8078845cb830eaebd1333bd7f9b9509": (
        _DecodeEquivalentAnalyzerMigration(
            legacy_analyzer_digest=(
                "14664ec334ef15c192a496aae269c927d0e11289f882e94217e0632a31a0afde"
            ),
            descriptor_component_name="visual_score_descriptor_ast_v1",
            legacy_descriptor_digest=(
                "3cdd076cf6c0f001f6d6de4d7228d85d259248e1bddb687c7173690f5bab7d39"
            ),
            current_descriptor_digest=(
                "9f3590797903f8bbb52fd57d992b7d7f93b5992081acb4b2c659c3948dee8834"
            ),
        ),
    ),
    # Exact-pixel request coalescing leaves describe_frame and its numeric
    # output unchanged (96 real half-frame requests compared byte-for-byte).
    # Do not bridge the earlier framing/schema changes: only this predecessor.
    "14664ec334ef15c192a496aae269c927d0e11289f882e94217e0632a31a0afde": (
        _DecodeEquivalentAnalyzerMigration(
            legacy_analyzer_digest=(
                "0bb5d3e1ae287ec7b59697727580b219dae5b857f6a0a25d8a55db5d59ec9921"
            ),
            descriptor_component_name="visual_score_descriptor_ast_v1",
            legacy_descriptor_digest=(
                "fe78347f24e895c2cc0ce57e7490706c46b9dd9ae08a3c87871a120838121ebd"
            ),
            current_descriptor_digest=(
                "3cdd076cf6c0f001f6d6de4d7228d85d259248e1bddb687c7173690f5bab7d39"
            ),
        ),
    ),
    "072871c33ea2610cd30f4210b7cc80a35e418ca7c563a4c10ce39ca2e5b52173": (
        _DecodeEquivalentAnalyzerMigration(
            legacy_analyzer_digest=(
                "0e2da6dae7879dea7d529dc36b87f0f7d87184a07c3e06f4bec15b752dc5ca41"
            ),
            descriptor_component_name="visual_score_descriptor_ast_v1",
            legacy_descriptor_digest=(
                "a2586f534c0ed3a8d7a018cec2a9cf0237598b9b4e51a3c0b84404d6cac0cdec"
            ),
            current_descriptor_digest=(
                "512bf2d767f5ca035f986f0e4e7b4693bb93e5aff828308537e0d8d795204c1e"
            ),
        ),
    ),
    "0e2da6dae7879dea7d529dc36b87f0f7d87184a07c3e06f4bec15b752dc5ca41": (
        _DecodeEquivalentAnalyzerMigration(
            legacy_analyzer_digest=(
                "eb2ead7937ad96cd7225e536b1527075258de9014be52c82ba8355dfa9e81b33"
            ),
            descriptor_component_name="visual_score_descriptor_ast_v1",
            legacy_descriptor_digest=(
                "d62354e350187d5e090d103858d15b3fcc95191703ab048178ca428ca7a6d186"
            ),
            current_descriptor_digest=(
                "a2586f534c0ed3a8d7a018cec2a9cf0237598b9b4e51a3c0b84404d6cac0cdec"
            ),
        ),
    ),
    "14fff1f9ee74dbaed4023c6fbbbd4cd94b67e078a94f3d272885669cfe0e6b30": (
        _DecodeEquivalentAnalyzerMigration(
            legacy_analyzer_digest=(
                "648cfb9801b2de54d4325a7c838e31bbadfbeaa8d71a68c37163d8c41f27b19c"
            ),
            descriptor_component_name="visual_score_descriptor_ast_v1",
            legacy_descriptor_digest=(
                "245edba6559b94d4855f676c3edc957994137fc42d2d2da0e3e0eb9f752a61c0"
            ),
            current_descriptor_digest=(
                "c0de5b604e22d12c713b02edfe2b5a0de8d19a80475b1547b861f975d8978d89"
            ),
        ),
    ),
    "753e2dbcee35d3bf8c72ed73005341b8b4ec3c142860e869904bc8cf5acac927": (
        _DecodeEquivalentAnalyzerMigration(
            legacy_analyzer_digest=(
                "14fff1f9ee74dbaed4023c6fbbbd4cd94b67e078a94f3d272885669cfe0e6b30"
            ),
            descriptor_component_name="visual_score_descriptor_ast_v1",
            legacy_descriptor_digest=(
                "c0de5b604e22d12c713b02edfe2b5a0de8d19a80475b1547b861f975d8978d89"
            ),
            current_descriptor_digest=(
                "909c66c8dbd903ca6ad1b1cc75c63710838fba88179b2b308f7c964d55f82f84"
            ),
        ),
    ),
    "eb2ead7937ad96cd7225e536b1527075258de9014be52c82ba8355dfa9e81b33": (
        _DecodeEquivalentAnalyzerMigration(
            legacy_analyzer_digest=(
                "753e2dbcee35d3bf8c72ed73005341b8b4ec3c142860e869904bc8cf5acac927"
            ),
            descriptor_component_name="visual_score_descriptor_ast_v1",
            legacy_descriptor_digest=(
                "909c66c8dbd903ca6ad1b1cc75c63710838fba88179b2b308f7c964d55f82f84"
            ),
            current_descriptor_digest=(
                "d62354e350187d5e090d103858d15b3fcc95191703ab048178ca428ca7a6d186"
            ),
        ),
    ),
}


def _decode_equivalent_analyzer_migration_closure(
    current_analyzer_digest: str,
) -> tuple[_DecodeEquivalentAnalyzerMigrationPath, ...]:
    """Resolve audited predecessors breadth-first and reject ambiguous graphs."""

    if _SHA256.fullmatch(current_analyzer_digest) is None:
        raise ValueError("current analyzer migration digest is invalid")
    # Queue entries retain edges oldest-to-newest even though the graph is
    # traversed backwards. Breadth-first discovery gives lookup precedence to
    # the current root, then the nearest legacy generation, then older roots.
    queue: list[
        tuple[
            str,
            tuple[tuple[str, _DecodeEquivalentAnalyzerMigration], ...],
            tuple[str, ...],
        ]
    ] = [(current_analyzer_digest, (), (current_analyzer_digest,))]
    queue_index = 0
    seen = {current_analyzer_digest}
    resolved: list[_DecodeEquivalentAnalyzerMigrationPath] = []
    while queue_index < len(queue):
        endpoint_digest, newer_edges, ancestry = queue[queue_index]
        queue_index += 1
        raw_edges = _DECODE_EQUIVALENT_ANALYZER_MIGRATIONS.get(
            endpoint_digest,
            (),
        )
        if not isinstance(raw_edges, tuple):
            raise ValueError("analyzer migration edges must be a tuple")
        local_legacy_digests: set[str] = set()
        for edge in raw_edges:
            if not isinstance(edge, _DecodeEquivalentAnalyzerMigration):
                raise ValueError("analyzer migration edge type is invalid")
            legacy_digest = edge.legacy_analyzer_digest
            if legacy_digest in ancestry:
                raise ValueError("legacy analyzer migration graph contains a cycle")
            if legacy_digest in local_legacy_digests or legacy_digest in seen:
                raise ValueError(
                    "legacy analyzer migration graph is duplicate or ambiguous"
                )
            local_legacy_digests.add(legacy_digest)
            path_edges = ((endpoint_digest, edge),) + newer_edges
            for (older_endpoint, older), (_newer_endpoint, newer) in zip(
                path_edges,
                path_edges[1:],
            ):
                if (
                    older_endpoint != newer.legacy_analyzer_digest
                    or older.descriptor_component_name
                    != newer.descriptor_component_name
                    or older.current_descriptor_digest
                    != newer.legacy_descriptor_digest
                ):
                    raise ValueError(
                        "legacy analyzer migration chain endpoints do not match"
                    )
            seen.add(legacy_digest)
            resolved.append(
                _DecodeEquivalentAnalyzerMigrationPath(
                    legacy_analyzer_digest=legacy_digest,
                    edges=path_edges,
                )
            )
            queue.append(
                (
                    legacy_digest,
                    path_edges,
                    ancestry + (legacy_digest,),
                )
            )
    return tuple(resolved)


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


def _analyzer_components_sha256(components: Mapping[str, Any]) -> str | None:
    """Return the canonical analyzer digest, or None for invalid components."""

    try:
        encoded = json.dumps(
            dict(components),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


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
    digest = _analyzer_components_sha256(components)
    assert digest is not None
    return AnalyzerFingerprint(
        digest=digest,
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
                # Bind an imported dependency to the one alias actually loaded
                # by the analyzer.  Hashing the complete import node made an
                # unrelated selector alias invalidate every persisted frame
                # descriptor merely because both aliases shared one statement.
                # Preserve the module and relative level for ImportFrom, and
                # preserve the imported name/asname for both forms, so every
                # semantic change to the referenced alias still invalidates.
                canonical_alias = ast.alias(
                    name=alias.name,
                    asname=alias.asname,
                )
                if isinstance(node, ast.ImportFrom):
                    definitions[local_name] = ast.ImportFrom(
                        module=node.module,
                        names=[canonical_alias],
                        level=node.level,
                    )
                else:
                    definitions[local_name] = ast.Import(
                        names=[canonical_alias],
                    )
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
            "contract": PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT,
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
        derived_cache_directory: Path | None = None,
    ) -> None:
        if derived_cache_directory is None:
            cache_root = Path(source_cache_directory).expanduser()
            if cache_root.is_symlink():
                raise ValueError("source cache directory cannot be a symbolic link")
            cache_root.mkdir(parents=True, exist_ok=True)
            if not cache_root.is_dir():
                raise ValueError("source cache directory must be a directory")
            derived_root = self._safe_directory(cache_root, "derived")
        else:
            direct_derived_root = Path(derived_cache_directory).expanduser()
            if direct_derived_root.is_symlink():
                raise ValueError(
                    "visual feature derived cache directory cannot be a "
                    "symbolic link"
                )
            try:
                derived_root = direct_derived_root.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError(
                    "visual feature derived cache directory must exist"
                ) from exc
            if not derived_root.is_dir():
                raise ValueError(
                    "visual feature derived cache directory must be a directory"
                )
        self.source_path = Path(source_path).expanduser().resolve(strict=True)
        self.source_sha256, self._source_stat_identity = _stable_sha256(
            self.source_path
        )
        self.analyzer = analyzer
        source_root = self._safe_directory(derived_root, "visual-v1")
        source_root = self._safe_directory(
            source_root,
            self.source_sha256[:2],
        )
        source_root = self._safe_directory(source_root, self.source_sha256)
        self._root = self._safe_directory(source_root, analyzer.digest)
        self._compatible_legacy_analyzer_migration_paths = (
            _decode_equivalent_analyzer_migration_closure(analyzer.digest)
        )
        self._compatible_legacy_analyzer_digests = tuple(
            path.legacy_analyzer_digest
            for path in self._compatible_legacy_analyzer_migration_paths
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
        try:
            (
                self._storage_root,
                self._storage_root_identity,
            ) = assert_visual_feature_derived_cache_root_available(
                derived_root,
            )
        except PersistentVisualFeatureCacheStorageUnavailableError:
            # Preserve the constructor's configuration-error surface so the
            # sampler can consistently translate required versus optional
            # initialization races.
            raise ValueError(
                "visual feature derived cache directory became unavailable"
            ) from None
        self._storage_io_failure_count = 0
        self._atomic_write_failure_count = 0

    @staticmethod
    def _safe_directory(parent: Path, name: str) -> Path:
        child = parent / name
        if child.is_symlink():
            raise ValueError(f"visual feature cache path is a symlink: {child}")
        child.mkdir(exist_ok=True)
        if not child.is_dir():
            raise ValueError(f"visual feature cache path is not a directory: {child}")
        return child

    def assert_storage_root_available(self) -> None:
        """Fail distinctly if the initialized cache mount/path is no longer live."""

        try:
            assert_visual_feature_derived_cache_root_available(
                self._storage_root,
                expected_identity=self._storage_root_identity,
            )
        except PersistentVisualFeatureCacheStorageUnavailableError:
            self._storage_io_failure_count += 1
            raise

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
        migration_path = next(
            (
                candidate
                for candidate in (
                    self._compatible_legacy_analyzer_migration_paths
                )
                if candidate.legacy_analyzer_digest == legacy_digest
            ),
            None,
        )
        if migration_path is None or not migration_path.edges:
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
        if set(actual_code) != set(expected_code):
            return False
        descriptor_names = {
            edge.descriptor_component_name
            for _endpoint_digest, edge in migration_path.edges
        }
        if len(descriptor_names) != 1:
            return False
        descriptor_name = next(iter(descriptor_names))
        if descriptor_name not in actual_code:
            return False
        if any(
            actual_code.get(name) != expected_code.get(name)
            for name in expected_code
            if name != descriptor_name
        ):
            return False
        # Replay the complete audited chain. Each intermediate component object
        # is reconstructed by changing only the allowlisted descriptor digest,
        # then hashed against that edge's canonical full analyzer endpoint.
        # The graph is therefore routing only; it cannot manufacture an
        # endpoint or bridge any schema/model/runtime/other-code difference.
        working_components = dict(actual_components)
        working_code = dict(actual_code)
        working_components["code_sha256"] = working_code
        working_analyzer_digest = legacy_digest
        for endpoint_digest, edge in migration_path.edges:
            if (
                working_analyzer_digest != edge.legacy_analyzer_digest
                or working_code.get(descriptor_name)
                != edge.legacy_descriptor_digest
                or _analyzer_components_sha256(working_components)
                != edge.legacy_analyzer_digest
            ):
                return False
            working_code[descriptor_name] = edge.current_descriptor_digest
            if (
                _analyzer_components_sha256(working_components)
                != endpoint_digest
            ):
                return False
            working_analyzer_digest = endpoint_digest
        return bool(
            working_analyzer_digest == self.analyzer.digest
            and working_components == expected_components
            and _analyzer_components_sha256(expected_components)
            == self.analyzer.digest
        )

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
        shards = sorted(
            lookup_root.iterdir(),
            key=lambda item: item.name,
        )
        for shard in shards:
            if (
                shard.is_symlink()
                or not shard.is_dir()
                or re.fullmatch(r"[0-9a-f]{2}", shard.name) is None
            ):
                continue
            shard_records = sorted(
                shard.iterdir(),
                key=lambda item: item.name,
            )
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

        self.assert_storage_root_available()
        if not self._source_is_unchanged():
            self._miss_count += 1
            return None
        try:
            locator_index_ready = self._ensure_locator_index()
        except (OSError, ValueError):
            self._storage_io_failure_count += 1
            raise PersistentVisualFeatureCacheStorageUnavailableError(
                "persistent visual feature cache index is unavailable"
            ) from None
        self.assert_storage_root_available()
        if not locator_index_ready:
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
            self.assert_storage_root_available()
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
        self.assert_storage_root_available()
        if not self._source_is_unchanged():
            self._miss_count += 1
            return None
        try:
            path = self._record_path(request)
        except (OSError, ValueError):
            self._storage_io_failure_count += 1
            raise PersistentVisualFeatureCacheStorageUnavailableError(
                "persistent visual feature cache path is unavailable"
            ) from None
        if path.is_symlink() or not path.is_file():
            self.assert_storage_root_available()
            self._miss_count += 1
            return None
        try:
            _actual_request, payload = self._read_record(
                path,
                expected_request=request,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            self.assert_storage_root_available()
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
        self.assert_storage_root_available()
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
        try:
            path = self._record_path(request)
            temporary = path.with_name(
                f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
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
        except (OSError, ValueError):
            # A partial temporary is deliberately ignored by readers.  Do not
            # delete it here: interruption evidence remains recoverable and no
            # cleanup operation is needed for correctness.
            self._atomic_write_failure_count += 1
            raise PersistentVisualFeatureCacheWriteError(
                "persistent visual feature cache atomic write failed"
            ) from None
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
            "storage_io_failure_count": self._storage_io_failure_count,
            "atomic_write_failure_count": self._atomic_write_failure_count,
            "raw_frames_persisted": False,
            "face_identity_embeddings_persisted": False,
        }


__all__ = [
    "AnalyzerFingerprint",
    "CanonicalFrameIdentity",
    "DescriptorRequestIdentity",
    "DescriptorRequestLocator",
    "PersistentVisualFeatureCacheIOError",
    "PersistentVisualFeatureCache",
    "PersistentVisualFeatureCacheStorageUnavailableError",
    "PersistentVisualFeatureCacheWriteError",
    "PYTHON_SYMBOL_DEPENDENCY_AST_CONTRACT",
    "VISUAL_FEATURE_CACHE_CONTRACT",
    "VISUAL_FEATURE_SCHEMA_VERSION",
    "build_analyzer_fingerprint",
    "assert_visual_feature_derived_cache_root_available",
    "python_symbol_dependency_sha256",
]
