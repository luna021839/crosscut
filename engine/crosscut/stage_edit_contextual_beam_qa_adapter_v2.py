"""Verified Beam-QA adapter for the contextual source-pair scorer v2.

This module is intentionally a leaf.  It may import the semi-Markov decoder,
post-render QA, and the contextual production DTOs.  Those modules must not
top-level import this adapter; the production activation boundary may perform a
function-local import only after its DTO module has initialized.  That keeps the
beam decoder free to consume the contextual runtime core without an import
cycle.

The production DTO in ``stage_edit_source_pair_contextual_v2_production`` is a
serializable inspection record.  Its constructor is deliberately *not* an
activation authority.  Only this adapter can attach the process-local seal in
``VerifiedStageEditContextualBeamQACaseV2`` after independently reopening and
revalidating the bound decoder and post-render artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from statistics import median
from types import SimpleNamespace
from typing import Any, Mapping

from .post_render_qa import (
    INPUT_ARTIFACT_FINGERPRINT_CONTRACT,
    PostRenderArtifactSet,
    PostRenderQAConfig,
    PostRenderQAReport,
    evaluate_post_render_qa,
    fingerprint_post_render_artifacts,
)
from .post_render_perceptual_quality import PerceptualQualityConfig
from .rendered_montage_grammar_review import RenderedMontageGrammarConfig
from .stage_edit_continuity_authority import (
    CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
)
from .stage_edit_semimarkov_beam import (
    DECODER_CONTRACT,
    DECODER_SCHEMA_VERSION,
    StageEditBeamTimeline,
    StageEditSemiMarkovResult,
)
from .stage_edit_source_pair_contextual_v2 import (
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME,
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT,
)
from .stage_edit_source_pair_contextual_v2_production import (
    StageEditContextualBeamQACaseV2,
    StageEditContextualSourcePairEvaluationBundleV2,
    load_stage_edit_contextual_source_pair_evaluation_bundle_v2,
    require_stage_edit_contextual_source_pair_evaluation_bundle_v2,
)


STAGE_EDIT_CONTEXTUAL_BEAM_QA_ADAPTER_V2_CONTRACT = (
    "typed_semimarkov_post_render_reverified_beam_qa_adapter_v2"
)
STAGE_EDIT_CONTEXTUAL_BEAM_QA_ARTIFACT_IDENTITY_V2_CONTRACT = (
    "stable_regular_json_file_byte_and_canonical_content_sha256_v1"
)
STAGE_EDIT_CONTEXTUAL_BEAM_QA_POST_RENDER_METRICS_V2_CONTRACT = (
    "post_render_bound_contextual_beam_qa_metrics_v2"
)
STAGE_EDIT_CONTEXTUAL_BEAM_QA_SOURCE_DIVERSITY_V2_CONTRACT = (
    "exp_shannon_effective_derivative_family_duration_v1"
)
STAGE_EDIT_CONTEXTUAL_BEAM_QA_MOTION_EVIDENCE_V2_CONTRACT = (
    "typed_temporal_pose_motion_plus_continuity_bound_render_proxy_v1"
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CLOSEUP_SHOT_SIZES = frozenset({"closeup", "extreme_closeup"})
_VERIFICATION_SEAL = object()
_EPSILON = 1e-9


class StageEditContextualBeamQAAdapterV2Error(RuntimeError):
    """A Beam-QA case could not be derived from verified production evidence."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            "Beam-QA evidence is not canonical JSON"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _strict_json_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageEditContextualBeamQAAdapterV2Error(f"{name} must be an object")
    return value


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON number: {token}")


def _normalized_post_render_revalidation_payload(
    value: Any,
    *,
    final_video_name: str,
) -> Any:
    """Remove only the evaluator snapshot directory from video path evidence."""

    if isinstance(value, Mapping):
        return {
            key: _normalized_post_render_revalidation_payload(
                item,
                final_video_name=final_video_name,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalized_post_render_revalidation_payload(
                item,
                final_video_name=final_video_name,
            )
            for item in value
        ]
    if (
        isinstance(value, str)
        and os.path.isabs(value)
        and Path(value).name == final_video_name
    ):
        return f"<post-render-video:{final_video_name}>"
    return value


def _required_mapping(root: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    return _strict_json_object(root.get(key), f"{path}.{key}")


def _required_bool(root: Mapping[str, Any], key: str, path: str) -> bool:
    value = root.get(key)
    if type(value) is not bool:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{path}.{key} must be a boolean"
        )
    return value


def _required_int(
    root: Mapping[str, Any], key: str, path: str, *, minimum: int = 0
) -> int:
    value = root.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{path}.{key} must be an integer >= {minimum}"
        )
    return value


def _required_float(
    root: Mapping[str, Any], key: str, path: str, *, minimum: float = 0.0
) -> float:
    value = root.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{path}.{key} must be numeric"
        )
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{path}.{key} must be finite and >= {minimum}"
        )
    return number


def _required_index_tuple(
    root: Mapping[str, Any], key: str, path: str
) -> tuple[int, ...]:
    value = root.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{path}.{key} must be a list of non-negative integer indices"
        )
    result = tuple(value)
    if len(result) != len(set(result)) or result != tuple(sorted(result)):
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{path}.{key} must be sorted and unique"
        )
    return result


def _strict_absolute_path(value: Path, name: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} must be a filesystem path"
        ) from exc
    if not isinstance(raw, str) or not raw or not os.path.isabs(raw):
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} must be an absolute path"
        )
    normalized = os.path.normpath(raw)
    if normalized != raw or any(part in {".", ".."} for part in Path(raw).parts):
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} must use one normalized lexical spelling"
        )
    return Path(normalized)


def _open_pinned_directory_fd(value: Path, name: str) -> tuple[Path, int]:
    """Walk an absolute directory from root using pinned no-follow fds."""

    directory = _strict_absolute_path(value, name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        current_fd = os.open(directory.anchor, flags)
    except OSError as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} root cannot be opened safely: {exc}"
        ) from exc
    try:
        for component in directory.parts[1:]:
            try:
                before = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise StageEditContextualBeamQAAdapterV2Error(
                    f"{name} cannot be inspected safely: {component}: {exc}"
                ) from exc
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise StageEditContextualBeamQAAdapterV2Error(
                    f"{name} contains a symlink or non-directory component: "
                    f"{component}"
                )
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise StageEditContextualBeamQAAdapterV2Error(
                    f"{name} component cannot be opened safely: {component}: {exc}"
                ) from exc
            opened = os.fstat(next_fd)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                os.close(next_fd)
                raise StageEditContextualBeamQAAdapterV2Error(
                    f"{name} component identity changed while opening: {component}"
                )
            os.close(current_fd)
            current_fd = next_fd
        return directory, current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _stable_file_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _require_regular_single_link(status: os.stat_result, name: str) -> None:
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} must be a regular single-link non-symlink file"
        )


@dataclass(frozen=True, slots=True)
class StageEditContextualBeamQAArtifactIdentityV2:
    path: str
    size_bytes: int
    file_sha256: str
    canonical_content_sha256: str
    contract: str = STAGE_EDIT_CONTEXTUAL_BEAM_QA_ARTIFACT_IDENTITY_V2_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != STAGE_EDIT_CONTEXTUAL_BEAM_QA_ARTIFACT_IDENTITY_V2_CONTRACT:
            raise StageEditContextualBeamQAAdapterV2Error(
                "Beam-QA artifact identity contract mismatch"
            )
        if not os.path.isabs(self.path):
            raise StageEditContextualBeamQAAdapterV2Error(
                "Beam-QA artifact identity path must be absolute"
            )
        if isinstance(self.size_bytes, bool) or self.size_bytes < 1:
            raise StageEditContextualBeamQAAdapterV2Error(
                "Beam-QA artifact identity size must be positive"
            )
        if not _valid_sha256(self.file_sha256) or not _valid_sha256(
            self.canonical_content_sha256
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "Beam-QA artifact identity SHA-256 is invalid"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "file_sha256": self.file_sha256,
            "canonical_content_sha256": self.canonical_content_sha256,
        }


def _read_stable_json_artifact(
    value: Path, name: str
) -> tuple[Mapping[str, Any], StageEditContextualBeamQAArtifactIdentityV2]:
    path = _strict_absolute_path(Path(value), name)
    _parent, parent_fd = _open_pinned_directory_fd(path.parent, f"{name} parent")
    digest = hashlib.sha256()
    byte_count = 0
    chunks: list[bytes] = []
    try:
        lexical_before = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        _require_regular_single_link(lexical_before, name)
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor_before = os.fstat(stream.fileno())
            _require_regular_single_link(descriptor_before, name)
            if (
                descriptor_before.st_dev != lexical_before.st_dev
                or descriptor_before.st_ino != lexical_before.st_ino
            ):
                raise StageEditContextualBeamQAAdapterV2Error(
                    f"{name} changed before it was opened"
                )
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                chunks.append(chunk)
                byte_count += len(chunk)
            descriptor_after = os.fstat(stream.fileno())
            _require_regular_single_link(descriptor_after, name)
        lexical_after = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        _require_regular_single_link(lexical_after, name)
    except StageEditContextualBeamQAAdapterV2Error:
        raise
    except OSError as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} could not be read: {exc}"
        ) from exc
    finally:
        os.close(parent_fd)
    if (
        _stable_file_identity(lexical_before)
        != _stable_file_identity(descriptor_before)
        or _stable_file_identity(descriptor_before)
        != _stable_file_identity(descriptor_after)
        or _stable_file_identity(descriptor_after)
        != _stable_file_identity(lexical_after)
        or byte_count != descriptor_after.st_size
        or byte_count < 1
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} changed while it was read"
        )
    try:
        payload = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} is not strict UTF-8 JSON"
        ) from exc
    root = _strict_json_object(payload, name)
    identity = StageEditContextualBeamQAArtifactIdentityV2(
        path=str(path),
        size_bytes=byte_count,
        file_sha256=digest.hexdigest(),
        canonical_content_sha256=_canonical_sha256(root),
    )
    return root, identity


def _fingerprint_stable_input_artifact(
    value: Path,
    name: str,
) -> dict[str, Any]:
    """Hash one non-JSON input through a pinned parent directory descriptor."""

    path = _strict_absolute_path(Path(value), name)
    _parent, parent_fd = _open_pinned_directory_fd(path.parent, f"{name} parent")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        lexical_before = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        _require_regular_single_link(lexical_before, name)
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor_before = os.fstat(stream.fileno())
            _require_regular_single_link(descriptor_before, name)
            if (
                descriptor_before.st_dev != lexical_before.st_dev
                or descriptor_before.st_ino != lexical_before.st_ino
            ):
                raise StageEditContextualBeamQAAdapterV2Error(
                    f"{name} changed before it was opened"
                )
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            descriptor_after = os.fstat(stream.fileno())
            _require_regular_single_link(descriptor_after, name)
        lexical_after = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        _require_regular_single_link(lexical_after, name)
    except StageEditContextualBeamQAAdapterV2Error:
        raise
    except OSError as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} could not be fingerprinted: {exc}"
        ) from exc
    finally:
        os.close(parent_fd)
    if (
        _stable_file_identity(lexical_before)
        != _stable_file_identity(descriptor_before)
        or _stable_file_identity(descriptor_before)
        != _stable_file_identity(descriptor_after)
        or _stable_file_identity(descriptor_after)
        != _stable_file_identity(lexical_after)
        or byte_count != descriptor_after.st_size
        or byte_count < 1
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} changed while it was fingerprinted"
        )
    return {
        "path": str(path),
        "size_bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _artifact_input_fingerprint(
    identity: StageEditContextualBeamQAArtifactIdentityV2,
) -> dict[str, Any]:
    return {
        "path": identity.path,
        "size_bytes": identity.size_bytes,
        "sha256": identity.file_sha256,
    }


@dataclass(frozen=True, slots=True)
class StageEditContextualBeamQADerivationV2:
    request_container: StageEditContextualBeamQAArtifactIdentityV2
    request_content_sha256: str
    decode_report: StageEditContextualBeamQAArtifactIdentityV2
    decode_result_content_sha256: str
    decoder_input_sha256: str
    post_render_report: StageEditContextualBeamQAArtifactIdentityV2
    render_plan: StageEditContextualBeamQAArtifactIdentityV2
    selection_report: StageEditContextualBeamQAArtifactIdentityV2
    post_render_input_fingerprints_sha256: str
    selected_timeline_rank: int
    selected_timeline_content_sha256: str
    metric_binding_sha256: str
    contextual_checkpoint: StageEditContextualBeamQAArtifactIdentityV2 | None = None
    contextual_corpus: StageEditContextualBeamQAArtifactIdentityV2 | None = None
    contextual_checkpoint_sha256: str | None = None
    contextual_corpus_sha256: str | None = None
    decode_continuity_authority: str | None = None
    decode_scorer_fingerprint: str | None = None
    contextual_runtime_equivalence_sha256: str | None = None
    contextual_token_schema_fingerprint: str | None = None
    contract: str = STAGE_EDIT_CONTEXTUAL_BEAM_QA_ADAPTER_V2_CONTRACT
    derivation_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.contract != STAGE_EDIT_CONTEXTUAL_BEAM_QA_ADAPTER_V2_CONTRACT:
            raise StageEditContextualBeamQAAdapterV2Error(
                "Beam-QA derivation contract mismatch"
            )
        for name in (
            "request_container",
            "decode_report",
            "post_render_report",
            "render_plan",
            "selection_report",
        ):
            if not isinstance(
                getattr(self, name),
                StageEditContextualBeamQAArtifactIdentityV2,
            ):
                raise TypeError(f"{name} must be a typed artifact identity")
        for name in (
            "request_content_sha256",
            "decode_result_content_sha256",
            "decoder_input_sha256",
            "post_render_input_fingerprints_sha256",
            "selected_timeline_content_sha256",
            "metric_binding_sha256",
        ):
            if not _valid_sha256(getattr(self, name)):
                raise StageEditContextualBeamQAAdapterV2Error(
                    f"{name} must be lowercase SHA-256"
                )
        checkpoint_binding = (
            self.contextual_checkpoint,
            self.contextual_corpus,
            self.contextual_checkpoint_sha256,
            self.contextual_corpus_sha256,
            self.decode_continuity_authority,
            self.decode_scorer_fingerprint,
            self.contextual_runtime_equivalence_sha256,
            self.contextual_token_schema_fingerprint,
        )
        if any(value is not None for value in checkpoint_binding):
            if any(value is None for value in checkpoint_binding):
                raise StageEditContextualBeamQAAdapterV2Error(
                    "Beam-QA contextual checkpoint binding is partial"
                )
            if not isinstance(
                self.contextual_checkpoint,
                StageEditContextualBeamQAArtifactIdentityV2,
            ) or not isinstance(
                self.contextual_corpus,
                StageEditContextualBeamQAArtifactIdentityV2,
            ):
                raise StageEditContextualBeamQAAdapterV2Error(
                    "Beam-QA contextual checkpoint binding lacks typed artifacts"
                )
            if self.decode_continuity_authority != (
                CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
            ):
                raise StageEditContextualBeamQAAdapterV2Error(
                    "Beam-QA checkpoint binding requires evaluation-only authority"
                )
            for name in (
                "contextual_checkpoint_sha256",
                "contextual_corpus_sha256",
                "decode_scorer_fingerprint",
                "contextual_runtime_equivalence_sha256",
                "contextual_token_schema_fingerprint",
            ):
                if not _valid_sha256(getattr(self, name)):
                    raise StageEditContextualBeamQAAdapterV2Error(
                        f"{name} must be lowercase SHA-256"
                    )
        if (
            isinstance(self.selected_timeline_rank, bool)
            or self.selected_timeline_rank < 1
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "selected timeline rank must be positive"
            )
        expected = _canonical_sha256(self.unsigned_dict())
        if self.derivation_sha256 is None:
            object.__setattr__(self, "derivation_sha256", expected)
        elif self.derivation_sha256 != expected:
            raise StageEditContextualBeamQAAdapterV2Error(
                "Beam-QA derivation digest mismatch"
            )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "request_container": self.request_container.to_dict(),
            "request_content_sha256": self.request_content_sha256,
            "decode_report": self.decode_report.to_dict(),
            "decode_result_content_sha256": self.decode_result_content_sha256,
            "decoder_input_sha256": self.decoder_input_sha256,
            "post_render_report": self.post_render_report.to_dict(),
            "render_plan": self.render_plan.to_dict(),
            "selection_report": self.selection_report.to_dict(),
            "post_render_input_fingerprints_sha256": (
                self.post_render_input_fingerprints_sha256
            ),
            "selected_timeline_rank": self.selected_timeline_rank,
            "selected_timeline_content_sha256": (
                self.selected_timeline_content_sha256
            ),
            "metric_binding_sha256": self.metric_binding_sha256,
            "contextual_checkpoint": (
                self.contextual_checkpoint.to_dict()
                if self.contextual_checkpoint is not None
                else None
            ),
            "contextual_corpus": (
                self.contextual_corpus.to_dict()
                if self.contextual_corpus is not None
                else None
            ),
            "contextual_checkpoint_sha256": self.contextual_checkpoint_sha256,
            "contextual_corpus_sha256": self.contextual_corpus_sha256,
            "decode_continuity_authority": self.decode_continuity_authority,
            "decode_scorer_fingerprint": self.decode_scorer_fingerprint,
            "contextual_runtime_equivalence_sha256": (
                self.contextual_runtime_equivalence_sha256
            ),
            "contextual_token_schema_fingerprint": (
                self.contextual_token_schema_fingerprint
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        assert self.derivation_sha256 is not None
        return {**self.unsigned_dict(), "derivation_sha256": self.derivation_sha256}


@dataclass(frozen=True, slots=True)
class VerifiedStageEditContextualBeamQACaseV2:
    """Non-persistable activation envelope issued only by the adapter."""

    case: StageEditContextualBeamQACaseV2
    derivation: StageEditContextualBeamQADerivationV2
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _VERIFICATION_SEAL:
            raise StageEditContextualBeamQAAdapterV2Error(
                "raw or deserialized Beam-QA cases are inspection-only"
            )
        if not isinstance(self.case, StageEditContextualBeamQACaseV2):
            raise TypeError("verified Beam-QA envelope needs the typed case DTO")
        if not isinstance(self.derivation, StageEditContextualBeamQADerivationV2):
            raise TypeError("verified Beam-QA envelope needs typed derivation evidence")
        if self.case.request_sha256 != self.derivation.request_content_sha256:
            raise StageEditContextualBeamQAAdapterV2Error(
                "Beam-QA request identity differs from its derivation"
            )
        if self.case.decode_report_sha256 != self.derivation.decode_report.file_sha256:
            raise StageEditContextualBeamQAAdapterV2Error(
                "Beam-QA decode report identity differs from its derivation"
            )


def is_verified_stage_edit_contextual_beam_qa_case_v2(value: Any) -> bool:
    return bool(
        isinstance(value, VerifiedStageEditContextualBeamQACaseV2)
        and value._seal is _VERIFICATION_SEAL
    )


def require_verified_stage_edit_contextual_beam_qa_case_v2(
    value: Any,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_corpus_sha256: str | None = None,
    expected_scorer_fingerprint: str | None = None,
    expected_runtime_equivalence_sha256: str | None = None,
    expected_token_schema_fingerprint: str | None = None,
) -> StageEditContextualBeamQACaseV2:
    if not is_verified_stage_edit_contextual_beam_qa_case_v2(value):
        raise StageEditContextualBeamQAAdapterV2Error(
            "production activation requires an adapter-issued verified Beam-QA case"
        )
    reverify_stage_edit_contextual_beam_qa_case_v2(value)
    expected_binding = (
        expected_checkpoint_sha256,
        expected_corpus_sha256,
        expected_scorer_fingerprint,
        expected_runtime_equivalence_sha256,
        expected_token_schema_fingerprint,
    )
    if any(item is not None for item in expected_binding):
        if any(item is None for item in expected_binding):
            raise StageEditContextualBeamQAAdapterV2Error(
                "production Beam-QA checkpoint expectation is partial"
            )
        derivation = value.derivation
        observed = (
            derivation.contextual_checkpoint_sha256,
            derivation.contextual_corpus_sha256,
            derivation.decode_scorer_fingerprint,
            derivation.contextual_runtime_equivalence_sha256,
            derivation.contextual_token_schema_fingerprint,
        )
        if (
            derivation.decode_continuity_authority
            != CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
            or observed != expected_binding
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "verified Beam-QA decode scorer/runtime identity differs from "
                "the contextual checkpoint"
            )
    return value.case


def reverify_stage_edit_contextual_beam_qa_case_v2(
    value: VerifiedStageEditContextualBeamQACaseV2,
) -> None:
    """Reopen every persisted identity immediately before activation use."""

    if not is_verified_stage_edit_contextual_beam_qa_case_v2(value):
        raise StageEditContextualBeamQAAdapterV2Error(
            "only an adapter-issued Beam-QA envelope can be reverified"
        )
    if not isinstance(value.case, StageEditContextualBeamQACaseV2) or not isinstance(
        value.derivation,
        StageEditContextualBeamQADerivationV2,
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified Beam-QA envelope was mutated after derivation"
        )
    derivation = value.derivation
    if derivation.derivation_sha256 != _canonical_sha256(
        derivation.unsigned_dict()
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified Beam-QA derivation digest changed after derivation"
        )
    if value.case.case_sha256 != _canonical_sha256(value.case.unsigned_dict()):
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified Beam-QA case digest changed after derivation"
        )
    if (
        value.case.request_sha256 != derivation.request_content_sha256
        or value.case.decode_report_sha256 != derivation.decode_report.file_sha256
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified Beam-QA case binding changed after derivation"
        )
    request_root, request_identity = _read_stable_json_artifact(
        Path(derivation.request_container.path), "verified request container"
    )
    if request_identity != derivation.request_container:
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified request container identity changed after derivation"
        )
    request_payload = _required_mapping(request_root, "request", "analysis")
    if _canonical_sha256(request_payload) != derivation.request_content_sha256:
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified request content changed after derivation"
        )

    decode_root, decode_identity = _read_stable_json_artifact(
        Path(derivation.decode_report.path), "verified decode report"
    )
    if decode_identity != derivation.decode_report:
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified decode report identity changed after derivation"
        )
    decode_result = _semimarkov_payload_from_report(decode_root)
    if (
        _canonical_sha256(decode_result)
        != derivation.decode_result_content_sha256
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified decode result changed after derivation"
        )
    if derivation.contextual_checkpoint is not None:
        assert derivation.contextual_corpus is not None
        checkpoint_root, checkpoint_identity = _read_stable_json_artifact(
            Path(derivation.contextual_checkpoint.path),
            "verified contextual evaluation checkpoint",
        )
        corpus_root, corpus_identity = _read_stable_json_artifact(
            Path(derivation.contextual_corpus.path),
            "verified contextual evaluation corpus",
        )
        if (
            checkpoint_identity != derivation.contextual_checkpoint
            or corpus_identity != derivation.contextual_corpus
            or Path(checkpoint_identity.path).name
            != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME
            or Path(corpus_identity.path).name
            != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME
            or Path(checkpoint_identity.path).parent
            != Path(corpus_identity.path).parent
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "verified contextual evaluation artifact identity changed"
            )
        evaluation_bundle = (
            load_stage_edit_contextual_source_pair_evaluation_bundle_v2(
                Path(checkpoint_identity.path).parent,
                expected_checkpoint_sha256=(
                    derivation.contextual_checkpoint_sha256
                ),
                expected_checkpoint_file_sha256=(
                    checkpoint_identity.file_sha256
                ),
                expected_corpus_sha256=derivation.contextual_corpus_sha256,
                expected_corpus_file_sha256=corpus_identity.file_sha256,
                expected_token_schema_fingerprint=(
                    derivation.contextual_token_schema_fingerprint
                ),
                expected_scorer_fingerprint=(
                    derivation.decode_scorer_fingerprint
                ),
                expected_runtime_equivalence_sha256=(
                    derivation.contextual_runtime_equivalence_sha256
                ),
            )
        )
        authority, scorer_fingerprint = _contextual_decode_identity_v2(
            decode_result
        )
        if (
            authority != derivation.decode_continuity_authority
            or scorer_fingerprint != evaluation_bundle.scorer.fingerprint
            or checkpoint_root != evaluation_bundle.checkpoint.to_dict()
            or corpus_root != evaluation_bundle.corpus.to_dict()
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "verified contextual evaluation scorer/runtime binding changed"
            )

    report_root, report_identity = _read_stable_json_artifact(
        Path(derivation.post_render_report.path), "verified post-render report"
    )
    if report_identity != derivation.post_render_report:
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified post-render report identity changed after derivation"
        )
    artifact_payload = _required_mapping(
        report_root, "artifact_set", "post_render_report"
    )
    try:
        artifact_set = PostRenderArtifactSet(
            base_name=str(artifact_payload["base_name"]),
            timestamp=str(artifact_payload["timestamp"]),
            analysis_path=Path(str(artifact_payload["analysis_path"])),
            render_plan_path=Path(str(artifact_payload["render_plan_path"])),
            selection_path=Path(str(artifact_payload["selection_path"])),
            video_path=Path(str(artifact_payload["video_path"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified post-render artifact set is malformed"
        ) from exc
    if artifact_set.analysis_path != Path(derivation.request_container.path):
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified post-render report names another request container"
        )
    _render_plan, render_plan_identity = _read_stable_json_artifact(
        artifact_set.render_plan_path,
        "verified render plan",
    )
    if render_plan_identity != derivation.render_plan:
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified render plan identity changed after derivation"
        )
    _selection, selection_identity = _read_stable_json_artifact(
        artifact_set.selection_path,
        "verified selection report",
    )
    if selection_identity != derivation.selection_report:
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified selection report identity changed after derivation"
        )
    live_fingerprints = fingerprint_post_render_artifacts(artifact_set)
    serialized_fingerprints = report_root.get("input_artifact_fingerprints")
    if (
        not isinstance(serialized_fingerprints, Mapping)
        or dict(serialized_fingerprints) != live_fingerprints
        or _canonical_sha256(live_fingerprints)
        != derivation.post_render_input_fingerprints_sha256
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified post-render input artifacts changed after derivation"
        )


def _semimarkov_payload_from_report(root: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = root.get("semimarkov")
    if isinstance(nested, Mapping):
        return nested
    if set(root) == {"timelines", "audit"}:
        return root
    raise StageEditContextualBeamQAAdapterV2Error(
        "decode report does not contain one exact semi-Markov result"
    )


def _contextual_decode_identity_v2(
    result_payload: Mapping[str, Any],
) -> tuple[str, str]:
    audit = _required_mapping(result_payload, "audit", "semi-Markov result")
    config = _required_mapping(audit, "config", "semi-Markov audit")
    score_provenance = _required_mapping(
        audit, "score_provenance", "semi-Markov audit"
    )
    boundary = _required_mapping(
        score_provenance,
        "boundary_continuity",
        "semi-Markov audit.score_provenance",
    )
    authority = config.get("continuity_authority")
    scorer_fingerprint = config.get("continuity_scorer_fingerprint")
    if (
        authority
        not in {
            PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
            CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
        }
        or boundary.get("authority") != authority
        or boundary.get("scorer_contract")
        != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT
        or boundary.get("scorer_fingerprint") != scorer_fingerprint
        or not _valid_sha256(scorer_fingerprint)
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "semi-Markov contextual scorer identity is missing or inconsistent"
        )
    return str(authority), str(scorer_fingerprint)


def _selected_timeline(
    result: StageEditSemiMarkovResult, selected_timeline_rank: int
) -> StageEditBeamTimeline:
    if not isinstance(result, StageEditSemiMarkovResult):
        raise TypeError("semimarkov must be StageEditSemiMarkovResult")
    audit = result.audit
    if (
        audit.schema_version != DECODER_SCHEMA_VERSION
        or audit.decoder != DECODER_CONTRACT
        or not _valid_sha256(audit.input_sha256)
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "semi-Markov decoder audit contract is invalid"
        )
    if not (
        audit.request_fulfilled
        and audit.returned_timeline_count == len(result.timelines)
        and audit.requested_timeline_count == len(result.timelines)
        and audit.complete_state_count >= len(result.timelines)
        and tuple(item.rank for item in result.timelines)
        == tuple(range(1, len(result.timelines) + 1))
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "semi-Markov full-decode structure is incomplete"
        )
    authority = audit.config.get("continuity_authority")
    if authority not in {
        PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
        CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
    }:
        raise StageEditContextualBeamQAAdapterV2Error(
            "Beam-QA result was not decoded by contextual continuity v2"
        )
    request_sha256 = audit.input_summary.get("production_request_sha256")
    if not _valid_sha256(request_sha256):
        raise StageEditContextualBeamQAAdapterV2Error(
            "semi-Markov audit is missing its canonical production request SHA-256; "
            "decoder/bridge hook required"
        )
    selected = tuple(
        item for item in result.timelines if item.rank == selected_timeline_rank
    )
    if len(selected) != 1:
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected timeline rank is absent or ambiguous"
        )
    timeline = selected[0]
    if not timeline.decisions or len(timeline.decisions) != len(timeline.edit.shots):
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected timeline decisions and StageEdit shots do not align"
        )
    starts = [item.master_start_seconds for item in timeline.decisions]
    ends = [item.master_end_seconds for item in timeline.decisions]
    if any(
        not math.isfinite(start)
        or not math.isfinite(end)
        or end <= start
        for start, end in zip(starts, ends, strict=True)
    ) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
        for left, right in zip(ends[:-1], starts[1:], strict=True)
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected timeline is not a finite contiguous full decode"
        )
    master_start = audit.input_summary.get("master_start_seconds")
    master_end = audit.input_summary.get("master_end_seconds")
    if not (
        isinstance(master_start, (int, float))
        and not isinstance(master_start, bool)
        and isinstance(master_end, (int, float))
        and not isinstance(master_end, bool)
        and math.isclose(starts[0], float(master_start), abs_tol=1e-6)
        and math.isclose(ends[-1], float(master_end), abs_tol=1e-6)
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected timeline does not cover the audited master range"
        )
    if tuple(item.source_index for item in timeline.decisions) != tuple(
        item.segment.source_index for item in timeline.edit.shots
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected timeline source path differs from its typed StageEdit"
        )
    return timeline


def _selected_rank_from_selection(selection: Mapping[str, Any]) -> int:
    if selection.get("final_selected_version") != "original":
        raise StageEditContextualBeamQAAdapterV2Error(
            "contextual Beam-QA attribution requires the original full decode; "
            "a partial or windowed repair cannot hide affine cadence regressions"
        )
    promotion = _required_mapping(
        selection, "stage_edit_render_promotion", "selection"
    )
    lineage = promotion.get("selected_candidate_lineage")
    if not isinstance(lineage, list):
        raise StageEditContextualBeamQAAdapterV2Error(
            "selection promotion lineage is missing"
        )
    ranks = {
        item.get("stage_edit_candidate_rank")
        for item in lineage
        if isinstance(item, Mapping)
        and isinstance(item.get("stage_edit_candidate_rank"), int)
        and not isinstance(item.get("stage_edit_candidate_rank"), bool)
    }
    if len(ranks) != 1:
        raise StageEditContextualBeamQAAdapterV2Error(
            "selection does not bind exactly one StageEdit candidate rank"
        )
    rank = next(iter(ranks))
    assert isinstance(rank, int)
    if rank < 1:
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected StageEdit candidate rank must be positive"
        )
    return rank


def _verify_timeline_render_source_path(
    *,
    analysis: Mapping[str, Any],
    render_plan: Mapping[str, Any],
    timeline: StageEditBeamTimeline,
) -> dict[str, float | int]:
    mapping = _required_mapping(analysis, "source_index_mapping", "analysis")
    eligible_to_request = mapping.get("eligible_to_request")
    if not isinstance(eligible_to_request, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in eligible_to_request
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "analysis source-index mapping is missing or malformed"
        )
    diversity = _required_mapping(
        analysis, "source_derivative_diversity", "analysis"
    )
    family_ids = diversity.get("family_ids")
    if not isinstance(family_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in family_ids
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "analysis derivative-family mapping is missing or malformed"
        )
    segments = render_plan.get("segments")
    if not isinstance(segments, list) or any(
        not isinstance(value, Mapping) for value in segments
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "render plan segments are missing or malformed"
        )
    if len(segments) != len(timeline.decisions):
        raise StageEditContextualBeamQAAdapterV2Error(
            "render plan segment count differs from the selected decode"
        )
    expected_request_sources: list[int] = []
    for decision in timeline.decisions:
        if not 0 <= decision.source_index < len(eligible_to_request):
            raise StageEditContextualBeamQAAdapterV2Error(
                "selected decode source is outside the eligible/request mapping"
            )
        if (
            decision.source_index >= len(family_ids)
            or family_ids[decision.source_index] != decision.derivative_family_id
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "selected decode derivative family differs from bound analysis evidence"
            )
        expected_request_sources.append(eligible_to_request[decision.source_index])
    plan_sources: list[int] = []
    durations: list[float] = []
    for index, raw in enumerate(segments):
        segment = _strict_json_object(raw, f"render plan segment {index}")
        source_index = segment.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise StageEditContextualBeamQAAdapterV2Error(
                "render plan segment source indices must be integers"
            )
        plan_sources.append(source_index)
        duration = segment.get("duration")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0.0
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "render plan segment durations must be positive finite numbers"
            )
        durations.append(float(duration))
    if plan_sources != expected_request_sources:
        raise StageEditContextualBeamQAAdapterV2Error(
            "final render source path differs from the selected semi-Markov timeline"
        )
    selected_families = [item.derivative_family_id for item in timeline.decisions]
    family_durations: dict[int, float] = {}
    source_runs: list[float] = []
    previous_source_index: int | None = None
    for source_index, family_id, duration in zip(
        plan_sources,
        selected_families,
        durations,
        strict=True,
    ):
        family_durations[family_id] = family_durations.get(family_id, 0.0) + duration
        if previous_source_index == source_index:
            source_runs[-1] += duration
        else:
            source_runs.append(duration)
            previous_source_index = source_index
    total_duration = sum(durations)
    shares = tuple(value / total_duration for value in family_durations.values())
    effective_count = math.exp(
        -sum(value * math.log(value) for value in shares if value > 0.0)
    )
    return {
        "duration_seconds": total_duration,
        "source_switch_count": sum(
            left != right
            for left, right in zip(
                plan_sources[:-1], plan_sources[1:], strict=True
            )
        ),
        "source_run_median_seconds": float(median(source_runs)),
        "longest_source_run_seconds": max(source_runs),
        "distinct_derivative_family_count": len(family_durations),
        "effective_derivative_family_count": effective_count,
        "dominant_derivative_family_share": max(shares),
    }


def _derived_boundary_indices(
    timeline: StageEditBeamTimeline,
    *,
    strict_highlight_section_indices: frozenset[int],
) -> dict[str, tuple[int, ...]]:
    highlight_opportunities: list[int] = []
    highlight_switches: list[int] = []
    closeup_motion_opportunities: list[int] = []
    closeup_motion_switches: list[int] = []
    for boundary_index, (outgoing, incoming) in enumerate(
        zip(timeline.decisions[:-1], timeline.decisions[1:], strict=True)
    ):
        switched = outgoing.derivative_family_id != incoming.derivative_family_id
        incoming_section_index = timeline.edit.shots[
            boundary_index + 1
        ].section_index
        if incoming_section_index in strict_highlight_section_indices:
            highlight_opportunities.append(boundary_index)
            if switched:
                highlight_switches.append(boundary_index)
        closeup_involved = (
            outgoing.shot_size in _CLOSEUP_SHOT_SIZES
            or incoming.shot_size in _CLOSEUP_SHOT_SIZES
        )
        motion_evidence_complete = bool(
            incoming.boundary_motion_amount_plus_continuity_evidence_complete
            and incoming.boundary_temporal_pose_motion_input_valid
            and incoming.boundary_temporal_sampling_contract_match
        )
        if closeup_involved and motion_evidence_complete:
            closeup_motion_opportunities.append(boundary_index)
        if (
            closeup_involved
            and switched
            and motion_evidence_complete
            and incoming.boundary_motion_amount_plus_continuity_supported
            and incoming.boundary_pose_direction_or_velocity_proven
        ):
            closeup_motion_switches.append(boundary_index)
    return {
        "highlight_switch_opportunity_boundary_indices": tuple(
            highlight_opportunities
        ),
        "highlight_switch_boundary_indices": tuple(highlight_switches),
        "closeup_motion_opportunity_boundary_indices": tuple(
            closeup_motion_opportunities
        ),
        "closeup_motion_switch_boundary_indices": tuple(
            closeup_motion_switches
        ),
    }


def _proxy_passed_boundary_indices(
    selection: Mapping[str, Any],
    *,
    expected_boundary_count: int,
) -> tuple[int, ...]:
    review = _required_mapping(selection, "final_selected_review", "selection")
    boundaries = review.get("boundaries")
    if not isinstance(boundaries, list) or any(
        not isinstance(item, Mapping) for item in boundaries
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "selection.final_selected_review.boundaries must be a list of objects"
        )
    if len(boundaries) != expected_boundary_count:
        raise StageEditContextualBeamQAAdapterV2Error(
            "selection final proxy boundary count differs from the selected decode"
        )
    passed: list[int] = []
    for position, raw in enumerate(boundaries):
        item = _strict_json_object(raw, f"selection final boundary {position}")
        if item.get("boundary_index") != position or type(item.get("passed")) is not bool:
            raise StageEditContextualBeamQAAdapterV2Error(
                "selection final proxy boundaries must be sequential with typed pass flags"
            )
        if item["passed"] is True:
            passed.append(position)
    return tuple(passed)


def _metric_sources(
    *,
    report: PostRenderQAReport,
    timeline: StageEditBeamTimeline,
    selection: Mapping[str, Any],
    decode_result_sha256: str,
    decoder_input_sha256: str,
    render_plan_file_sha256: str,
    independently_derived_cadence: Mapping[str, float | int],
) -> tuple[dict[str, Any], str]:
    metrics = _strict_json_object(report.metrics, "post-render metrics")
    cadence = _required_mapping(
        metrics, "cadence_and_repetition", "post_render.metrics"
    )
    proxy = _required_mapping(
        metrics, "proxy_quality_and_geometry", "post_render.metrics"
    )
    tool_checks = _required_mapping(metrics, "tool_checks", "post_render.metrics")
    full_decode_record = _required_mapping(
        tool_checks, "full_decode", "post_render.metrics.tool_checks"
    )
    production_sync = _required_mapping(
        metrics, "production_sync", "post_render.metrics"
    )
    montage = _required_mapping(
        metrics, "rendered_montage_grammar", "post_render.metrics"
    )
    recomputed_review = _required_mapping(
        montage, "recomputed_review", "post_render.metrics.rendered_montage_grammar"
    )
    highlight = _required_mapping(
        recomputed_review,
        "highlight_aggregate",
        "post_render.metrics.rendered_montage_grammar.recomputed_review",
    )
    highlight_observed = _required_mapping(
        highlight,
        "observed",
        "post_render.metrics.rendered_montage_grammar.recomputed_review.highlight_aggregate",
    )
    bound = _required_mapping(
        metrics, "stage_edit_contextual_beam_qa_v2", "post_render.metrics"
    )
    if bound.get("contract") != STAGE_EDIT_CONTEXTUAL_BEAM_QA_POST_RENDER_METRICS_V2_CONTRACT:
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render QA is missing the contextual Beam-QA v2 metric hook"
        )
    if bound.get("source_diversity_contract") != (
        STAGE_EDIT_CONTEXTUAL_BEAM_QA_SOURCE_DIVERSITY_V2_CONTRACT
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render Beam-QA source diversity contract mismatch"
        )
    if bound.get("motion_evidence_contract") != (
        STAGE_EDIT_CONTEXTUAL_BEAM_QA_MOTION_EVIDENCE_V2_CONTRACT
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render Beam-QA motion evidence contract mismatch"
        )
    timeline_sha256 = _canonical_sha256(timeline.to_dict())
    expected_hashes = {
        "decode_result_content_sha256": decode_result_sha256,
        "decoder_input_sha256": decoder_input_sha256,
        "render_plan_file_sha256": render_plan_file_sha256,
        "selected_timeline_content_sha256": timeline_sha256,
    }
    if any(bound.get(key) != value for key, value in expected_hashes.items()):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render Beam-QA metric hook is bound to another decode"
        )
    if _required_int(
        bound,
        "selected_timeline_rank",
        "post_render.metrics.stage_edit_contextual_beam_qa_v2",
        minimum=1,
    ) != timeline.rank:
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render Beam-QA metric hook selected another timeline"
        )

    strict_highlight_section_indices = _required_index_tuple(
        highlight,
        "strict_highlight_section_indices",
        "post_render.metrics.rendered_montage_grammar.recomputed_review."
        "highlight_aggregate",
    )
    highlight_window_count = _required_int(
        highlight,
        "strict_highlight_section_count",
        "post_render.metrics.rendered_montage_grammar.recomputed_review."
        "highlight_aggregate",
    )
    if highlight_window_count != len(strict_highlight_section_indices):
        raise StageEditContextualBeamQAAdapterV2Error(
            "rendered montage strict-highlight count differs from its indices"
        )
    timeline_section_indices = {
        item.section_index for item in timeline.edit.shots
    }
    if not set(strict_highlight_section_indices).issubset(
        timeline_section_indices
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "rendered montage names a highlight section absent from the decode"
        )
    derived_indices = _derived_boundary_indices(
        timeline,
        strict_highlight_section_indices=frozenset(
            strict_highlight_section_indices
        ),
    )
    proxy_passed_indices = frozenset(
        _proxy_passed_boundary_indices(
            selection,
            expected_boundary_count=max(0, len(timeline.decisions) - 1),
        )
    )
    observed_indices = {
        name: _required_index_tuple(
            bound, name, "post_render.metrics.stage_edit_contextual_beam_qa_v2"
        )
        for name in (
            *derived_indices.keys(),
            "proxy_verified_highlight_switch_boundary_indices",
            "proxy_verified_closeup_motion_switch_boundary_indices",
        )
    }
    for name, expected in derived_indices.items():
        if observed_indices[name] != expected:
            raise StageEditContextualBeamQAAdapterV2Error(
                f"post-render {name} differs from typed semi-Markov decisions"
            )
    expected_verified_highlight = tuple(
        index
        for index in observed_indices["highlight_switch_boundary_indices"]
        if index in proxy_passed_indices
    )
    expected_verified_closeup_motion = tuple(
        index
        for index in observed_indices["closeup_motion_switch_boundary_indices"]
        if index in proxy_passed_indices
    )
    if observed_indices[
        "proxy_verified_highlight_switch_boundary_indices"
    ] != expected_verified_highlight:
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified highlight switches differ from bound final proxy pass flags"
        )
    if observed_indices[
        "proxy_verified_closeup_motion_switch_boundary_indices"
    ] != expected_verified_closeup_motion:
        raise StageEditContextualBeamQAAdapterV2Error(
            "verified close-up motion switches differ from bound final proxy pass flags"
        )

    rendered_highlight_switch_count = _required_int(
        highlight_observed,
        "source_switch_count",
        "post_render.metrics.rendered_montage_grammar.recomputed_review."
        "highlight_aggregate.observed",
    )
    if rendered_highlight_switch_count != len(
        observed_indices["highlight_switch_boundary_indices"]
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "rendered montage highlight switches differ from the selected decode"
        )

    geometry_issue_count = (
        _required_int(
            proxy,
            "actual_boundary_issue_count",
            "post_render.metrics.proxy_quality_and_geometry",
        )
        + _required_int(
            proxy,
            "geometry_provider_failure_count",
            "post_render.metrics.proxy_quality_and_geometry",
        )
        + _required_int(
            proxy,
            "insufficient_transition_handle_count",
            "post_render.metrics.proxy_quality_and_geometry",
        )
        + int(
            not _required_bool(
                proxy,
                "boundary_sample_contract_valid",
                "post_render.metrics.proxy_quality_and_geometry",
            )
        )
    )
    per_source = production_sync.get("per_source")
    if not isinstance(per_source, list) or any(
        not isinstance(item, Mapping) for item in per_source
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post_render.metrics.production_sync.per_source must be a list of objects"
        )
    cut_local = _required_mapping(
        production_sync, "cut_local_sync", "post_render.metrics.production_sync"
    )
    cut_errors = cut_local.get("errors", [])
    if not isinstance(cut_errors, list):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render cut-local sync errors must be a list"
        )
    sync_issue_count = (
        sum(item.get("passed") is not True for item in per_source)
        + _required_int(
            cut_local,
            "unknown_boundary_count",
            "post_render.metrics.production_sync.cut_local_sync",
        )
        + len(cut_errors)
        + int(
            not _required_bool(
                production_sync, "passed", "post_render.metrics.production_sync"
            )
        )
    )

    values = {
        "full_decode": bool(
            _required_bool(
                full_decode_record,
                "passed",
                "post_render.metrics.tool_checks.full_decode",
            )
            and report.passed
        ),
        "duration_seconds": _required_float(
            cadence,
            "duration_seconds",
            "post_render.metrics.cadence_and_repetition",
            minimum=_EPSILON,
        ),
        "switch_count": _required_int(
            cadence,
            "source_switch_count",
            "post_render.metrics.cadence_and_repetition",
        ),
        "eligible_switch_opportunity_count": _required_int(
            cadence,
            "comparable_visual_boundary_count",
            "post_render.metrics.cadence_and_repetition",
            minimum=1,
        ),
        "visual_segment_median_seconds": _required_float(
            cadence,
            "visual_shot_median_seconds",
            "post_render.metrics.cadence_and_repetition",
        ),
        "source_run_median_seconds": _required_float(
            cadence,
            "source_run_median_seconds",
            "post_render.metrics.cadence_and_repetition",
        ),
        "longest_source_run_seconds": _required_float(
            cadence,
            "longest_source_run_seconds",
            "post_render.metrics.cadence_and_repetition",
        ),
        "distinct_source_count": _required_int(
            cadence,
            "distinct_derivative_family_count",
            "post_render.metrics.cadence_and_repetition",
            minimum=1,
        ),
        "effective_source_count": _required_float(
            cadence,
            "effective_derivative_family_count",
            "post_render.metrics.cadence_and_repetition",
        ),
        "dominant_source_ratio": _required_float(
            cadence,
            "dominant_derivative_family_share",
            "post_render.metrics.cadence_and_repetition",
        ),
        "quality_score_p10": _required_float(
            proxy,
            "overall_quality_p10",
            "post_render.metrics.proxy_quality_and_geometry",
        ),
        "quality_score_median_span": _required_float(
            proxy,
            "source_quality_median_span",
            "post_render.metrics.proxy_quality_and_geometry",
        ),
        "boundary_match_score_p10": _required_float(
            proxy,
            "boundary_score_p10",
            "post_render.metrics.proxy_quality_and_geometry",
        ),
        "weighted_proxy_score": _required_float(
            proxy,
            "weighted_score",
            "post_render.metrics.proxy_quality_and_geometry",
        ),
        "highlight_window_count": highlight_window_count,
        "highlight_switch_opportunity_count": len(
            observed_indices["highlight_switch_opportunity_boundary_indices"]
        ),
        "highlight_switch_count": len(
            observed_indices["highlight_switch_boundary_indices"]
        ),
        "verified_highlight_switch_count": len(
            observed_indices["proxy_verified_highlight_switch_boundary_indices"]
        ),
        "closeup_motion_opportunity_count": len(
            observed_indices["closeup_motion_opportunity_boundary_indices"]
        ),
        "closeup_motion_switch_count": len(
            observed_indices["closeup_motion_switch_boundary_indices"]
        ),
        "verified_closeup_motion_switch_count": len(
            observed_indices[
                "proxy_verified_closeup_motion_switch_boundary_indices"
            ]
        ),
        "geometry_issue_count": geometry_issue_count,
        "sync_issue_count": sync_issue_count,
    }
    cadence_field_map = {
        "duration_seconds": "duration_seconds",
        "switch_count": "source_switch_count",
        "source_run_median_seconds": "source_run_median_seconds",
        "longest_source_run_seconds": "longest_source_run_seconds",
        "distinct_source_count": "distinct_derivative_family_count",
        "effective_source_count": "effective_derivative_family_count",
        "dominant_source_ratio": "dominant_derivative_family_share",
    }
    for case_field, derived_field in cadence_field_map.items():
        observed = values[case_field]
        expected = independently_derived_cadence[derived_field]
        if isinstance(observed, int) and isinstance(expected, int):
            matches = observed == expected
        else:
            matches = math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1e-6
            )
        if not matches:
            raise StageEditContextualBeamQAAdapterV2Error(
                f"post-render {case_field} differs from the bound render plan"
            )
    if values["effective_source_count"] > values["distinct_source_count"] + _EPSILON:
        raise StageEditContextualBeamQAAdapterV2Error(
            "effective derivative-family count exceeds distinct families"
        )
    if values["dominant_source_ratio"] > 1.0 + _EPSILON:
        raise StageEditContextualBeamQAAdapterV2Error(
            "dominant derivative-family share exceeds one"
        )
    binding_payload = {
        "adapter_contract": STAGE_EDIT_CONTEXTUAL_BEAM_QA_ADAPTER_V2_CONTRACT,
        "metric_hook": dict(bound),
        "derived_boundary_indices": {
            key: list(value) for key, value in derived_indices.items()
        },
        "proxy_passed_boundary_indices": sorted(proxy_passed_indices),
        "values": values,
    }
    return values, _canonical_sha256(binding_payload)


@dataclass(frozen=True, slots=True)
class _ReplayTimelineV2:
    """Minimal typed view over a strictly validated persisted timeline."""

    rank: int
    decisions: tuple[SimpleNamespace, ...]
    edit: SimpleNamespace
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _artifact_identity_from_manifest_v2(
    value: Any,
    name: str,
) -> StageEditContextualBeamQAArtifactIdentityV2:
    raw = _strict_json_object(value, name)
    expected = {
        "contract",
        "path",
        "size_bytes",
        "file_sha256",
        "canonical_content_sha256",
    }
    if set(raw) != expected:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} artifact identity schema changed"
        )
    try:
        return StageEditContextualBeamQAArtifactIdentityV2(
            contract=raw["contract"],
            path=raw["path"],
            size_bytes=raw["size_bytes"],
            file_sha256=raw["file_sha256"],
            canonical_content_sha256=raw["canonical_content_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            f"{name} artifact identity is invalid"
        ) from exc


def _derivation_from_manifest_v2(
    value: Any,
) -> StageEditContextualBeamQADerivationV2:
    raw = _strict_json_object(value, "Beam-QA derivation manifest")
    expected = {
        "contract",
        "request_container",
        "request_content_sha256",
        "decode_report",
        "decode_result_content_sha256",
        "decoder_input_sha256",
        "post_render_report",
        "render_plan",
        "selection_report",
        "post_render_input_fingerprints_sha256",
        "selected_timeline_rank",
        "selected_timeline_content_sha256",
        "metric_binding_sha256",
        "contextual_checkpoint",
        "contextual_corpus",
        "contextual_checkpoint_sha256",
        "contextual_corpus_sha256",
        "decode_continuity_authority",
        "decode_scorer_fingerprint",
        "contextual_runtime_equivalence_sha256",
        "contextual_token_schema_fingerprint",
        "derivation_sha256",
    }
    if set(raw) != expected:
        raise StageEditContextualBeamQAAdapterV2Error(
            "Beam-QA derivation manifest schema changed"
        )
    try:
        return StageEditContextualBeamQADerivationV2(
            contract=raw["contract"],
            request_container=_artifact_identity_from_manifest_v2(
                raw["request_container"], "request container"
            ),
            request_content_sha256=raw["request_content_sha256"],
            decode_report=_artifact_identity_from_manifest_v2(
                raw["decode_report"], "decode report"
            ),
            decode_result_content_sha256=raw[
                "decode_result_content_sha256"
            ],
            decoder_input_sha256=raw["decoder_input_sha256"],
            post_render_report=_artifact_identity_from_manifest_v2(
                raw["post_render_report"], "post-render report"
            ),
            render_plan=_artifact_identity_from_manifest_v2(
                raw["render_plan"], "render plan"
            ),
            selection_report=_artifact_identity_from_manifest_v2(
                raw["selection_report"], "selection report"
            ),
            post_render_input_fingerprints_sha256=raw[
                "post_render_input_fingerprints_sha256"
            ],
            selected_timeline_rank=raw["selected_timeline_rank"],
            selected_timeline_content_sha256=raw[
                "selected_timeline_content_sha256"
            ],
            metric_binding_sha256=raw["metric_binding_sha256"],
            contextual_checkpoint=(
                _artifact_identity_from_manifest_v2(
                    raw["contextual_checkpoint"],
                    "contextual evaluation checkpoint",
                )
                if raw["contextual_checkpoint"] is not None
                else None
            ),
            contextual_corpus=(
                _artifact_identity_from_manifest_v2(
                    raw["contextual_corpus"],
                    "contextual evaluation corpus",
                )
                if raw["contextual_corpus"] is not None
                else None
            ),
            contextual_checkpoint_sha256=raw[
                "contextual_checkpoint_sha256"
            ],
            contextual_corpus_sha256=raw["contextual_corpus_sha256"],
            decode_continuity_authority=raw[
                "decode_continuity_authority"
            ],
            decode_scorer_fingerprint=raw["decode_scorer_fingerprint"],
            contextual_runtime_equivalence_sha256=raw[
                "contextual_runtime_equivalence_sha256"
            ],
            contextual_token_schema_fingerprint=raw[
                "contextual_token_schema_fingerprint"
            ],
            derivation_sha256=raw["derivation_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            "Beam-QA derivation manifest is invalid"
        ) from exc


def _post_render_config_from_manifest_v2(value: Any) -> PostRenderQAConfig:
    raw = _strict_json_object(value, "post-render policy")
    defaults = PostRenderQAConfig()
    if set(raw) != set(defaults.to_dict()):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render policy schema changed"
        )
    perceptual_raw = _strict_json_object(
        raw.get("perceptual_quality"), "post-render perceptual policy"
    )
    montage_raw = _strict_json_object(
        raw.get("rendered_montage_grammar"),
        "post-render montage policy",
    )
    if set(perceptual_raw) != set(PerceptualQualityConfig().__dataclass_fields__):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render perceptual policy schema changed"
        )
    if set(montage_raw) != set(
        RenderedMontageGrammarConfig().__dataclass_fields__
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render montage policy schema changed"
        )
    kwargs = dict(raw)
    try:
        kwargs["perceptual_quality"] = PerceptualQualityConfig(
            **dict(perceptual_raw)
        )
        kwargs["rendered_montage_grammar"] = RenderedMontageGrammarConfig(
            **dict(montage_raw)
        )
        config = PostRenderQAConfig(**kwargs)
    except (TypeError, ValueError) as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render policy is invalid"
        ) from exc
    if config.to_dict() != dict(raw):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render policy does not round-trip exactly"
        )
    return config


def _artifact_set_from_report_manifest_v2(
    report_root: Mapping[str, Any],
) -> PostRenderArtifactSet:
    raw = _required_mapping(report_root, "artifact_set", "post_render_report")
    expected = {
        "base_name",
        "timestamp",
        "analysis_path",
        "render_plan_path",
        "selection_path",
        "video_path",
    }
    if set(raw) != expected:
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render artifact-set schema changed"
        )
    try:
        artifact_set = PostRenderArtifactSet(
            base_name=raw["base_name"],
            timestamp=raw["timestamp"],
            analysis_path=_strict_absolute_path(
                Path(raw["analysis_path"]), "analysis artifact path"
            ),
            render_plan_path=_strict_absolute_path(
                Path(raw["render_plan_path"]), "render-plan artifact path"
            ),
            selection_path=_strict_absolute_path(
                Path(raw["selection_path"]), "selection artifact path"
            ),
            video_path=_strict_absolute_path(
                Path(raw["video_path"]), "video artifact path"
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render artifact set is invalid"
        ) from exc
    if artifact_set.to_dict() != dict(raw):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render artifact set does not round-trip exactly"
        )
    return artifact_set


def _replay_timeline_from_payload_v2(
    result_payload: Mapping[str, Any],
    *,
    selected_timeline_rank: int,
    request_sha256: str,
) -> tuple[_ReplayTimelineV2, str]:
    if set(result_payload) != {"timelines", "audit"}:
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted semi-Markov result schema changed"
        )
    timelines_raw = result_payload.get("timelines")
    audit = _strict_json_object(
        result_payload.get("audit"), "persisted semi-Markov audit"
    )
    audit_keys = {
        "schema_version",
        "decoder",
        "input_sha256",
        "readiness",
        "config",
        "input_summary",
        "endpoint_audit",
        "constraint_rejection_counts",
        "transition_hypothesis_count",
        "accepted_hypothesis_count",
        "beam_pruned_state_count",
        "complete_state_count",
        "duplicate_completion_count",
        "requested_timeline_count",
        "returned_timeline_count",
        "request_fulfilled",
        "score_provenance",
    }
    if set(audit) != audit_keys:
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted semi-Markov audit schema changed"
        )
    if not isinstance(timelines_raw, list) or not timelines_raw or any(
        not isinstance(value, Mapping) for value in timelines_raw
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted semi-Markov timelines are missing or malformed"
        )
    config = _required_mapping(audit, "config", "persisted semi-Markov audit")
    summary = _required_mapping(
        audit, "input_summary", "persisted semi-Markov audit"
    )
    input_sha256 = audit.get("input_sha256")
    if (
        audit.get("schema_version") != DECODER_SCHEMA_VERSION
        or audit.get("decoder") != DECODER_CONTRACT
        or not _valid_sha256(input_sha256)
        or config.get("continuity_authority")
        not in {
            PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
            CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2,
        }
        or summary.get("production_request_sha256") != request_sha256
        or audit.get("request_fulfilled") is not True
        or audit.get("requested_timeline_count") != len(timelines_raw)
        or audit.get("returned_timeline_count") != len(timelines_raw)
        or not isinstance(audit.get("complete_state_count"), int)
        or isinstance(audit.get("complete_state_count"), bool)
        or audit.get("complete_state_count") < len(timelines_raw)
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted semi-Markov full-decode audit is invalid"
        )
    ranks = tuple(value.get("rank") for value in timelines_raw)
    if ranks != tuple(range(1, len(timelines_raw) + 1)):
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted semi-Markov timeline ranks are not canonical"
        )
    selected_raw = tuple(
        value
        for value in timelines_raw
        if value.get("rank") == selected_timeline_rank
    )
    if len(selected_raw) != 1:
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected timeline rank is absent or ambiguous"
        )
    timeline_payload = dict(selected_raw[0])
    decisions_raw = timeline_payload.get("decisions")
    edit_raw = _strict_json_object(
        timeline_payload.get("stage_edit"), "persisted StageEdit"
    )
    shots_raw = edit_raw.get("shots")
    if (
        not isinstance(decisions_raw, list)
        or not decisions_raw
        or any(not isinstance(value, Mapping) for value in decisions_raw)
        or not isinstance(shots_raw, list)
        or len(shots_raw) != len(decisions_raw)
        or any(not isinstance(value, Mapping) for value in shots_raw)
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected timeline decisions and StageEdit shots do not align"
        )
    decisions: list[SimpleNamespace] = []
    shots: list[SimpleNamespace] = []
    starts: list[float] = []
    ends: list[float] = []
    for index, (decision_raw, shot_raw) in enumerate(
        zip(decisions_raw, shots_raw, strict=True)
    ):
        decision = _strict_json_object(
            decision_raw, f"persisted decision {index}"
        )
        shot = _strict_json_object(shot_raw, f"persisted StageEdit shot {index}")
        motion = _required_mapping(
            decision,
            "boundary_motion_amount_plus_continuity",
            f"persisted decision {index}",
        )
        temporal = _required_mapping(
            motion,
            "temporal_pose_motion",
            f"persisted decision {index}.boundary_motion_amount_plus_continuity",
        )
        for owner, key in (
            (motion, "evidence_complete"),
            (motion, "accent_supported"),
            (motion, "pose_direction_or_velocity_proven"),
            (temporal, "input_valid"),
            (temporal, "sampling_contract_match"),
        ):
            if type(owner.get(key)) is not bool:
                raise StageEditContextualBeamQAAdapterV2Error(
                    "persisted decision motion evidence is not typed"
                )
        source_index = decision.get("source_index")
        family_id = decision.get("derivative_family_id")
        shot_size = decision.get("shot_size")
        start = decision.get("master_start_seconds")
        end = decision.get("master_end_seconds")
        shot_source = shot.get("source_index")
        section_index = shot.get("section_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or isinstance(family_id, bool)
            or not isinstance(family_id, int)
            or family_id < 0
            or not isinstance(shot_size, str)
            or not shot_size
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(end) <= float(start)
            or shot_source != source_index
            or isinstance(section_index, bool)
            or not isinstance(section_index, int)
            or section_index < 0
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "persisted decision/StageEdit source or time evidence is invalid"
            )
        starts.append(float(start))
        ends.append(float(end))
        decisions.append(
            SimpleNamespace(
                source_index=source_index,
                derivative_family_id=family_id,
                shot_size=shot_size,
                boundary_motion_amount_plus_continuity_evidence_complete=(
                    motion["evidence_complete"]
                ),
                boundary_motion_amount_plus_continuity_supported=(
                    motion["accent_supported"]
                ),
                boundary_temporal_pose_motion_input_valid=(
                    temporal["input_valid"]
                ),
                boundary_temporal_sampling_contract_match=(
                    temporal["sampling_contract_match"]
                ),
                boundary_pose_direction_or_velocity_proven=(
                    motion["pose_direction_or_velocity_proven"]
                ),
            )
        )
        shots.append(
            SimpleNamespace(
                section_index=section_index,
                segment=SimpleNamespace(source_index=shot_source),
            )
        )
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
        for left, right in zip(ends[:-1], starts[1:], strict=True)
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected timeline is not contiguous"
        )
    master_start = summary.get("master_start_seconds")
    master_end = summary.get("master_end_seconds")
    if (
        isinstance(master_start, bool)
        or not isinstance(master_start, (int, float))
        or isinstance(master_end, bool)
        or not isinstance(master_end, (int, float))
        or not math.isclose(starts[0], float(master_start), abs_tol=1e-6)
        or not math.isclose(ends[-1], float(master_end), abs_tol=1e-6)
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "selected timeline does not cover its audited master range"
        )
    return (
        _ReplayTimelineV2(
            rank=selected_timeline_rank,
            decisions=tuple(decisions),
            edit=SimpleNamespace(shots=tuple(shots)),
            payload=timeline_payload,
        ),
        str(input_sha256),
    )


def rehydrate_stage_edit_contextual_beam_qa_case_v2(
    *,
    case_payload: Mapping[str, Any],
    derivation_payload: Mapping[str, Any],
) -> VerifiedStageEditContextualBeamQACaseV2:
    """Replay a persisted registry certificate and issue a fresh process seal.

    Persisted DTOs never become activation authority by construction.  Every
    artifact is reopened through the adapter's root-fd/no-follow reader, the
    post-render evaluator and the same metric derivation math are replayed, and
    only the freshly reconstructed values may receive the process-local seal.
    """

    try:
        persisted_case = StageEditContextualBeamQACaseV2.from_dict(
            _strict_json_object(case_payload, "Beam-QA case manifest")
        )
        persisted_derivation = _derivation_from_manifest_v2(
            derivation_payload
        )
    except StageEditContextualBeamQAAdapterV2Error:
        raise
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted Beam-QA manifest is invalid"
        ) from exc

    analysis_root, analysis_identity = _read_stable_json_artifact(
        Path(persisted_derivation.request_container.path),
        "rehydrated request container",
    )
    decode_root, decode_identity = _read_stable_json_artifact(
        Path(persisted_derivation.decode_report.path),
        "rehydrated decode report",
    )
    report_root, report_identity = _read_stable_json_artifact(
        Path(persisted_derivation.post_render_report.path),
        "rehydrated post-render report",
    )
    if (
        analysis_identity != persisted_derivation.request_container
        or decode_identity != persisted_derivation.decode_report
        or report_identity != persisted_derivation.post_render_report
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted Beam-QA primary artifact identity changed"
        )

    request_payload = _required_mapping(analysis_root, "request", "analysis")
    request_sha256 = _canonical_sha256(request_payload)
    if request_sha256 != persisted_derivation.request_content_sha256:
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted Beam-QA request content changed"
        )
    result_payload = _semimarkov_payload_from_report(decode_root)
    result_sha256 = _canonical_sha256(result_payload)
    timeline, decoder_input_sha256 = _replay_timeline_from_payload_v2(
        result_payload,
        selected_timeline_rank=persisted_derivation.selected_timeline_rank,
        request_sha256=request_sha256,
    )
    contextual_checkpoint_identity = None
    contextual_corpus_identity = None
    contextual_checkpoint_sha256 = None
    contextual_corpus_sha256 = None
    decode_continuity_authority = None
    decode_scorer_fingerprint = None
    contextual_runtime_equivalence_sha256 = None
    contextual_token_schema_fingerprint = None
    if persisted_derivation.contextual_checkpoint is not None:
        assert persisted_derivation.contextual_corpus is not None
        checkpoint_root, contextual_checkpoint_identity = (
            _read_stable_json_artifact(
                Path(persisted_derivation.contextual_checkpoint.path),
                "rehydrated contextual evaluation checkpoint",
            )
        )
        corpus_root, contextual_corpus_identity = _read_stable_json_artifact(
            Path(persisted_derivation.contextual_corpus.path),
            "rehydrated contextual evaluation corpus",
        )
        if (
            contextual_checkpoint_identity
            != persisted_derivation.contextual_checkpoint
            or contextual_corpus_identity
            != persisted_derivation.contextual_corpus
            or Path(contextual_checkpoint_identity.path).name
            != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME
            or Path(contextual_corpus_identity.path).name
            != STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME
            or Path(contextual_checkpoint_identity.path).parent
            != Path(contextual_corpus_identity.path).parent
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "persisted contextual evaluation artifact identity changed"
            )
        evaluation_bundle = (
            load_stage_edit_contextual_source_pair_evaluation_bundle_v2(
                Path(contextual_checkpoint_identity.path).parent,
                expected_checkpoint_sha256=(
                    persisted_derivation.contextual_checkpoint_sha256
                ),
                expected_checkpoint_file_sha256=(
                    contextual_checkpoint_identity.file_sha256
                ),
                expected_corpus_sha256=(
                    persisted_derivation.contextual_corpus_sha256
                ),
                expected_corpus_file_sha256=(
                    contextual_corpus_identity.file_sha256
                ),
                expected_token_schema_fingerprint=(
                    persisted_derivation.contextual_token_schema_fingerprint
                ),
                expected_scorer_fingerprint=(
                    persisted_derivation.decode_scorer_fingerprint
                ),
                expected_runtime_equivalence_sha256=(
                    persisted_derivation.contextual_runtime_equivalence_sha256
                ),
            )
        )
        decode_continuity_authority, decode_scorer_fingerprint = (
            _contextual_decode_identity_v2(result_payload)
        )
        if (
            decode_continuity_authority
            != CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
            or decode_scorer_fingerprint != evaluation_bundle.scorer.fingerprint
            or checkpoint_root != evaluation_bundle.checkpoint.to_dict()
            or corpus_root != evaluation_bundle.corpus.to_dict()
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "persisted decode scorer differs from contextual evaluation checkpoint"
            )
        contextual_checkpoint_sha256 = (
            evaluation_bundle.checkpoint.checkpoint_sha256
        )
        contextual_corpus_sha256 = evaluation_bundle.corpus.content_sha256
        contextual_runtime_equivalence_sha256 = (
            evaluation_bundle.checkpoint.runtime_equivalence_sha256
        )
        contextual_token_schema_fingerprint = (
            evaluation_bundle.checkpoint.token_schema_fingerprint
        )
    generation = _required_mapping(
        analysis_root, "stage_edit_generation", "analysis"
    )
    if (
        generation.get("semimarkov") != result_payload
        or generation.get("audit_path")
        != persisted_derivation.decode_report.path
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "analysis embeds another semi-Markov decode"
        )

    artifact_set = _artifact_set_from_report_manifest_v2(report_root)
    if artifact_set.analysis_path != Path(
        persisted_derivation.request_container.path
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render report names another request container"
        )
    render_plan_root, render_plan_identity = _read_stable_json_artifact(
        artifact_set.render_plan_path, "rehydrated render plan"
    )
    selection_root, selection_identity = _read_stable_json_artifact(
        artifact_set.selection_path, "rehydrated selection report"
    )
    if (
        render_plan_identity != persisted_derivation.render_plan
        or selection_identity != persisted_derivation.selection_report
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted Beam-QA render or selection identity changed"
        )
    if _selected_rank_from_selection(selection_root) != timeline.rank:
        raise StageEditContextualBeamQAAdapterV2Error(
            "rendered selection chose another timeline"
        )

    video_fingerprint = _fingerprint_stable_input_artifact(
        artifact_set.video_path,
        "rehydrated rendered video",
    )
    hardened_fingerprints = {
        "contract": INPUT_ARTIFACT_FINGERPRINT_CONTRACT,
        "analysis": _artifact_input_fingerprint(analysis_identity),
        "render_plan": _artifact_input_fingerprint(render_plan_identity),
        "selection": _artifact_input_fingerprint(selection_identity),
        "video": video_fingerprint,
    }
    live_fingerprints = fingerprint_post_render_artifacts(artifact_set)
    serialized_fingerprints = report_root.get("input_artifact_fingerprints")
    metrics = _required_mapping(report_root, "metrics", "post_render_report")
    stability = _required_mapping(
        metrics, "artifact_content_stability", "post_render_report.metrics"
    )
    if (
        not isinstance(serialized_fingerprints, Mapping)
        or dict(serialized_fingerprints) != hardened_fingerprints
        or live_fingerprints != hardened_fingerprints
        or stability.get("final_fingerprints") != hardened_fingerprints
        or _canonical_sha256(hardened_fingerprints)
        != persisted_derivation.post_render_input_fingerprints_sha256
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted post-render input artifacts changed"
        )

    policy = _post_render_config_from_manifest_v2(
        report_root.get("policy")
    )
    created_raw = report_root.get("created_at")
    if not isinstance(created_raw, str):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render report created_at is invalid"
        )
    try:
        created_at = datetime.fromisoformat(created_raw)
    except ValueError as exc:
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render report created_at is invalid"
        ) from exc
    if created_at.tzinfo is None:
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render report created_at lacks timezone"
        )
    recomputed_report = evaluate_post_render_qa(
        artifact_set,
        config=policy,
        created_at=created_at,
        trusted_registered_stage_edit=True,
    )
    _analysis_after, analysis_identity_after = _read_stable_json_artifact(
        artifact_set.analysis_path,
        "rehydrated request container after evaluation",
    )
    _render_plan_after, render_plan_identity_after = _read_stable_json_artifact(
        artifact_set.render_plan_path,
        "rehydrated render plan after evaluation",
    )
    _selection_after, selection_identity_after = _read_stable_json_artifact(
        artifact_set.selection_path,
        "rehydrated selection report after evaluation",
    )
    video_fingerprint_after = _fingerprint_stable_input_artifact(
        artifact_set.video_path,
        "rehydrated rendered video after evaluation",
    )
    if (
        analysis_identity_after != analysis_identity
        or render_plan_identity_after != render_plan_identity
        or selection_identity_after != selection_identity
        or video_fingerprint_after != video_fingerprint
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted post-render input artifacts changed during re-evaluation"
        )
    final_video_name = artifact_set.video_path.name
    if _normalized_post_render_revalidation_payload(
        recomputed_report.to_dict(), final_video_name=final_video_name
    ) != _normalized_post_render_revalidation_payload(
        report_root, final_video_name=final_video_name
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted post-render report differs from independent re-evaluation"
        )

    independently_derived_cadence = _verify_timeline_render_source_path(
        analysis=analysis_root,
        render_plan=render_plan_root,
        timeline=timeline,  # type: ignore[arg-type]
    )
    report_view = SimpleNamespace(
        metrics=metrics,
        passed=report_root.get("passed") is True,
    )
    values, metric_binding_sha256 = _metric_sources(
        report=report_view,  # type: ignore[arg-type]
        timeline=timeline,  # type: ignore[arg-type]
        selection=selection_root,
        decode_result_sha256=result_sha256,
        decoder_input_sha256=decoder_input_sha256,
        render_plan_file_sha256=render_plan_identity.file_sha256,
        independently_derived_cadence=independently_derived_cadence,
    )
    song_id = str(request_payload.get("song") or "").strip()
    if not song_id:
        raise StageEditContextualBeamQAAdapterV2Error(
            "analysis request is missing the song identity"
        )
    safe_song = re.sub(r"[^0-9A-Za-z가-힣]+", "-", song_id).strip("-") or "song"
    replayed_case = StageEditContextualBeamQACaseV2(
        case_id=(
            f"{safe_song.lower()}-rank-{timeline.rank}-{request_sha256[:12]}"
        ),
        song_id=song_id,
        request_sha256=request_sha256,
        decode_report_sha256=decode_identity.file_sha256,
        **values,
    )
    replayed_derivation = StageEditContextualBeamQADerivationV2(
        request_container=analysis_identity,
        request_content_sha256=request_sha256,
        decode_report=decode_identity,
        decode_result_content_sha256=result_sha256,
        decoder_input_sha256=decoder_input_sha256,
        post_render_report=report_identity,
        render_plan=render_plan_identity,
        selection_report=selection_identity,
        post_render_input_fingerprints_sha256=_canonical_sha256(
            hardened_fingerprints
        ),
        selected_timeline_rank=timeline.rank,
        selected_timeline_content_sha256=_canonical_sha256(
            timeline.to_dict()
        ),
        metric_binding_sha256=metric_binding_sha256,
        contextual_checkpoint=contextual_checkpoint_identity,
        contextual_corpus=contextual_corpus_identity,
        contextual_checkpoint_sha256=contextual_checkpoint_sha256,
        contextual_corpus_sha256=contextual_corpus_sha256,
        decode_continuity_authority=decode_continuity_authority,
        decode_scorer_fingerprint=decode_scorer_fingerprint,
        contextual_runtime_equivalence_sha256=(
            contextual_runtime_equivalence_sha256
        ),
        contextual_token_schema_fingerprint=(
            contextual_token_schema_fingerprint
        ),
    )
    if (
        replayed_case != persisted_case
        or replayed_derivation != persisted_derivation
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "persisted Beam-QA DTO differs from full artifact replay"
        )
    return VerifiedStageEditContextualBeamQACaseV2(
        case=replayed_case,
        derivation=replayed_derivation,
        _seal=_VERIFICATION_SEAL,
    )


def derive_stage_edit_contextual_beam_qa_case_v2(
    *,
    semimarkov: StageEditSemiMarkovResult,
    selected_timeline_rank: int,
    post_render_report: PostRenderQAReport,
    decode_report_path: Path,
    post_render_report_path: Path,
    contextual_evaluation_bundle_v2: (
        StageEditContextualSourcePairEvaluationBundleV2 | None
    ) = None,
) -> VerifiedStageEditContextualBeamQACaseV2:
    """Reopen, re-evaluate, and derive one activation-eligible Beam-QA case.

    No metric or pass boolean is accepted as an argument.  Missing post-render
    fields fail closed instead of being guessed from a nearby metric.
    """

    if not isinstance(post_render_report, PostRenderQAReport):
        raise TypeError("post_render_report must be PostRenderQAReport")
    timeline = _selected_timeline(semimarkov, selected_timeline_rank)
    result_payload = semimarkov.to_dict()
    result_sha256 = _canonical_sha256(result_payload)
    timeline_sha256 = _canonical_sha256(timeline.to_dict())
    contextual_checkpoint_identity = None
    contextual_corpus_identity = None
    contextual_checkpoint_sha256 = None
    contextual_corpus_sha256 = None
    decode_continuity_authority = None
    decode_scorer_fingerprint = None
    contextual_runtime_equivalence_sha256 = None
    contextual_token_schema_fingerprint = None
    if contextual_evaluation_bundle_v2 is not None:
        evaluation_bundle = (
            require_stage_edit_contextual_source_pair_evaluation_bundle_v2(
                contextual_evaluation_bundle_v2
            )
        )
        checkpoint_root, contextual_checkpoint_identity = (
            _read_stable_json_artifact(
                evaluation_bundle.directory
                / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CHECKPOINT_FILENAME,
                "contextual evaluation checkpoint",
            )
        )
        corpus_root, contextual_corpus_identity = _read_stable_json_artifact(
            evaluation_bundle.directory
            / STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_CORPUS_FILENAME,
            "contextual evaluation corpus",
        )
        decode_continuity_authority, decode_scorer_fingerprint = (
            _contextual_decode_identity_v2(result_payload)
        )
        if (
            contextual_checkpoint_identity.file_sha256
            != evaluation_bundle.checkpoint_file_sha256
            or contextual_corpus_identity.file_sha256
            != evaluation_bundle.corpus_file_sha256
            or checkpoint_root != evaluation_bundle.checkpoint.to_dict()
            or corpus_root != evaluation_bundle.corpus.to_dict()
            or decode_continuity_authority
            != CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
            or decode_scorer_fingerprint != evaluation_bundle.scorer.fingerprint
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "Beam-QA decode scorer differs from its pinned evaluation bundle"
            )
        contextual_checkpoint_sha256 = (
            evaluation_bundle.checkpoint.checkpoint_sha256
        )
        contextual_corpus_sha256 = evaluation_bundle.corpus.content_sha256
        contextual_runtime_equivalence_sha256 = (
            evaluation_bundle.checkpoint.runtime_equivalence_sha256
        )
        contextual_token_schema_fingerprint = (
            evaluation_bundle.checkpoint.token_schema_fingerprint
        )
    else:
        audit_config = _required_mapping(
            semimarkov.audit.to_dict(), "config", "semi-Markov audit"
        )
        if audit_config.get("continuity_authority") == (
            CONTEXTUAL_BEAM_QA_EVALUATION_CONTINUITY_AUTHORITY_V2
        ):
            raise StageEditContextualBeamQAAdapterV2Error(
                "evaluation-authority Beam-QA needs its loader-issued bundle"
            )

    decode_root, decode_identity = _read_stable_json_artifact(
        Path(decode_report_path), "decode_report_path"
    )
    embedded_decode = _semimarkov_payload_from_report(decode_root)
    if embedded_decode != result_payload:
        raise StageEditContextualBeamQAAdapterV2Error(
            "typed semi-Markov result differs from the immutable decode report"
        )

    report_root, report_identity = _read_stable_json_artifact(
        Path(post_render_report_path), "post_render_report_path"
    )
    if report_root != post_render_report.to_dict():
        raise StageEditContextualBeamQAAdapterV2Error(
            "typed post-render report differs from the immutable report artifact"
        )
    recomputed_report = evaluate_post_render_qa(
        post_render_report.artifact_set,
        config=post_render_report.policy,
        created_at=post_render_report.created_at,
        trusted_registered_stage_edit=True,
    )
    final_video_name = post_render_report.artifact_set.video_path.name
    recomputed_payload = _normalized_post_render_revalidation_payload(
        recomputed_report.to_dict(),
        final_video_name=final_video_name,
    )
    reported_payload = _normalized_post_render_revalidation_payload(
        post_render_report.to_dict(),
        final_video_name=final_video_name,
    )
    if recomputed_payload != reported_payload:
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render report does not match an independent re-evaluation"
        )
    current_fingerprints = fingerprint_post_render_artifacts(
        post_render_report.artifact_set
    )
    if (
        current_fingerprints != post_render_report.input_artifact_fingerprints
        or (
            _required_mapping(
                post_render_report.metrics,
                "artifact_content_stability",
                "post_render.metrics",
            ).get("final_fingerprints")
            != current_fingerprints
        )
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "post-render input artifact fingerprints are not stable and exact"
        )

    analysis_root, analysis_identity = _read_stable_json_artifact(
        post_render_report.artifact_set.analysis_path,
        "post_render_report.artifact_set.analysis_path",
    )
    request_payload = _required_mapping(analysis_root, "request", "analysis")
    request_sha256 = _canonical_sha256(request_payload)
    if semimarkov.audit.input_summary.get("production_request_sha256") != request_sha256:
        raise StageEditContextualBeamQAAdapterV2Error(
            "semi-Markov decode is bound to another production request"
        )
    generation = _required_mapping(analysis_root, "stage_edit_generation", "analysis")
    if generation.get("semimarkov") != result_payload:
        raise StageEditContextualBeamQAAdapterV2Error(
            "analysis artifact embeds a different semi-Markov result"
        )
    audit_path = generation.get("audit_path")
    if not isinstance(audit_path, str) or Path(audit_path) != Path(decode_report_path):
        raise StageEditContextualBeamQAAdapterV2Error(
            "analysis StageEdit audit path differs from the decode report"
        )
    render_plan_root, render_plan_identity = _read_stable_json_artifact(
        post_render_report.artifact_set.render_plan_path,
        "post_render_report.artifact_set.render_plan_path",
    )
    expected_render_plan_fingerprint = current_fingerprints.get("render_plan")
    if (
        not isinstance(expected_render_plan_fingerprint, Mapping)
        or expected_render_plan_fingerprint.get("sha256")
        != render_plan_identity.file_sha256
    ):
        raise StageEditContextualBeamQAAdapterV2Error(
            "render plan byte identity differs from post-render QA fingerprints"
        )
    independently_derived_cadence = _verify_timeline_render_source_path(
        analysis=analysis_root,
        render_plan=render_plan_root,
        timeline=timeline,
    )
    selection_root, selection_identity = _read_stable_json_artifact(
        post_render_report.artifact_set.selection_path,
        "post_render_report.artifact_set.selection_path",
    )
    if _selected_rank_from_selection(selection_root) != selected_timeline_rank:
        raise StageEditContextualBeamQAAdapterV2Error(
            "final rendered selection chose another StageEdit timeline"
        )

    values, metric_binding_sha256 = _metric_sources(
        report=post_render_report,
        timeline=timeline,
        selection=selection_root,
        decode_result_sha256=result_sha256,
        decoder_input_sha256=semimarkov.audit.input_sha256,
        render_plan_file_sha256=render_plan_identity.file_sha256,
        independently_derived_cadence=independently_derived_cadence,
    )
    song_id = str(request_payload.get("song") or "").strip()
    if not song_id:
        raise StageEditContextualBeamQAAdapterV2Error(
            "analysis request is missing the song identity"
        )
    safe_song = re.sub(r"[^0-9A-Za-z가-힣]+", "-", song_id).strip("-") or "song"
    case = StageEditContextualBeamQACaseV2(
        case_id=(
            f"{safe_song.lower()}-rank-{selected_timeline_rank}-"
            f"{request_sha256[:12]}"
        ),
        song_id=song_id,
        request_sha256=request_sha256,
        decode_report_sha256=decode_identity.file_sha256,
        **values,
    )
    derivation = StageEditContextualBeamQADerivationV2(
        request_container=analysis_identity,
        request_content_sha256=request_sha256,
        decode_report=decode_identity,
        decode_result_content_sha256=result_sha256,
        decoder_input_sha256=semimarkov.audit.input_sha256,
        post_render_report=report_identity,
        render_plan=render_plan_identity,
        selection_report=selection_identity,
        post_render_input_fingerprints_sha256=_canonical_sha256(
            current_fingerprints
        ),
        selected_timeline_rank=selected_timeline_rank,
        selected_timeline_content_sha256=timeline_sha256,
        metric_binding_sha256=metric_binding_sha256,
        contextual_checkpoint=contextual_checkpoint_identity,
        contextual_corpus=contextual_corpus_identity,
        contextual_checkpoint_sha256=contextual_checkpoint_sha256,
        contextual_corpus_sha256=contextual_corpus_sha256,
        decode_continuity_authority=decode_continuity_authority,
        decode_scorer_fingerprint=decode_scorer_fingerprint,
        contextual_runtime_equivalence_sha256=(
            contextual_runtime_equivalence_sha256
        ),
        contextual_token_schema_fingerprint=(
            contextual_token_schema_fingerprint
        ),
    )
    return VerifiedStageEditContextualBeamQACaseV2(
        case=case,
        derivation=derivation,
        _seal=_VERIFICATION_SEAL,
    )


__all__ = [
    "STAGE_EDIT_CONTEXTUAL_BEAM_QA_ADAPTER_V2_CONTRACT",
    "STAGE_EDIT_CONTEXTUAL_BEAM_QA_ARTIFACT_IDENTITY_V2_CONTRACT",
    "STAGE_EDIT_CONTEXTUAL_BEAM_QA_MOTION_EVIDENCE_V2_CONTRACT",
    "STAGE_EDIT_CONTEXTUAL_BEAM_QA_POST_RENDER_METRICS_V2_CONTRACT",
    "STAGE_EDIT_CONTEXTUAL_BEAM_QA_SOURCE_DIVERSITY_V2_CONTRACT",
    "StageEditContextualBeamQAAdapterV2Error",
    "StageEditContextualBeamQAArtifactIdentityV2",
    "StageEditContextualBeamQADerivationV2",
    "VerifiedStageEditContextualBeamQACaseV2",
    "derive_stage_edit_contextual_beam_qa_case_v2",
    "is_verified_stage_edit_contextual_beam_qa_case_v2",
    "rehydrate_stage_edit_contextual_beam_qa_case_v2",
    "reverify_stage_edit_contextual_beam_qa_case_v2",
    "require_verified_stage_edit_contextual_beam_qa_case_v2",
]
