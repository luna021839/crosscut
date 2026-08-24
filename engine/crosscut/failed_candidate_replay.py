"""Strict, QA-only replay of a serialized failed render candidate.

This module deliberately does *not* perform acquisition, synchronization,
visual optimization, or candidate selection.  It accepts only a failed
``rendered-candidate-selection.json`` artifact, binds every source to the
SHA-256 recorded by the acquisition manifest, reconstructs one complete
candidate, and renders the same fixed-height proxy used by production QA.

The output is evidence for engineering iteration, never a production final.
The CLI therefore refuses an output name that does not contain
``qa-replay`` and refuses to overwrite any existing file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import cv2

from .ive import AspectRatio
from .render import (
    BoundaryLookKeyframe,
    BoundaryLookTrack,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    SourceLookAdjustment,
    TransitionKind,
    output_size,
    segment_frame_counts,
)
from .rendered_proxy_review import (
    ANALYSIS_BASIS as CURRENT_PROXY_REVIEW_BASIS,
    RenderedProxyReview,
    proxy_dimensions,
    render_and_review_proxy,
)


ANALYSIS_BASIS = "strict_failed_complete_candidate_qa_replay_v1"
QA_OUTPUT_MARKER = "qa-replay"
EXPECTED_SELECTION_BASIS = "sequential_decoded_proxy_candidate_rerank_v1"
EXPECTED_REQUEST_INDEX_DOMAIN = "request_source_index"
EXPECTED_ELIGIBLE_INDEX_DOMAIN = "eligible_source_index"
EXPECTED_RENDER_INDEX_DOMAIN = "render_plan_sources_array_index"


class CandidateReplayValidationError(ValueError):
    """The serialized candidate is incomplete, inconsistent, or unbound."""


@dataclass(frozen=True, slots=True)
class VerifiedSource:
    request_source_index: int
    eligible_source_index: int
    path: Path
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_source_index": self.request_source_index,
            "eligible_source_index": self.eligible_source_index,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ReplayRepairContext:
    selection_report_path: Path
    acquisition_manifest_path: Path
    candidate_index: int
    candidate_payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayBoundaryRepairResult:
    """Return type for replay-only pixel repair extension hooks.

    A hook may change crop/editorial/look tracks, but the replay harness rejects
    changes to source clocks, cadence, source choices, transitions, audio, or
    output geometry.  This keeps a repaired replay comparable to the exact old
    candidate.
    """

    plan: RenderPlan
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, RenderPlan):
            raise TypeError("repair hook plan must be a RenderPlan")
        if not isinstance(self.audit, Mapping):
            raise TypeError("repair hook audit must be a mapping")


ReplayBoundaryRepairHook = Callable[
    [RenderPlan, ReplayRepairContext], ReplayBoundaryRepairResult
]


@dataclass(frozen=True, slots=True)
class ReconstructedFailedCandidate:
    plan: RenderPlan
    candidate_index: int
    candidate_payload: Mapping[str, Any]
    recorded_review: Mapping[str, Any]
    verified_sources: tuple[VerifiedSource, ...]
    selection_report_sha256: str
    acquisition_manifest_sha256: str
    reconstruction_checks: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateReplayArtifact:
    video_path: Path
    audit_path: Path
    payload: Mapping[str, Any]


RenderReview = Callable[..., tuple[RenderPlan, RenderedProxyReview]]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateReplayValidationError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise CandidateReplayValidationError(f"{name} must be an array")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CandidateReplayValidationError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateReplayValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateReplayValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise CandidateReplayValidationError(f"{name} must be >= {minimum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CandidateReplayValidationError(f"{name} must be a boolean")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateReplayValidationError(f"{name} must be a non-blank string")
    return value


def _optional_number(value: Any, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _optional_string(value: Any, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _same(
    left: float,
    right: float,
    name: str,
    *,
    tolerance: float = 1e-6,
) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise CandidateReplayValidationError(
            f"{name} mismatch: {left!r} != {right!r}"
        )


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateReplayValidationError(f"could not read {name}: {exc}") from exc
    return _mapping(value, name)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination.resolve()


def _candidate_for(
    report: Mapping[str, Any], candidate_index: int
) -> Mapping[str, Any]:
    candidates = _sequence(report.get("complete_candidates"), "complete_candidates")
    matches = [
        _mapping(item, "complete_candidates[]")
        for item in candidates
        if isinstance(item, Mapping) and item.get("candidate_index") == candidate_index
    ]
    if len(matches) != 1:
        raise CandidateReplayValidationError(
            "selection report must contain exactly one requested complete candidate"
        )
    indices = [
        _integer(_mapping(item, "complete_candidates[]").get("candidate_index"),
                 "complete_candidates[].candidate_index")
        for item in candidates
    ]
    if len(indices) != len(set(indices)):
        raise CandidateReplayValidationError("complete candidate indices are duplicated")
    return matches[0]


def _recorded_review_for(
    report: Mapping[str, Any], candidate_index: int
) -> Mapping[str, Any]:
    candidates = _sequence(report.get("candidates"), "candidates")
    matches = [
        _mapping(item, "candidates[]")
        for item in candidates
        if isinstance(item, Mapping) and item.get("candidate_index") == candidate_index
    ]
    if len(matches) != 1:
        raise CandidateReplayValidationError(
            "selection report must contain exactly one decoded review for the candidate"
        )
    item = matches[0]
    if item.get("error") is not None:
        raise CandidateReplayValidationError("recorded candidate review contains an error")
    return _mapping(item.get("review"), "candidate review")


def _verify_sources(
    report: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> tuple[tuple[Path, ...], tuple[VerifiedSource, ...], tuple[int, ...]]:
    mapping = _mapping(report.get("source_index_mapping"), "source_index_mapping")
    if mapping.get("request_index_domain") != EXPECTED_REQUEST_INDEX_DOMAIN:
        raise CandidateReplayValidationError("unexpected request source-index domain")
    if mapping.get("eligible_index_domain") != EXPECTED_ELIGIBLE_INDEX_DOMAIN:
        raise CandidateReplayValidationError("unexpected eligible source-index domain")
    if mapping.get("render_plan_index_domain") != EXPECTED_RENDER_INDEX_DOMAIN:
        raise CandidateReplayValidationError("unexpected render source-index domain")
    equivalences = _mapping(
        mapping.get("domain_equivalences"),
        "source_index_mapping.domain_equivalences",
    )
    if equivalences.get(EXPECTED_RENDER_INDEX_DOMAIN) != EXPECTED_REQUEST_INDEX_DOMAIN:
        raise CandidateReplayValidationError("render/request source domains are not equivalent")

    eligible_to_request = tuple(
        _integer(value, "eligible_to_request[]")
        for value in _sequence(
            mapping.get("eligible_to_request"), "eligible_to_request"
        )
    )
    request_to_eligible_raw = _sequence(
        mapping.get("request_to_eligible"), "request_to_eligible"
    )
    request_count = len(request_to_eligible_raw)
    if request_count < 1 or len(eligible_to_request) != request_count:
        raise CandidateReplayValidationError(
            "QA replay requires every request source to remain eligible and serialized"
        )
    if set(eligible_to_request) != set(range(request_count)):
        raise CandidateReplayValidationError("eligible_to_request is not a full permutation")
    for request_index, raw_eligible in enumerate(request_to_eligible_raw):
        eligible_index = _integer(raw_eligible, "request_to_eligible[]")
        if eligible_to_request[eligible_index] != request_index:
            raise CandidateReplayValidationError("source-index mapping is not bidirectional")

    eligible_entries = _sequence(
        mapping.get("eligible_sources"), "source_index_mapping.eligible_sources"
    )
    if len(eligible_entries) != request_count:
        raise CandidateReplayValidationError("eligible source entries are incomplete")
    path_by_request: list[Path | None] = [None] * request_count
    eligible_by_request: list[int | None] = [None] * request_count
    for position, raw_entry in enumerate(eligible_entries):
        entry = _mapping(raw_entry, "eligible_sources[]")
        eligible_index = _integer(
            entry.get("eligible_source_index"),
            "eligible_sources[].eligible_source_index",
        )
        request_index = _integer(
            entry.get("request_source_index"),
            "eligible_sources[].request_source_index",
        )
        if eligible_index != position or eligible_to_request[position] != request_index:
            raise CandidateReplayValidationError("eligible source entry order is inconsistent")
        path_by_request[request_index] = Path(
            _string(entry.get("source"), "eligible_sources[].source")
        ).resolve()
        eligible_by_request[request_index] = eligible_index
    if any(value is None for value in path_by_request):
        raise CandidateReplayValidationError("request source paths are incomplete")

    if acquisition.get("content_integrity_contract_version") != 1:
        raise CandidateReplayValidationError("unsupported acquisition integrity contract")
    summary = _mapping(acquisition.get("summary"), "acquisition.summary")
    if summary.get("status") != "complete":
        raise CandidateReplayValidationError("acquisition manifest is not complete")
    selected = _sequence(acquisition.get("selected"), "acquisition.selected")
    if len(selected) != request_count:
        raise CandidateReplayValidationError(
            "acquisition selected sources do not cover the replay source domain"
        )

    verified: list[VerifiedSource] = []
    for request_index, raw_selected in enumerate(selected):
        selected_item = _mapping(raw_selected, "acquisition.selected[]")
        selected_index = _integer(
            selected_item.get("source_index"),
            "acquisition.selected[].source_index",
        )
        if selected_index != request_index:
            raise CandidateReplayValidationError("acquisition source order is inconsistent")
        manifest_path = Path(
            _string(
                selected_item.get("local_path"),
                "acquisition.selected[].local_path",
            )
        ).resolve()
        expected_path = path_by_request[request_index]
        assert expected_path is not None
        if manifest_path != expected_path:
            raise CandidateReplayValidationError(
                "selection and acquisition source paths do not match"
            )
        integrity = _mapping(
            selected_item.get("content_integrity"),
            "acquisition.selected[].content_integrity",
        )
        if integrity.get("algorithm") != "sha256":
            raise CandidateReplayValidationError("source fingerprint must use SHA-256")
        expected_size = _integer(
            integrity.get("size_bytes"),
            "content_integrity.size_bytes",
            minimum=1,
        )
        expected_sha = _string(
            integrity.get("sha256"), "content_integrity.sha256"
        ).lower()
        if len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            raise CandidateReplayValidationError("source SHA-256 is malformed")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        actual_size = manifest_path.stat().st_size
        if actual_size != expected_size:
            raise CandidateReplayValidationError(
                f"source size fingerprint mismatch: {manifest_path}"
            )
        actual_sha = _sha256_path(manifest_path)
        if actual_sha != expected_sha:
            raise CandidateReplayValidationError(
                f"source SHA-256 fingerprint mismatch: {manifest_path}"
            )
        eligible_index = eligible_by_request[request_index]
        assert eligible_index is not None
        verified.append(
            VerifiedSource(
                request_source_index=request_index,
                eligible_source_index=eligible_index,
                path=manifest_path,
                size_bytes=actual_size,
                sha256=actual_sha,
            )
        )
    return (
        tuple(value for value in path_by_request if value is not None),
        tuple(verified),
        eligible_to_request,
    )


def _parse_source_look(value: Any, name: str) -> SourceLookAdjustment | None:
    if value is None:
        return None
    payload = _mapping(value, name)
    result = SourceLookAdjustment(
        brightness=_number(payload.get("brightness"), f"{name}.brightness"),
        contrast=_number(payload.get("contrast"), f"{name}.contrast"),
        saturation=_number(payload.get("saturation"), f"{name}.saturation"),
        temperature_balance=_number(
            payload.get("temperature_balance"), f"{name}.temperature_balance"
        ),
    )
    if _boolean(payload.get("identity"), f"{name}.identity") != result.is_identity:
        raise CandidateReplayValidationError(f"{name} identity flag is inconsistent")
    return result


def _parse_boundary_look(value: Any, name: str) -> BoundaryLookTrack | None:
    if value is None:
        return None
    payload = _mapping(value, name)
    if payload.get("post_global_residual") is not True:
        raise CandidateReplayValidationError(
            f"{name} is not marked as a post-global residual"
        )
    keyframes = tuple(
        BoundaryLookKeyframe(
            time_seconds=_number(item.get("time_seconds"), f"{name}.time_seconds"),
            brightness=_number(item.get("brightness"), f"{name}.brightness"),
            contrast=_number(item.get("contrast"), f"{name}.contrast"),
            saturation=_number(item.get("saturation"), f"{name}.saturation"),
        )
        for raw in _sequence(payload.get("keyframes"), f"{name}.keyframes")
        for item in (_mapping(raw, f"{name}.keyframes[]"),)
    )
    return BoundaryLookTrack(
        keyframes,
        reason=_string(payload.get("reason"), f"{name}.reason"),
    )


def _parse_pan_crop(value: Mapping[str, Any], name: str) -> PanCropTrack | None:
    applied = _boolean(value.get("applied"), f"{name}.applied")
    raw_keyframes = _sequence(value.get("crop_keyframes"), f"{name}.crop_keyframes")
    if applied != bool(raw_keyframes):
        raise CandidateReplayValidationError(
            f"{name} applied flag and crop keyframes disagree"
        )
    if not raw_keyframes:
        return None
    keyframes: list[PanCropKeyframe] = []
    for raw in raw_keyframes:
        item = _mapping(raw, f"{name}.crop_keyframes[]")
        crop_values = _sequence(
            item.get("crop_xywh"), f"{name}.crop_keyframes[].crop_xywh"
        )
        if len(crop_values) != 4:
            raise CandidateReplayValidationError("pan crop needs four values")
        keyframes.append(
            PanCropKeyframe(
                _number(item.get("time_seconds"), f"{name}.time_seconds"),
                NormalizedCrop(
                    *(
                        _number(component, f"{name}.crop_xywh[]")
                        for component in crop_values
                    )
                ),
            )
        )
    return PanCropTrack(tuple(keyframes))


def _parse_editorial_track(
    value: Mapping[str, Any], name: str
) -> EditorialTransformTrack | None:
    applied = _boolean(value.get("applied"), f"{name}.applied")
    raw_keyframes = _sequence(value.get("keyframes"), f"{name}.keyframes")
    if applied != bool(raw_keyframes):
        raise CandidateReplayValidationError(
            f"{name} applied flag and keyframes disagree"
        )
    reasons = tuple(
        _string(item, f"{name}.reasons[]")
        for item in _sequence(value.get("reasons"), f"{name}.reasons")
    )
    if not raw_keyframes:
        # Unapplied entries retain skip diagnostics in ``reasons``.  They are
        # not renderer inputs and therefore do not become a transform track.
        return None
    keyframes = tuple(
        EditorialTransformKeyframe(
            time_seconds=_number(item.get("time_seconds"), f"{name}.time_seconds"),
            zoom=_number(item.get("zoom"), f"{name}.zoom"),
            center_x=_number(item.get("center_x"), f"{name}.center_x"),
            center_y=_number(item.get("center_y"), f"{name}.center_y"),
            roll_degrees=_number(
                item.get("roll_degrees"), f"{name}.roll_degrees"
            ),
        )
        for raw in raw_keyframes
        for item in (_mapping(raw, f"{name}.keyframes[]"),)
    )
    # Older failed-candidate reports serialized every numeric keyframe but one
    # legacy pulse path omitted the track's diagnostic reason.  The reason is
    # not consumed by FFmpeg or any RenderPlan safety invariant, so a visible
    # replay-only sentinel preserves exact pixels without pretending the lost
    # metadata was recovered.
    return EditorialTransformTrack(
        keyframes,
        reason=(
            ";".join(reasons)
            if reasons
            else "qa_replay_serialized_editorial_reason_unavailable"
        ),
    )


def _parse_transition(
    value: Any, name: str, *, fps: float
) -> RenderTransition:
    payload = _mapping(value, name)
    try:
        kind = TransitionKind(_string(payload.get("kind"), f"{name}.kind"))
    except ValueError as exc:
        raise CandidateReplayValidationError(f"{name}.kind is invalid") from exc
    frame_count = _integer(payload.get("frame_count"), f"{name}.frame_count")
    result = RenderTransition(
        kind=kind,
        frame_count=frame_count,
        reason=_string(payload.get("reason"), f"{name}.reason"),
        face_mismatch=(
            None
            if payload.get("face_mismatch") is None
            else _number(payload.get("face_mismatch"), f"{name}.face_mismatch")
        ),
        tier=(
            None
            if payload.get("tier") is None
            else _string(payload.get("tier"), f"{name}.tier")
        ),
        appearance_min_cosine_similarity=(
            None
            if payload.get("appearance_min_cosine_similarity") is None
            else _number(
                payload.get("appearance_min_cosine_similarity"),
                f"{name}.appearance_min_cosine_similarity",
            )
        ),
        future_pose_max_mismatch=(
            None
            if payload.get("future_pose_max_mismatch") is None
            else _number(
                payload.get("future_pose_max_mismatch"),
                f"{name}.future_pose_max_mismatch",
            )
        ),
        future_silhouette_max_mismatch=(
            None
            if payload.get("future_silhouette_max_mismatch") is None
            else _number(
                payload.get("future_silhouette_max_mismatch"),
                f"{name}.future_silhouette_max_mismatch",
            )
        ),
    )
    _same(
        _number(payload.get("duration_seconds"), f"{name}.duration_seconds"),
        result.duration_seconds(fps),
        f"{name}.duration_seconds",
    )
    return result


def _infer_proxy_contract(review: Mapping[str, Any]) -> tuple[AspectRatio, float]:
    proxy = _mapping(review.get("proxy"), "candidate review.proxy")
    width = _integer(proxy.get("width"), "candidate review.proxy.width", minimum=2)
    height = _integer(proxy.get("height"), "candidate review.proxy.height", minimum=2)
    if (width, height) == proxy_dimensions(AspectRatio.LANDSCAPE):
        aspect = AspectRatio.LANDSCAPE
    elif (width, height) == proxy_dimensions(AspectRatio.PORTRAIT):
        aspect = AspectRatio.PORTRAIT
    else:
        raise CandidateReplayValidationError("recorded proxy dimensions are unsupported")
    return (aspect, _number(proxy.get("fps"), "candidate review.proxy.fps", minimum=1.0))


def _reconstruct_segments(
    candidate: Mapping[str, Any],
    *,
    eligible_to_request: Sequence[int],
) -> tuple[tuple[RenderSegment, ...], float]:
    visual = _mapping(candidate.get("visual_timeline"), "candidate.visual_timeline")
    choices = _sequence(visual.get("choices"), "visual_timeline.choices")
    final_segments = _sequence(
        candidate.get("final_render_segments"), "candidate.final_render_segments"
    )
    face_plan = _mapping(candidate.get("face_match"), "candidate.face_match")
    face_segments = _sequence(face_plan.get("segments"), "face_match.segments")
    editorial = _mapping(
        candidate.get("editorial_motion"), "candidate.editorial_motion"
    )
    editorial_segments = _sequence(
        editorial.get("segments"), "editorial_motion.segments"
    )
    count = len(choices)
    if count < 2 or not (
        len(final_segments) == len(face_segments) == len(editorial_segments) == count
    ):
        raise CandidateReplayValidationError(
            "candidate segment payload lengths are incomplete or inconsistent"
        )

    segments: list[RenderSegment] = []
    master_audio_starts: list[float] = []
    for index, raw_values in enumerate(
        zip(choices, final_segments, face_segments, editorial_segments, strict=True)
    ):
        raw_choice, raw_final, raw_face, raw_editorial = raw_values
        choice = _mapping(raw_choice, "visual_timeline.choices[]")
        final = _mapping(raw_final, "final_render_segments[]")
        face = _mapping(raw_face, "face_match.segments[]")
        editorial_item = _mapping(raw_editorial, "editorial_motion.segments[]")
        for payload, name in (
            (choice, "visual choice"),
            (final, "final render segment"),
            (face, "face-match segment"),
            (editorial_item, "editorial segment"),
        ):
            if _integer(payload.get("segment_index"), f"{name}.segment_index") != index:
                raise CandidateReplayValidationError(f"{name} order is inconsistent")

        eligible_index = _integer(choice.get("source_index"), "choice.source_index")
        if eligible_index >= len(eligible_to_request):
            raise CandidateReplayValidationError("choice source index is outside eligible domain")
        request_index = eligible_to_request[eligible_index]
        if _integer(
            final.get("request_source_index"),
            "final_render_segments[].request_source_index",
        ) != request_index:
            raise CandidateReplayValidationError("choice/request source mapping mismatch")
        for payload, name in (
            (face, "face_match"),
            (editorial_item, "editorial_motion"),
        ):
            if _integer(payload.get("source_index"), f"{name}.source_index") != eligible_index:
                raise CandidateReplayValidationError(f"{name} source index mismatch")

        canonical_start = _number(
            choice.get("canonical_master_start"), "choice.canonical_master_start"
        )
        canonical_end = _number(
            choice.get("canonical_master_end"), "choice.canonical_master_end"
        )
        duration = _number(choice.get("duration_seconds"), "choice.duration_seconds")
        if duration <= 0.0:
            raise CandidateReplayValidationError("choice duration must be positive")
        _same(canonical_end - canonical_start, duration, "choice canonical duration")
        output_start = _number(
            final.get("output_start_seconds"), "final.output_start_seconds"
        )
        output_end = _number(
            final.get("output_end_seconds"), "final.output_end_seconds"
        )
        _same(output_end - output_start, duration, "final output duration")
        _same(
            _number(final.get("canonical_master_start"), "final.canonical_master_start"),
            canonical_start,
            "canonical master start",
        )
        _same(
            _number(final.get("canonical_master_end"), "final.canonical_master_end"),
            canonical_end,
            "canonical master end",
        )
        expected_output_start = 0.0 if not segments else (
            segments[-1].master_start + segments[-1].duration
        )
        _same(output_start, expected_output_start, "contiguous output start")
        master_audio_starts.append(canonical_start - output_start)

        source_start = _number(
            final.get("source_local_start"), "final.source_local_start"
        )
        source_end = _number(final.get("source_local_end"), "final.source_local_end")
        source_duration = _number(
            final.get("source_duration_seconds"), "final.source_duration_seconds"
        )
        if source_duration <= 0.0:
            raise CandidateReplayValidationError("source duration must be positive")
        _same(source_end - source_start, source_duration, "source-local duration")
        for field, expected in (
            ("source_local_start", source_start),
            ("source_local_end", source_end),
            ("source_duration_seconds", source_duration),
        ):
            _same(
                _number(choice.get(field), f"choice.{field}"),
                expected,
                f"choice/final {field}",
            )
        rate = _number(final.get("source_playback_rate"), "final.source_playback_rate")
        _same(rate, source_duration / duration, "source playback rate", tolerance=1e-8)
        _same(
            _number(choice.get("source_playback_rate"), "choice.source_playback_rate"),
            rate,
            "choice/final source playback rate",
            tolerance=1e-8,
        )
        clock_basis = _string(final.get("source_clock_basis"), "source_clock_basis")
        if clock_basis == "local_monotonic_timewarp":
            render_source_duration: float | None = source_duration
        elif clock_basis == "global_scalar_offset":
            _same(source_duration, duration, "global scalar source duration")
            render_source_duration = None
        else:
            raise CandidateReplayValidationError("source clock basis is unsupported")

        pan_crop = _parse_pan_crop(face, f"face_match.segments[{index}]")
        editorial_track = _parse_editorial_track(
            editorial_item, f"editorial_motion.segments[{index}]"
        )
        source_look = _parse_source_look(
            final.get("source_look_adjustment"),
            f"final_render_segments[{index}].source_look_adjustment",
        )
        boundary_look = _parse_boundary_look(
            final.get("boundary_look_track"),
            f"final_render_segments[{index}].boundary_look_track",
        )
        expected_crop_mode = (
            "dynamic_pan_crop_plus_editorial_camera"
            if pan_crop is not None and editorial_track is not None
            else "editorial_camera"
            if editorial_track is not None
            else "dynamic_pan_crop"
            if pan_crop is not None
            else "full_frame"
        )
        if final.get("crop_mode") != expected_crop_mode:
            raise CandidateReplayValidationError(
                "serialized crop mode cannot be reconstructed exactly"
            )
        if _integer(
            final.get("editorial_transform_keyframe_count"),
            "editorial_transform_keyframe_count",
        ) != (0 if editorial_track is None else len(editorial_track.keyframes)):
            raise CandidateReplayValidationError("editorial keyframe count mismatch")

        segments.append(
            RenderSegment(
                source_index=request_index,
                source_start=source_start,
                duration=duration,
                master_start=output_start,
                pan_crop=pan_crop,
                editorial_transform=editorial_track,
                source_duration=render_source_duration,
                look_adjustment=source_look,
                boundary_look_track=boundary_look,
            )
        )

    master_audio_start = master_audio_starts[0]
    for value in master_audio_starts[1:]:
        _same(value, master_audio_start, "common master audio start")
    if master_audio_start < 0.0:
        raise CandidateReplayValidationError("master audio start is negative")

    face_applied = sum(item.pan_crop is not None for item in segments)
    if _integer(face_plan.get("applied_count"), "face_match.applied_count") != face_applied:
        raise CandidateReplayValidationError("face-match applied count mismatch")
    editorial_applied = sum(item.editorial_transform is not None for item in segments)
    if _integer(
        editorial.get("applied_segment_count"),
        "editorial_motion.applied_segment_count",
    ) != editorial_applied:
        raise CandidateReplayValidationError("editorial applied count mismatch")
    return (tuple(segments), master_audio_start)


def _reconstruct_transitions(
    candidate: Mapping[str, Any], *, fps: float, segment_count: int
) -> tuple[RenderTransition, ...]:
    payload = _mapping(candidate.get("group_transitions"), "group_transitions")
    raw_transitions = _sequence(payload.get("transitions"), "group_transitions.transitions")
    if len(raw_transitions) != segment_count - 1:
        raise CandidateReplayValidationError("transition count does not match segments")
    transitions = tuple(
        _parse_transition(value, f"group_transitions.transitions[{index}]", fps=fps)
        for index, value in enumerate(raw_transitions)
    )
    dissolve_count = sum(item.kind is TransitionKind.FACE_DISSOLVE for item in transitions)
    same_source_hold_count = sum(item.tier == "D" for item in transitions)
    if _integer(payload.get("dissolve_count"), "group_transitions.dissolve_count") != dissolve_count:
        raise CandidateReplayValidationError("serialized dissolve count mismatch")
    if _integer(
        payload.get("same_source_hold_count"),
        "group_transitions.same_source_hold_count",
    ) != same_source_hold_count:
        raise CandidateReplayValidationError("serialized same-source hold count mismatch")
    if _integer(
        payload.get("hard_cut_boundary_count"),
        "group_transitions.hard_cut_boundary_count",
    ) != sum(item.kind is TransitionKind.HARD_CUT for item in transitions):
        raise CandidateReplayValidationError("serialized hard-cut count mismatch")
    return transitions


def _validate_recorded_review(plan: RenderPlan, review: Mapping[str, Any]) -> dict[str, Any]:
    proxy = _mapping(review.get("proxy"), "candidate review.proxy")
    counts = segment_frame_counts(plan)
    expected_frame_count = sum(counts)
    if _integer(
        proxy.get("decoded_frame_count"), "candidate review.proxy.decoded_frame_count"
    ) != expected_frame_count:
        raise CandidateReplayValidationError("recorded proxy frame count does not match plan")
    _same(
        _number(proxy.get("duration_seconds"), "candidate review.proxy.duration_seconds"),
        plan.render_duration,
        "recorded proxy duration",
        tolerance=1e-5,
    )
    _same(
        _number(proxy.get("fps"), "candidate review.proxy.fps"),
        plan.fps,
        "recorded proxy fps",
    )
    if (
        _integer(proxy.get("width"), "candidate review.proxy.width"),
        _integer(proxy.get("height"), "candidate review.proxy.height"),
    ) != proxy_dimensions(plan.aspect_ratio):
        raise CandidateReplayValidationError("recorded proxy dimensions do not match plan")

    raw_boundaries = _sequence(review.get("boundaries"), "candidate review.boundaries")
    boundary_indices: list[int] = []
    cumulative = 0
    for raw in raw_boundaries:
        item = _mapping(raw, "candidate review.boundaries[]")
        boundary_index = _integer(item.get("boundary_index"), "review boundary index")
        if boundary_index >= len(plan.transitions):
            raise CandidateReplayValidationError("review boundary index is outside plan")
        boundary_indices.append(boundary_index)
        cumulative = sum(counts[: boundary_index + 1])
        transition = plan.transitions[boundary_index]
        if _integer(item.get("cut_frame_index"), "review cut frame index") != cumulative:
            raise CandidateReplayValidationError("review cut-frame index mismatch")
        if _integer(
            item.get("outgoing_source_index"), "review outgoing source index"
        ) != plan.segments[boundary_index].source_index:
            raise CandidateReplayValidationError("review outgoing source mismatch")
        if _integer(
            item.get("incoming_source_index"), "review incoming source index"
        ) != plan.segments[boundary_index + 1].source_index:
            raise CandidateReplayValidationError("review incoming source mismatch")
        if item.get("transition_kind") != transition.kind.value:
            raise CandidateReplayValidationError("review transition kind mismatch")
        if _integer(
            item.get("transition_frame_count"), "review transition frame count"
        ) != transition.frame_count:
            raise CandidateReplayValidationError("review transition frame count mismatch")
    if len(boundary_indices) != len(set(boundary_indices)):
        raise CandidateReplayValidationError("recorded review boundary indices are duplicated")
    if review.get("analysis") == "decoded_output_proxy_all_transition_quality_review_v2":
        if boundary_indices != list(range(len(plan.transitions))):
            raise CandidateReplayValidationError(
                "all-transition recorded review does not cover every plan boundary"
            )
    return {
        "segment_count": len(plan.segments),
        "transition_count": len(plan.transitions),
        "recorded_reviewed_boundary_count": len(boundary_indices),
        "output_frame_count": expected_frame_count,
        "duration_seconds": plan.render_duration,
        "fps": plan.fps,
        "source_sequence": [item.source_index for item in plan.segments],
        "all_serialized_time_source_boundary_invariants_match": True,
    }


def reconstruct_failed_candidate(
    selection_report_path: Path,
    acquisition_manifest_path: Path,
    *,
    candidate_index: int,
    reference_source: int,
    output: Path,
) -> ReconstructedFailedCandidate:
    """Reconstruct one exact old candidate and validate every binding."""

    candidate_index = _integer(candidate_index, "candidate_index")
    reference_source = _integer(reference_source, "reference_source")
    selection_path = Path(selection_report_path).resolve()
    acquisition_path = Path(acquisition_manifest_path).resolve()
    report = _load_json(selection_path, "selection report")
    acquisition = _load_json(acquisition_path, "acquisition manifest")
    if report.get("analysis_basis") != EXPECTED_SELECTION_BASIS:
        raise CandidateReplayValidationError("unsupported candidate-selection basis")
    if report.get("passed") is not False:
        raise CandidateReplayValidationError(
            "QA replay accepts only a fail-closed selection report"
        )
    if report.get("minimum_candidate_count_satisfied") is not True:
        raise CandidateReplayValidationError(
            "selection report did not complete its minimum candidate set"
        )
    complete_candidates = _sequence(
        report.get("complete_candidates"), "complete_candidates"
    )
    exact_count_fields = {
        "requested_review_count": 8,
        "reviewed_candidate_count": 8,
        "minimum_required_candidate_count": 8,
        "successful_review_count": 8,
        "complete_candidate_count": 8,
    }
    for field, expected in exact_count_fields.items():
        if _integer(report.get(field), field) != expected:
            raise CandidateReplayValidationError(
                f"failed selection report must preserve the exact-eight {field} contract"
            )
    if len(complete_candidates) != 8:
        raise CandidateReplayValidationError(
            "failed selection report must serialize exactly eight complete candidates"
        )
    if report.get("exact_complete_candidate_count_satisfied") is not True:
        raise CandidateReplayValidationError(
            "exact complete candidate count was not satisfied"
        )
    if report.get("exact_successful_decoded_review_count_satisfied") is not True:
        raise CandidateReplayValidationError(
            "exact decoded-review count was not satisfied"
        )
    candidate = _candidate_for(report, candidate_index)
    review = _recorded_review_for(report, candidate_index)
    aspect, fps = _infer_proxy_contract(review)
    sources, verified, eligible_to_request = _verify_sources(report, acquisition)
    if reference_source >= len(sources):
        raise CandidateReplayValidationError("reference source is outside source domain")
    segments, master_audio_start = _reconstruct_segments(
        candidate,
        eligible_to_request=eligible_to_request,
    )
    transitions = _reconstruct_transitions(
        candidate,
        fps=fps,
        segment_count=len(segments),
    )
    width, height = output_size(aspect)
    plan = RenderPlan(
        sources=sources,
        segments=segments,
        reference_source=reference_source,
        aspect_ratio=aspect,
        output=Path(output).resolve(),
        master_audio_start=master_audio_start,
        width=width,
        height=height,
        fps=fps,
        transitions=transitions,
    )
    checks = _validate_recorded_review(plan, review)
    return ReconstructedFailedCandidate(
        plan=plan,
        candidate_index=candidate_index,
        candidate_payload=candidate,
        recorded_review=review,
        verified_sources=verified,
        selection_report_sha256=_sha256_path(selection_path),
        acquisition_manifest_sha256=_sha256_path(acquisition_path),
        reconstruction_checks=checks,
    )


def _finite_vector(value: Any, name: str) -> tuple[float, ...]:
    return tuple(
        _number(item, f"{name}[]") for item in _sequence(value, name)
    )


def _fixed_vector(
    value: Any,
    name: str,
    *,
    length: int,
    integers: bool = False,
) -> tuple[Any, ...]:
    raw = _sequence(value, name)
    if len(raw) != length:
        raise CandidateReplayValidationError(f"{name} must contain {length} items")
    return tuple(
        _integer(item, f"{name}[]")
        if integers
        else _number(item, f"{name}[]")
        for item in raw
    )


def _optional_box(value: Any, name: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    raw = _sequence(value, name)
    if len(raw) != 4:
        raise CandidateReplayValidationError(f"{name} needs four values")
    x, y, width, height = tuple(_number(item, f"{name}[]") for item in raw)
    if not (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
        and x + width <= 1.000001
        and y + height <= 1.000001
    ):
        raise CandidateReplayValidationError(f"{name} is outside normalized bounds")
    return (x, y, width, height)


def _optional_point(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    raw = _sequence(value, name)
    if len(raw) != 2:
        raise CandidateReplayValidationError(f"{name} needs two values")
    point = tuple(_number(item, f"{name}[]") for item in raw)
    if any(not 0.0 <= item <= 1.0 for item in point):
        raise CandidateReplayValidationError(f"{name} is outside normalized bounds")
    return (point[0], point[1])


def _three_optional_boxes(value: Any, name: str) -> tuple[Any, Any, Any]:
    raw = _sequence(value, name)
    if len(raw) != 3:
        raise CandidateReplayValidationError(f"{name} must contain three samples")
    return tuple(
        _optional_box(item, f"{name}[{index}]") for index, item in enumerate(raw)
    )  # type: ignore[return-value]


def _three_optional_points(value: Any, name: str) -> tuple[Any, Any, Any]:
    raw = _sequence(value, name)
    if len(raw) != 3:
        raise CandidateReplayValidationError(f"{name} must contain three samples")
    return tuple(
        _optional_point(item, f"{name}[{index}]") for index, item in enumerate(raw)
    )  # type: ignore[return-value]


def _three_optional_numbers(value: Any, name: str) -> tuple[Any, Any, Any]:
    raw = _sequence(value, name)
    if len(raw) != 3:
        raise CandidateReplayValidationError(f"{name} must contain three samples")
    return tuple(
        _optional_number(item, f"{name}[{index}]")
        for index, item in enumerate(raw)
    )  # type: ignore[return-value]


def _three_strings(value: Any, name: str) -> tuple[str, str, str]:
    raw = _sequence(value, name)
    if len(raw) != 3:
        raise CandidateReplayValidationError(f"{name} must contain three samples")
    return tuple(_string(item, f"{name}[]") for item in raw)  # type: ignore[return-value]


def _geometry_window(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        _finite_vector(item, f"{name}[]")
        for item in _sequence(value, name)
    )


def _box_window(
    value: Any, name: str
) -> tuple[tuple[float, float, float, float], ...]:
    output: list[tuple[float, float, float, float]] = []
    for index, item in enumerate(_sequence(value, name)):
        box = _optional_box(item, f"{name}[{index}]")
        if box is None:
            raise CandidateReplayValidationError(f"{name} cannot contain null boxes")
        output.append(box)
    return tuple(output)


def _boundary_stability(value: Any, name: str) -> Any | None:
    if value is None:
        return None
    from .timeline import BoundarySideStability

    payload = _mapping(value, name)
    result = BoundarySideStability(
        sample_span_seconds=_number(
            payload.get("sample_span_seconds"), f"{name}.sample_span_seconds"
        ),
        scale_delta=_number(payload.get("scale_delta"), f"{name}.scale_delta"),
        structure_mismatch=_optional_number(
            payload.get("structure_mismatch"), f"{name}.structure_mismatch"
        ),
        pose_mismatch=_optional_number(
            payload.get("pose_mismatch"), f"{name}.pose_mismatch"
        ),
        shot_type_mismatch=_number(
            payload.get("shot_type_mismatch"), f"{name}.shot_type_mismatch"
        ),
    )
    available = tuple(
        _string(item, f"{name}.available_metrics[]")
        for item in _sequence(
            payload.get("available_metrics"), f"{name}.available_metrics"
        )
    )
    if available != result.available_metrics:
        raise CandidateReplayValidationError(f"{name} available metrics drifted")
    return result


def _shot_type(value: Any, name: str) -> Any:
    from .timeline import ShotType

    try:
        return ShotType(_string(value, name))
    except ValueError as exc:
        raise CandidateReplayValidationError(f"{name} is invalid") from exc


def _optional_shot_type(value: Any, name: str) -> Any | None:
    return None if value is None else _shot_type(value, name)


def _deserialize_visual_timeline(
    candidate: Mapping[str, Any],
) -> Any:
    """Strictly recover the non-biometric timeline fields needed by repair."""

    from .visual_score import VisualTimelineChoice, VisualTimelinePlan

    payload = _mapping(candidate.get("visual_timeline"), "candidate.visual_timeline")
    raw_choices = _sequence(payload.get("choices"), "visual_timeline.choices")
    choices: list[VisualTimelineChoice] = []
    for position, raw in enumerate(raw_choices):
        item = _mapping(raw, f"visual_timeline.choices[{position}]")
        if _integer(item.get("segment_index"), "choice.segment_index") != position:
            raise CandidateReplayValidationError("visual choice order is inconsistent")
        landmark_samples_raw = _sequence(
            item.get("face_landmark_samples_xy"), "choice.face_landmark_samples_xy"
        )
        if len(landmark_samples_raw) != 3:
            raise CandidateReplayValidationError(
                "face landmark samples must contain three entries"
            )
        landmark_samples: list[tuple[tuple[float, float], ...] | None] = []
        for sample_index, raw_sample in enumerate(landmark_samples_raw):
            if raw_sample is None:
                landmark_samples.append(None)
                continue
            points: list[tuple[float, float]] = []
            for point_index, raw_point in enumerate(
                _sequence(raw_sample, "face landmark sample")
            ):
                point = _optional_point(
                    raw_point,
                    f"face_landmark_samples_xy[{sample_index}][{point_index}]",
                )
                if point is None:
                    raise CandidateReplayValidationError(
                        "face landmark sample cannot contain null points"
                    )
                points.append(point)
            landmark_samples.append(tuple(points))

        entry_evidence = _mapping(
            item.get("dissolve_entry_evidence"), "choice.dissolve_entry_evidence"
        )
        exit_evidence = _mapping(
            item.get("dissolve_exit_evidence"), "choice.dissolve_exit_evidence"
        )
        for evidence, evidence_name in (
            (entry_evidence, "dissolve_entry_evidence"),
            (exit_evidence, "dissolve_exit_evidence"),
        ):
            if evidence.get("raw_embeddings_persisted") is not False:
                raise CandidateReplayValidationError(
                    f"{evidence_name} violates the no-raw-embedding contract"
                )
            # The manifest intentionally stores only the transient embedding
            # count, never SFace vectors.  Current replay geometry consumers
            # (face crop, exact-frame subject boxes, affine, editorial motion)
            # do not read those identity windows, and final transitions are
            # reused rather than rebuilt.  Validate the count type while
            # leaving the unavailable vectors empty.
            _integer(
                evidence.get("appearance_embedding_count"),
                f"{evidence_name}.appearance_embedding_count",
            )

        choice = VisualTimelineChoice(
            segment_index=position,
            source_index=_integer(item.get("source_index"), "choice.source_index"),
            canonical_master_start=_number(
                item.get("canonical_master_start"), "choice.canonical_master_start"
            ),
            canonical_master_end=_number(
                item.get("canonical_master_end"), "choice.canonical_master_end"
            ),
            source_local_start=_number(
                item.get("source_local_start"), "choice.source_local_start"
            ),
            source_local_end=_number(
                item.get("source_local_end"), "choice.source_local_end"
            ),
            quality_score=_number(item.get("quality_score"), "choice.quality_score"),
            composition_score=_number(
                item.get("composition_score"), "choice.composition_score"
            ),
            boundary_score=_number(
                item.get("boundary_score"), "choice.boundary_score"
            ),
            shot_scale=_number(item.get("shot_scale"), "choice.shot_scale"),
            shot_type=_shot_type(item.get("shot_type"), "choice.shot_type"),
            shot_id=_string(item.get("shot_id"), "choice.shot_id"),
            entry_shot_scale=_optional_number(
                item.get("entry_shot_scale"), "choice.entry_shot_scale"
            ),
            exit_shot_scale=_optional_number(
                item.get("exit_shot_scale"), "choice.exit_shot_scale"
            ),
            entry_shot_type=_optional_shot_type(
                item.get("entry_shot_type"), "choice.entry_shot_type"
            ),
            exit_shot_type=_optional_shot_type(
                item.get("exit_shot_type"), "choice.exit_shot_type"
            ),
            entry_scene_cut_near=_boolean(
                item.get("entry_scene_cut_near"), "choice.entry_scene_cut_near"
            ),
            exit_scene_cut_near=_boolean(
                item.get("exit_scene_cut_near"), "choice.exit_scene_cut_near"
            ),
            internal_scene_cut_master_seconds=_finite_vector(
                item.get("internal_scene_cut_master_seconds"),
                "choice.internal_scene_cut_master_seconds",
            ),
            entry_boundary_side_stability=_boundary_stability(
                item.get("entry_boundary_side_stability"),
                "choice.entry_boundary_side_stability",
            ),
            exit_boundary_side_stability=_boundary_stability(
                item.get("exit_boundary_side_stability"),
                "choice.exit_boundary_side_stability",
            ),
            face_samples_xywh=_three_optional_boxes(
                item.get("face_samples_xywh"), "choice.face_samples_xywh"
            ),
            face_match_candidate_failure=_optional_string(
                item.get("face_match_candidate_failure"),
                "choice.face_match_candidate_failure",
            ),
            face_detector_backends=_three_strings(
                item.get("face_detector_backends"), "choice.face_detector_backends"
            ),
            face_landmark_samples_xy=tuple(landmark_samples),  # type: ignore[arg-type]
            face_head_roll_samples_radians=_three_optional_numbers(
                item.get("face_head_roll_samples_radians"),
                "choice.face_head_roll_samples_radians",
            ),
            face_eye_scale_samples=_three_optional_numbers(
                item.get("face_eye_scale_samples"), "choice.face_eye_scale_samples"
            ),
            entry_face_geometry_window=_geometry_window(
                item.get("entry_face_geometry_window"),
                "choice.entry_face_geometry_window",
            ),
            exit_face_geometry_window=_geometry_window(
                item.get("exit_face_geometry_window"),
                "choice.exit_face_geometry_window",
            ),
            dissolve_entry_face_geometry_window=_geometry_window(
                item.get("dissolve_entry_face_geometry_window"),
                "choice.dissolve_entry_face_geometry_window",
            ),
            dissolve_exit_face_geometry_window=_geometry_window(
                item.get("dissolve_exit_face_geometry_window"),
                "choice.dissolve_exit_face_geometry_window",
            ),
            dissolve_entry_face_box_window=_box_window(
                item.get("dissolve_entry_face_box_window"),
                "choice.dissolve_entry_face_box_window",
            ),
            dissolve_exit_face_box_window=_box_window(
                item.get("dissolve_exit_face_box_window"),
                "choice.dissolve_exit_face_box_window",
            ),
            dissolve_entry_pose_backend_window=tuple(
                _string(value, "dissolve entry pose backend")
                for value in _sequence(
                    entry_evidence.get("pose_backends"),
                    "dissolve_entry_evidence.pose_backends",
                )
            ),
            dissolve_exit_pose_backend_window=tuple(
                _string(value, "dissolve exit pose backend")
                for value in _sequence(
                    exit_evidence.get("pose_backends"),
                    "dissolve_exit_evidence.pose_backends",
                )
            ),
            dissolve_entry_pose_confidence_window=_finite_vector(
                entry_evidence.get("pose_confidences"),
                "dissolve_entry_evidence.pose_confidences",
            ),
            dissolve_exit_pose_confidence_window=_finite_vector(
                exit_evidence.get("pose_confidences"),
                "dissolve_exit_evidence.pose_confidences",
            ),
            dissolve_entry_silhouette_backend_window=tuple(
                _string(value, "dissolve entry silhouette backend")
                for value in _sequence(
                    entry_evidence.get("silhouette_backends"),
                    "dissolve_entry_evidence.silhouette_backends",
                )
            ),
            dissolve_exit_silhouette_backend_window=tuple(
                _string(value, "dissolve exit silhouette backend")
                for value in _sequence(
                    exit_evidence.get("silhouette_backends"),
                    "dissolve_exit_evidence.silhouette_backends",
                )
            ),
            dissolve_entry_silhouette_confidence_window=_finite_vector(
                entry_evidence.get("silhouette_confidences"),
                "dissolve_entry_evidence.silhouette_confidences",
            ),
            dissolve_exit_silhouette_confidence_window=_finite_vector(
                exit_evidence.get("silhouette_confidences"),
                "dissolve_exit_evidence.silhouette_confidences",
            ),
            face_transition_minimax_mismatch_from_previous=_optional_number(
                item.get("face_transition_minimax_mismatch_from_previous"),
                "choice.face_transition_minimax_mismatch_from_previous",
            ),
            pose_backends=_three_strings(
                item.get("pose_backends"), "choice.pose_backends"
            ),
            pose_confidences=_fixed_vector(
                item.get("pose_confidences"), "choice.pose_confidences", length=3
            ),  # type: ignore[arg-type]
            detected_person_counts=_fixed_vector(
                item.get("detected_person_counts"),
                "choice.detected_person_counts",
                length=3,
                integers=True,
            ),  # type: ignore[arg-type]
            formation_person_counts=_fixed_vector(
                item.get("formation_person_counts"),
                "choice.formation_person_counts",
                length=3,
                integers=True,
            ),  # type: ignore[arg-type]
            formation_upper_body_counts=_fixed_vector(
                item.get("formation_upper_body_counts"),
                "choice.formation_upper_body_counts",
                length=3,
                integers=True,
            ),  # type: ignore[arg-type]
            shot_scale_bases=_three_strings(
                item.get("shot_scale_bases", ["legacy", "legacy", "legacy"]),
                "choice.shot_scale_bases",
            ),
            silhouette_samples_xywh=_three_optional_boxes(
                item.get("silhouette_samples_xywh"),
                "choice.silhouette_samples_xywh",
            ),
            silhouette_centroid_samples_xy=_three_optional_points(
                item.get("silhouette_centroid_samples_xy"),
                "choice.silhouette_centroid_samples_xy",
            ),
            silhouette_backends=_three_strings(
                item.get("silhouette_backends"), "choice.silhouette_backends"
            ),
            silhouette_confidences=_fixed_vector(
                item.get("silhouette_confidences"),
                "choice.silhouette_confidences",
                length=3,
            ),  # type: ignore[arg-type]
            layout_transition_mismatch_from_previous=_optional_number(
                item.get("layout_transition_mismatch_from_previous"),
                "choice.layout_transition_mismatch_from_previous",
            ),
            pose_transition_basis_from_previous=_optional_string(
                item.get("pose_transition_basis_from_previous"),
                "choice.pose_transition_basis_from_previous",
            ),
            pose_transition_mismatch_from_previous=_optional_number(
                item.get("pose_transition_mismatch_from_previous"),
                "choice.pose_transition_mismatch_from_previous",
            ),
            silhouette_transition_mismatch_from_previous=_optional_number(
                item.get("silhouette_transition_mismatch_from_previous"),
                "choice.silhouette_transition_mismatch_from_previous",
            ),
            formation_transition_mismatch_from_previous=_optional_number(
                item.get("formation_transition_mismatch_from_previous"),
                "choice.formation_transition_mismatch_from_previous",
            ),
        )
        _same(
            _number(item.get("duration_seconds"), "choice.duration_seconds"),
            choice.duration_seconds,
            "visual choice duration",
        )
        _same(
            _number(
                item.get("source_duration_seconds"),
                "choice.source_duration_seconds",
            ),
            choice.source_duration_seconds,
            "visual choice source duration",
        )
        _same(
            _number(item.get("source_playback_rate"), "choice.source_playback_rate"),
            choice.source_playback_rate,
            "visual choice playback rate",
            tolerance=1e-8,
        )
        choices.append(choice)

    summaries = tuple(
        dict(_mapping(item, "visual_timeline.candidate_review_summaries[]"))
        for item in _sequence(
            payload.get("candidate_review_summaries"),
            "visual_timeline.candidate_review_summaries",
        )
    )
    candidate_count = _integer(
        payload.get("candidate_count"), "visual_timeline.candidate_count", minimum=1
    )
    if candidate_count != (len(summaries) or 1):
        raise CandidateReplayValidationError("visual timeline candidate count drifted")
    return VisualTimelinePlan(
        choices=tuple(choices),
        optimizer_score=_number(
            payload.get("optimizer_score"), "visual_timeline.optimizer_score"
        ),
        analysis_basis=_string(
            payload.get("analysis_basis"), "visual_timeline.analysis_basis"
        ),
        selected_candidate_index=_integer(
            payload.get("selected_candidate_index"),
            "visual_timeline.selected_candidate_index",
        ),
        candidate_generation_basis=_string(
            payload.get("candidate_generation_basis"),
            "visual_timeline.candidate_generation_basis",
        ),
        candidate_review_summaries=summaries,
        source_clock_basis=_string(
            payload.get("source_clock_basis"), "visual_timeline.source_clock_basis"
        ),
    )


def _geometry_source_contract(
    context: ReplayRepairContext,
    plan: RenderPlan,
) -> tuple[tuple[Path, ...], tuple[float, ...], tuple[int, ...]]:
    report = _load_json(context.selection_report_path, "selection report")
    acquisition = _load_json(context.acquisition_manifest_path, "acquisition manifest")
    mapping = _mapping(report.get("source_index_mapping"), "source_index_mapping")
    eligible_to_request = tuple(
        _integer(item, "eligible_to_request[]")
        for item in _sequence(mapping.get("eligible_to_request"), "eligible_to_request")
    )
    selected = _sequence(acquisition.get("selected"), "acquisition.selected")
    aspects_by_request: list[float] = []
    for request_index, raw in enumerate(selected):
        item = _mapping(raw, "acquisition.selected[]")
        if _integer(item.get("source_index"), "selected.source_index") != request_index:
            raise CandidateReplayValidationError("acquisition source order drifted")
        media = _mapping(item.get("media_validation"), "selected.media_validation")
        if media.get("passed") is not True:
            raise CandidateReplayValidationError("source media validation did not pass")
        width = _integer(media.get("width"), "media width", minimum=2)
        height = _integer(media.get("height"), "media height", minimum=2)
        aspects_by_request.append(width / height)
    if len(aspects_by_request) != len(plan.sources):
        raise CandidateReplayValidationError("source aspect domain does not match plan")
    eligible_sources = tuple(plan.sources[index] for index in eligible_to_request)
    eligible_aspects = tuple(aspects_by_request[index] for index in eligible_to_request)
    return (eligible_sources, eligible_aspects, eligible_to_request)


def _same_tier_d_run_boundary(
    visual_timeline: Any,
    transitions: Sequence[RenderTransition],
    boundary_index: int,
) -> bool:
    left = visual_timeline.choices[boundary_index]
    right = visual_timeline.choices[boundary_index + 1]
    return bool(
        transitions[boundary_index].tier == "D"
        and left.source_index == right.source_index
        and (
            (left.shot_id and left.shot_id == right.shot_id)
            or (not left.shot_id and not right.shot_id)
        )
    )


def _visual_runs(
    visual_timeline: Any,
    transitions: Sequence[RenderTransition],
) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    first = 0
    for index in range(len(transitions)):
        if _same_tier_d_run_boundary(visual_timeline, transitions, index):
            continue
        runs.append((first, index))
        first = index + 1
    runs.append((first, len(visual_timeline.choices) - 1))
    return tuple(runs)


def _current_boundary_geometry_hook(
    plan: RenderPlan,
    context: ReplayRepairContext,
) -> ReplayBoundaryRepairResult:
    """Re-measure exact cut-frame geometry and overlay only its local ramps."""

    from .boundary_affine_plan import build_boundary_affine_plan
    from .boundary_subject_alignment import build_boundary_subject_alignment_plan
    from .editorial_motion import build_editorial_motion_plan
    from .face_match import build_face_match_plan

    if plan.aspect_ratio is not AspectRatio.LANDSCAPE:
        return ReplayBoundaryRepairResult(
            plan,
            {
                "hook": "current_boundary_geometry",
                "enabled": False,
                "blocked_reason": "non_landscape_replay",
            },
        )
    visual = _deserialize_visual_timeline(context.candidate_payload)
    if len(visual.choices) != len(plan.segments):
        raise CandidateReplayValidationError("visual/render segment count mismatch")
    eligible_sources, eligible_aspects, eligible_to_request = _geometry_source_contract(
        context,
        plan,
    )
    for position, (choice, segment) in enumerate(
        zip(visual.choices, plan.segments, strict=True)
    ):
        if choice.source_index >= len(eligible_to_request):
            raise CandidateReplayValidationError("visual source is outside eligible domain")
        if eligible_to_request[choice.source_index] != segment.source_index:
            raise CandidateReplayValidationError(
                f"visual/render source mismatch at segment {position}"
            )
        _same(choice.source_local_start, segment.source_start, "geometry source start")
        _same(choice.source_local_end, segment.source_end, "geometry source end")
        _same(choice.duration_seconds, segment.duration, "geometry duration")

    face_plan = build_face_match_plan(
        visual_timeline=visual,
        source_aspect_ratios=eligible_aspects,
    )
    affine_plan = build_boundary_affine_plan(
        visual_timeline=visual,
        sources=eligible_sources,
        transitions=plan.transitions,
        face_match_plan=face_plan,
        fps=plan.fps,
        output_width=proxy_dimensions(plan.aspect_ratio)[0],
        output_height=proxy_dimensions(plan.aspect_ratio)[1],
    )
    boundary_subject_alignment = None
    boundary_subject_alignment_error: str | None = None
    try:
        boundary_subject_alignment = build_boundary_subject_alignment_plan(
            visual_timeline=visual,
            sources=eligible_sources,
            transitions=plan.transitions,
            face_match_plan=face_plan,
            fps=plan.fps,
            output_width=plan.width,
            output_height=plan.height,
        )
    except (OSError, TypeError, ValueError, cv2.error) as exc:
        # Replay remains useful for old/partially serialized fixtures, but the
        # audit must make it explicit that no dense local proof was consumed.
        boundary_subject_alignment_error = str(exc)
    editorial_plan = build_editorial_motion_plan(
        visual_timeline=visual,
        transitions=plan.transitions,
        face_match_plan=face_plan,
        boundary_affine_plan=affine_plan,
        boundary_subject_alignment_plan=boundary_subject_alignment,
        phrase_times=(),
        section_plan=None,
        fps=plan.fps,
    )

    allowed_reasons = {
        "exact_frame_face_scale_position_matched_hard_cut",
        "exact_frame_silhouette_scale_position_matched_hard_cut",
        "orb_affine_subject_matched_hard_cut",
        "boundary_local_exact_face_alignment",
        "boundary_local_exact_silhouette_alignment",
    }
    boundary_reasons: dict[int, tuple[str, ...]] = {}
    for boundary_index in range(len(plan.transitions)):
        # The planner attaches one boundary reason to both sides of the real
        # edit.  Intersection avoids mislabeling the neighboring Tier-D split
        # merely because one of its segments carries the adjacent cut ramp.
        reasons = set(editorial_plan.segments[boundary_index].reasons)
        reasons.intersection_update(
            editorial_plan.segments[boundary_index + 1].reasons
        )
        selected = tuple(sorted(reasons & allowed_reasons))
        if selected:
            boundary_reasons[boundary_index] = selected

    runs = _visual_runs(visual, plan.transitions)
    run_for_segment = {
        segment_index: (first, last)
        for first, last in runs
        for segment_index in range(first, last + 1)
    }
    candidate_runs = {
        run_for_segment[segment]
        for boundary_index in boundary_reasons
        for segment in (boundary_index, boundary_index + 1)
    }
    segments = list(plan.segments)
    applied_runs: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []
    applied_boundaries: set[int] = set()
    if boundary_subject_alignment is not None:
        positions_by_segment_index = {
            choice.segment_index: position
            for position, choice in enumerate(visual.choices)
        }
        for alignment_segment in boundary_subject_alignment.segments:
            if alignment_segment.track is None:
                continue
            position = positions_by_segment_index.get(alignment_segment.segment_index)
            if position is None:
                raise CandidateReplayValidationError(
                    "boundary alignment references an unknown visual segment"
                )
            if (
                plan.segments[position].pan_crop
                != face_plan.pan_crop_for(alignment_segment.segment_index)
            ):
                skipped_runs.append(
                    {
                        "first_segment": position,
                        "last_segment": position,
                        "boundary_indices": list(
                            alignment_segment.boundary_indices
                        ),
                        "reason": (
                            "current_face_crop_differs_from_serialized_base_crop"
                        ),
                        "mismatched_segment_indices": [position],
                    }
                )
                continue
            try:
                trial_segment = replace(
                    segments[position],
                    editorial_transform=alignment_segment.track,
                )
                trial_segments = list(segments)
                trial_segments[position] = trial_segment
                trial = replace(plan, segments=tuple(trial_segments))
            except ValueError as exc:
                skipped_runs.append(
                    {
                        "first_segment": position,
                        "last_segment": position,
                        "boundary_indices": list(
                            alignment_segment.boundary_indices
                        ),
                        "reason": "render_plan_safety_rejected_overlay",
                        "error": str(exc),
                    }
                )
                continue
            segments = list(trial.segments)
            applied_boundaries.update(alignment_segment.boundary_indices)
            applied_runs.append(
                {
                    "first_segment": position,
                    "last_segment": position,
                    "boundary_indices": list(
                        alignment_segment.boundary_indices
                    ),
                    "overlaid_segment_indices": [position],
                    "scope": "exact_boundary_local_segment_only",
                }
            )
        # The dense v1 layer already produced complete, frame-validated local
        # tracks.  Do not run the legacy Tier-D run overlay as a second pass or
        # let one accepted cut rewrite neighboring same-source choices.
        candidate_runs = set()
    for first, last in sorted(candidate_runs):
        run_boundaries = tuple(
            index
            for index in boundary_reasons
            if first <= index <= last or first <= index + 1 <= last
        )
        run_items = editorial_plan.segments[first : last + 1]
        non_boundary_reasons = sorted(
            {
                reason
                for item in run_items
                for reason in item.reasons
                if reason not in allowed_reasons
                and reason != "optional_pan_roll_removed"
            }
        )
        if any(reason.startswith("reference_calibrated_") for reason in non_boundary_reasons):
            skipped_runs.append(
                {
                    "first_segment": first,
                    "last_segment": last,
                    "boundary_indices": list(run_boundaries),
                    "reason": "generated_run_contains_non_boundary_editorial_effect",
                    "non_boundary_reasons": non_boundary_reasons,
                }
            )
            continue
        crop_mismatch = tuple(
            index
            for index in range(first, last + 1)
            if plan.segments[index].pan_crop != face_plan.pan_crop_for(index)
        )
        if crop_mismatch:
            skipped_runs.append(
                {
                    "first_segment": first,
                    "last_segment": last,
                    "boundary_indices": list(run_boundaries),
                    "reason": "current_face_crop_differs_from_serialized_base_crop",
                    "mismatched_segment_indices": list(crop_mismatch),
                }
            )
            continue
        trial_segments = list(segments)
        for index in range(first, last + 1):
            trial_segments[index] = replace(
                trial_segments[index],
                editorial_transform=editorial_plan.track_for(index),
            )
        try:
            trial = replace(plan, segments=tuple(trial_segments))
        except ValueError as exc:
            skipped_runs.append(
                {
                    "first_segment": first,
                    "last_segment": last,
                    "boundary_indices": list(run_boundaries),
                    "reason": "render_plan_safety_rejected_overlay",
                    "error": str(exc),
                }
            )
            continue
        segments = list(trial.segments)
        applied_boundaries.update(run_boundaries)
        applied_runs.append(
            {
                "first_segment": first,
                "last_segment": last,
                "boundary_indices": list(run_boundaries),
                "overlaid_segment_indices": list(range(first, last + 1)),
            }
        )
    repaired = replace(plan, segments=tuple(segments))
    affine_payload = affine_plan.to_dict()
    serialized_choices = _sequence(
        _mapping(
            context.candidate_payload.get("visual_timeline"),
            "candidate.visual_timeline",
        ).get("choices"),
        "visual_timeline.choices",
    )
    omitted_appearance_embedding_count = sum(
        _integer(
            _mapping(
                _mapping(raw, "visual_timeline.choices[]").get(evidence_name),
                evidence_name,
            ).get("appearance_embedding_count"),
            f"{evidence_name}.appearance_embedding_count",
        )
        for raw in serialized_choices
        for evidence_name in (
            "dissolve_entry_evidence",
            "dissolve_exit_evidence",
        )
    )
    return ReplayBoundaryRepairResult(
        repaired,
        {
            "hook": "current_boundary_geometry",
            "enabled": True,
            "visual_timeline_reconstructed_from_failed_report": True,
            "privacy_omitted_appearance_embedding_count": (
                omitted_appearance_embedding_count
            ),
            "privacy_omitted_embeddings_consumed_by_repair": False,
            "serialized_transitions_reused_not_rebuilt": True,
            "face_match": {
                "applied_segment_count": face_plan.applied_count,
                "fallback_segment_count": face_plan.fallback_count,
            },
            "boundary_affine": {
                "decision_count": len(affine_plan.decisions),
                "analyzed_count": affine_plan.analyzed_count,
                "eligible_count": affine_plan.eligible_count,
                "exact_subject_pair_count": affine_plan.exact_subject_pair_count,
                "rejection_counts": affine_payload.get("rejection_counts", []),
            },
            "boundary_subject_alignment": (
                {
                    "enabled": True,
                    "analysis_basis": boundary_subject_alignment.analysis_basis,
                    "decision_count": len(boundary_subject_alignment.decisions),
                    "eligible_count": boundary_subject_alignment.eligible_count,
                    "applied_segment_count": (
                        boundary_subject_alignment.applied_segment_count
                    ),
                    "validated_output_frame_count": (
                        boundary_subject_alignment.validated_output_frame_count
                    ),
                    "optical_flow_containment_frame_count": (
                        boundary_subject_alignment.optical_flow_containment_frame_count
                    ),
                    "rejection_counts": boundary_subject_alignment.to_dict().get(
                        "rejection_counts", []
                    ),
                }
                if boundary_subject_alignment is not None
                else {
                    "enabled": False,
                    "error": boundary_subject_alignment_error,
                }
            ),
            "generated_editorial": {
                "applied_segment_count": editorial_plan.applied_count,
                "boundary_match_count": editorial_plan.boundary_match_count,
                "affine_boundary_match_count": (
                    editorial_plan.affine_boundary_match_count
                ),
                "skipped_boundary_reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in editorial_plan.skipped_boundary_reasons
                ],
            },
            "eligible_boundary_indices": sorted(boundary_reasons),
            "eligible_boundary_reasons": {
                str(index): list(reasons)
                for index, reasons in sorted(boundary_reasons.items())
            },
            "applied_boundary_indices": sorted(applied_boundaries),
            "applied_runs": applied_runs,
            "skipped_runs": skipped_runs,
            "source_cadence_audio_transitions_unchanged": True,
            "qa_thresholds_modified": False,
        },
    )


def _structural_signature(plan: RenderPlan) -> tuple[Any, ...]:
    """Signature of fields that a replay repair hook is forbidden to change."""

    return (
        tuple(path.resolve() for path in plan.sources),
        plan.reference_source,
        plan.aspect_ratio,
        plan.output.resolve(),
        plan.master_audio_start,
        plan.width,
        plan.height,
        plan.fps,
        tuple(
            (
                item.source_index,
                item.source_start,
                item.duration,
                item.master_start,
                item.source_duration,
                item.look_adjustment,
            )
            for item in plan.segments
        ),
        plan.transitions,
    )


def _current_boundary_look_hook(
    plan: RenderPlan,
    context: ReplayRepairContext,
) -> ReplayBoundaryRepairResult:
    del context
    from .boundary_look_match import analyze_boundary_look_match

    old_track_count = sum(item.boundary_look_track is not None for item in plan.segments)
    clean = replace(
        plan,
        segments=tuple(
            replace(item, boundary_look_track=None) for item in plan.segments
        ),
    )
    result = analyze_boundary_look_match(
        clean,
        group_16x9_safe=(clean.aspect_ratio is AspectRatio.LANDSCAPE),
    )
    return ReplayBoundaryRepairResult(
        plan=result.apply(clean),
        audit={
            "hook": "current_boundary_look_match",
            "serialized_tracks_cleared_before_reanalysis": old_track_count,
            "result": result.to_dict(),
        },
    )


def load_replay_repair_hook(specification: str) -> ReplayBoundaryRepairHook:
    """Load an explicit ``module:callable`` replay repair extension."""

    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name.strip() or not attribute_name.strip():
        raise CandidateReplayValidationError(
            "repair hook must use the form module:callable"
        )
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise CandidateReplayValidationError(
            f"could not load repair hook {specification!r}: {exc}"
        ) from exc
    if not callable(value):
        raise CandidateReplayValidationError("repair hook target is not callable")
    return value


def apply_current_boundary_repairs(
    plan: RenderPlan,
    context: ReplayRepairContext,
    *,
    extension_hooks: Sequence[tuple[str, ReplayBoundaryRepairHook]] = (),
) -> tuple[RenderPlan, tuple[Mapping[str, Any], ...]]:
    """Apply current geometry, optional extensions, then local look repair.

    The built-in exact-frame geometry repair always runs first.  Additional
    replay-only geometry work can be connected with
    ``--repair-hook module:callable``.  Every hook uses
    :class:`ReplayBoundaryRepairResult` and is guarded by the immutable
    structural signature below.  The built-in current look matcher always runs
    last.
    """

    baseline_signature = _structural_signature(plan)
    current = plan
    audits: list[Mapping[str, Any]] = []
    ordered_hooks = (
        ("current_boundary_geometry", _current_boundary_geometry_hook),
    ) + tuple(extension_hooks) + (
        ("current_boundary_look_match", _current_boundary_look_hook),
    )
    for name, hook in ordered_hooks:
        result = hook(current, context)
        if not isinstance(result, ReplayBoundaryRepairResult):
            raise CandidateReplayValidationError(
                f"repair hook {name!r} returned an invalid result"
            )
        if _structural_signature(result.plan) != baseline_signature:
            raise CandidateReplayValidationError(
                f"repair hook {name!r} changed protected timeline/source fields"
            )
        current = result.plan
        audits.append({"name": name, **dict(result.audit)})
    return (current, tuple(audits))


def _review_metric_comparison(
    recorded: Mapping[str, Any], replayed: Mapping[str, Any]
) -> dict[str, Any]:
    def quantile(values: Sequence[float], fraction: float) -> float:
        ordered = sorted(values)
        if not ordered:
            raise CandidateReplayValidationError("review metric domain is empty")
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        alpha = position - low
        return ordered[low] * (1.0 - alpha) + ordered[high] * alpha

    # V2 scored all serialized segment boundaries in one pool.  Recover its
    # actual introduced cross-source subset so a V3 replay is not compared to
    # the obsolete Tier-D/native-cut denominator.
    recorded_boundaries = [
        _mapping(item, "recorded boundaries[]")
        for item in _sequence(recorded.get("boundaries"), "recorded boundaries")
    ]
    old_introduced = [
        item
        for item in recorded_boundaries
        if _integer(item.get("outgoing_source_index"), "old outgoing source")
        != _integer(item.get("incoming_source_index"), "old incoming source")
    ]
    old_scores = [
        _number(item.get("weighted_score"), "old boundary weighted score")
        for item in old_introduced
    ]
    old_issue_count = sum(
        not _boolean(item.get("passed"), "old boundary passed")
        for item in old_introduced
    )
    old_boundary = {
        "p10": quantile(old_scores, 0.10),
        "issue_fraction": old_issue_count / len(old_introduced),
        "boundary_count": len(old_introduced),
        "issue_count": old_issue_count,
        "basis": "derived_introduced_cross_source_from_v2_boundaries",
    }
    raw_new_boundary = replayed.get("visible_boundary_score")
    if raw_new_boundary is None:
        raw_new_boundary = replayed.get("boundary_score")
    new_boundary = _mapping(raw_new_boundary, "replayed visible boundary score")
    fields = {
        "weighted_score": (
            _number(recorded.get("weighted_score"), "recorded weighted score"),
            _number(replayed.get("weighted_score"), "replayed weighted score"),
        ),
        "boundary_score_p10": (
            _number(old_boundary.get("p10"), "recorded boundary p10"),
            _number(new_boundary.get("p10"), "replayed boundary p10"),
        ),
        "boundary_issue_fraction": (
            _number(old_boundary.get("issue_fraction"), "recorded issue fraction"),
            _number(new_boundary.get("issue_fraction"), "replayed issue fraction"),
        ),
    }
    return {
        "recorded_analysis_basis": recorded.get("analysis"),
        "current_analysis_basis": replayed.get("analysis"),
        "same_analysis_basis": recorded.get("analysis") == replayed.get("analysis"),
        "boundary_metric_domain": "introduced_cross_source",
        "recorded_boundary_domain": old_boundary,
        "replayed_boundary_domain": {
            "p10": new_boundary.get("p10"),
            "issue_fraction": new_boundary.get("issue_fraction"),
            "boundary_count": _mapping(
                replayed.get("boundary_domains"), "replayed boundary domains"
            ).get("introduced_cross_source", {}).get("boundary_count")
            if isinstance(replayed.get("boundary_domains"), Mapping)
            else new_boundary.get("distribution", {}).get("count")
            if isinstance(new_boundary.get("distribution"), Mapping)
            else None,
            "issue_count": new_boundary.get("issue_count"),
            "basis": new_boundary.get("count_basis", "legacy_boundary_score"),
        },
        "metrics": {
            name: {
                "recorded": old,
                "replayed": new,
                "delta": new - old,
            }
            for name, (old, new) in fields.items()
        },
        "note": (
            "V2 introduced-cut metrics are re-derived from its boundary records; "
            "metric equality is evidence only because a current repair or reviewer "
            "revision may change scores without weakening QA thresholds"
        ),
    }


def replay_failed_candidate(
    selection_report_path: Path,
    acquisition_manifest_path: Path,
    *,
    candidate_index: int,
    reference_source: int,
    output: Path,
    audit_output: Path | None = None,
    apply_current_boundary_repair: bool = False,
    repair_hooks: Sequence[tuple[str, ReplayBoundaryRepairHook]] = (),
    professional_reference_profile: Path | None = None,
    render_review: RenderReview = render_and_review_proxy,
) -> CandidateReplayArtifact:
    """Render and decoded-review one verified failed candidate as QA evidence."""

    video_path = Path(output).resolve()
    if QA_OUTPUT_MARKER not in video_path.name.casefold():
        raise CandidateReplayValidationError(
            f"QA replay output filename must contain {QA_OUTPUT_MARKER!r}"
        )
    if video_path.suffix.casefold() != ".mp4":
        raise CandidateReplayValidationError("QA replay output must be an MP4")
    if video_path.exists():
        raise FileExistsError(video_path)
    audit_path = (
        Path(audit_output).resolve()
        if audit_output is not None
        else video_path.with_suffix(".audit.json")
    )
    if audit_path.exists():
        raise FileExistsError(audit_path)

    reference_profile_audit: Mapping[str, Any] | None = None
    resolved_reference_profile: Path | None = professional_reference_profile
    if professional_reference_profile is not None:
        profile_path = Path(professional_reference_profile).resolve()
        if not profile_path.is_file():
            raise FileNotFoundError(profile_path)
        resolved_reference_profile = profile_path
        reference_profile_audit = {
            "kind": "file",
            "path": str(profile_path),
            "sha256": _sha256_path(profile_path),
        }

    reconstructed = reconstruct_failed_candidate(
        selection_report_path,
        acquisition_manifest_path,
        candidate_index=candidate_index,
        reference_source=reference_source,
        output=video_path,
    )
    context = ReplayRepairContext(
        selection_report_path=Path(selection_report_path).resolve(),
        acquisition_manifest_path=Path(acquisition_manifest_path).resolve(),
        candidate_index=candidate_index,
        candidate_payload=reconstructed.candidate_payload,
    )
    plan = reconstructed.plan
    repair_audits: tuple[Mapping[str, Any], ...] = ()
    if repair_hooks and not apply_current_boundary_repair:
        raise CandidateReplayValidationError(
            "repair hooks require --apply-current-boundary-repair"
        )
    if apply_current_boundary_repair:
        plan, repair_audits = apply_current_boundary_repairs(
            plan,
            context,
            extension_hooks=repair_hooks,
        )

    try:
        if resolved_reference_profile is None:
            proxy_plan, review = render_review(plan, video_path)
        else:
            proxy_plan, review = render_review(
                plan,
                video_path,
                professional_reference_profile=resolved_reference_profile,
            )
        if proxy_plan.output.resolve() != video_path:
            raise CandidateReplayValidationError("renderer wrote an unexpected proxy path")
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise CandidateReplayValidationError("renderer did not create the QA proxy")
        expected_frames = plan.output_frame_count
        replay_proxy = _mapping(review.payload.get("proxy"), "replayed review.proxy")
        if _integer(
            replay_proxy.get("decoded_frame_count"),
            "replayed review.proxy.decoded_frame_count",
        ) != expected_frames:
            raise CandidateReplayValidationError("replayed proxy frame count drifted")
        comparison = _review_metric_comparison(
            reconstructed.recorded_review,
            review.payload,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "analysis_basis": ANALYSIS_BASIS,
            "artifact_class": "qa_only_failed_candidate_replay",
            "production_use_forbidden": True,
            "candidate_index": candidate_index,
            "mode": (
                "current_boundary_repair_qa_replay"
                if apply_current_boundary_repair
                else "exact_serialized_old_plan_fidelity_replay"
            ),
            "selection_report": {
                "path": str(Path(selection_report_path).resolve()),
                "sha256": reconstructed.selection_report_sha256,
            },
            "acquisition_manifest": {
                "path": str(Path(acquisition_manifest_path).resolve()),
                "sha256": reconstructed.acquisition_manifest_sha256,
            },
            "source_fingerprints": [
                item.to_dict() for item in reconstructed.verified_sources
            ],
            "reconstruction_checks": dict(reconstructed.reconstruction_checks),
            "repair_hooks": [dict(item) for item in repair_audits],
            "proxy": {
                "path": str(video_path),
                "size_bytes": video_path.stat().st_size,
                "sha256": _sha256_path(video_path),
                "fixed_height_pixels": proxy_plan.height,
                "frame_count": expected_frames,
                "duration_seconds": proxy_plan.render_duration,
                "qa_only_filename_marker_present": True,
            },
            "recorded_vs_replayed_review": comparison,
            "professional_reference_profile": reference_profile_audit,
            "decoded_proxy_review": review.payload,
            "qa_thresholds_weakened": False,
            "candidate_selection_rerun": False,
            "acquisition_or_analysis_rerun": False,
        }
        resolved_audit = _atomic_write_json(audit_path, payload)
        return CandidateReplayArtifact(video_path, resolved_audit, payload)
    except Exception:
        # The destination was guaranteed absent before this invocation, so any
        # file here belongs to this failed QA attempt and must not survive as a
        # plausible-looking replay artifact.
        video_path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly reconstruct one failed complete candidate and render a "
            "persistent 480p QA replay (never a production final)."
        )
    )
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--reference-source", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument(
        "--professional-reference-profile",
        type=Path,
        help=(
            "Validated five-reference same-song profile used by the decoded "
            "professional-relative selection gate; absolute QA remains diagnostic."
        ),
    )
    parser.add_argument(
        "--apply-current-boundary-repair",
        action="store_true",
        help=(
            "Apply current exact-frame geometry, optional replay extensions, "
            "and current boundary look before rendering; cadence and QA limits "
            "stay fixed."
        ),
    )
    parser.add_argument(
        "--repair-hook",
        action="append",
        default=[],
        metavar="MODULE:CALLABLE",
        help=(
            "Replay-only geometry/crop hook returning ReplayBoundaryRepairResult; "
            "requires --apply-current-boundary-repair and may be repeated."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    hooks = tuple(
        (specification, load_replay_repair_hook(specification))
        for specification in arguments.repair_hook
    )
    artifact = replay_failed_candidate(
        arguments.selection_report,
        arguments.acquisition_manifest,
        candidate_index=arguments.candidate_index,
        reference_source=arguments.reference_source,
        output=arguments.output,
        audit_output=arguments.audit_output,
        apply_current_boundary_repair=arguments.apply_current_boundary_repair,
        repair_hooks=hooks,
        professional_reference_profile=arguments.professional_reference_profile,
    )
    print(
        json.dumps(
            {
                "status": "qa_replay_complete",
                "production_use_forbidden": True,
                "video": str(artifact.video_path),
                "audit": str(artifact.audit_path),
                "passed": bool(
                    _mapping(
                        artifact.payload.get("decoded_proxy_review"),
                        "decoded_proxy_review",
                    ).get("passed")
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_BASIS",
    "CandidateReplayArtifact",
    "CandidateReplayValidationError",
    "ReconstructedFailedCandidate",
    "ReplayRepairContext",
    "ReplayBoundaryRepairHook",
    "ReplayBoundaryRepairResult",
    "apply_current_boundary_repairs",
    "load_replay_repair_hook",
    "reconstruct_failed_candidate",
    "replay_failed_candidate",
]
