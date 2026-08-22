"""Top-level offline analysis pipeline used by the native app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any, Iterator, Sequence

from numpy.typing import NDArray
import numpy as np
import cv2

from .audio import (
    DEFAULT_ANALYSIS_SAMPLE_RATE,
    AlignmentResult,
    SourceAlignment,
    extract_mono_pcm,
    normalized_cross_correlation_offset,
)
from .events import EventReporter
from .face_match import FaceMatchPlan, build_face_match_plan
from .group_transition import GroupTransitionPlan, build_group_transition_plan
from .draft import build_group_draft
from .ive import EditMode
from .media import MediaInfo, probe_media
from .member_draft import (
    MemberDraftTimeline,
    MemberSegmentKind,
    build_member_draft,
)
from .render import (
    RENDER_PLAN_SOURCE_INDEX_DOMAIN,
    RenderPlan,
    RenderTransition,
    aligned_render_segment,
    output_size,
    render,
    write_plan,
)
from .request import AnalysisRequest
from .rhythm import analyze_rhythm
from .tracking import (
    AnchoredTrackerConfig,
    MemberAnchor,
    MemberTrackingResult,
    TrackingState,
    track_anchored_member,
)
from .vision import (
    SceneDetectionConfig,
    detect_scene_changes_in_video,
    iter_video_frames,
)
from .visual_score import VisualTimelinePlan, build_visual_timeline


ANALYSIS_SAMPLE_RATE = min(DEFAULT_ANALYSIS_SAMPLE_RATE, 4_000)
REQUEST_SOURCE_INDEX_DOMAIN = "request_source_index"
ELIGIBLE_SOURCE_INDEX_DOMAIN = "eligible_source_index"


@dataclass(frozen=True, slots=True)
class AnalysisArtifact:
    manifest_path: Path
    manifest: dict[str, Any]
    output_paths: tuple[Path, ...]


def _source_index_mapping(
    sources: Sequence[Path],
    eligible_indices: Sequence[int],
) -> dict[str, Any]:
    """Describe both source-index domains used by the analysis artifacts.

    This pipeline copies ``request.sources`` into each render plan without
    reordering it. The standalone plan names those same numbers by their local
    ``sources[]`` array domain, and ``domain_equivalences`` records that the two
    domains are identical here. Analysis stages that operate only on
    synchronization-safe sources use a compact local index instead. The explicit
    bidirectional mapping prevents that local index from being mistaken for a
    request index after an excluded source creates a gap.
    """

    request_count = len(sources)
    eligible = tuple(eligible_indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= request_count
        for index in eligible
    ):
        raise ValueError("eligible source mapping contains an invalid request index")
    if len(set(eligible)) != len(eligible):
        raise ValueError("eligible source mapping contains duplicate request indices")

    request_to_eligible: list[int | None] = [None] * request_count
    entries: list[dict[str, Any]] = []
    for eligible_index, request_index in enumerate(eligible):
        request_to_eligible[request_index] = eligible_index
        entries.append(
            {
                "eligible_source_index": eligible_index,
                "request_source_index": request_index,
                "source": str(sources[request_index]),
            }
        )

    return {
        "request_index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
        "eligible_index_domain": ELIGIBLE_SOURCE_INDEX_DOMAIN,
        "render_plan_index_domain": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
        "domain_equivalences": {
            RENDER_PLAN_SOURCE_INDEX_DOMAIN: REQUEST_SOURCE_INDEX_DOMAIN,
        },
        "eligible_to_request": list(eligible),
        "request_to_eligible": request_to_eligible,
        "eligible_sources": entries,
        "payload_index_domains": {
            # Paths ending in ``[]`` without a child field describe the
            # zero-based array position.  The serialized media/alignment and
            # audio-offset items do not contain a synthetic ``position`` key.
            "request.sources[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "request.reference_source": REQUEST_SOURCE_INDEX_DOMAIN,
            "request.member_anchors[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "media[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "alignment.alignments[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "alignment_eligibility[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "scene_analysis[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_tracking[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_anchor_trim.anchors[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_anchor_trim.limiting_source_indices[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "group_draft.segments[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "group_draft.audio_offsets[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "group_draft.sync_scores[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "visual_timeline.choices[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "face_match.segments[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.segments[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.audio_offsets[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.group_wide_fallback_source": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.safe_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.quality_eligible_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.selected_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.no_safe_candidate_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.quality_guard_excluded_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.unused_quality_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.sources[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "render_plan.sources[]": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render_plan.reference_source": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render_plan.segments[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
        },
    }


def _with_index_domain(payload: dict[str, Any], domain: str) -> dict[str, Any]:
    """Attach a machine-readable source-index domain without mutating a payload."""

    if domain not in (REQUEST_SOURCE_INDEX_DOMAIN, ELIGIBLE_SOURCE_INDEX_DOMAIN):
        raise ValueError("unknown source index domain")
    result = dict(payload)
    result["index_domain"] = domain
    return result


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", value.strip()).strip("-._")
    return slug or "untitled"


def _member_render_selection_basis(selection_policy: str) -> str:
    """Name the complete anchored-tracking and global-selection contract."""

    policy = selection_policy.strip()
    if not policy:
        raise ValueError("member selection policy cannot be blank")
    return f"anchored_target_tracking_virtual_camera_{policy}"


def _capabilities_payload(
    *,
    group_ready: bool,
    edit_mode: EditMode,
    member_draft: MemberDraftTimeline | None,
    rendered_video_path: Path | None,
) -> dict[str, Any]:
    """Describe implemented analysis/render capabilities without claiming re-ID."""

    return {
        "group_timeline_inputs_ready": group_ready,
        "group_video_rendered": (
            edit_mode is EditMode.GROUP and rendered_video_path is not None
        ),
        "member_tracking_ready": (
            member_draft is not None
            and all(
                item.kind is MemberSegmentKind.TARGET_CROP
                for item in member_draft.segments
            )
        ),
        "member_video_rendered": (
            edit_mode is EditMode.MEMBER and rendered_video_path is not None
        ),
        "member_tracking_note": (
            "사용자 확인 앵커에서 시작한 전방향 연속 추적만 신뢰하며, "
            "카메라 컷 뒤 자동 재식별은 아직 사용하지 않습니다."
        ),
    }


def _validate_media(info: MediaInfo) -> None:
    if not info.video_streams:
        raise ValueError(f"video stream not found: {info.path}")
    if not info.audio_streams:
        raise ValueError(f"audio stream not found: {info.path}")
    if not info.duration_seconds or info.duration_seconds <= 0:
        raise ValueError(f"media duration is unavailable: {info.path}")


def _align_sequentially(
    request: AnalysisRequest,
    reporter: EventReporter,
) -> tuple[AlignmentResult, NDArray[np.float32]]:
    reference_path = request.sources[request.reference_source]
    reporter.status("기준 음원을 추출하는 중…")
    reference_pcm = extract_mono_pcm(reference_path, sample_rate=ANALYSIS_SAMPLE_RATE)
    alignments: list[SourceAlignment] = []

    for index, source in enumerate(request.sources):
        if index == request.reference_source:
            alignments.append(
                SourceAlignment(
                    source=str(source),
                    is_reference=True,
                    offset_samples=0,
                    offset_seconds=0.0,
                    score=1.0,
                    overlap_samples=int(reference_pcm.size),
                    overlap_seconds=reference_pcm.size / ANALYSIS_SAMPLE_RATE,
                    second_best_score=None,
                    score_margin=1.0,
                    overlap_ratio=1.0,
                )
            )
            continue

        reporter.status(f"음원 정렬 중: {source.name}")
        source_pcm = extract_mono_pcm(source, sample_rate=ANALYSIS_SAMPLE_RATE)
        estimate = normalized_cross_correlation_offset(
            reference_pcm,
            source_pcm,
            sample_rate=ANALYSIS_SAMPLE_RATE,
            max_offset_seconds=120.0,
            min_overlap_seconds=min(10.0, max(2.0, min(reference_pcm.size, source_pcm.size) / ANALYSIS_SAMPLE_RATE / 3.0)),
        )
        alignments.append(
            SourceAlignment(
                source=str(source),
                is_reference=False,
                offset_samples=estimate.offset_samples,
                offset_seconds=estimate.offset_seconds,
                score=estimate.score,
                overlap_samples=estimate.overlap_samples,
                overlap_seconds=estimate.overlap_seconds,
                second_best_score=estimate.second_best_score,
                score_margin=estimate.score_margin,
                overlap_ratio=estimate.overlap_ratio,
            )
        )
        reporter.log(
            f"{source.name}: 기준 대비 {estimate.offset_seconds:+.3f}초, "
            f"정렬 신뢰도 {estimate.score:.3f}, 후보 간격 {estimate.score_margin:.3f}"
        )
        completed_sources = index + 1
        reporter.progress(
            0.22 + 0.38 * completed_sources / len(request.sources),
            "음방 오디오를 정렬하는 중…",
        )

    # Keep request/source order, including when the reference is not the first file.
    by_source = {alignment.source: alignment for alignment in alignments}
    ordered = tuple(by_source[str(source)] for source in request.sources)
    return (
        AlignmentResult(
            reference=str(reference_path),
            sample_rate=ANALYSIS_SAMPLE_RATE,
            alignments=ordered,
            group="IVE",
        ),
        reference_pcm,
    )


def _common_master_range(
    media: list[MediaInfo],
    alignment: AlignmentResult,
    source_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[float, float]:
    starts: list[float] = [0.0]
    ends: list[float] = []
    indices = range(len(media)) if source_indices is None else source_indices
    for index in indices:
        info = media[index]
        source_alignment = alignment.alignments[index]
        assert info.duration_seconds is not None
        # master time t maps to source time t + offset.
        starts.append(-source_alignment.offset_seconds)
        ends.append(info.duration_seconds - source_alignment.offset_seconds)
    start = max(starts)
    end = min(ends)
    if end <= start:
        raise ValueError("aligned source videos have no common usable range")
    return (max(0.0, start), end)


def _member_anchor_trim(
    *,
    common_master_start: float,
    common_master_end: float,
    alignment: AlignmentResult,
    eligible_indices: list[int] | tuple[int, ...],
    tracking_results: dict[int, MemberTrackingResult],
) -> tuple[float, dict[str, Any]]:
    """Return the first canonical time covered by every forward-only track.

    Manual member tracking deliberately starts at the user-confirmed frame and
    never guesses backwards.  A source-local observation at ``s`` maps to
    canonical master time ``s - audio_offset``.  The safe shared start is
    therefore the latest canonical anchor among the eligible sources.
    """

    anchor_details: list[dict[str, Any]] = []
    for source_index in eligible_indices:
        result = tracking_results.get(source_index)
        if result is None or not result.observations:
            raise ValueError(
                "eligible member sources need a tracked anchor observation"
            )
        source_local_time = result.observations[0].timestamp_seconds
        offset = alignment.alignments[source_index].offset_seconds
        canonical_time = source_local_time - offset
        anchor_details.append(
            {
                "source_index": source_index,
                "source": alignment.alignments[source_index].source,
                "source_local_time_seconds": source_local_time,
                "audio_offset_seconds": offset,
                "canonical_anchor_time_seconds": canonical_time,
            }
        )

    if not anchor_details:
        effective_start = common_master_start
    else:
        effective_start = max(
            common_master_start,
            max(
                float(item["canonical_anchor_time_seconds"])
                for item in anchor_details
            ),
        )
    trimmed_seconds = max(0.0, effective_start - common_master_start)
    minimum_render_duration = 2.0 / 29.97
    usable = common_master_end - effective_start >= minimum_render_duration
    limiting_indices = [
        int(item["source_index"])
        for item in anchor_details
        if abs(
            float(item["canonical_anchor_time_seconds"]) - effective_start
        )
        <= 1e-7
    ]
    return effective_start, {
        "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
        "policy": "trim_to_latest_canonical_anchor",
        "reverse_tracking_used": False,
        "applied": trimmed_seconds > 1e-7,
        "aligned_start_seconds": common_master_start,
        "effective_start_seconds": effective_start,
        "end_seconds": common_master_end,
        "trimmed_seconds": trimmed_seconds,
        "usable_duration_seconds": max(0.0, common_master_end - effective_start),
        "usable_for_render": usable,
        "limiting_source_indices": limiting_indices,
        "anchors": anchor_details,
    }


def _alignment_eligibility(alignment: AlignmentResult) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for source_index, item in enumerate(alignment.alignments):
        reasons: list[str] = []
        if not item.is_reference:
            if item.score < 0.45:
                reasons.append(f"상관 점수 {item.score:.3f} < 0.450")
            if item.score_margin < 0.04:
                reasons.append(f"후보 간격 {item.score_margin:.3f} < 0.040")
            if item.overlap_ratio < 0.35:
                reasons.append(f"겹침 비율 {item.overlap_ratio:.3f} < 0.350")
        results.append(
            {
                "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
                "source_index": source_index,
                "source": item.source,
                "render_eligible": not reasons,
                "reasons": reasons,
            }
        )
    return results


def _tracking_frames(
    source: Path,
    *,
    sample_every: int = 2,
    maximum_width: int = 640,
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    for frame_index, frame in iter_video_frames(source, sample_every=sample_every):
        height, width = frame.shape[:2]
        if width > maximum_width:
            scale = maximum_width / width
            frame = cv2.resize(
                frame,
                (maximum_width, max(2, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        yield frame_index, frame


def _member_tracking_payload(
    source_index: int,
    result: MemberTrackingResult,
    anchor_time_seconds: float,
    anchor_bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    counts = {
        state.value: sum(item.state is state for item in result.observations)
        for state in TrackingState
    }
    # Keep a diagnostic path at roughly 6 Hz rather than writing every decoded
    # frame into the analysis manifest.
    keyframe_stride = max(1, int(round(result.fps / 6.0)))
    keyframes = [
        {
            "frame_index": item.frame_index,
            "time_seconds": item.timestamp_seconds,
            "bbox_xyxy": list(item.bbox_xyxy),
            "confidence": item.confidence,
            "state": item.state.value,
        }
        for item in result.observations
        if item.bbox_xyxy is not None
        and (
            item.frame_index == result.observations[0].frame_index
            or item.frame_index % keyframe_stride == 0
        )
    ]
    return {
        "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
        "source_index": source_index,
        "source": result.source_id,
        "member": result.member.value,
        "anchor": {
            "time_seconds": anchor_time_seconds,
            "bbox_xyxy": list(anchor_bbox),
        },
        "backend": result.backend_name,
        "terminal_failure": (
            result.terminal_failure.value if result.terminal_failure else None
        ),
        "observation_counts": counts,
        "keyframes": keyframes,
    }


def analyze(request: AnalysisRequest, reporter: EventReporter) -> AnalysisArtifact:
    reporter.progress(0.01, "요청을 확인하는 중…")
    request.output_directory.mkdir(parents=True, exist_ok=True)

    reporter.status("영상 메타데이터를 읽는 중…")
    media: list[MediaInfo] = []
    for index, source in enumerate(request.sources):
        info = probe_media(source)
        _validate_media(info)
        media.append(info)
        reporter.log(
            f"{source.name}: {info.duration_seconds:.2f}초, "
            f"{len(info.video_streams)} video / {len(info.audio_streams)} audio"
        )
        reporter.progress(0.05 + 0.15 * (index + 1) / len(request.sources), "영상 확인 중…")

    alignment, reference_pcm = _align_sequentially(request, reporter)
    alignment_eligibility = _alignment_eligibility(alignment)
    eligible_indices = [
        index
        for index, item in enumerate(alignment_eligibility)
        if item["render_eligible"]
    ]
    common_start, common_end = _common_master_range(
        media,
        alignment,
        eligible_indices,
    )

    rhythm_payload: dict[str, Any] | None = None
    rhythm_analysis = None
    rhythm_warning: str | None = None
    reporter.status("기준 곡의 박자와 프레이즈를 분석하는 중…")
    try:
        rhythm_analysis = analyze_rhythm(reference_pcm, ANALYSIS_SAMPLE_RATE)
        rhythm_payload = rhythm_analysis.to_dict()
        # The full onset envelope is useful for diagnostics but unnecessarily
        # large for the app contract. Keep its timing metadata and derived grid.
        onset = rhythm_payload.get("onset_envelope")
        if isinstance(onset, dict):
            onset.pop("times_seconds", None)
            onset.pop("values", None)
        reporter.log(
            f"추정 템포 {rhythm_analysis.bpm:.2f} BPM · "
            f"박자 {len(rhythm_analysis.beats)}개 · "
            f"프레이즈 후보 {len(rhythm_analysis.phrase_candidates)}개"
        )
    except ValueError as exc:
        rhythm_warning = f"박자 분석을 확정하지 못함: {exc}"
        reporter.emit("log", level="warning", message=rhythm_warning)
    reporter.progress(0.64, "박자 분석 완료")

    scene_payloads: list[dict[str, Any]] = []
    scene_config = SceneDetectionConfig(min_gap_frames=8)
    for index, (source, info) in enumerate(zip(request.sources, media, strict=True)):
        reporter.status(f"카메라 컷 분석 중: {source.name}")
        changes = detect_scene_changes_in_video(
            source,
            config=scene_config,
            sample_every=2,
        )
        frame_rate = info.video_streams[0].frame_rate or 30.0
        scene_payloads.append(
            {
                "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
                "source_index": index,
                "source": str(source),
                "frame_rate": frame_rate,
                "changes": [
                    {
                        "frame_index": change.frame_index,
                        "timestamp_seconds": change.frame_index / frame_rate,
                        "score": change.score,
                        "frame_difference": change.frame_difference,
                        "histogram_distance": change.histogram_distance,
                    }
                    for change in changes
                ],
            }
        )
        reporter.log(f"{source.name}: 방송 카메라 컷 {len(changes)}개 감지")
        reporter.progress(
            0.64 + 0.30 * (index + 1) / len(request.sources),
            "방송 카메라 컷을 분석하는 중…",
        )

    member_tracking_results: dict[int, MemberTrackingResult] = {}
    member_tracking_payloads: list[dict[str, Any]] = []
    member_common_start = common_start
    member_anchor_trim_payload: dict[str, Any] | None = None
    if request.edit_mode is EditMode.MEMBER:
        assert request.target_member is not None
        anchors_by_source = {
            anchor.source_index: anchor for anchor in request.member_anchors
        }
        for source_index, (source, info) in enumerate(
            zip(request.sources, media, strict=True)
        ):
            anchor_request = anchors_by_source[source_index]
            assert info.duration_seconds is not None
            if anchor_request.time_seconds >= info.duration_seconds:
                raise ValueError(
                    f"멤버 앵커 시각이 영상 길이를 벗어남: {source.name}"
                )
            if source_index not in eligible_indices:
                continue
            fps = info.video_streams[0].frame_rate or 30.0
            sample_every = 1
            anchor_frame = int(round(anchor_request.time_seconds * fps / sample_every)) * sample_every
            reporter.status(f"{request.target_member.value} 추적 중: {source.name}")
            scene_frames = {
                int(change["frame_index"])
                for change in scene_payloads[source_index]["changes"]
                if int(change["frame_index"]) > anchor_frame
            }
            result = track_anchored_member(
                _tracking_frames(source, sample_every=sample_every),
                anchor=MemberAnchor(
                    frame_index=anchor_frame,
                    bbox_xyxy=anchor_request.bbox_xyxy,
                    member=request.target_member,
                    source_id=str(source),
                ),
                fps=fps,
                config=AnchoredTrackerConfig(
                    minimum_confidence=0.42,
                    minimum_appearance_similarity=0.08,
                    maximum_center_jump=0.26,
                    hold_frames=4,
                ),
                scene_change_frames=scene_frames,
            )
            member_tracking_results[source_index] = result
            member_tracking_payloads.append(
                _member_tracking_payload(
                    source_index,
                    result,
                    anchor_request.time_seconds,
                    anchor_request.bbox_xyxy,
                )
            )
            tracked_count = sum(
                item.state is TrackingState.TRACKED for item in result.observations
            )
            reporter.log(
                f"{source.name}: 안전 추적 {tracked_count}프레임 · "
                f"종료 사유 {result.terminal_failure.value if result.terminal_failure else '없음'}"
            )
            reporter.progress(
                0.94 + 0.035 * (source_index + 1) / len(request.sources),
                "대상 멤버를 추적하는 중…",
            )
        member_common_start, member_anchor_trim_payload = _member_anchor_trim(
            common_master_start=common_start,
            common_master_end=common_end,
            alignment=alignment,
            eligible_indices=eligible_indices,
            tracking_results=member_tracking_results,
        )

    warnings: list[str] = []
    if rhythm_warning:
        warnings.append(rhythm_warning)
    for eligibility in alignment_eligibility:
        reasons = eligibility["reasons"]
        if reasons:
            warnings.append(
                f"자동 편집 후보에서 제외: {Path(str(eligibility['source'])).name} "
                f"({'; '.join(str(reason) for reason in reasons)})"
            )
    if (
        member_anchor_trim_payload is not None
        and member_anchor_trim_payload["applied"]
    ):
        warnings.append(
            "수동 멤버 앵커보다 앞선 구간은 뒤로 추적하지 않습니다. "
            f"가장 늦은 공통 앵커까지 "
            f"{float(member_anchor_trim_payload['trimmed_seconds']):.3f}초를 "
            "안전하게 제외했습니다."
        )

    group_draft = None
    visual_timeline: VisualTimelinePlan | None = None
    face_match_plan: FaceMatchPlan | None = None
    group_transition_plan: GroupTransitionPlan | None = None
    member_draft: MemberDraftTimeline | None = None
    render_payload: dict[str, Any] | None = None
    rendered_video_path: Path | None = None
    render_plan_path: Path | None = None
    timestamp = datetime.now(UTC)
    base_filename = f"IVE-{_safe_slug(request.song)}-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    group_ready = len(eligible_indices) >= 2 and rhythm_analysis is not None
    if request.edit_mode is EditMode.GROUP and group_ready:
        reporter.progress(0.95, "자동 교차편집 초안을 만드는 중…")
        offsets = tuple(
            alignment.alignments[index].offset_seconds for index in eligible_indices
        )
        sync_scores = tuple(
            alignment.alignments[index].score for index in eligible_indices
        )
        scene_cut_times = tuple(
            tuple(
                float(change["timestamp_seconds"])
                for change in scene_payloads[index]["changes"]
            )
            for index in eligible_indices
        )
        phrase_times = tuple(
            sorted(
                {
                    float(time_value)
                    for phrase in rhythm_analysis.phrase_candidates
                    for time_value in (phrase.start_seconds, phrase.end_seconds)
                }
            )
        )
        group_draft = build_group_draft(
            source_count=len(eligible_indices),
            common_master_start=common_start,
            common_master_end=common_end,
            audio_offsets=offsets,
            sync_scores=sync_scores,
            beat_times=rhythm_analysis.beat_times_seconds,
            phrase_times=phrase_times,
            scene_cut_times=scene_cut_times,
        )
        reporter.status("컷 전후 구도와 움직임을 비교하는 중…")
        try:
            visual_timeline = build_visual_timeline(
                sources=tuple(request.sources[index] for index in eligible_indices),
                canonical_spans=tuple(
                    (
                        item.canonical_master_start,
                        item.canonical_master_end,
                    )
                    for item in group_draft.segments
                ),
                audio_offsets=offsets,
                sync_scores=sync_scores,
                scene_cut_times=scene_cut_times,
                preferred_source_indices=tuple(
                    item.source_index for item in group_draft.segments
                ),
            )
            reporter.log(
                f"프레임 품질·구도·움직임 전역 선택 점수 "
                f"{visual_timeline.optimizer_score:.3f}"
            )
        except (ValueError, cv2.error) as exc:
            warnings.append(
                f"프레임 시각 점수를 확정하지 못해 리듬 초안을 사용함: {exc}"
            )
        if visual_timeline is not None:
            try:
                source_aspect_ratios: list[float] = []
                for index in eligible_indices:
                    stream = media[index].video_streams[0]
                    if not stream.width or not stream.height:
                        raise ValueError(
                            "얼굴 맞춤 크롭을 위한 원본 해상도를 확인할 수 없음"
                        )
                    source_aspect_ratios.append(stream.width / stream.height)
                face_match_plan = build_face_match_plan(
                    visual_timeline=visual_timeline,
                    source_aspect_ratios=source_aspect_ratios,
                )
                reporter.log(
                    "얼굴 크기·위치 맞춤 "
                    f"{face_match_plan.applied_count}개 적용 · "
                    f"{face_match_plan.fallback_count}개 원본 화면 유지"
                )
            except (ValueError, cv2.error) as exc:
                warnings.append(
                    "얼굴 맞춤을 안전하게 확정하지 못해 "
                    f"원본 화면 컷을 사용함: {exc}"
                )
            try:
                local_durations = tuple(
                    float(
                        media[index].video_streams[0].duration_seconds
                        or media[index].duration_seconds
                        or 0.0
                    )
                    for index in eligible_indices
                )
                group_transition_plan = build_group_transition_plan(
                    visual_timeline=visual_timeline,
                    face_match_plan=face_match_plan,
                    source_durations_seconds=local_durations,
                )
                tier_counts = {
                    tier: sum(
                        transition.tier == tier
                        for transition in group_transition_plan.transitions
                    )
                    for tier in ("A", "B", "C", "D")
                }
                reporter.log(
                    "전환 안전 등급 "
                    + " · ".join(
                        f"{tier} {tier_counts[tier]}개"
                        for tier in ("A", "B", "C", "D")
                    )
                )
            except ValueError as exc:
                warnings.append(
                    "전환 안전 등급을 확정하지 못해 모든 경계를 "
                    f"하드 컷으로 유지함: {exc}"
                )
        width, height = output_size(request.aspect_ratio)
        if visual_timeline is not None:
            segments = tuple(
                aligned_render_segment(
                    source_index=eligible_indices[item.source_index],
                    reference_start=item.canonical_master_start,
                    common_master_start=common_start,
                    source_offset=offsets[item.source_index],
                    duration=item.duration_seconds,
                    pan_crop=(
                        face_match_plan.pan_crop_for(item.segment_index)
                        if face_match_plan is not None
                        else None
                    ),
                )
                for item in visual_timeline.choices
            )
        else:
            segments = tuple(
                aligned_render_segment(
                    source_index=eligible_indices[item.source_index],
                    reference_start=item.canonical_master_start,
                    common_master_start=common_start,
                    source_offset=item.audio_offset_seconds,
                    duration=item.duration_seconds,
                )
                for item in group_draft.segments
            )
        if group_transition_plan is None:
            group_transition_plan = GroupTransitionPlan(
                tuple(
                    RenderTransition(
                        tier="C",
                        reason="rhythm_aligned_hard_cut_visual_analysis_unavailable",
                    )
                    for _ in range(max(0, len(segments) - 1))
                ),
                analysis_basis="rhythm_only_hard_cut_fallback_v1",
            )
        rendered_video_path = request.output_directory / f"{base_filename}-group-draft.mp4"
        render_plan_path = request.output_directory / f"{base_filename}-render-plan.json"
        plan = RenderPlan(
            sources=request.sources,
            segments=segments,
            reference_source=request.reference_source,
            aspect_ratio=request.aspect_ratio,
            output=rendered_video_path,
            master_audio_start=common_start,
            width=width,
            height=height,
            transitions=(
                group_transition_plan.transitions
                if group_transition_plan is not None
                else ()
            ),
        )
        write_plan(plan, render_plan_path)
        reporter.status("연속 기준 음원으로 교차편집 영상을 합성하는 중…")
        codec = render(plan)
        render_payload = {
            "video_path": str(rendered_video_path),
            "plan_path": str(render_plan_path),
            "video_codec": codec,
            "selection_basis": (
                visual_timeline.analysis_basis
                if visual_timeline is not None
                else group_draft.selection_basis
            ),
            "face_match": (
                {
                    "analysis_basis": face_match_plan.analysis_basis,
                    "applied_count": face_match_plan.applied_count,
                    "fallback_count": face_match_plan.fallback_count,
                    "count_basis": "selected_segments",
                    "aligned_segment_count": (
                        face_match_plan.aligned_segment_count
                    ),
                    "raw_segment_count": face_match_plan.raw_segment_count,
                }
                if face_match_plan is not None
                else None
            ),
            "group_transitions": (
                group_transition_plan.to_dict(plan.fps)
                if group_transition_plan is not None
                else None
            ),
        }
        reporter.progress(0.99, "교차편집 영상 합성 완료")
    elif request.edit_mode is EditMode.GROUP:
        warnings.append("안전하게 정렬된 소스 2개와 박자 분석이 필요해 영상 초안을 만들지 않았습니다.")

    member_range_ready = (
        member_anchor_trim_payload is not None
        and bool(member_anchor_trim_payload["usable_for_render"])
    )
    if request.edit_mode is EditMode.MEMBER and group_ready and member_range_ready:
        assert request.target_member is not None
        reporter.progress(0.98, "멤버 교차직캠 타임라인을 만드는 중…")
        offsets = tuple(
            alignment.alignments[index].offset_seconds for index in eligible_indices
        )
        sync_scores = tuple(
            alignment.alignments[index].score for index in eligible_indices
        )
        scene_cut_times = tuple(
            tuple(
                float(change["timestamp_seconds"])
                for change in scene_payloads[index]["changes"]
            )
            for index in eligible_indices
        )
        phrase_times = tuple(
            sorted(
                {
                    float(time_value)
                    for phrase in rhythm_analysis.phrase_candidates
                    for time_value in (phrase.start_seconds, phrase.end_seconds)
                }
            )
        )
        timing_draft = build_group_draft(
            source_count=len(eligible_indices),
            common_master_start=member_common_start,
            common_master_end=common_end,
            audio_offsets=offsets,
            sync_scores=sync_scores,
            beat_times=rhythm_analysis.beat_times_seconds,
            phrase_times=phrase_times,
            scene_cut_times=scene_cut_times,
        )
        aspect_ratios: list[float] = []
        for index in eligible_indices:
            video_stream = media[index].video_streams[0]
            if not video_stream.width or not video_stream.height:
                raise ValueError("멤버 크롭을 위한 원본 영상 해상도를 확인할 수 없습니다.")
            aspect_ratios.append(video_stream.width / video_stream.height)
        source_aspect_ratio = aspect_ratios[0]
        if any(abs(value - source_aspect_ratio) > 0.03 for value in aspect_ratios[1:]):
            raise ValueError("멤버 모드 원본 영상들의 화면 비율이 서로 다릅니다.")
        member_draft = build_member_draft(
            canonical_spans=timing_draft.segments,
            audio_offsets=offsets,
            tracking_by_source=tuple(
                member_tracking_results.get(index) for index in eligible_indices
            ),
            target_member=request.target_member,
            output_aspect=request.aspect_ratio,
            source_aspect_ratio=source_aspect_ratio,
        )
        unavailable = tuple(
            item
            for item in member_draft.segments
            if item.kind is not MemberSegmentKind.TARGET_CROP
        )
        if unavailable:
            warnings.append(
                f"대상 멤버를 안전하게 확인하지 못한 구간 {len(unavailable)}개가 있어 "
                "다른 멤버를 잘못 크롭하지 않도록 영상 합성을 중단했습니다. "
                "컷이 없는 멤버 직캠을 사용하거나 앵커 장면을 바꿔 주세요."
            )
        else:
            width, height = output_size(request.aspect_ratio)
            segments = tuple(
                aligned_render_segment(
                    source_index=eligible_indices[item.source_index],
                    reference_start=item.canonical_master_start,
                    common_master_start=member_common_start,
                    source_offset=item.audio_offset_seconds,
                    duration=item.duration_seconds,
                    pan_crop=item.to_pan_crop_track(),
                )
                for item in member_draft.segments
                if item.source_index is not None
                and item.audio_offset_seconds is not None
            )
            rendered_video_path = request.output_directory / (
                f"{base_filename}-{_safe_slug(request.target_member.value)}-member-draft.mp4"
            )
            render_plan_path = request.output_directory / (
                f"{base_filename}-{_safe_slug(request.target_member.value)}-render-plan.json"
            )
            plan = RenderPlan(
                sources=request.sources,
                segments=segments,
                reference_source=request.reference_source,
                aspect_ratio=request.aspect_ratio,
                output=rendered_video_path,
                master_audio_start=member_common_start,
                width=width,
                height=height,
            )
            write_plan(plan, render_plan_path)
            reporter.status(
                f"{request.target_member.value} 멤버 교차직캠을 합성하는 중…"
            )
            codec = render(plan)
            render_payload = {
                "video_path": str(rendered_video_path),
                "plan_path": str(render_plan_path),
                "video_codec": codec,
                "selection_basis": _member_render_selection_basis(
                    member_draft.selection_diagnostics.policy
                ),
                "effective_master_start_seconds": member_common_start,
            }
            reporter.progress(0.99, "멤버 교차직캠 합성 완료")
    elif request.edit_mode is EditMode.MEMBER:
        if group_ready and not member_range_ready:
            warnings.append(
                "가장 늦은 멤버 앵커 뒤에 합성할 공통 구간이 "
                "2프레임보다 짧아 멤버 영상을 만들지 않았습니다. "
                "모든 소스에서 더 앞쪽 장면을 앵커로 지정해 주세요."
            )
        else:
            warnings.append(
                "안전하게 정렬된 소스 2개와 박자 분석이 필요해 멤버 영상을 만들지 않았습니다."
            )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": timestamp.isoformat(),
        "request": request.as_dict(),
        "source_index_mapping": _source_index_mapping(
            request.sources,
            eligible_indices,
        ),
        "media": [info.to_dict() for info in media],
        "alignment": alignment.to_dict(),
        "alignment_eligibility": alignment_eligibility,
        "common_master_range": {
            "start_seconds": common_start,
            "end_seconds": common_end,
            "duration_seconds": common_end - common_start,
        },
        "scene_analysis": scene_payloads,
        "rhythm_analysis": rhythm_payload,
        "member_tracking": member_tracking_payloads,
        "member_anchor_trim": member_anchor_trim_payload,
        "group_draft": (
            _with_index_domain(
                group_draft.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if group_draft is not None
            else None
        ),
        "visual_timeline": (
            _with_index_domain(
                visual_timeline.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if visual_timeline is not None
            else None
        ),
        "face_match": (
            _with_index_domain(
                face_match_plan.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if face_match_plan is not None
            else None
        ),
        "group_transitions": (
            group_transition_plan.to_dict(29.97)
            if group_transition_plan is not None
            else None
        ),
        "member_draft": (
            _with_index_domain(
                member_draft.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if member_draft is not None
            else None
        ),
        "render": render_payload,
        "warnings": warnings,
        "capabilities": _capabilities_payload(
            group_ready=group_ready,
            edit_mode=request.edit_mode,
            member_draft=member_draft,
            rendered_video_path=rendered_video_path,
        ),
    }
    filename = f"{base_filename}-analysis.json"
    manifest_path = request.output_directory / filename
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reporter.progress(1.0, "분석 결과를 저장했습니다.")
    output_paths = [manifest_path]
    if render_plan_path is not None:
        output_paths.append(render_plan_path)
    if rendered_video_path is not None:
        output_paths.append(rendered_video_path)
    return AnalysisArtifact(
        manifest_path=manifest_path,
        manifest=manifest,
        output_paths=tuple(output_paths),
    )
