"""Deterministic timeline interchange exports for reviewed render plans.

The rendered MP4 remains authoritative because FCPXML, OTIO, and CMX 3600 do
not share one portable representation for CrossCut's sigmoid face dissolve,
dynamic crop, boundary look ramp, or per-shot rate correction.  These exports
preserve source identity and exact source/record clocks, carry the nonportable
decisions as metadata, and publish an explicit fidelity audit instead of
silently pretending that a lossy interchange file is the final render.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import quote
import xml.etree.ElementTree as ET

from .render import RenderPlan, TransitionKind, segment_frame_counts


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "reviewed_render_plan_interchange_bundle_v1"


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fps_fraction(fps: float) -> Fraction:
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("timeline interchange fps must be positive and finite")
    for candidate in (
        Fraction(24000, 1001),
        Fraction(30000, 1001),
        Fraction(60000, 1001),
    ):
        if abs(float(candidate) - fps) <= 1.0e-3:
            return candidate
    return Fraction(str(fps)).limit_denominator(1_000_000)


def _seconds_fraction(value: float) -> Fraction:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("timeline time must be non-negative and finite")
    return Fraction(str(value)).limit_denominator(1_000_000_000)


def _frame_seconds(frame_count: int, fps: Fraction) -> Fraction:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
        raise ValueError("timeline frame count must be a non-negative integer")
    return Fraction(frame_count * fps.denominator, fps.numerator)


def _fcpx_time(value: Fraction) -> str:
    if value == 0:
        return "0s"
    if value.denominator == 1:
        return f"{value.numerator}s"
    return f"{value.numerator}/{value.denominator}s"


def _file_url(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    return "file://" + quote(str(resolved), safe="/")


def _safe_name(value: str, fallback: str) -> str:
    normalized = re.sub(r"[\x00-\x1f/:]", "-", str(value)).strip()
    return normalized or fallback


def _segment_metadata(plan: RenderPlan, index: int) -> dict[str, Any]:
    segment = plan.segments[index]
    transition = plan.transitions[index - 1] if index > 0 else None
    return {
        "source_index": segment.source_index,
        "source_path": str(plan.sources[segment.source_index].resolve()),
        "source_start_seconds": segment.source_start,
        "source_duration_seconds": segment.effective_source_duration,
        "source_playback_rate": segment.source_playback_rate,
        "master_start_seconds": segment.master_start,
        "master_duration_seconds": segment.duration,
        "incoming_transition": (
            transition.to_dict(plan.fps) if transition is not None else None
        ),
        "crop": asdict(segment.crop) if segment.crop is not None else None,
        "pan_crop": asdict(segment.pan_crop) if segment.pan_crop is not None else None,
        "editorial_transform": (
            asdict(segment.editorial_transform)
            if segment.editorial_transform is not None
            else None
        ),
        "look_adjustment": (
            asdict(segment.look_adjustment)
            if segment.look_adjustment is not None
            else None
        ),
        "boundary_look_track": (
            asdict(segment.boundary_look_track)
            if segment.boundary_look_track is not None
            else None
        ),
    }


def _rational_time(value: float, rate: float) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "RationalTime.1",
        "value": value,
        "rate": rate,
    }


def _time_range(start: float, duration: float, rate: float) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "start_time": _rational_time(start, rate),
        "duration": _rational_time(duration, rate),
    }


def _external_media_references(
    source: Path,
    *,
    source_index: int,
) -> tuple[dict[str, Any], str]:
    key = "DEFAULT_MEDIA"
    return (
        {
            key: {
                "OTIO_SCHEMA": "ExternalReference.1",
                "name": source.name,
                "metadata": {"crosscut_source_index": source_index},
                "target_url": _file_url(source),
                "available_range": None,
                "available_image_bounds": None,
            }
        },
        key,
    )


def build_otio_payload(plan: RenderPlan, *, project_name: str) -> dict[str, Any]:
    """Build a standards-shaped OTIO JSON timeline with exact CrossCut metadata."""

    frame_counts = segment_frame_counts(plan)
    rate = float(_fps_fraction(plan.fps))
    video_children: list[dict[str, Any]] = []
    for index, (segment, frames) in enumerate(
        zip(plan.segments, frame_counts, strict=True)
    ):
        source = plan.sources[segment.source_index].resolve()
        media_references, active_media_reference_key = _external_media_references(
            source,
            source_index=segment.source_index,
        )
        metadata = _segment_metadata(plan, index)
        effects: list[dict[str, Any]] = []
        if not math.isclose(segment.source_playback_rate, 1.0, abs_tol=1.0e-9):
            effects.append(
                {
                    "OTIO_SCHEMA": "LinearTimeWarp.1",
                    "name": "CrossCut source clock correction",
                    "effect_name": "LinearTimeWarp",
                    "metadata": {"crosscut": metadata},
                    "time_scalar": segment.source_playback_rate,
                    "enabled": True,
                }
            )
        video_children.append(
            {
                "OTIO_SCHEMA": "Clip.2",
                "name": f"{index + 1:03d} {source.name}",
                "metadata": {"crosscut": metadata},
                "source_range": _time_range(
                    segment.source_start * rate,
                    frames,
                    rate,
                ),
                "effects": effects,
                "markers": [],
                "enabled": True,
                "color": None,
                "media_references": media_references,
                "active_media_reference_key": active_media_reference_key,
            }
        )
        # OTIO defines a LinearTimeWarp as changing how much source media is
        # played *inside* a fixed clip duration.  Therefore source_range uses
        # the canonical record-frame count; the effect's time_scalar and the
        # exact effective source duration in metadata preserve source clocks.
        metadata["record_frame_count"] = frames

    reference = plan.sources[plan.reference_source].resolve()
    audio_media_references, audio_active_media_reference_key = (
        _external_media_references(
            reference,
            source_index=plan.reference_source,
        )
    )
    audio_clip = {
        "OTIO_SCHEMA": "Clip.2",
        "name": "CrossCut master audio",
        "metadata": {
            "crosscut": {
                "source_index": plan.reference_source,
                "master_audio_start_seconds": plan.master_audio_start,
                "continuous_master_audio": True,
            }
        },
        "source_range": _time_range(
            plan.master_audio_start * rate,
            plan.output_frame_count,
            rate,
        ),
        "effects": [],
        "markers": [],
        "enabled": True,
        "color": None,
        "media_references": audio_media_references,
        "active_media_reference_key": audio_active_media_reference_key,
    }
    transition_metadata = [
        {
            "boundary_index": index,
            "record_frame": sum(frame_counts[: index + 1]),
            **transition.to_dict(plan.fps),
        }
        for index, transition in enumerate(plan.transitions)
    ]
    return {
        "OTIO_SCHEMA": "Timeline.1",
        "name": project_name,
        "metadata": {
            "crosscut": {
                "schema_version": SCHEMA_VERSION,
                "analysis_basis": ANALYSIS_BASIS,
                "authoritative_output": str(plan.output.resolve()),
                "fps": plan.fps,
                "width": plan.width,
                "height": plan.height,
                "aspect_ratio": plan.aspect_ratio.value,
                "output_frame_count": plan.output_frame_count,
                "transitions": transition_metadata,
                "nonportable_effects_preserved_in_clip_metadata": True,
            }
        },
        "global_start_time": _rational_time(0.0, rate),
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "name": "tracks",
            "metadata": {},
            "source_range": None,
            "effects": [],
            "markers": [],
            "enabled": True,
            "color": None,
            "children": [
                {
                    "OTIO_SCHEMA": "Track.1",
                    "name": "CrossCut video",
                    "metadata": {},
                    "source_range": None,
                    "effects": [],
                    "markers": [],
                    "enabled": True,
                    "color": None,
                    "children": video_children,
                    "kind": "Video",
                },
                {
                    "OTIO_SCHEMA": "Track.1",
                    "name": "Continuous master audio",
                    "metadata": {},
                    "source_range": None,
                    "effects": [],
                    "markers": [],
                    "enabled": True,
                    "color": None,
                    "children": [audio_clip],
                    "kind": "Audio",
                },
            ],
        },
    }


def build_fcpxml_bytes(plan: RenderPlan, *, project_name: str) -> bytes:
    """Build an FCPXML source/cut timeline plus exact CrossCut metadata."""

    fps = _fps_fraction(plan.fps)
    frame_counts = segment_frame_counts(plan)
    root = ET.Element("fcpxml", version="1.11")
    resources = ET.SubElement(root, "resources")
    format_id = "r1"
    ET.SubElement(
        resources,
        "format",
        id=format_id,
        name=f"CrossCut {plan.width}x{plan.height} {float(fps):.6f}p",
        frameDuration=_fcpx_time(_frame_seconds(1, fps)),
        width=str(plan.width),
        height=str(plan.height),
        colorSpace="1-1-1 (Rec. 709)",
    )
    maximum_end_by_source = [0.0] * len(plan.sources)
    for segment in plan.segments:
        maximum_end_by_source[segment.source_index] = max(
            maximum_end_by_source[segment.source_index], segment.source_end
        )
    asset_ids: list[str] = []
    for index, source in enumerate(plan.sources):
        asset_id = f"r{index + 2}"
        asset_ids.append(asset_id)
        duration = max(maximum_end_by_source[index], 1.0)
        ET.SubElement(
            resources,
            "asset",
            id=asset_id,
            name=source.name,
            start="0s",
            duration=_fcpx_time(_seconds_fraction(duration)),
            hasVideo="1",
            hasAudio="0",
            format=format_id,
            src=_file_url(source),
        )
    audio_id = f"r{len(plan.sources) + 2}"
    reference = plan.sources[plan.reference_source]
    ET.SubElement(
        resources,
        "asset",
        id=audio_id,
        name=f"{reference.name} master audio",
        start="0s",
        duration=_fcpx_time(
            _seconds_fraction(plan.master_audio_start + plan.render_duration)
        ),
        hasVideo="0",
        hasAudio="1",
        audioSources="1",
        audioChannels="2",
        audioRate="48000",
        src=_file_url(reference),
    )

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="CrossCut")
    project = ET.SubElement(event, "project", name=project_name)
    sequence = ET.SubElement(
        project,
        "sequence",
        format=format_id,
        duration=_fcpx_time(_frame_seconds(plan.output_frame_count, fps)),
        tcStart="0s",
        tcFormat="NDF",
        audioLayout="stereo",
        audioRate="48k",
    )
    spine = ET.SubElement(sequence, "spine")
    record_frame = 0
    for index, (segment, frames) in enumerate(
        zip(plan.segments, frame_counts, strict=True)
    ):
        duration = _frame_seconds(frames, fps)
        clip = ET.SubElement(
            spine,
            "asset-clip",
            name=f"{index + 1:03d} {plan.sources[segment.source_index].name}",
            ref=asset_ids[segment.source_index],
            offset=_fcpx_time(_frame_seconds(record_frame, fps)),
            start=_fcpx_time(_seconds_fraction(segment.source_start)),
            duration=_fcpx_time(duration),
            format=format_id,
        )
        metadata = ET.SubElement(clip, "metadata")
        ET.SubElement(
            metadata,
            "md",
            key="com.openai.crosscut.segment",
            value=json.dumps(
                _segment_metadata(plan, index),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        if index > 0:
            transition = plan.transitions[index - 1]
            marker = ET.SubElement(
                clip,
                "marker",
                start=_fcpx_time(_seconds_fraction(segment.source_start)),
                duration="1/1000s",
                value=f"CrossCut {transition.kind.value}",
            )
            marker.set("note", json.dumps(transition.to_dict(plan.fps), sort_keys=True))
        record_frame += frames

    ET.SubElement(
        spine,
        "asset-clip",
        name="Continuous master audio",
        ref=audio_id,
        lane="-1",
        offset="0s",
        start=_fcpx_time(_seconds_fraction(plan.master_audio_start)),
        duration=_fcpx_time(_frame_seconds(plan.output_frame_count, fps)),
        audioRole="music",
    )
    ET.indent(root, space="  ")
    return (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!DOCTYPE fcpxml>\n'
        + ET.tostring(root, encoding="utf-8")
        + b"\n"
    )


def _timecode(frame: int, nominal_fps: int) -> str:
    if frame < 0:
        raise ValueError("EDL frame cannot be negative")
    frames = frame % nominal_fps
    seconds_total = frame // nominal_fps
    seconds = seconds_total % 60
    minutes_total = seconds_total // 60
    minutes = minutes_total % 60
    hours = minutes_total // 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def build_edl_bytes(plan: RenderPlan, *, project_name: str) -> bytes:
    """Build a CMX 3600 video decision list with source-path comments."""

    nominal_fps = int(round(plan.fps))
    if nominal_fps not in {24, 25, 30, 50, 60}:
        raise ValueError("CMX 3600 export needs a conventional nominal frame rate")
    frame_counts = segment_frame_counts(plan)
    record_frame = nominal_fps * 60 * 60
    lines = [
        f"TITLE: {_safe_name(project_name, 'CROSSCUT').upper()}",
        "FCM: NON-DROP FRAME",
        "",
    ]
    for index, (segment, frames) in enumerate(
        zip(plan.segments, frame_counts, strict=True), start=1
    ):
        source_in = int(round(segment.source_start * nominal_fps))
        source_out = source_in + int(round(segment.effective_source_duration * nominal_fps))
        record_out = record_frame + frames
        transition = plan.transitions[index - 2] if index > 1 else None
        edit_code = "C"
        if transition is not None and transition.kind is TransitionKind.FACE_DISSOLVE:
            edit_code = f"D {transition.frame_count:03d}"
        reel = f"S{segment.source_index + 1:07d}"[-8:]
        lines.append(
            f"{index:03d}  {reel:<8} V     {edit_code:<5} "
            f"{_timecode(source_in, nominal_fps)} "
            f"{_timecode(source_out, nominal_fps)} "
            f"{_timecode(record_frame, nominal_fps)} "
            f"{_timecode(record_out, nominal_fps)}"
        )
        lines.append(f"* FROM CLIP NAME: {plan.sources[segment.source_index].name}")
        lines.append(f"* SOURCE FILE: {plan.sources[segment.source_index].resolve()}")
        lines.append(
            "* CROSSCUT: "
            + json.dumps(
                _segment_metadata(plan, index - 1),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        lines.append("")
        record_frame = record_out
    lines.extend(
        (
            f"* MASTER AUDIO SOURCE: {plan.sources[plan.reference_source].resolve()}",
            f"* MASTER AUDIO START SECONDS: {plan.master_audio_start:.9f}",
            "* NOTE: CrossCut sigmoid dissolve, crop, look, and camera tracks are metadata-only in CMX 3600.",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


@dataclass(frozen=True, slots=True)
class TimelineInterchangeBundle:
    fcpxml: bytes
    otio: bytes
    edl: bytes
    audit: bytes

    def payloads(self) -> Mapping[str, bytes]:
        return {
            "fcpxml": self.fcpxml,
            "otio": self.otio,
            "edl": self.edl,
            "audit": self.audit,
        }


def build_timeline_interchange_bundle(
    plan: RenderPlan,
    *,
    project_name: str,
) -> TimelineInterchangeBundle:
    name = _safe_name(project_name, "CrossCut")
    fcpxml = build_fcpxml_bytes(plan, project_name=name)
    otio = _pretty_json_bytes(build_otio_payload(plan, project_name=name))
    edl = build_edl_bytes(plan, project_name=name)
    audit_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_basis": ANALYSIS_BASIS,
        "project_name": name,
        "authoritative_output": str(plan.output.resolve()),
        "source_count": len(plan.sources),
        "segment_count": len(plan.segments),
        "output_frame_count": plan.output_frame_count,
        "fps": plan.fps,
        "continuous_master_audio": {
            "source_index": plan.reference_source,
            "start_seconds": plan.master_audio_start,
        },
        "fidelity": {
            "source_selection_and_record_timing": "portable",
            "source_clock_and_rate": "portable_or_explicit_metadata",
            "hard_cuts": "portable",
            "face_dissolve": "portable_duration_with_sigmoid_curve_in_metadata",
            "dynamic_crop_and_editorial_camera": "metadata_only",
            "look_adjustment_and_boundary_look": "metadata_only",
            "final_pixels": "authoritative_mp4_only",
        },
        "artifacts": {
            "fcpxml": {"sha256": _sha256(fcpxml), "size_bytes": len(fcpxml)},
            "otio": {"sha256": _sha256(otio), "size_bytes": len(otio)},
            "edl": {"sha256": _sha256(edl), "size_bytes": len(edl)},
        },
    }
    audit_payload["content_sha256"] = _sha256(
        _canonical_json_bytes({**audit_payload, "content_sha256": None})
    )
    return TimelineInterchangeBundle(
        fcpxml=fcpxml,
        otio=otio,
        edl=edl,
        audit=_pretty_json_bytes(audit_payload),
    )


def _write_no_replace_or_validate(path: Path, payload: bytes) -> str:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise FileExistsError(f"interchange destination is not a file: {destination}")
        if destination.read_bytes() != payload:
            raise FileExistsError(f"immutable interchange artifact differs: {destination}")
        return "validated_existing"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return "created"


def write_timeline_interchange_bundle(
    plan: RenderPlan,
    *,
    output_directory: Path,
    base_filename: str,
    project_name: str,
) -> tuple[Path, ...]:
    bundle = build_timeline_interchange_bundle(plan, project_name=project_name)
    root = Path(output_directory).expanduser().resolve()
    stem = _safe_name(base_filename, "crosscut")
    paths = (
        root / f"{stem}.fcpxml",
        root / f"{stem}.otio",
        root / f"{stem}.edl",
        root / f"{stem}-interchange-audit.json",
    )
    for path, payload in zip(paths, bundle.payloads().values(), strict=True):
        _write_no_replace_or_validate(path, payload)
    return paths


__all__ = [
    "ANALYSIS_BASIS",
    "SCHEMA_VERSION",
    "TimelineInterchangeBundle",
    "build_edl_bytes",
    "build_fcpxml_bytes",
    "build_otio_payload",
    "build_timeline_interchange_bundle",
    "write_timeline_interchange_bundle",
]
