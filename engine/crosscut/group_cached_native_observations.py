"""Read existing source-local GROUP observations without rebuilding an edit.

These numeric rows are sensing inputs, not actor correspondence or a quality
gate. HOG fallback is not a body pose; a missing face/pose remains a usable
partial observation. The caller still owns actor association and seam scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
import time

from .render_frame_schedule import VerifiedRenderFrameClock
from .visual_feature_cache import PersistentVisualFeatureCache, DescriptorRequestIdentity
from .visual_score import FrameVisualDescriptor, _descriptor_from_cache_payload


@dataclass(frozen=True, slots=True)
class CachedNativeObservation:
    source_index: int
    source_sha256: str
    native_frame_index: int
    native_frame_rate: Fraction
    producer_fingerprint: str
    record_path: Path
    quality: float
    composition: float
    shot_type: str
    shot_scale: float
    shot_scale_basis: str
    face_bbox: tuple[float, ...] | None
    face_landmarks: tuple[tuple[float, float], ...] | None
    pose: tuple[float, ...] | None
    pose_confidence: float
    pose_backend: str
    people_count: int
    formation_people: tuple
    color: tuple[float, ...]
    obstruction: float
    clipping: float
    following_native_frame: int | None
    image_motion: tuple[float, float] | None

    @property
    def native_seconds(self) -> float:
        return float(self.native_frame_index / self.native_frame_rate)

    @property
    def pose_is_face_owned(self) -> bool:
        # Legacy dominant-person pose has no cross-person ownership proof.
        return False


def observation_from_cached_descriptor(
    *, source_index: int, source_sha256: str, source_clock: VerifiedRenderFrameClock,
    request: DescriptorRequestIdentity, descriptor: FrameVisualDescriptor,
    producer_fingerprint: str, record_path: Path,
) -> CachedNativeObservation:
    if type(source_index) is not int or not 0 <= source_index < 64:
        raise ValueError("source index must be explicit and nonnegative")
    for digest in (source_sha256, producer_fingerprint):
        if not isinstance(digest, str) or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("cached observation requires a source and producer digest")
    if not isinstance(source_clock, VerifiedRenderFrameClock):
        raise ValueError("native cached-frame admission needs its real verified source clock")
    if not isinstance(request, DescriptorRequestIdentity) or not isinstance(descriptor, FrameVisualDescriptor):
        raise TypeError("cached observation requires a validated descriptor and request")
    primary = request.primary
    if Fraction(primary.fps_numerator, primary.fps_denominator) != source_clock.frame_rate:
        raise ValueError("cached frame uses another native rate")
    if primary.frame_index >= source_clock.frame_count:
        raise ValueError("cached primary frame lies outside source coverage")
    following = request.following
    if following is not None and (
        Fraction(following.fps_numerator, following.fps_denominator) != source_clock.frame_rate
        or not primary.frame_index <= following.frame_index < source_clock.frame_count
    ):
        raise ValueError("cached motion pair lies outside its forward native clock")
    has_pose = (
        descriptor.pose_backend != "structure_hog_fallback"
        and len(descriptor.pose_embedding) == 60 and descriptor.pose_confidence > 0
    )
    return CachedNativeObservation(
        source_index, source_sha256, primary.frame_index, source_clock.frame_rate,
        producer_fingerprint, record_path, descriptor.quality_score, descriptor.composition_score,
        descriptor.shot_type.value, descriptor.shot_scale, descriptor.shot_scale_basis,
        descriptor.primary_face_bbox, descriptor.primary_face_landmarks_xy,
        descriptor.pose_embedding if has_pose else None,
        descriptor.pose_confidence if has_pose else 0., descriptor.pose_backend,
        descriptor.detected_person_count, descriptor.formation_people, descriptor.color_embedding,
        descriptor.foreground_obstruction_score, descriptor.highlight_clipping_ratio,
        following.frame_index if following is not None else None,
        descriptor.motion if following is not None else None,
    )


@dataclass(frozen=True, slots=True)
class CachedNativeInventory:
    observations: tuple[CachedNativeObservation, ...]
    inspected_records: int
    invalid_records: int
    option_mismatches: int
    duplicate_primary_frames: int


def load_cached_native_inventory(
    cache: PersistentVisualFeatureCache, *, source_index: int,
    source_clock: VerifiedRenderFrameClock, maximum_records: int = 32768,
    deadline_seconds: float = 90.,
) -> CachedNativeInventory:
    """Reuse only current or explicitly decode-equivalent cache generations.

    No nearest-frame interpolation, pixel decoding, detector calls or cache
    writes. The rate/coverage check concerns physical frame identity, not a
    requirement that uncertain editorial or actor measurements be discarded.
    A non-CFR caller must keep its timestamp sensing path instead of excluding
    the source because this particular native-index adapter is unavailable.
    """
    if type(source_index) is not int or not 0 <= source_index < 64:
        raise ValueError("source index must be explicit and nonnegative")
    if not isinstance(source_clock, VerifiedRenderFrameClock):
        raise ValueError("native inventory needs its verified source clock")
    if type(maximum_records) is not int or not 1 <= maximum_records <= 262144:
        raise ValueError("cached inventory record bound is invalid")
    if isinstance(deadline_seconds, bool) or not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 600:
        raise ValueError("cached inventory deadline must be positive and bounded")
    if not isinstance(cache, PersistentVisualFeatureCache):
        raise TypeError("cached inventory requires the existing validated source cache")
    cache.assert_storage_root_available()
    if not cache._source_is_unchanged():
        raise OSError("cached inventory source changed")
    started = time.monotonic()
    paths = []
    for generation, root in enumerate(cache._lookup_roots):
        records, exceeded = cache._preflight_locator_record_paths(
            root, maximum_records=maximum_records-len(paths))
        if exceeded:
            raise ValueError("cached inventory exceeds its declared record bound")
        paths.extend((generation, path) for path in records)
    selected = {}
    invalid = mismatches = duplicates = 0
    for generation, path in paths:
        if time.monotonic()-started > deadline_seconds:
            raise TimeoutError("cached inventory read deadline exceeded")
        try:
            request, payload = cache._read_record(path)
            descriptor = _descriptor_from_cache_payload(payload)
            if not (request.include_subject_geometry and request.include_formation_pose and request.include_pose):
                mismatches += 1
                continue
            row = observation_from_cached_descriptor(
                source_index=source_index, source_sha256=cache.source_sha256,
                source_clock=source_clock, request=request, descriptor=descriptor,
                producer_fingerprint=path.parent.parent.name, record_path=path)
        except (ValueError, TypeError, KeyError):
            invalid += 1
            continue
        # Fixed provenance preference, never a best-quality/pose cherry-pick.
        # Static current-generation measurements beat old or motion-pair rows;
        # all retained image motion still owns its exact following frame.
        rank = (generation, row.following_native_frame is not None,
                row.following_native_frame if row.following_native_frame is not None else -1,
                path.name)
        previous = selected.get(row.native_frame_index)
        if previous is not None:
            duplicates += 1
        if previous is None or rank < previous[0]:
            selected[row.native_frame_index] = (rank, row)
    cache.assert_storage_root_available()
    if not cache._source_is_unchanged():
        raise OSError("cached inventory source changed during reading")
    return CachedNativeInventory(
        tuple(value[1] for _, value in sorted(selected.items())),
        len(paths), invalid, mismatches, duplicates)
