"""Transient rendered-coordinate observations for a prescribed learned shot.

This uses the same crop/zoom/roll projection as the final source-retention QA.
It is descriptor-stage evidence, not a substitute for decoded-render QA. Raw
source feature caches are never overwritten with transformed observations.
"""
from dataclasses import asdict, dataclass
import json

from .post_render_perceptual_quality import _Span, _reference_view_for_span

PRESCRIBED_RENDER_GEOMETRY = "prescribed_render_geometry_v1"


@dataclass(frozen=True)
class PrescribedRenderClock:
    """The renderer's linear shot clock, not the earlier alignment estimate."""

    segment: object
    master_start: float

    @property
    def rate(self):
        return (self.segment.source_duration or self.segment.duration) / self.segment.duration

    def master_to_source(self, seconds):
        return self.segment.source_start + (seconds - self.master_start) * self.rate

    def source_to_master(self, seconds):
        return self.master_start + (seconds - self.segment.source_start) / self.rate


class PrescribedGeometrySampler:
    def __init__(self, sampler, segment, *, master_start):
        self.sampler = sampler
        self.segment = segment
        self.master_start = master_start
        self.source_clock = PrescribedRenderClock(segment, master_start)
        self.frame_rate = getattr(sampler, "frame_rate", 29.97)
        def serialized(value):
            return json.loads(json.dumps(asdict(value))) if value is not None else None
        self.span = _Span(
            segment_index=0, source_index=segment.source_index,
            start_frame=0, end_frame=1, run_index=0,
            source_start=segment.source_start,
            source_rate=(segment.source_duration or segment.duration) / segment.duration,
            duration=segment.duration, incoming_transition_frames=0,
            crop=serialized(segment.crop), pan_crop=serialized(segment.pan_crop),
            editorial_transform=serialized(segment.editorial_transform),
        )

    def frame_at(self, seconds):
        frame = self.sampler.frame_at(seconds)
        if frame is None:
            return None
        master = self.source_clock.source_to_master(seconds)
        relative = min(self.segment.duration, max(0.0, master - self.master_start))
        # StageEdit GROUP output is landscape. Preserve available working-frame
        # resolution instead of inventing detail by upscaling sparse features.
        width = frame.shape[1]
        return _reference_view_for_span(
            frame, self.span, relative_time=relative,
            size=(width, max(1, round(width * 9 / 16))),
        )

    def feature_descriptor_at(self, seconds, *, following_seconds=None,
                              include_subject_geometry=True, include_formation_pose=False):
        from .visual_score import describe_frame
        frame = self.frame_at(seconds)
        following = self.frame_at(following_seconds) if following_seconds is not None else None
        if frame is None or (following_seconds is not None and following is None):
            return None
        return describe_frame(frame, following_frame=following,
                              include_subject_geometry=include_subject_geometry,
                              include_formation_pose=include_formation_pose)

    def descriptor_at(self, seconds):
        return self.feature_descriptor_at(seconds)

    def transition_descriptor_at(self, seconds):
        return self.feature_descriptor_at(seconds)

    def sface_embedding_at(self, seconds, descriptor):
        from .visual_score import extract_sface_embedding
        box, landmarks = descriptor.primary_face_bbox, descriptor.primary_face_landmarks_xy
        if descriptor.face_detector_backend != "yunet" or box is None or landmarks is None or len(landmarks) != 5:
            return ()
        frame = self.frame_at(seconds)
        return () if frame is None else extract_sface_embedding(frame, bbox_xywh=box, landmarks_xy=landmarks)
