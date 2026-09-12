"""Exact source-frame authority for face-hidden, decoded-output QA windows.

Every output body is detected again. Source-local tracking authorizes *who*,
never the output box or joints. Blended effect frames have no single source
owner and are deliberately outside this provider's contract.
"""
from dataclasses import dataclass, replace

from .body_pose_2d import BodyPose2DEvidence
from .direct_member_visual import _AttentionGeometry
from .pose import _formation_people_descriptor
from .post_render_perceptual_quality import (
    _parse_plan, _span_for_frame, _rendered_source_frame_index,
)
from .rendered_body_reference import (
    RenderedBodyMeasurement, build_source_body_reference,
    measure_referenced_output_body,
)


@dataclass(frozen=True, slots=True)
class RenderedBodyGeometry:
    """Output measurement, not a fabricated source-track certificate."""
    measurement: RenderedBodyMeasurement
    source_seconds: float
    source_index: int = 0
    face_bbox_xywh: None = None
    face_scale: None = None
    body_track: None = None
    pose_backend: str = 'rtmpose_body7'

    @property
    def person(self):
        return self.measurement.person

    @property
    def pose_screen_xyc(self):
        return self.measurement.pose.upper_body_xyc if self.measurement.pose else ()

    def to_dict(self):
        result = self.measurement.audit()
        result.pop('observed_output_pose_xyc', None)
        return result


def body_sample(sample, measurement):
    if measurement.reason != 'observed' or measurement.person is None or measurement.pose is None:
        raise ValueError('rendered body sample requires observed output pose')
    person = measurement.person
    return replace(sample,
        focal_geometry=RenderedBodyGeometry(measurement, sample.master_seconds),
        attention=_AttentionGeometry(person.bbox_xywh, person.centroid_xy,
            'source_authorized_rendered_body'))


class RenderedBodyWindowProvider:
    def __init__(self, render_plan, frame_provider, *, body_model_path,
                 pose_model_path, source_cache_arguments=None, source_provider=None,
                 body_execution_device='cpu'):
        (self.fps, _, _, self.output_frames, self.sources,
         self.spans) = _parse_plan(render_plan)
        self.transitions = tuple(render_plan['transitions'])
        self.frame_provider = frame_provider
        self.body_model_path = body_model_path
        self.pose_model_path = pose_model_path
        self.body_execution_device = body_execution_device
        self.source_cache_arguments = dict(source_cache_arguments or {})
        self._source = source_provider
        self._owned = source_provider is None
        self._cache = {}

    def _open(self):
        if self._source is None:
            from .source_first_focal_geometry import SourceFirstFocalGeometryProvider
            self._source = SourceFirstFocalGeometryProvider(self.sources,
                body_model_path=self.body_model_path, pose_model_path=self.pose_model_path,
                body_execution_device=self.body_execution_device,
                **self.source_cache_arguments)
        return self._source

    def measurement(self, output_frame):
        if (isinstance(output_frame, bool) or not isinstance(output_frame, int)
                or not 0 <= output_frame < self.output_frames):
            raise ValueError('output body frame is outside the render contract')
        span = _span_for_frame(self.spans, output_frame)
        if span.segment_index:
            transition = self.transitions[span.segment_index - 1]
            if (transition['kind'] != 'hard_cut' and
                    output_frame < span.start_frame + transition['frame_count']):
                return RenderedBodyMeasurement('blended_output_has_no_single_source_owner')
        return self._measurement_on_clock(output_frame,span,
            (output_frame-span.start_frame)/self.fps,cache_key=output_frame)

    def _measurement_on_clock(self,output_frame,span,relative_time,*,cache_key):
        source = self._open()
        sampler = source.faces._samplers[span.source_index]
        if sampler is None:
            return RenderedBodyMeasurement('source_frame_unavailable')
        source_frame = _rendered_source_frame_index(span,
            relative_time=relative_time,
            source_fps=sampler.frame_rate)
        seconds = source_frame / sampler.frame_rate
        # Validates source file identity even when our output measurement is cached.
        geometry = source.selector_geometry(span.source_index, seconds)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if (geometry.person is None or geometry.pose_backend != 'rtmpose_body7'
                or not geometry.pose_screen_xyc or
                (geometry.face_bbox_xywh is None and geometry.body_track is None)):
            return RenderedBodyMeasurement('source_target_body_unresolved')
        original = sampler.frame_at(seconds)
        output = self.frame_provider(output_frame / self.fps)
        if original is None or output is None:
            return RenderedBodyMeasurement('source_or_output_frame_unavailable')
        source_pose = BodyPose2DEvidence('observed', geometry.person.bbox_xywh,
            (original.shape[1], original.shape[0]), geometry.pose_screen_xyc)
        if source_pose.as_formation_person(geometry.person.detection_confidence) is None:
            return RenderedBodyMeasurement('source_target_pose_unobserved')
        reference = build_source_body_reference(original, geometry, source.encoder)
        people = _formation_people_descriptor(source.bodies.detector.infer(output),
            frame_width=output.shape[1], frame_height=output.shape[0],
            minimum_width=.015, minimum_height=.045, minimum_confidence=.35)[0]
        measured = measure_referenced_output_body(output, reference, people,
            source.encoder, source.pose)
        self._cache[cache_key] = measured
        return measured

    def endpoint_window(self,boundary_index,label):
        from .rendered_endpoint_component import (endpoint_component_clocks,
            EndpointComponentFrame,EndpointComponentPair)
        clocks=endpoint_component_clocks(self.spans,self.transitions,boundary_index,label)
        if clocks is None or any(c[3]<.90 for c in clocks):return None
        frames=[]
        for output_frame,span,relative_frames,weight in clocks:
            source=self._open()
            sampler=source.faces._samplers[span.source_index]
            if sampler is None:return None
            source_frame=_rendered_source_frame_index(span,relative_time=relative_frames/self.fps,
                source_fps=sampler.frame_rate)
            native_span=_span_for_frame(self.spans,output_frame)
            # The neighboring pure frame was usually measured by the ordinary
            # four-frame boundary QA already. Share that exact output result;
            # only the near-pure component needs an additional detector pass.
            pure_clock=(native_span.source_index==span.source_index and
                output_frame>=native_span.start_frame+native_span.incoming_transition_frames and
                _rendered_source_frame_index(native_span,
                    relative_time=(output_frame-native_span.start_frame)/self.fps,
                    source_fps=sampler.frame_rate)==source_frame)
            measured=self._measurement_on_clock(output_frame,span,relative_frames/self.fps,
                cache_key=output_frame if pure_clock else
                    ('endpoint_component',output_frame,span.source_index,source_frame))
            frames.append(EndpointComponentFrame(output_frame,span.source_index,source_frame,
                source_frame/sampler.frame_rate,weight,measured))
        return EndpointComponentPair(boundary_index,label,tuple(frames))

    def __call__(self, frame_indices):
        if len(frame_indices) != 4:
            raise ValueError('body motion correspondence requires four exact frames')
        return tuple(self.measurement(index) for index in frame_indices)

    def close(self):
        if self._owned and self._source is not None:
            self._source.close()
            self._source = None
        self._cache.clear()
