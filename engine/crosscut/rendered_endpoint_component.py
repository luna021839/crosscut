"""Source-component evidence at the near-pure ends of a decoded dissolve.

Identity authorizes a detected output person, never the whole blended frame.
Actual shoulder/hip geometry measures camera-scale continuity independently
of detector box padding and neighboring people's aggregate silhouette.
"""
from dataclasses import dataclass
import math

from .render import (FACE_DISSOLVE_ALPHA_CURVE, _FACE_DISSOLVE_LOGISTIC_ENDPOINT,
                     _FACE_DISSOLVE_LOGISTIC_SPAN)
from .rendered_body_reference import RenderedBodyMeasurement
from .timeline import _upper_body_shape_mismatch


def incoming_dissolve_weight(relative_frame, frame_count):
    if (isinstance(relative_frame,bool) or not isinstance(relative_frame,int) or
            isinstance(frame_count,bool) or not isinstance(frame_count,int) or
            frame_count<3 or not 0<=relative_frame<=frame_count):
        raise ValueError('dissolve weight requires exact bounded output frames')
    p=1-relative_frame/frame_count
    return min(1.,max(0.,1.-(1./(1.+math.exp(-12*(p-.5)))-_FACE_DISSOLVE_LOGISTIC_ENDPOINT)/_FACE_DISSOLVE_LOGISTIC_SPAN))


def endpoint_component_clocks(spans, transitions, boundary_index, label):
    """Return (output frame, source span, relative output-frame offset, weight).

    Unknown curves and overlapping previous effects cannot inherit this clock.
    The left handle has its own trim origin; do not extrapolate nominal frame
    indices across its independent ceil-to-source-frame reset.
    """
    from dataclasses import replace
    if (isinstance(boundary_index,bool) or not isinstance(boundary_index,int)
            or not 0<=boundary_index<len(transitions) or label not in ('effect_entry','effect_exit')):
        raise ValueError('invalid endpoint component boundary')
    transition=transitions[boundary_index]
    count=transition.get('frame_count')
    if (transition.get('kind')!='face_dissolve' or transition.get('alpha_curve')!=FACE_DISSOLVE_ALPHA_CURVE
            or isinstance(count,bool) or not isinstance(count,int) or count<3):
        return None
    left,right=spans[boundary_index:boundary_index+2]
    if left.source_index==right.source_index or right.end_frame-right.start_frame<=count:
        return None
    if label=='effect_exit':
        return ((right.start_frame+count-1,right,count-1,incoming_dissolve_weight(count-1,count)),
                (right.start_frame+count,right,count,1.))
    if left.end_frame-left.start_frame<=left.incoming_transition_frames:
        return None
    handle=replace(left,source_start=left.source_start+left.duration*left.source_rate)
    return ((right.start_frame-1,left,left.end_frame-left.start_frame-1,1.),
            (right.start_frame,handle,0,1.-incoming_dissolve_weight(0,count)))


@dataclass(frozen=True)
class EndpointComponentFrame:
    output_frame: int
    source_index: int
    source_frame: int
    source_seconds: float
    component_weight: float
    measurement: RenderedBodyMeasurement

    def __post_init__(self):
        if (any(isinstance(v,bool) or not isinstance(v,int) or v<0
                for v in (self.output_frame,self.source_index,self.source_frame)) or
                isinstance(self.source_seconds,bool) or not math.isfinite(self.source_seconds) or self.source_seconds<0 or
                isinstance(self.component_weight,bool) or
                not math.isfinite(self.component_weight) or not 0<=self.component_weight<=1.000000001 or
                not isinstance(self.measurement,RenderedBodyMeasurement)):
            raise ValueError('invalid decoded endpoint component evidence')

    def audit(self):
        measurement=self.measurement.audit()
        measurement.pop('observed_output_pose_xyc',None)
        return dict(output_frame=self.output_frame,source_index=self.source_index,
            source_frame=self.source_frame,source_seconds=self.source_seconds,
            component_weight=self.component_weight,measurement=measurement,
            whole_frame_single_source_claim=False)


def _observed_torso(frame):
    return observed_torso_geometry(frame.measurement) if frame.component_weight>=.90 else None


def observed_torso_geometry(m):
    """The same four measured anatomical points, never detector box padding."""
    if (not isinstance(m,RenderedBodyMeasurement) or m.reason!='observed' or m.pose is None or m.person is None
            or m.similarity is None or not math.isfinite(m.similarity) or m.similarity<.68
            or (m.margin is not None and (not math.isfinite(m.margin) or m.margin<.06))
            or m.source_output_pose_mismatch is None or not math.isfinite(m.source_output_pose_mismatch)
            or m.source_output_pose_mismatch>.30 or m.candidate_count<1
            or m.pose.as_formation_person(m.person.detection_confidence) is None):
        return None
    points=tuple(m.pose.upper_body_xyc[j] for j in (0,1,6,7))
    if any(not all(math.isfinite(v) for v in p) or p[2]<.3 or not 0<=p[0]<1 or not 0<=p[1]<1 for p in points):
        return None
    center=tuple(sum(p[k] for p in points)/4 for k in (0,1))
    radius_squared=sum((p[0]-center[0])**2+(p[1]-center[1])**2 for p in points)/4
    if radius_squared<=1e-8:return None
    return dict(centroid_xy=center,scale_area_proxy=radius_squared,
        scale_measurement='same_four_observed_torso_joints_rms_squared',
        common_joint_indices=[0,1,6,7],minimum_confidence=min(p[2] for p in points))


@dataclass(frozen=True)
class EndpointComponentPair:
    boundary_index: int
    label: str
    frames: tuple[EndpointComponentFrame, EndpointComponentFrame]

    def __post_init__(self):
        if (isinstance(self.boundary_index,bool) or not isinstance(self.boundary_index,int)
                or self.boundary_index<0 or self.label not in ('effect_entry','effect_exit')
                or len(self.frames)!=2 or any(not isinstance(f,EndpointComponentFrame) for f in self.frames)):
            raise ValueError('invalid decoded endpoint component pair')
        a,b=self.frames
        if (b.output_frame!=a.output_frame+1 or a.source_index!=b.source_index
                or b.source_seconds<a.source_seconds or b.source_frame<a.source_frame):
            raise ValueError('endpoint component pair is not one adjacent source clock')

    def metrics(self):
        geometries=tuple(_observed_torso(f) for f in self.frames)
        if any(g is None for g in geometries):return None
        left,right=geometries
        pose=_upper_body_shape_mismatch(*(f.measurement.pose.as_formation_person(
            f.measurement.person.detection_confidence) for f in self.frames))
        if pose is None:return None
        return dict(attention_position_step=min(1.,math.dist(left['centroid_xy'],right['centroid_xy'])/.42),
            attention_scale_step=min(1.,abs(math.log(left['scale_area_proxy']/right['scale_area_proxy']))/math.log(3.2)),
            pose_step=pose,attention_geometry=dict(left=left,right=right))

    def audit(self):
        return dict(contract='decoded_dissolve_endpoint_target_component_v1',
            boundary_index=self.boundary_index,label=self.label,frames=[f.audit() for f in self.frames],
            whole_frame_single_source_claim=False,quality_approval=False)
