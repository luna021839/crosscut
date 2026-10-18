"""Project genuine MEMBER source geometry into a real output-aspect crop.

Screen joints and torso-normalized shape joints are different inputs. Only
the former are projected. The original source observation and certificates
remain unchanged; this derived view never becomes a new sensor observation.
"""
from dataclasses import dataclass,field
import numpy as np
from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut.render import NormalizedCrop
from .aspect_viewport import project_points,_dimensions


def compose_camera(base,camera):
    """Compose a base aspect crop and a feasible zoom/pan in that base view."""
    if not isinstance(base,NormalizedCrop):raise TypeError('actual source crop required')
    value=np.asarray(camera,float)
    if value.shape!=(3,) or not np.isfinite(value).all() or value[0]<1.:
        raise ValueError('finite zoom of at least one and a base-view center required')
    zoom,x,y=value;half=.5/zoom
    if min(x,y)<half-1e-12 or max(x,y)>1.-half+1e-12:
        raise ValueError('camera escapes the actual base viewport')
    return NormalizedCrop(base.x+base.width*(x-half),base.y+base.height*(y-half),
        base.width/zoom,base.height/zoom)


def project_box(box,crop):
    if box is None:return None
    x,y,w,h=box;first=project_points([x,y],crop)
    return (float(first[0]),float(first[1]),float(w/crop.width),float(h/crop.height))


@dataclass(frozen=True)
class ViewportFocalObservation:
    source: FocalSelectorGeometry=field(repr=False)
    crop: NormalizedCrop
    output_size: tuple[int,int]
    face_bbox_xywh: tuple | None
    person_bbox_xywh: tuple | None
    pose_screen_xyc: tuple
    native_face_height_pixels: float | None

    @property
    def metric_pose_xyc(self):
        """Isotropic image-plane units: output height is one, not16:9 always."""
        aspect=self.output_size[0]/self.output_size[1]
        return tuple((x*aspect,y,c) for x,y,c in self.pose_screen_xyc)

    @property
    def face_visible_fraction(self):
        if self.face_bbox_xywh is None:return None
        x,y,w,h=self.face_bbox_xywh
        area=max(0.,min(1.,x+w)-max(0.,x))*max(0.,min(1.,y+h)-max(0.,y))
        return float(area/(w*h))


def project(observation,crop,*,source_size,output_size):
    if not isinstance(observation,FocalSelectorGeometry):raise TypeError('source-bound focal geometry required')
    width,height=_dimensions(source_size);out_width,out_height=_dimensions(output_size)
    if not isinstance(crop,NormalizedCrop):raise TypeError('actual rendered source crop required')
    ratio=crop.width*width/(crop.height*height)
    if not np.isclose(ratio,out_width/out_height,rtol=1e-6,atol=0.):
        raise ValueError('viewport must preserve the requested physical output aspect')
    raw=np.asarray(observation.pose_screen_xyc,float).reshape(-1,3)
    points=project_points(raw[:,:2],crop)
    pose=tuple((float(x),float(y),float(c)) for (x,y),c in zip(points,raw[:,2],strict=True))
    return ViewportFocalObservation(observation,crop,output_size,project_box(observation.face_bbox_xywh,crop),
        project_box(None if observation.person is None else observation.person.bbox_xywh,crop),pose,
        None if observation.face_bbox_xywh is None else float(observation.face_bbox_xywh[3]*height))
