"""Source-clock-bound support for a short missing focal detection, not identity."""
from dataclasses import dataclass
import math
from .cut_anchored_containment import track_cut_handle,_iou


@dataclass(frozen=True)
class FocalPixelContainmentSupport:
    source_index: int
    source_seconds: float
    bbox_xywh: tuple[float,float,float,float]
    anchor_source_seconds: tuple[float,float]
    bidirectional_iou: float

    def __post_init__(self):
        if isinstance(self.source_index,bool) or not isinstance(self.source_index,int) or self.source_index<0:
            raise ValueError('pixel support source index is invalid')
        values=(self.source_seconds,*self.bbox_xywh,*self.anchor_source_seconds,self.bidirectional_iou)
        if len(self.bbox_xywh)!=4 or len(self.anchor_source_seconds)!=2 or not all(
                isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values):
            raise ValueError('pixel support geometry or clocks are invalid')
        x,y,w,h=self.bbox_xywh;a,b=self.anchor_source_seconds
        if min(x,y,a)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001 or not a<self.source_seconds<b or b-a>.65+1e-9 or not .75<=self.bidirectional_iou<=1:
            raise ValueError('pixel support lacks bracketed geometry agreement')


def measure_focal_pixel_containment(*, source_index, frames, source_times,
        focal_boxes, query_index, fps, native_scene_cut_seconds=()):
    """Both anchor boxes must already belong to the same source's focal bank.

Pixel tracking cannot select a person. It supplies containment only when the
two independently followed observed anchors agree at the exact missing frame.
    """
    if not 0<query_index<len(source_times)-1 or len(frames)!=len(source_times) or len(focal_boxes)!=len(frames):
        raise ValueError('pixel containment query must be bracketed')
    if focal_boxes[0] is None or focal_boxes[-1] is None:return None
    if any(source_times[0]<t<=source_times[-1] for t in native_scene_cut_seconds):return None
    known=tuple((box,) if box is not None else () for box in focal_boxes)
    forward=track_cut_handle(frames=frames,times=source_times,face_boxes=known,anchor_box=focal_boxes[0],anchor_at_end=False,output_fps=fps)
    backward=track_cut_handle(frames=frames,times=source_times,face_boxes=known,anchor_box=focal_boxes[-1],anchor_at_end=True,output_fps=fps)
    if not forward.complete or not backward.complete:return None
    a=forward.samples[query_index].box;b=backward.samples[query_index].box
    agreement=_iou(a,b)
    if agreement<.75:return None
    return FocalPixelContainmentSupport(source_index,source_times[query_index],
        tuple((x+y)/2 for x,y in zip(a,b)),(source_times[0],source_times[-1]),agreement)
