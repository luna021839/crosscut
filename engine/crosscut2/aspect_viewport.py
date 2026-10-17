"""Source-to-output geometry for mixed portrait/landscape MEMBER sources.

The returned rectangle is a real source crop, suitable for RenderSegment.crop
or PanCropKeyframe.crop. Projection uses that same rectangle. It never squashes
source pixels, invents face/body observations or grants a quality approval.
"""
import math

import numpy as np

from crosscut.render import NormalizedCrop


def _dimensions(size):
    if (not isinstance(size,tuple) or len(size)!=2
            or any(type(v) is not int or not 1<=v<=16384 for v in size)):
        raise ValueError('explicit positive integer image width and height required')
    return size


def aspect_crop(source_size,output_size,*,center=(.5,.5),zoom=1.):
    """Largest output-aspect source rectangle, then the requested real crop.

    Zoom is geometric only. Its resolution/comfort policy belongs to the
    caller, not an arbitrary aspect-based admission gate. Centers are clamped
    to the physical image, and the returned rectangle is the realized view.
    """
    width,height=_dimensions(source_size);out_width,out_height=_dimensions(output_size)
    if (type(zoom) not in (int,float) or not math.isfinite(zoom) or zoom<1.
            or not isinstance(center,tuple) or len(center)!=2
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in center)):
        raise ValueError('finite source-normalized center and zoom of at least one required')
    ratio=(out_width/out_height)/(width/height)
    crop_width,crop_height=min(1.,ratio)/zoom,min(1.,1./ratio)/zoom
    x=min(1.-crop_width,max(0.,center[0]-crop_width/2.))
    y=min(1.-crop_height,max(0.,center[1]-crop_height/2.))
    return NormalizedCrop(x,y,crop_width,crop_height)


def project_points(points,crop):
    """Map genuine source-normalized positions into the actual output view.

    Offscreen positions remain offscreen; clamping them would disguise a
    clipped arm or excluded face. Confidence/evidence is not changed here.
    """
    if not isinstance(crop,NormalizedCrop):raise TypeError('real normalized source crop required')
    value=np.asarray(points,float)
    if value.shape[-1:]!=(2,) or not np.isfinite(value).all():
        raise ValueError('finite source-normalized xy positions required')
    return (value-[crop.x,crop.y])/[crop.width,crop.height]


def native_sampling_capacity(source_size,output_size):
    """Maximum enlargement before adding interpolation beyond native pixels.

    A value below one describes an already undersized input. It does not
    exclude that source; the selector can still use its measured evidence.
    """
    width,height=_dimensions(source_size);out_width,out_height=_dimensions(output_size)
    crop=aspect_crop(source_size,output_size)
    return min(crop.width*width/out_width,crop.height*height/out_height)
