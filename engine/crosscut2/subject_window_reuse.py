"""Retrieve an immutable source window by actual seed overlap, not filename.

Neighbouring music instants can reuse observed native pixels even when the
old cache key was a different cut proposal. Tracks, producer metadata and raw
points are not renamed or interpolated. Conflicting overlapping tracks cannot
win by convenient file order.
"""
import numpy as np
from .temporal_face_chain import _bounds,_iou


def retrieve(windows,*,source,seed_native,seed_head,needed):
    matches=[];needed=np.asarray(needed,int);box=_bounds(seed_head)
    if box is None:return None
    for path,window in windows:
        if window.source!=source or not window.ordinals[0]<=seed_native<=window.ordinals[-1]:continue
        at=int(seed_native-window.ordinals[0])
        if (not window.associated[at] or _iou(box,_bounds(window.heads[at]))<.7
                or needed.min()<window.ordinals[0] or needed.max()>window.ordinals[-1]):continue
        matches.append((path,window))
    if not matches:return None
    matches.sort(key=lambda v:(-len(v[1].ordinals),str(v[0])))
    path,window=matches[0]
    for _,other in matches[1:]:
        first=max(int(window.ordinals[0]),int(other.ordinals[0]));last=min(int(window.ordinals[-1]),int(other.ordinals[-1]))
        a=np.arange(first,last+1)-window.ordinals[0];b=np.arange(first,last+1)-other.ordinals[0]
        common=window.associated[a]&other.associated[b];joints=window.support[a]&other.support[b]&common[:,None]
        if (not np.array_equal(window.heads[a][common],other.heads[b][common])
                or not np.array_equal(window.vectors[a][common],other.vectors[b][common])
                or not np.array_equal(window.body[a][joints],other.body[b][joints])):return None
    return path,window


def covered_offsets(left,right,lc,rc,candidates,*,radius=6):
    result=[]
    for offset in candidates:
        selected=(lc.output_frames>=offset-radius)&(lc.output_frames<=offset+radius)
        if not selected.any():continue
        if all(window.ordinals[0]<=clock.native_indices[selected].min()
               and window.ordinals[-1]>=clock.native_indices[selected].max()
               for window,clock in ((left,lc),(right,rc))):result.append(offset)
    return result
