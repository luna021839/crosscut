"""Measured feature paths constrained by instance AND semantic-part rasters.

This is a distinct prediction-supported local surface basis. The legacy joint
patch API is not upgraded into part ownership. A mask prediction, even with
pixel flow, is not a named identity, complete extent or cut quality receipt.
"""
from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True, slots=True)
class NativePartSurfaceSupport:
    request_source_index: int
    native_frame_index: int
    source_clock_sha256: str
    native_pixel_sha256: str
    subject_fragment_reference: str
    part: str
    instance_support: np.ndarray
    semantic_part_support: np.ndarray
    competing_subject_support: tuple[tuple[str, np.ndarray], ...]
    instance_prediction_sha256: str
    semantic_prediction_sha256: str

    def __post_init__(self):
        if (any(type(n) is not int or n < 0 for n in (self.request_source_index,self.native_frame_index))
                or any(not _sha(v) for v in (self.source_clock_sha256,self.native_pixel_sha256,
                    self.instance_prediction_sha256,self.semantic_prediction_sha256))
                or not isinstance(self.subject_fragment_reference, str) or not self.subject_fragment_reference
                or self.part not in ('head_surface', 'left_arm_surface', 'right_arm_surface')
                or not isinstance(self.instance_support, np.ndarray) or self.instance_support.dtype != np.bool_
                or self.instance_support.ndim != 2 or min(self.instance_support.shape) < 3
                or not isinstance(self.semantic_part_support, np.ndarray)
                or self.semantic_part_support.dtype != np.bool_
                or self.semantic_part_support.shape != self.instance_support.shape
                or not isinstance(self.competing_subject_support, tuple)):
            raise ValueError('surface flow needs bound native instance/part rasters, not ROI box ownership')
        refs = []
        for ref, mask in self.competing_subject_support:
            if (not isinstance(ref, str) or not ref or ref == self.subject_fragment_reference or ref in refs
                    or not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
                    or mask.shape != self.instance_support.shape):
                raise ValueError('every competing subject support must retain its unique reference/native raster')
            refs.append(ref)
        for name in ('instance_support','semantic_part_support'):
            mask = getattr(self, name).copy()
            mask.setflags(write=False)
            object.__setattr__(self, name, mask)
        values = []
        for ref, mask in self.competing_subject_support:
            mask = mask.copy();mask.setflags(write=False);values.append((ref,mask))
        object.__setattr__(self, 'competing_subject_support', tuple(values))

    @property
    def witness(self):
        return dict(request_source_index=self.request_source_index,native_frame_index=self.native_frame_index,
            source_clock_sha256=self.source_clock_sha256,native_pixel_sha256=self.native_pixel_sha256,
            fragment_reference=self.subject_fragment_reference,part=self.part,
            instance_prediction_sha256=self.instance_prediction_sha256,
            semantic_prediction_sha256=self.semantic_prediction_sha256,
            instance_mask_sha256=hashlib.sha256(memoryview(self.instance_support)).hexdigest(),
            part_mask_sha256=hashlib.sha256(memoryview(self.semantic_part_support)).hexdigest(),
            competing_masks=[dict(reference=ref,sha256=hashlib.sha256(memoryview(mask)).hexdigest())
                for ref,mask in self.competing_subject_support])

    def memberships(self, point):
        height,width = self.instance_support.shape
        x,y = point
        if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x < width or not 0 <= y < height:
            return dict(on_native=False,subject=False,part=False,competing_subjects=[])
        # Conservative bilinear footprint: all four neighboring pixels must
        # belong to the subject and part. Any competing owner makes it unknown.
        xx,yy = math.floor(x),math.floor(y)
        right,bottom = min(xx+1,width-1),min(yy+1,height-1)
        points = ((yy,xx),(yy,right),(bottom,xx),(bottom,right))
        return dict(on_native=True,
            subject=all(bool(self.instance_support[a,b]) for a,b in points),
            part=all(bool(self.semantic_part_support[a,b]) for a,b in points),
            competing_subjects=[ref for ref,mask in self.competing_subject_support
                if any(bool(mask[a,b]) for a,b in points)])


def _membership_passes(value):
    return value['on_native'] and value['subject'] and value['part'] and not value['competing_subjects']


def measure_part_surface_paths(*, native_pixels, supports, maximum_features=96, minimum_features=3,
        maximum_forward_backward_error_pixels=1.25):
    """Keep every attempted feature and every actual native step, including failure.

    A caller supplies ALL intervening rasters, not endpoint interpolation.
    Features are selected only in subject ∩ part ∩ texture minus competitors.
    Both start and destination membership are checked on every LK step. A
    failed feature never becomes successful on a later frame. Source/camera
    identity and mask-model authority remain the owning worker's duty.
    """
    frames, rows = tuple(native_pixels),tuple(supports)
    if (not 2 <= len(frames) <= 14 or len(frames) != len(rows)
            or any(not isinstance(row, NativePartSurfaceSupport) for row in rows)
            or type(maximum_features) is not int or not 3 <= maximum_features <= 256
            or type(minimum_features) is not int or not 3 <= minimum_features <= maximum_features
            or type(maximum_forward_backward_error_pixels) not in (int,float)
            or not 0 < maximum_forward_backward_error_pixels <= 1.25):
        raise ValueError('part surface path needs explicit complete bounded native sensing')
    first = rows[0]
    for frame, row in zip(frames,rows,strict=True):
        if (not isinstance(frame,np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or frame.shape[:2] != first.instance_support.shape
                or row.instance_support.shape != first.instance_support.shape
                or (row.request_source_index,row.source_clock_sha256,row.subject_fragment_reference,row.part) !=
                    (first.request_source_index,first.source_clock_sha256,first.subject_fragment_reference,first.part)
                or hashlib.sha256(memoryview(np.ascontiguousarray(frame))).hexdigest() != row.native_pixel_sha256):
            raise ValueError('surface native pixels/source/clock/fragment/part binding differs')
    if any(b.native_frame_index != a.native_frame_index+1 for a,b in zip(rows,rows[1:])):
        raise ValueError('surface path must include each actual intervening native frame, without clones')
    mask = (first.instance_support & first.semantic_part_support).copy()
    for _, competing in first.competing_subject_support:
        mask &= ~competing
    # A texture corner's 3x3 derivative footprint must not straddle background.
    mask = cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    grayscale = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(grayscale,mask=mask.astype(np.uint8)*255,
        maxCorners=maximum_features,qualityLevel=.012,minDistance=3.,blockSize=3)
    originals = np.empty((0,2),np.float32) if corners is None else corners.reshape(-1,2)
    # Keep feature IDs stable, including rejected bilinear footprint seeds.
    attempts = [dict(feature_id=n,original_native_xy=p.tolist(),
        initial_membership=first.memberships(p),steps=[],survived=_membership_passes(first.memberships(p)))
        for n,p in enumerate(originals)]
    moving = originals.reshape(-1,1,2)
    opts = dict(winSize=(21,21),maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for index,(frame,row) in enumerate(zip(frames[1:],rows[1:]),1):
        if not len(moving): break
        current = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        forward,fs,_ = cv2.calcOpticalFlowPyrLK(grayscale,current,moving,None,**opts)
        backward,bs,_ = (cv2.calcOpticalFlowPyrLK(current,grayscale,forward,None,**opts)
            if forward is not None else (None,None,None))
        for n,item in enumerate(attempts):
            start = moving[n,0]
            end = None if forward is None else forward[n,0]
            back = None if backward is None else backward[n,0]
            finite = end is not None and back is not None and np.isfinite(end).all() and np.isfinite(back).all()
            error = float(np.linalg.norm(back-start)) if finite else None
            before = rows[index-1].memberships(start)
            after = row.memberships(end) if end is not None else dict(on_native=False,subject=False,part=False,competing_subjects=[])
            fstatus = bool(fs[n,0]) if fs is not None else False
            bstatus = bool(bs[n,0]) if bs is not None else False
            good = (finite and fstatus and bstatus and error <= maximum_forward_backward_error_pixels
                and _membership_passes(before) and _membership_passes(after))
            item['steps'].append(dict(from_native_frame=rows[index-1].native_frame_index,to_native_frame=row.native_frame_index,
                from_native_xy=start.tolist(),forward_native_xy=end.tolist() if end is not None and np.isfinite(end).all() else None,
                backward_native_xy=back.tolist() if back is not None and np.isfinite(back).all() else None,
                forward_status=fstatus,backward_status=bstatus,forward_backward_error_pixels=error,
                from_membership=before,to_membership=after,step_passed=bool(good)))
            item['survived'] = bool(item['survived'] and good)
        if forward is None:
            # Retain failure for every later native step as well; no restart.
            forward = moving.copy()
        moving = np.nan_to_num(forward,nan=-1.,posinf=-1.,neginf=-1.)
        grayscale = current
    surviving = [item for item in attempts if item['survived']]
    return dict(contract='instance_and_part_supported_all_native_feature_steps_v1',
        support_witnesses=[row.witness for row in rows],feature_attempts=attempts,
        surviving_feature_ids=[item['feature_id'] for item in surviving],
        measurement_status='observed_prediction_supported_surface_paths' if len(surviving) >= minimum_features else
            'unknown_insufficient_prediction_supported_texture_paths',
        maximum_forward_backward_error_pixels=maximum_forward_backward_error_pixels,
        minimum_features=minimum_features,instance_and_part_memberships_checked=True,
        mask_predictions_are_not_certified_ownership=True,whole_head_or_arm_extent_certified=False,
        named_identity_certified=False,cross_camera_motion_certified=False,quality_approved=False)
