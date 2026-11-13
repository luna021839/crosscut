"""Observed relative MEMBER roll and clockwise renderer-space containment.

Roll is an optional repair of a coherent screen-pose discrepancy, not a claim
about the world's horizon. No pose, missing joint, or angle is imputed. All
limits here are admission requirements additional to existing camera/QA gates.
"""
from __future__ import annotations

import math
import numpy as np

from .joint_dissolve_pose import OverlapPoseObservation, paired_overlap_pose_arrays

MEMBER_ROLL_CONTRACT = 'observed_focal_screen_relative_roll_v1'
MAXIMUM_MEMBER_ROLL_DEGREES = 1.6


def roll_ramp_safe_zoom(degrees):
    """Overscan for the entire neutral-to-roll ramp, not only its endpoint.

    The renderer's edge margin changes slope at one degree. Include that knot
    and conservatively bound the linear zoom ramp; the renderer still checks
    every keyframe/interpolation and may reject a panned camera.
    """
    from .render import minimum_roll_safe_zoom
    fractions={i/128 for i in range(1,129)}
    if abs(degrees)>1.:
        fractions.add(1/abs(degrees))
    return 1.+max((minimum_roll_safe_zoom(degrees*f)-1)/f for f in fractions)+.0001


def _size(output_size):
    if (len(output_size) != 2 or any(type(v) is not int or v <= 0 for v in output_size)):
        raise ValueError('roll projection requires positive pixel dimensions')
    return np.asarray(output_size, dtype=float)


def rotate_points(points, degrees, *, output_size):
    """FFmpeg clockwise rotation about the aspect-fill canvas centre."""
    size = _size(output_size)
    values = np.asarray(points, dtype=float)
    if values.shape[-1:] != (2,) or not np.isfinite(values).all() or not math.isfinite(degrees):
        raise ValueError('roll points and angle must be finite')
    a = math.radians(degrees)
    matrix = np.array(((math.cos(a),-math.sin(a)),(math.sin(a),math.cos(a))))
    return ((values-.5)*size) @ matrix.T / size + .5


def project_points(points, state, *, output_size):
    """Rotate first, then zoom/crop, matching the production filter order."""
    rotated = rotate_points(points, state.roll_degrees, output_size=output_size)
    return (rotated-(state.center_x,state.center_y))*state.zoom+.5


def _box(box):
    values = tuple(box)
    if (len(values) != 4 or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) for v in values) or min(values[2:]) <= 0):
        raise ValueError('roll containment requires a finite positive observed box')
    return values


def box_polygon(box):
    x,y,w,h = _box(box)
    return np.array(((x,y),(x+w,y),(x+w,y+h),(x,y+h)), dtype=float)


def project_box(box, state, *, output_size):
    polygon = project_points(box_polygon(box),state,output_size=output_size)
    low,high = polygon.min(axis=0),polygon.max(axis=0)
    return tuple(float(v) for v in (*low,*(high-low)))


def _area(points):
    if len(points)<3:
        return 0.
    p = np.asarray(points)
    return float(abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2)


def _clip(polygon):
    """Sutherland-Hodgman clipping against the visible unit viewport."""
    points = list(polygon)
    for axis,bound,sign in ((0,0.,1),(0,1.,-1),(1,0.,1),(1,1.,-1)):
        previous_points,points = points,[]
        if not previous_points:
            break
        previous = previous_points[-1]
        previous_inside = sign*(previous[axis]-bound)>=0
        for current in previous_points:
            inside = sign*(current[axis]-bound)>=0
            if inside != previous_inside:
                fraction = (bound-previous[axis])/(current[axis]-previous[axis])
                points.append(previous+fraction*(current-previous))
            if inside:
                points.append(current)
            previous,previous_inside=current,inside
    return points


def box_visible_fraction(box, state, *, output_size):
    """Visible area of the rotated quadrilateral, never of its axis-aligned hull.

    Crop to the original aspect-fill canvas first: rotation cannot recover
    pixels that were already discarded outside that canvas.
    """
    polygon = box_polygon(box)
    original_area = _area(polygon)*state.zoom**2
    source_visible = _clip(polygon)
    if not source_visible:
        return 0.
    projected = project_points(source_visible,state,output_size=output_size)
    return min(1.,max(0.,_area(_clip(projected))/original_area))


def estimate_pair_roll(observations, *, fps, output_size):
    """Return the measured clockwise correction RIGHT->LEFT, or None.

    At least three adjacent paired frames and six common high-confidence
    screen joints, including both shoulders and hips, must agree. A similarity
    fit must explain the whole pose and each torso axis; articulated limb
    differences, reflections, inconsistent motion and noise cannot author roll.
    """
    audit = dict(contract=MEMBER_ROLL_CONTRACT,accepted=False,quality_approval=False,
                 missing_joints_imputed=False,output_size=list(output_size))
    def reject(reason):
        return None,dict(audit,reason=reason)
    size = _size(output_size)
    if type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0:
        raise ValueError('roll evidence fps must be finite and positive')
    if (observations is None or len(observations)!=2
            or not 3<=len(observations[0])<=64 or len(observations[0])!=len(observations[1])
            or any(p is None for seq in observations for p in seq)):
        return reject('adjacent_focal_pose_window_missing')
    for seq in observations:
        if any(not isinstance(p,OverlapPoseObservation) for p in seq):
            raise TypeError('roll observations must be source-bound screen poses')
        if any(p.source_index!=seq[0].source_index for p in seq):
            raise ValueError('roll pose source changed inside a window')
        if any(abs((b.master_seconds-a.master_seconds)-1/fps)>1e-7
               or b.source_seconds<=a.source_seconds for a,b in zip(seq,seq[1:])):
            return reject('nonadjacent_or_nonmonotonic_pose_clock')
    if observations[0][0].source_index==observations[1][0].source_index:
        return reject('same_source_pose')
    if any(abs(a.master_seconds-b.master_seconds)>1e-7
           for a,b in zip(*observations)):
        raise ValueError('roll paired poses use different master clocks')
    paired_overlap_pose_arrays(observations,observations)
    values = np.array([[p.points_xyc for p in seq] for seq in observations])
    visible = ((values[:,:,:,2]>=.5)&(values[:,:,:,:2]>=0).all(axis=-1)
               &(values[:,:,:,:2]<=1).all(axis=-1)).all(axis=(0,1))
    indices = np.flatnonzero(visible).tolist()
    audit.update(common_joint_indices=indices,source_indices=[seq[0].source_index for seq in observations],
        master_seconds=[p.master_seconds for p in observations[0]],
        source_seconds=[[p.source_seconds for p in seq] for seq in observations])
    if len(indices)<6 or not {0,1,6,7}<=set(indices):
        return reject('insufficient_confident_torso_and_limb_joints')
    points = values[:,:,visible,:2]*size
    centred = points-points.mean(axis=2,keepdims=True)
    norms = np.sqrt(np.mean(np.sum(centred**2,axis=-1),axis=-1))
    if np.min(norms)<3.:
        return reject('pose_too_small')
    unit = centred/norms[:,:,None,None]
    for cloud in unit.reshape(-1,len(indices),2):
        eigen = np.linalg.eigvalsh(cloud.T @ cloud)
        if eigen[0] < .02*eigen[-1]:
            return reject('degenerate_pose_extent')
    angles=[];residuals=[];before=[]
    for a,b in zip(*unit):
        if np.linalg.det(b.T @ a)<=0:
            return reject('reflected_or_ambiguous_pose')
        angle=math.atan2(float(np.sum(b[:,0]*a[:,1]-b[:,1]*a[:,0])),float(np.sum(a*b)))
        angles.append(math.degrees(angle))
        c,s=math.cos(angle),math.sin(angle)
        projected=b @ np.array(((c,s),(-s,c)))
        residuals.append(float(np.sqrt(np.mean(np.sum((a-projected)**2,axis=-1)))))
        before.append(float(np.sqrt(np.mean(np.sum((a-b)**2,axis=-1)))))
    angle=float(np.median(angles))
    audit.update(relative_correction_degrees=angle,angle_range_degrees=max(angles)-min(angles),
        pose_rms_before=max(before),pose_rms_after=max(residuals))
    if not .8<=abs(angle)<=2*MAXIMUM_MEMBER_ROLL_DEGREES:
        return reject('angle_below_evidence_floor_or_above_renderer_cap')
    if max(abs(a-angle) for a in angles)>.25:
        return reject('temporally_inconsistent_tilt')
    if max(residuals)>.03 or any(after>.8*raw for after,raw in zip(residuals,before)):
        return reject('articulated_pose_not_explained_by_rotation')
    # Both shoulder/hip lines and the shoulder-to-hip axis must independently
    # support the same angle; one rotating wrist cannot move the whole camera.
    raw = values[:,:,:,:2]*size
    axes = [raw[:,:,1]-raw[:,:,0], raw[:,:,7]-raw[:,:,6],
            (raw[:,:,6]+raw[:,:,7]-raw[:,:,0]-raw[:,:,1])/2]
    for vectors in axes:
        for a,b in zip(*vectors):
            if min(np.linalg.norm(a),np.linalg.norm(b))<3.:
                return reject('unresolved_torso_axis')
            delta=math.degrees(math.atan2(float(b[0]*a[1]-b[1]*a[0]),float(a@b)))
            if abs(delta-angle)>.5:
                return reject('torso_axes_disagree_with_roll')
    # Compare measured shape velocity in a common orientation. Translation and
    # uniform scale are removed, but genuine opposite articulation is retained.
    rad=math.radians(angle);c,s=math.cos(rad),math.sin(rad)
    aligned=unit[1] @ np.array(((c,s),(-s,c)))
    left_motion=np.diff(unit[0],axis=0);right_motion=np.diff(aligned,axis=0)
    for a,b in zip(left_motion,right_motion):
        lengths=(np.linalg.norm(a),np.linalg.norm(b))
        if max(lengths)>.003:
            if min(lengths)<.5*max(lengths) or float(np.sum(a*b))<.8*lengths[0]*lengths[1]:
                return reject('incompatible_observed_articulation_motion')
    return angle,dict(audit,accepted=True,reason='coherent_observed_relative_screen_roll')


def compose_roll_state(base, roll_degrees, *, pivot, zoom_multiplier, output_size, time_seconds):
    """Add observed roll while retaining the previous output position of a pivot."""
    from .render import EditorialTransformKeyframe
    if abs(base.roll_degrees)>1e-9:
        raise ValueError('member roll cannot overwrite an existing roll owner')
    rotated=rotate_points((pivot,),roll_degrees,output_size=output_size)[0]
    old_position=(np.asarray(pivot)-(base.center_x,base.center_y))*base.zoom+.5
    zoom=base.zoom*zoom_multiplier
    centre=rotated-(old_position-.5)/zoom
    return EditorialTransformKeyframe(time_seconds,zoom,float(centre[0]),float(centre[1]),roll_degrees)
