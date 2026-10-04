"""Original-output polyline support, not inferred anatomy or occlusion.

Native coordinates, original camera visibility and editorial requirements are
different facts. A partial arm keeps its observed visible vertices and bone
portions; it does not become a required whole arm or an empty context arm.
The immutable original owns this mask, never the camera being fitted.
"""
from math import isfinite

CONTRACT = 'original_output_visible_joint_polyline_support_v1'
ARM_INDICES = {'left_arm': (0, 2, 4), 'right_arm': (1, 3, 5)}


def _point(value):
    if len(value) < 2 or any(type(v) not in (int, float) or not isfinite(v) for v in value[:2]):
        raise ValueError('visible support needs finite measured native coordinates')
    return tuple(value[:2])


def _clip(a, b, bounds=(0., 0., 1., 1.)):
    """Parameter interval of an actual measured line inside the raster."""
    a, b = _point(a), _point(b)
    lo, hi = 0., 1.
    x0, y0, x1, y1 = bounds
    for axis, low, high in ((0, x0, x1), (1, y0, y1)):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-15:
            if not low <= a[axis] <= high:
                return None
            continue
        left, right = sorted(((low-a[axis])/delta, (high-a[axis])/delta))
        lo, hi = max(lo, left), min(hi, right)
        if lo > hi:
            return None
    return (lo, hi) if hi-lo > 1e-12 else None


def _mix(a, b, t):
    return tuple(x+(y-x)*t for x, y in zip(a, b, strict=True))


def measured_visible_arm_support(*, native_points, project_point, base_crop=None):
    """Only positively observed raster support can be visible or outside.

    These are point/polyline supports, not silhouettes. Base-cropped pixels
    never become recoverable through the editorial transform.
    """
    if len(native_points) != 3:
        raise ValueError('arm support needs its three actually measured joints')
    native = tuple(_point(p) for p in native_points)
    projected = tuple(_point(project_point(p)) for p in native)
    crop = (0., 0., 1., 1.) if base_crop is None else base_crop
    if (len(crop) != 4 or any(not isfinite(v) for v in crop)
            or crop[0] >= crop[2] or crop[1] >= crop[3]):
        raise ValueError('original support base crop is invalid')
    inside = lambda p, c: c[0] <= p[0] <= c[2] and c[1] <= p[1] <= c[3]
    joints = [i for i, (p, q) in enumerate(zip(native, projected, strict=True))
        if inside(p, crop) and inside(q, (0., 0., 1., 1.))]
    bones = []
    for i in (0, 1):
        source = _clip(native[i], native[i+1], crop)
        output = _clip(projected[i], projected[i+1])
        if source is not None and output is not None:
            lo, hi = max(source[0], output[0]), min(source[1], output[1])
            if hi-lo > 1e-12:
                bones.append(dict(bone_index=i, first_parameter=lo, end_parameter=hi))
    full = len(joints) == 3 and len(bones) == 2 and all(
        b['first_parameter'] == 0. and b['end_parameter'] == 1. for b in bones)
    return dict(contract=CONTRACT, visibility='visible' if full else 'partial' if joints or bones else 'outside',
        joint_indices=joints, bone_portions=bones, native_joint_xy=[list(p) for p in native],
        original_projected_joint_xy=[list(p) for p in projected],
        original_base_crop_xyxy=list(crop), anatomical_mask_certified=False,
        occlusion_inferred=False)


def validate_support(value):
    if (not isinstance(value, dict) or value.get('contract') != CONTRACT
            or value.get('visibility') not in ('visible', 'partial', 'outside')
            or value.get('anatomical_mask_certified') is not False
            or value.get('occlusion_inferred') is not False):
        raise ValueError('original visible support lost its measured contract')
    reconstructed = measured_visible_arm_support(native_points=value['native_joint_xy'],
        project_point=lambda p: value['original_projected_joint_xy'][value['native_joint_xy'].index(list(p))],
        base_crop=value['original_base_crop_xyxy'])
    # Coincident joints may share native coordinates. Their affine original
    # projection must coincide too; a forged coordinate table is not accepted.
    if reconstructed != value:
        raise ValueError('original visible support was changed after projection')
    return value


def support_preserved(support, *, native_points, project_point, base_crop=None):
    """Require exactly the frozen visible support on the corresponding arm."""
    validate_support(support)
    points = tuple(_point(p) for p in native_points)
    if len(points) != 3:
        raise ValueError('candidate support needs the corresponding measured arm')
    return support_preserved_coordinates(support,native_points=points,
        projected_points=tuple(project_point(p) for p in points),base_crop=base_crop)


def support_positions(support, points):
    validate_support(support)
    points=tuple(_point(p) for p in points)
    locations = [points[i] for i in support['joint_indices']]
    for bone in support['bone_portions']:
        i = bone['bone_index']
        locations.extend(_mix(points[i], points[i+1], t)
            for t in (bone['first_parameter'], bone['end_parameter']))
    return tuple(locations)


def support_preserved_coordinates(support, *, native_points, projected_points, base_crop=None):
    crop = (0., 0., 1., 1.) if base_crop is None else base_crop
    inside = lambda p, c: c[0]-1e-9 <= p[0] <= c[2]+1e-9 and c[1]-1e-9 <= p[1] <= c[3]+1e-9
    native = support_positions(support,native_points)
    output = support_positions(support,projected_points)
    return all(inside(p,crop) and inside(q,(0.,0.,1.,1.)) for p,q in zip(native,output,strict=True))


def native_base_crop(state, tick):
    from .group_source_changing_owner import _at
    from .editorial_motion import _fixed_crop_at
    segment, time, _ = _at(state, tick)
    if segment.crop is not None:
        c = segment.crop
        crop = (c.x, c.y, c.width, c.height)
    elif segment.pan_crop is not None:
        crop = _fixed_crop_at(segment.pan_crop, time)
    else:
        return (0., 0., 1., 1.)
    x, y, w, h = crop
    return (max(0., x), max(0., y), min(1., x+w), min(1., y+h))


def restore_original_camera(value):
    """Replay a bound original camera's numbers, never a donor-selected crop."""
    from .group_performer_family_owner import digest
    from .group_joint_source_camera import GroupSourceCameraState, FixedGroupNeighborObservationState, GroupOutputRole
    from .render_plan_serialization import render_segment_from_json
    p = value['key_preimage']
    state_type = FixedGroupNeighborObservationState if value['contract'] == 'fixed_neighbor_readonly_fragment_state_v1' else GroupSourceCameraState
    result = state_type(p['slot'], p['first'], p['end'], p['source'],
        tuple(render_segment_from_json(s) for s in p['segments']), tuple(p['native']),
        GroupOutputRole(p['role']), p['clock'], p['fps'], tuple(p['output_size']),
        tuple(p['cell_frame_counts']), value.get('raw_shot_family'), p.get('editorial_contract_sha256'))
    if digest(result.to_dict()) != digest(value):
        raise ValueError('original visible support camera preimage changed')
    return result
