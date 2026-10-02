"""Freeze output-cell duties before observing any donor or virtual camera.

Incident boundary matches are observations of a pair, not ownership of both
entire shots. A wide with no retained stronger face/body declaration can use a
body-composition observation. This is an editorial declaration, not proof of
anatomy, cast membership or an author's otherwise unrecorded intention.
"""
from .group_joint_source_camera import GroupOutputRole
from .render import segment_frame_counts

CONTRACT = 'output_cell_owned_scene_intent_v2'
WIDE_TYPES = frozenset(('wide', 'group_wide'))
HEAD_TYPES = frozenset(('closeup', 'close_up', 'medium'))


def role_from_cells(cells):
    if not isinstance(cells, (list, tuple)) or not cells:
        raise ValueError('shot intent requires its actual output cells')
    roles = []
    for cell in cells:
        kind, subject = cell['shot_type'], cell['alignment_subject']
        if type(cell['face_applied']) is not bool:
            raise ValueError('shot intent face activity must be explicit')
        if subject in ('formation', 'silhouette'):
            role = GroupOutputRole.FORMATION
        elif subject == 'upper_body':
            role = GroupOutputRole.FOCAL_HEAD_ARMS
        elif subject == 'face':
            role = GroupOutputRole.FOCAL_HEAD
        elif cell['face_applied'] and subject is None:
            raise ValueError('active output crop has no explicit subject duty')
        elif subject not in (None, 'face', 'composition'):
            raise ValueError('unknown stronger output-cell subject declaration')
        elif kind in WIDE_TYPES:
            role = GroupOutputRole.COMPOSITION
        elif kind in HEAD_TYPES:
            role = GroupOutputRole.FOCAL_HEAD
        else:
            raise ValueError('output shot intent unresolved; do not choose from donor quality')
        roles.append(role)
    if len(set(roles)) != 1:
        raise ValueError('mixed output-cell duties require an owned split, not role weakening')
    return roles[0]


def _scope_interval(first, end, scope):
    if scope not in ('whole', 'entry', 'exit') or not 0 <= first < end:
        raise ValueError('scene intent needs an explicit whole/incident output scope')
    return ((first, end) if scope == 'whole' else
        (first, min(end, first+4)) if scope == 'entry' else (max(first, end-4), end))


def _overlapping(cells, first, end):
    return [c for c in cells if c['first_output_frame'] < end and c['end_output_frame'] > first]


def declared_shot_intent(context, shot, *, endpoint_scope='whole'):
    bundle = context.incumbent
    first, last = shot.first_segment_index, shot.last_segment_index
    offsets = [0]
    for count in segment_frame_counts(bundle.render_plan): offsets.append(offsets[-1]+count)
    cells = [dict(segment_index=i,
        first_output_frame=offsets[i], end_output_frame=offsets[i+1],
        shot_type=getattr(bundle.visual_timeline.choices[i].shot_type, 'value',
            bundle.visual_timeline.choices[i].shot_type),
        alignment_subject=bundle.face_match_plan.segments[i].alignment_subject,
        face_applied=bundle.face_match_plan.segments[i].applied)
        for i in range(first, last+1)]
    interval = _scope_interval(shot.first_output_frame, shot.end_output_frame, endpoint_scope)
    required = _overlapping(cells, *interval)
    # Fixed neighbors are not edited. Only their already-declared incident
    # pixels enter this seam comparison; a closeup elsewhere in the SAME
    # continuous camera shot cannot make its wide exit require a focal head.
    # Keep the entire original cell inventory and independently rederive in
    # the parent. Mutable states still require their whole, unmixed duty.
    return dict(contract=CONTRACT, role=role_from_cells(required).value,
        first_output_frame=shot.first_output_frame, end_output_frame=shot.end_output_frame,
        cells=required, retained_output_cells=cells, endpoint_scope=endpoint_scope,
        required_output_interval=list(interval),
        fixed_nonincident_pixels_unchanged=endpoint_scope!='whole',
        frozen_before_source_camera_search=True,
        incident_boundary_does_not_own_whole_shot=True, quality_approved=False)


def validate_intent(value, *, role, first, end, endpoint_scope='whole'):
    if (not isinstance(value, dict) or value.get('contract') != CONTRACT
            or value.get('frozen_before_source_camera_search') is not True
            or value.get('incident_boundary_does_not_own_whole_shot') is not True
            or value.get('quality_approved') is not False
            or (value.get('first_output_frame'), value.get('end_output_frame')) != (first, end)
            or value.get('endpoint_scope') != endpoint_scope
            or value.get('fixed_nonincident_pixels_unchanged') is not (endpoint_scope!='whole')
            or value.get('required_output_interval') != list(_scope_interval(first,end,endpoint_scope))
            or value.get('role') != role or role_from_cells(value.get('cells')).value != role):
        raise ValueError('scene role differs from its frozen output-cell duties')
    retained = value.get('retained_output_cells')
    if not isinstance(retained,list) or not retained:
        raise ValueError('incident intent lost the unchanged full-shot cell inventory')
    for inventory in (retained, value['cells']):
        indices = [c['segment_index'] for c in inventory]
        if (any(type(i) is not int or i < 0 for i in indices)
                or indices != list(range(indices[0], indices[-1]+1))
                or any(type(c[k]) is not int for c in inventory for k in ('first_output_frame','end_output_frame'))
                or any(not 0<=c['first_output_frame']<c['end_output_frame'] for c in inventory)
                or any(a['end_output_frame']!=b['first_output_frame'] for a,b in zip(inventory,inventory[1:]))):
            raise ValueError('shot intent cell inventory is incomplete or duplicated')
    if (retained[0]['first_output_frame']>first or retained[-1]['end_output_frame']<end
            or _overlapping(retained,*value['required_output_interval'])!=value['cells']):
        raise ValueError('incident intent borrowed cells outside its actual measured output interval')
