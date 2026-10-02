"""Freeze output-cell duties before observing any donor or virtual camera.

Incident boundary matches are observations of a pair, not ownership of both
entire shots. A wide with no retained stronger face/body declaration can use a
body-composition observation. This is an editorial declaration, not proof of
anatomy, cast membership or an author's otherwise unrecorded intention.
"""
from .group_joint_source_camera import GroupOutputRole

CONTRACT = 'output_cell_owned_scene_intent_v1'
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


def declared_shot_intent(context, shot):
    bundle = context.incumbent
    first, last = shot.first_segment_index, shot.last_segment_index
    cells = [dict(segment_index=i,
        shot_type=getattr(bundle.visual_timeline.choices[i].shot_type, 'value',
            bundle.visual_timeline.choices[i].shot_type),
        alignment_subject=bundle.face_match_plan.segments[i].alignment_subject,
        face_applied=bundle.face_match_plan.segments[i].applied)
        for i in range(first, last+1)]
    return dict(contract=CONTRACT, role=role_from_cells(cells).value,
        first_output_frame=shot.first_output_frame, end_output_frame=shot.end_output_frame,
        cells=cells, frozen_before_source_camera_search=True,
        incident_boundary_does_not_own_whole_shot=True, quality_approved=False)


def validate_intent(value, *, role, first, end):
    if (not isinstance(value, dict) or value.get('contract') != CONTRACT
            or value.get('frozen_before_source_camera_search') is not True
            or value.get('incident_boundary_does_not_own_whole_shot') is not True
            or value.get('quality_approved') is not False
            or (value.get('first_output_frame'), value.get('end_output_frame')) != (first, end)
            or value.get('role') != role or role_from_cells(value.get('cells')).value != role):
        raise ValueError('scene role differs from its frozen output-cell duties')
    indices = [c['segment_index'] for c in value['cells']]
    if (any(type(i) is not int or i < 0 for i in indices)
            or indices != list(range(indices[0], indices[-1]+1))):
        raise ValueError('shot intent cell inventory is incomplete or duplicated')
