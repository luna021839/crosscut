"""Declared shot duties are not the inventory of everything a detector found.

The selected head is a pre-camera, frame-local hypothesis, not a named actor.
Raw background observations remain evidence, including their unknown ownership.
No camera geometry or continuity score is an input to this scope decision.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .group_joint_source_camera import GroupOutputRole


CONTRACT = 'declared_scene_role_obligations_v2'
ARM_PARTS = ('left_arm', 'right_arm')


def _get(row, key, default=None):
    return row.get(key, default) if isinstance(row, Mapping) else getattr(row, key, default)


@dataclass(frozen=True)
class SceneRoleObligations:
    role: str
    selected_head_id: int | None
    required_head_ids: tuple[int, ...]
    required_person_ids: tuple[int, ...]
    required_parts: tuple[tuple[int, str], ...]
    focal_owner_id: int | None
    focal_owner_state: str
    context_head_ids: tuple[int, ...]
    context_person_ids: tuple[int, ...]

    def to_dict(self):
        return dict(contract=CONTRACT, **asdict(self),
            scope_chosen_before_camera=True, raw_context_must_be_retained=True,
            named_identity_certified=False, physical_extent_certified=False)


def scene_role_obligations(raw, role, selected_head_id):
    role = GroupOutputRole(role)
    heads, people = _get(raw, 'heads'), _get(raw, 'people')
    if heads is None or people is None:
        raise ValueError('scene role needs the complete native observation inventory')
    head_ids = tuple(_get(h, 'observation_id') for h in heads)
    person_ids = tuple(_get(p, 'observation_id') for p in people)
    if (any(type(i) is not int for i in (*head_ids, *person_ids))
            or len(set(head_ids)) != len(head_ids) or len(set(person_ids)) != len(person_ids)):
        raise ValueError('scene role raw observation IDs are invalid or duplicated')
    if role == GroupOutputRole.COMPOSITION:
        if selected_head_id is not None or heads or len(people) < 2:
            raise ValueError('composition duty needs body support, never a fake focal head')
        # This v1 conservative inventory is NOT a semantic cast/cohort solver.
        # Unknown poses remain unknown and cannot create an arm obligation.
        return SceneRoleObligations(role.value, None, (), person_ids, (), None,
            'not_requested_body_support_only', (), ())
    if role == GroupOutputRole.FORMATION:
        if selected_head_id is not None or len(people) < 2:
            raise ValueError('formation duty cannot be replaced by a selected focal head')
        return SceneRoleObligations(role.value, None, head_ids, person_ids,
            tuple((i, part) for i in person_ids for part in ARM_PARTS), None,
            'not_focal', (), ())
    if type(selected_head_id) is not int or selected_head_id not in head_ids:
        raise ValueError('declared focal hypothesis is missing from original observations')
    owners = tuple(p for p in people if _get(p, 'unique_head_observation_id') == selected_head_id)
    owner = owners[0] if len(owners) == 1 else None
    owner_id = None if owner is None else _get(owner, 'observation_id')
    state = 'observed' if owner is not None else 'ambiguous' if owners else 'unknown'
    parts = {} if owner is None else dict(_get(owner, 'part_states', ()))
    if role == GroupOutputRole.FOCAL_HEAD_ARMS and (owner is None
            or any(parts.get(part) != 'observed' for part in ARM_PARTS)):
        raise ValueError('scene focal required arms lack unique original-pixel ownership')
    # A known arm of the selected owner remains a duty even on a head role.
    # An unbound OTHER body is neither assigned to this head nor erased.
    required_parts = tuple((owner_id, part) for part in ARM_PARTS
        if owner is not None and parts.get(part) == 'observed')
    return SceneRoleObligations(role.value, selected_head_id, (selected_head_id,),
        () if owner_id is None else (owner_id,), required_parts, owner_id, state,
        tuple(i for i in head_ids if i != selected_head_id),
        tuple(i for i in person_ids if i != owner_id))
