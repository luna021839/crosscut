"""Original per-tick review duties, including unrequested/mixed-role ticks.

Producer runs before donor sensing. Missing track/pose never erases known
visible support; declared focal-with-arms cannot be reduced to a face-only
role. This does not solve required-formation cast or anonymous correspondence.
"""
from .group_joint_source_camera import GroupOutputRole
from .group_original_visible_support import ARM_INDICES, measured_visible_arm_support
from .group_performer_family_owner import digest


def freeze_review_original_duties(original, measured, *, role_origin='human_declared'):
    if role_origin not in ('human_declared','automatic_observed_support_v1'):
        raise ValueError('original duty policy needs a frozen human/automatic role origin')
    if measured['original_program_sha256'] != original.program_sha256:
        raise ValueError('review duty observations borrowed another original programme')
    count = sum(original.global_cell_frame_counts)
    requested = tuple(measured['requested_output_ticks'])
    if (tuple(sorted(set(requested))) != requested or any(type(t) is not int or not 0 <= t < count for t in requested)
            or set(map(int, measured['rows'])) != set(requested)):
        raise ValueError('review original observations changed their requested tick inventory')
    epochs = {t:(u,index,e) for u in original.units for index,e in enumerate(u.role_program['epochs'])
        for t in range(e['first_output_frame'],e['end_output_frame'])}
    if set(epochs) != set(range(count)):
        raise ValueError('review original role programme has a duty gap')
    latched, tokens = {}, {}
    focal_roles = (GroupOutputRole.FOCAL_HEAD.value,GroupOutputRole.FOCAL_HEAD_ARMS.value)
    for tick in range(count):
        u,index,e = epochs[tick]; role = e['role']; key = (u.unit_id,index)
        parts = latched.setdefault(key,{})
        if role == GroupOutputRole.FOCAL_HEAD_ARMS.value and role_origin=='human_declared':
            for name in ARM_INDICES: parts.setdefault(name,None)
        row = measured['rows'].get(str(tick)); head = measured['original_linked_heads'].get(str(tick))
        focal = role in focal_roles
        if row is not None:
            projection = row['projection']; raw = row['native_observation']
            if (projection['original_unit_id'],projection['original_epoch_index'],projection['original_role']) != (u.unit_id,index,role):
                raise ValueError('review original measurement changed an external unit/epoch/role')
            owners = [p for p in raw['people'] if head is not None and p['unique_head_observation_id'] == head]
            if focal and len(owners) == 1:
                p = owners[0]; projected = next(q for q in projection['people'] if q['observation_id'] == p['observation_id'])
                for name,indices in ARM_INDICES.items():
                    if dict(p['part_states']).get(name) != 'observed': continue
                    points = [p['upper_body_xyc'][j][:2] for j in indices]
                    target = [projected['projected_upper_body_xyc'][j][:2] for j in indices]
                    support = measured_visible_arm_support(native_points=points,
                        project_point=lambda point:target[points.index(list(point))])
                    # A new positive original observation owns this tick's
                    # actual visible support. Missing/outside observations
                    # retain prior known support; explicit whole-arm duties
                    # remain whole and cannot be replaced by a partial mask.
                    if support['visibility'] != 'outside' and not (name in parts and parts[name] is None):
                        parts[name]=support
        tokens[str(tick)] = dict(head_required=focal,parts=dict(parts) if focal else {},
            role_origin=role_origin,
            original_role=role,original_unit_id=u.unit_id,original_epoch_index=index,
            original_observation_sha256=None if row is None else digest(row['native_observation']),
            conditional_original_head=head,
            observation='not_requested' if row is None else 'actual_requested_original_native',
            unresolved_role_or_formation=role in (None,GroupOutputRole.FORMATION.value),
            cast_correspondence_certified=False)
    return dict(contract='frozen_original_review_part_duties_v1',
        automatic_role_policy=role_origin,
        original_program_sha256=original.program_sha256,rows=tokens,
        full_original_output_ticks_observed=len(requested) == count,
        missing_actor_does_not_erase_latched_parts=True,mixed_and_subcell_original_roles_preserved=True,
        original_cast_correspondence_certified=False,quality_approved=False)
