"""Local original-programme storage and received closure, not sensor history.

Replacing the active window must not erase a prior duty programme still named
by retained frames/pose links. Storage versions do not constitute a parent-
owned pre-work journal, a coherent actor programme or original pixel visibility.
"""
from . import group_performer_family_owner as family

CONTRACT='local_original_role_and_native_duty_inventory_v1'


def programme_records(*,roles,native,active):
    return family.plain(dict(contract=CONTRACT,original_role_programmes=roles,
        native_duty_programmes=native,active_native_duty_programme_sha256=active,
        controller_owned_history_certified=False,pre_receipt_history_completeness_certified=False,
        coherent_original_actor_programme_certified=False,original_output_visibility_certified=False,
        global_mixed_role_camera_owner_implemented=False,quality_approved=False))


def validate_programme_records(records,*,raw_inventory,reference_roots=(),required_programme=None,
        original_plan=None):
    from .group_scene_original_duties import validate_original_duty_program,duty_at
    from .group_scene_epoch_program import validate_original_role_epochs
    if (not isinstance(records,dict) or records.get('contract')!=CONTRACT
            or not isinstance(records.get('original_role_programmes'),dict)
            or not isinstance(records.get('native_duty_programmes'),dict)
            or any(records.get(name) is not False for name in ('controller_owned_history_certified',
                'pre_receipt_history_completeness_certified','coherent_original_actor_programme_certified',
                'original_output_visibility_certified','global_mixed_role_camera_owner_implemented','quality_approved'))):
        raise ValueError('original programme inventory lost its local/uncertainty boundary')
    native=records['native_duty_programmes']
    for identity,program in native.items():
        if identity!=program.get('program_sha256'):
            raise ValueError('original native programme stored under another version')
        validate_original_duty_program(program,raw_inventory=raw_inventory)
    for identity,program in records['original_role_programmes'].items():
        if identity!=family.digest(program):
            raise ValueError('original role programme stored under another version')
        validate_original_role_epochs(program,original_plan=original_plan)
    active=records.get('active_native_duty_programme_sha256')
    if active is not None and active not in native or native and active is None:
        raise ValueError('active original duty programme disappeared from its received versions')
    if required_programme is not None and native.get(required_programme['program_sha256'])!=required_programme:
        raise ValueError('selected original duty programme disappeared from its received versions')
    def walk(value):
        if isinstance(value,dict):
            ref=value.get('original_duty_reference')
            if ref is not None:
                if not isinstance(ref,dict) or ref.get('program_sha256') not in native:
                    raise ValueError('retained original duty reference lost its own programme version')
                program=native[ref['program_sha256']]
                units=[unit for unit in program['units'] if unit['original_unit_id']==ref.get('original_unit_id')]
                if len(units)!=1:
                    raise ValueError('retained original duty reference borrowed another unit')
                unit=units[0]
                expected=duty_at(program,dict(first=unit['first_output_frame'],end=unit['end_output_frame'],
                    role=unit['role']),ref.get('output_tick'))
                if ref!=expected:
                    raise ValueError('retained original duty reference borrowed another tick/part row')
            identity=value.get('original_duty_program_sha256')
            if identity is not None and identity not in native:
                raise ValueError('retained pose/endpoint lost its original duty programme version')
            for child in value.values():walk(child)
        elif isinstance(value,(list,tuple)):
            for child in value:walk(child)
    for root in reference_roots:walk(root)
    return records
