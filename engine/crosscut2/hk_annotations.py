"""Import optional real browser H/K choices without fitting or inventing labels.

The public manifest digest binds the browser export. The caller must separately
pin the private original A/B mapping digest; that mapping must never be served
by the comparison viewer. Self-reported AV is provenance, not proof of hearing.
These rows are NOT the three-option preference-model training schema.
"""
import hashlib
import json
import re


def import_annotations(annotations, *, manifest_bytes, grounding_bytes,
                       expected_grounding_sha256, regression_event_ids=()):
    if (not isinstance(annotations, dict) or type(manifest_bytes) is not bytes
            or type(grounding_bytes) is not bytes
            or not isinstance(expected_grounding_sha256, str)
            or re.fullmatch('[a-f0-9]{64}', expected_grounding_sha256) is None
            or hashlib.sha256(grounding_bytes).hexdigest() != expected_grounding_sha256):
        raise ValueError('exact original public packet and pinned private mapping required')
    packet_sha256=hashlib.sha256(manifest_bytes).hexdigest()
    try:
        packet=json.loads(manifest_bytes);grounding=json.loads(grounding_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('packet and private mapping must be complete UTF8 JSON') from error
    if (not isinstance(packet, dict) or not isinstance(grounding, dict)
            or packet.get('contract') != 'unlabelled_same_event_hk_av_pairs_v2'
            or packet.get('actual_human_preferences_obtained') is not False
            or annotations.get('contract') != 'same_event_user_hk_annotations_v2'
            or annotations.get('packet_sha256') != packet_sha256
            or not isinstance(packet.get('events'), list)
            or not isinstance(grounding.get('events'), list)
            or not isinstance(annotations.get('labels'), list)):
        raise ValueError('H/K annotation schema or exact public packet binding differs')
    events={};mapping={}
    for row in packet['events']:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
                or not row['id'] or row['id'] in events
                or not isinstance(row.get('clips'), dict) or set(row['clips']) != {'A','B'}
                or any(not isinstance(v, str) or not v for v in row['clips'].values())):
            raise ValueError('unique H/K event IDs and both original clips required')
        events[row['id']]=row
    for row in grounding['events']:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
                or row['id'] in mapping or not isinstance(row.get('variant_map'), dict)
                or set(row['variant_map']) != {'A','B'}
                or any(not isinstance(v,str) for v in row['variant_map'].values())
                or set(row['variant_map'].values()) != {'H','K'}
                or type(row.get('global_frame')) is not int or row['global_frame']<0):
            raise ValueError('unique private event mapping must contain one H and one K')
        mapping[row['id']]=row
    sealed=set(regression_event_ids)
    if not events or events.keys()!=mapping.keys() or not sealed<=events.keys():
        raise ValueError('private mapping and explicit regression subset must match the packet')
    imported=[];other=[];seen=set();answered=set()
    for label in annotations['labels']:
        if not isinstance(label, dict) or not isinstance(label.get('id'), str):
            raise ValueError('actual label needs an existing event ID')
        identity=label['id'];choice=label.get('choice')
        if identity not in events or identity in seen:
            raise ValueError('unknown or duplicate actual H/K annotation')
        if not isinstance(choice, str) or choice not in ('A','B','','tie','all_bad','uncertain'):
            raise ValueError('unknown actual H/K choice')
        seen.add(identity)
        if choice:answered.add(identity)
        row=dict(id=identity,choice=choice,packet_sha256=packet_sha256,
            private_mapping_sha256=expected_grounding_sha256,
            label_provenance=label.get('provenance'),
            self_reported_normal_av=label.get('self_reported_normal_av') is True)
        if identity in sealed or label.get('regression_only') is True:
            reason='sealed_regression_not_training'
        elif choice not in ('A','B'):
            reason='no_strict_preference_'+(choice or 'unanswered')
        elif label.get('provenance')!='user_browser_choice' or label.get('self_reported_normal_av') is not True:
            reason='missing_actual_browser_choice_or_self_reported_normal_av'
        else:
            reason=None
        if reason:
            other.append(dict(row,reason=reason));continue
        imported.append(dict(row,global_frame=mapping[identity]['global_frame'],
            chosen_action=mapping[identity]['variant_map'][choice],
            supervision_kind='human_av_hk_pair_annotation_only'))
    return imported,dict(imported_strict_preferences=len(imported),non_strict_or_ineligible=other,
        unanswered_event_ids=sorted(events.keys()-answered),preference_fit_executed=False,
        requires_separate_bound_numeric_features_before_training=True,
        self_report_is_not_technical_proof_of_listening=True,
        professional_quality_approved=False,blind_song_validation_claimed=False)
