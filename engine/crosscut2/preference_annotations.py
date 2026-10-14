"""Bind actual optional browser choices to the exact same-event features.

Self-reported AV viewing is provenance, not a technical proof of listening.
Ties, all-bad/uncertain answers and sealed regression examples are preserved
in the import summary, never silently converted into a winning cut/hold.
"""
import re
import numpy as np

from .preference_features import FEATURE_NAMES, SCHEMA


def materialize(annotations, features, *, packet_sha256, split_by_group, regression_event_ids=()):
    if (not isinstance(packet_sha256, str) or not re.fullmatch('[0-9a-f]{64}', packet_sha256)
            or annotations.get('contract') != 'same_event_user_preference_annotations_v1'
            or features.get('contract') != 'unlabelled_same_event_preference_features_v1'
            or any(row.get('packet_sha256') != packet_sha256 for row in (annotations, features))
            or features.get('feature_schema') != SCHEMA
            or features.get('feature_names') != list(FEATURE_NAMES)):
        raise ValueError('annotation packet and current raw feature binding differ')
    events = features.get('events', [])
    rows = {row['id']: row for row in events}
    if len(rows) != len(events) or not set(regression_event_ids) <= rows.keys():
        raise ValueError('unique event IDs and an explicit existing regression subset required')
    choices = {'hold_A': 0, 'cut_A_B': 1, 'hold_B': 2}
    expected_order = list(choices)
    result, skipped, seen = [], [], set()
    for label in annotations.get('labels', []):
        event_id, choice = label.get('id'), label.get('choice')
        if event_id not in rows or event_id in seen:
            raise ValueError('unknown or duplicate actual annotation event')
        seen.add(event_id)
        if choice not in (*choices, '', 'tie', 'all_bad', 'uncertain'):
            raise ValueError('unknown actual annotation choice')
        reason = None
        if event_id in regression_event_ids or label.get('regression_only') is True:
            reason = 'sealed_regression_not_training'
        elif choice not in choices:
            reason = 'no_strict_preference_' + (choice or 'unanswered')
        elif (label.get('provenance') != 'user_browser_choice'
                or label.get('self_reported_normal_av') is not True):
            reason = 'missing_actual_browser_choice_or_self_reported_normal_av'
        if reason:
            skipped.append(dict(id=event_id, choice=choice, reason=reason))
            continue
        row = rows[event_id]
        group = row.get('physical_group')
        split = split_by_group.get(group)
        values = np.asarray(row.get('features'), float)
        if (not isinstance(group, str) or not group.strip() or split not in ('train', 'dev', 'held_out')
                or row.get('option_order') != expected_order
                or values.shape != (3, len(FEATURE_NAMES)) or not np.isfinite(values).all()):
            raise ValueError('same-event option order, finite features and physical-group split required')
        result.append(dict(id=event_id, physical_group=group, split=split,
            supervision_kind='human_av_preference', self_reported_normal_av=True,
            features=values.tolist(), chosen=choices[choice], packet_sha256=packet_sha256,
            label_provenance='user_browser_choice'))
    return result, dict(imported_strict_preferences=len(result), skipped=skipped,
        unanswered_event_ids=sorted(rows.keys()-seen),
        self_report_is_not_technical_proof_of_listening=True,
        professional_quality_approved=False, blind_song_validation_claimed=False)
