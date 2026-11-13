"""Train-only diagnostic of human editorial timing agreement.

This consumes supervision, never produces inference features or replacement
labels. Unknown attribution is not a vote for holding the shot. Disagreement
between editors is not by itself an inconsistent-label or learnability proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('reference clock must be numeric')
    if not math.isfinite(value):
        raise ValueError('reference clock must be finite')
    return float(value)


@dataclass(frozen=True)
class EditorialReference:
    reference_id: str
    song_id: str
    style_id: str
    positives: tuple[float, ...]
    negatives: tuple[float, ...]
    boundaries: tuple[float, ...]
    known_intervals: tuple[tuple[float, float], ...]


def reference_editorial_evidence(payload: Mapping, *, song_id: str) -> EditorialReference:
    identifier = payload.get('reference_id')
    if not isinstance(identifier, str) or not identifier.strip() or not song_id:
        raise ValueError('reference and song identities are required')
    positives, negatives, boundaries, intervals = [], [], [], []
    for record in payload['detected_boundary_records']:
        time = _time(record['cut_master_time'])
        boundaries.append(time)
        if record.get('loss_masked') is not False or record.get('target_masks', {}).get('cut') is not True:
            continue
        if record.get('is_editorial_source_switch_positive') is True and record.get('cut_target') == 1.0:
            positives.append(time)
        elif record.get('is_native_source_camera_cut_hard_negative') is True and record.get('cut_target') == 0.0:
            negatives.append(time)
    for shot in payload['shot_labels']:
        labels = shot.get('edit_labels', {})
        if not shot.get('selected_source_id') or labels.get('dense_attribution', {}).get('ready') is not True:
            continue
        cut = labels['cut']
        start, end = _time(cut['cut_master_time']), _time(cut['end_master_time'])
        if end <= start:
            raise ValueError('known reference shot must have a positive duration')
        intervals.append((start, end))
    return EditorialReference(identifier, song_id, str(payload.get('style_id', '')),
                              tuple(sorted(set(positives))), tuple(sorted(set(negatives))),
                              tuple(sorted(set(boundaries))), tuple(sorted(intervals)))


def classify_reference_neighborhood(reference: EditorialReference, time: float, tolerance: float) -> str:
    time, tolerance = _time(time), _time(tolerance)
    if tolerance < 0:
        raise ValueError('tolerance cannot be negative')
    if any(abs(t-time) <= tolerance for t in reference.positives):
        return 'editorial_cut'
    nearby = tuple(t for t in reference.boundaries if abs(t-time) <= tolerance)
    if nearby:
        # Every nearby detector boundary must be a verified non-editorial cut.
        return 'native_cut_only' if all(t in reference.negatives for t in nearby) else 'unknown'
    if any(start < time-tolerance and time+tolerance < end for start, end in reference.known_intervals):
        return 'known_hold'
    return 'unknown'


def editorial_agreement_report(references: Sequence[EditorialReference], *, tolerance_seconds: float) -> dict:
    refs = tuple(references)
    if not refs or len({r.reference_id for r in refs}) != len(refs):
        raise ValueError('nonempty unique references required')
    tolerance = _time(tolerance_seconds)
    if tolerance < 0:
        raise ValueError('tolerance cannot be negative')
    counts = dict(editorial_cut=0, native_cut_only=0, known_hold=0, unknown=0)
    events = []
    for anchor in refs:
        peers = tuple(r for r in refs if r.song_id == anchor.song_id and r.reference_id != anchor.reference_id)
        for time in anchor.positives:
            votes = {r.reference_id: classify_reference_neighborhood(r, time, tolerance) for r in peers}
            for vote in votes.values():
                counts[vote] += 1
            events.append(dict(reference_id=anchor.reference_id, song_id=anchor.song_id,
                               master_time_seconds=time, peer_votes=votes))
    observed = sum(counts[k] for k in ('editorial_cut', 'native_cut_only', 'known_hold'))
    return dict(contract='observed_cross_reference_editorial_agreement_diagnostic_v1',
                tolerance_seconds=tolerance, reference_count=len(refs),
                positive_event_count=len(events), peer_vote_counts=counts,
                observed_peer_cut_agreement=counts['editorial_cut']/observed if observed else None,
                unknown_is_negative=False, labels_changed=False,
                target_disagreement_is_not_inconsistency_proof=True,
                events=events)
