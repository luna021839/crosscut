"""Fill absent primary-face inputs from measured, exact source-native frames.

The query domain is every available input candidate, never a teacher-selected
source or trajectory. Existing measurements, source gates, pose and GROUP's
intentionally absent target-member coordinates are not changed.
"""
from dataclasses import replace
import hashlib
import json
import math
import re

from .reference_evidence_builder import ReferenceSourceClock
from .stage_edit_face_cues import FEATURE_NAMES, FaceCueTable, schema_with_face_cues
from .stage_edit_feature_materializer import StageEditFeatureSnapshot

CONTRACT = 'input_domain_missing_primary_native_face_v1'
GEOMETRY = ('face_center_x', 'face_center_y', 'face_scale')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sources_from_request(request):
    sources = {s['source_id']: s for s in request['sources']}
    if not sources or len(sources) != len(request['sources']) or len(sources) > 32:
        raise ValueError('one distinct physical source request is required')
    for source in sources.values():
        if not re.fullmatch('[0-9a-f]{64}', source['media_content_sha256']):
            raise ValueError('source media SHA-256 is required')
        fps = source['frame_rate']
        if type(fps) not in (float, int) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('a positive actual source frame rate is required')
    return sources


def missing_queries(snapshots, request, *, schema=None):
    """Deduplicate physical frames across the complete target-free input domain."""
    schema = schema or schema_with_face_cues()
    indices = tuple(schema.source_feature_names.index(n) for n in (*GEOMETRY, *FEATURE_NAMES))
    sources = sources_from_request(request)
    clocks = {s: ReferenceSourceClock.from_mapping(v['clock']) for s, v in sources.items()}
    result = {}
    for snapshot in snapshots:
        if snapshot.token_schema_fingerprint != schema.fingerprint:
            raise ValueError('face completion schema differs from the snapshot')
        if {c.source_id for t in snapshot.sequence.tokens for c in t.source_candidates} != set(sources):
            raise ValueError('face completion source pool differs')
        for token in snapshot.sequence.tokens:
            for candidate in token.source_candidates:
                if (not candidate.available or not candidate.hard_gate_passed
                        or not all(candidate.missing_mask[i] for i in indices)):
                    continue
                source = sources[candidate.source_id]
                seconds = clocks[candidate.source_id].master_to_source(token.master_time_seconds)
                native = round(seconds*source['frame_rate'])
                if seconds < 0 or native < 0:
                    raise ValueError('an available source has a negative native clock')
                key = candidate.source_id, native
                result.setdefault(key, set()).add(float(token.master_time_seconds))
    return {key: tuple(sorted(times)) for key, times in sorted(result.items())}


def augment_missing_native_faces(snapshot, old_face_table, report, request_bytes, *, schema=None):
    schema = schema or schema_with_face_cues()
    if not isinstance(snapshot, StageEditFeatureSnapshot) or not isinstance(old_face_table, FaceCueTable):
        raise TypeError('an actual feature snapshot and its existing face table are required')
    request = json.loads(request_bytes)
    sources = sources_from_request(request)
    upstream = dict(snapshot.provenance.upstream_sha256)
    if upstream.get('face_member_cues') != old_face_table.content_sha256:
        raise ValueError('existing face input provenance differs')
    if 'source_native_face_completion' in upstream:
        raise ValueError('face completion already applied')
    if (report.get('contract') != CONTRACT
            or report.get('request_file_sha256') != hashlib.sha256(request_bytes).hexdigest()
            or snapshot.snapshot_sha256 not in report.get('input_snapshot_sha256', [])
            or report.get('source_media_sha256') != {s: v['media_content_sha256'] for s, v in sources.items()}
            or dict(old_face_table.source_media_sha256) != report.get('source_media_sha256')
            or report.get('gallery_identity') != old_face_table.gallery_identity
            or report.get('descriptor_space') != old_face_table.descriptor_space.to_dict()
            or report.get('current_source_contents_verified_against_request') is not True
            or report.get('selection') != 'all_available_snapshot_candidates_with_absent_face_inputs'
            or report.get('teacher_labels_consumed') is not False):
        raise ValueError('source-native observations do not bind this input domain, gallery and request')
    rows = {}
    for row in report['rows']:
        key = row['source_id'], row['native_frame']
        if (key[0] not in sources or type(key[1]) is not int or key[1] < 0 or key in rows
                or row['media_content_sha256'] != sources[key[0]]['media_content_sha256']):
            raise ValueError('foreign, repeated or unbound native face measurement')
        if row['state'] not in ('OBSERVED', 'NO_FACE', 'LANDMARKS_UNAVAILABLE', 'FRAME_UNAVAILABLE'):
            raise ValueError('an explicit face observation state is required')
        if row['state'] == 'OBSERVED':
            box = row['face_bbox_xywh']; scores = row['member_cosines']
            if (len(box) != 4 or len(scores) != 6
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in (*box, *scores))
                    or min(box[:2]) < 0 or min(box[2:]) <= 0
                    or box[0]+box[2] > 1.+1e-9 or box[1]+box[3] > 1.+1e-9
                    or any(not -1 <= v <= 1 for v in scores)):
                raise ValueError('actual normalized face geometry and six raw cosines are required')
        rows[key] = row
    domain = missing_queries((snapshot,), request, schema=schema)
    indices = tuple(schema.source_feature_names.index(n) for n in (*GEOMETRY, *FEATURE_NAMES))
    clocks = {s: ReferenceSourceClock.from_mapping(v['clock']) for s, v in sources.items()}
    tokens = []
    for token in snapshot.sequence.tokens:
        candidates = []
        for candidate in token.source_candidates:
            if (not candidate.available or not candidate.hard_gate_passed
                    or not all(candidate.missing_mask[i] for i in indices)):
                candidates.append(candidate)
                continue
            seconds = clocks[candidate.source_id].master_to_source(token.master_time_seconds)
            key = candidate.source_id, round(seconds*sources[candidate.source_id]['frame_rate'])
            row = rows.get(key) if key in domain else None
            if row is not None and row['state'] == 'OBSERVED':
                x, y, w, h = row['face_bbox_xywh']
                values, masks = list(candidate.features), list(candidate.missing_mask)
                for i, value in zip(indices, (x+w/2., y+h/2., h, *row['member_cosines']), strict=True):
                    values[i], masks[i] = float(value), False
                candidate = replace(candidate, features=tuple(values), missing_mask=tuple(masks))
            candidates.append(candidate)
        tokens.append(replace(token, source_candidates=tuple(candidates)))
    upstream['source_native_face_completion'] = digest(report)
    return StageEditFeatureSnapshot.build(replace(snapshot.sequence, tokens=tuple(tokens)), schema=schema,
        provenance=replace(snapshot.provenance, upstream_sha256=tuple(sorted(upstream.items())),
            contracts=snapshot.provenance.contracts+(CONTRACT,)),
        causal_sensing_binding=snapshot.causal_sensing_binding,
        inspection_only_without_causal_sensing=not snapshot.causal_sensing_binding.production_complete)
