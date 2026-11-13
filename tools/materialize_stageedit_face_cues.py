"""Append actual source-native raw face scores to an existing feature snapshot.

Does not decode, infer, train, rename members, change sources or edit a movie.
Reuse one completed face-only observation report across reference snapshots
whose exact times occur in its source-native domain.
"""
import argparse
import hashlib
import json
from pathlib import Path

from crosscut.ive import IVEMember
from crosscut.native_face_descriptor import NativeFaceDescriptorSpace
from crosscut.reference_evidence_builder import ReferenceSourceClock
from crosscut.stage_edit_face_cues import FaceCue, FaceCueTable, augment_snapshot, schema_with_face_cues
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot


def build_table(snapshot, report, request, *, request_file_sha256):
    if (report['request_file_sha256'] != request_file_sha256
            or report['graph_content_sha256'] != snapshot.causal_sensing_binding.member_state_graph_content_sha256
            or report.get('current_source_contents_verified_against_request') is not True
            or report.get('raw_query_or_reference_vectors_serialized') is not False):
        raise ValueError('source-native observations do not bind this snapshot graph/request/current media')
    sources = {s['source_id']: s for s in request['sources']}
    source_ids = {c.source_id for token in snapshot.sequence.tokens for c in token.source_candidates}
    if source_ids != set(sources):
        raise ValueError('snapshot and observation request source pools differ')
    domain = {(c.source_id, token.master_time_seconds) for token in snapshot.sequence.tokens for c in token.source_candidates}
    records, seen, spaces, galleries = [], set(), {}, set()
    for row in report['rows']:
        key = row['source_id'], row['master_seconds']
        if key in seen:
            raise ValueError('observation report repeats one physical source/time')
        seen.add(key)
        if key[0] not in sources:
            raise ValueError('observation report borrowed another source')
        source = sources[key[0]]
        expected = ReferenceSourceClock.from_mapping(source['clock']).master_to_source(key[1])
        if (abs(row['source_seconds']-expected) > 1e-9
                or row['native_frame'] != round(expected * source['frame_rate'])
                or row['media_content_sha256'] != source['media_content_sha256']):
            raise ValueError('observation report changed the exact source/media/native clock')
        if row['state'] != 'OBSERVED' or key not in domain:
            continue
        hint = row['arcface']
        space = NativeFaceDescriptorSpace(**hint['descriptor_space'])
        spaces[tuple(space.to_dict().items())] = space
        galleries.add(hint['gallery_identity'])
        # Even a completely UNKNOWN decision contributes its ORIGINAL six
        # scalar likenesses. No threshold, softmax, winner or renormalization.
        records.append(FaceCue(*key, row['source_seconds'], row['native_frame'],
            tuple(hint['member_scores'][member.value] for member in IVEMember)))
    if len(spaces) != 1 or len(galleries) != 1:
        raise ValueError('one consistent real ArcFace/gallery domain is required')
    return FaceCueTable(snapshot.snapshot_sha256,
        tuple(sorted((s, sources[s]['media_content_sha256']) for s in sources)),
        next(iter(galleries)), next(iter(spaces.values())), tuple(sorted(records, key=lambda row: row.key)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'observations', 'request', 'output-directory'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--edit-mode', choices=('group','member'), default='group')
    p.add_argument('--target-member')
    a = p.parse_args()
    snapshot = StageEditFeatureSnapshot.from_dict(json.loads(a.snapshot.read_bytes()))
    snapshot.require_production_causal_sensing(edit_mode=a.edit_mode, target_member=a.target_member)
    raw_request = a.request.read_bytes()
    table = build_table(snapshot, json.loads(a.observations.read_bytes()), json.loads(raw_request),
        request_file_sha256=hashlib.sha256(raw_request).hexdigest())
    augmented = augment_snapshot(snapshot, table)
    schema = schema_with_face_cues()
    augmented.require_production_causal_sensing(edit_mode=a.edit_mode, target_member=a.target_member, schema=schema)
    restored = StageEditFeatureSnapshot.from_dict(augmented.to_dict(), schema=schema)
    if restored != augmented:
        raise AssertionError('enriched snapshot failed its strict roundtrip')
    destination = a.output_directory
    if not destination.is_relative_to(Path('/Volumes/CrossCutMedia')) or not Path('/Volumes/CrossCutMedia').is_mount():
        raise OSError('derived inputs must remain on the connected external SSD')
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, digest, value in (('face-cues', table.content_sha256, table.to_dict()),
            ('feature-snapshot', augmented.snapshot_sha256, augmented.to_dict()),
            ('token-schema', schema.fingerprint, schema.to_dict())):
        path = destination/f'{name}-{digest}.json'
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError('refusing to overwrite a different derived input')
        else:
            with path.open('xb') as stream:
                stream.write(encoded)
        results[name] = str(path)
    print(json.dumps(dict(outputs=results, observed_face_cues=len(table.records),
        candidate_rows=sum(len(t.source_candidates) for t in snapshot.sequence.tokens),
        raw_source_dimension=schema.raw_source_dimension, encoded_source_dimension=schema.encoded_source_dimension,
        clocks_sources_original_features_pose_and_gates_unchanged=True, new_media_decode=0, new_neural_calls=0)))


if __name__ == '__main__':
    main()
