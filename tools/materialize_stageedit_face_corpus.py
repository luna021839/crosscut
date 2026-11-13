"""Refresh scalar center cues and add real face inputs for one frozen training launch.

No media decoding, neural inference, fitting, split changes or supervision
rewrites. Only the explicit launch's reference snapshots are consumed.
"""
import argparse
import hashlib
import json
from pathlib import Path

from crosscut.causal_shooting_grammar import load_causal_shooting_grammar_artifact
from crosscut.ive_performer_state import (
    build_ive_performer_state_artifact, load_ive_performer_state_artifact, write_ive_performer_state_artifact,
)
from crosscut.member_state_graph import load_member_state_graph_artifact
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_face_cues import augment_snapshot, schema_with_face_cues
from materialize_stageedit_face_cues import build_table


def _publish(directory, name, value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    path = directory/name
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError('refusing to replace a different prepared input')
    else:
        with path.open('xb') as stream:
            stream.write(encoded)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', type=Path, required=True)
    p.add_argument('--observation-directory', type=Path, required=True)
    p.add_argument('--output-directory', type=Path, required=True)
    a = p.parse_args()
    volume = Path('/Volumes/CrossCutMedia')
    if not volume.is_mount() or not a.output_directory.is_relative_to(volume):
        raise OSError('prepared inputs must remain on the external SSD')
    device = volume.stat().st_dev
    raw_launch = a.launch.read_bytes()
    args = json.loads(raw_launch)['argv']
    inputs = [args[i+1] for i, option in enumerate(args) if option == '--feature-snapshot']
    if len(inputs) != len(set(value.split('=',1)[0] for value in inputs)):
        raise ValueError('launch repeats a reference snapshot')
    a.output_directory.mkdir(parents=True, exist_ok=True)
    schema = schema_with_face_cues()
    prepared, groups = {}, {}
    rows = []
    for value in inputs:
        if not volume.is_mount() or volume.stat().st_dev != device:
            raise OSError('authorized SSD changed')
        reference, name = value.split('=', 1)
        path = Path(name)
        song_root = path.parent.parent
        song = song_root.name
        snapshot = StageEditFeatureSnapshot.from_dict(json.loads(path.read_bytes()))
        if song not in groups:
            binding = snapshot.causal_sensing_binding
            root = song_root/'causal-sensing'
            def artifact(kind, digest):
                return root/kind/digest[:2]/(digest+'.json')
            previous = load_ive_performer_state_artifact(artifact('performer-state/ive_performer_state_v1',
                binding.ive_performer_state_content_sha256))
            member_graph = load_member_state_graph_artifact(artifact('member-state/ive_member_state_graph_v1',
                binding.member_state_graph_content_sha256))
            grammar = load_causal_shooting_grammar_artifact(artifact('shooting-grammar',
                binding.shooting_grammar_content_sha256))
            current = build_ive_performer_state_artifact(member_graph.graph, grammar.records,
                provenance=previous.provenance)
            if previous.input_binding != current.input_binding:
                raise ValueError('cached performer refresh would change its causal evidence')
            performer_path = write_ive_performer_state_artifact(a.output_directory/'performer-state', current)
            requests = [v for v in (song_root/'requests').rglob('*.json') if not v.name.startswith('._')]
            if len(requests) != 1:
                raise ValueError('one exact existing source/clock request is required per song')
            request_bytes = requests[0].read_bytes()
            report_path = a.observation_directory/(song+'-primary-face-cosines.json')
            report_bytes = report_path.read_bytes()
            groups[song] = previous, current, json.loads(report_bytes), json.loads(request_bytes), hashlib.sha256(request_bytes).hexdigest()
            print(json.dumps(dict(song=song, performer_state=str(performer_path),
                observation_file_sha256=hashlib.sha256(report_bytes).hexdigest())), flush=True)
        previous, current, report, request, request_sha = groups[song]
        refreshed = snapshot.refresh_ive_performer_state(previous, current)
        table = build_table(refreshed, report, request, request_file_sha256=request_sha)
        augmented = augment_snapshot(refreshed, table)
        augmented.require_production_causal_sensing(edit_mode='group', schema=schema)
        if StageEditFeatureSnapshot.from_dict(augmented.to_dict(), schema=schema) != augmented:
            raise AssertionError('prepared input failed strict roundtrip')
        target = a.output_directory/reference
        target.mkdir(exist_ok=True)
        cue_path = _publish(target, f'face-cues-{table.content_sha256}.json', table.to_dict())
        snapshot_path = _publish(target, f'feature-snapshot-{augmented.snapshot_sha256}.json', augmented.to_dict())
        prepared[reference] = str(snapshot_path)
        rows.append(dict(reference_id=reference, song=song, original_snapshot_sha256=snapshot.snapshot_sha256,
            refreshed_base_snapshot_sha256=refreshed.snapshot_sha256, feature_snapshot_sha256=augmented.snapshot_sha256,
            face_cue_path=str(cue_path), face_cue_content_sha256=table.content_sha256,
            measured_faces=len(table.records), candidate_rows=sum(len(t.source_candidates) for t in augmented.sequence.tokens)))
        print(json.dumps(rows[-1]), flush=True)
    schema_path = _publish(a.output_directory, f'token-schema-{schema.fingerprint}.json', schema.to_dict())
    manifest = dict(contract='existing_launch_center_refresh_and_arcface_inputs_v1',
        original_launch_file_sha256=hashlib.sha256(raw_launch).hexdigest(), source_launch=str(a.launch),
        feature_snapshot_by_reference=prepared, token_schema_path=str(schema_path),
        token_schema_fingerprint=schema.fingerprint, references=rows,
        supervision_files_changed=False, source_pool_clock_pose_and_gates_changed=False,
        original_caches_overwritten=False, media_decodes=0, neural_forwards=0)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = _publish(a.output_directory, f'face-corpus-{digest}.json', manifest)
    print(json.dumps(dict(manifest=str(manifest_path), references=len(prepared),
        measured_faces=sum(row['measured_faces'] for row in rows))), flush=True)


if __name__ == '__main__':
    main()
