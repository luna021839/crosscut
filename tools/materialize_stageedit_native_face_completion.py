"""Publish completed source-native face inputs, without fitting or reanalysing media."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from crosscut.stage_edit_face_cues import FaceCueTable, FEATURE_NAMES
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_native_face_completion import GEOMETRY, augment_missing_native_faces, digest
from crosscut.stage_edit_transformer import StageEditTokenSchema


def require_input_only_change(before, after, schema):
    """Census the actual full snapshot, not only a synthetic unit fixture."""
    allowed={schema.source_feature_names.index(n) for n in (*GEOMETRY,*FEATURE_NAMES)}
    changed=0
    if before.causal_sensing_binding!=after.causal_sensing_binding:
        raise ValueError('face completion changed causal source/clock binding')
    for a,b in zip(before.sequence.tokens,after.sequence.tokens,strict=True):
        if replace(b,source_candidates=a.source_candidates)!=a:
            raise ValueError('face completion changed token time, music, context or history')
        for x,y in zip(a.source_candidates,b.source_candidates,strict=True):
            if replace(y,features=x.features,missing_mask=x.missing_mask)!=x:
                raise ValueError('face completion changed source, availability, gate or temporal pose')
            for j,(u,v,mu,mv) in enumerate(zip(x.features,y.features,x.missing_mask,y.missing_mask,strict=True)):
                if (u,mu) != (v,mv):
                    if j not in allowed or not mu or mv:
                        raise ValueError('face completion overwrote an observed or unrelated input')
            changed+=x!=y
    return changed


def publish(path,value):
    raw=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('refusing to replace different derived inputs')
    else:
        with path.open('xb') as f:f.write(raw)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('prepared-inputs','observation-directory','output-directory'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--ssd-device',type=int,required=True)
    a=p.parse_args();volume=Path('/Volumes/CrossCutMedia')
    if (not volume.is_mount() or volume.stat().st_dev!=a.ssd_device
            or not a.output_directory.resolve().is_relative_to(volume.resolve())):
        raise OSError('derived inputs must remain on the authorized SSD')
    raw_parent=a.prepared_inputs.read_bytes();parent=json.loads(raw_parent)
    schema=StageEditTokenSchema.from_dict(json.loads(Path(parent['token_schema_path']).read_bytes()))
    plan=json.loads((a.observation_directory/'plan.json').read_bytes())
    group_by_snapshot={};reports=dict(parent.get('native_face_observation_reports',{}))
    for g in plan['groups']:
        path=a.observation_directory/(g['request_file_sha256']+'-observations.json')
        report=json.loads(path.read_bytes())
        if {(r['source_id'],r['native_frame']) for r in report['rows']}!={(r['source_id'],r['native_frame']) for r in g['queries']}:
            raise ValueError('native observation group is incomplete or from another query domain')
        request_raw=Path(g['request_path']).read_bytes()
        if hashlib.sha256(request_raw).hexdigest()!=g['request_file_sha256']:
            raise ValueError('native observation request changed')
        report_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        reports[report_sha]=dict(path=str(path),sha256=report_sha,
            content_sha256=digest(report),request_file_sha256=g['request_file_sha256'])
        for sha in g['input_snapshot_sha256']:
            if sha in group_by_snapshot:raise ValueError('snapshot belongs to multiple observation requests')
            group_by_snapshot[sha]=(report,request_raw)
    face_paths={r['reference_id']:r['face_cue_path'] for r in parent['references']}
    parent_shas={r['reference_id']:r['feature_snapshot_sha256'] for r in parent['references']}
    snapshots=dict(parent['feature_snapshot_by_reference']);rows=[];changed_refs={}
    # Finish all validation/censuses before writing any new snapshot.
    prepared=[]
    for ref,path in snapshots.items():
        if parent_shas[ref] not in group_by_snapshot:continue
        old=StageEditFeatureSnapshot.from_dict(json.loads(Path(path).read_bytes()),schema=schema)
        if old.snapshot_sha256!=parent_shas[ref]:raise ValueError('parent snapshot identity changed')
        report,request=group_by_snapshot[old.snapshot_sha256]
        face=FaceCueTable.from_dict(json.loads(Path(face_paths[ref]).read_bytes()))
        new=augment_missing_native_faces(old,face,report,request,schema=schema)
        new.require_production_causal_sensing(edit_mode='group',schema=schema)
        if StageEditFeatureSnapshot.from_dict(new.to_dict(),schema=schema)!=new:
            raise ValueError('completed native-face snapshot failed strict roundtrip')
        changed=require_input_only_change(old,new,schema)
        prepared.append((ref,new));changed_refs[ref]=dict(previous_snapshot_sha256=old.snapshot_sha256,
            feature_snapshot_sha256=new.snapshot_sha256,newly_observed_candidate_rows=changed)
    if len(prepared)!=len(group_by_snapshot):raise ValueError('some observed input snapshots were not supplied by the parent')
    a.output_directory.mkdir(parents=True,exist_ok=True)
    for ref,new in prepared:
        destination=a.output_directory/ref;destination.mkdir(exist_ok=True)
        snapshots[ref]=str(publish(destination/('feature-snapshot-'+new.snapshot_sha256+'.json'),new.to_dict()))
    for r in parent['references']:
        r=dict(r)
        if r['reference_id'] in changed_refs:r.update(changed_refs[r['reference_id']])
        rows.append(r)
    output=dict(parent,contract='source_exact_native_face_completion_corpus_v1',
        prepared_inputs_parent=str(a.prepared_inputs),prepared_inputs_parent_sha256=hashlib.sha256(raw_parent).hexdigest(),
        feature_snapshot_by_reference=snapshots,references=rows,native_face_observation_reports=reports,
        newly_observed_candidate_rows_by_reference={**parent.get('newly_observed_candidate_rows_by_reference',{}),
            **{r:v['newly_observed_candidate_rows'] for r,v in changed_refs.items()}},
        primary_face_geometry_and_cosines_updated=True,source_pool_clock_pose_and_gates_changed=False,
        supervision_files_changed=False,original_caches_overwritten=False,media_decodes=0,neural_forwards=0)
    target=publish(a.output_directory/('face-corpus-'+digest(output)+'.json'),output)
    print(json.dumps(dict(prepared_inputs=str(target),updated_references=len(prepared),
        newly_observed_candidate_rows=output['newly_observed_candidate_rows_by_reference'],
        new_media_decodes=0,new_neural_calls=0,training_performed=False,professional_quality_approved=False)))


if __name__=='__main__':main()
