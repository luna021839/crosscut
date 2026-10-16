"""Bounded native-face completion over target-free candidate inputs only.

No supervision/trajectory files, training, downloads or renders are loaded.
An explicit --observe starts NN work. Per-source journals preserve completed
measurements after interruption; --resume recovers only the identical plan.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time

from crosscut.ive import IVEMember
from crosscut.stage_edit_face_cues import FaceCueTable
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_native_face_completion import CONTRACT, digest, missing_queries, sources_from_request
from crosscut.stage_edit_transformer import StageEditTokenSchema


def pairs(values):
    result = {}
    for value in values:
        ref, sep, name = value.partition('=')
        if not sep or not ref or not name or ref in result:
            raise ValueError('one distinct REFERENCE=/absolute/path is required')
        if not Path(name).is_absolute():
            raise ValueError('an absolute input path is required')
        result[ref] = Path(name)
    return result


def prepare(snapshot_paths, face_paths, request_paths, schema):
    if set(snapshot_paths) != set(face_paths) or not snapshot_paths or len(snapshot_paths) > 32:
        raise ValueError('one existing face table per explicit input snapshot is required')
    snapshots = {r: StageEditFeatureSnapshot.from_dict(json.loads(p.read_bytes()), schema=schema)
        for r, p in snapshot_paths.items()}
    faces = {r: FaceCueTable.from_dict(json.loads(p.read_bytes())) for r, p in face_paths.items()}
    spaces = {digest(t.descriptor_space.to_dict()) for t in faces.values()}
    galleries = {t.gallery_identity for t in faces.values()}
    if len(spaces) != 1 or len(galleries) != 1:
        raise ValueError('one consistent original gallery and descriptor space is required')
    for r, s in snapshots.items():
        if dict(s.provenance.upstream_sha256).get('face_member_cues') != faces[r].content_sha256:
            raise ValueError('the face table is not this snapshot input')
    groups = []; assigned = set()
    for path in request_paths:
        raw = path.read_bytes(); request = json.loads(raw); sources = sources_from_request(request)
        refs = [r for r,s in snapshots.items()
            if {c.source_id for t in s.sequence.tokens for c in t.source_candidates} == set(sources)]
        if not refs or assigned.intersection(refs):
            raise ValueError('source requests must partition the explicit input snapshots')
        for r in refs:
            if dict(faces[r].source_media_sha256) != {s:v['media_content_sha256'] for s,v in sources.items()}:
                raise ValueError('existing face inputs belong to different source contents')
        assigned.update(refs)
        queries = missing_queries([snapshots[r] for r in refs], request, schema=schema)
        groups.append(dict(request_file_sha256=hashlib.sha256(raw).hexdigest(),
            request_path=str(path), request=request, refs=sorted(refs),
            input_snapshot_sha256=sorted(snapshots[r].snapshot_sha256 for r in refs),
            queries=[dict(source_id=s,native_frame=n,master_seconds=list(t)) for (s,n),t in queries.items()]))
    if assigned != set(snapshots):
        raise ValueError('some input snapshots have no exact source request')
    return groups, next(iter(faces.values()))


def choose_primary(faces, width=960, height=540):
    """Deterministic original-viewport focal; no label, member or identity rank."""
    if faces is None or not len(faces):
        return None
    def key(face):
        x,y,w,h = map(float, face[:4]); cx=(x+w/2)/width
        return (-(w*h)*math.exp(-((cx-.5)/.4)**2), x, y)
    return min(faces, key=key)


def read_journal(path, source, desired, *, resume):
    """Keep complete native measurements; preserve an interrupted tail separately."""
    if not path.exists():return {}
    if not resume:raise FileExistsError('a measurement journal requires explicit recovery')
    data=path.read_bytes();rows={};offset=0
    for line in data.splitlines(keepends=True):
        try:row=json.loads(line)
        except (ValueError,UnicodeError):
            if line.endswith(b'\n') or offset+len(line)!=len(data):raise
            backup=path.with_name(path.name+'.interrupted-tail')
            if backup.exists() and backup.read_bytes()!=line:
                raise ValueError('a different interrupted journal tail already exists')
            if not backup.exists():backup.write_bytes(line)
            with path.open('r+b') as f:f.truncate(offset)
            break
        if (row['source_id']!=source['source_id'] or row['media_content_sha256']!=source['media_content_sha256']
                or type(row['native_frame']) is not int or row['native_frame'] not in desired
                or row['native_frame'] in rows):
            raise ValueError('source-native measurement journal differs')
        rows[row['native_frame']]=row;offset+=len(line)
        if not line.endswith(b'\n'):
            with path.open('ab') as f:f.write(b'\n')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', action='append', required=True)
    p.add_argument('--face-table', action='append', required=True)
    p.add_argument('--request', type=Path, action='append', required=True)
    for key in ('token-schema','output','member-bank','yujin-bank','arcface-model'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--ssd-device',type=int,required=True)
    p.add_argument('--seconds',type=float,default=900.)
    p.add_argument('--maximum-frames',type=int,default=8000)
    p.add_argument('--observe',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(); volume=Path('/Volumes/CrossCutMedia'); started=time.monotonic()
    if not 0<a.seconds<=1800 or not 1<=a.maximum_frames<=20000:
        p.error('one bounded observation allowance is required')
    if (not volume.is_mount() or volume.stat().st_dev!=a.ssd_device
            or not a.output.resolve().is_relative_to(volume.resolve())):
        raise OSError('explicit output must remain on the authorized SSD')
    schema=StageEditTokenSchema.from_dict(json.loads(a.token_schema.read_bytes()))
    groups,table=prepare(pairs(a.snapshot),pairs(a.face_table),a.request,schema)
    count=sum(len(g['queries']) for g in groups)
    if count>a.maximum_frames:raise ValueError('complete input query domain exceeds the explicit frame allowance')
    plan=dict(contract=CONTRACT,groups=groups,gallery_identity=table.gallery_identity,
        descriptor_space=table.descriptor_space.to_dict(),
        detector_sha256=hashlib.sha256(Path('engine/crosscut/models/face_detection_yunet_2023mar.onnx').read_bytes()).hexdigest(),
        primary_selection='largest_visible_face_with_original_viewport_center_weight',
        input_selection='all_available_snapshot_candidates_with_absent_face_inputs')
    plan_sha=digest(plan)
    print(json.dumps(dict(prepared_frames=count,groups=len(groups),snapshot_count=len(a.snapshot),
        query_plan_sha256=plan_sha,observation_started=a.observe,teacher_labels_consumed=False)),flush=True)
    if not a.observe:return
    if a.resume:
        if digest(json.loads((a.output/'plan.json').read_bytes()))!=plan_sha:
            raise ValueError('resume input/source/gallery/query plan changed')
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        with (a.output/'plan.json').open('x') as f:json.dump(plan,f,allow_nan=False)
    import cv2
    import psutil
    from audit_stageedit_face_spaces import verify_source, _file_identity, observed_face_geometry
    from crosscut.ive_local_arcface_gallery import LocalIVEArcFaceGallery
    from crosscut.native_face_descriptor import NativeArcFaceR50, NativeFaceDescriptorSpace, ARCFACE_MODEL_NAME
    process=psutil.Process(); peak=0; calls=dict(decoded_frames=0,detector=0,arcface=0,reused=0,frame_unavailable=0)
    def live():
        nonlocal peak
        peak=max(peak,process.memory_info().rss)
        if (not volume.is_mount() or volume.stat().st_dev!=a.ssd_device
                or time.monotonic()-started>a.seconds or peak>3*2**30
                or shutil.disk_usage(volume).free<2**30+32*2**20):
            raise RuntimeError('declared face observation resource/source allowance exhausted')
    finished=threading.Event()
    def watch():
        while not finished.wait(1.):
            try:live()
            except Exception as exc:
                try:
                    with (a.output/'resource-stop.json').open('w') as f:json.dump(dict(error=str(exc),calls=calls),f)
                finally:os._exit(75)
    threading.Thread(target=watch,daemon=True).start()
    try:
        gallery=LocalIVEArcFaceGallery.from_existing_photo_banks(a.member_bank,a.yujin_bank)
        arc=NativeArcFaceR50(a.arcface_model)
        space=NativeFaceDescriptorSpace(ARCFACE_MODEL_NAME,512,arc.asset_sha256,
            hashlib.sha256(Path('engine/crosscut/native_face_descriptor.py').read_bytes()).hexdigest())
        if gallery.identity!=table.gallery_identity or space!=table.descriptor_space:
            raise ValueError('current model/gallery differs from the existing input space')
        cv2.setNumThreads(2)
        detector=cv2.FaceDetectorYN.create('engine/crosscut/models/face_detection_yunet_2023mar.onnx','',(960,540),.75,.3,5000)
        for group in groups:
            request=group['request']; sources=sources_from_request(request); results=[]
            for sid,source in sources.items():
                desired={q['native_frame'] for q in group['queries'] if q['source_id']==sid}
                if not desired:continue
                cache=a.output/(digest([group['request_file_sha256'],sid])+'-rows.jsonl')
                retained=read_journal(cache,source,desired,resume=a.resume)
                calls['reused']+=len(retained)
                identity=verify_source(source,device=a.ssd_device,check=live)
                capture=cv2.VideoCapture(str(source['path']))
                if not capture.isOpened():raise OSError('cannot open retained source')
                try:
                    if abs(capture.get(cv2.CAP_PROP_FPS)-source['frame_rate'])>max(.001,source['frame_rate']*.0001):
                        raise ValueError('actual source cadence differs from the native request')
                    with cache.open('a') as f:
                        for native in sorted(desired-set(retained)):
                            live();current=round(capture.get(cv2.CAP_PROP_POS_FRAMES))
                            if current<=native and native-current<=30:
                                for _ in range(native-current):
                                    if not capture.grab():raise OSError('native source advance failed')
                            elif not capture.set(cv2.CAP_PROP_POS_FRAMES,native):
                                raise OSError('native source seek failed')
                            ok,pixels=capture.read();live()
                            if identity!=_file_identity(Path(source['path'])):raise OSError('source changed during observation')
                            row=dict(source_id=sid,native_frame=native,media_content_sha256=source['media_content_sha256'])
                            if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES))-1!=native:
                                row['state']='FRAME_UNAVAILABLE'
                                calls['frame_unavailable']+=1
                            else:
                                calls['decoded_frames']+=1
                                small=cv2.resize(pixels,(960,540),interpolation=cv2.INTER_AREA)
                                _,faces=detector.detect(small);calls['detector']+=1
                                face=choose_primary(faces)
                                row['state']='NO_FACE'
                                if face is not None:
                                    face=face.copy();height,width=pixels.shape[:2]
                                    face[:14:2]*=width/960;face[1:14:2]*=height/540
                                    try:box,landmarks,_=observed_face_geometry(face,width,height)
                                    except ValueError:row['state']='LANDMARKS_UNAVAILABLE'
                                    else:
                                        vector=arc.embedding(pixels,bbox_xywh=box,landmarks_xy=landmarks);calls['arcface']+=1
                                        hint=gallery.classify(vector,descriptor_space=space)
                                        row.update(state='OBSERVED',face_bbox_xywh=box,
                                            member_cosines=[hint['member_scores'][m.value] for m in IVEMember])
                                        del vector
                            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();retained[native]=row
                            if sum(calls[k] for k in ('detector','reused','frame_unavailable'))%200==0:
                                print(json.dumps(dict(source=sid,calls=calls,seconds=time.monotonic()-started)),flush=True)
                finally:capture.release()
                if identity!=_file_identity(Path(source['path'])):raise OSError('source changed after observation')
                results.extend(retained[n] for n in sorted(retained))
            report=dict(contract=CONTRACT,request_file_sha256=group['request_file_sha256'],
                input_snapshot_sha256=group['input_snapshot_sha256'],gallery_identity=gallery.identity,
                descriptor_space=space.to_dict(),source_media_sha256={s:v['media_content_sha256'] for s,v in sources.items()},
                current_source_contents_verified_against_request=True,
                selection=plan['input_selection'],teacher_labels_consumed=False,
                raw_query_or_reference_vectors_serialized=False,rows=results)
            target=a.output/(group['request_file_sha256']+'-observations.json')
            encoded=json.dumps(report,allow_nan=False)
            if target.exists():
                if target.read_text()!=encoded:raise ValueError('completed face observation report differs')
            else:
                with target.open('x') as f:f.write(encoded)
        print(json.dumps(dict(completed=True,calls=calls,elapsed_seconds=time.monotonic()-started,
            peak_RSS_bytes=peak,professional_quality_approved=False)),flush=True)
    finally:finished.set()


if __name__=='__main__':main()
