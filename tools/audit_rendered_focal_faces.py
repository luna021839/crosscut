"""Measure the encoded film's faces independently of its planned targets.

Keep ALL output detections. Matching a planned background face must not hide
a larger visible foreground face. This is diagnostic geometry, not identity,
continuous audiovisual/professional acceptance, or a render rejection gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from crosscut2 import sparse_native_frames as sparse
from crosscut2.focal_output_correspondence import match,predictions_by_tick


def checksum(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4*1024**2):digest.update(block)
    return digest.hexdigest()


def output_raster(movie):
    """Keep native square-pixel16:9 geometry; never squash portrait members.

    The current focal correspondence uses a16:9 physical metric. Other
    aspects need their own matching geometry, not a resized diagnostic.
    """
    result=subprocess.run([shutil.which('ffprobe') or 'ffprobe','-v','error',
        '-select_streams','v:0','-show_entries',
        'stream=width,height,sample_aspect_ratio:stream_side_data=rotation','-of','json',str(movie)],
        check=True,capture_output=True,timeout=10.)
    streams=json.loads(result.stdout)['streams']
    if len(streams)!=1:raise ValueError('one encoded video raster required')
    stream=streams[0];width,height=stream['width'],stream['height']
    if (type(width) is not int or type(height) is not int or min(width,height)<1
            or width*height>3840*2160 or width*9!=height*16
            or stream.get('sample_aspect_ratio')!='1:1'
            or any(row.get('rotation',0)!=0 for row in stream.get('side_data_list',[]))):
        raise ValueError('diagnostic currently requires native square-pixel unrotated16:9 GROUP output')
    return width,height


def run(movie,plan,output,*,authorized_root,device,expected_sha256):
    started=time.monotonic();movie=Path(movie);plan=Path(plan);output=Path(output)
    root=Path(authorized_root).resolve(strict=True)
    if (not all(p.resolve().is_relative_to(root) for p in (movie,plan,output))
            or output.exists() or output.with_suffix('.claim.json').exists()):
        raise ValueError('fresh final-face diagnostic inside authorized media root required')
    if checksum(movie)!=expected_sha256:raise ValueError('exact completed encoded film required')
    record=json.loads(plan.read_bytes());prediction=record['render_plan']
    if Path(prediction['output'])!=movie:raise ValueError('encoded film and selected plan differ')
    width,height=output_raster(movie)
    identity=sparse.file_identity(movie)
    def check():
        if root.stat().st_dev!=device or sparse.file_identity(movie)!=identity:raise OSError('diagnostic movie or SSD changed')
        if time.monotonic()-started>150.:raise TimeoutError('one finite final-output face pass exhausted')
    check();clock=sparse.probe(movie,check=check)
    count=prediction['output_frame_count'];seams=record['seams']
    if count!=clock.count:raise ValueError('completed output frame count differs')
    wanted=sorted({t for row in seams for t in (row['output_frame']-1,row['output_frame'])})
    if not wanted or len(wanted)>512:raise ValueError('one bounded all-authored-boundary pass required')
    expected=predictions_by_tick(seams,count);model=Path('engine/crosscut/models/face_detection_yunet_2023mar.onnx')
    with output.with_suffix('.claim.json').open('x') as stream:json.dump(dict(movie_sha256=expected_sha256,
        maximum_output_face_forwards=len(wanted),maximum_seconds=150.,original_source_NN=0,
        automatic_retry=False),stream,indent=2)
    cv2.setNumThreads(2)
    detector=cv2.FaceDetectorYN.create(str(model),'',(width,height),.75,.3,5000)
    observed={};proof={}
    for tick,pixels in sparse.frames(movie,wanted,clock=clock,check=check,width=width,height=height,
            receipt=proof,maximum_seconds=120.,maximum_decode_seconds=12.):
        check();_,found=detector.detect(pixels);faces=[]
        if found is not None:
            for f in sorted(found,key=lambda v:float(v[3]),reverse=True)[:8]:
                if f[3]/height>.019:
                    faces.append(dict(center=[float((f[0]+f[2]/2)/width),float((f[1]+f[3]/2)/height)],
                        height=float(f[3]/height),confidence=float(f[-1])))
        selected,correspondence=match(faces,expected[tick])
        observed[tick]=dict(faces=faces,planned_correspondence=correspondence,
            planned_target_is_largest_detected=(selected==0) if selected is not None else None,
            raster_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    if sorted(observed)!=wanted:raise ValueError('incomplete final-output face measurements')
    rows=[]
    for row in seams:
        tick=row['output_frame'];a,b=observed[tick-1],observed[tick]
        actual_ratio=None
        if a['faces'] and b['faces']:
            ha,hb=a['faces'][0]['height'],b['faces'][0]['height'];actual_ratio=max(ha/hb,hb/ha)
        pa,pb=expected[tick-1]['height'],expected[tick]['height']
        rows.append(dict(output_frame=tick,seconds=row['seconds'],source_pair=[row['left_source'],row['right_source']],
            planned_face_ratio=max(pa/pb,pb/pa) if min(pa,pb)>0 else None,
            actual_largest_face_ratio=actual_ratio,
            planned_target_is_largest_detected=[a['planned_target_is_largest_detected'],b['planned_target_is_largest_detected']]))
    result=dict(movie=str(movie),movie_sha256=expected_sha256,plan_sha256=checksum(plan),
        detector_sha256=checksum(model),detector_input_raster=[width,height],output_face_forwards=len(observed),
        original_source_NN=0,original_source_decodes=0,elapsed_seconds=time.monotonic()-started,
        planned_targets_matched=sum(o['planned_correspondence']['matched'] for o in observed.values()),
        planned_targets_are_largest_detected=sum(o['planned_target_is_largest_detected'] is True for o in observed.values()),
        authored_cuts=rows,observations=observed,decode=proof,
        largest_face_is_not_named_identity=True,professional_quality_approved=False,automatic_render_gate=False)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('authored_cuts','observations','decode')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('movie','plan','output','authorized-root'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',type=int,required=True);parser.add_argument('--expected-sha256',required=True)
    args=parser.parse_args()
    run(args.movie,args.plan,args.output,authorized_root=args.authorized_root,device=args.device,expected_sha256=args.expected_sha256)
