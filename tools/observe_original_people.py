"""One finite original-frame whole-person pass; explicit input/output only.

No acquisition, training, editorial selection or render. Checkpoints are
source/producer-bound; a selector change does not invalidate the raw sensor.
"""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time
import cv2
import numpy as np
import psutil
from crosscut2 import sparse_native_frames as sparse
from crosscut2 import multi_person_pose,rtmo_original_people
from crosscut2.original_people_bank import FAMILY,save_chunk,load_chunk


def run(request,output,*,plan_only=False):
    raw=Path(request).read_bytes();spec=json.loads(raw)
    root=Path(spec['authorized_root']).resolve(strict=True)
    output=Path(output);cache=output.with_suffix('')
    if not output.parent.resolve(strict=True).is_relative_to(root) or output.exists():
        raise ValueError('fresh output in the authorized project required')
    started=time.monotonic();process=psutil.Process();peak=0;sensor=None;next_memory_check=0.
    maximum=spec['maximum_seconds'];limit=spec['maximum_NN_calls']
    if (not 1<=maximum<=1200 or type(limit) is not int or not 1<=limit<=12000
            or len(spec['sources'])>16):raise ValueError('finite original-frame allowance required')
    def check():
        nonlocal peak,next_memory_check
        if not Path(spec['mount']).is_mount() or root.stat().st_dev!=spec['ssd_device']:
            raise OSError('authorized original-source SSD disconnected')
        # read_frames calls this while draining every pipe chunk. A full
        # process-table census per read can cost more than the model itself.
        now=time.monotonic()
        if now>=next_memory_check:
            rss=0
            for p in [process,*process.children(recursive=True)]:
                try:rss+=p.memory_info().rss
                except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
            peak=max(peak,rss);next_memory_check=now+.5
        if peak>4*2**30:raise MemoryError('original-frame observation exceeded4GiB')
        if time.monotonic()-started>maximum:raise TimeoutError('finite original-frame allowance exhausted')
        if sensor is not None and sensor.forwards>limit:raise ValueError('original-frame NN limit exceeded')
    prepared=[];total=0
    # Consumer association/DP fixes must not demand new sensor inference.
    producer=hashlib.sha256((inspect.getsource(multi_person_pose.tensor)+
        inspect.getsource(multi_person_pose.decode)+inspect.getsource(rtmo_original_people.Sensor)).encode()).hexdigest()
    for s,entry in enumerate(spec['sources']):
        check();source=Path(entry['path'])
        if source.is_symlink() or not source.resolve(strict=True).is_relative_to(root):raise ValueError('source outside project')
        if list(sparse.file_identity(source))!=entry['identity']:raise ValueError('declared source changed')
        clock=sparse.probe(source,check=check)
        wanted=entry['native_indices'];width,height=entry['raster']
        windows=sparse.windows(wanted,clock,maximum_decode_seconds=12.,
            maximum_selected_frames=min(32,64*1024**2//(width*height*3)))
        total+=len(wanted)
        binding=dict(family=FAMILY,source_sha256=entry['sha256'],model_sha256=multi_person_pose.MODEL_SHA256,
            numerical_producer_sha256=producer,minimum_score=.1,NMS_IoU=.65,raster=[width,height])
        prepared.append((source,clock,binding))
        print(json.dumps(dict(phase='original_people_source_preflight',source=s,frames=len(wanted),
            sparse_windows=len(windows),raster=[width,height])),flush=True)
    if total>limit:raise ValueError('declared all-source request exceeds NN limit')
    if plan_only:
        print(json.dumps(dict(phase='original_people_plan_only',frames=total,new_NN=0)),flush=True);return
    cache.mkdir(exist_ok=True);cv2.setNumThreads(2)
    references=[];reused=0;completed=0
    try:
        for s,(entry,(source,clock,binding)) in enumerate(zip(spec['sources'],prepared,strict=True)):
            check();wanted=entry['native_indices'];chunks=[];decode_receipts=[]
            for first in range(0,len(wanted),64):
                check();selected=wanted[first:first+64]
                key=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()+np.asarray(selected,np.int64).tobytes()).hexdigest()
                path=cache/(key+'.npz')
                if path.exists():
                    rows=load_chunk(path,binding=binding)
                    if sorted(rows)!=selected:raise ValueError('checkpoint native schedule differs')
                    reused+=len(selected)
                else:
                    if sensor is None:sensor=rtmo_original_people.Sensor(spec['model'],coreml_cache=spec['coreml_cache'],minimum_score=.1)
                    rows=[];receipt={}
                    for native,image in sparse.frames(source,selected,clock=clock,check=check,
                            width=binding['raster'][0],height=binding['raster'][1],receipt=receipt,
                            maximum_seconds=120.,maximum_decode_seconds=12.):
                        check()
                        if sensor.forwards>=limit:raise ValueError('original-frame NN limit exhausted')
                        rows.append(dict(native=native,raster_sha256=hashlib.sha256(image.tobytes()).hexdigest(),**sensor.describe(image)))
                    if [v['native'] for v in rows]!=selected:raise ValueError('source-first frame coverage differs')
                    save_chunk(path,rows,binding=binding);decode_receipts.append(receipt)
                completed+=len(selected)
                chunks.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                print(json.dumps(dict(phase='original_people_checkpoint',source=s,completed=completed,total=total,
                    new_NN=0 if sensor is None else sensor.forwards,elapsed_seconds=time.monotonic()-started)),flush=True)
            references.append(dict(source_identity=entry['identity'],binding=binding,chunks=chunks,decode_receipts=decode_receipts))
        check()
        record=dict(family=FAMILY,request_sha256=hashlib.sha256(raw).hexdigest(),sources=[r['path'] for r in spec['sources']],
            source_sha256=[r['sha256'] for r in spec['sources']],source_records=references,
            new_NN_calls=0 if sensor is None else sensor.forwards,reused_original_frames=reused,
            observation_frames=completed,elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=peak,
            sensor_receipt=None if sensor is None else sensor.receipt,anatomical_ownership_certified=False,
            final_selection_or_render_executed=False)
        with output.open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
        print(json.dumps(dict(phase='original_people_completed',frames=completed,new_NN=record['new_NN_calls'],
            elapsed_seconds=record['elapsed_seconds'],peak_RSS_bytes=peak,reference=str(output))),flush=True)
    finally:
        if sensor is not None:sensor.close()
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    run(args.request,args.output,plan_only=args.plan_only)
