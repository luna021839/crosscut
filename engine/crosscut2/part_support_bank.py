"""Additive native-pixel part support; never rerun/invalidate pose models.

One small portable source snapshot retains completed physical intervals across
interruption and changed editorial trials. It stores no pictures or new neural
features. Part paths corroborate observations, not named/anatomical identity.
"""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic
import cv2
import numpy as np
from . import temporal_parts
from .grid_perception import file_identity
from .source_cuts import read_frames
from .observation_cache_contract import function_text


def interval_key(a,b,first,final,heights):
    return hashlib.sha256(np.asarray([a,b,*heights],np.float64).tobytes()
        +np.asarray([first,final],np.float32).tobytes()).hexdigest()


def _sequential_rasters(path,needed,*,check):
    values=sorted(needed);ranges=[]
    for value in values:
        if ranges and value==ranges[-1][1]+1:ranges[-1][1]=value
        else:ranges.append([value,value])
    expression='+'.join(f'between(n\\,{a}\\,{b})' for a,b in ranges)
    with tempfile.TemporaryFile() as errors:
        p=subprocess.Popen([shutil.which('ffmpeg'),'-nostdin','-v','error','-threads','2','-i',str(path),
            '-map','0:v:0','-an','-filter_threads','1','-vf','select='+expression+',scale=960:540:flags=area',
            '-fps_mode','passthrough','-pix_fmt','bgr24','-threads','1','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=errors)
        try:
            count=0
            for pixels in read_frames(p.stdout,check=check,width=960,height=540):
                if count>=len(values):raise OSError('part-support decoder emitted an undeclared physical ordinal')
                yield values[count],pixels;count+=1
            code=p.wait(timeout=10.);errors.seek(0);diagnostic=errors.read(4096)
            if code or diagnostic or count!=len(values):raise OSError('native part support decode failed: '+diagnostic.decode(errors='replace'))
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5.)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5.)
            p.stdout.close()


def cache_implementation():
    """Preserve canonical pixel/part caches across a bit-exact seek change.

    The entire temporal algorithm, interval key and consumer source are still
    hashed. Only the operational reader is excluded. Any other algorithm edit
    creates a new key, rather than silently trusting an old observation.
    """
    here=Path(__file__)
    consumer=function_text(here,'corroborate').replace('implementation=cache_implementation()',
        'implementation=hashlib.sha256(Path(temporal_parts.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()')
    canonical_reader=function_text(here,'_sequential_rasters').replace('def _sequential_rasters(','def _rasters(')
    semantic=hashlib.sha256(Path(temporal_parts.__file__).read_bytes()+function_text(here,'interval_key').encode()+consumer.encode()
        +canonical_reader.encode()+function_text(here.with_name('source_cuts.py'),'read_frames').encode()).hexdigest()
    if semantic=='579a27339f6227f191fed135f2930563fdce3f72ae99a063b811d149d6d07c13':
        return '447432eab91d837e40edf01d482a3ef7302c210e24efa59a189266c55aba4abb'
    return semantic


def _rasters(path,needed,*,check,maximum_seconds=120.,allow_sequential_fallback=True,receipt=None):
    """Verified sparse native frames; strict queues cannot hide a full scan."""
    from . import sparse_native_frames
    values=sorted(needed)
    if not values:return
    report={} if receipt is None else receipt
    try:
        clock=sparse_native_frames.probe(path,check=check)
        sparse_native_frames.windows(values,clock)
    except sparse_native_frames.UnsupportedSeek as error:
        report.update(strategy='explicit_legacy_sequential_fallback',sparse_unavailable_reason=str(error),
            whole_source_sequential_strategy_requested=True)
        if not allow_sequential_fallback:
            raise sparse_native_frames.UnsupportedSeek('bounded part queue refuses full-source fallback: '+str(error)) from error
        yield from _sequential_rasters(path,values,check=check)
    else:
        yield from sparse_native_frames.frames(path,values,clock=clock,check=check,
            maximum_seconds=maximum_seconds,receipt=report)


def corroborate(*,sources,states,data,census,cache,check,maximum_seconds=180.,maximum_new_intervals=6000):
    if not 1<=maximum_seconds<=600 or not 0<=maximum_new_intervals<=12000:
        raise ValueError('finite additive native part support allowance required')
    started=monotonic();last=started;cache=Path(cache);rows=[dict(d) for d in data];receipts=[];spent=0
    implementation=cache_implementation()
    def live():
        nonlocal last
        now=monotonic()
        if now>=last:check();last=now+.1
        if now-started>=maximum_seconds:raise TimeoutError('native part-support allowance exhausted; completed source intervals retained')
    for source,path in enumerate(sources):
        live();identity=file_identity(path);clock=census[source];cuts=np.asarray(clock['frames'],int)
        binding=json.dumps(dict(source=identity,clock={k:clock[k] for k in ('fps','frame_count')},
            implementation=implementation,decoder=file_identity(Path(shutil.which('ffmpeg')).resolve()),
            raster=[960,540],filter='area',opencv=cv2.__version__,numpy=np.__version__),sort_keys=True)
        target=cache/(hashlib.sha256(binding.encode()).hexdigest()+'.npz');saved={};dirty=False
        if target.is_file():
            with np.load(target,allow_pickle=False) as archive:
                if str(archive['binding'])!=binding:raise ValueError('part-support source/runtime binding changed')
                keys=archive['keys'];support=archive['support'];observed=archive['observed'];disagrees=archive['disagrees']
            if (keys.ndim!=1 or len(keys)!=len(set(keys)) or keys.dtype.kind!='U'
                    or support.shape!=(len(keys),8) or observed.shape!=support.shape or disagrees.shape!=support.shape
                    or any(x.dtype!=bool for x in (support,observed,disagrees))):
                raise ValueError('native part-support cache shape differs')
            saved={str(k):(s,o,d) for k,s,o,d in zip(keys,support,observed,disagrees)}
        def checkpoint():
            nonlocal dirty
            if not dirty:return
            check()
            if file_identity(path)!=identity:raise OSError('source changed before part-support checkpoint')
            cache.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.parts-',suffix='.npz',dir=cache)
            try:
                keys=sorted(saved)
                with os.fdopen(fd,'wb') as stream:
                    np.savez_compressed(stream,binding=binding,keys=np.asarray(keys,dtype='U64'),
                        support=np.asarray([saved[k][0] for k in keys],bool),observed=np.asarray([saved[k][1] for k in keys],bool),
                        disagrees=np.asarray([saved[k][2] for k in keys],bool))
                    stream.flush();os.fsync(stream.fileno())
                check()
                if file_identity(path)!=identity:raise OSError('source changed during part-support checkpoint')
                os.replace(tmp,target);dirty=False
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        requests={};links=[];source_rows=[]
        for i,(state,row) in enumerate(zip(states,rows)):
            if state.source!=source:continue
            source_rows.append(i);N=len(row['body']);row['part_pair_support']=np.ones((N,3,8),bool)
            row['pixel_part_scope']=np.zeros(N,bool)
            clock_indices=row.get('pose_native_indices')
            if clock_indices is None:raise ValueError('part support needs the exact pose physical ordinals')
            for n in range(N):
                if max(row['heads'][n,:,2])<.12 or row['formation'][n].any():continue
                body=row['body'][n]
                if not (body[:,:6,2]>=.3).any():continue
                row['pixel_part_scope'][n]=True;row['part_pair_support'][n]=False
                for pair,(first,final) in enumerate(((0,1),(1,2),(2,3))):
                    a,b=map(int,clock_indices[n,[first,final]])
                    if not 0<=a<b<clock['frame_count'] or b-a>12 or np.any((cuts>a)&(cuts<=b)):continue
                    heights=row['heads'][n,[first//2,final//2],2]
                    if np.any(heights<=.019) or not ((body[first,:,2]>=.3)&(body[final,:,2]>=.3)).any():continue
                    key=interval_key(a,b,body[first],body[final],heights);links.append((i,n,pair,key))
                    if key not in saved:requests[key]=(a,b,body[first],body[final],heights)
        if spent+len(requests)>maximum_new_intervals:raise ValueError('new native intervals exceed the declared allowance')
        decoded=0;new=0
        try:
            if requests:
                needed=set();ends={}
                for key,(a,b,*_) in requests.items():needed.update(range(a,b+1));ends.setdefault(b,[]).append(key)
                buffer=deque()
                for native,pixels in _rasters(path,needed,check=live):
                    live();buffer.append((native,pixels));decoded+=1
                    while buffer and buffer[0][0]<native-12:buffer.popleft()
                    frames=dict(buffer)
                    for key in ends.get(native,()):
                        a,b,first,final,heights=requests[key]
                        sample=temporal_parts.observe([frames[f] for f in range(a,b+1)],first,final,head_heights=heights)
                        saved[key]=(sample.supported,sample.pixel_path_observed,sample.model_pixel_disagreement)
                        dirty=True;new+=1;spent+=1
                        if new%32==0:checkpoint()
                        if new%128==0:print(json.dumps(dict(phase='native_part_paths',source=source,completed_intervals=new,
                            requested_intervals=len(requests),elapsed_seconds=monotonic()-started)),flush=True)
            checkpoint();live()
            if file_identity(path)!=identity:raise OSError('source changed during part-support observation')
            for i,n,pair,key in links:rows[i]['part_pair_support'][n,pair]=saved[key][0]
            supported=0;unknown=0
            for i in source_rows:
                row=rows[i];raw=row['body'];filtered=raw.copy();pair=row['part_pair_support']
                phase=np.stack((pair[:,0],pair[:,0]|pair[:,1],pair[:,1]|pair[:,2],pair[:,2]),axis=1)
                filtered[...,2]=np.where(phase,raw[...,2],0.)
                row['body']=filtered;row['raw_body_hypotheses']=raw
                supported+=int((phase&row['pixel_part_scope'][:,None,None]).sum())
                unknown+=int((~phase&row['pixel_part_scope'][:,None,None]).sum())
            receipt=dict(source=source,new_physical_intervals=new,reused_physical_intervals=len({x[3] for x in links})-new,
                decoded_native_rasters=decoded,new_neural_calls=0,corroborated_part_slots=supported,unknown_part_slots=unknown,
                source_model_caches_unchanged=True,raw_joint_coordinates_unchanged=True,anatomical_ownership_certified=False,
                cache_bytes=target.stat().st_size if target.exists() else 0,elapsed_seconds=monotonic()-started)
            receipts.append(receipt);print(json.dumps(dict(phase='source_part_paths_complete',**receipt)),flush=True)
        finally:checkpoint()
    return rows,receipts
