"""Retain independent source-cut evidence, with current source/clock binding.

An adaptive detector miss does not revoke a cached TransNet observation. The
union is still heuristic, not full recall proof. No new NN/source decode and
no face/body cache key changes. Native first-incoming indices map through the
actual nearest-native sampling clock, not rounded timestamp labels.
"""
import hashlib
from pathlib import Path
import numpy as np
from .grid_perception import file_identity


def merge_frames(adaptive,transnet,*,count):
    a,t=np.asarray(adaptive,int),np.asarray(transnet,int)
    if any(v.ndim!=1 or np.any(v<=0) or np.any(v>=count) or np.any(np.diff(v)<=0) for v in (a,t)):
        raise ValueError('strict source-native first-incoming cut domains required')
    # Two detectors can localize the same transition slightly differently.
    # Keep the declared TransNet first-incoming frame, not two artificial
    # 1/2-frame camera units; never cluster two TransNet observations together.
    extra=[int(f) for f in a if not len(t) or np.min(np.abs(t-f))>2]
    return np.array(sorted(set(map(int,t))|set(extra)),int)


def nearest_indices(warp,*,master_start,fps,final_frame,native_fps,native_count):
    values=np.array([int(warp.master_to_source(master_start+tick/fps)*native_fps+.5)
        for tick in range(final_frame)],int)
    if np.any(np.diff(values)<0) or values.min()<0 or values.max()>=native_count:
        raise ValueError('source union escapes canonical physical sampling coverage')
    return values


def combine(*,sources,manifest,census,warps,master_start,fps,final_frame,check=lambda:None):
    records=manifest['scene_analysis']
    if not len(sources)==len(records)==len(census)==len(warps):raise ValueError('source cut evidence domains differ')
    events=[];bound=[];receipts=[]
    for s,(path,record,clock,warp) in enumerate(zip(sources,records,census,warps)):
        check();identity=file_identity(path)
        if Path(record['source']).resolve()!=Path(path).resolve() or record.get('source_index')!=s:
            raise ValueError('retained scene observations belong to another source')
        if record.get('detection_policy',{}).get('contract')!='transnetv2_source_native_first_incoming_frame_v1':
            raise ValueError('unrecognized legacy scene-clock domain; do not invent native indices')
        expected=record.get('persistent_scalar_cache',{}).get('source_sha256')
        if not isinstance(expected,str) or len(expected)!=64:raise ValueError('retained scene source digest required')
        digest=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):check();digest.update(chunk)
        if digest.hexdigest()!=expected or file_identity(path)!=identity:
            raise OSError('legacy native scene observations no longer match current source pixels')
        frames=[]
        for change in record['changes']:
            f=change['frame_index']
            if type(f) is not int:raise ValueError('first incoming native frame must be an integer')
            frames.append(f)
        union=merge_frames(clock['frames'],frames,count=clock['frame_count'])
        schedule=nearest_indices(warp,master_start=master_start,fps=fps,final_frame=final_frame,
            native_fps=clock['fps'],native_count=clock['frame_count'])
        output=np.unique(np.searchsorted(schedule,union,side='left'));output=output[(output>0)&(output<final_frame)]
        events.append(output.astype(int))
        bound.append(dict(source=s,source_file_identity=list(identity),source_sha256=expected,
            fps=clock['fps'],frame_count=clock['frame_count'],frames=union.tolist(),
            contract='bound_transnet_adaptive_first_incoming_union_v1',complete_semantic_cut_recall_claimed=False))
        receipts.append(dict(source=s,source_sha256_verified=True,adaptive_count=len(clock['frames']),
            retained_transnet_count=len(frames),union_count=len(union),mapped_output_events=len(output),
            new_neural_calls=0,new_source_decoded_frames=0,base_face_body_cache_invalidated=False,
            cut_mapping='first_output_tick_whose_nearest_native_ordinal_reaches_first_incoming',
            near_adaptive_events_deferred_to_transnet_first_incoming=True))
    return tuple(events),bound,receipts
