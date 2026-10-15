"""Admit the actual hold's affine renderer clock BEFORE source selection.

Cached grids may use ties-to-even or a nonlinear warp. Neither can be
relabelled as the renderer's segment arithmetic. Reject only the mismatching
hold alternative, retaining other sources, durations and cameras. No offsets,
audio resync, cache rewrite or post-selection repair is performed.
"""
import numpy as np


def attach(banks,warps,clocks,nodes,*,master_start,fps):
    for bank,warp,clock in zip(banks,warps,clocks,strict=True):
        bank['programme_render_clock']=dict(fps=float(clock['fps']),count=int(clock['frame_count']),
            source_times=np.asarray([warp.master_to_source(master_start+int(n)/fps) for n in nodes],float))


def indices(first,rate,ticks,*,fps,native_fps):
    # Preserve even the production timestamp->milliseconds->seconds step.
    seconds=first+(np.asarray(ticks)/fps)*rate
    return (((seconds*1000.)/1000.)*native_fps+.5).astype(np.int64)


def admit(bank,nodes,i,j,entries,exits,*,fps):
    shape=(len(entries),len(exits));result=np.ones(shape,bool)
    clock=bank.get('programme_render_clock')
    if clock is None:return result
    first,final=map(int,np.asarray(nodes)[[i,j]]);a,b=clock['source_times'][[i,j]]
    duration=(final-first)/fps;rate=float((b-a)/duration)
    if a<0 or rate<=0 or not np.isfinite([a,b,rate]).all():return np.zeros(shape,bool)
    points=np.clip(np.asarray(nodes)[:,None]+[-7,-1,0,6],0,int(nodes[-1])-1)
    known=(points>=first)&(points<final)
    actual=indices(a,rate,points[known]-first,fps=fps,native_fps=clock['fps'])
    if np.any(actual!=bank['pose_native_indices'][known]):return np.zeros(shape,bool)
    # Face inputs are independently bound even when their indices happen to
    # equal the body endpoints; do not infer that from a successful body test.
    points=np.clip(np.asarray(nodes)[:,None]+[-1,0],0,int(nodes[-1])-1);known=(points>=first)&(points<final)
    actual=indices(a,rate,points[known]-first,fps=fps,native_fps=clock['fps'])
    if np.any(actual!=bank['native_indices'][known]):return np.zeros(shape,bool)
    left=np.ones(len(entries),bool);right=np.ones(len(exits),bool)
    for ports,mask,n in ((entries,left,i),(exits,right,j)):
        for k,port in enumerate(ports):
            context=port.observation.get('dense_subject_context')
            if context is None:continue
            offsets=context['clock'].output_frames
            start=a+(int(nodes[n])-first+int(offsets[0]))/fps*rate
            wanted=context['clock'].native_indices
            actual=indices(start,rate,np.arange(len(offsets)),fps=fps,native_fps=clock['fps'])
            mask[k]=bool(start>=0 and np.array_equal(actual,wanted))
    return result&left[:,None]&right[None,:]
