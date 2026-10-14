"""Measured two-sided crop proposals, frozen within each native camera shot.

This is a camera domain, not an edit or ownership verdict. No beat strength,
reference-film time or desired cut count selects its representatives. Native
remains available; the three other paths cover actual proposed transforms.
"""
import math
import numpy as np
from .paired_geometry import Camera


def pair_proposals(a,b,*,caps):
    """Keep feasible common-size candidates, not one comfort-prior argmin."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if (a.shape!=(4,) or b.shape!=(4,) or not np.isfinite([*a,*b,*caps]).all()
            or min(a[2],b[2])<=0 or any(not 1<=v<=2 for v in caps)):
        raise ValueError('actual finite faces and source zoom capacities required')
    lo=max(a[2],b[2]);hi=min(a[2]*caps[0],b[2]*caps[1])
    targets=np.linspace(lo,hi,3) if lo<=hi else np.array([lo])
    results=[]
    for target in targets:
        za,zb=np.clip(target/a[2],1.,caps[0]),np.clip(target/b[2],1.,caps[1])
        lows=[1.+z*(h[:2]-1.) for h,z in ((a,za),(b,zb))]
        highs=[z*h[:2] for h,z in ((a,za),(b,zb))]
        low,high=np.maximum(*lows),np.minimum(*highs)
        middle=(a[:2]+b[:2])/2
        common=np.where(low<=high,np.clip(middle,low,high),(low+high)/2)
        cameras=[]
        for head,z,left,right in ((a,za,lows[0],highs[0]),(b,zb,lows[1],highs[1])):
            position=np.clip(common,left,right)
            center=head[:2]+(.5-position)/z
            delta=center-.5;center=.5+delta*min(1.,.25/max(1e-12,float(np.linalg.norm(delta))))
            center=np.clip(center,.5/z,1.-.5/z)
            cameras.append(Camera(float(z),float(center[0]),float(center[1])))
        results.append(tuple(cameras))
    return tuple(results)


def representatives(candidates,*,limit=3):
    """Deterministic existing-camera k-centres, no synthesized average crop."""
    values=sorted(set(candidates)-{(1.,.5,.5)})
    if not values:return [],0.
    raw=np.asarray(values,float)
    feature=np.column_stack((np.log(raw[:,0]),2.*raw[:,1:]))
    first=int(np.argmin(np.linalg.norm(feature-np.median(feature,axis=0),axis=1)))
    chosen=[first];distance=np.linalg.norm(feature-feature[first],axis=1)
    while len(chosen)<min(limit,len(raw)):
        index=int(np.argmax(distance))
        if index in chosen:break
        chosen.append(index);distance=np.minimum(distance,np.linalg.norm(feature-feature[index],axis=1))
    return [tuple(map(float,raw[i])) for i in chosen],float(distance.max())


def make_paths(nodes,banks,source_events,*,check=lambda:None,maximum_candidates=2_000_000):
    nodes=np.asarray(nodes,int);final=int(nodes[-1]);count=0
    if len(banks)!=len(source_events) or nodes[0]!=0 or np.any(np.diff(nodes)<=0):
        raise ValueError('source-matched canonical camera grid required')
    boundaries=[np.unique(np.r_[0,np.asarray(v,int),final]) for v in source_events]
    if any(v[0]!=0 or v[-1]!=final or np.any(np.diff(v)<=0) for v in boundaries):
        raise ValueError('native camera units escape canonical source domain')
    pools=[[set() for _ in range(len(v)-1)] for v in boundaries]
    def heads(source,n,side):
        rows=banks[source]['heads'][n,side]
        group=rows[(rows[:,2]>.019)&(rows[:,3]>=.75)]
        if len(group)>=3 and group[:,2].max()<1.6*np.sort(group[:,2])[-2]:return ()
        return [r for r in rows[:3] if r[2]>=.055 and r[3]>=.75]
    for n,tick in enumerate(nodes[1:-1],start=1):
        check();left=[heads(s,n,0) for s in range(len(banks))];right=[heads(s,n,1) for s in range(len(banks))]
        for a in range(len(banks)):
            ua=int(np.searchsorted(boundaries[a],tick-1,side='right')-1)
            for b in range(len(banks)):
                if a==b:continue
                ub=int(np.searchsorted(boundaries[b],tick,side='right')-1)
                for fa in left[a]:
                    for fb in right[b]:
                        for ca,cb in pair_proposals(fa,fb,caps=(banks[a]['zoom_cap'],banks[b]['zoom_cap'])):
                            count+=2
                            if count>maximum_candidates:raise ValueError('measured camera proposal budget exceeded; no silent domain reduction')
                            pools[a][ua].add((ca.zoom,ca.x,ca.y));pools[b][ub].add((cb.zoom,cb.x,cb.y))
    paths=[];reports=[]
    for s,units in enumerate(pools):
        check();chosen=[];errors=[]
        for pool in units:
            reps,error=representatives(pool);chosen.append(reps);errors.append(error)
        source_paths=[]
        for family in range(4):
            knots={}
            for unit,(first,end) in enumerate(zip(boundaries[s][:-1],boundaries[s][1:])):
                options=chosen[unit]
                camera=(1.,.5,.5) if family==0 or not options else options[min(family-1,len(options)-1)]
                knots[int(first)]=camera;knots[int(end)-1]=camera
            ticks=np.array(sorted(knots),int);values=np.array([knots[t] for t in ticks])
            source_paths.append((ticks,values))
        paths.append(source_paths)
        reports.append(dict(source=s,native_units=len(units),unique_proposals=sum(map(len,units)),
            retained_non_native_representatives=sum(map(len,chosen)),maximum_numeric_covering_error=max(errors,default=0.),
            stable_inside_native_shots=True,actor_ownership_certified=False))
    return paths,dict(total_raw_proposals=count,paths_per_source=4,source_reports=reports,
        proposal_selection_used_music_or_reference_times=False,perfect_camera_coverage_claimed=False,roll_supported=False)
