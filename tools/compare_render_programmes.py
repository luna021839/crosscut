"""Compare actual output-frame source/native/camera programmes, not quality.

An added or reordered source is joined by its original path and content hash.
This reads saved numeric plans only; it neither decodes a film nor reruns NN.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from crosscut.render import segment_frame_counts
from crosscut.render_frame_schedule import sampler_frame_schedule
from crosscut.render_plan_serialization import render_plan_from_json
from render_frozen_subject_programme import camera_values,materialize_scored_cameras


def source_bindings(record):
    paths=record['render_plan']['sources'];hashes=record['source_sha256']
    if len(paths)!=len(hashes) or len(set(paths))!=len(paths):
        raise ValueError('unique original sources with one content hash each required')
    return dict(zip(paths,hashes,strict=True))


def verify_common_sources(old,new):
    a,b=source_bindings(old),source_bindings(new)
    if any(a[p]!=b[p] for p in a.keys()&b.keys()):
        raise ValueError('a common original source changed content')
    return {p:i for i,p in enumerate(sorted(a.keys()|b.keys()))}


def materialize(record,spec,ids):
    plan=render_plan_from_json(record['render_plan']);plan,_=materialize_scored_cameras(plan,record)
    lookup={row['path']:row for row in spec['sources']}
    if len(lookup)!=len(spec['sources']):raise ValueError('ambiguous physical source specification')
    ss=[];nn=[];cc=[]
    for segment,count in zip(plan.segments,segment_frame_counts(plan),strict=True):
        path=str(plan.sources[segment.source_index]);source=lookup[path]
        ss.extend([ids[path]]*count)
        nn.extend(sampler_frame_schedule(source_start=segment.source_start,
            source_playback_rate=segment.source_playback_rate,native_fps=source['fps'],output_fps=plan.fps,
            frame_count=count,source_frame_count=source['frame_count']).source_frame_indices)
        cc.extend(camera_values(segment,count,plan.fps))
    return plan,(np.asarray(ss),np.asarray(nn),np.asarray(cc))


def compare(old,new,spec):
    ids=verify_common_sources(old,new)
    a_plan,a=materialize(old,spec,ids);b_plan,b=materialize(new,spec,ids)
    if len(a[0])!=len(b[0]) or a_plan.fps!=b_plan.fps:raise ValueError('different output clock or duration')
    physical=(a[0]!=b[0])|(a[1]!=b[1]);camera=(np.abs(a[2]-b[2])>1e-9).any(-1)
    changed=physical|camera;edges=np.diff(np.r_[False,changed,False].astype(int));fps=b_plan.fps
    return dict(old_authored_cuts=len(old['seams']),new_authored_cuts=len(new['seams']),output_frames=len(changed),
        changed_source_or_native_frames=int(physical.sum()),changed_source_clock_or_camera_frames=int(changed.sum()),
        changed_seconds=float(changed.sum()/fps),changed_fraction=float(changed.mean()),
        changed_intervals_seconds=[[float(first/fps),float(final/fps)] for first,final in
            zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1),strict=True)],
        source_seconds=[dict(source=p,seconds=float((b[0]==ids[p]).sum()/fps)) for p in new['render_plan']['sources']],
        physical_sources_joined_by_path_and_content=True,source_NN=0,source_rasters=0,
        professional_quality_approved=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-plan',type=Path,required=True);parser.add_argument('--new-plan',type=Path,required=True)
    parser.add_argument('--source-spec',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('recover the existing comparison instead of overwriting it')
    result=compare(*(json.loads(p.read_bytes()) for p in (args.old_plan,args.new_plan,args.source_spec)))
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
