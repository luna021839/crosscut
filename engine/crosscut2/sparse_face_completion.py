"""Bounded acquisition requests and exact reuse of missing face endpoints.

An unobserved source frame is not a detector-observed empty frame. The first
programme may reveal which missing endpoints matter; request them once across
the available sources before the final selection. No rendered pixels, teacher
timestamps, invented pose, source exclusion or audiovisual approval is used.
Actual decoding/inference is an explicit caller-owned operation, not a cache
loader fallback.
"""
import json
from pathlib import Path
import numpy as np


def requests(seams,nodes,banks,*,maximum_centres=4,maximum_frames=64):
    if (type(maximum_centres) is not int or not 0<=maximum_centres<=8
            or type(maximum_frames) is not int or not 0<=maximum_frames<=128):
        raise ValueError('Explicit finite face-endpoint completion allowance required')
    nodes=np.asarray(nodes,int);candidates={}
    for seam in seams:
        tick=int(seam['output_frame']);n=int(np.searchsorted(nodes,tick))
        if n>=len(nodes) or nodes[n]!=tick:raise ValueError('Selected cut outside the actual decision grid')
        missing=[]
        for side,key in ((0,'left_source'),(1,'right_source')):
            s=seam[key]
            if type(s) is not int or not 0<=s<len(banks):raise ValueError('Physical selected source required')
            known=np.asarray(banks[s].get('face_observation_valid',np.ones((len(nodes),2),bool)))
            if not known[n,side]:missing.append((s,side))
        if missing:
            risk=float(seam.get('played_geometry_rate',1.))
            if not np.isfinite(risk) or risk<0:raise ValueError('Finite actual selected geometry rate required')
            candidates[n]=(-len(missing),-risk,tick)
    selected=[];wanted=set()
    for n in sorted(candidates,key=lambda n:candidates[n]):
        if len(selected)>=maximum_centres:break
        pending=set()
        for s,bank in enumerate(banks):
            known=np.asarray(bank.get('face_observation_valid',np.ones((len(nodes),2),bool)))
            for side in (0,1):
                if not known[n,side]:pending.add((s,int(bank['native_indices'][n,side])))
        if len(wanted|pending)>maximum_frames:continue
        wanted|=pending;selected.append(int(nodes[n]))
    return [dict(source=s,native_frame=k) for s,k in sorted(wanted)],dict(
        selected_centres=selected,missing_selected_cut_centres=len(candidates),
        maximum_centres=maximum_centres,maximum_frames=maximum_frames,
        actual_frame_requests=len(wanted),bounded_single_pass=True,
        selected_source_sequence_is_proposal_not_quality_approval=True,
        rendered_movie_or_reference_pixels_used=False)


def apply(banks,poses,styles,states,rows,*,source_hashes):
    """Merge source-bound observations ONLY into explicitly unobserved slots.

    Both observed faces and observed-empty frames stay byte exact. New faces
    receive no old body's coordinates; body/hand support remains unknown until
    an independently bound subject window can provide it.
    """
    if poses is None:poses=[np.zeros((len(b['heads']),4,3,8,3),np.float32) for b in banks]
    if states is None:states=[np.zeros((len(b['heads']),4,3),np.uint8) for b in banks]
    if not len(banks)==len(poses)==len(styles)==len(states)==len(source_hashes):
        raise ValueError('One exact source domain per completion input required')
    if len(rows)>128:raise ValueError('Bounded face-endpoint batch required')
    seen=set();validated=[]
    for row in rows:
        s,k=row['source'],row['native_frame']
        if (type(s) is not int or not 0<=s<len(banks) or type(k) is not int or k<0
                or row['source_sha256']!=source_hashes[s] or len(row['source_sha256'])!=64
                or len(row['raster_sha256'])!=64
                or any(c not in '0123456789abcdef' for c in row['raster_sha256'])):
            raise ValueError('Exact source hash/native ordinal/raster binding required')
        if (s,k) in seen:raise ValueError('Duplicate endpoint observations must be resolved before merging')
        seen.add((s,k));arrays={}
        for key,shape in (('heads',(8,4)),('vectors',(3,128)),('native_readability',(8,4)),
                ('native_readability_valid',(8,)),('colour',(8,2,40)),('colour_valid',(8,)),('presence',(2,))):
            value=np.asarray(row[key])
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Actual bounded face/style measurements required')
            if key.endswith('_valid') and value.dtype!=bool:raise ValueError('Separate boolean observation validity required')
            arrays[key]=value
        validated.append((s,k,arrays))
    output_banks=[];output_poses=[];output_styles=[];output_states=[]
    for bank,pose,style,state in zip(banks,poses,styles,states):
        copied=dict(bank);N=len(bank['heads'])
        copied['face_observation_valid']=np.asarray(bank.get('face_observation_valid',np.ones((N,2),bool))).copy()
        for key,shape,dtype in (('heads',(N,2,8,4),float),('vectors',(N,2,3,128),float),
                ('native_readability',(N,2,8,4),float),('native_readability_valid',(N,2,8),bool)):
            copied[key]=np.asarray(bank[key]).copy() if key in bank else np.zeros(shape,dtype)
        copied['fine_grid_old_indices']=np.arange(N,dtype=np.int64)
        output_banks.append(copied);output_poses.append(np.array(pose,copy=True))
        output_styles.append({key:np.array(value,copy=True) for key,value in style.items()})
        output_states.append(np.array(state,copy=True))
    filled=0;preserved=0
    for s,k,values in validated:
        bank=output_banks[s]
        for n,side in zip(*np.nonzero(np.asarray(bank['native_indices'])==k)):
            if bank['face_observation_valid'][n,side]:preserved+=1;continue
            if np.any(bank['heads'][n,side]) or np.any(bank['vectors'][n,side]):
                raise ValueError('Unobserved endpoint contains conflicting legacy face coordinates')
            for key in ('heads','vectors','native_readability','native_readability_valid'):
                bank[key][n,side]=values[key]
            for key in ('colour','colour_valid','presence'):output_styles[s][key][n,side]=values[key]
            bank['face_observation_valid'][n,side]=True
            sl=slice(2*side,2*side+2)
            output_poses[s][n,sl]=0.;output_states[s][n,sl]=0
            filled+=1
    return (output_banks,output_poses,output_styles,output_states,dict(
        new_observed_endpoints=filled,already_observed_endpoints_preserved=preserved,
        source_bound_native_records=len(validated),old_input_arrays_rewritten=False,
        face_or_style_measurements_do_not_certify_body_or_identity=True,
        neural_calls_in_merge=0,decoder_calls_in_merge=0))


def load(reference,*,sources,source_hashes,master_start,fps,authorized_root):
    """Explicit source/native records only; never start a detector on a miss."""
    from .grid_perception import file_identity
    root=Path(authorized_root).resolve(strict=True);reference=Path(reference)
    if reference.is_symlink() or not reference.resolve(strict=True).is_relative_to(root):
        raise ValueError('Face endpoint reference outside the authorized project')
    record=json.loads(reference.read_bytes())
    identities=[list(file_identity(Path(p))) for p in sources]
    if (record['sources']!=list(map(str,sources)) or record['source_sha256']!=list(source_hashes)
            or record['source_file_identities']!=identities or record['fps']!=fps
            or record['master_audio_start']!=master_start or len(record['records'])>128):
        raise ValueError('Face endpoint source/native music domain changed')
    rows=[]
    for item in record['records']:
        s,k=item['source'],item['native_frame'];path=Path(item['path'])
        if (type(s) is not int or not 0<=s<len(sources) or type(k) is not int or k<0
                or path.is_symlink() or not path.resolve(strict=True).is_relative_to(root)):
            raise ValueError('Face endpoint cache address changed')
        binding=json.dumps(dict(source_sha256=source_hashes[s],source_identity=identities[s],
            native_frame=k,producer=record['producer']),sort_keys=True)
        with np.load(path,allow_pickle=False) as saved:
            if str(saved['binding'])!=binding or int(saved['source'])!=s or int(saved['native_frame'])!=k:
                raise ValueError('Face endpoint observation producer/ordinal changed')
            rows.append(dict(source=s,native_frame=k,source_sha256=str(saved['source_sha256']),
                raster_sha256=str(saved['raster_sha256']),**{key:saved[key].copy() for key in
                    ('heads','vectors','native_readability','native_readability_valid','colour','colour_valid','presence')}))
    return rows,dict(records_loaded=len(rows),prior_probe_calls=record['used'],
        prior_probe_seconds=record['elapsed_seconds'],new_NN_or_decode_in_loader=False)
