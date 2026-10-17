"""Exact current-face ArcFace observations, separate from six-member scores.

Only the original source/native/face rectangle can retrieve a query. Runtime
provenance stays with each snapshot; raw vectors are local numeric inputs, not
named identities, transferable labels, or material for review reports.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from crosscut.native_face_descriptor import ARCFACE_R50_SHA256

MAXIMUM_QUERIES=4096
MAXIMUM_SNAPSHOT_BYTES=16*1024**2


def query_key(native,head):
    head=np.asarray(head)
    if (type(native) is not int or native<0 or head.shape!=(4,) or not np.isfinite(head).all()):
        raise ValueError('actual source native ordinal and finite original face required')
    return hashlib.sha256(np.array([native,*head],np.float64).tobytes()).hexdigest()


def descriptor_family(binding):
    if (binding.get('model')!=ARCFACE_R50_SHA256 or binding.get('raster')!=[960,540]
            or any(not isinstance(binding.get(k),str) or len(binding[k])!=64
                   or any(c not in '0123456789abcdef' for c in binding[k])
                   for k in ('source_sha256','adapter','detector','association','decoder_implementation'))):
        raise ValueError('pinned original-source ArcFace observation binding required')
    return hashlib.sha256(json.dumps({k:binding[k] for k in
        ('model','adapter','detector','association','raster','decoder_implementation')},sort_keys=True).encode()).hexdigest()


def validate_values(keys,vectors,valid):
    keys=np.asarray(keys);vectors=np.asarray(vectors);valid=np.asarray(valid)
    if (keys.ndim!=1 or keys.dtype.kind!='U' or len(keys)>MAXIMUM_QUERIES
            or len(set(map(str,keys)))!=len(keys) or any(len(k)!=64 for k in keys)
            or vectors.shape!=(len(keys),512) or vectors.dtype.kind!='f' or not np.isfinite(vectors).all()
            or valid.shape!=(len(keys),) or valid.dtype!=bool
            or np.any(np.abs(np.linalg.norm(vectors[valid],axis=-1)-1.)>.001)
            or np.any(vectors[~valid]!=0.)):
        raise ValueError('bounded actual512D unit queries or explicit failed observations required')
    result={}
    for key,vector,ok in zip(map(str,keys),vectors,valid):
        vector=np.asarray(vector,np.float32).copy();vector.setflags(write=False)
        result[key]=(vector,bool(ok))
    return result


def read_snapshot(path,*,binding):
    path=Path(path);descriptor_family(binding)
    if path.is_symlink() or path.stat().st_size>MAXIMUM_SNAPSHOT_BYTES:
        raise ValueError('bounded local query snapshot required')
    with np.load(path,allow_pickle=False) as saved:
        if str(saved['binding'])!=json.dumps(binding,sort_keys=True):
            raise ValueError('actual query snapshot producer differs')
        return validate_values(saved['keys'],saved['vectors'],saved['valid'])


def publish_snapshot(path,values,*,binding,check=lambda:None):
    """Checkpoint only queries produced by THIS runtime, never relabel parents."""
    path=Path(path);descriptor_family(binding);check()
    keys=np.array(sorted(values),dtype='U64')
    vectors=np.array([values[k][0] for k in keys],np.float32).reshape(-1,512)
    valid=np.array([values[k][1] for k in keys],bool)
    validate_values(keys,vectors,valid)
    descriptor,temporary=tempfile.mkstemp(prefix='.direct-face-',suffix='.npz',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            np.savez_compressed(stream,binding=json.dumps(binding,sort_keys=True),keys=keys,vectors=vectors,valid=valid)
            stream.flush();os.fsync(stream.fileno())
        if Path(temporary).stat().st_size>MAXIMUM_SNAPSHOT_BYTES:
            raise ValueError('direct face snapshot exceeded its explicit16MiB budget')
        check();os.replace(temporary,path)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()


def fields(bank,values):
    """No nearest-face/other-frame transfer, fabricated query, or source veto."""
    heads=np.asarray(bank['heads']);natives=np.asarray(bank['native_indices'])
    if (heads.ndim!=4 or heads.shape[-1]!=4 or natives.shape!=heads.shape[:2]
            or natives.dtype.kind not in 'iu' or np.any(natives<0) or not np.isfinite(heads).all()):
        raise ValueError('original source-bound endpoint face bank required')
    vectors=np.zeros(heads.shape[:-1]+(512,),np.float32);known=np.zeros(heads.shape[:-1],bool)
    for address in np.ndindex(natives.shape):
        for h,head in enumerate(heads[address]):
            item=values.get(query_key(int(natives[address]),head))
            if item is None or not item[1]:continue
            vector=np.asarray(item[0])
            if vector.shape!=(512,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector)-1.)>.001:
                raise ValueError('current face observation is not a512D unit query')
            vectors[address+(h,)]=vector;known[address+(h,)]=True
    vectors.setflags(write=False);known.setflags(write=False)
    return vectors,known


def recover_source(binding,*,legacy_cache,cache):
    """Merge source-bound parents without changing their runtime provenance."""
    from .recurrent_member_hints import recover_query_snapshots
    family=descriptor_family(binding)
    values,old=recover_query_snapshots(legacy_cache,binding)
    conflicts=set();parents=[];total_bytes=0
    files=sorted(p for p in Path(cache).glob('*.npz') if not p.name.startswith('.') and not p.is_symlink())
    if len(files)>128:raise ValueError('direct query snapshot inventory exceeded128files')
    for path in files:
        if path.stat().st_size>MAXIMUM_SNAPSHOT_BYTES:raise ValueError('direct query snapshot too large')
        with np.load(path,allow_pickle=False) as saved:producer=json.loads(str(saved['binding']))
        if (producer.get('contract')!='actual_native_face_pair_queries_v2'
                or any(producer.get(k)!=v for k,v in binding.items() if k not in ('ort','opencv'))):continue
        if descriptor_family(producer)!=family:raise ValueError('direct descriptor family changed')
        total_bytes+=path.stat().st_size
        if total_bytes>64*1024**2 or len(parents)>=32:raise ValueError('bounded direct query parents exceeded')
        found=read_snapshot(path,binding=producer)
        for key,(vector,ok) in found.items():
            if key in values and (not np.array_equal(vector,values[key][0]) or ok!=values[key][1]):
                conflicts.add(key)
            else:values[key]=(vector,ok)
        parents.append(dict(file=str(path),producer_ort=producer.get('ort'),producer_opencv=producer.get('opencv')))
    for key in conflicts:values.pop(key,None)
    if len(values)>MAXIMUM_QUERIES:raise ValueError('direct source query domain exceeds4096')
    return values,dict(legacy_parents=old,direct_parents=parents,ambiguous_query_inputs=len(conflicts),
        recovered_queries=len(values),valid_recovered_queries=sum(v[1] for v in values.values()),
        query_recovery_NN=0,source_rasters=0,historical_producers_not_relabelled=True)


def bind_banks(banks,bindings,*,source_hashes,legacy_cache,cache):
    """Attach optional, exact512D fields; no change to original face/pose data."""
    if not len(banks)==len(bindings)==len(source_hashes):raise ValueError('one actual source binding per bank required')
    reports=[]
    # Check all source domains before mutating any bank.
    for binding,digest in zip(bindings,source_hashes):
        descriptor_family(binding)
        if binding['source_sha256']!=digest:raise ValueError('direct query belongs to another physical source')
    for s,(bank,binding) in enumerate(zip(banks,bindings)):
        values,receipt=recover_source(binding,legacy_cache=legacy_cache,cache=cache)
        vectors,known=fields(bank,values)
        bank['direct_face_vectors']=vectors;bank['direct_face_known']=known
        bank['direct_face_descriptor_family']=descriptor_family(binding)
        reports.append(dict(source=s,matched_current_face_slots=int(known.sum()),
            total_current_face_slots=known.size,missing_face_excludes_no_source=True,**receipt))
    return dict(sources=reports,query_recovery_NN=0,source_rasters=0,raw_vectors_exported=False,
        all_original_face_and_body_observations_retained=True,
        source_named_identity_certified=False,nearest_or_other_frame_query_transfer=False)


def bind_manifest(path,*,banks,nodes,sources,source_hashes,master_start,fps,authorized_root,legacy_cache,cache):
    """Cache-only CLI boundary: verify the physical programme before reuse."""
    from .grid_perception import file_identity
    path=Path(path);root=Path(authorized_root).resolve(strict=True)
    if (path.is_symlink() or not path.resolve(strict=True).is_relative_to(root)
            or not 0<path.stat().st_size<=128*1024):
        raise ValueError('bounded source-grid face manifest in authorized project required')
    raw=path.read_bytes();record=json.loads(raw);nodes=np.asarray(nodes)
    if (record.get('contract')!='all_preselection_actor_proposal_faces_on_fixed_source_grid_v1'
            or record.get('source_paths')!=list(map(str,sources))
            or record.get('source_file_identities')!=[list(file_identity(p)) for p in sources]
            or record.get('source_sha256')!=list(source_hashes)
            or record.get('grid_nodes_sha256')!=hashlib.sha256(nodes.tobytes()).hexdigest()
            or record.get('output_fps')!=fps or record.get('master_start')!=master_start
            or record.get('original_node_count')!=len(nodes)):
        raise ValueError('direct face manifest physical source, clock or candidate grid differs')
    bindings=record.get('bindings',[])
    receipt=bind_banks(banks,bindings,source_hashes=source_hashes,legacy_cache=legacy_cache,cache=cache)
    receipt['request_manifest_sha256']=hashlib.sha256(raw).hexdigest()
    return bindings,receipt


def refresh_member_scores(banks,bindings,hints,*,gallery,check=lambda:None):
    """Classify cached CURRENT faces, not their nearest SFace neighbours.

    These six scores help choose the featured subject before camera/DP. They
    are separate from direct512D pair comparison, which must not count the
    same descriptor twice. Unknown/failed queries keep the existing fallback;
    neither a label nor source admission is forced by this operation.
    """
    from crosscut.ive import IVEMember
    from crosscut.native_face_descriptor import ARCFACE_MODEL_NAME,NativeFaceDescriptorSpace
    from .fixed_holds import local_actor_hypotheses
    from .source_member_scores import MemberScoreSeeds,from_queries
    if not len(banks)==len(bindings) or (hints is not None and len(hints)!=len(banks)):
        raise ValueError('one source-local hint domain per actual face bank required')
    result=[];receipts=[]
    names=tuple(member.value for member in IVEMember)
    for s,(bank,binding) in enumerate(zip(banks,bindings,strict=True)):
        check();heads=np.asarray(bank['heads']);native=np.asarray(bank['native_indices'])
        vectors=np.asarray(bank['direct_face_vectors']);known=np.asarray(bank['direct_face_known'])
        if (bank.get('direct_face_descriptor_family')!=descriptor_family(binding)
                or heads.ndim!=4 or heads.shape[-1]!=4 or native.shape!=heads.shape[:2]
                or native.dtype.kind not in 'iu' or np.any(native<0)
                or vectors.shape!=heads.shape[:-1]+(512,) or known.shape!=heads.shape[:-1]
                or known.dtype!=bool or not np.isfinite(vectors).all()
                or np.any(np.abs(np.linalg.norm(vectors[known],axis=-1)-1.)>.001)):
            raise ValueError('bound actual current face queries required before member scoring')
        original=None if hints is None else hints[s]
        if isinstance(original,MemberScoreSeeds):
            original.verify(bank);clusters=original.cluster_hints
        elif original is not None:clusters=np.asarray(original)
        else:
            _,count=local_actor_hypotheses(bank);clusters=np.full(count,-1,np.int16)
        space=NativeFaceDescriptorSpace(ARCFACE_MODEL_NAME,512,ARCFACE_R50_SHA256,binding['adapter'])
        physical={}
        for n,side,h in zip(*np.nonzero(known)):
            head=heads[n,side,h];ordinal=int(native[n,side]);key=query_key(ordinal,head)
            value=(ordinal,head,vectors[n,side,h])
            if key in physical and not np.array_equal(physical[key][2],value[2]):
                raise ValueError('one actual face has contradictory direct observations')
            physical[key]=value
        if len(physical)>MAXIMUM_QUERIES:raise ValueError('bounded current-face score domain exceeded')
        requests=[];votes=[]
        for i,key in enumerate(sorted(physical)):
            if i%64==0:check()
            ordinal,head,vector=physical[key]
            requests.append((ordinal,head))
            votes.append(gallery.classify(tuple(map(float,vector)),descriptor_space=space))
        seeds=from_queries(bank,requests,votes,clusters,gallery_identity=gallery.identity,
            member_names=names,minimum_similarity=gallery.minimum_similarity,
            minimum_member_margin=gallery.minimum_member_margin)
        result.append(seeds)
        receipts.append(dict(source=s,unique_actual_face_scores=len(seeds.direct_queries),
            old_cluster_hints_preserved=True,named_member_identity_certified=False))
    return result,dict(sources=receipts,source_NN=0,source_rasters=0,
        actual_current_queries_used=True,reference_photos_or_vectors_exported=False,
        source_exclusion=False,unknown_not_forced_to_a_member=True)
