"""Small source-first COCO17 observations, independent of a selected actor.

The producer records whole original images. Selection only associates these
already observed people to faces; a different chosen face never needs another
model call. No RTMO point is re-labelled as RTMW133 or pixel-certified anatomy.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

FAMILY='original_full_image_RTMO_COCO17_v1'


def save_chunk(path,rows,*,binding):
    """Exclusive finite checkpoint; empty detections remain real observations."""
    path=Path(path);natives=[r['native'] for r in rows]
    if not rows or len(rows)>64 or natives!=sorted(set(natives)):
        raise ValueError('one to64 sorted unique original frames required')
    offsets=np.r_[0,np.cumsum([len(r['scores']) for r in rows])]
    with path.open('xb') as f:
        np.savez_compressed(f,binding=json.dumps(binding,sort_keys=True),
            native=np.asarray(natives,np.int64),offsets=offsets,
            boxes=np.concatenate([r['boxes'] for r in rows]),
            scores=np.concatenate([r['scores'] for r in rows]),
            keypoints17=np.concatenate([r['keypoints17'] for r in rows]),
            raster_sha256=np.asarray([r['raster_sha256'] for r in rows],dtype='U64'))


def load_chunk(path,*,binding):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>8*1024**2:
        raise ValueError('bounded real original-frame checkpoint required')
    with np.load(path,allow_pickle=False) as z:
        if json.loads(str(z['binding']))!=binding:raise ValueError('original-frame producer/source binding changed')
        native,offsets,boxes,scores,points,rasters=(z[k] for k in
            ('native','offsets','boxes','scores','keypoints17','raster_sha256'))
    if (native.ndim!=1 or not 1<=len(native)<=64 or native.dtype.kind not in 'iu'
            or np.any(native<0) or np.any(np.diff(native)<=0)
            or offsets.shape!=(len(native)+1,) or offsets.dtype.kind not in 'iu'
            or offsets[0]!=0 or offsets[-1]!=len(scores) or np.any(np.diff(offsets)<0)
            or np.any(np.diff(offsets)>32) or boxes.shape!=(len(scores),4)
            or scores.ndim!=1 or points.shape!=(len(scores),17,3)
            or rasters.shape!=native.shape or rasters.dtype.kind!='U'
            or not all(np.isfinite(a).all() for a in (boxes,scores,points))
            or np.any((scores<0)|(scores>1)) or np.any((points[...,2]<0)|(points[...,2]>1))
            or any(len(str(v))!=64 or any(c not in '0123456789abcdef' for c in str(v)) for v in rasters)):
        raise ValueError('original COCO17 checkpoint tensor/clock changed')
    return {int(n):dict(boxes=boxes[a:b],scores=scores[a:b],keypoints17=points[a:b],
        raster_sha256=str(r)) for n,a,b,r in zip(native,offsets[:-1],offsets[1:],rasters)}


def load_reference(path,*,sources,source_hashes,banks,authorized_root,check=lambda:None):
    """Read only these named checkpoints; never load a model or scan caches."""
    from .grid_perception import file_identity
    root=Path(authorized_root).resolve(strict=True);path=Path(path)
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError('original people reference outside the project')
    record=json.loads(path.read_bytes())
    if (record['family']!=FAMILY or record['sources']!=list(map(str,sources))
            or record['source_sha256']!=list(source_hashes)
            or len(record['source_records'])!=len(sources)):
        raise ValueError('original people physical source domain changed')
    result=[]
    for s,(source,bank,entry) in enumerate(zip(sources,banks,record['source_records'],strict=True)):
        check()
        from .multi_person_pose import MODEL_SHA256
        binding=entry['binding'];raster=binding.get('raster')
        if (binding.get('source_sha256')!=source_hashes[s] or binding.get('family')!=FAMILY
                or binding.get('model_sha256')!=MODEL_SHA256 or not isinstance(raster,list)
                or len(raster)!=2 or any(type(v) is not int or not 2<=v<=8192 for v in raster)):
            raise ValueError('original people checkpoint source/model/raster family changed')
        if list(file_identity(source))!=entry['source_identity']:
            # USB remounts/copies can change file identity without changing
            # a single model input. Verify CONTENT instead of re-running NN.
            digest=hashlib.sha256()
            with Path(source).open('rb') as stream:
                while block:=stream.read(4*1024**2):check();digest.update(block)
            if digest.hexdigest()!=source_hashes[s]:
                raise ValueError('original people source bytes/identity changed')
        rows={}
        for chunk in entry['chunks']:
            check();p=Path(chunk['path'])
            if p.is_symlink() or not p.resolve(strict=True).is_relative_to(root):
                raise ValueError('original people checkpoint outside project')
            if p.stat().st_size>8*1024**2:raise ValueError('original people checkpoint exceeds bounded size')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=chunk['sha256']:
                raise ValueError('original people checkpoint content changed')
            new=load_chunk(p,binding=entry['binding'])
            if rows.keys()&new.keys():raise ValueError('duplicate original native observations')
            rows.update(new)
        wanted=set(map(int,np.asarray(bank['pose_native_indices']).ravel()))
        if not wanted<=rows.keys():raise ValueError('actual source-clock pose schedule was not observed')
        result.append(dict(by_native=rows,source=s,source_sha256=source_hashes[s],
            raster=entry['binding']['raster'],family=FAMILY,reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return result
