"""Optional recurrent IVE hints using existing local ArcFace/photo assets.

At most three preselected native faces per anonymous source-local actor.
Raw query vectors stay in a small local NPZ, never in reports/review packets;
reference photos/banks are not copied. No threshold lowering or forced label.
"""
import hashlib,inspect,json,os,tempfile
from importlib.metadata import version
from pathlib import Path
from time import monotonic
import cv2,numpy as np
from crosscut.ive import IVEMember
from crosscut.ive_local_arcface_gallery import LocalIVEArcFaceGallery
from crosscut.native_face_descriptor import NativeArcFaceR50,NativeFaceDescriptorSpace,ARCFACE_MODEL_NAME,ARCFACE_R50_SHA256
from crosscut import native_face_descriptor
from crosscut.face_identity import _validate_normalized_face
from .fixed_holds import local_actor_hypotheses
from .grid_perception import file_identity
from .supplementary_pose_bank import _frames,_DECODER_IMPLEMENTATION


def select_faces(bank,labels,count,*,native_fps):
    """Choose by original size/readability, never member or cross-source fit."""
    requests=[];groups=[]
    for actor in range(count):
        candidates=[]
        for n,side,h in zip(*np.nonzero(labels==actor)):
            head=bank['heads'][n,side,h]
            if head[2]<.055:continue
            quality=1.
            if 'native_readability' in bank and bank['native_readability_valid'][n,side,h]:
                contrast,edge,clipped,_=bank['native_readability'][n,side,h]
                quality=min(1.,edge/.5)*min(1.,contrast/.075)*(1.-clipped)
            candidates.append((-float(head[2])*quality,int(bank['native_indices'][n,side]),n,side,h))
        used=[];indices=[]
        for _,native,n,side,h in sorted(candidates):
            if any(abs(native-v)<max(6,round(native_fps*.25)) for v in used):continue
            used.append(native);indices.append(len(requests));requests.append((native,bank['heads'][n,side,h].copy()))
            if len(used)==3:break
        groups.append(indices)
    return requests,groups


def associate(found,head):
    """Match the actual cached face rectangle, not nearest other performer."""
    if found is None:return None
    matches=[]
    for face in found:
        measured=np.array([(face[0]+face[2]/2)/960,(face[1]+face[3]/2)/540,face[3]/540])
        if (measured[2]>.019 and np.linalg.norm((measured[:2]-head[:2])*[16/9,1.])/max(.02,head[2])<=.2
                and abs(np.log(measured[2]/head[2]))<=.25):matches.append(face)
    return matches[0] if len(matches)==1 else None


def consensus(hints):
    known=[h['member'] for h in hints if h['state']=='LIKELY_MEMBER']
    if len(known)<2 or len(set(known))!=1:return -1
    return tuple(m.value for m in IVEMember).index(known[0])


def valid_query_geometry(box,points):
    """Off-raster/degenerate face is UNKNOWN, not a whole-source failure."""
    try:
        _validate_normalized_face(box,points)
        native_face_descriptor.face_similarity_affine(np.asarray(points)*[960,540])
    except ValueError:return False
    return True


def measure(*,sources,banks,census,photo_tool,models,cache,check,maximum_seconds=240.,maximum_calls=288):
    if not len(sources)==len(banks)==len(census) or not 1<=maximum_seconds<=600 or not 0<=maximum_calls<=1024:
        raise ValueError('bounded source-local member-hint observation allowance required')
    started=monotonic();photo_tool=Path(photo_tool);models=Path(models);cache=Path(cache)
    gallery=LocalIVEArcFaceGallery.from_existing_photo_banks(photo_tool/'face_models/user_member_arcface_embeddings.json',
        photo_tool/'face_models/yujin_arcface_embeddings.json')
    model_path=photo_tool/'insightface_models/buffalo_l/w600k_r50.onnx'
    adapter=hashlib.sha256(Path(native_face_descriptor.__file__).read_bytes()).hexdigest()
    space=NativeFaceDescriptorSpace(ARCFACE_MODEL_NAME,512,ARCFACE_R50_SHA256,adapter)
    detector_path=models/'face_detection_yunet_2023mar.onnx'
    detector_sha=hashlib.sha256(detector_path.read_bytes()).hexdigest()
    arc=detector=None;calls=detector_calls=0;result=[];receipts=[]
    def live():
        check()
        if monotonic()-started>=maximum_seconds:raise TimeoutError('one recurrent-hint pass exhausted; completed query snapshots retained')
    for s,(path,bank,clock) in enumerate(zip(sources,banks,census)):
        live();identity=file_identity(path);digest=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):live();digest.update(chunk)
        source_sha=digest.hexdigest();labels,count=local_actor_hypotheses(bank)
        requests,groups=select_faces(bank,labels,count,native_fps=clock['fps'])
        binding=json.dumps(dict(source_sha256=source_sha,model=ARCFACE_R50_SHA256,adapter=adapter,detector=detector_sha,
            raster=[960,540],decoder_implementation=_DECODER_IMPLEMENTATION,
            association=hashlib.sha256(inspect.getsource(associate).encode()).hexdigest(),
            ort=version('onnxruntime'),opencv=cv2.__version__,native_fps=clock['fps'],native_count=clock['frame_count']),sort_keys=True)
        cache.mkdir(parents=True,exist_ok=True);target=cache/(hashlib.sha256(binding.encode()).hexdigest()+'.npz')
        values={};dirty=False
        if target.is_file():
            with np.load(target,allow_pickle=False) as saved:
                if str(saved['binding'])!=binding:raise ValueError('local member query cache binding differs')
                keys,vectors,valid=(saved[k] for k in ('keys','vectors','valid'))
            if (keys.ndim!=1 or keys.dtype.kind!='U' or len(set(keys))!=len(keys) or vectors.shape!=(len(keys),512)
                    or valid.shape!=(len(keys),) or valid.dtype!=bool or not np.isfinite(vectors).all() or len(keys)>1024):
                raise ValueError('local member query snapshot domain differs')
            values={str(k):(v.copy(),bool(ok)) for k,v,ok in zip(keys,vectors,valid)}
        keys=[];needed={};source_calls=source_detectors=0
        for native,head in requests:
            key=hashlib.sha256(np.array([native,*head],np.float64).tobytes()).hexdigest();keys.append(key)
            if key not in values:needed.setdefault(native,{})[key]=head
        if sum(map(len,needed.values()))>maximum_calls-calls:
            raise ValueError('recurrent member query allowance exceeded before neural inference')
        def checkpoint():
            nonlocal dirty
            if not dirty:return
            # Flush completed query observations even after the inference
            # deadline, without authorizing another model call or source read.
            check()
            if file_identity(path)!=identity:raise OSError('source changed during member query observations')
            descriptor,temporary=tempfile.mkstemp(prefix='.member-query-',suffix='.npz',dir=cache)
            try:
                ordered=sorted(values)
                with os.fdopen(descriptor,'wb') as stream:
                    np.savez_compressed(stream,binding=binding,keys=np.array(ordered,dtype='U64'),
                        vectors=np.asarray([values[k][0] for k in ordered],np.float32),valid=np.asarray([values[k][1] for k in ordered],bool))
                    stream.flush();os.fsync(stream.fileno())
                check();os.replace(temporary,target);dirty=False
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
        try:
            if needed:
                if arc is None:arc=NativeArcFaceR50(model_path)
                if detector is None:detector=cv2.FaceDetectorYN.create(str(detector_path),'',(960,540),.75,.3,5000)
                for native,pixels in _frames(path,sorted(needed),check=live):
                    live();_,found=detector.detect(pixels);detector_calls+=1;source_detectors+=1
                    for key,head in needed[native].items():
                        face=associate(found,head);vector=np.zeros(512,np.float32);ok=face is not None
                        if ok:
                            box=tuple(float(v) for v in (face[0]/960,face[1]/540,face[2]/960,face[3]/540))
                            points=tuple((float(face[k]/960),float(face[k+1]/540)) for k in range(4,14,2))
                            ok=valid_query_geometry(box,points)
                            if ok:
                                vector=np.array(arc.embedding(pixels,bbox_xywh=box,landmarks_xy=points),np.float32)
                                calls+=1;source_calls+=1
                        values[key]=(vector,ok);dirty=True
                        if (source_calls+source_detectors)%8==0:checkpoint()
            checkpoint()
        finally:
            if dirty and file_identity(path)==identity:checkpoint()
        hints=[]
        for indices in groups:
            votes=[]
            for index in indices:
                vector,valid=values[keys[index]]
                votes.append(gallery.classify(tuple(map(float,vector)) if valid else None,descriptor_space=space))
            hints.append(consensus(votes))
        if file_identity(path)!=identity:raise OSError('source changed before member-hint result')
        result.append(np.asarray(hints,np.int16))
        receipt=dict(source=s,anonymous_source_clusters=count,preselected_faces=len(requests),
            labelled_by_recurrent_consensus=sum(h>=0 for h in hints),new_arcface_calls=source_calls,
            new_supplementary_detector_calls=source_detectors,reused_query_inputs=len(keys)-sum(map(len,needed.values())),
            raw_query_cache_bytes=0 if not target.exists() else target.stat().st_size,
            base_face_body_cache_invalidated=False,private_references_copied_or_exported=False,
            stage_video_hint_accuracy_calibrated=False,named_identity_certified=False,elapsed_seconds=monotonic()-started)
        receipts.append(receipt);print(json.dumps(dict(phase='source_recurrent_member_hints',**receipt)),flush=True)
    return result,receipts
