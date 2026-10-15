"""One explicit bounded experimental input packet; NEVER a whole-film render.

Coarse retrieval visits every source/head, not a source-native focal. It is
only a search proposal: one-frame action then multi-frame subject matching
must follow. Failed experiments remain failures, not successful safe holds.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import monotonic
from types import SimpleNamespace
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
from crosscut2 import sparse_native_frames as sparse
from crosscut2.grid_perception import file_identity
from crosscut2.rtmw_pose import WholeBodySensor,MODEL_SHA256
from crosscut2.partial_pose import face_proposals,face_guided_hypothesis
from crosscut2.temporal_face_chain import _bounds,_iou
from crosscut2 import subject_window_bank as bank
from crosscut2 import subject_action_match as matcher


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--ssd-device',type=int,required=True)
    args=parser.parse_args();spec=json.loads(args.input.read_text());started=monotonic()
    root=Path('/Volumes/CrossCutMedia/CrossCut-Validation').resolve(strict=True)
    if args.output.exists():raise FileExistsError('recover existing one-shot experiment; do not repeat it')
    if any(not p.resolve().is_relative_to(root) for p in (args.cache,args.output)):
        raise ValueError('large observations/output must remain in the authorized SSD project')
    sources=[Path(v['path']).resolve(strict=True) for v in spec['sources']]
    if any(not p.is_relative_to(root) for p in sources):raise ValueError('source outside explicit project')
    identities=[file_identity(p) for p in sources]
    def check():
        if not Path('/Volumes/CrossCutMedia').is_mount() or root.stat().st_dev!=args.ssd_device:
            raise OSError('authorized SSD disconnected or remounted')
        if any(file_identity(p)!=identity for p,identity in zip(sources,identities)):
            raise OSError('a source changed during the single observation batch')
    check();args.cache.mkdir(parents=True,exist_ok=True);args.output.parent.mkdir(parents=True,exist_ok=True)
    # 768 rather than the review's suggested512 detector calls permits three
    # dense incoming hypotheses per five contexts. Pose256/raster4096/time240
    # remain shared hard limits. This is not a default app resource policy.
    budget=bank.ProbeBudget(face_limit=spec.get('face_limit',768),check=check)
    sensor=WholeBodySensor(spec['pose_model']);cv2.setNumThreads(2)
    detector=cv2.FaceDetectorYN.create(spec['detector_model'],'',(960,540),.75,.3,5000)
    recognizer=cv2.FaceRecognizerSF.create(spec['recognizer_model'],'')
    hash_start=monotonic();source_hashes=[];hash_bytes=0
    for path in sources:
        check()
        with path.open('rb') as stream:source_hashes.append(hashlib.file_digest(stream,'sha256').hexdigest())
        hash_bytes+=path.stat().st_size
    model_hashes={}
    for key in ('pose_model','detector_model','recognizer_model'):
        path=Path(spec[key]);hash_bytes+=path.stat().st_size
        with path.open('rb') as stream:model_hashes[key]=hashlib.file_digest(stream,'sha256').hexdigest()
    if model_hashes['pose_model']!=MODEL_SHA256:raise ValueError('unexpected RTMW asset')
    hash_seconds=monotonic()-hash_start
    raw_binding=dict(source_hashes=source_hashes,models=model_hashes,opencv=cv2.__version__,numpy=np.__version__,
        sensor=hashlib.sha256(Path(inspect.getfile(WholeBodySensor)).read_bytes()).hexdigest(),raster=[960,540],
        runtime=__import__('onnxruntime').__version__,reader=hashlib.sha256(Path(sparse.__file__).read_bytes()).hexdigest())
    clocks={};decode_receipts=[];incomplete=[];raw={};returned_rasters=0
    pose_file=args.cache/(digest(raw_binding)+'-raw133.npz')
    if pose_file.exists():
        with np.load(pose_file,allow_pickle=False) as z:raw.update(zip(map(str,z['keys']),z['raw133']))
    raw_before=set(raw);dirty=False
    def checkpoint():
        nonlocal dirty
        if not dirty:return
        check();fd,temp=tempfile.mkstemp(prefix='.subject-raw-',suffix='.npz',dir=args.cache)
        try:
            with os.fdopen(fd,'wb') as stream:
                keys=sorted(raw);np.savez_compressed(stream,keys=np.asarray(keys,dtype='U64'),raw133=np.asarray([raw[k] for k in keys]))
                stream.flush();os.fsync(stream.fileno())
            check();os.replace(temp,pose_file);dirty=False
        finally:
            if os.path.exists(temp):os.unlink(temp)
    def rasters(s,requested):
        nonlocal returned_rasters
        budget.live();wanted=sorted(set(map(int,requested)))
        if s not in clocks:clocks[s]=sparse.probe(sources[s],check=budget.live)
        # Reserve the entire request BEFORE decoding, not after receiving an
        # over-budget frame. Include conservative decoder/coarse-buffer room
        # before retaining any raster arrays.
        budget.reserve('raster',len(wanted))
        budget.arrays((2*len(wanted)+8)*960*540*3)
        receipt={};result={}
        for native,pixels in sparse.frames(sources[s],wanted,clock=clocks[s],check=budget.live,receipt=receipt,
                                           maximum_seconds=min(30.,max(1.,budget.seconds-(monotonic()-budget.started)))):
            returned_rasters+=1;result[native]=pixels
        decode_receipts.append(dict(source=s,**receipt));return result
    def pose(s,native,head,pixel_sha,get_pixels):
        nonlocal dirty
        box=face_proposals(head)
        if not box:return np.zeros((133,3),np.float32)
        key=digest(dict(binding=raw_binding,source=s,native=int(native),raster_sha256=pixel_sha,roi=box[0]))
        budget.reserve('pose')
        if key not in raw:
            pixels=get_pixels();budget.live();value=sensor.describe(pixels,bbox_xywh=box[0]).wholebody_xyc.copy()
            raw[key]=value;dirty=True
            if sensor.calls%16==0:checkpoint()
        return raw[key]
    coarse_memory={}
    def coarse(s,native):
        key=(s,int(native))
        if key in coarse_memory:return coarse_memory[key]
        cache=args.cache/(digest(dict(binding=raw_binding,source=s,native=int(native)))+'-faces.npz')
        pixels=None
        if cache.exists():
            with np.load(cache,allow_pickle=False) as z:heads,vectors,pixel_sha=z['heads'],z['vectors'],str(z['pixel_sha'])
        else:
            pixels=rasters(s,[native])[native];heads,vectors=bank.face_observation(pixels,detector,recognizer,budget)
            pixel_sha=hashlib.sha256(pixels.tobytes()).hexdigest()
            with cache.open('xb') as stream:np.savez_compressed(stream,heads=heads,vectors=vectors,pixel_sha=pixel_sha)
        def pixels_once():
            nonlocal pixels
            if pixels is None:pixels=rasters(s,[native])[native]
            if hashlib.sha256(pixels.tobytes()).hexdigest()!=pixel_sha:raise OSError('cached face raster changed')
            return pixels
        value=(heads,vectors,pixel_sha,pixels_once);coarse_memory[key]=value;return value
    def make_clock(s,start,rate,phase=0):
        source=spec['sources'][s]
        return matcher.clock(source=s,binding=source_hashes[s],relative_frames=np.arange(-14,15),
            source_start=start,playback_rate=rate,output_fps=spec['fps'],native_fps=source['fps'],
            native_count=source['frame_count'],phase_frames=phase)
    pose_ticks=np.array([-12,-8,-4,-1,0,4,8,10,12])
    def observed(s,clock,seed_native,seed_head):
        wanted=list(range(int(clock.native_indices.min()),int(clock.native_indices.max())+1))
        samples=[int(v) for v in clock.native_indices[np.isin(clock.output_frames,pose_ticks)]]
        binding=digest(dict(raw=raw_binding,source=s,ordinals=wanted,seed_native=int(seed_native),seed_head=seed_head.tolist(),
            pose_samples=samples,producer=hashlib.sha256(Path(bank.__file__).read_bytes()).hexdigest()))
        target=args.cache/(binding+'-subject.npz');failure=target.with_suffix('.failure.json')
        if target.exists():
            cached=bank.load(target,binding=source_hashes[s])
            expected_track=hashlib.sha256(str((s,int(seed_native))).encode()+np.asarray(seed_head,float).tobytes()).hexdigest()
            if cached.track!=expected_track or not np.array_equal(cached.ordinals,wanted):
                raise ValueError('cached subject seed/ordinal binding changed')
            return cached,dict(reused=True,target=str(target))
        if failure.exists():return None,json.loads(failure.read_text())
        pixels=rasters(s,wanted);frames=[pixels[n] for n in wanted];heads=[];vectors=[]
        budget.arrays(sum(p.nbytes for p in frames)*2)
        for p in frames:
            h,v=bank.face_observation(p,detector,recognizer,budget);heads.append(h);vectors.append(v)
        by_id={id(p):(native,hashlib.sha256(p.tobytes()).hexdigest()) for native,p in pixels.items()}
        class RawSensor:
            def describe(self,pixels,*,bbox_xywh):
                # build() already reserves the request. Reserve only here in
                # the raw cache helper, without double-counting cold forwards.
                nonlocal dirty
                native,pixel_sha=by_id[id(pixels)]
                key=digest(dict(binding=raw_binding,source=s,native=int(native),raster_sha256=pixel_sha,roi=tuple(bbox_xywh)))
                if key not in raw:
                    value=sensor.describe(pixels,bbox_xywh=bbox_xywh).wholebody_xyc.copy();raw[key]=value;dirty=True
                    if sensor.calls%16==0:checkpoint()
                return SimpleNamespace(reason='observed',raw_keypoints_xyc=raw[key][:17])
        value,receipt=bank.build(source=s,binding=source_hashes[s],frames=frames,detections=heads,vectors=vectors,
            ordinals=np.asarray(wanted),seed_native=int(seed_native),seed_head=seed_head,pose_natives=samples,
            sensor=RawSensor(),budget=budget,native_cuts=spec['sources'][s]['native_cuts'])
        checkpoint()
        if value is None:
            with failure.open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
        else:bank.save(value,target,provenance=receipt)
        return value,dict(**receipt,target=str(target),reused=False)
    results=[];error=None
    try:
        for case in spec['cases']:
            coarse_memory.clear();budget.live();s=case['source_A'];lc=make_clock(s,case['source_start_A'],case['rate_A'])
            left,receipt=observed(s,lc,case['seed_native_A'],np.asarray(case['seed_head_A']))
            if left is None:
                results.append(dict(id=case['id'],status='NO_SUPPORTED_MATCH',left_receipt=receipt));continue
            left_index=int(case['seed_native_A']-left.ordinals[0]);valid=left.associated&(np.linalg.norm(left.vectors,axis=1)>.5)
            prototype=left.vectors[valid].mean(0) if valid.any() else np.zeros(128)
            prototype=prototype/max(1e-12,np.linalg.norm(prototype));proposals=[]
            for b,entry in enumerate(case['incoming']):
                if b==s:continue
                rc=make_clock(b,entry['start'],entry['rate']);native=int(rc.native_indices[rc.output_frames==0][0])
                try:heads,vectors,pixel_sha,get_pixels=coarse(b,native)
                except sparse.UnsupportedSeek as exc:
                    incomplete.append(dict(case=case['id'],source=b,reason=str(exc),not_source_wide_exclusion=True));continue
                for h,head in enumerate(heads):
                    appearance=float(prototype@vectors[h]) if np.linalg.norm(vectors[h])>.5 and valid.any() else None
                    proposals.append(dict(source=b,head_index=h,head=head,appearance=appearance,clock=rc,native=native,
                        pixel_sha=pixel_sha,get_pixels=get_pixels,others=np.delete(heads,h,axis=0)))
            # Search cue only, not same-person certification; every actual head
            # was considered, with no first-three or native-focal restriction.
            proposals.sort(key=lambda p:(-(p['appearance'] if p['appearance'] is not None else -.5),p['source'],p['head_index']))
            coarse_rows=[]
            a_body=left.body[left_index];a_valid=left.support[left_index]
            root_a=a_body[:2,:2].mean(0) if a_valid[:2].all() else left.heads[left_index,:2]
            for p in proposals[:12]:
                raw133=pose(p['source'],p['native'],p['head'],p['pixel_sha'],p['get_pixels'])
                hypothesis=face_guided_hypothesis(SimpleNamespace(reason='observed',raw_keypoints_xyc=raw133[:17]),p['head'],p['others'])
                common=a_valid&hypothesis.supported
                phase=None
                if common.sum()>=2:
                    root_b=hypothesis.points[:2,:2].mean(0) if hypothesis.supported[:2].all() else p['head'][:2]
                    diff=(a_body[:,:2]-root_a)/max(.02,left.heads[left_index,2])-(hypothesis.points[:,:2]-root_b)/max(.02,p['head'][2])
                    phase=float(np.linalg.norm(diff*[16/9,1.],axis=-1)[common].mean())
                p['coarse_phase']=phase
                # Unknown stays unmeasured; it cannot outrank an observed
                # action solely through a zero face residual.
                p['coarse_rank']=(phase is None,100. if phase is None else phase+1.-(p['appearance'] or 0.))
                coarse_rows.append(p)
            coarse_rows.sort(key=lambda p:(p['coarse_rank'],p['source'],p['head_index']))
            matches=[];receipts=[]
            for p in coarse_rows[:3]:
                right,rreceipt=observed(p['source'],p['clock'],p['native'],p['head']);receipts.append(rreceipt)
                if right is None:continue
                value=matcher.compare(left,right,lc,p['clock'],left_cameras=np.asarray(case['left_cameras']),
                    incoming_cap=spec['sources'][p['source']]['zoom_cap'])
                matches.append(value)
            ranked=matcher.rank(matches)
            results.append(dict(id=case['id'],**ranked,left_receipt=receipt,incoming_receipts=receipts,
                all_source_head_proposals=len(proposals),coarse_action_probes=len(coarse_rows),
                nonzero_phase_executed=False,following_four_seconds_executed=False,
                original_failed_K=case['original_failed_K']))
            checkpoint();print(json.dumps(dict(case=case['id'],status=ranked['status'],used=budget.used,
                actual_new_pose=sensor.calls,elapsed=monotonic()-started)),flush=True)
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        checkpoint();check()
        report=dict(status='failed' if error else 'completed_candidate_probe_not_quality_approval',error=error,cases=results,
            used=budget.used,actual_new_pose_forwards=sensor.calls,raw_pose_inputs_added=len(set(raw)-raw_before),
            returned_native_rasters=returned_rasters,raster_used_is_predecode_reservation=True,
            peak_python_array_bytes=budget.peak_array_bytes,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            full_RSS_or_ORT_workspace_200MiB_proven=False,hash_io_bytes=hash_bytes,hash_io_seconds=hash_seconds,
            hash_io_is_explicit_preflight_only=True,ORT_model_load_and_decoder_IO_not_fully_instrumented=True,
            elapsed_seconds=monotonic()-started,strict_sparse_decode_receipts=decode_receipts,incomplete_sources=incomplete,
            full_source_fallback_used=False,old_raw_caches_modified=False,weights_fitted=False,
            full_programme_DP_executed=False,video_encoded=False,professional_quality_approved=False,
            user_test_result_created=False,automatic_retry=False,nonzero_phase_not_yet_executed=True,
            explicit_detector_limit_reason='768 for up to three dense incoming tracks across five contexts; other shared review limits unchanged')
        with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
