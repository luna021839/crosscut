"""One bounded raw-subject acquisition BEFORE final source/camera selection.

Default to still-unobserved fixed opportunities. An explicit provisional
programme can instead request its missing endpoints before a final solve;
this is bounded active perception, not teacher timestamps or a movie patch.
Existing observations remain immutable. No source-phase search or model fit.
"""
import hashlib
import inspect
import json
from pathlib import Path
from time import monotonic
import cv2
import numpy as np
from . import subject_window_bank as bank,sparse_native_frames as sparse
from .dense_subject_catalogue import DenseSubjects
from .grid_perception import file_identity
from .rtmw_pose import WholeBodySensor,MODEL_SHA256


def candidates(catalogues,opportunities,*,maximum=12,include_observed=False,explore_missing_action=False):
    """At most one physical pair per fixed music episode; no shot quota."""
    if type(maximum) is not int or maximum<0:raise ValueError('nonnegative observation centre limit required')
    if maximum==0:return []
    best={}
    requests=opportunities.missing_action_requests if explore_missing_action else opportunities.bindings
    for key,value in requests.items():
        n,a,b,ha,hb=key
        if any(catalogues[s].subjects is None for s in (a,b)):continue
        if (explore_missing_action or not include_observed) and all(catalogues[s].subjects.sample(n,side,h) is not None for s,side,h in ((a,0,ha),(b,1,hb))):continue
        episode=int(opportunities.episodes[n]);item=(float(value),key)
        if episode not in best or (-item[0],item[1])<(-best[episode][0],best[episode][1]):best[episode]=item
    ranked=sorted(best.values(),key=lambda v:(-v[0],v[1]))
    if not explore_missing_action:return [key for _,key in ranked[:maximum]]
    # Spread the SAME finite measurement allowance across the whole clock,
    # then fill unused slots by priority. This schedules observations only:
    # no cut, hold length, or editorial utility is prescribed by these bins.
    bins={};nodes=opportunities.nodes;span=max(1,int(nodes[-1]-nodes[0]))
    for item in ranked:
        slot=min(maximum-1,int((nodes[item[1][0]]-nodes[0])*maximum//span))
        bins.setdefault(slot,item)
    chosen=list(bins.values());seen={item[1] for item in chosen}
    chosen.extend(item for item in ranked if item[1] not in seen)
    return [key for _,key in chosen[:maximum]]


def observation_limits(maximum_centres,maximum_seconds):
    """One explicit whole-programme allowance, never per-cut retry budgets."""
    if (type(maximum_centres) is not int or not 0<=maximum_centres<=64
            or isinstance(maximum_seconds,(bool,np.bool_)) or not np.isscalar(maximum_seconds)
            or not np.isfinite(maximum_seconds) or not 0<maximum_seconds<=900):
        raise ValueError('Explicit bounded opportunity observation centres/time required')
    return dict(face_limit=max(768,64*maximum_centres),pose_limit=max(512,32*maximum_centres),
        raster_limit=max(2048,128*maximum_centres),recognition_limit=max(4096,256*maximum_centres),
        seconds=float(maximum_seconds))


def provisional_candidates(catalogues,nodes,seams,*,maximum):
    """Measure unobserved prospective endpoints once, across the whole clock.

    These are requests from an automatic provisional programme. They do not
    freeze a cut, modify its cost, prescribe a final edit, or grant a credit.
    """
    if type(maximum) is not int or not 0<=maximum<=64 or not isinstance(seams,list) or len(seams)>2048:
        raise ValueError('bounded provisional programme endpoints required')
    grid=np.asarray(nodes);keys=[];seen=set()
    for row in seams:
        values=[row[k] for k in ('output_frame','left_source','right_source','left_head_index','right_head_index')]
        if any(type(v) is not int for v in values):raise ValueError('integer provisional endpoint addresses required')
        tick,a,b,ha,hb=values;n=int(np.searchsorted(grid,tick))
        if not 0<n<len(grid)-1 or grid[n]!=tick or not 0<=a<len(catalogues) or not 0<=b<len(catalogues) or a==b:
            raise ValueError('provisional endpoint must use the actual grid and physical sources')
        # A legs/formation endpoint has no face-guided seed. Do not invent
        # one or turn its absence into a source exclusion.
        if min(ha,hb)<0:continue
        if any(h>=catalogues[s].bank['heads'].shape[2] for s,h in ((a,ha),(b,hb))):
            raise ValueError('provisional face address outside actual source bank')
        key=(n,a,b,ha,hb)
        if key in seen:continue
        seen.add(key)
        if any(catalogues[s].subjects is None for s in (a,b)):continue
        if all(catalogues[s].subjects.sample(n,side,h) is not None for s,side,h in ((a,0,ha),(b,1,hb))):continue
        keys.append(key)
    keys.sort()
    if not maximum:return []
    if len(keys)<=maximum:return keys
    return [keys[i] for i in np.linspace(0,len(keys)-1,maximum).round().astype(int)]


def endpoint_availability(catalogues,selected):
    """Actual usable requested windows, not process exit or model-call counts.

    Deferred requests may legitimately remain UNKNOWN. Report that state
    explicitly; a zero-call/cache-only pass must not imply data completion.
    """
    addresses={(n,s,side,h) for n,a,b,ha,hb in selected for s,side,h in ((a,0,ha),(b,1,hb))}
    missing=sum(catalogues[s].subjects.sample(n,side,h) is None for n,s,side,h in addresses)
    return dict(requested_endpoints=len(addresses),unobserved_endpoints=missing,
        all_requested_endpoints_available=missing==0)


def observer(specification,*,sources,source_hashes,identities,cache,authorized_root,device,image_changes=False,
             explore_missing_action=False,maximum_centres=12,maximum_seconds=180.,provisional_plan=None,master_start=None):
    """Explicit local model/media spec; sessions stay lazy until a cache miss."""
    limits=observation_limits(maximum_centres,maximum_seconds)
    spec=json.loads(Path(specification).read_text());root=Path(authorized_root).resolve(strict=True);cache=Path(cache)
    if [str(p) for p in sources]!=[v['path'] for v in spec['sources']]:raise ValueError('probe source specification differs')
    if not cache.resolve().is_relative_to(root):raise ValueError('subject cache outside authorized SSD project')
    provisional=None;provisional_sha=None
    if provisional_plan is not None:
        path=Path(provisional_plan)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('provisional programme outside authorized source project')
        raw=path.read_bytes();provisional=json.loads(raw);provisional_sha=hashlib.sha256(raw).hexdigest()
        if (provisional.get('source_sha256')!=list(source_hashes)
                or provisional.get('source_file_identities')!=[list(v) for v in identities]
                or provisional.get('render_plan',{}).get('sources')!=list(map(str,sources))
                or provisional.get('render_plan',{}).get('fps')!=spec['fps']
                or isinstance(master_start,(bool,np.bool_)) or not isinstance(master_start,(int,float))
                or not np.isfinite(master_start)
                or provisional.get('render_plan',{}).get('master_audio_start')!=master_start):
            raise ValueError('provisional programme physical source/clock domain differs')
    models={k:hashlib.sha256(Path(spec[k]).read_bytes()).hexdigest() for k in ('pose_model','detector_model','recognizer_model')}
    if models['pose_model']!=MODEL_SHA256:raise ValueError('subject opportunity pose model changed')
    producer=hashlib.sha256(json.dumps(dict(models=models,opencv=cv2.__version__,ort=__import__('onnxruntime').__version__,
        sensor=hashlib.sha256(Path(inspect.getfile(WholeBodySensor)).read_bytes()).hexdigest(),
        window=hashlib.sha256(Path(bank.__file__).read_bytes()).hexdigest(),
        adapter=hashlib.sha256(inspect.getsource(bank.face_guided_hypothesis).encode()).hexdigest(),raster=[960,540]),sort_keys=True).encode()).hexdigest()
    sessions=[None];native_clocks={};face_memory={}
    def run(*,catalogues,opportunities,check):
        started=monotonic()
        if spec['fps']!=opportunities.fps:raise ValueError('opportunity music clock differs')
        if any(c.subjects is None or c.subjects.source!=s or c.subjects.binding!=source_hashes[s]
                for s,c in enumerate(catalogues)):
            raise ValueError('actual source-bound subject catalogue required before probing')
        def live():
            check()
            if not Path('/Volumes/CrossCutMedia').is_mount() or root.stat().st_dev!=device:raise OSError('authorized SSD changed')
            if any(file_identity(p)!=v for p,v in zip(sources,identities)):raise OSError('probe source changed')
        budget=bank.ProbeBudget(**limits,check=live)
        selected=(candidates(catalogues,opportunities,maximum=maximum_centres,include_observed=image_changes,
            explore_missing_action=explore_missing_action) if provisional is None else
            provisional_candidates(catalogues,opportunities.nodes,provisional['seams'],maximum=maximum_centres))
        initial_availability=endpoint_availability(catalogues,selected)
        receipts=[];decodes=[];new=[[] for _ in sources];windows=0
        cache.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
        for n,a,b,ha,hb in selected:
            live()
            for s,side,h in ((a,0,ha),(b,1,hb)):
                cat=catalogues[s]
                if cat.subjects.sample(n,side,h) is not None:continue
                requested=cat.subjects.indices[n];wanted=np.arange(int(requested.min()),int(requested.max())+1)
                seed=int(cat.bank['native_indices'][n,side]);head=cat.bank['heads'][n,side,h]
                signature=hashlib.sha256(json.dumps(dict(source=source_hashes[s],producer=producer,
                    wanted=wanted.tolist(),requested=requested.tolist(),seed=seed,head=head.tolist()),sort_keys=True).encode()).hexdigest()
                path=cache/(signature+'-window.npz');window=None
                if path.exists():
                    window=bank.load(path,binding=source_hashes[s]);reused=True
                    expected=hashlib.sha256(str((s,seed)).encode()+np.asarray(head,float).tobytes()).hexdigest()
                    if window.track!=expected or not np.array_equal(window.ordinals,wanted):raise ValueError('cached probe seed/clock changed')
                else:
                    reused=False
                    if windows>=2*maximum_centres or any(budget.used[k]+count>getattr(budget,k+'_limit') for k,count in
                            (('face',len(wanted)),('pose',len(np.unique(requested))),('raster',len(wanted)),('recognition',8*len(wanted)))):
                        receipts.append(dict(source=s,status='OBSERVATION_DEFERRED',reason='shared_allowance'));continue
                    try:
                        budget.live();windows+=1
                        if sessions[0] is None:
                            sessions[0]=(WholeBodySensor(spec['pose_model']),
                                cv2.FaceDetectorYN.create(spec['detector_model'],'',(960,540),.75,.3,5000),
                                cv2.FaceRecognizerSF.create(spec['recognizer_model'],''))
                        sensor,detector,recognizer=sessions[0]
                        if s not in native_clocks:native_clocks[s]=sparse.probe(sources[s],check=budget.live)
                        if (float(native_clocks[s].rate)!=spec['sources'][s]['fps']
                                or native_clocks[s].count!=spec['sources'][s]['frame_count']):
                            raise ValueError('actual native source clock differs from probe specification')
                        budget.reserve('raster',len(wanted));budget.arrays((2*len(wanted)+8)*960*540*3)
                        decode={};pixels=dict(sparse.frames(sources[s],wanted.tolist(),clock=native_clocks[s],check=budget.live,
                            receipt=decode,maximum_decode_seconds=12.,
                            maximum_seconds=min(30.,max(1.,budget.seconds-(monotonic()-budget.started)))))
                        decodes.append(dict(source=s,**decode));heads=[];vectors=[]
                        for native in wanted:
                            key=(s,int(native))
                            if key not in face_memory:face_memory[key]=bank.face_observation(pixels[int(native)],detector,recognizer,budget)
                            rows,features=face_memory[key];heads.append(rows);vectors.append(features)
                        raw=[];native_rows=[];rois=[];raster_hashes=[];by_id={id(v):n for n,v in pixels.items()}
                        class RecordingSensor:
                            def describe(self,image,*,bbox_xywh):
                                observed=sensor.describe(image,bbox_xywh=bbox_xywh)
                                raw.append(observed.wholebody_xyc.copy());native_rows.append(by_id[id(image)])
                                rois.append(bbox_xywh);raster_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                                return observed
                        window,proof=bank.build(source=s,binding=source_hashes[s],frames=[pixels[int(v)] for v in wanted],
                            detections=heads,vectors=vectors,ordinals=wanted,seed_native=seed,seed_head=head,
                            pose_natives=np.unique(requested),sensor=RecordingSensor(),budget=budget,native_cuts=spec['sources'][s]['native_cuts'])
                        if window is not None:bank.save(window,path,provenance=dict(**proof,producer=producer))
                        if raw:
                            with path.with_name(path.stem+'-raw133.npz').open('xb') as stream:
                                np.savez_compressed(stream,raw133=np.asarray(raw),native=native_rows,rois=np.asarray(rois),
                                    raster_sha256=raster_hashes,producer=producer)
                    except TimeoutError:
                        receipts.append(dict(source=s,status='OBSERVATION_DEFERRED',reason='shared_allowance'));continue
                if window is not None:
                    new[s].append(window);receipts.append(dict(source=s,path=str(path),reused=reused,
                        status='raw_subject_observed_before_final_selection',output_frame=int(cat.nodes[n])))
            print(json.dumps(dict(phase='preselection_subject_opportunity',centres_done=selected.index((n,a,b,ha,hb))+1,
                centres=len(selected),used=budget.used,elapsed_seconds=monotonic()-started)),flush=True)
        for s,cat in enumerate(catalogues):
            if not new[s]:continue
            old=cat.subjects;existing={id(w):w for rows in old.by_native.values() for w,_ in rows}
            cat.subjects=DenseSubjects(source=old.source,binding=old.binding,indices=old.indices,
                windows=[*existing.values(),*new[s]],bank=cat.bank,fps=cat.fps)
        image_receipts=[]
        if image_changes:
            from .observed_reframing import measure
            requested=[{} for _ in sources]
            for n,a,b,ha,hb in selected:
                for s in (a,b):
                    pair=tuple(map(int,catalogues[s].bank['native_indices'][n]))
                    if pair[0]<pair[1]:requested[s].setdefault(pair,[]).append(int(catalogues[s].nodes[n]))
            for s,pairs in enumerate(requested):
                live();rows=[]
                if pairs:
                    try:
                        measured,proof=measure(sources[s],np.asarray(list(pairs),np.int64),
                            source_sha256=source_hashes[s],source_identity=identities[s],
                            native_cuts=spec['sources'][s]['native_cuts'],cache=cache/'observed-image-pairs',
                            budget=budget,check=live)
                    except TimeoutError:
                        measured=[];proof=dict(status='OBSERVATION_DEFERRED',reason='same_shared_observation_allowance')
                    for pair,item in zip(pairs,measured):
                        if item['binding']['native']!=list(pair):raise ValueError('image pair producer changed request order')
                        for tick in pairs[pair]:
                            rows.append(dict(output_frames=[tick-1,tick],native_pair=list(pair),
                                source_sha256=source_hashes[s],observation=item['observation']))
                    image_receipts.append(dict(source=s,pairs=len(pairs),
                        states=[r['observation']['state'] for r in rows],measurement=proof))
                catalogues[s].observed_image_records=tuple(rows)
        return dict(contract='bounded_preselection_fixed_opportunity_subject_probe_v1',centres=len(selected),new_windows=windows,
            endpoint_availability_before=initial_availability,
            endpoint_availability_after=endpoint_availability(catalogues,selected),
            maximum_centres=maximum_centres,maximum_new_windows=2*maximum_centres,maximum_seconds=maximum_seconds,
            used=budget.used,window_receipts=receipts,decode_receipts=decodes,elapsed_seconds=monotonic()-started,
            neural_model_sessions_initialized=sessions[0] is not None,source_NN_calls=sum(budget.used[k] for k in ('face','pose','recognition')),
            selected_movie_or_reference_timestamps_used=provisional is not None,old_observations_rewritten=False,
            request_policy='fixed_music_opportunities' if provisional is None else 'provisional_programme_missing_endpoints',
            provisional_plan_sha256=provisional_sha,provisional_cuts_are_not_frozen_final_cuts=provisional is not None,
            user_or_teacher_timestamps_used=False,automatic_retry=False,
            missing_action_exploration=bool(explore_missing_action),
            observation_requests_are_not_action_credit=True,
            observation_centres=[list(key) for key in selected],
            image_changes_observed_on_same_preselection_centres=bool(image_changes),
            image_observation_receipts=image_receipts,image_measurements_share_original_budget=True,
            explicit_array_peak_bytes=budget.peak_array_bytes,total_process_RSS_not_certified=True)
    return run
