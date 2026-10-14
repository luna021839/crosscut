"""Bounded source evidence for unexplained output detections, not EDL repair.

An output detector proposes a location, never a trusted source cut. Confirm a
persistent structural discontinuity on exact ORIGINAL native rasters, then add
that physical observation to later source unions. No timestamp blacklist,
post-DP retiming, new NN or implicit full-source decode. This heuristic is not
complete cut recall, semantic proof, or a professional-quality certificate.
"""
import hashlib,json,os,shutil,tempfile
from pathlib import Path
from time import monotonic
import cv2,numpy as np
from . import sparse_native_frames
from .critic import signature
from .grid_perception import file_identity
from .observation_cache_contract import function_text


def _binding(path):
    decoder=shutil.which('ffmpeg')
    if not decoder:raise OSError('existing decoder required')
    here=Path(__file__)
    seal=hashlib.sha256(here.read_bytes()+Path(sparse_native_frames.__file__).read_bytes()
        +function_text(here.with_name('critic.py'),'signature').encode()
        +function_text(here.with_name('source_cuts.py'),'read_frames').encode()
        +function_text(here.parent.parent/'crosscut/render_frame_schedule.py','render_frame_clock_from_probe').encode()).hexdigest()
    return dict(source=list(file_identity(path)),decoder=list(file_identity(Path(decoder).resolve())),
        implementation=seal,opencv=cv2.__version__,numpy=np.__version__,raster=[320,180],filter='area')


def persistent_structure_cut(frames):
    """Six chronological raw frames: three before, three after the proposal.

    Fixed noncalibrated confirmation rule: internally coherent sides, abrupt
    cross-side structure change. Brightness-only flashes don't qualify. Slow
    pans and textureless frames remain unknown rather than invented cuts.
    """
    if len(frames)!=6:raise ValueError('exact3+3 native confirmation frames required')
    values=np.asarray([signature(frame)[0] for frame in frames])
    if not np.isfinite(values).all() or (np.linalg.norm(values,axis=1)<.5).any():return False
    before,after=values[:3],values[3:]
    within=np.array([before[0]@before[1],before[1]@before[2],after[0]@after[1],after[1]@after[2]])
    return bool(within.min()>=.55 and np.median(before@after.T)<=.4
        and float(before[-1]@after[0])<=.4)


def observed_native_cuts(sources,cache):
    """Read only exact source/runtime/implementation-bound completed evidence."""
    bindings=[_binding(path) for path in sources];result=[[] for _ in sources]
    for path in sorted(Path(cache).glob('*.json')):
        if len(path.stem)!=64 or any(c not in '0123456789abcdef' for c in path.stem):continue
        value=json.loads(path.read_text());binding=value.get('binding')
        if binding not in bindings:continue
        if (value.get('contract')!='raw_confirmed_output_proposed_native_cut_v1'
                or value.get('confirmed') is not True or type(value.get('native_frame')) is not int
                or value['native_frame']<3 or value.get('new_neural_calls')!=0):
            raise ValueError('invalid completed source-cut evidence')
        result[bindings.index(binding)].append(value['native_frame'])
    return [sorted(set(xs)) for xs in result]


def discover_from_output(*,movie,record,critic,cache,check,maximum_events=8,maximum_native_frames=64,
                         maximum_seconds=30.):
    """At most one declared refinement pass; caller owns whether to reselect."""
    from crosscut.render_plan_serialization import render_plan_from_json
    from crosscut.render_frame_schedule import probe_render_frame_clock
    from .programme_critic import programme_schedules,rendered_source_frame
    if not 0<=maximum_events<=8 or not 0<=maximum_native_frames<=64 or not 1<=maximum_seconds<=60:
        raise ValueError('bounded one-pass source refinement required')
    started=monotonic();movie=Path(movie);cache=Path(cache)
    plan=render_plan_from_json(record['render_plan']);identities=[file_identity(p) for p in plan.sources]
    if (plan.output!=movie or tuple(critic.get('source_file_identity',()))!=file_identity(movie)
            or identities!=[tuple(r['source_file_identity']) for r in record['sparse_perception']]):
        raise ValueError('current output and source pixel binding required')
    def live():
        check()
        if monotonic()-started>maximum_seconds:raise TimeoutError('one source refinement allowance exhausted')
        if identities!=[file_identity(p) for p in plan.sources]:raise OSError('source changed during refinement')
    clocks=[probe_render_frame_clock(p) for p in plan.sources]
    schedules,authored,_=programme_schedules(plan,clocks,[[] for _ in plan.sources])
    candidates=critic['boundary_coverage']['unexplained_detections']
    if len(candidates)>maximum_events:raise ValueError('output proposals exceed one declared refinement allowance')
    if any(type(t) is not int or not 0<t<plan.output_frame_count for t in candidates):
        raise ValueError('exact interior output proposal ordinals required')
    reports=[];decoded=0;confirmed=[];source_clocks={}
    for tick in sorted(set(candidates)):
        live();a=rendered_source_frame(schedules,tick-1);b=rendered_source_frame(schedules,tick)
        if tick in authored or a['source']!=b['source'] or b['native_frame']!=a['native_frame']+1:
            reports.append(dict(output_frame=tick,confirmed=False,reason='not_one_consecutive_native_source_pair'));continue
        s=int(b['source']);f=int(b['native_frame']);source=plan.sources[s];binding=_binding(source)
        key=hashlib.sha256(json.dumps(dict(binding=binding,native_frame=f),sort_keys=True).encode()).hexdigest()
        target=cache/(key+'.json')
        if target.exists():
            saved=json.loads(target.read_text())
            if saved.get('binding')!=binding or saved.get('native_frame')!=f:raise ValueError('source-cut cache binding differs')
            reports.append(dict(output_frame=tick,source=s,native_frame=f,reused=True,confirmed=saved['confirmed']))
            if saved['confirmed']:confirmed.append((s,f))
            continue
        if decoded+6>maximum_native_frames:raise ValueError('source refinement native frame allowance exceeded')
        if s not in source_clocks:source_clocks[s]=sparse_native_frames.probe(source,check=live)
        clock=source_clocks[s];expected=record['bound_source_cut_events'][s]
        if clock.count!=expected['frame_count'] or abs(float(clock.rate)-expected['fps'])>1e-8:
            raise ValueError('source refinement clock differs from selected source domain')
        needed=list(range(f-3,f+3));execution={}
        if needed[0]<0 or needed[-1]>=clock.count:
            reports.append(dict(output_frame=tick,source=s,native_frame=f,confirmed=False,reason='insufficient_source_context'));continue
        pixels=dict(sparse_native_frames.frames(source,needed,clock=clock,check=live,width=320,height=180,receipt=execution))
        decoded+=len(pixels)
        accepted=persistent_structure_cut([pixels[t] for t in needed]);live()
        value=dict(contract='raw_confirmed_output_proposed_native_cut_v1',binding=binding,native_frame=f,
            confirmed=accepted,new_neural_calls=0,original_native_indices=needed,execution=execution,
            source_event_is_heuristic_not_semantic_certification=True)
        # Rejected proposals are reports, not trusted cut observations.
        if accepted:
            cache.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.cut-',suffix='.json',dir=cache)
            try:
                with os.fdopen(fd,'w') as stream:json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
                live();os.replace(tmp,target)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
            confirmed.append((s,f))
        reports.append(dict(output_frame=tick,source=s,native_frame=f,reused=False,confirmed=accepted,execution=execution))
    return dict(candidates=len(candidates),confirmed_source_events=confirmed,proposals=reports,
        decoded_native_frames=decoded,new_neural_calls=0,source_full_scan_used=False,
        edited_existing_plan=False,post_selection_retimed=False,elapsed_seconds=monotonic()-started)
