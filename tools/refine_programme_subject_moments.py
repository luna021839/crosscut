"""Observe only proposed cut neighbourhoods, then jointly freeze edit moments.

The coarse automatically selected source sequence is retained, not an old
manual montage. Both music clocks move together. One bounded observation
batch is cached independently; a warm run only re-ranks immutable observations.
"""
import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
from time import monotonic
from types import SimpleNamespace

import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
from crosscut.render import EditorialTransformKeyframe,EditorialTransformTrack,render
from crosscut.render_frame_schedule import sampler_frame_schedule
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut2 import sparse_native_frames as sparse,subject_window_bank as bank
from crosscut2.grid_perception import file_identity
from crosscut2.subject_action_match import clock
from crosscut2.subject_cut_moments import search
from crosscut2.subject_moment_sequence import choose,hold_curve,curve_digest,camera_motion
from crosscut2.fixed_holds import _curve_at
from crosscut2.rtmw_pose import WholeBodySensor,MODEL_SHA256


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','source-spec','cache','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ssd-device',type=int,required=True);parser.add_argument('--render',action='store_true')
    args=parser.parse_args();started=monotonic()
    root=Path('/Volumes/CrossCutMedia/CrossCut-Validation').resolve(strict=True)
    if any(not p.resolve().is_relative_to(root) for p in (args.plan,args.cache,args.output)):
        raise ValueError('media and new observations must remain in the SSD project')
    if args.output.exists() or args.output.with_suffix('.plan.json').exists():raise FileExistsError('recover the saved plan')
    parent_bytes=args.plan.read_bytes();parent=json.loads(parent_bytes)
    plan=render_plan_from_json(parent['render_plan']);spec=json.loads(args.source_spec.read_text())
    if any(not p.resolve(strict=True).is_relative_to(root) for p in plan.sources):
        raise ValueError('source media outside the explicitly authorized SSD project')
    if [str(p) for p in plan.sources]!=[s['path'] for s in spec['sources']] or plan.fps!=spec['fps']:
        raise ValueError('source/clock specification differs from the actual coarse programme')
    identities=[file_identity(p) for p in plan.sources]
    if identities!=[tuple(v) for v in parent['source_file_identities']]:raise OSError('coarse programme source changed')
    def check():
        if not Path('/Volumes/CrossCutMedia').is_mount() or root.stat().st_dev!=args.ssd_device:raise OSError('authorized SSD changed')
        if any(file_identity(p)!=v for p,v in zip(plan.sources,identities)):raise OSError('source changed')
    check();args.cache.mkdir(parents=True,exist_ok=True)
    budget=bank.ProbeBudget(face_limit=9000,recognition_limit=60000,pose_limit=4000,raster_limit=10000,
        seconds=600.,check=check)
    sensor=WholeBodySensor(spec['pose_model']);cv2.setNumThreads(2)
    detector=cv2.FaceDetectorYN.create(spec['detector_model'],'',(960,540),.75,.3,5000)
    recognizer=cv2.FaceRecognizerSF.create(spec['recognizer_model'],'')
    models={key:hashlib.sha256(Path(spec[key]).read_bytes()).hexdigest()
            for key in ('pose_model','detector_model','recognizer_model')}
    if models['pose_model']!=MODEL_SHA256:raise ValueError('pose model changed')
    producer=digest(dict(models=models,opencv=cv2.__version__,ort=__import__('onnxruntime').__version__,
        sensor=hashlib.sha256(Path(inspect.getfile(WholeBodySensor)).read_bytes()).hexdigest(),
        window=hashlib.sha256(Path(bank.__file__).read_bytes()).hexdigest(),
        adapter=hashlib.sha256(inspect.getsource(bank.face_guided_hypothesis).encode()).hexdigest(),raster=[960,540]))
    native_clocks={};face_memory={};decode_receipts=[];window_receipts=[]
    final=plan.output_frame_count;fps=plan.fps
    starts=np.array([round(s.master_start*fps) for s in plan.segments],int)
    ends=np.r_[starts[1:],final]
    interiors=[]
    for first,last,segment in zip(starts,ends,plan.segments):
        track=segment.editorial_transform
        if track is None:ticks=np.array([first,last-1]);values=np.tile([1.,.5,.5],(2,1))
        else:
            ticks=np.array([first+round(k.time_seconds*fps) for k in track.keyframes])
            values=np.array([[k.zoom,k.center_x,k.center_y] for k in track.keyframes])
        interiors.append((ticks,values))
    def at_source(segment,first,tick):
        return segment.source_start+(tick-first)/fps*segment.source_playback_rate
    def source_clock(segment,first,anchor):
        source=spec['sources'][segment.source_index]
        return clock(source=segment.source_index,binding=parent['source_sha256'][segment.source_index],
            relative_frames=np.arange(-14,15),source_start=at_source(segment,first,anchor),
            playback_rate=segment.source_playback_rate,output_fps=fps,native_fps=source['fps'],
            native_count=source['frame_count'])
    def raw_head(seam,side):
        index=1 if side==0 else 2
        expected=seam['expected_output_focal'][index]
        camera=np.asarray(seam['actual_camera_strips'][side][-1 if side==0 else 0])
        return np.r_[camera[1:]+(np.asarray(expected['center'])-.5)/camera[0],
                     expected['height']/camera[0],expected['confidence']]
    def observed(s,hypothesis,seed_native,seed_head):
        wanted=np.arange(int(hypothesis.native_indices.min()),int(hypothesis.native_indices.max())+1)
        requests=np.unique(hypothesis.native_indices)
        binding=digest(dict(source=parent['source_sha256'][s],identity=identities[s],producer=producer,
            ordinals=wanted.tolist(),requests=requests.tolist(),seed_native=seed_native,seed_head=seed_head.tolist()))
        target=args.cache/(binding+'-window.npz');failure=target.with_suffix('.failure.json')
        if target.exists():
            result=bank.load(target,binding=parent['source_sha256'][s])
            expected=hashlib.sha256(str((s,int(seed_native))).encode()+np.asarray(seed_head,float).tobytes()).hexdigest()
            if result.track!=expected or not np.array_equal(result.ordinals,wanted):raise ValueError('cached subject seed or clock changed')
            return result,dict(source=s,reused=True,path=str(target))
        if failure.exists():return None,json.loads(failure.read_text())
        if s not in native_clocks:native_clocks[s]=sparse.probe(plan.sources[s],check=budget.live)
        budget.reserve('raster',len(wanted));budget.arrays((2*len(wanted)+8)*960*540*3)
        receipt={}
        pixels=dict(sparse.frames(plan.sources[s],wanted.tolist(),clock=native_clocks[s],check=budget.live,
            receipt=receipt,maximum_seconds=min(30.,max(1.,budget.seconds-(monotonic()-budget.started)))))
        decode_receipts.append(dict(source=s,**receipt));frames=[pixels[int(n)] for n in wanted]
        heads=[];vectors=[]
        for native,pixel in zip(wanted,frames):
            key=s,int(native)
            if key not in face_memory:face_memory[key]=bank.face_observation(pixel,detector,recognizer,budget)
            h,v=face_memory[key];heads.append(h);vectors.append(v)
        by_id={id(pixel):int(n) for n,pixel in pixels.items()};raw=[];raw_natives=[];rois=[];raster_hashes=[]
        class Sensor:
            def describe(self,pixel,*,bbox_xywh):
                value=sensor.describe(pixel,bbox_xywh=bbox_xywh).wholebody_xyc.copy()
                raw.append(value);raw_natives.append(by_id[id(pixel)]);rois.append(bbox_xywh)
                raster_hashes.append(hashlib.sha256(pixel.tobytes()).hexdigest())
                return SimpleNamespace(reason='observed',raw_keypoints_xyc=value[:17])
        result,receipt=bank.build(source=s,binding=parent['source_sha256'][s],frames=frames,detections=heads,
            vectors=vectors,ordinals=wanted,seed_native=seed_native,seed_head=seed_head,
            pose_natives=requests,sensor=Sensor(),budget=budget,native_cuts=spec['sources'][s]['native_cuts'])
        if result is None:
            with failure.open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
        else:bank.save(result,target,provenance=dict(**receipt,producer=producer))
        if raw:
            raw_path=target.with_name(target.stem+'-raw133.npz')
            with raw_path.open('xb') as stream:np.savez_compressed(stream,raw133=np.asarray(raw),native=raw_natives,
                rois=np.asarray(rois),raster_sha256=raster_hashes,producer=producer)
        return result,dict(**receipt,source=s,reused=False,path=str(target))
    def anchor(i):
        tick=int(starts[i]) if i<len(starts) else final
        left=None if i==0 else _curve_at(*interiors[i-1],np.clip(tick+np.arange(-7,0),starts[i-1],ends[i-1]-1))
        right=None if i==len(starts) else _curve_at(*interiors[i],np.clip(tick+np.arange(7),starts[i],ends[i]-1))
        return dict(frame=tick,left_strip=left,right_strip=right,configuration_cost=0.,connection_cost=0.,
            observed=False,metrics=None)
    levels=[[anchor(0)]];cases=[]
    for i,seam in enumerate(parent['seams'],start=1):
        budget.live();tick=int(starts[i]);a,b=plan.segments[i-1:i+1]
        radius=max(0,min(6,(tick-starts[i-1]-14)//2,(ends[i]-tick-14)//2))
        if min(tick-starts[i-1],ends[i]-tick)<14:
            levels.append([anchor(i)]);cases.append(dict(frame=tick,status='short_original_arc_unchanged'));continue
        lc,rc=source_clock(a,starts[i-1],tick),source_clock(b,starts[i],tick)
        left,lreceipt=observed(a.source_index,lc,int(lc.native_indices[lc.output_frames==-1][0]),raw_head(seam,0))
        right,rreceipt=observed(b.source_index,rc,int(rc.native_indices[rc.output_frames==0][0]),raw_head(seam,1))
        window_receipts.extend([lreceipt,rreceipt]);result=None;alternatives=[]
        if left is not None and right is not None:
            result=search(left,right,lc,rc,candidate_frames=list(range(-radius,radius+1)),
                outgoing_cap=spec['sources'][a.source_index]['zoom_cap'],incoming_cap=spec['sources'][b.source_index]['zoom_cap'])
            for value in result['supported']:
                alternatives.append(dict(frame=tick+value['cut_offset_frames'],observed=True,
                    left_strip=np.tile(value['outgoing_camera'],(7,1)),right_strip=np.tile(value['incoming_camera'],(7,1)),
                    configuration_cost=value['seam_arm_direction_residual'],connection_cost=value['cut_moment_score'],metrics=value))
        levels.append(alternatives or [anchor(i)])
        cases.append(dict(frame=tick,supported_alternatives=len(alternatives),result=result))
        print(json.dumps(dict(phase='actual_cut_neighbourhood',cut=i,total=len(parent['seams']),frame=tick,
            supported_alternatives=len(alternatives),new_pose=sensor.calls,face_calls=budget.used['face'],
            elapsed_seconds=monotonic()-started)),flush=True)
    levels.append([anchor(len(starts))])
    def arc(i,a,b):
        segment=plan.segments[i];s=segment.source_index
        value=hold_curve(a['frame'],b['frame'],interiors[i],a['right_strip'],b['left_strip'],cap=spec['sources'][s]['zoom_cap'])
        if value is None:return None
        native=sampler_frame_schedule(source_start=at_source(segment,starts[i],a['frame']),
            source_playback_rate=segment.source_playback_rate,native_fps=spec['sources'][s]['fps'],output_fps=fps,
            frame_count=b['frame']-a['frame'],source_frame_count=spec['sources'][s]['frame_count']).source_frame_indices
        return dict(camera_cost=camera_motion(value.values,native,spec['sources'][s]['native_cuts'],fps=fps),
            camera_sha256=curve_digest(value.values))
    selected,edge_receipts,receipt=choose(levels,arc,check=check)
    moments=[values[k] for values,k in zip(levels,selected)];segments=[];seams=[];proof=[]
    for i,(a,b,edge) in enumerate(zip(moments[:-1],moments[1:],edge_receipts)):
        old=plan.segments[i];first,last=a['frame'],b['frame'];s=old.source_index
        curve=hold_curve(first,last,interiors[i],a['right_strip'],b['left_strip'],cap=spec['sources'][s]['zoom_cap'])
        if curve is None or curve_digest(curve.values)!=edge['camera_sha256']:raise AssertionError('scored camera replay changed')
        track=EditorialTransformTrack(tuple(EditorialTransformKeyframe(k/fps,*map(float,cam))
            for k,cam in enumerate(curve.values)),reason='joint_subject_music_moment_frozen_before_render')
        segment=replace(old,master_start=first/fps,source_start=at_source(old,starts[i],first),duration=(last-first)/fps,
            source_duration=(last-first)/fps*old.source_playback_rate,editorial_transform=track)
        segments.append(segment)
        native=sampler_frame_schedule(source_start=segment.source_start,source_playback_rate=segment.source_playback_rate,
            native_fps=spec['sources'][s]['fps'],output_fps=fps,frame_count=last-first,
            source_frame_count=spec['sources'][s]['frame_count']).source_frame_indices
        if a['metrics'] is not None and int(native[0])!=a['metrics']['native_B'][6]:raise AssertionError('incoming subject native input changed')
        if b['metrics'] is not None and int(native[-1])!=b['metrics']['native_A'][5]:raise AssertionError('outgoing subject native input changed')
        proof.append(dict(source=s,first_frame=first,final_frame=last,scored_camera_sha256=edge['camera_sha256'],
            native_first=int(native[0]),native_last=int(native[-1]),observed_entry=a['observed'],observed_exit=b['observed']))
        if i:
            seams.append(dict(output_frame=first,seconds=first/fps,left_source=plan.segments[i-1].source_index,
                right_source=s,cut_moment_observed=a['observed'],metrics=a['metrics'],source_retimed=False))
    refined=replace(plan,segments=tuple(segments),output=args.output)
    receipt.update(elapsed_seconds=monotonic()-started,source_NN_calls=sensor.calls,
        new_face_calls=budget.used['face'],used=budget.used,window_receipts=window_receipts,
        decode_receipts=decode_receipts,cut_cases=cases,selected_moments=[v['frame'] for v in moments],
        parent_moments=[int(v) for v in np.r_[starts,final]],camera_native_replay=proof,
        parent_plan_sha256=hashlib.sha256(parent_bytes).hexdigest(),old_observations_rewritten=False,
        source_hashes_reused_from_identity_verified_parent=True,normal_app_path_validated=False)
    record={k:v for k,v in parent.items() if k not in ('render_plan','seams','fixed_actor_camera_hold_receipt',
        'selected_clock_check','source_NN_calls','source_analysis_decode','historical_observations_replayed')}
    record.update(render_plan=refined.to_json(),seams=seams,subject_moment_refinement_receipt=receipt,
        source_NN_calls=sensor.calls+budget.used['face']+budget.used['recognition'],
        source_analysis_decode=budget.used['raster'],coarse_programme_source_NN_calls=parent.get('source_NN_calls'),
        selected_clock_check=dict(camera_replay_exact=True,observed_subject_native_ordinals_exact=True,
            all_original_sparse_grid_inputs_rechecked=False,normal_speed_audio_playback_performed=False))
    with args.output.with_suffix('.plan.json').open('x') as stream:
        stream.write(json.dumps(record,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x,allow_nan=False)+'\n')
    print(json.dumps(dict(phase='refined_programme_frozen',moved_cuts=sum(v['frame']!=int(starts[i])
        for i,v in enumerate(moments[1:-1],start=1)),observed_cuts=sum(v['observed'] for v in moments[1:-1]),
        cuts=len(seams),new_pose=sensor.calls,elapsed_seconds=monotonic()-started)),flush=True)
    if args.render:
        last=[-10.]
        def progress(event):
            check();elapsed=monotonic()-started
            if elapsed-last[0]>=10. or event.get('phase')=='end':
                print(json.dumps(dict(phase='render',elapsed_seconds=elapsed,event=event)),flush=True);last[0]=elapsed
        render(refined,maximum_runtime_seconds=900.,video_bitrate='24M',progress_callback=progress)
        print(json.dumps(dict(phase='movie_ready',path=str(args.output),elapsed_seconds=monotonic()-started)),flush=True)


if __name__=='__main__':main()
