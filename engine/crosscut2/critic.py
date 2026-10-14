"""Output-only cut diagnostics; no source plan, gallery, or quality certificate.

The first baseline measures geometry and anonymous appearance consistently on
any encoded movie. It is NOT the learned, held-out action critic. In particular
SFace cosine is not a member probability and image structure is not body pose.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from time import monotonic

import cv2
import numpy as np


def signature(pixels):
    small=cv2.resize(pixels,(160,90),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY).astype(np.float32)
    centered=(gray-gray.mean())/max(5.,float(gray.std()))
    structure=cv2.resize(centered,(32,18),interpolation=cv2.INTER_AREA).ravel()
    norm=float(np.linalg.norm(structure))
    if norm>1e-8:structure/=norm
    lab=cv2.cvtColor(small,cv2.COLOR_BGR2LAB)
    hist=np.concatenate([np.histogram(lab[:,:,i],bins=16,range=(0,256))[0] for i in range(3)]).astype(float)
    hist/=hist.sum()
    return structure,hist


def cut_strength(left,right):
    return float(np.sqrt(np.maximum(0.,1.-np.sqrt(left[1]*right[1]).sum())))


def looks_like_flash(before,after,*,return_similarity=.93):
    # A recovered spatial image across a brief photometric event is evidence
    # against a new shot, not a promise of TransNet-level cut recall.
    a,b=before[0],after[0]
    return bool(np.linalg.norm(a)>.5 and np.linalg.norm(b)>.5 and float(a@b)>=return_similarity)


def classify(faces):
    if not faces:return 'unobserved'
    heights=sorted((f['height'] for f in faces),reverse=True)
    if len(heights)>=3 and heights[0]<1.6*heights[1]:return 'formation'
    return 'close' if heights[0]>=.12 else 'medium' if heights[0]>=.06 else 'wide'


def pair_features(left,right,*,appearance_cosine=None):
    """Pure numeric comparison; missing measurements are not perfect matches."""
    a,b=classify(left),classify(right)
    result=dict(left_shot=a,right_shot=b,left_people=len(left),right_people=len(right),
        role='formation' if a==b=='formation' else 'focal',appearance_cosine=appearance_cosine,
        named_member_identity_measured=False,body_action_measured=False)
    if not left or not right:
        return dict(result,geometry_measured=False,heuristic_discontinuity=None)
    la,lb=max(left,key=lambda f:f['height']),max(right,key=lambda f:f['height'])
    result.update(face_height_ratio=max(la['height'],lb['height'])/min(la['height'],lb['height']),
        face_center_displacement=math.hypot(la['center'][0]-lb['center'][0],
            (la['center'][1]-lb['center'][1])*9/16),
        close_to_extreme_wide=max(la['height'],lb['height'])>=.12 and min(la['height'],lb['height'])<.045)
    if result['role']=='formation':
        ca=np.asarray([f['center'] for f in sorted(left,key=lambda f:f['center'][0])])
        cb=np.asarray([f['center'] for f in sorted(right,key=lambda f:f['center'][0])])
        ha,hb=median(f['height'] for f in left),median(f['height'] for f in right)
        xa,xb=ca.mean(0),cb.mean(0)
        result.update(group_center_displacement=float(np.linalg.norm((xa-xb)*[1.,9/16])),
            group_scale_log_jump=abs(math.log(ha/hb)))
        if len(ca)==len(cb):
            # Centered normalized shape is anonymous formation geometry, not
            # proof that x-sorted people are the same named performers.
            residual=float(np.mean(np.linalg.norm(((ca-xa)/ha-(cb-xb)/hb)*[1.,9/16],axis=1)))
        else:residual=None
        result['anonymous_group_layout_residual']=residual
        score=result['group_center_displacement']/.08+result['group_scale_log_jump']/.3
        score+=abs(len(left)-len(right))*.4+(0. if residual is None else min(4.,residual))
    else:
        score=result['face_center_displacement']/.08+math.log(result['face_height_ratio'])/.3
        score+=3. if result['close_to_extreme_wide'] else 0.
    return dict(result,geometry_measured=True,heuristic_discontinuity=score,
        heuristic_score_is_not_calibrated_quality=True)


def _identity(path):
    s=path.stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns


def scan_cuts(path,*,check,threshold=.40,detector='adaptive'):
    cap=cv2.VideoCapture(str(path));recent=deque(maxlen=7);candidates=[];tick=0
    if not cap.isOpened():raise OSError('movie decoder unavailable')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps<=0:raise ValueError('movie clock unavailable')
    expected_count=round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if detector not in ('adaptive','histogram'):raise ValueError('explicit output cut detector required')
    adaptive=None
    if detector=='adaptive':
        from scenedetect import FrameTimecode
        from scenedetect.detectors import AdaptiveDetector
        # Keep short-shot mistakes observable rather than hiding them with a
        # half-second minimum. Other values are the upstream fixed defaults.
        adaptive=AdaptiveDetector(adaptive_threshold=3.,min_scene_len=5,
            window_width=2,min_content_val=15.)
    last=None
    try:
        while True:
            check()
            ok,pixels=cap.read()
            if not ok:break
            sig=signature(pixels)
            score=0. if last is None else cut_strength(last,sig)
            if adaptive is None:
                if score>=threshold:candidates.append([tick,score])
            else:
                incoming=adaptive.process_frame(FrameTimecode(tick,fps=fps),
                    cv2.resize(pixels,(320,180),interpolation=cv2.INTER_AREA))
                candidates.extend([cut.frame_num,1.] for cut in incoming)
            recent.append((tick,sig))
            if len(recent)==7:
                first,final=recent[0],recent[-1]
                if looks_like_flash(first[1],final[1]):
                    for row in candidates[-6:]:
                        if first[0]<row[0]<=final[0]:row[1]=-1.
            last=sig;tick+=1
            if tick%600==0:print(json.dumps(dict(phase='cut_scan',movie=path.name,decoded_frames=tick)),flush=True)
    finally:cap.release()
    cuts=[]
    for frame,score in candidates:
        if score<(threshold if adaptive is None else 0.):continue
        if cuts and frame-cuts[-1][0]<=3:
            if score>cuts[-1][1]:cuts[-1]=(frame,score)
        else:cuts.append((frame,score))
    if tick<2:raise ValueError('movie has no decoded frame sequence')
    if expected_count>0 and tick!=expected_count:
        raise OSError('movie decoder did not reach the complete encoded frame count')
    return fps,tick,tuple(frame for frame,_ in cuts)


class PixelFaces:
    def __init__(self,detector_path,appearance_path):
        self.detector=cv2.FaceDetectorYN.create(str(detector_path),'',(960,540),.75,.3,5000)
        self.appearance=cv2.FaceRecognizerSF.create(str(appearance_path),'')

    def read(self,pixels,*,expected_focal=None):
        h,w=pixels.shape[:2];image=cv2.resize(pixels,(960,540),interpolation=cv2.INTER_AREA)
        _,found=self.detector.detect(image);result=[];raw=[];dominant_vector=None
        if found is None:found=[]
        for face in sorted(found,key=lambda f:float(f[3]),reverse=True)[:8]:
            if face[3]/540<.02:continue
            result.append(dict(height=float(face[3]/540),
                center=(float((face[0]+face[2]/2)/960),float((face[1]+face[3]/2)/540)),
                confidence=float(face[-1])))
            raw.append(face)
        chosen=0 if result else None
        self.last_focal_receipt=dict(policy='largest_detected_face_output_only',candidates=list(result),
            selected_output_face_index=chosen,matched=False,ownership_certified=False)
        if expected_focal is not None:
            from .focal_output_correspondence import match
            chosen,self.last_focal_receipt=match(result,expected_focal)
        if chosen is not None and raw[chosen][3]*h/540>=60:
            native=raw[chosen].copy();native[:14:2]*=w/960;native[1:14:2]*=h/540
            aligned=self.appearance.alignCrop(pixels,native)
            vector=self.appearance.feature(aligned).ravel().astype(float)
            norm=float(np.linalg.norm(vector))
            if math.isfinite(norm) and norm>1e-9:dominant_vector=vector/norm
        selected=tuple(result) if expected_focal is None else (() if chosen is None else (result[chosen],))
        return selected,dominant_vector


def evaluate(path,*,detector_path,appearance_path,maximum_seconds=240.,threshold=.40,cut_detector='adaptive'):
    path=Path(path);binding=_identity(path);started=monotonic()
    def check():
        if monotonic()-started>=maximum_seconds:raise TimeoutError('single output-critic deadline reached')
    fps,count,cuts=scan_cuts(path,check=check,threshold=threshold,detector=cut_detector)
    if len(cuts)>512:raise ValueError('cut census exceeds the declared finite observation bound')
    model=PixelFaces(detector_path,appearance_path);cap=cv2.VideoCapture(str(path));rows=[]
    def at(tick):
        check()
        if not cap.set(cv2.CAP_PROP_POS_FRAMES,tick):raise OSError('movie ordinal seek unavailable')
        ok,pixels=cap.read()
        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1!=tick:
            raise OSError('movie ordinal decoder returned another frame')
        return model.read(pixels)
    try:
        for frame in cuts:
            left,lv=at(max(0,frame-2));right,rv=at(min(count-1,frame+2))
            similarity=None if lv is None or rv is None else float(np.clip(lv@rv,-1.,1.))
            rows.append(dict(frame=frame,seconds=frame/fps,left_faces=left,right_faces=right,
                **pair_features(left,right,appearance_cosine=similarity)))
    finally:cap.release()
    if binding!=_identity(path):raise OSError('movie changed during pixel evaluation')
    lengths=[(b-a)/fps for a,b in zip((0,*cuts),(*cuts,count),strict=True)]
    measured=[r for r in rows if r['geometry_measured']]
    worst=sorted(measured,key=lambda r:(-r['heuristic_discontinuity'],r['frame']))[:10]
    return dict(contract='output_only_geometry_critic_baseline_v1',movie=path.name,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_file_identity=binding,
        model_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (detector_path,appearance_path)},
        output_frame_count=count,fps=fps,duration_seconds=count/fps,detected_cut_count=len(cuts),
        cuts_per_minute=len(cuts)*60*fps/count,median_shot_seconds=median(lengths),
        shots_under_half_second=sum(t<.5 for t in lengths),shots_under_one_second=sum(t<1 for t in lengths),
        measured_geometry_cuts=len(measured),unobserved_geometry_cuts=len(rows)-len(measured),
        formation_cuts=sum(r['role']=='formation' for r in measured),
        close_to_extreme_wide_cuts=sum(r.get('close_to_extreme_wide',False) for r in measured),
        median_face_size_ratio=None if not measured else median(r['face_height_ratio'] for r in measured),
        median_face_center_displacement=None if not measured else median(r['face_center_displacement'] for r in measured),
        worst_cuts=worst,cuts=rows,cut_detector=cut_detector+'-with-structure-flash-suppression',
        cut_detector_parameters=(dict(adaptive_threshold=3.,min_scene_len_frames=5,
            window_width=2,min_content_val=15.) if cut_detector=='adaptive' else dict(threshold=threshold)),
        cut_detection_recall_validated=False,private_gallery_accessed=False,private_vectors_serialized=False,
        body_action_measured=False,learned_quality_critic=False,held_out_quality_validated=False,
        professional_quality_approved=False,normal_speed_audio_playback_performed=False,
        elapsed_seconds=monotonic()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('movie',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--maximum-seconds',type=float,default=240.)
    parser.add_argument('--cut-threshold',type=float,default=.40)
    parser.add_argument('--cut-detector',choices=('adaptive','histogram'),default='adaptive')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('do not overwrite a completed evaluation')
    if not math.isfinite(args.maximum_seconds) or not 1<=args.maximum_seconds<=600:
        raise ValueError('one finite output evaluation time allowance required')
    if not .05<=args.cut_threshold<=.9:raise ValueError('explicit finite cut threshold required')
    cv2.setNumThreads(2)
    result=evaluate(args.movie,detector_path=args.models/'face_detection_yunet_2023mar.onnx',
        appearance_path=args.models/'face_recognition_sface_2021dec.onnx',
        maximum_seconds=args.maximum_seconds,threshold=args.cut_threshold,cut_detector=args.cut_detector)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cuts','worst_cuts','model_sha256')}),flush=True)


if __name__=='__main__':main()
