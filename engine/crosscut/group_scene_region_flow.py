"""Nonrigid native ROI-point distributions, not articulated body paths.

Every original detector rectangle remains in the measurement. Features are
tracked on every intervening native raster without a whole-body rigid fit.
Their median translation supplies only a CONDITIONAL detector-ROI association;
it does not measure a body outline, identify a performer or certify an arm.
Independent parent/final must still reobserve original pixels and detections.
"""
from hashlib import sha256
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .cut_anchored_containment import _iou

CONTRACT='native_detector_roi_nonrigid_feature_distribution_v1'
POLICY=dict(maximum_native_gap=12,maximum_regions=12,maximum_features_per_region=80,
    minimum_seed_features=12,minimum_continuing_features=10,
    quality_level=.012,minimum_feature_distance=4.,block_size=5,
    lk_window=[21,21],lk_maximum_level=3,maximum_forward_backward_pixels=1.25,
    minimum_translated_roi_overlap=.60,minimum_terminal_roi_point_fraction=.60,
    minimum_roi_overlap_uniqueness_margin=.10,minimum_observed_roi_pairs=2,
    camera_or_anatomical_quality_calibrated=False)


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def policy_sha256():
    return digest(dict(policy=POLICY,opencv_version=cv2.__version__,
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))


def _point(value,*,allow_none=False):
    if value is None and allow_none:return None
    if (not isinstance(value,list) or len(value)!=2 or any(type(v) not in (int,float)
            or not math.isfinite(v) for v in value)):
        raise ValueError('region flow coordinates must be actual finite numeric points')
    return np.asarray(value,float)


def _step_summary(step,*,previous_points,size):
    attempts=step.get('attempts')
    if (not isinstance(attempts,list) or [a.get('feature_id') for a in attempts]!=list(previous_points)):
        raise ValueError('region flow omitted, reordered or revived a native feature')
    usable={}
    for attempt in attempts:
        fid=attempt['feature_id'];start=_point(attempt.get('source_feature_xy'))
        if not np.array_equal(start,previous_points[fid]):
            raise ValueError('region flow feature lost its exact adjacent native source occurrence')
        target=_point(attempt.get('tracked_target_xy'),allow_none=True)
        back=_point(attempt.get('backward_feature_xy'),allow_none=True)
        flags=(attempt.get('forward_lk_succeeded'),attempt.get('backward_lk_succeeded'))
        if any(type(v) is not bool for v in flags):
            raise ValueError('region flow omitted measured forward/backward status')
        good=all(flags) and target is not None and back is not None
        good=bool(good and np.linalg.norm(back-start)<=POLICY['maximum_forward_backward_pixels']
            and np.all(target>=0) and np.all(target<size))
        if good:usable[fid]=target
    return usable


def _track_summary(track,*,size,gap):
    seeds=track.get('seed_points_xy');steps=track.get('steps');box=track.get('original_box_xywh')
    if (not isinstance(seeds,list) or len(seeds)>POLICY['maximum_features_per_region']
            or not isinstance(steps,list) or len(steps)>gap
            or not isinstance(box,list) or len(box)!=4
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in box)):
        raise ValueError('region distribution lost bounded original seed/ROI/native steps')
    x,y,w,h=box
    if min(x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001:
        raise ValueError('region distribution original ROI is outside its native raster')
    low=np.floor(np.asarray((x,y))*size);high=np.ceil(np.asarray((x+w,y+h))*size)
    points={i:_point(p) for i,p in enumerate(seeds)}
    if (len({tuple(p) for p in seeds})!=len(seeds)
            or any(np.any(p<low) or np.any(p>=high) or np.any(p<0) or np.any(p>=size) for p in points.values())):
        raise ValueError('region distribution features escaped or duplicated their actual detector ROI')
    previous=points
    if len(seeds)<POLICY['minimum_seed_features']:
        if steps:raise ValueError('insufficient native texture cannot invent a flow history')
        reason='insufficient_roi_texture';usable=False
    else:
        usable=True;reason='conditional_nonrigid_roi_points_not_body_extent'
        for index,step in enumerate(steps):
            if step.get('native_step_offset')!=index:
                raise ValueError('region distribution skipped an intervening native raster')
            current=_step_summary(step,previous_points=previous,size=size)
            if step.get('usable_feature_ids')!=list(current):
                raise ValueError('region distribution changed its actual continuing image support')
            previous=current
            if len(current)<POLICY['minimum_continuing_features']:
                if index!=len(steps)-1:raise ValueError('unknown region motion cannot revive later features')
                usable=False;reason='insufficient_continuing_roi_points';break
        if len(steps)!=gap:usable=False;reason='incomplete_native_region_flow'
    if not usable:
        return dict(usable=False,reason=reason,terminal_feature_ids=list(previous),
            median_translation=None,displacement_quantiles=None,predicted_translation_only_roi=None)
    ids=list(previous)
    shifts=np.asarray([previous[i]-points[i] for i in ids])/size
    median=np.median(shifts,axis=0)
    return dict(usable=True,reason=reason,terminal_feature_ids=ids,median_translation=median.tolist(),
        displacement_quantiles=np.quantile(shifts,(.1,.5,.9),axis=0).tolist(),
        predicted_translation_only_roi=[x+float(median[0]),y+float(median[1]),w,h])


def _associations(tracks,previous,current,size):
    people=current['people'];count=len(people)
    overlaps=[];fractions=[]
    for track in tracks:
        row=[];support=[]
        terminal={a['feature_id']:a['tracked_target_xy'] for a in track['steps'][-1]['attempts']} if track['steps'] else {}
        for person in people:
            if not track['usable']:row.append(0.);support.append(0.);continue
            row.append(_iou(track['predicted_translation_only_roi'],person['box_xywh']))
            points=np.asarray([terminal[i] for i in track['terminal_feature_ids']])/size
            x,y,w,h=person['box_xywh']
            support.append(float(np.mean((points[:,0]>=x)&(points[:,0]<=x+w)&
                (points[:,1]>=y)&(points[:,1]<=y+h))))
        overlaps.append(row);fractions.append(support)
    pairs=[]
    if tracks and count:
        matrix=np.asarray(overlaps);aa,bb=linear_sum_assignment(1.-matrix)
        for a,b in zip(aa,bb):
            score=matrix[a,b]
            rivals=[matrix[a,j] for j in range(count) if j!=b]
            rivals.extend(matrix[i,b] for i in range(len(tracks)) if i!=a)
            if (score>=POLICY['minimum_translated_roi_overlap']
                    and fractions[a][b]>=POLICY['minimum_terminal_roi_point_fraction']
                    and all(score-v>=POLICY['minimum_roi_overlap_uniqueness_margin'] for v in rivals)):
                pairs.append([previous['people'][a]['observation_id'],people[b]['observation_id']])
    ids=[[p['observation_id'] for p in row['people']] for row in (previous,current)]
    matched=[[p[i] for p in pairs] for i in (0,1)]
    return dict(conditional_roi_overlaps=overlaps,terminal_roi_point_fraction=fractions,assignments=pairs,
        observed_body_observation_ids=matched,
        unmatched_body_observation_ids=[[i for i in ids[side] if i not in matched[side]] for side in (0,1)],
        observed_support_paths=len(pairs)>=POLICY['minimum_observed_roi_pairs'],
        complete_support_paths=len(pairs)==len(ids[0])==len(ids[1]) and len(pairs)>=2)


def _binding(previous,current):
    if (any(previous[k]!=current[k] for k in ('request_source_index','source_clock_identity',
            'source_file_identity','frame_size_wh')) or not 0<current['native_frame_index']-previous['native_frame_index']
            <=POLICY['maximum_native_gap'] or max(len(previous['people']),len(current['people']))>POLICY['maximum_regions']):
        raise ValueError('region distribution requires one bounded same-source native clock/raster domain')
    return dict(contract=CONTRACT,policy_sha256=policy_sha256(),
        raw_observation_sha256=[digest(previous),digest(current)],
        native_frame_indices=[previous['native_frame_index'],current['native_frame_index']],
        request_source_index=previous['request_source_index'],frame_size_wh=previous['frame_size_wh'],
        required_body_observation_ids=[[p['observation_id'] for p in r['people']] for r in (previous,current)],
        support_scope='conditional_detector_roi_points_not_semantic_cast',
        articulated_motion_certified=False,rigid_body_transform_certified=False,anatomical_extent_certified=False,
        named_identity_certified=False,semantic_cast_membership_certified=False,body_phase_certified=False,
        quality_approved=False)


def measure_region_distribution(sensors,previous,current):
    before,after=(json.loads(json.dumps(row.to_dict(),allow_nan=False)) for row in (previous,current))
    value=_binding(before,after)
    key=(CONTRACT,previous.sha256,current.sha256,policy_sha256())
    if key in sensors.link_cache:return sensors.link_cache[key]
    value.update(native_pixel_sha256=[],region_tracks=[],reason=None,observed_support_paths=False,
        complete_support_paths=False,assignments=[],observed_body_observation_ids=[[],[]],
        unmatched_body_observation_ids=value['required_body_observation_ids'])
    def finish(reason):
        value['reason']=reason
        sensors.session.budget.spend('pose_evidence_bytes',len(json.dumps(value,sort_keys=True,allow_nan=False).encode()))
        sensors.link_cache[key]=value
        return value
    if min(len(previous.people),len(current.people))<2:return finish('fewer_than_two_original_detector_regions')
    source=previous.request_source_index;pixels=sensors.session.native_pixels(source,previous.native_frame_index)
    if pixels is None:return finish('native_region_pixels_unavailable')
    value['native_pixel_sha256'].append(sha256(memoryview(pixels)).hexdigest())
    if value['native_pixel_sha256'][0]!=previous.native_pixel_sha256:
        raise ValueError('region source raster differs from actual native model observation')
    gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY);del pixels
    size=np.asarray((gray.shape[1],gray.shape[0]),float);tracks=[];live=[]
    for person in previous.people:
        x,y,w,h=person.box_xywh;mask=np.zeros_like(gray)
        mask[max(0,math.floor(y*size[1])):min(gray.shape[0],math.ceil((y+h)*size[1])),
            max(0,math.floor(x*size[0])):min(gray.shape[1],math.ceil((x+w)*size[0]))]=255
        sensors.session.budget.spend('pose_comparisons')
        points=cv2.goodFeaturesToTrack(gray,maxCorners=POLICY['maximum_features_per_region'],
            qualityLevel=POLICY['quality_level'],minDistance=POLICY['minimum_feature_distance'],mask=mask,blockSize=5)
        seeds=[] if points is None else points[:,0].tolist()
        tracks.append(dict(previous_body_observation_id=person.observation_id,
            original_box_xywh=list(person.box_xywh),seed_points_xy=seeds,steps=[]))
        live.append({i:np.asarray(p) for i,p in enumerate(seeds)} if len(seeds)>=12 else {})
    opts=dict(winSize=tuple(POLICY['lk_window']),maxLevel=POLICY['lk_maximum_level'],
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for offset,native in enumerate(range(previous.native_frame_index+1,current.native_frame_index+1)):
        pixels=sensors.session.native_pixels(source,native)
        if pixels is None:value['region_tracks']=tracks;return finish('intermediate_native_region_pixels_unavailable')
        value['native_pixel_sha256'].append(sha256(memoryview(pixels)).hexdigest())
        target_gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY);del pixels
        if target_gray.shape!=gray.shape:raise ValueError('native region raster dimensions changed')
        for i,track in enumerate(tracks):
            if len(live[i])<10:continue
            ids=list(live[i]);points=np.asarray(list(live[i].values()),np.float32).reshape(-1,1,2)
            # Feature updates are pixel primitives, not pose/ROI comparisons.
            # Both actual LK directions are charged BEFORE either is invoked.
            sensors.session.budget.spend('region_flow_feature_updates',2*len(ids))
            forward,fs,_=cv2.calcOpticalFlowPyrLK(gray,target_gray,points,None,**opts)
            backward,bs,_=(cv2.calcOpticalFlowPyrLK(target_gray,gray,forward,None,**opts)
                if forward is not None and np.isfinite(forward).all() else (None,None,None))
            attempts=[]
            for j,fid in enumerate(ids):
                finite=lambda data:data is not None and np.isfinite(data[j,0]).all()
                attempts.append(dict(feature_id=fid,source_feature_xy=live[i][fid].tolist(),
                    tracked_target_xy=forward[j,0].tolist() if finite(forward) else None,
                    backward_feature_xy=backward[j,0].tolist() if finite(backward) else None,
                    forward_lk_succeeded=bool(fs is not None and fs[j,0] and finite(forward)),
                    backward_lk_succeeded=bool(bs is not None and bs[j,0] and finite(backward))))
            step=dict(native_step_offset=offset,attempts=attempts)
            current_points=_step_summary(step,previous_points=live[i],size=size)
            step['usable_feature_ids']=list(current_points);track['steps'].append(step);live[i]=current_points
        gray=target_gray
    if value['native_pixel_sha256'][-1]!=current.native_pixel_sha256:
        raise ValueError('region target raster differs from actual native model observation')
    gap=current.native_frame_index-previous.native_frame_index
    for track in tracks:track.update(_track_summary(track,size=size,gap=gap))
    value['region_tracks']=tracks
    sensors.session.budget.spend('pose_comparisons',len(tracks)*len(current.people))
    value.update(_associations(tracks,before,after,size))
    return finish('conditional_nonrigid_roi_distribution_not_articulation_or_identity')


def validate_region_distribution(value,previous,current):
    """Recompute received algebra; it cannot authenticate pixel/model history."""
    previous,current=(json.loads(json.dumps(row,allow_nan=False)) for row in (previous,current))
    binding=_binding(previous,current);digest(value)
    if (any(value.get(k)!=v for k,v in binding.items()) or value.get('observed_support_paths') is not True
            or value.get('reason')!='conditional_nonrigid_roi_distribution_not_articulation_or_identity'):
        raise ValueError('region distribution changed its source/policy or promoted unknown authority')
    gap=current['native_frame_index']-previous['native_frame_index'];hashes=value.get('native_pixel_sha256')
    if (not isinstance(hashes,list) or len(hashes)!=gap+1 or hashes[0]!=previous['native_pixel_sha256']
            or hashes[-1]!=current['native_pixel_sha256'] or any(not isinstance(h,str) or len(h)!=64 for h in hashes)):
        raise ValueError('region distribution lost every actual intervening native raster')
    tracks=value.get('region_tracks');people=previous['people'];size=np.asarray(previous['frame_size_wh'],float)
    if (not isinstance(tracks,list) or len(tracks)!=len(people)
            or [t.get('previous_body_observation_id') for t in tracks]!=[p['observation_id'] for p in people]
            or [t.get('original_box_xywh') for t in tracks]!=[p['box_xywh'] for p in people]):
        raise ValueError('region distribution omitted or substituted an original detector rectangle')
    for track in tracks:
        summary=_track_summary(track,size=size,gap=gap)
        if any(track.get(k)!=v for k,v in summary.items()):
            raise ValueError('region distribution changed its measured flow summary')
    expected=_associations(tracks,previous,current,size)
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('region distribution altered its unique supported ROI assignment')
