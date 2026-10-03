"""Source-local head paths through actual native rasters, never actor identity.

A direct path is tried first. If it has no unique link, bounded adjacent-native
steps may recover a fast gesture. Every step retains the same original-pixel
FB limit and the final independently detected landmarks must still agree.
Raw LK returns are pre-reserved in the LOCAL primitive journal. This is not
controller-owned history, raw NN retention or a permanent raster archive.
"""
from hashlib import sha256

import cv2
import numpy as np

from . import group_performer_family_owner as family

CONTRACT='source_native_head_adjacent_pixel_paths_v1'
LABELS=('opencv_head_lk_forward','opencv_head_lk_backward')
POLICY=dict(contract='direct_then_all_head_adjacent_recovery_if_no_unique_v1',
    maximum_native_gap=12,maximum_heads=24,minimum_eye_span_pixels=4.,
    minimum_agreeing_landmarks=4,maximum_forward_backward_pixels=1.25,
    minimum_detector_tolerance_pixels=2.,detector_tolerance_eye_fraction=.25,
    window_size=[15,15],maximum_pyramid_level=4,maximum_iterations=30,epsilon=.01)
MAXIMUM_RETURN_BYTES=8192


def policy_sha256():return family.digest(POLICY)


def _raw(row):return family.plain(row.to_dict() if hasattr(row,'to_dict') else row)


def _hash(value):return sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _points(row):
    size=np.asarray(row['frame_size_wh'],np.float32)
    ids=[];points=[]
    if len(row['heads'])>POLICY['maximum_heads']:
        raise ValueError('head pixel path cannot truncate its native head inventory')
    for head in row['heads']:
        value=np.asarray(head['landmarks_xy'],np.float32)
        if (value.shape!=(5,2) or not np.isfinite(value).all() or
                (value<0).any() or (value>1).any()):continue
        value=value*size
        if np.linalg.norm(value[0]-value[1])<POLICY['minimum_eye_span_pixels']:continue
        ids.append(head['observation_id']);points.extend(value)
    return ids,np.asarray(points,np.float32).reshape(-1,1,2)


def _unique(pairs):
    return [pair for pair in pairs if sum(p[0]==pair[0] for p in pairs)==1 and
        sum(p[1]==pair[1] for p in pairs)==1]


def _match(a,b,ids,points,valid):
    if points is None or valid is None:return []
    size=np.asarray(a['frame_size_wh'],np.float32)
    original={head['observation_id']:head for head in a['heads']}
    pairs=[]
    for index,identity in enumerate(ids):
        before=np.asarray(original[identity]['landmarks_xy'],np.float32)*size
        predicted=points[index*5:(index+1)*5,0]
        supported=valid[index*5:(index+1)*5]
        for head in b['heads']:
            after=np.asarray(head['landmarks_xy'],np.float32)
            if (after.shape!=(5,2) or not np.isfinite(after).all() or
                    (after<0).any() or (after>1).any()):continue
            after=after*size
            span=min(np.linalg.norm(before[0]-before[1]),np.linalg.norm(after[0]-after[1]))
            if span<POLICY['minimum_eye_span_pixels']:continue
            tolerance=max(POLICY['minimum_detector_tolerance_pixels'],
                POLICY['detector_tolerance_eye_fraction']*span)
            agreement=supported&(np.linalg.norm(predicted-after,axis=1)<=tolerance)
            if int(agreement.sum())>=POLICY['minimum_agreeing_landmarks']:
                pairs.append([identity,head['observation_id']])
    return pairs


def _binding(a,b,*,mode,native_step,ids,point_sha,hashes,direction):
    return dict(raw_pair=[family.digest(a),family.digest(b)],
        source_index=a['request_source_index'],source_clock_identity=a['source_clock_identity'],
        source_file_identity=a['source_file_identity'],native_frames=[a['native_frame_index'],b['native_frame_index']],
        native_step=native_step,head_observation_ids=ids,source_points_sha256=point_sha,
        source_pixel_sha256=hashes[0 if direction=='forward' else 1],
        target_pixel_sha256=hashes[1 if direction=='forward' else 0],
        mode=mode,policy_sha256=policy_sha256())


def _arrays(result,count):
    if type(result) is not tuple or len(result)!=3:return None
    points,status,_=result
    if (not isinstance(points,np.ndarray) or points.shape!=(count,1,2)
            or points.dtype.kind!='f' or not np.isfinite(points).all()
            or not isinstance(status,np.ndarray) or status.shape!=(count,1)
            or status.dtype.kind not in 'bui'):return None
    return points,status[:,0].astype(bool)


def _valid_step(before,forward,backward,size):
    aa=_arrays(forward,len(before));bb=_arrays(backward,len(before))
    if aa is None or bb is None:return None,None
    after,fs=aa;back,bs=bb
    width,height=size
    valid=fs&bs&(np.linalg.norm(back[:,0]-before[:,0],axis=1)<=
        POLICY['maximum_forward_backward_pixels'])
    valid &= (after[:,0,0]>=0)&(after[:,0,0]<width)&(after[:,0,1]>=0)&(after[:,0,1]<height)
    return after,valid


def _execute_step(sensors,a,b,*,mode,native_step,ids,points,grays,hashes):
    budget=sensors.session.budget
    options=dict(winSize=tuple(POLICY['window_size']),maxLevel=POLICY['maximum_pyramid_level'],
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,
            POLICY['maximum_iterations'],POLICY['epsilon']))
    step=dict(mode=mode,native_step=list(native_step),source_pixel_sha256=hashes[0],
        target_pixel_sha256=hashes[1],forward_primitive_ref=None,backward_primitive_ref=None)
    def measure(direction,inputs,producer):
        sequence=len(budget.primitive_journal.attempts)
        result=budget.measure_primitive(label='opencv_head_lk_'+direction,
            input_binding=inputs,work=dict(region_flow_feature_updates=len(points)),
            maximum_result_bytes=MAXIMUM_RETURN_BYTES,producer=producer)
        step[direction+'_primitive_ref']=budget.primitive_journal.attempts[sequence]['measurement_ref_sha256']
        return result
    forward=measure('forward',_binding(a,b,mode=mode,native_step=list(native_step),ids=ids,
        point_sha=_hash(points),hashes=hashes,direction='forward'),
        lambda:cv2.calcOpticalFlowPyrLK(grays[0],grays[1],points,None,**options))
    parsed=_arrays(forward,len(points))
    if parsed is None:return step,None,None
    after=parsed[0]
    backward=measure('backward',_binding(a,b,mode=mode,native_step=list(native_step),ids=ids,
        point_sha=_hash(after),hashes=hashes,direction='backward'),
        lambda:cv2.calcOpticalFlowPyrLK(grays[1],grays[0],after,None,**options))
    after,valid=_valid_step(points,forward,backward,a['frame_size_wh'])
    return step,after,valid


def _base(a,b):
    return dict(contract=CONTRACT,raw_pair=[family.digest(a),family.digest(b)],
        native_frame_indices=[a['native_frame_index'],b['native_frame_index']],
        frame_size_wh=a['frame_size_wh'],policy_sha256=policy_sha256(),
        source_head_observation_ids=[head['observation_id'] for head in a['heads']],
        target_head_observation_ids=[head['observation_id'] for head in b['heads']],
        direct_steps=[],direct_measured_links=[],adjacent_steps=[],adjacent_measured_links=[],
        adjacent_recovery_requested=False,adjacent_recovery_complete=False,
        measured_links=[],unique_links=[],reason=None,repeated_native_alias=False,
        cross_camera_identity_certified=False,named_identity_certified=False,
        pre_receipt_history_completeness_certified=False,decoded_raster_bytes_retained=False,
        quality_approved=False)


def _check_pair(a,b):
    if (a['request_source_index']!=b['request_source_index'] or
            a['source_clock_identity']!=b['source_clock_identity'] or
            a['source_file_identity']!=b['source_file_identity'] or
            a['frame_size_wh']!=b['frame_size_wh'] or
            a['native_frame_index']>=b['native_frame_index'] or
            any(not _sha(row.get('native_pixel_sha256')) for row in (a,b))):
        raise ValueError('head path changed its exact native source/clock/file/raster ordering')


def measure_head_pixel_links(sensors,previous,current):
    a,b=_raw(previous),_raw(current);_check_pair(a,b)
    value=_base(a,b);first,last=value['native_frame_indices']
    if last-first>POLICY['maximum_native_gap']:
        value['reason']='native_head_gap_unrequested';return value
    domain=getattr(sensors.session,'domain',None)
    if domain is not None:
        clock=domain.clocks_by_request[a['request_source_index']]
        if family.digest(clock.to_dict())!=a['source_clock_identity']:
            raise ValueError('head path source clock differs from its current native domain')
        if any(first/float(clock.frame_rate)<cut<=last/float(clock.frame_rate)
                for cut in domain.cuts_for_request(a['request_source_index'])):
            value['reason']='native_broadcaster_cut';return value
    ids,points=_points(a)
    if not ids or not b['heads']:
        value['reason']='head_landmark_scope_unknown';return value
    def gray(native,expected=None):
        pixels=sensors.session.native_pixels(a['request_source_index'],native)
        if (not isinstance(pixels,np.ndarray) or pixels.dtype!=np.uint8 or
                pixels.shape!=(a['frame_size_wh'][1],a['frame_size_wh'][0],3)):
            raise ValueError('actual native head raster is missing or changed')
        identity=_hash(pixels)
        identities=getattr(sensors.session,'identities',None)
        if isinstance(identities,dict) and identities.get(a['request_source_index'])!=tuple(a['source_file_identity']):
            raise ValueError('head path original file identity differs from its current native decoder')
        if expected is not None and identity!=expected:
            raise ValueError('actual head endpoint raster differs from its native observation')
        return cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY),identity
    before,first_hash=gray(first,a['native_pixel_sha256'])
    after,last_hash=gray(last,b['native_pixel_sha256'])
    sensors.session.budget.spend('pose_comparisons',len(ids)*len(b['heads']))
    step,moved,valid=_execute_step(sensors,a,b,mode='direct',native_step=[first,last],
        ids=ids,points=points,grays=(before,after),hashes=(first_hash,last_hash))
    value['direct_steps'].append(step)
    value['direct_measured_links']=_match(a,b,ids,moved,valid)
    if not _unique(value['direct_measured_links']) and last-first>1:
        # An adaptive query is explicit. Never discard conflicting direct links
        # to make a newly convenient adjacent path unique.
        value['adjacent_recovery_requested']=True
        moving=points;alive=np.ones(len(points),dtype=bool)
        previous_gray,previous_hash=before,first_hash
        for native in range(first+1,last+1):
            current_gray,current_hash=(after,last_hash) if native==last else gray(native)
            step,moved,valid=_execute_step(sensors,a,b,mode='adjacent',native_step=[native-1,native],
                ids=ids,points=moving,grays=(previous_gray,current_gray),hashes=(previous_hash,current_hash))
            value['adjacent_steps'].append(step)
            if moved is None:break
            alive &= valid;moving=moved
            if native==last:
                value['adjacent_recovery_complete']=True
                sensors.session.budget.spend('pose_comparisons',len(ids)*len(b['heads']))
                value['adjacent_measured_links']=_match(a,b,ids,moving,alive)
            if not any(int(alive[i:i+5].sum())>=POLICY['minimum_agreeing_landmarks']
                    for i in range(0,len(alive),5)):break
            previous_gray,previous_hash=current_gray,current_hash
    value['measured_links']=sorted({tuple(pair) for pair in
        (*value['direct_measured_links'],*value['adjacent_measured_links'])})
    value['measured_links']=[list(pair) for pair in value['measured_links']]
    value['unique_links']=_unique(value['measured_links'])
    return value


def validate_head_primitive_inputs(binding,label,raw_inventory):
    if label not in LABELS or not isinstance(binding,dict):
        raise ValueError('head primitive label/input is absent')
    refs=binding.get('raw_pair')
    if not isinstance(refs,list) or len(refs)!=2 or any(ref not in raw_inventory for ref in refs):
        raise ValueError('head primitive borrowed another original native observation')
    a,b=(_raw(raw_inventory[ref]) for ref in refs);_check_pair(a,b)
    first,last=a['native_frame_index'],b['native_frame_index']
    native=binding.get('native_step');mode=binding.get('mode')
    if (not isinstance(native,list) or len(native)!=2 or any(type(n) is not int for n in native)
            or not first<=native[0]<native[1]<=last or last-first>POLICY['maximum_native_gap']
            or mode not in ('direct','adjacent') or mode=='direct' and native!=[first,last]
            or mode=='adjacent' and native[1]-native[0]!=1):
        raise ValueError('head primitive escaped its exact bounded native step')
    hashes=[binding.get('source_pixel_sha256'),binding.get('target_pixel_sha256')]
    if label==LABELS[1]:hashes.reverse()
    if (any(not _sha(ref) for ref in hashes) or not _sha(binding.get('source_points_sha256'))
            or native[0]==first and hashes[0]!=a['native_pixel_sha256']
            or native[1]==last and hashes[1]!=b['native_pixel_sha256']):
        raise ValueError('head primitive changed the bound original endpoint pixels')
    ids,_=_points(a)
    expected=_binding(a,b,mode=mode,native_step=native,ids=ids,
        point_sha=binding['source_points_sha256'],hashes=hashes,
        direction='forward' if label==LABELS[0] else 'backward')
    if binding!=expected:raise ValueError('head primitive changed its source/seed/policy input')


def validate_head_pixel_link(value,inventory,raw_inventory):
    """Reconstruct every received link from retained raw LK coordinate/status bits.

    Intermediate raster hashes bind local input records, not permanently stored
    raster bytes. Fresh selected/baseline pixel replay remains mandatory.
    """
    refs=value.get('raw_pair') if isinstance(value,dict) else None
    if not isinstance(refs,list) or len(refs)!=2 or any(ref not in raw_inventory for ref in refs):
        raise ValueError('head pixel view lost its original raw pair')
    a,b=(_raw(raw_inventory[ref]) for ref in refs);_check_pair(a,b)
    expected=_base(a,b)
    if set(value)!=set(expected):raise ValueError('head pixel view schema/authority changed')
    ids,points=_points(a);first,last=expected['native_frame_indices']
    reason=value.get('reason')
    if reason is not None:
        if (reason not in ('native_head_gap_unrequested','native_broadcaster_cut','head_landmark_scope_unknown')
                or reason=='native_head_gap_unrequested' and last-first<=POLICY['maximum_native_gap']
                or reason=='head_landmark_scope_unknown' and ids and b['heads']):
            raise ValueError('head pixel view invented its unresolved scope')
        expected['reason']=reason
        if value!=expected:raise ValueError('unrequested head view invented a pixel path')
        return value
    if not ids or not b['heads'] or last-first>POLICY['maximum_native_gap']:
        raise ValueError('head pixel view ignored an unsupported native scope')
    last_sequence=-1
    def replay(step,mode,interval,moving):
        nonlocal last_sequence
        if (not isinstance(step,dict) or set(step)!={'mode','native_step','source_pixel_sha256',
                'target_pixel_sha256','forward_primitive_ref','backward_primitive_ref'}
                or step['mode']!=mode or step['native_step']!=interval):
            raise ValueError('head view skipped/replaced an actual native step')
        hashes=[step['source_pixel_sha256'],step['target_pixel_sha256']]
        forward,binding,sequence=inventory.returned(step['forward_primitive_ref'],LABELS[0],after_sequence=last_sequence)
        validate_head_primitive_inputs(binding,LABELS[0],raw_inventory)
        if binding!=_binding(a,b,mode=mode,native_step=interval,ids=ids,
                point_sha=_hash(moving),hashes=hashes,direction='forward'):
            raise ValueError('head forward view borrowed another actual seed or raster')
        last_sequence=sequence
        parsed=_arrays(forward,len(moving))
        if parsed is None:
            if step['backward_primitive_ref'] is not None:
                raise ValueError('invalid head forward return invented backward work')
            return None,None
        backward,binding,sequence=inventory.returned(step['backward_primitive_ref'],LABELS[1],after_sequence=sequence)
        validate_head_primitive_inputs(binding,LABELS[1],raw_inventory)
        if binding!=_binding(a,b,mode=mode,native_step=interval,ids=ids,
                point_sha=_hash(parsed[0]),hashes=hashes,direction='backward'):
            raise ValueError('head backward view borrowed another actual forward result')
        last_sequence=sequence
        return _valid_step(moving,forward,backward,a['frame_size_wh'])
    if len(value['direct_steps'])!=1:raise ValueError('head view lost its direct query')
    expected['direct_steps']=value['direct_steps']
    moved,valid=replay(value['direct_steps'][0],'direct',[first,last],points)
    expected['direct_measured_links']=_match(a,b,ids,moved,valid)
    recover=not _unique(expected['direct_measured_links']) and last-first>1
    expected['adjacent_recovery_requested']=recover
    if recover:
        moving=points;alive=np.ones(len(points),bool);previous_hash=a['native_pixel_sha256'];stopped=False
        steps=value['adjacent_steps']
        if not isinstance(steps,list) or not 1<=len(steps)<=last-first:
            raise ValueError('head recovery lost its bounded actual native steps')
        expected['adjacent_steps']=steps
        for index,step in enumerate(steps):
            native=first+index+1
            if stopped or step['source_pixel_sha256']!=previous_hash:
                raise ValueError('head recovery skipped a failed step or changed its native raster chain')
            moved,valid=replay(step,'adjacent',[native-1,native],moving)
            if moved is None:stopped=True;continue
            alive &= valid;moving=moved;previous_hash=step['target_pixel_sha256']
            if native==last:
                expected['adjacent_recovery_complete']=True
                expected['adjacent_measured_links']=_match(a,b,ids,moving,alive)
            if not any(int(alive[i:i+5].sum())>=POLICY['minimum_agreeing_landmarks']
                    for i in range(0,len(alive),5)):stopped=True
        if not stopped and len(steps)!=last-first:
            raise ValueError('head view erased a still-required native recovery step')
    expected['measured_links']=[list(pair) for pair in sorted({tuple(pair) for pair in
        (*expected['direct_measured_links'],*expected['adjacent_measured_links'])})]
    expected['unique_links']=_unique(expected['measured_links'])
    if value!=expected:raise ValueError('head link differs from raw primitives or promoted identity authority')
    return value
