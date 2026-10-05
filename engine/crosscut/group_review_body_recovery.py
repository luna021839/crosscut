"""Source-local, fixed-ROI auxiliary body measurements for GROUP reviews.

Duplicate body detections are kept, never NMS-merged into an owner. A new
top-down observation can supplement a selected single-face occurrence, with
the ROI frozen from the prior inventory before pose or match costs are seen.
It is not production person ownership, named identity, or quality approval.
"""
from dataclasses import dataclass, replace
import json
import math
import numpy as np

from .group_performer_family_owner import digest
from .body_pose_2d import RTMPOSE_BODY7_MODEL_SHA256


def head_fingerprint(head):
    return digest({k:head[k] for k in ('box_xywh','landmarks_xy','confidence','extent_basis')})


def native_binding(raw):
    return dict(source=raw['request_source_index'],native=raw['native_frame_index'],
        clock=raw['source_clock_identity'],file=list(raw['source_file_identity']),
        pixel=raw['native_pixel_sha256'],size=list(raw['frame_size_wh']))


@dataclass(frozen=True,slots=True)
class FrozenReviewBodyWindow:
    token_json:str

    def __post_init__(self):
        value=json.loads(self.token_json);requests=value.get('requests',())
        if (value.get('contract')!='frozen_prior_inventory_body_review_window_v1'
                or not 2<=len(requests)<=16 or value.get('roi_selection_inspected_new_pose') is not False
                or any(len(v.get('roi',()))!=4 or min(v['roi'][2:])<=0
                    or any(not math.isfinite(x) for x in v['roi']) for v in requests)
                or any(a['binding']['native']>=b['binding']['native'] for a,b in zip(requests,requests[1:]))
                or any(b['binding']['native']-a['binding']['native']>12 for a,b in zip(requests,requests[1:]))
                or len({(v['binding']['source'],v['binding']['clock'],tuple(v['binding']['file']))
                    for v in requests})!=1):
            raise ValueError('fixed body ROI needs exact bounded source-local prior requests')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def freeze_review_body_window(raws,*,native_camera_unit,head_ids=None,minimum_native_head_pixels=None):
    """No candidate error, pose success, or editorial crop selects this ROI."""
    if (not isinstance(raws,(list,tuple)) or not 2<=len(raws)<=16
            or type(native_camera_unit) is not int or native_camera_unit<0):
        raise ValueError('auxiliary review body scope is an actual single-face native close-up')
    explicit=head_ids is not None
    if minimum_native_head_pixels is not None and (not explicit
            or type(minimum_native_head_pixels) not in (int,float)
            or not math.isfinite(minimum_native_head_pixels) or minimum_native_head_pixels<32):
        raise ValueError('pixel-scoped auxiliary body needs an explicit prior actor and at least 32 native face pixels')
    if not explicit and any(len(raw['heads'])!=1 for raw in raws):
        raise ValueError('auxiliary review body scope is an actual single-face native close-up')
    if explicit and (not isinstance(head_ids,(list,tuple)) or len(head_ids)!=len(raws)
            or any(type(i) is not int or i<0 for i in head_ids)):
        raise ValueError('explicit auxiliary body needs an independently selected occurrence at every request')
    heads=[]
    for index,raw in enumerate(raws):
        selected=([h for h in raw['heads'] if h['observation_id']==head_ids[index]] if explicit else raw['heads'])
        normalized_small=len(selected)==1 and selected[0]['box_xywh'][3]<(0.05 if explicit else .13)
        if minimum_native_head_pixels is not None:
            size=raw.get('frame_size_wh')
            if (not isinstance(size,(list,tuple)) or len(size)!=2 or any(type(v) is not int or v<=0 for v in size)):
                raise ValueError('native pixel scope lost its actual raster dimensions')
            normalized_small=len(selected)==1 and selected[0]['box_xywh'][3]*size[1]<minimum_native_head_pixels
        if len(selected)!=1 or normalized_small:
            raise ValueError('selected auxiliary head is absent, repeated or below its declared measurement scope')
        heads.append(selected[0])
    boxes=[]
    for raw,head in zip(raws,heads,strict=True):
        ids=set(head['candidate_body_observation_ids'])
        boxes.extend(p['box_xywh'] for p in raw['people'] if p['observation_id'] in ids
            and p.get('face_anchor_consistent') is True and p.get('pose_reason')=='observed')
    if not boxes:raise ValueError('body window has no previously measured head-compatible ROI')
    boxes=np.asarray(boxes,float)
    center=np.median(boxes[:,:2]+boxes[:,2:]/2,axis=0);size=np.median(boxes[:,2:],axis=0)
    low=np.maximum(0.,center-size/2);high=np.minimum(1.,center+size/2)
    roi=tuple(map(float,np.r_[low,high-low]))
    requests=[dict(binding=native_binding(raw),raw_parent_sha256=digest(raw),
        head_fingerprint=head_fingerprint(head),head_id=head['observation_id'],roi=roi)
        for raw,head in zip(raws,heads,strict=True)]
    value=dict(contract='frozen_prior_inventory_body_review_window_v1',native_camera_unit=native_camera_unit,
        requests=requests,prior_body_roi_alternatives=boxes.tolist(),
        roi_derivation='coordinate_median_of_all_prior_head_compatible_boxes',
        roi_selection_inspected_new_pose=False,raw_detector_rows_rewritten=False,
        head_selection_scope='explicit_prior_selected_occurrences' if explicit else 'sole_native_face',
        body_ownership_certified=False,quality_approved=False)
    if minimum_native_head_pixels is not None:
        value.update(measurement_scope='explicit_actual_native_face_pixel_scope_v1',
            minimum_native_head_pixels=minimum_native_head_pixels,
            native_face_heights_pixels=[h['box_xywh'][3]*raw['frame_size_wh'][1] for h,raw in zip(heads,raws)])
    return FrozenReviewBodyWindow(json.dumps(value,sort_keys=True,allow_nan=False))


def plan_partial_review_body_windows(raws, *, requested_ticks, native_signatures,
        head_ids, minimum_native_head_pixels=32):
    """Measure available occurrences; a missing slot must not erase its peers.

    Split on the actual source/clock/file/camera run, not pose success. Unknown
    actors are not interpolated. Every originally requested slot is retained
    in the report, while only exact available native occurrences can be sent
    to the auxiliary model. This does not complete a motion comparison.
    """
    ticks=tuple(requested_ticks);signatures=tuple(native_signatures)
    if (not 2<=len(ticks)<=16 or len(raws)!=len(ticks) or len(head_ids)!=len(ticks)
            or len(signatures)!=len(ticks)
            or any(type(t) is not int or t<0 for t in ticks)
            or any(a>=b for a,b in zip(ticks,ticks[1:]))):
        raise ValueError('partial body scope needs its complete requested slot inventory')
    slots=[];runs=[];current=[];previous=None
    for tick,raw,head,signature in zip(ticks,raws,head_ids,signatures,strict=True):
        if signature is not None:
            if (len(signature)!=5 or type(signature[0]) is not int or signature[0]<0
                    or type(signature[3]) is not int or signature[3]<0
                    or type(signature[4]) is not int or signature[4]<0):
                raise ValueError('partial body scope lost its actual native sampler address')
            signature=tuple(signature)
        if raw is not None:
            expected=(raw['request_source_index'],raw['source_clock_identity'],
                digest(raw['source_file_identity']))
            if signature is None or signature[:3]!=expected or signature[4]!=raw['native_frame_index']:
                raise ValueError('partial body slot borrowed another source/clock/file/native frame')
        if head is not None and (raw is None or type(head) is not int or head<0):
            raise ValueError('partial body slot invented an unobserved selected actor')
        state='UNOBSERVED_NATIVE' if raw is None else 'UNRESOLVED_ACTOR' if head is None else 'AVAILABLE'
        slots.append(dict(tick=tick,state=state,native_signature=signature,head_id=head))
        # A known physical run may bridge a missing detector row; an unknown
        # sampler address or an actual native cut may not be bridged.
        connected=(signature is not None and previous is not None
            and signature[:4]==previous[:4] and 0<=signature[4]-previous[4]<=12)
        if not connected and current:
            runs.append(current);current=[]
        if signature is not None:current.append((tick,raw,head,signature))
        previous=signature
    if current:runs.append(current)
    windows=[];unavailable=[]
    for run in runs:
        available={}
        for tick,raw,head,signature in run:
            if raw is None or head is None:continue
            native=signature[4]
            if native in available:
                old_raw,old_head=available[native]
                if old_head!=head or digest(old_raw)!=digest(raw):
                    raise ValueError('partial body native aliases disagree about the actual actor or raster')
            else:available[native]=(raw,head)
        if len(available)<2:
            unavailable.append(dict(ticks=[v[0] for v in run],reason='fewer_than_two_available_native_occurrences'))
            continue
        pairs=[available[n] for n in sorted(available)]
        try:
            window=freeze_review_body_window([v[0] for v in pairs],
                native_camera_unit=run[0][3][3],head_ids=[v[1] for v in pairs],
                minimum_native_head_pixels=minimum_native_head_pixels)
        except ValueError as error:
            unavailable.append(dict(ticks=[v[0] for v in run],reason=str(error)));continue
        windows.append(window)
    return tuple(windows),dict(requested_slots=slots,physical_run_count=len(runs),
        available_native_window_count=len(windows),unavailable=unavailable,
        missing_or_ambiguous_slots=[v['tick'] for v in slots if v['state']!='AVAILABLE'],
        missing_actor_slots_interpolated=False,complete_motion_observation=False,
        quality_approved=False)


def bind_review_body_receipts(bundle):
    """Reject duplicate or foreign receipts; retain unsuccessful actual poses."""
    if (bundle.get('contract')!='bounded_actual_fixed_roi_body_review_receipts_v1'
            or not isinstance(bundle.get('windows'),list) or not isinstance(bundle.get('receipts'),list)
            or len(bundle['receipts'])>512):
        raise ValueError('body recovery needs the bounded independently frozen actual inventory')
    windows={}
    for row in bundle['windows']:
        window=FrozenReviewBodyWindow(json.dumps(row['token'],sort_keys=True,allow_nan=False))
        if row['identity']!=window.identity or window.identity in windows:
            raise ValueError('body window seal changed or repeated')
        windows[window.identity]=window
    receipts={}
    for receipt in bundle['receipts']:
        window=windows.get(receipt.get('window_identity'))
        if (window is None or receipt.get('model_content_sha256')!=RTMPOSE_BODY7_MODEL_SHA256
                or receipt.get('contract')!='actual_fixed_roi_body_review_receipt_v1'):
            raise ValueError('body receipt has no matching pre-inference window/model')
        request=next((r for r in window.payload['requests'] if r['binding']==receipt.get('binding')),None)
        if (request is None or request['head_fingerprint']!=receipt.get('head_fingerprint')
                or request['raw_parent_sha256']!=receipt.get('raw_parent_sha256')
                or tuple(request['roi'])!=tuple(receipt.get('pose',{}).get('bbox_xywh',()))):
            raise ValueError('body receipt changed its prior actor/native/ROI request')
        b=receipt['binding'];key=(b['source'],b['native'],b['clock'],tuple(b['file']),b['pixel'],receipt['head_fingerprint'])
        if key in receipts:raise ValueError('two body receipts compete for one exact selected raster')
        receipts[key]=(window,receipt)
    return receipts


def _primary_native_points(view):
    """Selected raw owned body only, never invert/merge normalized shapes."""
    d=view['descriptor'];raw=view['row']['native_observation']
    owners=[p for p in view['row'].get('projection',{}).get('people',())
        if p.get('unique_head_observation_id')==d.frame_local_head_id]
    if len(owners)!=1 or not d.owner_known:return {} if d.pose_shape is None else None
    body=next((p for p in raw['people'] if p['observation_id']==owners[0]['observation_id']),None)
    points=() if body is None else body.get('upper_body_xyc',())
    if len(points)!=8:return {} if d.pose_shape is None else None
    requested=set() if d.pose_shape is None else {p[0] for p in d.pose_shape}
    if any(not 0<=i<6 or len(points[i])!=3 or any(not math.isfinite(v) for v in points[i])
            or points[i][2]<.42 or not all(0<=v<1 for v in points[i][:2]) for i in requested):return None
    parts=dict(body.get('part_states',()))
    # Whole-arm UNKNOWN often means ONE joint was missed, not that every
    # other directly observed joint disappeared. Preserve explicit ambiguity
    # and adverse part states; a raw model's remaining selected-owner joints
    # can contribute individually without claiming complete arm ownership.
    observed={i:tuple(p) for i,p in enumerate(points[:6]) if len(p)==3
        and all(math.isfinite(v) for v in p) and p[2]>=.42 and all(0<=v<1 for v in p[:2])
        and parts.get('shoulder_anchor' if i<2 else 'left_arm' if i in (2,4) else 'right_arm')
            in ('observed','unknown')}
    if not requested<=observed.keys():return None
    return observed


def apply_review_primary_joint_observation(view,*,project_point):
    """Keep directly measured selected-owner joints without whole-arm gating.

    This does not infer a missing wrist or promote UNKNOWN ownership/support.
    No auxiliary receipt must first succeed merely to retain a primary elbow.
    """
    if view is None:return None
    primary=_primary_native_points(view)
    if primary is None or not {0,1}<=primary.keys():return view
    projected={i:(*project_point(p[:2]),p[2]) for i,p in primary.items()}
    if any(not all(0<=v<=1 for v in projected[i][:2]) for i in (0,1)):return view
    center=np.mean(np.asarray([projected[i][:2] for i in (0,1)]),axis=0)
    scale=float(np.linalg.norm((np.asarray(projected[0][:2])-projected[1][:2])*(1.,9/16)))
    if scale<=1e-6:return view
    shape=tuple((i,(p[0]-center[0])/scale,(p[1]-center[1])*9/16/scale)
        for i,p in sorted(projected.items()) if all(0<=v<=1 for v in p[:2]))
    known={} if view['descriptor'].pose_shape is None else {p[0]:np.asarray(p[1:]) for p in view['descriptor'].pose_shape}
    actual={p[0]:np.asarray(p[1:]) for p in shape}
    if any(i not in actual or np.linalg.norm(point-actual[i])>1e-6 for i,point in known.items()):return view
    raw=view['row']['native_observation'];head=next(h for h in raw['heads']
        if h['observation_id']==view['descriptor'].frame_local_head_id)
    provenance={i:dict(basis='PRIMARY_NATIVE_MODEL',raw_parent_sha256=digest(raw),
        model_confidence=p[2],xy_native=list(p[:2])) for i,p in primary.items()}
    identity=digest(dict(contract='selected_owner_individual_primary_joint_review_v1',
        binding=native_binding(raw),head=head_fingerprint(head),provenance=provenance,
        production_ownership_unchanged=True))
    return dict(view,descriptor=replace(view['descriptor'],pose_shape=shape),
        review_primary_joint_evidence_identity=identity,review_joint_evidence_identity=identity,
        review_joint_provenance=provenance,primary_whole_arm_ownership_unchanged=True)


def apply_review_body_receipt(view,receipt,*,project_point,expected_window):
    """Bind auxiliary evidence to this exact actor/raster/camera projection.

    Original conflicting observations are not discarded. A low-confidence
    wrist remains absent; no candidate-dependent intersection grants a pass.
    """
    if view is None:return None
    if not isinstance(expected_window,FrozenReviewBodyWindow):
        raise ValueError('auxiliary body consumer needs its independently frozen request')
    raw=view['row']['native_observation'];d=view['descriptor']
    result=dict(view)
    selected=next((h for h in raw['heads'] if h['observation_id']==d.frame_local_head_id),None)
    request=next((v for v in expected_window.payload['requests'] if v['binding']==native_binding(raw)),None)
    if (receipt.get('contract')!='actual_fixed_roi_body_review_receipt_v1' or request is None
            or receipt.get('window_identity')!=expected_window.identity
            or receipt.get('model_content_sha256')!=RTMPOSE_BODY7_MODEL_SHA256
            or receipt.get('binding')!=native_binding(raw) or selected is None
            or receipt.get('raw_parent_sha256')!=digest(raw)
            or request['head_id']!=selected['observation_id']
            or request['head_fingerprint']!=head_fingerprint(selected)
            or (len(raw['heads'])!=1 and expected_window.payload.get('head_selection_scope')
                !='explicit_prior_selected_occurrences')
            or receipt.get('head_fingerprint')!=head_fingerprint(selected)):
        raise ValueError('auxiliary body receipt borrowed another actor/native raster')
    result['review_body_receipt_sha256']=digest(receipt)
    result['auxiliary_body_is_production_ownership']=False
    pose=receipt['pose'];points=pose.get('upper_body_xyc',())
    if tuple(pose.get('bbox_xywh',()))!=tuple(request['roi']):
        raise ValueError('auxiliary body inference changed the predeclared ROI')
    raw_points=pose.get('raw_keypoints_xyc',())
    # BodyPose2DAnalyzer retains unclipped SimCC scores in raw17 but clips
    # the public eight-joint confidence to [0,1]. Coordinates are identical;
    # an actual model score over 1 must not be mistaken for fabricated joints.
    if len(raw_points)==17 and len(points)==8 and any(
            tuple(points[i])!=(raw_points[j][0],raw_points[j][1],
                max(0.,min(1.,raw_points[j][2]))) for i,j in enumerate((5,6,7,8,9,10,11,12))):
        raise ValueError('auxiliary upper-body evidence changed its actual RTMPose raw joint mapping')
    if (pose.get('reason')!='observed' or pose.get('face_anchor_consistent') is not True
            or len(points)!=8 or len(pose.get('raw_keypoints_xyc',()))!=17
            or any(len(p)!=3 or any(not math.isfinite(v) for v in p) or not 0<=p[2]<=1 for p in points)):
        result['review_body_recovery_state']='UNRESOLVED_ACTUAL_POSE';return result
    primary=_primary_native_points(view)
    if primary is None:
        result['review_body_recovery_state']='CONFLICTING_OR_MISSING_KNOWN_PRIMARY_BODY';return result
    combined=[tuple(p) for p in points]
    provenance={i:dict(basis='FIXED_ROI_NATIVE_MODEL',receipt_sha256=digest(receipt),raw_parent_sha256=digest(raw),
        model_confidence=p[2],xy_native=list(p[:2])) for i,p in enumerate(points[:6])
        if p[2]>=.42 and all(0<=v<1 for v in p[:2])}
    for i,p in primary.items():
        combined[i]=p
        provenance[i]=dict(basis='PRIMARY_NATIVE_MODEL',raw_parent_sha256=digest(raw),
            model_confidence=p[2],xy_native=list(p[:2]))
    if min(combined[0][2],combined[1][2])<.42:
        result['review_body_recovery_state']='UNRESOLVED_ACTUAL_POSE';return result
    projected=[(*project_point(p[:2]),p[2]) for p in combined]
    if any(not all(0.<=v<=1. for v in projected[i][:2]) for i in (0,1)):
        result['review_body_recovery_state']='SHOULDER_OUTSIDE_ACTUAL_CAMERA';return result
    center=np.mean(np.asarray(projected[:2])[:,:2],axis=0)
    scale=float(np.linalg.norm((np.asarray(projected[0][:2])-np.asarray(projected[1][:2]))*(1.,9/16)))
    if scale<=1e-6:
        result['review_body_recovery_state']='DEGENERATE_SHOULDER';return result
    shape=tuple((i,(p[0]-center[0])/scale,(p[1]-center[1])*9/16/scale)
        for i,p in enumerate(projected[:6]) if p[2]>=.42 and all(0.<=v<=1. for v in p[:2]))
    expected_primary={} if d.pose_shape is None else {p[0]:np.asarray(p[1:]) for p in d.pose_shape}
    resolved={p[0]:np.asarray(p[1:]) for p in shape}
    # Every joint is resolved in one native coordinate system and normalized
    # once with the retained common shoulder basis. Auxiliary absence does
    # not erase a primary wrist; known coordinate conflicts remain failures.
    if any(k not in resolved or np.linalg.norm(expected_primary[k]-resolved[k])>1e-6
            for k in expected_primary):
        result['review_body_recovery_state']='CONFLICTING_OR_MISSING_KNOWN_PRIMARY_BODY';return result
    disagreements={}
    for i,p in primary.items():
        secondary=points[i]
        if secondary[2]<.42 or not all(0<=v<1 for v in secondary[:2]):continue
        a=np.asarray(project_point(p[:2]));b=np.asarray(project_point(secondary[:2]))
        disagreements[i]=float(np.linalg.norm((a-b)*(1.,9/16))/scale)
    result['primary_auxiliary_joint_disagreement']=disagreements
    if any(v>.35 for v in disagreements.values()):
        result['review_body_recovery_state']='CONFLICTING_OR_MISSING_KNOWN_PRIMARY_BODY';return result
    result['descriptor']=replace(d,pose_shape=shape)
    result['review_body_native_xyc']=tuple(combined)
    result['review_body_projected_xyc']=tuple(tuple(p) for p in projected)
    result['review_joint_provenance']=provenance
    result['review_joint_evidence_identity']=digest(dict(contract='native_joint_model_union_review_v1',
        binding=native_binding(raw),head=head_fingerprint(selected),
        provenance=provenance,conflicts=disagreements,raw_parent=digest(raw),receipt=digest(receipt),
        production_ownership_unchanged=True))
    result['review_body_recovery_state']='OBSERVED_HEAD_CONDITIONED_BODY_REVIEW_HYPOTHESIS'
    return result


def consume_head_conditioned_review_support(token,view,*,key):
    """Review-only polyline support, with unchanged raw ownership authority.

    Only the verified adapter's successful selected-head measurement can fill
    an otherwise unknown part. Primary FAIL and known conflicting evidence
    cannot be hidden. No production unique-head ownership is manufactured.
    """
    from .group_review_program_evaluator import bind_review_part_duties,consume_frozen_part_duties
    from .group_review_camera_repair import project_review_editorial_point
    from .group_original_visible_support import ARM_INDICES,support_preserved_coordinates
    bound=bind_review_part_duties(token)
    if view is None:raise ValueError('head-conditioned support needs the exact output observation')
    row=view['row'];head=view['descriptor'].frame_local_head_id
    result=consume_frozen_part_duties(bound,row,candidate_head_ids=(head,),key=key)
    if view.get('review_body_recovery_state')!='OBSERVED_HEAD_CONDITIONED_BODY_REVIEW_HYPOTHESIS':return result
    if not view.get('review_body_receipt_sha256') or view.get('auxiliary_body_is_production_ownership') is not False:
        raise ValueError('head-conditioned support lost its verified auxiliary evidence binding')
    raw=view['review_body_native_xyc'];projected=view['review_body_projected_xyc']
    states=dict(result['part_states']);required=bound.row(row['output_tick'])
    for name,support in required['parts'].items():
        if states[name]!='UNKNOWN':continue
        indices=ARM_INDICES[name]
        if any(raw[i][2]<.42 for i in indices):continue
        native=[raw[i][:2] for i in indices]
        actual=[project_review_editorial_point(projected[i][:2],key) for i in indices]
        passed=(all(all(0<=v<=1 for v in p) for p in (*native,*actual)) if support is None else
            support_preserved_coordinates(support,native_points=native,projected_points=actual))
        states[name]='PASS' if passed else 'FAIL'
    state='FAIL' if 'FAIL' in states.values() else 'UNKNOWN' if 'UNKNOWN' in states.values() else 'PASS'
    return dict(result,state=state,part_states=states,review_head_conditioned_parts=True,
        auxiliary_body_is_production_ownership=False,review_body_receipt_sha256=view['review_body_receipt_sha256'])
