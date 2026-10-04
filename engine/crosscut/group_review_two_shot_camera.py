"""Realize actual donor neighbours jointly, before selecting a two-shot block.

Original entry/exit and donor/donor boundary have distinct targets. The camera
is a whole selected transform, not a post-selection cosmetic refit. Numeric
geometry and native choreography are separate evidence; no quality approval.
"""
from dataclasses import asdict,dataclass,replace
import json
import math
from .group_performer_family_owner import digest
from .group_review_camera_repair import fit_whole_camera_track,transformed_descriptor,geometry_cost


def _joint_target(left,right,maximum_zoom):
    if left is None or right is None or left.head_center is None or right.head_center is None:return None
    if any(d.face_height is None or d.face_width is None for d in (left,right)):return None
    destination=(.5,.45)
    # Both donors may need magnification to pan legally. Merely choosing the
    # larger native face leaves that donor at zoom 1 and unable to translate.
    required=[]
    for d in (left,right):
        x,y=d.head_center
        if not 0<x<1 or not 0<y<1:return None
        minimum_zoom=max(1.,destination[0]/x,(1-destination[0])/(1-x),
            destination[1]/y,(1-destination[1])/(1-y))
        required.append(d.face_height*minimum_zoom)
    height=min(max(required),maximum_zoom*min(left.face_height,right.face_height))
    aspect=math.sqrt(left.face_width/left.face_height*right.face_width/right.face_height)
    roll=(None if left.head_roll_radians is None or right.head_roll_radians is None else
        left.head_roll_radians+math.remainder(right.head_roll_radians-left.head_roll_radians,2*math.pi)/2)
    return replace(left,head_center=destination,face_height=height,face_width=height*aspect,
        body_box=None,pose_shape=None,head_roll_radians=roll)


def realize_review_two_shot_camera_pair(left_views,right_views,*,first,boundary,end,fps,
        original_entry,original_exit,required_support_check,maximum_zoom=2.):
    if (any(type(t) is not int for t in (first,boundary,end)) or not first<boundary<end
            or not 2<=boundary-first<=64 or not 2<=end-boundary<=64
            or len(left_views)!=boundary-first or len(right_views)!=end-boundary
            or type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or not callable(required_support_check)):
        raise ValueError('two-shot camera needs bounded actual donor intervals and original entry/exit')
    for expected,views in ((range(first,boundary),left_views),(range(boundary,end),right_views)):
        if any(v is not None and v['row']['output_tick']!=t for t,v in zip(expected,views)):
            raise ValueError('joint camera borrowed another selected donor instant')
    result=dict(contract='actual_neighbour_two_shot_camera_realization_v1',state='UNKNOWN',
        interval=[first,end],boundary=boundary,camera_fitted_before_selection=True,
        post_selection_refit_permitted=False,native_choreography_match_certified=False,quality_approved=False)
    if any(v is None for v in (*left_views,*right_views)):return None,None,result
    joint=_joint_target(left_views[-1]['descriptor'],right_views[0]['descriptor'],maximum_zoom)
    if joint is None:return None,None,result
    tracks=[];support=[]
    for side,views,start,target_a,target_b in (
            ('left',left_views,first,original_entry,joint),
            ('right',right_views,boundary,joint,original_exit)):
        by_tick={v['row']['output_tick']:v for v in views}
        def allowed(row,key):
            verdict=required_support_check(side,by_tick[row['output_tick']],key)
            if verdict not in ('PASS','FAIL','UNKNOWN'):raise ValueError('joint camera requires actual three-state support')
            return verdict!='FAIL'  # UNKNOWN remains explicit, never certification.
        duration=(len(views)-1)/fps
        track=fit_whole_camera_track([v['row'] for v in views],target_a,target_b,duration=duration,
            sample_times=tuple(i/fps for i in range(len(views))),
            head_ids=tuple(v['descriptor'].frame_local_head_id for v in views),
            descriptors=tuple(v['descriptor'] for v in views),maximum_zoom=maximum_zoom,
            required_support_check=allowed)
        if track is None:return None,None,dict(result,reason='joint_whole_camera_support_failed')
        tracks.append(track)
        support.extend(dict(side=side,tick=v['row']['output_tick'],
            state=required_support_check(side,v,track.evaluate((v['row']['output_tick']-start)/fps))) for v in views)
    left_track,right_track=tracks
    a=transformed_descriptor(left_views[-1]['descriptor'],left_track.evaluate((len(left_views)-1)/fps))
    b=transformed_descriptor(right_views[0]['descriptor'],right_track.evaluate(0.))
    entry=transformed_descriptor(left_views[0]['descriptor'],left_track.evaluate(0.))
    exit=transformed_descriptor(right_views[-1]['descriptor'],right_track.evaluate((len(right_views)-1)/fps))
    costs=dict(entry=geometry_cost(original_entry,entry),internal=geometry_cost(a,b),
        exit=geometry_cost(exit,original_exit))
    bindings=[dict(tick=v['row']['output_tick'],raw_parent=digest(v['row']['native_observation']),
        head=v['descriptor'].frame_local_head_id,joint_evidence=v.get('review_joint_evidence_identity'))
        for v in (*left_views,*right_views)]
    result.update(state='REALIZED',joint_target=joint.to_dict(),left_track=asdict(left_track),
        right_track=asdict(right_track),geometry_costs=costs,support=support,
        actual_donor_binding_identity=digest(bindings),
        actual_support_state='FAIL' if any(s['state']=='FAIL' for s in support) else
            'UNKNOWN' if any(s['state']=='UNKNOWN' for s in support) else 'PASS')
    result['realization_identity']=digest(result)
    return left_track,right_track,result


@dataclass(frozen=True,slots=True)
class FrozenRealizedTwoShotCandidate:
    token_json:str

    def __post_init__(self):
        p=self.payload
        if (p.get('contract')!='sealed_realized_two_shot_candidate_v1'
                or p.get('technical_state') not in ('PASS','FAIL','UNKNOWN')
                or type(p.get('changed_frames')) is not int or p['changed_frames']<0
                or p.get('camera_fitted_before_selection') is not True
                or p.get('post_selection_camera_refit') is not False
                or any(type(p.get(k)) not in (int,float) or not math.isfinite(p[k]) or p[k]<0
                    for k in ('native_motion_error','projected_geometry_error','camera_instability'))):
            raise ValueError('two-shot candidate needs sealed actual realization and separate finite scores')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)

    def assert_unchanged(self,payload):
        if payload!=self.payload:raise ValueError('selected two-shot camera/sampler was changed after sealing')


def select_review_two_shot_diagnostic(candidates):
    if not isinstance(candidates,(list,tuple)) or len(candidates)>336:
        raise ValueError('two-shot diagnostic requires a finite predeclared candidate inventory')
    if any(not isinstance(c,FrozenRealizedTwoShotCandidate) for c in candidates):
        raise ValueError('cannot select an unrealized two-shot camera')
    complete=[c for c in candidates if c.payload['technical_state']=='PASS' and c.payload['changed_frames']>0]
    # No KEEP/old UNKNOWN penalty; one complete changed diagnostic hypothesis.
    return min(complete,key=lambda c:(c.payload['native_motion_error'],c.payload['projected_geometry_error'],
        c.payload['camera_instability'],c.identity)) if complete else None
