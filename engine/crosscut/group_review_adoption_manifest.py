"""Complete keyed review adoption, separate from expert media approval.

The producer freezes actual entry/exit/cut-union and support requests before
fresh evaluation. No continuous original is assigned a perfect match cost.
This does not create motion evidence or calibrate a perceptual threshold.
"""
from dataclasses import dataclass
import json
import math

from .group_performer_family_owner import digest


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


COMPARATIVE='COMPARATIVE_MEASURED_GAIN'
SUPPORTED='SUPPORTED_WITHOUT_IMPROVEMENT_CLAIM'


@dataclass(frozen=True,slots=True)
class FrozenOriginalMotionInventory:
    """Actual old measurements, sealed BEFORE candidate costs are available."""
    token_json:str

    def __post_init__(self):
        value=json.loads(self.token_json);rows=value.get('rows')
        if (value.get('contract')!='original_motion_measurements_before_candidate_scoring_v1'
                or not _sha(value.get('original_plan_identity')) or not _sha(value.get('original_event_identity'))
                or not isinstance(rows,list) or not rows or len({r.get('tick') for r in rows})!=len(rows)):
            raise ValueError('noncomparative mode needs genuine independently sealed original measurements')
        for row in rows:
            state,cost=row.get('state'),row.get('motion_cost')
            if (type(row.get('tick')) is not int or row['tick']<1
                    or not _sha(row.get('original_binding')) or not _sha(row.get('measurement_evidence_identity'))
                    or state not in ('MEASURED','UNMEASURED')
                    or state=='MEASURED' and (type(cost) not in (int,float) or not math.isfinite(cost) or cost<0)
                    or state=='UNMEASURED' and (cost is not None or not isinstance(row.get('unknown_reason'),str)
                        or not row['unknown_reason'])):
                raise ValueError('original known costs cannot be nulled or synthesized into unknown provenance')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def freeze_original_motion_inventory(*,original_plan_identity,original_event_identity,measurements):
    return FrozenOriginalMotionInventory(json.dumps(dict(
        contract='original_motion_measurements_before_candidate_scoring_v1',
        original_plan_identity=original_plan_identity,original_event_identity=original_event_identity,
        rows=list(measurements)),sort_keys=True,allow_nan=False))


@dataclass(frozen=True,slots=True)
class FrozenReviewAdoptionManifest:
    token_json: str

    def __post_init__(self):
        value=json.loads(self.token_json)
        rows=value.get('requests',[])
        if (value.get('contract')!='frozen_keyed_motion_adoption_manifest_v1'
                or not all(_sha(value.get(k)) for k in ('evidence_plan_identity',
                    'original_plan_identity','candidate_plan_identity','motion_policy_identity'))
                or not isinstance(rows,list) or not 1<=len(rows)<=8192
                or value.get('adoption_mode',COMPARATIVE) not in (COMPARATIVE,SUPPORTED)
                or len({r.get('request_id') for r in rows})!=len(rows)
                or any(not _sha(r.get('request_id')) or r.get('domain') not in ('SUPPORT','MOTION') for r in rows)):
            raise ValueError('adoption needs a bounded complete frozen keyed request manifest')
        if value.get('adoption_mode')==SUPPORTED:
            inventory=FrozenOriginalMotionInventory(json.dumps(value.get('sealed_original_motion_inventory')))
            if (inventory.identity!=value.get('original_motion_inventory_identity')
                    or inventory.payload['original_plan_identity']!=value['original_plan_identity']
                    or not _sha(value.get('protected_reference_identity'))
                    or {r['tick'] for r in inventory.payload['rows']}!=
                        {r['tick'] for r in rows if r['domain']=='MOTION'}
                    or any(old['original_binding']!=next(r['bindings']['original'] for r in rows
                        if r['domain']=='MOTION' and r['tick']==old['tick']) for old in inventory.payload['rows'])):
                raise ValueError('supported mode changed its sealed original or protected reference')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def freeze_review_adoption_manifest(evidence,*,original_plan_identity,candidate_plan_identity,
        original_cut_ticks,candidate_cut_ticks,incident_bindings,motion_policy_identity,support_part_inventory,
        adoption_mode=COMPARATIVE,original_motion_inventory=None,protected_reference_identity=None):
    """Incident bindings seal each actual sampler/camera/actor window separately.

    A binding is a producer-derived content fingerprint, not a selected view
    number. Full common incident inventory is computed here, including exit.
    Motion suitability policy must be frozen independently of this candidate.
    """
    if not all(_sha(v) for v in (original_plan_identity,candidate_plan_identity,motion_policy_identity)):
        raise ValueError('adoption requires exact plan and independent motion policy identities')
    old,new=set(original_cut_ticks),set(candidate_cut_ticks)
    if any(type(t) is not int or not 0<t<evidence.output_frame_count for t in old|new):
        raise ValueError('adoption cut union contains a foreign output incident')
    expected=tuple(sorted({t for t in (evidence.first,evidence.end,*old,*new)
        if evidence.first<=t<=evidence.end and 0<t<evidence.output_frame_count}))
    if tuple(evidence.incident_ticks)!=expected or set(incident_bindings)!=set(expected):
        raise ValueError('adoption manifest omitted entry/exit or an original/candidate cut')
    if adoption_mode not in (COMPARATIVE,SUPPORTED):raise ValueError('unknown frozen adoption mode')
    inventory=None
    if adoption_mode==SUPPORTED:
        if not isinstance(original_motion_inventory,FrozenOriginalMotionInventory) or not _sha(protected_reference_identity):
            raise ValueError('supported mode requires sealed original measurements and protected reference identity')
        inventory=original_motion_inventory.payload
        if (inventory['original_plan_identity']!=original_plan_identity
                or {r['tick'] for r in inventory['rows']}!=set(expected)
                or any(row['original_binding']!=incident_bindings[row['tick']]['original'] for row in inventory['rows'])):
            raise ValueError('sealed original measurements omitted or changed an exact original incident binding')
    requests=[]
    if set(support_part_inventory)!=set(evidence.interior_ticks):
        raise ValueError('adoption support inventory omitted an exact interior tick')
    for tick in evidence.interior_ticks:
        parts=support_part_inventory[tick]
        if (not isinstance(parts,(tuple,list)) or not parts or len(set(parts))!=len(parts)
                or not set(parts)<=set(('head','left_arm','right_arm','no_anatomical_duty'))):
            raise ValueError('adoption requires independently frozen per-part support requests')
        for part in parts:requests.append(dict(domain='SUPPORT',tick=tick,part=part,duty_root=evidence.original_duty_root))
    for tick in expected:
        binding=incident_bindings[tick]
        if set(binding)!=set(('original','candidate')) or not all(_sha(v) for v in binding.values()):
            raise ValueError('incident must bind actual original and candidate sampler/camera/actor windows')
        kind=('UNCHANGED' if binding['original']==binding['candidate'] else
            'NEW_CUT' if tick not in old and tick in new else
            'REMOVED_CUT' if tick in old and tick not in new else
            'REPLACED_CUT' if tick in old and tick in new else 'CONTINUOUS_CAMERA_EDIT')
        requests.append(dict(domain='MOTION',tick=tick,kind=kind,bindings=dict(binding)))
    root=dict(contract='frozen_keyed_motion_adoption_manifest_v1',
        adoption_mode=adoption_mode,
        evidence_plan_identity=evidence.identity,original_plan_identity=original_plan_identity,
        candidate_plan_identity=candidate_plan_identity,motion_policy_identity=motion_policy_identity,
        original_duty_root=evidence.original_duty_root,editorial_intent_root=evidence.editorial_intent_root,
        interval=[evidence.first,evidence.end],quality_approved=False)
    if inventory is not None:root.update(sealed_original_motion_inventory=inventory,
        original_motion_inventory_identity=original_motion_inventory.identity,
        protected_reference_identity=protected_reference_identity)
    requests=[dict(r,request_id=digest(dict(root=root,request=r))) for r in requests]
    return FrozenReviewAdoptionManifest(json.dumps(dict(root,requests=requests),sort_keys=True,allow_nan=False))


def validate_keyed_adoption_receipts(manifest,receipts):
    if not isinstance(manifest,FrozenReviewAdoptionManifest) or not isinstance(receipts,(list,tuple)):
        raise ValueError('adoption needs its frozen manifest and actual keyed receipts')
    requests=manifest.payload['requests'];expected={r['request_id']:r for r in requests}
    ids=[r.get('request_id') for r in receipts]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('adoption receipt inventory duplicated, omitted or added an expected request')
    value=manifest.payload
    for receipt in receipts:
        request=expected[receipt['request_id']]
        if (receipt.get('manifest_identity')!=manifest.identity
                or receipt.get('evidence_plan_identity')!=value['evidence_plan_identity']
                or receipt.get('original_plan_identity')!=value['original_plan_identity']
                or receipt.get('candidate_plan_identity')!=value['candidate_plan_identity']
                or receipt.get('domain')!=request['domain'] or receipt.get('tick')!=request['tick']):
            raise ValueError('adoption receipt borrowed another manifest/plan/tick/domain')
        if request['domain']=='MOTION' and (receipt.get('kind')!=request['kind']
                or receipt.get('bindings')!=request['bindings']
                or receipt.get('motion_policy_identity')!=value['motion_policy_identity']):
            raise ValueError('motion receipt changed its frozen incident kind or actual binding/policy')
        if request['domain']=='MOTION' and value.get('adoption_mode')==SUPPORTED:
            old=next(r for r in value['sealed_original_motion_inventory']['rows'] if r['tick']==request['tick'])
            if (receipt.get('original_motion_inventory_identity')!=value['original_motion_inventory_identity']
                    or receipt.get('original_motion_measurement_evidence_identity')!=old['measurement_evidence_identity']
                    or receipt.get('original_motion_cost')!=old['motion_cost']):
                raise ValueError('motion receipt erased or rewrote a sealed original measurement')
        if request['domain']=='SUPPORT' and receipt.get('part')!=request['part']:
            raise ValueError('support receipt borrowed another required anatomical part')
    return expected


def decide_keyed_review_adoption(*,manifest,receipts,comparison,protected_reference_verification=None):
    """Return NO_EDIT/HOLD/REJECT/REVIEW_ENCODE_ELIGIBLE without fabricating proof."""
    expected=validate_keyed_adoption_receipts(manifest,receipts)
    value=manifest.payload
    supported=value.get('adoption_mode')==SUPPORTED
    if comparison.get('evidence_plan_sha256')!=value['evidence_plan_identity']:
        raise ValueError('adoption comparison borrowed another exact evidence plan')
    result=dict(contract='keyed_group_review_adoption_decision_v1',manifest_identity=manifest.identity,
        numeric_review_encode_permitted=False,quality_approved=False,normal_complete_bundle_created=False,
        adoption_mode=value.get('adoption_mode',COMPARATIVE),comparative_improvement_claimed=False)
    if value['original_plan_identity']==value['candidate_plan_identity']:
        return dict(result,state='NO_EDIT',reasons=['identical_actual_plan'])
    reasons=[];unresolved=[];unknown_original=0
    if supported and (not isinstance(protected_reference_verification,dict)
            or protected_reference_verification.get('state')!='PASS'
            or protected_reference_verification.get('reference_identity')!=value['protected_reference_identity']
            or not _sha(protected_reference_verification.get('actual_output_evidence_identity'))):
        unresolved.append(('protected_reference','actual_protected_output_verification_required'))
    for receipt in receipts:
        request=expected[receipt['request_id']]
        if request['domain']=='SUPPORT':
            before,after=receipt.get('original_state'),receipt.get('candidate_state')
            if before not in ('PASS','FAIL','UNKNOWN') or after not in ('PASS','FAIL','UNKNOWN'):
                raise ValueError('support receipt needs separate actual original and candidate states')
            if after=='FAIL':reasons.append((request['tick'],'support_failed'))
            elif supported and after!='PASS':unresolved.append((request['tick'],'complete_candidate_support_required'))
            elif before=='PASS' and after=='UNKNOWN':unresolved.append((request['tick'],'known_support_lost'))
            elif before=='UNKNOWN' and after=='UNKNOWN':unknown_original+=1
            continue
        verdict=receipt.get('verdict');kind=request['kind']
        if verdict in ('UNMEASURED','UNKNOWN'):
            unresolved.append((request['tick'],'motion_unmeasured'));continue
        if verdict=='REGRESSING':reasons.append((request['tick'],'motion_regressed'));continue
        if kind=='UNCHANGED':
            if verdict!='UNCHANGED':raise ValueError('unchanged receipt must consume equal actual window bindings')
        elif kind=='NEW_CUT':
            if (verdict!='SUPPORTED_NEW_MATCH' or receipt.get('same_time_action_supported') is not True
                    or receipt.get('independent_suitability_passed') is not True):
                unresolved.append((request['tick'],'new_cut_suitability_not_supported'))
        elif kind=='REPLACED_CUT':
            before,after=receipt.get('original_motion_cost'),receipt.get('candidate_motion_cost')
            tolerance=receipt.get('frozen_nonregression_tolerance')
            original=(next(r for r in value['sealed_original_motion_inventory']['rows']
                if r['tick']==request['tick']) if supported else None)
            if supported and (receipt.get('independent_suitability_passed') is not True
                    or receipt.get('candidate_motion_intent_kind')!='BODY'
                    or receipt.get('candidate_complete_body_observation') is not True
                    or not _sha(receipt.get('candidate_motion_evidence_identity'))):
                unresolved.append((request['tick'],'independent_complete_candidate_body_required'))
            if supported and original['state']=='UNMEASURED':
                if (verdict!='SUPPORTED_REPLACEMENT' or receipt.get('same_time_action_supported') is not True
                        or receipt.get('independent_suitability_passed') is not True
                        or receipt.get('candidate_motion_intent_kind')!='BODY'
                        or receipt.get('candidate_complete_body_observation') is not True
                        or not _sha(receipt.get('candidate_motion_evidence_identity'))
                        or type(after) not in (int,float) or not math.isfinite(after) or after<0):
                    unresolved.append((request['tick'],'independent_complete_supported_replacement_required'))
            elif (verdict!='NONREGRESSING' or receipt.get('same_time_action_supported') is not True
                    or any(type(v) not in (int,float) or not math.isfinite(v) or v<0
                        for v in (before,after,tolerance))):
                unresolved.append((request['tick'],'replaced_cut_nonregression_unmeasured'))
            elif after>before+tolerance:reasons.append((request['tick'],'measured_motion_regressed'))
        else:
            required='CONTINUITY_SUPPORTED' if kind=='REMOVED_CUT' else 'CAMERA_CONTINUITY_SUPPORTED'
            if verdict!=required or receipt.get('actual_sampler_continuity_verified') is not True:
                unresolved.append((request['tick'],'actual_continuity_not_supported'))
    if comparison.get('verdict')=='WORSE':reasons.append(('comparison','paired_regression'))
    elif not supported and comparison.get('verdict')!='BETTER':unresolved.append(('comparison','no_comparable_measured_gain'))
    if reasons:return dict(result,state='REJECT',reasons=reasons,unresolved=unresolved)
    if unresolved:return dict(result,state='HOLD',reasons=unresolved)
    return dict(result,state='REVIEW_ENCODE_ELIGIBLE',numeric_review_encode_permitted=True,
        comparative_improvement_claimed=not supported,
        unresolved_original_support=unknown_original,reasons=[],professional_quality_certified=False)
