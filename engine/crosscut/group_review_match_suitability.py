"""Independent control-calibrated suitability, not observation or style rank.

Controls have exact retained evidence identities and lie outside the target.
Failure to separate them yields an unavailable policy, not relaxed thresholds.
This narrow development policy is not a general perceptual-quality model.
"""
from dataclasses import dataclass
import json
import math
from .group_performer_family_owner import digest


def _finite(value):return type(value) in (int,float) and math.isfinite(value)


@dataclass(frozen=True,slots=True)
class FrozenReviewMatchPolicy:
    token_json:str

    def __post_init__(self):
        value=json.loads(self.token_json)
        expected=_policy_payload(value.get('controls'),target_interval=value.get('target_interval',()),
            action_schema=value.get('action_schema'))
        if value!=expected:
            raise ValueError('match policy changed its independently calibrated controls or thresholds')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def _policy_payload(controls,*,target_interval,action_schema='upper_body_six_v1'):
    """Separate actual positive / wrong-person / wrong-phase controls once.

    The caller is the evidence producer, not the target score optimizer.
    All controls, including a failed separation, remain in the policy seal.
    """
    from .group_review_declared_action import ACTION_SCHEMAS
    if (action_schema not in ACTION_SCHEMAS
            or not isinstance(controls,(tuple,list)) or not 3<=len(controls)<=32
            or len(target_interval)!=2 or not 0<=target_interval[0]<target_interval[1]):
        raise ValueError('match policy needs finite independent controls and exact target interval')
    for c in controls:
        coverage_adversary=(c.get('label')=='WRONG_PHASE'
            and c.get('control_role')=='BODY_COVERAGE_ADVERSARY')
        cost_unmeasured=(c.get('motion_cost') is None and
            (c.get('label')=='WRONG_ACTOR' or coverage_adversary))
        if (c.get('label') not in ('POSITIVE_MOVING_MATCH','WRONG_ACTOR','WRONG_PHASE')
                or c.get('action_schema','upper_body_six_v1')!=action_schema
                or not isinstance(c.get('evidence_identity'),str) or len(c['evidence_identity'])!=64
                or any(v not in '0123456789abcdef' for v in c['evidence_identity'])
                or not isinstance(c.get('output_ticks'),list) or not c['output_ticks']
                or any(type(t) is not int or t<0 or target_interval[0]<=t<target_interval[1] for t in c['output_ticks'])
                or not _finite(c.get('appearance_cosine')) or not -1<=c['appearance_cosine']<=1
                or not cost_unmeasured and (not _finite(c.get('motion_cost')) or c['motion_cost']<0)
                or coverage_adversary and (c.get('motion_cost') is not None
                    or c.get('expected_body_coverage_state') not in ('FAIL','UNKNOWN')
                    or c.get('actual_body_coverage_state') not in ('PASS','FAIL','UNKNOWN'))):
            raise ValueError('calibration borrowed target results or lacks exact actual control measurements')
    positive=[c for c in controls if c['label']=='POSITIVE_MOVING_MATCH']
    wrong_actor=[c for c in controls if c['label']=='WRONG_ACTOR']
    wrong_phase=[c for c in controls if c['label']=='WRONG_PHASE']
    if not positive or not wrong_actor or not wrong_phase:
        raise ValueError('calibration omitted a required positive or adversarial control class')
    weakest_identity=min(c['appearance_cosine'] for c in positive)
    strongest_wrong_actor=max(c['appearance_cosine'] for c in wrong_actor)
    worst_positive_motion=max(c['motion_cost'] for c in positive)
    numeric_phase=[c for c in wrong_phase if c.get('control_role')!='BODY_COVERAGE_ADVERSARY']
    coverage=[c for c in wrong_phase if c.get('control_role')=='BODY_COVERAGE_ADVERSARY']
    best_wrong_phase_motion=min((c['motion_cost'] for c in numeric_phase),default=None)
    identity_separated=weakest_identity>strongest_wrong_actor
    phase_separated=best_wrong_phase_motion is not None and worst_positive_motion<best_wrong_phase_motion
    coverage_state=('NOT_REQUESTED' if not coverage else
        'CONTROL_FAILED' if any(c['actual_body_coverage_state']=='PASS' for c in coverage) else
        'UNMEASURED' if any(c['actual_body_coverage_state']=='UNKNOWN' for c in coverage) else
        'COVERAGE_NEGATIVE_CONFIRMED')
    separable=(identity_separated and phase_separated
        and coverage_state in ('NOT_REQUESTED','COVERAGE_NEGATIVE_CONFIRMED'))
    failed=(not identity_separated or best_wrong_phase_motion is not None and not phase_separated
        or coverage_state=='CONTROL_FAILED')
    token=dict(contract='independent_control_calibrated_review_match_policy_v1',
        state='CONTROL_SEPARATED' if separable else 'CONTROLS_NOT_SEPARATED' if failed else 'CONTROLS_PARTIAL',controls=list(controls),
        target_interval=list(target_interval),basis='frozen_requested_body_common_time_and_actual_output_coverage',
        action_schema=action_schema,requested_joint_indices=list(ACTION_SCHEMAS[action_schema]),
        action_kind='BODY' if action_schema=='upper_body_six_v1' else 'PARTIAL_ACTION',
        identity_control_state='CONTROL_SEPARATED' if identity_separated else 'CONTROLS_NOT_SEPARATED',
        body_coverage_adversary_state=coverage_state,
        numeric_phase_control_state=('UNMEASURED' if best_wrong_phase_motion is None else
            'CONTROL_SEPARATED' if phase_separated else 'CONTROLS_NOT_SEPARATED'),
        minimum_appearance_cosine=(weakest_identity+strongest_wrong_actor)/2 if identity_separated else None,
        maximum_motion_cost=(worst_positive_motion+best_wrong_phase_motion)/2 if phase_separated else None,
        control_policy_frozen_before_target_scoring=True,expert_quality_certified=False,
        unseen_song_generalization_validated=False,quality_approved=False)
    return token


def freeze_control_calibrated_review_match_policy(controls,*,target_interval,action_schema='upper_body_six_v1'):
    value=_policy_payload(controls,target_interval=target_interval,action_schema=action_schema)
    return FrozenReviewMatchPolicy(json.dumps(value,sort_keys=True,allow_nan=False))


def evaluate_review_match_suitability(*,policy,appearance_cosine,mutual_appearance,motion,
        actual_native_alias=False,actual_required_support_state='UNKNOWN'):
    if not isinstance(policy,FrozenReviewMatchPolicy):raise ValueError('suitability requires its independent frozen policy')
    p=policy.payload
    result=dict(contract='independent_review_match_suitability_v1',policy_identity=policy.identity,
        state='UNMEASURED',match_suitable=False,quality_approved=False)
    if actual_native_alias:return dict(result,state='UNSUITABLE',reason='native_alias_cannot_observe_action')
    if actual_required_support_state=='FAIL':return dict(result,state='UNSUITABLE',reason='required_output_support_lost')
    if (p['identity_control_state']=='CONTROL_SEPARATED' and _finite(appearance_cosine)
            and (not mutual_appearance or appearance_cosine<p['minimum_appearance_cosine'])):
        return dict(result,state='UNSUITABLE',reason=['actor_control_failed'])
    if p['state']!='CONTROL_SEPARATED':return dict(result,reason='independent_controls_not_separated',
        identity_control_state=p['identity_control_state'],numeric_phase_control_state=p['numeric_phase_control_state'],
        body_coverage_adversary_state=p['body_coverage_adversary_state'])
    if (not _finite(appearance_cosine) or motion.get('intent_kind')!=p['action_kind']
            or motion.get('same_time_action_supported') is not True
            or not _finite(motion.get('exploration_cost'))
            or motion.get('nonstationary_native_action_observed') is not True
            or motion.get('complete_on_declared_basis') is not True
            or p['action_kind']=='BODY' and motion.get('full_BODY_contract_complete') is not True
            or p['action_kind']=='PARTIAL_ACTION' and motion.get('declared_action_schema')!=p['action_schema']
            or actual_required_support_state!='PASS'):
        return dict(result,reason='required_actor_body_or_output_evidence_unmeasured')
    failures=[]
    if not mutual_appearance or appearance_cosine<p['minimum_appearance_cosine']:failures.append('actor_control_failed')
    if motion['exploration_cost']>p['maximum_motion_cost']:failures.append('action_control_failed')
    if failures:return dict(result,state='UNSUITABLE',reason=failures)
    return dict(result,state='MATCH_SUITABLE',match_suitable=True,
        named_actor_certified=False,review_encode_permitted=False)


def evaluate_review_diagnostic_encode_permission(*,policy,appearance_cosine,mutual_appearance,motion,
        actual_required_support_state,complete_keyed_incident_state,protected_reference_state,
        candidate_full_output_replay_state,actual_native_alias=False,
        partial_action_declaration=None,original_known_duty_state='UNKNOWN',actor_relation_state='UNKNOWN'):
    """One diagnostic is not adoption or a substitute numeric phase threshold."""
    if not isinstance(policy,FrozenReviewMatchPolicy):raise ValueError('diagnostic requires independent sealed controls')
    p=policy.payload
    result=dict(contract='separate_technical_body_diagnostic_encode_permission_v1',
        policy_identity=policy.identity,diagnostic_encode_permitted=False,maximum_encodes=1,
        match_suitable=False,adoption_permitted=False,quality_approved=False,
        normal_speed_independent_human_judgment_still_required=True)
    if (p['identity_control_state']!='CONTROL_SEPARATED'
            or p['body_coverage_adversary_state']=='CONTROL_FAILED'
            or p['numeric_phase_control_state']=='CONTROLS_NOT_SEPARATED'
            or actual_native_alias or not mutual_appearance or not _finite(appearance_cosine)
            or appearance_cosine<p['minimum_appearance_cosine']):
        return dict(result,state='NOT_PERMITTED',reason='independent_identity_or_known_adversary_failed')
    kind=motion.get('intent_kind')
    if original_known_duty_state!='PASS' or actor_relation_state!='PASS':
        return dict(result,state='NOT_PERMITTED',reason='original_known_or_actor_relation_duty_not_passed')
    partial=False
    if kind=='PARTIAL_ACTION':
        from .group_review_declared_action import FrozenReviewActionDeclaration
        partial=(isinstance(partial_action_declaration,FrozenReviewActionDeclaration)
            and partial_action_declaration.payload['action_kind']=='PARTIAL_ACTION'
            and p['action_schema']==partial_action_declaration.payload['schema']
            and motion.get('declaration_identity')==partial_action_declaration.identity
            and motion.get('declared_action_schema')==partial_action_declaration.payload['schema']
            and motion.get('complete_on_declared_basis') is True
            and motion.get('full_BODY_contract_complete') is False
            and motion.get('all_declared_comparison_velocity_windows_measured') is True)
    if (kind=='BODY' and p['action_kind']!='BODY' or kind!='BODY' and not partial
            or motion.get('same_time_action_supported') is not True
            or not _finite(motion.get('exploration_cost'))
            or any(s!='PASS' for s in (actual_required_support_state,complete_keyed_incident_state,
                protected_reference_state,candidate_full_output_replay_state))):
        return dict(result,state='NOT_PERMITTED',reason='complete_actual_body_output_incident_evidence_required')
    if (motion.get('nonstationary_native_action_observed') is not True
            or kind=='BODY' and motion.get('full_BODY_contract_complete') is not True):
        return dict(result,state='NOT_PERMITTED',reason='stationary_or_incomplete_scope_is_not_moving_action')
    return dict(result,state='DIAGNOSTIC_ONLY',diagnostic_encode_permitted=True,
        declared_action_kind=kind,full_BODY_contract_complete=not partial,
        numeric_phase_control_state=p['numeric_phase_control_state'],comparative_improvement_claimed=False)
