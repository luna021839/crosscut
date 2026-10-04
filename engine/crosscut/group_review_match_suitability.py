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
        expected=_policy_payload(value.get('controls'),target_interval=value.get('target_interval',()))
        if value!=expected:
            raise ValueError('match policy changed its independently calibrated controls or thresholds')

    @property
    def payload(self):return json.loads(self.token_json)

    @property
    def identity(self):return digest(self.payload)


def _policy_payload(controls,*,target_interval):
    """Separate actual positive / wrong-person / wrong-phase controls once.

    The caller is the evidence producer, not the target score optimizer.
    All controls, including a failed separation, remain in the policy seal.
    """
    if (not isinstance(controls,(tuple,list)) or not 3<=len(controls)<=32
            or len(target_interval)!=2 or not 0<=target_interval[0]<target_interval[1]):
        raise ValueError('match policy needs finite independent controls and exact target interval')
    for c in controls:
        if (c.get('label') not in ('POSITIVE_MOVING_MATCH','WRONG_ACTOR','WRONG_PHASE')
                or not isinstance(c.get('evidence_identity'),str) or len(c['evidence_identity'])!=64
                or any(v not in '0123456789abcdef' for v in c['evidence_identity'])
                or not isinstance(c.get('output_ticks'),list) or not c['output_ticks']
                or any(type(t) is not int or t<0 or target_interval[0]<=t<target_interval[1] for t in c['output_ticks'])
                or not _finite(c.get('appearance_cosine')) or not -1<=c['appearance_cosine']<=1
                or (c['label']!='WRONG_ACTOR' or c.get('motion_cost') is not None)
                    and (not _finite(c.get('motion_cost')) or c['motion_cost']<0)):
            raise ValueError('calibration borrowed target results or lacks exact actual control measurements')
    positive=[c for c in controls if c['label']=='POSITIVE_MOVING_MATCH']
    wrong_actor=[c for c in controls if c['label']=='WRONG_ACTOR']
    wrong_phase=[c for c in controls if c['label']=='WRONG_PHASE']
    if not positive or not wrong_actor or not wrong_phase:
        raise ValueError('calibration omitted a required positive or adversarial control class')
    weakest_identity=min(c['appearance_cosine'] for c in positive)
    strongest_wrong_actor=max(c['appearance_cosine'] for c in wrong_actor)
    worst_positive_motion=max(c['motion_cost'] for c in positive)
    best_wrong_phase_motion=min(c['motion_cost'] for c in wrong_phase)
    separable=(weakest_identity>strongest_wrong_actor and worst_positive_motion<best_wrong_phase_motion)
    token=dict(contract='independent_control_calibrated_review_match_policy_v1',
        state='CONTROL_SEPARATED' if separable else 'CONTROLS_NOT_SEPARATED',controls=list(controls),
        target_interval=list(target_interval),basis='frozen_requested_body_common_time_and_actual_output_coverage',
        minimum_appearance_cosine=(weakest_identity+strongest_wrong_actor)/2 if separable else None,
        maximum_motion_cost=(worst_positive_motion+best_wrong_phase_motion)/2 if separable else None,
        control_policy_frozen_before_target_scoring=True,expert_quality_certified=False,
        unseen_song_generalization_validated=False,quality_approved=False)
    return token


def freeze_control_calibrated_review_match_policy(controls,*,target_interval):
    value=_policy_payload(controls,target_interval=target_interval)
    return FrozenReviewMatchPolicy(json.dumps(value,sort_keys=True,allow_nan=False))


def evaluate_review_match_suitability(*,policy,appearance_cosine,mutual_appearance,motion,
        actual_native_alias=False,actual_required_support_state='UNKNOWN'):
    if not isinstance(policy,FrozenReviewMatchPolicy):raise ValueError('suitability requires its independent frozen policy')
    p=policy.payload
    result=dict(contract='independent_review_match_suitability_v1',policy_identity=policy.identity,
        state='UNMEASURED',match_suitable=False,quality_approved=False)
    if p['state']!='CONTROL_SEPARATED':return dict(result,reason='independent_controls_not_separated')
    if actual_native_alias:return dict(result,state='UNSUITABLE',reason='native_alias_cannot_observe_action')
    if actual_required_support_state=='FAIL':return dict(result,state='UNSUITABLE',reason='required_output_support_lost')
    if (not _finite(appearance_cosine) or motion.get('intent_kind')!='BODY'
            or motion.get('same_time_action_supported') is not True
            or not _finite(motion.get('exploration_cost'))
            or actual_required_support_state!='PASS'):
        return dict(result,reason='required_actor_body_or_output_evidence_unmeasured')
    failures=[]
    if not mutual_appearance or appearance_cosine<p['minimum_appearance_cosine']:failures.append('actor_control_failed')
    if motion['exploration_cost']>p['maximum_motion_cost']:failures.append('action_control_failed')
    if failures:return dict(result,state='UNSUITABLE',reason=failures)
    return dict(result,state='MATCH_SUITABLE',match_suitable=True,
        named_actor_certified=False,review_encode_permitted=False)
