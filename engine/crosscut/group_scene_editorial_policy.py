"""An explicit, uncalibrated composition utility; not a semantic QA verdict.

Required observation availability, containment and local support paths are
checked first. The five scene-framing statistics are tradeable editorial
costs, not five independently mandatory nonregression duties. UNKNOWN is
never a zero-cost observation. A baseline is one whole, realizable path,
including a single middle hypothesis shared by both incident edges. An
improved entry must not hide a new worst exit: compare the worst known
incident boundary before comparing the aggregate utility.
"""
from itertools import islice, product
import math

CONTRACT = 'declared_composition_required_evidence_then_utility_v1'
GEOMETRY_METRICS = ('occupancy', 'attention', 'scale', 'span', 'screen_motion')
ARTICULATION_METRICS = ('composition_pose_shape', 'composition_pose_velocity')


def known_metrics(row):
    values = row.get('metrics')
    return (row.get('available') is True and isinstance(values, dict)
        and all(k in values for k in GEOMETRY_METRICS)
        and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values()))


def composition_cost(row):
    if not known_metrics(row):
        raise ValueError('UNKNOWN/nonfinite scene metric is not an editorial cost')
    # Equal diagnostic weights, deliberately NOT calibrated professional taste.
    return sum(row['metrics'][k]/(1.+row['metrics'][k]) for k in GEOMETRY_METRICS)/len(GEOMETRY_METRICS)


def coherent_path_reference(groups, read, *, limit=16):
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError('baseline enumeration must have a bounded explicit budget')
    alternatives = []
    # The cap is hypothesis exploration, never a claim that omitted people or
    # complete baseline paths do not exist. Original hypotheses stay archived.
    for path in islice(product(*(g[:2] for g in groups)), limit):
        rows = [read(a, b) for a, b in zip(path, path[1:])]
        known = [known_metrics(r) for r in rows]
        alternatives.append(dict(state_keys=[s.key for s in path], edges=rows,
            complete=all(known), unknown_edge_count=known.count(False)))
    if not alternatives:
        return dict(contract=CONTRACT, state_keys=None, edges=[None]*(len(groups)-1),
            complete=False, alternatives=[], uncertainty='baseline_hypotheses_unavailable',
            uncalibrated_editorial_policy=True, quality_approved=False)
    chosen = min(alternatives, key=lambda a:(a['unknown_edge_count'],
        sum(composition_cost(r) for r in a['edges'] if known_metrics(r)), tuple(a['state_keys'])))
    return dict(contract=CONTRACT, state_keys=chosen['state_keys'], edges=chosen['edges'],
        complete=chosen['complete'], alternatives=[dict(state_keys=a['state_keys'],
            complete=a['complete'], unknown_edge_count=a['unknown_edge_count']) for a in alternatives],
        baseline_hypothesis_limit=limit, uncalibrated_editorial_policy=True, quality_approved=False)


def evaluate_composition_path(reference, measured):
    result = dict(contract=CONTRACT, admitted=False, reason=None,
        baseline_state_keys=reference.get('state_keys'),
        required_semantics_traded=False, unknown_imputed_as_zero=False,
        componentwise_geometry_nonregression_required=False,
        calibrated_professional_utility=False, quality_approved=False)
    if len(measured) != len(reference['edges']) or any(not known_metrics(r)
            or r.get('required_semantics_available') is not True for r in measured):
        return dict(result, reason='new_required_measurement_unavailable')
    baseline_known, candidate_known = [], []
    baseline_articulation, candidate_articulation = [], []
    unknown = False
    for before, after in zip(reference['edges'], measured, strict=True):
        if before is None or not known_metrics(before):
            unknown = True
            continue
        # Retain separately known non-composition motion/part measurements.
        # Geometry improvement cannot buy disappearance/worsening of these.
        for key, value in before['metrics'].items():
            if key not in GEOMETRY_METRICS and (key not in after['metrics']
                    or after['metrics'][key] > value+1e-9):
                return dict(result, reason='known_non_geometry_evidence_lost_or_worse')
        baseline_known.append(composition_cost(before))
        candidate_known.append(composition_cost(after))
        for key in ARTICULATION_METRICS:
            if key in before['metrics']:
                baseline_articulation.append(before['metrics'][key]/(1.+before['metrics'][key]))
                candidate_articulation.append(after['metrics'][key]/(1.+after['metrics'][key]))
    before_cost, after_cost = sum(baseline_known), sum(candidate_known)
    if candidate_known and max(candidate_known) > max(baseline_known)+1e-9:
        return dict(result, reason='known_worst_boundary_composition_utility_worse',
            known_baseline_worst_boundary_cost=max(baseline_known),
            matched_candidate_worst_boundary_cost=max(candidate_known))
    if candidate_known and after_cost > before_cost+1e-9:
        return dict(result, reason='known_edge_aggregate_composition_utility_worse')
    pose_before, pose_after = sum(baseline_articulation), sum(candidate_articulation)
    pose_gain = bool(baseline_articulation) and pose_after < pose_before-1e-6
    gain = unknown or after_cost < before_cost-1e-6 or pose_gain
    return dict(result, admitted=gain,
        reason='newly_observable_not_proven_improvement' if unknown else
            'known_partial_articulation_improved' if pose_gain else
            'aggregate_composition_utility_improved' if gain else 'no_editorial_gain',
        comparison_kind='newly_observable' if unknown else 'complete_observed_baseline',
        known_baseline_cost=before_cost if baseline_known else None,
        matched_candidate_cost=after_cost if candidate_known else None,
        known_baseline_worst_boundary_cost=max(baseline_known) if baseline_known else None,
        matched_candidate_worst_boundary_cost=max(candidate_known) if candidate_known else None,
        known_baseline_articulation_cost=pose_before if baseline_articulation else None,
        matched_candidate_articulation_cost=pose_after if candidate_articulation else None,
        articulated_identity_or_complete_cohort_certified=False,
        candidate_cost=sum(composition_cost(r) for r in measured)+pose_after)
