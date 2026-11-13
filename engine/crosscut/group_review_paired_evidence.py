"""Equal review observations and non-fabricating paired quality comparisons.

Exploration may rank UNKNOWN hypotheses. Comparative improvement cannot be
manufactured by deleting an inconvenient measurement or changing time grids.
This is not a calibrated expert-quality score or normal GROUP admission.
"""
from dataclasses import dataclass
import math

from .group_performer_family_owner import digest

CONTRACT = 'common_output_tick_paired_group_review_evidence_v1'
VISIBILITY_STATES = ('not_requested', 'detector_missing', 'association_unresolved', 'observed')


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True, slots=True)
class PairedEvidencePlan:
    output_frame_count: int
    fps: float
    first: int
    end: int
    original_duty_root: str
    editorial_intent_root: str
    context_frames: int = 3
    appearance_stride: int = 6
    extra_appearance_ticks: tuple[int, ...] = ()
    incident_ticks: tuple[int, ...] = ()

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.output_frame_count, self.first, self.end,
                self.context_frames, self.appearance_stride))
                or not 0 <= self.first < self.end <= self.output_frame_count <= 36000
                or self.end-self.first > 4096 or not 1 <= self.context_frames <= 12
                or not 1 <= self.appearance_stride <= 12
                or type(self.fps) not in (int, float) or not math.isfinite(self.fps) or self.fps <= 0
                or not _sha(self.original_duty_root) or not _sha(self.editorial_intent_root)):
            raise ValueError('paired evidence needs exact common ticks, original duties and editorial intent')
        if (not isinstance(self.extra_appearance_ticks, tuple) or len(self.extra_appearance_ticks) > 512
                or tuple(sorted(set(self.extra_appearance_ticks))) != self.extra_appearance_ticks
                or any(type(t) is not int or not max(0, self.first-self.context_frames) <= t
                    < min(self.output_frame_count, self.end+self.context_frames)
                    for t in self.extra_appearance_ticks)):
            raise ValueError('paired extra appearance requests need a frozen common incident scope')
        if (not isinstance(self.incident_ticks,tuple) or len(self.incident_ticks)>512
                or tuple(sorted(set(self.incident_ticks)))!=self.incident_ticks
                or any(type(t) is not int or not self.first<=t<=self.end
                    or not 0<t<self.output_frame_count for t in self.incident_ticks)):
            raise ValueError('paired incidents need a frozen common entry/interior/exit cut union')

    @property
    def interior_ticks(self): return tuple(range(self.first, self.end))

    @property
    def left_ticks(self): return tuple(range(max(0, self.first-self.context_frames), self.first))

    @property
    def right_ticks(self): return tuple(range(self.end, min(self.output_frame_count, self.end+self.context_frames)))

    @property
    def appearance_ticks(self):
        values = {self.first, (self.first+self.end)//2, self.end-1,
            *range(self.first, self.end, self.appearance_stride), *self.extra_appearance_ticks}
        if self.left_ticks: values.add(self.left_ticks[-1])
        if self.right_ticks: values.add(self.right_ticks[0])
        return tuple(sorted(values))

    def to_dict(self):
        return dict(contract=CONTRACT, output_frame_count=self.output_frame_count, fps=self.fps,
            **({'incident_ticks':list(self.incident_ticks)} if self.incident_ticks else {}),
            first=self.first, end=self.end, original_duty_root=self.original_duty_root,
            editorial_intent_root=self.editorial_intent_root, context_frames=self.context_frames,
            appearance_stride=self.appearance_stride, interior_ticks=list(self.interior_ticks),
            extra_appearance_ticks=list(self.extra_appearance_ticks),
            left_ticks=list(self.left_ticks), right_ticks=list(self.right_ticks),
            appearance_ticks=list(self.appearance_ticks), native_aliases_deduplicated_only_for_inference=True,
            sampled_output_duration_preserved=True, quality_approved=False)

    @property
    def identity(self): return digest(self.to_dict())

    def require_observed_requests(self, *, interior_ticks, left_ticks, right_ticks, appearance_ticks):
        if (tuple(interior_ticks) != self.interior_ticks or tuple(left_ticks) != self.left_ticks
                or tuple(right_ticks) != self.right_ticks or tuple(appearance_ticks) != self.appearance_ticks):
            raise ValueError('paired side changed its grid, incident context or appearance requests')


def time_normalized_center_acceleration(ticks, centers, fps):
    """Measured finite differences in output seconds, not sample indices."""
    if (len(ticks) != len(centers) or not ticks or any(type(t) is not int for t in ticks)
            or any(a >= b for a, b in zip(ticks, ticks[1:])) or not math.isfinite(fps) or fps <= 0):
        raise ValueError('temporal review needs ordered exact output samples')
    result = []
    for ta, tb, tc, a, b, c in zip(ticks, ticks[1:], ticks[2:], centers, centers[1:], centers[2:]):
        if a is None or b is None or c is None:
            result.append(None); continue
        if any(len(p) != 2 or any(not math.isfinite(v) for v in p) for p in (a, b, c)):
            raise ValueError('temporal centers must be finite two-coordinate observations')
        dt0, dt1 = (tb-ta)/fps, (tc-tb)/fps
        ax = 2*((c[0]-b[0])/dt1-(b[0]-a[0])/dt0)/(dt0+dt1)
        ay = 2*((c[1]-b[1])/dt1-(b[1]-a[1])/dt0)/(dt0+dt1)*9/16
        result.append(math.hypot(ax, ay))
    return tuple(result)


def validate_paired_native_request_inventory(plans, capture, *, expected_render_plan_identity,
        expected_native_addresses=None):
    """Check captured rows/appearance keys, not requests copied into a score.

    Inference aliases may share a native frame; every requested OUTPUT tick
    still has its own measured row. An empty appearance detection list is an
    actual result, whereas an absent result key is an omitted request.
    """
    if not plans or any(not isinstance(p,PairedEvidencePlan) for p in plans):
        raise ValueError('paired capture needs its frozen evidence plans')
    ticks = sorted({t for p in plans for t in (*p.interior_ticks,*p.left_ticks,*p.right_ticks)})
    appearance = sorted({t for p in plans for t in p.appearance_ticks})
    if (capture.get('full_plan_identity') != expected_render_plan_identity
            or capture.get('requested_output_ticks') != ticks
            or capture.get('requested_appearance_ticks') != appearance
            or set(capture.get('rows',{})) != {str(t) for t in ticks}
            or set(capture.get('patches',{})) != {str(t) for t in ticks}
            or set(capture.get('appearance',{})) != {str(t) for t in appearance}):
        raise ValueError('paired actual capture omitted or changed output/context/appearance requests')
    if expected_native_addresses is not None and set(expected_native_addresses)!=set(ticks):
        raise ValueError('paired sampler address inventory omitted an exact output request')
    for tick in ticks:
        row = capture['rows'][str(tick)]
        raw = row['native_observation']; projection = row['projection']
        if (row.get('output_tick') != tick
                or raw['native_frame_index'] != projection['native_frame_index']
                or raw['request_source_index'] != projection['source_request_index']
                or raw['source_clock_identity'] != projection['source_clock_identity']):
            raise ValueError('paired actual row borrowed another output/source/native clock')
        if expected_native_addresses is not None and tuple(expected_native_addresses[tick])!=(
                raw['request_source_index'],raw['native_frame_index'],raw['source_clock_identity']):
            raise ValueError('paired actual capture changed its compiled renderer native address')
    return dict(requested_output_ticks=len(ticks),requested_appearance_ticks=len(appearance),
        actual_capture_keys_verified=True,native_aliases_deleted_output_ticks=False,
        fresh_native_authority_certified=False,quality_approved=False)


def compare_paired_metric_series(*, plan, original, candidate, weights, minimum_gain=.05):
    """Compare aligned measured risks; lost known evidence cannot buy BETTER.

    Both operands must come from the same producer-frozen request plan. Every
    series contains either one risk per output tick or explicit incident risk
    items shared by the two operands. Smaller risk is better. Newly available
    candidate evidence is recorded, not compared with an invented baseline.
    """
    if not isinstance(plan, PairedEvidencePlan) or not weights or not math.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError('paired comparison needs a frozen evidence plan and explicit weights')
    for side in (original, candidate):
        if side.get('evidence_plan_sha256') != plan.identity:
            raise ValueError('paired comparison borrowed another evidence plan')
        plan.require_observed_requests(**side['requests'])
        if set(side['metrics']) != set(weights):
            raise ValueError('paired comparison omitted an expected metric domain')
    lost, gained, common, components = [], [], [], {}
    weighted_delta, measured_weight = 0., 0.
    for name, weight in weights.items():
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError('paired weights must be finite nonnegative risks')
        before, after = original['metrics'][name], candidate['metrics'][name]
        if (not isinstance(before, (tuple, list)) or not before or len(before) != len(after)
                or len(before) not in (len(plan.interior_ticks),
                    len(plan.incident_ticks) if plan.incident_ticks else 2)):
            raise ValueError('paired metric changed its output/incident sample domain')
        differences = []
        for i, (a, b) in enumerate(zip(before, after, strict=True)):
            for value in (a, b):
                if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
                    raise ValueError('paired measurement must be finite or explicit UNKNOWN')
            if a is not None and b is None:
                lost.append(dict(metric=name, item=i, original=a))
            elif a is None and b is not None:
                gained.append(dict(metric=name, item=i, candidate=b))
            elif a is not None:
                differences.append(b-a); common.append(dict(metric=name, item=i))
        # Full original time denominator: deleting a bad middle sample cannot
        # renormalize the remaining good observations to occupy the whole shot.
        delta = None if not differences else sum(differences)/len(before)
        components[name] = dict(delta=delta, comparable_items=len(differences), total_items=len(before))
        if delta is not None:
            weighted_delta += weight*delta; measured_weight += weight
    if lost or not measured_weight:
        verdict, reason = 'INCONCLUSIVE', 'known_evidence_lost' if lost else 'no_comparable_measured_risks'
    elif weighted_delta < -minimum_gain:
        verdict, reason = 'BETTER', 'measured_paired_tradeoff_improves'
    elif weighted_delta > minimum_gain:
        verdict, reason = 'WORSE', 'measured_paired_tradeoff_regresses'
    else:
        verdict, reason = 'INCONCLUSIVE', 'no_measured_superiority'
    return dict(contract='nonfabricating_paired_group_review_comparison_v1',
        verdict=verdict, reason=reason, evidence_plan_sha256=plan.identity,
        weighted_candidate_minus_original=weighted_delta if measured_weight else None,
        components=components, lost_known_evidence=lost, gained_known_evidence=gained,
        new_coverage_is_quality_gain=False, source_identity_certified=False,
        motion_nonregression_certified=False, quality_approved=False)
