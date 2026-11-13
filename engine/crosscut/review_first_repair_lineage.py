"""Record actual singleton boundary repairs without changing any QA verdict.

The JSON quality floor must see the same successful repair history as the
ordinary candidate producer. This consumes typed, final timeline evidence;
serialized pass flags or an unrelated repair report cannot authorize a record.
"""
from collections.abc import Mapping

from .ive_product_quality_floor import PRE_REVIEW_REPAIR_LINEAGE_CONTRACT
from .timeline_review import TimelineReview
from .visual_boundary_repair import VisualBoundaryRepairPlan
from .visual_score import VisualTimelinePlan


def _clocks(timeline):
    return tuple((c.segment_index, c.source_index, c.canonical_master_start,
                  c.canonical_master_end, c.source_local_start, c.source_local_end)
                 for c in timeline.choices)


def review_first_repair_lineage(*, candidate_index, timeline, repairs, review):
    """Return one genuine successful pre-review repair event, or no event.

    Timeline geometry may acquire extra dissolve descriptors after repair, so
    source and master clocks (not metadata object equality) bind its result.
    The caller still validates the complete render bundle before publishing.
    """
    if (type(candidate_index) is not int or candidate_index < 0
            or not isinstance(timeline, VisualTimelinePlan)
            or not isinstance(review, TimelineReview)
            or repairs is not None and not isinstance(repairs, VisualBoundaryRepairPlan)):
        raise TypeError('repair lineage requires typed candidate/timeline/review evidence')
    summaries = tuple(s for s in timeline.candidate_review_summaries
                      if isinstance(s, Mapping) and s.get('selected') is True)
    if not summaries or not review.passed or repairs is None or repairs.changed_count <= 0:
        return None
    if len(summaries) != 1:
        raise ValueError('repair lineage requires one selected pre-review')
    summary = summaries[0]
    failures = summary.get('hard_failure_codes')
    if (type(summary.get('candidate_index')) is not int
            or summary['candidate_index'] != candidate_index
            or type(summary.get('passed')) is not bool
            or not isinstance(failures, list)
            or any(not isinstance(code, str) or not code for code in failures)
            or len(failures) != len(set(failures))
            or summary['passed'] == bool(failures)):
        raise ValueError('repair lineage pre-review identity or failures are invalid')
    if summary['passed']:
        return None
    if _clocks(repairs.timeline) != _clocks(timeline):
        raise ValueError('repair lineage does not belong to the selected timeline clocks')
    before, after = _clocks(repairs.attempt_input_timelines[0]), _clocks(timeline)
    if len(before) != len(after) or any(a[:2] != b[:2] for a,b in zip(before, after)):
        raise ValueError('singleton boundary repair changed the source path')
    if before == after:
        return None  # A changed intermediate attempt is not a final change.
    return {
        'stage': 'pre_review_failure_repair_validation',
        'contract': PRE_REVIEW_REPAIR_LINEAGE_CONTRACT,
        'candidate_index': candidate_index,
        'pre_review_passed': False,
        'pre_review_hard_failure_codes': list(failures),
        'boundary_repair_analysis_basis': repairs.analysis_basis,
        'boundary_repair_changed_count': repairs.changed_count,
        'local_source_substitution_count': 0,
        'post_repair_timeline_review_passed': True,
        'post_repair_hard_failure_codes': [],
        'output_geometry_pre_proxy_eligible': False,
        'proxy_review_eligible': True,
        'delivery_eligible': True,
        'decoded_output_quality_approval': False,
    }
