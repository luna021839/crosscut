"""Bounded, model-free scene trials; never a retained-family/quality owner.

An explicit normal-app sibling, not a fallback which promotes an uncertified
bundle. Original identity/clock/cut authority is mandatory. Scene texture is
rank evidence ONLY: no detector rectangle is certified as a performer, and
unknown actors, head/arms and beat phase remain unknown in the artifact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import tempfile
from time import monotonic
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from . import group_performer_family_owner as family
from . import group_source_changing_owner as source
from .group_actual_shots import visible_group_shots
from .group_joint_observations import GroupJointObservationSession, _file_identity
from .group_joint_source_camera import GroupOutputRole, JointGroupWorkBudget
from .group_joint_state_factory import candidate_joint_group_state, assemble_single_group_source_camera_state
from .group_scene_intent import declared_shot_intent
from .group_scene_source_owner import (freeze_derivatives, validate_bound_derivative_path,
    observed_wide_repair_priority, time_stratified_window_frontier)
from .group_source_domain import GroupSourceDomain
from .optional_native_worker import current_optional_deadline, isolated_optional
from .render import RenderPlan, render, segment_frame_counts
from .render_frame_schedule import probe_render_frame_clock

CONTRACT = 'normal_scene_review_proposal_v2'
MAX_WINDOWS = 3
MAX_MUTABLE_FRAMES = 240  # Includes producer + independent replay within 1200 source reads.
MAX_OUTPUT_FRAMES = 18000
MAX_OUTPUT_SECONDS = 300.
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
WORK_LIMITS = dict(maximum_states=24, maximum_edges=128, maximum_dense_paths=4,
    maximum_new_decodes=1200, maximum_model_calls=1,
    maximum_projected_cache_bytes=32 * 1024 * 1024)
UNKNOWN = dict(performer_identity=None, stage_cohort=None, detected_person_count=None,
    articulated_motion=None, head_extent=None, arm_containment=None, beat_phase=None,
    human_quality=None, semantic_authority=False)


@dataclass(frozen=True, slots=True)
class SceneReviewContext:
    incumbent: Any
    domain: GroupSourceDomain
    source_identities: tuple
    code_dependencies: tuple
    token: Any
    deadline: float
    source_derivative_binding: Any


def _remaining(context):
    value = context.deadline - monotonic()
    if value <= 0:
        raise ValueError('scene review shared absolute deadline expired')
    return value


def _fresh(context):
    _remaining(context)
    if context.source_identities != tuple(_file_identity(p) for p in context.domain.eligible_sources):
        raise ValueError('scene review source preimage changed')
    if context.code_dependencies != tuple(sorted(family.dependencies().items())):
        raise ValueError('scene review code/installed-dependency preimage changed')


def freeze_context(*, controller, eligible_indices, offsets, timewarps, scene_cut_times,
                   source_derivatives, **unused):
    from .pipeline import _validate_review_first_group_bundle
    bundle = controller.incumbent
    family._supported(bundle)
    plan = bundle.render_plan
    if not 3 <= len(eligible_indices) <= 8 or plan.output_frame_count > MAX_OUTPUT_FRAMES or plan.render_duration > MAX_OUTPUT_SECONDS:
        raise ValueError('scene review bounded source/output domain unavailable')
    _validate_review_first_group_bundle(bundle, eligible_indices=eligible_indices,
        offsets=offsets, timewarps=timewarps, common_start=plan.master_audio_start,
        phrase_times=bundle.phrase_times)
    deadline = current_optional_deadline()
    if deadline is None or deadline <= monotonic():
        raise ValueError('scene review requires the existing optional deadline')
    identities = tuple(_file_identity(plan.sources[i]) for i in eligible_indices)
    clocks = tuple(probe_render_frame_clock(plan.sources[i],
        timeout_seconds=max(.001, min(10., deadline-monotonic()))) for i in eligible_indices)
    domain = GroupSourceDomain(tuple(plan.sources), tuple(eligible_indices), tuple(offsets),
        tuple(timewarps), tuple(tuple(v) for v in scene_cut_times), clocks)
    value = SceneReviewContext(bundle, domain, identities, tuple(sorted(family.dependencies().items())),
        controller.transaction_token(), deadline, freeze_derivatives(source_derivatives, domain, identities))
    _fresh(value)
    return value


def _session(context, previous_counts=None):
    # Independent phase quotas enforce a 1200-read hard aggregate even if a
    # child's diagnostic counters are false. Parent never resets a 1200 cap.
    limits = dict(WORK_LIMITS, maximum_new_decodes=500 if previous_counts is not None else 700,
                  maximum_edges=4 if previous_counts is not None else 124)
    budget = JointGroupWorkBudget(absolute_deadline=context.deadline, **limits)
    if previous_counts is not None:
        if set(previous_counts) != set(budget.counts):
            raise ValueError('scene review work ledger domain changed')
        parent_limits = JointGroupWorkBudget(absolute_deadline=context.deadline, **WORK_LIMITS).limits
        for key, count in previous_counts.items():
            maximum = 700 if key == 'new_decodes' else 124 if key == 'edges' else parent_limits[key]
            if type(count) is not int or not 0 <= count <= maximum:
                raise ValueError('scene review malformed cumulative work count')
        if previous_counts['model_calls'] != 0 or previous_counts['dense_paths'] != 1:
            raise ValueError('scene review child phase/model work contract changed')
    return GroupJointObservationSession(domain=context.domain, budget=budget,
        maximum_raw_cache_bytes=32*1024*1024)


def _intents_and_guard(context, shots):
    """Stronger incident or mutable duties cannot be relabelled as texture.

    Fixed nonincident cells stay pixel-identical. A declared formation duty is
    also stronger than this trial. A known focal boundary observation vetoes
    even a stale/incorrect WIDE label; failed decisions retain their duty.
    """
    intents = tuple(declared_shot_intent(context, shot,
        endpoint_scope='exit' if i == 0 else 'entry' if i == 2 else 'whole')
        for i, shot in enumerate(shots))
    if any(value['role'] != GroupOutputRole.COMPOSITION.value for value in intents):
        raise ValueError('scene review cannot drop stronger head/arm/formation duty')
    middle = shots[1]
    incident_indices = {middle.first_segment_index-1, middle.last_segment_index}
    alignment = context.incumbent.editorial_motion_plan.boundary_subject_alignment_plan
    for decision in alignment.decisions:
        if (decision.boundary_index in incident_indices
                and decision.subject_kind in ('face', 'upper_body', 'formation', 'silhouette')):
            raise ValueError('scene review cannot discard a known incident subject observation')
    repairs = context.incumbent.boundary_repairs
    for repair in getattr(repairs, 'repairs', ()):
        if (middle.first_segment_index-1 <= repair.boundary_index <= middle.last_segment_index
                and repair.selected_visual_evidence.closeup_gesture is not None):
            raise ValueError('scene review cannot discard known selected arm/gesture evidence')
    # Other retained dense/head owners were rejected by _supported;

    # do not infer no people from a missing declaration or an empty detector.
    return intents


def _review_states(context, shots):
    """Read fixed incident support, not a far-away fractional cell fragment.

    The outer shots are unchanged. If an interior canonical boundary leaves
    all four incident samples available, use that bounded original-cell view.
    This does not split, trim or publish the fixed shots in the output plan.
    Mutable bounds remain the complete original shot and must be cell-owned.
    The independent assembler rederives the same fixed view from the full plan.
    """
    offsets = [0]
    for count in segment_frame_counts(context.incumbent.render_plan):
        offsets.append(offsets[-1]+count)
    views = list(shots)
    left, right = shots[0], shots[-1]
    starts = [t for t in offsets if left.first_output_frame <= t <= left.end_output_frame-4]
    ends = [t for t in offsets if right.first_output_frame+4 <= t <= right.end_output_frame]
    if starts:
        views[0] = SimpleNamespace(first_output_frame=starts[0], end_output_frame=left.end_output_frame)
    if ends:
        views[-1] = SimpleNamespace(first_output_frame=right.first_output_frame, end_output_frame=ends[-1])
    return source._states(context.incumbent.render_plan, context.domain, views,
        (GroupOutputRole.COMPOSITION,)*3, fixed_subcell_views=True)


def _path(context, shots, donor):
    intents = _intents_and_guard(context, shots)
    baseline = _review_states(context, shots)
    if type(donor) is not int or not 0 <= donor < len(context.domain.eligible_indices):
        raise ValueError('scene review donor index domain invalid')
    if donor == baseline[1].eligible_source_index:
        raise ValueError('scene review cannot rename the incumbent')
    state = candidate_joint_group_state(plan=context.incumbent.render_plan, domain=context.domain,
        incumbent_state=baseline[1], eligible_source_index=donor)
    looks = source.source_static_looks(context)
    index = context.domain.request_index(donor)
    if index not in looks:
        raise ValueError('destination static grade unavailable; not invented or borrowed')
    # Complete whole-frame camera. No arbitrary ROI or old source face crop.
    state = replace(state, segments=tuple(replace(s, look_adjustment=looks[index])
        for s in state.segments)).bind_source(context.domain)
    path = (baseline[0], state, baseline[2])
    validate_bound_derivative_path(context.source_derivative_binding,
        [s.eligible_source_index for s in path])
    # A review may replace an existing same-source seek/rate seam with a
    # source switch AT THAT SAME OUTPUT TICK. Native broadcast events, every
    # other seam, cadence and fixed pixels still retain their strict proof.
    # Requiring the old source-specific seam *kind* would forbid the intended
    # source change even when no output cut was added, removed or moved.
    assembled = assemble_single_group_source_camera_state(plan=context.incumbent.render_plan,
        domain=context.domain, path=path, replace_incident_clock_jumps=True)
    return baseline, path, assembled, intents


def scene_descriptor(image):
    """Whole-raster spatial contrast/color, NOT occupancy or articulated pose."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8):
        raise ValueError('scene descriptor requires an actual decoded BGR raster')
    small = cv2.resize(image, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mass = cv2.resize(np.sqrt(dx*dx+dy*dy), (8, 6), interpolation=cv2.INTER_AREA)
    total = float(mass.sum())
    # A flat/blank scene has UNKNOWN spatial support, not a perfect zero score.
    texture = (mass / total).ravel().astype(float).tolist() if total > .01 else None
    return dict(spatial_texture=texture, mean_bgr=small.mean(axis=(0, 1)).astype(float).tolist(),
        contrast=float(gray.std()), basis='whole_raster_texture_color_not_people_v1')


def scene_edge(a, b):
    texture = None
    if a['spatial_texture'] is not None and b['spatial_texture'] is not None:
        texture = float(np.abs(np.array(a['spatial_texture'])-b['spatial_texture']).sum()/2)
    color = float(np.mean(np.abs(np.array(a['mean_bgr'])-b['mean_bgr'])))
    contrast = abs(a['contrast']-b['contrast'])
    values = [texture, color, contrast]
    if any(v is not None and (not math.isfinite(v) or not 0 <= v <= 1.00001) for v in values):
        raise ValueError('scene metric must be finite or explicitly unknown')
    return dict(metrics=dict(spatial_texture=texture, color=color, contrast=contrast),
        unknown_metric_count=sum(v is None for v in values),
        # Explicit uncertainty charge, not imputation of a measured value.
        ranking_cost=sum(1. if v is None else v for v in values)/3,
        quality_improvement_established=False, semantics=dict(UNKNOWN))


def compare_review_path(baseline_edges, candidate_edges):
    """Keep the incumbent unless the two-sided incident path has a gain.

    This is only an explicit review-artifact ranking rule, not a human pose
    verdict. Unknown texture cannot be charged as a measured baseline cost.
    Component costs may trade; a lower total cannot hide a new worst seam.
    """
    result = dict(contract='scene_review_incumbent_two_sided_ranking_v1',
        admitted=False, quality_approved=False, unknown_imputed_as_zero=False)
    if (not isinstance(baseline_edges, (list, tuple))
            or not isinstance(candidate_edges, (list, tuple))
            or len(baseline_edges) != 2 or len(candidate_edges) != 2):
        return dict(result, reason='two_incident_edges_required')
    before, after = [], []
    for edges, costs in ((baseline_edges, before), (candidate_edges, after)):
        for row in edges:
            if (not isinstance(row, dict) or row.get('unknown_metric_count') != 0
                    or type(row.get('unknown_metric_count')) is not int):
                return dict(result, reason='unknown_texture_is_not_measured_gain')
            metrics = row.get('metrics')
            if (not isinstance(metrics, dict)
                    or set(metrics) != {'spatial_texture', 'color', 'contrast'}
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                        or not 0 <= v <= 1.00001 for v in metrics.values())):
                return dict(result, reason='texture_measurement_unavailable')
            cost = sum(metrics.values())/3
            if (type(row.get('ranking_cost')) not in (int, float)
                    or not math.isfinite(row['ranking_cost'])
                    or abs(row['ranking_cost']-cost) > 1e-12):
                return dict(result, reason='texture_ranking_cost_not_measured')
            costs.append(cost)
    result.update(baseline_cost=sum(before)/2, candidate_cost=sum(after)/2,
        baseline_worst_cost=max(before), candidate_worst_cost=max(after),
        gain=(sum(before)-sum(after))/2)
    if max(after) > max(before)+1e-9:
        return dict(result, reason='new_worst_incident_texture_seam')
    if result['gain'] <= 1e-6:
        return dict(result, reason='incumbent_has_no_lower_two_sided_cost')
    return dict(result, admitted=True, reason='two_sided_texture_gain_not_quality_approval')


def _sample(session, state, tick):
    native = state.native_frame_indices[tick-state.first_output_frame]
    index = session.domain.request_index(state.eligible_source_index)
    raw = session.native_pixels(index, native)
    image = session.projected_pixels(state, tick)
    if raw is None or image is None:
        raise ValueError('scene review missing exact native/projected pixels')
    row = dict(output_tick=tick, request_source_index=index, native_frame_index=native,
        raw_shape=list(raw.shape), raw_pixels_sha256=sha256(raw.tobytes()).hexdigest(),
        projected_pixels_sha256=sha256(image.tobytes()).hexdigest())
    return row, image


def _measure(session, path):
    # Exactly the declared four incident ticks, never a whole fixed shot gate.
    endpoints = []
    for a, b in zip(path, path[1:]):
        if min(a.end_output_frame-a.first_output_frame, b.end_output_frame-b.first_output_frame) < 4:
            raise ValueError('scene proposal incident window too short')
        ticks = (a.end_output_frame-4, a.end_output_frame-1,
                 b.first_output_frame, b.first_output_frame+3)
        observations = [_sample(session, s, t) for s, t in zip((a, a, b, b), ticks)]
        edge = scene_edge(scene_descriptor(observations[1][1]), scene_descriptor(observations[2][1]))
        # Pixel change over three output ticks is observable, but not a limb,
        # optical-flow identity or musical phase estimate. It is diagnostic.
        edge['raster_change'] = [float(np.mean(np.abs(cv2.resize(observations[j][1], (96,54)).astype(float)
            - cv2.resize(observations[j+1][1], (96,54)).astype(float)))/255) for j in (0, 2)]
        edge['samples'] = [item[0] for item in observations]
        endpoints.append(edge)
        session.budget.spend('edges')
    return endpoints


def _dense(session, baseline, path):
    rows = []
    for tick in range(path[1].first_output_frame, path[1].end_output_frame):
        before, _ = _sample(session, baseline[1], tick)
        after, _ = _sample(session, path[1], tick)
        rows.append(dict(output_tick=tick, before=before, after=after))
        # Dense streams must not retain every projected 4K frame until eviction.
        session.retain_native_pixels(((before['request_source_index'], before['native_frame_index']),
                                      (after['request_source_index'], after['native_frame_index'])))
    if not any(row['before']['projected_pixels_sha256'] != row['after']['projected_pixels_sha256'] for row in rows):
        raise ValueError('scene review has no changed pixels; incumbent alias rejected')
    return rows


@isolated_optional()
def propose(context):
    _fresh(context)
    plan = context.incumbent.render_plan
    shots = visible_group_shots(plan, context.domain)
    prepared, errors = [], []
    for i in range(max(0, len(shots)-2)):
        window = shots[i:i+3]
        if not 4 <= window[1].frame_count <= MAX_MUTABLE_FRAMES:
            continue
        try:
            _intents_and_guard(context, window)
            baseline = _review_states(context, window)
            prepared.append((baseline, window, dict(observed_wide_repair_priority=
                observed_wide_repair_priority(context.incumbent.visual_timeline, window[1:2]))))
        except ValueError as exc:
            errors.append(str(exc))
    distributed = list(time_stratified_window_frontier(prepared, MAX_WINDOWS))
    if distributed:
        highest = max(enumerate(prepared), key=lambda row: row[1][2]['observed_wide_repair_priority'])
        distributed = [highest] + [row for row in distributed if row[0] != highest[0]]
        distributed = distributed[:MAX_WINDOWS]
    session = _session(context)
    candidates = []
    try:
        for _, (baseline, window, _) in distributed:
            baseline_edges = _measure(session, baseline)
            for donor in range(len(context.domain.eligible_indices)):
                if session.budget.counts['states'] >= session.budget.limits['states']:
                    break
                try:
                    old, path, assembly, intents = _path(context, window, donor)
                except ValueError as exc:
                    errors.append(str(exc)); continue
                session.budget.spend('states')
                edges = _measure(session, path)
                comparison = compare_review_path(baseline_edges, edges)
                if not comparison['admitted']:
                    errors.append(comparison['reason']); continue
                candidates.append((-comparison['gain'],
                    window[1].first_output_frame, donor, window, assembly, intents, baseline_edges, edges, comparison))
        if not candidates:
            return None, dict(contract=CONTRACT, status='unavailable', errors=errors,
                work=session.budget.to_dict(), semantics=dict(UNKNOWN))
        # The unchanged incumbent is an explicit zero-gain option. Rank the
        # measured improvement, not absolute costs of different baseline shots.
        best = min(candidates, key=lambda value: value[:3])
        _, _, donor, window, assembly, intents, old_edges, edges, comparison = best
        cost = comparison['candidate_cost']
        baseline, path, check, _ = _path(context, window, donor)
        if check[0] != assembly[0]:
            raise ValueError('scene numeric assembly is not deterministic')
        session.budget.spend('dense_paths')
        dense = _dense(session, baseline, path)
        schedule = family._schedule(assembly[0], context.domain)
        original_schedule = family._schedule(plan, context.domain)
        first, end = window[1].first_output_frame, window[1].end_output_frame
        if any(asdict(schedule[t]) != asdict(original_schedule[t]) for t in schedule if not first <= t < end):
            raise ValueError('scene review changed an outside source/output clock')
        receipt = dict(contract=CONTRACT, status='PROPOSED_NOT_RENDERED', artifact_role='scene_review_only',
            quality_approved=False, delivery_completion_message_allowed=False,
            original_plan_sha256=family.digest(family.plan_payload(plan)),
            original_metadata_sha256=family.digest(family.metadata_payload(context.incumbent)),
            candidate_plan_sha256=family.digest(family.plan_payload(assembly[0])),
            original_schedule_sha256=family._schedule_digest(original_schedule),
            candidate_schedule_sha256=family._schedule_digest(schedule),
            bounds=[[s.first_output_frame,s.end_output_frame] for s in window],
            donor_eligible_index=donor, intents=intents, dense=dense, baseline_edges=old_edges,
            candidate_edges=edges, ranking_cost=cost,
            measured_surrogate_gain=comparison['gain'], path_comparison=comparison,
            surrogate_is_not_a_quality_verdict=True, semantics=dict(UNKNOWN),
            derivative_audit=context.source_derivative_binding,
            source_identities=context.source_identities, code_dependencies=context.code_dependencies,
            work=session.budget.to_dict(), search_errors=errors,
            refreshed_retained_family_authority=False,
            invalidated_changed_families=['face','pose','motion','boundary_look','timeline_descriptors','editorial_QA'])
        _fresh(context)
        return (assembly[0], receipt), None
    finally:
        session.close()


def _rederive(context, plan, receipt):
    if (receipt.get('contract') != CONTRACT or receipt.get('quality_approved') is not False
            or receipt.get('delivery_completion_message_allowed') is not False
            or receipt.get('refreshed_retained_family_authority') is not False
            or family.digest(receipt.get('semantics')) != family.digest(UNKNOWN)
            or receipt.get('artifact_role') != 'scene_review_only'):
        raise ValueError('scene review tier/uncertainty contract changed')
    comparison = compare_review_path(receipt.get('baseline_edges', []), receipt.get('candidate_edges', []))
    if (not comparison['admitted']
            or family.digest(receipt.get('path_comparison')) != family.digest(comparison)):
        raise ValueError('scene review incumbent/two-sided comparison unavailable or changed')
    original = context.incumbent.render_plan
    shots = visible_group_shots(original, context.domain)
    bounds = receipt.get('bounds')
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 3
            or any(not isinstance(row, (list, tuple)) or len(row) != 2
                   or any(type(t) is not int for t in row) for row in bounds)):
        raise ValueError('scene review bounds must be exact integer output ticks')
    matching = [shots[i:i+3] for i in range(max(0,len(shots)-2))
        if [[s.first_output_frame,s.end_output_frame] for s in shots[i:i+3]] == bounds]
    if len(matching) != 1 or not 4 <= matching[0][1].frame_count <= MAX_MUTABLE_FRAMES:
        raise ValueError('scene review bounds are not original whole-shot ownership')
    baseline, path, assembly, intents = _path(context, matching[0], receipt.get('donor_eligible_index'))
    expected = dict(original_plan_sha256=family.digest(family.plan_payload(original)),
        original_metadata_sha256=family.digest(family.metadata_payload(context.incumbent)),
        candidate_plan_sha256=family.digest(family.plan_payload(assembly[0])),
        original_schedule_sha256=family._schedule_digest(family._schedule(original,context.domain)),
        candidate_schedule_sha256=family._schedule_digest(family._schedule(assembly[0],context.domain)),
        intents=family.plain(intents), derivative_audit=family.plain(context.source_derivative_binding),
        source_identities=family.plain(context.source_identities), code_dependencies=family.plain(context.code_dependencies))
    if family.digest(family.plan_payload(plan)) != family.digest(family.plan_payload(assembly[0])):
        raise ValueError('scene review consumer plan differs from independently assembled source/camera')
    if any(family.digest(receipt.get(k)) != family.digest(v) for k,v in expected.items()):
        raise ValueError('scene review source/metadata/intent/clock preimage differs')
    return baseline, path


@isolated_optional()
def replay(context, plan, receipt):
    """Fresh reader/worker; exact every-tick equality, not set membership."""
    _fresh(context)
    baseline, path = _rederive(context, plan, receipt)
    session = _session(context, receipt['work']['counts'])
    try:
        session.budget.spend('dense_paths')
        old_edges, edges = _measure(session, baseline), _measure(session, path)
        comparison = compare_review_path(old_edges, edges)
        if not comparison['admitted']:
            raise ValueError('scene review incumbent wins fresh two-sided comparison: '+comparison['reason'])
        if family.digest(receipt.get('path_comparison')) != family.digest(comparison):
            raise ValueError('scene review two-sided comparison changed')
        dense = _dense(session, baseline, path)
        if (family.digest(dense) != family.digest(receipt.get('dense'))
                or family.digest(old_edges) != family.digest(receipt.get('baseline_edges'))
                or family.digest(edges) != family.digest(receipt.get('candidate_edges'))):
            raise ValueError('scene review exact ordered native/projected evidence changed')
        expected_cost = comparison['candidate_cost']
        expected_gain = comparison['gain']
        if (family.digest(receipt.get('ranking_cost')) != family.digest(expected_cost)
                or family.digest(receipt.get('measured_surrogate_gain')) != family.digest(expected_gain)):
            raise ValueError('scene review ranking metrics changed')
        _fresh(context)
        phase = session.budget.to_dict()
        combined = JointGroupWorkBudget(absolute_deadline=context.deadline, **WORK_LIMITS).to_dict()
        combined['counts'] = {k: receipt['work']['counts'][k]+phase['counts'][k] for k in phase['counts']}
        if any(combined['counts'][k] > combined['limits'][k] for k in combined['counts']):
            raise ValueError('scene review aggregate phase budget exceeded')
        combined.update(scope='scene_trial_producer_and_parent_only',
            phase_work=[receipt['work'],phase], independent_native_read_quotas=[700,500],
            producer_counters_independently_remeasured=False,
            hard_aggregate_does_not_trust_child_counters=True)
        return combined
    finally:
        session.close()


def _not_requested_geometry(image):
    from .rendered_proxy_review import OutputSubjectGeometry
    # Legacy scalar QA uses a zero-length observation inventory. Its zero
    # counts are explicitly NOT measured person absence and are not exported
    # as semantic evidence; the new contract uses nulls in UNKNOWN above.
    return OutputSubjectGeometry(face_backend='not_requested_scene_review',
        silhouette_backend='not_requested_scene_review', shot_scale_basis='unknown_not_requested')


def _technical_qa(context, proxy, review):
    from .proxy_review_contract import proxy_integrity_evidence, proxy_integrity_evidence_valid
    from .post_render_qa import (PostRenderQAConfig, _IssueCollector,
        _probe_and_decode, _validate_ffprobe_against_plan)
    integrity = proxy_integrity_evidence(review.payload)
    if not proxy_integrity_evidence_valid(integrity):
        raise ValueError('scene review nested technical corruption gate: '+str(integrity))
    config = PostRenderQAConfig()
    collector = _IssueCollector()
    def run(command, timeout):
        return subprocess.run(command, capture_output=True, text=True,
            timeout=min(timeout,_remaining(context)), check=False)
    tools = _probe_and_decode(proxy.output, config=config, runner=run, collector=collector)
    checks = _validate_ffprobe_against_plan(tools, {'render':{'video_codec':'libx264'}},
        proxy.to_json(), config, collector)
    if collector.issues or not checks.get('available') or not all(checks['checks'].values()):
        raise ValueError('scene review full-decode/stream/plan technical gate: '+str([i.code for i in collector.issues]))
    return dict(technical_only=True, professional_qa_passed=False, nested_proxy_integrity=integrity,
        final_media=checks, full_decode=tools, issue_codes=[],
        legacy_subject_counts_are_empty_observations_not_person_absence=True)


def _witness_plan(context, proxy, first, end, output):
    # Make a bounded independent encode of ONLY changed cells. Rebase output
    # time, never native source time; prove exact schedules again after slicing.
    offsets=[0]
    for count in segment_frame_counts(proxy): offsets.append(offsets[-1]+count)
    if first not in offsets or end not in offsets:
        raise ValueError('witness slice needs actual owned cell boundaries')
    a,b=offsets.index(first),offsets.index(end)
    start=proxy.segments[a].master_start
    clipped=replace(proxy,output=output,master_audio_start=proxy.master_audio_start+start,
        segments=tuple(replace(s,master_start=s.master_start-start) for s in proxy.segments[a:b]),
        transitions=proxy.transitions[a:b-1])
    reference=family._schedule(proxy,context.domain)
    witness=family._schedule(clipped,context.domain)
    if len(witness)!=end-first:
        raise ValueError('witness output count differs')
    for tick,row in witness.items():
        original=reference[first+tick]
        if (row.source_index,row.native_frame_index,row.terminal_clone)!=(
                original.source_index,original.native_frame_index,original.terminal_clone):
            raise ValueError('witness encode would use another native source/time')
    return clipped


def compare_decoded_witness(full_path, witness_path, first, end, context, incumbent_witness_path):
    """Lossy-codec roundtrip check of EVERY changed output tick, not a hash claim.

    Independent reference encode is reconstructed from original source/time.
    Pixel-error bounds are technical tolerances, not editorial quality scores.
    Hashes bind the actual decoded images; compressed/raw SHA equality is NOT
    claimed. Both decoder streams are sequential, never unchecked random seek.
    """
    cap, ref=cv2.VideoCapture(str(full_path)),cv2.VideoCapture(str(witness_path))
    old_ref=cv2.VideoCapture(str(incumbent_witness_path))
    rows=[]; distinguishable=0
    try:
        for tick in range(end):
            _remaining(context)
            ok,image=cap.read()
            if not ok: raise ValueError('proposal final decode ended before changed ticks')
            if tick<first: continue
            ok,expected=ref.read()
            if not ok or expected.shape!=image.shape:
                raise ValueError('proposal witness decode dimensions/count differ')
            old_ok,old_image=old_ref.read()
            if not old_ok or old_image.shape!=image.shape:
                raise ValueError('incumbent witness decode dimensions/count differ')
            change=float(np.abs(expected.astype(np.int16)-old_image.astype(np.int16)).mean())
            old_error=float(np.abs(image.astype(np.int16)-old_image.astype(np.int16)).mean())
            error=np.abs(image.astype(np.int16)-expected.astype(np.int16))
            mean=float(error.mean()); p99=float(np.percentile(error,99))
            if mean>5. or p99>24.:
                raise ValueError(f'proposal pixels differ from selected source/camera at {tick}: mean={mean},p99={p99}')
            if change > 10.:
                if old_error <= mean+5.:
                    raise ValueError('proposal encoded pixels match incumbent rather than selected camera')
                distinguishable+=1
            rows.append(dict(output_tick=tick,decoded_pixels_sha256=sha256(image.tobytes()).hexdigest(),
                candidate_incumbent_mean_code_difference=change,incumbent_mean_absolute_code_error=old_error,
                reference_decoded_pixels_sha256=sha256(expected.tobytes()).hexdigest(),
                mean_absolute_code_error=mean,p99_absolute_code_error=p99))
        if ref.read()[0] or old_ref.read()[0]: raise ValueError('proposal witness has trailing output frames')
    finally:
        cap.release();ref.release();old_ref.release()
    if not distinguishable:
        raise ValueError('proposal pixels are not distinguishable from incumbent within codec tolerance')
    if [r['output_tick'] for r in rows]!=list(range(first,end)):
        raise ValueError('proposal final decoded ticks incomplete')
    return dict(contract='independent_source_render_roundtrip_v1',rows=rows,
        distinguishable_from_incumbent_frames=distinguishable,
        maximum_mean_code_error=5.,maximum_p99_code_error=24.,semantic_quality_approved=False)


@isolated_optional()
def render_review(context, plan, receipt, output):
    from .rendered_proxy_review import render_and_review_proxy, derive_proxy_render_plan
    _fresh(context)
    _rederive(context,plan,receipt)
    output=Path(output)
    first,end=receipt['bounds'][1]
    proxy,review=render_and_review_proxy(plan,output,geometry_provider=_not_requested_geometry,
        video_codec='libx264',maximum_runtime_seconds=_remaining(context))
    if not output.is_file() or not 0<output.stat().st_size<=MAX_ARTIFACT_BYTES:
        raise ValueError('scene review output absent/oversize')
    technical=_technical_qa(context,proxy,review)
    with tempfile.TemporaryDirectory(prefix='scene-source-witness-',dir=output.parent) as folder:
        witness=_witness_plan(context,proxy,first,end,Path(folder)/'witness.mp4')
        render(witness,video_codec='libx264',video_bitrate='8M',video_maxrate='12M',
            video_buffer_size='16M',audio_bitrate='128k',maximum_runtime_seconds=_remaining(context))
        original_proxy=derive_proxy_render_plan(context.incumbent.render_plan,output)
        incumbent_witness=_witness_plan(context,original_proxy,first,end,Path(folder)/'incumbent-witness.mp4')
        render(incumbent_witness,video_codec='libx264',video_bitrate='8M',video_maxrate='12M',
            video_buffer_size='16M',audio_bitrate='128k',maximum_runtime_seconds=_remaining(context))
        corroboration=compare_decoded_witness(output,witness.output,first,end,context,incumbent_witness.output)
    _fresh(context)
    return dict(proxy_plan=proxy.to_json(),technical_qa=technical,
        proxy_review=review.payload,changed_pixel_witness=corroboration,
        video_sha256=family._file_hash(output),video_bytes=output.stat().st_size,
        encoded_work_bound=dict(preview_output_frames=proxy.output_frame_count,
            witness_output_frames=2*(end-first),source_observation_budget_is_separate=True,
            full_video_decode_passes=2,changed_region_decode_frames=end+2*(end-first),
            shared_absolute_deadline=context.deadline))


def _write(path,value):
    data=json.dumps(family.plain(value),ensure_ascii=False,indent=2,allow_nan=False).encode()
    if len(data)>MAX_MANIFEST_BYTES: raise ValueError('scene review metadata byte budget exceeded')
    path=Path(path); temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_bytes(data);temporary.replace(path)


def _publish_review_clock(staging, video, proxy_plan, video_sha256):
    """Preserve the renderer's content-bound clock across staging rename.

    Only the output destination changes. Source clocks, filter graph and all
    source/camera sampling remain the actual renderer's archived evidence.
    The manifest cannot expose a modern video while its required sidecar is
    left behind under the unpublished filename.
    """
    from .render_clock_artifact import render_clock_path, RENDER_CLOCK_ARTIFACT_CONTRACT
    original=render_clock_path(staging)
    if original.is_symlink() or not original.is_file() or original.stat().st_size>MAX_MANIFEST_BYTES:
        raise ValueError('scene review missing bounded original renderer clock')
    clock=json.loads(original.read_text())
    if (clock.get('artifact_contract')!=RENDER_CLOCK_ARTIFACT_CONTRACT
            or clock.get('video_content_sha256')!=video_sha256
            or clock.get('quality_approval') is not False
            or family.digest(clock.get('plan'))!=family.digest(proxy_plan)):
        raise ValueError('scene review renderer clock differs from encoded source plan')
    destination=render_clock_path(video)
    if destination.exists() or destination.is_symlink():
        raise ValueError('scene review renderer clock destination already exists')
    clock=dict(clock,plan=dict(proxy_plan,output=str(video)))
    _write(destination,clock)
    return dict(render_clock_path=str(destination),render_clock_sha256=family._file_hash(destination))


def attempt_review_proposal(*, controller, reporter, archive_path, **arguments):
    """Normal app publication, separate from both incumbent and result events.

    Called before costly strict all-person certification; same controller cutoff
    and at most one full preview plus two short independent witness encodes.
    No retries, new model loader, or approval promotion.
    The existing full final QA and output disposition remain untouched.
    """
    archive_path=Path(archive_path)
    bundle=controller.incumbent
    if getattr(bundle,'performer_family_review',None) is not None: return None
    choices=getattr(bundle.visual_timeline,'choices',())
    if sum(getattr(c.shot_type,'value',c.shot_type) in ('wide','group_wide') for c in choices)<3:
        return None
    video=archive_path.with_suffix('.mp4')
    staging=video.with_name(video.stem+'.unpublished.mp4')
    from .render_clock_artifact import render_clock_path
    if any(p.exists() or p.is_symlink() for p in (
            archive_path,video,staging,render_clock_path(video),render_clock_path(staging))):
        reporter.log('scene review destination already exists; existing artifact preserved')
        return None
    archive_path.parent.mkdir(parents=True,exist_ok=True)
    record=dict(contract=CONTRACT,status='UNAVAILABLE',quality_approved=False,
        delivery_completion_message_allowed=False,artifact_role='scene_review_only',semantics=dict(UNKNOWN))
    def operation():
        nonlocal record
        context=freeze_context(controller=controller,**arguments)
        reporter.log('구도 제안: 기존 소스·시간표로 후보 탐색을 시작합니다.')
        proposal,rejection=propose(context)
        if proposal is None:
            record['rejection']=rejection
            return None
        plan,receipt=proposal
        reporter.log('구도 제안: 변경 후보를 선택했습니다. 원본 프레임을 독립 재검증합니다.')
        family.digest(receipt)  # Nonfinite/malformed child JSON is not archival truth.
        record.update(status='PROPOSED_NOT_RENDERED',original_metadata=family.metadata_payload(context.incumbent),
            original_plan=context.incumbent.render_plan.to_json(),candidate_plan=plan.to_json(),receipt=receipt)
        controller.validate_plan(plan)
        _rederive(context,plan,receipt)
        replay_work=replay(context,plan,receipt)
        if controller.transaction_token()!=context.token:
            raise ValueError('scene review incumbent changed before encode')
        receipt=dict(receipt,work=replay_work)
        reporter.log('구도 제안: 독립 검증 통과. 전체 480p 검토 영상 렌더와 기술 검사를 시작합니다.')
        result=render_review(context,plan,receipt,staging)
        _fresh(context)
        if controller.transaction_token()!=context.token:
            raise ValueError('scene review incumbent changed before publication')
        # Recheck both source and output after the worker exits. No CAS/adoption.
        if result['video_sha256']!=family._file_hash(staging):
            raise ValueError('scene review video changed after final decode')
        clock_binding=_publish_review_clock(staging,video,result['proxy_plan'],result['video_sha256'])
        staging.replace(video)
        render_clock_path(staging).unlink()
        proxy_plan=dict(result.pop('proxy_plan'),output=str(video))
        record=dict(record,status='REVIEW_REQUIRED',video_path=str(video),
            video_sha256=result['video_sha256'],video_bytes=result['video_bytes'],
            original_metadata=family.metadata_payload(context.incumbent),
            original_plan=context.incumbent.render_plan.to_json(),candidate_plan=plan.to_json(),
            proxy_plan=proxy_plan,receipt=receipt,**clock_binding,
            **{k:v for k,v in result.items() if k not in ('video_sha256','video_bytes')})
        _write(archive_path,record)
        return dict(contract=CONTRACT,status='REVIEW_REQUIRED',artifact_role='scene_review_only',
            quality_approved=False,delivery_completion_message_allowed=False,
            video_path=str(video),video_sha256=record['video_sha256'],
            manifest_path=str(archive_path),manifest_sha256=family._file_hash(archive_path))
    result=controller.run_optional('scene_review_proposal',operation)
    if result is None:
        # An encoded partial/stale file is not playable/exposed. A failed
        # optional step does not change the original complete owner.
        staging.unlink(missing_ok=True)
        video.unlink(missing_ok=True)
        render_clock_path(staging).unlink(missing_ok=True)
        render_clock_path(video).unlink(missing_ok=True)
        record.update(status='UNAVAILABLE',controller_attempt=controller.attempts[-1] if controller.attempts else None)
        _write(archive_path,record)
    else:
        reporter.emit('scene_review_proposal',scene_review_proposal=result,
            message='480p 구도 제안: 출연자·동작·박자·편집 품질 미승인. 기존 결과와 별도입니다.')
    return result
