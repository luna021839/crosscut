"""Opt-in GROUP C / R0-camera / R1 numeric integration review owner.

This is deliberately NOT a semantic/body owner. Pixel contrast moments rank
inspectable camera proposals; unavailable identity/phase/parts remain null.
All source clocks and cuts are unchanged. Optional native work, live parent
replay and CAS run under the *same* controller deadline. Unreplayable existing
receipt families are rejected, not erased. No model or song-specific inputs.
Ordinary app requests do NOT enable this contrast-only selection policy.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any, Mapping

import cv2
import numpy as np

from .editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from .editorial_transform_events import rendered_transform_event_records
from .group_actual_shots import bounded_actual_shot_windows, visible_group_cut_inventory, visible_group_shots
from .group_interval_materialization import split_group_render_intervals, _slice_camera
from .group_joint_observations import GroupJointObservationSession, _file_identity
from .group_joint_source_camera import JointGroupWorkBudget
from .group_source_domain import GroupSourceDomain
from .group_transition import GroupTransitionPlan
from .optional_native_worker import isolated_optional, current_optional_deadline
from .post_render_perceptual_quality import _reference_view_for_span
from .prescribed_render_geometry import PrescribedGeometrySampler
from .render import EditorialTransformKeyframe, EditorialTransformTrack, segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import probe_render_frame_clock
from .timeline import ShotType
from .timeline_review import review_timeline
from .visual_boundary_repair import _thaw_audit_value
from .visual_score import VisualTimelineChoice

CONTRACT = 'normal_interval_pixel_review_owner_v1'
MOTION_BASIS = 'normal_interval_review_numeric_tracks_v1'
QUALITY_CODE = 'normal_interval_review_quality_unapproved'
# Resource limits describe this slice, not musical time or semantic thresholds.
MAX_OUTPUT_FRAMES = 36000
MAX_REFRESHED_CELLS = 24
MAX_SAMPLES = 3 * MAX_REFRESHED_CELLS + 8
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
UNKNOWN = dict(identity=None, body_visibility=None, required_parts=None,
    motion_phase=None, head_orientation=None, semantic_correspondence=None)


def _plain(value):
    return _thaw_audit_value(value)


def _digest(value):
    return sha256(json.dumps(_plain(value), sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _plan_payload(plan):
    value = plan.to_json() if hasattr(plan, 'to_json') else dict(plan)
    # Immutable QA snapshots relocate the artifact, not source pixels/geometry.
    value = dict(value)
    value.pop('output', None)
    return value


def _metadata_payload(bundle):
    value = bundle.to_dict()
    value.pop('normal_interval_review', None)
    return value


def _dependencies():
    root = Path(__file__).parent
    names = ('group_normal_interval_owner.py', 'group_interval_materialization.py',
        'group_actual_shots.py', 'render.py', 'render_geometry.py',
        'render_boundary_clock.py', 'render_frame_schedule.py',
        'exact_native_frame_reader.py', 'prescribed_render_geometry.py',
        'post_render_perceptual_quality.py', 'group_joint_observations.py',
        'group_source_domain.py', 'timeline_review.py', 'editorial_transform_events.py',
        'visual_score.py', 'group_boundary_grammar_adapter.py', 'group_transition.py',
        'review_first_selection_contract.py', 'pipeline.py', 'editorial_motion.py',
        'timeline.py', 'timewarp.py', 'post_render_qa.py', 'review_first_runtime.py',
        'request.py')
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in names}


@dataclass(frozen=True, slots=True)
class NormalIntervalContext:
    incumbent: Any
    domain: GroupSourceDomain
    source_identities: tuple[tuple[int, ...], ...]
    dependencies: tuple[tuple[str, str], ...]
    token: Any
    deadline: float


def _supported(bundle):
    """Do not make old independent authority disappear to pass admission."""
    blocked = ('boundary_repairs', 'face_match_plan', 'boundary_affine_plan',
        'boundary_look_match', 'dynamic_crop_application', 'output_geometry_deferral',
        'splice_graph_authorization', 'cut_local_sync', 'stage_edit_dynamic_crop_version',
        'stage_edit_dynamic_crop_v3_decodes', 'dynamic_crop_motion_timeline_v3',
        'normal_interval_review')
    present = [name for name in blocked if getattr(bundle, name, None) is not None]
    motion = bundle.editorial_motion_plan
    if motion is None:
        present.append('complete_motion_inventory_missing')
    else:
        for name in ('boundary_subject_alignment_plan', 'shot_context_cameras',
                'coupled_shot_cameras', 'accepted_boundary_indices',
                'natural_match_boundary_indices', 'boundary_residual_attention_mismatches',
                'required_boundary_failures', 'member_roll_audit',
                'member_containment_trial_audit', 'skipped_boundary_reasons'):
            if getattr(motion, name):
                present.append('editorial_motion.' + name)
        if motion.boundary_match_count or motion.affine_boundary_match_count:
            present.append('editorial_motion.boundary_proof_counts')
    if any(c.transition_evidence_authority_from_previous is not None
            or c.transition_evidence_contract_from_previous is not None
            for c in bundle.visual_timeline.choices):
        present.append('timeline.transition_evidence')
    if (bundle.render_plan.authorized_shot_video_clocks
            or any(s.boundary_look_track is not None or s.look_adjustment is not None or s.outgoing_overlap_transform is not None
                for s in bundle.render_plan.segments)):
        present.append('render.local_clock_or_look_or_overlap')
    if present:
        raise ValueError('normal interval owner cannot rebind existing authority: ' + ', '.join(present))


def freeze_context(*, controller, eligible_indices, offsets, timewarps, scene_cut_times):
    incumbent = controller.incumbent
    _supported(incumbent)
    from .pipeline import _validate_review_first_group_bundle
    _validate_review_first_group_bundle(incumbent, eligible_indices=eligible_indices,
        offsets=offsets, timewarps=timewarps,
        common_start=incumbent.render_plan.master_audio_start,
        phrase_times=incumbent.phrase_times)
    deadline = current_optional_deadline()
    if deadline is None or deadline <= monotonic():
        raise ValueError('normal interval owner requires the shared optional transaction deadline')
    sources = incumbent.render_plan.sources
    identities = tuple(_file_identity(sources[i]) for i in eligible_indices)
    clocks = []
    for i in eligible_indices:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ValueError('normal interval native clock preparation exhausted its deadline')
        clocks.append(probe_render_frame_clock(sources[i], timeout_seconds=min(10., remaining)))
    if identities != tuple(_file_identity(sources[i]) for i in eligible_indices):
        raise ValueError('normal interval source changed during native clock preparation')
    domain = GroupSourceDomain(tuple(sources), tuple(eligible_indices), tuple(offsets),
        tuple(timewarps), tuple(tuple(cuts) for cuts in scene_cut_times), tuple(clocks))
    return NormalIntervalContext(incumbent, domain, identities,
        tuple(sorted(_dependencies().items())), controller.transaction_token(), deadline)


def _fresh(context):
    if monotonic() >= context.deadline:
        raise ValueError('normal interval transaction deadline expired')
    if (tuple(_file_identity(p) for p in context.domain.eligible_sources) != context.source_identities
            or tuple(sorted(_dependencies().items())) != context.dependencies):
        raise ValueError('normal interval live source/code preimage changed')


def _scaffold(context):
    before = context.incumbent.render_plan
    if before.output_frame_count > MAX_OUTPUT_FRAMES:
        raise ValueError('normal interval whole-output numeric replay budget exceeded')
    windows = bounded_actual_shot_windows(before, context.domain, maximum_windows=1,
        maximum_mutable_shots=1, maximum_mutable_frames=300)
    if not windows:
        raise ValueError('normal interval needs three observed actual epochs, not canonical cell labels')
    window = windows[0]
    ticks = tuple(sorted({t for shot in window for t in
        (shot.first_output_frame, shot.end_output_frame) if 0 < t < before.output_frame_count}))
    scaffold = split_group_render_intervals(plan=before, domain=context.domain,
        split_output_frames=ticks, maximum_replay_frames=MAX_OUTPUT_FRAMES)
    r0 = window[1]
    changed = tuple(row.new_segment_index for row in scaffold.cell_map
        if r0.first_output_frame <= row.first_output_frame < r0.end_output_frame)
    if not changed or any(scaffold.plan.segments[i].editorial_transform is not None for i in changed):
        raise ValueError('normal interval cannot replace an existing authored R0 camera without replay')
    refreshed = set(scaffold.refreshed_segment_indices) | set(changed)
    # Both incident edge descriptors belong to the refreshed topology.
    for index in tuple(refreshed):
        refreshed.update(i for i in (index-1, index+1) if 0 <= i < len(scaffold.cell_map))
    if len(refreshed) > MAX_REFRESHED_CELLS:
        raise ValueError('normal interval observation-cell budget exceeded')
    return scaffold, window, changed, tuple(sorted(refreshed))


def _recipes():
    # Rank-only candidates; retain the existing default QA safety envelope:
    # zoom <= 1.18, center shift <= .025, no roll. No gate is relaxed.
    result = [None]
    for zoom in (1.04, 1.10, 1.16):
        margin = (1. - 1. / zoom) / 2.
        for direction in (-1., 0., 1.):
            result.append(dict(zoom=zoom, end_zoom=zoom + .015,
                center_x=.5 + direction * min(.025, margin * .8), center_y=.5))
    return tuple(result)


def _overlay(scaffold, window, changed, recipe):
    if recipe is None:
        return scaffold.plan
    if recipe not in _recipes()[1:]:
        raise ValueError('normal interval camera recipe is outside the bounded grammar')
    r0 = window[1]
    duration = (r0.end_output_frame-r0.first_output_frame) / scaffold.plan.fps
    track = EditorialTransformTrack((
        EditorialTransformKeyframe(0., recipe['zoom'], recipe['center_x'], recipe['center_y']),
        EditorialTransformKeyframe(duration, recipe['end_zoom'], recipe['center_x'], recipe['center_y']),
    ), reason='normal_interval_unapproved_pixel_rank_camera')
    segments = list(scaffold.plan.segments)
    for index in changed:
        row = scaffold.cell_map[index]
        segment = segments[index]
        segments[index] = replace(segment, editorial_transform=_slice_camera(track,
            offset_seconds=(row.first_output_frame-r0.first_output_frame)/scaffold.plan.fps,
            duration=segment.duration))
    return replace(scaffold.plan, segments=tuple(segments))


def _schedule(plan, domain):
    rows = native_frame_clocks_for_output(plan, domain.clocks_by_request,
        range(plan.output_frame_count))
    if len(rows) != plan.output_frame_count or any(row.terminal_clone for row in rows.values()):
        raise ValueError('normal interval global native schedule is incomplete or terminal-cloned')
    return rows


def _schedule_digest(rows):
    return _digest([(tick, row.source_index, row.native_frame_index, row.source_request_seconds,
        row.terminal_clone) for tick, row in sorted(rows.items())])


def _pixel_descriptor(frame):
    """Low-level contrast moments only; no person/face/body interpretation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.
    weight = np.abs(gray - float(np.median(gray)))
    mass = float(weight.sum())
    if not math.isfinite(mass) or mass < .0001 * weight.size:
        return None
    yy, xx = np.indices(gray.shape, dtype=np.float64)
    xx /= max(1, gray.shape[1]-1)
    yy /= max(1, gray.shape[0]-1)
    x, y = float((xx*weight).sum()/mass), float((yy*weight).sum()/mass)
    sx = math.sqrt(float(((xx-x)**2*weight).sum()/mass))
    sy = math.sqrt(float(((yy-y)**2*weight).sum()/mass))
    return (x, y, sx, sy)


def _measure(session, plan, domain, schedule, ticks):
    ends = np.cumsum(segment_frame_counts(plan)).tolist()
    result = []
    for tick in sorted(set(ticks)):
        session.budget.check()
        row = schedule[tick]
        frame = session.native_pixels(row.source_index, row.native_frame_index)
        if frame is None:
            raise ValueError('normal interval exact native frame unavailable; no nearby-frame substitution')
        index = bisect_right(ends, tick)
        local = (tick - (ends[index-1] if index else 0)) / plan.fps
        sampler = PrescribedGeometrySampler(SimpleNamespace(frame_rate=float(
            domain.clocks_by_request[row.source_index].frame_rate)), plan.segments[index],
            master_start=plan.master_audio_start)
        projected = _reference_view_for_span(frame, sampler.span, relative_time=local,
            size=(160, 90), render_size=(plan.width, plan.height))
        descriptor = _pixel_descriptor(projected)
        result.append(dict(output_frame=tick, source_index=row.source_index,
            native_frame_index=row.native_frame_index,
            source_request_seconds=row.source_request_seconds,
            native_sha256=sha256(frame.tobytes()).hexdigest(),
            native_shape=list(frame.shape), projected_sha256=sha256(projected.tobytes()).hexdigest(),
            pixel_contrast=float(cv2.cvtColor(projected, cv2.COLOR_BGR2GRAY).std()/255.),
            contrast_moments=list(descriptor) if descriptor is not None else None))
    return result


def _rank(measurements, window):
    c, r0, r1 = window
    by_tick = {row['output_frame']: row['contrast_moments'] for row in measurements}
    pairs = ((c.end_output_frame-1, r0.first_output_frame),
        (r0.end_output_frame-1, r1.first_output_frame))
    values = []
    for a, b in pairs:
        if by_tick.get(a) is None or by_tick.get(b) is None:
            return None
        values.append(float(np.linalg.norm(np.asarray(by_tick[a])-by_tick[b])))
    return sum(values) / len(values)


def _observation_ticks(scaffold, window, refreshed):
    ticks = {window[0].end_output_frame-1, window[1].first_output_frame,
        window[1].end_output_frame-1, window[2].first_output_frame}
    for index in refreshed:
        row = scaffold.cell_map[index]
        ticks.update((row.first_output_frame, (row.first_output_frame+row.end_output_frame-1)//2,
            row.end_output_frame-1))
    if len(ticks) > MAX_SAMPLES:
        raise ValueError('normal interval native observation budget exceeded')
    return tuple(sorted(ticks))


def _complete(context, scaffold, window, changed, refreshed, plan, measurements, ranks, recipe):
    """Variable topology metadata, not a fixed-cardinality source-index edit."""
    from .pipeline import _deterministic_exact_shot_clock_from_timeline
    old = context.incumbent
    by_tick = {row['output_frame']: row for row in measurements}
    choices = []
    epochs = visible_group_shots(plan, context.domain)
    native_events = tuple(event for event in visible_group_cut_inventory(plan, context.domain)
        if event.kind == 'native_camera_cut')
    for row, segment in zip(scaffold.cell_map, plan.segments, strict=True):
        prior = old.visual_timeline.choices[row.old_segment_index]
        midpoint = (row.first_output_frame+row.end_output_frame-1)//2
        epoch = next(shot for shot in epochs if shot.first_output_frame <= midpoint < shot.end_output_frame)
        shot_id = f'normal-native-epoch-{epoch.first_output_frame}-{epoch.end_output_frame}'
        internal = tuple(plan.master_audio_start+segment.master_start
            +(event.first_incoming_output_frame-row.first_output_frame)/plan.fps
            for event in native_events if row.first_output_frame < event.first_incoming_output_frame < row.end_output_frame)
        if row.new_segment_index not in refreshed:
            choices.append(replace(prior, segment_index=row.new_segment_index,
                shot_id=shot_id, internal_scene_cut_master_seconds=internal))
            continue
        taps = (row.first_output_frame, (row.first_output_frame+row.end_output_frame-1)//2,
            row.end_output_frame-1)
        # Numeric camera inventory has no measured semantic shot scale/type.
        # Original measurements remain in the retained complete predecessor.
        moments = [by_tick[t]['contrast_moments'] for t in taps]
        composition = (1. - min(1., float(np.mean([math.hypot(v[0]-.5, v[1]-.5)
            for v in moments if v is not None]))*2.)) if all(v is not None for v in moments) else None
        previous = by_tick.get(row.first_output_frame-1)
        entry = by_tick[row.first_output_frame]['contrast_moments']
        boundary_score = (1. - min(1., float(np.linalg.norm(np.asarray(previous['contrast_moments'])
            - np.asarray(entry))))) if previous is not None and previous['contrast_moments'] is not None and entry is not None else None
        choices.append(VisualTimelineChoice(segment_index=row.new_segment_index,
            source_index=context.domain.eligible_index(segment.source_index),
            canonical_master_start=plan.master_audio_start+segment.master_start,
            canonical_master_end=plan.master_audio_start+segment.master_start+segment.duration,
            source_local_start=segment.source_start, source_local_end=segment.source_end,
            quality_score=float(np.mean([by_tick[t]['pixel_contrast'] for t in taps])),
            composition_score=composition, boundary_score=boundary_score,
            shot_scale=None, shot_type=ShotType.UNKNOWN,
            shot_id=shot_id, internal_scene_cut_master_seconds=internal,
            shot_scale_bases=('unknown_pixel_only',)*3))
    timeline = replace(old.visual_timeline, choices=tuple(choices),
        optimizer_score=old.visual_timeline.optimizer_score,
        analysis_basis='normal_interval_global_native_pixel_observations_v1',
        candidate_generation_basis='bounded_actual_epoch_camera_rank_only_v1',
        candidate_review_summaries=(), geometry_basis='prescribed_render_geometry_v1',
        source_clock_basis='renderer_global_native_epoch_clock_v1',
        actual_epoch_clock=epoch_clock_payload(plan, context.domain))
    # A legacy scalar selector score is retained only as selector history, not
    # a fresh quality or boundary measurement; see receipt's explicit scope.
    events = rendered_transform_event_records(plan.segments, plan.fps)
    motion = EditorialMotionPlan(segments=tuple(EditorialMotionSegment(i, c.source_index,
        plan.segments[i].editorial_transform, ('numeric_inventory_not_semantic_authority',))
        for i, c in enumerate(choices)), boundary_match_count=0,
        zoom_event_count=len(events['zoom']), pan_event_count=len(events['pan']),
        roll_event_count=len(events['roll']), blocked_dissolve_segment_count=0,
        # Numeric framing targets are not authored heuristic/repair gestures.
        # This zero is a structural inventory, NOT missing semantic motion.
        # Actual renderer-visible counts above remain independently replayed.
        authored_event_counts=(('zoom', 0), ('pan', 0), ('roll', 0)),
        analysis_basis=MOTION_BASIS, eligible_request_indices=context.domain.eligible_indices)
    review = review_timeline(timeline, available_source_count=len(context.domain.eligible_indices))
    candidate = replace(old, visual_timeline=timeline, render_plan=plan,
        timeline_review=review, pre_render_score=float(review.weighted_score),
        editorial_motion_plan=motion, group_transition_plan=GroupTransitionPlan(plan.transitions,
            analysis_basis='normal_interval_numeric_hard_cuts_quality_unapproved'),
        stage_edit_exact_shot_clock=_deterministic_exact_shot_clock_from_timeline(timeline,
            section_plan=old.section_plan, phrase_times=old.phrase_times),
        candidate_origin=CONTRACT, cadence_profile=CONTRACT,
        candidate_lineage=(*old.candidate_lineage, dict(stage=CONTRACT,
            predecessor_sha256=_digest(dict(owner=_metadata_payload(old), plan=_plan_payload(old.render_plan))),
            changed_segment_indices=list(changed), refreshed_segment_indices=list(refreshed),
            preserved_original_measurements_in_predecessor=True,
            numeric_inventory_only=True, quality_approved=False)))
    receipt = dict(contract=CONTRACT, plan_sha256=_digest(_plan_payload(plan)),
        metadata_sha256=_digest(_metadata_payload(candidate)),
        predecessor_sha256=_digest(dict(owner=_metadata_payload(old), plan=_plan_payload(old.render_plan))),
        source_domain=context.domain.to_dict(),
        source_identities=[list(v) for v in context.source_identities],
        dependencies=dict(context.dependencies),
        window=[shot.to_dict() for shot in window], cell_map=[asdict(row) for row in scaffold.cell_map],
        changed_segment_indices=list(changed), refreshed_segment_indices=list(refreshed),
        recipe=recipe, rankings=ranks, raw_samples=measurements,
        global_native_schedule_sha256=_schedule_digest(_schedule(plan, context.domain)),
        actual_cuts=[cut.to_dict() for cut in visible_group_cut_inventory(plan, context.domain)],
        uncertain_semantics=dict(UNKNOWN), quality_approved=False,
        final_quality_status='UNAPPROVED_REVIEW_CANDIDATE',
        observation_scope='sampled_exact_native_pixels_and_projected_contrast_moments_not_body_or_decoded_output',
        numeric_quality_scope='projected_pixel_contrast_and_moments_not_performer_or_editorial_quality',
        optimizer_score_scope='retained_selector_history_not_new_camera_quality',
        numeric_replay_scope='whole_output_schedule_and_fixed_C_R1_geometry',
        clock_domains=dict(window='integer_global_output_ticks',
            timeline='canonical_segment_clock_not_quantized_output_seconds'),
        native_runtime=dict(opencv=cv2.__version__, numpy=np.__version__),
        limits=dict(maximum_output_frames=MAX_OUTPUT_FRAMES, maximum_refreshed_cells=MAX_REFRESHED_CELLS,
            maximum_native_samples=MAX_SAMPLES, camera_recipes=len(_recipes())-1))
    return replace(candidate, normal_interval_review=receipt)


@isolated_optional()
def propose_normal_interval(context):
    _fresh(context)
    scaffold, window, changed, refreshed = _scaffold(context)
    schedule = _schedule(scaffold.plan, context.domain)
    budget = JointGroupWorkBudget(absolute_deadline=context.deadline, maximum_new_decodes=MAX_SAMPLES,
        maximum_projected_cache_bytes=4*1024*1024)
    session = GroupJointObservationSession(domain=context.domain, budget=budget)
    ticks = _observation_ticks(scaffold, window, refreshed)
    ranks, best = [], None
    try:
        baseline_cost = None
        for number, recipe in enumerate(_recipes()):
            plan = _overlay(scaffold, window, changed, recipe)
            measurements = _measure(session, plan, context.domain, schedule, ticks)
            cost = _rank(measurements, window)
            ranks.append(dict(recipe_index=number, contrast_moment_cost=cost,
                semantic_motion_cost=None, identity_cost=None))
            if number == 0:
                baseline_cost = cost
            elif cost is not None and baseline_cost is not None and cost < baseline_cost-1e-6:
                if best is None or cost < best[0]:
                    best = (cost, plan, measurements, recipe)
        if best is None:
            raise ValueError('normal interval has no measured pixel-rank improvement; UNKNOWN is not zero cost')
        _, plan, measurements, recipe = best
        candidate = _complete(context, scaffold, window, changed, refreshed, plan, measurements, ranks, recipe)
        _fresh(context)
        return candidate
    finally:
        session.close()


def validate_bundle_binding(bundle):
    """Cheap downstream guard: no stale complete bundle after another owner."""
    receipt = _plain(getattr(bundle, 'normal_interval_review', None))
    marked = (bundle.candidate_origin == CONTRACT or
        getattr(bundle.editorial_motion_plan, 'analysis_basis', None) == MOTION_BASIS)
    if receipt is None:
        if marked:
            raise ValueError('normal interval receipt was erased from the complete bundle')
        return
    if (receipt.get('contract') != CONTRACT or receipt.get('quality_approved') is not False
            or receipt.get('uncertain_semantics') != UNKNOWN
            or receipt.get('plan_sha256') != _digest(_plan_payload(bundle.render_plan))
            or receipt.get('metadata_sha256') != _digest(_metadata_payload(bundle))):
        raise ValueError('normal interval complete plan/metadata/UNKNOWN binding changed downstream')


def validate_parent(candidate, context):
    """Parent owns grammar/topology, clock preservation, fresh pixel replay and CAS.

    Re-observation uses a new killable process, never the worker's cached raster
    or a serialized passed flag. The shared scalar routines remain a common-mode
    limitation; no independent semantic measurement is claimed.
    """
    _fresh(context)
    validate_bundle_binding(candidate)
    scaffold, window, changed, refreshed = _scaffold(context)
    receipt = _plain(candidate.normal_interval_review)
    expected_plan = _overlay(scaffold, window, changed, receipt['recipe'])
    if candidate.render_plan != expected_plan:
        raise ValueError('normal interval parent rejected out-of-grammar geometry or source changes')
    before_schedule = _schedule(context.incumbent.render_plan, context.domain)
    after_schedule = _schedule(candidate.render_plan, context.domain)
    if _schedule_digest(before_schedule) != _schedule_digest(after_schedule):
        raise ValueError('normal interval parent rejected changed global native output sampling')
    if visible_group_cut_inventory(context.incumbent.render_plan, context.domain) != \
            visible_group_cut_inventory(candidate.render_plan, context.domain):
        raise ValueError('normal interval parent rejected changed actual cuts')
    # Numeric scaffold is independently reconstructed from incumbent. R0 is
    # the only overlay; C/R1/unrelated cell samples and sampling canvas stay fixed.
    if (receipt['source_domain'] != context.domain.to_dict()
            or receipt['source_identities'] != [list(v) for v in context.source_identities]
            or receipt['dependencies'] != dict(context.dependencies)):
        raise ValueError('normal interval parent source/domain/code preimages disagree')
    replayed = propose_normal_interval(context)
    if (candidate.to_dict() != replayed.to_dict() or candidate.render_plan != replayed.render_plan):
        raise ValueError('normal interval parent fresh pixel/ranking/metadata replay disagrees')
    from .pipeline import _validate_review_first_group_bundle
    _validate_review_first_group_bundle(candidate, eligible_indices=context.domain.eligible_indices,
        offsets=context.domain.audio_offsets, timewarps=context.domain.source_timewarps,
        common_start=context.incumbent.render_plan.master_audio_start,
        phrase_times=context.incumbent.phrase_times)
    _fresh(context)


def attempt_normal_interval(*, controller, eligible_indices, offsets, timewarps,
        scene_cut_times, archive_path):
    """Called by normal analyze even when a learned candidate already exists."""
    def transaction():
        context = freeze_context(controller=controller, eligible_indices=eligible_indices,
            offsets=offsets, timewarps=timewarps, scene_cut_times=scene_cut_times)
        candidate = propose_normal_interval(context)
        # Archive full predecessor and proposal before CAS; not a quality seal.
        archive = dict(contract=CONTRACT, status='proposal_pending_parent_replay',
            original_bundle=context.incumbent.to_dict(), original_plan=context.incumbent.render_plan.to_json(),
            candidate_bundle=candidate.to_dict(), candidate_plan=candidate.render_plan.to_json(),
            transaction_token=asdict(context.token), quality_approved=False)
        encoded = json.dumps(archive, sort_keys=True, allow_nan=False).encode()
        if len(encoded) > MAX_ARCHIVE_BYTES:
            raise ValueError('normal interval complete replay archive exceeds explicit byte budget')
        from .pipeline import _write_analysis_manifest_atomically
        _write_analysis_manifest_atomically(Path(archive_path), archive)
        controller.compare_and_promote(candidate, plan=candidate.render_plan, authority=CONTRACT,
            token=context.token, validate=lambda value: validate_parent(value, context))
        return candidate
    return controller.run_optional('normal_actual_epoch_camera_owner', transaction)


def validate_serialized_metadata(complete):
    if not isinstance(complete, Mapping):
        raise ValueError('normal interval selected complete bundle is missing')
    metadata = dict(complete)
    receipt = metadata.pop('normal_interval_review', None)
    if (not isinstance(receipt, Mapping) or receipt.get('contract') != CONTRACT
            or complete.get('candidate_origin') != CONTRACT
            or receipt.get('metadata_sha256') != _digest(metadata)
            or receipt.get('quality_approved') is not False
            or receipt.get('uncertain_semantics') != UNKNOWN
            or receipt.get('final_quality_status') != 'UNAPPROVED_REVIEW_CANDIDATE'):
        raise ValueError('normal interval serialized complete metadata/unknown semantics is inconsistent')
    return receipt


@isolated_optional()
def _replay_final_native(plan_payload, receipt, deadline):
    """Independent artifact consumer: no candidate object or cached worker pixels."""
    from .render_plan_serialization import render_plan_from_json
    from .render_frame_schedule import VerifiedRenderFrameClock
    from .exact_native_frame_reader import ExactNativeFrameReader
    from fractions import Fraction
    plan = render_plan_from_json(plan_payload)
    if receipt.get('plan_sha256') != _digest(_plan_payload(plan)):
        raise ValueError('normal interval final renderer differs from selected camera plan')
    if receipt.get('dependencies') != _dependencies():
        raise ValueError('normal interval final source/renderer code dependencies changed')
    domain = receipt['source_domain']
    indices = domain['eligible_request_indices']
    if (not isinstance(indices, list) or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(plan.sources) for i in indices)
            or domain['eligible_sources'] != [str(plan.sources[i]) for i in indices]):
        raise ValueError('normal interval final eligible source paths/order changed')
    before = [list(_file_identity(plan.sources[i])) for i in indices]
    if before != receipt['source_identities']:
        raise ValueError('normal interval final source file identity changed')
    budget = JointGroupWorkBudget(absolute_deadline=deadline, maximum_new_decodes=MAX_SAMPLES)
    clocks = {}
    for i, raw in zip(indices, domain['render_frame_clocks'], strict=True):
        budget.check()
        expected = VerifiedRenderFrameClock(Fraction(raw['frame_rate']), raw['frame_count'],
            raw['packet_grid_sha256'], raw['grid_basis'])
        observed = probe_render_frame_clock(plan.sources[i],
            timeout_seconds=max(.01, min(10., deadline-monotonic())))
        if observed != expected:
            raise ValueError('normal interval final verified native PTS grid changed')
        clocks[i] = observed
    if plan.output_frame_count > MAX_OUTPUT_FRAMES:
        raise ValueError('normal interval final numeric schedule exceeds budget')
    schedule = native_frame_clocks_for_output(plan, clocks, range(plan.output_frame_count))
    if _schedule_digest(schedule) != receipt['global_native_schedule_sha256']:
        raise ValueError('normal interval final global native sample inventory changed')
    samples = receipt['raw_samples']
    if (not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES
            or any(type(row.get('output_frame')) is not int
                or not 0 <= row['output_frame'] < plan.output_frame_count for row in samples)
            or len({row['output_frame'] for row in samples}) != len(samples)):
        raise ValueError('normal interval final exact native samples malformed or unbounded')
    captures, readers = {}, {}
    def pixels(source, native):
        budget.spend('new_decodes')
        if source not in readers:
            capture = cv2.VideoCapture(str(plan.sources[source]))
            captures[source] = capture
            if not capture.isOpened():
                raise ValueError('normal interval final native decoder unavailable')
            readers[source] = ExactNativeFrameReader(capture, clocks[source])
        if list(_file_identity(plan.sources[source])) != before[indices.index(source)]:
            raise ValueError('normal interval source changed during final native replay')
        return readers[source].frame_at(native)
    try:
        # The final render plan supplies explicit sample times. This adapter
        # carries ONLY native clocks, not invented scalar/timewarp alignment.
        measured = _measure(SimpleNamespace(budget=budget, native_pixels=pixels), plan,
            SimpleNamespace(clocks_by_request=clocks), schedule,
            tuple(row['output_frame'] for row in samples))
        if measured != samples:
            raise ValueError('normal interval final source pixels/projection/moments differ from parent-admitted samples')
        if before != [list(_file_identity(plan.sources[i])) for i in indices]:
            raise ValueError('normal interval sources changed during final replay')
        result = dict(contract=CONTRACT, binding_valid=True, source_live_replay=True,
            plan_sha256=receipt['plan_sha256'], metadata_sha256=receipt['metadata_sha256'],
            global_native_schedule_sha256=receipt['global_native_schedule_sha256'],
            sampled_native_pixel_count=len(samples), quality_approved=False,
            uncertain_semantics=dict(UNKNOWN), final_quality_status='UNAPPROVED_REVIEW_CANDIDATE',
            decoded_output_quality_certified=False,
            scope='source_native_samples_and_numeric_camera_binding_only')
        result['evidence_sha256'] = _digest(result)
        return result
    finally:
        for capture in captures.values():
            capture.release()


def replay_final_sources(plan_payload, receipt):
    from .optional_native_worker import optional_native_scope
    deadline = min(current_optional_deadline() or float('inf'), monotonic()+45.)
    with optional_native_scope(deadline):
        return _replay_final_native(plan_payload, _plain(receipt), deadline)


def qa_metrics_valid(metrics):
    if not isinstance(metrics, Mapping):
        return False
    value = dict(metrics)
    identity = value.pop('evidence_sha256', None)
    return bool(value.get('contract') == CONTRACT and value.get('binding_valid') is True
        and value.get('source_live_replay') is True and value.get('quality_approved') is False
        and value.get('uncertain_semantics') == UNKNOWN
        and value.get('decoded_output_quality_certified') is False
        and value.get('final_quality_status') == 'UNAPPROVED_REVIEW_CANDIDATE'
        and type(value.get('sampled_native_pixel_count')) is int
        and 1 <= value['sampled_native_pixel_count'] <= MAX_SAMPLES
        and all(isinstance(value.get(key), str) and len(value[key]) == 64
            for key in ('plan_sha256', 'metadata_sha256', 'global_native_schedule_sha256'))
        and identity == _digest(value))


def consume_normal_interval_qa(analysis, plan_payload, selection):
    render = analysis.get('render') or {}
    complete_rows = selection.get('complete_candidates') or ()
    complete = complete_rows[0] if len(complete_rows) == 1 else {}
    receipt = render.get('normal_interval_review')
    marked = (selection.get('selected_path_authority') == CONTRACT
        or render.get('candidate_origin') == CONTRACT
        or (render.get('editorial_motion') or {}).get('analysis_basis') == MOTION_BASIS
        or complete.get('normal_interval_review') is not None)
    if receipt is None and not marked:
        return None
    selected = validate_serialized_metadata(complete)
    if receipt != selected:
        raise ValueError('normal interval independent render/selection receipt copies differ or are missing')
    if _digest(_plan_payload(plan_payload)) != receipt['plan_sha256']:
        raise ValueError('normal interval final plan is not the complete selected plan')
    timeline = dict(analysis.get('visual_timeline') or {})
    timeline.pop('index_domain', None)
    if (timeline != complete['visual_timeline']
            or render.get('editorial_motion') != complete['editorial_motion']
            or render.get('group_transitions') != complete['group_transitions']
            or render.get('final_render_segments') != complete['final_render_segments']):
        raise ValueError('normal interval analysis/render contains stale selected topology or motion metadata')
    return replay_final_sources(plan_payload, receipt)


def epoch_clock_payload(plan, domain):
    return dict(contract='normal_global_native_epoch_clock_v1',
        plan_sha256=_digest(_plan_payload(plan)),
        eligible_request_indices=list(domain.eligible_indices),
        native_clocks=[clock.to_dict() for clock in domain.render_frame_clocks],
        source_cut_times=[list(cuts) for cuts in domain.scene_cut_times],
        actual_events=[event.to_dict() for event in visible_group_cut_inventory(plan, domain)])


def validate_epoch_clock(plan, timeline, source_scene_cut_times, eligible_to_request):
    """Static consumer replays all actual events with the renderer's sampler.

    A serialized clock is a numeric input here. Live source-grid/pixel binding
    is separately required at parent adoption and final QA; this is not a new
    source/reference audio alignment certificate.
    """
    from .render_frame_schedule import VerifiedRenderFrameClock
    from fractions import Fraction
    raw = _plain(timeline.get('actual_epoch_clock'))
    if not isinstance(raw, dict) or raw.get('contract') != 'normal_global_native_epoch_clock_v1':
        raise ValueError('normal interval timeline lacks its global native epoch clock')
    if (raw.get('plan_sha256') != _digest(_plan_payload(plan))
            or raw.get('eligible_request_indices') != list(eligible_to_request)
            or raw.get('source_cut_times') != [list(source_scene_cut_times[i]) for i in eligible_to_request]
            or plan.output_frame_count > MAX_OUTPUT_FRAMES):
        raise ValueError('normal interval epoch clock differs from caller source/cut/plan domain')
    clocks = tuple(VerifiedRenderFrameClock(Fraction(c['frame_rate']), c['frame_count'],
        c['packet_grid_sha256'], c['grid_basis']) for c in raw['native_clocks'])
    if [clock.to_dict() for clock in clocks] != raw['native_clocks']:
        raise ValueError('normal interval epoch clock has malformed native grid declarations')
    domain = GroupSourceDomain(tuple(plan.sources), tuple(eligible_to_request),
        (0.,)*len(clocks), (None,)*len(clocks),
        tuple(tuple(source_scene_cut_times[i]) for i in eligible_to_request), clocks)
    events = visible_group_cut_inventory(plan, domain)
    if raw.get('actual_events') != [event.to_dict() for event in events]:
        raise ValueError('normal interval native epoch events fail independent global sampler replay')
    return tuple(event.first_incoming_output_frame/plan.fps for event in events
        if event.kind == 'native_camera_cut')
