"""Complete the selected native edit, without installing partial metadata.

The supported retained families are face recipes, legacy subject alignment,
disjoint shot-context cameras, and absent boundary look. Other active owners
are explicitly declined, never erased. This is construction, not raw admission
or professional-quality approval; it must run inside the existing controller.
"""
from dataclasses import replace
import json
from statistics import median

from .boundary_subject_alignment import BoundarySubjectAlignmentDecision, BoundarySubjectAlignmentSegment
from .editorial_motion import EditorialMotionSegment
from .editorial_transform_events import rendered_transform_event_records
from .face_match import FaceMatchSegment
from .group_boundary_evidence_preservation import preserve_boundary_evidence
from .group_joint_bundle_metadata import seal_native_motion_production
from .group_joint_native_preparation import _plan_identity
from .group_native_motion_timeline import build_native_motion_timeline, NativeMotionTimelineMetadata
from .group_transition import GroupTransitionPlan
from .timeline_review import review_timeline

CONTRACT = 'selected_native_motion_complete_metadata_v1'


def _metadata_payload(bundle):
    value = bundle.to_dict()
    value.pop('performer_family_review', None)
    value.pop('native_motion_production', None)  # The complete selection seal owns this backlink separately.
    return value


def validate_native_metadata_binding(bundle):
    """Content binding only; never replace raw/selected/final media replay."""
    from . import group_performer_family_owner as family
    receipt = family.plain(bundle.performer_family_review)
    dense = family.plain(bundle.editorial_motion_plan.source_camera_refresh)
    if (not isinstance(receipt, dict) or not isinstance(dense, dict)
            or receipt.get('contract') != CONTRACT or bundle.candidate_origin != CONTRACT
            or dense.get('contract') != CONTRACT or receipt.get('source_camera_refresh') != dense
            or receipt.get('plan_sha256') != family.digest(family.plan_payload(bundle.render_plan))
            or receipt.get('metadata_sha256') != family.digest(_metadata_payload(bundle))
            or any(receipt.get(k) is not False for k in ('quality_approved', 'raw_parent_admitted',
                'cross_camera_identity_certified', 'body_phase_certified', 'decoded_output_quality_certified'))):
        raise ValueError('selected native complete metadata/family binding changed')
    if bundle.boundary_look_match is not None or any(s.boundary_look_track is not None for s in bundle.render_plan.segments):
        raise ValueError('selected native metadata has an unsupported live boundary-look owner')
    evidence = dense['timeline_evidence']
    if evidence['selected_plan_identity'] != _plan_identity(bundle.render_plan):
        raise ValueError('selected native metadata has stale source/camera observations')


def _retained_shot_context(payload, before, after, mapping):
    """Move a disjoint receipt only after replaying EVERY original head/frame.

    An index shift is not a new observation. The receipt and both fixed
    neighbors need intact cells, absolute output clocks and renderer numbers.
    The existing validator independently replays its full new native grid.
    """
    from .visual_boundary_repair import _thaw_audit_value
    from .shot_context_binding import (shot_context_binding_valid, shot_context_geometry_identity, _digest)
    from .render import segment_frame_counts
    row = json.loads(json.dumps(_thaw_audit_value(payload), allow_nan=False))
    if not shot_context_binding_valid(row, before):
        raise ValueError('native metadata predecessor shot-context receipt is not independently valid')
    first, last = row['segment_index'], row.get('last_segment_index', row['segment_index'])
    indices = tuple(range(first-1, last+2))
    if any(i not in mapping for i in indices):
        raise ValueError('native metadata changed an active shot-context camera or its fixed neighbor')
    mapped = tuple(mapping[i] for i in indices)
    if mapped != tuple(range(mapped[0], mapped[0]+len(indices))):
        raise ValueError('native metadata split a retained shot-context camera')
    nf, nl = mapping[first], mapping[last]
    old_counts, new_counts = segment_frame_counts(before), segment_frame_counts(after)
    names = ('source_index', 'crop', 'pan_crop', 'editorial_transform',
        'outgoing_overlap_transform', 'boundary_look_track', 'look_adjustment')
    if (sum(old_counts[:first-1]) != sum(new_counts[:nf-1])
            or old_counts[first-1:last+2] != new_counts[nf-1:nl+2]
            or any(any(getattr(before.segments[oi], n) != getattr(after.segments[ni], n) for n in names)
                for oi, ni in zip(indices, mapped, strict=True))
            or before.transitions[first-1:last+1] != after.transitions[nf-1:nl+1]):
        raise ValueError('native metadata changed a retained shot-context pixel/absolute-clock preimage')
    row['segment_index'] = nf
    if 'last_segment_index' in row:
        row['last_segment_index'] = nl
        row['boundary_indices'] = [nf-1, nl]
    row['geometry_identity'] = shot_context_geometry_identity(after, nf, nl)
    row['content_sha256'] = _digest({k:v for k,v in row.items() if k != 'content_sha256'})
    if not shot_context_binding_valid(row, after):
        raise ValueError('native metadata retained shot-context failed independent full native/head replay')
    return row


def _fresh_motion(old, plan, timeline, cells, evidence):
    mapping = {c.retained_old_segment_index:c.segment_index for c in cells if c.retained_old_segment_index is not None}
    edges = {cells[i].retained_old_segment_index:i for i in range(len(cells)-1)
        if cells[i].retained_old_segment_index is not None
        and cells[i+1].retained_old_segment_index == cells[i].retained_old_segment_index+1}
    changed = {c.segment_index for c in cells if c.refresh_source_observations}
    segments = tuple(replace(old.editorial_motion_plan.segments[c.retained_old_segment_index],
            segment_index=c.segment_index, source_index=timeline.choices[c.segment_index].source_index,
            track=plan.segments[c.segment_index].editorial_transform)
        if c.retained_old_segment_index is not None else EditorialMotionSegment(c.segment_index,
            timeline.choices[c.segment_index].source_index, plan.segments[c.segment_index].editorial_transform,
            ('fresh_native_metadata_numeric_camera_not_legacy_boundary_certificate',)) for c in cells)
    motion = old.editorial_motion_plan
    alignment = motion.boundary_subject_alignment_plan
    retired = []
    if alignment is not None:
        decisions = []
        for d in alignment.decisions:
            i = edges.get(d.boundary_index)
            if i is None:
                retired.append(d.to_dict())
            else:
                if any(j not in edges for j in d.preserved_one_sided_boundary_indices):
                    raise ValueError('native metadata changed the neighbor of a paired alignment certificate')
                decisions.append(replace(d, boundary_index=i, outgoing_segment_index=i,
                    incoming_segment_index=i+1, preserved_one_sided_boundary_indices=tuple(
                        edges[j] for j in d.preserved_one_sided_boundary_indices)))
        for i in range(len(plan.transitions)):
            if i not in edges.values():
                decisions.append(BoundarySubjectAlignmentDecision(i, i, i+1,
                    timeline.choices[i].source_index, timeline.choices[i+1].source_index,
                    plan.segments[i].source_end, plan.segments[i+1].source_start, False,
                    'separately_bound_selected_native_motion_not_legacy_boundary_certificate'))
        alignment = replace(alignment, decisions=tuple(sorted(decisions, key=lambda d:d.boundary_index)),
            segments=tuple(BoundarySubjectAlignmentSegment(i, c.source_index, plan.segments[i].editorial_transform,
                tuple(edges[j] for j in motion.boundary_subject_alignment_plan.segments[cell.retained_old_segment_index].boundary_indices
                    if j in edges) if cell.retained_old_segment_index is not None else ())
                for i, (cell, c) in enumerate(zip(cells, timeline.choices, strict=True))))
    contexts = tuple(_retained_shot_context(row, old.render_plan, plan, mapping) for row in motion.shot_context_cameras)
    if motion.coupled_shot_cameras:
        raise ValueError('native metadata needs a topology-aware coupled-shot refresh; incumbent retained')
    if motion.source_camera_refresh is not None:
        raise ValueError('native metadata needs a prior native-family support replay; incumbent retained')
    events = rendered_transform_event_records(plan.segments, plan.fps)
    reasons = [{r.split(':',1)[0] for r in row.reasons} for row in segments]
    matched = {i for i, (a, b) in enumerate(zip(reasons, reasons[1:]))
        if any(r.endswith('_matched_hard_cut') for r in a & b)}
    if alignment is not None:
        matched.update(d.boundary_index for d in alignment.decisions if d.eligible)
    affine = {i for i, (a, b) in enumerate(zip(reasons, reasons[1:]))
        if any(r.startswith('orb_affine_') and r.endswith('_matched_hard_cut') for r in a & b)}
    def runs(reason):
        return sum(reason in row and (i == 0 or reason not in reasons[i-1]) for i, row in enumerate(reasons))
    interior = runs('reference_calibrated_same_source_interior_zoom_pulse')
    authored = (('zoom', len(matched)+interior+sum(runs(r) for r in
            ('reference_calibrated_push_in', 'reference_calibrated_pull_out'))),
        ('pan', 0 if alignment is None else alignment.pan_event_count),
        ('roll', 0 if alignment is None else alignment.roll_event_count),
        ('separately_bound_selected_native_cameras', evidence.get('selected_shot_count', 2)))
    history = dict(retired_alignment_decisions=retired,
        retired_required_boundary_failures=[list(row) for row in motion.required_boundary_failures if row[0] not in edges],
        retired_accepted_boundaries=[i for i in motion.accepted_boundary_indices if i not in edges],
        retired_natural_match_boundaries=[i for i in motion.natural_match_boundary_indices if i not in edges],
        retired_residual_attention_mismatches=[list(row) for row in motion.boundary_residual_attention_mismatches if row[0] not in edges],
        historical_only_not_new_source_certificates=True)
    fresh = dict(contract=CONTRACT, timeline_evidence=evidence, retired_boundary_history=history,
        changed_segment_indices=sorted(changed), quality_approved=False)
    return replace(motion, segments=segments, boundary_subject_alignment_plan=alignment,
        shot_context_cameras=contexts, zoom_event_count=len(events['zoom']), pan_event_count=len(events['pan']),
        roll_event_count=len(events['roll']), authored_event_counts=authored,
        boundary_match_count=len(matched), affine_boundary_match_count=len(affine), interior_zoom_pulse_count=interior,
        accepted_boundary_indices=tuple(edges[i] for i in motion.accepted_boundary_indices if i in edges),
        natural_match_boundary_indices=tuple(edges[i] for i in motion.natural_match_boundary_indices if i in edges),
        boundary_residual_attention_mismatches=tuple((edges[row[0]], *row[1:])
            for row in motion.boundary_residual_attention_mismatches if row[0] in edges),
        required_boundary_failures=tuple((edges[row[0]], *row[1:])
            for row in motion.required_boundary_failures if row[0] in edges), source_camera_refresh=fresh)


def _fresh_face(old, plan, timeline, cells, evidence):
    if old.face_match_plan is None:
        return None
    samples = {s['output_tick']:s for s in evidence['samples']}
    rows = []
    for cell, choice, segment in zip(cells, timeline.choices, plan.segments, strict=True):
        if cell.retained_old_segment_index is not None:
            rows.append(replace(old.face_match_plan.segments[cell.retained_old_segment_index],
                segment_index=cell.segment_index, source_index=choice.source_index, pan_crop=segment.pan_crop))
            continue
        heights = [box[3] for box in choice.face_samples_xywh if box is not None]
        rows.append(FaceMatchSegment(cell.segment_index, choice.source_index, False,
            'selected_editorial_camera_no_new_face_pan_recipe', median(heights) if heights else None,
            old.face_match_plan.target_face_height, None, None, segment.pan_crop,
            alignment_subject='face' if heights else None,
            alignment_purpose='actual_projected_selected_actor_observations_not_face_head_extent_certificate',
            projected_output_observations=dict(contract=CONTRACT,
                samples=[samples[t] for t in cell.taps], quality_approved=False)))
    return replace(old.face_match_plan, segments=tuple(rows),
        eligible_face_segment_count=sum(r.observed_face_height is not None for r in rows),
        eligible_silhouette_segment_count=sum(r.observed_silhouette_height is not None for r in rows),
        analysis_basis='retained_face_targets_and_selected_native_camera_fresh_rows_v1')


def complete_native_motion_metadata(*, context, selected, metadata, domain, protected_output_intervals=()):
    """Atomically construct/seal the full selected bundle; never neutralize it."""
    from .pipeline import _GroupRenderCandidateBundle, _deterministic_exact_shot_clock_from_timeline
    old = context.incumbent
    if not isinstance(old, _GroupRenderCandidateBundle) or not isinstance(metadata, NativeMotionTimelineMetadata):
        raise ValueError('native complete metadata needs the typed predecessor and fresh timeline producer result')
    plan = selected.render_plan
    evidence = json.loads(metadata.evidence_json)
    if (evidence['original_plan_identity'] != _plan_identity(old.render_plan)
            or evidence['selected_plan_identity'] != _plan_identity(plan)
            or old.native_motion_production is not None):
        raise ValueError('native complete metadata belongs to a different selected programme')
    unsupported = [n for n in ('dynamic_crop_application', 'cut_local_sync', 'splice_graph_authorization',
        'output_geometry_deferral', 'stage_edit_dynamic_crop_version', 'stage_edit_dynamic_crop_v3_decodes',
        'dynamic_crop_motion_timeline_v3', 'normal_interval_review', 'performer_family_review', 'boundary_look_match')
        if getattr(old, n) is not None]
    if unsupported:
        raise ValueError('native complete metadata needs retained owner support refresh: '+', '.join(unsupported))
    timeline = metadata.timeline
    motion = _fresh_motion(old, plan, timeline, metadata.cells, evidence)
    face = _fresh_face(old, plan, timeline, metadata.cells, evidence)
    repairs, affine, preservation = preserve_boundary_evidence(old_plan=old.render_plan, new_plan=plan,
        old_timeline=old.visual_timeline, new_timeline=timeline,
        boundary_repairs=old.boundary_repairs, boundary_affine_plan=old.boundary_affine_plan)
    review = review_timeline(timeline, available_source_count=len(domain.eligible_indices))
    candidate = replace(old, visual_timeline=timeline, render_plan=plan, editorial_motion_plan=motion,
        face_match_plan=face, boundary_repairs=repairs, boundary_affine_plan=affine,
        timeline_review=review, pre_render_score=float(review.weighted_score),
        group_transition_plan=GroupTransitionPlan(plan.transitions, analysis_basis=CONTRACT),
        stage_edit_exact_shot_clock=_deterministic_exact_shot_clock_from_timeline(timeline,
            section_plan=old.section_plan, phrase_times=old.phrase_times),
        candidate_origin=CONTRACT, candidate_lineage=(*old.candidate_lineage, dict(stage=CONTRACT,
            original_plan_identity=_plan_identity(old.render_plan), preservation=preservation,
            source_dependent_rows_fresh=True, full_bundle_constructed=True, quality_approved=False)))
    from . import group_performer_family_owner as family
    candidate = replace(candidate, performer_family_review=dict(contract=CONTRACT,
        plan_sha256=family.digest(family.plan_payload(plan)), metadata_sha256=family.digest(_metadata_payload(candidate)),
        source_camera_refresh=family.plain(motion.source_camera_refresh), source_domain=domain.to_dict(),
        complete_metadata_constructed=True, quality_approved=False, raw_parent_admitted=False,
        cross_camera_identity_certified=False, body_phase_certified=False, decoded_output_quality_certified=False))
    production = seal_native_motion_production(complete_bundle=candidate, source_domain=domain,
        shot_clock_context=selected.shot_clock_context, selected_shots=selected.selected_shots,
        original_plan_identity=_plan_identity(old.render_plan), protected_output_intervals=protected_output_intervals,
        state='SELECTED_FOR_REVIEW', selection_evidence=dict(contract=CONTRACT,
            selected_camera_evidence=json.loads(selected.evidence_json), timeline_evidence=evidence,
            complete_metadata_constructed=True, candidate_admitted=False, quality_approved=False),
        selection_scope=selected.selection_scope)
    return replace(candidate, native_motion_production=production)


def produce_complete_native_motion_metadata(*, context, selected, domain, views, protected_output_intervals=()):
    metadata = build_native_motion_timeline(context=context, selected_plan=selected.render_plan,
        domain=domain, selected_shots=selected.selected_shots, views=views,
        selection_scope=selected.selection_scope)
    return complete_native_motion_metadata(context=context, selected=selected, metadata=metadata,
        domain=domain, protected_output_intervals=protected_output_intervals)
