"""Final paired measurements consume selected actors, including the EXIT.

This producer preserves unknown values and does not decide aesthetic success.
An original continuous shot is measured, never assigned an invented zero
motion cost for an added cut. Added-match admission needs a separate contract.
"""
from .group_review_camera_repair import effective_face_detail_pixels
from .group_review_measured_features import pair_risk
from .group_review_motion_seam import (score_review_motion_seam,score_review_head_seam,
    review_native_window_continuous)


def selected_pair_side(evidence,views,*,association_states,full_plan_identity):
    metrics=dict(closeup_distance=[],native_detail=[],blur=[],actor_association_uncertainty=[],
        incident_geometry=[])
    for tick in evidence.interior_ticks:
        if tick not in views or tick not in association_states:
            raise ValueError('selected paired consumer omitted an actual requested occurrence')
        view=views[tick];descriptor=None if view is None else view['descriptor']
        detail=effective_face_detail_pixels(descriptor)
        metrics['closeup_distance'].append(None if descriptor is None else 1.-view['near'])
        metrics['native_detail'].append(None if detail is None else max(0.,1.-detail/150.))
        metrics['blur'].append(None if view is None else view['detail_risk'])
        metrics['actor_association_uncertainty'].append(0. if association_states[tick] in
            ('SAME_RASTER_OCCURRENCE','SOURCE_LOCAL_APPEARANCE_HYPOTHESIS') else 1.)
    incidents=evidence.incident_ticks or (evidence.first,evidence.end)
    for tick in incidents:
        if tick-1 not in views or tick not in views:
            raise ValueError('selected paired consumer omitted actual entry/exit neighbors')
        risk=pair_risk(views[tick-1],views[tick])
        metrics['incident_geometry'].append(None if risk['face_geometry_risk'] is None
            or risk['appearance_cosine'] is None else risk['risk'])
    return dict(evidence_plan_sha256=evidence.identity,
        requests=dict(interior_ticks=evidence.interior_ticks,left_ticks=evidence.left_ticks,
            right_ticks=evidence.right_ticks,appearance_ticks=evidence.appearance_ticks),
        metrics=metrics,whole_render_plan_identity=full_plan_identity,
        selected_occurrences_not_default_salience=True,entry_and_exit_consumed=True,
        continuous_original_assigned_perfect_zero=False,quality_approved=False)


def _physical(sequence):
    return review_native_window_continuous(sequence)


def selected_incident_measurements(evidence,views,*,actual_cut_ticks):
    """Use actual flanks, clipping at source/native physical shot boundaries."""
    result=[]
    for tick in evidence.incident_ticks:
        outgoing=[views.get(t) for t in range(max(0,tick-3),tick)]
        incoming=[views.get(t) for t in range(tick,min(evidence.output_frame_count,tick+3))]
        # A physical cut inside a flank must not be treated as a continuous
        # track. Prefer its genuine nearest two observations if available.
        if not _physical(outgoing):outgoing=outgoing[-2:]
        if not _physical(incoming):incoming=incoming[:2]
        connected=_physical(outgoing) and _physical(incoming)
        unknown=dict(state='INCONCLUSIVE',exploration_cost=None)
        body=(score_review_motion_seam(outgoing,incoming,fps=evidence.fps) if connected else unknown)
        head=(score_review_head_seam(outgoing,incoming,fps=evidence.fps) if connected else unknown)
        result.append(dict(tick=tick,actual_visible_cut=tick in actual_cut_ticks,
            endpoint=pair_risk(views.get(tick-1),views.get(tick)),body=body,head=head,
            selected_actual_flanks_used=True,continuous_original_assigned_zero=False,
            quality_approved=False))
    return tuple(result)


def paired_selected_support(original,candidate):
    """Distinguish originally UNKNOWN from loss of known original support."""
    if (not isinstance(original,(list,tuple)) or not original or len(original)!=len(candidate)
            or any(v.get('state') not in ('PASS','FAIL','UNKNOWN') for v in (*original,*candidate))
            or any(a['output_tick']!=b['output_tick'] for a,b in zip(original,candidate))):
        raise ValueError('paired selected support needs the same nonempty exact tick inventory')
    rows=[]
    for before,after in zip(original,candidate):
        lost=before['state']=='PASS' and after['state']=='UNKNOWN'
        state='FAIL' if after['state']=='FAIL' or lost else after['state']
        rows.append(dict(output_tick=before['output_tick'],original_state=before['state'],
            candidate_state=after['state'],state=state,lost_known_support=lost,
            original_unobservability_preserved=before['state']=='UNKNOWN',quality_approved=False))
    return tuple(rows)
