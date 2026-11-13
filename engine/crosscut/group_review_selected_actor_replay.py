"""Bind fresh review heads to frozen anonymous occurrence intent.

No frame-local ID persists across detector runs. A same-raster fingerprint
may bind a renumbered detection; other frames need an explicit appearance
hypothesis. Neither route certifies a name or turns unmatched into salience.
"""
from .group_performer_family_owner import digest
from .group_review_native_window import cosine


def _raw(view):
    return view['row']['native_observation']


def _head_fingerprint(view):
    identity=view['descriptor'].frame_local_head_id
    head=next((h for h in _raw(view).get('heads',()) if h['observation_id']==identity),None)
    return None if head is None else digest({k:v for k,v in head.items() if k!='observation_id'})


def bind_review_selected_actor(template,candidates,*,maximum_native_gap=12,
        minimum_cosine=.25,ambiguity_margin=.025):
    """Return one explicit selected view, or UNMATCHED/AMBIGUOUS evidence.

    The caller supplies only candidates from one actual requested raw raster
    and chooses a frozen template from the SAME observed native camera unit.
    A camera-unit bridge cannot be invented here; this function does not LK.
    """
    if (not isinstance(candidates,(list,tuple)) or len(candidates)>24
            or type(maximum_native_gap) is not int or not 0<=maximum_native_gap<=12
            or not 0<=minimum_cosine<=1 or not 0<=ambiguity_margin<=1):
        raise ValueError('selected replay needs a bounded actual head inventory')
    result=dict(state='UNMATCHED',selected=None,appearance_cosine=None,
        frame_local_ids_treated_as_persistent=False,anonymous_actor_certified=False,
        quality_approved=False)
    if template is None or not candidates:return result
    reference=_raw(template)
    for candidate in candidates:
        raw=_raw(candidate)
        if (raw['request_source_index']!=reference['request_source_index']
                or raw['source_clock_identity']!=reference['source_clock_identity']
                or raw['source_file_identity']!=reference['source_file_identity']):
            raise ValueError('selected actor replay changed frozen source/clock/file intent')
        if raw!=_raw(candidates[0]):
            raise ValueError('selected replay inventory contains different raw rasters')
    actual=_raw(candidates[0])
    if abs(actual['native_frame_index']-reference['native_frame_index'])>maximum_native_gap:
        return dict(result,reason='frozen_template_native_gap_unmeasured')
    fingerprint=_head_fingerprint(template)
    if actual['native_frame_index']==reference['native_frame_index']:
        if actual.get('native_pixel_sha256')!=reference.get('native_pixel_sha256'):
            raise ValueError('selected replay exact native raster changed its pixels')
        matches=[c for c in candidates if fingerprint is not None and _head_fingerprint(c)==fingerprint]
        if len(matches)==1:
            return dict(result,state='SAME_RASTER_OCCURRENCE',selected=matches[0],
                frozen_occurrence_fingerprint=fingerprint)
        # ID equality without matching physical head geometry is NOT recovery.
    scored=[(cosine(template['vector'],c['vector']),c) for c in candidates]
    ranked=sorted((p for p in scored if p[0] is not None),key=lambda p:-p[0])
    if not ranked or ranked[0][0]<minimum_cosine:
        return dict(result,reason='frozen_actor_appearance_not_observed')
    margin=None if len(ranked)==1 else ranked[0][0]-ranked[1][0]
    if margin is not None and margin<ambiguity_margin:
        return dict(result,state='AMBIGUOUS',appearance_cosine=ranked[0][0],
            competing_margin=margin,reason='fresh_selected_actor_competition_unresolved')
    return dict(result,state='SOURCE_LOCAL_APPEARANCE_HYPOTHESIS',selected=ranked[0][1],
        appearance_cosine=ranked[0][0],competing_margin=margin,
        frozen_occurrence_fingerprint=fingerprint)
