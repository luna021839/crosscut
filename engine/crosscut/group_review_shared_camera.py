"""One measured head/body view drives camera fitting AND its support checks.

This review adapter neither reacquires an actor nor grants app admission. Raw
ownership stays unchanged, UNKNOWN stays UNKNOWN, and primary FAIL survives.
"""
from .group_review_body_recovery import consume_head_conditioned_review_support
from .group_review_camera_repair import fit_whole_camera_track


def fit_shared_head_body_review_camera(views,left,right,*,duration,sample_times,
        original_part_duties,maximum_zoom=2.):
    if (not isinstance(views,(list,tuple)) or len(views)<3 or any(v is None for v in views)
            or len(sample_times)!=len(views)):
        raise ValueError('shared camera requires the same complete actual selected actor sample inventory')
    by_tick={v['row']['output_tick']:v for v in views}
    if len(by_tick)!=len(views):raise ValueError('shared camera cannot collapse two requested output occurrences')
    support_receipts=[]
    def support(row,key):
        view=by_tick.get(row['output_tick'])
        if view is None or view['row']!=row:raise ValueError('camera support borrowed another actual selected raster')
        receipt=consume_head_conditioned_review_support(original_part_duties,view,key=key)
        support_receipts.append(receipt)
        return receipt['state']!='FAIL'
    track=fit_whole_camera_track([v['row'] for v in views],left,right,duration=duration,
        maximum_zoom=maximum_zoom,sample_times=sample_times,
        head_ids=tuple(v['descriptor'].frame_local_head_id for v in views),
        descriptors=tuple(v['descriptor'] for v in views),required_support_check=support)
    return track,dict(contract='same_head_conditioned_body_camera_and_support_adapter_v1',
        requested_output_ticks=list(by_tick),support_receipts=support_receipts,
        body_evidence_identities=[v.get('review_joint_evidence_identity') for v in views],
        unknown_support_certified_pass=False,body_ownership_rewritten=False,quality_approved=False)
