"""Independent schema binding for a complete native-shot camera application.

This binds containment/renderer geometry, not face identity, choreography,
professional quality, a learned checkpoint, or the old fourteen-frame planner.
Post-render QA must still inspect the actual encoded montage independently.
"""
from collections.abc import Mapping
import hashlib
import json
import math
from fractions import Fraction

from .boundary_subject_alignment import _TransformState, _project_box
from .render import segment_frame_counts, _can_share_source_clock
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import VerifiedRenderFrameClock


CONTRACT = 'native_shot_context_head_containment_binding_v1'
RUN_CONTRACT = 'native_multi_cell_shot_context_head_containment_binding_v2'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def shot_context_geometry_identity(plan, index, last_index=None):
    """The complete changed shot and both fixed neighbors, excluding grading.

    A later color-only pass does not change the containment coordinates. A
    different source, clock, base crop, camera, overlap or output geometry does.
    """
    raw=plan.to_json();last=index if last_index is None else last_index
    names=('source_index','source_start','source_local_end','source_duration','master_start','duration',
        'crop','pan_crop','editorial_transform','outgoing_overlap_transform')
    return _digest(dict(fps=plan.fps,width=plan.width,height=plan.height,
        sources=[str(p) for p in plan.sources],reference_source=plan.reference_source,
        master_audio_start=plan.master_audio_start,aspect_ratio=plan.aspect_ratio.value,
        frame_counts=list(segment_frame_counts(plan)),index=index,
        segments=[{name:row.get(name) for name in names} for row in raw['segments'][index-1:last+2]],
        transitions=raw['transitions'][index-1:last+1],
        **({'last_segment_index':last} if last!=index else {})))


def _numeric(value):
    return type(value) in (float,int) and math.isfinite(value)


def build_shot_context_binding(proposal):
    """Bind a real proposal without promoting its unproved editorial claims."""
    if proposal.local_focal_attention:
        raise ValueError('source-local focal shot cameras remain an unpromoted experiment')
    plan,index,last=proposal.plan,proposal.segment_index,proposal.end_segment_index
    payload=dict(contract=CONTRACT if last==index else RUN_CONTRACT,
        segment_index=index,source_index=plan.segments[index].source_index,
        geometry_identity=shot_context_geometry_identity(plan,index,last),
        native_frame_ids=list(proposal.native_frame_ids),head_evidence=[dict(row) for row in proposal.head_evidence],
        source_clock=proposal.source_clock.to_dict() if proposal.source_clock else None,
        frame_count=proposal.camera.frame_count,protected_subject='head_envelope_not_full_body',
        maximum_center_distance=.10,minimum_head_visible_fraction=.995,
        native_camera_cuts_in_changed_shot=0,audio_or_source_phase_changed=False,
        identity_or_gesture_certified=False,professional_quality_approved=False,
        learned_transformer_inference=False,
        **({'last_segment_index':last,'boundary_indices':[index-1,last]} if last!=index else {}))
    # Freeze into the same JSON-native types used after the final manifest is
    # written. A producer cannot gain a different tuple/list interpretation.
    payload=json.loads(json.dumps(payload,allow_nan=False))
    payload['content_sha256']=_digest(payload)
    if not shot_context_binding_valid(payload,plan):
        raise ValueError('shot-context proposal cannot bind its complete camera geometry')
    return payload


def shot_context_binding_valid(payload, plan):
    """Recalculate all active-frame bounds and complete context identity."""
    if not isinstance(payload,Mapping) or payload.get('contract') not in (CONTRACT,RUN_CONTRACT):
        return False
    index=payload.get('segment_index')
    multi=payload.get('contract')==RUN_CONTRACT
    last=payload.get('last_segment_index') if multi else index
    if (type(index) is not int or not 0<index<len(plan.segments)-1
            or type(last) is not int or not index<=last<len(plan.segments)-1
            or multi and (last==index or payload.get('boundary_indices')!=[index-1,last])
            or not multi and ('last_segment_index' in payload or 'boundary_indices' in payload)
            or type(payload.get('source_index')) is not int
            or payload.get('source_index')!=plan.segments[index].source_index
            or payload.get('protected_subject')!='head_envelope_not_full_body'
            or payload.get('maximum_center_distance')!=.10
            or payload.get('minimum_head_visible_fraction')!=.995
            or type(payload.get('native_camera_cuts_in_changed_shot')) is not int
            or payload.get('native_camera_cuts_in_changed_shot')!=0
            or any(payload.get(key) is not False for key in ('audio_or_source_phase_changed',
                'identity_or_gesture_certified','professional_quality_approved','learned_transformer_inference'))):
        return False
    try:
        if (_digest({k:v for k,v in payload.items() if k!='content_sha256'})!=payload.get('content_sha256')
                or shot_context_geometry_identity(plan,index,last)!=payload.get('geometry_identity')):
            return False
        counts=segment_frame_counts(plan);count=sum(counts[index:last+1])
        native=payload.get('native_frame_ids');rows=payload.get('head_evidence')
        if (any(s.pan_crop is not None or s.crop is not None or s.editorial_transform is None
                    or s.outgoing_overlap_transform is not None for s in plan.segments[index:last+1])
                or any(t.kind.value!='hard_cut' for t in plan.transitions[index-1:last+1])
                or any(plan.segments[n].source_index==plan.segments[index].source_index for n in (index-1,last+1))
                or not _can_share_source_clock(plan,first_index=index,candidate_last_index=last)
                or not 2<=count<=300 or payload.get('frame_count')!=count
                or not isinstance(native,list) or len(native)!=count
                or any(type(n) is not int or n<0 for n in native)
                or any(a>b for a,b in zip(native,native[1:]))
                or not isinstance(rows,list) or len(rows)!=count):
            return False
        clock_data=payload.get('source_clock')
        if not isinstance(clock_data,dict):return False
        clock=VerifiedRenderFrameClock(Fraction(clock_data['frame_rate']),clock_data['frame_count'],
            clock_data['packet_grid_sha256'],clock_data['grid_basis'])
        if clock.to_dict()!=clock_data:return False
        begin=sum(counts[:index])
        samples=native_frame_clocks_for_output(plan,{plan.segments[index].source_index:clock},range(begin,begin+count))
        if (native!=[samples[n].native_frame_index for n in range(begin,begin+count)]
                or any(tick.terminal_clone for tick in samples.values())):
            return False
        observed=set();segment_index=index;segment_begin=0
        for frame,row in enumerate(rows):
            if not isinstance(row,Mapping) or type(row.get('frame')) is not int or row['frame']!=frame:
                return False
            basis=row.get('basis');box=row.get('head_envelope_xywh')
            if (not isinstance(box,list) or len(box)!=4 or not all(_numeric(v) for v in box)
                    or box[0]<0 or box[1]<0 or box[2]<=0 or box[3]<=0
                    or box[0]+box[2]>1.000001 or box[1]+box[3]>1.000001):
                return False
            if basis=='observed_face_head_envelope':
                observed.add(frame)
            elif basis=='two_neighbor_pixel_tracked_head_union':
                refs=row.get('reference_frames')
                if (not isinstance(refs,list) or len(refs)<2 or len(set(refs))!=len(refs)
                        or any(type(ref) is not int or not 0<=ref<count or not 0<abs(ref-frame)<=2 for ref in refs)
                        or row.get('face_landmarks_observed') is not False):
                    return False
            else:return False
            while frame>=segment_begin+counts[segment_index]:
                segment_begin+=counts[segment_index];segment_index+=1
            key=plan.segments[segment_index].editorial_transform.evaluate((frame-segment_begin)/plan.fps)
            if math.hypot(key.center_x-.5,key.center_y-.5)>.100001:
                return False
            if _project_box(tuple(box),_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees))[1]<.995:
                return False
        if not {0,count-1}<=observed:
            return False
        return all(set(row.get('reference_frames',()))<=observed for row in rows)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,ZeroDivisionError):
        return False


def assert_unchanged_camera_context(before, after, index, last):
    """Reuse sensing only when the local output and native domains are exact.

    The legacy digest deliberately includes every shot's frame count. An
    unrelated two-cell retime can change that digest without changing this
    camera. Do not rerun sensing or simply bless the new digest: prove the
    complete camera and both fixed neighbors, absolute output ticks, source
    domain and output/audio geometry first. Grading in this context is also
    preserved; this is not a color-only receipt migration.
    """
    fields=('sources','reference_source','master_audio_start','fps','width',
        'height','aspect_ratio','member_pose_dissolve_requested',
        'member_pose_dissolve_effective','member_pose_dissolve_count')
    if (type(index) is not int or type(last) is not int
            or not 0<index<=last<len(before.segments)-1
            or len(before.segments)!=len(after.segments)
            or any(getattr(before,name)!=getattr(after,name) for name in fields)
            or before.segments[index-1:last+2]!=after.segments[index-1:last+2]
            or before.transitions[index-1:last+1]!=after.transitions[index-1:last+1]):
        raise ValueError('camera context changed; prior sensing cannot be reused')
    old,new=segment_frame_counts(before),segment_frame_counts(after)
    if (sum(old)!=sum(new) or sum(old[:index-1])!=sum(new[:index-1])
            or old[index-1:last+2]!=new[index-1:last+2]):
        raise ValueError('camera context output clock changed; prior sensing cannot be reused')


def rebind_unchanged_shot_context(payload, before, after):
    """Retain a valid receipt across an unrelated edit, never create evidence."""
    from .visual_boundary_repair import _thaw_audit_value
    row=json.loads(json.dumps(_thaw_audit_value(payload),allow_nan=False))
    if not shot_context_binding_valid(row,before):
        raise ValueError('prior shot-context receipt is not independently valid')
    index=row['segment_index'];last=row.get('last_segment_index',index)
    assert_unchanged_camera_context(before,after,index,last)
    row['geometry_identity']=shot_context_geometry_identity(after,index,last)
    row['content_sha256']=_digest({k:v for k,v in row.items() if k!='content_sha256'})
    # Independently recalculate native ticks and EVERY transformed head box.
    if not shot_context_binding_valid(row,after):
        raise ValueError('unchanged shot-context receipt failed final independent validation')
    return row


def bound_shot_context_cameras(motion, plan):
    if not isinstance(motion,Mapping):return {}
    rows=motion.get('shot_context_cameras',())
    if not isinstance(rows,(list,tuple)):
        return {}
    result={}
    motion_segments=motion.get('segments',())
    serialized=plan.to_json()['segments']
    for row in rows:
        if not shot_context_binding_valid(row,plan):return {}
        for index in range(row['segment_index'],row.get('last_segment_index',row['segment_index'])+1):
            if index in result:return {}
            if not isinstance(motion_segments,(list,tuple)) or index>=len(motion_segments):return {}
            item=motion_segments[index]
            if (not isinstance(item,Mapping) or item.get('segment_index')!=index
                    or item.get('source_index')!=plan.segments[index].source_index
                    or item.get('applied') is not True
                    or item.get('sampling_canvas_zoom')!=serialized[index]['editorial_transform'].get('sampling_canvas_zoom')
                    or item.get('keyframes')!=list(serialized[index]['editorial_transform']['keyframes'])):return {}
            result[index]=row
    # Adjacent replaced shots alter each other's supposedly fixed targets.
    if any(index+1 in result and result[index] is not result[index+1] for index in result):return {}
    return result
