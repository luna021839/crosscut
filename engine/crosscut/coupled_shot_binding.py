"""Atomic geometry/clock binding for a whole shot plus a tracked entry handle.

Both cameras must be present. This cannot promote source-local tracking into
member identity, moving-pose quality or a qualified learned checkpoint.
"""
from collections.abc import Mapping
from fractions import Fraction
import json
import math

from .boundary_subject_alignment import _TransformState, _project_box, _close_face_head_envelope
from .render import segment_frame_counts
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import VerifiedRenderFrameClock
from .shot_context_binding import (
    _digest, _numeric, build_shot_context_binding, shot_context_binding_valid,
    shot_context_geometry_identity,
)

CONTRACT='native_coupled_whole_shot_and_entry_handle_binding_v1'


def coupled_geometry_identity(plan,index):
    return _digest([shot_context_geometry_identity(plan,index),shot_context_geometry_identity(plan,index+1)])


def build_coupled_shot_binding(proposal):
    index=proposal.segment_index
    payload=dict(contract=CONTRACT,segment_index=index,boundary_indices=[index-1,index],
        geometry_identity=coupled_geometry_identity(proposal.plan,index),
        outgoing=build_shot_context_binding(proposal.outgoing),
        incoming_source_index=proposal.plan.segments[index+1].source_index,
        incoming_source_clock=proposal.incoming_source_clock.to_dict(),
        incoming_native_frame_ids=list(proposal.incoming_native_frame_ids),
        incoming_head_evidence=list(proposal.incoming_head_evidence),
        incoming_neutral_from_frame=len(proposal.incoming_head_evidence)-1,
        incoming_native_cuts_in_active_window=0,
        protected_subject='cut_anchor_head_not_every_background_face',
        audio_or_source_phase_changed=False,identity_or_gesture_certified=False,
        professional_quality_approved=False,learned_transformer_inference=False)
    payload=json.loads(json.dumps(payload,allow_nan=False));payload['content_sha256']=_digest(payload)
    if not coupled_shot_binding_valid(payload,proposal.plan):
        raise ValueError('coupled camera cannot bind both complete active-frame domains')
    return payload


def coupled_shot_binding_valid(payload,plan):
    if not isinstance(payload,Mapping) or payload.get('contract')!=CONTRACT:return False
    index=payload.get('segment_index')
    if (type(index) is not int or not 0<index<len(plan.segments)-2
            or payload.get('boundary_indices')!=[index-1,index]
            or payload.get('protected_subject')!='cut_anchor_head_not_every_background_face'
            or type(payload.get('incoming_native_cuts_in_active_window')) is not int
            or payload['incoming_native_cuts_in_active_window']!=0
            or any(payload.get(key) is not False for key in ('audio_or_source_phase_changed',
                'identity_or_gesture_certified','professional_quality_approved','learned_transformer_inference'))):return False
    try:
        if (_digest({k:v for k,v in payload.items() if k!='content_sha256'})!=payload.get('content_sha256')
                or coupled_geometry_identity(plan,index)!=payload.get('geometry_identity')):return False
        outgoing=payload.get('outgoing')
        if (not shot_context_binding_valid(outgoing,plan) or outgoing.get('segment_index')!=index):return False
        segment=plan.segments[index+1];counts=segment_frame_counts(plan);track=segment.editorial_transform
        rows=payload.get('incoming_head_evidence');native=payload.get('incoming_native_frame_ids')
        neutral=payload.get('incoming_neutral_from_frame')
        if (segment.crop is not None or segment.pan_crop is not None or segment.outgoing_overlap_transform is not None
                or track is None or any(t.kind.value!='hard_cut' for t in plan.transitions[index-1:index+2])
                or type(payload.get('incoming_source_index')) is not int
                or payload['incoming_source_index']!=segment.source_index
                or type(neutral) is not int or not 1<=neutral<min(14,counts[index+1])
                or neutral/plan.fps>.65 or not isinstance(rows,list) or len(rows)!=neutral+1
                or not isinstance(native,list) or len(native)!=len(rows)
                or any(type(n) is not int or n<0 for n in native)):return False
        data=payload.get('incoming_source_clock')
        if not isinstance(data,dict):return False
        clock=VerifiedRenderFrameClock(Fraction(data['frame_rate']),data['frame_count'],data['packet_grid_sha256'],data['grid_basis'])
        if clock.to_dict()!=data:return False
        start=sum(counts[:index+1]);ticks=native_frame_clocks_for_output(plan,{segment.source_index:clock},range(start,start+len(rows)))
        if (native!=[ticks[n].native_frame_index for n in range(start,start+len(rows))]
                or any(tick.terminal_clone for tick in ticks.values())):return False
        previous_misses=0
        for frame,row in enumerate(rows):
            if not isinstance(row,Mapping) or type(row.get('frame')) is not int or row['frame']!=frame:return False
            face=row.get('face_box_xywh');head=row.get('head_envelope_xywh');basis=row.get('basis');misses=row.get('consecutive_detector_misses')
            if (not isinstance(face,list) or len(face)!=4 or not all(_numeric(v) for v in face)
                    or face[0]<0 or face[1]<0 or face[2]<=0 or face[3]<=0
                    or face[0]+face[2]>1.000001 or face[1]+face[3]>1.000001
                    or head!=list(_close_face_head_envelope(tuple(face)))
                    or type(misses) is not int or not 0<=misses<=2):return False
            if frame==0:
                if basis!='observed_cut_anchor' or misses!=0:return False
            elif basis=='pixel_flow_plus_observed_geometry':
                if misses!=0:return False
            elif basis=='bounded_pixel_flow_containment':
                if misses!=previous_misses+1:return False
            else:return False
            previous_misses=misses;key=track.evaluate(frame/plan.fps)
            if (math.hypot(key.center_x-.5,key.center_y-.5)>.100001
                    or _project_box(tuple(head),_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees))[1]<.995):return False
        for frame in range(neutral,counts[index+1]):
            key=track.evaluate(frame/plan.fps)
            if (abs(key.zoom-1.)>1e-8 or abs(key.center_x-.5)>1e-8
                    or abs(key.center_y-.5)>1e-8 or abs(key.roll_degrees)>1e-8):return False
        return True
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,ZeroDivisionError):return False


def bound_coupled_shot_cameras(motion,plan):
    if not isinstance(motion,Mapping):return {}
    rows=motion.get('coupled_shot_cameras',());segments=motion.get('segments',())
    if not isinstance(rows,(list,tuple)) or not isinstance(segments,(list,tuple)):return {}
    serialized=plan.to_json()['segments'];result={}
    for row in rows:
        if not coupled_shot_binding_valid(row,plan):return {}
        index=row['segment_index']
        for i in (index,index+1):
            if i in result or i>=len(segments):return {}
            item=segments[i]
            if (not isinstance(item,Mapping) or item.get('segment_index')!=i
                    or item.get('source_index')!=plan.segments[i].source_index
                    or item.get('applied') is not True
                    or item.get('keyframes')!=list(serialized[i]['editorial_transform']['keyframes'])):return {}
            result[i]=row
    return result


def context_boundary_indices(contexts):
    return {boundary for index,row in contexts.items()
        for boundary in (row['boundary_indices'] if row.get('contract')==CONTRACT else (index-1,index))}
