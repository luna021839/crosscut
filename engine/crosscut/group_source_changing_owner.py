"""Normal-app whole-shot SOURCE alternatives with fresh retained families.

This is a bounded, inspectable focal-camera producer, NOT a named-performer,
choreographic-phase, physical-head-mask or professional-edit authority.  The
existing renderer/QA and their unconditional performer-review quality blocker
remain authoritative.  Native detector alternatives are frozen BEFORE cameras;
all incident comparisons use the SAME complete mutable camera state.

Supported slice: one/two whole cell-aligned shots, external source clocks,
face-pan + measured head/arm motion + all-tick native focal ROI/part checks +
projected descriptors + enabled boundary look. Incident legacy learned/affine/
formation/shot-context authorities are declined, not silently carried over.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import math
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from . import group_performer_family_owner as family
from .boundary_subject_alignment import (
    BoundarySubjectAlignmentDecision, _TransformState, _project_box, _project_point,
    _close_face_head_envelope, _action_envelope, BoundarySubjectAlignmentConfig)
from .editorial_motion import EditorialMotionSegment, _fixed_crop_at
from .editorial_transform_events import rendered_transform_event_records
from .face_match import FaceMatchSegment
from .group_actual_shots import visible_group_shots, visible_group_cut_inventory
from .group_joint_observations import GroupJointObservationSession, _file_identity
from .group_joint_source_camera import (
    GroupOutputRole, JointGroupWorkBudget, JointGroupBudgetExhausted,
    JointGroupEdgeObservation, SingleGroupDenseVerification, JointGroupDenseVerification,
    joint_state_from_render_plan, select_single_group_source_camera_path,
    select_joint_group_source_camera_path)
from .group_joint_state_factory import (candidate_joint_group_state,
    assemble_single_group_source_camera_state, assemble_joint_group_states)
from .group_joint_focal_camera import propose_endpoint_focal_face_roi_camera
from .group_joint_native_observer import observe_native_group, register_joint_native_models
from .group_native_head_orientation import (SUPPORTED_ANALYSIS_ROLLS,
    register_native_head_orientation_models)
from .group_focal_attention import focal_landmarks_track
from .group_native_head_window import observe_native_head_window_policy, _foreground
from .cut_anchored_containment import _iou
from .optional_native_worker import isolated_optional
from .render_boundary_clock import native_frame_clocks_for_output
from .render import segment_frame_counts

CONTRACT = 'normal_source_changing_family_owner_v1'
DENSE_CONTRACT = 'source_native_focal_whole_camera_refresh_v1'
SCENE_DENSE_CONTRACT = 'source_native_scene_whole_camera_refresh_v2'
DENSE_CONTRACTS = (DENSE_CONTRACT, SCENE_DENSE_CONTRACT)
MAX_REFRESHED_CELLS = 24
MAX_MUTABLE_FRAMES = 300
MAX_SOURCES = 8
MAX_RECORDED_FAILURES = 48
# These counters cover the shared native session, not legacy descriptor/model,
# matched-render subprocess, parent IPC, or final decoded QA allocations.
WORK_LIMITS = dict(maximum_states=24, maximum_edges=128, maximum_dense_paths=4,
    maximum_new_decodes=1200, maximum_model_calls=2400,
    maximum_projected_cache_bytes=32*1024*1024)


def geometry_payload(plan):
    payload = family.plan_payload(plan)
    for segment in payload['segments']:
        segment.pop('boundary_look_track', None)
        segment.pop('look_adjustment', None)
    return payload


def installed_native_assets(context):
    """Resolve only installed, pinned assets. Never download or infer a model."""
    from .pose import FORMATION_PERSON_DETECTOR_MODEL, _EXPECTED_SHA256, _MODEL_DIRECTORY
    from .visual_score import _YUNET_MODEL, YUNET_EXPECTED_SHA256
    # RTMPose is installed in the source-first SSD cache, while NanoDet is
    # shipped in the package model directory. Its location is not inferred
    # from the unrelated pose weight's parent. Both assets remain hash-pinned
    # and bound by the normal parent replay; nothing is downloaded here.
    person = _MODEL_DIRECTORY / FORMATION_PERSON_DETECTOR_MODEL
    assets = dict(person_model_path=person, pose_model_path=context.model_path,
        head_model_path=_YUNET_MODEL)
    expected = dict(person_model_path=_EXPECTED_SHA256[FORMATION_PERSON_DETECTOR_MODEL],
        pose_model_path=context.model_sha256, head_model_path=YUNET_EXPECTED_SHA256)
    for name, path in assets.items():
        if not path.is_file() or family._file_hash(path) != expected[name]:
            raise ValueError('source-changing installed pinned native asset unavailable: '+str(path))
    return assets


def asset_payload(assets):
    return {name:dict(path=str(path), identity=list(_file_identity(path)),
        sha256=family._file_hash(path)) for name,path in sorted(assets.items())}


def _at(state, tick):
    relative = tick-state.first_output_frame
    if not 0 <= relative < len(state.native_frame_indices):
        raise ValueError('source-family sample outside complete state')
    native = state.native_frame_indices[relative]
    for segment, count in zip(state.segments, state.segment_output_frame_counts, strict=True):
        if relative < count:
            return segment, relative/state.output_fps, native
        relative -= count
    raise AssertionError('complete state frame inventory mismatch')


def _base_box(box, segment, time):
    crop = segment.crop
    if crop is not None:
        crop = (crop.x, crop.y, crop.width, crop.height)
    elif segment.pan_crop is not None:
        crop = _fixed_crop_at(segment.pan_crop, time)
    if crop is None:
        return tuple(box)
    x,y,w,h = crop
    if abs(w/h-1.) > .04:
        raise ValueError('source family needs aspect-preserving base crop')
    return ((box[0]-x)/w, (box[1]-y)/h, box[2]/w, box[3]/h)


def project_box(box, state, tick):
    segment,time,_ = _at(state,tick)
    base = _base_box(box,segment,time)
    key = segment.editorial_transform.evaluate(time) if segment.editorial_transform else _TransformState()
    projected, fraction = _project_box(base,_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees))
    # Pre-crop visibility is independently required; a later zoom cannot recover
    # pixels already removed by the actual base crop.
    _,base_fraction = _project_box(base,_TransformState())
    return projected,min(fraction,base_fraction)


def project_point(point,state,tick):
    segment,time,_ = _at(state,tick)
    x,y,_,_ = _base_box((*point,0.,0.),segment,time)
    key = segment.editorial_transform.evaluate(time) if segment.editorial_transform else _TransformState()
    return _project_point((x,y),_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees))


def _dominant(row, role):
    if not row.heads:
        raise ValueError('native_focal_head_unknown')
    # Inverse-rotated detector boxes inflate in native axes. Reuse the
    # native-window's original-pixel eye-area ranking, not that inflated ROI.
    head = _foreground(row.heads,*row.frame_size_wh)
    if head is None:
        raise ValueError('native_focal_head_ambiguous_not_cropped_away')
    x,y,w,h = head.box_xywh
    if (x<0 or y<0 or x+w>1.000001 or y+h>1.000001 or len(head.landmarks_xy)!=5):
        raise ValueError('native_focal_face_roi_or_landmarks_incomplete')
    owners = [p for p in row.people if getattr(p,'unique_head_observation_id',None)==head.observation_id]
    person = owners[0] if len(owners)==1 else None
    if role == GroupOutputRole.FORMATION:
        raise ValueError('required_formation_refresh_not_implemented_no_focal_relabel')
    if role == GroupOutputRole.FOCAL_HEAD_ARMS:
        if (person is None or any(dict(person.part_states).get(part)!='observed'
                for part in ('left_arm','right_arm'))):
            raise ValueError('predeclared_native_arm_ownership_or_parts_unknown')
    return head,person


class NativeFamilySensors:
    """One live raw-native cache/model owner; no camera-dependent cohort choice."""
    def __init__(self,session,assets,*,frozen_declarations=None):
        self.session,self.assets = session,assets
        register_joint_native_models(session,**assets)
        register_native_head_orientation_models(session,assets['head_model_path'])
        self.declarations = {}
        self.raw_rows = {}
        if frozen_declarations is not None:
            for row in frozen_declarations:
                self.declarations[tuple(row['span_key'])] = row

    @staticmethod
    def span_key(state):
        return (state.eligible_source_index,state.first_output_frame,state.end_output_frame,
            state.native_frame_indices[0],state.native_frame_indices[-1],state.role.value)

    def raw(self,state,tick,roll):
        source = self.session.domain.request_index(state.eligible_source_index)
        native = state.native_frame_indices[tick-state.first_output_frame]
        key = source,native,roll
        if key not in self.raw_rows:
            row = observe_native_group(self.session,request_source_index=source,
                native_frame_index=native,head_analysis_roll=roll)
            self.session.budget.spend('pose_evidence_bytes',len(__import__('json').dumps(row.to_dict(),
                sort_keys=True,separators=(',',':'),allow_nan=False).encode()))
            self.raw_rows[key]=row
        return self.raw_rows[key]

    def declare(self,state):
        key = self.span_key(state)
        if key in self.declarations:
            declaration=self.declarations[key]
            if declaration.get('available') is False:
                raise ValueError(declaration['reason'])
            return declaration
        # Native-only, generic bounded policy, declared before ANY camera score.
        # Preserve every variant and competing head; never combine them into a
        # fictitious multi-person scene. Conflict stays ambiguity, not +60 rescue.
        ticks = tuple(sorted({state.first_output_frame,
            min(state.end_output_frame-1,state.first_output_frame+3),
            max(state.first_output_frame,state.end_output_frame-4),state.end_output_frame-1}))
        # Discover on three consecutive DISTINCT original native frames. Do
        # not compare long-gap LK or require every rotated background face to
        # equal the winning foreground face. All variants remain in evidence.
        native_indices=tuple(dict.fromkeys(state.native_frame_indices))[:3]
        if len(native_indices)!=3:
            raise ValueError('unique_native_window_samples_unavailable')
        source=self.session.domain.request_index(state.eligible_source_index)
        policy=observe_native_head_window_policy(self.session,
            request_source_index=source,native_frame_indices=native_indices)
        policy_payload=policy.to_dict()
        self.session.budget.spend('pose_evidence_bytes',len(__import__('json').dumps(
            policy_payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()))
        chosen=policy.selected_analysis_roll_degrees
        declaration=dict(span_key=list(key),anchor_output_ticks=list(ticks),analysis_roll=chosen,
            native_head_window_policy=policy_payload,native_body_anchor_observations=[],
            available=policy.selected,reason=policy.reason,
            policy='shared_pre_camera_native_eye_area_pixel_window_v1',
            role=state.role.value,pre_camera_subject_policy=True,
            whole_head_extent_certified=False,cross_camera_identity_certified=False)
        # Preserve unavailable variants too; an absent result is not evidence
        # that this source was physically unusable throughout the song.
        self.declarations[key]=declaration
        if not policy.selected:
            raise ValueError(policy.reason)
        try:
            for tick in ticks:
                raw=self.raw(state,tick,chosen)
                declaration['native_body_anchor_observations'].append(raw.to_dict())
                _dominant(raw,state.role)
        except ValueError as exc:
            declaration.update(available=False,reason=str(exc))
            raise
        return declaration

    def frame(self,state,tick):
        declaration=self.declare(state)
        raw=self.raw(state,tick,declaration['analysis_roll'])
        if abs(raw.frame_size_wh[0]/raw.frame_size_wh[1]-16/9) > .04:
            raise ValueError('native source aspect/padding projection not supported')
        head,person=_dominant(raw,state.role)
        box,fraction=project_box(head.box_xywh,state,tick)
        if fraction < BoundarySubjectAlignmentConfig().minimum_face_visible_fraction:
            raise ValueError('fresh_native_focal_face_roi_clipped')
        # Preserve the legacy measured-face + headroom FRAMING obligation. It
        # remains a face-derived envelope, never a segmented physical head.
        headroom=_close_face_head_envelope(head.box_xywh)
        projected_headroom,headroom_fraction=project_box(headroom,state,tick)
        if headroom_fraction < BoundarySubjectAlignmentConfig().minimum_face_visible_fraction:
            raise ValueError('fresh_native_face_derived_headroom_would_be_clipped')
        arms=[]
        if person is not None:
            # Observed arms remain obligations even on a nominal head role.
            for name,joints in (('left_arm',(0,2,4)),('right_arm',(1,3,5))):
                if dict(person.part_states).get(name)!='observed':
                    continue
                points=tuple(project_point(person.upper_body_xyc[j][:2],state,tick) for j in joints)
                if any(not (0.<=x<=1. and 0.<=y<=1.) for x,y in points):
                    raise ValueError('fresh_native_observed_arm_would_be_clipped')
                # Reuse the existing wrist/hand padding policy; joint points
                # alone must not silently weaken the old dense arm envelope.
                envelope=_action_envelope(head.box_xywh,tuple(person.upper_body_xyc[j][:2] for j in joints))
                _,part_fraction=project_box(envelope,state,tick)
                if part_fraction < BoundarySubjectAlignmentConfig().minimum_performer_visible_fraction:
                    raise ValueError('fresh_native_observed_arm_envelope_would_be_clipped')
                arms.append(dict(part=name,projected_joint_xy=[list(p) for p in points],
                    native_action_envelope_xywh=list(envelope),visible_fraction=part_fraction))
        return dict(output_tick=tick,request_source_index=raw.request_source_index,
            native_frame_index=raw.native_frame_index,native_observation=raw.to_dict(),
            selected_frame_local_head_id=head.observation_id,
            raw_face_roi_xywh=list(head.box_xywh),raw_landmarks_xy=[list(p) for p in head.landmarks_xy],
            projected_face_roi_xywh=list(box),face_roi_visible_fraction=fraction,
            face_derived_headroom_xywh=list(headroom),projected_headroom_xywh=list(projected_headroom),
            headroom_visible_fraction=headroom_fraction,headroom_basis='legacy_face_derived_framing_not_anatomy',
            projected_observed_arms=arms,role=state.role.value,analysis_roll=declaration['analysis_roll'],
            whole_head_extent_certified=False,cross_camera_identity_certified=False,body_phase_certified=False)

    def dense(self,path):
        result=[]
        for state in path[1:-1]:
            rows=[];previous=None;previous_pixels=None
            for tick in range(state.first_output_frame,state.end_output_frame):
                self.session.budget.check()
                row=self.frame(state,tick)
                pixels=self.session.native_pixels(row['request_source_index'],row['native_frame_index'])
                if previous is not None:
                    same_native=(row['native_frame_index']==previous['native_frame_index'])
                    if not same_native and not focal_landmarks_track(previous_pixels,pixels,
                            tuple(map(tuple,previous['raw_landmarks_xy'])),tuple(map(tuple,row['raw_landmarks_xy']))):
                        raise ValueError('native_focal_temporal_pixel_path_unknown')
                projected=self.session.projected_pixels(state,tick)
                if projected is None: raise ValueError('dense_projected_native_pixels_unavailable')
                row['projected_pixels_sha256']=sha256(memoryview(projected)).hexdigest()
                row['same_source_previous_pixel_link_observed']=previous is not None
                rows.append(row);previous=row;previous_pixels=pixels
            result.append(dict(first_output_frame=state.first_output_frame,end_output_frame=state.end_output_frame,
                role=state.role.value,rows=rows))
        return result


def _states(plan,domain,shots,roles):
    return tuple(joint_state_from_render_plan(plan=plan,domain=domain,slot=i,
        first_output_frame=s.first_output_frame,end_output_frame=s.end_output_frame,role=roles[i])
        for i,s in enumerate(shots))


def _shot_roles(context,shots):
    # An incoming source's raw wide/close label never chooses its duty. Retain
    # existing declared face/formation obligations before visiting alternatives.
    alignment=context.incumbent.editorial_motion_plan.boundary_subject_alignment_plan
    roles=[]
    for shot in shots:
        decisions=[d for d in alignment.decisions if d.eligible and
            shot.first_segment_index-1<=d.boundary_index<=shot.last_segment_index]
        kinds={d.subject_kind for d in decisions}
        if kinds & {'formation','silhouette'}:
            role=GroupOutputRole.FORMATION
        elif kinds and kinds <= {'face'}:
            role=GroupOutputRole.FOCAL_HEAD
        else:
            # Explicit output closeups in the incumbent's face family are a
            # pre-existing duty, not a source-ID or raw candidate-family gate.
            choices=context.incumbent.visual_timeline.choices[shot.first_segment_index:shot.last_segment_index+1]
            face=context.incumbent.face_match_plan.segments[shot.first_segment_index:shot.last_segment_index+1]
            role=(GroupOutputRole.FOCAL_HEAD if all(getattr(c.shot_type,'value',c.shot_type) in
                ('closeup','close_up','medium') for c in choices) and all(f.alignment_subject=='face' for f in face)
                else GroupOutputRole.FOCAL_HEAD_ARMS)
        roles.append(role)
    return tuple(roles)


def source_static_looks(context):
    """Reuse a grade only for its SAME source in the frozen full incumbent.

    A source-independent disabled family is explicit. Otherwise only a unique
    per-source recipe across the entire old plan is reusable. Missing/locally
    varying recipes are unavailable, never an invented neutral correction.
    This is pre-existing renderer authority, not fresh color/quality approval;
    the incident boundary look is still measured and encoded again.
    """
    plan=context.incumbent.render_plan
    if all(s.look_adjustment is None for s in plan.segments):
        return {i:None for i in context.domain.eligible_indices}
    values={}
    for index in context.domain.eligible_indices:
        recipes={s.look_adjustment for s in plan.segments if s.source_index==index}
        if len(recipes)==1:values[index]=next(iter(recipes))
    return values


def source_static_look_bindings(context):
    plan=context.incumbent.render_plan
    disabled=all(s.look_adjustment is None for s in plan.segments)
    return [dict(request_source_index=index,
        look_adjustment=None if value is None else asdict(value),
        original_cell_indices=[i for i,s in enumerate(plan.segments) if s.source_index==index],
        basis='entire_incumbent_static_family_disabled' if disabled else 'same_source_unique_incumbent_grade',
        fresh_color_quality_approved=False)
        for index,value in sorted(source_static_looks(context).items())]


def _hypothesis(context,incumbent,source,track=None):
    state=candidate_joint_group_state(plan=context.incumbent.render_plan,domain=context.domain,
        incumbent_state=incumbent,eligible_source_index=source,camera_track=track)
    # Numeric pan is a camera hypothesis only. No old source observation, face
    # metadata, static source grade, or dense receipt is copied into it.
    if any(s.crop is not None or s.outgoing_overlap_transform is not None
            for s in incumbent.segments):
        raise ValueError('incident_static_crop_or_overlap_needs_dedicated_source_refresh')
    request_source=context.domain.request_index(source)
    looks=source_static_looks(context)
    if request_source not in looks:
        raise ValueError('destination_source_static_look_unavailable_or_locally_varying')
    return replace(state,segments=tuple(replace(s,pan_crop=old.pan_crop,
        look_adjustment=looks[request_source])
        for s,old in zip(state.segments,incumbent.segments,strict=True))).bind_source(context.domain)


def _assembler(path):
    return assemble_single_group_source_camera_state if len(path)==3 else assemble_joint_group_states


def _observe_boundary(plan,domain,a,b,sensors,session,provider):
    from .group_motion_camera_refinement import observe_projected_boundary,measured_output_face_box
    from .observed_head_motion import HEAD_MOTION_BASIS
    from .closeup_gesture import MOTION_CONTRACT
    ticks=(b.first_output_frame-4,b.first_output_frame-1,b.first_output_frame,b.first_output_frame+3)
    if ticks[0]<a.first_output_frame or ticks[-1]>=b.end_output_frame:
        return None,None
    states=(a,a,b,b);frames=[];samples=[];raw=[]
    offsets=family._frame_offsets(plan)
    from bisect import bisect_right
    from .render_boundary_clock import RenderedBoundaryNativeFrame
    for state,tick in zip(states,ticks,strict=True):
        try: row=sensors.frame(state,tick)
        except ValueError: return None,None
        image=session.projected_pixels(state,tick)
        face=measured_output_face_box(image)
        # Output face finder cannot silently switch to a newly foregrounded
        # background dancer. The native selected face remains the authority.
        if face is None or _iou(tuple(row['projected_face_roi_xywh']),face)<.60:
            return None,None
        index=bisect_right(offsets,tick)-1
        # Same concrete clock type as hard_cut_boundary_sample_clocks.
        segment,relative,_ = _at(state,tick)
        clock=domain.clocks_by_request[row['request_source_index']]
        samples.append(RenderedBoundaryNativeFrame(tick,row['request_source_index'],
            row['native_frame_index'],segment.source_start+relative*segment.source_playback_rate,
            row['native_frame_index']/float(clock.frame_rate),tick-state.first_output_frame,
            bisect_right(offsets,state.first_output_frame)-1,False))
        frames.append(image);raw.append(row)
    required=(HEAD_MOTION_BASIS if all(s.role==GroupOutputRole.FOCAL_HEAD for s in states) else MOTION_CONTRACT)
    observed=observe_projected_boundary(tuple(frames),tuple(samples),provider,
        output_fps=plan.fps,required_basis=required)
    return observed,raw


def _fresh_motion(context,plan,timeline,changed,edges,dense):
    from .shot_context_binding import rebind_unchanged_shot_context
    from .coupled_shot_binding import rebind_unchanged_coupled_shot
    old=context.incumbent.editorial_motion_plan
    contexts=tuple(rebind_unchanged_shot_context(row,context.incumbent.render_plan,plan)
        for row in old.shot_context_cameras)
    coupled=tuple(rebind_unchanged_coupled_shot(row,context.incumbent.render_plan,plan)
        for row in old.coupled_shot_cameras)
    alignment=old.boundary_subject_alignment_plan
    rows=[]
    for decision in alignment.decisions:
        if decision.boundary_index not in edges:
            rows.append(decision);continue
        i=decision.boundary_index
        # Do not source-index-mutate a v5 certificate. This new dense family has
        # a separate schema/consumer and retains the original in the archive.
        rows.append(BoundarySubjectAlignmentDecision(i,i,i+1,timeline.choices[i].source_index,
            timeline.choices[i+1].source_index,plan.segments[i].source_end,
            plan.segments[i+1].source_start,False,
            'replaced_by_separately_bound_fresh_native_whole_camera_family',subject_kind=decision.subject_kind))
    alignment=replace(alignment,decisions=tuple(rows),segments=tuple(
        replace(row,source_index=timeline.choices[i].source_index,track=plan.segments[i].editorial_transform,
            boundary_indices=tuple(j for j in row.boundary_indices if j not in edges))
        if i in changed else row for i,row in enumerate(alignment.segments)))
    segments=tuple(EditorialMotionSegment(i,timeline.choices[i].source_index,segment.editorial_transform,
        ('fresh_source_native_whole_camera_not_legacy_dense_certificate',)) if i in changed else old.segments[i]
        for i,segment in enumerate(plan.segments))
    events=rendered_transform_event_records(plan.segments,plan.fps)
    # Legacy authored-gesture counts and frame-visible targets are DIFFERENT
    # inventories. Recount surviving legacy reasons instead of preserving stale
    # old-source gesture totals or inventing a v5 'matched' reason for this new
    # independently bound camera. New cameras have their own dense inventory.
    reasons=[{r.split(':',1)[0] for r in row.reasons} for row in segments]
    matched={i for i,(a,b) in enumerate(zip(reasons,reasons[1:]))
        if any(r.endswith('_matched_hard_cut') for r in a & b)}
    matched.update(d.boundary_index for d in alignment.decisions if d.eligible)
    affine_matches={i for i,(a,b) in enumerate(zip(reasons,reasons[1:]))
        if any(r.startswith('orb_affine_') and r.endswith('_matched_hard_cut') for r in a & b)}
    def runs(reason):
        return sum(reason in row and (i==0 or reason not in reasons[i-1])
            for i,row in enumerate(reasons))
    interior=runs('reference_calibrated_same_source_interior_zoom_pulse')
    authored_zoom=len(matched)+interior+sum(runs(r) for r in (
        'reference_calibrated_push_in','reference_calibrated_pull_out'))
    authored=(('zoom',authored_zoom),('pan',alignment.pan_event_count),
        ('roll',alignment.roll_event_count),
        ('separately_bound_source_whole_cameras',len(dense['mutable_shots'])))
    return replace(old,segments=segments,boundary_subject_alignment_plan=alignment,
        shot_context_cameras=contexts,coupled_shot_cameras=coupled,
        zoom_event_count=len(events['zoom']),pan_event_count=len(events['pan']),roll_event_count=len(events['roll']),
        authored_event_counts=authored,affine_boundary_match_count=len(affine_matches),
        interior_zoom_pulse_count=interior,
        # An old accepted/residual scalar must not describe a NEW source.
        # The full old values remain in the mandatory original-bundle archive.
        # New observations live in source_camera_refresh, never in a v5 header.
        accepted_boundary_indices=tuple(i for i in old.accepted_boundary_indices if i not in edges),
        natural_match_boundary_indices=tuple(i for i in old.natural_match_boundary_indices if i not in edges),
        boundary_residual_attention_mismatches=tuple(row for row in old.boundary_residual_attention_mismatches
            if row[0] not in edges),
        boundary_match_count=len(matched),
        source_camera_refresh=dense)


def _fresh_face(context,plan,timeline,observed,dense,changed,refreshed):
    samples={}
    for span in dense['mutable_shots']:
        for row in span['rows']: samples[row['output_tick']]=row
    offsets=family._frame_offsets(plan);rows=[]
    for i,old in enumerate(context.incumbent.face_match_plan.segments):
        projected=[s for s in observed['samples'] if s['segment_index']==i]
        if i not in changed:
            rows.append(replace(old,projected_output_observations=dict(samples=projected,
                basis=observed['basis'])) if i in refreshed else old)
            continue
        raw=[samples[t] for t in range(offsets[i],offsets[i+1])]
        is_formation = all(r['role']==GroupOutputRole.FORMATION.value for r in raw)
        height=None if is_formation else median(r['raw_face_roi_xywh'][3] for r in raw)
        silhouette=(median(max(p['box_xywh'][1]+p['box_xywh'][3] for p in r['native_observation']['people'])
            -min(p['box_xywh'][1] for p in r['native_observation']['people']) for r in raw) if is_formation else None)
        pan=plan.segments[i].pan_crop
        zoom=median(1./_fixed_crop_at(pan,(t-offsets[i])/plan.fps)[3]
            for t in range(offsets[i],offsets[i+1])) if pan else 1.
        rows.append(FaceMatchSegment(i,timeline.choices[i].source_index,pan is not None,
            None if pan else 'fresh_face_fit_owned_by_whole_editorial_camera',height,
            None if is_formation else old.target_face_height,old.target_face_height/height if old.target_face_height and height else None,
            zoom,pan,alignment_subject='formation' if is_formation else 'face',
            observed_silhouette_height=silhouette,continuous_run_stitched=any(
                span['first_output_frame']<=offsets[i] and span['end_output_frame']>=offsets[i+1]
                and (span['first_output_frame']<offsets[i] or span['end_output_frame']>offsets[i+1])
                for span in dense['mutable_shots']),
            alignment_purpose='fresh_native_source_geometry_and_retested_camera_not_copied_recipe',
            projected_output_observations=dict(samples=projected,basis=observed['basis'],
                native_dense_sample_count=len(raw),raw_native_digest=family.digest(raw))))
    return replace(context.incumbent.face_match_plan,segments=tuple(rows))


def complete_source_candidate(context,path,sensors,session,assets,dense_rows,edge_rows,search_record):
    """Every enabled supported family is fresh before a candidate reaches CAS."""
    from .group_joint_render_observations import refresh_joint_group_render_observations
    from .group_boundary_evidence_preservation import preserve_boundary_evidence
    from .pipeline import _deterministic_exact_shot_clock_from_timeline,_validate_review_first_group_bundle
    from .timeline_review import review_timeline
    from .group_transition import GroupTransitionPlan
    if context.source_derivative_binding is not None:
        from .group_scene_source_owner import validate_bound_derivative_path
        validate_bound_derivative_path(context.source_derivative_binding,[s.eligible_source_index for s in path])
    old=context.incumbent
    observations=refresh_joint_group_render_observations.__wrapped__(plan=old.render_plan,
        timeline=old.visual_timeline,domain=context.domain,path=path,sync_scores=context.sync_scores,
        feature_cache_directory=None,feature_derived_cache_directory=None,
        require_persistent_feature_cache=False)
    proposed,changed=observations.plan,observations.changed_segment_indices
    if not changed or not any(proposed.segments[i].source_index!=old.render_plan.segments[i].source_index for i in changed):
        raise ValueError('source-changing owner requires an actual source change')
    refreshed=observations.refreshed_segment_indices
    if len(refreshed)>MAX_REFRESHED_CELLS: raise ValueError('source-family refresh cell budget exceeded')
    edges=tuple(sorted({e for i in changed for e in (i-1,i) if 0<=e<len(proposed.transitions)}))
    before_observed=family.observe_scope(old.render_plan,context.domain,refreshed,edges,context.model_path,session)
    observed=family.observe_scope(proposed,context.domain,refreshed,edges,context.model_path,session)
    plan,look,look_evidence=family.refresh_look(old,proposed,edges,session,context.domain)
    if geometry_payload(plan)!=geometry_payload(proposed):
        raise ValueError('look refresh changed an admitted source clock/camera')
    final_states=tuple(joint_state_from_render_plan(plan=plan,domain=context.domain,slot=s.slot,
        first_output_frame=s.first_output_frame,end_output_frame=s.end_output_frame,role=s.role,
        editorial_contract_sha256=s.editorial_contract_sha256) for s in path)
    scene = getattr(sensors,'scene_family',False)
    dense=dict(contract=SCENE_DENSE_CONTRACT if scene else DENSE_CONTRACT,geometry_sha256=family.digest(geometry_payload(plan)),
        changed_cells=list(changed),incident_edges=list(edges),state_keys=[s.key for s in final_states],
        window_bounds=[[s.first_output_frame,s.end_output_frame] for s in final_states],
        roles=[s.role.value for s in final_states],mutable_shots=dense_rows,
        incident_projected_observations=edge_rows,pre_camera_native_declarations=list(sensors.declarations.values()),
        source_domain=context.domain.to_dict(),native_assets=asset_payload(assets),
        predeclared_source_static_looks=source_static_look_bindings(context),
        every_mutable_output_tick_observed=True,face_roi_is_not_physical_head_extent=True,
        semantic_raw_v2_admission=False,cross_camera_identity_certified=False,body_phase_certified=False,
        decoded_output_quality_certified=False,quality_approved=False)
    if context.source_derivative_binding is not None:
        dense.update(source_derivative_binding=context.source_derivative_binding,
            derivative_window_source_indices=[s.eligible_source_index for s in final_states])
    if scene:
        dense.update(scene_hypotheses={s.editorial_contract_sha256:sensors.hypothesis(s) for s in path},
            hypothesis_keys=[s.editorial_contract_sha256 for s in path],
            scene_discovery=sensors.archive(),source_derivative_binding=context.source_derivative_binding)
    motion=_fresh_motion(context,plan,observations.timeline,changed,edges,dense)
    face=_fresh_face(context,plan,observations.timeline,observed,dense,changed,refreshed)
    repairs,affine,preserved=preserve_boundary_evidence(old_plan=old.render_plan,new_plan=plan,
        old_timeline=old.visual_timeline,new_timeline=observations.timeline,
        boundary_repairs=old.boundary_repairs,boundary_affine_plan=old.boundary_affine_plan)
    review=review_timeline(observations.timeline,available_source_count=len(context.domain.eligible_indices))
    candidate=replace(old,render_plan=plan,visual_timeline=observations.timeline,timeline_review=review,
        pre_render_score=float(review.weighted_score),editorial_motion_plan=motion,face_match_plan=face,
        boundary_look_match=look,boundary_repairs=repairs,boundary_affine_plan=affine,
        group_transition_plan=GroupTransitionPlan(plan.transitions,analysis_basis=CONTRACT),
        stage_edit_exact_shot_clock=_deterministic_exact_shot_clock_from_timeline(observations.timeline,
            section_plan=old.section_plan,phrase_times=old.phrase_times),
        candidate_origin=CONTRACT,cadence_profile=CONTRACT,
        candidate_lineage=(*old.candidate_lineage,dict(stage=CONTRACT,changed_cells=list(changed),
            refreshed_cells=list(refreshed),source_changes_supported=True,quality_approved=False)))
    for i in range(len(plan.segments)):
        if i not in refreshed and any(a!=b for a,b in (
                (plan.segments[i],old.render_plan.segments[i]),
                (candidate.visual_timeline.choices[i],old.visual_timeline.choices[i]),
                (face.segments[i],old.face_match_plan.segments[i]),
                (motion.segments[i],old.editorial_motion_plan.segments[i]))):
            raise ValueError('source family changed an unrelated retained cell')
    receipt=dict(contract=CONTRACT,quality_approved=False,source_changes_supported=True,
        cross_camera_identity_certified=False,body_phase_certified=False,decoded_output_quality_certified=False,
        changed_cells=list(changed),refreshed_cells=list(refreshed),incident_edges=list(edges),
        plan_sha256=family.digest(family.plan_payload(plan)),metadata_sha256=family.digest(family.metadata_payload(candidate)),
        original_plan_sha256=family.digest(family.plan_payload(old.render_plan)),
        original_metadata_sha256=family.digest(family.metadata_payload(old)),
        source_domain=context.domain.to_dict(),synchronization_scores=list(context.sync_scores),
        source_identities=[list(v) for v in context.source_identities],dependencies=dict(context.code_dependencies),
        model_path=str(context.model_path),model_identity=list(context.model_identity),model_sha256=context.model_sha256,
        global_native_schedule_sha256=family._schedule_digest(family._schedule(plan,context.domain)),
        before_observations=before_observed,final_observations=observed,look_refresh=look_evidence,
        boundary_evidence_preservation=preserved,source_camera_refresh=dense,
        search=search_record,native_work=session.budget.to_dict(),
        resource_scope='single_native_session_only_legacy_refresh_parent_and_final_qa_counters_not_aggregated',
        final_quality_status='UNAPPROVED_PERFORMER_REVIEW_CANDIDATE')
    candidate=replace(candidate,performer_family_review=receipt)
    _validate_review_first_group_bundle(candidate,eligible_indices=context.domain.eligible_indices,
        offsets=context.domain.audio_offsets,timewarps=context.domain.source_timewarps,
        common_start=plan.master_audio_start,phrase_times=old.phrase_times)
    family._fresh(context)
    return candidate


def propose_source_changes(context):
    """Invoked in the NORMAL isolated producer; real source alternatives first."""
    from .closeup_gesture import CloseupGestureProvider
    from .group_motion_camera_refinement import projected_motion_preserved,camera_motion_rank
    family._fresh(context)
    attempts=[];all_candidates=[];session=None;provider=None;sensors=None
    def rejection_evidence():
        # A run with ZERO complete candidates still owns its observations.
        # Do not discard failed variant/cohort discovery at the IPC boundary.
        return dict(contract=CONTRACT,attempts=attempts,candidate_alternatives=all_candidates,
            native_work=session.budget.to_dict() if session is not None else None,
            pre_camera_native_declarations=list(sensors.declarations.values()) if sensors is not None else [],
            original_native_observations=[row.to_dict() for row in sensors.raw_rows.values()]
                if sensors is not None else [],quality_approved=False)
    try:
        if len(context.domain.eligible_indices)>MAX_SOURCES:
            raise ValueError('source-changing native source budget exceeded; no silent source truncation')
        assets=installed_native_assets(context)
        session=GroupJointObservationSession(domain=context.domain,budget=JointGroupWorkBudget(
            absolute_deadline=context.deadline,**WORK_LIMITS),maximum_raw_cache_bytes=32*1024*1024)
        sensors=NativeFamilySensors(session,assets)
        provider=CloseupGestureProvider(context.incumbent.render_plan.sources,context.model_path,
            verify_native_global_focal=True)
        plan=context.incumbent.render_plan
        shots=visible_group_shots(plan,context.domain)
        # A paired window does not depend on ANY earlier successful solo path.
        # The incumbent remains one option in each slot, so one-change paths
        # and true two-change paths compete in the same bounded product.
        windows=[shots[i:i+4] for i in range(max(0,len(shots)-3))]
        windows += [shots[i:i+3] for i in range(max(0,len(shots)-2))]
        for window in windows:
            if any(s.frame_count>MAX_MUTABLE_FRAMES for s in window[1:-1]): continue
            if len({i for s in window[1:-1] for i in range(s.first_segment_index,s.last_segment_index+1)})>MAX_REFRESHED_CELLS-2: continue
            try:
                roles=_shot_roles(context,window)
                if GroupOutputRole.FORMATION in roles:
                    raise ValueError('required_formation_or_silhouette_not_relabelled_focal')
                baseline=_states(plan,context.domain,window,roles)
                for state in (baseline[0],baseline[-1]): sensors.declare(state)
                states=[];generated=[]
                # Enumerate every eligible numeric source for EACH mutable
                # shot before observing/rejecting candidate head geometry.
                for incumbent in baseline[1:-1]:
                    values=[incumbent]
                    for source in range(len(context.domain.eligible_indices)):
                        try:
                            state=_hypothesis(context,incumbent,source)
                            generated.append(dict(slot=incumbent.slot,source=source,state_key=state.key))
                            if source!=incumbent.eligible_source_index: values.append(state)
                        except ValueError as exc:
                            attempts.append(dict(stage='numeric_source',slot=incumbent.slot,source=source,error=str(exc)))
                    states.append(values)
                # Fixed outer target ROIs, before alternative camera scoring.
                left=sensors.frame(baseline[0],baseline[0].end_output_frame-1)['projected_face_roi_xywh']
                right=sensors.frame(baseline[-1],baseline[-1].first_output_frame)['projected_face_roi_xywh']
                interior=tuple((a+b)/2 for a,b in zip(left,right,strict=True))
                camera_slots=max(0,(24-2-sum(map(len,states)))//len(states))
                for slot,values in enumerate(states,1):
                    cameras=[]
                    for state in tuple(values):
                        try:
                            sensors.declare(state)
                            first=sensors.frame(state,state.first_output_frame)
                            last=sensors.frame(state,state.end_output_frame-1)
                            a,ta,_=_at(state,state.first_output_frame);b,tb,_=_at(state,state.end_output_frame-1)
                            boxes=(_base_box(tuple(first['raw_face_roi_xywh']),a,ta),
                                _base_box(tuple(last['raw_face_roi_xywh']),b,tb))
                            targets=(tuple(left),tuple(right)) if len(baseline)==3 else (
                                (tuple(left),interior) if slot==1 else (interior,tuple(right)))
                            proposal=propose_endpoint_focal_face_roi_camera(source_face_boxes=boxes,
                                fixed_neighbor_face_boxes=targets,frame_count=len(state.native_frame_indices),fps=plan.fps)
                            camera=_hypothesis(context,baseline[slot],state.eligible_source_index,proposal.track)
                            cameras.append(camera)
                        except ValueError as exc:
                            if len(attempts)<MAX_RECORDED_FAILURES:
                                attempts.append(dict(stage='native_camera',slot=slot,source=state.eligible_source_index,error=str(exc)))
                    # Fair deterministic source order, not raw-family labels or
                    # a requirement that this source won a previous solo test.
                    values.extend(cameras[:camera_slots])
                native_edge_cache={};baseline_observed={};edge_payload={}
                def read_edge(a,b):
                    key=a.key,b.key
                    if key not in native_edge_cache:
                        native_edge_cache[key]=_observe_boundary(plan,context.domain,a,b,sensors,session,provider)
                    return native_edge_cache[key]
                for a,b in zip(baseline,baseline[1:]):
                    baseline_observed[b.first_output_frame]=read_edge(a,b)[0]
                def edge(a,b,budget):
                    if a.eligible_source_index==b.eligible_source_index: return None
                    if context.source_derivative_binding is not None:
                        from .group_scene_source_owner import relation
                        from .source_derivative_diversity import DERIVATIVE_DISTINCT
                        if relation(context.source_derivative_binding,a.eligible_source_index,b.eligible_source_index)!=DERIVATIVE_DISTINCT:
                            return None
                    observed,raw=read_edge(a,b)
                    if observed is None or observed.face_metrics is None: return None
                    before=baseline_observed.get(b.first_output_frame)
                    known_preserved=True
                    if before is not None:
                        if before.face_metrics is not None and any(new>old+1e-9 for old,new in
                                zip(before.face_metrics,observed.face_metrics,strict=True)):
                            known_preserved=False
                        if before.motion is not None and not projected_motion_preserved(observed.motion,before.motion):
                            known_preserved=False
                    complete=(camera_motion_rank(observed.motion,observed.face_metrics) is not None and known_preserved)
                    edge_payload[(a.key,b.key)]=dict(first_incoming_output_frame=b.first_output_frame,
                        before_motion=None if before is None or before.motion is None else before.motion.to_dict(),
                        before_face_metrics=None if before is None else before.face_metrics,
                        proposed_motion=None if observed.motion is None else observed.motion.to_dict(),
                        proposed_face_metrics=observed.face_metrics,known_components_preserved=known_preserved,
                        native_focal_samples=raw)
                    return JointGroupEdgeObservation(a.key,b.key,b.first_output_frame,
                        sum(observed.face_metrics)/2,True,complete)
                completed={}
                def verify(path,edges,budget):
                    verification_type=SingleGroupDenseVerification if len(path)==3 else JointGroupDenseVerification
                    ticks=tuple(tuple(range(s.first_output_frame,s.end_output_frame)) for s in path[1:-1])
                    alternative=None
                    try:
                        assembled,changed=_assembler(path)(plan=plan,domain=context.domain,path=path)
                        if not any(assembled.segments[i].source_index!=plan.segments[i].source_index for i in changed):
                            raise ValueError('camera_only_not_a_source_changing_candidate')
                        alternative=dict(plan=assembled.to_json(),states=[s.to_dict() for s in path],
                            changed_cells=list(changed),status='pending_native_and_complete_family_refresh')
                        all_candidates.append(alternative)
                        dense_rows=sensors.dense(path)
                        alternative['fresh_native_dense']=dense_rows
                        # Preserve EACH incumbent observed arm at its actual
                        # output tick, even when the declared role is head-only.
                        known=[]
                        for original,new,span in zip(baseline[1:-1],path[1:-1],dense_rows,strict=True):
                            for row in span['rows']:
                                tick=row['output_tick']
                                try:
                                    declaration=sensors.declare(original)
                                    native=sensors.raw(original,tick,declaration['analysis_roll'])
                                    _,person=_dominant(native,GroupOutputRole.FOCAL_HEAD)
                                except ValueError:
                                    known.append(dict(output_tick=tick,known_parts=None));continue
                                parts=tuple(name for name in ('left_arm','right_arm') if person is not None
                                    and dict(person.part_states).get(name)=='observed')
                                new_parts={p['part'] for p in row['projected_observed_arms']}
                                if not set(parts)<=new_parts:
                                    raise ValueError('source_change_lost_an_incumbent_observed_arm')
                                known.append(dict(output_tick=tick,known_parts=list(parts),
                                    native_observation_sha256=native.sha256))
                        alternative['incumbent_known_part_preservation']=known
                        record=dict(generated_numeric_states=generated,
                            prior_solo_success_required=False,raw_family_gate=False,
                            baseline_unknown_not_imputed=True,incumbent_known_parts=known,
                            prior_complete_rejections=family.plain(all_candidates[:-1]),window_bounds=[[s.first_output_frame,s.end_output_frame] for s in path])
                        candidate=complete_source_candidate(context,path,sensors,session,assets,dense_rows,
                            [edge_payload[(a.key,b.key)] for a,b in zip(path,path[1:])],record)
                        completed[tuple(s.key for s in path)]=candidate
                        return verification_type(tuple(s.key for s in path),ticks[0] if len(path)==3 else ticks,
                            True,'complete_source_and_all_supported_active_families_refreshed_unapproved')
                    except (ValueError,RuntimeError,OSError) as exc:
                        if alternative is not None:
                            alternative.update(status='rejected',error=str(exc))
                        # Try the next complete path, not just the first numeric
                        # winner that later fails look/face/dense refresh.
                        attempts.append(dict(stage='complete_family_refresh',state_keys=[s.key for s in path],error=str(exc)))
                        return verification_type(tuple(s.key for s in path),ticks[0] if len(path)==3 else ticks,False,str(exc))
                options=dict(domain=context.domain,left=baseline[0],right=baseline[-1],declared_roles=roles,
                    observe_edge=edge,verify_dense=verify,budget=session.budget,baseline_path=baseline,
                    baseline_unknown_policy='observe_candidate')
                selection=(select_single_group_source_camera_path(states=tuple(states[0]),**options) if len(baseline)==3
                    else select_joint_group_source_camera_path(states_by_slot=tuple(tuple(v) for v in states),**options))
                attempts.append(dict(stage='whole_source_camera_selection',result=selection.to_dict(),
                    generated_numeric_states=generated))
                if selection.path is not None:
                    return completed[tuple(s.key for s in selection.path)]
                if selection.status=='budget_retained_baseline': break
            except (ValueError,RuntimeError,OSError) as exc:
                attempts.append(dict(stage='window',bounds=[s.to_dict() for s in window],error=str(exc)))
            session.budget.check()
        return family.PerformerFamilyRejection('no complete source-changing supported-family candidate',
            rejection_evidence())
    except (ValueError,RuntimeError,OSError) as exc:
        return family.PerformerFamilyRejection(str(exc),rejection_evidence())
    finally:
        if provider is not None: provider.close()
        if session is not None: session.close()


def validate_dense_uncertainty(dense):
    if (not isinstance(dense,dict) or dense.get('contract') not in DENSE_CONTRACTS
            or dense.get('every_mutable_output_tick_observed') is not True
            or dense.get('face_roi_is_not_physical_head_extent') is not True
            or any(dense.get(k) is not False for k in ('semantic_raw_v2_admission',
                'cross_camera_identity_certified','body_phase_certified',
                'decoded_output_quality_certified','quality_approved'))
            or not isinstance(dense.get('mutable_shots'),list) or not 1<=len(dense['mutable_shots'])<=2):
        raise ValueError('fresh native family cannot acquire semantic or quality authority')
    if dense.get('source_derivative_binding') is not None:
        from .group_scene_source_owner import validate_bound_derivative_path
        validate_bound_derivative_path(dense['source_derivative_binding'],dense.get('derivative_window_source_indices',[]))
    for span in dense['mutable_shots']:
        rows=span.get('rows')
        if not isinstance(rows,list) or not 1<=len(rows)<=MAX_MUTABLE_FRAMES:
            raise ValueError('fresh native family dense extent is invalid')
        if any(any(row.get(k) is not False for k in ('whole_head_extent_certified',
                'cross_camera_identity_certified','body_phase_certified')) for row in rows):
            raise ValueError('native ROI rows cannot acquire anatomy, identity or phase authority')


def validate_binding(bundle):
    receipt=family.plain(bundle.performer_family_review)
    dense=family.plain(bundle.editorial_motion_plan.source_camera_refresh)
    if not isinstance(receipt,dict) or not isinstance(dense,dict):
        raise ValueError('source-changing receipt and full native family are required')
    if (bundle.candidate_origin!=CONTRACT or receipt.get('contract')!=CONTRACT
            or receipt.get('source_changes_supported') is not True
            or any(receipt.get(k) is not False for k in ('quality_approved','body_phase_certified',
                'cross_camera_identity_certified','decoded_output_quality_certified'))
            or receipt.get('plan_sha256')!=family.digest(family.plan_payload(bundle.render_plan))
            or receipt.get('metadata_sha256')!=family.digest(family.metadata_payload(bundle))
            or dense!=receipt.get('source_camera_refresh')
            or dense.get('contract') not in DENSE_CONTRACTS
            or dense.get('geometry_sha256')!=family.digest(geometry_payload(bundle.render_plan))):
        raise ValueError('source-changing complete camera/family/uncertainty binding changed')
    validate_dense_uncertainty(dense)
    if dense['contract']==SCENE_DENSE_CONTRACT:
        from .group_scene_observations import validate_scene_dense
        validate_scene_dense(dense)
    static=dense.get('predeclared_source_static_looks')
    if (not isinstance(static,list) or any(not isinstance(row,dict)
            or type(row.get('request_source_index')) is not int
            or row.get('fresh_color_quality_approved') is not False for row in static)
            or len({row['request_source_index'] for row in static})!=len(static)):
        raise ValueError('source-specific retained static-look binding missing or altered')
    bindings={row['request_source_index']:row['look_adjustment'] for row in static}
    for i in receipt['changed_cells']:
        segment=bundle.render_plan.segments[i]
        if (segment.source_index not in bindings or bindings[segment.source_index]!=
                (None if segment.look_adjustment is None else asdict(segment.look_adjustment))):
            raise ValueError('source change copied a different source static-look recipe')
    for span in dense['mutable_shots']:
        if [r['output_tick'] for r in span['rows']]!=list(range(span['first_output_frame'],span['end_output_frame'])):
            raise ValueError('source-changing dense family omitted an output tick')
        if any((r['face_roi_visible_fraction'] is None if r['role']!=GroupOutputRole.FORMATION.value else False)
                or (r['face_roi_visible_fraction'] is not None and r['face_roi_visible_fraction']<.995)
                or r['role']!=span['role'] for r in span['rows']):
            raise ValueError('source-changing dense family lost required role/face visibility')


def validate_parent(candidate,context):
    family._fresh(context);validate_binding(candidate)
    receipt=family.plain(candidate.performer_family_review)
    if (receipt['source_domain']!=context.domain.to_dict()
            or receipt['source_identities']!=[list(v) for v in context.source_identities]
            or receipt['dependencies']!=dict(context.code_dependencies)
            or receipt['synchronization_scores']!=list(context.sync_scores)
            or receipt['original_metadata_sha256']!=family.digest(family.metadata_payload(context.incumbent))):
        raise ValueError('source-changing parent preimage changed')
    if context.source_derivative_binding is not None and receipt['source_camera_refresh'].get('source_derivative_binding')!=context.source_derivative_binding:
        raise ValueError('scene parent derivative preimage changed')
    replay=family.propose_performer_family(context)  # independent isolated worker, same absolute deadline
    if isinstance(replay,family.PerformerFamilyRejection):
        raise ValueError('source-changing parent independent replay rejected: '+replay.error)
    if candidate.to_dict()!=replay.to_dict() or candidate.render_plan!=replay.render_plan:
        raise ValueError('source-changing parent independent complete sensor/family replay disagrees')
    family._fresh(context)


@isolated_optional()
def final_live_replay(plan_json,receipt,deadline):
    """Final LIVE original-native replay; not a decoded-output quality pass."""
    from fractions import Fraction
    from .render_plan_serialization import render_plan_from_json
    from .render_frame_schedule import VerifiedRenderFrameClock
    from .group_source_domain import GroupSourceDomain
    # Retain the original complete sparse projected face/motion replay, including
    # code/model/source binding, rather than replacing it with a child hash.
    metrics=family._final_live_replay.__wrapped__(plan_json,receipt,deadline)
    plan=render_plan_from_json(plan_json);raw=receipt['source_domain'];dense=receipt['source_camera_refresh']
    validate_dense_uncertainty(dense)
    indices=tuple(raw['eligible_request_indices'])
    clocks=tuple(VerifiedRenderFrameClock(Fraction(v['frame_rate']),v['frame_count'],
        v['packet_grid_sha256'],v['grid_basis']) for v in raw['render_frame_clocks'])
    domain=GroupSourceDomain(tuple(plan.sources),indices,(0.,)*len(indices),(None,)*len(indices),
        tuple(tuple(c) for c in raw['scene_cut_times']),clocks)
    assets={k:Path(v['path']) for k,v in dense['native_assets'].items()}
    if asset_payload(assets)!=dense['native_assets']: raise ValueError('source-changing final native assets changed')
    session=GroupJointObservationSession(domain=domain,budget=JointGroupWorkBudget(
        absolute_deadline=deadline,**WORK_LIMITS),maximum_raw_cache_bytes=32*1024*1024)
    try:
        if dense['contract']==SCENE_DENSE_CONTRACT:
            from .group_scene_observations import SceneSensors, validate_scene_dense
            validate_scene_dense(dense)
            sensors=SceneSensors(session,assets)
            sensors.hypotheses=dict(dense['scene_hypotheses'])
            keys=dense['hypothesis_keys']
        else:
            sensors=NativeFamilySensors(session,assets,frozen_declarations=dense['pre_camera_native_declarations'])
            keys=[None]*len(dense['window_bounds'])
        path=tuple(joint_state_from_render_plan(plan=plan,domain=domain,slot=i,first_output_frame=b[0],
            end_output_frame=b[1],role=GroupOutputRole(dense['roles'][i]),
            editorial_contract_sha256=keys[i]) for i,b in enumerate(dense['window_bounds']))
        if dense['contract']==SCENE_DENSE_CONTRACT:
            # Fresh original pixels and observations also bind FIXED neighbors;
            # child hashes alone are insufficient to establish scene hypotheses.
            for state in path:
                h=sensors.hypothesis(state)
                fresh=[sensors.raw(state,t,h['analysis_roll']).sha256 for t in h['anchor_output_ticks']]
                if fresh!=h['raw_observation_sha256']:
                    raise ValueError('scene final fixed/mutable anchor observations changed')
                if state.role!=GroupOutputRole.FORMATION:
                    for i in range(0,len(h['anchor_output_ticks']),2):
                        j=i+1
                        a=sensors.raw(state,h['anchor_output_ticks'][i],h['analysis_roll'])
                        b=sensors.raw(state,h['anchor_output_ticks'][j],h['analysis_roll'])
                        if h['anchor_head_ids'][i:i+2] not in sensors._head_links(a,b)['unique_links']:
                            raise ValueError('scene final fixed/mutable head pixel hypothesis changed')
        if [s.key for s in path]!=dense['state_keys'] or family.digest(sensors.dense(path))!=family.digest(dense['mutable_shots']):
            raise ValueError('source-changing final all-tick native/camera replay disagrees')
    finally: session.close()
    if [list(_file_identity(p)) for p in domain.eligible_sources]!=receipt['source_identities']:
        raise ValueError('source-changing source changed during final dense replay')
    if asset_payload(assets)!=dense['native_assets']: raise ValueError('source-changing native assets changed during final replay')
    metrics.update(contract=CONTRACT,dense_native_output_tick_count=sum(len(s['rows']) for s in dense['mutable_shots']),
        semantic_raw_v2_admission=False,cross_camera_identity_certified=False,body_phase_certified=False)
    metrics.pop('evidence_sha256',None);metrics['evidence_sha256']=family.digest(metrics)
    return metrics
