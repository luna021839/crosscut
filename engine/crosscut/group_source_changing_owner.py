"""Normal-app whole-shot SOURCE/CAMERA edits with fresh retained families.

This is a bounded, inspectable focal-camera producer, NOT a named-performer,
choreographic-phase, physical-head-mask or professional-edit authority.  The
existing renderer/QA and their unconditional performer-review quality blocker
remain authoritative.  Native detector alternatives are frozen BEFORE cameras;
all incident comparisons use the SAME complete mutable camera state.

Supported slice: one/two whole cell-aligned shots, external source clocks,
face-pan + measured head/arm motion + all-tick native focal ROI/part checks +
projected descriptors + retained enabled OR originally absent boundary look. Incident legacy learned/affine/
formation/shot-context authorities are declined, not silently carried over.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    assemble_single_group_source_camera_state, assemble_joint_group_states,
    fixed_neighbor_state_from_render_plan)
from .group_joint_focal_camera import propose_endpoint_focal_face_roi_camera
from .group_joint_native_observer import NativeGroupObservation, observe_native_group, register_joint_native_models
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
SCENE_DENSE_CONTRACT = 'source_native_scene_role_whole_camera_refresh_v6'
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
    from .group_scene_camera_queries import GroupSceneRoleQuery
    if isinstance(state,GroupSceneRoleQuery):
        return state.sample_at(tick)
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


@dataclass(frozen=True,slots=True)
class NativeMotionDemand:
    source_index: int
    native_frame_index: int
    output_tick: int
    source_family: str
    opportunity: str
    analysis_roll: float | None = None
    require_appearance: bool = False
    require_member_features: bool = False

    def __post_init__(self):
        if (any(type(n) is not int or n<0 for n in
                (self.source_index,self.native_frame_index,self.output_tick))
                or not isinstance(self.source_family,str) or not self.source_family
                or self.opportunity not in ('moving_closeup','other')
                or type(self.require_appearance) is not bool
                or type(self.require_member_features) is not bool):
            raise ValueError('native motion demand needs a physical address and original opportunity')
        from .group_native_head_orientation import native_head_analysis_spec
        native_head_analysis_spec(self.analysis_roll)

    @property
    def native_key(self):
        return self.source_index,self.native_frame_index,self.analysis_roll


@dataclass(frozen=True,slots=True)
class NativeMotionSupply:
    demand: NativeMotionDemand
    observation: NativeGroupObservation | None
    appearance: dict | None = None
    recorded_input_state: str | None = None
    source_equivalence_json: str | None = None
    member_features: dict | None = None

    @property
    def missing_modalities(self):
        if self.recorded_input_state in ('CONFLICTING_NATIVE_RECORDS','CONFLICTING_FEATURE_RECORDS'):
            return ()  # Conflicting measurements are not an invitation to retry until one wins.
        if self.observation is None:
            return ('RAW',) + (('APPEARANCE',) if self.demand.require_appearance else ()) + (
                ('MEMBER_FEATURES',) if self.demand.require_member_features else ())
        # A measured decode failure is retained, not a fresh completion miss.
        if self.observation.native_pixel_sha256 is None:return ()
        return (('APPEARANCE',) if self.demand.require_appearance and self.appearance is None else ()) + (
            ('MEMBER_FEATURES',) if self.demand.require_member_features and self.member_features is None else ())

    @property
    def state(self):
        if self.recorded_input_state in ('CONFLICTING_NATIVE_RECORDS','CONFLICTING_FEATURE_RECORDS'):
            return self.recorded_input_state
        if self.observation is None:return 'UNKNOWN_NOT_OBSERVED'
        if self.observation.native_pixel_sha256 is None:return 'EXACT_DECODE_UNAVAILABLE'
        if self.missing_modalities:
            return ('RAW_AVAILABLE_APPEARANCE_UNOBSERVED' if 'APPEARANCE' in self.missing_modalities
                else 'RAW_AVAILABLE_MEMBER_FEATURES_UNOBSERVED')
        # Available native rows can still have absent/ambiguous heads or limbs.
        return 'NATIVE_ROWS_AVAILABLE'


@dataclass(frozen=True,slots=True)
class NativeMotionInventory:
    supplies: tuple[NativeMotionSupply,...]
    attempted_native_keys: tuple
    completion_consumed: bool
    status: str
    completion_schedule_json: str | None = None


def _native_actor_anchor_cache_identity(anchor):
    if anchor is None:return None
    return family.digest(dict(raw=anchor['row']['native_observation'],
        head=anchor['descriptor'].frame_local_head_id,vector=anchor.get('vector'),
        ambiguity=anchor.get('reacquisition_ambiguous',False),
        association_state=anchor.get('selected_actor_association_state'),
        original_role=anchor.get('original_role'),scope=anchor.get('original_attention_scope')))


class NativeMotionViews:
    """Project one completed request inventory into explicit actor occurrences.

    All visible heads survive absent appearance. Only heads with a vector in
    the pinned space enter temporal association. The same connected path is
    returned to camera, hold and incident consumers; no default salience or
    framewise cosine reselection is performed here.
    """
    def __init__(self,sensors,inventory,*,body_receipt_bundles=()):
        import json
        from .group_review_body_recovery import merge_review_body_receipts
        if (not isinstance(inventory,NativeMotionInventory)
                or not inventory.completion_consumed or sensors.session.closed):
            raise ValueError('native motion views need the completed live request inventory')
        self.sensors=sensors;self.domain=sensors.session.domain
        self.supplies={};self.payloads={};self.records={};self.views={};self.paths={}
        for supply in inventory.supplies:
            # Live observers inherit typed tuple containers from raw.to_dict;
            # persisted inputs have JSON arrays. Canonicalize containers at
            # this consumer seam, exactly as the raw payload below. Do not
            # rewrite a physical parent, descriptor, coordinate or member hint.
            supply=replace(supply,
                appearance=None if supply.appearance is None else json.loads(json.dumps(
                    supply.appearance,sort_keys=True,allow_nan=False)),
                member_features=None if supply.member_features is None else json.loads(json.dumps(
                    supply.member_features,sort_keys=True,allow_nan=False)))
            key=supply.demand.native_key
            previous=self.supplies.get(key)
            if previous is not None:
                if previous.observation!=supply.observation:
                    raise ValueError('native motion aliases changed their raw observation')
                if previous.appearance is not None and supply.appearance is not None and previous.appearance!=supply.appearance:
                    raise ValueError('native motion aliases changed their appearance observation')
                if (previous.member_features is not None and supply.member_features is not None
                        and previous.member_features != supply.member_features):
                    raise ValueError('native motion aliases changed their member feature observation')
                supply=replace(supply,appearance=previous.appearance if supply.appearance is None else supply.appearance,
                    member_features=previous.member_features if supply.member_features is None else supply.member_features)
            self.supplies[key]=supply
            if supply.observation is not None:
                self.payloads[key]=json.loads(json.dumps(supply.observation.to_dict(),sort_keys=True,allow_nan=False))
        self.body_receipts,self.body_merge_audit=merge_review_body_receipts(*tuple(body_receipt_bundles))

    def record(self,query,tick,*,analysis_roll=None):
        from .group_scene_static_query import reproject_recorded_static_query
        from .group_review_recorded_index import validate_recorded_appearance
        from .native_face_descriptor import native_descriptor_space,SFACE_MODEL_NAME
        self.sensors.session.budget.check()
        if self.sensors.session.closed:raise ValueError('native motion views lost their live request session')
        query.bind_source(self.domain)
        _,_,native=query.sample_at(tick);source=self.domain.request_index(query.eligible_source_index)
        address=(source,native,analysis_roll)
        key=(query.key,tick,analysis_roll)
        if key in self.records:return self.records[key]
        supply=self.supplies.get(address);raw=self.payloads.get(address)
        if supply is None or raw is None or raw['native_pixel_sha256'] is None:
            result=dict(state='UNKNOWN_NOT_OBSERVED' if supply is None else supply.state,
                row=None,appearance=None,head_inventory=(),quality_approved=False)
        else:
            appearance=supply.appearance
            if appearance is not None:
                validate_recorded_appearance(raw,appearance,expected_space=native_descriptor_space(
                    self.sensors.session,SFACE_MODEL_NAME))
            member_features=supply.member_features
            if member_features is not None:
                from .ive_local_arcface_gallery import validate_recorded_member_features
                from .native_face_descriptor import ARCFACE_MODEL_NAME
                validate_recorded_member_features(raw,member_features,gallery=self.sensors.member_gallery,
                    expected_space=native_descriptor_space(self.sensors.session,ARCFACE_MODEL_NAME))
            row=reproject_recorded_static_query(query,tick,raw,domain=self.domain,compact=True)
            from .group_review_source_equivalence import freeze_verified_native_file_binding
            file_binding=freeze_verified_native_file_binding(self.sensors,raw,supply.source_equivalence_json)
            result=dict(state=supply.state,row=row,appearance=appearance,member_features=member_features,
                source_equivalence_json=supply.source_equivalence_json,
                verified_native_file_binding=file_binding,
                head_inventory=tuple(h['observation_id'] for h in raw['heads']),quality_approved=False)
        self.records[key]=result
        return result

    def native_signature(self,query,tick,*,analysis_roll=None):
        from bisect import bisect_right
        import json
        source=self.domain.request_index(query.eligible_source_index);_,_,native=query.sample_at(tick)
        self.sensors.session.budget.check()
        if self.sensors.session.closed:raise ValueError('native signature lost its live request session')
        # One live physical file defines the camera run, including unobserved
        # ticks. Recorded device aliases are independently bound in record();
        # their actual raw/body/feature parents are NOT rewritten here.
        identity=self.sensors.session._assert_source_identity(source)
        clock=self.domain.clocks_by_request[source]
        cuts=tuple(math.ceil(t*float(clock.frame_rate)-1e-9) for t in self.domain.cuts_for_request(source))
        return source,query.source_clock_identity,family.digest(identity),bisect_right(cuts,native),native

    def observation_status(self,query,tick,*,selected=None):
        """A compact exact-address diagnostic, never an identity certificate.

        Missing raw, missing features, an unmatched DP occurrence and an
        ambiguous selected occurrence are different failure layers. No
        embedding or private-gallery content leaves this receipt.
        """
        found=self.record(query,tick)
        candidates=self.candidates(query,tick)
        raw=None if found['row'] is None else found['row']['native_observation']
        return dict(output_tick=tick,request_source_index=self.domain.request_index(query.eligible_source_index),
            requested_native_frame_index=query.sample_at(tick)[2],native_supply_state=found['state'],
            raw_observed=raw is not None,visible_head_occurrences=len(candidates),
            appearance_ready_head_occurrences=sum(v['vector'] is not None for v in candidates),
            selected_head_id=None if selected is None else selected['descriptor'].frame_local_head_id,
            selected_actor_state=('UNMATCHED' if selected is None else
                'AMBIGUOUS' if selected.get('reacquisition_ambiguous',False) else 'OBSERVED_PATH_HYPOTHESIS'),
            original_role=None if query.role is None else query.role.value,
            private_embeddings_included=False,quality_approved=False)

    def candidates(self,query,tick,*,analysis_roll=None,association_ready=False):
        from .group_review_camera_repair import ReviewActorSelection,framing_descriptor,effective_face_detail_pixels
        from .group_review_body_recovery import (apply_review_primary_joint_observation,
            apply_review_body_receipt,head_fingerprint)
        self.sensors.session.budget.check()
        if self.sensors.session.closed:raise ValueError('native candidates lost their live request session')
        key=(query.key,tick,analysis_roll)
        if key not in self.views:
            found=self.record(query,tick,analysis_roll=analysis_roll);result=[]
            row=found['row']
            if row is not None:
                raw=row['native_observation'];appearance=found['appearance']
                embeddings={} if appearance is None else {h['observation_id']:h['embedding'] for h in appearance['heads']}
                member_features=found.get('member_features')
                member_hints={} if member_features is None else {
                    h['observation_id']:h['member_hint'] for h in member_features['heads']}
                for head in raw['heads']:
                    head_id=head['observation_id']
                    descriptor=framing_descriptor(row,actor_selection=ReviewActorSelection.occurrence(head_id))
                    if descriptor is None:continue
                    detail=effective_face_detail_pixels(descriptor)
                    near=min(1.,max(0.,(descriptor.face_height-.13)/.13))
                    if detail is not None:near*=min(1.,detail/150.)
                    view=dict(row=row,descriptor=descriptor,vector=embeddings.get(head_id),
                        verified_native_file_binding=found['verified_native_file_binding'],
                        appearance=appearance,member_hint=member_hints.get(head_id),near=near,detail_risk=None,
                        native_camera_unit_index=self.native_signature(query,tick,analysis_roll=analysis_roll)[3],
                        actor_association_state='APPEARANCE_UNOBSERVED' if embeddings.get(head_id) is None
                            else 'EXACT_HEAD_APPEARANCE_AVAILABLE',quality_approved=False)
                    view=apply_review_primary_joint_observation(view,project_point=lambda p:project_point(p,query,tick))
                    physical=(raw['request_source_index'],raw['native_frame_index'],raw['source_clock_identity'],
                        tuple(raw['source_file_identity']),raw['native_pixel_sha256'],head_fingerprint(head))
                    receipt=self.body_receipts.get(physical)
                    if receipt is not None:
                        window,evidence=receipt
                        view=apply_review_body_receipt(view,evidence,expected_window=window,
                            project_point=lambda p:project_point(p,query,tick))
                    elif any(v['physical_occurrence_identity']==family.digest(physical)
                            for v in self.body_merge_audit['conflicts']):
                        view=dict(view,review_body_recovery_state='CONFLICTING_FROZEN_BODY_RECEIPTS')
                    result.append(view)
            self.views[key]=tuple(result)
        values=self.views[key]
        return tuple(v for v in values if v['vector'] is not None) if association_ready else values

    def connected_actor_views(self,queries,ticks,anchor,*,analysis_roll=None):
        from .group_review_actor_path import connected_source_actor_views
        ticks=tuple(ticks)
        key=(tuple((t,queries[t].key) for t in ticks),analysis_roll,_native_actor_anchor_cache_identity(anchor))
        if key not in self.paths:
            frames=[self.candidates(queries[t],t,analysis_roll=analysis_roll,association_ready=True) for t in ticks]
            units={t:self.native_signature(queries[t],t,analysis_roll=analysis_roll)[:4] for t in ticks}
            self.paths[key]=connected_source_actor_views(frames,ticks=ticks,native_units=units,
                anchor=anchor,maximum_frames=512,work_port=self.sensors.session.budget.request_work_phase)
        return self.paths[key]

    def reproject_selected_actor_path(self,queries,selected,*,analysis_roll=None):
        """Keep the exact selected occurrences under a newly authored camera.

        This is not a second actor search. An unmatched tick stays unmatched,
        and an unavailable/changed raw parent cannot be replaced by the nearest
        face. Native phase must already have been selected before this call.
        Connected-path uncertainty is retained, not erased by reprojection.
        """
        if (not isinstance(selected,dict) or not selected or len(selected)>512
                or not set(selected)<=set(queries)):
            raise ValueError('selected actor projection needs one bounded explicit tick path')
        result={}
        association_fields=('source_actor_path_identity','reacquisition_ambiguous',
            'selected_actor_association_state','original_role',
            'original_attention_hypothesis_is_focal_duty','original_actor_inventory_seed',
            'original_attention_scope')
        for tick,prior in selected.items():
            self.sensors.session.budget.check()
            if prior is None:
                result[tick]=None
                continue
            raw=prior['row']['native_observation'];query=queries[tick]
            source=self.domain.request_index(query.eligible_source_index)
            if (prior['row']['output_tick']!=tick or raw['request_source_index']!=source
                    or raw['native_frame_index']!=query.sample_at(tick)[2]
                    or raw['source_clock_identity']!=query.source_clock_identity):
                raise ValueError('selected camera changed the already selected actor native schedule')
            head=prior['descriptor'].frame_local_head_id
            candidates=[v for v in self.candidates(query,tick,analysis_roll=analysis_roll)
                if v['descriptor'].frame_local_head_id==head
                and family.digest(v['row']['native_observation'])==family.digest(raw)]
            if len(candidates)!=1:
                raise ValueError('selected camera lost the exact observed actor raw parent')
            result[tick]=dict(candidates[0],**{k:prior[k] for k in association_fields if k in prior})
        return result

    def freeze_original_actors(self,queries,ticks,*,original_program_identity):
        from .group_review_reference_actor import freeze_original_review_actor_inventory
        ticks=tuple(sorted(set(ticks)))
        frames={t:self.candidates(queries[t],t,association_ready=True) for t in ticks}
        units={t:self.native_signature(queries[t],t)[:4] for t in ticks}
        roles={t:'UNSPECIFIED' if queries[t].role is None else queries[t].role.value for t in ticks}
        return freeze_original_review_actor_inventory(frames,native_units=units,
            original_program_identity=original_program_identity,
            original_attention_scopes={t:queries[t].original_unit_id for t in ticks},
            original_roles_by_tick=roles)


def _balanced_native_motion_tuples(sources,cut_ticks,native_phases,maximum_tuples,*,check=None):
    """A declared finite subset, NOT an endpoint-cost ranking or full optimum."""
    from itertools import product
    if (not isinstance(sources,tuple) or not 2<=len(sources)<=12 or len(set(sources))!=len(sources)
            or any(type(s) is not int or s<0 for s in sources)
            or not isinstance(cut_ticks,tuple) or not 1<=len(cut_ticks)<=3
            or cut_ticks!=tuple(sorted(set(cut_ticks)))
            or any(type(t) is not int or t<0 for t in cut_ticks)
            or not isinstance(native_phases,tuple) or not 1<=len(native_phases)<=5
            or len(set(native_phases))!=len(native_phases)
            or any(type(p) is not int or abs(p)>2 for p in native_phases)
            or type(maximum_tuples) is not int or not 1<=maximum_tuples<=64):
        raise ValueError('native motion frontier needs finite externally declared strata')
    remaining=[];ties={}
    for i,row in enumerate(product(sources,sources,cut_ticks,native_phases,native_phases,(0,1),(0,1))):
        if check is not None and i%128==0:check()
        if row[0]!=row[1]:
            remaining.append(row);ties[row]=family.digest(row)
    counts={};result=[]
    # Pure tie-break hashes are immutable. Do not serialize/hash the same
    # theoretical tuple again during every balanced admission round.
    def strata(row):
        a,b,cut,pa,pb,ha,hb=row
        return (('left_source',a),('right_source',b),('cut',cut),
            ('left_phase',pa),('right_phase',pb),('left_actor',ha),('right_actor',hb))
    def priority(row):
        keys=strata(row)
        return (max(counts.get(k,0) for k in keys[:2]),
            sum(counts.get(k,0) for k in keys[:2]),
            sum(counts.get(k,0) for k in keys[2:]),
            abs(row[3])+abs(row[4]),row[5]+row[6],ties[row])
    for _ in range(min(maximum_tuples,len(remaining))):
        # Cancellation/deadline remain responsive inside the bounded linear
        # pass as well as enumeration. This is still a finite rank-0/1 subset.
        chosen=None;best=None
        for i,row in enumerate(remaining):
            if check is not None and i%128==0:check()
            score=priority(row)
            if best is None or score<best:chosen=row;best=score
        remaining.remove(chosen);result.append(chosen)
        for key in strata(chosen):counts[key]=counts.get(key,0)+1
    return tuple(result)


def realize_normal_native_motion_frontier(*,context,sensors,first_output_frame,end_output_frame,
        cut_ticks,native_phases=(-1,0,1),maximum_tuples=64,maximum_camera_candidates=2,
        maximum_new_observations=0,maximum_new_raw_observations=0,
        protected_output_intervals=(),body_receipt_bundles=(),source_families=None):
    """Own the real source/actor/native-phase/cut/camera frontier in ONE pass.

    Native demand comes from each actual materialized topology, not an old
    whole-song discovery sampler. Missing facts are completed at most once.
    This is the normal producer's selection stage: it returns exact evaluated
    whole plans and actor references. It does not substitute for full fresh
    bundle metadata, final replay, encoding or expert-quality admission.
    """
    from .group_scene_owned_program import freeze_owned_scene_program
    from .group_scene_epoch_program import compile_common_clock_camera_queries
    from .group_native_motion_programme import NativeMotionSide,realize_native_motion_programme
    from .group_native_motion_camera_producer import produce_two_sided_native_cameras
    from .request_work_lease import current_request_work_phase
    from .group_native_motion_completion_schedule import NativeMotionCompletionPackage
    from time import monotonic
    started=monotonic();stage_seconds={};stage_start=started
    domain=sensors.session.domain;plan=context.incumbent.render_plan
    port=sensors.session.budget.request_work_phase
    if getattr(sensors.session,'_native_motion_completion_consumed',False):
        raise ValueError('native motion completion already consumed this session')
    if (port is None or current_request_work_phase() is not port or sensors.session.closed
            or any(type(t) is not int for t in (first_output_frame,end_output_frame))
            or not 3<=first_output_frame<end_output_frame<=plan.output_frame_count-3
            or not isinstance(cut_ticks,tuple) or any(not first_output_frame+4<=t<=end_output_frame-4
                or t-first_output_frame>160 or end_output_frame-t>160 for t in cut_ticks)):
        raise ValueError('normal native frontier requires its active bounded request-owned interval')
    original=freeze_owned_scene_program(context,domain)
    original_shots=tuple(s.to_dict() for s in visible_group_shots(plan,domain))
    halo=tuple(range(first_output_frame-3,end_output_frame+3))
    if type(maximum_tuples) is not int or not 1<=maximum_tuples<=64:
        raise ValueError('native motion frontier needs its declared tuple capacity')
    effective_tuples=min(maximum_tuples,(16384-len(halo))//(end_output_frame-first_output_frame))
    proposed=_balanced_native_motion_tuples(domain.eligible_indices,cut_ticks,native_phases,effective_tuples,
        check=sensors.session.budget.check)
    def queries(candidate):
        compiled=compile_common_clock_camera_queries(context=context,domain=domain,original_program=original,
            candidate_plan=candidate,output_intervals=((halo[0],halo[-1]+1),))
        return {t:q for q in compiled.queries for t in range(q.first_output_frame,q.end_output_frame)}
    base=queries(plan)
    families=({s:'physical-file-'+family.digest(sensors.session._assert_source_identity(s))
        for s in domain.eligible_indices} if source_families is None else source_families)
    if (not isinstance(families,dict) or set(families)!=set(domain.eligible_indices)
            or any(not isinstance(v,str) or not v for v in families.values())):
        raise ValueError('native frontier source families differ from its frozen source domain')
    demands=[]
    def request(q,t):
        source=domain.request_index(q.eligible_source_index)
        opportunity=('moving_closeup' if q.role in (GroupOutputRole.FOCAL_HEAD,GroupOutputRole.FOCAL_HEAD_ARMS)
            else 'other')
        return NativeMotionDemand(source,q.sample_at(t)[2],t,families[source],opportunity,
            require_appearance=True,require_member_features=sensors.member_gallery is not None)
    demands.extend(request(base[t],t) for t in halo)
    declarations=[];trials=[]
    # Native topology, duty/protection and alias inventories precede all donor
    # actor/camera costs. Invalid physical shots are recorded, not retried with
    # a widened interval or a nearby native raster.
    for index,spec in enumerate(proposed):
        sensors.session.budget.check();port.charge('states')
        a,b,cut,pa,pb,ha,hb=spec
        sides=(NativeMotionSide(first_output_frame,cut,a,pa),NativeMotionSide(cut,end_output_frame,b,pb))
        try:
            numeric=realize_native_motion_programme(original=plan,domain=domain,sides=sides,
                protected_output_intervals=protected_output_intervals)
        except ValueError as exc:
            if str(exc) not in ('joint state crosses an observed actual native camera cut',
                    'one motion side crosses an actual native broadcaster cut',
                    'phase base grid differs from the complete actual renderer clock',
                    'native motion programme intersects protected original pixels'):
                raise
            trials.append(dict(tuple_index=index,spec=spec,state='INADMISSIBLE_NATIVE_TOPOLOGY',reason=str(exc)))
            continue
        qs=queries(numeric.render_plan)
        demands.extend(request(qs[t],t) for t in range(first_output_frame,end_output_frame))
        declarations.append((index,spec,sides,qs))
    if len(demands)>16384:
        raise ValueError('declared native motion alias inventory exceeds its request cap')
    stage_seconds['numeric_declaration']=monotonic()-stage_start;stage_start=monotonic()
    cached=sensors.motion_inventory(tuple(demands))
    stage_seconds['cache_census']=monotonic()-stage_start;stage_start=monotonic()
    # The original context/KEEP programme is shared by every alternative. Its
    # exact entry/return flanks come first, followed by all original actor
    # inputs. Donor packages are frozen numeric topologies, not match winners.
    original_order=tuple(dict.fromkeys((first_output_frame-1,end_output_frame,
        *range(first_output_frame-3,first_output_frame),
        *range(end_output_frame,end_output_frame+3),*halo)))
    required_keys=tuple(dict.fromkeys(request(base[t],t).native_key for t in original_order))
    packages=tuple(NativeMotionCompletionPackage(f'joint-tuple-{index:03d}',
        tuple(dict.fromkeys(request(qs[t],t).native_key
            for t in range(first_output_frame,end_output_frame))))
        for index,spec,sides,qs in declarations)
    completed=sensors.complete_motion_inventory_once(cached,maximum_new_observations=maximum_new_observations,
        maximum_new_raw_observations=maximum_new_raw_observations,
        required_native_keys=required_keys,completion_packages=packages,
        critical_required_native_keys=tuple(dict.fromkeys(request(base[t],t).native_key
            for t in (first_output_frame-1,end_output_frame))),member_features_optional=True)
    stage_seconds['one_completion']=monotonic()-stage_start;stage_start=monotonic()
    views=NativeMotionViews(sensors,completed,body_receipt_bundles=body_receipt_bundles)
    original_actors,original_receipt=views.freeze_original_actors(base,halo,
        original_program_identity=original.program_sha256)
    stage_seconds['original_actor_freeze']=monotonic()-stage_start;stage_start=monotonic()
    # Freeze each source-local actor hypothesis before fitting a donor camera.
    # Equivalent physical paths reuse the same DP, then exactly reproject into
    # each new query scope. A camera cannot trigger a more favourable actor.
    path_cache={};candidates=[];actor_seconds=0.;camera_seconds=0.
    from .group_review_projected_action import project_observed_review_face
    fixed_context_faces={t:project_observed_review_face(original_actors[t],base[t],t)
        for t in (first_output_frame-1,end_output_frame)}
    missing_fixed_landmarks=tuple(t for t,f in fixed_context_faces.items()
        if f is None or f['landmarks_xy'] is None)
    for index,spec,sides,qs in declarations:
        sensors.session.budget.check()
        trial_start=monotonic()
        paths=[];receipts=[]
        for side,actor_rank in zip(sides,spec[-2:],strict=True):
            ticks=tuple(range(side.first_output_frame,side.end_output_frame));middle=(ticks[0]+ticks[-1])/2
            seed_tick=next((t for t in sorted(ticks,key=lambda t:(abs(t-middle),t))
                if views.candidates(qs[t],t,association_ready=True)),None)
            anchors=(() if seed_tick is None else tuple(sorted(views.candidates(qs[seed_tick],seed_tick,
                association_ready=True),key=lambda v:(family.digest(v['row']['native_observation']),
                    v['descriptor'].frame_local_head_id))))
            anchor=None if actor_rank>=len(anchors) else anchors[actor_rank]
            key=(tuple((t,views.native_signature(qs[t],t)) for t in ticks),_native_actor_anchor_cache_identity(anchor))
            if key not in path_cache:path_cache[key]=views.connected_actor_views(qs,ticks,anchor)
            path,receipt=path_cache[key]
            paths.append(views.reproject_selected_actor_path(qs,path));receipts.append(receipt)
        actor_seconds+=monotonic()-trial_start
        endpoint_inputs=tuple(views.observation_status(qs[t],t,selected=path[t])
            for side,path in zip(sides,paths,strict=True)
            for t in (side.first_output_frame,side.end_output_frame-1))
        if any(r.get('state')!='CONNECTED_SOURCE_LOCAL_HYPOTHESIS' for r in receipts):
            trials.append(dict(tuple_index=index,spec=spec,state='ACTOR_PATH_UNKNOWN',paths=receipts,
                endpoint_inputs=endpoint_inputs));continue
        missing_actors=tuple(t for side,path in zip(sides,paths,strict=True)
            for t in (side.first_output_frame,side.end_output_frame-1) if path[t] is None)
        missing_landmarks=tuple(t for side,path in zip(sides,paths,strict=True)
            for t in (side.first_output_frame,side.end_output_frame-1)
            if (f:=project_observed_review_face(path[t],qs[t],t)) is None or f['landmarks_xy'] is None)
        if missing_fixed_landmarks or missing_landmarks:
            # There is no camera score to compute without its exact fitting
            # inputs. Do not rebuild/measure an identical full KEEP programme
            # for every known-missing tuple; these are UNKNOWN inputs, not a
            # geometry failure or a source-wide exclusion.
            trials.append(dict(tuple_index=index,spec=spec,state='UNOBSERVED_ENDPOINT_ANATOMY',
                endpoint_inputs=endpoint_inputs,receipt=dict(
                    missing_actor_ticks=missing_actors,
                    missing_landmark_ticks=(*missing_landmarks,*missing_fixed_landmarks),
                    measured_keep_incidents=None,camera_fitting_started=False,
                    missing_values_used_as_perfect_keep_score=False,quality_approved=False)))
            continue
        trial_start=monotonic()
        realized,receipt=produce_two_sided_native_cameras(context=context,domain=domain,views=views,
            original_program=original,original_actors=original_actors,original_actor_receipt=original_receipt,
            sides=sides,side_paths=tuple(paths),side_path_receipts=tuple(receipts),
            protected_output_intervals=protected_output_intervals,maximum_camera_candidates=maximum_camera_candidates)
        camera_seconds+=monotonic()-trial_start
        trials.append(dict(tuple_index=index,spec=spec,state=receipt['state'],receipt=receipt,
            endpoint_inputs=endpoint_inputs,
            camera_candidates=[dict(selected_plan_identity=family.digest(c.render_plan.to_json()),
                state=c.evidence['state'],score=c.evidence['score'],measured_keep_score=c.evidence['measured_keep_score'],
                known_incident_failures=c.evidence['incident_comparison']['failures'],
                known_hold_observation_losses=c.evidence['known_hold_observation_losses'],
                known_hold_motion_failures=c.evidence['known_hold_motion_failures'],
                local_member_hint_conflict_ticks=c.evidence['local_member_hint_conflict_ticks'])
                for c in realized]))
        candidates.extend(realized)
    eligible=tuple(sorted((c for c in candidates if c.evidence['state']=='REALIZED_FOR_REVIEW'),
        key=lambda c:(c.evidence['score']-c.evidence['measured_keep_score'],family.digest(c.render_plan.to_json()))))
    from collections import Counter
    stage_seconds['actor_paths_and_reprojection']=actor_seconds
    stage_seconds['joint_camera_and_incident_measurement']=camera_seconds
    report=dict(contract='normal_joint_native_motion_frontier_selection_stage_v1',
        original_program_identity=original.program_sha256,original_actor_inventory=original_receipt,
        original_whole_song_shot_inventory=original_shots,
        frozen_source_families=families,distinct_frozen_source_families=len(set(families.values())),
        original_protected_output_intervals=protected_output_intervals,
        proposed_joint_tuples=len(proposed),physical_topologies=len(declarations),
        requested_maximum_joint_tuples=maximum_tuples,effective_alias_budget_tuple_cap=effective_tuples,
        native_alias_requests=len(demands),unique_native_requests=len(views.supplies),
        native_completion_status=completed.status,native_completion_consumed=completed.completion_consumed,
        native_completion_schedule=__import__('json').loads(completed.completion_schedule_json),
        native_supply_states=dict(Counter(s.state for s in completed.supplies)),
        unique_native_supply_states=dict(Counter(s.state for s in views.supplies.values())),
        original_context_inputs=tuple(views.observation_status(base[t],t,selected=original_actors[t])
            for t in (first_output_frame-1,end_output_frame)),
        actual_actor_dp_paths=len(path_cache),realized_camera_candidates=len(candidates),
        nonregressing_camera_candidates=len(eligible),trials=sorted(trials,key=lambda r:r['tuple_index']),
        demand_basis='exact_new_topology_authorized_native_sampler_not_old_discovery_clock',
        full_joint_optimum_claimed=False,full_normal_metadata_constructed=False,
        stage_seconds=stage_seconds,selection_elapsed_seconds=monotonic()-started,
        video_rendered=False,quality_approved=False)
    return eligible,report,views


class NativeFamilySensors:
    """One live raw-native cache/model owner; no camera-dependent cohort choice."""
    def __init__(self,session,assets,*,frozen_declarations=None,member_gallery=None):
        self.session,self.assets = session,assets
        register_joint_native_models(session,**assets)
        register_native_head_orientation_models(session,assets['head_model_path'])
        self.declarations = {}
        self.raw_rows = {}
        self.appearance_rows = {}
        self.member_feature_rows = {}
        self.recorded_member_feature_cache = None
        self.member_gallery = member_gallery
        if member_gallery is not None:
            from .ive_local_arcface_gallery import LocalIVEArcFaceGallery
            if not isinstance(member_gallery,LocalIVEArcFaceGallery):
                raise ValueError('native member hints need the explicitly supplied local gallery')
        self.recorded_motion_index = None
        self.recorded_motion_bindings = {}
        if frozen_declarations is not None:
            for row in frozen_declarations:
                self.declarations[tuple(row['span_key'])] = row

    @staticmethod
    def span_key(state):
        return (state.eligible_source_index,state.first_output_frame,state.end_output_frame,
            state.native_frame_indices[0],state.native_frame_indices[-1],state.role.value)

    def raw(self,state,tick,roll):
        native = state.native_frame_indices[tick-state.first_output_frame]
        return self.raw_native(state,native,roll)

    def raw_native(self,state,native,roll):
        """Read one exact native fact; editorial boundaries are not cache keys."""
        source = self.session.domain.request_index(state.eligible_source_index)
        if (type(native) is not int or not state.native_frame_indices[0]<=native<=state.native_frame_indices[-1]):
            raise ValueError('native observation escaped the complete source-camera state')
        return self.native_at(source, native, roll)

    def native_at(self,source,native,roll=None,*,cached_only=False):
        """Use the existing raw_rows cache; never choose a dominant performer.

        cached_only does not decode a missing frame or invoke any detector.
        Actual cached rows still require their owning live session and exact
        source/clock/model/adapter bindings. A saved focal box is not a raw row.
        """
        from .group_native_head_orientation import native_head_analysis_spec
        if type(cached_only) is not bool:
            raise ValueError('native cache mode must be explicit')
        self.session.budget.check()
        if self.session.closed:
            raise ValueError('native motion cannot reuse a closed session')
        local = self.session.domain.eligible_index(source)
        clock = self.session.domain.render_frame_clocks[local]
        if type(native) is not int or not 0 <= native < clock.frame_count:
            raise ValueError('native motion request is outside its source clock')
        head_model,_ = native_head_analysis_spec(roll)
        required_models = ('nanodet_people','rtmpose_body7',head_model)
        key = source,native,roll
        row = self.raw_rows.get(key)
        if row is None and cached_only:
            return None
        if row is not None:
            identity = self.session._assert_source_identity(source)
            expected_clock = family.digest(clock.to_dict())
            if (row.request_source_index != source or row.native_frame_index != native
                    or row.source_clock_identity != expected_clock
                    or row.source_file_identity != identity):
                raise ValueError('cached native fact changed source/clock/file/frame')
            if (tuple(name for name,_ in row.model_content_sha256) != required_models
                    or tuple(name for name,_ in row.model_adapter_fingerprints) != required_models):
                raise ValueError('cached native fact changed the declared model/analysis variant')
            for name, model_sha in row.model_content_sha256:
                self.session._assert_model_assets(name)
                if self.session.models[name][2] != model_sha:
                    raise ValueError('cached native fact changed model content')
            if tuple((name,self.session.model_adapters[name]) for name,_ in
                    row.model_adapter_fingerprints) != row.model_adapter_fingerprints:
                raise ValueError('cached native fact changed model adapter')
            return row
        row = self.session.budget.normalized_native_views.measure(
            label='native_focal_observation_view',
            input_binding=dict(request_source_index=source,native_frame_index=native,
                source_clock_identity=family.digest(clock.to_dict()),head_analysis_roll=roll),
            producer=lambda:observe_native_group(self.session,request_source_index=source,
                native_frame_index=native,head_analysis_roll=roll))
        self.raw_rows[key]=row
        return row

    def appearance_at(self,raw,*,cached_only=False):
        """An exact raw-parent SFace result; absent vectors do not erase heads."""
        from .group_review_actor_appearance import observe_recorded_head_appearances
        from .group_review_recorded_index import validate_recorded_appearance
        from .native_face_descriptor import native_descriptor_space,SFACE_MODEL_NAME
        payload=raw.to_dict();key=family.digest(payload)
        value=self.appearance_rows.get(key)
        if value is None and cached_only:return None
        space=native_descriptor_space(self.session,SFACE_MODEL_NAME)
        if value is None:
            value=observe_recorded_head_appearances(self.session,payload)
            self.appearance_rows[key]=value
        validate_recorded_appearance(payload,value,expected_space=space)
        return value

    def member_features_at(self,supply,*,cached_only=False):
        from .ive_local_arcface_gallery import (observe_recorded_member_features,
            validate_recorded_member_features)
        from .native_face_descriptor import native_descriptor_space,ARCFACE_MODEL_NAME
        import json
        if self.member_gallery is None:
            raise ValueError('member feature request has no explicit local photo gallery')
        payload=json.loads(json.dumps(supply.observation.to_dict(),sort_keys=True,allow_nan=False))
        key=family.digest(payload),self.member_gallery.identity
        value=self.member_feature_rows.get(key)
        if value is None and self.recorded_member_feature_cache is not None:
            value=self.recorded_member_feature_cache.get(payload)
            if value is not None:self.member_feature_rows[key]=value
        if value is None and cached_only:return None
        space=native_descriptor_space(self.session,ARCFACE_MODEL_NAME)
        if value is None:
            value=observe_recorded_member_features(self.session,supply,self.member_gallery)
            self.member_feature_rows[key]=value
            if self.recorded_member_feature_cache is not None:
                self.recorded_member_feature_cache.add(payload,value)
        validate_recorded_member_features(payload,value,gallery=self.member_gallery,expected_space=space)
        return value

    def bind_recorded_member_feature_cache(self,cache):
        from .ive_local_arcface_gallery import LocalIVEArcFaceFeatureCache
        from .native_face_descriptor import native_descriptor_space,ARCFACE_MODEL_NAME
        if (not isinstance(cache,LocalIVEArcFaceFeatureCache) or self.member_gallery is None
                or self.recorded_member_feature_cache is not None
                or getattr(self.session,'_native_motion_completion_consumed',False)
                or cache.gallery.identity!=self.member_gallery.identity
                or cache.space!=native_descriptor_space(self.session,ARCFACE_MODEL_NAME)):
            raise ValueError('recorded member features need one exact pre-completion input binding')
        self.recorded_member_feature_cache=cache

    def bind_recorded_motion_index(self,index,*,source_identities,source_content_sha256,
            additional_source_identities=None):
        """Bind the request's explicit archive index without rewriting raw parents.

        No directory search, extra cache DB or inference occurs. Device-only
        remounts require the independently retained complete content hash.
        This is reusable observation input, not fresh pixel or quality proof.
        """
        from .group_review_recorded_index import RecordedGroupReviewIndex,verify_recorded_source_file
        import json
        if (getattr(self,'recorded_motion_index',None) is not None
                or not isinstance(index,RecordedGroupReviewIndex)
                or not isinstance(source_identities,dict) or not source_identities
                or not set(source_identities)<=set(self.session.domain.eligible_indices)
                or not isinstance(source_content_sha256,dict)):
            raise ValueError('recorded motion input needs one explicit request-owned source manifest')
        self.session.budget.check()
        if self.session.closed:raise ValueError('recorded motion input needs a live request session')
        additions={} if additional_source_identities is None else additional_source_identities
        if not isinstance(additions,dict) or not set(additions)<=set(source_identities):
            raise ValueError('additional native input identities escaped the frozen source manifest')
        bindings={};additional_bindings={}
        for source,identity in sorted(source_identities.items()):
            receipt=verify_recorded_source_file(self.session.domain.request_sources[source],identity,
                content_sha256=source_content_sha256.get(source),work_port=self.session.budget.request_work_phase)
            if tuple(receipt['current_file_identity'])!=self.session._assert_source_identity(source):
                raise ValueError('recorded source changed while binding it to the live request')
            bindings[source]=json.dumps(receipt,sort_keys=True,separators=(',',':'),allow_nan=False)
            if source in additions and tuple(additions[source])!=tuple(identity):
                added=verify_recorded_source_file(self.session.domain.request_sources[source],additions[source],
                    content_sha256=source_content_sha256.get(source),work_port=self.session.budget.request_work_phase)
                if tuple(added['current_file_identity'])!=self.session._assert_source_identity(source):
                    raise ValueError('additional native input source changed during request binding')
                additional_bindings[source]=json.dumps(added,sort_keys=True,separators=(',',':'),allow_nan=False)
        self.recorded_motion_bindings=bindings
        self.recorded_motion_additional_bindings=additional_bindings
        self.recorded_motion_index=index

    def _recorded_motion_supply(self,demand):
        from .group_review_recorded_index import native_analysis_variant,validate_recorded_appearance
        from .group_native_head_orientation import native_head_analysis_spec
        from .group_joint_person_ownership import ownership_policy_fingerprint
        from .group_scene_focal_pose_motion import restore_native_row
        from .native_face_descriptor import native_descriptor_space,SFACE_MODEL_NAME
        import json
        index=getattr(self,'recorded_motion_index',None)
        binding=getattr(self,'recorded_motion_bindings',{}).get(demand.source_index)
        if index is None or binding is None:return NativeMotionSupply(demand,None)
        receipt=json.loads(binding)
        if self.session.closed:raise ValueError('recorded motion cannot reuse a closed session')
        self.session.budget.check()
        if tuple(receipt['current_file_identity'])!=self.session._assert_source_identity(demand.source_index):
            raise ValueError('recorded motion source changed after request binding')
        names=('nanodet_people','rtmpose_body7',native_head_analysis_spec(demand.analysis_roll)[0])
        for name in names:self.session._assert_model_assets(name)
        variant=native_analysis_variant(dict(
            model_content_sha256=tuple((name,self.session.models[name][2]) for name in names),
            model_adapter_fingerprints=tuple((name,self.session.model_adapters[name]) for name in names),
            person_ownership_policy_sha256=ownership_policy_fingerprint()))
        clock=self.session.domain.clocks_by_request[demand.source_index]
        found=index.lookup(source=demand.source_index,native=demand.native_frame_index,
            clock=family.digest(clock.to_dict()),identity=receipt['recorded_file_identity'],
            analysis_variant_sha256=variant)
        # A later one-pass completion has live-device parents, while the
        # older input archive may have an earlier mount identity. Reuse only
        # an explicitly bound additional manifest for a genuinely ABSENT raw
        # address. Never replace measured failures, conflicting rows or an
        # existing bad actor result with a more favourable capture.
        extra=getattr(self,'recorded_motion_additional_bindings',{}).get(demand.source_index)
        if found['raw'] is None and found['state']=='MISSING' and extra is not None:
            alternative=json.loads(extra)
            if tuple(alternative['current_file_identity'])!=self.session._assert_source_identity(demand.source_index):
                raise ValueError('additional native input source changed after request binding')
            added=index.lookup(source=demand.source_index,native=demand.native_frame_index,
                clock=family.digest(clock.to_dict()),identity=alternative['recorded_file_identity'],
                analysis_variant_sha256=variant)
            found=added;receipt=alternative;binding=extra
        raw=None if found['raw'] is None else restore_native_row(found['raw'])
        appearance=None
        if raw is not None and demand.require_appearance:
            appearance=self.appearance_rows.get(family.digest(raw.to_dict()),found['appearance'])
            if appearance is not None:
                space=native_descriptor_space(self.session,SFACE_MODEL_NAME)
                actual=validate_recorded_appearance(found['raw'],appearance)
                if actual!=space:
                    # Keep the actual raw heads, but never mix descriptor spaces.
                    appearance=None;found=dict(found,state='APPEARANCE_SPACE_MISMATCH')
        supply=NativeMotionSupply(demand,raw,appearance,found['state'],binding)
        if demand.require_member_features and raw is not None and raw.native_pixel_sha256 is not None:
            supply=replace(supply,member_features=self.member_features_at(supply,cached_only=True))
        return supply

    def _joint_motion_supply(self,demand,live_raw):
        """Census live and archived facts; neither may hide adverse evidence.

        Only the verified device-remount file address is normalized for the
        comparison. The selected archive and every feature keep their exact
        original parents. Distinct declared model variants are not conflicts.
        """
        import json
        recorded=self._recorded_motion_supply(demand)
        if live_raw is None:
            return recorded
        if recorded.state in ('CONFLICTING_NATIVE_RECORDS','CONFLICTING_FEATURE_RECORDS'):
            return replace(recorded,appearance=None,member_features=None)
        live=NativeMotionSupply(demand,live_raw,
            self.appearance_at(live_raw,cached_only=True) if demand.require_appearance
                and live_raw.native_pixel_sha256 is not None else None,
            recorded_input_state=recorded.recorded_input_state)
        if demand.require_member_features and live_raw.native_pixel_sha256 is not None:
            live=replace(live,member_features=self.member_features_at(live,cached_only=True))
        if recorded.observation is None:
            return live
        before=json.loads(json.dumps(recorded.observation.to_dict(),sort_keys=True,allow_nan=False))
        after=json.loads(json.dumps(live_raw.to_dict(),sort_keys=True,allow_nan=False))
        same_parent=before['source_file_identity']==after['source_file_identity']
        if not same_parent:
            receipt=(None if recorded.source_equivalence_json is None else
                json.loads(recorded.source_equivalence_json))
            if (receipt is None or receipt.get('state')!='CONTENT_VERIFIED_DEVICE_REMOUNT'
                    or receipt.get('recorded_file_identity')!=before['source_file_identity']
                    or receipt.get('current_file_identity')!=after['source_file_identity']
                    or before['source_file_identity'][1:]!=after['source_file_identity'][1:]):
                return NativeMotionSupply(demand,None,recorded_input_state='CONFLICTING_NATIVE_RECORDS')
            after=dict(after,source_file_identity=before['source_file_identity'])
        if before!=after:
            return NativeMotionSupply(demand,None,recorded_input_state='CONFLICTING_NATIVE_RECORDS')
        # A feature measured on a different raw parent is never copied or
        # relabelled just because the underlying source file is equivalent.
        if not same_parent:
            return recorded
        for field in ('appearance','member_features'):
            a,b=getattr(recorded,field),getattr(live,field)
            if a is not None and b is not None and family.digest(a)!=family.digest(b):
                return replace(recorded,appearance=None,member_features=None,
                    recorded_input_state='CONFLICTING_FEATURE_RECORDS')
        return replace(recorded,
            appearance=recorded.appearance if recorded.appearance is not None else live.appearance,
            member_features=recorded.member_features if recorded.member_features is not None else live.member_features)

    def _complete_motion_appearance(self,supply):
        """Complete an archived raw parent's appearance on verified live pixels."""
        raw=supply.observation
        if supply.source_equivalence_json is None:return self.appearance_at(raw)
        import hashlib,json
        from .face_identity import _validate_normalized_face
        from .native_face_descriptor import native_descriptor_space,SFACE_MODEL_NAME
        from .group_review_actor_appearance import CONTRACT
        from .group_review_recorded_index import validate_recorded_appearance
        receipt=json.loads(supply.source_equivalence_json)
        # The archive's canonical JSON containers remain the parent. Restore
        # returns typed tuples; do not let that representation change make a
        # completed appearance fail its next exact-parent cache lookup.
        payload=json.loads(json.dumps(raw.to_dict(),sort_keys=True,allow_nan=False))
        source=raw.request_source_index;native=raw.native_frame_index
        if (self.session.closed or tuple(receipt['recorded_file_identity'])!=raw.source_file_identity
                or tuple(receipt['current_file_identity'])!=self.session._assert_source_identity(source)):
            raise ValueError('archived appearance lost its live source equivalence')
        if raw.source_clock_identity!=family.digest(self.session.domain.clocks_by_request[source].to_dict()):
            raise ValueError('archived appearance changed its native source clock')
        pixels=self.session.native_pixels(source,native)
        if pixels is None or hashlib.sha256(memoryview(pixels)).hexdigest()!=raw.native_pixel_sha256:
            raise ValueError('archived appearance exact native pixels changed')
        space=native_descriptor_space(self.session,SFACE_MODEL_NAME);rows=[]
        for head in raw.heads:
            try:_validate_normalized_face(head.box_xywh,head.landmarks_xy)
            except ValueError:
                rows.append(dict(observation_id=head.observation_id,embedding=None,
                    reason='recorded_geometry_outside_native_descriptor_input'));continue
            vector=self.session.observe_native(source,native,model=SFACE_MODEL_NAME,
                arguments=dict(bbox_xywh=head.box_xywh,landmarks_xy=head.landmarks_xy))
            rows.append(dict(observation_id=head.observation_id,embedding=vector,
                reason='actual_native_sface' if vector is not None else 'native_embedding_unavailable'))
        if tuple(receipt['current_file_identity'])!=self.session._assert_source_identity(source):
            raise ValueError('archived appearance source changed during completion')
        self.session._assert_model_assets(SFACE_MODEL_NAME)
        result=dict(contract=CONTRACT,**{k:payload[k] for k in ('request_source_index','native_frame_index',
            'source_clock_identity','source_file_identity','native_pixel_sha256')},
            recorded_native_observation_sha256=family.digest(payload),descriptor_space=space.to_dict(),heads=rows,
            actual_native_raster_redecoded=True,detector_input_is_recorded=True,head_detection_reexecuted=False,
            named_member_identity_certified=False,persistent_actor_identity_certified=False,
            motion_certified=False,candidate_admitted=False,quality_approved=False,
            verified_source_equivalence=receipt,archived_observation_rewritten=False)
        validate_recorded_appearance(payload,result,expected_space=space)
        self.appearance_rows[family.digest(payload)]=result
        return result

    def motion_inventory(self,demands):
        """Inspect every declared time/source opportunity without new NN work."""
        demands = tuple(demands)
        if (not demands or len(demands)>16384
                or any(not isinstance(d,NativeMotionDemand) for d in demands)):
            raise ValueError('native motion inventory needs bounded typed demands')
        family_by_source={}
        for d in demands:
            if d.require_member_features and self.member_gallery is None:
                raise ValueError('member feature request has no explicit local photo gallery')
            previous=family_by_source.setdefault(d.source_index,d.source_family)
            if previous!=d.source_family:
                raise ValueError('native demand aliases changed a frozen source family')
        # Inspect physical facts once, not once per camera/cut/output alias.
        # Union modalities BEFORE inspection: a raw-only first alias must not
        # hide a later appearance/member-feature request at the same raster.
        union={}
        for d in demands:
            prior=union.get(d.native_key)
            union[d.native_key]=(d if prior is None else replace(prior,
                require_appearance=prior.require_appearance or d.require_appearance,
                require_member_features=prior.require_member_features or d.require_member_features))
        physical={}
        for key,d in union.items():
            raw=self.native_at(d.source_index,d.native_frame_index,d.analysis_roll,cached_only=True)
            physical[key]=self._joint_motion_supply(d,raw)
        supplies=tuple(replace(physical[d.native_key],demand=d) for d in demands)
        return NativeMotionInventory(supplies,(),False,'CACHE_ONLY')

    def complete_motion_inventory_once(self,inventory,*,maximum_new_observations,
            maximum_new_raw_observations=None,required_native_keys=(),completion_packages=(),
            critical_required_native_keys=(),member_features_optional=False):
        """One request-owned, time/source/opportunity-balanced missing pass.

        Existing successes, observed failures and absent facts remain distinct.
        The entire inventory is inspected BEFORE admitting any missing row.
        Camera count, actor alternatives and phase count do not multiply NN work.
        The caller keeps this one session through realization and selection.
        """
        from .group_scene_native_storage import MAX_NATIVE_JSON_BYTES
        from .group_joint_primitive_journal import primitive_reservation_bytes
        from .request_work_lease import current_request_work_phase
        from .group_native_motion_completion_schedule import (validate_completion_packages,
            coherent_completion_schedule,NativeMotionCompletionCost,COMPLETION_RESOURCES)
        port = self.session.budget.request_work_phase
        if port is None or current_request_work_phase() is not port:
            raise ValueError('normal motion completion requires the active whole-request work lease')
        if (not isinstance(inventory,NativeMotionInventory) or inventory.completion_consumed
                or type(member_features_optional) is not bool
                or type(maximum_new_observations) is not int
                or not 0<=maximum_new_observations<=1200
                or maximum_new_raw_observations is not None and (
                    type(maximum_new_raw_observations) is not int
                    or not 0<=maximum_new_raw_observations<=maximum_new_observations)):
            raise ValueError('one native completion needs the frozen cache-only inventory')
        if getattr(self.session,'_native_motion_completion_consumed',False):
            raise ValueError('native motion completion already consumed this session')
        validate_completion_packages(inventory,required_native_keys,completion_packages,critical_required_native_keys)
        # Consume before the first model call, including a zero-budget pass.
        port.charge('native_completion_passes')
        self.session._native_motion_completion_consumed=True
        demands=tuple(s.demand for s in inventory.supplies)
        # A cache can go stale between the first inventory and admission. The
        # pass is already consumed, but no newly inferred observation exists.
        unvalidated=tuple(NativeMotionSupply(d,None) for d in demands)
        self.motion_completion_result=NativeMotionInventory(
            unvalidated,(),True,'CACHE_REVALIDATION_STARTED')
        try:
            current=self.motion_inventory(demands)
        except BaseException:
            self.motion_completion_result=NativeMotionInventory(
                unvalidated,(),True,'CACHE_REVALIDATION_FAILED')
            raise
        core_current=(replace(current,supplies=tuple(replace(s,
            demand=replace(s.demand,require_member_features=False)) for s in current.supplies))
            if member_features_optional else current)
        missing={}
        for supply in core_current.supplies:
            if supply.missing_modalities:
                missing.setdefault(supply.demand.native_key,[]).append(supply.demand)
        first=min(d.output_tick for d in demands)
        end=max(d.output_tick for d in demands)+1
        budget=self.session.budget
        def remaining():
            parent=port.remaining_capacity(('new_decodes','model_calls','pose_evidence_bytes'))
            return {k:max(0,min(budget.limits[k]-budget.counts[k]-budget.reserved[k],parent[k]))
                for k in parent}
        limit=min(maximum_new_observations,len(missing))
        evidence_reserve=primitive_reservation_bytes(MAX_NATIVE_JSON_BYTES)
        known={s.demand.native_key:s.observation for s in current.supplies}
        bindings={s.demand.native_key:s for s in current.supplies}
        appearances={s.demand.native_key:s.appearance for s in current.supplies if s.appearance is not None}
        member_features={s.demand.native_key:s.member_features for s in current.supplies if s.member_features is not None}
        need_appearance={d.native_key for d in demands if d.require_appearance}
        need_member_features={d.native_key for d in demands if d.require_member_features}
        core_demands={s.demand.native_key:replace(s.demand,
            require_appearance=s.demand.native_key in need_appearance,
            require_member_features=not member_features_optional and s.demand.native_key in need_member_features)
            for s in core_current.supplies}
        def work_cost(key,*,optional_member_only=False):
            raw=known[key]
            probe=getattr(self.session,'native_raster_cache_available',None)
            cached=bool(probe is not None and probe(key[0],key[1],
                expected_pixel_sha256=None if raw is None else raw.native_pixel_sha256))
            heads=24 if raw is None else len(raw.heads)
            raw_calls=14 if raw is None else 0
            appearance_calls=heads if key in need_appearance and key not in appearances else 0
            member_calls=heads if key in need_member_features and key not in member_features else 0
            if optional_member_only:raw_calls=appearance_calls=0
            elif member_features_optional:member_calls=0
            # Only NEW normalized raw observations reserve the raw-view prefix.
            # Existing raw-parent descriptor completion does not execute that
            # primitive, and must not be rejected for an unrelated 512KiB hold.
            return NativeMotionCompletionCost(int(not cached),raw_calls+appearance_calls+member_calls,
                evidence_reserve if raw is None and not optional_member_only else 0)
        self.motion_completion_result=replace(current,completion_consumed=True,status='ADMISSION_STARTED')
        try:
            resource_costs={key:work_cost(key) for key in missing}
            available=remaining()  # Capacity after live-raster verification, not a copied allocation.
        except BaseException:
            self.motion_completion_result=replace(current,completion_consumed=True,
                status='ADMISSION_INTERRUPTED_PARTIAL')
            raise
        counts={}; pending=dict(missing); scheduled=[]
        raw_limit=(maximum_new_observations if maximum_new_raw_observations is None
            else maximum_new_raw_observations)
        raw_scheduled=0
        raw_by_key=known
        admission_remaining=dict(available)
        # Bit-reversed order visits distributed time bands before adjacent
        # bands. A later wall/IPC cutoff does not leave a chronological prefix.
        bits=max(1,(limit-1).bit_length())
        bands=sorted(range(limit),key=lambda n:int(format(n,'0'+str(bits)+'b')[::-1],2))
        for band in bands:
            eligible={key:rows for key,rows in pending.items()
                if (raw_by_key[key] is not None or raw_scheduled<raw_limit)
                and all(getattr(resource_costs[key],r)<=admission_remaining[r] for r in COMPLETION_RESOURCES)}
            if not eligible:break
            target=first+(band+.5)*(end-first)/max(1,limit)
            def priority(d):
                return (counts.get((d.source_family,d.opportunity),0),
                    abs(d.output_tick-target),d.source_family,d.opportunity,
                    d.source_index,d.native_frame_index,
                    -1000. if d.analysis_roll is None else d.analysis_roll)
            key,demand=min(((key,min(rows,key=priority)) for key,rows in eligible.items()),
                key=lambda item:priority(item[1]))
            aliases=pending.pop(key)
            scheduled.append(demand)
            for r in COMPLETION_RESOURCES:admission_remaining[r]-=getattr(resource_costs[key],r)
            if raw_by_key[key] is None:raw_scheduled+=1
            # One native fact services every output use. Do not erase a later
            # moving-closeup use by selecting an earlier 'other' representative.
            for lane in {(d.source_family,d.opportunity) for d in aliases}:
                counts[lane]=counts.get(lane,0)+1
        schedule_receipt=dict(policy='source_opportunity_resource_feasible_time_band_balanced_v2',
            initial_live_resource_capacity=available,
            admitted_resource_upper_bounds={r:available[r]-admission_remaining[r] for r in COMPLETION_RESOURCES},
            resource_admission_is_not_atomic_parent_reservation=True,
            successful_observations_or_connected_actors_certified=False,quality_approved=False)
        if required_native_keys or completion_packages:
            scheduled,schedule_receipt=coherent_completion_schedule(core_current,
                required_native_keys=required_native_keys,packages=completion_packages,
                maximum_observations=limit,maximum_raw_observations=raw_limit,
                resource_costs=resource_costs,remaining_resources=available,
                critical_required_native_keys=critical_required_native_keys)
        admitted=tuple(d.native_key for d in scheduled)
        # Consume already-verified live rasters before any new decode can evict
        # them. Never pin extra raster copies or grant a stale zero-decode claim.
        scheduled=tuple(sorted(scheduled,key=lambda d:resource_costs[d.native_key].new_decodes))
        attempted=[];core_attempted=[];optional_attempted=[];optional_unavailable=[];preflight_skipped=[]
        schedule_receipt.update(optional_member_features_are_core_admission_authority=not member_features_optional)
        schedule_receipt.update(admitted_core_native_keys=admitted,
            execution_schedule_core_native_keys=tuple(d.native_key for d in scheduled),
            execution_order_policy='verified_live_rasters_before_cache_misses_no_editorial_rescoring',
            admission_resource_costs=[dict(native_key=k,**resource_costs[k].resources()) for k in admitted])
        status=('BUDGET_PARTIAL' if len(scheduled)<len(missing)
            else 'COMPLETED_ONCE')
        if schedule_receipt.get('terminal_required_dependencies'):status='TERMINAL_REQUIRED_INPUT_BLOCKER'
        def snapshot(state):
            # These rows were validated in this pass. Do not reopen source/model
            # I/O after its deadline merely to construct the result container.
            core_states={k:NativeMotionSupply(d,known[k],appearances.get(k),bindings[k].recorded_input_state,
                bindings[k].source_equivalence_json,member_features.get(k)) for k,d in core_demands.items()}
            completed={k for k,s in core_states.items() if not s.missing_modalities and s.state=='NATIVE_ROWS_AVAILABLE'}
            schedule_receipt.update(attempted_optional_member_native_keys=list(optional_attempted),
                optional_member_unavailable_inputs=list(optional_unavailable),
                attempted_unique_native_keys=list(attempted),attempted_core_native_keys=list(core_attempted),
                unattempted_admitted_core_native_keys=[k for k in admitted if k not in core_attempted],
                completed_attempted_core_native_keys=[k for k in core_attempted if k in completed],
                incomplete_attempted_core_inputs=[dict(native_key=k,state=core_states[k].state)
                    for k in core_attempted if k not in completed],
                live_preflight_skipped_native_inputs=list(preflight_skipped),
                completion_interrupted=state=='INTERRUPTED_PARTIAL',
                package_execution=[dict(identity=p['identity'],
                    attempted_admitted_native_keys=[k for k in p['admitted_missing_native_keys'] if k in core_attempted],
                    unattempted_admitted_native_keys=[k for k in p['admitted_missing_native_keys'] if k not in core_attempted],
                    entire_admitted_core_inputs_completed=all(k in completed for k in p['admitted_missing_native_keys']))
                    for p in schedule_receipt.get('selected_packages',[])])
            schedule_json=__import__('json').dumps(schedule_receipt,sort_keys=True,allow_nan=False)
            value=NativeMotionInventory(tuple(NativeMotionSupply(d,known[d.native_key],appearances.get(d.native_key),
                bindings[d.native_key].recorded_input_state,bindings[d.native_key].source_equivalence_json,
                member_features.get(d.native_key))
                for d in demands),tuple(attempted),True,state,schedule_json)
            self.motion_completion_result=value
            return value
        snapshot('RUNNING')
        try:
            for demand in scheduled:
                budget.check()
                raw=known[demand.native_key]
                required=work_cost(demand.native_key).resources()
                available=remaining()
                if any(available[k]<n for k,n in required.items()):
                    preflight_skipped.append(dict(native_key=demand.native_key,
                        state='LIVE_RESOURCE_CAPACITY_CHANGED',required=required,remaining=available))
                    status='BUDGET_PARTIAL';continue
                attempted.append(demand.native_key)
                core_attempted.append(demand.native_key)
                snapshot('RUNNING')
                if raw is None:
                    known[demand.native_key]=self.native_at(demand.source_index,
                        demand.native_frame_index,demand.analysis_roll)
                    bindings[demand.native_key]=NativeMotionSupply(demand,known[demand.native_key])
                raw=known[demand.native_key]
                if (demand.native_key in need_appearance and demand.native_key not in appearances
                        and raw is not None and raw.native_pixel_sha256 is not None):
                    appearances[demand.native_key]=self._complete_motion_appearance(replace(bindings[demand.native_key],
                        observation=raw))
                if (demand.native_key in need_member_features and demand.native_key not in member_features
                        and raw is not None and raw.native_pixel_sha256 is not None and not member_features_optional):
                    member_features[demand.native_key]=self.member_features_at(replace(bindings[demand.native_key],
                        observation=raw))
                snapshot('RUNNING')
            if member_features_optional and not schedule_receipt.get('terminal_required_dependencies'):
                # Core raw/SFace for every admitted package precede optional
                # photo hints in THIS same consumed pass. No raw re-inference,
                # second completion, or optional budget can displace core.
                for key in dict.fromkeys(d.native_key for d in demands if d.require_member_features):
                    budget.check();raw=known[key]
                    if raw is None or raw.native_pixel_sha256 is None or key in member_features:continue
                    if key in need_appearance and key not in appearances:continue
                    if key not in attempted and len(attempted)>=maximum_new_observations:continue
                    required=work_cost(key,optional_member_only=True).resources()
                    available=remaining()
                    if any(available[k]<n for k,n in required.items()):continue
                    equivalence=bindings[key].source_equivalence_json
                    if equivalence is not None and __import__('json').loads(equivalence)['state']=='CONTENT_VERIFIED_DEVICE_REMOUNT':
                        optional_unavailable.append(dict(native_key=key,state='ARCHIVED_OPTIONAL_RECOMPUTE_UNAVAILABLE'))
                        continue
                    if key not in attempted:attempted.append(key)
                    optional_attempted.append(key);snapshot('RUNNING')
                    member_features[key]=self.member_features_at(replace(bindings[key],observation=raw))
                    snapshot('RUNNING')
                if status=='COMPLETED_ONCE' and any(k not in member_features and known[k] is not None
                        and known[k].native_pixel_sha256 is not None for k in need_member_features):
                    status='CORE_COMPLETED_OPTIONAL_FEATURES_PARTIAL'
        except BaseException:
            snapshot('INTERRUPTED_PARTIAL')
            raise  # Preserve cancellation/cutoff; never synthesize successful completion.
        return snapshot(status)

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


def _states(plan,domain,shots,roles,*,fixed_subcell_views=False):
    return tuple((fixed_neighbor_state_from_render_plan
        if fixed_subcell_views and i in (0,len(shots)-1) else joint_state_from_render_plan)(
        plan=plan,domain=domain,slot=i,
        first_output_frame=s.first_output_frame,end_output_frame=s.end_output_frame,role=roles[i])
        for i,s in enumerate(shots))


def _shot_roles(context, shots):
    from .group_scene_intent import declared_shot_intent
    return tuple(GroupOutputRole(declared_shot_intent(context, shot)['role']) for shot in shots)


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
    from .group_scene_execution_scope import GlobalSceneExecutionPath
    if isinstance(path,GlobalSceneExecutionPath):return path.assemble
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
        is_composition = all(r['role']==GroupOutputRole.COMPOSITION.value for r in raw)
        is_formation = all(r['role']==GroupOutputRole.FORMATION.value for r in raw)
        no_face = is_formation or is_composition
        height=None if no_face else median(r['raw_face_roi_xywh'][3] for r in raw)
        silhouette=(median(max(p['box_xywh'][1]+p['box_xywh'][3] for p in r['native_observation']['people'])
            -min(p['box_xywh'][1] for p in r['native_observation']['people']) for r in raw) if is_formation else None)
        pan=plan.segments[i].pan_crop
        zoom=median(1./_fixed_crop_at(pan,(t-offsets[i])/plan.fps)[3]
            for t in range(offsets[i],offsets[i+1])) if pan else 1.
        rows.append(FaceMatchSegment(i,timeline.choices[i].source_index,pan is not None,
            None if pan else 'fresh_face_fit_owned_by_whole_editorial_camera',height,
            None if no_face else old.target_face_height,old.target_face_height/height if old.target_face_height and height else None,
            zoom,pan,alignment_subject='composition' if is_composition else 'formation' if is_formation else 'face',
            observed_silhouette_height=silhouette,continuous_run_stitched=any(
                span['first_output_frame']<=offsets[i] and span['end_output_frame']>=offsets[i+1]
                and (span['first_output_frame']<offsets[i] or span['end_output_frame']>offsets[i+1])
                for span in dense['mutable_shots']),
            alignment_purpose='fresh_native_source_geometry_and_retested_camera_not_copied_recipe',
            projected_output_observations=dict(samples=projected,basis=observed['basis'],
                native_dense_sample_count=len(raw),raw_native_digest=family.digest(raw))))
    return replace(context.incumbent.face_match_plan,segments=tuple(rows))


def complete_source_candidate(context,path,sensors,session,assets,dense_rows,edge_rows,search_record,
        *,frozen_scene_discovery=None):
    """Every enabled supported family is fresh before a candidate reaches CAS."""
    from .group_joint_render_observations import refresh_joint_group_render_observations
    from .group_boundary_evidence_preservation import preserve_boundary_evidence
    from .pipeline import _deterministic_exact_shot_clock_from_timeline,_validate_review_first_group_bundle
    from .timeline_review import review_timeline
    from .group_transition import GroupTransitionPlan
    from .group_scene_execution_scope import (GlobalSceneExecutionPath,mutable_states,incident_pairs,
        MAX_REFRESHED_CELLS as GLOBAL_MAX_REFRESHED_CELLS)
    global_execution=isinstance(path,GlobalSceneExecutionPath)
    if global_execution:path.validate_external(context,context.domain)
    if global_execution:
        from .group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy
        physical_policy=PreparedPhysicalBoundaryPolicy(original=path.original_plan,observed=path.numeric_plan,
            domain=context.domain,derivative_binding=context.source_derivative_binding)
        physical_policies=physical_policy.require(incident_pairs(path))
    elif context.source_derivative_binding is not None:
        from .group_scene_source_owner import validate_bound_derivative_path
        validate_bound_derivative_path(context.source_derivative_binding,[s.eligible_source_index for s in path])
    old=context.incumbent
    scene = getattr(sensors,'scene_family',False)
    # Numeric assembly and scope size are knowable BEFORE expensive projected
    # descriptors, look fitting or native family refresh. The materializer
    # independently derives the same scope below; no oversized path gets work.
    from .group_scene_dependency_support import derive_support
    numeric,planned_changed=_assembler(path)(plan=old.render_plan,domain=context.domain,path=path)
    planned_operations=None
    if scene:
        from .group_scene_edit_operations import derive_scene_edit_operations
        planned_operations=derive_scene_edit_operations(original=old.render_plan,proposed=numeric,
            domain=context.domain,mutable_intervals=tuple((s.first_output_frame,s.end_output_frame) for s in mutable_states(path)))
        if not planned_operations['has_numeric_edit']:
            raise ValueError('scene completion requires an actual source/camera edit before descriptor work')
    elif not any(numeric.segments[i].source_index!=old.render_plan.segments[i].source_index for i in planned_changed):
        raise ValueError('legacy source-changing owner requires an actual source change')
    retained_look_edges=family.retained_applied_look_edges(old)
    planned_support=derive_support(old.render_plan,numeric,retained_applied_look_edges=retained_look_edges)
    refresh_cap=GLOBAL_MAX_REFRESHED_CELLS if global_execution else MAX_REFRESHED_CELLS
    if len(planned_support['descriptor_refresh_cells'])>refresh_cap:
        raise ValueError('source-family refresh cell budget exceeded before descriptor work')
    observations=refresh_joint_group_render_observations.__wrapped__(plan=old.render_plan,
        timeline=old.visual_timeline,domain=context.domain,path=path,sync_scores=context.sync_scores,
        feature_cache_directory=None,feature_derived_cache_directory=None,
        require_persistent_feature_cache=False,maximum_refreshed_cells=refresh_cap,
        retained_applied_look_edges=retained_look_edges)
    proposed,changed=observations.plan,observations.changed_segment_indices
    if (proposed!=numeric or changed!=planned_changed or observations.dependency_support!=planned_support):
        raise ValueError('source family descriptor work changed its preflight numeric/dependency scope')
    if not changed:
        raise ValueError('source/camera completion has no numeric changed cells')
    refreshed=observations.refreshed_segment_indices
    if len(refreshed)>refresh_cap: raise ValueError('source-family refresh cell budget exceeded')
    edges=tuple(sorted({e for i in changed for e in (i-1,i) if 0<=e<len(proposed.transitions)}))
    before_observed=family.observe_scope(old.render_plan,context.domain,refreshed,edges,context.model_path,session)
    observed=family.observe_scope(proposed,context.domain,refreshed,edges,context.model_path,session)
    plan,look,look_evidence=family.refresh_look(old,proposed,edges,session,context.domain)
    if geometry_payload(plan)!=geometry_payload(proposed):
        raise ValueError('look refresh changed an admitted source clock/camera')
    final_states=(path.with_plan(plan,context.domain) if global_execution else tuple((fixed_neighbor_state_from_render_plan if scene and i in (0,len(path)-1)
        else joint_state_from_render_plan)(plan=plan,domain=context.domain,slot=s.slot,
        first_output_frame=s.first_output_frame,end_output_frame=s.end_output_frame,role=s.role,
        editorial_contract_sha256=s.editorial_contract_sha256) for i,s in enumerate(path)))
    if scene:
        from .group_scene_observations import compare_scene_edge
        fresh_edges = [compare_scene_edge(sensors,a,b) for a,b in incident_pairs(final_states)]
        if global_execution:
            final_policy=PreparedPhysicalBoundaryPolicy(original=path.original_plan,observed=plan,
                domain=context.domain,derivative_binding=context.source_derivative_binding)
            if final_policy.require(incident_pairs(final_states))!=physical_policies:
                raise ValueError('complete family refresh changed original physical event/sampler policy')
            for row,(a,b) in zip(fresh_edges,incident_pairs(final_states),strict=True):final_policy.attach(row,a,b)
        if any(not fresh.get('available') or family.digest(fresh.get('metrics'))!=family.digest(before.get('metrics'))
                for before,fresh in zip(edge_rows,fresh_edges,strict=True)):
            raise ValueError('scene selected incident metrics changed during complete family refresh')
        edge_rows = fresh_edges
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
    if global_execution:
        dense.update(global_execution_scope=path.to_dict(),
            global_execution_original_numeric_plan=old.render_plan.to_json(),
            global_execution_selected_numeric_plan=proposed.to_json(),
            physical_boundary_policies=physical_policies,
            derivative_window_source_indices=[s.eligible_source_index for s in final_states])
    if context.source_derivative_binding is not None:
        dense.update(source_derivative_binding=context.source_derivative_binding,
            derivative_window_source_indices=[s.eligible_source_index for s in final_states])
    selected_program=None
    if scene:
        selected_program=dict(search_record['scene_replay_program'],numeric_plan=proposed.to_json(),
            original_numeric_plan=old.render_plan.to_json(),dependency_support=planned_support,
            edit_operations=planned_operations)
        if global_execution:selected_program['global_execution_scope']=path.to_dict()
        if family.digest(selected_program['selected_states']) != family.digest([s.to_dict() for s in path]):
            raise ValueError('scene completion changed the frozen selected program')
        dense.update(scene_hypotheses={s.editorial_contract_sha256:sensors.hypothesis(s) for s in path},
            hypothesis_keys=[s.editorial_contract_sha256 for s in path],
            original_duty_program=sensors.original_duty_program,
            scene_discovery=sensors.archive() if frozen_scene_discovery is None else frozen_scene_discovery,
            source_derivative_binding=context.source_derivative_binding)
        if selected_program.get('original_duty_program')!=sensors.original_duty_program:
            raise ValueError('scene completion changed the presearch original duty programme')
        from .group_scene_archive import archive_binding, validate_archive
        selected_program['discovery_archive_binding']=archive_binding(dense['scene_discovery'])
        selected_program['program_sha256']=family.digest(selected_program)
        validate_archive(selected_program, dense)
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
        external_original_reference=family.external_reference_binding(context,candidate),
        external_original_selected_boundary_look_metadata=None if candidate.boundary_look_match is None
            else family.plain(candidate.boundary_look_match.to_dict()),
        source_domain=context.domain.to_dict(),synchronization_scores=list(context.sync_scores),
        source_identities=[list(v) for v in context.source_identities],dependencies=dict(context.code_dependencies),
        model_path=str(context.model_path),model_identity=list(context.model_identity),model_sha256=context.model_sha256,
        global_native_schedule_sha256=family._schedule_digest(family._schedule(plan,context.domain)),
        before_observations=before_observed,final_observations=observed,look_refresh=look_evidence,
        boundary_evidence_preservation=preserved,source_camera_refresh=dense,
        search=search_record,native_work=session.budget.to_dict(),
        resource_scope='single_native_session_only_legacy_refresh_parent_and_final_qa_counters_not_aggregated',
        final_quality_status='UNAPPROVED_PERFORMER_REVIEW_CANDIDATE')
    if scene:
        receipt['selected_scene_program']=selected_program
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
    from .group_scene_execution_scope import MAX_MUTABLE_UNITS,dense_layout
    global_scope=isinstance(dense,dict) and dense.get('global_execution_scope') is not None
    mutable_cap=MAX_MUTABLE_UNITS if global_scope else 2
    if (not isinstance(dense,dict) or dense.get('contract') not in DENSE_CONTRACTS
            or dense.get('every_mutable_output_tick_observed') is not True
            or dense.get('face_roi_is_not_physical_head_extent') is not True
            or any(dense.get(k) is not False for k in ('semantic_raw_v2_admission',
                'cross_camera_identity_certified','body_phase_certified',
                'decoded_output_quality_certified','quality_approved'))
            or not isinstance(dense.get('mutable_shots'),list) or not 1<=len(dense['mutable_shots'])<=mutable_cap):
        raise ValueError('fresh native family cannot acquire semantic or quality authority')
    if global_scope:
        if dense.get('contract')!=SCENE_DENSE_CONTRACT:
            raise ValueError('global execution requires the native scene duty consumer')
        mutable,pairs=dense_layout(dense)
        if len(dense['mutable_shots'])!=len(mutable) or len(dense['incident_projected_observations'])!=len(pairs):
            raise ValueError('global dense lost an explicitly owned mutable shot or actual incident edge')
    else:
        # This is the COMPLETE receipt boundary. Standalone registered-motion
        # unit consumers may have no incident edge slice; a full production
        # family cannot omit any edge from its unchanged legacy scope.
        keys=dense.get('state_keys')
        if (not isinstance(keys,list) or len(keys) not in (3,4)
                or len(dense.get('incident_projected_observations',()))!=len(keys)-1):
            raise ValueError('complete native family lost its actual incident edge inventory')
    if global_scope:
        from .group_scene_physical_boundaries import validate_dense_physical_boundaries
        validate_dense_physical_boundaries(dense)
    elif dense.get('source_derivative_binding') is not None:
        from .group_scene_source_owner import validate_bound_derivative_path
        indices=dense.get('derivative_window_source_indices',[])
        validate_bound_derivative_path(dense['source_derivative_binding'],indices)
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
    family.validate_look_refresh(bundle.render_plan,bundle.boundary_look_match,receipt)
    family.validate_external_reference_binding(bundle,receipt)
    validate_dense_uncertainty(dense)
    if dense['contract']==SCENE_DENSE_CONTRACT:
        from .group_scene_observations import validate_scene_dense
        from .group_scene_selected_replay import validate_program
        validate_scene_dense(dense)
        validate_program(receipt.get('selected_scene_program'),dense)
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
        if any((r['face_roi_visible_fraction'] is None if r['role'] not in (GroupOutputRole.FORMATION.value,GroupOutputRole.COMPOSITION.value) else False)
                or (r['face_roi_visible_fraction'] is not None and r['face_roi_visible_fraction']<.995)
                or r['role']!=span['role'] for r in span['rows']):
            raise ValueError('source-changing dense family lost required role/face visibility')


def validate_parent(candidate,context):
    family._fresh(context);validate_binding(candidate)
    receipt=family.plain(candidate.performer_family_review)
    family.validate_look_refresh(candidate.render_plan,candidate.boundary_look_match,receipt,
        original_bundle=context.incumbent)
    family.validate_external_reference_binding(candidate,receipt,context=context)
    if (receipt['source_domain']!=context.domain.to_dict()
            or receipt['source_identities']!=[list(v) for v in context.source_identities]
            or receipt['dependencies']!=dict(context.code_dependencies)
            or receipt['synchronization_scores']!=list(context.sync_scores)
            or receipt['original_metadata_sha256']!=family.digest(family.metadata_payload(context.incumbent))):
        raise ValueError('source-changing parent preimage changed')
    if context.source_derivative_binding is not None and receipt['source_camera_refresh'].get('source_derivative_binding')!=context.source_derivative_binding:
        raise ValueError('scene parent derivative preimage changed')
    if receipt['source_camera_refresh']['contract']==SCENE_DENSE_CONTRACT:
        from .group_scene_intent import declared_shot_intent
        original_shots={(s.first_output_frame,s.end_output_frame):s
            for s in visible_group_shots(context.incumbent.render_plan,context.domain)}
        for h in receipt['source_camera_refresh']['scene_hypotheses'].values():
            shot=original_shots.get((h['span_key'][1],h['span_key'][2]))
            if shot is None or h.get('shot_intent')!=declared_shot_intent(context,shot,
                    endpoint_scope=h.get('endpoint_scope','whole')):
                raise ValueError('scene child role intent differs from the original output-owned shot')
    if receipt['source_camera_refresh']['contract']==SCENE_DENSE_CONTRACT:
        from .group_scene_selected_replay import replay_selected_scene
        proof=replay_selected_scene(context,candidate)
        family._fresh(context)
        return proof
    replay=family.propose_performer_family(context)  # legacy independent isolated worker
    if isinstance(replay,family.PerformerFamilyRejection):
        raise ValueError('source-changing parent independent replay rejected: '+replay.error)
    if candidate.to_dict()!=replay.to_dict() or candidate.render_plan!=replay.render_plan:
        raise ValueError('source-changing parent independent complete sensor/family replay disagrees')
    family._fresh(context)


@isolated_optional()
def final_live_replay(plan_json,receipt,deadline):
    """Final LIVE original-native replay; not a decoded-output quality pass."""
    receipt=family.plain(receipt)
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
            from .group_scene_selected_replay import validate_program
            validate_scene_dense(dense)
            validate_program(receipt.get('selected_scene_program'),dense)
            sensors=SceneSensors(session,assets)
            # Reobserve originals first; a selected/candidate-only receipt
            # cannot redefine already known original parts at final replay.
            from .group_scene_selected_replay import _path
            from .group_scene_original_duties import freeze_original_duty_program
            program=receipt['selected_scene_program']
            original_plan=render_plan_from_json(program['original_numeric_plan'])
            from .group_scene_execution_scope import (GlobalSceneExecutionPath,endpoint_scope,incident_pairs)
            global_scope=program.get('global_execution_scope')
            baseline=_path(original_plan,domain,program['baseline_states'],
                global_execution_scope=global_scope,original_plan=original_plan)
            from .group_scene_edit_operations import validate_scene_edit_operations
            validate_scene_edit_operations(program['edit_operations'],original=original_plan,proposed=plan,
                source_domain=raw,mutable_intervals=(baseline.mutable_intervals if isinstance(baseline,GlobalSceneExecutionPath)
                    else tuple(tuple(span) for span in dense['window_bounds'][1:-1])))
            original_groups=[]
            for index,state in enumerate(baseline):
                scope=endpoint_scope(baseline,index)
                unit=program['original_duty_program']['units'][index]
                intent=unit['branches'][0]['original_hypothesis']['shot_intent']
                sensors.shot_intents[state.first_output_frame,state.end_output_frame,scope]=intent
                original_groups.append(tuple(sensors.bind(state,h) for h in sensors.discover(state,
                    rolls=tuple(program['analysis_rolls']),endpoint_scope=scope)))
            if [[s.editorial_contract_sha256 for s in group] for group in original_groups]!=program['baseline_hypothesis_keys']:
                raise ValueError('scene final original hypothesis inventory changed')
            original_duties=freeze_original_duty_program(sensors,original_groups,
                original_plan_identity=family.digest(family.plan_payload(original_plan)),
                global_execution_scope=global_scope)
            if original_duties!=program['original_duty_program']:
                raise ValueError('scene final fresh original required duty lifecycle changed')
            sensors.install_original_duty_program(original_duties)
            sensors.hypotheses.update(dense['scene_hypotheses'])
            keys=dense['hypothesis_keys']
        else:
            sensors=NativeFamilySensors(session,assets,frozen_declarations=dense['pre_camera_native_declarations'])
            keys=[None]*len(dense['window_bounds'])
        if dense.get('global_execution_scope') is not None:
            selected_numeric=render_plan_from_json(program['numeric_plan'])
            path=_path(selected_numeric,domain,program['selected_states'],global_execution_scope=global_scope,
                original_plan=original_plan).with_plan(plan,domain)
        else:path=tuple((fixed_neighbor_state_from_render_plan if dense['contract']==SCENE_DENSE_CONTRACT
            and i in (0,len(dense['window_bounds'])-1) else joint_state_from_render_plan)(
            plan=plan,domain=domain,slot=i,first_output_frame=b[0],
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
                if state.role not in (GroupOutputRole.FORMATION,GroupOutputRole.COMPOSITION):
                    for i in range(0,len(h['anchor_output_ticks']),2):
                        j=i+1
                        a=sensors.raw(state,h['anchor_output_ticks'][i],h['analysis_roll'])
                        b=sensors.raw(state,h['anchor_output_ticks'][j],h['analysis_roll'])
                        if h['anchor_head_ids'][i:i+2] not in sensors._head_links(a,b)['unique_links']:
                            raise ValueError('scene final fixed/mutable head pixel hypothesis changed')
                sides=(tuple(sorted({True for a,b in incident_pairs(path) if a is state}
                    | {False for a,b in incident_pairs(path) if b is state})) if isinstance(path,GlobalSceneExecutionPath) else
                    (True,) if state is path[0] else (False,) if state is path[-1] else (False,True))
                for outgoing in sides:
                    sensors.endpoint(state,outgoing=outgoing)
        if [s.key for s in path]!=dense['state_keys'] or family.digest(sensors.dense(path))!=family.digest(dense['mutable_shots']):
            raise ValueError('source-changing final all-tick native/camera replay disagrees')
        if dense['contract']==SCENE_DENSE_CONTRACT:
            from .group_scene_observations import compare_scene_edge
            fresh_edges=[compare_scene_edge(sensors,a,b) for a,b in incident_pairs(path)]
            if isinstance(path,GlobalSceneExecutionPath):
                from .group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy
                physical_policy=PreparedPhysicalBoundaryPolicy(original=original_plan,observed=plan,domain=domain,
                    derivative_binding=dense.get('source_derivative_binding'))
                if physical_policy.require(incident_pairs(path))!=dense['physical_boundary_policies']:
                    raise ValueError('scene final original physical event/sampler policy changed')
                for row,(a,b) in zip(fresh_edges,incident_pairs(path),strict=True):physical_policy.attach(row,a,b)
            if family.digest(fresh_edges)!=family.digest(dense['incident_projected_observations']):
                raise ValueError('scene final selected incident articulation/geometry replay disagrees')
    finally: session.close()
    if [list(_file_identity(p)) for p in domain.eligible_sources]!=receipt['source_identities']:
        raise ValueError('source-changing source changed during final dense replay')
    if asset_payload(assets)!=dense['native_assets']: raise ValueError('source-changing native assets changed during final replay')
    metrics.update(contract=CONTRACT,dense_native_output_tick_count=sum(len(s['rows']) for s in dense['mutable_shots']),
        semantic_raw_v2_admission=False,cross_camera_identity_certified=False,body_phase_certified=False)
    metrics.pop('evidence_sha256',None);metrics['evidence_sha256']=family.digest(metrics)
    return metrics
