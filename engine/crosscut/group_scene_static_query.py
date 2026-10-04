"""One actual native/static query over a registered full physical camera.

One-frame original role epochs are legal observations. They do not invent
temporal identity or motion, and do not alter renderer source/camera phase.
This V2 review measurement is deliberately not the legacy dense certificate.
"""
from dataclasses import dataclass
import json

from .group_scene_camera_queries import GroupSceneRoleQuery
from .group_joint_native_observer import observe_native_group
from .group_performer_family_owner import digest
from .group_source_changing_owner import project_point,project_box

CONTRACT='registered_original_role_static_query_v2'


@dataclass(frozen=True,slots=True)
class StaticRoleMeasurement:
    query_json:str
    output_tick:int
    native_json:str
    projection_json:str

    def to_dict(self):
        return dict(contract=CONTRACT,query=json.loads(self.query_json),output_tick=self.output_tick,
            native_observation=json.loads(self.native_json),projection=json.loads(self.projection_json),
            one_frame_static_observation=True,motion_observed=False,
            actor_identity_certified=False,candidate_admitted=False,quality_approved=False)


def observe_static_role_query(session,query,tick,*,cache=None):
    """Actual inference with all raw context retained; no endpoint fake rows."""
    if not isinstance(query,GroupSceneRoleQuery):
        raise ValueError('static role observation needs a registered whole-camera query')
    query.bind_source(session.domain)
    # Unresolved original intent remains unresolved, rather than selecting a
    # convenient role from the donor. It is still a measurable static frame.
    segment,phase,native=query.sample_at(tick)
    request=session.domain.request_index(query.eligible_source_index)
    key=(request,native)
    if cache is None or key not in cache:
        raw=observe_native_group(session,request_source_index=request,native_frame_index=native)
        if raw.native_pixel_sha256 is None:raise ValueError('static role query exact native raster unavailable')
        if cache is not None:cache[key]=(session,raw)
    else:
        owner,raw=cache[key]
        if owner is not session:raise ValueError('static query cache belongs to another native owner')
        if (raw.request_source_index!=request or raw.native_frame_index!=native
                or raw.source_clock_identity!=query.source_clock_identity):
            raise ValueError('static role query cache borrowed another source/clock/frame')
        # Cached normalized rows never own file/model reuse after their live
        # session closes. The caller retains this session, not a saved receipt.
        if session.closed:raise ValueError('static query cannot reuse a closed native session')
        session._assert_source_identity(request)
        for name,_ in raw.model_content_sha256:session._assert_model_assets(name)
    heads=[dict(observation_id=h.observation_id,raw_box_xywh=list(h.box_xywh),
        projected_box_xywh=list(project_box(h.box_xywh,query,tick)[0]),
        visible_fraction=project_box(h.box_xywh,query,tick)[1],
        projected_landmarks_xy=[list(project_point(p,query,tick)) for p in h.landmarks_xy]) for h in raw.heads]
    people=[dict(observation_id=p.observation_id,raw_box_xywh=list(p.box_xywh),
        projected_box_xywh=list(project_box(p.box_xywh,query,tick)[0]),
        visible_fraction=project_box(p.box_xywh,query,tick)[1],
        projected_upper_body_xyc=[[*project_point(v[:2],query,tick),v[2]] for v in p.upper_body_xyc],
        unique_head_observation_id=p.unique_head_observation_id,part_states=dict(p.part_states)) for p in raw.people]
    freeze=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    projected=dict(camera_sha256=query.camera.key,whole_renderer_identity=query.camera.full_renderer_identity,
        original_role_program_sha256=digest(query.original_role_program),
        original_unit_id=query.original_unit_id,original_epoch_index=query.epoch_index,
        original_role=None if query.role is None else query.role.value,
        source_request_index=request,native_frame_index=native,source_clock_identity=raw.source_clock_identity,
        canonical_segment_phase_seconds=phase,heads=heads,people=people,
        unknown_owner_ids=[p.observation_id for p in raw.people if p.unique_head_observation_id is None],
        physical_limb_masks_observed=False)
    return StaticRoleMeasurement(freeze(query.to_dict()),tick,freeze(raw.to_dict()),freeze(projected))


def reproject_recorded_static_query(query,tick,raw,*,domain,compact=False):
    """Numeric review reuse only; NEVER a fresh/live native observation.

    A persisted detector result may rank another camera over the exact same
    registered source/native frame. New selected/parent consumers must still
    perform their own native measurements. A self-consistent archive cannot
    grant a live session, motion, actor, admission or quality certificate.
    """
    if not isinstance(query,GroupSceneRoleQuery):raise ValueError('recorded projection needs a registered camera query')
    query.bind_source(domain);_,phase,native=query.sample_at(tick)
    request=domain.request_index(query.eligible_source_index)
    if (raw['request_source_index']!=request or raw['native_frame_index']!=native
            or raw['source_clock_identity']!=query.source_clock_identity
            or not isinstance(raw['native_pixel_sha256'],str) or len(raw['native_pixel_sha256'])!=64
            or raw.get('quality_approved') is not False):
        raise ValueError('recorded numeric projection borrowed a source/native/clock or approval')
    heads=[dict(observation_id=h['observation_id'],raw_box_xywh=list(h['box_xywh']),
        projected_box_xywh=list(project_box(h['box_xywh'],query,tick)[0]),
        visible_fraction=project_box(h['box_xywh'],query,tick)[1],
        projected_landmarks_xy=[list(project_point(p,query,tick)) for p in h['landmarks_xy']]) for h in raw['heads']]
    people=[dict(observation_id=p['observation_id'],raw_box_xywh=list(p['box_xywh']),
        projected_box_xywh=list(project_box(p['box_xywh'],query,tick)[0]),
        visible_fraction=project_box(p['box_xywh'],query,tick)[1],
        projected_upper_body_xyc=[[*project_point(v[:2],query,tick),v[2]] for v in p['upper_body_xyc']],
        unique_head_observation_id=p['unique_head_observation_id'],part_states=dict(p['part_states'])) for p in raw['people']]
    projection=dict(camera_sha256=query.camera.key,whole_renderer_identity=query.camera.full_renderer_identity,
        original_role_program_sha256=digest(query.original_role_program),original_unit_id=query.original_unit_id,
        original_epoch_index=query.epoch_index,original_role=None if query.role is None else query.role.value,
        source_request_index=request,native_frame_index=native,source_clock_identity=raw['source_clock_identity'],
        canonical_segment_phase_seconds=phase,heads=heads,people=people,
        unknown_owner_ids=[p['observation_id'] for p in raw['people'] if p['unique_head_observation_id'] is None],
        physical_limb_masks_observed=False)
    if type(compact) is not bool:raise ValueError('recorded projection compact mode must be explicit')
    query_data=(dict(camera=dict(output_fps=query.camera.output_fps,output_size=list(query.camera.output_size)),
        registered_query_sha256=query.key) if compact else query.to_dict())
    return dict(contract='recorded_source_native_static_query_reprojection_for_review_v1',
        query=query_data,output_tick=tick,native_observation=raw,projection=projection,
        one_frame_static_observation=False,recorded_observation_reprojected=True,
        fresh_inference_performed=False,live_native_authority=False,motion_observed=False,
        actor_identity_certified=False,candidate_admitted=False,quality_approved=False)
