"""Measured conditional ROI texture movement, not detector jitter or anatomy."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import cv2
import pytest

from crosscut.group_joint_source_camera import GroupOutputRole, GroupSourceCameraState
from crosscut.render import RenderSegment, EditorialTransformTrack, EditorialTransformKeyframe
from engine.tests.test_scene_region_flow import world


def pixel_edge_fixture(*,opposite_pixels=False,detector_jitter=False):
    """Explicit non-human rows/cell intent; real LK/projection/consumer math."""
    from crosscut import group_performer_family_owner as family
    from crosscut.group_scene_observations import compare_scene_edge, SCENE_CONTRACT
    from crosscut.group_scene_region_flow import measure_region_distribution
    from crosscut.group_scene_intent import CONTRACT as INTENT_CONTRACT
    from crosscut.group_source_domain import GroupSourceDomain
    from crosscut.render_frame_schedule import VerifiedRenderFrameClock
    clock = VerifiedRenderFrameClock(Fraction(30), 12, 'a'*64, 'explicit_nonhuman_clock')
    domain = GroupSourceDomain((Path('EXPLICIT_CHART_NOT_OPENED'),),(0,),(0.,),(None,),((),),(clock,))
    clock_hash = family.digest(clock.to_dict())
    sensor, original, images = world()
    rows = tuple(replace(r,source_clock_identity=clock_hash) for r in original)
    flows={0:measure_region_distribution(sensor,rows[0],rows[3])}
    distinct=opposite_pixels or detector_jitter
    if distinct:
        base=images[0].copy();base[90:]=0
        yy,xx=np.indices(base.shape[:2],dtype=np.float32)
        images={n:cv2.remap(base,xx-(0 if detector_jitter else (1 if n<4 else -1)*2*(n%4)),
            yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for n in range(8)}
        sensor.session.native_pixels=lambda source,n:images[n]
        rows=tuple(replace(original[n%4],native_frame_index=n,source_clock_identity=clock_hash,
            native_pixel_sha256=sha256(memoryview(images[n])).hexdigest(),
            people=tuple(replace(p,box_xywh=(p.box_xywh[0]+(.004*(n%4) if detector_jitter and n>=4 else 0),
                *p.box_xywh[1:])) for p in original[n%4].people)) for n in range(8))
        sensor.link_cache={}
        flows={0:measure_region_distribution(sensor,rows[0],rows[3]),
            4:measure_region_distribution(sensor,rows[4],rows[7])}
    fixed = EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),EditorialTransformKeyframe(.1,1.1)))
    moved = EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),EditorialTransformKeyframe(.1,1.1,.51)))
    states=[]; hypotheses={}
    incoming=state(camera=fixed if distinct else moved,first=4)
    if distinct:incoming=replace(incoming,native_frame_indices=(4,5,6,7),
        segments=(replace(incoming.segments[0],source_start=4/30),))
    for chosen in (state(camera=fixed),incoming):
        first,end=chosen.first_output_frame,chosen.end_output_frame
        cell=dict(segment_index=0,first_output_frame=first,end_output_frame=end,
            shot_type='group_wide',alignment_subject=None,face_applied=False)
        intent=dict(contract=INTENT_CONTRACT,role=chosen.role.value,first_output_frame=first,end_output_frame=end,
            cells=[cell],retained_output_cells=[cell],endpoint_scope='whole',required_output_interval=[first,end],
            fixed_nonincident_pixels_unchanged=False,frozen_before_source_camera_search=True,
            incident_boundary_does_not_own_whole_shot=True,quality_approved=False)
        h=dict(contract=SCENE_CONTRACT,span_key=[0,first,end,chosen.native_frame_indices[0],
            chosen.native_frame_indices[-1],chosen.role.value],
            role=chosen.role.value,shot_intent=intent,endpoint_scope='whole',quality_approved=False)
        key=family.digest(h);hypotheses[key]=h
        states.append(replace(chosen,source_clock_identity=clock_hash,editorial_contract_sha256=key).bind_source(domain))
    def frame(chosen,tick):
        raw=rows[chosen.native_frame_indices[tick-chosen.first_output_frame]]
        geometry=dict(occupancy=.42,attention_xy=[.5,.5],
            subject_height_quantiles=[.75]*3,horizontal_span=.78)
        if detector_jitter:
            from crosscut.group_scene_observations import _scene_geometry
            from crosscut.group_source_changing_owner import project_box
            geometry=_scene_geometry([],[
                dict(projected_box_xywh=project_box(p.box_xywh,chosen,tick)[0]) for p in raw.people])
        return dict(output_tick=tick,role=chosen.role.value,
            native_observation=raw.to_dict(),scene_geometry=geometry)
    def endpoint(chosen,*,outgoing):
        return dict(output_ticks=[chosen.first_output_frame,chosen.end_output_frame-1],
            composition_support_link=flows[4 if distinct and chosen.first_output_frame==4 else 0],
            required_semantics_available=True,
            verified_source_clock=clock.to_dict())
    import crosscut.group_scene_articulation as articulation
    before=dict(sensor.session.budget.counts)
    with patch.object(articulation,'compare_composition_articulation',return_value=dict(available=False)):
        edge=compare_scene_edge(NS(frame=frame,endpoint=endpoint),*states)
    assert sensor.session.budget.counts==before
    return edge,dict(hypotheses=hypotheses,hypothesis_keys=[s.editorial_contract_sha256 for s in states],
        state_keys=[s.key for s in states],source_domain=domain.to_dict())


def state(*, camera=None, first=0):
    return GroupSourceCameraState(1, first, first+4, 0,
        (RenderSegment(0, 0., 4/30, first/30, editorial_transform=camera),),
        (0, 1, 2, 3), GroupOutputRole.COMPOSITION, 'c'*64,
        output_size=(320, 180))


def test_fixed_detector_rectangles_do_not_turn_nonrigid_pixel_motion_into_zero():
    from crosscut.group_scene_region_flow import measure_region_distribution, project_region_distribution
    sensors, rows, _ = world()
    value = measure_region_distribution(sensors, rows[0], rows[3])
    before = dict(sensors.session.budget.counts)
    observed = project_region_distribution(value, rows[0].to_dict(), rows[3].to_dict(),
        state=state(), output_ticks=(0, 3))
    assert observed['basis'] == 'conditional_native_roi_feature_distribution_projected_by_complete_camera'
    assert observed['velocity_quantiles'][0][0] < -.10
    assert observed['velocity_quantiles'][2][0] > .10
    assert observed['named_identity_certified'] is False
    assert observed['articulated_motion_certified'] is False
    assert observed['quality_approved'] is False
    assert sensors.session.budget.counts == before  # No new decode, NN or LK.


def test_projection_uses_the_real_continuous_camera_instead_of_subtracting_native_medians():
    from crosscut.group_scene_region_flow import measure_region_distribution, project_region_distribution
    sensors, rows, _ = world()
    value = measure_region_distribution(sensors, rows[0], rows[3])
    camera = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.1, .50, .50),
        EditorialTransformKeyframe(.1, 1.1, .51, .50)))
    fixed = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.1),
        EditorialTransformKeyframe(.1, 1.1)))
    native = project_region_distribution(value, rows[0].to_dict(), rows[3].to_dict(),
        state=state(camera=fixed), output_ticks=(0, 3))
    moved = project_region_distribution(value, rows[0].to_dict(), rows[3].to_dict(),
        state=state(camera=camera), output_ticks=(0, 3))
    delta = np.asarray(moved['velocity_quantiles'])-native['velocity_quantiles']
    assert delta[:,0] == pytest.approx([-.11]*3, abs=1e-6)
    assert delta[:,1] == pytest.approx([0.]*3, abs=1e-6)


@pytest.mark.parametrize('change', ['clock', 'source', 'tick', 'flow'])
def test_native_flow_cannot_be_projected_through_a_different_source_occurrence(change):
    from crosscut.group_scene_region_flow import measure_region_distribution, project_region_distribution
    sensors, rows, _ = world()
    from crosscut.group_scene_region_storage import expand_region_tracks
    value = deepcopy(expand_region_tracks(measure_region_distribution(sensors, rows[0], rows[3])))
    chosen = state(); ticks = (0,3)
    if change == 'clock': chosen = replace(chosen, source_clock_identity='d'*64)
    elif change == 'source': chosen = replace(chosen, segments=(replace(chosen.segments[0], source_index=1),))
    elif change == 'tick': ticks = (0,2)
    else: value['region_tracks'][0]['terminal_feature_ids'] = []
    with pytest.raises(ValueError):
        project_region_distribution(value, rows[0].to_dict(), rows[3].to_dict(),
            state=chosen, output_ticks=ticks)


def test_unavailable_pixel_motion_is_not_a_detector_center_success_fallback():
    from crosscut.group_scene_region_flow import measure_region_distribution, project_region_distribution
    sensors, rows, _ = world(flat=True)
    value = measure_region_distribution(sensors, rows[0], rows[3])
    with pytest.raises(ValueError):
        project_region_distribution(value, rows[0].to_dict(), rows[3].to_dict(),
            state=state(), output_ticks=(0,3))


def test_composition_edge_uses_supported_pixel_velocity_not_constant_detector_attention():
    from crosscut.group_scene_observations import compare_scene_edge
    from crosscut.group_scene_region_flow import measure_region_distribution
    import crosscut.group_scene_articulation as articulation
    left, right = state(), state(first=4)
    a, rows, _ = world()
    flow = measure_region_distribution(a, rows[0], rows[3])
    def frame(chosen, tick):
        raw = rows[tick-chosen.first_output_frame]
        return dict(output_tick=tick, role=chosen.role.value, native_observation=raw.to_dict(),
            scene_geometry=dict(occupancy=.42, attention_xy=[.5,.5],
                subject_height_quantiles=[.75]*3, horizontal_span=.78))
    def endpoint(chosen, *, outgoing):
        return dict(output_ticks=[chosen.first_output_frame,chosen.end_output_frame-1],
            composition_support_link=flow, required_semantics_available=True)
    sensor = NS(frame=frame,endpoint=endpoint)
    # Reverse actual supported image movement on the incoming side, with the
    # same static detector rectangles. The observer, not a fabricated cost,
    # owns the mismatch. Projection helpers are tested independently above.
    with patch.object(articulation,'compare_composition_articulation',return_value=dict(available=False)), \
            patch('crosscut.group_scene_observations.composition_endpoint_screen_motion',side_effect=[
                dict(velocity_quantiles=[[-.2,0.],[0.,0.],[.2,0.]],basis='actual_pixel_probe'),
                dict(velocity_quantiles=[[0.,-.2],[0.,0.],[0.,.2]],basis='actual_pixel_probe')]):
        edge = compare_scene_edge(sensor,left,right)
    assert edge['metrics']['screen_motion'] > .1
    assert edge['camera_direction_basis'] == 'conditional_supported_pixel_distribution_not_detector_attention_or_3d_camera'
    assert edge['detector_attention_velocity_outgoing'] == [0.,0.]
    assert edge['detector_attention_velocity_incoming'] == [0.,0.]


def test_real_nonrigid_lk_projection_reaches_received_edge_consumer_without_new_native_work():
    from crosscut.group_scene_observations import validate_composition_edge_motion
    edge,external=pixel_edge_fixture()
    assert edge['available'] is True and edge['metrics']['screen_motion']==pytest.approx(.11,abs=1e-6)
    assert edge['detector_attention_velocity_outgoing']==[0.,0.]
    assert edge['detector_attention_velocity_incoming']==[0.,0.]
    validate_composition_edge_motion(edge,**external)


def test_actual_opposite_pixel_flow_changes_the_normal_edge_score_with_fixed_boxes():
    from crosscut.group_scene_observations import validate_composition_edge_motion
    edge,external=pixel_edge_fixture(opposite_pixels=True)
    assert edge['available'] and edge['metrics']['screen_motion']>.25
    assert edge['detector_attention_velocity_outgoing']==[0.,0.]
    assert edge['detector_attention_velocity_incoming']==[0.,0.]
    assert edge['composition_pixel_motion'][0]['velocity_quantiles'][1][0]>0.
    assert edge['composition_pixel_motion'][1]['velocity_quantiles'][1][0]<0.
    validate_composition_edge_motion(edge,**external)


def test_static_pixels_with_jittering_detector_boxes_are_not_supported_image_motion():
    from crosscut.group_scene_observations import validate_composition_edge_motion
    edge,external=pixel_edge_fixture(detector_jitter=True)
    assert edge['available'] and edge['metrics']['screen_motion']<1e-5
    assert abs(edge['detector_attention_velocity_incoming'][0])>.1
    validate_composition_edge_motion(edge,**external)


def test_verified_native_time_and_editing_camera_contribution_remain_separate():
    from crosscut.group_performer_family_owner import digest
    from crosscut.group_scene_region_flow import measure_region_distribution,project_region_distribution
    from crosscut.render_frame_schedule import VerifiedRenderFrameClock
    clock=VerifiedRenderFrameClock(Fraction(30),12,'a'*64)
    sensor,original,_=world()
    rows=tuple(replace(r,source_clock_identity=digest(clock.to_dict())) for r in original)
    flow=measure_region_distribution(sensor,rows[0],rows[3])
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),
        EditorialTransformKeyframe(.1,1.1,.51)))
    chosen=replace(state(camera=camera),source_clock_identity=digest(clock.to_dict()))
    view=project_region_distribution(flow,rows[0].to_dict(),rows[3].to_dict(),
        state=chosen,output_ticks=(0,3),native_clock=clock)
    assert view['source_native_times']==['0','1/10']
    assert view['source_native_duration_seconds']==pytest.approx(.1)
    assert view['output_duration_seconds']==pytest.approx(.1)
    assert view['source_native_time_basis']=='verified_exact_cfr_native_presentation_time'
    assert np.asarray(view['editing_camera_contribution_quantiles'])[:,0]==pytest.approx([-.11]*3)
    assert np.asarray(view['image_displacement_at_same_output_transform_quantiles'])[:,0]==pytest.approx(
        np.asarray(view['native_velocity_quantiles'])[:,0]*1.1,abs=1e-6)
    assert view['articulated_motion_certified'] is False


def test_quantized_index_grid_does_not_invent_exact_native_timestamps_or_drop_screen_motion():
    from crosscut.group_performer_family_owner import digest
    from crosscut.group_scene_region_flow import measure_region_distribution,project_region_distribution
    from crosscut.render_frame_schedule import VerifiedRenderFrameClock
    clock=VerifiedRenderFrameClock(Fraction(30),12,'a'*64,'bijective_nearest_index_presentation_grid')
    sensor,original,_=world()
    rows=tuple(replace(r,source_clock_identity=digest(clock.to_dict())) for r in original)
    flow=measure_region_distribution(sensor,rows[0],rows[3])
    view=project_region_distribution(flow,rows[0].to_dict(),rows[3].to_dict(),
        state=replace(state(),source_clock_identity=digest(clock.to_dict())),output_ticks=(0,3),
        native_clock=clock)
    assert view['source_native_times'] is None and view['native_velocity_quantiles'] is None
    assert view['source_native_time_basis']=='native_pts_not_available_from_index_only_grid'
    assert np.asarray(view['velocity_quantiles'])[2,0]>.1


@pytest.mark.parametrize('change',['quantile','metric','camera','missing','basis','raw','claim','native_clock'])
def test_received_motion_cost_cannot_override_its_bound_pixels_clock_or_camera(change):
    from crosscut.group_scene_observations import validate_composition_edge_motion
    edge,external=pixel_edge_fixture();edge=deepcopy(edge)
    if change=='quantile':edge['composition_pixel_motion'][0]['velocity_quantiles'][0][0]+=.3
    elif change=='metric':edge['metrics']['screen_motion']=0.
    elif change=='camera':
        edge['composition_pixel_motion'][0]['complete_state_preimage']['key_preimage']['segments'][0]['source_index']=1
    elif change=='missing':edge.pop('composition_pixel_motion')
    elif change=='basis':edge['camera_direction_basis']='measured_screen_attention_motion_not_inferred_3d_camera'
    elif change=='raw':edge['composition_pixel_motion'][0]['raw_observation_sha256'][0]='e'*64
    elif change=='native_clock':edge['required_endpoint_evidence'][0]['verified_source_clock']['frame_rate']='60'
    else:edge['composition_pixel_motion'][0]['body_phase_certified']=True
    with pytest.raises(ValueError):validate_composition_edge_motion(edge,**external)


def test_unknown_pixel_motion_stays_unknown_and_keeps_separately_known_geometry():
    from crosscut.group_scene_observations import validate_composition_edge_motion
    edge,external=pixel_edge_fixture()
    edge.update(available=False,composition_pixel_motion=None)
    edge['metrics']['screen_motion']=None
    validate_composition_edge_motion(edge,**external)
    assert edge['metrics']['occupancy']==0.
    edge['metrics']['screen_motion']=0.
    with pytest.raises(ValueError,match='imputed'):validate_composition_edge_motion(edge,**external)
