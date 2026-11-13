"""Real scene projection/rejection on explicit non-human observation fixtures.

Keeping known geometry does not admit cropped required parts or certify a
real performer, source occlusion, body ownership, or professional quality.
"""
from copy import deepcopy
from dataclasses import replace
import pytest

from crosscut import group_scene_observations as scene
from crosscut.render import RenderSegment, EditorialTransformTrack, EditorialTransformKeyframe
from engine.tests.test_scene_focal_pose_motion import scene_world


def rejected_world(kind):
    sensors,original,h,rows,_=scene_world()
    rows.update({n:replace(rows[n-8],native_frame_index=n) for n in range(8,16)})
    sensors.raw=lambda chosen,tick,roll:rows[chosen.native_frame_indices[tick-chosen.first_output_frame]]
    right=replace(original,slot=2,first_output_frame=8,end_output_frame=16,
        native_frame_indices=tuple(range(8,16)),
        segments=(RenderSegment(0,8/30,8/30,8/30),))
    intent=deepcopy(h['shot_intent'])
    intent.update(first_output_frame=8,end_output_frame=16,required_output_interval=[8,16])
    for cell in intent['cells']:
        cell.update(first_output_frame=8,end_output_frame=16)
    for cell in intent['retained_output_cells']:
        cell.update(first_output_frame=8,end_output_frame=16)
    sensors.shot_intents[8,16,'whole']=intent
    right=sensors.bind(right,sensors.discover(right,rolls=(None,))[0])
    if kind in ('head','arm'):
        center_y=.6 if kind=='head' else .25
        camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,2.,.5,center_y),
            EditorialTransformKeyframe(7/30,2.,.5,center_y)))
        left=replace(original,segments=(replace(original.segments[0],editorial_transform=camera),))
    else:
        left=original
        for n in range(8):
            if kind=='native_face':
                rows[n]=replace(rows[n],heads=tuple(replace(head,box_xywh=(-.02,.29,.06,.08)) for head in rows[n].heads))
            elif kind=='missing_head':rows[n]=replace(rows[n],heads=())
            elif kind=='raster':rows[n]=replace(rows[n],native_pixel_sha256=None)
            elif kind=='aspect':rows[n]=replace(rows[n],frame_size_wh=(1000,800))
            else:raise AssertionError('unknown explicit fixture')
    return sensors,left,right


@pytest.mark.parametrize('kind',['head','arm','native_face','missing_head'])
def test_actual_frame_rejection_keeps_known_whole_camera_geometry_without_admission(kind):
    sensors,left,right=rejected_world(kind)
    # No endpoint LK/NN work is needed to expose the already known clipping.
    endpoint_calls=[]
    sensors.endpoint=lambda *a,**kw:endpoint_calls.append((a,kw))
    edge=scene.compare_scene_edge(sensors,left,right)
    assert edge['available'] is False and edge['required_semantics_available'] is False
    assert edge['metrics'] is not None  # RED before F21: the old frame throws first.
    assert edge['metrics']['occupancy'] is not None
    assert edge['metrics']['attention'] is not None
    assert edge['metrics']['scale'] is not None
    assert edge['known_geometry_retained_on_frame_rejection'] is True
    assert edge['frame_admission_violations']
    assert edge['required_endpoint_evidence'] is None and endpoint_calls==[]
    assert edge['cross_camera_identity_certified'] is False
    bad=edge['rows'][0]
    assert bad['role_geometry_constraints_satisfied'] is False
    assert bad['frame_admission_violations']
    assert bad['native_observation']['native_frame_index']==left.native_frame_indices[4]
    assert bad['scene_geometry'] is not None
    if kind=='missing_head':
        assert bad['role_obligations'] is None  # Unknown duty is not an empty duty.
        assert bad['scene_people'][0]['required_by_role'] is None
        assert bad['role_obligations_unknown_reason']
    elif kind=='arm':
        assert bad['role_obligations']['required_parts']
        assert any(a['required_by_role'] and not a['measured_joints_visible']
            for a in bad['projected_observed_arms'])


@pytest.mark.parametrize('kind',['head','arm','native_face'])
def test_rejected_frame_cache_rethrows_and_does_not_make_second_use_admissible(kind):
    sensors,left,_=rejected_world(kind)
    caught=[]
    for _ in range(2):
        with pytest.raises(ValueError) as value:sensors.frame(left,4)
        assert hasattr(value.value,'row'), 'rejected projection must carry its known geometry'
        caught.append(value.value.row)
        if len(caught)==1:counts=dict(sensors.session.budget.counts)
        else:assert sensors.session.budget.counts==counts
    assert caught[0] is caught[1]
    assert caught[0]['frame_admission_violations']
    with pytest.raises(ValueError):sensors.endpoint(left,outgoing=True)
    with pytest.raises(ValueError):sensors.dense((left,left,left))


@pytest.mark.parametrize('kind',['raster','aspect'])
def test_unavailable_raster_or_unsupported_projection_never_fabricates_known_geometry(kind):
    sensors,left,right=rejected_world(kind)
    result=scene.compare_scene_edge(sensors,left,right)
    assert not result['available'] and result['metrics'] is None
    assert result.get('known_geometry_retained_on_frame_rejection') is not True


def test_valid_frame_retains_strict_constraints_and_explicit_no_quality_claim():
    sensors,chosen,_,_,_=scene_world()
    row=sensors.frame(chosen,4)
    assert row['role_geometry_constraints_satisfied'] is True
    assert row['frame_admission_violations']==[]
    assert row['role_obligations_unknown_reason'] is None
    assert row['quality_approved'] is False
    assert row['role_obligations']['required_parts']
    assert row is sensors.frame(chosen,4)


def test_partial_rejection_geometry_cannot_buy_editorial_gain_by_becoming_available():
    sensors,left,right=rejected_world('arm')
    edge=scene.compare_scene_edge(sensors,left,right)
    assert edge['metrics'] is not None
    candidate=dict(edge,available=True,required_semantics_available=True)
    assert not scene.has_editorial_gain(edge,candidate)
