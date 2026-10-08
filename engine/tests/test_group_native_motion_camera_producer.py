"""Actual numeric projection/camera production over synthetic native evidence.

No decoder/model or real-media quality claim. The source-changing paths are
real production DP/projection paths, not monkeypatched fitting/selection.
"""
from dataclasses import replace
from fractions import Fraction
import json

import pytest

from crosscut.group_native_motion_camera_producer import produce_two_sided_native_cameras
from crosscut.group_native_motion_programme import NativeMotionSide,realize_native_motion_programme
from crosscut.group_source_changing_owner import NativeMotionDemand,NativeMotionSupply,NativeMotionInventory,NativeMotionViews
from crosscut.group_scene_epoch_program import compile_common_clock_camera_queries
from crosscut.group_scene_owned_program import freeze_owned_scene_program
from crosscut.group_scene_focal_pose_motion import restore_native_row
from crosscut.group_performer_family_owner import digest
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from tests.test_normal_group_motion_views import world
from tests.test_scene_role_epoch_program import fixture
from tests.test_group_review_recorded_index import native_record


def inputs(*,missing_tick=None):
    context,domain,_=fixture(native_fps=Fraction(30))
    context.incumbent.face_match_plan=None
    original=freeze_owned_scene_program(context,domain)
    sides=(NativeMotionSide(30,45,2),NativeMotionSide(45,60,3))
    numeric=realize_native_motion_programme(original=context.incumbent.render_plan,domain=domain,sides=sides)
    def queries(plan):
        compiled=compile_common_clock_camera_queries(context=context,domain=domain,original_program=original,
            candidate_plan=plan,output_intervals=((27,63),))
        return {t:q for q in compiled.queries for t in range(q.first_output_frame,q.end_output_frame)}
    before,proposed=queries(context.incumbent.render_plan),queries(numeric.render_plan)
    supplies={}
    for query_set in (before,proposed):
        for tick,q in query_set.items():
            raw,app=native_record()
            source=domain.request_index(q.eligible_source_index);native=q.sample_at(tick)[2]
            raw=restore_native_row(raw)
            if source==3:
                head=raw.heads[0];center=(.45,.3)
                points=tuple((center[0]+(x-center[0])*.625,center[1]+(y-center[1])*.625)
                    for x,y in head.landmarks_xy)
                raw=replace(raw,heads=(replace(head,box_xywh=(.41875,.2375,.0625,.125),landmarks_xy=points),))
            raw=replace(raw,request_source_index=source,native_frame_index=native,source_clock_identity=q.source_clock_identity)
            payload=json.loads(json.dumps(raw.to_dict()))
            app.update({k:payload[k] for k in ('request_source_index','native_frame_index','source_clock_identity',
                'source_file_identity','native_pixel_sha256')})
            app['recorded_native_observation_sha256']=digest(payload)
            demand=NativeMotionDemand(source,native,tick,'source-'+str(source),'moving_closeup',require_appearance=True)
            supplies[demand.native_key]=NativeMotionSupply(demand,None if tick==missing_tick else raw,
                None if tick==missing_tick else app)
    model_owner=world()[0].sensors
    views=NativeMotionViews(model_owner,NativeMotionInventory(tuple(supplies.values()),(),True,'COMPLETED_ONCE'))
    actors,receipt=views.freeze_original_actors(before,tuple(before),original_program_identity=original.program_sha256)
    paths=[];path_receipts=[]
    for side in sides:
        ticks=tuple(range(side.first_output_frame,side.end_output_frame))
        anchors=views.candidates(proposed[ticks[0]],ticks[0],association_ready=True)
        path,proof=views.connected_actor_views(proposed,ticks,None if not anchors else anchors[0])
        paths.append(path);path_receipts.append(proof)
    return dict(context=context,domain=domain,views=views,original_program=original,original_actors=actors,
        original_actor_receipt=receipt,sides=sides,side_paths=tuple(paths),side_path_receipts=tuple(path_receipts))


def test_actual_source_changing_camera_does_not_reselect_the_actor_or_original_donor_schedule():
    args=inputs()
    candidates,receipt=produce_two_sided_native_cameras(**args)
    assert candidates and receipt['state']=='REALIZED'
    before=args['context'].incumbent.render_plan
    original=native_frame_clocks_for_output(before,args['domain'].clocks_by_request,range(before.output_frame_count))
    for candidate in candidates:
        proof=candidate.evidence
        assert proof['state']=='REALIZED_FOR_REVIEW'
        assert not proof['actor_reselected_after_camera_fit'] and not proof['camera_refit_after_selection']
        assert len(proof['actual_incidents'])==3 and len(proof['actual_holds'])==2
        assert len(proof['measured_keep_incidents'])==3
        assert proof['actual_holds'][0]['original_unknown_requests']
        assert not proof['quality_approved']
        assert all(i['face_position_error']<1e-8 and i['face_scale_log_error']<1e-8
            for i in proof['actual_incidents'])
        actual=native_frame_clocks_for_output(candidate.render_plan,args['domain'].clocks_by_request,
            range(before.output_frame_count))
        for tick in (*range(30),*range(60,before.output_frame_count)):
            assert actual[tick].native_frame_index==original[tick].native_frame_index
            assert actual[tick].source_index==original[tick].source_index
        assert {actual[t].source_index for t in range(45,60)}=={3}
        for shot in candidate.selected_shots:
            assert len(shot.actor_frames)==shot.end_output_frame-shot.first_output_frame
            assert all(f.frame_local_head_id==7 and f.native_observation_sha256 for f in shot.actor_frames)
        assert candidate.selected_shots[1].request_source_index==3
        assert args['views'].sensors.session.budget.counts['model_calls']==0


def test_partial_or_unconnected_actor_address_cannot_enter_camera_fit():
    args=inputs()
    shortened=dict(args['side_paths'][0]);shortened.pop(32)
    with pytest.raises(ValueError,match='unconnected or partially addressed'):
        produce_two_sided_native_cameras(**dict(args,side_paths=(shortened,args['side_paths'][1])))
    contaminated=dict(args['side_paths'][0]);contaminated[32]=dict(contaminated[32],source_actor_path_identity='a'*64)
    with pytest.raises(ValueError,match='unconnected or partially addressed'):
        produce_two_sided_native_cameras(**dict(args,side_paths=(contaminated,args['side_paths'][1])))


def test_unknown_interior_actor_survives_as_missing_without_camera_reacquisition():
    args=inputs(missing_tick=51)
    candidates,_=produce_two_sided_native_cameras(**args)
    assert candidates
    for candidate in candidates:
        frame=next(f for f in candidate.selected_shots[1].actor_frames if f.output_tick==51)
        assert frame.frame_local_head_id is None and frame.native_observation_sha256 is None
        assert 6 in candidate.evidence['whole_camera_receipts'][1]['unmeasured_support_frame_indices']


def test_original_actor_selection_must_be_independent_of_donor_costs():
    args=inputs()
    with pytest.raises(ValueError,match='independent original'):
        produce_two_sided_native_cameras(**dict(args,original_actor_receipt=dict(
            args['original_actor_receipt'],donor_costs_used=True)))


def test_independent_original_from_a_wider_query_scope_is_rebound_without_reselection():
    args=inputs()
    compiled=compile_common_clock_camera_queries(context=args['context'],domain=args['domain'],
        original_program=args['original_program'],candidate_plan=args['context'].incumbent.render_plan,
        output_intervals=((0,120),))
    qs={t:q for q in compiled.queries for t in range(q.first_output_frame,q.end_output_frame)}
    actors=args['views'].reproject_selected_actor_path(qs,args['original_actors'])
    candidates,_=produce_two_sided_native_cameras(**dict(args,original_actors=actors))
    assert candidates and all(not c.evidence['actor_reselected_after_camera_fit'] for c in candidates)


def test_native_articulation_uses_actual_raster_aspect():
    from crosscut.group_review_two_shot_camera import measure_review_partial_joint_trajectory
    left={0:(.4,.5),1:(.6,.5),2:(.3,.2),3:(.7,.2)}
    right={j:(x,y/2) for j,(x,y) in left.items()}
    same=measure_review_partial_joint_trajectory([left]*3,[right]*3,ticks=(0,1,2),fps=30.,
        left_aspects=(.5,)*3,right_aspects=(1.,)*3)
    assert same['observed_subset_position_error']==pytest.approx(0.)
    with pytest.raises(ValueError,match='actual raster aspect'):
        measure_review_partial_joint_trajectory([left]*3,[right]*3,ticks=(0,1,2),fps=30.,left_aspects=(.5,))


def test_member_hints_are_an_additional_conflict_check_not_a_named_actor_certificate():
    from crosscut.group_native_motion_camera_producer import _appearance_incidents
    from tests.test_ive_local_arcface_gallery import gallery,vector
    from crosscut.native_face_descriptor import NativeFaceDescriptorSpace,ARCFACE_MODEL_NAME,ARCFACE_R50_SHA256
    space=NativeFaceDescriptorSpace(ARCFACE_MODEL_NAME,512,ARCFACE_R50_SHA256,'e'*64)
    bank=gallery()
    left=dict(vector=[1.]+[0.]*127,member_hint=bank.classify(vector(0),descriptor_space=space))
    right=dict(vector=[1.]+[0.]*127,member_hint=bank.classify(vector(1),descriptor_space=space))
    selected={t:left for t in range(27,63)};selected.update({t:right for t in range(45,60)})
    evidence=_appearance_incidents(selected,30,45,60)
    assert [r['local_member_hint_conflict'] for r in evidence]==[False,True,True]
    assert all(not r['named_actor_certified'] and not r['quality_approved'] for r in evidence)
    selected[45]=dict(right,member_hint=None)
    assert not _appearance_incidents(selected,30,45,60)[1]['local_member_hint_conflict']
