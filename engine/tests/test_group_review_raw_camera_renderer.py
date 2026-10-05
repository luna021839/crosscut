"""Explicit synthetic NONMEDIA rasters, not IVE or editorial quality proof."""
import hashlib
from dataclasses import replace
import numpy as np
import pytest
from engine.tests.test_scene_owned_program import fixture
from crosscut.group_scene_owned_program import freeze_owned_scene_program
from crosscut.group_scene_epoch_program import compile_common_clock_camera_queries
from crosscut.group_review_interval_compile import realize_interval_in_plan
from crosscut.group_review_raw_camera_renderer import render_selected_review_rasters
from crosscut.group_performer_family_owner import digest
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe


def setup():
    context,domain=fixture();old=context.incumbent.render_plan
    camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.4),),reason='preselection_only')
    plan=realize_interval_in_plan(old,10,20,camera=camera)
    queries=compile_common_clock_camera_queries(context=context,domain=domain,
        original_program=freeze_owned_scene_program(context,domain),candidate_plan=plan,
        output_intervals=((10,20),)).queries
    mapping={t:q for q in queries for t in range(q.first_output_frame,q.end_output_frame) if 10<=t<20}
    class Session:
        def __init__(self):self.domain=domain
        def _assert_source_identity(self,source):return (1,2,3,4,5)
        def native_pixels(self,source,native):
            x=np.tile(np.linspace(0,255,320,dtype=np.uint8),(180,1));return np.stack((x,x,x),axis=2)
    session=Session()
    def row(q,t):
        index=q.sample_at(t)[2];pixels=session.native_pixels(0,index)
        return dict(output_tick=t,native_observation=dict(request_source_index=0,native_frame_index=index,
            source_clock_identity=q.source_clock_identity,source_file_identity=[1,2,3,4,5],
            native_pixel_sha256=hashlib.sha256(memoryview(pixels)).hexdigest()))
    return plan,mapping,session,row


def test_selected_camera_renders_real_pixels_and_fresh_replay_is_identical():
    plan,queries,session,row=setup()
    a,ra=render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
        session=session,native_row_for_query=row,expected_program_identity=digest(plan.to_json()))
    b,rb=render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
        session=session,native_row_for_query=row,expected_program_identity=digest(plan.to_json()))
    assert a.identity==b.identity and ra['rendered_output_pixel_sha256']==rb['rendered_output_pixel_sha256']
    assert ra['output_frames']==10 and not ra['quality_approved']
    assert not np.array_equal(a.frame(10),session.native_pixels(0,queries[10].sample_at(10)[2]))
    assert a.frame(10).flags.writeable is False


def test_selected_raw_pixel_hash_cannot_borrow_a_saved_native_receipt():
    plan,queries,session,row=setup()
    def wrong(q,t):
        value=row(q,t);value['native_observation']['native_pixel_sha256']='a'*64;return value
    with pytest.raises(ValueError,match='measured parent'):
        render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
            session=session,native_row_for_query=wrong,expected_program_identity=digest(plan.to_json()))


def test_post_selection_program_or_camera_slicing_cannot_be_hidden():
    plan,queries,session,row=setup()
    with pytest.raises(ValueError,match='selected program'):
        render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
            session=session,native_row_for_query=row,expected_program_identity='b'*64)
    with pytest.raises(ValueError,match='post-select slice'):
        render_selected_review_rasters(plan=plan,queries={t:q for t,q in queries.items() if t<19},changed_intervals=((10,19),),
            session=session,native_row_for_query=row,expected_program_identity=digest(plan.to_json()))
