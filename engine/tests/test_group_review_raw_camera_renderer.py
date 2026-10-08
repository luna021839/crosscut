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
from crosscut.group_source_changing_owner import project_point


def setup(fps=30.):
    context,domain=fixture(fps=fps);old=context.incumbent.render_plan
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


def test_device_remount_reuses_only_exact_decoded_native_pixels_without_rewriting_archive():
    plan,queries,session,row=setup()
    session._assert_source_identity=lambda source:(99,2,3,4,5)
    _,receipt=render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
        session=session,native_row_for_query=row,expected_program_identity=digest(plan.to_json()))
    assert all(b['recorded_source_file_identity']==[1,2,3,4,5] and b['current_source_file_identity']==[99,2,3,4,5]
        and b['file_binding_basis']=='device_remount_and_exact_native_pixel_hash' for b in receipt['native_bindings'])
    def wrong(q,t):
        result=row(q,t);result['native_observation']['native_pixel_sha256']='a'*64;return result
    with pytest.raises(ValueError,match='measured parent'):
        render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
            session=session,native_row_for_query=wrong,expected_program_identity=digest(plan.to_json()))
    session._assert_source_identity=lambda source:(99,2,4,4,5)
    with pytest.raises(ValueError,match='source/frame/file/clock'):
        render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
            session=session,native_row_for_query=row,expected_program_identity=digest(plan.to_json()))


def test_post_selection_program_or_camera_slicing_cannot_be_hidden():
    plan,queries,session,row=setup()
    with pytest.raises(ValueError,match='selected program'):
        render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
            session=session,native_row_for_query=row,expected_program_identity='b'*64)
    with pytest.raises(ValueError,match='post-select slice'):
        render_selected_review_rasters(plan=plan,queries={t:q for t,q in queries.items() if t<19},changed_intervals=((10,19),),
            session=session,native_row_for_query=row,expected_program_identity=digest(plan.to_json()))


@pytest.mark.parametrize('fps',(30.,30000/1001))
def test_distinct_canonical_rasters_keep_every_ordinal_and_replay_identically(fps):
    plan,queries,session,row=setup(fps)
    indices=[queries[t].sample_at(t)[2] for t in range(10,20)]
    ordinal={n:i for i,n in enumerate(indices)}
    assert len(ordinal)==10
    session.native_pixels=lambda source,native:np.full((180,320,3),20+ordinal[native]*20,dtype=np.uint8)
    a,ra=render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
        session=session,native_row_for_query=row,expected_program_identity=digest(plan.to_json()))
    b,rb=render_selected_review_rasters(plan=plan,queries=queries,changed_intervals=((10,20),),
        session=session,native_row_for_query=row,expected_program_identity=digest(plan.to_json()))
    values=[float(a.frame(t).mean()) for t in range(10,20)]
    assert all(abs(value-(20+i*20))<=3 for i,value in enumerate(values))
    assert a.identity==b.identity and ra['output_frames']==10 and not ra['quality_approved']


@pytest.mark.parametrize('roll',(0.,1.3,-1.3))
def test_registered_dynamic_camera_points_agree_with_actual_ffmpeg_raster_centroids(roll):
    # Independent geometric oracle: colored native markers go through the
    # production FFmpeg filters, then their centroids are located from pixels.
    # A repeated renderer hash is NOT this projection-parity check.
    import cv2
    context,domain=fixture();old=context.incumbent.render_plan
    camera=EditorialTransformTrack((
        EditorialTransformKeyframe(0.,1.25,.48,.46,roll),
        EditorialTransformKeyframe(9/old.fps,1.65,.53,.54,roll)),reason='explicit_nonmedia_camera_parity')
    plan=realize_interval_in_plan(old,10,20,camera=camera)
    queries=compile_common_clock_camera_queries(context=context,domain=domain,
        original_program=freeze_owned_scene_program(context,domain),candidate_plan=plan,
        output_intervals=((10,20),)).queries
    mapping={t:q for q in queries for t in range(q.first_output_frame,q.end_output_frame)}
    markers=((.42,.43),(.59,.44),(.48,.62));colors=((0,0,255),(0,255,0),(255,0,0))
    native=np.zeros((720,1280,3),dtype=np.uint8)
    for p,c in zip(markers,colors):cv2.circle(native,(round(p[0]*1280),round(p[1]*720)),9,c,-1)
    class Session:
        def __init__(self):self.domain=domain
        def _assert_source_identity(self,source):return (1,2,3,4,5)
        def native_pixels(self,source,index):return native
    session=Session()
    def row(q,t):
        return dict(output_tick=t,native_observation=dict(request_source_index=0,
            native_frame_index=q.sample_at(t)[2],source_clock_identity=q.source_clock_identity,
            source_file_identity=[1,2,3,4,5],native_pixel_sha256=hashlib.sha256(memoryview(native)).hexdigest()))
    output,_=render_selected_review_rasters(plan=plan,queries=mapping,changed_intervals=((10,20),),
        session=session,native_row_for_query=row,expected_program_identity=digest(plan.to_json()))
    deviations=[]
    for t in range(10,20):
        frame=output.frame(t).astype(float)
        for p,c in zip(markers,colors):
            channel=int(np.argmax(c));others=[i for i in range(3) if i!=channel]
            weight=np.maximum(0.,frame[:,:,channel]-np.maximum(frame[:,:,others[0]],frame[:,:,others[1]])-30.)
            assert weight.sum()>100.,'actual projected marker disappeared'
            y,x=np.indices(weight.shape);center=(float((x*weight).sum()/weight.sum()),float((y*weight).sum()/weight.sum()))
            expected=project_point(p,mapping[t],t)
            # Working-canvas and chroma rounding are bounded independently;
            # pan/zoom/time/roll sign errors cannot hide within this tolerance.
            wanted=(expected[0]*plan.width,expected[1]*plan.height)
            deviations.append(dict(tick=t,marker=p,actual=center,expected=wanted,
                error=float(np.linalg.norm(np.asarray(center)-wanted))))
    assert max(d['error'] for d in deviations)<=2., deviations
