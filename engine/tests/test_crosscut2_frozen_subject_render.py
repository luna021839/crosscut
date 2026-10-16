from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from render_frozen_subject_programme import validate_camera_clock,materialize_scored_cameras,camera_values
from crosscut.render import (RenderPlan,RenderSegment,AspectRatio,
    EditorialTransformTrack,EditorialTransformKeyframe)
from crosscut2.subject_moment_sequence import curve_digest


def fixture():
    plan=RenderPlan((Path('source.mp4'),),
        (RenderSegment(0,2.,1.,0.),RenderSegment(0,3.,1.,1.)),0,
        AspectRatio.LANDSCAPE,Path('output.mp4'),fps=30.)
    camera=curve_digest(np.tile([1.,.5,.5],(30,1)))
    proof=[dict(source=0,first_frame=k*30,final_frame=(k+1)*30,
        native_first=60+k*30,native_last=89+k*30,scored_camera_sha256=camera) for k in range(2)]
    return plan,proof,[dict(fps=30.,frame_count=300,native_cuts=[75,100])]


def test_saved_subject_programme_checks_every_camera_and_exact_native_endpoints_without_reselection():
    plan,proof,sources=fixture()
    result=validate_camera_clock(plan,proof,sources)
    assert result['output_frame_count']==60 and result['evaluated_boundary_count']==3
    assert result['cuts']==[dict(frame=15,origins=['retained_native']),
        dict(frame=30,origins=['authored']),dict(frame=40,origins=['retained_native'])]


@pytest.mark.parametrize('key,value,message',[
    ('scored_camera_sha256','0'*64,'camera changed'),
    ('native_first',61,'native clock changed'),
    ('final_frame',31,'interval changed')])
def test_saved_programme_mutations_fail_before_encoding(key,value,message):
    plan,proof,sources=fixture();proof=deepcopy(proof);proof[0][key]=value
    with pytest.raises(ValueError,match=message):validate_camera_clock(plan,proof,sources)


def test_unscored_rotation_is_not_ignored_by_the_camera_hash():
    plan,proof,sources=fixture()
    track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1,.5,.5,1.),))
    plan=replace(plan,segments=(replace(plan.segments[0],editorial_transform=track),plan.segments[1]))
    with pytest.raises(ValueError,match='unscored rotation'):validate_camera_clock(plan,proof,sources)


def test_global_camera_receipt_uses_the_preselected_full_frame_hash_not_a_new_fit():
    plan,proof,sources=fixture()
    global_proof=[dict(source=row['source'],first_frame=row['first_frame'],final_frame=row['final_frame'],
        camera_path_sha256=row['scored_camera_sha256'],per_frame_camera_count=30) for row in proof]
    assert validate_camera_clock(plan,global_proof,sources)['output_frame_count']==60
    global_proof[0]['per_frame_camera_count']=29
    with pytest.raises(ValueError,match='incomplete global camera proof'):
        validate_camera_clock(plan,global_proof,sources)
    del proof[0]['native_first']
    with pytest.raises(ValueError,match='incomplete subject'):
        validate_camera_clock(plan,proof,sources)


def test_compact_roundoff_recovers_exact_original_hash_without_selection_or_fitting():
    import hashlib
    n=70;curve=np.tile([1.2,.5,.5],(n,1));compact=curve.copy()
    compact[10,1]=np.nextafter(compact[10,1],1.)
    track=EditorialTransformTrack(tuple(EditorialTransformKeyframe(k/30.,*row) for k,row in enumerate(compact)))
    plan=RenderPlan((Path('source.mp4'),),(RenderSegment(0,2.,n/30.,0.,editorial_transform=track),),
        0,AspectRatio.LANDSCAPE,Path('output.mp4'),fps=30.)
    row=dict(source=0,first_frame=0,final_frame=n,camera_arc_code=1,per_frame_camera_count=n,
        camera_path_sha256=curve_digest(curve),entry_geometry_hash=hashlib.sha256(curve[:7].tobytes()).hexdigest(),
        exit_geometry_hash=hashlib.sha256(curve[-7:].tobytes()).hexdigest())
    record=dict(fixed_actor_camera_hold_receipt=dict(frozen_selected_camera_arcs=[row]),seams=[])
    with pytest.raises(ValueError,match='camera changed'):validate_camera_clock(plan,[row],[dict(fps=30.,frame_count=300,native_cuts=[])])
    exact,receipt=materialize_scored_cameras(plan,record)
    np.testing.assert_array_equal(camera_values(exact.segments[0],n,30.),curve)
    assert receipt[0]['maximum_difference']<=np.finfo(float).eps
    assert not receipt[0]['source_or_cut_or_camera_choice_changed']
    assert validate_camera_clock(exact,[row],[dict(fps=30.,frame_count=300,native_cuts=[])])['output_frame_count']==n
    # Actual editing mutations and forged hashes are NOT repaired/accepted.
    larger=replace(plan.segments[0],editorial_transform=EditorialTransformTrack(tuple(
        replace(k,center_x=k.center_x+.001) if i==10 else k for i,k in enumerate(track.keyframes))))
    with pytest.raises(ValueError,match='more than replay roundoff'):
        materialize_scored_cameras(replace(plan,segments=(larger,)),record)
    bad=deepcopy(record);bad['fixed_actor_camera_hold_receipt']['frozen_selected_camera_arcs'][0]['camera_path_sha256']='0'*64
    with pytest.raises(ValueError,match='exact frozen camera hash'):materialize_scored_cameras(plan,bad)
