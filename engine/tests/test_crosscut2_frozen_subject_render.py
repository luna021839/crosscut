from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from render_frozen_subject_programme import validate_camera_clock
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
