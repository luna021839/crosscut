import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from crosscut.ive import IVEMember,EditMode,AspectRatio
from crosscut2.member_app_job import INPUT_CONTRACT,load_inputs,try_analyze
from crosscut2.member_geometry_inputs import identity
from crosscut2.visible_shot_duration import fit
from test_crosscut2_member_geometry_inputs import fixture as geometry_fixture


def inputs(tmp_path):
    source,ids,row,database,face=geometry_fixture(tmp_path)
    second=tmp_path/'second.mp4';second.write_bytes(b'other original, never decoded')
    second_id=identity(second)
    face=copy.deepcopy(face);face['sources'].append(dict(face['sources'][0],source=1,source_file_identity=second_id))
    faces=tmp_path/'faces.json';faces.write_text(json.dumps(face))
    model=tmp_path/'pacing.json';model.write_text(json.dumps(fit([dict(reference_id='TRAIN',split='train',
        input_sha256='a'*64,cut_seconds=[0.,1.,2.,4.])])) )
    def bind(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    entries=[dict(path=str(p),identity=bound,fps=30.,width=1080,height=1920,frame_count=100)
        for p,bound in ((source.path,ids[0]),(second,second_id))]
    value=dict(contract=INPUT_CONTRACT,member='가을',song='BLACKHOLE',reference_source=0,sources=entries,
        geometry=dict(path=str(database),namespace='frozen'),face_inputs=[bind(faces)],body_by_source=[[],[]],
        duration_model=bind(model),local_timewarps=[],master_start=0.,duration_frames=90,fps=30.,beat_times=[0.,1.,2.])
    path=tmp_path/'inputs.json';path.write_text(json.dumps(value))
    request=SimpleNamespace(target_member=IVEMember.GAEUL,song='BLACKHOLE',sources=(source.path,second),reference_source=0)
    return path,request,value


def test_original_observation_sources_can_be_reordered_without_aliasing_producers(tmp_path):
    path,request,value=inputs(tmp_path)
    request.sources=tuple(reversed(request.sources));request.reference_source=1
    warps=(object(),object())
    with patch('crosscut2.member_app_job.load_warps',return_value=warps):
        data=load_inputs(path,request)
    assert data['source_order']==[1,0] and data['warps']==[warps[1],warps[0]]
    assert data['sources'][0].path==request.sources[0]
    assert data['observations'][0][30]['face_bbox_xywh']==data['observations'][1][30]['face_bbox_xywh']
    assert data['observations'][0][30]['pose_screen_xyc']==[]  # no original body on this source
    assert data['observations'][1][30]['pose_screen_xyc']


def test_wrong_song_or_master_audio_is_a_cache_miss(tmp_path):
    path,request,_=inputs(tmp_path)
    request.song='another song'
    assert load_inputs(path,request) is None
    request.song='BLACKHOLE';request.reference_source=1
    assert load_inputs(path,request) is None


def test_changed_sources_and_face_dependencies_cannot_be_reused(tmp_path):
    path,request,value=inputs(tmp_path)
    request.sources[0].write_bytes(b'changed')
    with pytest.raises(OSError,match='source bytes'):
        load_inputs(path,request)


def test_absent_current_cache_does_not_displace_a_cold_job(tmp_path):
    request=SimpleNamespace(source_cache_directory=tmp_path,target_member=IVEMember.GAEUL,
        edit_mode=EditMode.MEMBER,aspect_ratio=AspectRatio.PORTRAIT)
    assert try_analyze(request,None) is None
