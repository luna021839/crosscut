import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from crosscut2 import torch_person_observations as people
from crosscut2.grid_perception import file_identity


def world(tmp_path):
    source=tmp_path/'source.mp4';source.write_bytes(b'exact original source')
    bound=dict(source=0,source_file_identity=list(file_identity(source)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),fps=30.,frame_count=300)
    record=dict(contract=people.CONTRACT,implementation_sha256=people._implementation(),
        weights_sha256=people.WEIGHTS_SHA256,source_sha256=bound['source_sha256'],
        source_file_identity=bound['source_file_identity'],fps=30.,frame_count=300,raster=[640,360],
        raster_semantics='original_normalized_xyxy_score',backend='torch_MPS_float32',maximum_side=1280,
        confidence=.35,nms_iou=.55,rectangles={'20':[[.1,.1,.7,.8,.9]],'21':[]})
    cache=tmp_path/'cache';cache.mkdir()
    path=cache/(bound['source_sha256']+'.json');path.write_text(json.dumps(record))
    return source,bound,cache,path,record


def test_warm_person_bank_needs_no_model_or_producer_and_keeps_empty_separate(tmp_path,monkeypatch):
    source,bound,cache,path,_=world(tmp_path);before=path.read_bytes()
    def forbidden(*a,**kw):raise AssertionError('warm bank must not infer, probe or decode')
    monkeypatch.setattr(people,'_detector',forbidden)
    monkeypatch.setattr(people,'_raster',forbidden)
    monkeypatch.setattr(people.sparse,'probe',forbidden)
    monkeypatch.setattr(people.sparse,'frames',forbidden)
    values,receipt=people.observe(sources=[source],bound=[bound],requests=[[20,21]],cache=cache,
        weights=tmp_path/'missing-model.pt',maximum_new_frames=0)
    assert set(values[0])=={20,21} and values[0][21].shape==(0,5)
    assert 22 not in values[0] and receipt['new_person_forwards']==receipt['source_rasters']==0
    assert receipt['complete'] and path.read_bytes()==before


@pytest.mark.parametrize('key,value',[
    ('weights_sha256','different'),('source_file_identity',[0,1,2,3]),('fps',60.),
    ('raster_semantics','scaled'),('backend','onnx'),('implementation_sha256','different'),
    ('confidence',.1),('nms_iou',.8),
])
def test_changed_person_provenance_is_not_silently_relabelled(tmp_path,key,value):
    source,bound,cache,path,record=world(tmp_path);record[key]=value;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='observation changed'):
        people.load(sources=[source],bound=[bound],cache=cache)


def test_complete_missing_domain_is_preflighted_before_model_construction(tmp_path,monkeypatch):
    source,bound,cache,_,_=world(tmp_path)
    monkeypatch.setattr(people,'_detector',lambda *a:pytest.fail('must preflight before inference'))
    with pytest.raises(ValueError,match='needs 1 new frames'):
        people.observe(sources=[source],bound=[bound],requests=[[20,22]],cache=cache,
            weights=tmp_path/'absent',maximum_new_frames=0)
    with pytest.raises(ValueError,match='exact source/native'):
        people.observe(sources=[source],bound=[bound],requests=[[20,20]],cache=cache,
            weights=tmp_path/'absent',maximum_new_frames=2)


def test_ssd_remount_reads_original_producer_after_content_verification_without_rewriting(tmp_path):
    source,bound,cache,path,record=world(tmp_path)
    record['source_file_identity'][0]+=1;path.write_text(json.dumps(record));before=path.read_bytes()
    result=people.load(sources=[source],bound=[bound],cache=cache)
    assert set(result[0])=={20,21} and path.read_bytes()==before
    record['source_sha256']=bound['source_sha256']='f'*64;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='remounted person source content changed'):
        people.load(sources=[source],bound=[bound],cache=cache)


def test_source_contents_are_checked_even_before_a_warm_return(tmp_path):
    source,bound,cache,_,_=world(tmp_path);bound['source_sha256']='f'*64
    with pytest.raises(ValueError,match='source content changed'):
        people.observe(sources=[source],bound=[bound],requests=[[20]],cache=cache,
            weights=tmp_path/'absent',maximum_new_frames=0)


def test_conflicting_snapshots_and_unphysical_boxes_fail_not_arbitrary_selection(tmp_path):
    source,bound,cache,path,record=world(tmp_path)
    record['rectangles']['20'][0][2]=.8
    second=cache/'other.json';second.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='conflicting'):
        people.load(sources=[source],bound=[bound],cache=cache)
    second.unlink();record['rectangles']['20'][0][2]=1.1;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='malformed normalized'):
        people.load(sources=[source],bound=[bound],cache=cache)
