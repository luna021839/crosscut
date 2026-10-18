import copy,hashlib,json
from dataclasses import replace
from pathlib import Path

import pytest
from crosscut2.member_stage_mix import Source
from crosscut2.member_wardrobe_finish import binding,signature,match,validate
from crosscut.render import NormalizedCrop
from test_crosscut2_member_programme import inputs


def cached(tmp_path):
    sources,plan,_=inputs()
    sources=[replace(s,path=tmp_path/s.path.name) for s in sources]
    for source in sources:source.path.write_bytes(b'physical original')
    plan=replace(plan,sources=tuple(s.path for s in sources))
    value=binding(plan,sources,0,frame_count=7,lead_frames=3)
    movie=tmp_path/'handle.mp4';movie.write_bytes(b'previously measured and encoded handle')
    handle=dict(binding=value,binding_sha256=signature(value),movie=str(movie),
        movie_sha256=hashlib.sha256(movie.read_bytes()).hexdigest())
    return sources,plan,handle


def test_handles_match_physical_crops_and_clocks_not_old_cut_indices(tmp_path):
    sources,plan,handle=cached(tmp_path)
    chosen,report=match(plan,sources,[handle])
    assert report['matched_windows']==1 and chosen[0]['first_frame']==27
    assert chosen[0]['frame_count']==7 and report['source_NN']==0
    changed=replace(plan,segments=(replace(plan.segments[0],crop=NormalizedCrop(.1,.1,.8,.8)),plan.segments[1]))
    assert match(changed,sources,[handle])[1]['matched_windows']==0
    changed=replace(plan,segments=(replace(plan.segments[0],source_start=.1),plan.segments[1]))
    assert match(changed,sources,[handle])[1]['matched_windows']==0


def test_changed_pixels_or_physical_source_cannot_reuse_an_old_handle(tmp_path):
    sources,plan,handle=cached(tmp_path)
    Path(handle['movie']).write_bytes(b'changed effect pixels')
    with pytest.raises(ValueError,match='movie changed'):match(plan,sources,[handle])
    sources[0].path.write_bytes(b'changed original')
    assert match(plan,sources,[handle])[1]['matched_windows']==0
    changed=copy.deepcopy(handle);changed['binding']['fps']=60
    with pytest.raises(ValueError,match='binding changed'):match(plan,sources,[changed])


def test_same_sources_reordered_keep_the_same_physical_handle(tmp_path):
    sources,plan,handle=cached(tmp_path)
    sources.reverse()
    plan=replace(plan,sources=tuple(s.path for s in sources),
        segments=tuple(replace(s,source_index=1-s.source_index) for s in plan.segments))
    assert match(plan,sources,[handle])[1]['matched_windows']==1


def test_post_effect_receipt_does_not_pretend_the_nominal_plan_changed(tmp_path):
    base=tmp_path/'base.mp4';base.write_bytes(b'original complete film')
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    plan=dict(output=str(base),output_frame_count=60)
    assert validate({},plan)==base
    with pytest.raises(ValueError,match='lacks its effect receipt'):
        validate(dict(base_video_path=str(base)),plan)
    handles=tmp_path/'handles.json';handles.write_text(json.dumps(dict(handles=[dict(movie='patch.mp4',
        movie_sha256='e'*64,binding_sha256='c'*64)])))
    inputs=dict(path=str(handles),sha256=digest(handles.read_bytes()))
    value=dict(contract='member_spatial_garment_first_post_effect_v1',base_movie=str(base),
        base_movie_sha256=digest(base.read_bytes()),output='final.mp4',output_sha256='a'*64,
        output_frames=60,audio_packet_payloads_and_timestamps_preserved=True,full_AV_decode_exit_code=0,
        effects=[dict(movie='patch.mp4',sha256='e'*64,binding_sha256='c'*64)])
    receipt=tmp_path/'effect.json';receipt.write_text(json.dumps(value))
    rendering=dict(wardrobe_receipt=str(receipt),wardrobe_receipt_sha256=digest(receipt.read_bytes()),
        base_video_path=str(base),base_movie_sha256=value['base_movie_sha256'],video_path='final.mp4',movie_sha256='a'*64)
    assert validate(rendering,plan,inputs)==base
    base.write_bytes(b'changed base movie')
    with pytest.raises(ValueError,match='base film changed'):validate(rendering,plan,inputs)
