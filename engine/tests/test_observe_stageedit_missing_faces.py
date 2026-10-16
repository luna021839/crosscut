import json
import pytest
from observe_stageedit_missing_faces import pairs, choose_primary, read_journal, prepare
from test_stage_edit_native_face_completion import fixture
from crosscut.stage_edit_face_cues import schema_with_face_cues


def test_query_domain_uses_every_available_snapshot_candidate_not_selected_labels(tmp_path):
    snapshot,face,raw,_=fixture()
    s=tmp_path/'snapshot.json';f=tmp_path/'faces.json';r=tmp_path/'request.json'
    s.write_text(json.dumps(snapshot.to_dict()));f.write_text(json.dumps(face.to_dict()));r.write_bytes(raw)
    groups,_=prepare({'ref':s},{'ref':f},[r],schema_with_face_cues())
    assert groups[0]['queries']==[dict(source_id='source-a',native_frame=30,master_seconds=[0.])]
    assert groups[0]['input_snapshot_sha256']==[snapshot.snapshot_sha256]


def test_input_mapping_refuses_duplicates_and_implicit_relative_paths():
    with pytest.raises(ValueError):pairs(['ref=/tmp/a','ref=/tmp/b'])
    with pytest.raises(ValueError):pairs(['ref=relative'])
    assert str(pairs(['ref=/tmp/a'])['ref'])=='/tmp/a'


def test_original_viewport_primary_selection_has_no_member_or_label_argument():
    central=[420.,200.,120.,150.];edge=[900.,200.,60.,80.]
    assert choose_primary([edge,central]) is central
    assert choose_primary(None) is None and choose_primary([]) is None


def row(native):
    return dict(source_id='s',media_content_sha256='5'*64,native_frame=native,state='NO_FACE')


def test_recovery_preserves_completed_rows_and_backs_up_only_an_incomplete_tail(tmp_path):
    p=tmp_path/'source.jsonl';good=json.dumps(row(3)).encode()+b'\n';tail=b'{"source_id":'
    p.write_bytes(good+tail)
    assert read_journal(p,dict(source_id='s',media_content_sha256='5'*64),{3,4},resume=True)=={3:row(3)}
    assert p.read_bytes()==good
    assert p.with_name(p.name+'.interrupted-tail').read_bytes()==tail


def test_complete_last_record_without_newline_is_retained_not_reinferred(tmp_path):
    p=tmp_path/'source.jsonl';p.write_text(json.dumps(row(3)))
    assert read_journal(p,dict(source_id='s',media_content_sha256='5'*64),{3},resume=True)=={3:row(3)}
    assert p.read_bytes().endswith(b'\n')


@pytest.mark.parametrize('kind',['foreign','duplicate','middle','no_resume'])
def test_journal_cannot_be_silently_rebound_or_skip_invalid_complete_rows(tmp_path,kind):
    p=tmp_path/'source.jsonl';payload=json.dumps(row(3))+'\n'
    if kind=='foreign':payload=json.dumps(dict(row(3),source_id='other'))+'\n'
    if kind=='duplicate':payload*=2
    if kind=='middle':payload+='invalid\n'+json.dumps(row(4))+'\n'
    p.write_text(payload)
    with pytest.raises((ValueError,FileExistsError)):
        read_journal(p,dict(source_id='s',media_content_sha256='5'*64),{3,4},resume=kind!='no_resume')
