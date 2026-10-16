import json
import pytest
from observe_stageedit_missing_faces import pairs, choose_primary, read_journal, prepare, reusable_rows
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


def reuse_fixture(tmp_path):
    plan=dict(contract='c',gallery_identity='g',descriptor_space={'model':'r50'},
        detector_sha256='d',primary_selection='p',groups=[dict(request_file_sha256='request',
            request={'sources':[dict(source_id='s',media_content_sha256='5'*64)]},
            queries=[dict(source_id='s',native_frame=3),dict(source_id='s',native_frame=4)])])
    report=dict(current_source_contents_verified_against_request=True,teacher_labels_consumed=False,
        gallery_identity='g',descriptor_space={'model':'r50'},request_file_sha256='request',
        rows=[row(3),dict(row(4),state='FRAME_UNAVAILABLE')])
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'request-observations.json').write_text(json.dumps(report))
    return plan,report


def test_physical_native_reuse_does_not_reuse_unavailable_or_nearest_frames(tmp_path):
    plan,_=reuse_fixture(tmp_path)
    rows,receipts=reusable_rows([tmp_path],plan)
    assert list(rows)==[('s','5'*64,3)]
    assert rows['s','5'*64,3]['reused_from_observation_report_sha256']==receipts[0]['sha256']
    plan['groups'][0]['queries']=[dict(source_id='s',native_frame=2)]
    assert reusable_rows([tmp_path],plan)==({},[])


@pytest.mark.parametrize('kind',['model','labels','incomplete','foreign_media'])
def test_reuse_keeps_its_original_model_input_and_observation_scope(tmp_path,kind):
    plan,report=reuse_fixture(tmp_path)
    if kind=='model':plan=dict(plan,detector_sha256='other')
    elif kind=='labels':report['teacher_labels_consumed']=True
    elif kind=='incomplete':report['rows'].pop()
    elif kind=='foreign_media':
        plan['groups'][0]['request']['sources'][0]['media_content_sha256']='9'*64
        assert reusable_rows([tmp_path],plan)==({},[])
        return
    (tmp_path/'request-observations.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):reusable_rows([tmp_path],plan)
