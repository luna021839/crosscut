import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest

from crosscut.ive import IVEMember
from crosscut2 import recurrent_member_hints as member
from crosscut2.source_member_scores import load_programme_endpoint_requests
from crosscut2.grid_perception import file_identity
from test_crosscut2_event_holds import inputs


def test_additional_exact_queries_are_classified_once_without_relabelling_clusters(tmp_path,monkeypatch):
    _,banks,_,_=inputs(count=2);bank=banks[0]
    source=tmp_path/'source.mp4';source.write_bytes(b'source-identity')
    models=tmp_path/'models';models.mkdir();(models/'face_detection_yunet_2023mar.onnx').write_bytes(b'detector')
    queries=[(int(bank['native_indices'][n,1]),bank['heads'][n,1,0]) for n in (0,1)]
    names=tuple(m.value for m in IVEMember);calls=[];decoded=[]
    class Gallery:
        identity='gallery';minimum_similarity=.45;minimum_member_margin=.08
        def classify(self,value,**kw):
            scores=dict.fromkeys(names,.1);scores[names[0]]=.6
            return dict(state='LIKELY_MEMBER',member=names[0],member_scores=scores)
    class Arc:
        def __init__(self,*a):pass
        def embedding(self,*a,**kw):
            calls.append(kw);v=np.zeros(512);v[0]=1.;return v
    class Detector:
        def detect(self,*a):return None,np.zeros((1,15))
    monkeypatch.setattr(member.LocalIVEArcFaceGallery,'from_existing_photo_banks',lambda *a,**kw:Gallery())
    monkeypatch.setattr(member,'select_faces',lambda *a,**kw:([queries[0]],[[0]]))
    monkeypatch.setattr(member,'NativeArcFaceR50',Arc)
    monkeypatch.setattr(member.cv2,'FaceDetectorYN',SimpleNamespace(create=lambda *a:Detector()))
    face=np.array([.45*960,.2*540,.1*960,.2*540,*([.5*960,.3*540]*5),.99])
    monkeypatch.setattr(member,'associate',lambda found,head:face)
    monkeypatch.setattr(member,'valid_query_geometry',lambda *a:True)
    from crosscut2 import sparse_native_frames as sparse
    monkeypatch.setattr(sparse,'probe',lambda *a,**kw:SimpleNamespace(count=200,rate=30.))
    def frames(path,natives,**kw):
        decoded.extend(natives);kw['receipt'].update(strategy='test-exact-native')
        for native in natives:yield native,np.zeros((540,960,3),np.uint8)
    monkeypatch.setattr(sparse,'frames',frames)
    args=dict(sources=[source],banks=[bank],census=[dict(fps=30.,frame_count=200)],photo_tool=tmp_path,
        models=models,cache=tmp_path/'cache',check=lambda:None,maximum_calls=2,
        preserve_member_scores=True,additional_requests=[[queries[1],queries[1]]])
    result,receipt=member.measure(**args)
    assert len(calls)==2 and len(decoded)==2
    assert result[0].cluster_hints.tolist()==[-1]  # one original vote is still not recurrent consensus
    assert len(result[0].direct_queries)==2 and receipt[0]['additional_current_frame_queries']==1
    cached={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args['cache'].glob('*.npz')}
    calls.clear();decoded.clear()
    replay,receipt=member.measure(**dict(args,maximum_calls=0,cache_only_partial=True))
    assert not calls and not decoded and receipt[0]['new_arcface_calls']==0
    assert len(replay[0].direct_queries)==2
    assert cached=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args['cache'].glob('*.npz')}


def test_provisional_member_plan_must_match_the_real_source_and_audio_clock(tmp_path):
    nodes,banks,_,_=inputs(count=2);sources=[]
    for s in range(2):
        p=tmp_path/f'{s}.mp4';p.write_bytes(bytes([s]));sources.append(p)
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
    record=dict(source_sha256=list(hashes),source_file_identities=[list(file_identity(p)) for p in sources],
        render_plan=dict(sources=list(map(str,sources)),fps=30.,master_audio_start=2.),
        seams=[dict(output_frame=int(nodes[1]))])
    path=tmp_path/'plan.json';path.write_text(json.dumps(record))
    args=dict(nodes=nodes,banks=banks,sources=sources,source_hashes=hashes,master_start=2.,fps=30.,authorized_root=tmp_path)
    requests,receipt=load_programme_endpoint_requests(path,**args)
    assert all(requests) and len(receipt['provisional_plan_sha256'])==64
    with pytest.raises(ValueError,match='physical source/clock'):
        load_programme_endpoint_requests(path,**dict(args,master_start=3.))
    record['source_sha256'][0]='0'*64;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='physical source/clock'):load_programme_endpoint_requests(path,**args)


def test_cli_current_member_queries_need_explicit_photo_scores_before_io(monkeypatch):
    from crosscut2.stage_mix import main
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused','--cache','unused',
        '--output','unused','--models','unused','--provisional-member-observation-plan','unused'])
    with pytest.raises(ValueError,match='individual member scores and local photo assets'):main()
