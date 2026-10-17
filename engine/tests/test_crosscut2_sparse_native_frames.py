from fractions import Fraction
from pathlib import Path
import shutil,subprocess
import numpy as np
import pytest
from crosscut2 import sparse_native_frames as sparse
from crosscut2.supplementary_pose_bank import _frames as sequential


def clock():return sparse.SeekClock(Fraction(30),1000,(0,60,120,180,240,300,360,420,480,540,600,660,720,780,840,900,960),(), 'a'*64)


def test_sparse_windows_include_only_bounded_local_groups_and_known_preroll():
    plan=sparse.windows([2,3,699,700,998],clock())
    assert len(plan)==3 and plan[1]==(697,702,660,(699,700))
    assert all(last-key+1<=240 for first,last,key,group in plan)
    assert sum(last-key+1 for first,last,key,group in plan)<200


def test_unproven_long_gop_cannot_be_advertised_as_local_bounded_decode():
    c=sparse.SeekClock(Fraction(30),1000,(0,),(),'a'*64)
    with pytest.raises(sparse.UnsupportedSeek,match='preroll'):sparse.windows([900],c)
    for values in ([3,2],[2,2],[-1],[1000],[]):
        with pytest.raises(ValueError):sparse.windows(values,clock())


def test_explicit_preroll_budget_can_admit_a_verified_long_gop_without_a_fallback():
    c=sparse.SeekClock(Fraction(30),1000,(0,),(),'a'*64)
    with pytest.raises(sparse.UnsupportedSeek,match='preroll'):sparse.windows([270,271],c)
    actual=sparse.windows([270,271],c,maximum_decode_seconds=12.)
    assert actual==((268,273,0,(270,271)),)
    # The requested frame identities and known IDR stay exact. Raising this
    # explicit planning bound neither changes the clock nor scans blindly.
    with pytest.raises(sparse.UnsupportedSeek):sparse.windows([900],c,maximum_decode_seconds=12.)


def test_actual_reader_accepts_explicit_long_preroll_and_matches_sequential_pixels(tmp_path):
    encoder=shutil.which('ffmpeg')
    if not encoder:pytest.skip('existing FFmpeg required')
    source=tmp_path/'long-idr-gap.mp4'
    subprocess.run([encoder,'-nostdin','-v','error','-f','lavfi','-i',
        'testsrc2=size=160x90:rate=30','-frames:v','330','-an','-c:v','libx264',
        '-bf','3','-g','330','-keyint_min','330','-sc_threshold','0','-pix_fmt','yuv420p',str(source)],check=True)
    c=sparse.probe(source);wanted=[257,258];receipt={}
    with pytest.raises(sparse.UnsupportedSeek,match='preroll'):
        list(sparse.frames(source,wanted,clock=c))
    actual=dict(sparse.frames(source,wanted,clock=c,maximum_decode_seconds=9.,receipt=receipt))
    reference=dict(sequential(source,wanted,check=lambda:None))
    for tick in wanted:np.testing.assert_array_equal(actual[tick],reference[tick])
    assert receipt['maximum_declared_preroll_seconds']==9.
    assert receipt['maximum_window_presentation_span_frames']>240


def test_AVC_open_GOP_key_picture_is_not_a_random_access_IDR():
    import io
    packet=lambda kinds:b''.join((2).to_bytes(4,'big')+bytes([kind,0]) for kind in kinds)
    assert sparse.avc_idr_packet(packet([6,5]),4)
    assert not sparse.avc_idr_packet(packet([6,1]),4)
    for kinds in ([6,5],[6,1],[9,6,5]):
        raw=packet(kinds);idr,read=sparse.avc_idr_headers(io.BytesIO(raw),0,len(raw),4)
        assert idr==sparse.avc_idr_packet(raw,4) and read==len(kinds)*5
    with pytest.raises(sparse.UnsupportedSeek):sparse.avc_idr_packet(b'\x00\x00\x00\x10\x65',4)


def test_IDR_inspection_skips_large_payload_without_spending_packet_read_budget():
    import io
    raw=(2_000_000).to_bytes(4,'big')+b'\x65'+b'\x00'*1_999_999
    idr,read=sparse.avc_idr_headers(io.BytesIO(raw),0,len(raw),4)
    assert idr and read==5


def test_dense_requested_frames_are_split_into_bounded_validation_buffers():
    plan=sparse.windows(list(range(100)),clock())
    assert all(len(group)<=32 for first,last,key,group in plan)
    assert [n for first,last,key,group in plan for n in group]==list(range(100))


def test_decoder_error_window_cannot_leak_pixels_to_model_or_cache(tmp_path,monkeypatch):
    import io
    from types import SimpleNamespace
    source=tmp_path/'bound';source.write_bytes(b'target')
    c=sparse.SeekClock(Fraction(30),100,(0,),sparse.file_identity(source),'a'*64)
    def fake(args,stdout,stderr):
        stderr.write(b'mmco: incomplete reference');stderr.flush()
        return SimpleNamespace(stdout=io.BytesIO(),wait=lambda timeout:0,poll=lambda:0)
    monkeypatch.setattr(sparse.subprocess,'Popen',fake)
    monkeypatch.setattr(sparse,'read_frames',lambda *a,**kw:iter([np.zeros((540,960,3),np.uint8)]))
    delivered=[]
    with pytest.raises(OSError,match='decode incomplete'):
        for row in sparse.frames(source,[1],clock=c):delivered.append(row)
    assert delivered==[]


@pytest.mark.parametrize('codec,rate,extra',[
    ('libx264','30000/1001',['-bf','3','-g','30']),
    ('libvpx-vp9','30',['-deadline','realtime','-cpu-used','8','-g','30']),
    ('libaom-av1','30',['-cpu-used','8','-g','30'])])
def test_real_late_frame_native_rasters_equal_canonical_sequential_decoder(tmp_path,codec,rate,extra):
    encoder=shutil.which('ffmpeg')
    if not encoder:pytest.skip('existing FFmpeg required')
    source=tmp_path/(codec+'.mp4')
    process=subprocess.run([encoder,'-nostdin','-v','error','-f','lavfi','-i',
        f'testsrc2=size=160x90:rate={rate}','-frames:v','150','-an','-c:v',codec,*extra,
        '-pix_fmt','yuv420p',str(source)],capture_output=True)
    if process.returncode and b'Unknown encoder' in process.stderr:pytest.skip('optional encoder absent')
    assert process.returncode==0,process.stderr.decode()
    c=sparse.probe(source);wanted=[0,1,47,48,115,116,149];receipt={}
    actual=dict(sparse.frames(source,wanted,clock=c,receipt=receipt))
    reference=dict(sequential(source,wanted,check=lambda:None))
    assert actual.keys()==reference.keys()
    for tick in wanted:np.testing.assert_array_equal(actual[tick],reference[tick])
    assert not receipt['whole_source_sequential_strategy_requested'] and receipt['maximum_window_presentation_span_frames']<150


def test_source_identity_change_is_rejected_before_a_decoder_starts(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'changed')
    with pytest.raises(OSError,match='source changed'):
        list(sparse.frames(source,[12],clock=clock()))


def test_real_supplementary_cache_miss_uses_sparse_reader_then_warm_does_not_probe_or_decode(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from crosscut2 import supplementary_pose_bank as bank
    import hashlib
    source=tmp_path/'real-cold.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=30',
        '-frames:v','90','-an','-c:v','libx264','-bf','3','-g','30','-pix_fmt','yuv420p',str(source)],check=True)
    def forbidden(*a,**kw):raise AssertionError('real cold source must not use whole-source fallback')
    monkeypatch.setattr(bank,'_frames',forbidden)
    class Sensor:
        calls=0
        def describe(self,*args,**kw):
            self.calls+=1;return SimpleNamespace(wholebody_xyc=np.zeros((133,3),np.float32))
    sensor=Sensor();args=dict(path=source,requests=[(2,np.array([.5,.3,.2,.99])),(80,np.array([.5,.3,.2,.99]))],
        source_clock=dict(fps=30.,frame_count=90),cache=tmp_path/'cache',check=lambda:None,sensor=sensor)
    _,cold=bank.observe_requests(**args)
    assert sensor.calls==2 and cold['native_raster_execution']['strategy']=='verified_original_PTS_bounded_native_seek_v1'
    saved=Path(cold['cache_file']);before=hashlib.sha256(saved.read_bytes()).hexdigest()
    monkeypatch.setattr(sparse,'probe',forbidden);monkeypatch.setattr(sparse,'frames',forbidden)
    _,warm=bank.observe_requests(**args)
    assert warm['new_neural_calls']==0 and warm['written_checkpoints']==0 and sensor.calls==2
    assert warm['cache_file']==cold['cache_file'] and hashlib.sha256(saved.read_bytes()).hexdigest()==before


@pytest.mark.parametrize('outer_allowance',[600.,200.])
def test_decoder_budget_excludes_consumer_inference_but_outer_budget_still_applies(tmp_path,monkeypatch,outer_allowance):
    from types import SimpleNamespace
    from crosscut2 import supplementary_pose_bank as bank
    source=tmp_path/'slow-consumer.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=30',
        '-frames:v','90','-an','-c:v','libx264','-bf','3','-g','30','-pix_fmt','yuv420p',str(source)],check=True)
    now=[0.];monkeypatch.setattr(sparse,'monotonic',lambda:now[0]);monkeypatch.setattr(bank,'monotonic',lambda:now[0])
    class SlowSensor:
        def describe(self,*a,**kw):
            now[0]+=130.;return SimpleNamespace(wholebody_xyc=np.zeros((133,3),np.float32))
    args=dict(path=source,requests=[(2,np.array([.5,.3,.2,.99])),(80,np.array([.5,.3,.2,.99]))],
        source_clock=dict(fps=30.,frame_count=90),cache=tmp_path/'cache',check=lambda:None,sensor=SlowSensor(),maximum_seconds=outer_allowance)
    if outer_allowance==600.:
        _,receipt=bank.observe_requests(**args)
        assert receipt['new_neural_calls']==2 and receipt['native_raster_execution']['decoder_active_seconds']==0.
    else:
        with pytest.raises(TimeoutError,match='supplementary observation budget'):bank.observe_requests(**args)
