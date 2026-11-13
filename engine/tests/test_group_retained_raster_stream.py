"""Tiny nonhuman codec fixtures; never professional editing-quality proof."""
import subprocess
import numpy as np
import pytest
from crosscut.group_retained_raster_review import content_sha
from crosscut.group_retained_raster_stream import construct_retained_stream_movie,verify_retained_stream_recipe
from crosscut.group_retained_raster_stream import bounded_review_encoder_bytes


def fixture_movie(path,offset):
    frames=np.zeros((5,24,40,3),dtype=np.uint8)
    for n in range(5):
        frames[n,:,:,0]=n*25+offset;frames[n,:,:,1]=90+offset
        frames[n,:,:,2]=np.arange(40,dtype=np.uint8)[None,:]
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pixel_format','bgr24',
        '-video_size','40x24','-framerate','10','-i','pipe:0','-f','lavfi','-i','anullsrc=r=48000:cl=stereo',
        '-map','0:v:0','-map','1:a:0','-c:v','libx264','-pix_fmt','yuv420p','-crf','16','-c:a','aac',
        '-t','0.5',str(path)],input=frames.tobytes(),check=True,timeout=20.)


def options(tmp_path):
    a=tmp_path/'original.mp4';b=tmp_path/'proposed.mp4';fixture_movie(a,0);fixture_movie(b,17)
    return dict(reference_movie=a,proposed_movie=b,output=tmp_path/'delivery.mp4',frame_count=5,
        fps=10.,width=40,height=24,changed_intervals=((1,2),),protected_intervals=((3,5),),
        expected_reference_sha256=content_sha(a),expected_proposed_sha256=content_sha(b),
        minimum_free_bytes=1,maximum_output_bytes=1024*1024,maximum_seconds=30.)


def test_stream_recipe_is_exact_but_delivered_mp4_is_not_called_lossless(tmp_path):
    report=construct_retained_stream_movie(**options(tmp_path))
    verified=verify_retained_stream_recipe(report,maximum_seconds=30.)
    assert all(verified['checks'].values()) and verified['decoded_recipe_frames']==5
    assert verified['preencode_recipe_independently_verified']
    assert not report['persisted_lossless_master'] and report['compatible_movie_is_lossy']
    assert not verified['compatible_movie_decode_verified']
    assert not verified['compatible_mp4_decoded_identity_claimed']
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames',
        '-of','csv=p=0',report['movie']],capture_output=True,text=True,check=True,timeout=10.)
    assert int(probe.stdout.strip())==5


def test_stream_cannot_patch_protected_frames_or_reuse_changed_content(tmp_path):
    values=options(tmp_path);values['protected_intervals']=((1,3),)
    with pytest.raises(ValueError,match='protected decoded frame'):construct_retained_stream_movie(**values)
    assert not values['output'].exists()
    values['protected_intervals']=((3,5),);values['expected_reference_sha256']='0'*64
    with pytest.raises(ValueError,match='input content changed'):construct_retained_stream_movie(**values)


def test_recipe_checksum_tampering_is_not_a_stream_pass(tmp_path):
    report=construct_retained_stream_movie(**options(tmp_path));report['protected_raster_sha256']='0'*64
    with pytest.raises(ValueError,match='recipe verification failed'):verify_retained_stream_recipe(report)


def test_encoder_bound_includes_vbv_audio_container_and_exact_frame_duration():
    assert bounded_review_encoder_bytes(frame_count=5,fps=10.,maximum_video_bitrate=32000000,
        video_buffer_bits=48000000,audio_payload_bytes=123,container_reserve_bytes=100)==8000223
    expected=bounded_review_encoder_bytes(frame_count=4357,fps=29.97,maximum_video_bitrate=32000000,
        video_buffer_bits=48000000,audio_payload_bytes=10000000)
    assert expected<640*1024**2


def test_unbounded_or_incomplete_encoder_policy_is_rejected_before_start(tmp_path):
    values=options(tmp_path)
    with pytest.raises(ValueError,match='nonnegative payloads'):
        construct_retained_stream_movie(**values,maximum_video_bitrate=32000000)
    with pytest.raises(ValueError,match='exceeds the output cap'):
        construct_retained_stream_movie(**values,maximum_video_bitrate=32000000,
            video_buffer_bits=48000000,audio_payload_bound=123)


def provider_options(tmp_path):
    import json
    from crosscut.group_retained_raster_stream import FrozenRetainedRasterProvider
    a=tmp_path/'original.mp4';fixture_movie(a,0)
    token=dict(contract='selected_raw_raster_provider_v1',frame_count=5,frame_size_wh=[40,24],
        changed_intervals=[[1,2]],selected_program_identity='a'*64,native_evidence_root='b'*64,
        renderer_identity='c'*64,post_selection_camera_refit=False)
    calls=[]
    def frame(t):calls.append(t);return np.full((24,40,3),137,np.uint8)
    provider=FrozenRetainedRasterProvider(json.dumps(token,sort_keys=True),frame)
    return dict(reference_movie=a,output=tmp_path/'raw-delivery.mp4',frame_count=5,fps=10.,
        width=40,height=24,changed_intervals=((1,2),),protected_intervals=((3,5),),
        expected_reference_sha256=content_sha(a),selected_raw_provider=provider,
        expected_provider_identity=provider.identity,minimum_free_bytes=1,
        maximum_output_bytes=1024*1024,maximum_seconds=30.),calls


def test_raw_provider_encodes_once_without_a_proposed_movie_and_replays_exact_recipe(tmp_path):
    values,calls=provider_options(tmp_path)
    report=construct_retained_stream_movie(**values)
    assert calls==[1] and report['proposed_movie'] is None
    assert not report['proposed_movie_encode_required']
    verified=verify_retained_stream_recipe(report,selected_raw_provider=values['selected_raw_provider'])
    assert calls==[1,1] and all(verified['checks'].values())
    assert not verified['compatible_movie_decode_verified']


def test_raw_provider_cannot_refit_its_camera_or_borrow_another_binding(tmp_path):
    import json
    from crosscut.group_retained_raster_stream import FrozenRetainedRasterProvider
    values,_=provider_options(tmp_path);provider=values['selected_raw_provider']
    p=provider.payload;p['post_selection_camera_refit']=True
    with pytest.raises(ValueError,match='immutable exact'):
        FrozenRetainedRasterProvider(json.dumps(p),provider.frame_for_tick)
    values['expected_provider_identity']='e'*64
    with pytest.raises(ValueError,match='identity'):construct_retained_stream_movie(**values)


def test_independent_raw_recipe_replay_needs_the_same_provider_and_detects_changed_pixels(tmp_path):
    values,_=provider_options(tmp_path);provider=values['selected_raw_provider']
    report=construct_retained_stream_movie(**values)
    with pytest.raises(ValueError,match='frozen selected raw provider'):verify_retained_stream_recipe(report)
    from crosscut.group_retained_raster_stream import FrozenRetainedRasterProvider
    bad=FrozenRetainedRasterProvider(provider.token_json,lambda t:np.full((24,40,3),138,np.uint8))
    with pytest.raises(ValueError,match='recipe verification failed'):
        verify_retained_stream_recipe(report,selected_raw_provider=bad)
