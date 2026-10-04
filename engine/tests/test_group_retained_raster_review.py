"""Tiny nonhuman FFV1 fixtures; no artist media or professional QA claim."""
import subprocess
import numpy as np
import pytest

from crosscut.group_retained_raster_review import (content_sha,construct_retained_raster_master,
    verify_retained_raster_master)

def fixture_movie(path,offset):
    frames=np.zeros((5,24,40,3),dtype=np.uint8)
    for n in range(5):
        frames[n,:,:,0]=n*25+offset
        frames[n,2:20,4:30,1]=90+offset
        frames[n,:,:,2]=np.arange(40,dtype=np.uint8)[None,:]
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pixel_format','bgr24',
        '-video_size','40x24','-framerate','10','-i','pipe:0','-f','lavfi','-i','anullsrc=r=48000:cl=stereo',
        '-map','0:v:0','-map','1:a:0','-c:v','ffv1','-pix_fmt','bgr0','-c:a','pcm_s16le',
        '-t','0.5',str(path)],input=frames.tobytes(),check=True,timeout=20.)

def options(tmp_path):
    a=tmp_path/'original.mkv';b=tmp_path/'proposed.mkv'
    fixture_movie(a,0);fixture_movie(b,17)
    return dict(reference_movie=a,proposed_movie=b,output=tmp_path/'master.mkv',frame_count=5,
        fps=10.,width=40,height=24,changed_intervals=((1,2),),protected_intervals=((3,5),),
        expected_reference_sha256=content_sha(a),expected_proposed_sha256=content_sha(b),
        minimum_free_bytes=1,maximum_output_bytes=1024*1024,maximum_seconds=30.)

def test_lossless_actual_decoded_outside_and_protected_identity(tmp_path):
    report=construct_retained_raster_master(**options(tmp_path))
    verified=verify_retained_raster_master(report,maximum_seconds=30.)
    assert verified['master_decoded_identity_verified']
    assert verified['checks']['every_outside_frame_equal']
    assert verified['protected_frames']==2 and report['outside_patch_frames']==4
    assert not report['compatible_mp4_decoded_identity_claimed']
    assert not verified['quality_approved']

def test_protected_overlap_rejected_before_master_write(tmp_path):
    values=options(tmp_path);values['protected_intervals']=((1,3),)
    with pytest.raises(ValueError,match='protected decoded frame'):
        construct_retained_raster_master(**values)
    assert not values['output'].exists()

def test_forged_original_content_rejected_before_decode(tmp_path):
    values=options(tmp_path);values['expected_reference_sha256']='0'*64
    with pytest.raises(ValueError,match='input content changed'):
        construct_retained_raster_master(**values)
    assert not values['output'].exists()
