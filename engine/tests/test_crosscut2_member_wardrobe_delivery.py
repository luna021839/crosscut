import shutil
import subprocess
import numpy as np
import pytest

from crosscut2.member_wardrobe_delivery import render


@pytest.mark.parametrize('edge',[0,2])
def test_actual_patch_frame_bounds_and_duration(tmp_path,edge):
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:pytest.skip('local FFmpeg required')
    base=tmp_path/'base.mp4';patch=tmp_path/'patch.mp4'
    for path,color,frames in ((base,'red',20),(patch,'blue',5)):
        subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i',f'color={color}:s=160x90:r=30:d={frames/30}',
            '-f','lavfi','-i',f'sine=frequency=440:duration={frames/30}',
            '-frames:v',str(frames),'-c:v','libx264','-bf','0','-g','5','-pix_fmt','yuv420p','-c:a','aac',
            '-video_track_timescale','30000','-n',str(path)],capture_output=True,check=True,timeout=10.)
    out=tmp_path/'result.mp4'
    result=render(base,[dict(movie=str(patch),first_frame=7,frame_count=5,edge_blend_frames=edge)],out,
        ffmpeg=ffmpeg,maximum_seconds=20.,video_codec='libx264')
    assert result['output_frames']==20 and result['source_NN']==0
    assert result['audio_packet_payloads_and_timestamps_preserved']
    assert result['storage_reserve_bytes']==768*2**20
    raw=subprocess.run([ffmpeg,'-v','error','-i',str(out),'-an','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],
        capture_output=True,check=True,timeout=10.)
    frames=np.frombuffer(raw.stdout,np.uint8).reshape(20,90,160,3)
    colors=frames[:,45,80]
    assert np.all(colors[:7,0]>240) and np.all(colors[12:,0]>240)
    if edge:
        assert colors[7,0]>240 and colors[11,0]>240
        assert colors[9,2]>240 and colors[9,0]<10
    else:assert np.all(colors[7:12,2]>240) and np.all(colors[7:12,0]<10)
    assert not (tmp_path/'.result.encoding.mp4').exists()
    with pytest.raises(FileExistsError):
        render(base,[dict(movie=str(patch),first_frame=7,frame_count=5)],out,ffmpeg=ffmpeg)


@pytest.mark.parametrize('option',[dict(reserve_bytes=0),dict(reserve_bytes=256*2**20-1),
    dict(video_bitrate=0),dict(video_bitrate=80_000_001)])
def test_explicit_low_storage_settings_are_still_bounded(tmp_path,option):
    with pytest.raises(ValueError):render(tmp_path/'base',[],tmp_path/'new.mp4',**option)
