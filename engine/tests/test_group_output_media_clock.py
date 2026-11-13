"""Real tiny codecs prove clocks/ordinals, never human pose-match quality."""
from fractions import Fraction
import copy
import subprocess
import cv2
import numpy as np
import pytest

from crosscut.group_output_media_clock import (ordinal_video_encoder_arguments,
    output_rate, verify_final_output_media_clock, _audio_signature)


def fixture_movie(path):
    frames = np.zeros((24, 64, 96, 3), np.uint8)
    palette = [(25, 45, 210), (45, 200, 35), (220, 50, 40), (160, 180, 20)]
    for n in range(24):
        frames[n] = palette[n // 6]
        # A visible ordinal barcode makes a same-shot duplicate detectable too.
        for bit in range(5):
            if n & (1 << bit): frames[n, 8:56, 8+bit*16:20+bit*16] = (240, 240, 240)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo',
        '-pixel_format', 'bgr24', '-video_size', '96x64', '-framerate', '2997/100',
        '-i', 'pipe:0', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-crf', '16',
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-t', str(24/29.97), str(path)],
        input=frames.tobytes(), check=True, timeout=20.)


def decoded_frames(path):
    cap = cv2.VideoCapture(str(path)); rows = []
    try:
        while True:
            ok, image = cap.read()
            if not ok: break
            rows.append(image)
    finally: cap.release()
    return np.array(rows)


def test_rational_clock_rebuilt_from_ordinals_not_rounded_mkv_timestamps(tmp_path):
    original = tmp_path/'original.mp4'; fixture_movie(original)
    master = tmp_path/'millisecond-master.mkv'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(original),
        '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'ffv1', '-c:a', 'copy', str(master)],
        check=True, timeout=20.)
    delivered = tmp_path/'delivered.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(master), '-i', str(original),
        '-map', '0:v:0', '-map', '1:a:0', *ordinal_video_encoder_arguments(29.97),
        '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
        str(delivered)], check=True, timeout=20.)
    report = verify_final_output_media_clock(movie=delivered, original_audio_movie=original,
        frame_count=24, fps=29.97)
    assert report['original_audio_preroll_and_skip_samples_equal']
    assert report['final_zero_origin_video_clock']['frame_rate'] == '2997/100'
    a, b = decoded_frames(original), decoded_frames(delivered)
    assert len(a) == len(b) == 24
    # All actual decoded labels keep their ordinal, including every cut edge.
    nearest = [int(np.argmin(np.mean((a.astype(float)-frame.astype(float))**2, axis=(1,2,3))))
        for frame in b]
    assert nearest == list(range(24))
    assert report['editorial_pose_match_approved'] is False


def test_video_origin_shift_cannot_pass_by_matching_duration_and_frame_count(tmp_path):
    original = tmp_path/'original.mp4'; fixture_movie(original)
    shifted = tmp_path/'shifted.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(original),
        '-map', '0:v:0', '-map', '0:a:0', '-vf', 'setpts=PTS+1/TB',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
        '-copyts', '-avoid_negative_ts', 'disabled', '-c:a', 'copy', str(shifted)],
        check=True, timeout=20.)
    with pytest.raises(ValueError, match='zero-origin'):
        verify_final_output_media_clock(movie=shifted, original_audio_movie=original,
            frame_count=24, fps=29.97)


def test_audio_signature_keeps_negative_preroll_and_skip_evidence():
    payload = dict(streams=[dict(codec_name='aac', sample_rate='48000', channels=2,
        channel_layout='stereo', time_base='1/48000')], packets=[dict(pts=-1024, dts=-1024,
        duration=1024, size=100, data_hash='SHA256:'+'a'*64,
        side_data_list=[dict(side_data_type='Skip Samples', skip_samples=1024, discard_padding=0)])])
    original = _audio_signature(payload)
    changed = copy.deepcopy(payload); changed['packets'][0]['pts'] = 0
    assert _audio_signature(changed) != original
    changed = copy.deepcopy(payload); changed['packets'][0]['side_data_list'] = []
    assert _audio_signature(changed) != original
    changed = copy.deepcopy(payload); changed['packets'][0]['duration'] = 1017
    assert _audio_signature(changed) != original


def test_declared_rate_and_encoder_pts_are_exact_and_bounded():
    assert output_rate(29.97) == Fraction(2997,100)
    arguments = ordinal_video_encoder_arguments(29.97)
    assert 'scale=out_color_matrix=bt709:out_range=tv,settb=expr=1/2997,setpts=N*100' in arguments
    assert 'passthrough' in arguments and 'disabled' in arguments
    for value in (True, 0, -1, float('nan')):
        with pytest.raises(ValueError): output_rate(value)
