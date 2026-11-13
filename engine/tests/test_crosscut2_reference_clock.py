import hashlib

import pytest

from crosscut2.reference_clock import SharedContentClockBridge


def bridge(tmp_path, *, old=((0., 13.), (50., 63.), (100., 113.)),
           current=((0., 0.), (70., 70.), (140., 140.))):
    path = tmp_path/'same-video.mp4'
    path.write_bytes(b'test physical asset, not actual media')
    rows = lambda points: [dict(master_seconds=a, source_seconds=b) for a, b in points]
    return SharedContentClockBridge.from_source_file(path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        old_points=rows(old), current_points=rows(current))


def test_exact_clock_transfer_does_not_snap_to_nearby_editorial_node(tmp_path):
    value = bridge(tmp_path).output_event(50.4, master_start=26.7, fps=30., frame_count=4357)
    assert value['output_frame_float'] == pytest.approx(1101.)
    assert value['output_frame'] == 1101  # NOT an existing nearby 1103/1106 grid node
    assert value['exact_event_observations_required']
    assert not value['human_av_preference_created']


def test_piecewise_drift_inversion_and_only_output_frame_quantization(tmp_path):
    value = bridge(tmp_path, old=((0., 13.), (50., 63.1), (100., 113.3)),
        current=((0., 0.), (70., 70.07), (140., 140.))).output_event(
            50.4, master_start=26.7, fps=30000/1001, frame_count=4357)
    current = (63.1+.4*1.004)/1.001
    assert value['current_master_seconds'] == pytest.approx(current)
    assert value['output_frame_float'] == pytest.approx((current-26.7)*30000/1001)
    assert abs(value['quantization_frames']) <= .5


def test_clock_bridge_requires_same_bytes_and_strictly_monotonic_anchors(tmp_path):
    path = tmp_path/'same-video-id.mp4'
    path.write_bytes(b'different transcode')
    points = [dict(master_seconds=x, source_seconds=x) for x in (0., 10.)]
    with pytest.raises(ValueError, match='physical reference asset differs'):
        SharedContentClockBridge.from_source_file(path, expected_sha256='a'*64,
            old_points=points, current_points=points)
    with pytest.raises(ValueError, match='strictly monotonic'):
        bridge(tmp_path, old=((0., 13.), (10., 12.)))
    with pytest.raises(ValueError, match='strictly monotonic'):
        bridge(tmp_path, current=((0., 0.), (float('nan'), 5.)))


def test_no_extrapolation_and_no_silent_out_of_movie_label(tmp_path):
    value = bridge(tmp_path)
    with pytest.raises(ValueError, match='cannot extrapolate'):
        value.transfer(100.01)
    with pytest.raises(ValueError, match='cannot extrapolate'):
        bridge(tmp_path, current=((0., 0.), (60., 60.))).transfer(50.)
    with pytest.raises(ValueError, match='movie interior'):
        value.output_event(0., master_start=26.7, fps=30., frame_count=4357)
