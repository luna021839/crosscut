import os

import cv2
import numpy as np
import pytest

from crosscut2.source_cuts import census, read_frames


def test_packed_frame_reader_preserves_byte_order_and_counts():
    first, final = os.pipe()
    os.write(final, bytes(range(24)))
    os.close(final)
    with os.fdopen(first, 'rb') as reader:
        frames = list(read_frames(reader, check=lambda: None, width=2, height=2))
    assert len(frames) == 2
    assert frames[0].ravel().tolist() == list(range(12))
    assert frames[1].ravel().tolist() == list(range(12, 24))


def test_packed_frame_reader_rejects_partial_eof_without_imputing_pixels():
    first, final = os.pipe()
    os.write(final, b'\x00' * 11)
    os.close(final)
    with os.fdopen(first, 'rb') as reader:
        with pytest.raises(OSError, match='incomplete'):
            list(read_frames(reader, check=lambda: None, width=2, height=2))


def test_real_streaming_census_preserves_clock_and_reuses_complete_results(tmp_path, monkeypatch):
    path = tmp_path / 'camera.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64, 36))
    assert writer.isOpened()
    try:
        for color in ((0, 0, 255), (0, 255, 0), (255, 0, 0)):
            for _ in range(20):
                writer.write(np.full((36, 64, 3), color, dtype=np.uint8))
    finally:
        writer.release()
    original, = census((path,), cache=tmp_path / 'cache', check=lambda: None)
    assert original['all_encoded_frames_decoded'] is True
    assert original['frame_count'] == 60 and original['fps'] == pytest.approx(10.)
    assert original['frames'] == [20, 40]
    def no_decode(*args, **kwargs):
        raise AssertionError('completed camera census must not decode again')
    monkeypatch.setattr('crosscut2.source_cuts.subprocess.Popen', no_decode)
    repeated, = census((path,), cache=tmp_path / 'cache', check=lambda: None)
    assert repeated == original
