from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from crosscut2 import pose_grid


def test_native_action_windows_keep_distinct_left_and_right_flanks():
    warp = SimpleNamespace(master_to_source=lambda t: t + .2)
    indices = pose_grid.pose_indices(warp, nodes=np.array([0, 20, 40]),
        master_start=0., fps=10., native_fps=20., native_count=100)
    assert indices[1].tolist() == [30, 42, 44, 56]
    assert indices[0, :3].tolist() == [4, 4, 4]
    assert indices[-1, 2:].tolist() == [82, 82]


def test_action_windows_cannot_read_outside_physical_source():
    with pytest.raises(ValueError, match='physical source'):
        pose_grid.pose_indices(SimpleNamespace(master_to_source=lambda t: t + 10),
            nodes=np.array([0, 20]), master_start=0., fps=10., native_fps=10., native_count=30)


def sensor(detector_rows, evidence):
    instance = pose_grid.PoseSensor.__new__(pose_grid.PoseSensor)
    instance.detector = SimpleNamespace(infer=lambda image: np.asarray(detector_rows))
    instance.body = SimpleNamespace(describe=lambda *args, **kwargs: evidence)
    instance.calls = 0
    return instance


def test_ambiguous_face_body_does_not_choose_another_available_pose():
    instance = sensor([[20, 10, 80, 98, .9], [25, 10, 75, 95, .9]], None)
    heads = np.array([[.5, .25, .15, .9], [0, 0, 0, 0], [0, 0, 0, 0]])
    poses, states = instance.observe(np.zeros((100, 100, 3), np.uint8), heads)
    assert states[0] == 2 and not poses.any() and instance.calls == 0


def test_model_face_contradiction_stays_unknown_not_a_zero_motion_match():
    evidence = SimpleNamespace(reason='observed', face_anchor_consistent=False,
        upper_body_xyc=tuple((.5, .5, .9) for _ in range(8)))
    instance = sensor([[20, 10, 80, 98, .9]], evidence)
    heads = np.array([[.5, .25, .15, .9], [0, 0, 0, 0], [0, 0, 0, 0]])
    poses, states = instance.observe(np.zeros((100, 100, 3), np.uint8), heads)
    assert states[0] == 3 and not poses.any() and instance.calls == 1


def test_actual_streamed_pose_grid_cache_reuses_without_model_calls(tmp_path, monkeypatch):
    source = tmp_path / 'synthetic-body-window.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (160, 90))
    if not writer.isOpened(): pytest.skip('fixture MJPG encoder unavailable')
    for n in range(40): writer.write(np.full((90, 160, 3), n * 4, np.uint8))
    writer.release()
    class SyntheticSensor:
        def __init__(self, *args): self.calls = 0
        def observe(self, pixels, heads):
            self.calls += 1
            points = np.zeros((3, 8, 3), np.float32)
            points[0] = [[.4, .4, .9], [.6, .4, .9], [.3, .5, .9], [.7, .5, .9],
                [.2, .6, .9], [.8, .6, .9], [.43, .75, .9], [.57, .75, .9]]
            return points, np.array([1, 0, 0], np.uint8)
    monkeypatch.setattr(pose_grid, 'PoseSensor', SyntheticSensor)
    heads = np.zeros((4, 2, 8, 4), np.float32); heads[:, :, 0] = [.5, .3, .12, .9]
    args = dict(sources=(source,), warps=(SimpleNamespace(master_to_source=lambda t:t),),
        nodes=np.array([0, 10, 20, 30]), master_start=0., fps=10., banks=[dict(heads=heads)],
        models=tmp_path, body_model=tmp_path/'not-a-real-model.onnx', cache=tmp_path/'body-cache',
        check=lambda: None, camera_census=[dict(fps=10., frame_count=40)], maximum_seconds=20.)
    first, receipts = pose_grid.measure_pose_grid(**args)
    assert not receipts[0]['reused'] and receipts[0]['neural_pose_calls'] > 0
    def forbidden(*args): raise AssertionError('warm body cache loaded a model')
    monkeypatch.setattr(SyntheticSensor, '__init__', forbidden)
    reused, warm = pose_grid.measure_pose_grid(**args)
    assert warm[0]['reused'] and warm[0]['neural_pose_calls'] == 0
    assert np.array_equal(first[0], reused[0])
