"""Bounded source selection must not use the later recognition scores."""
import hashlib
from types import SimpleNamespace

import pytest

from audit_stageedit_face_spaces import select_observations, verify_source, observed_face_geometry, _iou


def obs(source, time, *, height=.3, identifier=None):
    return SimpleNamespace(source_id=source, timestamp_seconds=time,
        observation_id=identifier or f'{source}:{time}',
        face_bbox=SimpleNamespace(top=.1, bottom=.1+height))


def test_quantiles_separate_nearby_frames_and_preserve_actual_observations():
    original = [obs('b', .1), obs('a', 0), obs('a', .1), obs('a', 1),
        obs('a', 2), obs('a', 3), obs('a', 4), obs('a', 5), obs('a', 6)]
    result = select_observations(reversed(original))
    assert [(v.source_id, v.timestamp_seconds) for v in result] == [
        ('a', 0), ('a', 3), ('a', 6), ('b', .1)]
    assert all(any(v is old for old in original) for v in result)


def test_geometry_only_preselection_small_faces_and_no_faces_stay_out():
    no_face = obs('a', 0)
    no_face.face_bbox = None
    assert select_observations([no_face, obs('a', 1, height=.14)]) == ()
    assert select_observations([obs('a', 0), obs('a', 1), obs('a', 2)],
        per_source=1)[0].timestamp_seconds == 1


@pytest.mark.parametrize('count', [0, 7, True, 1.5])
def test_invalid_count_cannot_expand_analysis(count):
    with pytest.raises(ValueError):
        select_observations([], per_source=count)


@pytest.mark.parametrize('height', [0, -1, 1.1, float('nan')])
def test_invalid_height_rejected(height):
    with pytest.raises(ValueError):
        select_observations([], minimum_face_height=height)


def test_iou_requires_the_same_physical_face_not_merely_largest_face():
    assert _iou((.1,.1,.3,.3), (.1,.1,.3,.3)) == pytest.approx(1.)
    assert _iou((.1,.1,.3,.3), (.5,.5,.9,.9)) == 0.


def test_verify_source_hashes_current_bytes_and_rejects_same_size_replacement(tmp_path):
    path = tmp_path/'input.mp4'
    path.write_bytes(b'original')
    source = dict(path=str(path), size_bytes=8,
        media_content_sha256=hashlib.sha256(b'original').hexdigest())
    assert verify_source(source, device=path.stat().st_dev, root=tmp_path)[2] == 8
    path.write_bytes(b'replaced')
    with pytest.raises(OSError, match='content differs'):
        verify_source(source, device=path.stat().st_dev, root=tmp_path)


def test_verify_source_rejects_symlink_and_wrong_device(tmp_path):
    path = tmp_path/'input.mp4'
    path.write_bytes(b'a')
    source = dict(path=str(path), size_bytes=1,
        media_content_sha256=hashlib.sha256(b'a').hexdigest())
    with pytest.raises(OSError, match='differs'):
        verify_source(source, device=-1, root=tmp_path)
    link = tmp_path/'link.mp4'
    link.symlink_to(path)
    source['path'] = str(link)
    with pytest.raises(OSError, match='nonsymlink'):
        verify_source(source, device=path.stat().st_dev, root=tmp_path)


def test_detector_rectangle_can_spill_without_changing_observed_alignment_points():
    face = [-2., 10., 50., 60., 5., 20., 35., 20., 20., 35., 8., 50., 32., 50., .9]
    box, landmarks, raw = observed_face_geometry(face, 100, 100)
    assert raw[0] == -.02 and box == (0., .1, .48, .6)
    assert landmarks == ((.05,.2),(.35,.2),(.2,.35),(.08,.5),(.32,.5))
    face[4] = -1.
    with pytest.raises(ValueError, match='inside the frame'):
        observed_face_geometry(face, 100, 100)
