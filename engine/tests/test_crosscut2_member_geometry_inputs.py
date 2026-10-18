import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import pytest

from crosscut.visual_feature_cache import CanonicalFrameIdentity
from crosscut2.member_geometry_inputs import augment_faces, identity, load_geometry
from crosscut2.member_stage_mix import Source


def fixture(tmp_path):
    path = tmp_path/'source.mp4'; path.write_bytes(b'original fixture, never decoded')
    source = Source(path, 30., 1080, 1920, 100)
    ids = [identity(path)]
    row = dict(source_seconds=1., face_bbox_xywh=[.4,.2,.1,.1], face_scale=.1,
               pose_screen_xyc=[[.4,.5,.9]]*8)
    native = CanonicalFrameIdentity.from_seconds(1., frame_rate=30.)
    key = hashlib.sha256(json.dumps((str(path), *ids[0][1:], native.frame_index,
        native.fps_numerator, native.fps_denominator), separators=(',', ':')).encode()).hexdigest()
    raw = json.dumps(row).encode(); database = tmp_path/'geometry.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE observations(namespace TEXT,frame_key TEXT,digest TEXT,payload BLOB)')
        connection.execute('INSERT INTO observations VALUES(?,?,?,?)',
            ('frozen', key, hashlib.sha256(raw).hexdigest(), zlib.compress(raw)))
    supplement = dict(gallery_identity='gallery-sha', descriptor_space=dict(kind='r50', dimensions=512), sources=[
        dict(source=0, source_file_identity=ids[0], observations=[dict(native=30, source_seconds=1., raster_sha256='b'*64,
            faces=[dict(member='가을', member_scores={'가을': .7}, bbox_xywh=[.4,.2,.1,.1])])])])
    return source, ids, row, database, supplement


def test_frozen_native_rows_replay_without_changing_database_or_producer(tmp_path):
    source, ids, row, database, _ = fixture(tmp_path)
    before = database.read_bytes()
    rows = load_geometry(database, namespace='frozen', sources=[source], identities=ids)
    assert rows == [{30: row}] and database.read_bytes() == before
    assert load_geometry(database, namespace='another-producer', sources=[source], identities=ids) == [{}]


def test_digest_or_source_identity_change_cannot_relabel_old_geometry(tmp_path):
    source, ids, _, database, _ = fixture(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE observations SET digest='invalid'")
    with pytest.raises(ValueError, match='invalid frozen'):
        load_geometry(database, namespace='frozen', sources=[source], identities=ids)
    source.path.write_bytes(b'changed original')
    with pytest.raises(OSError, match='physical member'):
        load_geometry(database, namespace='frozen', sources=[source], identities=ids)


def test_current_face_retains_arms_only_on_same_observed_head_without_mutating_inputs(tmp_path):
    source, ids, row, _, supplement = fixture(tmp_path)
    original = [{30: row}]; before = copy.deepcopy(original)
    updated, receipt = augment_faces(original, sources=[source], identities=ids, supplements=[supplement], member='가을')
    assert original == before and updated[0][30]['pose_screen_xyc'] == row['pose_screen_xyc']
    assert receipt['old_screen_poses_retained_on_same_observed_face'] == 1
    supplement['sources'][0]['observations'][0]['faces'][0]['bbox_xywh'][0] = .8
    other, receipt = augment_faces(original, sources=[source], identities=ids, supplements=[supplement], member='가을')
    assert other[0][30]['pose_screen_xyc'] == [] and not receipt['old_screen_poses_retained_on_same_observed_face']


def test_current_unresolved_member_hint_is_unknown_not_a_source_veto(tmp_path):
    source, ids, row, _, supplement = fixture(tmp_path)
    supplement['sources'][0]['observations'][0]['faces'] = []
    updated, receipt = augment_faces([{30: row}], sources=[source], identities=ids, supplements=[supplement], member='가을')
    assert updated[0][30]['face_bbox_xywh'] is None
    assert updated[0][30]['current_target_hint_unresolved']
    assert not updated[0][30]['body_observation_fabricated']
    assert receipt['current_target_hints_unresolved'] == 1


def test_duplicate_pixels_descriptor_spaces_and_source_indices_are_checked(tmp_path):
    source, ids, row, _, supplement = fixture(tmp_path)
    other = copy.deepcopy(supplement)
    other['sources'][0]['observations'][0]['raster_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='physical pixels'):
        augment_faces([{30: row}], sources=[source], identities=ids, supplements=[supplement, other], member='가을')
    other['descriptor_space']['dimensions'] = 128
    with pytest.raises(ValueError, match='descriptor/reference'):
        augment_faces([{30: row}], sources=[source], identities=ids, supplements=[supplement, other], member='가을')
    supplement['sources'][0]['source'] = -1
    with pytest.raises(ValueError, match='out of range'):
        augment_faces([{30: row}], sources=[source], identities=ids, supplements=[supplement], member='가을')
