"""Read physically bound MEMBER geometry without aliasing old producers.

This adapter never decodes an image or opens a model. Fresh member hints can
replace a face, but an old arm pose survives only on the same observed head.
Unresolved member hints remain UNKNOWN rather than excluding that source.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

from crosscut.visual_feature_cache import CanonicalFrameIdentity


def identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def check_sources(sources, identities):
    if len(sources) != len(identities) or any(identity(s.path) != list(i) for s, i in zip(sources, identities, strict=True)):
        raise OSError('physical member sources changed')


def load_geometry(database, *, namespace, sources, identities, check=lambda: None):
    """Read one explicit frozen namespace; validate digest and native binding."""
    check_sources(sources, identities)
    result = [{} for _ in sources]
    query = Path(database).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(query, uri=True) as connection:
        connection.execute('PRAGMA query_only=ON')
        for frame_key, digest, compressed in connection.execute(
                'SELECT frame_key,digest,payload FROM observations WHERE namespace=?', (namespace,)):
            check()
            decoder = zlib.decompressobj(); raw = decoder.decompress(compressed, 32769)
            if (len(raw) > 32768 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                    or hashlib.sha256(raw).hexdigest() != digest):
                raise ValueError('invalid frozen member geometry record')
            value = json.loads(raw)
            for s, (source, bound) in enumerate(zip(sources, identities, strict=True)):
                native = CanonicalFrameIdentity.from_seconds(value['source_seconds'], frame_rate=source.fps)
                binding = (str(source.path), *bound[1:], native.frame_index, native.fps_numerator, native.fps_denominator)
                if hashlib.sha256(json.dumps(binding, separators=(',', ':')).encode()).hexdigest() != frame_key:
                    continue
                if native.frame_index in result[s]:
                    raise ValueError('ambiguous frozen member native observation')
                result[s][native.frame_index] = value
                break
    check_sources(sources, identities)
    return result


def augment_faces(observations, *, sources, identities, supplements, member):
    """Return copies; do not rewrite SQLite, raw measurements or namespaces."""
    check_sources(sources, identities)
    if len(observations) != len(sources) or not isinstance(member, str) or not member:
        raise ValueError('one source geometry bank and explicit target member required')
    supplements = tuple(supplements)
    result = [dict(rows) for rows in observations]
    spaces = {(data['gallery_identity'], json.dumps(data['descriptor_space'], sort_keys=True)) for data in supplements}
    if len(spaces) > 1:
        raise ValueError('current member faces use different descriptor/reference spaces')
    seen = {}; queried = positive = retained = unresolved = 0
    for data in supplements:
        for entry in data['sources']:
            s = entry['source']
            if type(s) is not int or not 0 <= s < len(sources):
                raise ValueError('current member face source index is out of range')
            source = sources[s]
            if entry['source_file_identity'] != list(identities[s]):
                raise OSError('current member face source binding changed')
            for value in entry['observations']:
                n = value['native']; key = s, n
                if type(n) is not int or not 0 <= n < source.frame_count:
                    raise ValueError('current member face requires an original native frame')
                if key in seen:
                    if seen[key] != value['raster_sha256']:
                        raise ValueError('two current member supplements disagree on physical pixels')
                    continue
                seen[key] = value['raster_sha256']; queried += 1; old = result[s].get(n)
                candidates = [face for face in value['faces'] if face['member'] == member]
                target = max(candidates, key=lambda face: face['member_scores'][member]) if candidates else None
                pose = []; scale = None
                if target is not None:
                    positive += 1
                    if old is not None and old['face_bbox_xywh'] is not None:
                        a, b = np.array(old['face_bbox_xywh']), np.array(target['bbox_xywh'])
                        distance = np.linalg.norm((a[:2] + a[2:]/2 - b[:2] - b[2:]/2) *
                                   [source.width/source.height, 1.]) / max(.01, b[3])
                        if distance <= .5 and abs(np.log(a[3]/b[3])) <= .4:
                            pose = old.get('pose_screen_xyc') or []; scale = old.get('face_scale')
                            retained += bool(pose)
                else:
                    unresolved += 1
                result[s][n] = dict(source_seconds=value['source_seconds'],
                    face_bbox_xywh=None if target is None else target['bbox_xywh'], face_scale=scale,
                    pose_screen_xyc=pose, supplemental_current_face_observation=True,
                    current_target_hint_unresolved=target is None,
                    pose_kept_only_on_original_head_geometry_match=bool(pose), body_observation_fabricated=False)
    check_sources(sources, identities)
    return result, dict(actual_current_source_queries=queried, current_target_faces=positive,
        current_target_hints_unresolved=unresolved, old_screen_poses_retained_on_same_observed_face=retained)
