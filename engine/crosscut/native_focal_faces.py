"""Independent native-pixel face evidence for source-first member tracking.

Detection uses at most 960 pixels of width. SFace alignment samples the
original decoded raster, not an already downscaled 480-pixel thumbnail.
Neither this feature basis nor its face banks may be mixed with legacy ones.
Each source owns one native raster cache; appearance vectors remain transient.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import math
from pathlib import Path

import cv2

from .direct_member_target_face import FocalFaceCandidate, _landmark_rms_scale
from .face_identity import extract_sface_embedding, SFACE_SHA256
from .native_frame_reader import native_locality_reader
from .visual_score import _working_frame, _detect_faces, _normalized_face_metadata


class NativeFocalFaceProvider:
    def __init__(self, sources, maximum_detection_width=960):
        if maximum_detection_width not in (480, 960):
            raise ValueError('native focal face detection width is unsupported')
        self.sources = tuple(Path(p) for p in sources)
        self.maximum_detection_width = maximum_detection_width
        self._captures = {}
        self._readers = {}
        self._cache = OrderedDict()
        self.inference_frame_count = 0

    def fingerprint(self):
        return {'contract': 'native_raster_focal_face_evidence_v1',
            'maximum_detection_width': self.maximum_detection_width,
            'alignment_input': 'original_decoded_raster', 'sface_sha256': SFACE_SHA256,
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'face_embeddings_persisted': False}

    def frame_at(self, source, seconds):
        if (isinstance(source, bool) or not isinstance(source, int)
                or not 0<=source<len(self.sources) or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds<0):
            raise ValueError('native focal query must be source/time bound')
        if source not in self._captures:
            capture = cv2.VideoCapture(str(self.sources[source]))
            if not capture.isOpened():
                capture.release()
                raise OSError('native focal source could not be opened')
            self._captures[source] = capture
            self._readers[source] = native_locality_reader(capture, self.sources[source])
        return self._readers[source].frame_at(seconds)

    def candidates_at(self, source, seconds):
        if (isinstance(source, bool) or not isinstance(source, int)
                or not 0<=source<len(self.sources) or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds<0):
            raise ValueError('native focal query must be source/time bound')
        # Exact requested times are retained for VFR sources; there is no
        # rounding into a different provider's frame-clock coordinate system.
        key = (source, seconds)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        native = self.frame_at(source, seconds)
        candidates = []
        if native is not None:
            working = _working_frame(native, maximum_width=self.maximum_detection_width)
            faces, backend = _detect_faces(working, cv2.cvtColor(working, cv2.COLOR_BGR2GRAY))
            if backend == 'yunet':
                for face in faces:
                    bbox, landmarks, _roll, eye_scale, _geometry = _normalized_face_metadata(
                        face, frame_width=working.shape[1], frame_height=working.shape[0])
                    if bbox is None or landmarks is None:
                        continue
                    features = extract_sface_embedding(native, bbox_xywh=bbox, landmarks_xy=landmarks)
                    if features:
                        candidates.append(FocalFaceCandidate(bbox, features, eye_scale,
                            _landmark_rms_scale(landmarks, frame_width=working.shape[1], frame_height=working.shape[0])))
            self.inference_frame_count += 1
        self._cache[key] = tuple(candidates)
        if len(self._cache)>4096:
            self._cache.popitem(last=False)
        return self._cache[key]

    def close(self):
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
        self._readers.clear()
        self._cache.clear()
