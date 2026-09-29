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
    def __init__(self, sources, maximum_detection_width=960, *, shared_source_frames=None):
        from .shared_source_frames import validate_shared_sources
        self._shared_source_frames = validate_shared_sources(sources, shared_source_frames)
        if maximum_detection_width not in (480, 960):
            raise ValueError('native focal face detection width is unsupported')
        self.sources = tuple(Path(p) for p in sources)
        self.maximum_detection_width = maximum_detection_width
        self._captures = {}
        self._readers = {}
        self._cache = OrderedDict()
        self._candidate_completeness = {}
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
        if self._shared_source_frames is not None:
            owner = self._shared_source_frames[source]
            frame = owner.native_at(seconds)
            self._readers[source] = owner.reader
            return frame
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
        if self._shared_source_frames is not None:
            self._shared_source_frames[source].assert_source_available()
        # Exact requested times are retained for VFR sources; there is no
        # rounding into a different provider's frame-clock coordinate system.
        key = (source, seconds)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        native = self.frame_at(source, seconds)
        candidates = []
        complete = False
        if native is not None:
            working = _working_frame(native, maximum_width=self.maximum_detection_width)
            faces, backend = _detect_faces(working, cv2.cvtColor(working, cv2.COLOR_BGR2GRAY))
            if backend == 'yunet':
                complete = True
                for face in faces:
                    bbox, landmarks, _roll, eye_scale, _geometry = _normalized_face_metadata(
                        face, frame_width=working.shape[1], frame_height=working.shape[0])
                    if bbox is None or landmarks is None:
                        complete = False
                        continue
                    features = extract_sface_embedding(native, bbox_xywh=bbox, landmarks_xy=landmarks)
                    if features:
                        candidates.append(FocalFaceCandidate(bbox, features, eye_scale,
                            _landmark_rms_scale(landmarks, frame_width=working.shape[1], frame_height=working.shape[0])))
                    else:
                        complete = False
            self.inference_frame_count += 1
        self._cache[key] = tuple(candidates)
        # Scalar completion evidence shares the exact bounded candidate cache;
        # it does not repeat detection or persist pixels/identity features.
        self._candidate_completeness[key] = complete
        if len(self._cache)>4096:
            evicted, _ = self._cache.popitem(last=False)
            self._candidate_completeness.pop(evicted, None)
        return self._cache[key]

    def candidate_observation_complete(self, source, seconds):
        return (type(source) is int and type(seconds) in (int, float)
                and self._candidate_completeness.get((source, seconds)) is True)

    def close(self):
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
        self._readers.clear()
        self._cache.clear()
        self._candidate_completeness.clear()
