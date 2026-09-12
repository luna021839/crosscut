"""Pixel observations for the bounded source-local focal body tracker.

The source face provider is shared with the selector and dense camera planner.
Only frozen source seeds create the body appearance bank. Query and recovered
tracks cannot update it. Small neighboring windows share frame observations,
so changing an edit path does not reanalyse complete source videos.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import math
from pathlib import Path

import cv2

from .body_appearance import BodyAppearanceEncoder
from .focal_body_track import (FocalBodyFrame, FocalBodyObservation,
    body_reference_similarity, resolve_focal_body_track)
from .focal_pose import face_associated_person
from .pose import _NanoDetFormationDetector, _formation_people_descriptor


class SourceLocalFocalBodyProvider:
    def __init__(self, face_provider, encoder: BodyAppearanceEncoder, *, pose_analyzer=None):
        if not getattr(face_provider, '_require_focal_consensus', False):
            raise ValueError('focal body tracks require immutable source face banks')
        self.face_provider = face_provider
        self.encoder = encoder
        self.pose_analyzer = pose_analyzer
        self.detector = _NanoDetFormationDetector(Path(__file__).parent/'models/object_detection_nanodet_2022nov.onnx')
        self._references = {}
        self._frame_cache = OrderedDict()
        self._track_cache = OrderedDict()
        self.frame_inference_count = 0
        self.frame_cache_hit_count = 0

    def _people_and_features(self, frame):
        people = _formation_people_descriptor(self.detector.infer(frame),
            frame_width=frame.shape[1], frame_height=frame.shape[0], minimum_width=.015,
            minimum_height=.045, minimum_confidence=.35)[0]
        return people, self.encoder.describe(frame, tuple(p.bbox_xywh for p in people))

    def _ensure_references(self, source_index):
        if source_index in self._references:
            return self._references[source_index]
        bank = self.face_provider._ensure_focal_consensus(source_index)
        sampler = self.face_provider._samplers[source_index]
        references = []
        if bank and sampler:
            for observation in bank.references:
                frame = sampler.frame_at(observation.source_seconds)
                if frame is None:
                    continue
                people, features = self._people_and_features(frame)
                person = face_associated_person(observation.candidate.bbox_xywh, people)
                if person is not None:
                    references.append(features[people.index(person)])
        self._references[source_index] = tuple(references)
        return self._references[source_index]

    def frame_observations(self, source_index, seconds):
        sampler = self.face_provider._samplers[source_index]
        if sampler is None:
            return FocalBodyFrame(source_index, seconds, ())
        key = self.face_provider._identity_key(source_index, sampler, seconds)
        if key in self._frame_cache:
            self.frame_cache_hit_count += 1
            frame = self._frame_cache[key]
            self._frame_cache.move_to_end(key)
            # Frame pixels, not caller float spelling, are the cache key.
            return FocalBodyFrame(source_index, seconds, tuple(replace(o, source_seconds=seconds) for o in frame.observations))
        references = self._ensure_references(source_index)
        bank = self.face_provider._ensure_focal_consensus(source_index)
        pixels = sampler.frame_at(seconds)
        observations = []
        if pixels is not None and len(references) >= 3:
            people, features = self._people_and_features(pixels)
            faces = self.face_provider._focal_candidates(source_index, seconds)
            frozen = bank.select(faces) if bank else None
            for person, features in zip(people, features):
                associated = tuple(face for face in faces if face_associated_person(face.bbox_xywh, people) is person)
                face = associated[0] if len(associated) == 1 else None
                similarity = body_reference_similarity(features, references)
                pose = ()
                if self.pose_analyzer is not None and similarity >= .60:
                    measured = self.pose_analyzer.describe(pixels, bbox_xywh=person.bbox_xywh,
                        face_bbox_xywh=face.bbox_xywh if face else None)
                    if measured.reason == 'observed':
                        pose = measured.upper_body_xyc
                        if measured.face_anchor_consistent is False:
                            face = None
                observations.append(FocalBodyObservation(source_index, seconds, person,
                    features, similarity, face, bool(face is not None and face == frozen), pose))
            self.frame_inference_count += 1
        result = FocalBodyFrame(source_index, seconds, tuple(observations))
        self._frame_cache[key] = result
        if len(self._frame_cache)>4096:
            self._frame_cache.popitem(last=False)
        return result

    def track_at(self, source_index, seconds):
        if (isinstance(source_index, bool) or not isinstance(source_index, int)
                or not 0 <= source_index < len(self.face_provider._sources)
                or isinstance(seconds, bool) or not isinstance(seconds, (float, int))
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError('source-local focal query must be source/time bound')
        sampler = self.face_provider._samplers[source_index]
        if sampler is None:
            return resolve_focal_body_track((), source_index=source_index, query_seconds=seconds)
        key = self.face_provider._identity_key(source_index, sampler, seconds)
        if key in self._track_cache:
            result = self._track_cache[key]
            self._track_cache.move_to_end(key)
            return replace(result, source_seconds=seconds,
                observation=replace(result.observation, source_seconds=seconds) if result.observation else None)
        fps = float(sampler.frame_rate)
        count = sampler.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps<=0 or not math.isfinite(count) or count<=0:
            raise ValueError('source-local focal track frame clock unavailable')
        # 21 observations across approximately 2.4 seconds, with exact source-frame offsets.
        # No full-source retry and no repeated frame as independent evidence.
        step_frames = max(1, round(.12*fps))
        # Neighboring exact queries share one source-frame grid. Centering a
        # fresh grid on every adjacent query previously multiplied inference
        # by the stride, defeating reuse during dense camera checks.
        query_identity = sampler.frame_identity(seconds)
        grid_center = query_identity.frame_index//step_frames*step_frames
        frame_indices = [grid_center+step*step_frames for step in range(-10, 11)]
        times = [i/fps for i in frame_indices if 0<=i<count-1 and i != query_identity.frame_index]
        times.append(seconds)
        times.sort()
        result = resolve_focal_body_track(tuple(self.frame_observations(source_index, t) for t in times),
            source_index=source_index, query_seconds=seconds, maximum_window_seconds=min(2., 10*step_frames/fps+.000001))
        self._track_cache[key] = result
        if len(self._track_cache)>4096:
            self._track_cache.popitem(last=False)
        return result
