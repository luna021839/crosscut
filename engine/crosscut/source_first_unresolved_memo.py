"""Run-local reuse of completely observed, unresolved target-track queries.

Positive geometry keeps its existing persistent namespace. Only scalar missing
results are retained here: either every tracker frame completed native faces,
body detections and requested poses, or complete native-face observations prove
both direct admission and the required two recovery anchors impossible.
Failed decodes, absent models and exceptions must remain retryable. No missing result
is published to disk or promoted to source exclusion or cut approval.
"""
from collections import Counter, OrderedDict
from dataclasses import replace
import math

import cv2

from .focal_selector_geometry import FocalSelectorGeometry


def _verified_empty_native_detection(provider, source, seconds):
    from .visual_score import _detect_faces, _working_frame
    native = provider.native_faces.frame_at(source, seconds)
    if native is None:
        return False
    working = _working_frame(native, maximum_width=provider.native_faces.maximum_detection_width)
    faces, backend = _detect_faces(working, cv2.cvtColor(working, cv2.COLOR_BGR2GRAY))
    # The native provider also increments its counter for Haar fallback or
    # failed SFace extraction. Neither proves a valid empty YuNet observation.
    return backend == 'yunet' and not faces


def _verified_complete_native_candidates(provider, source, seconds, candidates):
    """Partial SFace extraction is not evidence that the target is absent.

    Inspect the same still-cached native raster while the provider is observing
    it. This never repeats SFace extraction or performs a neighboring-frame
    lookup; every detected box must have yielded a native identity candidate.
    """
    from .visual_score import _detect_faces, _working_frame, _normalized_face_metadata
    native = provider.native_faces.frame_at(source, seconds)
    if native is None:
        return False
    working = _working_frame(native, maximum_width=provider.native_faces.maximum_detection_width)
    faces, backend = _detect_faces(working, cv2.cvtColor(working, cv2.COLOR_BGR2GRAY))
    if backend != 'yunet' or len(faces) != len(candidates):
        return False
    boxes = []
    for face in faces:
        box, landmarks, *_ = _normalized_face_metadata(face,
            frame_width=working.shape[1], frame_height=working.shape[0])
        if box is None or landmarks is None:
            return False
        boxes.append(box)
    return Counter(boxes) == Counter(candidate.bbox_xywh for candidate in candidates)


class SourceFirstUnresolvedMemo:
    def __init__(self, provider, *, maximum_entries=65536):
        if isinstance(maximum_entries, bool) or not isinstance(maximum_entries, int) or maximum_entries <= 0:
            raise ValueError('unresolved query memo requires a positive bound')
        self.provider = provider
        self.original = provider.selector_geometry
        self.original_close = provider.close
        self.original_diagnostics = provider.diagnostics
        self.maximum_entries = maximum_entries
        self.namespace = provider.observation_namespace
        self.entries = OrderedDict()
        self.completed_face_frames = OrderedDict()
        self.original_candidates = getattr(provider.faces, '_external_focal_candidate_provider', None)
        self.incomplete_reasons = {}
        self.hits = self.records = self.evictions = 0
        self.empty_face_validation_count = 0
        self.complete_face_validation_count = 0
        self.native_completion_receipt_count = 0
        self.anchor_bound_records = 0
        self.closed = False

    def candidates_at(self, source, seconds):
        before = self.provider.native_faces.inference_frame_count
        result = self.original_candidates(source, seconds)
        # A counter increase proves a raster was decoded, but not that YuNet
        # succeeded. Verify empty results on that same native frame so Haar
        # fallback and absent/failed models never become observed negatives.
        if self.provider.native_faces.inference_frame_count > before:
            completion = getattr(self.provider.native_faces, 'candidate_observation_complete', None)
            if callable(completion):
                self.native_completion_receipt_count += 1
                if completion(source, seconds) is not True:
                    return result
            elif not result:
                self.empty_face_validation_count += 1
                if not _verified_empty_native_detection(self.provider, source, seconds):
                    return result
            else:
                self.complete_face_validation_count += 1
                if not _verified_complete_native_candidates(self.provider, source, seconds, result):
                    return result
            sampler = self.provider.faces._samplers[source]
            key = self.provider.faces._identity_key(source, sampler, seconds)
            self.completed_face_frames[key] = True
            self.completed_face_frames.move_to_end(key)
            if len(self.completed_face_frames) > 4096:
                self.completed_face_frames.popitem(last=False)
        return result

    def _incomplete(self, reason):
        self.incomplete_reasons[reason] = self.incomplete_reasons.get(reason, 0) + 1
        return False

    def _remember(self, key, value):
        self.entries[key] = value
        self.entries.move_to_end(key)
        self.records += 1
        if len(self.entries) > self.maximum_entries:
            self.entries.popitem(last=False)
            self.evictions += 1

    def _fully_observed_failure(self, source, seconds, key, sampler):
        body = self.provider.bodies
        track = body._track_cache.get(key)
        if (track is None or track.observation is not None or track.reason not in {
                'independent_face_body_anchors_missing', 'target_trajectory_unresolved',
                'competing_target_trajectories'}):
            return self._incomplete('unproven_track_failure')
        if len(body._references.get(source, ())) < 3:
            return self._incomplete('body_references_unavailable')
        fps = float(sampler.frame_rate)
        count = sampler.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
            return self._incomplete('native_clock_unavailable')
        # Match the existing tracker's native-frame lattice exactly. Looking
        # up the completed observations below never reads or infers a frame.
        stride = max(1, round(.12 * fps))
        query_frame = sampler.frame_identity(seconds).frame_index
        center = query_frame // stride * stride
        indices = [center + step * stride for step in range(-10, 11)]
        times = [index / fps for index in indices if 0 <= index < count - 1 and index != query_frame]
        times.append(seconds)
        for time in times:
            frame_key = self.provider.faces._identity_key(source, sampler, time)
            frame = body._frame_cache.get(frame_key)
            faces = self.provider.faces._focal_candidate_cache.get(frame_key)
            if frame is None or not frame.observations:
                return self._incomplete('body_frame_unobserved')
            if faces is None or not faces and frame_key not in self.completed_face_frames:
                return self._incomplete('native_face_decode_unproven')
            if any((observation.reference_similarity >= .60
                    or getattr(observation, 'frozen_face_match', False))
                   and len(observation.pose_screen_xyc) != 8
                   for observation in frame.observations):
                return self._incomplete('pose_inference_unproven')
        return True

    def __call__(self, source, seconds, descriptor=None):
        if self.closed:
            raise RuntimeError('source-first session is closed')
        provider = self.provider
        if (isinstance(source, bool) or not isinstance(source, int) or not 0 <= source < len(provider.sources)
                or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            # Preserve the original validation/error semantics.
            return self.original(source, seconds, descriptor)
        if provider._identity(provider.sources[source]) != provider._identities[source]:
            raise OSError('source-first focal source changed during analysis')
        if provider.observation_namespace != self.namespace:
            self.entries.clear()
            self.completed_face_frames.clear()
            self.namespace = provider.observation_namespace
        sampler = provider.faces._samplers[source]
        if sampler is None:
            return self.original(source, seconds, descriptor)
        key = provider.faces._identity_key(source, sampler, seconds)
        if key in self.entries:
            if provider._persistent is not None:
                # Another session can publish a positive observation in the
                # same namespace. A run-local negative must never hide it.
                published = provider._persistent.get(provider._persistent_key(source, key), source, seconds)
                if published is not None and (published.face_bbox_xywh is not None or published.person is not None):
                    self.entries.pop(key)
                    # The scalar resolver may still hold the original missing
                    # result. Do not resurrect it on the very next query after
                    # this memo entry has yielded to a published positive.
                    getattr(provider, '_cache', {}).pop(key, None)
                    return published
            self.entries.move_to_end(key)
            self.hits += 1
            return replace(self.entries[key], source_seconds=seconds)
        prefilter = getattr(provider, '_source_first_anchor_prefilter', None)
        if (key not in getattr(provider, '_cache', {}) and prefilter is not None
                and prefilter.query_cannot_resolve(source, seconds)):
            # A concurrently published positive must outrank a new negative
            # certificate just as it does an existing run-local memo entry.
            if provider._persistent is not None:
                published = provider._persistent.get(provider._persistent_key(source, key), source, seconds)
                if published is not None:
                    return published
            value = FocalSelectorGeometry(source, seconds, None, pose_backend='rtmpose_body7')
            self._remember(key, value)
            self.anchor_bound_records += 1
            return value
        value = self.original(source, seconds, descriptor)
        if (isinstance(value, FocalSelectorGeometry) and value.source_index == source
                and math.isclose(value.source_seconds, seconds, abs_tol=1e-9, rel_tol=0)
                and value.face_bbox_xywh is None and value.person is None
                and self._fully_observed_failure(source, seconds, key, sampler)):
            self._remember(key, value)
        return value

    def diagnostics(self):
        return dict(contract='run_local_completely_observed_unresolved_query_memo_v1',
                    entries=len(self.entries), maximum_entries=self.maximum_entries,
                    hits=self.hits, records=self.records, evictions=self.evictions,
                    incomplete_window_reasons=dict(self.incomplete_reasons),
                    empty_native_detection_validation_count=self.empty_face_validation_count,
                    complete_native_detection_validation_count=self.complete_face_validation_count,
                    native_completion_receipt_count=self.native_completion_receipt_count,
                    complete_anchor_bound_records=self.anchor_bound_records,
                    negative_observations_persisted=False, render_authorized=False)

    def provider_diagnostics(self):
        return {**self.original_diagnostics(), 'unresolved_query_memo': self.diagnostics()}

    def close(self):
        if not self.closed:
            self.closed = True
            self.entries.clear()
            self.completed_face_frames.clear()
            self.original_close()


def install_source_first_unresolved_memo(provider):
    if isinstance(getattr(provider, '_source_first_unresolved_memo', None), SourceFirstUnresolvedMemo):
        return
    memo = SourceFirstUnresolvedMemo(provider)
    provider._source_first_unresolved_memo = memo
    provider.selector_geometry = memo
    if callable(memo.original_candidates):
        provider.faces._external_focal_candidate_provider = memo.candidates_at
    provider.diagnostics = memo.provider_diagnostics
    provider.close = memo.close
