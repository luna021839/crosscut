"""One target-observation authority shared before and after path selection.

This replaces independent dominant-person decisions in MEMBER proposal and
dense camera stages. Raw descriptor caches remain unchanged: the target's
native face, source-local body track and RTMPose joints have a separate basis.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import replace
from importlib.metadata import version
import hashlib
import inspect
import json
import math
from pathlib import Path

from .body_appearance import BodyAppearanceEncoder
from .body_pose_2d import BodyPose2DAnalyzer, BodyPose2DEvidence
from .direct_member_editorial_motion import DirectMemberDenseFaceProvider, DirectMemberFaceSample
from .direct_member_target_face import FOCAL_FACE_BASIS
from .focal_body_track_provider import SourceLocalFocalBodyProvider, select_direct_focal_observation
from .focal_body_track import FocalBodyGeometryCertificate
from .focal_selector_geometry import FocalSelectorGeometry, FOCAL_SELECTOR_BODY_BASIS
from .native_focal_faces import NativeFocalFaceProvider
from .source_first_published_cache import SourceFirstPublishedGeometryCache as SourceFirstGeometryCache


class SourceFirstFocalGeometryProvider:
    def __init__(self, sources, *, body_model_path: Path, pose_model_path: Path,
                 feature_cache_directory=None, feature_derived_cache_directory=None,
                 require_persistent_feature_cache=False, body_execution_device='cpu',
                 share_source_frames=True):
        from .shared_source_frames import SharedSourceFrames
        if not isinstance(share_source_frames, bool):
            raise TypeError('share_source_frames must be boolean')
        from .source_first_execution import resolve_body_execution_device, body_observation_namespace
        self.body_execution_device = resolve_body_execution_device(body_execution_device)
        self.sources = tuple(Path(p) for p in sources)
        self._identities = tuple(self._identity(p) for p in self.sources)
        with ExitStack() as cleanup:
            self.shared_source_frames = None
            if share_source_frames:
                shared = []
                for source in self.sources:
                    owner = SharedSourceFrames(source)
                    cleanup.callback(owner.close)
                    shared.append(owner)
                self.shared_source_frames = tuple(shared)
            self.native_faces = NativeFocalFaceProvider(self.sources,
                shared_source_frames=self.shared_source_frames)
            cleanup.callback(self.native_faces.close)
            self.faces = DirectMemberDenseFaceProvider(self.sources, require_focal_consensus=True,
                focal_candidate_provider=self.native_faces.candidates_at,
                feature_cache_directory=feature_cache_directory,
                feature_derived_cache_directory=feature_derived_cache_directory,
                require_persistent_feature_cache=require_persistent_feature_cache,
                shared_source_frames=self.shared_source_frames)
            cleanup.callback(self.faces.close)
            self.pose = BodyPose2DAnalyzer(pose_model_path)
            if self.body_execution_device == 'mps':
                from .mps_body_appearance import MPSBodyAppearanceEncoder
                self.encoder = MPSBodyAppearanceEncoder(body_model_path)
            else:
                self.encoder = BodyAppearanceEncoder(body_model_path)
            self.bodies = SourceLocalFocalBodyProvider(self.faces, self.encoder, pose_analyzer=self.pose)
            self._cache = OrderedDict()
            self._boundary_references = []
            self._reasons = {}
            self.observation_namespace = body_observation_namespace(
                self.observation_fingerprint(), self.encoder, self.body_execution_device)
            self._persistent = (SourceFirstGeometryCache(feature_cache_directory,self.observation_namespace)
                if feature_cache_directory is not None else None)
            cleanup.pop_all()

    @classmethod
    def observation_fingerprint(cls):
        """Camera policy and path costs do not invalidate measured observations."""
        from .direct_member_editorial_motion import _enable_dense_sequential_frame_reads
        paths=('body_pose_2d.py','body_appearance.py','_osnet_ain_network.py','focal_body_track.py',
            'focal_body_track_provider.py','focal_reference_consensus.py','focal_pose.py',
            'focal_selector_geometry.py','native_focal_faces.py','native_frame_reader.py',
            'seek_locality.py','source_first_geometry_cache.py','shared_source_frames.py')
        components={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in paths}
        for method in (cls.selector_geometry,DirectMemberDenseFaceProvider._ensure_focal_consensus,
                DirectMemberDenseFaceProvider._focal_candidates,_enable_dense_sequential_frame_reads):
            components[method.__qualname__]=hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
        import cv2,numpy
        components.update(opencv=cv2.__version__,numpy=numpy.__version__,
            torch=version('torch'),pillow=version('Pillow'),schema='source_first_focal_observation_v1')
        return hashlib.sha256(json.dumps(components,sort_keys=True).encode()).hexdigest()

    def _persistent_key(self,source_index,key):
        _device,inode,size,mtime=self._identities[source_index]
        value=(str(self.sources[source_index].resolve()),inode,size,mtime,*key[1:])
        return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()

    @staticmethod
    def _identity(path):
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

    def selector_geometry(self, source_index, seconds, descriptor=None):
        if (isinstance(source_index, bool) or not isinstance(source_index, int)
                or not 0<=source_index<len(self.sources) or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds<0):
            raise ValueError('source-first observation must be source/time bound')
        # Even a cache hit must not disguise a disconnected or changed SSD.
        if self._identity(self.sources[source_index]) != self._identities[source_index]:
            raise OSError('source-first focal source changed during analysis')
        sampler = self.faces._samplers[source_index]
        if sampler is None:
            return FocalSelectorGeometry(source_index, seconds, None)
        key = self.faces._identity_key(source_index, sampler, seconds)
        if key in self._cache:
            self._cache.move_to_end(key)
            value = self._cache[key]
            track = value.body_track
            if track is not None:
                if isinstance(track,FocalBodyGeometryCertificate):
                    track=replace(track,source_seconds=seconds)
                else:
                    track = replace(track, source_seconds=seconds,
                        observation=replace(track.observation, source_seconds=seconds))
            return replace(value, source_seconds=seconds, body_track=track)
        persistent_key=self._persistent_key(source_index,key)
        if self._persistent is not None:
            cached=self._persistent.get(persistent_key,source_index,seconds)
            if cached is not None:
                self._cache[key]=cached
                if len(self._cache)>4096:self._cache.popitem(last=False)
                return cached
        bank = self.faces._ensure_focal_consensus(source_index)
        candidates = self.faces._focal_candidates(source_index, seconds)
        face = bank.select(candidates) if bank else None
        frame = self.bodies.frame_observations(source_index, seconds)
        selected, reason = select_direct_focal_observation(face, candidates, frame)
        track = None
        if selected is None and reason != 'frozen_solo_face_body_unobserved':
            recovered = self.bodies.track_at(source_index, seconds)
            if recovered.observation is not None:
                track, selected = recovered, recovered.observation
                face, reason = selected.face, recovered.reason
            else:
                face, reason = None, recovered.reason
        person = None
        screen = ()
        if selected is not None:
            person = selected.person
            screen = selected.pose_screen_xyc
            pixels = sampler.frame_at(seconds)
            if not screen and pixels is not None:
                measured = self.pose.describe(pixels, bbox_xywh=person.bbox_xywh,
                    face_bbox_xywh=face.bbox_xywh if face else None)
                if measured.reason == 'observed':
                    screen = measured.upper_body_xyc
            if screen and pixels is not None:
                measured = BodyPose2DEvidence('observed', person.bbox_xywh,
                    (pixels.shape[1], pixels.shape[0]), screen)
                # No fallback into the other 60D pose feature basis.
                person = measured.as_formation_person(person.detection_confidence) or person
        value = FocalSelectorGeometry(source_index, seconds, face.bbox_xywh if face else None,
            (face.eye_scale or face.landmark_scale) if face else None, person, track,
            'rtmpose_body7', screen)
        self._reasons[reason] = self._reasons.get(reason, 0)+1
        self._cache[key] = value
        # Positive observations can survive a new edit path. Do not persist
        # a transient model/decoder failure as permanent source exclusion.
        if self._persistent is not None and (value.face_bbox_xywh is not None or value.person is not None):
            self._persistent.put(persistent_key,value)
        if len(self._cache)>4096:
            self._cache.popitem(last=False)
        return value

    def sample(self, source_index, seconds):
        value = self.selector_geometry(source_index, seconds)
        if value.face_bbox_xywh is None:
            return self._body_sample(value)
        person = value.person
        return DirectMemberFaceSample(value.face_bbox_xywh, 1, 'yunet',
            person.bbox_xywh if person else None,
            person.detection_confidence if person else 0.,
            'face_associated_formation_person_box' if person else 'none', FOCAL_FACE_BASIS)

    @staticmethod
    def _body_sample(value):
        if value.person is None or (value.face_bbox_xywh is None and value.body_track is None):
            return None
        return DirectMemberFaceSample(value.person.bbox_xywh, 0, 'none',
            value.person.bbox_xywh, value.person.detection_confidence,
            'source_local_focal_body_track', FOCAL_SELECTOR_BODY_BASIS)

    def paired_samples(self, outgoing_source_index, outgoing_seconds, incoming_source_index, incoming_seconds):
        left = self.sample(outgoing_source_index, outgoing_seconds)
        right = self.sample(incoming_source_index, incoming_seconds)
        if left is not None and right is not None and (left.face_count == 0 or right.face_count == 0):
            left = self._body_sample(self.selector_geometry(outgoing_source_index, outgoing_seconds))
            right = self._body_sample(self.selector_geometry(incoming_source_index, incoming_seconds))
        # Capture the exact source-native faces selected by this authority.
        faces = []
        for source, seconds in ((outgoing_source_index, outgoing_seconds),
                (incoming_source_index, incoming_seconds)):
            # Alignment may use a common BODY box on both sides. That does
            # not erase an independently observed FACE on the visible side.
            # Conversely, a neighboring face cannot authorize the hidden side.
            geometry = self.selector_geometry(source, seconds)
            candidates = self.faces._focal_candidates(source, seconds)
            observed = [c for c in candidates
                if geometry.face_bbox_xywh is not None
                and c.bbox_xywh == geometry.face_bbox_xywh]
            face = observed[0] if len(observed) == 1 else None
            faces.append(face)
        self._boundary_references.append((outgoing_source_index, incoming_source_index, *faces))
        return left, right

    def begin_focal_boundary_reference_capture(self):
        self._boundary_references.clear()

    def recover_exact_focal_pair_from_registered_references(self, outgoing_source_index,
            outgoing_seconds, incoming_source_index, incoming_seconds):
        # Compatibility with the app's exact-boundary QA-reference recovery.
        # Never reintroduce the old independent dominant-person decision or
        # substitute an anchor's face box for an occluded query frame.
        pair=[]
        for source,seconds in ((outgoing_source_index,outgoing_seconds),
                (incoming_source_index,incoming_seconds)):
            geometry=self.selector_geometry(source,seconds)
            if geometry.face_bbox_xywh is None:return None
            face=next((c for c in self.faces._focal_candidates(source,seconds)
                if c.bbox_xywh==geometry.face_bbox_xywh),None)
            if face is None:return None
            pair.append(face)
        return tuple(pair)

    def focal_boundary_reference_capture(self):
        return tuple(self._boundary_references)

    def diagnostics(self):
        return {'contract': 'shared_source_first_member_geometry_v1',
            'body_execution_device': getattr(self, 'body_execution_device', 'cpu'),
            'observation_namespace': getattr(self, 'observation_namespace', None),
            'observation_cache_count': len(self._cache), 'reasons': dict(self._reasons),
            'body_frame_inference_count': self.bodies.frame_inference_count,
            'body_frame_cache_hit_count': self.bodies.frame_cache_hit_count,
            'native_face_inference_frame_count': self.native_faces.inference_frame_count,
            'pose': self.pose.fingerprint(), 'body_appearance': self.encoder.fingerprint(),
            'native_face': self.native_faces.fingerprint(),
            'shared_source_frames': [owner.diagnostics() for owner in self.shared_source_frames or ()],
            'persistent_geometry':self._persistent.diagnostics() if self._persistent else None,
            'raw_descriptor_cache_changed': False, 'final_quality_authority': False}

    def close(self):
        try:
            if self._persistent is not None:self._persistent.close()
        finally:
            try:
                self.native_faces.close()
                self.faces.close()
                self._cache.clear()
                self._boundary_references.clear()
                self.bodies._references.clear()
                self.bodies._frame_cache.clear()
                self.bodies._track_cache.clear()
            finally:
                for owner in self.shared_source_frames or ():
                    owner.close()
