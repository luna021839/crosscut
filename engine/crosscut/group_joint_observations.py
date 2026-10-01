"""One worker-owned native/pixel/model session for joint GROUP state sensing.

Sources are lazy and eligible-only. Exact verified native IDs and file identity
are checked before reads/cache hits and after reads. Projection has its own
whole-camera byte LRU and never modifies raw or persistent feature caches.
No callback or model object is sent across an optional-worker boundary. All
native misses remain unavailable, without neighboring-frame fallback.
"""
from collections import OrderedDict
import hashlib
import json
import pickle
import stat
from types import SimpleNamespace

import cv2

from .exact_native_frame_reader import ExactNativeFrameReader
from .group_joint_source_camera import JointGroupWorkBudget,JointProjectedPixelCache
from .group_source_domain import GroupSourceDomain
from .prescribed_render_geometry import PrescribedGeometrySampler
from .post_render_perceptual_quality import _reference_view_for_span
from .render_frame_schedule import probe_render_frame_clock
from .optional_native_worker import _encode_message


def _file_identity(path):
    value=path.stat()
    if not stat.S_ISREG(value.st_mode):raise ValueError('joint observation source is not a regular file')
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


class GroupJointObservationSession:
    def __init__(self,*,domain,budget,maximum_raw_cache_bytes=64*1024*1024,
            maximum_model_observation_bytes=8*1024*1024):
        if (not isinstance(domain,GroupSourceDomain) or not isinstance(budget,JointGroupWorkBudget)
                or any(type(n) is not int or n<1 for n in (maximum_raw_cache_bytes,maximum_model_observation_bytes))):
            raise ValueError('joint observation session needs bound sources and finite cache/work budgets')
        self.domain=domain;self.budget=budget;self.raw_limit=maximum_raw_cache_bytes
        self.captures={};self.readers={};self.identities={};self.raw=OrderedDict();self.raw_bytes=0
        self.projected=JointProjectedPixelCache(budget);self.models={};self.geometry={}
        self.model_observations=OrderedDict();self.observation_bytes=0
        self.observation_limit=maximum_model_observation_bytes;self.model_reuse=0
        self.cache_reuse=0;self.closed=False

    def _assert_source_identity(self,source):
        self.budget.check()
        if self.closed:raise ValueError('joint observation session is closed')
        local=self.domain.eligible_index(source)
        path=self.domain.request_sources[source]
        identity=_file_identity(path)
        if source not in self.identities:
            self.identities[source]=identity
            capture=None
            try:
                clock=probe_render_frame_clock(path,
                    timeout_seconds=max(.01,min(10.,self.budget.deadline-self.budget.clock())))
                self.budget.check()
                if clock!=self.domain.render_frame_clocks[local] or _file_identity(path)!=identity:
                    raise ValueError('joint observation native clock/file changed before decoder opening')
                capture=cv2.VideoCapture(str(path))
                if not capture.isOpened():raise ValueError('joint eligible source decoder unavailable')
                reader=ExactNativeFrameReader(capture,clock)
                if _file_identity(path)!=identity:raise ValueError('joint source changed during decoder opening')
                self.captures[source]=capture;self.readers[source]=reader
            except BaseException:
                if capture is not None:capture.release()
                self.identities.pop(source,None)
                raise
        elif identity!=self.identities[source]:
            raise ValueError('joint source identity changed; cached/native observations cannot be reused')
        return identity

    def native_pixels(self,source,native_frame):
        identity=self._assert_source_identity(source)
        local=self.domain.eligible_index(source);clock=self.domain.render_frame_clocks[local]
        if type(native_frame) is not int or not 0<=native_frame<clock.frame_count:
            raise ValueError('joint observation native frame is outside the verified source')
        key=(source,identity,clock.packet_grid_sha256,native_frame)
        if key in self.raw:
            value=self.raw.pop(key);self.raw[key]=value;self.cache_reuse+=1;return value
        # Count materialized native reads, not invisible codec/keyframe work.
        # ExactNativeFrameReader separately exposes skipped RGB grabs and seeks.
        self.budget.spend('new_decodes')
        frame=self.readers[source].frame_at(native_frame)
        self._assert_source_identity(source)
        if frame is not None:
            frame=frame.copy();frame.setflags(write=False)
        size=0 if frame is None else frame.nbytes
        if size<=self.raw_limit:
            self.raw[key]=frame;self.raw_bytes+=size
            while self.raw_bytes>self.raw_limit or len(self.raw)>1200:
                _,old=self.raw.popitem(last=False);self.raw_bytes-=0 if old is None else old.nbytes
        return frame

    def projected_pixels(self,state,output_tick):
        state.bind_source(self.domain)
        if type(output_tick) is not int or not state.first_output_frame<=output_tick<state.end_output_frame:
            raise ValueError('joint output observation is outside its complete camera state')
        source=self.domain.request_index(state.eligible_source_index)
        identity=self._assert_source_identity(source)
        offset=output_tick-state.first_output_frame
        native=state.native_frame_indices[offset]
        def observe():
            frame=self.native_pixels(source,native)
            if frame is None:return None
            relative=offset;index=0
            for count in state.segment_output_frame_counts:
                if relative<count:break
                relative-=count;index+=1
            key=(state.key,index)
            if key not in self.geometry:
                segment=state.segments[index]
                self.geometry[key]=PrescribedGeometrySampler(SimpleNamespace(frame_rate=float(
                    self.domain.render_frame_clocks[state.eligible_source_index].frame_rate)),segment,
                    master_start=segment.master_start)
            return _reference_view_for_span(frame,self.geometry[key].span,
                relative_time=relative/state.output_fps,size=state.output_size)
        pixels=self.projected.get_or_observe(state=state,output_tick=output_tick,
            raw_identity=(source,identity,native),observer=observe)
        self._assert_source_identity(source)
        return pixels

    def register_model(self,name,asset_path,observer):
        """The worker loads each actual model once and supplies its observer."""
        self.budget.check()
        if not isinstance(name,str) or not name or name in self.models or not callable(observer):
            raise ValueError('joint model needs one explicit worker-owned name/observer')
        identity=_file_identity(asset_path)
        with asset_path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if _file_identity(asset_path)!=identity:raise ValueError('joint model changed while binding its content identity')
        self.models[name]=(asset_path,identity,digest,observer)

    def observe(self,state,output_tick,*,model,arguments=None):
        if model not in self.models:raise ValueError('joint observation model is not owned by this worker')
        path,identity,_,observer=self.models[model]
        if _file_identity(path)!=identity:raise ValueError('joint observation model identity changed')
        state.bind_source(self.domain)
        if type(output_tick) is not int or not state.first_output_frame<=output_tick<state.end_output_frame:
            raise ValueError('model observation is outside its complete camera state')
        source=self.domain.request_index(state.eligible_source_index)
        source_identity=self._assert_source_identity(source)
        if arguments is not None and (not isinstance(arguments,dict)
                or any(not isinstance(name,str) for name in arguments)):
            raise ValueError('joint model arguments must be explicit JSON-native named values')
        # One frame may have multiple top-down bodies. Their ROI arguments
        # must not collide and lend one dancer's cached pose to another.
        encoded_arguments=json.dumps(arguments,sort_keys=True,separators=(',',':'),allow_nan=False)
        bound_arguments=json.loads(encoded_arguments)
        key=(state.key,output_tick,source_identity,model,self.models[model][2],encoded_arguments)
        if key in self.model_observations:
            payload=self.model_observations.pop(key);self.model_observations[key]=payload;self.model_reuse+=1
            # Private in-process data from this registered worker observer only.
            # Never deserialize downloaded files, Pro material or shared caches.
            return pickle.loads(payload)
        pixels=self.projected_pixels(state,output_tick)
        if pixels is None:return None
        self.budget.spend('model_calls',model=model)
        value=observer(pixels) if bound_arguments is None else observer(pixels,**bound_arguments)
        self.budget.check()
        if _file_identity(path)!=identity:raise ValueError('joint model changed while observing projected pixels')
        self._assert_source_identity(source)
        payload=_encode_message(value)
        if len(payload)<=self.observation_limit:
            self.model_observations[key]=payload;self.observation_bytes+=len(payload)
            while self.observation_bytes>self.observation_limit or len(self.model_observations)>2400:
                _,old=self.model_observations.popitem(last=False);self.observation_bytes-=len(old)
        return value

    def diagnostics(self):
        return dict(contract='worker_owned_eligible_native_and_projected_observations_v1',
            opened_request_sources=sorted(self.captures),raw_pixel_cache_bytes=self.raw_bytes,
            projected_pixel_cache_bytes=self.projected.bytes,native_pixel_cache_reuse=self.cache_reuse,
            source_readers={str(s):r.diagnostics() for s,r in self.readers.items()},
            model_content_sha256={name:row[2] for name,row in self.models.items()},
            reused_model_observations=self.model_reuse,model_observation_serialized_cache_bytes=self.observation_bytes,
            counted_decodes='materialized_exact_native_requests_not_hidden_codec_keyframe_work',
            persistent_raw_feature_caches_modified=False,budget=self.budget.to_dict())

    def close(self):
        for capture in self.captures.values():capture.release()
        self.captures.clear();self.readers.clear();self.raw.clear();self.raw_bytes=0
        self.projected.rows.clear();self.projected.bytes=0;self.geometry.clear();self.models.clear()
        self.model_observations.clear();self.observation_bytes=0
        self.closed=True

    def __enter__(self):return self

    def __exit__(self,*_):self.close()
