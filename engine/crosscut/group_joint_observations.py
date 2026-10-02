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
        self.projected=JointProjectedPixelCache(budget);self.models={};self.model_adapters={};self.geometry={}
        self.model_asset_dependencies={}
        self.model_adapter_preimages={}
        self.model_observations=OrderedDict();self.observation_bytes=0
        self.observation_limit=maximum_model_observation_bytes;self.model_reuse=0
        self.cache_reuse=0;self.closed=False
        self.raw_peak_bytes=0;self.raw_evictions=0;self.native_frame_bytes={}
        self.peak_owned_raw_decode_bytes=0;self.native_pixel_copies=0

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
        expected=self.native_frame_bytes.get(source,0)
        # Evict before the allocation, not after briefly inserting a third
        # 4K raster into a 64MiB cache. Unknown first-source dimensions are
        # reported honestly; pair-streaming callers release obsolete frames.
        while self.raw and self.raw_bytes+expected>self.raw_limit:
            _,old=self.raw.popitem(last=False)
            self.raw_bytes-=0 if old is None else old.nbytes;self.raw_evictions+=1
        frame=self.readers[source].frame_at(native_frame)
        self._assert_source_identity(source)
        if frame is not None:
            size=frame.nbytes
            self.peak_owned_raw_decode_bytes=max(self.peak_owned_raw_decode_bytes,self.raw_bytes+size)
            # OpenCV owns each returned ndarray and ExactNativeFrameReader
            # retains no raster. A view needs a private copy; an owned buffer
            # can be frozen without allocating another entire 4K frame.
            if not frame.flags.owndata:
                self.peak_owned_raw_decode_bytes=max(self.peak_owned_raw_decode_bytes,self.raw_bytes+2*size)
                frame=frame.copy();self.native_pixel_copies+=1
            frame.setflags(write=False)
            self.native_frame_bytes[source]=size
        size=0 if frame is None else frame.nbytes
        if size<=self.raw_limit:
            self.raw[key]=frame;self.raw_bytes+=size
            while self.raw_bytes>self.raw_limit or len(self.raw)>1200:
                _,old=self.raw.popitem(last=False);self.raw_bytes-=0 if old is None else old.nbytes
                self.raw_evictions+=1
            self.raw_peak_bytes=max(self.raw_peak_bytes,self.raw_bytes)
        return frame

    def retain_native_pixels(self,keys):
        """Drop obsolete raster cache entries, not model/raw-feature evidence.

        Explicit source/native keys are checked without opening a decoder.
        Later needs are real counted reads; no reference is relabelled or
        silently substituted. Callers must not retain unbounded raster copies.
        """
        self.budget.check()
        if self.closed or not isinstance(keys,tuple) or len(keys)>1200:
            raise ValueError('native raster retention requires a live bounded key set')
        wanted=set()
        for key in keys:
            if not isinstance(key,tuple) or len(key)!=2 or any(type(n) is not int for n in key):
                raise ValueError('native raster retention key is invalid')
            source,native=key
            clock=self.domain.render_frame_clocks[self.domain.eligible_index(source)]
            if not 0<=native<clock.frame_count:
                raise ValueError('native raster retention key is outside its clock')
            wanted.add(key)
        for key in tuple(self.raw):
            if (key[0],key[3]) not in wanted:
                old=self.raw.pop(key);self.raw_bytes-=0 if old is None else old.nbytes;self.raw_evictions+=1

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

    def register_model(self,name,asset_path,observer,*,adapter_fingerprint=None,asset_dependencies=(),
            expected_asset_sha256=None):
        """The worker loads each actual model once and supplies its observer."""
        self.budget.check()
        if not isinstance(name,str) or not name or name in self.models or not callable(observer):
            raise ValueError('joint model needs one explicit worker-owned name/observer')
        if expected_asset_sha256 is not None and (not isinstance(expected_asset_sha256,str)
                or len(expected_asset_sha256)!=64
                or any(c not in '0123456789abcdef' for c in expected_asset_sha256)):
            raise ValueError('joint model loaded-asset identity must be a SHA256 digest')
        identity=_file_identity(asset_path)
        with asset_path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if _file_identity(asset_path)!=identity:raise ValueError('joint model changed while binding its content identity')
        if expected_asset_sha256 is not None and digest!=expected_asset_sha256:
            raise ValueError('joint model asset differs from the bytes loaded by its observer')
        dependencies=tuple(asset_dependencies)
        if (len(dependencies)>16 or len(set(dependencies))!=len(dependencies)
                or asset_path in dependencies):
            raise ValueError('joint model external assets must be explicit, distinct and bounded')
        bound=[]
        for dependency in dependencies:
            self.budget.check()
            before=_file_identity(dependency)
            with dependency.open('rb') as stream:
                dependency_sha=hashlib.file_digest(stream,'sha256').hexdigest()
            if _file_identity(dependency)!=before:
                raise ValueError('joint model external asset changed while binding')
            bound.append((dependency,before,dependency_sha))
        if bound:
            adapter_fingerprint={'adapter':adapter_fingerprint,'external_asset_dependencies':[
                {'filename':path.name,'sha256':sha} for path,_,sha in bound]}
        encoded=json.dumps(adapter_fingerprint,sort_keys=True,separators=(',',':'),allow_nan=False)
        if _file_identity(asset_path)!=identity:
            raise ValueError('joint model changed while binding external assets')
        if any(_file_identity(path)!=before for path,before,_ in bound):
            raise ValueError('joint model external asset changed while binding')
        self.models[name]=(asset_path,identity,digest,observer)
        self.model_asset_dependencies[name]=tuple(bound)
        self.model_adapters[name]=hashlib.sha256(encoded.encode()).hexdigest()
        self.model_adapter_preimages[name]=json.loads(encoded)

    def _assert_model_assets(self,name):
        path,identity,_,_=self.models[name]
        if _file_identity(path)!=identity:raise ValueError('joint observation model identity changed')
        if any(_file_identity(path)!=identity for path,identity,_ in self.model_asset_dependencies.get(name,())):
            raise ValueError('joint observation model external asset identity changed')

    def observe(self,state,output_tick,*,model,arguments=None):
        if model not in self.models:raise ValueError('joint observation model is not owned by this worker')
        path,identity,_,observer=self.models[model]
        self._assert_model_assets(model)
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
        key=(state.key,output_tick,source_identity,model,self.models[model][2],
            self.model_adapters[model],encoded_arguments)
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
        self._assert_model_assets(model)
        self._assert_source_identity(source)
        payload=_encode_message(value)
        if len(payload)<=self.observation_limit:
            self.model_observations[key]=payload;self.observation_bytes+=len(payload)
            while self.observation_bytes>self.observation_limit or len(self.model_observations)>2400:
                _,old=self.model_observations.popitem(last=False);self.observation_bytes-=len(old)
        return value

    def observe_native(self,source,native_frame,*,model,arguments=None):
        """Camera-independent raw source measurement, shared by all cameras.

        A projected detection's enumeration is not a native subject track.
        This separate cache binds file/native clock/model/ROI BEFORE camera
        search and cannot borrow projected pixels or another body's pose.
        """
        if model not in self.models:
            raise ValueError('native observation model is not owned by this worker')
        path,identity,digest,observer=self.models[model]
        self._assert_model_assets(model)
        source_identity=self._assert_source_identity(source)
        local=self.domain.eligible_index(source)
        clock=self.domain.render_frame_clocks[local]
        if type(native_frame) is not int or not 0<=native_frame<clock.frame_count:
            raise ValueError('native observation is outside its verified source clock')
        if arguments is not None and (not isinstance(arguments,dict)
                or any(not isinstance(name,str) for name in arguments)):
            raise ValueError('native model arguments must be explicit named JSON-native values')
        encoded=json.dumps(arguments,sort_keys=True,separators=(',',':'),allow_nan=False)
        bound_arguments=json.loads(encoded)
        key=('source_native',source,source_identity,clock.packet_grid_sha256,native_frame,model,digest,
            self.model_adapters[model],encoded)
        if key in self.model_observations:
            payload=self.model_observations.pop(key);self.model_observations[key]=payload;self.model_reuse+=1
            return pickle.loads(payload)
        pixels=self.native_pixels(source,native_frame)
        if pixels is None:return None
        self.budget.spend('model_calls',model=model)
        value=observer(pixels) if bound_arguments is None else observer(pixels,**bound_arguments)
        self.budget.check()
        self._assert_model_assets(model)
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
            maximum_raw_pixel_cache_bytes=self.raw_limit,peak_raw_pixel_cache_bytes=self.raw_peak_bytes,
            raw_pixel_cache_eviction_count=self.raw_evictions,native_pixel_copy_count=self.native_pixel_copies,
            peak_owned_raw_raster_bytes_during_decode=self.peak_owned_raw_decode_bytes,
            raw_decode_peak_excludes_codec_model_and_caller_retained_memory=True,process_peak_rss_measured=False,
            projected_pixel_cache_bytes=self.projected.bytes,native_pixel_cache_reuse=self.cache_reuse,
            source_readers={str(s):r.diagnostics() for s,r in self.readers.items()},
            model_content_sha256={name:row[2] for name,row in self.models.items()},
            model_external_asset_sha256={name:{path.name:sha for path,_,sha in rows}
                for name,rows in self.model_asset_dependencies.items() if rows},
            model_adapter_fingerprints=dict(self.model_adapters),
            model_adapter_preimages=json.loads(json.dumps(self.model_adapter_preimages)),
            reused_model_observations=self.model_reuse,model_observation_serialized_cache_bytes=self.observation_bytes,
            counted_decodes='materialized_exact_native_requests_not_hidden_codec_keyframe_work',
            persistent_raw_feature_caches_modified=False,budget=self.budget.to_dict())

    def close(self):
        for capture in self.captures.values():capture.release()
        self.captures.clear();self.readers.clear();self.raw.clear();self.raw_bytes=0
        self.projected.rows.clear();self.projected.bytes=0;self.geometry.clear();self.models.clear();self.model_adapters.clear()
        self.model_adapter_preimages.clear()
        self.model_asset_dependencies.clear()
        self.native_frame_bytes.clear()
        self.model_observations.clear();self.observation_bytes=0
        self.closed=True

    def __enter__(self):return self

    def __exit__(self,*_):self.close()
