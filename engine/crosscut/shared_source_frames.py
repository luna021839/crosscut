"""One exact source decoder for native-face and working-frame consumers.

Only transient pixels are shared. Times are never snapped or interpolated,
and one source cannot lend frames to another. The source-owning session closes
this object after its borrowers; no borrowed sampler closes its decoder.
"""
from collections import OrderedDict
import math
from pathlib import Path
from threading import RLock
from time import monotonic

import cv2

from .seek_locality import SeekLocalityReader, probe_constant_frame_clock
from .visual_score import _working_frame


def validate_shared_sources(sources, shared):
    if shared is None:
        return None
    shared = tuple(shared)
    if len(shared) != len(sources):
        raise ValueError('shared decoder count does not match sources')
    for source, owner in zip(sources, shared, strict=True):
        if not isinstance(owner, SharedSourceFrames) or Path(source).resolve(strict=True) != owner.source:
            raise ValueError('shared decoder belongs to another source')
        owner.assert_source_available()
    return shared


class SharedSourceFrames:
    def __init__(self, source: Path, *, maximum_working_frames: int = 128):
        if (isinstance(maximum_working_frames, bool) or not isinstance(maximum_working_frames, int)
                or not 1 <= maximum_working_frames <= 128):
            raise ValueError('shared working-frame capacity must be in [1, 128]')
        self._requested_source = Path(source).absolute()
        self.source = self._requested_source.resolve(strict=True)
        if not self.source.is_file():
            raise ValueError('shared source must be a file')
        self._identity = self._source_identity()
        self.maximum_working_frames = maximum_working_frames
        self._working = OrderedDict()
        self._capture = None
        self.reader = None
        self._last_native = self._last_working = None
        self._lock = RLock()
        self.closed = False
        self.native_requests = self.working_requests = self.working_cache_hits = 0
        self.working_materializations = 0
        self.working_seconds = 0.

    def _source_identity(self):
        stat = self.source.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def assert_source_available(self):
        if self.closed:
            raise RuntimeError('shared source decoder is closed')
        if (self._requested_source.resolve(strict=True) != self.source
                or self._source_identity() != self._identity):
            raise OSError('shared source changed during analysis')

    def _time(self, seconds):
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError('shared frame needs an exact nonnegative source time')
        self.assert_source_available()
        return float(seconds)

    def native_at(self, seconds):
        with self._lock:
            key = self._time(seconds)
            self.native_requests += 1
            if self.reader is None:
                capture = cv2.VideoCapture(str(self.source))
                if not capture.isOpened():
                    capture.release()
                    raise OSError('shared source decoder could not be opened')
                try:
                    reader = SeekLocalityReader(capture, clock=probe_constant_frame_clock(self.source),
                        process_frame=lambda frame: frame, maximum_cached_frames=1)
                except BaseException:
                    capture.release()
                    raise
                self._capture, self.reader = capture, reader
            native = self.reader.frame_at(key)
            if native is not None:
                if key not in self._working:
                    if native is not self._last_native:
                        started = monotonic()
                        try:
                            self._last_working = _working_frame(native)
                        finally:
                            self.working_seconds += monotonic()-started
                        self._last_native = native
                        self.working_materializations += 1
                    self._working[key] = self._last_working
                self._working.move_to_end(key)
                while len(self._working) > self.maximum_working_frames:
                    self._working.popitem(last=False)
            return native

    def working_at(self, seconds):
        with self._lock:
            key = self._time(seconds)
            self.working_requests += 1
            if key in self._working:
                self.working_cache_hits += 1
                self._working.move_to_end(key)
                return self._working[key]
            # native_at also publishes the corresponding exact working pixels.
            self.native_at(key)
            return self._working.get(key)

    def diagnostics(self):
        reader = self.reader
        return dict(contract='shared_native_and_working_source_frames_v1', closed=self.closed,
            initialized=reader is not None, native_requests=self.native_requests,
            working_requests=self.working_requests, working_cache_hits=self.working_cache_hits,
            working_materializations=self.working_materializations,
            working_seconds=self.working_seconds,
            working_cache_entries=len(self._working), maximum_working_frames=self.maximum_working_frames,
            read_count=reader.read_count if reader else 0,
            seek_count=reader.seek_count if reader else 0,
            fallback_count=reader.fallback_count if reader else 0,
            enabled=bool(reader and reader.enabled), raw_pixels_persisted=False)

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed = True
                if self._capture is not None:
                    self._capture.release()
                if self.reader is not None:
                    self.reader.frames.clear()
                self._working.clear()
                self._last_native = self._last_working = None
