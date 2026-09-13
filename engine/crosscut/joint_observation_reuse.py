"""Bounded, non-persistent pixel observations across fresh joint searches.

Only successful numeric observations survive. Decisions, no-goods, optimizer
states and renderer proofs still belong to each fresh provider. Publication
participates in the base provider's transaction journal; aborted measurements
cannot seed the next search. The base provider/cache fingerprint is unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import pickle
import sys
import zlib
from typing import Any

from .joint_boundary_provider import (
    ProgressDecodedBoundaryOptionProvider, _source_runtime_stat_identity,
    _SCALAR_BRIDGE_CACHE_NAMESPACE, _SCALAR_BRIDGE_MAXIMUM_ENTRIES_PER_SHARD,
)
from .joint_scalar_reader_compatibility import previous_reader_scalar_analyzer
from .source_scalar_cache import PersistentSourceScalarCache


class JointObservationBank:
    """One analysis-run RAM bank, never an artifact or a source of QA passes."""

    def __init__(self, *, maximum_bytes: int = 256 * 1024 * 1024,
                 maximum_entries: int = 100_000) -> None:
        if any(type(value) is not int or value < 1
               for value in (maximum_bytes, maximum_entries)):
            raise ValueError("observation limits must be positive integers")
        self.maximum_bytes = maximum_bytes
        self.maximum_entries = maximum_entries
        self._values: OrderedDict[bytes, tuple[bytes, int]] = OrderedDict()
        self.accounted_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def lookup(self, key: tuple) -> tuple[bool, Any]:
        digest = hashlib.sha256(pickle.dumps(key, protocol=5)).digest()
        record = self._values.get(digest)
        if record is None:
            self.misses += 1
            return False, None
        self.hits += 1
        # FIFO admissions are deliberate: reads inside an aborted provider
        # transaction must not change future eviction order.
        # These bytes are created only by publish() in this process. There is
        # deliberately no file/network loader or persistent pickle surface.
        return True, pickle.loads(zlib.decompress(record[0]))

    def publish(self, key: tuple, value: Any) -> None:
        digest = hashlib.sha256(pickle.dumps(key, protocol=5)).digest()
        if digest in self._values:
            return
        # Python float/tuple overhead made a 256 MiB raw-object bank retain only
        # a small tail of a full search. Compress numeric observations in RAM;
        # byte accounting includes object and ordered-map overhead per entry.
        encoded = zlib.compress(pickle.dumps(value, protocol=5), level=1)
        size = sys.getsizeof(digest) + sys.getsizeof(encoded) + 256
        if size > self.maximum_bytes:
            return
        while self._values and (
            self.accounted_bytes + size > self.maximum_bytes
            or len(self._values) >= self.maximum_entries
        ):
            _, (_, removed_size) = self._values.popitem(last=False)
            self.accounted_bytes -= removed_size
            self.evictions += 1
        self._values[digest] = encoded, size
        self.accounted_bytes += size

    def diagnostics(self) -> dict[str, Any]:
        return dict(contract="joint_process_local_observation_reuse_v1",
                    entry_count=len(self._values), accounted_bytes=self.accounted_bytes,
                    maximum_bytes=self.maximum_bytes, maximum_entries=self.maximum_entries,
                    hits=self.hits, misses=self.misses, evictions=self.evictions,
                    encoding="compressed_numeric_observations_process_memory_only",
                    persistent=False, raw_pixels_retained=False,
                    renderer_proofs_reused=False, final_media_authority=False)


class ReusingDecodedBoundaryOptionProvider(ProgressDecodedBoundaryOptionProvider):
    def __init__(self, *, observation_bank: JointObservationBank, **kwargs: Any) -> None:
        if not isinstance(observation_bank, JointObservationBank):
            raise TypeError("observation_bank must be a JointObservationBank")
        super().__init__(**kwargs)
        self._observation_bank = observation_bank
        self._pending_observations: dict[tuple, Any] = {}
        self._legacy_scalar_entries_admitted = 0
        self._legacy_scalar_conflicts = 0
        self._containment_observation_hits = 0
        self._containment_observation_misses = 0
        self._observation_namespace = (
            self._configuration_identity, self._source_runtime_stat_identities,
        )

    def _observation_key(self, kind, source_index, seconds, *geometry):
        self._require_open()
        # Stat and persistent-storage checks apply even on RAM hits, including
        # SSD removal, replacement and in-place source edits during a run.
        current = _source_runtime_stat_identity(self.sources[source_index])
        expected = self._source_runtime_stat_identities[source_index]
        if current != expected:
            raise RuntimeError("joint observation source changed during selection")
        if current is None and self._descriptor_provider is None:
            raise OSError("joint observation source is unavailable")
        self._assert_required_persistent_cache_available(source_index)
        return (kind, self._observation_namespace, source_index,
                float(seconds), *geometry)

    def _remember_observation(self, key, value):
        # Missing descriptors/identities must remain missing, not become a
        # cached permanent exclusion after a transient decoder/model failure.
        if value is None or value == ():
            return
        if self._active_transaction_journal is None:
            self._observation_bank.publish(key, value)
        else:
            self._transactional_set(self._pending_observations, key, value)

    def _sample_descriptor(self, source_index, seconds):
        if seconds < 0:
            return super()._sample_descriptor(source_index, seconds)
        key = self._observation_key("raw", source_index, seconds)
        hit, value = self._observation_bank.lookup(key)
        if hit:
            self.raw_descriptor_cache_hit_count += 1
            return value
        value = super()._sample_descriptor(source_index, seconds)
        self._remember_observation(key, value)
        return value

    def _sample_transformed_descriptor(self, source_index, seconds, raw_descriptor, transform):
        if self._descriptor_provider is None and self._transformed_descriptor_provider is None:
            # Production remeasurement depends on decoded pixels and crop/roll,
            # not the boundary proof/track label or the supplied raw descriptor.
            # Keep exact clocks and geometry; no approximate crop/time aliasing.
            # The namespace still binds the provider, model and source context.
            key = self._observation_key(
                "production-transformed-raster-v1", source_index, seconds,
                transform.crop_xywh, float(transform.rotation_degrees),
            )
        else:
            # Injected providers may legitimately interpret every field.
            key = self._observation_key("transformed", source_index, seconds,
                                        raw_descriptor, transform)
        hit, value = self._observation_bank.lookup(key)
        if hit:
            self.transformed_descriptor_cache_hit_count += 1
            return value
        value = super()._sample_transformed_descriptor(
            source_index, seconds, raw_descriptor, transform)
        self._remember_observation(key, value)
        return value

    def _sample_face_identity(self, source_index, seconds, descriptor):
        key = self._observation_key("face", source_index, seconds, descriptor)
        hit, value = self._observation_bank.lookup(key)
        if hit:
            self.face_identity_cache_hit_count += 1
            return value
        value = super()._sample_face_identity(source_index, seconds, descriptor)
        self._remember_observation(key, value)
        return value

    def _cut_handle_containment(self, **arguments):
        resolver = arguments["descriptor_resolver"]
        # Reuse only the production selector's numeric tracking observations.
        # Injected observations and full-proof rehydration callbacks retain
        # their own execution; this is not a renderer-proof cache.
        if (self._descriptor_provider is not None
                or self._containment_frame_provider is not None
                or getattr(resolver, "__self__", None) is not self
                or getattr(resolver, "__func__", None)
                is not ReusingDecodedBoundaryOptionProvider._sample_descriptor):
            return super()._cut_handle_containment(**arguments)
        key = self._observation_key(
            "production-cut-handle-numeric-observations-v1",
            arguments["source_index"], arguments["start_master_seconds"],
            tuple(arguments["sample_times"]),
            tuple(arguments["required_master_sample_times"]),
            arguments["entry_transform"].subject_basis,
            arguments["exit_transform"].subject_basis,
            tuple(arguments["editorial_track"].keyframes),
        )
        hit, samples = self._observation_bank.lookup(key)
        if hit:
            self._containment_observation_hits += 1
            return dict(samples), None
        self._containment_observation_misses += 1
        observations, failure = super()._cut_handle_containment(**arguments)
        if failure is None and observations:
            # Freeze the mapping before the caller can mutate it. Samples are
            # frozen numeric dataclasses; no frames or flow keypoints survive.
            self._remember_observation(key, tuple(sorted(observations.items())))
        return observations, failure

    def _load_scalar_bridge_persistent_cache(self, segment_index):
        was_loaded = segment_index in self._scalar_bridge_persistent_cache_loaded_segments
        super()._load_scalar_bridge_persistent_cache(segment_index)
        if was_loaded or segment_index not in self._scalar_bridge_persistent_cache_loaded_segments:
            return
        current_cache = self._scalar_bridge_persistent_caches[segment_index]
        legacy_analyzer = previous_reader_scalar_analyzer(current_cache.analyzer)
        if legacy_analyzer is None:
            return
        direct_root = self.feature_derived_cache_directory
        legacy_cache = PersistentSourceScalarCache(
            source_cache_directory=direct_root, source_path=self.sources[0],
            namespace=_SCALAR_BRIDGE_CACHE_NAMESPACE, analyzer=legacy_analyzer,
            derived_cache_directory=direct_root,
        )
        legacy = self._decode_scalar_bridge_shard_payload(
            legacy_cache.get(), segment_index=segment_index)
        current = self._scalar_bridge_persistent_entries_by_segment[segment_index]
        for key, value in sorted(legacy.items()):
            if key in current:
                if current[key] != value:
                    self._legacy_scalar_conflicts += 1
                continue  # Fresh exact evaluation has precedence.
            if len(current) >= _SCALAR_BRIDGE_MAXIMUM_ENTRIES_PER_SHARD:
                break
            current[key] = value
            self._legacy_scalar_entries_admitted += 1

    def _enforce_transient_cache_limits(self):
        super()._enforce_transient_cache_limits()
        if self._active_transaction_journal is None:
            for key, value in self._pending_observations.items():
                self._observation_bank.publish(key, value)
            self._pending_observations.clear()

    def close(self):
        # Never flush a pending/aborted transaction from a finally block.
        self._pending_observations.clear()
        super().close()

    def to_dict(self):
        return {**super().to_dict(),
                "containment_observation_reuse": {
                    "hit_count": self._containment_observation_hits,
                    "miss_count": self._containment_observation_misses,
                    "successful_numeric_tracks_only": True,
                    "full_proof_callbacks_reused": False,
                    "raw_pixels_retained": False,
                },
                "scalar_bridge_reader_compatibility": {
                    "admitted_entries": self._legacy_scalar_entries_admitted,
                    "conflicting_legacy_entries_ignored": self._legacy_scalar_conflicts,
                    "renderer_proofs_reused": False,
                },
                "observation_reuse": self._observation_bank.diagnostics()}
