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
)


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
                "observation_reuse": self._observation_bank.diagnostics()}
