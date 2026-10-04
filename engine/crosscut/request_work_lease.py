"""Parent-owned editing work, with phase reserves fixed before optional work.

This is an execution/accounting contract, not a runtime calibration, peak-RSS
bound, complete raw-network history or editorial-quality certificate. Legacy
callers without an explicitly installed lease remain diagnostic callers.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from uuid import uuid4

from .review_first_runtime import ReviewFirstSearchCutoff

CONTRACT = "controller_owned_editing_phase_lease_v1"
PHASES = ("original", "heads", "discovery", "selected_completion", "parent",
          "final", "encode", "decoded_qa", "terminal")
RESOURCES = ("states", "edges", "dense_paths", "new_decodes", "model_calls", "model_loads",
             "pose_comparisons", "pose_evidence_bytes", "region_flow_feature_updates",
             "association_nodes", "association_edges", "workers", "ipc_bytes",
             "identity_reads", "model_hash_bytes", "source_probes", "decoder_opens",
             "journal_bytes")
TERMINAL_BYTES = 4096
PRIMITIVE_TERMINAL_IPC_BYTES = 2048
MAX_WORKER_MESSAGE_BYTES = 32 * 1024 * 1024
NATIVE_PRIMITIVE_RESOURCES = frozenset(("dense_paths", "new_decodes", "pose_comparisons",
    "pose_evidence_bytes", "region_flow_feature_updates", "association_nodes", "association_edges"))
_phase = ContextVar("controller_request_work_phase", default=None)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _amounts(value):
    if (not isinstance(value, (dict, MappingProxyType)) or not value
            or any(k not in RESOURCES or type(n) is not int or not 0 <= n <= 2**63-1
                   for k, n in value.items())):
        raise ValueError("phase work needs exact named nonnegative integer resources")
    return dict(value)


class RequestWorkExhausted(ReviewFirstSearchCutoff):
    """Scheduling result, never a measured negative geometry observation."""
    def __init__(self, phase, dimension, *, partial=False):
        self.phase, self.dimension, self.partial = phase, dimension, bool(partial)
        super().__init__(f"request phase {phase} has no {dimension} admission budget")

    def to_dict(self):
        return dict(contract=CONTRACT, phase=self.phase, dimension=self.dimension,
                    status="budget_partial" if self.partial else "budget_not_started",
                    observed_geometry_rejected=False)


@dataclass(frozen=True, slots=True)
class PhaseAllocation:
    resources: object
    wall_seconds: float
    maximum_worker_message_bytes: int = 0

    def __post_init__(self):
        values = _amounts(self.resources)
        if (type(self.wall_seconds) not in (float, int)
                or not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0
                or type(self.maximum_worker_message_bytes) is not int
                or not 0 <= self.maximum_worker_message_bytes <= MAX_WORKER_MESSAGE_BYTES
                or values.get("journal_bytes", 0) < 2 * TERMINAL_BYTES):
            raise ValueError("phase allocation needs bounded work, wall time and terminal space")
        if self.maximum_worker_message_bytes and values.get("ipc_bytes", 0) < 2 * self.maximum_worker_message_bytes + 1:
            raise ValueError("worker request/result capacity must be reserved together")
        object.__setattr__(self, "resources", MappingProxyType({k: values.get(k, 0) for k in RESOURCES}))

    def to_dict(self):
        return dict(resources=dict(self.resources), wall_seconds=float(self.wall_seconds),
                    maximum_worker_message_bytes=self.maximum_worker_message_bytes)


def current_request_work_phase():
    return _phase.get()


@contextmanager
def bind_request_work_phase(port):
    token = _phase.set(port)
    try:
        yield port
    finally:
        _phase.reset(token)


class RequestWorkLease:
    def __init__(self, *, runtime, controller_identity, total_limits, allocations, journal_path):
        totals = _amounts(total_limits)
        if (not isinstance(controller_identity, str) or len(controller_identity) != 64
                or any(c not in "0123456789abcdef" for c in controller_identity)
                or set(allocations) != set(PHASES)
                or any(not isinstance(v, PhaseAllocation) for v in allocations.values())):
            raise ValueError("lease must bind one controller and every declared editing phase")
        if any(sum(a.resources[k] for a in allocations.values()) > totals.get(k, 0) for k in RESOURCES):
            raise RequestWorkExhausted("request", "whole_phase_resources")
        if sum(a.wall_seconds for a in allocations.values()) > runtime.remaining_total_seconds:
            raise RequestWorkExhausted("request", "whole_phase_wall_reserve")
        self.runtime, self.controller_identity = runtime, controller_identity
        self.allocations = MappingProxyType(dict(allocations))
        self.total_limits = MappingProxyType({k: totals.get(k, 0) for k in RESOURCES})
        self.deadline = float(runtime.deadline_monotonic_seconds)
        self._clock = runtime._clock
        self.owner_pid, self.lease_id = os.getpid(), uuid4().hex
        self.journal_path = Path(journal_path)
        self.states = {p: "not_requested" for p in PHASES}
        self.ports = {}
        self.active = None
        self.sequence, self.prefix_sha256 = 0, "0" * 64
        self.journal_failed = False
        self._lock = RLock()
        setup=PhaseWorkPort(self,"terminal",self.deadline)
        header=dict(controller_identity=controller_identity,total_limits=dict(self.total_limits),
            allocations={p:self.allocations[p].to_dict() for p in PHASES})
        header_bytes=len(_wire(self._journal_row("lease",header,setup)))+1
        # The initial archive is part of the request too. Charge its exact
        # bytes to terminal/setup before opening a file or acknowledging work.
        setup._capacity({"journal_bytes":header_bytes+TERMINAL_BYTES})
        self.ports["terminal"]=setup
        # Exclusive creation prevents accidental history reset after failure.
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(fd, "wb", buffering=0)
        self._append("lease",header,setup)

    def _owned(self):
        if os.getpid() != self.owner_pid:
            raise ValueError("request lease authority cannot be copied into a child")
        if self.journal_failed:
            raise OSError("controller work journal failed; no further work admitted")

    @property
    def remaining_seconds(self):
        return max(0., self.deadline-float(self._clock()))

    def _journal_row(self,kind,payload,port):
        return dict(lease_id=self.lease_id, sequence=self.sequence, previous_sha256=self.prefix_sha256,
                   phase=None if port is None else port.name, kind=kind, payload=payload)

    def _append(self, kind, payload, port=None, terminal_key=None):
        self._owned()
        data = _wire(self._journal_row(kind,payload,port)) + b"\n"
        if port is not None:
            port._admit_journal(len(data), terminal_key)
        try:
            # A raw regular-file write may return a short count (disk full,
            # interrupted I/O). Do not ACK admission or advance the hash until
            # the complete record is durable; an incomplete prefix is failure.
            remaining = memoryview(data)
            while remaining:
                written = self._stream.write(remaining)
                if type(written) is not int or not 0 < written <= len(remaining):
                    raise OSError("controller work journal did not write its complete record")
                remaining = remaining[written:]
            os.fsync(self._stream.fileno())
        except BaseException:
            self.journal_failed = True
            raise
        if port is not None:
            port._commit_journal(len(data), terminal_key)
        self.prefix_sha256 = sha256(data).hexdigest()
        self.sequence += 1

    @contextmanager
    def phase(self, name):
        with self._lock:
            self._owned()
            if name not in self.allocations or self.active is not None or self.states[name] != "not_requested":
                raise ValueError("request phase is unknown, reentrant or already consumed")
            remaining = sum(a.wall_seconds for p, a in self.allocations.items()
                            if p != name and self.states[p] == "not_requested")
            allocation = self.allocations[name]
            if allocation.wall_seconds + remaining > self.remaining_seconds:
                raise RequestWorkExhausted(name, "protected_future_phase_wall")
            now = float(self._clock())
            deadline = min(now + allocation.wall_seconds, self.deadline - remaining)
            port = self.ports.get(name) if name=="terminal" else None
            if port is None:port=PhaseWorkPort(self,name,deadline)
            port.deadline=deadline
            port._reserve_terminal("phase")
            self._append("phase_begin", dict(deadline=deadline), port)
            self.ports[name], self.active, self.states[name] = port, port, "running"
        error = None
        try:
            with bind_request_work_phase(port):
                yield port
        except BaseException as exc:
            error = exc
            raise
        finally:
            with self._lock:
                status = ("returned" if error is None else
                          error.to_dict()["status"] if isinstance(error, RequestWorkExhausted) else "failed")
                if port.live_workers:
                    status = "worker_not_reaped"
                if port._reservations and error is None:
                    status = "native_prefix_incomplete"
                # Terminal persistence has its own capacity and does not need a
                # renewed deadline. A failed journal is not a successful phase.
                try:
                    self._append("phase_terminal", dict(status=status, charged=dict(port.charged),
                                 reserved=dict(port.reserved), function_entered=port.function_entered,
                                 error_type=None if error is None else type(error).__name__), port, "phase")
                finally:
                    self.states[name] = "journal_failed" if self.journal_failed else status
                    self.active = None
                if port.live_workers and error is None:
                    raise ValueError("phase cannot complete before its owned workers are reaped")
                if port._reservations and error is None:
                    raise ValueError("phase cannot complete with an unreturned native primitive prefix")

    def to_dict(self):
        with self._lock:
            return dict(contract=CONTRACT, lease_id=self.lease_id, owner_pid=self.owner_pid,
                        controller_identity=self.controller_identity, states=dict(self.states),
                        total_limits=dict(self.total_limits),
                        allocated={p: self.allocations[p].to_dict() for p in PHASES},
                        phases={p: v.to_dict() for p, v in self.ports.items()},
                        journal_path=str(self.journal_path), journal_event_count=self.sequence,
                        journal_prefix_sha256=self.prefix_sha256, journal_failed=self.journal_failed,
                        counter_copies_are_authority=False, retry_renews_budget=False,
                        all_model_call_sites_metered=False, all_raw_results_journaled=False,
                        runtime_calibrated=False, process_peak_rss_bounded=False, quality_approved=False)

    def close(self):
        with self._lock:
            if self.active is not None:
                raise ValueError("cannot close an active request phase")
            self._stream.close()


class PhaseWorkPort:
    def __init__(self, lease, name, deadline):
        self.lease, self.name, self.deadline = lease, name, deadline
        self.allocation = lease.allocations[name]
        self.charged = {k: 0 for k in RESOURCES}
        self.reserved = {k: 0 for k in RESOURCES}
        self._terminals, self._reservations = {}, {}
        self.model_calls, self.live_workers = {}, {}
        self.function_entered = False
        self.admission_cutoff = None
        self._serial = 0

    @property
    def maximum_worker_message_bytes(self):
        return self.allocation.maximum_worker_message_bytes

    @property
    def partial_work(self):
        return self.function_entered or any(self.charged[k] for k in
            ("new_decodes", "model_calls", "model_loads", "pose_comparisons", "region_flow_feature_updates",
             "model_hash_bytes", "source_probes", "decoder_opens"))

    def _check(self):
        self.lease._owned()
        if self.lease.active is not self:
            raise ValueError("phase port is not the controller's live phase")
        now = float(self.lease._clock())
        if now >= self.deadline:
            self._deny("phase_wall")

    def _deny(self,dimension):
        exc=RequestWorkExhausted(self.name,dimension,partial=self.partial_work)
        if self.admission_cutoff is None:self.admission_cutoff=exc
        raise exc

    def _capacity(self, amounts):
        for k, n in _amounts(amounts).items():
            if self.charged[k] + self.reserved[k] + n > self.allocation.resources[k]:
                self._deny(k)

    def _reserve_terminal(self, key):
        self._capacity({"journal_bytes": TERMINAL_BYTES})
        self.reserved["journal_bytes"] += TERMINAL_BYTES
        self._terminals[key] = TERMINAL_BYTES

    def _admit_journal(self, count, terminal_key):
        if terminal_key is None:
            self._capacity({"journal_bytes": count})
        elif terminal_key not in self._terminals or count > self._terminals[terminal_key]:
            raise ValueError("controller terminal record exceeded its pre-reserved capacity")

    def _commit_journal(self, count, terminal_key):
        if terminal_key is not None:
            self.reserved["journal_bytes"] -= self._terminals.pop(terminal_key)
        self.charged["journal_bytes"] += count

    def charge(self, kind, count=1, *, model=None):
        with self.lease._lock:
            self._check()
            amounts = _amounts({kind: count})
            if kind == "journal_bytes":
                raise ValueError("journal work is charged by the parent writer")
            if kind in ("model_calls","model_loads") and (not isinstance(model, str) or not 0 < len(model) <= 256):
                raise ValueError("model work must name its measured model")
            self._capacity(amounts)
            self.lease._append("work_admitted", dict(amounts=amounts, model=model), self)
            self.charged[kind] += count
            if kind == "model_calls":
                self.model_calls[model] = self.model_calls.get(model, 0) + count

    def reserve(self, amounts):
        with self.lease._lock:
            self._check()
            values = _amounts(amounts)
            if (not set(values) <= NATIVE_PRIMITIVE_RESOURCES
                    or values.get("pose_evidence_bytes",0) < 1):
                raise ValueError("native primitive cannot reserve unnamed model or controller work")
            self._capacity(values)
            self._capacity(dict(values,ipc_bytes=values.get("ipc_bytes",0)+PRIMITIVE_TERMINAL_IPC_BYTES))
            key = f"reservation-{self._serial}"
            self._serial += 1
            self._reserve_terminal(key)
            try:
                self.lease._append("primitive_reserved", dict(key=key, amounts=values), self)
            except BaseException:
                self.reserved["journal_bytes"] -= self._terminals.pop(key)
                raise
            for k, n in values.items():
                self.reserved[k] += n
            # A settlement ACK remains available after the compute deadline.
            self.charged["ipc_bytes"] += PRIMITIVE_TERMINAL_IPC_BYTES
            self._reservations[key] = dict(amounts=values, started=False)
            return key

    def start(self, key):
        with self.lease._lock:
            self._check()
            row = self._reservations.get(key)
            if row is None or row["started"]:
                raise ValueError("native reservation was forged or started twice")
            self.lease._append("primitive_started", dict(key=key), self)
            row["started"] = True
            for k, n in row["amounts"].items():
                if k != "pose_evidence_bytes":
                    self.reserved[k] -= n
                    self.charged[k] += n

    def settle(self, key, evidence_bytes):
        with self.lease._lock:
            self.lease._owned()
            if self.lease.active is not self:
                raise ValueError("native terminal does not belong to the live controller phase")
            row = self._reservations.get(key)
            if (row is None or type(evidence_bytes) is not int
                    or not 0 <= evidence_bytes <= row["amounts"].get("pose_evidence_bytes", 0)):
                raise ValueError("native terminal evidence exceeded the parent reservation")
            self.lease._append("primitive_terminal", dict(key=key, started=row["started"],
                               evidence_bytes=evidence_bytes, raw_result_in_this_record=False), self, key)
            for k, n in row["amounts"].items():
                if k == "pose_evidence_bytes" or not row["started"]:
                    self.reserved[k] -= n
            self.charged["pose_evidence_bytes"] += evidence_bytes
            del self._reservations[key]

    def begin_worker(self, module, name):
        with self.lease._lock:
            self._check()
            limit = self.maximum_worker_message_bytes
            if not limit:
                self._deny("worker_transport")
            values = dict(workers=1, ipc_bytes=2 * limit + 1)
            self._capacity(values)
            key = f"worker-{self._serial}"
            self._serial += 1
            self._reserve_terminal(key)
            try:
                self.lease._append("worker_admitted", dict(key=key, module=module, name=name,
                                   charged_ipc_upper_bound_bytes=2 * limit + 1), self)
            except BaseException:
                self.reserved["journal_bytes"] -= self._terminals.pop(key)
                raise
            for k, n in values.items():
                self.charged[k] += n
            self.live_workers[key] = dict(module=module, name=name, function_entered=False)
            return key

    def worker_entered(self, key):
        with self.lease._lock:
            self._check()
            row = self.live_workers.get(key)
            if row is None or row["function_entered"]:
                raise ValueError("worker execution identity was omitted or duplicated")
            self.lease._append("worker_entered", dict(key=key), self)
            row["function_entered"] = self.function_entered = True

    def finish_worker(self, key, *, status, sent_bytes, received_bytes):
        with self.lease._lock:
            self.lease._owned()
            if (key not in self.live_workers or status not in ("returned", "failed", "cutoff")
                    or any(type(n) is not int or n < 0 for n in (sent_bytes, received_bytes))):
                raise ValueError("invalid owned worker terminal record")
            row = self.live_workers[key]
            self.lease._append("worker_terminal", dict(key=key, status=status, sent_bytes=sent_bytes,
                               received_bytes=received_bytes, function_entered=row["function_entered"],
                               computation_result_unknown=status != "returned",
                               budget_refunded=False), self, key)
            del self.live_workers[key]

    def to_dict(self):
        return dict(deadline=self.deadline, charged=dict(self.charged), reserved=dict(self.reserved),
                    model_calls=dict(self.model_calls), function_entered=self.function_entered,
                    pending_native_reservations=len(self._reservations),
                    live_workers=list(self.live_workers), raw_results_complete=False)
