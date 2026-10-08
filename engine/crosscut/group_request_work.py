"""One normal GROUP controller lease, retained through render and decoded QA.

These are declared allocation limits, not a measured runtime SLA. Legacy
observers are not all metered, and installing this lifecycle does not implement
native union completion, raw-parent admission or artistic-quality approval.
"""
from dataclasses import dataclass
import math

from .request_work_lease import PHASES, RESOURCES, PhaseAllocation, RequestWorkExhausted


@dataclass(frozen=True)
class GroupWorkPolicy:
    safety_fraction: float = .9

    def allocations(self, remaining_seconds):
        if (type(remaining_seconds) not in (int, float) or not math.isfinite(remaining_seconds)
                or remaining_seconds <= 0 or type(self.safety_fraction) not in (int, float)
                or not math.isfinite(self.safety_fraction) or not 0 < self.safety_fraction < 1):
            raise ValueError('normal GROUP work needs remaining original time and explicit finite reserves')
        weights = dict(original=.005, heads=.005, discovery=.12, selected_completion=.20,
            parent=.005, final=.06, encode=.48, decoded_qa=.12, terminal=.005)
        # Capacities belong to this parent once; new sessions cannot reset them.
        # Unmetered legacy call sites remain explicitly unmetered in the lease.
        caps = dict(states=65536, edges=131072, dense_paths=128, new_decodes=1024,
            model_calls=8192, model_loads=16, pose_comparisons=65536,
            pose_evidence_bytes=64*1024**2, region_flow_feature_updates=131072,
            association_nodes=65536, association_edges=131072, workers=16,
            ipc_bytes=128*1024**2, identity_reads=8192, model_hash_bytes=512*1024**2,
            source_probes=32, decoder_opens=32, journal_bytes=32*1024**2)
        result = {}
        for name in PHASES:
            resources = caps if name in ('heads','discovery','selected_completion','parent','final','decoded_qa') else {'journal_bytes':1024*1024}
            if name == 'encode':
                # Frozen source-clock checks/decoders may still run during the
                # encoder. Reserve their work; forbid new NN camera search.
                resources = dict(caps, states=0,edges=0,dense_paths=0,
                    new_decodes=36000,model_calls=0,model_loads=0,pose_comparisons=0,
                    pose_evidence_bytes=0,region_flow_feature_updates=0,
                    association_nodes=0,association_edges=0)
            resources = dict(resources, native_completion_passes=1 if name=='selected_completion' else 0)
            result[name] = PhaseAllocation(resources,
                remaining_seconds*self.safety_fraction*weights[name],
                16*1024**2 if name in ('heads','discovery','selected_completion','parent','final','encode','decoded_qa') else 0)
        return result


class GroupRequestWork:
    """Scope the real normal controller, never a second budget/controller.

    Optional refiners share one live discovery phase. Other phases are entered
    at most once with the same lease; unused phases remain not_requested.
    ``ExitStack`` at the public analyze boundary closes this on every exit,
    including cancellation and exceptions outside an optional callback.
    """
    def __init__(self, controller, *, journal_path, policy=GroupWorkPolicy()):
        allocations = policy.allocations(controller.runtime.remaining_total_seconds)
        totals = {k:sum(a.resources[k] for a in allocations.values()) for k in RESOURCES}
        self.controller = controller
        self.lease = controller.reserve_request_work(total_limits=totals,
            allocations=allocations, journal_path=journal_path)
        self._optional_context = None
        self.closed = False
        self.last_entered_phase = None
        self.last_requested_phase = None

    def __enter__(self):
        return self

    def start_optional(self):
        if self.closed or self._optional_context is not None:
            raise ValueError('normal GROUP optional phase cannot be renewed')
        self.last_requested_phase = 'discovery'
        context = self.lease.phase('discovery')
        context.__enter__()
        self._optional_context = context
        self.last_entered_phase = 'discovery'
        self.controller._persist()

    def finish_optional(self, exc_type=None, exc=None, traceback=None):
        context, self._optional_context = self._optional_context, None
        if context is None:
            return
        if exc is None and self.lease.active is not None:
            # A callback may already have caught its typed scheduling cutoff.
            # Keep that outcome in the phase terminal rather than laundering it
            # into a successful phase just because the full baseline survived.
            cutoff = self.lease.active.admission_cutoff
            if cutoff is not None:
                exc_type, exc, traceback = type(cutoff), cutoff, cutoff.__traceback__
        try:
            context.__exit__(exc_type, exc, traceback)
        finally:
            self.controller._persist()

    def run(self, name, operation):
        if self.closed:
            raise ValueError('closed normal GROUP work cannot admit another phase')
        self.finish_optional()
        self.last_requested_phase = name
        try:
            return self.controller.run_work_phase(name, operation)
        finally:
            # An expired total clock or future-reserve rejection may happen
            # before phase_begin. Do not claim that function/phase ran then.
            if name in self.lease.ports and self.lease.states[name]!='not_requested':
                self.last_entered_phase = name

    @property
    def live_remaining_seconds(self):
        port = self.lease.active
        if port is None:
            raise ValueError('normal GROUP timeout needs its current declared phase')
        port._check()
        return max(0., port.deadline-float(self.controller.runtime._clock()))

    def __exit__(self, exc_type, exc, traceback):
        if self.closed:
            return False
        cleanup_failure = None
        try:
            self.finish_optional(exc_type, exc, traceback)
        except BaseException as failure:
            cleanup_failure = failure
            raise
        finally:
            failure = exc if exc is not None else cleanup_failure
            outcome = ('returned' if failure is None else
                failure.to_dict()['status'] if isinstance(failure,RequestWorkExhausted) else
                'cancelled' if isinstance(failure,KeyboardInterrupt) else 'failed')
            try:
                self.lease.finish_request(outcome=outcome,last_entered_phase=self.last_entered_phase,
                    last_requested_phase=self.last_requested_phase,
                    error_type=None if failure is None else type(failure).__name__,
                    error=None if failure is None else str(failure)[:2000])
            finally:
                self.lease.close()
                self.closed = True
                self.controller._persist()
        return False


def install_group_request_work(*, controller, resources, journal_path):
    session = GroupRequestWork(controller, journal_path=journal_path)
    resources.enter_context(session)
    session.start_optional()
    return session
