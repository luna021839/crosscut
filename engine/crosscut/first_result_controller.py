"""A retained complete plan and one admission policy for optional editing work.

This controller is not a quality approver. It protects a validated render plan
from failed proposals; decoded technical and editorial QA still run afterward.
"""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass,replace
from enum import Enum
import json
import hashlib
import os
import signal
import threading
from pathlib import Path
import tempfile
from typing import Callable

from .render import RenderPlan, segment_frame_counts
from .review_first_runtime import (
    ReviewFirstDeadlineExceeded, ReviewFirstSearchCutoff, ReviewFirstRuntime,
    enforce_review_first_deadline,
)


class FirstResultStage(str, Enum):
    INPUT_VERIFIED = "INPUT_VERIFIED"
    BASELINE_PLAN_READY = "BASELINE_PLAN_READY"
    RENDERED = "RENDERED"
    TECHNICALLY_VERIFIED = "TECHNICALLY_VERIFIED"
    REVIEW_READY = "REVIEW_READY"
    APPROVED = "APPROVED"


@dataclass(frozen=True,slots=True)
class IncumbentTransactionToken:
    revision: int
    content_sha256: str


@contextmanager
def _defer_owner_publication_signals():
    """Defer POSIX deadline/cancel delivery only across the short commit.

    No model work or arbitrary validator runs here. This is not hardware-level
    atomicity or full crash recovery; the journal is a controller audit record.
    """
    supported = (os.name == 'posix' and threading.current_thread() is threading.main_thread()
        and hasattr(signal, 'pthread_sigmask') and hasattr(signal, 'SIGALRM'))
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGINT}) if supported else None
    try:
        yield
    finally:
        if supported:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class FirstResultController:
    def __init__(self, runtime: ReviewFirstRuntime, *, sources, source_durations,
                 master_start: float, duration: float, allowed_source_indices=None, journal_path=None,
                 external_original_reference=None):
        self.runtime = runtime
        self.sources = tuple(sources)
        self.source_durations = tuple(source_durations)
        if len(self.sources) != len(self.source_durations):
            raise ValueError("verified source duration domain changed")
        self.master_start = master_start
        self.duration = duration
        self.allowed_source_indices = frozenset(range(len(self.sources)) if allowed_source_indices is None
                                                else allowed_source_indices)
        if not self.allowed_source_indices or not self.allowed_source_indices <= set(range(len(self.sources))):
            raise ValueError("first result source eligibility domain is invalid")
        self._render_configuration = None
        self.stage = FirstResultStage.INPUT_VERIFIED
        self.baseline = None
        self.incumbent = None
        self.attempts = []
        self.skipped_count = 0
        self._selected_authority = None
        self._incumbent_revision = 0
        self._incumbent_plan = None
        self._history = [self.stage.value]
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self.omitted_attempt_count = 0
        self._journal_warnings = []
        self._request_work_lease = None
        self._request_work_lease_attempted = False
        self._external_original_reference = external_original_reference
        if external_original_reference is not None:
            from .group_external_original_reference import GroupExternalOriginalReference
            if not isinstance(external_original_reference, GroupExternalOriginalReference):
                raise TypeError('controller external original must be a frozen typed GROUP reference')
            external_original_reference.validate_original()

    @property
    def external_original_reference(self):
        return self._external_original_reference

    @property
    def boundary_look_creation_allowed(self):
        # A fresh normal project may author a new look. An explicit external
        # reference keeps its original absent/active policy across promotions.
        if self.external_original_reference is None:
            return True
        self.external_original_reference.validate_original()
        return False

    @property
    def request_work_lease(self):
        return self._request_work_lease

    def reserve_request_work(self, *, total_limits, allocations, journal_path):
        """Own all editing phases once, before any optional producer.

        Allocation values are explicit declared work, not measured WCET or an
        automatically guessed media SLA. Missing admission never launches work.
        """
        from .request_work_lease import RequestWorkLease
        if self._request_work_lease_attempted or self.attempts or self.baseline is None:
            raise ValueError("whole-request work must be admitted once after baseline and before optional producers")
        self._request_work_lease_attempted = True
        self._request_work_lease = RequestWorkLease(runtime=self.runtime,
            controller_identity=self.transaction_token().content_sha256,
            total_limits=total_limits, allocations=allocations, journal_path=journal_path)
        self._persist()
        return self._request_work_lease

    def run_work_phase(self, name, operation):
        from .optional_native_worker import optional_native_scope
        from .request_work_lease import RequestWorkExhausted
        if self._request_work_lease is None or not callable(operation):
            raise ValueError("declared request work phase needs its live controller lease and operation")
        self.runtime.require_total_budget("request_phase_" + str(name))
        try:
            with self._request_work_lease.phase(name) as port:
                with optional_native_scope(port.deadline), enforce_review_first_deadline(port.deadline,
                        stage="request_phase_" + name):
                    try:
                        value = operation()
                        port._check()
                        if port.admission_cutoff is not None:
                            raise port.admission_cutoff
                    except (ReviewFirstDeadlineExceeded, ReviewFirstSearchCutoff) as exc:
                        self.runtime.require_total_budget("request_phase_" + name)
                        if isinstance(exc,RequestWorkExhausted):raise
                        # Convert inside the phase context so its durable
                        # terminal records the same typed scheduling outcome.
                        raise RequestWorkExhausted(name,"phase_wall",partial=port.partial_work) from exc
                return value
        finally:
            self._persist()

    def _validate_external_candidate(self, value, plan, *, baseline=False):
        reference = self.external_original_reference
        if reference is not None:
            self.runtime.require_total_budget('external_original_policy_validation')
            if baseline:
                reference.validate_baseline(value, plan)
            else:
                reference.validate_candidate(value, plan)
            self.runtime.require_total_budget('external_original_policy_validation')

    def _cleanup_journal(self, temporary, *, phase):
        if temporary is None:
            return
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            # A successful publication must not become a rejected proposal
            # because removing an already-renamed temporary path failed.
            self._journal_warnings.append(dict(phase=phase, error_type=type(exc).__name__, error=str(exc)[:2000]))
            self._journal_warnings = self._journal_warnings[-16:]

    def _prepare_journal(self, snapshot):
        if self.journal_path is None:
            return None
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.' + self.journal_path.name, suffix='.tmp',
                                    dir=self.journal_path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            return temporary
        except BaseException:
            self._cleanup_journal(temporary, phase='precommit_cleanup')
            raise

    def _persist(self):
        temporary = self._prepare_journal(self.to_dict())
        published = False
        try:
            if temporary is not None:
                temporary.replace(self.journal_path)
            published = True
        finally:
            self._cleanup_journal(temporary, phase='postcommit_cleanup' if published else 'precommit_cleanup')

    def _publish_owner(self, changes, snapshot, *, token=None):
        """Prepare first; journal replace is the declared publication point.

        Before that point, any failure preserves the old whole owner/journal.
        Afterwards memory must match the published revision even if a pending
        control exception or cleanup error is delivered. No rollback launders a
        committed disk state back into an old memory owner.
        """
        def recheck():
            if token is not None:
                self.runtime.require_first_render_budget('candidate_publication')
                if self.transaction_token() != token:
                    raise ValueError('atomic candidate baseline changed before publication')
        recheck()
        temporary = self._prepare_journal(snapshot)
        published = False
        delayed_error = None
        expected_bytes = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
        try:
            recheck()
            with _defer_owner_publication_signals():
                recheck()
                if temporary is not None:
                    try:
                        temporary.replace(self.journal_path)
                    except BaseException as exc:
                        # A wrapper/control exception can occur after rename
                        # took effect. Resolve actual publication, not just the
                        # return value, before deciding which owner to retain.
                        try:
                            committed = self.journal_path.read_bytes() == expected_bytes
                        except OSError:
                            committed = False
                        if not committed:
                            raise
                        delayed_error = exc
                published = True
                self.__dict__.update(changes)
            if delayed_error is not None:
                raise delayed_error
        finally:
            self._cleanup_journal(temporary, phase='postcommit_cleanup' if published else 'precommit_cleanup')

    def validate_plan(self, plan: RenderPlan) -> None:
        if not isinstance(plan, RenderPlan):
            raise TypeError("first result requires a complete RenderPlan")
        # Re-run geometry, transition and output-clock validation, not just a
        # serializable timeline or an optimistic quality flag.
        replace(plan)
        configuration = (plan.reference_source, plan.aspect_ratio, plan.width, plan.height, plan.fps)
        if self._render_configuration is not None and configuration != self._render_configuration:
            raise ValueError("candidate changed reference audio or output geometry/frame clock")
        if (tuple(plan.sources) != self.sources or abs(plan.master_audio_start-self.master_start) > 1e-6
                or abs(plan.duration-self.duration) > 1 / plan.fps
                or any(count <= 0 for count in segment_frame_counts(plan))):
            raise ValueError("candidate changed the verified source/interval/frame domain")
        for segment in plan.segments:
            if segment.source_index not in self.allowed_source_indices:
                raise ValueError("candidate uses a source outside verified eligibility")
            if segment.source_end > self.source_durations[segment.source_index] + 1e-6:
                raise ValueError("candidate source clock exceeds verified media duration")

    def install_baseline(self, value, *, plan: RenderPlan, authority: str):
        if self.baseline is not None:
            raise RuntimeError("baseline may only be installed once")
        self.validate_plan(plan)
        self._validate_external_candidate(value, plan, baseline=True)
        stage = FirstResultStage.BASELINE_PLAN_READY
        history = [*self._history, stage.value]
        revision = self._incumbent_revision+1
        snapshot = dict(self.to_dict(), baseline_retained=True, stage=stage.value,
            stage_history=history, incumbent_revision=revision, selected_authority=authority)
        self._publish_owner(dict(_render_configuration=(plan.reference_source, plan.aspect_ratio, plan.width, plan.height, plan.fps),
            baseline=value, incumbent=value, _incumbent_plan=plan, _incumbent_revision=revision,
            _selected_authority=authority, stage=stage, _history=history), snapshot)
        return value

    def run_optional(self, stage: str, operation: Callable, *, validate: Callable | None = None):
        if self.baseline is None:
            raise RuntimeError("optional work cannot precede the complete baseline")
        self.runtime.require_total_budget(stage)
        from .request_work_lease import current_request_work_phase, RequestWorkExhausted
        if self._request_work_lease_attempted and (self._request_work_lease is None
                or current_request_work_phase() is not self._request_work_lease.active
                or self._request_work_lease.active is None):
            self.skipped_count += 1
            if len(self.attempts) < 64:
                self.attempts.append(dict(stage=stage, status="budget_not_started",
                    dimension="declared_live_request_phase_required", elapsed_seconds=0.,
                    observed_geometry_rejected=False))
            else:
                self.omitted_attempt_count += 1
            self._persist()
            return None
        if not self.runtime.optional_refinement_allowed or len(self.attempts) >= 64:
            self.skipped_count += 1
            if len(self.attempts) < 64:
                self.attempts.append(dict(stage=stage, status="skipped_search_cutoff", elapsed_seconds=0.))
            else:
                self.omitted_attempt_count += 1
            self._persist()
            return None
        started = self.runtime.elapsed_seconds
        started_revision = self._incumbent_revision
        row = dict(stage=stage, status="running")
        self.attempts.append(row)
        self._persist()
        cutoff = min(self.runtime.deadline_monotonic_seconds,
            self.runtime.deadline_monotonic_seconds - self.runtime.total_budget_seconds
            + min(self.runtime.first_render_budget_seconds, self.runtime.refinement_stop_seconds))
        if self._request_work_lease is not None:
            cutoff = min(cutoff,self._request_work_lease.active.deadline)
        try:
            from .optional_native_worker import optional_native_scope
            with optional_native_scope(cutoff), enforce_review_first_deadline(cutoff, stage=stage):
                value = operation()
                self.runtime.require_first_render_budget(stage)
                if validate is not None:
                    validate(value)
            row["status"] = "validated_proposal" if value is not None else "unavailable"
            return value
        except RequestWorkExhausted as exc:
            row.update(exc.to_dict())
        except ReviewFirstDeadlineExceeded:
            committed = self._incumbent_revision != started_revision
            row["status"] = "committed_deadline_pending" if committed else "deadline_exceeded"
            self.runtime.require_total_budget(stage)
            if not committed:
                row["status"] = "search_cutoff"
        except ReviewFirstSearchCutoff:
            row["status"] = "committed_search_cutoff" if self._incumbent_revision != started_revision else "search_cutoff"
        except Exception as exc:
            row.update(status="committed_optional_error" if self._incumbent_revision != started_revision else "rejected",
                error_type=type(exc).__name__, error=str(exc)[:2000])
        except BaseException:
            row["status"] = "committed_cancelled" if self._incumbent_revision != started_revision else "cancelled"
            raise
        finally:
            row["elapsed_seconds"] = max(0., self.runtime.elapsed_seconds-started)
            self._persist()
        return None

    def promote(self, value, *, plan: RenderPlan, authority: str, _token=None):
        self.validate_plan(plan)
        self._validate_external_candidate(value, plan)
        revision = self._incumbent_revision+1
        snapshot = dict(self.to_dict(), incumbent_revision=revision, selected_authority=authority)
        self._publish_owner(dict(incumbent=value, _incumbent_plan=plan, _incumbent_revision=revision,
            _selected_authority=authority), snapshot, token=_token)
        return value

    def transaction_token(self):
        """Freeze full owner content, not just a plan/PID or object identity."""
        if self.incumbent is None or self._incumbent_plan is None:
            raise RuntimeError('candidate transaction cannot precede a complete incumbent')
        if isinstance(self.incumbent,RenderPlan):
            payload=self.incumbent.to_json()
        elif hasattr(self.incumbent,'to_dict'):
            payload=self.incumbent.to_dict()
        elif isinstance(self.incumbent,dict):
            payload=self.incumbent
        else:
            raise TypeError('candidate transaction needs a serializable complete typed owner')
        preimage=dict(owner=payload,render_plan=self._incumbent_plan.to_json(),
            external_original_reference=None if self.external_original_reference is None
                else self.external_original_reference.to_dict())
        identity=hashlib.sha256(json.dumps(preimage,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        return IncumbentTransactionToken(self._incumbent_revision,identity)

    def compare_and_promote(self,value,*,plan,authority,token,validate):
        """Validate and replace the whole candidate under controller ownership.

        A worker's result cannot overwrite a newer or mutated incumbent. The
        mandatory validator independently rechecks all external evidence; the
        second token comparison also rejects reentrant validator mutations.
        No partial plan or source-dependent metadata is installed on failure.
        """
        if not isinstance(token,IncumbentTransactionToken) or not callable(validate):
            raise ValueError('atomic candidate adoption requires its frozen token and complete validator')
        if self.transaction_token()!=token:
            raise ValueError('atomic candidate baseline changed before parent validation')
        self.validate_plan(plan)
        validate(value)
        if self.transaction_token()!=token:
            raise ValueError('atomic candidate baseline changed during parent validation')
        return self.promote(value,plan=plan,authority=authority,_token=token)

    def render_complete(self):
        if self.stage is not FirstResultStage.BASELINE_PLAN_READY:
            raise RuntimeError("first result render completion is out of order")
        self.stage = FirstResultStage.RENDERED
        self._history.append(self.stage.value)
        self._persist()

    def to_dict(self):
        return dict(contract="retained_complete_first_result_plan_v1", stage=self.stage.value,
            stage_history=list(self._history),
            baseline_retained=self.baseline is not None, selected_authority=self._selected_authority,
            incumbent_revision=self._incumbent_revision,
            optional_attempt_count=sum(r["status"] != "skipped_search_cutoff" for r in self.attempts),
            rejected_attempt_count=sum(r["status"] == "rejected" for r in self.attempts),
            skipped_count=self.skipped_count, attempts=deepcopy(self.attempts),
            omitted_attempt_count=self.omitted_attempt_count,
            journal_path=str(self.journal_path) if self.journal_path is not None else None,
            external_original_reference=None if self.external_original_reference is None
                else self.external_original_reference.to_dict(),
            journal_warnings=deepcopy(self._journal_warnings),
            request_work_lease_attempted=self._request_work_lease_attempted,
            request_work_lease=None if self._request_work_lease is None else self._request_work_lease.to_dict(),
            quality_approved=False, decoded_qa_required=True)


def finalize_first_result_manifest(manifest, disposition):
    """Persist only the already content-bound final QA disposition, never infer it.

    Kept outside proposal promotion: render completion is not technical or
    editorial approval. The final GROUP/MEMBER QA owner calls this after all
    content binding and integrity classification checks.
    """
    from .review_first_runtime import ReviewFirstStatus
    locations = []
    group = manifest.get("review_first_visual_selection")
    if isinstance(group, dict):
        locations.append(group)
    render = manifest.get("render")
    if isinstance(render, dict) and isinstance(render.get("cadence"), dict):
        locations.append(render["cadence"])
    member = manifest.get("member_direct_full_frame")
    if isinstance(member, dict) and isinstance(member.get("cadence"), dict):
        locations.append(member["cadence"])
    for location in locations:
        payload = location.get("first_result_controller")
        if not isinstance(payload, dict) or payload.get("stage") != FirstResultStage.RENDERED.value:
            continue
        history = list(payload.get("stage_history", []))
        if disposition.status in (ReviewFirstStatus.REVIEW_READY, ReviewFirstStatus.APPROVED):
            history.append(FirstResultStage.TECHNICALLY_VERIFIED.value)
        history.append(disposition.status.value)
        payload.update(stage=disposition.status.value, stage_history=history,
            quality_approved=disposition.status is ReviewFirstStatus.APPROVED,
            decoded_qa_completed=True, video_output_exposed=disposition.video_output_exposed)
