"""Killable optional native work with a guardian and anonymous-pipe IPC.

Only our own parent/child messages use pickle. No downloaded input is unpickled
and no raw descriptor, face vector, or model result is persisted for IPC.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from collections import deque
from functools import wraps
import copyreg
import importlib
import inspect
import io
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
from threading import Event, Thread
from time import monotonic
from types import MappingProxyType

_optional_deadline = ContextVar("optional_native_deadline", default=None)
_WORKER_MODULE = "crosscut.optional_native_worker"


def _worker_environment():
    # `python engine/crosscut_cli.py` adds engine only to that interpreter's
    # sys.path. A fresh `python -m crosscut...` does not inherit it. Bind both
    # guardian and execute to this checkout, independently of cwd/test setup.
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parent.parent)
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = package_root + (os.pathsep + inherited if inherited else "")
    return environment


class _WorkerStderrTail:
    """Drain native diagnostics without unbounded buffers or disk persistence."""
    def __init__(self):
        self.read_fd, self.write_fd = os.pipe()
        self.chunks = deque(maxlen=8)
        self.thread = Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        with os.fdopen(self.read_fd, "rb", buffering=0) as stream:
            while chunk := stream.read(1024):
                self.chunks.append(chunk)

    def close_writer(self):
        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None

    def text(self):
        self.thread.join(timeout=.25)
        return b"".join(tuple(self.chunks)).decode("utf-8", errors="replace").strip()

    def close(self):
        self.close_writer()
        self.thread.join(timeout=1.)


def _restore_readonly_mapping(value):
    return MappingProxyType(value)


def _reduce_readonly_mapping(value):
    # The worker entry point runs as __main__. Always emit the canonical
    # importable function, not a reference to the caller's unrelated __main__.
    from .optional_native_worker import _restore_readonly_mapping
    return _restore_readonly_mapping, (dict(value),)


class _BoundedMessageStream(io.BytesIO):
    def __init__(self, maximum_bytes):
        super().__init__()
        self.maximum_bytes = maximum_bytes

    def write(self, data):
        if self.maximum_bytes is not None and self.tell() + len(data) > self.maximum_bytes:
            raise ValueError("optional IPC message exceeded its pre-reserved byte bound")
        return super().write(data)


def _encode_message(value, *, maximum_bytes=None):
    """Private IPC only: preserve immutable maps and shared object identity.

    Visual candidates and repair evidence deliberately contain mappingproxy
    fields. Do not flatten those contracts to mutable dictionaries or install
    a process-global pickle reducer. In particular, authority seals and custom
    provider callbacks receive no special serialization/bypass here.
    """
    stream = _BoundedMessageStream(maximum_bytes)
    writer = pickle.Pickler(stream, protocol=pickle.HIGHEST_PROTOCOL)
    writer.dispatch_table = copyreg.dispatch_table.copy()
    writer.dispatch_table[MappingProxyType] = _reduce_readonly_mapping
    writer.dump(value)
    return stream.getvalue()


@contextmanager
def optional_native_scope(deadline):
    token = _optional_deadline.set(deadline)
    try:
        yield
    finally:
        _optional_deadline.reset(token)


def current_optional_deadline():
    return _optional_deadline.get()


def isolated_optional(*, progress_callbacks=()):
    """Run opted-in, serializable native boundaries in a fresh interpreter.

    In-process callbacks are not sent across the process boundary. Only named
    progress notifications may be omitted; identity/provider callbacks are
    never silently removed. An unserializable proposal is rejected normally.
    Strict/non-optional callers retain the original synchronous behavior.
    """
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            deadline = current_optional_deadline()
            if deadline is None:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            for name in progress_callbacks:
                bound.arguments[name] = None
            return run_isolated_python_call(function.__module__, function.__name__,
                dict(bound.arguments), deadline=deadline)
        return wrapped
    return decorate


def run_isolated_python_call(module: str, name: str, kwargs: dict, *, deadline: float):
    from .review_first_runtime import ReviewFirstSearchCutoff, defer_deadline_during_owned_native_wait
    if os.name != "posix":
        raise RuntimeError("optional native isolation requires POSIX process groups")
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("optional worker deadline must be finite")
    if deadline <= monotonic():
        raise ReviewFirstSearchCutoff("optional native work has no admission budget")
    from .request_work_lease import current_request_work_phase, RequestWorkExhausted
    phase = current_request_work_phase()
    if phase is None:
        request = _encode_message((module, name, kwargs))
        # The worker/guardian enforce the absolute cutoff themselves. Defer our
        # competing Python alarm until the process and IPC handles are reaped.
        with defer_deadline_during_owned_native_wait():
            return _communicate_isolated(request, deadline)
    from .request_work_transport import ParentWorkTransport
    if not hasattr(phase, "begin_worker"):
        raise ValueError("a child cannot create another isolated phase owner")
    deadline = min(deadline, phase.deadline)
    key = phase.begin_worker(module, name)  # Durable parent admission BEFORE pickle/model work.
    transport = None
    stats = dict(sent_bytes=0, received_bytes=0)
    status = "failed"
    error = None
    result = None
    try:
        transport = ParentWorkTransport(phase, key)
        request = _encode_message((module, name, kwargs, transport.descriptor()),
                                  maximum_bytes=phase.maximum_worker_message_bytes)
        with defer_deadline_during_owned_native_wait():
            result = _communicate_isolated(request, deadline, transport=transport,
                maximum_result_bytes=phase.maximum_worker_message_bytes, wire_stats=stats)
        status = "returned"
    except BaseException as exc:
        error = exc
        status = "cutoff" if isinstance(exc, ReviewFirstSearchCutoff) else "failed"
    finally:
        if transport is not None:
            transport.close()  # The native group and its inherited pipe holders are reaped already.
            if transport.error is not None:
                error = transport.error
                status = "cutoff" if isinstance(error, RequestWorkExhausted) else "failed"
        phase.finish_worker(key, status=status, **stats)
    if error is not None:
        raise error
    return result


def _bounded_communicate(worker, request, deadline, maximum_result_bytes, wire_stats):
    """Bound parent output allocation even if a child violates its serializer."""
    from .review_first_runtime import ReviewFirstSearchCutoff
    done = Event()
    chunks, errors = [], []

    def send():
        try:
            # Record the measured accepted prefix, including a pipe closing
            # mid-message. Do not turn partial input into a fictitious zero.
            remaining=memoryview(request)
            while remaining:
                written=worker.stdin.write(remaining[:65536])
                if type(written) is not int or not 0<written<=min(len(remaining),65536):
                    raise OSError("owned worker input did not accept its complete prefix")
                wire_stats["sent_bytes"]+=written
                remaining=remaining[written:]
            worker.stdin.flush()
        except BrokenPipeError:
            pass
        except BaseException as exc:
            errors.append(exc)
        finally:
            try:
                worker.stdin.close()
            except BrokenPipeError:
                pass

    def receive():
        try:
            while data := worker.stdout.read(min(65536, maximum_result_bytes - wire_stats["received_bytes"] + 1)):
                wire_stats["received_bytes"] += len(data)
                if wire_stats["received_bytes"] > maximum_result_bytes:
                    raise ValueError("optional worker returned more IPC bytes than reserved")
                chunks.append(data)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    sender = Thread(target=send, daemon=True)
    receiver = Thread(target=receive, daemon=True)
    sender.start()
    receiver.start()
    while worker.poll() is None or not done.is_set():
        if errors:
            raise errors[0]
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ReviewFirstSearchCutoff("optional native worker deadline")
        done.wait(min(.025, remaining))
    if errors:
        raise errors[0]
    sender.join(timeout=.25)
    receiver.join(timeout=.25)
    if sender.is_alive() or receiver.is_alive():
        raise RuntimeError("owned bounded IPC threads did not finish")
    return b"".join(chunks), None


def _communicate_isolated(request: bytes, deadline: float, *, transport=None,
                          maximum_result_bytes=None, wire_stats=None):
    from .review_first_runtime import ReviewFirstSearchCutoff
    from .workflow_supervisor import terminate_owned_process_group
    command = [sys.executable, "-m", _WORKER_MODULE, "guard", repr(deadline), str(os.getpid())]
    inherited = () if transport is None else transport.pass_fds
    command.extend(str(fd) for fd in inherited)
    diagnostics = _WorkerStderrTail()
    worker = None
    try:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=diagnostics.write_fd, start_new_session=True,
                                  env=_worker_environment(), pass_fds=inherited)
        if transport is not None:
            transport.child_spawned()
        diagnostics.close_writer()
        try:
            if transport is None:
                result, _ = worker.communicate(input=request, timeout=max(.001, deadline-monotonic()))
            else:
                result, _ = _bounded_communicate(worker, request, deadline, maximum_result_bytes, wire_stats)
        except subprocess.TimeoutExpired as exc:
            raise ReviewFirstSearchCutoff("optional native worker deadline") from exc
        if worker.returncode != 0 or not result:
            if monotonic() >= deadline:
                raise ReviewFirstSearchCutoff("optional native worker deadline")
            # Leave room for the error prefix inside the controller's 2KB
            # audit field; the actual fatal line is normally at the end.
            detail = diagnostics.text()[-1500:]
            raise RuntimeError(f"optional native worker failed with exit code {worker.returncode}"
                               + (f"; stderr: {detail}" if detail else ""))
        ok, payload = pickle.loads(result)
        if not ok:
            raise RuntimeError(f"optional native call failed: {payload}")
        return payload
    finally:
        # Includes a worker blocked in MLX/ONNX and inherited-pipe descendants.
        # No second communicate() can wait indefinitely on a surviving pipe.
        try:
            if worker is not None:
                terminate_owned_process_group(worker)
        finally:
            if worker is not None and worker.stdin:
                try:
                    worker.stdin.close()
                except BrokenPipeError:
                    pass
            if worker is not None and worker.stdout:
                worker.stdout.close()
            diagnostics.close()


def _execute():
    from .request_work_lease import RequestWorkExhausted
    # Keep IPC on an owned fd; both Python prints and native stdout now go to
    # stderr, so model diagnostics cannot corrupt the binary return channel.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    remote = None
    maximum_bytes = None
    try:
        request = pickle.load(sys.stdin.buffer)
        if len(request) == 4:
            from .request_work_transport import RemoteWorkPhase
            from .request_work_lease import bind_request_work_phase
            module, name, kwargs, descriptor = request
            remote = RemoteWorkPhase(descriptor)
            maximum_bytes = remote.maximum_worker_message_bytes
            scope = bind_request_work_phase(remote)
        else:
            module, name, kwargs = request
            scope = nullcontext()
        with scope:
            if remote is not None:
                remote.worker_entered()  # Parent ACK precedes the target function.
            value = getattr(importlib.import_module(module), name)(**kwargs)
            encoded = _encode_message((True, value), maximum_bytes=maximum_bytes)
    except (Exception, RequestWorkExhausted) as exc:
        # Encoding itself can fail. Report that concrete failure without
        # writing a partial success payload to the anonymous IPC channel.
        encoded = _encode_message((False, f"{type(exc).__name__}: {exc}"), maximum_bytes=maximum_bytes)
    finally:
        if remote is not None:
            remote.close()
    with output:
        output.write(encoded)


def _guard(deadline: float, owner: int, inherited_fds=()):
    # The guardian stays outside the work group so it can kill ALL inherited
    # pipe holders even after the execute leader returned successfully.
    if os.getpid() != os.getpgrp():
        raise RuntimeError("native guard must own its process group")
    def cancelled(_signum, _frame):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, cancelled)
    process = None
    try:
        # Do not allow TERM between fork and recording the owned PID. SIGKILL
        # is reserved for work-group cleanup (including TERM-ignoring native code).
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            process = subprocess.Popen([sys.executable, "-m", _WORKER_MODULE, "execute"],
                                       start_new_session=True, pass_fds=inherited_fds)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        while True:
            if os.getppid() != owner or monotonic() >= deadline:
                return 124
            try:
                return process.wait(timeout=min(.05, max(.001, deadline-monotonic())))
            except subprocess.TimeoutExpired:
                pass
    finally:
        # Normal result, parent death and cancellation all close descendants'
        # inherited IPC fds BEFORE the guardian exits. Preserve the leader's
        # success return code; do not kill the guardian's own group here.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1.)


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "execute":
        _execute()
    elif mode == "guard":
        raise SystemExit(_guard(float(arguments[0]), int(arguments[1]), tuple(map(int, arguments[2:]))))
    else:
        raise SystemExit("invalid private worker mode")
