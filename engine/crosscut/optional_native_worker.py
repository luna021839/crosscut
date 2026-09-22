"""Killable optional native work with a guardian and anonymous-pipe IPC.

Only our own parent/child messages use pickle. No downloaded input is unpickled
and no raw descriptor, face vector, or model result is persisted for IPC.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import copyreg
import importlib
import inspect
import io
import math
import os
import pickle
import signal
import subprocess
import sys
from time import monotonic
from types import MappingProxyType

_optional_deadline = ContextVar("optional_native_deadline", default=None)
_WORKER_MODULE = "crosscut.optional_native_worker"


def _restore_readonly_mapping(value):
    return MappingProxyType(value)


def _reduce_readonly_mapping(value):
    # The worker entry point runs as __main__. Always emit the canonical
    # importable function, not a reference to the caller's unrelated __main__.
    from .optional_native_worker import _restore_readonly_mapping
    return _restore_readonly_mapping, (dict(value),)


def _encode_message(value):
    """Private IPC only: preserve immutable maps and shared object identity.

    Visual candidates and repair evidence deliberately contain mappingproxy
    fields. Do not flatten those contracts to mutable dictionaries or install
    a process-global pickle reducer. In particular, authority seals and custom
    provider callbacks receive no special serialization/bypass here.
    """
    stream = io.BytesIO()
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
    request = _encode_message((module, name, kwargs))
    # The worker/guardian enforce the absolute cutoff themselves. Defer our
    # competing Python alarm until the process and IPC handles are reaped.
    with defer_deadline_during_owned_native_wait():
        return _communicate_isolated(request, deadline)


def _communicate_isolated(request: bytes, deadline: float):
    from .review_first_runtime import ReviewFirstSearchCutoff
    from .workflow_supervisor import terminate_owned_process_group
    command = [sys.executable, "-m", _WORKER_MODULE, "guard", repr(deadline), str(os.getpid())]
    worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        try:
            result, _ = worker.communicate(input=request, timeout=max(.001, deadline-monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise ReviewFirstSearchCutoff("optional native worker deadline") from exc
        if worker.returncode != 0 or not result:
            if monotonic() >= deadline:
                raise ReviewFirstSearchCutoff("optional native worker deadline")
            raise RuntimeError(f"optional native worker failed with exit code {worker.returncode}")
        ok, payload = pickle.loads(result)
        if not ok:
            raise RuntimeError(f"optional native call failed: {payload}")
        return payload
    finally:
        # Includes a worker blocked in MLX/ONNX and inherited-pipe descendants.
        # No second communicate() can wait indefinitely on a surviving pipe.
        try:
            terminate_owned_process_group(worker)
        finally:
            if worker.stdin:
                try:
                    worker.stdin.close()
                except BrokenPipeError:
                    pass
            if worker.stdout:
                worker.stdout.close()


def _execute():
    # Keep IPC on an owned fd; both Python prints and native stdout now go to
    # stderr, so model diagnostics cannot corrupt the binary return channel.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        module, name, kwargs = pickle.load(sys.stdin.buffer)
        value = getattr(importlib.import_module(module), name)(**kwargs)
        encoded = _encode_message((True, value))
    except Exception as exc:
        # Encoding itself can fail. Report that concrete failure without
        # writing a partial success payload to the anonymous IPC channel.
        encoded = _encode_message((False, f"{type(exc).__name__}: {exc}"))
    with output:
        output.write(encoded)


def _guard(deadline: float, owner: int):
    # The guard and native child share this *owned* session. Killing this
    # group also stops the native child if the app/analysis parent vanished.
    if os.getpid() != os.getpgrp():
        raise RuntimeError("native guard must own its process group")
    process = subprocess.Popen([sys.executable, "-m", _WORKER_MODULE, "execute"])
    while True:
        if os.getppid() != owner or monotonic() >= deadline:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        try:
            return process.wait(timeout=min(.05, max(.001, deadline-monotonic())))
        except subprocess.TimeoutExpired:
            pass


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "execute":
        _execute()
    elif mode == "guard":
        raise SystemExit(_guard(float(arguments[0]), int(arguments[1])))
    else:
        raise SystemExit("invalid private worker mode")
