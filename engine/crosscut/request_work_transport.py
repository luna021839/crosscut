"""Anonymous-pipe requests to a live parent phase; no copied budget authority."""
import json
import os
import struct
from threading import Lock, Thread
from uuid import uuid4

from .request_work_lease import RequestWorkExhausted

MAX_CONTROL_BYTES = 4096
MAX_ACK_BYTES = 1024


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read_exact(stream, count):
    chunks, remaining = [], count
    while remaining:
        data = stream.read(remaining)
        if not data:
            if remaining == count:
                return None
            raise EOFError("owned work pipe lost a message prefix")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _read_frame(stream, maximum, before_payload=None):
    header = _read_exact(stream, 4)
    if header is None:
        return None
    count = struct.unpack(">I", header)[0]
    if not 0 < count <= maximum:
        raise ValueError("owned work control message exceeds its declared bound")
    if before_payload is not None:before_payload(count)
    data = _read_exact(stream, count)
    if data is None:
        raise EOFError("owned work pipe omitted its declared message")
    return data


def _write_frame(stream, data):
    # Buffered writes handle partial pipe writes without retaining unbounded data.
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


class ParentWorkTransport:
    def __init__(self, phase, worker_key):
        self.phase, self.worker_key = phase, worker_key
        self.request_read, self.request_write = os.pipe()
        self.reply_read, self.reply_write = os.pipe()
        self.nonce = uuid4().hex
        self.sequence = 0
        self.error = None
        self.closed = False
        self.thread = Thread(target=self._serve, daemon=True)
        self.thread.start()

    @property
    def pass_fds(self):
        return self.request_write, self.reply_read

    def descriptor(self):
        return dict(request_fd=self.request_write, reply_fd=self.reply_read,
                    nonce=self.nonce, worker_key=self.worker_key, phase=self.phase.name,
                    deadline=self.phase.deadline,
                    maximum_worker_message_bytes=self.phase.maximum_worker_message_bytes)

    def child_spawned(self):
        for name in ("request_write", "reply_read"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def _dispatch(self, row):
        if (not isinstance(row, dict) or set(row) != {"nonce", "sequence", "action", "arguments"}
                or row["nonce"] != self.nonce or type(row["sequence"]) is not int
                or row["sequence"] != self.sequence or not isinstance(row["arguments"], dict)):
            raise ValueError("worker omitted or changed its parent-owned request sequence")
        self.sequence += 1
        action, arguments = row["action"], row["arguments"]
        if action == "worker_entered" and arguments == {"key": self.worker_key}:
            return self.phase.worker_entered(self.worker_key)
        permitted = {"charge": {"kind", "count", "model"}, "reserve": {"amounts"},
                     "start": {"key"}, "settle": {"key", "evidence_bytes"},"remaining_capacity":{"kinds"}}
        if action not in permitted or set(arguments) != permitted[action]:
            raise ValueError("worker requested an undeclared phase operation")
        return getattr(self.phase, action)(**arguments)

    def _serve(self):
        with os.fdopen(self.request_read, "rb") as requests, os.fdopen(self.reply_write, "wb") as replies:
            try:
                while mode := _read_exact(requests,1):
                    if mode not in (b'\x00',b'\x01'):
                        raise ValueError('owned work pipe changed its admission/terminal mode')
                    terminal=mode==b'\x01'
                    # Decide capacity from the bounded header BEFORE allocating
                    # the declared control payload. A terminal uses the 2048B
                    # allowance already charged when its primitive was issued.
                    def admit(count):
                        if not terminal:self.phase.charge('ipc_bytes',count+5+MAX_ACK_BYTES)
                    data=_read_frame(requests,2048-5-MAX_ACK_BYTES if terminal else MAX_CONTROL_BYTES,admit)
                    if data is None:raise EOFError('owned work pipe omitted its framed operation')
                    try:
                        # Parent charges the bounded control envelope too, before
                        # granting native work. ACK capacity is a conservative
                        # upper bound, not fictitious measured wire traffic.
                        row = json.loads(data)
                        if terminal!=(isinstance(row,dict) and row.get('action')=='settle'):
                            raise ValueError('native terminal cannot borrow ordinary admission IPC')
                        result = self._dispatch(row)
                        ack = dict(ok=True, result=result)
                    except BaseException as exc:
                        if self.error is None:
                            self.error = exc
                        ack = dict(ok=False, error_type=type(exc).__name__,
                                   error=str(exc)[:256],
                                   exhausted=exc.to_dict() if isinstance(exc, RequestWorkExhausted) else None)
                    payload = _wire(ack)
                    if len(payload) > MAX_ACK_BYTES:
                        raise ValueError("work ACK exceeded its precharged bound")
                    _write_frame(replies, payload)
            except BaseException as exc:
                if self.error is None:
                    self.error = exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.child_spawned()
        # The worker's entire process group must have been reaped first.
        self.thread.join(timeout=1.)
        if self.thread.is_alive() and self.error is None:
            self.error = RuntimeError("owned work pipe did not close after worker cleanup")


class RemoteWorkPhase:
    def __init__(self, descriptor):
        required = {"request_fd", "reply_fd", "nonce", "worker_key", "phase", "deadline",
                    "maximum_worker_message_bytes"}
        if (not isinstance(descriptor, dict) or set(descriptor) != required
                or any(type(descriptor[k]) is not int or descriptor[k] < 0
                       for k in ("request_fd", "reply_fd"))
                or not isinstance(descriptor["nonce"], str) or len(descriptor["nonce"]) != 32):
            raise ValueError("worker phase transport is not an owned anonymous-pipe descriptor")
        self.owner_pid = os.getpid()
        self.name, self.deadline = descriptor["phase"], descriptor["deadline"]
        self.maximum_worker_message_bytes = descriptor["maximum_worker_message_bytes"]
        self.worker_key, self.nonce = descriptor["worker_key"], descriptor["nonce"]
        self.sequence = 0
        self.requests = os.fdopen(descriptor["request_fd"], "wb")
        self.replies = os.fdopen(descriptor["reply_fd"], "rb")
        self._lock = Lock()

    def _call(self, action, **arguments):
        if os.getpid() != self.owner_pid:
            raise ValueError("forked descendants cannot clone a phase request cursor")
        with self._lock:
            data = _wire(dict(nonce=self.nonce, sequence=self.sequence, action=action, arguments=arguments))
            if len(data) > MAX_CONTROL_BYTES:
                raise ValueError("worker work request exceeded the bounded control envelope")
            self.requests.write(b'\x01' if action=='settle' else b'\x00')
            _write_frame(self.requests, data)
            response = _read_frame(self.replies, MAX_ACK_BYTES)
            if response is None:
                raise RuntimeError("parent closed admission before granting work")
            ack = json.loads(response)
            self.sequence += 1
            if ack.get("ok") is not True:
                exhausted = ack.get("exhausted")
                if exhausted is not None:
                    raise RequestWorkExhausted(exhausted["phase"], exhausted["dimension"],
                                               partial=exhausted["status"] == "budget_partial")
                raise RuntimeError("parent work admission failed: " + str(ack.get("error", "")))
            return ack.get("result")

    def charge(self, kind, count=1, *, model=None):
        return self._call("charge", kind=kind, count=count, model=model)

    def remaining_capacity(self,kinds):
        return self._call('remaining_capacity',kinds=list(kinds))

    def reserve(self, amounts):
        return self._call("reserve", amounts=amounts)

    def start(self, key):
        return self._call("start", key=key)

    def settle(self, key, evidence_bytes):
        return self._call("settle", key=key, evidence_bytes=evidence_bytes)

    def worker_entered(self):
        return self._call("worker_entered", key=self.worker_key)

    def close(self):
        self.requests.close()
        self.replies.close()
