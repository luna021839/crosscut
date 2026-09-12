"""Small, read-only runtime observations for the automatic joint editor.

These observations never authorize a candidate, change a cache key, or retain
frames. They expose existing counters without building the provider's full
proof audit at every progress event.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import re
from time import monotonic
from typing import Any, Callable

from .candidate_stream_verdict import TerminalCandidateVerdict


_PROVIDER_COUNTERS = (
    "descriptor_decode_count",
    "transformed_descriptor_decode_count",
    "face_identity_model_call_count",
    "boundary_materialization_rehydration_count",
    "bridge_materialization_rehydration_count",
)
_PERSISTENT_COUNTERS = {
    "request_hit_count": "scalar_bridge_persistent_cache_hit_count",
    "request_miss_count": "scalar_bridge_persistent_cache_miss_count",
    "atomic_write_count": "scalar_bridge_persistent_cache_write_count",
    "invalid_entry_count": "scalar_bridge_persistent_cache_invalid_entry_count",
}
_REASON_FIELDS = frozenset((
    "issue_reasons", "failure_codes", "reason_codes",
    "technical_integrity_issue_reasons",
))
_REASON_CODE = re.compile(r"[a-z][a-z0-9_:-]{0,159}\Z")


def _counter(provider: Any, name: str) -> int | None:
    value = getattr(provider, name, None)
    return value if type(value) is int and value >= 0 else None


class JointSelectionRuntime:
    """Elapsed time and provider-local counters for one selection batch."""

    def __init__(
        self, provider: Any, *, batch_index: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.provider = provider
        self.batch_index = batch_index
        self._clock = clock
        self._started = clock()

    def snapshot(self) -> dict[str, Any]:
        bank = getattr(self.provider, "_observation_bank", None)
        return {
            "contract": "joint_selection_runtime_v1",
            "batch_index": self.batch_index,
            "batch_elapsed_seconds": round(max(0.0, self._clock() - self._started), 3),
            "counter_scope": "current_provider_instance",
            "provider_work": {
                name: _counter(self.provider, name) for name in _PROVIDER_COUNTERS
            },
            "persistent_scalar_cache": {
                label: _counter(self.provider, name)
                for label, name in _PERSISTENT_COUNTERS.items()
            },
            "final_media_authority": False,
            "observation_reuse": bank.diagnostics() if bank is not None else None,
        }


def joint_candidate_terminal_payload(
    verdict: TerminalCandidateVerdict,
    *, elapsed_seconds: float,
    failures: Sequence[Mapping[str, Any]],
    budget: Mapping[str, Any],
    decoded_pass_count: int,
) -> dict[str, Any]:
    """Summarize the existing verdict and reported issue codes, not new QA.

    Reported issues may include tolerated outliers. Only ``status`` and
    ``blocker_codes`` identify the terminal decision. Nested file paths,
    descriptors, free-text errors and proof records are never forwarded.
    """
    matching = [
        item for item in failures
        if item.get("candidate_index") == verdict.candidate_index
    ]
    counts: Counter[str] = Counter()
    pending: list[Any] = list(matching)
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if not isinstance(node, (Mapping, list, tuple)) or id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in _REASON_FIELDS and isinstance(value, (list, tuple)):
                    counts.update(
                        code for code in value
                        if isinstance(code, str) and _REASON_CODE.fullmatch(code)
                    )
                elif isinstance(value, (Mapping, list, tuple)):
                    pending.append(value)
        else:
            pending.extend(node)
    stages = sorted({
        value for item in matching
        if isinstance(value := item.get("stage"), str)
        and _REASON_CODE.fullmatch(value)
    })
    return {
        "contract": "joint_candidate_terminal_v1",
        "candidate_index": verdict.candidate_index,
        "status": verdict.status.value,
        "elapsed_seconds": round(max(0.0, elapsed_seconds), 3),
        "blocker_codes": [item.code for item in verdict.blockers],
        "failure_stages": stages,
        "reported_issue_counts": dict(sorted(counts.items())),
        "reported_issues_are_not_additional_gate_decisions": True,
        "decoded_render_budget": dict(budget),
        "decoded_pass_count": decoded_pass_count,
        "final_media_authority": False,
    }
