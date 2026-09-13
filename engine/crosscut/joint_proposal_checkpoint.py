"""Recover source/shot proposals, never process-local verification proofs.

A render interruption must not require rediscovering an already found route.
Only identifiers are loaded; every candidate is rebound to the current lattice
and all exact boundary, bridge, editorial and decoded-output gates still run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from .timeline import TimelineCandidate
from .visual_candidate_checkpoint import canonical_json_sha256


@dataclass(frozen=True, slots=True)
class JointProposalRestart:
    paths: tuple[tuple[TimelineCandidate, ...], ...]
    checkpoint_sha256: str

    def scalar_audit(self) -> dict[str, Any]:
        return {"contract": "input_bound_source_shot_proposal_restart_v1",
                "checkpoint_sha256": self.checkpoint_sha256,
                "proposal_count": len(self.paths), "proofs_restored": False,
                "quality_gates_waived": False, "final_media_authority": False}


def rebind_joint_proposal_checkpoint(
    payload: Mapping[str, Any], *, expected_input_identity: Mapping[str, Any],
    lattice: Sequence[Sequence[TimelineCandidate]],
) -> tuple[tuple[TimelineCandidate, ...], ...]:
    """Reject changed files/clocks/policy/code and ambiguous or stale shot IDs."""
    if (payload.get("status") != "candidate_selection_complete_derivation_pending"
            or payload.get("analysis_basis") != "joint_decoded_visual_candidate_selection_v1"):
        raise ValueError("not a completed joint selection audit")
    identity = payload.get("input_identity")
    expected_sha = canonical_json_sha256(expected_input_identity)
    if (not isinstance(identity, Mapping)
            or payload.get("input_identity_sha256") != expected_sha
            or canonical_json_sha256(identity) != expected_sha):
        raise ValueError("joint proposal input identity changed")
    rows = tuple(tuple(row) for row in lattice)
    spans = identity.get("canonical_spans")
    if (not rows or any(not row for row in rows) or not isinstance(spans, list)
            or len(spans) != len(rows)):
        raise ValueError("joint proposal timeline is incomplete")
    for index, (row, span) in enumerate(zip(rows, spans, strict=True)):
        if any(c.segment_index != index or [c.start_seconds, c.end_seconds] != span for c in row):
            raise ValueError("joint proposal current lattice clock changed")
    selection = payload.get("joint_visual_selection")
    batches = selection.get("selection_batches") if isinstance(selection, Mapping) else None
    if not isinstance(batches, list) or not 1 <= len(batches) <= 32:
        raise ValueError("joint proposal selection batches missing")
    paths = []
    for batch in batches:
        raw_paths = batch.get("source_shot_paths") if isinstance(batch, Mapping) else None
        if not isinstance(raw_paths, list) or not 1 <= len(raw_paths) <= 32:
            raise ValueError("joint proposal source paths missing")
        for raw_path in raw_paths:
            if not isinstance(raw_path, list) or len(raw_path) != len(rows):
                raise ValueError("joint proposal source path is incomplete")
            path = []
            for row, pair in zip(rows, raw_path, strict=True):
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(not isinstance(value, str) or not value for value in pair)):
                    raise ValueError("joint proposal source/shot identifiers invalid")
                matches = [c for c in row if [c.source_id, c.shot_id] == pair]
                if len(matches) != 1:
                    raise ValueError("joint proposal source/shot no longer uniquely available")
                path.append(matches[0])
            path = tuple(path)
            if path not in paths:
                paths.append(path)
            if len(paths) == 3:
                return tuple(paths)
    if not paths:
        raise ValueError("joint proposal audit has no complete source path")
    return tuple(paths)


def find_joint_proposal_restart(
    output_directory: Path, *, expected_input_identity: Mapping[str, Any],
    lattice: Sequence[Sequence[TimelineCandidate]],
) -> JointProposalRestart | None:
    """Best-effort optional hints; absent/stale hints retain the normal search."""
    root = Path(output_directory)
    if root.is_symlink() or not root.is_dir():
        return None
    entries = []
    try:
        for path in root.glob("*-pre-derivation-candidate-selection.json"):
            if path.name.startswith("._"):
                continue
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode) and 0 < metadata.st_size <= 32 * 1024 * 1024:
                entries.append((metadata.st_mtime_ns, path))
    except OSError:
        return None
    for _, path in sorted(entries, reverse=True)[:16]:
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                continue
            data = path.read_bytes()
            after = path.lstat()
            if ((before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_ino, after.st_size, after.st_mtime_ns)):
                continue
            payload = json.loads(data)
            if not isinstance(payload, Mapping):
                continue
            paths = rebind_joint_proposal_checkpoint(payload,
                expected_input_identity=expected_input_identity, lattice=lattice)
            return JointProposalRestart(paths, hashlib.sha256(data).hexdigest())
        except (OSError, UnicodeError, ValueError, TypeError):
            continue
    return None
