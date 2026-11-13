"""Align crop-v2 artifacts with StageEdit causal training tokens.

Dynamic crop supervision is shot-indexed while StageEdit batches are token
indexed.  This module performs the only allowed bridge: it verifies each
immutable artifact and its referenced attribution file, matches the exact
canonical shot start to one token, and retains only targets that the real
renderer adapter can consume without a synthesized start or unsafe clamp.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .dynamic_crop_input_verification import verify_dynamic_crop_input_files
from .dynamic_crop_source_frame_binding import SOURCE_FRAME_BINDING_CONTRACT
from .reference_attribution import REFERENCE_EVIDENCE_ANALYSIS_BASIS
from .reference_dynamic_crop_supervision import (
    load_reference_dynamic_crop_supervision,
)
from .stage_edit_dynamic_crop_adapter import (
    build_stage_edit_dynamic_crop_target,
)
from .stage_edit_dynamic_crop_head_v2 import (
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropTargetSequenceV2,
)
from .stage_edit_dynamic_crop_semimarkov_adapter import (
    decode_dynamic_crop_target_v2,
)
from .stage_edit_reference_adapter import ReferenceTrainingExample


DYNAMIC_CROP_TRAINING_CORPUS_SCHEMA_VERSION = 1
DYNAMIC_CROP_TRAINING_CORPUS_CONTRACT = (
    "renderer_decodable_crop_v2_targets_aligned_to_stage_edit_tokens_v1"
)
_SAFE_REFERENCE_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reference_id_from_attribution(payload: Mapping[str, Any]) -> str:
    provenance = payload.get("provenance")
    attribution = (
        provenance.get("attribution") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(attribution, Mapping):
        raise ValueError("dynamic crop artifact lacks attribution provenance")
    path = Path(str(attribution.get("path") or ""))
    if path.is_symlink() or not path.is_file():
        raise ValueError("dynamic crop attribution provenance path is unavailable")
    content = path.read_bytes()
    expected = str(attribution.get("file_sha256") or "")
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("dynamic crop attribution provenance hash changed")
    root = json.loads(content)
    if not isinstance(root, Mapping):
        raise ValueError("dynamic crop attribution provenance root is invalid")

    # Current canonical final attributions intentionally carry their identity
    # inside the immutable reference-evidence caller contract rather than in a
    # mutable convenience field at the document root.  Older fixtures and
    # pre-canonical artifacts can still expose ``reference_id`` at the root;
    # when both identities exist they must agree exactly.
    canonical_reference_id = ""
    evidence_provenance = root.get("reference_evidence_provenance")
    if evidence_provenance is not None:
        if not isinstance(evidence_provenance, Mapping):
            raise ValueError(
                "dynamic crop attribution reference-evidence provenance is invalid"
            )
        if evidence_provenance.get("analysis") != (
            REFERENCE_EVIDENCE_ANALYSIS_BASIS
        ):
            raise ValueError(
                "dynamic crop attribution reference-evidence contract is invalid"
            )
        nested = evidence_provenance.get("provenance")
        caller = nested.get("caller") if isinstance(nested, Mapping) else None
        if not isinstance(caller, Mapping):
            raise ValueError(
                "dynamic crop attribution evidence caller provenance is missing"
            )
        canonical_reference_id = str(
            caller.get("reference_video_id") or ""
        ).strip()

    legacy_reference_id = str(root.get("reference_id") or "").strip()
    if (
        canonical_reference_id
        and legacy_reference_id
        and canonical_reference_id != legacy_reference_id
    ):
        raise ValueError("dynamic crop attribution reference ids conflict")
    reference_id = canonical_reference_id or legacy_reference_id
    if not reference_id:
        raise ValueError("dynamic crop attribution has no reference_id")
    if _SAFE_REFERENCE_ID.fullmatch(reference_id) is None:
        raise ValueError("dynamic crop attribution reference_id is unsafe")

    artifact_caller = provenance.get("caller")
    if artifact_caller is not None:
        if not isinstance(artifact_caller, Mapping):
            raise ValueError("dynamic crop artifact caller provenance is invalid")
        artifact_reference_id = str(
            artifact_caller.get("reference_id") or ""
        ).strip()
        if artifact_reference_id and artifact_reference_id != reference_id:
            raise ValueError(
                "dynamic crop artifact and attribution reference ids differ"
            )
    return reference_id


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropTrainingCorpusV2:
    sequences: Mapping[str, StageEditDynamicCropTargetSequenceV2]
    data_id: str
    artifact_paths: tuple[str, ...]
    statistics: Mapping[str, Any]
    contract: str = DYNAMIC_CROP_TRAINING_CORPUS_CONTRACT

    def __post_init__(self) -> None:
        if not self.sequences or not self.data_id.strip():
            raise ValueError("dynamic crop training corpus cannot be empty")
        if self.contract != DYNAMIC_CROP_TRAINING_CORPUS_CONTRACT:
            raise ValueError("dynamic crop training corpus contract is invalid")

    def sequence_for(self, sequence_id: str) -> StageEditDynamicCropTargetSequenceV2:
        try:
            return self.sequences[sequence_id]
        except KeyError as exc:
            raise ValueError(
                f"dynamic crop targets are missing sequence {sequence_id}"
            ) from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": DYNAMIC_CROP_TRAINING_CORPUS_SCHEMA_VERSION,
            "contract": self.contract,
            "data_id": self.data_id,
            "input_policy": {
                "crop_v2_artifacts_are_targets_only": True,
                "reference_match_diagnostics_enter_model_inputs": False,
                "exact_canonical_shot_start_token_alignment": True,
                "renderer_undecodable_targets_are_loss_masked": True,
                "missing_start_keyframes_are_not_synthesized": True,
                "unsafe_values_are_not_clamped_into_targets": True,
                "source_frame_binding_contract": SOURCE_FRAME_BINDING_CONTRACT,
                "source_frame_binding_is_model_input": False,
                "geometry_only_heads_reconcile_source_phase": False,
            },
            "artifact_paths": list(self.artifact_paths),
            "sequences": [
                {
                    "sequence_id": sequence_id,
                    "token_count": len(sequence.targets),
                    "track_label_count": sum(
                        target is not None for target in sequence.targets
                    ),
                    "keyframe_label_count": sum(
                        len(target.keyframes)
                        for target in sequence.targets
                        if target is not None
                    ),
                }
                for sequence_id, sequence in sorted(self.sequences.items())
            ],
            "statistics": dict(self.statistics),
        }


def build_stage_edit_dynamic_crop_training_corpus_v2(
    examples: Sequence[ReferenceTrainingExample],
    *,
    artifact_paths_by_reference: Mapping[str, Sequence[Path]],
    verify_input_files: bool = True,
    token_time_tolerance_seconds: float = 1.0e-6,
) -> StageEditDynamicCropTrainingCorpusV2:
    """Build K=8 target sequences aligned to the adapted StageEdit corpus."""

    normalized_examples = tuple(examples)
    if not normalized_examples:
        raise ValueError("dynamic crop corpus needs StageEdit examples")
    by_reference = {example.reference_id: example for example in normalized_examples}
    if len(by_reference) != len(normalized_examples):
        raise ValueError("StageEdit reference ids must be unique")
    unknown = set(artifact_paths_by_reference) - set(by_reference)
    if unknown:
        raise ValueError(
            "dynamic crop artifacts name unknown references: "
            + ", ".join(sorted(unknown))
        )
    if token_time_tolerance_seconds < 0.0:
        raise ValueError("dynamic crop token-time tolerance cannot be negative")

    sequences: dict[str, StageEditDynamicCropTargetSequenceV2] = {}
    artifact_paths: list[str] = []
    digest = hashlib.sha256(DYNAMIC_CROP_TRAINING_CORPUS_CONTRACT.encode())
    digest.update(SOURCE_FRAME_BINDING_CONTRACT.encode())
    total_tracks = 0
    total_keyframes = 0
    total_motion_labels = 0
    artifact_ineligible = 0
    renderer_rejected = 0
    renderer_rejection_reasons: dict[str, int] = {}
    references_with_tracks: set[str] = set()
    songs_with_tracks: set[str] = set()
    source_frame_observations = 0
    source_frame_mismatches = 0
    tracks_requiring_frame_reconciliation = 0
    verified_input_sha256_cache: dict[str, str] = {}

    for reference_id, example in sorted(by_reference.items()):
        aligned: list[DynamicCropTrajectoryTargetV2 | None] = [
            None
        ] * len(example.sequence.tokens)
        seen_shots: set[int] = set()
        for raw_path in sorted(
            (Path(value) for value in artifact_paths_by_reference.get(reference_id, ())),
            key=lambda value: str(value),
        ):
            path = raw_path.resolve()
            artifact = load_reference_dynamic_crop_supervision(
                path, verify_input_files=False
            )
            payload = artifact.payload
            if verify_input_files:
                verify_dynamic_crop_input_files(
                    payload,
                    verified_sha256_cache=verified_input_sha256_cache,
                )
            observed_reference = _reference_id_from_attribution(payload)
            if observed_reference != reference_id:
                raise ValueError(
                    f"dynamic crop artifact/reference mismatch: {path}"
                )
            shot = payload.get("shot")
            if not isinstance(shot, Mapping):
                raise ValueError("dynamic crop artifact shot is missing")
            shot_index = int(shot["shot_index"])
            if shot_index in seen_shots:
                raise ValueError(
                    f"duplicate dynamic crop shot {reference_id}:{shot_index}"
                )
            seen_shots.add(shot_index)
            artifact_paths.append(str(path))
            content = path.read_bytes()
            digest.update(reference_id.encode())
            digest.update(hashlib.sha256(content).digest())

            adapter_target = build_stage_edit_dynamic_crop_target(payload)
            if adapter_target is None:
                artifact_ineligible += 1
                continue
            trajectory = DynamicCropTrajectoryTargetV2.from_adapter_target(
                adapter_target
            )
            renderer_review = decode_dynamic_crop_target_v2(trajectory)
            if not renderer_review.eligible:
                renderer_rejected += 1
                for reason in renderer_review.rejection_reasons:
                    renderer_rejection_reasons[reason] = (
                        renderer_rejection_reasons.get(reason, 0) + 1
                    )
                continue
            master_start = float(shot["master_start_seconds"])
            matches = tuple(
                index
                for index, token in enumerate(example.sequence.tokens)
                if abs(token.master_time_seconds - master_start)
                <= token_time_tolerance_seconds
            )
            if len(matches) != 1:
                raise ValueError(
                    "dynamic crop shot start has no unique StageEdit token: "
                    f"{reference_id}:{shot_index}@{master_start:.9f}s"
                )
            token_index = matches[0]
            if aligned[token_index] is not None:
                raise ValueError("multiple dynamic crop targets map to one token")
            aligned[token_index] = trajectory
            total_tracks += 1
            binding = trajectory.source_frame_binding
            if binding is None:
                raise AssertionError("artifact adapter lost source-frame provenance")
            source_frame_observations += len(binding.observations)
            source_frame_mismatches += sum(delta != 0 for delta in binding.frame_deltas)
            tracks_requiring_frame_reconciliation += int(binding.requires_source_frame_reconciliation)
            total_keyframes += len(trajectory.keyframes)
            total_motion_labels += int(
                not trajectory.motion_label_loss_masked
            )
            references_with_tracks.add(reference_id)
            if example.song_id is not None:
                songs_with_tracks.add(str(example.song_id))
        sequence = StageEditDynamicCropTargetSequenceV2(
            sequence_id=example.sequence.sequence_id,
            targets=tuple(aligned),
        )
        sequences[sequence.sequence_id] = sequence
        digest.update(sequence.sequence_id.encode())
        digest.update(
            _canonical_bytes(
                [
                    None
                    if target is None
                    else {
                        "shot_index": target.shot_index,
                        "artifact_content_sha256": (
                            target.artifact_content_sha256
                        ),
                    }
                    for target in sequence.targets
                ]
            )
        )

    statistics = {
        "reference_count": len(normalized_examples),
        "song_count": len(
            {
                str(example.song_id)
                for example in normalized_examples
                if example.song_id is not None
            }
        ),
        "artifact_count": len(artifact_paths),
        "artifact_ineligible_count": artifact_ineligible,
        "renderer_rejected_artifact_count": renderer_rejected,
        "renderer_rejection_reasons": dict(
            sorted(renderer_rejection_reasons.items())
        ),
        "track_label_count": total_tracks,
        "source_frame_observation_count": source_frame_observations,
        "source_frame_mismatch_count": source_frame_mismatches,
        "tracks_requiring_source_frame_reconciliation": tracks_requiring_frame_reconciliation,
        "keyframe_label_count": total_keyframes,
        "motion_label_count": total_motion_labels,
        "reference_count_with_tracks": len(references_with_tracks),
        "song_count_with_tracks": len(songs_with_tracks),
    }
    return StageEditDynamicCropTrainingCorpusV2(
        sequences=sequences,
        data_id=f"stageedit-dynamic-crop-v2-{digest.hexdigest()[:20]}",
        artifact_paths=tuple(sorted(artifact_paths)),
        statistics=statistics,
    )


__all__ = [
    "DYNAMIC_CROP_TRAINING_CORPUS_CONTRACT",
    "DYNAMIC_CROP_TRAINING_CORPUS_SCHEMA_VERSION",
    "StageEditDynamicCropTrainingCorpusV2",
    "build_stage_edit_dynamic_crop_training_corpus_v2",
]
