"""Immutable Dynamic Crop v2 evidence promoted to simplified v3 targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    StageEditDynamicCropTargetSequenceV3,
)
from .stage_edit_dynamic_crop_target_v3 import (
    DYNAMIC_CROP_V3_SIMPLIFICATION_CONTRACT,
)
from .stage_edit_dynamic_crop_training import (
    StageEditDynamicCropTrainingCorpusV2,
)
from .dynamic_crop_source_frame_binding import SOURCE_FRAME_BINDING_CONTRACT


DYNAMIC_CROP_TRAINING_CORPUS_V3_SCHEMA_VERSION = 1
DYNAMIC_CROP_TRAINING_CORPUS_V3_CONTRACT = (
    "immutable_v2_renderer_evidence_to_simplified_source_exact_v3_targets_v1"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropTrainingCorpusV3:
    sequences: Mapping[str, StageEditDynamicCropTargetSequenceV3]
    data_id: str
    source_v2_data_id: str
    artifact_paths: tuple[str, ...]
    statistics: Mapping[str, Any]
    target_contract: str = DYNAMIC_CROP_V3_TARGET_CONTRACT
    contract: str = DYNAMIC_CROP_TRAINING_CORPUS_V3_CONTRACT

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("dynamic crop v3 training corpus cannot be empty")
        if not self.data_id.strip() or not self.source_v2_data_id.strip():
            raise ValueError("dynamic crop v3 corpus identities cannot be blank")
        if self.target_contract != DYNAMIC_CROP_V3_TARGET_CONTRACT:
            raise ValueError("dynamic crop v3 corpus target contract differs")
        if self.contract != DYNAMIC_CROP_TRAINING_CORPUS_V3_CONTRACT:
            raise ValueError("dynamic crop v3 training corpus contract differs")
        if set(self.sequences) != {
            sequence.sequence_id for sequence in self.sequences.values()
        }:
            raise ValueError("dynamic crop v3 corpus sequence keys differ")

    def sequence_for(self, sequence_id: str) -> StageEditDynamicCropTargetSequenceV3:
        try:
            return self.sequences[sequence_id]
        except KeyError as exc:
            raise ValueError(
                f"dynamic crop v3 targets are missing sequence {sequence_id}"
            ) from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": DYNAMIC_CROP_TRAINING_CORPUS_V3_SCHEMA_VERSION,
            "contract": self.contract,
            "target_contract": self.target_contract,
            "data_id": self.data_id,
            "source_v2_data_id": self.source_v2_data_id,
            "input_policy": {
                "v2_artifacts_remain_immutable": True,
                "new_keyframes_synthesized": False,
                "observed_keyframes_moved_or_clamped": False,
                "exact_renderer_track_revalidated_after_simplification": True,
                "target_source_id_retained_for_exact_slot_resolution": True,
                "source_frame_binding_contract": SOURCE_FRAME_BINDING_CONTRACT,
                "all_original_source_frame_observations_retained": True,
                "source_frame_binding_is_model_input": False,
                "geometry_only_heads_reconcile_source_phase": False,
                "simplification_contract": (
                    DYNAMIC_CROP_V3_SIMPLIFICATION_CONTRACT
                ),
            },
            "artifact_paths": list(self.artifact_paths),
            "sequences": [
                {
                    "sequence_id": sequence_id,
                    "token_count": len(sequence.targets),
                    "track_label_count": sum(
                        target is not None for target in sequence.targets
                    ),
                    "simplified_keyframe_label_count": sum(
                        len(target.keyframes)
                        for target in sequence.targets
                        if target is not None
                    ),
                }
                for sequence_id, sequence in sorted(self.sequences.items())
            ],
            "statistics": dict(self.statistics),
        }


def build_stage_edit_dynamic_crop_training_corpus_v3(
    source: StageEditDynamicCropTrainingCorpusV2,
) -> StageEditDynamicCropTrainingCorpusV3:
    """Promote verified v2 tracks without altering their artifact evidence."""

    if not isinstance(source, StageEditDynamicCropTrainingCorpusV2):
        raise TypeError("dynamic crop v3 corpus builder requires a v2 corpus")
    sequences = {
        sequence_id: StageEditDynamicCropTargetSequenceV3.from_v2(sequence)
        for sequence_id, sequence in sorted(source.sequences.items())
    }
    original_counts: Counter[int] = Counter()
    simplified_counts: Counter[int] = Counter()
    motion_original_counts: dict[str, Counter[int]] = {
        "static": Counter(),
        "dynamic": Counter(),
    }
    motion_simplified_counts: dict[str, Counter[int]] = {
        "static": Counter(),
        "dynamic": Counter(),
    }
    reference_track_count = 0
    original_keyframes = 0
    simplified_keyframes = 0
    maximum_error = 0.0
    fingerprints: set[str] = set()
    target_payloads: list[dict[str, object]] = []
    for sequence_id, sequence in sorted(sequences.items()):
        for token_index, target in enumerate(sequence.targets):
            if target is None:
                continue
            reference_track_count += 1
            original_keyframes += target.original_keyframe_count
            simplified_keyframes += len(target.keyframes)
            original_counts[target.original_keyframe_count] += 1
            simplified_counts[len(target.keyframes)] += 1
            motion_original_counts[target.motion_label][
                target.original_keyframe_count
            ] += 1
            motion_simplified_counts[target.motion_label][len(target.keyframes)] += 1
            maximum_error = max(
                maximum_error,
                target.maximum_normalized_reconstruction_error,
            )
            fingerprints.add(target.simplification_config_fingerprint)
            target_payloads.append(
                {
                    "sequence_id": sequence_id,
                    "token_index": token_index,
                    **target.to_dict(),
                }
            )
    if reference_track_count < 1:
        raise ValueError("dynamic crop v3 corpus contains no trajectory targets")
    if len(fingerprints) != 1:
        raise ValueError("dynamic crop v3 simplification configs are inconsistent")
    statistics: dict[str, Any] = {
        "reference_count": source.statistics.get("reference_count"),
        "song_count": source.statistics.get("song_count"),
        "artifact_count": source.statistics.get("artifact_count"),
        "track_label_count": reference_track_count,
        "original_keyframe_label_count": original_keyframes,
        "source_frame_observation_count": source.statistics.get("source_frame_observation_count"),
        "source_frame_mismatch_count": source.statistics.get("source_frame_mismatch_count"),
        "tracks_requiring_source_frame_reconciliation": source.statistics.get("tracks_requiring_source_frame_reconciliation"),
        "simplified_keyframe_label_count": simplified_keyframes,
        "removed_keyframe_count": original_keyframes - simplified_keyframes,
        "original_keyframe_count_histogram": {
            str(key): value for key, value in sorted(original_counts.items())
        },
        "simplified_keyframe_count_histogram": {
            str(key): value for key, value in sorted(simplified_counts.items())
        },
        "motion_keyframe_count_histograms": {
            motion: {
                "original": {
                    str(key): value
                    for key, value in sorted(motion_original_counts[motion].items())
                },
                "simplified": {
                    str(key): value
                    for key, value in sorted(
                        motion_simplified_counts[motion].items()
                    )
                },
            }
            for motion in ("static", "dynamic")
        },
        "maximum_normalized_reconstruction_error": maximum_error,
        "simplification_config_fingerprint": next(iter(fingerprints)),
        "new_keyframe_count": 0,
        "renderer_revalidation_required": True,
    }
    identity_payload = {
        "contract": DYNAMIC_CROP_TRAINING_CORPUS_V3_CONTRACT,
        "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
        "source_v2_data_id": source.data_id,
        "artifact_paths": list(source.artifact_paths),
        "statistics": statistics,
        "targets": target_payloads,
    }
    data_id = f"stageedit-dynamic-crop-v3-{_canonical_sha256(identity_payload)[:20]}"
    return StageEditDynamicCropTrainingCorpusV3(
        sequences=sequences,
        data_id=data_id,
        source_v2_data_id=source.data_id,
        artifact_paths=tuple(source.artifact_paths),
        statistics=statistics,
    )


__all__ = [
    "DYNAMIC_CROP_TRAINING_CORPUS_V3_CONTRACT",
    "DYNAMIC_CROP_TRAINING_CORPUS_V3_SCHEMA_VERSION",
    "StageEditDynamicCropTrainingCorpusV3",
    "build_stage_edit_dynamic_crop_training_corpus_v3",
]
