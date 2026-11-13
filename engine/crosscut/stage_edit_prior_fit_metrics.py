"""Read-only fit diagnosis on the already materialized TRAIN trajectories.

This does not run the encoder, construct an optimizer, select a checkpoint or
consume validation data. It separates failing to fit TRAIN from failing to
generalize. Original observations and the real renderer own the curve metric.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .stage_edit_decoded_curve_metrics import (
    decoded_observation_error,
    summarize_decoded_observations,
)
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION,
    dynamic_crop_v3_times_by_count,
    renderer_safe_values_from_raw_v3,
)
from .stage_edit_dynamic_crop_semimarkov_adapter_v3 import DynamicCropHeadPredictionV3

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional away from Apple Silicon.
    mx = None


def evaluate_materialized_prior_fit(
    tracks: Sequence[Any],
    snapshot: Mapping[str, Any],
    *,
    reference_song_ids: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate the exact merged affine head on its detached TRAIN inputs.

    Target source and duration are conditional, exactly as in the validation
    curve metric. No end-to-end source-selection or visual-quality claim is
    implied. Rejected curves stay rejected and make coverage incomplete.
    """
    if mx is None:
        raise RuntimeError("Materialized prior fit diagnosis requires MLX")
    if not tracks:
        raise ValueError("A nonempty frozen TRAIN inventory is required")
    for track in tracks:
        if (reference_song_ids.get(track.reference_id) != track.song_id
                or not track.original_observed_keyframes
                or track.context_embedding is None or track.source_embedding is None
                or not np.isfinite(track.shot_duration_seconds)
                or track.shot_duration_seconds <= 0):
            raise ValueError("Foreign or incomplete TRAIN observation in prior fit diagnosis")
    contexts = mx.array(np.stack([t.context_embedding for t in tracks]))
    sources = mx.array(np.stack([t.source_embedding for t in tracks]))
    # Exactly Linear+bias/Linear without constructing randomly initialized
    # modules: a read-only report must not perturb the training dropout RNG.
    raw = mx.addmm(snapshot['context_bias'], contexts - snapshot['context_center'],
                   snapshot['context_weight'].T) + mx.matmul(
        sources - snapshot['source_center'], snapshot['source_weight'].T)
    if raw.shape != (len(tracks), DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION):
        raise ValueError("One complete actual merged head output per TRAIN track required")
    values = renderer_safe_values_from_raw_v3(raw[:, :32].reshape(len(tracks), 8, 4))
    times = dynamic_crop_v3_times_by_count(raw[:, 40:75])
    mx.eval(raw, values, times)
    raw_np, values_np, times_np = map(np.asarray, (raw, values, times))
    if not all(np.isfinite(a).all() for a in (raw_np, values_np, times_np)):
        raise ValueError("Nonfinite actual merged prior outputs")
    errors = []
    by_song: dict[str, list[Any]] = {}
    rejected = Counter()
    targets, predictions, time_errors, motion_rows = [], [], [], []
    for i, track in enumerate(tracks):
        count = len(track.values)
        targets.append(count - 1)
        predictions.append(int(raw_np[i, 32:40].argmax()))
        time_errors.extend(np.abs(times_np[i, count - 1, :count] - track.normalized_times).tolist())
        if not track.motion_label_loss_masked:
            motion_rows.append((track.motion_label == 'dynamic', bool(raw_np[i, 75] >= 0)))
        prediction = DynamicCropHeadPredictionV3(
            source_id=track.source_id, source_index=0,
            values=tuple(map(tuple, values_np[i].astype(float))),
            count_logits=tuple(raw_np[i, 32:40].astype(float)),
            time_interval_logits=tuple(raw_np[i, 40:75].astype(float)),
            motion_logit=float(raw_np[i, 75]),
        )
        decoded = prediction.decode(float(track.shot_duration_seconds))
        if not decoded.eligible or decoded.minimum_safe_zoom_projection_count:
            rejected['renderer_rejected_or_projected'] += 1
            continue
        row = decoded_observation_error(decoded, track.original_observed_keyframes,
                                        expected_source_id=track.source_id)
        errors.append(row)
        by_song.setdefault(track.song_id, []).append(row)
    confusion = np.zeros((8, 8), int)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target, prediction] += 1
    recalls = [float(confusion[i, i] / confusion[i].sum()) for i in range(8) if confusion[i].sum()]
    motion_recalls = [float(np.mean([p == t for t, p in motion_rows if t == value]))
                      for value in (False, True) if any(t == value for t, _ in motion_rows)]
    song_counts = Counter(t.song_id for t in tracks)
    return dict(
        contract='all_materialized_train_tracks_exact_merged_head_original_clock_fit_v1',
        diagnostic_only=True, training_reference_ids=sorted(reference_song_ids),
        track_count=len(tracks), decoder_rejections=dict(rejected),
        decoded_observed_time=summarize_decoded_observations(errors, expected_tracks=len(tracks), source_mismatches=0),
        per_song_decoded_observed_time={song: summarize_decoded_observations(by_song.get(song, []),
            expected_tracks=count, source_mismatches=0) for song, count in sorted(song_counts.items())},
        count_exact_accuracy=float(np.mean(np.asarray(targets) == predictions)),
        count_macro_recall=float(np.mean(recalls)),
        count_confusion_target_rows_predicted_columns=confusion.tolist(),
        teacher_forced_count_normalized_time_mae=float(np.mean(time_errors)),
        motion_balanced_accuracy=float(np.mean(motion_recalls)) if motion_recalls else None,
        conditional_on_target_source_and_duration=True, encoder_forward_count=0,
        parameter_updates=0, validation_or_heldout_data_consumed=False,
        checkpoint_selection_or_qualification_authority=False,
        raw_features_or_embeddings_exported=False,
    )
