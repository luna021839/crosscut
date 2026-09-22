"""Measurement completeness for both proxy and final member continuity reports.

A PASS flag is not evidence. This contract validates the serialized observation
domain independently of editorial thresholds, which remain owned by the actual
continuity evaluator.
"""
from collections.abc import Mapping
import math


def _measurement(record: Mapping, key: str) -> bool:
    """Explicitly absent observations are not the same as missing fields."""
    return key in record and (record[key] is None or (
        type(record[key]) in (int, float) and math.isfinite(record[key])
        and 0 <= record[key] <= 1))


def _matches(value: object, expected: float | None) -> bool:
    if expected is None:
        return value is None
    return (type(value) in (int, float) and math.isfinite(value)
            and math.isclose(value, expected, rel_tol=0., abs_tol=1e-9))


def _p95(values: list) -> float | None:
    observed = sorted(value for value in values if value is not None)
    return observed[math.ceil(.95 * len(observed)) - 1] if observed else None


def _motion_measurements_valid(value: object, *, frame: int, fps: float) -> bool:
    """Validate observed four-frame scalars, not an editorial motion threshold."""
    if not isinstance(value, Mapping):
        return False
    basis = value.get('motion_basis')
    contracts = {
        'aggregate_pose_silhouette_subject_v1': 'cut_adjacent_exact_entry_output_space_pose_motion_v3',
        'focal_common_joint_articulation_v1': 'cut_adjacent_focal_articulation_separate_framing_v4',
    }
    if (not isinstance(basis, str) or basis not in contracts
            or value.get('contract') != contracts[basis]
            or value.get('subject_motion_separate_from_articulation') is not
                (basis == 'focal_common_joint_articulation_v1')
            or any(type(value.get(key)) is not bool for key in ('timestamp_bound_adjacent', 'both_moving'))
            or any(value.get(key) is not True for key in ('output_space_subject_motion_used',
                'direction_cosine_influenced_gate_or_bonus', 'motion_speed_and_pose_phase_influenced_gate_or_bonus'))
            or any(value.get(key) is not False for key in ('sparse_or_nonadjacent_evidence_used', 'raw_vectors_persisted'))):
        return False
    # These are samples of the final decoded video, not source-media clocks.
    for key in ('master_timestamps_seconds', 'source_local_timestamps_seconds'):
        times = value.get(key)
        if not isinstance(times, Mapping) or any(not _matches(times.get(name), index / fps)
                for name, index in zip(('outgoing_previous', 'outgoing_endpoint', 'incoming_start', 'incoming_next'),
                                       (frame-2, frame-1, frame, frame+1))):
            return False
    magnitudes, counts = value.get('motion_magnitudes'), value.get('motion_component_counts')
    if not isinstance(magnitudes, Mapping) or not isinstance(counts, Mapping):
        return False
    for side in ('outgoing', 'incoming'):
        for name in ('pose', 'silhouette', 'subject', 'aggregate', 'decision'):
            key = side + '_' + name
            if key not in magnitudes:
                return False
            number = magnitudes[key]
            if number is not None and (type(number) not in (int, float)
                    or not math.isfinite(number) or number < 0):
                return False
        if (not _matches(magnitudes[side+'_decision'], magnitudes[side+'_aggregate'])
                or type(counts.get(side)) is not int
                or counts[side] != sum(magnitudes[side+'_'+name] is not None
                    for name in ('pose', 'silhouette', 'subject'))):
            return False
    for key in ('endpoint_pose_shape_mismatch', 'endpoint_pose_shape_compatibility',
                'motion_speed_ratio_mismatch', 'subject_motion_speed_ratio_mismatch',
                'pose_phase_continuation_mismatch'):
        if not _measurement(value, key):
            return False
    mismatch = value['endpoint_pose_shape_mismatch']
    if not _matches(value['endpoint_pose_shape_compatibility'], None if mismatch is None else 1.-mismatch):
        return False
    for key in ('direction_cosine_audit_only', 'direction_cosine', 'subject_direction_cosine'):
        if key not in value:
            return False
        number = value[key]
        if number is not None and (type(number) not in (int, float)
                or not math.isfinite(number) or not -1 <= number <= 1):
            return False
    preference = value.get('soft_preference_applied')
    return (type(preference) in (int, float) and math.isfinite(preference) and preference >= 0
            and _matches(value['direction_cosine'], value['direction_cosine_audit_only']))


def member_continuity_measurements_valid(value: object, *, plan: Mapping | None = None) -> bool:
    if not isinstance(value, Mapping):
        return False
    metrics = value.get("metrics")
    records = value.get("boundaries")
    reasons = value.get("failure_reasons")
    failures = value.get("failure_boundary_indices")
    passed = value.get("passed")
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("analysis") != "automatic_member_exact_rendered_boundary_continuity_v1"
            or type(passed) is not bool
            or not isinstance(metrics, Mapping) or not isinstance(records, list)
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) or not reason.strip() for reason in reasons)
            or len(set(reasons)) != len(reasons)
            or not isinstance(failures, Mapping)
            or passed != (not reasons)
            or (passed and failures)):
        return False
    frame_count = metrics.get("output_frame_count")
    fps = metrics.get("fps")
    counts = [metrics.get(name) for name in (
        "source_switch_boundary_count", "hard_cut_source_switch_count", "effect_source_switch_count")]
    if (type(frame_count) is not int or frame_count <= 4
            or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or metrics.get("exact_incoming_first_frame_evaluated") is not True
            or metrics.get("final_render_transform_domain_evaluated") is not True
            or any(type(count) is not int or count < 0 for count in counts)
            or counts[0] != counts[1] + counts[2] or counts[0] != len(records)
            or (passed and not records)):
        return False
    observed = {}
    hard_count = 0
    hard_records = []
    effect_steps = []
    for record in records:
        if not isinstance(record, Mapping):
            return False
        index = record.get("boundary_index")
        frame = record.get("output_frame_index")
        outgoing, incoming = record.get("outgoing_source_index"), record.get("incoming_source_index")
        kind = record.get("transition_kind")
        if (type(index) is not int or index < 0 or index in observed
                or type(frame) is not int or not 0 < frame < frame_count
                or any(type(source) is not int or source < 0 for source in (outgoing, incoming))
                or outgoing == incoming or not isinstance(kind, str) or kind not in {
                    "hard_cut", "face_dissolve", "member_zoom_bridge", "member_blur_bridge"}
                or type(record.get("descriptors_available")) is not bool
                or (passed and record["descriptors_available"] is not True)):
            return False
        if (not _matches(record.get("output_seconds"), frame / fps)
                or not all(_measurement(record, key) for key in (
                    "attention_position_mismatch", "attention_scale_mismatch", "pose_mismatch"))
                or any(type(record.get(key)) is not bool for key in (
                    "focal_face_reference_guided", "body_reference_guided"))):
            return False
        if kind == "hard_cut":
            if (not _measurement(record, "silhouette_mismatch")
                    or record.get("sample_frame_indices") != [frame-2, frame-1, frame, frame+1]
                    or frame < 2 or frame + 1 >= frame_count
                    or not _motion_measurements_valid(record.get("motion"), frame=frame, fps=fps)):
                return False
            hard_records.append(record)
        else:
            count = record.get("effect_frame_count")
            steps = record.get("endpoint_steps")
            if (type(count) is not int or count < 1 or frame + count >= frame_count
                    or record.get("pure_endpoint_sample_frame_indices") != [frame-1, frame+count]
                    or not isinstance(steps, list) or len(steps) != 2):
                return False
            for step, label, frames in zip(steps, ("effect_entry", "effect_exit"),
                    ([frame-1, frame], [frame+count-1, frame+count])):
                if (not isinstance(step, Mapping) or step.get("label") != label
                        or step.get("sample_frame_indices") != frames
                        or not all(_measurement(step, key) for key in (
                            "attention_position_step", "attention_scale_step", "pose_step", "silhouette_step"))):
                    return False
                effect_steps.append(step)
        hard_count += kind == "hard_cut"
        observed[index] = (frame, outgoing, incoming, kind,
                           0 if kind == "hard_cut" else count)
    if hard_count != counts[1]:
        return False
    pose_count = sum(row["pose_mismatch"] is not None for row in hard_records)
    attention_rows = [row for row in hard_records if row["attention_position_mismatch"] is not None
                      and row["attention_scale_mismatch"] is not None]
    derived_counts = {
        "actual_pose_boundary_count": pose_count,
        "attention_boundary_count": len(attention_rows),
        "effect_endpoint_pair_count": len(effect_steps),
        "effect_endpoint_attention_count": sum(row["attention_position_step"] is not None
            and row["attention_scale_step"] is not None for row in effect_steps),
        "effect_endpoint_pose_count": sum(row["pose_step"] is not None for row in effect_steps),
    }
    if any(type(metrics.get(key)) is not int or metrics[key] != expected
           for key, expected in derived_counts.items()):
        return False
    if any(not _matches(metrics.get(key), count / hard_count if hard_count else 0.)
           for key, count in (("actual_pose_coverage_fraction", pose_count),
                              ("attention_coverage_fraction", len(attention_rows)))):
        return False
    if type(metrics.get("focal_reference_authority_active")) is not bool:
        return False
    strict_rows = ([row for row in attention_rows if row["focal_face_reference_guided"]
                    or row["body_reference_guided"]]
                   if metrics["focal_reference_authority_active"] else attention_rows)
    for field, rows in (("attention_position_mismatch", strict_rows),
                        ("attention_scale_mismatch", strict_rows),
                        ("pose_mismatch", hard_records), ("silhouette_mismatch", hard_records)):
        key = field + "_p95"
        if key not in metrics or not _matches(metrics[key], _p95([row[field] for row in rows])):
            return False
    for reason, indices in failures.items():
        if (reason not in reasons or not isinstance(indices, list) or not indices
                or any(type(index) is not int or index not in observed for index in indices)
                or len(set(indices)) != len(indices)):
            return False
    if plan is not None:
        segments, transitions = plan.get("segments"), plan.get("transitions")
        if (not isinstance(segments, list) or not isinstance(transitions, list)
                or len(transitions) != len(segments) - 1
                or frame_count != plan.get("output_frame_count") or fps != plan.get("fps")):
            return False
        expected = {}
        seconds = 0.0
        for index, (left, right, transition) in enumerate(zip(segments, segments[1:], transitions)):
            if not all(isinstance(item, Mapping) for item in (left, right, transition)):
                return False
            duration = left.get("duration")
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
                return False
            seconds += duration
            if left.get("source_index") != right.get("source_index"):
                expected[index] = (round(seconds * fps), left.get("source_index"),
                                   right.get("source_index"), transition.get("kind"),
                                   transition.get("frame_count"))
        if observed != expected:
            return False
    return True
