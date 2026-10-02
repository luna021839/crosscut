"""Grounded, alternative-preserving instance prompts from actual raw rows.

Prompts are model inputs, never a semantic identity or visible-surface proof.
Each ambiguous face/body association gets its OWN hypothesis, not the union of
all possible bodies. Raw off-frame or low-confidence joints stay in the input
row, with explicit reasons for not using them as positive prompt coordinates.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True, slots=True)
class NativeInstancePromptHypothesis:
    raw_observation_sha256: str
    face_observation_id: int
    body_observation_id: int
    point_coords_xy: tuple
    point_labels: tuple
    box_xyxy: tuple
    used_raw_joint_indices: tuple
    unused_raw_joints: tuple
    policy_sha256: str

    def to_dict(self):
        return dict(contract='native_raw_pose_grounded_instance_prompt_hypothesis_v2',
            raw_observation_sha256=self.raw_observation_sha256,
            face_observation_id=self.face_observation_id, body_observation_id=self.body_observation_id,
            point_coords_xy=self.point_coords_xy, point_labels=self.point_labels,
            box_xyxy=self.box_xyxy, used_raw_joint_indices=self.used_raw_joint_indices,
            unused_raw_joints=self.unused_raw_joints, policy_sha256=self.policy_sha256,
            raw_score_basis='rtmpose_simcc_raw_peak_not_probability_v1', raw_scores_clipped=False,
            body_alternatives_not_merged=True, prompt_is_identity_certificate=False,
            prompt_is_surface_certificate=False, quality_approved=False)


def native_instance_prompt_hypotheses(*, raw_observation, face_observation_id,
        minimum_joint_confidence=.55):
    """Keep all body candidates for an EXPLICIT inherited/diagnostic face seed.

    A caller must establish the seed and cross-source correspondence separately.
    No largest-face re-selection, best-score mask selection, joint clamping or
    lowered confidence floor occurs here. Return empty for a missing seed/body.
    """
    if (not isinstance(raw_observation, Mapping)
            or raw_observation.get('contract') != 'actual_source_native_owned_person_body_face_observation_rows_v3'
            or type(face_observation_id) is not int or face_observation_id < 0
            or not _number(minimum_joint_confidence) or not .55 <= minimum_joint_confidence <= 1.
            or not _sha(raw_observation.get('native_pixel_sha256'))):
        raise ValueError('instance prompts need a bound v3 raw row, explicit face ID and unchanged joint floor')
    heads, people = raw_observation.get('heads'), raw_observation.get('people')
    if (not isinstance(heads, (list, tuple)) or len(heads) > 24
            or not isinstance(people, (list, tuple)) or len(people) > 12
            or any(not isinstance(row, Mapping) or type(row.get('observation_id')) is not int
                or row['observation_id'] < 0 for row in (*heads, *people))
            or len({row['observation_id'] for row in heads}) != len(heads)
            or len({row['observation_id'] for row in people}) != len(people)):
        raise ValueError('instance prompt raw head/body inventories are not bounded unique observations')
    head = next((h for h in heads if h['observation_id'] == face_observation_id), None)
    if head is None:
        return ()
    def center(row):
        box = row.get('box_xywh')
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(not _number(v) for v in box) or min(box[2:]) <= 0):
            raise ValueError('instance prompt must retain the actual finite native detector box')
        point = (box[0]+box[2]/2, box[1]+box[3]/2)
        return point if all(0 <= v < 1 for v in point) else None
    focal = center(head)
    if focal is None:
        return ()
    negative = tuple(p for row in heads if row['observation_id'] != face_observation_id
        for p in (center(row),) if p is not None)
    owners = head.get('candidate_body_observation_ids')
    if (not isinstance(owners, (list, tuple)) or any(type(i) is not int or i < 0 for i in owners)
            or len(set(owners)) != len(owners)):
        raise ValueError('instance prompt must retain all distinct raw body ownership alternatives')
    by_id = {p['observation_id']: p for p in people}
    if any(i not in by_id for i in owners):
        raise ValueError('instance prompt refers to an absent raw body alternative')
    raw_sha = hashlib.sha256(json.dumps(dict(raw_observation), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    policy_sha = hashlib.sha256(json.dumps(dict(contract='raw_pose_positive_instance_prompts_v2',
        minimum_joint_confidence=minimum_joint_confidence,
        raw_score_basis='rtmpose_simcc_raw_peak_not_probability_v1',
        prompt_joint_inventory=(5, 6, 7, 8, 9, 10, 11, 12),
        no_body_union=True, native_points_not_clamped=True), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    hypotheses = []
    for owner in owners:
        body = by_id[owner]
        center(body)  # Validate, but do not replace raw geometry with its center.
        box = body['box_xywh']
        joints = body.get('raw_pose_xyc')
        if (not isinstance(joints, (list, tuple)) or len(joints) != 17
                or any(not isinstance(j, (list, tuple)) or len(j) != 3
                    or any(not _number(v) for v in j) for j in joints)):
            raise ValueError('instance prompt needs every original raw17 joint, not a simplified skeleton')
        affine = body.get('source_pixel_to_pose_input_affine')
        if (not isinstance(affine, (list, tuple)) or len(affine) != 2
                or any(not isinstance(row, (list, tuple)) or len(row) != 3
                    or any(not _number(v) for v in row) for row in affine)):
            raise ValueError('instance prompt raw17 must retain its actual source-to-pose affine')
        positive, used, unused = [focal], [], []
        for index in (5, 6, 7, 8, 9, 10, 11, 12):
            x, y, confidence = joints[index]
            if not 0 <= x < 1 or not 0 <= y < 1:
                unused.append((index, 'raw_joint_outside_native_frame'))
            elif confidence < minimum_joint_confidence:
                unused.append((index, 'raw_joint_below_unchanged_prompt_floor'))
            else:
                positive.append((x, y)); used.append(index)
        hypotheses.append(NativeInstancePromptHypothesis(raw_sha, face_observation_id, owner,
            (*positive, *negative), (1,)*len(positive)+(0,)*len(negative),
            (box[0], box[1], box[0]+box[2], box[1]+box[3]), tuple(used), tuple(unused), policy_sha))
    return tuple(hypotheses)
