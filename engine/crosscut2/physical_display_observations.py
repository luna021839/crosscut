"""Bind GROUP costs to the displayed face, never to an actor-state alias.

Cached top-down pose is only available for the first three detected faces.
A fourth focal face does NOT borrow the first person's body. Pixel tracking
and anatomical ownership are separate; duplicated wrists remain tentative.
"""
from dataclasses import replace
import numpy as np
from .event_states import displayed_focal_faces, crop_added_loss
from .fixed_holds import _curve_at

UNMEASURED, MODEL_ONLY, PIXEL_SUPPORTED, CONTRADICTED = range(4)
SCENE_NAMES = ('NO_FACE_DETECTION', 'FOCAL', 'FORMATION', 'AMBIGUOUS', 'KNOWN_LOWER_BODY_ONLY')


def focal_metadata(heads, cameras):
    chosen, indices = displayed_focal_faces(heads, cameras)
    center = .5 + cameras[:, :, None, :1] * (heads[..., :2] - cameras[:, :, None, 1:])
    size = heads[..., 2] * cameras[:, :, None, 0]
    half = size[..., None] * [.45 * 9 / 16, .5]
    extent = np.maximum(0., np.minimum(1., center + half) - np.maximum(0., center - half))
    scores = np.prod(extent, -1) * np.exp(-((center[..., 0] - .5) / .4) ** 2)
    scores = np.where((heads[..., 2] > .019) & (heads[..., 3] >= .75), scores, 0.)
    ordered = np.sort(scores, axis=-1)
    ambiguity = ordered[..., -2] / np.maximum(1e-12, ordered[..., -1]) if scores.shape[-1] > 1 else np.zeros(indices.shape)
    count = np.sum(scores > 0., axis=-1)
    sorted_sizes = np.sort(np.where(scores > 0., size, 0.), axis=-1)
    formation = (count >= 3) & (sorted_sizes[..., -1] < 1.6 * sorted_sizes[..., -2])
    scene = np.where(indices < 0, 0, np.where(formation, 2, np.where(ambiguity >= .8, 3, 1))).astype(np.uint8)
    # No zero-head value can produce KNOWN_LOWER_BODY_ONLY.
    return chosen, indices, ambiguity, scene


def duplicate_wrists(bodies, *, tolerance=.015):
    """Same semantic wrist attributed to multiple face ROIs is ambiguous."""
    bodies = np.asarray(bodies)
    if bodies.ndim != 5 or bodies.shape[1:] != (4, 3, 8, 3):
        raise ValueError('source pose bank N x4 x3 x8 x3 required')
    result = np.zeros(bodies.shape[:-1], bool)
    for joint in (4, 5):
        xy = bodies[..., joint, :2] * [16 / 9, 1.]
        valid = (bodies[..., joint, 2] >= .3) & (bodies[..., joint, :2] >= 0).all(-1) & (bodies[..., joint, :2] <= 1).all(-1)
        distance = np.linalg.norm(xy[..., :, None, :] - xy[..., None, :, :], axis=-1)
        collision = (distance <= tolerance) & valid[..., :, None] & valid[..., None, :]
        collision &= ~np.eye(3, dtype=bool)
        result[..., joint] = collision.any(-1)
    return result


def bind(nodes, states, rows, banks, styles, *, check=lambda: None):
    """Collapse aliases only AFTER all costs use a physical source/crop.

    All neural/part observations already exist. This adds no inference, no
    pixel decode, no identity claim and no global reanalysis.
    """
    nodes = np.asarray(nodes, int); N = len(nodes)
    groups = {}; source_rows = {}
    for state, row in zip(states, rows):
        groups.setdefault((state.source, state.family), (state, row))
        if state.family == 'native':source_rows.setdefault(state.source, []).append(row)
    result_states = []; result = []
    for (source, _), (state, template) in groups.items():
        check(); bank = banks[source]; row = dict(template); cameras = row['cameras']
        head, index, ambiguity, scene = focal_metadata(bank['heads'], cameras)
        body = np.zeros((N, 4, 8, 3), np.float32)
        vectors = np.zeros((N, 2, 128), np.float32)
        phase = np.full((N, 4, 8), .5, np.float32)
        motion = np.full((N, 2, 8), .5, np.float32)
        scope = np.zeros(N, bool); support = np.zeros((N, 3, 8), bool)
        disagreement = np.zeros((N, 3, 8), bool)
        salience = np.ones((N, 2), np.float32)
        # The same physical face's raw body must be identical for all aliases.
        for n, side in zip(*np.nonzero((index >= 0) & (index < 3))):
            h = int(index[n, side]); vectors[n, side] = bank['vectors'][n, side, h]
            candidates = [d for d in source_rows[source] if int(d['indices'][n, side]) == h]
            if not candidates:continue
            d = candidates[0]; sl = slice(side * 2, side * 2 + 2)
            raw = d.get('raw_body_hypotheses', d['body'])
            if any(not np.array_equal(raw[n, sl], v.get('raw_body_hypotheses', v['body'])[n, sl]) for v in candidates[1:]):
                raise ValueError('actor aliases disagree on the same physical face pose')
            body[n, sl] = raw[n, sl]
            phase[n, sl] = d.get('body_reliability', np.full((N, 4, 8), .5))[n, sl]
            motion[n, side] = d.get('motion_reliability', np.full((N, 2, 8), .5))[n, side]
            if 'utility_parts' in d:salience[n, side] = np.exp(-d['utility_parts']['salience'][n, side] / 1.1)
            scope[n] |= bool(d.get('pixel_part_scope', np.zeros(N, bool))[n])
            if 'part_pair_support' in d:
                # Cross-cut middle support is deliberately not shared across
                # two different displayed subjects.
                support[n, side * 2] = d['part_pair_support'][n, side * 2]
                if 'part_pair_model_disagreement' in d:disagreement[n, side * 2] = d['part_pair_model_disagreement'][n, side * 2]
        # Compare all physically available top-three hypotheses, including
        # ones that were not the displayed focal in this camera family.
        all_bodies = np.zeros((N, 4, 3, 8, 3), np.float32)
        for d in source_rows[source]:
            raw = d.get('raw_body_hypotheses', d['body'])
            for n, side in zip(*np.nonzero((d['indices'] >= 0) & (d['indices'] < 3))):
                all_bodies[n, side * 2:side * 2 + 2, int(d['indices'][n, side])] = raw[n, side * 2:side * 2 + 2]
        duplicates = duplicate_wrists(all_bodies)
        ownership = np.zeros((N, 4, 8), bool)
        for n, side in zip(*np.nonzero((index >= 0) & (index < 3))):
            ownership[n, side * 2:side * 2 + 2] = duplicates[n, side * 2:side * 2 + 2, index[n, side]]
        valid = (body[..., 2] >= .3) & (body[..., :2] >= 0).all(-1) & (body[..., :2] <= 1).all(-1)
        evidence = np.where(valid, MODEL_ONLY, UNMEASURED).astype(np.uint8)
        evidence = np.where(valid & scope[:, None, None] & (phase == 1.), PIXEL_SUPPORTED, evidence)
        evidence = np.where(valid & scope[:, None, None] & (phase == 0.), CONTRADICTED, evidence).astype(np.uint8)
        quality = np.full((N, 2), .5, np.float32)
        qvalid = bank.get('native_readability_valid')
        if qvalid is not None:
            for n, side in zip(*np.nonzero(index >= 0)):
                h = int(index[n, side])
                if qvalid[n, side, h]:
                    contrast, edge, clipped, height = bank['native_readability'][n, side, h]
                    quality[n, side] = min(1., height / 120.) * min(1., edge / .5) * min(1., contrast / .075) * (1. - clipped)
        center = .5 + cameras[..., :1] * (head[..., :2] - cameras[..., 1:])
        half = head[..., 2, None] * cameras[..., :1] * [.45 * 9 / 16, .5]
        extent = np.maximum(0., np.minimum(1., center + half) - np.maximum(0., center - half))
        fraction = np.prod(extent / np.maximum(1e-8, 2 * half), axis=-1)
        display = np.minimum(1., head[..., 2] * cameras[..., 0] / .22)
        black = np.clip((styles[source]['presence'][..., 0] - .8) / .2, 0, 1)
        availability = np.clip(np.sqrt(quality) * display * np.sqrt(salience) * fraction * (1. - black), 0, 1)
        four = _curve_at(row['ticks'], row['curve'], np.clip(nodes[:, None] + [-7, -1, 0, 6], 0, nodes[-1] - 1))
        projected = body.copy(); projected[..., :2] = .5 + four[:, :, None, :1] * (body[..., :2] - four[:, :, None, 1:])
        crop = crop_added_loss(body, projected).reshape(N, 2, 2).max(2)
        loss = 1. - availability + crop + .1 * np.log(cameras[..., 0]) / np.log(2.)
        row.update(heads=head, displayed_focal_heads=head, displayed_focal_indices=index, indices=index,
            focal_ambiguity=ambiguity, scene_kind=scene, scene_kind_names=SCENE_NAMES,
            scene_evidence='cached_face_geometry_only; no lower-body semantics inferred',
            vectors=vectors, body=body, raw_body_hypotheses=body, projected=projected,
            body_reliability=phase, motion_reliability=motion, primitive_evidence_kind=evidence,
            primitive_ownership_ambiguity=ownership, pixel_part_scope=scope,
            part_pair_support=support, part_pair_model_disagreement=disagreement,
            availability=availability, native_quality=quality, added_crop=crop,
            displayed_attention_dominant=index >= 0, displayed_attention_ratio=np.where(index >= 0, 1., 0.),
            formation=scene == 2, local_member_hint=np.full((N, 2), -1, np.int16),
            event_loss_rate=(loss[:-1, 1] + loss[1:, 0]) / 2.,
            physical_group_focal_bound=True, actor_alias_changes_cost=False,
            anatomical_ownership_certified=False)
        result_states.append(replace(state, actor=None)); result.append(row)
    return tuple(result_states), result, dict(physical_states=len(result_states), actor_aliases_removed=len(states) - len(result_states),
        group_content_and_pose_share_displayed_focal=True, pose_face_bank_capacity=3,
        scene_lower_body_observations_used=False, new_neural_calls=0, source_decode=0)
