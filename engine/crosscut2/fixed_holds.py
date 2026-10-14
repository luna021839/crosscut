"""Whole-programme actor/camera-state DP on reusable conditional observations.

Camera alternatives are fixed source-local curves BEFORE search, not separately
refitted at the selected boundaries. Local appearance paths are hypotheses, not
named membership or dense actor certificates. Native NULL ownership keeps every
physical source/hold possible when appearance or action is unobserved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from time import monotonic

import numpy as np

from crosscut.render import EditorialTransformKeyframe, EditorialTransformTrack, RenderSegment
from .action_geometry import (action_crop_cost, compare_action, compare_screen_motion,
    normalized_body, synchronous_velocity, complete_arm_motion)
from .paired_geometry import Camera
from .stage_mix import native_event_cost, node_quality
from .hold_utility import DEFAULT_POLICY, known_switch, source_local_salience, soft_hold_utility,native_readability_cost

MISSING_PORT_COST = 6.  # Finite uncertainty, not a measured identity/pose mismatch.


@dataclass(frozen=True)
class State:
    source: int
    actor: int | None
    family: str


def local_actor_hypotheses(bank, *, maximum_clusters=12, minimum_cosine=.55, ambiguity_margin=.03):
    """Reuse anonymous native appearance; order changes do not change owners.

    Recurrent native appearances seed the finite hypothesis budget BEFORE
    labeling. First-frame singletons cannot consume every slot permanently.
    Adjacent exit/entry frames alone are not independent recurrence. Gaps
    remain -1; no gallery, names, nearest-position fallback or forced six-way
    assignment is used. This offline hypothesis is not a dense track certificate.
    """
    vectors = bank['vectors']
    if vectors.shape != (len(bank['heads']), 2, 3, 128) or not np.isfinite(vectors).all():
        raise ValueError('the bound anonymous source appearance bank is required')
    labels = np.full(vectors.shape[:3], -1, np.int16)
    # The wider bounded draft pass is not the DP hypothesis set. Its sole job
    # is to allocate those slots by real recurrence instead of arrival order.
    draft, supports, seen = [], [], set()
    for n in range(len(vectors)):
        for side in (0,1):
            frame = int(bank['native_indices'][n,side])
            if frame in seen: continue
            seen.add(frame)
            used = set()
            for head,value in enumerate(vectors[n,side]):
                if bank['heads'][n,side,head,2] < .035 or np.linalg.norm(value) < .5:
                    continue
                scores = np.array([float(value @ prototype) for prototype in draft])
                order = np.argsort(-scores)
                owner = None
                if len(order) and scores[order[0]] >= minimum_cosine:
                    best = int(order[0])
                    ambiguous = len(order)>1 and scores[best]-scores[order[1]] < ambiguity_margin
                    if best not in used and not ambiguous: owner=best
                elif len(draft) < max(128,maximum_clusters):
                    owner=len(draft)
                    draft.append(value.astype(float).copy())
                    supports.append([])
                if owner is not None:
                    supports[owner].append(frame)
                    used.add(owner)
                    updated=.9*draft[owner]+.1*value
                    draft[owner]=updated/max(1e-9,float(np.linalg.norm(updated)))
    eligible=[i for i,frames in enumerate(supports) if max(frames)-min(frames)>=2]
    chosen=sorted(eligible,key=lambda i:(-len(supports[i]),i))[:maximum_clusters]
    chosen.sort()  # Preserve first-observed local numbering, not density rank.
    prototypes=[draft[i].copy() for i in chosen]
    repeated = {}
    for n in range(len(vectors)):
        for side in (0, 1):
            key = int(bank['native_indices'][n, side])
            if key in repeated:
                labels[n, side] = repeated[key]
                continue
            used = set()
            for head, value in enumerate(vectors[n, side]):
                if bank['heads'][n, side, head, 2] < .035 or np.linalg.norm(value) < .5:
                    continue
                scores = np.array([float(value @ p) for p in prototypes])
                order = np.argsort(-scores)
                owner = None
                if len(order) and scores[order[0]] >= minimum_cosine:
                    best = int(order[0])
                    ambiguous = len(order) > 1 and scores[best] - scores[order[1]] < ambiguity_margin
                    if best not in used and not ambiguous:
                        owner = best
                if owner is not None:
                    labels[n, side, head] = owner
                    used.add(owner)
                    updated = .9 * prototypes[owner] + .1 * value
                    prototypes[owner] = updated / max(1e-9, np.linalg.norm(updated))
            repeated[key] = labels[n, side].copy()
    return labels, len(prototypes)


def clamped_camera(head, cap, support, *, alignment_target=None):
    """Small fixed family: fit a focal head while preserving observed action.

    Unknown hands impose no fabricated clipping and no all-source ban. Native
    framing is a candidate in every family and remains separately available.
    """
    if head[2] < .035:
        return Camera()
    goal = DEFAULT_POLICY.focal_lower if alignment_target is None else alignment_target
    desired = min(cap, max(1., goal / float(head[2])))
    points = np.asarray(support, float).reshape(2, 8, 3)
    visible = (points[:, :, 2] >= .3) & (points[:, :, :2] >= 0).all(2) & (points[:, :, :2] <= 1).all(2)
    xy = points[:, :, :2][visible]
    # Include head top/bottom, not just the face center. Source-edge clipping
    # is not retrospectively made into an invalid source.
    xy = np.vstack((xy, [head[0], max(0., head[1] - head[2] / 2)],
        [head[0], min(1., head[1] + head[2] / 2)]))
    candidates = []
    for zoom in sorted({1., (1. + desired) / 2, desired}):
        low, high = .5 / zoom, 1. - .5 / zoom
        center = np.clip(np.array([head[0], head[1] + .2 / zoom], float), low, high)
        # Shift toward a feasible visible-action envelope when one exists.
        lo, hi = xy.max(0) - .5 / zoom, xy.min(0) + .5 / zoom
        for axis in (0, 1):
            if max(low, lo[axis]) <= min(high, hi[axis]):
                center[axis] = np.clip(center[axis], max(low, lo[axis]), min(high, hi[axis]))
        delta = center - .5
        center = .5 + delta * min(1., .25 / max(1e-12, float(np.linalg.norm(delta))))
        camera = Camera(float(zoom), float(center[0]), float(center[1]))
        # There is no enlargement reward inside the declared comfortable band.
        residual=math.log(goal / (float(head[2]) * zoom))
        cost = 1.4 * (max(0.,residual) if alignment_target is None else abs(residual)) + .65 * math.log(zoom)
        cost += float(action_crop_cost([zoom], [center], points)[0])
        cost += 6. * max(0., float(head[2]) * zoom - max(.24, float(head[2])))
        candidates.append((cost, zoom, camera))
    return min(candidates, key=lambda row: row[:2])[2]


def _curve_at(ticks, values, query):
    return np.array([np.interp(query, ticks, values[:, i]) for i in range(3)]).T


def make_states(nodes, banks, poses, *, fps, source_events=None, body_states=None):
    """Freeze source-local actor paths, camera curves and whole-hold costs."""
    states, data, cluster_counts = [], [], []
    if body_states is not None and (len(body_states)!=len(banks) or any(
            np.asarray(value).shape!=(len(nodes),4,3) or np.any((value<0)|(value>3)) for value in body_states)):
        raise ValueError('source-bound cached body observation states required')
    for source, bank in enumerate(banks):
        native_quality=bank.get('native_readability');native_quality_valid=bank.get('native_readability_valid')
        if native_quality is not None and (np.asarray(native_quality).shape!=bank['heads'].shape
                or np.asarray(native_quality_valid).shape!=bank['heads'].shape[:3]
                or not np.isfinite(native_quality).all()):
            raise ValueError('source-bound native readability domain required')
        labels, count = local_actor_hypotheses(bank)
        cluster_counts.append(count)
        events = np.array([],int) if source_events is None else source_events[source]
        salience = source_local_salience(bank,labels,nodes,events,fps=fps)
        # Alignment cannot require surrendering an observed local actor to
        # NULL/head-0 ownership. Freeze the same alternatives for every owner.
        families = ('native', 'action_fit', 'align_medium', 'align_close')
        for actor, family in [(a, f) for a in (None, *range(count)) for f in families]:
            heads = np.zeros((len(nodes), 2, 4), float)
            vectors = np.zeros((len(nodes), 2, 128), float)
            body = np.zeros((len(nodes), 4, 8, 3), float)
            indices = np.full((len(nodes), 2), -1, int)
            formation = np.zeros((len(nodes), 2), bool)
            formation_heads = np.zeros((len(nodes), 2, 8, 4), float)
            quality = np.full((len(nodes), 2), 1.2)
            cameras = np.tile([1., .5, .5], (len(nodes), 2, 1))
            observed = np.zeros((len(nodes), 2), bool)
            owner = np.full((len(nodes),2),-1,np.int16)
            utility_parts = {key:np.zeros((len(nodes),2)) for key in ('scale','salience','visibility_uncertainty',
                'ownership_uncertainty','crop_effort','crop_clipping','framing_position','native_readability')}
            utility_parts['scale'][:]=1.2
            for n in range(len(nodes)):
                for side in (0, 1):
                    visible = bank['heads'][n, side]
                    if actor is None:
                        group = visible[(visible[:, 2] > .019) & (visible[:, 3] >= .75)]
                        formation[n, side] = len(group) >= 3 and np.max(group[:, 2]) < 1.6 * np.sort(group[:, 2])[-2]
                        if formation[n, side]:
                            group = group[group[:, 2] >= .55 * np.median(group[:, 2])]
                            formation_heads[n, side, :len(group)] = group[np.argsort(group[:, 0])]
                            heads[n, side] = (*group[:, :2].mean(0), np.median(group[:, 2]), .9)
                            parts=soft_hold_utility(float(heads[n,side,2]),formation=True)
                            if native_quality is not None:
                                relevant=(visible[:,2]>=.55*np.median(group[:,2]))&native_quality_valid[n,side]
                                if relevant.any():parts['native_readability']=float(np.mean([
                                    native_readability_cost(value) for value in native_quality[n,side,relevant]]))
                            quality[n,side]=sum(parts.values())
                            for key,value in parts.items():utility_parts[key][n,side]=value
                            observed[n, side] = True
                            continue
                        index = 0 if visible[0, 2] > .019 else -1
                    else:
                        matches = np.flatnonzero(labels[n, side] == actor)
                        index = int(matches[0]) if len(matches) else -1
                    indices[n, side] = index
                    if index < 0:
                        if actor is not None and np.max(visible[:,2]) >= .055:
                            # This hypothesis is absent while another readable
                            # foreground head is observed. Do not infer a new
                            # owner, or make the absent endpoint cheaper to cut.
                            quality[n,side] += .65
                            utility_parts['ownership_uncertainty'][n,side]=.65
                        continue  # a true NULL, not another head silently chosen
                    observed[n, side] = True
                    head = visible[index]
                    owner[n,side] = labels[n,side,index]
                    heads[n, side] = head
                    vectors[n, side] = bank['vectors'][n, side, index]
                    support = np.zeros((2, 8, 3)) if poses is None else poses[source][n, side * 2:side * 2 + 2, index]
                    body[n, side * 2:side * 2 + 2] = support
                    target={'align_medium':.18,'align_close':.24}.get(family)
                    camera = Camera() if family == 'native' else clamped_camera(head, bank['zoom_cap'], support,
                        alignment_target=target)
                    cameras[n, side] = (camera.zoom, camera.x, camera.y)
                    cached = None if body_states is None else body_states[source][n,side*2:side*2+2,index]
                    parts=soft_hold_utility(float(head[2]),camera.zoom,
                        salience=salience[n,side,index],body_states=cached,projected_center=camera.project(head[:2]),
                        native_readability=None if native_quality is None or not native_quality_valid[n,side,index]
                            else native_quality[n,side,index])
                    for key,value in parts.items():utility_parts[key][n,side]=value
                    quality[n,side]=sum(parts.values())
                    if actor is None:
                        # Finite ownership uncertainty, also present in the
                        # action-disabled control. NULL must stay possible but
                        # cannot freely follow a different foreground actor.
                        quality[n, side] += .18
                        utility_parts['ownership_uncertainty'][n,side]=.18
                    effort=.65*math.log(camera.zoom)
                    clipping=float(action_crop_cost([camera.zoom], [[camera.x, camera.y]], support)[0])
                    quality[n,side] += effort + clipping
                    utility_parts['crop_effort'][n,side]=effort
                    utility_parts['crop_clipping'][n,side]=clipping
            # The complete curve is frozen here. Duration/neighbor choices do
            # not refit it; the exact same curve will be sent to the renderer.
            knots = {}
            for n, tick in enumerate(nodes):
                for side, offset in ((0, -1), (1, 0)):
                    # Missing observations are not instructions to snap to
                    # native framing. Do not overwrite the real endpoint with
                    # the clamped duplicate entry/exit of a nonexistent side.
                    if not observed[n, side] or (n, side) in ((0, 0), (len(nodes)-1, 1)):
                        continue
                    t = max(0, min(int(nodes[-1]) - 1, int(tick) + offset))
                    knots[t] = cameras[n, side]
            if not knots:
                knots[0] = np.array([1., .5, .5])
            ticks = np.array(sorted(knots), int)
            values = np.array([knots[t] for t in ticks])
            for n,tick in enumerate(nodes):
                cameras[n] = _curve_at(ticks, values, np.clip(int(tick)+np.array([-1,0]), 0, nodes[-1]-1))
            # A continuing hold exposes both sides of an interior node. The
            # inter-node unary below omits this one-frame knot pair entirely.
            through = np.zeros(len(nodes))
            jump = cameras[1:-1, 1] - cameras[1:-1, 0]
            through[1:-1] = 6. * (np.linalg.norm(jump[:, 1:] * [16/9, 1.], axis=1)
                + np.abs(np.log(cameras[1:-1, 1, 0] / cameras[1:-1, 0, 0])))
            camera_through=through.copy()
            owner_through=DEFAULT_POLICY.owner_switch*known_switch(owner[:,0],owner[:,1])
            owner_through[[0,-1]]=0.
            through += owner_through
            # Same-source camera acceleration is not free inside a hold.
            lengths = np.diff(nodes) / fps
            a, b = cameras[:-1, 1], cameras[1:, 0]
            speed = np.linalg.norm((a[:, 1:] - b[:, 1:]) * [16 / 9, 1.], axis=1) / lengths
            zoom_speed = np.abs(np.log(a[:, 0] / b[:, 0])) / lengths
            unary = (quality[:-1, 1] + quality[1:, 0]) / 2
            readability_tail=.35*np.maximum(0.,np.maximum(quality[:-1,1],quality[1:,0]))
            unary += readability_tail
            interval_owner_switch=DEFAULT_POLICY.owner_switch*known_switch(owner[:-1,1],owner[1:,0])
            unary += interval_owner_switch/lengths
            unary += .8 * np.maximum(0., speed - .15) + .3 * np.maximum(0., zoom_speed - .25)
            projected = body.copy()
            for n, tick in enumerate(nodes):
                selected = _curve_at(ticks, values, np.clip(int(tick) + np.array([-7, -1, 0, 6]), 0, nodes[-1] - 1))
                projected[n, :, :, :2] = .5 + selected[:, None, :1] * (body[n, :, :, :2] - selected[:, None, 1:])
            motion = np.zeros((len(nodes), 2, 8, 2))
            motion_valid = np.zeros((len(nodes), 2, 8), bool)
            arm_motion=np.zeros_like(motion);arm_valid=np.zeros_like(motion_valid)
            for n in range(len(nodes)):
                for window in (0, 1):
                    if (body[n,2*window,:,2]>=.3).any() and (body[n,2*window+1,:,2]>=.3).any():
                        arm_motion[n,window],arm_valid[n,window]=complete_arm_motion(
                            body[n,2*window],body[n,2*window+1],seconds=6/fps)
                    early, ev = normalized_body(body[n, 2*window])
                    late, lv = normalized_body(body[n, 2*window+1])
                    if early is not None and late is not None:
                        motion[n, window] = (late-early) * fps/6.
                        motion_valid[n, window] = ev & lv
                        motion_valid[n, window, :2] = False
                        motion_valid[n, window, 6:] = False
            ha, hb = heads[:, 0], heads[:, 1]
            continuous = ((np.minimum(ha[:, 2], hb[:, 2]) > .019)
                & (np.abs(ha[:, :2]-hb[:, :2]).max(1) < .03)
                & (np.abs(np.log(np.maximum(ha[:, 2], .001)/np.maximum(hb[:, 2], .001))) < .2))
            # A known appearance disagreement cannot certify the unrealized
            # opposite-side window as this actor, even at identical positions.
            va, vb = vectors[:, 0], vectors[:, 1]
            known = (np.linalg.norm(va, axis=1) >= .5) & (np.linalg.norm(vb, axis=1) >= .5)
            continuous &= ~known | ((va*vb).sum(1) >= .55)
            if source_events is not None:
                cuts = np.asarray(source_events[source], float)
                for n, tick in enumerate(nodes):
                    if np.any((cuts > tick-7) & (cuts <= tick-1)):
                        motion_valid[n, 0] = False
                        arm_valid[n,0]=False
                    if np.any((cuts > tick) & (cuts <= tick+6)):
                        motion_valid[n, 1] = False
                        arm_valid[n,1]=False
                    if np.any((cuts > tick-1) & (cuts <= tick)):
                        continuous[n] = False
            states.append(State(source, actor, family))
            data.append(dict(heads=heads, vectors=vectors, body=body, projected=projected,
                indices=indices, formation=formation, formation_heads=formation_heads,
                cameras=cameras, ticks=ticks, curve=values, unary=unary, through=through,
                motion=motion, motion_valid=motion_valid, continuous=continuous,
                arm_motion=arm_motion,arm_valid=arm_valid,
                owner=owner,owner_through=owner_through,camera_through=camera_through,
                interval_owner_switch=interval_owner_switch,utility_parts=utility_parts,
                quality=quality,readability_tail=readability_tail))
    return tuple(states), data, cluster_counts


def _action_refinement(action, synchronous, screen, *, moving_threshold=2):
    velocity, moving = synchronous
    agreement = 0.
    if action.phase_loss is not None:
        agreement = 3.*action.phase_loss - 2. + (1. if velocity is None else 1.5*min(velocity, .4))
    # Keep screen position/prediction, not the confounded pre->post direction.
    spatial = screen['cost'] - (.75*screen['velocity_loss'] if screen['velocity_loss'] is not None else 0.)
    agreement += .6*(spatial-2.)
    active = velocity is not None and moving >= moving_threshold
    if active and velocity > .4:
        # Contrary SAME-CLOCK action is never canceled by a pose/position credit.
        return max(0., agreement) + 2.5*(velocity-.4)
    return max(-.8 if active else -.6, agreement)


def port_cost(n, left, right, *, fps, action_enabled=True):
    a, b = left['heads'][n, 0], right['heads'][n, 1]
    result = dict(geometry_loss=4., appearance_cosine=None, conditional_pose_loss=None,
        conditional_velocity_loss=None, comparable_action_joints=0, projected_screen_motion=None,
        role='unobserved', soft_cost=MISSING_PORT_COST)
    if min(a[2], b[2]) <= .019:
        # A one-sided missing port is not a measured connection of a readable
        # foreground subject. Keep it finite, and never penalize all-UNKNOWN
        # sources as a fabricated mismatch or forbid their native paths.
        readable=(a[2]>=.055 and not left['formation'][n,0]) or (b[2]>=.055 and not right['formation'][n,1])
        if readable:result['soft_cost']+=2.
        return result
    ca, cb = Camera(*left['cameras'][n, 0]), Camera(*right['cameras'][n, 1])
    displacement = float(np.linalg.norm((ca.project(a[:2]) - cb.project(b[:2])) * [1., 9/16]))
    geometry = displacement / .06 + abs(math.log(float(a[2] * ca.zoom / (b[2] * cb.zoom)))) / .25
    group_a, group_b = bool(left['formation'][n, 0]), bool(right['formation'][n, 1])
    role = 'formation' if group_a and group_b else 'focal'
    similarity = None
    if not group_a and not group_b:
        va, vb = left['vectors'][n, 0], right['vectors'][n, 1]
        if min(np.linalg.norm(va), np.linalg.norm(vb)) >= .5:
            similarity = float(np.clip(va @ vb, -1., 1.))
    identity = (.35 if group_a and group_b else 3. if group_a != group_b else
        1.4 if similarity is None else 5. * max(0., .65 - similarity))
    layout = None
    if group_a and group_b:
        ga, gb = left['formation_heads'][n, 0], right['formation_heads'][n, 1]
        ga, gb = ga[ga[:, 2] > .019], gb[gb[:, 2] > .019]
        pa, pb = ca.project(ga[:, :2]), cb.project(gb[:, :2])
        if len(pa) == len(pb):
            layout = float(np.linalg.norm((pa - pb) * [1., 9 / 16], axis=1).mean()) / .08
        else:
            distances = np.linalg.norm((pa[:, None] - pb[None]) * [1., 9 / 16], axis=2)
            layout = float((distances.min(0).mean() + distances.min(1).mean()) / 2.) / .08
        layout += .45 * abs(len(pa) - len(pb))
        geometry += layout
    action = compare_action(left['body'][n, :2], right['body'][n, 2:], seconds=6/fps)
    screen = compare_screen_motion(left['projected'][n, :2], right['projected'][n, 2:],
        seconds=6/fps, frame_seconds=1/fps)
    lv, rv = left['motion_valid'][n].copy(), right['motion_valid'][n].copy()
    lv[1] &= left['continuous'][n]
    rv[0] &= right['continuous'][n]
    la,ra=left['arm_valid'][n].copy(),right['arm_valid'][n].copy()
    la[1] &= left['continuous'][n];ra[0] &= right['continuous'][n]
    # Before/after windows are independent comparisons. Both sources use the
    # SAME basis within each window; no joint vector is compared to a bone.
    full=(lv&rv).sum(1)>=2
    lm=np.where(full[:,None,None],left['motion'][n],left['arm_motion'][n])
    rm=np.where(full[:,None,None],right['motion'][n],right['arm_motion'][n])
    lvalid=np.where(full[:,None],lv,la);rvalid=np.where(full[:,None],rv,ra)
    synchronous = synchronous_velocity(lm,lvalid,rm,rvalid)
    arm_active=any(not full[w] and (lvalid[w]&rvalid[w]).sum()>=2 and np.any(
        (lvalid[w]&rvalid[w]) & (np.maximum(np.linalg.norm(lm[w],axis=1),np.linalg.norm(rm[w],axis=1))>=.3))
        for w in (0,1))
    refinement = _action_refinement(action, synchronous, screen,moving_threshold=1 if arm_active else 2) if action_enabled else 0.
    return dict(result, geometry_loss=float(geometry), appearance_cosine=similarity, role=role,
        conditional_pose_loss=action.phase_loss, conditional_velocity_loss=action.velocity_loss,
        conditional_action_basis=action.measurement_basis, formation_layout_loss=layout,
        comparable_action_joints=action.common_joints, projected_screen_motion=screen,
        synchronous_velocity_loss=synchronous[0], action_refinement=float(refinement),
        synchronous_basis_per_window=[('shoulder_relative_articulation' if full[w] else 'directed_complete_arm_segments')
            if (lvalid[w]&rvalid[w]).sum()>=2 else None for w in (0,1)],
        soft_cost=float(geometry + identity + refinement))


def coupled_semi_markov(nodes, *, fps, unary, seam_cost, state_sources, source_events=None,
        minimum_hold=.9, maximum_hold=4.8, target_hold=2.4, return_penalty=.7, through=None):
    """Exact fixed-curve DP retains actor/camera state AND previous source.

    ``seam_cost`` may be a dense tensor or a node->matrix callable. A callable
    is consumed once per reached node; short/long incoming summaries are kept
    instead of retaining every quadratic port matrix for the whole song.
    Equivalent-state dominance is safe because every state's entry/exit camera
    and one-sided motion port at a node were fixed before search, independent
    of that hold's duration. Do not replace state by source-only argmin.
    """
    nodes, state_sources = np.asarray(nodes, int), np.asarray(state_sources, int)
    q, sources = len(state_sources), int(state_sources.max()) + 1
    lazy_ports = callable(seam_cost)
    if (nodes[0] != 0 or np.any(np.diff(nodes) <= 0) or not 0 < minimum_hold <= maximum_hold
            or not math.isfinite(fps) or fps <= 0 or unary.shape != (q, len(nodes)-1)
            or not np.isfinite(unary).all() or np.any(state_sources < 0)
            or (not lazy_ports and (seam_cost.shape != (len(nodes), q, q)
                or not np.isfinite(seam_cost).all()))):
        raise ValueError('finite fixed-state complete programme graph required')
    through = np.zeros((q, len(nodes))) if through is None else np.asarray(through, float)
    if through.shape != (q, len(nodes)) or not np.isfinite(through).all() or np.any(through < 0):
        raise ValueError('finite non-negative interior camera costs required')
    crossing = np.pad(np.cumsum(through, axis=1), ((0,0),(1,0)))
    lengths = np.diff(nodes)/fps
    prefix = np.pad(np.cumsum(unary * lengths, axis=1), ((0,0),(1,0)))
    if sources == 1:
        total = prefix[:,-1] + crossing[:,len(nodes)-1] - crossing[:,1]
        best = int(np.argmin(total))
        return ((0,len(nodes)-1,best),), float(total[best])
    cost = np.full((len(nodes), q, sources+1), np.inf)
    previous, incoming_cache = {}, {}
    by_source = [np.flatnonzero(state_sources == s) for s in range(sources)]
    def incoming(start, short):
        key = (start, short)
        if key in incoming_cache: return incoming_cache[key]
        matrix = np.asarray(seam_cost(start) if lazy_ports else seam_cost[start])
        if matrix.shape != (q,q) or not np.isfinite(matrix).all():
            raise ValueError('finite fixed-state node port matrix required')
        # cost[start] is final before any later end reaches it. Both duration
        # classes can therefore be summarized now without rebuilding the ports.
        for short_class in (False, True):
            values, old_state, old_prior = np.full((q,sources), np.inf), np.full((q,sources),-1,int), np.full((q,sources),-1,int)
            for new_source, targets in enumerate(by_source):
                shifted = cost[start].copy()
                if short_class: shifted[:, new_source] += return_penalty
                priors = np.argmin(shifted, axis=1)
                scores = shifted[np.arange(q), priors]
                for old_source, old in enumerate(by_source):
                    if old_source == new_source or not len(old) or not len(targets): continue
                    joined = scores[old, None] + matrix[np.ix_(old, targets)]
                    arg = np.argmin(joined, axis=0)
                    picked = old[arg]
                    values[targets, old_source] = joined[arg, np.arange(len(targets))]
                    old_state[targets, old_source] = picked
                    old_prior[targets, old_source] = priors[picked]
            incoming_cache[start,short_class] = values, old_state, old_prior
        return incoming_cache[key]
    for end in range(1, len(nodes)):
        for start in range(end-1, -1, -1):
            duration = (nodes[end]-nodes[start])/fps
            if duration > maximum_hold: break
            if duration < minimum_hold and end != len(nodes)-1: continue
            hold = (prefix[:,end] - prefix[:,start] + crossing[:,end] - crossing[:,start+1]
                + .23*(duration-target_hold)**2 + 2.*max(0.,minimum_hold-duration))
            if source_events is not None:
                event = np.array([native_event_cost(nodes[start],nodes[end],events,fps) for events in source_events])
                hold += event[state_sources]
            if start == 0:
                cost[end,:,sources] = hold
                for state in range(q): previous[end,state,sources] = (0,-1,-1)
                continue
            base, old, prior = incoming(start, duration < 3.)
            candidate = base + hold[:,None] + .35
            better = candidate < cost[end,:,:sources]
            cost[end,:,:sources][better] = candidate[better]
            for state, source in zip(*np.nonzero(better)):
                previous[end,int(state),int(source)] = (start,int(old[state,source]),int(prior[state,source]))
    state, prior = np.unravel_index(np.argmin(cost[-1]), cost[-1].shape)
    score = float(cost[-1,state,prior])
    if not math.isfinite(score): raise ValueError('no complete fixed-state programme under declared hold grammar')
    runs, end = [], len(nodes)-1
    while end:
        start, old, before = previous[end,int(state),int(prior)]
        runs.append((start,end,int(state)))
        end, state, prior = start, old, before
    return tuple(reversed(runs)), score


def select_fixed_holds(nodes, banks, poses, *, fps, source_events, broadcast_slots, check,
                       action_enabled=True, body_states=None):
    started = monotonic()
    states, data, cluster_counts = make_states(nodes, banks, poses, fps=fps,
        source_events=source_events,body_states=body_states)
    count = len(states)
    matrix_bytes = count * count * np.dtype(np.float32).itemsize
    if matrix_bytes > 200 * 1024**2:
        raise ValueError('fixed-state node port matrix exceeded its explicit 200MiB bound')
    same_show = np.zeros((count,count),np.float32)
    if broadcast_slots is not None:
        for a,sa in enumerate(states):
            for b,sb in enumerate(states):
                if sa.source != sb.source and broadcast_slots[sa.source] is not None and (
                        broadcast_slots[sa.source]==broadcast_slots[sb.source]):
                    same_show[a,b]=.9
    built_nodes = 0
    def costs(n):
        nonlocal built_nodes
        check()
        if monotonic()-started > 240.:
            raise TimeoutError('single fixed actor/camera selection exceeded 240 seconds')
        matrix = np.full((count,count), MISSING_PORT_COST, np.float32)
        # Unobserved endpoints share the finite missing port, not an NN veto.
        left = [i for i,d in enumerate(data) if d['heads'][n,0,2] > .019]
        right = [i for i,d in enumerate(data) if d['heads'][n,1,2] > .019]
        left_readable=np.array([d['heads'][n,0,2]>=.055 and not d['formation'][n,0] for d in data])
        right_readable=np.array([d['heads'][n,1,2]>=.055 and not d['formation'][n,1] for d in data])
        # The observed-only nested loop must not bypass missing-port costs.
        # Known/known cells below replace this default; all UNKNOWN stay finite.
        matrix += 2.*(left_readable[:,None] | right_readable[None,:])
        for a in left:
            for b in right:
                if states[a].source == states[b].source: continue
                value = port_cost(n,data[a],data[b],fps=fps,action_enabled=action_enabled)['soft_cost']
                matrix[a,b] = value
        matrix += same_show
        built_nodes += 1
        if built_nodes % 40 == 0:
            print(__import__('json').dumps(dict(phase='fixed_actor_camera_ports',nodes=built_nodes,total_nodes=len(nodes),
                fixed_states=count,elapsed_seconds=monotonic()-started)),flush=True)
        return matrix

    def selected_port(n, left, right):
        # Recompute only selected scalar ports for the cost report, with the
        # same float32 rounding as the transient matrix. Do not rebuild it.
        value = port_cost(n,data[left],data[right],fps=fps,action_enabled=action_enabled)['soft_cost']
        return float(np.float32(value) + same_show[left,right])
    runs, score = coupled_semi_markov(nodes, fps=fps, unary=np.array([d['unary'] for d in data]),
        seam_cost=costs, state_sources=np.array([s.source for s in states]), source_events=source_events,
        through=np.array([d['through'] for d in data]))
    decomposition=[];cumulative=0.
    for i,(start,end,state_index) in enumerate(runs):
        row=data[state_index];state=states[state_index]
        seconds=np.diff(nodes)[start:end]/fps
        duration=float(seconds.sum())
        parts={key:float(((value[start:end,1]+value[start+1:end+1,0])/2) @ seconds)
            for key,value in row['utility_parts'].items()}
        unary_total=float(row['unary'][start:end] @ seconds)
        parts['readability_tail']=float(row['readability_tail'][start:end] @ seconds)
        parts['interval_owner_switch']=float(row['interval_owner_switch'][start:end].sum())
        parts['interval_camera_motion']=unary_total-sum(parts.values())
        parts['camera_through']=float(row['camera_through'][start+1:end].sum())
        parts['owner_through']=float(row['owner_through'][start+1:end].sum())
        parts['native_fragments']=0. if len(banks)==1 else native_event_cost(nodes[start],nodes[end],source_events[state.source],fps)
        parts['duration_grammar']=0. if len(banks)==1 else .23*(duration-2.4)**2+2.*max(0.,.9-duration)
        parts['entry_port']=0. if not i else selected_port(start,runs[i-1][2],state_index)
        parts['editorial_switch']=0. if not i else .35
        parts['short_source_return']=.7 if i>=2 and duration<3. and states[runs[i-2][2]].source==state.source else 0.
        prior=cumulative;cumulative+=sum(parts.values())
        next_exit=None if i==len(runs)-1 else selected_port(end,state_index,runs[i+1][2])
        decomposition.append(dict(first_frame=int(nodes[start]),final_frame=int(nodes[end]),source=state.source,
            actor_hypothesis=state.actor,camera_family=state.family,prior_cumulative_cost=prior,
            cumulative_cost=cumulative,next_selected_exit_cost=next_exit,components=parts,
            ownership_is_not_dense_actor_certification=True))
    if not math.isclose(cumulative,score,rel_tol=1e-7,abs_tol=1e-6):
        raise AssertionError('selected programme cost decomposition differs from exact DP objective')
    return runs, states, data, score, dict(fixed_states=count,source_local_appearance_clusters=cluster_counts,
        source_local_actor_hypotheses_are_not_certified=True,camera_alternatives_fixed_before_dp=True,
        actor_budget_seeded_by_recurrent_native_observations=True,
        actor_camera_endpoint_state_retained=True,whole_hold_observed_action_support_consumed=True,
        dense_actor_ownership_validated=False,action_enabled=action_enabled,
        cached_body_states_consumed=body_states is not None,
        native_readability_consumed=all('native_readability' in bank for bank in banks),
        native_readability_is_not_occlusion_or_quality_certificate=True,
        body_states_are_not_occlusion_certificates=True,
        missing_port_uncertainty_cost=MISSING_PORT_COST,
        missing_ports_and_broadcast_costs_consumed_before_dp=True,
        known_owner_changes_are_finite_events=True,
        selected_hold_cost_decomposition=decomposition,
        selection_seconds=monotonic()-started,cost_tensor_bytes=matrix_bytes if built_nodes else 0,
        port_matrix_nodes_built=built_nodes,port_matrices_built_once_per_node=True,
        whole_song_quadratic_tensor_allocated=False,
        eager_port_tensor_bytes_avoided=len(nodes)*matrix_bytes,
        actor_owned_alignment_families_available=True)


def realize_fixed_holds(original, warps, nodes, runs, states, data, *, output, action_enabled=True, boundary_frames=None):
    segments, seams = [], []
    for i,(start,end,state_index) in enumerate(runs):
        state, row = states[state_index], data[state_index]
        first, final = (int(nodes[start]),int(nodes[end])) if boundary_frames is None else (
            int(boundary_frames[i]),int(boundary_frames[i+1]))
        duration = (final-first)/original.fps
        last = final-1
        knots = sorted({first,last,*[int(t) for t in row['ticks'] if first<t<last]})
        values = _curve_at(row['ticks'],row['curve'],knots)
        transform = None if np.all(values == [1.,.5,.5]) else EditorialTransformTrack(tuple(
            EditorialTransformKeyframe((tick-first)/original.fps,float(z),float(x),float(y))
            for tick,(z,x,y) in zip(knots,values)), reason='fixed_source_actor_action_camera_candidate')
        source_first = warps[state.source].master_to_source(original.master_audio_start+first/original.fps)
        source_final = warps[state.source].master_to_source(original.master_audio_start+final/original.fps)
        segments.append(RenderSegment(state.source,float(source_first),float(duration),float(first/original.fps),
            editorial_transform=transform,source_duration=float(source_final-source_first)))
        if i:
            left_index = runs[i-1][2]
            left_state = states[left_index]
            measurement = port_cost(start,data[left_index],row,fps=original.fps,action_enabled=action_enabled)
            seams.append(dict(measurement,output_frame=first,seconds=first/original.fps,
                sparse_observation_frame=int(nodes[start]),
                pose_port_retimed_unvalidated=first!=int(nodes[start]),
                left_source=left_state.source,right_source=state.source,
                left_actor_hypothesis=left_state.actor,right_actor_hypothesis=state.actor,
                left_camera_family=left_state.family,right_camera_family=state.family,
                left_head_index=int(data[left_index]['indices'][start,0]),right_head_index=int(row['indices'][start,1]),
                anonymous_appearance_is_not_member_probability=True,framing_loss=None))
    return replace(original,segments=tuple(segments),output=output,transitions=(),authorized_shot_video_clocks=(),
        member_pose_dissolve_requested=None,member_pose_dissolve_effective=None,member_pose_dissolve_count=None), seams
