"""Pre-camera native scene hypotheses and measured screen continuity.

A hypothesis is a detector-variant/cohort/endpoint-association alternative, not
an actor label. Fixed neighbors can have several hypotheses with IDENTICAL
render geometry. Every detected head/body remains in the raw archive; variants
are never concatenated into people. Missing observations stay None. This
module produces diagnostic native screen evidence, not semantic/quality QA.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from collections import Counter
from itertools import product
import math
import json
from statistics import median
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import group_source_changing_owner as source
from . import group_performer_family_owner as family
from .group_joint_source_camera import GroupOutputRole,JointGroupBudgetExhausted
from .group_joint_formation import (ObservedFormationPerson, DeclaredFormationView,
    ObservedFormationTemporalMotion, compare_projected_formation, _bones)
from .group_joint_native_joint_flow import native_joint_pixel_flow_stream, native_joint_assignment_cost
from .group_joint_person_ownership import trackable_person_ownership
from .group_native_head_orientation import SUPPORTED_ANALYSIS_ROLLS
from .boundary_subject_alignment import BoundarySubjectAlignmentConfig, _action_envelope, _close_face_head_envelope
from .group_scene_role_obligations import scene_role_obligations, CONTRACT as ROLE_CONTRACT

SCENE_CONTRACT = 'native_scene_role_cohort_hypotheses_v3'
DENSE_CONTRACT = 'source_native_scene_role_whole_camera_refresh_v6'
JOINT_LINK_CONTRACT = 'native_required_body_joint_pixel_path_v2'
MAX_JOINT_NATIVE_GAP = 12
MAX_HYPOTHESES_PER_SPAN = 32
# Bounded exploration, NOT detection/confidence/visibility thresholds.
VARIANTS = (None, *SUPPORTED_ANALYSIS_ROLLS)
DISCOVERY_QUERY_CONTRACT = 'ordered_native_scene_analysis_roll_query_v1'


def _discovery_rolls(rolls):
    """Query ownership is narrower than the shared raw/model observation cache.

    Keep order explicit, reject ambiguous inputs before any sensor work, and
    normalize equivalent numeric rolls. A warm broader query must not add
    hypotheses to a later narrow query or erase the earlier discovery archive.
    """
    if (not isinstance(rolls, tuple) or not rolls or len(rolls) > len(VARIANTS)
            or any(roll is not None and (type(roll) not in (int, float)
                or not math.isfinite(roll) or roll not in SUPPORTED_ANALYSIS_ROLLS)
                for roll in rolls)):
        raise ValueError('scene discovery needs an explicit ordered supported roll tuple')
    normalized = tuple(None if roll is None else float(roll) for roll in rolls)
    if len(set(normalized)) != len(normalized):
        raise ValueError('scene discovery roll tuple must have distinct alternatives')
    return normalized


def scene_anchor_ticks(first, end, endpoint_scope='whole'):
    """Observe only an unchanged neighbor's incident edge, both for a new shot.

    A left fixed neighbor supplies EXIT motion; a right neighbor supplies ENTRY
    motion. Their far edges are not changed by this transaction and cannot veto
    its candidate discovery. Mutable shots still require both endpoints and a
    complete dense temporal path. The scope is hash-bound into each hypothesis.
    """
    if endpoint_scope not in ('whole', 'entry', 'exit'):
        raise ValueError('unknown scene endpoint scope')
    if end-first < (7 if endpoint_scope == 'whole' else 4):
        raise ValueError('scene span is too short for its endpoint scope')
    entry, exit_ = (first, first+3), (end-4, end-1)
    return entry+exit_ if endpoint_scope == 'whole' else entry if endpoint_scope == 'entry' else exit_


def _box_union(boxes):
    """Exact area of a union of axis-aligned screen rectangles, not sum(area)."""
    if not boxes:
        return None
    xs = sorted({v for x, y, w, h in boxes for v in (x, x+w)})
    area = 0.
    for left, right in zip(xs, xs[1:]):
        ys = sorted((y, y+h) for x, y, w, h in boxes if x < right and x+w > left)
        total = 0.; start = end = None
        for a, b in ys:
            if start is None:
                start, end = a, b
            elif a > end:
                total += end-start; start, end = a, b
            else:
                end = max(end, b)
        if start is not None:
            total += end-start
        area += (right-left)*total
    return float(area)


def _visible_native_box(box):
    x, y, w, h = box
    left, top, right, bottom = max(0., x), max(0., y), min(1., x+w), min(1., y+h)
    return None if right <= left or bottom <= top else (left, top, right-left, bottom-top)


def _scene_geometry(heads, people):
    # Context outside the actual output remains archived, but cannot occupy
    # screen area or move the on-screen attention statistic from off screen.
    head_boxes = [b for h in heads if (b := _visible_native_box(h['projected_box_xywh'])) is not None]
    boxes = [b for p in people if (b := _visible_native_box(p['projected_box_xywh'])) is not None] or head_boxes
    if not boxes:
        return None
    attention_boxes = head_boxes or boxes
    centers = [(x+w/2, y+h/2) for x, y, w, h in attention_boxes]
    return dict(occupancy=_box_union(boxes),
        attention_xy=[median(p[n] for p in centers) for n in (0, 1)],
        subject_height_quantiles=[float(v) for v in np.quantile([b[3] for b in boxes], (.25, .5, .75))],
        horizontal_span=max(b[0]+b[2] for b in boxes)-min(b[0] for b in boxes),
        head_count=len(heads), person_count=len(people),
        basis='all_observed_native_cohort_projected_not_one_face_or_camera_label')


def _known_parts(person):
    return tuple(name for name in ('left_arm', 'right_arm')
        if person is not None and dict(getattr(person, 'part_states', ())).get(name) == 'observed')


class SceneFrameRejection(ValueError):
    """A measured projection failed admission; its known pixels are not erased.

    The row cannot be used by endpoint/dense admission. Only the conditional
    editorial comparison may retain its separately measured geometry.
    """
    def __init__(self, row):
        self.row = row
        super().__init__(row['frame_admission_violations'][0])


class SceneSensors(source.NativeFamilySensors):
    """One raw cache shared by alternatives; no dominant-face prerequisite."""
    scene_family = True

    def __init__(self, session, assets):
        super().__init__(session, assets)
        self.hypotheses = {}
        self.discoveries = {}
        self.link_cache = {}
        self.frame_cache = {}
        self.dense_cache = {}
        self.shot_intents = {}
        self.endpoint_cache = {}
        self.articulation_raw_rows = {}
        self.articulation_endpoint_cache = {}
        self.original_duty_program = None
        self.original_native_duty_programmes = {}
        self.original_role_programmes = {}

    def install_original_duty_program(self, program):
        from .group_scene_original_duties import validate_original_duty_program
        validate_original_duty_program(program, raw_inventory={r.sha256:r.to_dict()
            for r in self.raw_rows.values()})
        # The caller's mutable object is not this local programme version.
        # Keep earlier versions when switching windows; retained frame/pose
        # references may still name them. This is not controller history.
        owned=family.plain(program)
        if not hasattr(self,'original_native_duty_programmes'):self.original_native_duty_programmes={}
        self.original_duty_program=self.original_native_duty_programmes.setdefault(owned['program_sha256'],owned)

    def record_original_role_program(self,program):
        from .group_scene_epoch_program import validate_original_role_epochs
        validate_original_role_epochs(program)
        owned=family.plain(program);identity=family.digest(owned)
        if not hasattr(self,'original_role_programmes'):self.original_role_programmes={}
        if identity not in self.original_role_programmes:
            self.session.budget.spend('pose_evidence_bytes',len(json.dumps(owned,sort_keys=True,allow_nan=False).encode()))
            self.original_role_programmes[identity]=owned
        return identity

    def duty_binding(self, state, tick):
        from .group_scene_original_duties import bind_row_obligations
        return bind_row_obligations(getattr(self,'original_duty_program',None),state,tick)

    def obligations(self, raw, state, tick, head_id):
        _,names = self.duty_binding(state,tick)
        return scene_role_obligations(raw,state.role,head_id,required_part_names=names)

    def raw(self, state, tick, roll):
        if state.role != GroupOutputRole.COMPOSITION:
            return super().raw(state, tick, roll)
        from .group_scene_composition import observe_native_composition
        if roll is not None:
            raise ValueError('body-composition detector has no head-roll variants')
        request = self.session.domain.request_index(state.eligible_source_index)
        native = state.native_frame_indices[tick-state.first_output_frame]
        key = ('body_composition', request, native)
        if key not in self.raw_rows:
            row = self.session.budget.normalized_native_views.measure(label='native_composition_observation_view',
                input_binding=dict(request_source_index=request,native_frame_index=native,
                    source_clock_identity=state.source_clock_identity),
                producer=lambda:observe_native_composition(self.session, request_source_index=request,
                    native_frame_index=native))
            self.raw_rows[key] = row
        return self.raw_rows[key]

    def endpoint(self, state, *, outgoing):
        """Own-side required pixels BEFORE donors or cross-camera utility."""
        from .group_scene_composition import body_support_links
        ticks = (state.end_output_frame-4, state.end_output_frame-1) if outgoing else (
            state.first_output_frame, state.first_output_frame+3)
        program = getattr(self,'original_duty_program',None)
        key = state.key, outgoing, None if program is None else program['program_sha256']
        if key in self.endpoint_cache:
            return self.endpoint_cache[key]
        h = self.hypothesis(state)
        rows = [self.frame(state, t) for t in ticks]
        raw = [self.raw(state, t, h['analysis_roll']) for t in ticks]
        if raw[0].native_frame_index == raw[1].native_frame_index:
            raise ValueError('repeated native endpoint is not measured motion')
        evidence = dict(output_ticks=list(ticks), required_semantics_available=True,
            head_pixel_link=None, joint_pixel_link=None, composition_support_link=None,
            verified_source_clock=self.session.domain.render_frame_clocks[state.eligible_source_index].to_dict(),
            original_duty_program_sha256=None if program is None else program['program_sha256'],
            quality_approved=False)
        if state.role == GroupOutputRole.COMPOSITION:
            link = body_support_links(self, *raw)
            if not link['observed_support_paths']:
                raise ValueError('composition endpoint native support path unknown')
            evidence['composition_support_link'] = link
        else:
            if state.role != GroupOutputRole.FORMATION:
                ids = [r['selected_frame_local_head_id'] for r in rows]
                link = self._head_links(*raw)
                if ids not in link['unique_links']:
                    raise ValueError('required focal endpoint head pixel path unknown')
                evidence['head_pixel_link'] = link
            parts = set(p for row in rows for p in map(tuple,row['role_obligations']['required_parts']))
            if state.role == GroupOutputRole.FORMATION or parts:
                ids = None if state.role == GroupOutputRole.FORMATION else tuple(
                    r['selected_frame_local_head_id'] for r in rows)
                required = None if ids is None else tuple(sorted({name for _, name in parts}))
                link = (native_body_links(self, *raw) if ids is None else
                    focal_required_motion(self,state,*raw,output_ticks=ticks,
                        selected_head_ids=ids,selected_part_names=required))
                from .group_scene_focal_pose_motion import VIEW_CONTRACT,required_motion_available
                if not required_motion_available(link):
                    if link.get('contract')==VIEW_CONTRACT:
                        raise ValueError('required endpoint registered pose unavailable: '+str(link.get('reason')))
                    raise ValueError('required endpoint joint pixel paths unknown')
                if link.get('contract')==JOINT_LINK_CONTRACT:evidence['joint_pixel_link'] = link
                else:evidence['registered_pose_sequence'] = link
        self.endpoint_cache[key] = evidence
        return evidence

    def _head_links(self, a, b):
        key = a.sha256, b.sha256
        if key in self.link_cache:
            return self.link_cache[key]
        if (a.request_source_index != b.request_source_index or a.native_frame_index > b.native_frame_index
                or a.frame_size_wh != b.frame_size_wh):
            raise ValueError('scene head link changed source/native ordering/raster')
        if a.native_frame_index == b.native_frame_index:
            if a.sha256 != b.sha256:
                raise ValueError('repeated native scene has conflicting raw observations')
            pairs = [(h.observation_id, h.observation_id) for h in a.heads]
        else:
            from .group_scene_head_pixel_flow import measure_head_pixel_links
            value=measure_head_pixel_links(self,a,b)
            self.link_cache[key]=value
            return value
        # Keep ALL possible measured links. Resolve only one-to-one edges;
        # ambiguous edges are still archived and never get a fabricated cost.
        unique = tuple((x, y) for x, y in pairs
            if sum(u == x for u, v in pairs) == 1 and sum(v == y for u, v in pairs) == 1)
        value = dict(raw_pair=[a.sha256, b.sha256], measured_links=[list(p) for p in pairs],
            unique_links=[list(p) for p in unique],
            repeated_native_alias=a.native_frame_index == b.native_frame_index,
            cross_camera_identity_certified=False)
        self.link_cache[key] = value
        return value

    def discover(self, state, *, rolls=VARIANTS, endpoint_scope='whole'):
        rolls = _discovery_rolls(rolls)
        span_key = self.span_key(state)
        if state.role == GroupOutputRole.COMPOSITION:
            rolls = (None,)
        key = (*span_key, endpoint_scope, rolls)
        if key in self.discoveries:
            return tuple(self.hypotheses[h] for h in self.discoveries[key]['hypothesis_ids'])
        try:
            ticks = scene_anchor_ticks(state.first_output_frame, state.end_output_frame, endpoint_scope)
        except ValueError:
            if endpoint_scope not in ('whole', 'entry', 'exit'):
                raise
            self.discoveries[key] = dict(span_key=list(span_key), endpoint_scope=endpoint_scope,
                query_contract=DISCOVERY_QUERY_CONTRACT, analysis_roll_query=list(rolls),
                variants=[], hypothesis_ids=[], reason='scene_span_needs_endpoint_window')
            return ()
        record = dict(span_key=list(span_key), endpoint_scope=endpoint_scope,
            query_contract=DISCOVERY_QUERY_CONTRACT, analysis_roll_query=list(rolls),
            anchor_output_ticks=list(ticks), variants=[],
            hypothesis_ids=[], unresolved_subject_node_retained=True,
            variant_is_not_an_actor=True, quality_approved=False)
        record = self.discoveries.setdefault(key, record)
        for roll in rolls:
            if roll in [v['analysis_roll'] for v in record['variants']]: continue
            rows = tuple(self.raw(state, t, roll) for t in ticks)
            variant = dict(analysis_roll=roll, raw_observation_sha256=[r.sha256 for r in rows],
                heads_by_anchor=[[h.observation_id for h in r.heads] for r in rows],
                people_by_anchor=[[p.observation_id for p in r.people] for r in rows],
                unassociated_head_possible=True, unassociated_body_possible=True,
                long_gap_association='unresolved_endpoint_hypotheses_not_identity')
            record['variants'].append(variant)
            # Observe every source/variant BEFORE judging a complete assignment.
            # A formation uses all bodies, not the global face winner.
            if state.role == GroupOutputRole.COMPOSITION:
                from .group_scene_composition import body_support_links
                if any(len(r.people) < 2 for r in rows):
                    variant['unavailable'] = 'composition_body_support_unknown'; continue
                links = [body_support_links(self, rows[i], rows[i+1])
                    if rows[i].native_frame_index < rows[i+1].native_frame_index else None
                    for i in range(0, len(rows), 2)]
                variant['body_support_links'] = links
                if any(link is None or not link['observed_support_paths'] for link in links):
                    variant['unavailable'] = 'composition_endpoint_pixel_support_unknown'; continue
                candidates = [(None,)*len(ticks)]
            elif state.role == GroupOutputRole.FORMATION:
                if any(len(r.people) < 2 for r in rows):
                    variant['unavailable'] = 'formation_cohort_unknown'; continue
                candidates = [(None,)*len(ticks)]
            else:
                links = [self._head_links(rows[i], rows[i+1]) for i in range(0, len(rows), 2)]
                variant['endpoint_links'] = links
                count = math.prod(len(link['unique_links']) for link in links)
                variant['possible_endpoint_hypotheses'] = count
                if count+len(record['hypothesis_ids']) > MAX_HYPOTHESES_PER_SPAN:
                    variant['unavailable'] = 'scene_hypothesis_budget_unresolved_not_truncated_people'
                    continue
                candidates = [tuple(head for pair in pairs for head in pair)
                    for pairs in product(*(link['unique_links'] for link in links))]
                if not candidates:
                    variant['unavailable'] = 'endpoint_head_pixel_links_unknown'; continue
            # Store compact endpoint possibilities first; no unbounded body-pose
            # Cartesian materialization. Overflow remains explicitly unresolved.
            variant['possible_endpoint_hypotheses'] = len(candidates)
            if len(record['hypothesis_ids'])+len(candidates) > MAX_HYPOTHESES_PER_SPAN:
                variant['unavailable'] = 'scene_hypothesis_budget_unresolved_not_truncated_people'
                continue
            for ids in candidates:
                value = dict(contract=SCENE_CONTRACT, span_key=list(span_key),
                    endpoint_scope=endpoint_scope,
                    anchor_output_ticks=list(ticks), analysis_roll=roll,
                    anchor_head_ids=list(ids), raw_observation_sha256=[r.sha256 for r in rows],
                    role=state.role.value, required_cohorts=ROLE_CONTRACT,
                    shot_intent=self.shot_intents.get((state.first_output_frame,state.end_output_frame,endpoint_scope)),
                    body_anchor_ids=[[p.observation_id for p in r.people] for r in rows]
                        if state.role == GroupOutputRole.COMPOSITION else None,
                    temporal_identity_certified=False, cross_camera_identity_certified=False,
                    whole_head_extent_certified=False, body_phase_certified=False, quality_approved=False)
                identity = family.digest(value)
                self.hypotheses[identity] = value
                record['hypothesis_ids'].append(identity)
        return tuple(self.hypotheses[h] for h in record['hypothesis_ids'])

    def bind(self, state, hypothesis):
        identity = family.digest(hypothesis)
        if self.hypotheses.get(identity) != hypothesis or list(self.span_key(state)) != hypothesis['span_key']:
            raise ValueError('scene hypothesis does not belong to this source/native span')
        return replace(state, editorial_contract_sha256=identity).bind_source(self.session.domain)

    def hypothesis(self, state):
        value = self.hypotheses.get(state.editorial_contract_sha256)
        if value is None or list(self.span_key(state)) != value['span_key']:
            raise ValueError('scene state lacks its frozen pre-camera cohort hypothesis')
        return value

    def frame(self, state, tick, *, head_id='anchor'):
        hypothesis = self.hypothesis(state)
        if head_id == 'anchor':
            if tick not in hypothesis['anchor_output_ticks']:
                raise ValueError('non-anchor scene frame requires the dense measured head link')
            head_id = hypothesis['anchor_head_ids'][hypothesis['anchor_output_ticks'].index(tick)]
        duty_reference,required_names = self.duty_binding(state,tick)
        cache_key = state.key, tick, head_id, None if duty_reference is None else duty_reference['program_sha256']
        if cache_key in self.frame_cache:
            row = self.frame_cache[cache_key]
            if row.get('frame_admission_violations'):
                raise SceneFrameRejection(row)
            return row
        raw = self.raw(state, tick, hypothesis['analysis_roll'])
        if abs(raw.frame_size_wh[0]/raw.frame_size_wh[1]-16/9) > .04:
            raise ValueError('scene native aspect/padding projection unsupported')
        if raw.native_pixel_sha256 is None:
            raise ValueError('scene original raster unavailable')
        violations = []
        def reject(reason):
            if reason not in violations:
                violations.append(reason)
        primary = next((h for h in raw.heads if h.observation_id == head_id), None)
        if state.role not in (GroupOutputRole.FORMATION, GroupOutputRole.COMPOSITION) and primary is None:
            reject('scene focal hypothesis missing, not replaced by background actor')
        try:
            obligations = scene_role_obligations(raw, state.role, head_id,required_part_names=required_names)
            obligations_error = None
        except ValueError as exc:
            # A missing/ambiguous duty is UNKNOWN, never a zero-part duty.
            # Raw context can still have measured, supported output geometry.
            obligations = None
            obligations_error = str(exc)
            reject(obligations_error)
        required_parts = None if obligations is None else set(obligations.required_parts)
        heads = []
        minimum = BoundarySubjectAlignmentConfig().minimum_face_visible_fraction
        for head in raw.heads:
            visible = _visible_native_box(head.box_xywh)
            if visible is None:
                reject('scene native head has no visible raster support')
            box, fraction = source.project_box(head.box_xywh if visible is None else visible, state, tick)
            required = None if obligations is None else head.observation_id in obligations.required_head_ids
            if required and fraction < minimum:
                reject('scene camera crops a role-required native head/cohort member')
            headroom = _close_face_head_envelope(head.box_xywh)
            if head is primary:
                # Existing focal gate, not a lowered threshold or inferred hair.
                x,y,w,h=head.box_xywh
                if x<0 or y<0 or x+w>1 or y+h>1:
                    reject('required focal face is native-cropped; whole face unknown')
                _, fraction_room = source.project_box(headroom, state, tick)
                if fraction_room < minimum:
                    reject('scene focal face-derived headroom clipped')
            else:
                fraction_room = None
            heads.append(dict(observation_id=head.observation_id,
                raw_box_xywh=list(head.box_xywh), native_visible_support_xywh=None if visible is None else list(visible),
                native_cropped=(head.box_xywh[0]<0 or head.box_xywh[1]<0 or head.box_xywh[0]+head.box_xywh[2]>1 or head.box_xywh[1]+head.box_xywh[3]>1), projected_box_xywh=list(box),
                visible_fraction=fraction, headroom_visible_fraction=fraction_room,
                required_by_role=required,
                physical_head_extent=None))
        people = []; observed_arms = []
        for person in raw.people:
            box, fraction = source.project_box(person.box_xywh, state, tick)
            points = tuple((*source.project_point(p[:2], state, tick), p[2]) for p in person.upper_body_xyc)
            parts = dict(getattr(person, 'part_states', ()))
            if state.role == GroupOutputRole.COMPOSITION and (
                    fraction < BoundarySubjectAlignmentConfig().minimum_performer_visible_fraction):
                reject('composition camera clips required measured native body support')
            if state.role == GroupOutputRole.FORMATION:
                if (not trackable_person_ownership(person)
                        or fraction < BoundarySubjectAlignmentConfig().minimum_performer_visible_fraction
                        or set(_known_parts(person)) != {'left_arm', 'right_arm'}):
                    reject('formation body/required arm evidence unknown or camera-clipped')
            arms = []
            anchor = next((h.box_xywh for h in raw.heads
                if h.observation_id == getattr(person, 'unique_head_observation_id', None)), None)
            for name, indices in (('left_arm', (0, 2, 4)), ('right_arm', (1, 3, 5))):
                if name not in _known_parts(person):
                    continue
                required = None if required_parts is None else (person.observation_id, name) in required_parts
                joints_visible = all(0 <= points[n][0] <= 1 and 0 <= points[n][1] <= 1 for n in indices)
                if required and not joints_visible:
                    reject('scene camera crops a role-required observed arm')
                # No face/body association => cannot invent a face-derived hand
                # envelope. Keep the unknown explicitly, never a neutral box.
                if required and anchor is None:
                    reject('scene observed arm has no bound face-envelope anchor')
                envelope = None if anchor is None else _action_envelope(
                    anchor, tuple(person.upper_body_xyc[n][:2] for n in indices))
                visibility = None if envelope is None else source.project_box(envelope, state, tick)[1]
                if required and visibility is not None and visibility < BoundarySubjectAlignmentConfig().minimum_performer_visible_fraction:
                    reject('scene camera clips a required measured arm envelope')
                item = dict(part=name, person_observation_id=person.observation_id,
                    projected_joint_xy=[list(points[n][:2]) for n in indices],
                    native_action_envelope_xywh=None if envelope is None else list(envelope),
                    visible_fraction=visibility, required_by_role=required,
                    measured_joints_visible=joints_visible,
                    envelope_state='unknown_unassociated_head' if anchor is None else 'face_derived_framing_not_anatomy')
                arms.append(item); observed_arms.append(item)
            people.append(dict(observation_id=person.observation_id,
                projected_box_xywh=list(box), visible_fraction=fraction,
                projected_upper_body_xyc=[list(p) for p in points], confidence=person.confidence,
                unique_head_observation_id=getattr(person, 'unique_head_observation_id', None),
                part_states=parts, observed_arms=arms,
                required_by_role=None if obligations is None else person.observation_id in obligations.required_person_ids))
        owners = [p for p in raw.people if getattr(p, 'unique_head_observation_id', None) == head_id]
        if state.role == GroupOutputRole.FOCAL_HEAD_ARMS and (len(owners) != 1
                or set(_known_parts(owners[0])) != {'left_arm', 'right_arm'}):
            reject('scene focal required arms lack unique original-pixel ownership')
        if state.role == GroupOutputRole.FORMATION and len(people) < 2:
            reject('formation cannot be certified by one face ROI')
        selected = next((h for h in heads if h['observation_id'] == head_id), None)
        result = dict(output_tick=tick, request_source_index=raw.request_source_index,
            native_frame_index=raw.native_frame_index, native_observation=raw.to_dict(),
            scene_hypothesis_sha256=state.editorial_contract_sha256,
            selected_frame_local_head_id=head_id, analysis_roll=hypothesis['analysis_roll'],
            raw_face_roi_xywh=None if primary is None else list(primary.box_xywh),
            raw_landmarks_xy=None if primary is None else [list(p) for p in primary.landmarks_xy],
            projected_landmarks_xy=None if primary is None else [list(source.project_point(p, state, tick)) for p in primary.landmarks_xy],
            projected_face_roi_xywh=None if selected is None else selected['projected_box_xywh'],
            face_roi_visible_fraction=None if selected is None else selected['visible_fraction'],
            projected_observed_arms=observed_arms, scene_heads=heads, scene_people=people,
            role_obligations=None if obligations is None else obligations.to_dict(),
            role_obligations_unknown_reason=obligations_error,
            original_duty_reference=duty_reference,
            frame_admission_violations=violations,
            role_geometry_constraints_satisfied=not violations,
            scene_geometry=_scene_geometry(heads, people), role=state.role.value,
            whole_head_extent_certified=False, cross_camera_identity_certified=False,
            body_phase_certified=False, anatomical_occlusion_inferred=False, quality_approved=False)
        from .group_scene_native_blob_storage import frame_wire_storage
        stored,wire_bytes,native_view=frame_wire_storage(result,self.session.budget.normalized_native_views)
        self.session.budget.spend('pose_evidence_bytes',wire_bytes)
        result=dict(stored)
        result['native_observation']=native_view
        self.frame_cache[cache_key] = result
        if violations:
            raise SceneFrameRejection(result)
        return result

    def mutable_camera_inputs(self,state,tick):
        """Conditional raw duties BEFORE admission of a replaceable camera.

        Camera violations remain in the exact measured row. Missing source
        observations, cropped native faces and unresolved required owners do
        not become repairable geometry; fixed neighbors still use frame().
        """
        if state.role not in (GroupOutputRole.FOCAL_HEAD,GroupOutputRole.FOCAL_HEAD_ARMS):
            return self.frame(state,tick),()
        try:row=self.frame(state,tick)
        except SceneFrameRejection as exc:row=exc.row
        selected=next((h for h in row['scene_heads']
            if h['observation_id']==row['selected_frame_local_head_id']),None)
        if (row['role_obligations'] is None or selected is None
                or selected['native_cropped'] or row['raw_face_roi_xywh'] is None):
            if row['frame_admission_violations']:raise SceneFrameRejection(row)
            raise ValueError('mutable fitting requires observed original-duty owner and whole native face')
        head=tuple(row['raw_face_roi_xywh'])
        envelopes=[_close_face_head_envelope(head)]
        for arm in row['projected_observed_arms']:
            if not arm['required_by_role']:continue
            if arm['native_action_envelope_xywh'] is None:
                raise SceneFrameRejection(row)
            envelopes.append(tuple(arm['native_action_envelope_xywh']))
        # Exact raw head/arm records, not an accepted old projection. The
        # caller converts these through the preserved source base crop.
        return row,tuple(envelopes)

    def dense(self, path):
        from .group_scene_execution_scope import mutable_states
        result = []
        for state in mutable_states(path):
            hypothesis = self.hypothesis(state)
            if hypothesis.get('endpoint_scope', 'whole') != 'whole':
                raise ValueError('mutable scene shot cannot borrow fixed-neighbor endpoint scope')
            rows = []; previous = None; current_head = hypothesis['anchor_head_ids'][0]
            for tick in range(state.first_output_frame, state.end_output_frame):
                raw = self.raw(state, tick, hypothesis['analysis_roll'])
                previous_head = current_head
                if previous is not None and state.role not in (GroupOutputRole.FORMATION, GroupOutputRole.COMPOSITION):
                    links = self._head_links(previous, raw)
                    choices = [b for a, b in links['unique_links'] if a == current_head]
                    if len(choices) != 1:
                        raise ValueError('scene dense focal path unknown/competing; no actor substitution')
                    current_head = choices[0]
                if tick in hypothesis['anchor_output_ticks']:
                    expected = hypothesis['anchor_head_ids'][hypothesis['anchor_output_ticks'].index(tick)]
                    if current_head != expected:
                        raise ValueError('scene left/right endpoint hypotheses lack one whole temporal path')
                body_link=None; support_link=None
                if previous is not None and state.role == GroupOutputRole.COMPOSITION:
                    if previous.native_frame_index == raw.native_frame_index:
                        if previous.sha256 != raw.sha256: raise ValueError('repeated composition raw raster differs')
                    else:
                        from .group_scene_composition import body_support_links
                        support_link = body_support_links(self, previous, raw)
                        if not support_link['observed_support_paths']:
                            raise ValueError('composition dense body support lifecycle unknown')
                current_parts = self.obligations(raw,state,tick,current_head).required_parts
                previous_parts = () if previous is None else self.obligations(
                    previous,state,tick-1,previous_head).required_parts
                required_names = tuple(sorted({name for _,name in (*previous_parts,*current_parts)}))
                if previous is not None and (state.role == GroupOutputRole.FORMATION or required_names):
                    if previous.native_frame_index == raw.native_frame_index:
                        if previous.sha256 != raw.sha256: raise ValueError('repeated scene body raster differs')
                    else:
                        body_link=(native_body_links(self,previous,raw) if state.role==GroupOutputRole.FORMATION
                            else focal_required_motion(self,state,previous,raw,output_ticks=(tick-1,tick),
                                selected_head_ids=(previous_head,current_head),selected_part_names=required_names))
                        from .group_scene_focal_pose_motion import VIEW_CONTRACT,required_motion_available
                        if not required_motion_available(body_link):
                            if body_link.get('contract')==VIEW_CONTRACT:
                                raise ValueError('scene dense registered focal pose unavailable: '+str(body_link.get('reason')))
                            raise ValueError('scene dense required body/arm joint pixel lifecycle unknown')
                row = dict(self.frame(state, tick, head_id=current_head))
                # Dense/public JSON remains the original observation schema.
                # The read-through frame cache keeps only its wire packet;
                # frozen owners repack this temporary consumer view at transfer.
                row['native_observation']=family.plain(row['native_observation'])
                row['required_body_temporal_link']=body_link
                row['composition_support_link']=support_link
                projected = self.session.projected_pixels(state, tick)
                if projected is None:
                    raise ValueError('scene dense projected pixels unavailable')
                row['projected_pixels_sha256'] = source.sha256(memoryview(projected)).hexdigest()
                row['same_source_previous_pixel_link_observed'] = (support_link is not None if state.role == GroupOutputRole.COMPOSITION
                    else previous is not None and state.role != GroupOutputRole.FORMATION)
                row['native_repeated_alias'] = previous is not None and previous.native_frame_index == raw.native_frame_index
                rows.append(row); previous = raw
            result.append(dict(first_output_frame=state.first_output_frame, end_output_frame=state.end_output_frame,
                role=state.role.value, rows=rows))
        return result

    def archive(self):
        from .group_scene_native_storage import pack_native_observation
        from .group_scene_programme_archive import programme_records
        views=self.session.budget.normalized_native_views
        native=[pack_native_observation(r.to_dict()) for r in self.raw_rows.values()]
        parallel=[None if r is None else pack_native_observation(r.to_dict())
            for r in self.articulation_raw_rows.values()]
        listed={r['observation_sha256'] for r in (*native,*parallel) if r is not None}
        native.extend(packet for ref,packet in views.views.items() if ref not in listed)
        return dict(contract=SCENE_CONTRACT, discoveries=list(self.discoveries.values()),
            hypotheses=self.hypotheses, native_rows=native,
            parallel_articulation_rows=parallel,
            endpoint_pixel_links=list(self.link_cache.values()),
            primitive_measurements=self.session.budget.primitive_journal.to_dict(),
            normalized_native_view_records=views.to_dict(),
            original_programme_records=programme_records(roles=getattr(self,'original_role_programmes',{}),
                native=getattr(self,'original_native_duty_programmes',{}),
                active=None if getattr(self,'original_duty_program',None) is None else self.original_duty_program['program_sha256']),
            quality_approved=False, association_hypotheses_are_not_named_actors=True)


def formation_view(row, state):
    people = tuple(ObservedFormationPerson(p['observation_id'], tuple(p['projected_box_xywh']),
        tuple(tuple(v) for v in p['projected_upper_body_xyc']), p['confidence']) for p in row['scene_people'])
    return DeclaredFormationView(people, tuple(p.observation_id for p in people), state.output_size)



def native_body_links(sensors, previous, current, *, selected_head_ids=None,selected_part_names=None):
    """Measured joint-local paths for every required body/arm, never head-only authority.

    Unique head paths and frame-local head/body mapping are retained as a
    DIFFERENT diagnostic hypothesis. They cannot set an LK cost to zero or
    mint ObservedFormationTemporalMotion.same_source_pixel_path_verified.
    """
    gap=current.native_frame_index-previous.native_frame_index
    if (previous.request_source_index != current.request_source_index
            or previous.source_clock_identity != current.source_clock_identity
            or previous.source_file_identity != current.source_file_identity
            or previous.frame_size_wh != current.frame_size_wh
            or not 0 < gap <= MAX_JOINT_NATIVE_GAP):
        raise ValueError('scene temporal body measurement needs a chronological original-source pair')
    if selected_head_ids is not None and (not isinstance(selected_head_ids,tuple)
            or len(selected_head_ids)!=2 or any(type(i) is not int for i in selected_head_ids)):
        raise ValueError('focal body scope requires two measured frame-local head IDs')
    if selected_part_names is not None and (selected_head_ids is None
            or not isinstance(selected_part_names,tuple) or not selected_part_names
            or len(set(selected_part_names))!=len(selected_part_names)
            or not set(selected_part_names)<={'left_arm','right_arm'}):
        raise ValueError('focal joint path requires explicit observed arm names')
    required_names=tuple(sorted(selected_part_names or ('left_arm','right_arm')))
    key = (JOINT_LINK_CONTRACT, previous.sha256, current.sha256, selected_head_ids,required_names)
    if key in sensors.link_cache: return sensors.link_cache[key]
    pixel_hashes={}
    def pixels(native):
        image=sensors.session.native_pixels(previous.request_source_index,native)
        if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3
                or image.shape!=(previous.frame_size_wh[1],previous.frame_size_wh[0],3)):
            raise ValueError('required joint path needs every actual same-raster native frame')
        pixel_hashes[native]=source.sha256(memoryview(image)).hexdigest()
        return image
    first,last=pixels(previous.native_frame_index),pixels(current.native_frame_index)
    if [pixel_hashes[r.native_frame_index] for r in (previous,current)]!=[
            previous.native_pixel_sha256,current.native_pixel_sha256]:
        raise ValueError('required joint path native pixels differ from raw observations')
    # Follow the three-output-tick endpoint through its ACTUAL native rasters,
    # with identical LK/error/assignment limits. Keep only the endpoints and
    # one live intermediate here; the session owns its bounded shared LRU.
    # Early no-feature failures do not imply that unread frames passed.
    flows=native_joint_pixel_flow_stream(first,last,previous.people,
        intermediate_pixels=(pixels(n) for n in range(previous.native_frame_index+1,current.native_frame_index)))
    native_indices=sorted(pixel_hashes)
    steps_complete=native_indices==list(range(previous.native_frame_index,current.native_frame_index+1))
    costs = [[native_joint_assignment_cost(f,p,current.frame_size_wh) for p in current.people] for f in flows]
    matrix=np.asarray([[1e6 if c is None else c for c in row] for row in costs])
    assignments=[]
    if matrix.size:
        sensors.session.budget.spend('pose_comparisons',int(matrix.size))
        aa,bb=linear_sum_assignment(matrix)
        for a,b in zip(aa,bb):
            value=costs[a][b]
            if value is None or value>.12: continue
            rivals=[v for i,v in enumerate(costs[a]) if i!=b and v is not None]
            reverse=[row[b] for i,row in enumerate(costs) if i!=a and row[b] is not None]
            if any(v-value<.025 for v in (*rivals,*reverse)): continue
            assignments.append((int(a),int(b)))
    heads=sensors._head_links(previous,current)
    if selected_head_ids is None:
        required_assignments=assignments
        required_ids=[[p.observation_id for p in previous.people],[p.observation_id for p in current.people]]
        complete=bool(previous.people) and len(assignments)==len(previous.people)==len(current.people)
    else:
        owners=[tuple(i for i,p in enumerate(row.people)
            if getattr(p,'unique_head_observation_id',None)==head_id)
            for row,head_id in zip((previous,current),selected_head_ids,strict=True)]
        required_assignments=[] if any(len(v)!=1 for v in owners) else [(owners[0][0],owners[1][0])]
        required_ids=[[row.people[i].observation_id for i in ids]
            for row,ids in zip((previous,current),owners,strict=True)]
        complete=bool(required_assignments) and all(pair in assignments for pair in required_assignments)
    contradicted=False
    for a,b in required_assignments:
        # A known single arm on a head role needs its own measured joints, not
        # an invented obligation for an unknown other arm. Formation/both-arm
        # roles retain all six required arm joints.
        required_joints={joint for name in required_names
            for joint in ((0,2,4) if name=='left_arm' else (1,3,5))}
        if flows[a] is None or not required_joints<={item[0] for item in flows[a].joints}:
            complete=False
        if selected_head_ids is not None and any(
                dict(getattr(person,'part_states',())).get(name)!='observed'
                for person in (previous.people[a],current.people[b]) for name in required_names):
            complete=False
        x=getattr(previous.people[a],'unique_head_observation_id',None)
        y=getattr(current.people[b],'unique_head_observation_id',None)
        observed=[v for u,v in heads['unique_links'] if u==x]
        if x is not None and y is not None and observed and observed != [y]: contradicted=True
    complete=complete and not contradicted and steps_complete
    evidence=dict(contract=JOINT_LINK_CONTRACT,
        raw_observation_sha256=[previous.sha256,current.sha256],
        request_source_index=previous.request_source_index,
        source_clock_identity=previous.source_clock_identity,
        source_file_identity=list(previous.source_file_identity),frame_size_wh=list(previous.frame_size_wh),
        native_frame_indices=native_indices,native_pixel_sha256=[pixel_hashes[n] for n in native_indices],
        native_steps_complete=steps_complete,costs=costs,
        assignments=[list(v) for v in assignments],joint_pixel_flow=[None if f is None else f.to_dict() for f in flows],
        head_pixel_links=heads,head_body_hypotheses_are_not_joint_path_authority=True,
        body_association_basis='native_joint_local_pixel_paths_all_required_arm_joints',
        required_body_scope='all_formation_people' if selected_head_ids is None else 'selected_focal_owner',
        required_body_observation_ids=required_ids,selected_head_ids=selected_head_ids,
        required_part_names=list(required_names),
        complete_required_joint_paths=complete,head_joint_assignment_contradiction=contradicted,
        named_identity_certified=False,body_phase_certified=False,quality_approved=False)
    sensors.session.budget.spend('pose_evidence_bytes',len(__import__('json').dumps(evidence,sort_keys=True).encode()))
    sensors.link_cache[key]=evidence
    return evidence


def validate_required_body_native_sequence(link,previous,current):
    """Bind the required path's actual native sequence; fresh pixel replay remains mandatory."""
    first,last=previous['native_frame_index'],current['native_frame_index']
    sha=lambda value:isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    if (not isinstance(link,dict) or link.get('contract')!=JOINT_LINK_CONTRACT
            or link.get('complete_required_joint_paths') is not True
            or link.get('native_steps_complete') is not True
            or any(type(n) is not int for n in (first,last)) or not 0 < last-first <= MAX_JOINT_NATIVE_GAP
            or previous['request_source_index']!=current['request_source_index']
            or link.get('request_source_index')!=previous['request_source_index']
            or any(family.plain(previous.get(k))!=family.plain(current.get(k))
                or family.plain(link.get(k))!=family.plain(previous.get(k)) for k in (
                    'source_clock_identity','source_file_identity','frame_size_wh'))
            or link.get('raw_observation_sha256')!=[family.digest(previous),family.digest(current)]
            or link.get('native_frame_indices')!=list(range(first,last+1))
            or any(link.get(k) is not False for k in ('named_identity_certified',
                'body_phase_certified','quality_approved','head_joint_assignment_contradiction'))):
        raise ValueError('required body joint path changed its original source/native sequence or authority')
    hashes=link.get('native_pixel_sha256')
    if (not isinstance(hashes,list) or len(hashes)!=last-first+1 or any(not sha(v) for v in hashes)
            or hashes[0]!=previous['native_pixel_sha256'] or hashes[-1]!=current['native_pixel_sha256']):
        raise ValueError('required body joint path skipped or changed an actual native raster')


def endpoint_focal_part_paths(sensors,state,first,last):
    """Known focal arms at fixed endpoints need pixels, not static skeletons."""
    before=tuple(sorted({name for _,name in first['role_obligations']['required_parts']}))
    after=tuple(sorted({name for _,name in last['role_obligations']['required_parts']}))
    if not before and not after:return None
    if before!=after and (getattr(sensors,'original_duty_program',None) is None or not set(before)<=set(after)):
        raise ValueError('scene endpoint selected-arm observation lifecycle unknown')
    h=sensors.hypothesis(state)
    raw=tuple(sensors.raw(state,row['output_tick'],h['analysis_roll']) for row in (first,last))
    if raw[0].native_frame_index==raw[1].native_frame_index:
        raise ValueError('scene repeated native endpoint is not observed arm motion')
    evidence=focal_required_motion(sensors,state,*raw,
        output_ticks=(first['output_tick'],last['output_tick']),
        selected_head_ids=(first['selected_frame_local_head_id'],last['selected_frame_local_head_id']),
        selected_part_names=tuple(sorted(set(before)|set(after))))
    from .group_scene_focal_pose_motion import VIEW_CONTRACT,required_motion_available
    if not required_motion_available(evidence):
        if evidence.get('contract')==VIEW_CONTRACT:
            raise ValueError('scene endpoint registered focal pose unavailable: '+str(evidence.get('reason')))
        raise ValueError('scene endpoint required arm native pixel path unknown')
    return evidence


def focal_required_motion(sensors,state,previous,current,*,output_ticks,selected_head_ids,selected_part_names):
    """Registered model2D temporal type or unchanged legacy pixel producer.

    Backend choice is based on the raw binding contract, not which result wins
    or has a convenient cost. Unknown new-model duties never fall back to a
    more convenient old certificate. Formation has a different explicit path.
    """
    from .group_scene_focal_pose_motion import (registered_raw17_basis,focal_pose_view)
    if all(registered_raw17_basis(row) for row in (previous,current)):
        program=getattr(sensors,'original_duty_program',None)
        key=('registered_focal_pose_view',state.key,previous.sha256,current.sha256,
            output_ticks,selected_head_ids,selected_part_names,
            None if program is None else program['program_sha256'])
        if key not in sensors.link_cache:
            sensors.link_cache[key]=focal_pose_view(sensors,state,previous,current,output_ticks=output_ticks,
                selected_head_ids=selected_head_ids,required_part_names=selected_part_names)
        return sensors.link_cache[key]
    return native_body_links(sensors,previous,current,selected_head_ids=selected_head_ids,
        selected_part_names=selected_part_names)


def temporal_formation(sensors, state, outgoing):
    """Original-pixel joint flow + one-to-one body ownership, not rigid ROI LK."""
    ticks = (state.end_output_frame-4, state.end_output_frame-1) if outgoing else (
        state.first_output_frame, state.first_output_frame+3)
    h = sensors.hypothesis(state)
    raw = tuple(sensors.raw(state, t, h['analysis_roll']) for t in ticks)
    projected = tuple(sensors.frame(state, t) for t in ticks)
    views = tuple(formation_view(r, state) for r in projected)
    if raw[0].native_frame_index == raw[1].native_frame_index:
        return None, dict(reason='repeated_native_is_not_body_motion', motion=None)
    evidence = native_body_links(sensors, *raw)
    if not evidence['complete_required_joint_paths']:
        return None, dict(evidence, reason='required_formation_temporal_joint_paths_unknown')
    assignments = evidence['assignments']
    people = list(views[1 if outgoing else 0].people)
    native = tuple(r.native_frame_index for r in raw)
    for a, b in assignments:
        x, y = views[0].people[a], views[1].people[b]
        bones_a, bones_b = _bones(x, state.output_size), _bones(y, state.output_size)
        common = tuple(k for k in bones_a if k in bones_b)
        if len(common) < 2:
            return None, dict(evidence, reason='required_formation_bone_motion_unknown')
        velocity = tuple((k, *map(float, (bones_b[k]-bones_a[k])/(3/state.output_fps))) for k in common)
        motion = ObservedFormationTemporalMotion(state.key, ticks, native, state.output_fps,
            (x.observation_id, y.observation_id), velocity, True)
        index = b if outgoing else a
        people[index] = replace(people[index], temporal_motion=motion)
    return DeclaredFormationView(tuple(people), tuple(p.observation_id for p in people), state.output_size), evidence


def selected_focal_motion(rows, states):
    """Reuse the existing five-point and arm-bone measurements/gates.

    Only the selection input changes: a hash-bound scene hypothesis, not the
    projected global-largest face. No legacy global-focal certificate is minted.
    Unknown joints remain masked before the existing arm measurement functions.
    """
    from .visual_score import _normalized_face_metadata
    from .observed_head_motion import head_motion_signature, measure_head_motion
    from .closeup_gesture import measure_arm_motion, observed_arm_articulation
    from .rendered_proxy_review import ProxyReviewConfig
    from .group_motion_camera_refinement import _box_discontinuity
    signatures = []; arms = []
    for row, state in zip(rows, states, strict=True):
        points = row['projected_landmarks_xy']; box = row['projected_face_roi_xywh']
        if points is None or box is None or any(not 0 <= v < 1 for p in points for v in p):
            signatures.append(None)
        else:
            # Same 854x480 sensor geometry as observe_projected_boundary.
            face = SimpleNamespace(bbox_xywh=tuple(v*d for v,d in zip(box,(854,480,854,480))),
                landmarks_xy=tuple((x*854,y*480) for x,y in points))
            _, landmarks, _, _, geometry = _normalized_face_metadata(face, frame_width=854, frame_height=480)
            signatures.append(head_motion_signature(geometry, landmarks))
        owners = [p for p in row['scene_people'] if p['unique_head_observation_id'] == row['selected_frame_local_head_id']]
        if len(owners) != 1:
            arms.append(None); continue
        p = owners[0]; upper = [list(v) for v in p['projected_upper_body_xyc']]
        for name, indices in (('left_arm',(0,2,4)),('right_arm',(1,3,5))):
            if p['part_states'].get(name) != 'observed':
                for i in indices: upper[i][2] = 0.  # Explicit MASK, never an observation/cost.
        arms.append(SimpleNamespace(reason='observed',face_anchor_consistent=True,
            upper_body_xyc=tuple(tuple(v) for v in upper),frame_size_wh=state.output_size))
    native = tuple(r['native_frame_index'] for r in rows)
    intervals = (3/states[0].output_fps,1/states[0].output_fps,3/states[0].output_fps)
    # Discovery proved each endpoint's original-pixel same-source head path.
    head = measure_head_motion(tuple(signatures),native_frames=native,
        output_intervals_seconds=intervals,same_source_face_paths_pixel_verified=True)
    arm = measure_arm_motion(tuple(arms),native_frames=native,output_intervals_seconds=intervals)
    metrics = {}
    if head is not None:
        metrics.update(head_predicted=head.predicted_pose_cost,head_velocity=head.velocity_jump_per_output_tick)
    if arm is not None:
        for i in arm.observed_bones:
            metrics['arm_predicted_'+str(i)] = arm.predicted_jump_radians[i]
            metrics['arm_velocity_'+str(i)] = arm.velocity_jump_per_output_tick_radians[i]
    # Preserve static endpoint bones too, without labelling them four-frame motion.
    endpoint = [observed_arm_articulation(a) for a in arms[1:3]]
    if all(v is not None for v in endpoint):
        for i,(a,b) in enumerate(zip(*endpoint)):
            if a is not None and b is not None:
                metrics['arm_endpoint_'+str(i)] = min(1.,max(0.,(1.-sum(x*y for x,y in zip(a,b)))/2.))
    boxes = [r['projected_face_roi_xywh'] for r in rows]
    face_metrics = None if boxes[1] is None or boxes[2] is None else _box_discontinuity(boxes[1],boxes[2])
    limits = ProxyReviewConfig()
    face_ok = face_metrics is not None and face_metrics[0] <= limits.maximum_face_scale_discontinuity and face_metrics[1] <= limits.maximum_face_position_discontinuity
    required_arms = (any(s.role == GroupOutputRole.FOCAL_HEAD_ARMS for s in states)
        or any(row['role_obligations']['required_parts'] for row in rows))
    required_bones={bone for row in rows for _,name in row['role_obligations']['required_parts']
        for bone in ((0,1) if name=='left_arm' else (2,3))}
    if any(s.role==GroupOutputRole.FOCAL_HEAD_ARMS for s in states):required_bones.update(range(4))
    complete = face_ok and head is not None and (not required_arms or
        arm is not None and required_bones<=set(arm.observed_bones))
    return bool(complete),metrics,dict(head_motion=None if head is None else head.to_dict(),
        arm_motion=None if arm is None else arm.to_dict(),face_metrics=face_metrics,
        face_limits=dict(scale=limits.maximum_face_scale_discontinuity,position=limits.maximum_face_position_discontinuity),
        focal_basis='bound_scene_hypothesis_not_legacy_global_focal',quality_approved=False)


def composition_endpoint_screen_motion(state,rows,endpoint):
    """Already measured own-side pixels viewed through ONE complete camera.

    Modalities remain explicit. Native ROI particles, legacy measured ROI LK
    and registered focal landmarks do not become anatomical/cast authority.
    A missing pixel producer must never borrow detector attention as motion.
    """
    from .group_scene_region_flow import CONTRACT as REGION_CONTRACT,project_region_distribution
    from .group_scene_composition import LINK_CONTRACT,validate_support_link
    ticks=tuple(row['output_tick'] for row in rows)
    raw=tuple(family.plain(row['native_observation']) for row in rows)
    if (endpoint.get('output_ticks')!=list(ticks) or len(ticks)!=2
            or any(type(t) is not int for t in ticks)
            or not state.first_output_frame<=ticks[0]<ticks[1]<state.end_output_frame
            or any(row['request_source_index']!=state.segments[0].source_index
                or row['source_clock_identity']!=state.source_clock_identity for row in raw)
            or [state.native_frame_indices[t-state.first_output_frame] for t in ticks]
                !=[row['native_frame_index'] for row in raw]):
        raise ValueError('composition screen motion borrowed another source/native/output occurrence')
    if state.role==GroupOutputRole.COMPOSITION:
        link=endpoint.get('composition_support_link') or {}
        if link.get('contract')==REGION_CONTRACT:
            from fractions import Fraction
            from .render_frame_schedule import VerifiedRenderFrameClock
            clock=endpoint.get('verified_source_clock')
            if not isinstance(clock,dict):
                raise ValueError('native region motion lacks its bound source clock, not an output-fps substitute')
            try:
                native_clock=VerifiedRenderFrameClock(Fraction(clock['frame_rate']),clock['frame_count'],
                    clock['packet_grid_sha256'],clock['grid_basis'])
            except (KeyError,TypeError,ValueError,ZeroDivisionError) as exc:
                raise ValueError('native region motion source clock malformed') from exc
            if family.digest(clock)!=family.digest(native_clock.to_dict()):
                raise ValueError('native region motion changed its actual source clock preimage')
            measured=project_region_distribution(link,*raw,state=state,output_ticks=ticks,native_clock=native_clock)
        elif link.get('contract')==LINK_CONTRACT:
            # Legacy contracts use their actual LK prediction, NOT a fresh
            # destination detector box whose jitter could fake motion.
            validate_support_link(link,*raw);velocity=[]
            lookup={p['observation_id']:i for i,p in enumerate(raw[0]['people'])}
            for old_id,new_id in link['assignments']:
                index=lookup[old_id];before=raw[0]['people'][index]['box_xywh']
                after=link['predicted_boxes'][index]
                a=(before[0]+before[2]/2,before[1]+before[3]/2)
                b=(after[0]+after[2]/2,after[1]+after[3]/2)
                velocity.append((np.asarray(source.project_point(b,state,ticks[1]))-
                    source.project_point(a,state,ticks[0]))/((ticks[1]-ticks[0])/state.output_fps))
            measured=dict(basis='legacy_native_pixel_lk_roi_centres_not_detector_attention',
                flow_sha256=family.digest(link),
                velocity_quantiles=np.quantile(velocity,(.1,.5,.9),axis=0).tolist())
        else:raise ValueError('composition has no explicit supported pixel motion modality')
    elif state.role==GroupOutputRole.FORMATION:
        link=endpoint.get('joint_pixel_link') or {}
        validate_required_body_native_sequence(link,*raw)
        velocity=[]
        for index,destination in link['assignments']:
            flow=link['joint_pixel_flow'][index]
            if (flow is None or flow.get('observation_id')!=raw[0]['people'][index]['observation_id']
                    or not {0,1,2,3,4,5}<={j[0] for j in flow['joints']}):
                raise ValueError('composition formation side lost its actual required joint image movement')
            for item in flow['joints']:
                joint,x,y,tx,ty,error,features=item
                if joint not in (0,1,2,3,4,5):continue
                if ([x,y]!=raw[0]['people'][index]['upper_body_xyc'][joint][:2]
                        or error>1.25 or features<3):
                    raise ValueError('composition formation motion borrowed another native joint feature')
                velocity.append((np.asarray(source.project_point((tx,ty),state,ticks[1]))-
                    source.project_point((x,y),state,ticks[0]))/((ticks[1]-ticks[0])/state.output_fps))
        measured=dict(basis='required_native_joint_local_pixel_flow_projected_by_complete_camera',
            flow_sha256=family.digest(link),
            velocity_quantiles=np.quantile(velocity,(.1,.5,.9),axis=0).tolist())
    else:
        link=endpoint.get('head_pixel_link') or {};heads=[r['selected_frame_local_head_id'] for r in rows]
        if (heads not in link.get('unique_links',())
                or link.get('raw_pair')!=[family.digest(row) for row in raw]):
            raise ValueError('composition cut lacks its other side registered focal pixel path')
        selected=[next((h for h in row['heads'] if h['observation_id']==head),None)
            for row,head in zip(raw,heads,strict=True)]
        if any(h is None or len(h.get('landmarks_xy',()))!=5 for h in selected):
            raise ValueError('composition focal motion has no complete registered native landmarks')
        velocity=[(np.asarray(source.project_point(b,state,ticks[1]))-
            source.project_point(a,state,ticks[0]))/((ticks[1]-ticks[0])/state.output_fps)
            for a,b in zip(selected[0]['landmarks_xy'],selected[1]['landmarks_xy'],strict=True)]
        measured=dict(basis='registered_focal_landmarks_with_unique_original_pixel_path',
            flow_sha256=family.digest(link),
            velocity_quantiles=np.quantile(velocity,(.1,.5,.9),axis=0).tolist())
    result=dict(measured)
    result.update(contract='complete_camera_supported_screen_motion_distribution_v1',
        complete_state_preimage=state.to_dict(),output_ticks=list(ticks),
        raw_observation_sha256=[family.digest(row) for row in raw],
        cross_camera_identity_certified=False,body_phase_certified=False,quality_approved=False)
    return result


def validate_composition_edge_motion(edge,*,hypotheses,hypothesis_keys,state_keys,source_domain):
    """Rederive projection and utility from bound point paths and cameras.

    This received-algebra check does not authenticate the physical history of
    its LK values. Normal fresh selected parent/final pixel replay is mandatory.
    """
    from fractions import Fraction
    from pathlib import Path
    from .group_source_domain import GroupSourceDomain
    from .render_frame_schedule import VerifiedRenderFrameClock
    from .group_scene_focal_pose_motion import restore_view_state
    from .group_scene_region_flow import CONTRACT as REGION_CONTRACT
    endpoints=edge.get('required_endpoint_evidence')
    has_native=bool(endpoints and any((e.get('composition_support_link') or {}).get('contract')
        ==REGION_CONTRACT for e in endpoints))
    basis='conditional_supported_pixel_distribution_not_detector_attention_or_3d_camera'
    if edge.get('camera_direction_basis')!=basis:
        if has_native:raise ValueError('native region cut cannot use detector attention as its motion utility')
        return  # Explicit legacy receipts do not gain this new native modality.
    views=edge.get('composition_pixel_motion')
    if views is None and edge.get('available') is False:
        if (edge.get('metrics') or {}).get('screen_motion') is not None:
            raise ValueError('unknown composition pixel motion was imputed as a known detector cost')
        return
    if (not isinstance(views,list) or len(views)!=2 or not isinstance(endpoints,list)
            or len(endpoints)!=2 or len(edge.get('rows',()))!=4):
        raise ValueError('composition cut omitted its measured pixel motion views')
    eligible=source_domain['eligible_request_indices'];clocks=source_domain['render_frame_clocks']
    files=[Path('EXPLICIT_EXCLUDED_SOURCE_NOT_OPENED') for _ in range(max(eligible)+1)]
    for i,path in zip(eligible,source_domain['eligible_sources'],strict=True):files[i]=Path(path)
    domain=GroupSourceDomain(tuple(files),tuple(eligible),(0.,)*len(eligible),(None,)*len(eligible),
        tuple(tuple(c) for c in source_domain['scene_cut_times']),tuple(
            VerifiedRenderFrameClock(Fraction(c['frame_rate']),c['frame_count'],
                c['packet_grid_sha256'],c['grid_basis']) for c in clocks))
    expected=[]
    for index,(view,endpoint) in enumerate(zip(views,endpoints,strict=True)):
        h=hypotheses[hypothesis_keys[index]]
        chosen=restore_view_state(view.get('complete_state_preimage'),domain,h,state_keys[index])
        rebuilt=composition_endpoint_screen_motion(chosen,edge['rows'][2*index:2*index+2],endpoint)
        if family.digest(view)!=family.digest(rebuilt):
            raise ValueError('composition output velocity differs from its actual native paths/complete camera')
        expected.append(np.asarray(rebuilt['velocity_quantiles']))
    motion=float(np.mean(np.linalg.norm(expected[0]-expected[1],axis=1)))
    if (edge['metrics'].get('screen_motion')!=motion
            or edge.get('screen_velocity_outgoing')!=expected[0][1].tolist()
            or edge.get('screen_velocity_incoming')!=expected[1][1].tolist()):
        raise ValueError('composition screen utility changed its actual projected pixel distribution')


def compare_scene_edge(sensors, left, right):
    """Both sides use the SAME complete state/hypothesis, never per-edge actors."""
    ticks = (left.end_output_frame-4, left.end_output_frame-1,
        right.first_output_frame, right.first_output_frame+3)
    rows = []; frame_rejections = []
    try:
        for s, t in zip((left, left, right, right), ticks):
            try:
                rows.append(sensors.frame(s, t))
            except SceneFrameRejection as exc:
                rows.append(exc.row)
                frame_rejections.append(dict(state_key=s.key,output_tick=t,
                    reasons=exc.row['frame_admission_violations']))
    except JointGroupBudgetExhausted as exc:
        exc.partial_evidence=dict(left_state=left.key,right_state=right.key,rows=rows,
            metrics=None,available=False,required_semantics_available=False,
            outcome='BUDGET_PARTIAL' if rows else 'BUDGET_NOT_STARTED',
            unexecuted_measurements_are_unknown=True)
        raise
    except ValueError as exc:
        return dict(available=False, required_semantics_available=False, reason=str(exc), metrics=None,
            cross_camera_identity_certified=False, body_phase_certified=False)
    a, b = rows[1]['scene_geometry'], rows[2]['scene_geometry']
    if a is None or b is None:
        return dict(available=False, required_semantics_available=False, reason='scene geometry unknown', metrics=None)
    metrics = dict(occupancy=abs(a['occupancy']-b['occupancy']),
        attention=float(np.linalg.norm(np.asarray(a['attention_xy'])-b['attention_xy'])),
        scale=float(np.mean(np.abs(np.log(np.asarray(a['subject_height_quantiles'])/b['subject_height_quantiles'])))),
        span=abs(math.log(max(a['horizontal_span'], 1e-9)/max(b['horizontal_span'], 1e-9))))
    def velocity(pair, fps):
        if any(r['scene_geometry'] is None for r in pair):return None
        return (np.asarray(pair[1]['scene_geometry']['attention_xy'])-
            pair[0]['scene_geometry']['attention_xy'])*(fps/3)
    va = velocity(rows[:2],left.output_fps); vb = velocity(rows[2:],right.output_fps)
    composition_cut=GroupOutputRole.COMPOSITION in (left.role,right.role)
    metrics['screen_motion'] = None if composition_cut or va is None or vb is None else float(np.linalg.norm(va-vb))
    formation = left.role == GroupOutputRole.FORMATION or right.role == GroupOutputRole.FORMATION
    evidence = dict(left_state=left.key, right_state=right.key, rows=rows,
        metrics=metrics, available=False, required_semantics_available=False,
        required_endpoint_evidence=None, first_incoming_output_frame=right.first_output_frame,
        screen_velocity_outgoing=None if composition_cut or va is None else va.tolist(),
        screen_velocity_incoming=None if composition_cut or vb is None else vb.tolist(),
        camera_direction_basis='measured_screen_attention_motion_not_inferred_3d_camera',
        cross_camera_identity_certified=False, body_phase_certified=False)
    # Projected geometry is independently known even when required native
    # motion is UNKNOWN. Never erase it or convert UNKNOWN into a zero cost.
    if not formation and GroupOutputRole.COMPOSITION not in (left.role, right.role):
        boxes = [r['projected_face_roi_xywh'] for r in rows]
        if any(v is None for v in boxes):
            if not frame_rejections:
                return dict(evidence, reason='focal hypothesis unobserved')
        else:
            centers = [np.asarray((v[0]+v[2]/2, v[1]+v[3]/2)) for v in boxes]
            metrics['focal_position'] = float(np.linalg.norm(centers[1]-centers[2]))
            metrics['focal_scale'] = abs(math.log(boxes[1][3]/boxes[2][3]))
            velocities = [(centers[1]-centers[0])*left.output_fps/3, (centers[3]-centers[2])*right.output_fps/3]
            metrics['focal_motion'] = float(np.linalg.norm(velocities[0]-velocities[1]))
    if frame_rejections:
        return dict(evidence, reason=frame_rejections[0]['reasons'][0],
            frame_admission_violations=frame_rejections,
            known_geometry_retained_on_frame_rejection=True,
            required_motion_unknown_not_zero=True)
    try:
        endpoints = [sensors.endpoint(left, outgoing=True), sensors.endpoint(right, outgoing=False)]
    except JointGroupBudgetExhausted as exc:
        exc.partial_evidence=dict(evidence,outcome='BUDGET_PARTIAL',
            unexecuted_measurements_are_unknown=True,
            known_geometry_retained_on_missing_motion=True,required_motion_unknown_not_zero=True)
        raise
    except ValueError as exc:
        return dict(evidence, reason=str(exc), known_geometry_retained_on_missing_motion=True,
            required_motion_unknown_not_zero=True)
    evidence.update(available=True, required_semantics_available=True, required_endpoint_evidence=endpoints)
    if composition_cut:
        from .group_scene_editorial_policy import composition_cost
        from .group_scene_articulation import compare_composition_articulation
        evidence.update(detector_attention_velocity_outgoing=va.tolist(),
            detector_attention_velocity_incoming=vb.tolist(),composition_pixel_motion=None,
            camera_direction_basis='conditional_supported_pixel_distribution_not_detector_attention_or_3d_camera')
        try:
            motion=[composition_endpoint_screen_motion(s,pair,endpoint)
                for s,pair,endpoint in zip((left,right),(rows[:2],rows[2:]),endpoints,strict=True)]
        except JointGroupBudgetExhausted as exc:
            exc.partial_evidence=dict(evidence,available=False,outcome='BUDGET_PARTIAL',
                unexecuted_measurements_are_unknown=True)
            raise
        except ValueError as exc:
            return dict(evidence,available=False,reason=str(exc),
                known_geometry_retained_on_missing_motion=True,required_motion_unknown_not_zero=True)
        outgoing,incoming=[np.asarray(m['velocity_quantiles']) for m in motion]
        metrics['screen_motion']=float(np.mean(np.linalg.norm(outgoing-incoming,axis=1)))
        evidence.update(composition_pixel_motion=motion,
            screen_velocity_outgoing=outgoing[1].tolist(),screen_velocity_incoming=incoming[1].tolist())
        articulation = compare_composition_articulation(sensors, left, right)
        if articulation['available']:
            metrics.update(articulation['metrics'])
        evidence.update(comparison_mode='predeclared_composition_cut',
            articulated_cross_camera_motion=articulation, cross_camera_actor_correspondence=None,
            framing_cost=composition_cost(evidence), quality_approved=False)
        return evidence
    if formation:
        outgoing, eo = temporal_formation(sensors, left, True)
        incoming, ei = temporal_formation(sensors, right, False)
        evidence['formation_temporal'] = [eo, ei]
        if outgoing is None or incoming is None:
            return dict(evidence, available=False, reason='formation_required_body_motion_unknown')
        match = compare_projected_formation(outgoing, incoming, use_temporal_correspondence=True)
        evidence['formation_match'] = match.to_dict()
        if not match.observed:
            return dict(evidence, available=False, reason=match.status)
        metrics['formation_pose'] = match.pose_mismatch
        metrics['formation_framing'] = match.framing_cost
    else:
        # Raw face-landmark shape, roll and screen velocity are measured. This
        # is a visual correspondence hypothesis, never a named actor/phase.
        complete, measured, motion = selected_focal_motion(rows, (left,left,right,right))
        metrics.update(measured); evidence['selected_focal_motion'] = motion
        if not complete:
            return dict(evidence,available=False,reason='required_selected_motion_or_unchanged_face_limits_unavailable')
    evidence['framing_cost'] = sum(min(1., v) for v in metrics.values())/len(metrics)
    return evidence


def coherent_baseline(values, *, focal_assignment_unique=True):
    """One observed conditional reference, never a best-of-different-actors vector.

    Prefer an available whole comparison, ranked by the existing capped mean.
    Ambiguous focal assignment still cannot create an unconditional actor fact:
    only its observed scene components become nonregression requirements. Keep
    every alternative and the chosen keys so this ranking remains auditable.
    """
    scene_components={'occupancy','attention','scale','span','screen_motion',
        'formation_pose','formation_framing'}
    measured=[row for row in values if row.get('metrics') and any(
        value is not None for value in row['metrics'].values()) and all(
        value is None or type(value) in (int,float) and math.isfinite(value) and value>=0
        for value in row['metrics'].values())]
    complete=[row for row in measured if row.get('available')]
    def rank(row):
        numbers=[v for v in row['metrics'].values() if v is not None]
        return (sum(min(1.,v) for v in numbers)/len(numbers),
            str(row.get('left_state','')),str(row.get('right_state','')))
    selected=min(complete or measured,key=rank) if measured else None
    metrics={} if selected is None else {key:value for key,value in selected['metrics'].items()
        if value is not None and (focal_assignment_unique or key in scene_components)}
    return dict(metrics=metrics,available=bool(complete),observed_alternatives=values,
        coherent_reference_state_keys=None if selected is None else
            [selected.get('left_state'),selected.get('right_state')],
        focal_assignment_unique=focal_assignment_unique,
        focal_conditional_cost_not_imputed=True,
        conditional_reference_is_not_actor_certainty=True,
        componentwise_minimum_across_incompatible_hypotheses=False)


def preserves_known(before, after):
    """Semantic unknown is distinct from a technical invariant and style cost."""
    if (not after.get('available') or not after.get('metrics') or any(
            value is not None and (type(value) not in (int,float)
                or not math.isfinite(value) or value<0)
            for value in after['metrics'].values())):
        return False
    known = before.get('metrics') or {}
    return all(value is None or (key in after['metrics'] and after['metrics'][key] is not None
        and after['metrics'][key] <= value+1e-9) for key, value in known.items())


def has_editorial_gain(before, after):
    """Actual known-value improvement, never availability-only coverage gain.

    The required-duty gate and known nonregression remain independent. A
    missing old arm measurement is not a perfect arm cost or an automatic
    improvement when a candidate finally has a measurement.
    """
    known = before.get('metrics') or {}
    if any(value is not None and (type(value) not in (int,float)
            or not math.isfinite(value) or value < 0) for value in known.values()):
        return False
    return preserves_known(before, after) and any(value is not None
        and after['metrics'][key] < value-1e-6 for key,value in known.items())


def validate_scene_dense(dense):
    """Consumer-side structure and per-tick evidence; fresh replay still required."""
    dense=family.plain(dense)
    from .group_scene_source_owner import relation
    from .source_derivative_diversity import DERIVATIVE_DISTINCT
    if dense.get('contract')!=DENSE_CONTRACT:
        raise ValueError('scene dense consumer requires its explicit role contract')
    family.digest(dense)  # Recursively reject nonfinite evidence before comparisons.
    from .group_scene_intent import validate_intent
    from .group_scene_composition import validate_raw_composition,validate_support_link
    from .group_scene_articulation import validate_articulation_evidence
    from .group_native_detector_result import validate_serialized_detector_result
    from .group_scene_focal_pose_motion import VIEW_CONTRACT,required_motion_available,validate_focal_pose_view
    raw_inventory={family.digest(row):row for row in (dense.get('scene_discovery') or {}).get('native_rows',())}
    program=dense.get('original_duty_program')
    if program is not None:
        from .group_scene_original_duties import validate_original_duty_program,duty_at,FOCAL
        validate_original_duty_program(program,raw_inventory=raw_inventory)
        if program['eligible_request_indices']!=dense['source_domain']['eligible_request_indices']:
            raise ValueError('original duty programme changed source-domain ownership')
    for edge in dense.get('incident_projected_observations', ()):
        if (edge.get('frame_admission_violations') or any(
                row.get('frame_admission_violations')
                or row.get('role_geometry_constraints_satisfied',True) is not True
                for row in edge.get('rows',()))):
            raise ValueError('scene incident cannot admit a measured frame with role/visibility rejection')
        if edge.get('comparison_mode') == 'predeclared_composition_cut':
            validate_articulation_evidence(edge.get('articulated_cross_camera_motion'),
                edge.get('metrics'), dense.get('native_assets'))
    hypotheses=dense.get('scene_hypotheses');keys=dense.get('hypothesis_keys')
    if not isinstance(hypotheses,dict) or not isinstance(keys,list) or len(keys)!=len(dense['state_keys']):
        raise ValueError('scene dense hypothesis inventory missing')
    from .group_scene_execution_scope import dense_layout
    mutable_indices,pair_indices=dense_layout(dense)
    if (dense.get('global_execution_scope') is not None
            and len(pair_indices)!=len(dense.get('incident_projected_observations',()))):
        raise ValueError('scene dense lost its explicit incident edge inventory')
    for index, key in enumerate(keys):
        h=hypotheses.get(key)
        if not isinstance(h,dict) or family.digest(h)!=key or h.get('contract')!=SCENE_CONTRACT:
            raise ValueError('scene hypothesis hash/contract changed')
        if h.get('required_cohorts')!=ROLE_CONTRACT:
            raise ValueError('scene hypothesis does not bind declared role obligations')
        if h['role']!=h['span_key'][-1]:
            raise ValueError('scene hypothesis role differs from its bound source span')
        scope=h.get('endpoint_scope','whole')
        validate_intent(h.get('shot_intent'),role=h['role'],first=h['span_key'][1],end=h['span_key'][2],endpoint_scope=scope)
        allowed=(('whole',) if dense.get('global_execution_scope') is not None else
            ('whole','exit') if index==0 else ('whole','entry') if index==len(keys)-1 else ('whole',))
        if scope not in allowed:
            raise ValueError('scene endpoint scope does not own this fixed/mutable position')
        expected=scene_anchor_ticks(h['span_key'][1],h['span_key'][2],scope)
        if (h['anchor_output_ticks']!=list(expected) or len(h['anchor_head_ids'])!=len(expected)
                or len(h['raw_observation_sha256'])!=len(expected)):
            raise ValueError('scene hypothesis omitted required endpoint observations')
        if any(h.get(k) is not False for k in ('temporal_identity_certified','cross_camera_identity_certified',
                'whole_head_extent_certified','body_phase_certified','quality_approved')):
            raise ValueError('scene hypotheses cannot become anatomy, identity, phase or quality proof')
    if program is not None:
        for index,edge in enumerate(dense.get('incident_projected_observations',())):
            pair=pair_indices[index]
            rows=edge.get('rows');endpoints=edge.get('required_endpoint_evidence')
            if (index+1>=len(keys) or not isinstance(rows,list) or len(rows)!=4
                    or not isinstance(endpoints,list) or len(endpoints)!=2
                    or any(e.get('original_duty_program_sha256')!=program['program_sha256'] for e in endpoints)):
                raise ValueError('scene incident lost the original required endpoint duty programme')
            for position,row in enumerate(rows):
                side=position//2;h=hypotheses[keys[pair[side]]]
                first,end=h['span_key'][1:3]
                expected_tick=(end-4,end-1)[position] if side==0 else (first,first+3)[position-2]
                if row.get('output_tick')!=expected_tick or row.get('role')!=h['role']:
                    raise ValueError('scene incident duty borrowed another output tick/role')
                ref=duty_at(program,dict(first=first,end=end,role=h['role']),expected_tick)
                if row.get('original_duty_reference')!=ref:
                    raise ValueError('scene incident changed its original required duty reference')
                names=tuple(ref['required_part_names']) if h['role'] in FOCAL else None
                obligations=scene_role_obligations(row['native_observation'],h['role'],
                    row['selected_frame_local_head_id'],required_part_names=names)
                if family.digest(row.get('role_obligations'))!=family.digest(obligations.to_dict()):
                    raise ValueError('scene incident changed original required/context part scope')
    for index,edge in enumerate(dense.get('incident_projected_observations',())):
        if index+1>=len(keys):raise ValueError('scene edge inventory exceeds its complete state path')
        validate_composition_edge_motion(edge,hypotheses=hypotheses,
            hypothesis_keys=[keys[i] for i in pair_indices[index]],state_keys=[dense['state_keys'][i] for i in pair_indices[index]],
            source_domain=dense['source_domain'])
    for index,edge in enumerate(dense.get('incident_projected_observations',())):
        endpoints=edge.get('required_endpoint_evidence')
        if endpoints is None:continue
        if not isinstance(endpoints,list) or len(endpoints)!=2 or index+1>=len(keys):
            raise ValueError('scene incident endpoint evidence has another path scope')
        for offset,endpoint in enumerate(endpoints):
            view=endpoint.get('registered_pose_sequence')
            if view is None:continue
            state_index=pair_indices[index][offset]
            h=hypotheses[keys[state_index]]
            pair=view.get('primary_raw_observation_sha256')
            if (not isinstance(pair,list) or len(pair)!=2 or any(ref not in raw_inventory for ref in pair)
                    or endpoint.get('joint_pixel_link') is not None
                    or view.get('output_ticks')!=endpoint.get('output_ticks')):
                raise ValueError('scene registered incident evidence lost original native samples/type/clock')
            expected_ticks=([h['span_key'][2]-4,h['span_key'][2]-1] if offset==0
                else [h['span_key'][1],h['span_key'][1]+3])
            if endpoint['output_ticks']!=expected_ticks:
                raise ValueError('scene registered incident evidence borrowed another endpoint side')
            validate_focal_pose_view(view,raw_inventory[pair[0]],raw_inventory[pair[1]],
                raw_inventory=raw_inventory,source_domain=dense['source_domain'],hypothesis=h,
                state_key=dense['state_keys'][state_index],original_duty_program=program)
    for a,b in ((keys[i],keys[j]) for i,j in pair_indices):
        if relation(dense.get('source_derivative_binding'),hypotheses[a]['span_key'][0],hypotheses[b]['span_key'][0])!=DERIVATIVE_DISTINCT:
            raise ValueError('scene cut lacks a bound distinct-camera pair; duplicate/UNKNOWN cannot count as diversity')
    minimum=BoundarySubjectAlignmentConfig().minimum_face_visible_fraction
    def fraction(value):
        return type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
    for span,key in zip(dense['mutable_shots'],[keys[i] for i in mutable_indices],strict=True):
        h=hypotheses[key]
        if (span['first_output_frame'],span['end_output_frame'],span['role'])!=(h['span_key'][1],h['span_key'][2],h['role']):
            raise ValueError('scene dense role/clock differs from frozen hypothesis')
        ticks=[r.get('output_tick') for r in span['rows']]
        if (any(type(t) is not int for t in ticks) or
                ticks!=list(range(span['first_output_frame'],span['end_output_frame']))):
            raise ValueError('scene dense omitted/duplicated a mutable output tick')
        previous_row=None
        for row in span['rows']:
            if (row.get('frame_admission_violations')
                    or row.get('role_geometry_constraints_satisfied',True) is not True):
                raise ValueError('scene dense cannot admit a measured frame with role/visibility rejection')
            if (row.get('scene_hypothesis_sha256')!=key or row.get('anatomical_occlusion_inferred') is not False
                    or row.get('role')!=h['role']):
                raise ValueError('scene dense row borrowed another hypothesis/anatomy claim')
            native=row.get('native_observation') or {}
            validate_serialized_detector_result(native)
            names=None
            if program is not None:
                preimage=dict(first=span['first_output_frame'],end=span['end_output_frame'],role=span['role'])
                ref=duty_at(program,preimage,row['output_tick'])
                if row.get('original_duty_reference')!=ref:
                    raise ValueError('scene dense changed its externally frozen original duty reference')
                if row['role'] in FOCAL:names=tuple(ref['required_part_names'])
            elif row.get('original_duty_reference') is not None:
                raise ValueError('scene dense lost its referenced original duty programme')
            obligations=scene_role_obligations(native,row['role'],row['selected_frame_local_head_id'],required_part_names=names)
            if family.plain(row.get('role_obligations'))!=family.plain(obligations.to_dict()):
                raise ValueError('scene dense required/context scope changed after native observation')
            original_heads=Counter(p['observation_id'] for p in native.get('heads',[]))
            original_people=Counter(p['observation_id'] for p in native.get('people',[]))
            if (any(n!=1 for n in (*original_heads.values(),*original_people.values()))
                    or not original_heads and row['role']!=GroupOutputRole.COMPOSITION.value
                    or original_heads != Counter(p['observation_id'] for p in row['scene_heads'])
                    or original_people != Counter(p['observation_id'] for p in row['scene_people'])):
                raise ValueError('scene dense omitted or duplicated an original head/body cohort member')
            if any(x.get('required_by_role')!=(x['observation_id'] in obligations.required_head_ids)
                    or x.get('physical_head_extent') is not None
                    or not fraction(x['visible_fraction'])
                    or x['observation_id'] in obligations.required_head_ids and x['visible_fraction']<minimum
                    for x in row['scene_heads']):
                raise ValueError('scene camera lost a head or invented physical head extent')
            for projected in (*row['scene_heads'],*row['scene_people']):
                box=projected.get('projected_box_xywh')
                if (not fraction(projected.get('visible_fraction'))
                        or not isinstance(box,(list,tuple)) or len(box)!=4 or min(box[2:])<=0):
                    raise ValueError('scene projected support has invalid finite visibility/geometry')
            for head in row['scene_heads']:
                if head['observation_id']==row['selected_frame_local_head_id'] and (
                        head.get('native_cropped') is not False
                        or not fraction(head.get('headroom_visible_fraction'))
                        or head['headroom_visible_fraction']<minimum):
                    raise ValueError('scene selected focal native face/headroom is unavailable')
            required_parts=set(obligations.required_parts)
            measured_parts=Counter((a['person_observation_id'],a['part']) for a in row['projected_observed_arms'])
            observed_parts=Counter((p['observation_id'],name) for p in native.get('people',[])
                for name,status in p.get('part_states',[]) if name in ('left_arm','right_arm') and status=='observed')
            if measured_parts!=observed_parts or not required_parts<=set(measured_parts):
                raise ValueError('scene dense omitted, duplicated or promoted original observed parts')
            for arm in row['projected_observed_arms']:
                required=(arm['person_observation_id'],arm['part']) in required_parts
                if arm['visible_fraction'] is not None and not fraction(arm['visible_fraction']):
                    raise ValueError('scene dense arm visibility is not a finite fraction')
                if arm.get('required_by_role')!=required or required and (
                        arm.get('measured_joints_visible') is not True or arm['visible_fraction'] is None
                        or arm['visible_fraction']<BoundarySubjectAlignmentConfig().minimum_performer_visible_fraction
                        or arm['native_action_envelope_xywh'] is None):
                    raise ValueError('scene dense lost required arm visibility or borrowed context')
                if required and (len(arm['projected_joint_xy'])!=3 or any(
                        len(point)!=2 or any(not fraction(v) for v in point)
                        for point in arm['projected_joint_xy'])):
                    raise ValueError('scene dense required arm joints are not measured visible coordinates')
            if any(p.get('required_by_role')!=(p['observation_id'] in obligations.required_person_ids)
                    for p in row['scene_people']):
                raise ValueError('scene dense required body scope changed')
            flattened=[]
            for person in row['scene_people']:
                original=next(p for p in native['people'] if p['observation_id']==person['observation_id'])
                if (person['part_states']!=dict(original.get('part_states',[])) or
                        person['unique_head_observation_id']!=original.get('unique_head_observation_id')
                        or not fraction(person['visible_fraction'])):
                    raise ValueError('scene dense altered original body ownership or visibility')
                if any(a['person_observation_id']!=person['observation_id'] for a in person['observed_arms']):
                    raise ValueError('scene dense borrowed another body observed-arm inventory')
                flattened.extend(person['observed_arms'])
            if Counter(family.digest(a) for a in flattened)!=Counter(family.digest(a) for a in row['projected_observed_arms']):
                raise ValueError('scene dense body and flat arm inventories disagree')
            previous_parts=() if previous_row is None else tuple(sorted({name for _,name in
                previous_row['role_obligations']['required_parts']}))
            current_parts=tuple(sorted({name for _,name in obligations.required_parts}))
            if row['role']==GroupOutputRole.FORMATION.value or previous_parts or current_parts:
                if row['output_tick']>span['first_output_frame'] and not row['native_repeated_alias']:
                    link=row.get('required_body_temporal_link')
                    if not required_motion_available(link):
                        raise ValueError('scene dense required arm temporal pixel producer missing')
                    if link.get('contract')==VIEW_CONTRACT:
                        raw_inventory.update({family.digest(previous_row['native_observation']):previous_row['native_observation'],
                            family.digest(native):native})
                        validate_focal_pose_view(link,previous_row['native_observation'],native,
                            raw_inventory=raw_inventory,source_domain=dense['source_domain'],hypothesis=h,
                            state_key=dense['state_keys'][dense['hypothesis_keys'].index(key)],
                            original_duty_program=program)
                    else:validate_required_body_native_sequence(link,previous_row['native_observation'],native)
                    expected_scope=('all_formation_people' if row['role']==GroupOutputRole.FORMATION.value
                        else 'selected_focal_owner')
                    if link.get('required_body_scope')!=expected_scope:
                        raise ValueError('scene dense temporal body evidence has another role scope')
                    if expected_scope=='selected_focal_owner':
                        required_names=sorted(set(previous_parts)|set(current_parts))
                        if (not set(previous_parts)<=set(current_parts)
                                or link.get('required_part_names')!=required_names):
                            raise ValueError('scene dense required arm lifecycle or pixel mask changed')
                        expected_ids=[[previous_row['role_obligations']['focal_owner_id']],[obligations.focal_owner_id]]
                        if link.get('required_body_observation_ids')!=expected_ids or None in (
                                previous_row['role_obligations']['focal_owner_id'],obligations.focal_owner_id):
                            raise ValueError('scene dense borrowed another body temporal path')
            if row['role']==GroupOutputRole.FORMATION.value:
                if row['raw_face_roi_xywh'] is not None or len(row['scene_people'])<2:
                    raise ValueError('formation was replaced by a fake single face')
                for person in row['scene_people']:
                    if person['visible_fraction']<BoundarySubjectAlignmentConfig().minimum_performer_visible_fraction or any(
                            person['part_states'].get(k)!='observed' for k in ('left_arm','right_arm')):
                        raise ValueError('scene formation loses required original body/arms')
            if row['role']==GroupOutputRole.COMPOSITION.value:
                validate_raw_composition(native)
                if any(row.get(k) is not None for k in ('selected_frame_local_head_id',
                        'raw_face_roi_xywh','raw_landmarks_xy','projected_landmarks_xy',
                        'projected_face_roi_xywh','face_roi_visible_fraction')) or row['projected_observed_arms']:
                    raise ValueError('composition row fabricated face or arm authority')
                if any(p['visible_fraction']<BoundarySubjectAlignmentConfig().minimum_performer_visible_fraction
                        for p in row['scene_people']):
                    raise ValueError('composition dense lost required native body support visibility')
                if previous_row is not None:
                    before=previous_row['native_observation']
                    repeated=before['native_frame_index']==native['native_frame_index']
                    if row.get('native_repeated_alias')!=repeated:
                        raise ValueError('composition repeated-native marker changed')
                    if repeated:
                        if before!=native or row.get('composition_support_link') is not None:
                            raise ValueError('native alias cannot manufacture composition motion')
                    else:
                        validate_support_link(row.get('composition_support_link'),before,native)
                elif row.get('composition_support_link') is not None:
                    raise ValueError('first composition row has no previous mutable native support')
                if row['output_tick'] in h['anchor_output_ticks']:
                    n=h['anchor_output_ticks'].index(row['output_tick'])
                    if (family.digest(native)!=h['raw_observation_sha256'][n]
                            or [p['observation_id'] for p in native['people']]!=h['body_anchor_ids'][n]):
                        raise ValueError('composition dense does not join the frozen endpoint hypothesis')
            previous_row=row
