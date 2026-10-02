"""Bounded equal-song-time focal source ordering from owned native pixels.

This is a sensing/shortlist producer, not production admission. It never grants
named identity, whole-head visibility, required-part coverage or cut quality.
All target references are chosen before comparing source appearance/cameras;
ambiguous/missing rows remain unavailable rather than becoming a zero score.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from .face_identity import SFaceEmbedder, SFACE_SHA256, _validate_normalized_face
from .group_joint_native_observer import observe_native_group
from .group_joint_native_joint_flow import native_joint_pixel_flows, native_joint_assignment_cost
from .group_focal_attention import focal_landmarks_track
from .boundary_subject_alignment import _optical_flow_box
from .cut_anchored_containment import _iou
from .group_joint_person_ownership import (OwnedPersonObservation, ownership_policy_fingerprint,
    trackable_person_ownership, bind_person_ownership)
from .render import aligned_render_segment
from .render_frame_schedule import sampler_frame_schedule
from .visual_score import _working_frame


def locally_associated_face_roi_target(before_heads, after_heads, reference):
    """Existing local recovery geometry, BEFORE counted appearance sensing.

    The .7/.55 IoU, .08 displacement and .85..1.18 scale limits are the normal
    nearby-face recovery policy, not a new cross-camera identity threshold.
    All competing freshly detected faces take part in uniqueness checks.
    """
    def unique_near(heads, minimum):
        ranked = sorted(((_iou(head.box_xywh, reference.box_xywh), head) for head in heads),
            key=lambda item: item[0], reverse=True)
        if (not ranked or ranked[0][0] < minimum or len(ranked) > 1
                and ranked[1][0] >= .35 and ranked[0][0]-ranked[1][0] < .1):
            return None
        return ranked[0][1]
    for head in (*before_heads, *after_heads, reference):
        box = head.box_xywh
        if (min(box[:2]) < 0 or box[0]+box[2] > 1.000001 or box[1]+box[3] > 1.000001):
            return None
    left, right = unique_near(before_heads, .7), unique_near(after_heads, .55)
    if left is None or left.observation_id != reference.observation_id or right is None:
        return None
    a, b = reference.box_xywh, right.box_xywh
    shift = math.hypot(b[0]+b[2]/2-a[0]-a[2]/2, b[1]+b[3]/2-a[1]-a[3]/2)
    if shift > .08 or any(not .85 <= b[n]/a[n] <= 1.18 for n in (2, 3)):
        return None
    return right


@dataclass(frozen=True, slots=True)
class OwnedFocalAppearance:
    person_observation_id: int
    face_observation_id: int
    embedding: tuple[float, ...]

    def __post_init__(self):
        if (any(type(n) is not int or n < 0 for n in (self.person_observation_id, self.face_observation_id))
                or not isinstance(self.embedding, tuple) or len(self.embedding) != 128
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.embedding)
                or abs(float(np.linalg.norm(self.embedding))-1.) > 1e-5):
            raise ValueError('focal appearance needs an actual normalized owned face embedding')


@dataclass(frozen=True, slots=True)
class NativeFaceAppearance:
    """A detected face keeps ALL body alternatives; none is a chosen owner."""
    face_observation_id: int
    embedding: tuple[float, ...]
    candidate_body_observation_ids: tuple[int, ...]

    def __post_init__(self):
        OwnedFocalAppearance(0, self.face_observation_id, self.embedding)
        if (not isinstance(self.candidate_body_observation_ids, tuple)
                or len(set(self.candidate_body_observation_ids)) != len(self.candidate_body_observation_ids)
                or any(type(n) is not int or n < 0 for n in self.candidate_body_observation_ids)):
            raise ValueError('face appearance must retain all explicit raw body alternatives')

    @property
    def person_observation_id(self):
        return None  # a native face track does not resolve body/arm ownership


@dataclass(frozen=True, slots=True)
class InheritedFocalReference:
    """Parent-frozen semantic seed; not an appearance-derived identity claim.

    The normal parent still owns live authority validation. This discovery
    helper only checks that its first raw native observation is the one the
    parent named; it cannot infer or certify the semantic reference itself.
    """
    semantic_reference: str
    parent_authority_id: str
    request_source_index: int
    native_frame_index: int
    observation_sha256: str
    face_observation_id: int

    def __post_init__(self):
        if (any(not isinstance(value, str) or not value.strip() or len(value) > 512
                for value in (self.semantic_reference, self.parent_authority_id))
                or any(type(value) is not int or value < 0 for value in (
                    self.request_source_index, self.native_frame_index, self.face_observation_id))
                or not isinstance(self.observation_sha256, str) or len(self.observation_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.observation_sha256)):
            raise ValueError('inherited focal reference needs parent authority and an exact raw native seed')

    def to_dict(self):
        return asdict(self)


def register_owned_focal_appearance_model(session, model_path):
    """SFace feature calls use the SAME owner budget, including each face ROI."""
    path = Path(model_path)
    session.budget.check()
    if hashlib.sha256(path.read_bytes()).hexdigest() != SFACE_SHA256:
        raise ValueError('focal appearance model differs from its pinned asset')
    embedder = SFaceEmbedder(path)
    session.register_model('sface_owned_face', path, embedder.embedding, adapter_fingerprint={
        'contract': 'source_native_owned_yunet_face_sface_appearance_not_identity_v1',
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'embedder_sha256': hashlib.sha256(Path(__file__).with_name('face_identity.py').read_bytes()).hexdigest(),
        'opencv_version': cv2.__version__, 'native_normalized_geometry_not_clamped': True})


def observe_owned_focal_appearances(session, row):
    if row.person_ownership_policy_sha256 != ownership_policy_fingerprint():
        raise ValueError('focal ordering cannot upgrade stale/legacy part ownership')
    rebound, owners = bind_person_ownership(row.people, row.heads, row.frame_size_wh)
    if rebound != row.people or any(head.candidate_body_observation_ids != tuple(owners[head.observation_id])
            for head in row.heads):
        raise ValueError('focal appearance raw part ownership does not independently replay')
    heads = {head.observation_id: head for head in row.heads}
    values = []
    for person in row.people:
        if (not isinstance(person, OwnedPersonObservation) or not trackable_person_ownership(person)
                or person.unique_head_observation_id is None):
            continue
        head = heads[person.unique_head_observation_id]
        if head.candidate_body_observation_ids != (person.observation_id,):
            raise ValueError('focal appearance cannot borrow another body face owner')
        try:
            _validate_normalized_face(head.box_xywh, head.landmarks_xy)
        except ValueError:
            # Off-native face/landmark geometry is absent, never clamped or
            # substituted with a neighboring/background face.
            continue
        embedding = session.observe_native(row.request_source_index, row.native_frame_index,
            model='sface_owned_face', arguments={'bbox_xywh': list(head.box_xywh),
                'landmarks_xy': [list(point) for point in head.landmarks_xy]})
        if embedding is not None:
            values.append(OwnedFocalAppearance(person.observation_id, head.observation_id, tuple(embedding)))
    return tuple(values)


def observe_native_face_appearances(session, row):
    """Body duplicates do not erase a separately detected native face.

    This is HEAD-ROI sensing, not changing the inherited editorial role to
    HEAD or relaxing any required body/arm duty. Ambiguous body rows stay in
    the result, and complete required-part coverage is never granted here.
    """
    if row.person_ownership_policy_sha256 != ownership_policy_fingerprint():
        raise ValueError('native face ordering cannot upgrade stale part ownership')
    values = []
    for head in row.heads:
        try:
            _validate_normalized_face(head.box_xywh, head.landmarks_xy)
        except ValueError:
            continue
        embedding = session.observe_native(row.request_source_index, row.native_frame_index,
            model='sface_owned_face', arguments={'bbox_xywh': list(head.box_xywh),
                'landmarks_xy': [list(point) for point in head.landmarks_xy]})
        if embedding is not None:
            values.append(NativeFaceAppearance(head.observation_id, tuple(embedding),
                head.candidate_body_observation_ids))
    return tuple(values)


def _native_for_canonical(domain, source, canonical):
    local = domain.eligible_index(source)
    clock = domain.render_frame_clocks[local]
    segment = aligned_render_segment(source_index=source, reference_start=canonical,
        common_master_start=canonical, duration=1/30., source_offset=domain.audio_offsets[local],
        timewarp=domain.source_timewarps[local])
    schedule = sampler_frame_schedule(source_start=segment.source_start,
        source_playback_rate=segment.source_playback_rate, native_fps=float(clock.frame_rate),
        output_fps=30., frame_count=1, source_frame_count=clock.frame_count,
        maximum_terminal_hold_frames=0)
    return schedule.source_frame_indices[0]


def _dominant_owned_face(row, appearances):
    # Preserve the existing global 1.8 area margin. Not the largest BODY box,
    # screen-center prior, successful crop or candidate-dependent target.
    heads = sorted(row.heads, key=lambda h: h.box_xywh[2]*h.box_xywh[3], reverse=True)
    if not heads or len(heads) > 1 and heads[0].box_xywh[2]*heads[0].box_xywh[3] < 1.8*heads[1].box_xywh[2]*heads[1].box_xywh[3]:
        return None
    values = tuple(item for item in appearances if item.face_observation_id == heads[0].observation_id)
    return values[0] if len(values) == 1 else None


def _native_pair_is_continuous(session, left, right):
    if (left.request_source_index != right.request_source_index
            or left.native_frame_index >= right.native_frame_index):
        return False
    clock = session.domain.render_frame_clocks[session.domain.eligible_index(left.request_source_index)]
    return not any(left.native_frame_index < math.ceil(t*float(clock.frame_rate)-1e-9) <= right.native_frame_index
        for t in session.domain.cuts_for_request(left.request_source_index))


class NativeFocalLinkGraph:
    """Window-local pair sensing, never a semantic/part/identity receipt.

    Build adjacent pairs as each row arrives, while at most two native
    rasters are needed. All face/body alternatives are measured once, then
    triplet ranking reads this graph without any decode/resize/flow calls.
    Cached edges include raw-row identities; false/unknown edges are cached
    too. Raw LK feature ownership still requires the separate part producer.
    """
    def __init__(self, session, *, maximum_pair_observations=128,
            maximum_analysis_pair_bytes=16*1024*1024):
        if any(type(value) is not int or value < 1 for value in (
                maximum_pair_observations, maximum_analysis_pair_bytes)):
            raise ValueError('native focal graph needs explicit finite pair and analysis budgets')
        self.session = session
        self.maximum_pairs = maximum_pair_observations
        self.maximum_analysis_bytes = maximum_analysis_pair_bytes
        self.pairs = {}
        self.unique_pairs = self.lookups = self.peak_analysis_bytes = self.peak_raw_pair_bytes = 0
        self.pair_sensing_seconds = 0.

    @staticmethod
    def _appearance_key(item):
        if not isinstance(item, (NativeFaceAppearance, OwnedFocalAppearance)):
            raise ValueError('native focal graph requires explicit face/body appearance alternatives')
        return (type(item).__name__, item.face_observation_id, item.person_observation_id)

    def _assert_row_binding(self, row):
        identity = self.session._assert_source_identity(row.request_source_index)
        clock = self.session.domain.render_frame_clocks[self.session.domain.eligible_index(row.request_source_index)]
        clock_sha = hashlib.sha256(json.dumps(clock.to_dict(), sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        if identity != row.source_file_identity or clock_sha != row.source_clock_identity:
            raise ValueError('native focal graph raw row source/file/clock binding changed')
        for name, digest in row.model_content_sha256:
            self.session._assert_model_assets(name)
            if self.session.models[name][2] != digest:
                raise ValueError('native focal graph raw model binding changed')
        for name, digest in row.model_adapter_fingerprints:
            if self.session.model_adapters[name] != digest:
                raise ValueError('native focal graph raw adapter binding changed')

    def add_pair(self, left, right, before_values, after_values):
        self.session.budget.check()
        self._assert_row_binding(left); self._assert_row_binding(right)
        key = (left.sha256, right.sha256)
        if key in self.pairs:
            raise ValueError('native focal graph pair may be measured only once per window')
        alternatives = tuple(product(before_values, after_values))
        if self.unique_pairs+len(alternatives) > self.maximum_pairs:
            raise ValueError('native focal graph pair overflow, not success-biased truncation')
        started = monotonic()
        observed = {}
        continuous = _native_pair_is_continuous(self.session, left, right)
        if not alternatives or not continuous:
            self.pairs[key] = observed
            return
        pixels_before = self.session.native_pixels(left.request_source_index, left.native_frame_index)
        pixels_after = self.session.native_pixels(right.request_source_index, right.native_frame_index)
        if pixels_before is None or pixels_after is None:
            self.pairs[key] = observed
            return
        if (hashlib.sha256(memoryview(pixels_before)).hexdigest() != left.native_pixel_sha256
                or hashlib.sha256(memoryview(pixels_after)).hexdigest() != right.native_pixel_sha256
                or pixels_before.shape != pixels_after.shape
                or pixels_before.shape[1::-1] != left.frame_size_wh
                or pixels_after.shape[1::-1] != right.frame_size_wh):
            raise ValueError('native focal graph pixels differ from the measured raw rows')
        self.peak_raw_pair_bytes = max(self.peak_raw_pair_bytes, pixels_before.nbytes+pixels_after.nbytes)
        tracking_before = _working_frame(pixels_before, maximum_width=960)
        tracking_after = _working_frame(pixels_after, maximum_width=960)
        analysis_bytes = tracking_before.nbytes+tracking_after.nbytes
        if analysis_bytes > self.maximum_analysis_bytes:
            raise ValueError('native focal graph analysis pair exceeds its byte budget')
        self.peak_analysis_bytes = max(self.peak_analysis_bytes, analysis_bytes)
        before_heads = {head.observation_id: head for head in left.heads}
        after_heads = {head.observation_id: head for head in right.heads}
        flows = None
        for before, after in alternatives:
            self.session.budget.check()
            self.unique_pairs += 1
            pair = (self._appearance_key(before), self._appearance_key(after))
            if pair in observed:
                raise ValueError('duplicate native appearance IDs cannot add competing pair evidence')
            basis = None
            if isinstance(before, NativeFaceAppearance) and isinstance(after, NativeFaceAppearance):
                a, b = before_heads[before.face_observation_id], after_heads[after.face_observation_id]
                if focal_landmarks_track(tracking_before, tracking_after, a.landmarks_xy, b.landmarks_xy):
                    basis = 'native_face_landmark_path_not_whole_head_or_part_ownership'
                else:
                    box = _optical_flow_box(tracking_before, tracking_after, a.box_xywh)
                    if box is not None and _iou(box, b.box_xywh) >= .60:
                        basis = 'native_face_box_path_not_whole_head_or_part_ownership'
            elif isinstance(before, OwnedFocalAppearance) and isinstance(after, OwnedFocalAppearance):
                if flows is None:
                    flows = dict(zip((p.observation_id for p in left.people),
                        native_joint_pixel_flows(pixels_before, pixels_after, left.people), strict=True))
                target = next(p for p in right.people if p.observation_id == after.person_observation_id)
                flow = flows[before.person_observation_id]
                cost = native_joint_assignment_cost(flow, target, right.frame_size_wh)
                alternatives_cost = [native_joint_assignment_cost(flow, p, right.frame_size_wh)
                    for p in right.people if p.observation_id != target.observation_id]
                if cost is not None and cost <= .2 and not any(
                        value is not None and value <= cost+.02 for value in alternatives_cost):
                    basis = 'native_joint_patch_path_not_owned_visible_surface'
            observed[pair] = basis
        self.pairs[key] = observed
        self.pair_sensing_seconds += monotonic()-started

    def path_available(self, rows, chosen):
        self.session.budget.check()
        if len(rows) != len(chosen) or len(rows) < 2:
            raise ValueError('native focal graph path must name every measured anchor')
        for row in rows:
            self._assert_row_binding(row)
        for left, right, before, after in zip(rows, rows[1:], chosen, chosen[1:]):
            self.lookups += 1
            values = self.pairs.get((left.sha256, right.sha256))
            if values is None or values.get((self._appearance_key(before), self._appearance_key(after))) is None:
                return False
        return True

    def to_dict(self):
        return dict(contract='window_local_native_focal_pair_discovery_graph_v1',
            unique_pair_observations=self.unique_pairs, cached_pair_lookups=self.lookups,
            peak_raw_pair_raster_bytes=self.peak_raw_pair_bytes,
            peak_analysis_pair_raster_bytes=self.peak_analysis_bytes,
            pair_sensing_seconds=self.pair_sensing_seconds,
            pixel_reads_during_path_ranking=0, model_calls_during_path_ranking=0,
            part_feature_ownership_verified=False, quality_approved=False)


def _same_source_links(session, rows, chosen):
    for left, right, before, after in zip(rows, rows[1:], chosen, chosen[1:]):
        if not _native_pair_is_continuous(session, left, right):
            return False
        pixels_before = session.native_pixels(left.request_source_index, left.native_frame_index)
        pixels_after = session.native_pixels(right.request_source_index, right.native_frame_index)
        if isinstance(before, NativeFaceAppearance) and isinstance(after, NativeFaceAppearance):
            head_before = next(head for head in left.heads if head.observation_id == before.face_observation_id)
            head_after = next(head for head in right.heads if head.observation_id == after.face_observation_id)
            # Use the native detector's 960px analysis raster. The fixed LK
            # window/FB-error units are analysis pixels, not arbitrary 4K/8K
            # source pixels. Normalized landmark coordinates are unchanged.
            tracking_before = _working_frame(pixels_before, maximum_width=960)
            tracking_after = _working_frame(pixels_after, maximum_width=960)
            if not focal_landmarks_track(tracking_before, tracking_after,
                    head_before.landmarks_xy, head_after.landmarks_xy):
                # Same existing head-only pixel-path alternative as the normal
                # projected motion camera. Not a rigid WHOLE BODY identity fit.
                box = _optical_flow_box(tracking_before, tracking_after, head_before.box_xywh)
                if box is None or _iou(box, head_after.box_xywh) < .60:
                    return False
            continue  # face-local path does not grant arm/whole-head motion
        people = tuple(p for p in left.people if p.observation_id == before.person_observation_id)
        target = next(p for p in right.people if p.observation_id == after.person_observation_id)
        flow = native_joint_pixel_flows(pixels_before, pixels_after, people)[0]
        cost = native_joint_assignment_cost(flow, target, right.frame_size_wh)
        if cost is None or cost > .2:
            return False
        # An equally plausible competing current person must not be silently
        # discarded just because appearance ordering picked one of them.
        alternatives = [native_joint_assignment_cost(flow, p, right.frame_size_wh)
            for p in right.people if p.observation_id != target.observation_id]
        if any(value is not None and value <= cost+.02 for value in alternatives):
            return False
    return True


def match_focal_appearance_anchors(reference, alternatives, *, minimum_similarity=.5, minimum_margin=.06):
    """Shortlist only. Policies are NOT calibrated identity/quality confidence."""
    if len(reference) != 3 or len(alternatives) != 3:
        raise ValueError('focal source ordering needs exactly three synchronized anchors')
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
            for v in (minimum_similarity, minimum_margin)):
        raise ValueError('focal appearance ordering policy must be finite and explicit')
    chosen, similarities, margins = [], [], []
    for ref, values in zip(reference, alternatives):
        if ref is None or not values:
            return None, 'missing_owned_focal_appearance', (), ()
        ranked = sorted(((float(np.dot(ref.embedding, item.embedding)), item) for item in values),
            key=lambda pair: pair[0], reverse=True)
        similarity = ranked[0][0]
        margin = similarity-ranked[1][0] if len(ranked) > 1 else None
        similarities.append(similarity); margins.append(margin)
        if similarity < minimum_similarity or margin is not None and margin < minimum_margin:
            return None, 'appearance_missing_or_ambiguous_not_identity', tuple(similarities), tuple(margins)
        chosen.append(ranked[0][1])
    return tuple(chosen), 'appearance_shortlist_only', tuple(similarities), tuple(margins)


def _resolve_seeded_focal_reference(session, rows, appearances, seed, *, pixel_path_available=None):
    if seed is None:
        return None
    chosen, info = match_focal_trajectory((seed,) * 3, appearances,
        path_available=lambda path: path[0].face_observation_id == seed.face_observation_id
            and (pixel_path_available(path) if pixel_path_available is not None else _same_source_links(session, rows, path)))
    return chosen if chosen is not None and min(info['similarities']) >= .5 else None


def resolve_diagnostic_focal_reference(session, rows, appearances, *, pixel_path_available=None):
    """Diagnostic area seed before donor search, not an inherited subject.

    The target cannot switch to a different/larger face mid-gesture. A global
    area margin at EVERY frame would erase an already measured focal subject
    when a background face briefly grows; native appearance+pixel links, not
    subsequent screen centrality/dominance, resolve the retained reference.
    """
    return _resolve_seeded_focal_reference(session, rows, appearances,
        _dominant_owned_face(rows[0], appearances[0]), pixel_path_available=pixel_path_available)


def resolve_inherited_focal_reference(session, rows, appearances, *, inherited_reference,
        pixel_path_available=None):
    """Bind the caller's frozen subject, never reselect by face size/score.

    A missing/stale native seed is unavailable even if a convenient larger
    face is detected. The return value remains discovery, not identity or a
    whole-shot required-part receipt.
    """
    if not isinstance(inherited_reference, InheritedFocalReference):
        raise ValueError('inherited focal discovery requires an explicit parent-frozen reference')
    if len(rows) != 3 or len(appearances) != 3:
        raise ValueError('inherited focal discovery requires three native anchors')
    first = rows[0]
    if (first.request_source_index != inherited_reference.request_source_index
            or first.native_frame_index != inherited_reference.native_frame_index
            or first.sha256 != inherited_reference.observation_sha256):
        return None
    if session is not None:
        session.domain.eligible_index(inherited_reference.request_source_index)
    seeds = tuple(item for item in appearances[0]
        if item.face_observation_id == inherited_reference.face_observation_id)
    heads = tuple(head for head in first.heads
        if head.observation_id == inherited_reference.face_observation_id)
    if len(seeds) != 1 or len(heads) != 1:
        return None
    return _resolve_seeded_focal_reference(session, rows, appearances, seeds[0],
        pixel_path_available=pixel_path_available)


def match_focal_trajectory(reference, alternatives, *, path_available, maximum_hypotheses=64):
    """Resolve local native trajectories BEFORE cross-source appearance rank.

    Every measured path is retained, including competing paths. Independent
    frame-wise argmax can switch people during a gesture and must not become
    persistent target correspondence. An overflow is explicit unavailable.
    """
    if len(reference) != 3 or len(alternatives) != 3 or not callable(path_available):
        raise ValueError('focal trajectory requires three anchors and actual pixel-path sensing')
    if type(maximum_hypotheses) is not int or not 1 <= maximum_hypotheses <= 256:
        raise ValueError('focal trajectory hypothesis cap must be bounded and explicit')
    count = math.prod(len(values) for values in alternatives)
    if not count or any(item is None for item in reference):
        return None, dict(reason='missing_native_face_path', similarities=(), hypotheses=[])
    if count > maximum_hypotheses:
        return None, dict(reason='native_face_hypothesis_overflow_not_truncated', similarities=(),
            hypothesis_count=count, hypotheses=[])
    paths = []
    for path in product(*alternatives):
        if not path_available(path):
            continue
        similarities = tuple(float(np.dot(a.embedding, b.embedding)) for a, b in zip(reference, path))
        paths.append((float(np.mean(similarities)), similarities, path))
    paths.sort(key=lambda row: row[0], reverse=True)
    if not paths:
        return None, dict(reason='native_face_paths_unavailable', similarities=(), hypotheses=[])
    hypotheses = [dict(face_observation_ids=[item.face_observation_id for item in path],
        similarities=similarities, mean_similarity=score) for score, similarities, path in paths]
    score, similarities, chosen = paths[0]
    return chosen, dict(reason='native_trajectory_appearance_rank_only', similarities=similarities,
        mean_similarity=score, competing_trajectory_margin=score-paths[1][0] if len(paths)>1 else None,
        hypotheses=hypotheses, identity_certified=False, required_part_coverage_certified=False)


def native_pose_hypotheses_for_face_paths(*, domain, rows, face_paths, duties, maximum_hypotheses=64):
    """Measure ALL raw body alternatives of supplied face paths, independently of LK.

    The face path remains discovery, not whole-head/arm or named identity. An
    ambiguous body can have measurable geometry without becoming visible duty.
    No successful camera, lowest pose cost or LK survivor chooses a body owner.
    """
    from .group_joint_temporal_pose import native_pose_windows_from_rows, TemporalPoseUnavailable
    if (not isinstance(face_paths, tuple) or type(maximum_hypotheses) is not int
            or not 1 <= maximum_hypotheses <= 256):
        raise ValueError('native face/body motion needs an immutable bounded face-path inventory')
    inputs = []; count = 0
    for faces in face_paths:
        if (not isinstance(faces, tuple) or len(faces) != len(rows)
                or any(type(n) is not int or n < 0 for n in faces)):
            raise ValueError('temporal face path must identify every actual raw anchor')
        choices = []
        for row, face in zip(rows, faces, strict=True):
            head = next((h for h in row.heads if h.observation_id == face), None)
            if head is None:
                raise ValueError('temporal face path names an absent raw face')
            choices.append(head.candidate_body_observation_ids or (None,))
        count += math.prod(len(values) for n, values in enumerate(choices)
            if n == 0 or rows[n].native_frame_index != rows[n-1].native_frame_index)
        inputs.append((faces, tuple(choices)))
    if count > maximum_hypotheses:
        raise TemporalPoseUnavailable('all-face/body pose hypothesis overflow, not success-biased truncation')
    return tuple((faces, window) for faces, choices in inputs for window in
        native_pose_windows_from_rows(domain=domain, rows=rows, choices=choices, duties=duties,
            maximum_hypotheses=maximum_hypotheses))


def observe_synchronized_focal_frontier(session, *, reference_source, canonical_times, face_roi_only=False,
        inherited_reference=None):
    """At most three rasters per eligible source; no full-song reanalysis.

    The normal caller supplies its parent-frozen reference BEFORE discovery.
    Without it, area-based seeding is explicitly diagnostic, never labelled
    an inherited subject. Neither path proves a dense final render schedule.
    """
    times = tuple(canonical_times)
    if (len(times) != 3 or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
            or not times[0] < times[1] < times[2] or times[-1]-times[0] > .35):
        raise ValueError('focal frontier requires three distinct bounded canonical music times')
    session.domain.eligible_index(reference_source)
    if type(face_roi_only) is not bool:
        raise ValueError('face-ROI-only sensing must be an explicit non-admission policy')
    if inherited_reference is not None and (not isinstance(inherited_reference, InheritedFocalReference)
            or inherited_reference.request_source_index != reference_source):
        raise ValueError('focal discovery reference differs from the predeclared source')
    reference_basis = ('parent_frozen_raw_native_seed_not_identity_certification'
        if inherited_reference is not None else 'diagnostic_area_seed_not_inherited_subject')
    from .group_joint_temporal_pose import (FrozenPoseMotionDutiesV1,
        TemporalPoseUnavailable, compare_pose_phase_windows, independent_native_pose_hypotheses)
    # Fixed SENSOR obligations before source ordering, not a narrower
    # editorial head-only role derived from failing body/arm measurements.
    pose_duties = FrozenPoseMotionDutiesV1('native-focal-discovery-head-and-arm-sensor-not-semantic-parent',
        (0, 5, 6, 7, 8, 9, 10), .3)
    observed = {}; appearances = {}; candidates = []; graphs = []; pose_windows = {}
    # Declare reference first, not after seeing favorable donor/camera scores.
    sources = (reference_source, *(s for s in session.domain.eligible_indices if s != reference_source))
    for source in sources:
        rows, values = [], []
        graph = NativeFocalLinkGraph(session)
        appearance_observer = observe_native_face_appearances if face_roi_only else observe_owned_focal_appearances
        for time in times:
            session.retain_native_pixels(((source,rows[-1].native_frame_index),) if rows else ())
            row = observe_native_group(session, request_source_index=source,
                native_frame_index=_native_for_canonical(session.domain, source, time))
            rows.append(row); values.append(appearance_observer(session, row))
            # Stream a pair BEFORE decoding the third raster. An overflow
            # remains the trajectory selector's explicit unavailable result.
            if len(rows) > 1 and math.prod(len(items) for items in values) <= 64:
                graph.add_pair(rows[-2], rows[-1], values[-2], values[-1])
        rows, values = tuple(rows), tuple(values)
        observed[source] = rows; appearances[source] = values
        pixel_path_available = lambda path: graph.path_available(rows, path)
        if source == reference_source:
            reference = (resolve_diagnostic_focal_reference(session, rows, appearances[source],
                    pixel_path_available=pixel_path_available)
                if inherited_reference is None else resolve_inherited_focal_reference(session, rows,
                    appearances[source], inherited_reference=inherited_reference,
                    pixel_path_available=pixel_path_available))
            if reference is None:
                return dict(contract='bounded_synchronized_owned_focal_source_frontier_v1',
                    status='inherited_focal_reference_unavailable', canonical_times=times,
                    reference_basis=reference_basis,
                    candidates=[], raw_observations=[row.to_dict() for row in rows],
                    production_admitted=False, cross_source_identity_certified=False, quality_approved=False)
        # Only the already sensed pair graph is needed now. Three 4K native
        # rasters need not fit together in the default 64MiB raw LRU.
        provisional, path_info = match_focal_trajectory(reference, appearances[source],
            path_available=lambda path: (source != reference_source
                or path[0].face_observation_id == reference[0].face_observation_id)
                and pixel_path_available(path))
        ranking_similarities = path_info['similarities']
        trajectory_margin = path_info.get('competing_trajectory_margin')
        linked = (provisional is not None and min(ranking_similarities) >= .5
            and (trajectory_margin is None or trajectory_margin >= .06))
        reason = 'appearance_shortlist_only' if linked else path_info['reason']
        # Candidate ordering is not identity admission. Retain weak/ambiguous
        # appearance as a provisional hypothesis rather than permanently
        # excluding a source using an uncalibrated bank threshold. Required
        # parts/all competing hypotheses are still the normal owner's duty.
        ranking_linked = provisional is not None
        anchors = provisional
        session.budget.check()
        try:
            seed_body_ids = None
            if source == reference_source:
                seed = next(head for head in rows[0].heads if head.observation_id == reference[0].face_observation_id)
                seed_body_ids = seed.candidate_body_observation_ids or (None,)
            pose_windows[source] = tuple(((), window) for window in independent_native_pose_hypotheses(
                domain=session.domain,rows=rows,duties=pose_duties,initial_person_ids=seed_body_ids))
            pose_status = ('raw_temporal_hypotheses_measured_not_association_or_phase_certification'
                if pose_windows[source] else 'no_eligible_native_face_path_for_temporal_body_hypotheses')
        except TemporalPoseUnavailable as exc:
            pose_windows[source] = ()
            pose_status = str(exc)
        session.budget.check()
        pose_payload = [dict(face_observation_ids=faces, track=window.to_dict())
            for faces, window in pose_windows[source]]
        session.budget.spend('pose_evidence_bytes',len(json.dumps(pose_payload,sort_keys=True,
            separators=(',', ':'),allow_nan=False).encode()))
        candidates.append(dict(source_index=source, shortlist_available=linked, reason=reason if linked else
            reason, similarities=ranking_similarities, competing_trajectory_margin=trajectory_margin,
            ranking_only_path_available=ranking_linked, ranking_similarities=ranking_similarities,
            native_trajectory_hypotheses=path_info['hypotheses'], source_eligibility_changed=False,
            independent_native_pose_motion_status=pose_status,
            independent_native_pose_motion_association_basis='all_raw_person_alternatives_not_conditioned_on_face_or_joint_lk',
            independent_native_pose_motion=pose_payload,
            anchors=[] if anchors is None else [dict(native_frame_index=row.native_frame_index,
                native_presentation_seconds=row.native_frame_index/float(session.domain.clocks_by_request[source].frame_rate),
                observation_sha256=row.sha256, person_observation_id=item.person_observation_id,
                candidate_body_observation_ids=(item.candidate_body_observation_ids if isinstance(item, NativeFaceAppearance)
                    else (item.person_observation_id,)),
                face_observation_id=item.face_observation_id,
                face_box_xywh=next(head.box_xywh for head in row.heads if head.observation_id == item.face_observation_id))
                for row, item in zip(observed[source], anchors)]))
        graphs.append(dict(source_index=source, **graph.to_dict()))
    comparisons = []
    for candidate in candidates:
        source = candidate['source_index']
        session.budget.spend('pose_comparisons', len(pose_windows[reference_source])*len(pose_windows[source]))
        pairs = product(pose_windows[reference_source], pose_windows[source])
        comparison = dict(source_index=source, all_owner_pair_hypotheses=[dict(
            reference_face_observation_ids=a[0], candidate_face_observation_ids=b[0],
            evidence=compare_pose_phase_windows(a[1], b[1])) for a, b in pairs],
            empty_is_zero_cost=False, minimum_over_convenient_owner_used=False,
            source_eligibility_or_appearance_ranking_changed=False)
        session.budget.spend('pose_evidence_bytes',len(json.dumps(comparison,sort_keys=True,
            separators=(',', ':'),allow_nan=False).encode()))
        comparisons.append(comparison)
        session.budget.check()
    return dict(contract='bounded_synchronized_owned_focal_source_frontier_v1', status='measured_shortlist',
        canonical_times=times, reference_source_index=reference_source, candidates=candidates,
        reference_basis=reference_basis,
        inherited_reference=None if inherited_reference is None else inherited_reference.to_dict(),
        source_local_pair_graphs=graphs,
        independent_raw17_temporal_pose_comparisons=comparisons,
        raw_observations=[row.to_dict() for rows in observed.values() for row in rows],
        target_declared_before_donor_comparison=True, evidence_basis=(
            'native_face_roi_appearance_and_local_landmark_paths_not_body_ownership' if face_roi_only else
            'native_owned_face_appearance_and_local_joint_paths'),
        face_roi_only=face_roi_only, inherited_role_or_required_parts_downgraded=False,
        source_local_tracking_policy=dict(face_raster_maximum_width=960,
            implementation_sha256=hashlib.sha256(Path(__file__).with_name('group_focal_attention.py').read_bytes()).hexdigest(),
            basis='four_of_five_landmark_flow_or_actual_face_box_flow_iou_060_not_whole_head_motion'),
        appearance_policy=dict(minimum_similarity=.5, minimum_margin=.06, calibrated_identity_confidence=False),
        source_clock_authority=session.domain.to_dict(), production_admitted=False,
        whole_head_extent_verified=False, full_required_part_coverage_verified=False,
        final_render_schedule_join_verified=False,
        cross_source_identity_certified=False, quality_approved=False)
