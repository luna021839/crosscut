"""Bounded equal-song-time focal source ordering from owned native pixels.

This is a sensing/shortlist producer, not production admission. It never grants
named identity, whole-head visibility, required-part coverage or cut quality.
All target references are chosen before comparing source appearance/cameras;
ambiguous/missing rows remain unavailable rather than becoming a zero score.
"""
from dataclasses import dataclass
import hashlib
import math
from itertools import product
from pathlib import Path

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


def _same_source_links(session, rows, chosen):
    for left, right, before, after in zip(rows, rows[1:], chosen, chosen[1:]):
        if left.native_frame_index >= right.native_frame_index:
            return False
        clock = session.domain.render_frame_clocks[session.domain.eligible_index(left.request_source_index)]
        if any(left.native_frame_index < math.ceil(t*float(clock.frame_rate)-1e-9) <= right.native_frame_index
                for t in session.domain.cuts_for_request(left.request_source_index)):
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


def resolve_inherited_focal_reference(session, rows, appearances):
    """Seed before donor search, then keep that actual native face track.

    The target cannot switch to a different/larger face mid-gesture. A global
    area margin at EVERY frame would erase an already measured focal subject
    when a background face briefly grows; native appearance+pixel links, not
    subsequent screen centrality/dominance, resolve the retained reference.
    """
    seed = _dominant_owned_face(rows[0], appearances[0])
    if seed is None:
        return None
    chosen, info = match_focal_trajectory((seed,) * 3, appearances,
        path_available=lambda path: _same_source_links(session, rows, path))
    return chosen if chosen is not None and min(info['similarities']) >= .5 else None


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


def observe_synchronized_focal_frontier(session, *, reference_source, canonical_times, face_roi_only=False):
    """At most three rasters per eligible source; no full-song reanalysis.

    Reference motive must be inherited by the caller BEFORE this procedure.
    A global dominant face plus local paths can order candidates, not replace
    semantic duties or prove the complete B/C/neighbor camera receipt.
    """
    times = tuple(canonical_times)
    if (len(times) != 3 or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
            or not times[0] < times[1] < times[2] or times[-1]-times[0] > .35):
        raise ValueError('focal frontier requires three distinct bounded canonical music times')
    session.domain.eligible_index(reference_source)
    if type(face_roi_only) is not bool:
        raise ValueError('face-ROI-only sensing must be an explicit non-admission policy')
    observed = {}; appearances = {}; candidates = []
    # Declare reference first, not after seeing favorable donor/camera scores.
    sources = (reference_source, *(s for s in session.domain.eligible_indices if s != reference_source))
    for source in sources:
        rows = tuple(observe_native_group(session, request_source_index=source,
            native_frame_index=_native_for_canonical(session.domain, source, time)) for time in times)
        observed[source] = rows
        appearance_observer = observe_native_face_appearances if face_roi_only else observe_owned_focal_appearances
        appearances[source] = tuple(appearance_observer(session, row) for row in rows)
        if source == reference_source:
            reference = resolve_inherited_focal_reference(session, rows, appearances[source])
            if reference is None:
                return dict(contract='bounded_synchronized_owned_focal_source_frontier_v1',
                    status='inherited_focal_reference_unavailable', canonical_times=times,
                    candidates=[], raw_observations=[row.to_dict() for row in rows],
                    production_admitted=False, cross_source_identity_certified=False, quality_approved=False)
        # Resolve this donor while its three native rasters are in the byte
        # LRU. A second all-source pass would evict/redecode 4K sources just
        # to compare their already measured temporal paths.
        provisional, path_info = match_focal_trajectory(reference, appearances[source],
            path_available=lambda path: _same_source_links(session, observed[source], path))
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
        candidates.append(dict(source_index=source, shortlist_available=linked, reason=reason if linked else
            reason, similarities=ranking_similarities, competing_trajectory_margin=trajectory_margin,
            ranking_only_path_available=ranking_linked, ranking_similarities=ranking_similarities,
            native_trajectory_hypotheses=path_info['hypotheses'], source_eligibility_changed=False,
            anchors=[] if anchors is None else [dict(native_frame_index=row.native_frame_index,
                native_presentation_seconds=row.native_frame_index/float(session.domain.clocks_by_request[source].frame_rate),
                observation_sha256=row.sha256, person_observation_id=item.person_observation_id,
                candidate_body_observation_ids=(item.candidate_body_observation_ids if isinstance(item, NativeFaceAppearance)
                    else (item.person_observation_id,)),
                face_observation_id=item.face_observation_id,
                face_box_xywh=next(head.box_xywh for head in row.heads if head.observation_id == item.face_observation_id))
                for row, item in zip(observed[source], anchors)]))
    return dict(contract='bounded_synchronized_owned_focal_source_frontier_v1', status='measured_shortlist',
        canonical_times=times, reference_source_index=reference_source, candidates=candidates,
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
        cross_source_identity_certified=False, quality_approved=False)
