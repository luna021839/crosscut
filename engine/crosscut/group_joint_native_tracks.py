"""Source-native person tracklets, independent of searched virtual cameras.

Only measured one-to-one pixel-flow links carry a persistent local reference.
Ambiguous links retain their alternatives and required tracks retain missing
rows. This is not named identity, cross-camera phase or an editing approval.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boundary_subject_alignment import _optical_flow_box
from .cut_anchored_containment import _iou
from .group_joint_native_observer import NativeGroupObservation, observe_native_group
from .group_joint_native_joint_flow import native_joint_pixel_flows, native_joint_assignment_cost
from .group_joint_person_ownership import trackable_person_ownership


@dataclass(frozen=True, slots=True)
class NativePersonTrackPoint:
    native_frame_index: int
    observation_sha256: str
    observation_id: int | None
    basis: str
    predicted_box_xywh: tuple | None = None
    observed_iou: float | None = None
    alternative_observation_ids: tuple[int, ...] = ()
    measured_joint_pixel_flow: tuple = ()
    joint_assignment_cost: float | None = None
    lifecycle_state: str = 'tentative'


@dataclass(frozen=True, slots=True)
class NativePersonTrack:
    reference: str
    initial_observation_id: int
    points: tuple[NativePersonTrackPoint, ...]
    birth_native_frame_index: int = 0
    lifecycle_state: str = 'tentative'
    end_reason: str | None = None


@dataclass(frozen=True, slots=True)
class NativeGroupTracklets:
    request_source_index: int
    observations: tuple[NativeGroupObservation, ...]
    tracks: tuple[NativePersonTrack, ...]
    required_references: tuple[str, ...]

    def to_dict(self):
        return dict(contract='source_native_bounded_person_fragments_with_retained_unknowns_v2',
            **asdict(self), source_local_references_are_not_named_identity=True,
            required_references_shrink_after_dropout=False, cross_camera_correspondence_verified=False,
            fragment_births_do_not_recover_required_identity=True,
            fragment_aliases=(), gap_policy_basis='native_frames_not_array_steps',
            motion_certified=False, visibility_certified=False, quality_approved=False)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    @property
    def complete_required_native_paths(self):
        by_ref = {item.reference: item for item in self.tracks}
        return bool(self.required_references) and all(
            len(by_ref[ref].points) == len(self.observations)
            and all(point.observation_id is not None for point in by_ref[ref].points)
            for ref in self.required_references)


def track_native_group(session, *, request_source_index, native_frame_indices,
        required_initial_observation_ids=None, maximum_gap_native_frames=2,
        maximum_fragments=384):
    """Track original person detections in one continuous native shot.

    The caller may predeclare a subset once at the original first view; camera
    results and failed matches are not inputs. Detection ordering may change.
    Births consume later observations, including after a cut or a long loss.
    A reappearance is a NEW fragment, not proof that a missing duty recovered.
    All remaining one-to-one alternatives are recorded, not treated as zero.
    """
    frames = tuple(native_frame_indices)
    if (len(frames) < 2 or len(frames) > 312
            or any(type(n) is not int or n < 0 for n in frames)
            or any(a > b for a, b in zip(frames, frames[1:]))):
        raise ValueError('native person tracks require a bounded chronological source grid')
    if (type(maximum_gap_native_frames) is not int or not 1 <= maximum_gap_native_frames <= 6
            or type(maximum_fragments) is not int or not 12 <= maximum_fragments <= 384):
        raise ValueError('native fragment lifecycle needs bounded frame-gap and birth policies')
    source = request_source_index
    local = session.domain.eligible_index(source)
    fps = float(session.domain.render_frame_clocks[local].frame_rate)
    cuts = tuple(math.ceil(value*fps-1e-9) for value in session.domain.scene_cut_times[local])
    first = observe_native_group(session, request_source_index=source, native_frame_index=frames[0])
    clock_identity=hashlib.sha256(json.dumps(session.domain.render_frame_clocks[local].to_dict(),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (not isinstance(first, NativeGroupObservation) or first.request_source_index != source
            or first.native_frame_index != frames[0] or first.source_clock_identity!=clock_identity
            or getattr(session,'identities',{}).get(source,first.source_file_identity)!=first.source_file_identity
            or any(native>=session.domain.render_frame_clocks[local].frame_count for native in frames)):
        raise ValueError('native observation differs from requested first source/native frame/file/clock')
    if hasattr(session,'models') and (
            {name for name,_ in first.model_content_sha256}!={'nanodet_people','rtmpose_body7','yunet_head_faces'}
            or any(name not in session.models or name not in session.model_adapters for name,_ in first.model_content_sha256)
            or first.model_content_sha256!=tuple((name,session.models[name][2]) for name,_ in first.model_content_sha256)
            or first.model_adapter_fingerprints!=tuple((name,session.model_adapters[name])
                for name,_ in first.model_content_sha256)):
        raise ValueError('first native model/adapter differs from the registered worker authority')
    observations = [first]
    initial = {item.observation_id: item for item in first.people}
    required = tuple(initial) if required_initial_observation_ids is None else tuple(required_initial_observation_ids)
    if (len(set(required)) != len(required) or any(type(n) is not int or n not in initial for n in required)):
        raise ValueError('required initial native cohort must be declared before temporal tracking')
    references, points, confirmed, births, states, ends = {}, {}, {}, {}, {}, {}
    required_references = []

    def birth(person, step, observation):
        index = len(references)
        if index >= maximum_fragments:
            raise ValueError('native fragment budget exhausted without dropping unresolved tracks')
        native = frames[step]
        references[index] = f'source-{source}/native-{native}/body-fragment-{index}'
        births[index] = native
        states[index] = 'tentative'
        ends[index] = None
        # Explicit absence before birth is not a retrospectively invented pose.
        points[index] = [NativePersonTrackPoint(row.native_frame_index,row.sha256,None,
            'native_fragment_not_yet_born',lifecycle_state='tentative') for row in observations[:step]]
        points[index].append(NativePersonTrackPoint(native,observation.sha256,person.observation_id,
            'initial_native_detection' if step == 0 else 'new_native_fragment_detection'))
        confirmed[index] = (step, person)
        return index

    for person in first.people:
        index = birth(person, 0, first)
        if person.observation_id in required:
            required_references.append(references[index])
    for step, native in enumerate(frames[1:], 1):
        session.budget.check()
        # Stream observation and flow together. Materializing all raw rows
        # first evicts early rasters and seeks through the same shot twice.
        current = observe_native_group(session, request_source_index=source, native_frame_index=native)
        if (not isinstance(current, NativeGroupObservation)
                or current.native_frame_index != native
                or (current.request_source_index, current.source_clock_identity, current.source_file_identity,
                    current.model_content_sha256,getattr(current,'model_adapter_fingerprints',()),
                    current.person_ownership_policy_sha256)
                    != (source, first.source_clock_identity, first.source_file_identity,
                        first.model_content_sha256,getattr(first,'model_adapter_fingerprints',()),
                        first.person_ownership_policy_sha256)
                or current.native_pixel_sha256 is not None and first.native_pixel_sha256 is not None
                    and current.frame_size_wh != first.frame_size_wh):
            raise ValueError('native track observations changed source/file/clock/raster/model/requested-frame domain')
        if native == frames[step-1] and current.sha256 != observations[-1].sha256:
            raise ValueError('one repeated source-native frame returned contradictory observations')
        observations.append(current)
        broken = any(frames[step-1] < cut <= native for cut in cuts)
        people = current.people
        ended_now = {}
        for index in references:
            last_step,_ = confirmed[index]
            if states[index] == 'ended':
                continue
            if broken or native-frames[last_step] > maximum_gap_native_frames:
                states[index] = 'ended'
                ends[index] = ('native_camera_cut_breaks_person_path' if broken else
                    'native_fragment_gap_exceeds_native_frame_policy')
                ended_now[index] = ends[index]
        rows = [index for index in references if states[index] != 'ended']
        current_pixels = session.native_pixels(source, frames[step])
        predictions = []
        flows = {}
        batches = {}
        for index in rows:
            last_step, person = confirmed[index]
            if current_pixels is not None and native != frames[last_step]:
                batches.setdefault(last_step, []).append((index, person))
        for last_step, items in batches.items():
            previous_pixels = session.native_pixels(source, frames[last_step])
            if previous_pixels is not None:
                observed = native_joint_pixel_flows(previous_pixels, current_pixels,
                    tuple(person for _, person in items))
                flows.update({index: flow for (index, _), flow in zip(items, observed, strict=True)})
        for index in rows:
            last_step, person = confirmed[index]
            if current_pixels is None:
                predictions.append(None)
                continue
            previous_pixels = session.native_pixels(source, frames[last_step])
            if previous_pixels is None:
                predictions.append(None)
                continue
            predictions.append(person.box_xywh if frames[last_step] == frames[step] else
                None if flows.get(index) is not None else
                _optical_flow_box(previous_pixels, current_pixels, person.box_xywh))
        overlaps = np.asarray([[(_iou(box, person.box_xywh) if box is not None else 0.)
            for person in people] for box in predictions], dtype=float)
        # Normalize separately declared local policies. Joint evidence and a
        # rigid containment fallback are not interchangeable measurements.
        costs = np.full((len(rows), len(people)), 10., dtype=float)
        joint_costs = {}
        for a, index in enumerate(rows):
            flow = flows.get(index)
            for b, person in enumerate(people):
                if not trackable_person_ownership(confirmed[index][1]) or not trackable_person_ownership(person):
                    continue
                if flow is not None:
                    value = native_joint_assignment_cost(flow, person, current.frame_size_wh)
                    if value is not None:
                        joint_costs[a, b] = value
                        costs[a, b] = value/.12
                elif overlaps[a, b] >= .60:
                    costs[a, b] = (1.-overlaps[a, b])/.40
        assignment = {}
        if rows and people:
            aa, bb = linear_sum_assignment(costs)
            for a, b in zip(aa, bb):
                competitors = np.delete(costs[a], b)
                owners = np.delete(costs[:, b], a)
                margin = .025/.12 if flows.get(rows[a]) is not None else .10/.40
                if (costs[a, b] <= 1.
                        and (not len(competitors) or float(competitors.min())-costs[a, b] >= margin)
                        and (not len(owners) or float(owners.min())-costs[a, b] >= margin)):
                    assignment[a] = b
        for index in references:
            if states[index] == 'ended':
                points[index].append(NativePersonTrackPoint(native,current.sha256,None,
                    ended_now.get(index,'native_fragment_already_ended'),lifecycle_state='ended'))
        for a, index in enumerate(rows):
            alternatives = tuple(person.observation_id for b, person in enumerate(people)
                if costs[a, b] <= 1.) if people else ()
            measured = flows[index].joints if flows.get(index) is not None else ()
            if a not in assignment:
                states[index] = 'missing'
                points[index].append(NativePersonTrackPoint(frames[step], current.sha256, None,
                    'native_pixel_link_ambiguous' if alternatives else 'native_pixel_link_unobserved',
                    predictions[a], None, alternatives, measured,lifecycle_state='missing'))
                continue
            b = assignment[a]
            person = people[b]
            previous_step,_ = confirmed[index]
            if native != frames[previous_step]:
                states[index] = 'confirmed'
            confirmed[index] = (step, person)
            points[index].append(NativePersonTrackPoint(frames[step], current.sha256, person.observation_id,
                'observed_unique_source_native_joint_pixel_link' if measured else
                'observed_unique_source_native_pixel_link', predictions[a],
                None if measured else float(overlaps[a, b]), alternatives,
                measured, joint_costs.get((a, b)),states[index]))
        consumed = set(assignment.values())
        if native != frames[step-1]:
            for b,person in enumerate(people):
                if b not in consumed:
                    birth(person,step,current)
    tracks = tuple(NativePersonTrack(references[index],
        next(point.observation_id for point in points[index] if point.observation_id is not None),
        tuple(points[index]),births[index],states[index],ends[index]) for index in references)
    return NativeGroupTracklets(source,tuple(observations),tracks,tuple(required_references))
