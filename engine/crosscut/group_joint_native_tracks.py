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


@dataclass(frozen=True, slots=True)
class NativePersonTrack:
    reference: str
    initial_observation_id: int
    points: tuple[NativePersonTrackPoint, ...]


@dataclass(frozen=True, slots=True)
class NativeGroupTracklets:
    request_source_index: int
    observations: tuple[NativeGroupObservation, ...]
    tracks: tuple[NativePersonTrack, ...]
    required_references: tuple[str, ...]

    def to_dict(self):
        return dict(contract='source_native_unique_pixel_flow_tracklets_with_retained_unknowns_v1',
            **asdict(self), source_local_references_are_not_named_identity=True,
            required_references_shrink_after_dropout=False, cross_camera_correspondence_verified=False,
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
        required_initial_observation_ids=None):
    """Track original person detections in one continuous native shot.

    The caller may predeclare a subset once at the original first view; camera
    results and failed matches are not inputs. Detection ordering may change.
    A native camera cut explicitly breaks paths rather than linking two takes.
    All remaining one-to-one alternatives are recorded, not treated as zero.
    """
    frames = tuple(native_frame_indices)
    if (len(frames) < 2 or len(frames) > 312
            or any(type(n) is not int or n < 0 for n in frames)
            or any(a > b for a, b in zip(frames, frames[1:]))):
        raise ValueError('native person tracks require a bounded chronological source grid')
    source = request_source_index
    local = session.domain.eligible_index(source)
    fps = float(session.domain.render_frame_clocks[local].frame_rate)
    cuts = tuple(math.ceil(value*fps-1e-9) for value in session.domain.scene_cut_times[local])
    first = observe_native_group(session, request_source_index=source, native_frame_index=frames[0])
    observations = [first]
    initial = {item.observation_id: item for item in first.people}
    required = tuple(initial) if required_initial_observation_ids is None else tuple(required_initial_observation_ids)
    if (len(set(required)) != len(required) or any(type(n) is not int or n not in initial for n in required)):
        raise ValueError('required initial native cohort must be declared before temporal tracking')
    references = {index: f'source-{source}/native-{frames[0]}/body-track-{index}' for index in initial}
    points = {index: [NativePersonTrackPoint(frames[0], first.sha256, index, 'initial_native_detection')]
        for index in initial}
    confirmed = {index: (0, person) for index, person in initial.items()}
    broken = False
    rows = list(initial)
    for step, native in enumerate(frames[1:], 1):
        session.budget.check()
        # Stream observation and flow together. Materializing all raw rows
        # first evicts early rasters and seeks through the same shot twice.
        current = observe_native_group(session, request_source_index=source, native_frame_index=native)
        if ((current.request_source_index, current.source_clock_identity, current.source_file_identity
                ) != (source, first.source_clock_identity, first.source_file_identity)
                or current.native_pixel_sha256 is not None and first.native_pixel_sha256 is not None
                    and current.frame_size_wh != first.frame_size_wh):
            raise ValueError('native track observations changed source/file/clock/raster domain')
        observations.append(current)
        broken = broken or any(frames[step-1] < cut <= frames[step] for cut in cuts)
        people = current.people
        if broken:
            for index in initial:
                points[index].append(NativePersonTrackPoint(frames[step], current.sha256, None,
                    'native_camera_cut_breaks_person_path'))
            continue
        current_pixels = session.native_pixels(source, frames[step])
        predictions = []
        flows = {}
        batches = {}
        for index in rows:
            last_step, person = confirmed[index]
            if current_pixels is not None and step-last_step <= 2:
                batches.setdefault(last_step, []).append((index, person))
        for last_step, items in batches.items():
            previous_pixels = session.native_pixels(source, frames[last_step])
            if previous_pixels is not None:
                observed = native_joint_pixel_flows(previous_pixels, current_pixels,
                    tuple(person for _, person in items))
                flows.update({index: flow for (index, _), flow in zip(items, observed, strict=True)})
        for index in rows:
            last_step, person = confirmed[index]
            if current_pixels is None or step-last_step > 2:
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
        for a, index in enumerate(rows):
            alternatives = tuple(person.observation_id for b, person in enumerate(people)
                if costs[a, b] <= 1.) if people else ()
            measured = flows[index].joints if flows.get(index) is not None else ()
            if a not in assignment:
                points[index].append(NativePersonTrackPoint(frames[step], current.sha256, None,
                    'native_pixel_link_ambiguous' if alternatives else 'native_pixel_link_unobserved',
                    predictions[a], None, alternatives, measured))
                continue
            b = assignment[a]
            person = people[b]
            confirmed[index] = (step, person)
            points[index].append(NativePersonTrackPoint(frames[step], current.sha256, person.observation_id,
                'observed_unique_source_native_joint_pixel_link' if measured else
                'observed_unique_source_native_pixel_link', predictions[a],
                None if measured else float(overlaps[a, b]), alternatives,
                measured, joint_costs.get((a, b))))
    tracks = tuple(NativePersonTrack(references[index], index, tuple(points[index])) for index in rows if initial)
    return NativeGroupTracklets(source, tuple(observations), tracks, tuple(references[index] for index in required))
