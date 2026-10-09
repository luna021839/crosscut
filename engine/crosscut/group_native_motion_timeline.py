"""Fresh variable-topology timeline rows for an already selected native edit.

This producer does not search actors, re-fit cameras, run a model or approve
editing quality. It consumes the selected physical occurrences and the one
completed native inventory. Changed source/camera rows are constructed anew;
an incoming edge may not borrow the predecessor's face/pose transition proof.
The complete bundle/family owner must still retain or refresh its other fields.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from statistics import mean, median

import cv2

from .group_actual_shots import visible_group_shots, visible_group_cut_inventory
from .group_joint_native_preparation import _plan_identity
from .group_review_interval_compile import output_offsets, verify_realized_sampler
from .group_source_domain import GroupSourceDomain
from .render import RenderPlan
from .timeline import ShotType
from .visual_score import VisualTimelineChoice, VisualTimelinePlan

CONTRACT = 'selected_native_motion_fresh_timeline_v1'
MAX_REFRESHED_CELLS = 8
MAX_NATIVE_SAMPLES = 24
MAX_PROGRAMME_REFRESHED_CELLS = 512
MAX_PROGRAMME_NATIVE_SAMPLES = 1536


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _same_parameters(a, b):
    if type(a) in (int, float) and type(b) in (int, float):
        return abs(a-b) <= 1e-9
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same_parameters(x, y) for x, y in zip(a, b))
    return a == b


@dataclass(frozen=True, slots=True)
class NativeMotionMetadataCell:
    segment_index: int
    first_output_frame: int
    end_output_frame: int
    retained_old_segment_index: int | None
    refresh_source_observations: bool
    refresh_incoming_edge: bool

    @property
    def taps(self):
        return (self.first_output_frame,
            (self.first_output_frame+self.end_output_frame-1)//2, self.end_output_frame-1)


@dataclass(frozen=True, slots=True)
class NativeMotionTimelineMetadata:
    timeline: VisualTimelinePlan
    cells: tuple[NativeMotionMetadataCell, ...]
    evidence_json: str

    def to_dict(self):
        return dict(contract=CONTRACT, timeline=self.timeline.to_dict(),
            cells=[dict(segment_index=c.segment_index, first_output_frame=c.first_output_frame,
                end_output_frame=c.end_output_frame, retained_old_segment_index=c.retained_old_segment_index,
                refresh_source_observations=c.refresh_source_observations,
                refresh_incoming_edge=c.refresh_incoming_edge) for c in self.cells],
            observations=json.loads(self.evidence_json), complete_bundle_constructed=False,
            camera_refit=False, actor_reselection=False, quality_approved=False)


def metadata_cells(*, original, selected, domain, selected_shots, selection_scope='adjacent_two_shots'):
    """Declare the finite refresh footprint before reading a single pixel."""
    from .group_joint_bundle_metadata import SelectedNativeMotionShot
    variable = selection_scope == 'variable_programme'
    if (selection_scope not in ('adjacent_two_shots','variable_programme')
            or not isinstance(original, RenderPlan) or not isinstance(selected, RenderPlan)
            or not isinstance(domain, GroupSourceDomain) or original.sources != domain.request_sources
            or selected.sources != original.sources or not isinstance(selected_shots, tuple)
            or (not 1 <= len(selected_shots) <= 1024 if variable else len(selected_shots) != 2)
            or any(not isinstance(s, SelectedNativeMotionShot) for s in selected_shots)):
        raise ValueError('native metadata requires both exact selected shots and their original renderer domain')
    spans = tuple((s.first_output_frame,s.end_output_frame) for s in selected_shots)
    if (any(type(a) is not int or type(b) is not int or not 0<=a<b<=original.output_frame_count for a,b in spans)
            or any(a_end>b_start if variable else a_end!=b_start
                for (_,a_end),(b_start,_) in zip(spans,spans[1:]))
            or selected.output_frame_count != original.output_frame_count
            or any(getattr(original, n) != getattr(selected, n) for n in
                ('fps', 'width', 'height', 'aspect_ratio', 'master_audio_start', 'reference_source'))):
        raise ValueError('native metadata changed its complete output/audio/source domain')
    outside = tuple((a,b) for a,b in zip((0,*(b for _,b in spans)),
        (*(a for a,_ in spans), original.output_frame_count)) if a<b)
    if outside:
        verify_realized_sampler(original, selected, domain,
            (tick for a, b in outside for tick in range(a, b)), require_source_request_time=False)
    old_starts, starts = output_offsets(original), output_offsets(selected)
    from .group_scene_epoch_program import _pixel_parameters
    from .request_work_lease import current_request_work_phase
    port = current_request_work_phase()
    for a, b in outside:
        for tick in range(a, b):
            if port is not None and tick % 128 == 0:
                port._check()
            oi, ni = bisect_right(old_starts, tick)-1, bisect_right(starts, tick)-1
            old_camera = _pixel_parameters(original.segments[oi], (tick-old_starts[oi])/original.fps)
            new_camera = _pixel_parameters(selected.segments[ni], (tick-starts[ni])/selected.fps)
            if not _same_parameters(old_camera, new_camera):
                raise ValueError('immutable native metadata camera/look changed outside the selected shots')
    original_cells = {(a, b): i for i, (a, b) in enumerate(zip(old_starts, old_starts[1:]))}
    retained = []
    for i, (a, b) in enumerate(zip(starts, starts[1:])):
        old = original_cells.get((a, b))
        if any(a<end and b>first for first,end in spans):
            old = None
        elif old is not None and selected.segments[i].source_index != original.segments[old].source_index:
            raise ValueError('immutable native metadata changed its original request source')
        retained.append(old)
    result = tuple(NativeMotionMetadataCell(i, a, b, retained[i], retained[i] is None,
        i > 0 and not (retained[i] is not None and retained[i-1] == retained[i]-1))
        for i, (a, b) in enumerate(zip(starts, starts[1:])))
    refreshed = tuple(c for c in result if c.refresh_source_observations)
    if (len(refreshed) > (MAX_PROGRAMME_REFRESHED_CELLS if variable else MAX_REFRESHED_CELLS)
            or len({t for c in refreshed for t in c.taps}) > (MAX_PROGRAMME_NATIVE_SAMPLES if variable else MAX_NATIVE_SAMPLES)):
        raise ValueError('selected native metadata exceeded its predeclared finite pixel footprint')
    return result


def projected_pixel_quality(pixels):
    """Measured sharpness/exposure/contrast, not a person or editing score."""
    if pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 2:
        raise ValueError('native metadata needs an actual projected BGR raster')
    width = min(480, pixels.shape[1])
    working = cv2.resize(pixels, (width, max(2, round(pixels.shape[0]*width/pixels.shape[1]))))
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    sharpness = 1.-math.exp(-float(cv2.Laplacian(gray, cv2.CV_32F).var())/180.)
    clipped = float(((gray < 8) | (gray > 247)).mean())
    exposure = math.exp(-((float(gray.mean())-118.)/105.)**2)*(1.-min(.8, clipped))
    contrast = min(1., float(gray.std())/58.)
    quality = max(0., min(1., .45*sharpness+.32*exposure+.23*contrast))
    edges = cv2.Canny(gray, 60, 150)
    moments = cv2.moments(edges)
    composition = None if moments['m00'] == 0 else max(0., 1.-2.*math.hypot(
        moments['m10']/moments['m00']/gray.shape[1]-.5,
        moments['m01']/moments['m00']/gray.shape[0]-.5))
    return dict(quality_score=quality, composition_score=composition,
        sharpness=sharpness, exposure=exposure, contrast=contrast,
        quality_basis='actual_projected_pixel_sharpness_exposure_contrast_not_professional_quality')


def _fresh_choice(cell, segment, domain, plan, samples, shot_id, native_cuts):
    rows = [samples[t] for t in cell.taps]
    faces = tuple(row['face_box'] for row in rows)
    scales = tuple(None if box is None else min(1., box[3]/.42) for box in faces)
    def shot_type(scale):
        return ShotType.UNKNOWN if scale is None else ShotType.CLOSE_UP if scale >= .52 else (
            ShotType.MEDIUM if scale >= .22 else ShotType.WIDE)
    observed = [s for s in scales if s is not None]
    scale = median(observed) if observed else None
    values = [row['pixels']['composition_score'] for row in rows]
    return VisualTimelineChoice(segment_index=cell.segment_index,
        source_index=domain.eligible_index(segment.source_index),
        canonical_master_start=plan.master_audio_start+segment.master_start,
        canonical_master_end=plan.master_audio_start+segment.master_start+segment.duration,
        source_local_start=segment.source_start, source_local_end=segment.source_end,
        quality_score=mean(row['pixels']['quality_score'] for row in rows),
        composition_score=mean(values) if all(v is not None for v in values) else None,
        boundary_score=None, shot_scale=scale, shot_type=shot_type(scale), shot_id=shot_id,
        entry_shot_scale=scales[0], exit_shot_scale=scales[-1],
        entry_shot_type=shot_type(scales[0]), exit_shot_type=shot_type(scales[-1]),
        face_samples_xywh=faces,
        face_landmark_samples_xy=tuple(row['landmarks'] for row in rows),
        face_head_roll_samples_radians=tuple(row['head_roll'] for row in rows),
        face_eye_scale_samples=tuple(row['eye_scale'] for row in rows),
        face_detector_backends=tuple(row['face_backend'] for row in rows),
        face_match_candidate_failure=None if all(box is not None for box in faces) else
            'selected_actor_observation_unknown_no_salience_reselection',
        pose_backends=('native_inventory_no_legacy_pose_embedding',)*3,
        detected_person_counts=tuple(row['person_count'] for row in rows),
        shot_scale_bases=tuple('selected_native_actor_face_roi_not_head_extent' if box is not None
            else 'unknown_selected_actor_geometry' for box in faces),
        internal_scene_cut_master_seconds=tuple(plan.master_audio_start+segment.master_start
            +(cut.first_incoming_output_frame-cell.first_output_frame)/plan.fps for cut in native_cuts
            if cell.first_output_frame < cut.first_incoming_output_frame < cell.end_output_frame))


def _clear_incoming_edge(choice):
    # Geometry belonging to this unchanged source remains intact. Only facts
    # about its previous donor cease to describe the selected edge.
    return replace(choice, boundary_score=None, entry_boundary_side_stability=None,
        entry_face_geometry_window=(), dissolve_entry_face_geometry_window=(),
        dissolve_entry_face_box_window=(), dissolve_entry_face_identity_window=(),
        dissolve_entry_pose_embedding_window=(), dissolve_entry_pose_backend_window=(),
        dissolve_entry_pose_confidence_window=(), dissolve_entry_silhouette_embedding_window=(),
        dissolve_entry_silhouette_backend_window=(), dissolve_entry_silhouette_confidence_window=(),
        face_transition_minimax_mismatch_from_previous=None,
        transition_evidence_authority_from_previous=None, transition_evidence_contract_from_previous=None,
        scale_transition_mismatch_from_previous=None, focus_transition_mismatch_from_previous=None,
        layout_transition_mismatch_from_previous=None, pose_transition_basis_from_previous=None,
        pose_transition_mismatch_from_previous=None, silhouette_transition_mismatch_from_previous=None,
        formation_transition_mismatch_from_previous=None)


def build_native_motion_timeline(*, context, selected_plan, domain, selected_shots, views,
        selection_scope='adjacent_two_shots'):
    """Read the finite pixel footprint; consume EXACT selected actor parents.

    Raw/model cache misses remain unknown. Only actual raster decoding is
    allowed here, debited to the existing session. No extra NN completion or
    appearance ranking can replace the selected occurrence after camera fit.
    """
    from .group_scene_camera_queries import PreparedGroupCameraQueries
    from .group_scene_owned_program import freeze_owned_scene_program
    from .group_scene_epoch_program import compile_common_clock_camera_queries
    from .group_review_camera_repair import framing_descriptor, ReviewActorSelection
    from .group_source_changing_owner import NativeMotionViews
    incumbent = context.incumbent
    if (not isinstance(incumbent.visual_timeline, VisualTimelinePlan)
            or len(incumbent.visual_timeline.choices) != len(incumbent.render_plan.segments)
            or not isinstance(views, NativeMotionViews) or views.domain != domain):
        raise ValueError('native timeline producer needs the completed live inventory and complete predecessor')
    cells = metadata_cells(original=incumbent.render_plan, selected=selected_plan,
        domain=domain, selected_shots=selected_shots, selection_scope=selection_scope)
    prepared = PreparedGroupCameraQueries(selected_plan, domain)
    requested_ticks = tuple(sorted({t for c in cells if c.refresh_source_observations for t in c.taps}))
    original = freeze_owned_scene_program(context, domain)
    compiled = compile_common_clock_camera_queries(context=context, domain=domain, original_program=original,
        candidate_plan=selected_plan, output_intervals=tuple((t, t+1) for t in requested_ticks))
    queries = {t: q for q in compiled.queries for t in range(q.first_output_frame, q.end_output_frame)}
    actor_rows = {}
    for shot in selected_shots:
        if len(shot.actor_frames) != shot.end_output_frame-shot.first_output_frame:
            raise ValueError('native timeline lost an exact selected actor path')
        for tick, row in zip(range(shot.first_output_frame, shot.end_output_frame), shot.actor_frames, strict=True):
            actual = prepared.samples[tick]
            if (row.output_tick != tick or row.request_source_index != shot.request_source_index
                    or actual.source_index != row.request_source_index
                    or actual.native_frame_index != row.native_frame_index or actual.terminal_clone
                    or tick in actor_rows):
                raise ValueError('native timeline actor parent differs from the selected sampler')
            actor_rows[tick] = row
    samples = {}
    session = views.sensors.session
    for tick in requested_ticks:
        session.budget.check()
        query = queries[tick]
        actual = prepared.samples[tick]
        found = views.record(query, tick)
        row = found['row']
        raw = None if row is None else row['native_observation']
        actor = actor_rows.get(tick)
        if actor is not None and actor.native_observation_sha256 is not None and (
                raw is None or _digest(raw) != actor.native_observation_sha256):
            raise ValueError('native timeline lost the selected actor exact raw parent')
        native = session.native_pixels(actual.source_index, actual.native_frame_index)
        if native is None:
            raise ValueError('native timeline exact raster unavailable; no adjacent substitute')
        native_sha = sha256(memoryview(native)).hexdigest()
        if raw is not None and native_sha != raw['native_pixel_sha256']:
            raise ValueError('native timeline raster differs from the completed inventory raw parent')
        projected = session.projected_pixels(query, tick)
        pixels = projected_pixel_quality(projected)
        descriptor = None if actor is None or actor.frame_local_head_id is None else framing_descriptor(
            row, actor_selection=ReviewActorSelection.occurrence(actor.frame_local_head_id))
        head = None if descriptor is None else next((h for h in row['projection']['heads']
            if h['observation_id'] == actor.frame_local_head_id), None)
        box = None if head is None else tuple(head['projected_box_xywh'])
        landmarks = None if head is None else tuple(tuple(p) for p in head['projected_landmarks_xy'])
        # No silhouette/formation/member certificate can be inferred from a
        # face descriptor or a detection count. Unknown geometry stays null.
        samples[tick] = dict(output_tick=tick, request_source_index=actual.source_index,
            native_frame_index=actual.native_frame_index, native_pixel_sha256=native_sha,
            projected_pixel_sha256=sha256(memoryview(projected)).hexdigest(),
            query_identity=query.key, raw_observation_sha256=None if raw is None else _digest(raw),
            selected_frame_local_head_id=None if actor is None else actor.frame_local_head_id,
            selected_actor_state='UNMATCHED_OR_UNOBSERVED' if box is None else 'SELECTED_OCCURRENCE_HYPOTHESIS',
            face_box=box, landmarks=landmarks,
            head_roll=None if landmarks is None else math.atan2(landmarks[1][1]-landmarks[0][1],
                (landmarks[1][0]-landmarks[0][0])*selected_plan.width/selected_plan.height),
            eye_scale=None if landmarks is None else math.hypot(landmarks[1][0]-landmarks[0][0],
                (landmarks[1][1]-landmarks[0][1])*selected_plan.height/selected_plan.width),
            face_backend='none' if box is None else 'selected_source_native_yunet_inventory',
            person_count=0 if raw is None else len(raw['people']),
            person_count_state='UNKNOWN' if raw is None else 'RAW_DETECTION_COUNT_NOT_VISIBLE_CAST', pixels=pixels)
    epochs = visible_group_shots(selected_plan, domain)
    native_cuts = tuple(e for e in visible_group_cut_inventory(selected_plan, domain) if e.kind == 'native_camera_cut')
    choices = []
    for cell, segment in zip(cells, selected_plan.segments, strict=True):
        epoch = next(s for s in epochs if s.first_output_frame <= cell.taps[1] < s.end_output_frame)
        shot_id = f'native-motion-visible-{epoch.first_output_frame}-{epoch.end_output_frame}-{epoch.source_index}'
        if cell.refresh_source_observations:
            choice = _fresh_choice(cell, segment, domain, selected_plan, samples, shot_id, native_cuts)
        else:
            choice = replace(incumbent.visual_timeline.choices[cell.retained_old_segment_index],
                segment_index=cell.segment_index, shot_id=shot_id,
                canonical_master_start=selected_plan.master_audio_start+segment.master_start,
                canonical_master_end=selected_plan.master_audio_start+segment.master_start+segment.duration,
                source_local_start=segment.source_start, source_local_end=segment.source_end)
        if cell.refresh_incoming_edge:
            choice = _clear_incoming_edge(choice)
        choices.append(choice)
    timeline = VisualTimelinePlan(choices=tuple(choices), optimizer_score=incumbent.visual_timeline.optimizer_score,
        selected_candidate_index=incumbent.visual_timeline.selected_candidate_index, candidate_review_summaries=(),
        analysis_basis=CONTRACT, candidate_generation_basis='already_selected_native_actor_phase_cut_camera',
        source_clock_basis='exact_selected_renderer_native_clock', geometry_basis='prescribed_render_geometry_v1',
        actual_epoch_clock=None)
    evidence = dict(contract=CONTRACT, original_plan_identity=_plan_identity(incumbent.render_plan),
        selection_scope=selection_scope, selected_shot_count=len(selected_shots),
        selected_plan_identity=_plan_identity(selected_plan), samples=[samples[t] for t in sorted(samples)],
        live_source_binding=dict(contract='native_metadata_live_sources_v1', source_domain=domain.to_dict(),
            source_identities=[list(session._assert_source_identity(i)) for i in domain.eligible_indices]),
        limits=dict(maximum_refreshed_cells=(MAX_PROGRAMME_REFRESHED_CELLS
                if selection_scope=='variable_programme' else MAX_REFRESHED_CELLS),
            maximum_native_samples=(MAX_PROGRAMME_NATIVE_SAMPLES
                if selection_scope=='variable_programme' else MAX_NATIVE_SAMPLES)),
        new_model_observations=0, source_observations_fresh=True, private_vectors_exported=False,
        family_metadata_complete=False, candidate_admitted=False, quality_approved=False)
    return NativeMotionTimelineMetadata(timeline, cells, json.dumps(evidence, sort_keys=True,
        separators=(',', ':'), allow_nan=False))
