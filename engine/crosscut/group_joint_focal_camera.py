"""Whole-shot face-ROI camera proposals against BOTH fixed cut neighbors.

Pure geometry, not subject selection, head-extent evidence, arm/phase proof
or production admission. A source-changing owner must supply independently
bound native observations and validate every required part after projection.
"""
from dataclasses import dataclass, replace
import math

from .boundary_subject_alignment import (
    BoundarySubjectAlignmentConfig, _TransformState, _target_state,
    _track_for_segment, _follow_face_camera_path, _project_box, _center,
    _project_point, _clamp_center,
)
from .render import EditorialTransformTrack, EditorialTransformKeyframe


def _valid_box(box):
    return (isinstance(box, tuple) and len(box) == 4
        and all(type(v) in (int, float) and math.isfinite(v) for v in box)
        and min(box[:2]) >= 0 and min(box[2:]) > 0
        and box[0]+box[2] <= 1.000001 and box[1]+box[3] <= 1.000001)


def _residual(source, target, state):
    projected, visible = _project_box(source, state)
    return dict(face_height_log_residual=abs(math.log(projected[3]/target[3])),
        face_center_distance=math.dist(_center(projected), _center(target)),
        observed_face_roi_visible_fraction=visible)


@dataclass(frozen=True, slots=True)
class FocalFaceROICameraProposal:
    track: EditorialTransformTrack
    endpoint_before: tuple[dict, dict]
    endpoint_after: tuple[dict, dict]
    frame_count: int
    minimum_observed_face_roi_visible_fraction: float
    observed_geometry_sample_count: int
    preserved_entry_camera_frame_count: int = 0
    unobserved_geometry_frame_indices: tuple[int, ...] = ()
    endpoint_required_envelopes: tuple[tuple[tuple[float, float, float, float], ...], ...] = ()

    def to_dict(self):
        return dict(contract=('authored_partial_native_face_roi_camera_proposal_v1'
                if self.unobserved_geometry_frame_indices else
                'both_fixed_neighbors_whole_shot_native_face_roi_camera_proposal_v1'),
            endpoint_before=self.endpoint_before, endpoint_after=self.endpoint_after,
            frame_count=self.frame_count,
            observed_geometry_sample_count=self.observed_geometry_sample_count,
            preserved_entry_camera_frame_count=self.preserved_entry_camera_frame_count,
            camera_parameter_retention_is_not_encoded_pixel_identity=True,
            every_output_frame_face_roi_observed=self.observed_geometry_sample_count == self.frame_count,
            unobserved_geometry_frame_indices=self.unobserved_geometry_frame_indices,
            interpolated_camera_parameters_are_not_subject_observations=bool(self.unobserved_geometry_frame_indices),
            minimum_observed_face_roi_visible_fraction=self.minimum_observed_face_roi_visible_fraction,
            endpoint_required_envelopes=self.endpoint_required_envelopes,
            required_envelopes_are_fitting_constraints_not_dense_evidence=True,
            face_inputs_are_geometry_only=True, native_subject_correspondence_certified=False,
            whole_head_extent_certified=False, required_arm_visibility_certified=False,
            gesture_or_phase_continuity_certified=False, source_clock_certified=False,
            production_admitted=False, quality_approved=False)


def propose_focal_face_roi_camera(*, source_face_boxes, fixed_neighbor_face_boxes,
        fps, incumbent_track=None, maximum_zoom=2., maximum_center_shift=.10):
    """Retain the actual nonlinear face path; never use only the exit anchor.

    The bounded clamp is a proposal with explicit residuals, not exact-match
    or quality approval. No missing sample is interpolated into evidence, and
    a face rectangle is never inflated or relabelled as a whole head.
    """
    boxes, targets = tuple(source_face_boxes), tuple(fixed_neighbor_face_boxes)
    if (not 2 <= len(boxes) <= 300 or len(targets) != 2
            or any(not _valid_box(box) for box in (*boxes, *targets))
            or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or type(maximum_zoom) not in (int, float) or not math.isfinite(maximum_zoom)
            or not 1 <= maximum_zoom <= 2
            or type(maximum_center_shift) not in (int, float) or not math.isfinite(maximum_center_shift)
            or not 0 < maximum_center_shift <= .25
            or incumbent_track is not None and not isinstance(incumbent_track, EditorialTransformTrack)):
        raise ValueError('focal camera needs every actual native face ROI and both fixed neighbors')
    # Keep the conservative caller default, but permit an explicit same
    # physical quarter-frame envelope as the endpoint sibling. Every actual
    # ROI and renderer keyframe still has to pass the native support checks.
    # This is a proposal limit, not a motion/part-containment waiver.
    endpoints = (boxes[0], boxes[-1])
    states = []
    for source, target in zip(endpoints, targets):
        zoom = min(maximum_zoom, max(1., target[3]/source[3]))
        state, _ = _target_state(box=source, target_box=target, zoom=zoom,
            roll_degrees=0., maximum_center_shift=maximum_center_shift,
            settings=BoundarySubjectAlignmentConfig())
        if state is None:
            raise ValueError('two-sided focal proposal has no native-safe endpoint camera')
        states.append(state)
    duration = (len(boxes)-1)/fps
    track = _track_for_segment(last_frame_time=duration, ramp_frames=14, fps=fps,
        entry=(0, states[0]), exit=(1, states[1]), settle_policy='shot_continuous')
    if track is None:
        raise ValueError('both focal cut cameras are already neutral')
    track = _follow_face_camera_path(track, boxes, fps=fps, maximum_center_shift=maximum_center_shift)
    track = replace(track, reason='both_fixed_neighbors_native_face_roi_following_not_extent_certificate_v1')
    before, after, fractions = [], [], []
    for n, box in enumerate(boxes):
        key = track.evaluate(n/fps)
        state = _TransformState(key.zoom, key.center_x, key.center_y, key.roll_degrees)
        fraction = _project_box(box, state)[1]
        if fraction < .995:
            raise ValueError(f'focal proposal clips an actually supplied face ROI at frame {n}')
        fractions.append(fraction)
    for n, box, target in ((0, boxes[0], targets[0]), (len(boxes)-1, boxes[-1], targets[-1])):
        key = track.evaluate(n/fps)
        after.append(_residual(box, target, _TransformState(key.zoom, key.center_x, key.center_y, key.roll_degrees)))
        old = incumbent_track.evaluate(n/fps) if incumbent_track is not None else _TransformState()
        before.append(_residual(box, target, _TransformState(old.zoom, old.center_x, old.center_y, old.roll_degrees)))
    return FocalFaceROICameraProposal(track, tuple(before), tuple(after), len(boxes), min(fractions), len(boxes))


def propose_partial_face_roi_camera(*, source_face_boxes, fixed_neighbor_face_boxes,
        fps, incumbent_track=None, maximum_zoom=2., maximum_center_shift=.25,
        maximum_unobserved_run=4):
    """Author camera-only gap parameters; never fill a missing face measurement.

    A short motion-blurred/undetected interval need not prevent a diagnostic
    camera render. It still CANNOT provide dense role/extent/identity admission.
    Every input None remains declared. Endpoint/cut geometry is measured, and
    all supplied visible ROIs must pass the unchanged visibility check. Unknown
    frames get interpolated CAMERA centers, not invented boxes or body poses.
    """
    boxes = tuple(source_face_boxes)
    if (not 2 <= len(boxes) <= 300 or boxes[0] is None or boxes[-1] is None
            or type(maximum_unobserved_run) is not int or not 1 <= maximum_unobserved_run <= 4
            or any(box is not None and not _valid_box(box) for box in boxes)):
        raise ValueError('partial camera needs real endpoints and explicit bounded missing geometry')
    missing = tuple(i for i, box in enumerate(boxes) if box is None)
    run = 0
    for box in boxes:
        run = run+1 if box is None else 0
        if run > maximum_unobserved_run:
            raise ValueError('partial camera missing interval exceeds its authored-parameter policy')
    endpoint = propose_endpoint_focal_face_roi_camera(source_face_boxes=(boxes[0], boxes[-1]),
        fixed_neighbor_face_boxes=fixed_neighbor_face_boxes, frame_count=len(boxes), fps=fps,
        incumbent_track=incumbent_track, maximum_zoom=maximum_zoom,
        maximum_center_shift=maximum_center_shift)
    keys = tuple(endpoint.track.evaluate(i/fps) for i in range(len(boxes)))
    targets = tuple(_project_point(_center(box), _TransformState(key.zoom,
        key.center_x, key.center_y, key.roll_degrees))
        for box, key in ((boxes[0], keys[0]), (boxes[-1], keys[-1])))
    centers = []
    for i, (box, key) in enumerate(zip(boxes, keys, strict=True)):
        if box is None:
            centers.append(None)
            continue
        amount = i/(len(boxes)-1)
        target = tuple(a*(1-amount)+b*amount for a, b in zip(*targets))
        point = _project_point(_center(box), _TransformState(1., .5, .5, key.roll_degrees))
        centers.append(tuple(point[j]-(target[j]-.5)/key.zoom for j in (0, 1)))
    # Only centers are authored between observed neighbors. `boxes` stays
    # untouched and None; no subject sample is supplied to the dense API.
    for i in missing:
        left = max(j for j in range(i) if boxes[j] is not None)
        right = min(j for j in range(i+1, len(boxes)) if boxes[j] is not None)
        amount = (i-left)/(right-left)
        centers[i] = tuple(a*(1-amount)+b*amount for a, b in zip(centers[left], centers[right]))
    smoothed = [tuple(sum(weight*centers[min(len(boxes)-1, max(0, i+offset-2))][j]
        for offset, weight in enumerate((1., 2., 3., 2., 1.)))/9. for j in (0, 1))
        for i in range(len(boxes))]
    output = []
    for i, (key, center) in enumerate(zip(keys, smoothed, strict=True)):
        if i in (0, len(boxes)-1):
            output.append(key)
            continue
        amount = i/(len(boxes)-1)
        corrected = tuple(center[j]+(1-amount)*(centers[0][j]-smoothed[0][j])
            +amount*(centers[-1][j]-smoothed[-1][j]) for j in (0, 1))
        cx, cy = _clamp_center(*corrected, zoom=key.zoom, maximum_shift=maximum_center_shift)
        output.append(EditorialTransformKeyframe(i/fps, key.zoom, cx, cy, key.roll_degrees))
    track = EditorialTransformTrack(tuple(output),
        reason='observed_face_rois_with_explicit_unobserved_camera_parameter_gaps_not_dense_part_proof_v1')
    visible = []
    for i, box in enumerate(boxes):
        if box is None:
            continue
        key = track.evaluate(i/fps)
        fraction = _project_box(box, _TransformState(key.zoom, key.center_x, key.center_y, key.roll_degrees))[1]
        if fraction < .995:
            raise ValueError(f'partial camera clips a supplied real face ROI at frame {i}')
        visible.append(fraction)
    return replace(endpoint, track=track, observed_geometry_sample_count=len(visible),
        minimum_observed_face_roi_visible_fraction=min(visible),
        unobserved_geometry_frame_indices=missing)


def propose_endpoint_focal_face_roi_camera(*, source_face_boxes, fixed_neighbor_face_boxes,
        frame_count, fps, incumbent_track=None, maximum_zoom=2., maximum_center_shift=.25,
        required_endpoint_envelopes=None):
    """Authored camera interpolation from TWO measurements, not dense proof.

    This deliberately separate proposal permits an actual render/inspection
    when dense sensing is unavailable. Only camera parameters are interpolated;
    no middle-frame face, head, arm or motion observation is manufactured.
    The normal source-changing owner cannot use this as complete part coverage.
    """
    boxes, targets = tuple(source_face_boxes), tuple(fixed_neighbor_face_boxes)
    if (type(frame_count) is not int or not 2 <= frame_count <= 300 or len(boxes) != 2 or len(targets) != 2
            or any(not _valid_box(box) for box in (*boxes, *targets))
            or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or type(maximum_zoom) not in (int, float) or not math.isfinite(maximum_zoom)
            or not 1 <= maximum_zoom <= 2
            or type(maximum_center_shift) not in (int, float) or not math.isfinite(maximum_center_shift)
            or not 0 < maximum_center_shift <= .25
            or incumbent_track is not None and not isinstance(incumbent_track, EditorialTransformTrack)):
        raise ValueError('endpoint camera proposal needs two measurements and a bounded output interval')
    envelopes=() if required_endpoint_envelopes is None else tuple(tuple(group)
        for group in required_endpoint_envelopes)
    if envelopes and (len(envelopes)!=2 or any(not _valid_box(box)
            for group in envelopes for box in group)):
        # Coordinates are AFTER the immutable source crop. Expanding an
        # editorial camera cannot restore its already discarded pixels.
        raise ValueError('required endpoint envelope is unobserved or outside the retained base raster')
    states = []
    for index,(source, target) in enumerate(zip(boxes, targets)):
        # The renderer already admits this physical quarter-frame envelope.
        # Do not inherit a narrower planner style cap and call its residual
        # unavoidable. Real-pixel corners and endpoint ROI support still apply.
        zoom = min(maximum_zoom, max(1., target[3]/source[3]))
        if envelopes:
            state=_required_envelope_camera(source,target,(source,*envelopes[index]),
                desired_zoom=zoom,maximum_center_shift=maximum_center_shift)
        else:
            state, _ = _target_state(box=source, target_box=target, zoom=zoom,
                roll_degrees=0., maximum_center_shift=maximum_center_shift,
                settings=BoundarySubjectAlignmentConfig())
            if state is None:
                raise ValueError('endpoint camera has no real-pixel-safe crop')
        states.append(state)
    duration = (frame_count-1)/fps
    track = _track_for_segment(last_frame_time=duration, ramp_frames=14, fps=fps,
        entry=(0, states[0]), exit=(1, states[1]), settle_policy='shot_continuous')
    if track is None:
        raise ValueError('both endpoint cameras are already neutral')
    track = replace(track, reason='two_observed_endpoints_authored_camera_not_dense_native_part_evidence_v1')
    before, after = [], []
    for time, box, target, state in zip((0., duration), boxes, targets, states):
        current = _residual(box, target, state)
        if current['observed_face_roi_visible_fraction'] < .995:
            raise ValueError('endpoint camera clips its actual observed face ROI')
        after.append(current)
        old = incumbent_track.evaluate(time) if incumbent_track is not None else _TransformState()
        before.append(_residual(box, target, _TransformState(old.zoom, old.center_x, old.center_y, old.roll_degrees)))
    return FocalFaceROICameraProposal(track, tuple(before), tuple(after), frame_count,
        min(row['observed_face_roi_visible_fraction'] for row in after), 2,
        endpoint_required_envelopes=envelopes)


def _required_envelope_camera(face,target,envelopes,*,desired_zoom,maximum_center_shift):
    """Continuous feasible endpoint fit, no gate/cap reduction or pixel synthesis.

    Preserve all supplied measured boxes; minimize face-target displacement
    inside the intersecting canvas/support/pan domain at the closest possible
    zoom to the face-scale target. Middle-frame admission stays the owner's job.
    """
    lo=tuple(min(box[axis] for box in envelopes) for axis in (0,1))
    hi=tuple(max(box[axis]+box[axis+2] for box in envelopes) for axis in (0,1))
    zoom=min(desired_zoom,*(1/(hi[j]-lo[j]) for j in (0,1)))
    if zoom<1.-1e-9:
        raise ValueError('required endpoint envelopes have no retained-raster camera domain')
    def domain(at_zoom):
        half=.5/at_zoom
        bounds=tuple((max(half,hi[j]-half),min(1.-half,lo[j]+half)) for j in (0,1))
        if any(low>high for low,high in bounds):return None
        nearest=tuple(min(high,max(low,.5)) for low,high in bounds)
        return bounds if math.dist(nearest,(.5,.5))<=maximum_center_shift else None
    zoom=max(1.,zoom);bounds=domain(zoom)
    if bounds is None:
        # Lower zoom expands every canvas/support center interval. Search its
        # monotonic feasible domain rather than reject a repairable framing
        # or raise the radial pan limit. This is only bounded geometry work.
        low,high=1.,zoom
        if domain(low) is None:
            raise ValueError('required endpoint camera has no retained-raster/pan domain')
        for _ in range(40):
            middle=(low+high)/2
            if domain(middle) is None:high=middle
            else:low=middle
        zoom=low;bounds=domain(zoom)
    clamp=lambda point:tuple(min(high,max(low,point[j])) for j,(low,high) in enumerate(bounds))
    origin=clamp((.5,.5))
    desired=tuple(_center(face)[j]-(_center(target)[j]-.5)/zoom for j in (0,1))
    center=clamp(desired)
    if math.dist(center,(.5,.5))>maximum_center_shift:
        delta=tuple(center[j]-origin[j] for j in (0,1))
        offset=tuple(origin[j]-.5 for j in (0,1))
        aa=sum(x*x for x in delta);bb=2*sum(x*y for x,y in zip(delta,offset))
        cc=sum(x*x for x in offset)-maximum_center_shift**2
        amount=min(1.,max(0.,(-bb+math.sqrt(max(0.,bb*bb-4*aa*cc)))/(2*aa)))
        center=tuple(origin[j]+amount*delta[j] for j in (0,1))
    state=_TransformState(zoom,*center,0.)
    state.keyframe(0.)  # Actual renderer's structural/real-pixel bounds.
    if any(_project_box(box,state)[1]<.999999 for box in envelopes):
        raise ValueError('required envelope fit failed containment; no visibility waiver')
    return state


def retain_entry_camera_proposal(proposal, *, incumbent_track, fps, preserved_frames=4):
    """Keep the appreciated entry framing while proposing a better exit.

    Parameters, not pixel identity or dense part evidence, are retained. The
    renderer's shared sampling canvas can still affect resampling, so the
    actual encoded entry must be inspected separately by the owner.
    """
    if (not isinstance(proposal, FocalFaceROICameraProposal)
            or proposal.observed_geometry_sample_count != 2
            or not isinstance(incumbent_track, EditorialTransformTrack)
            or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or type(preserved_frames) is not int or not 4 <= preserved_frames < proposal.frame_count-1):
        raise ValueError('entry retention needs a bounded endpoint proposal and actual old camera')
    count = proposal.frame_count
    first = incumbent_track.evaluate((preserved_frames-1)/fps)
    last = proposal.track.evaluate((count-1)/fps)
    keys = []
    for n in range(count):
        if n < preserved_frames:
            keys.append(incumbent_track.evaluate(n/fps))
            continue
        progress = (n-preserved_frames+1)/(count-preserved_frames)
        amount = progress**3*(10.+progress*(-15.+6.*progress))
        values = tuple(getattr(first, name)*(1.-amount)+getattr(last, name)*amount
            for name in ('zoom', 'center_x', 'center_y', 'roll_degrees'))
        keys.append(EditorialTransformKeyframe(n/fps, *values))
    track = EditorialTransformTrack(tuple(keys),
        reason='four_tick_entry_camera_parameters_retained_then_native_safe_exit_fit_not_dense_part_evidence_v1')
    return replace(proposal, track=track, endpoint_after=(proposal.endpoint_before[0], proposal.endpoint_after[1]),
        minimum_observed_face_roi_visible_fraction=min(proposal.endpoint_before[0]['observed_face_roi_visible_fraction'],
            proposal.endpoint_after[1]['observed_face_roi_visible_fraction']), preserved_entry_camera_frame_count=preserved_frames)
