"""Experimental constant face registration over the actual dissolve overlap.

NOT connected to production: the sixteen-edge real replay admitted zero
registrations. A one-sided constant camera and all-frame face requirement do
not cover actual moving/partly occluded fancams. Retain this measured baseline
for a joint temporal registration comparison, not as the final app policy.

The ordinary cut camera reserves overlap edges. This complementary pass owns
only otherwise untouched FACE_DISSOLVE cameras and retains all effect/cut
decisions. Failed measurements remain explicit, not permission for a hard cut.
"""
from dataclasses import replace
import math
from statistics import fmean

from .direct_member_editorial_motion import DirectMemberEditorialMotionConfig, DirectMemberFaceSample
from .direct_member_transform import attention_mismatch, box_visible_fraction, map_box_to_portrait_fill, project_box
from .direct_member_target_face import FOCAL_FACE_BASIS
from .render import EditorialTransformKeyframe as Keyframe, EditorialTransformTrack as Track, TransitionKind


def align_member_dissolve_cameras(*, render_segments, transitions, source_indices,
        eligible_request_source_indices, source_geometries, common_master_start,
        geometry_provider, fps=29.97, config=None):
    """Return measured camera additions and audit; never overwrite another track.

One constant registration is held through every blended frame. The renderer
holds an outgoing final keyframe through its extended overlap; incoming
registration stays fixed until the overlap ends, then returns to identity.
Every affected ramp/overlap frame requires the same source-bound focal basis.
    """
    settings = config or DirectMemberEditorialMotionConfig()
    segments = list(render_segments)
    sources = tuple(source_indices)
    eligible = tuple(eligible_request_source_indices)
    if (len(segments) != len(sources) or len(transitions) != len(segments)-1
            or not math.isfinite(fps) or fps <= 0
            or not math.isfinite(common_master_start) or common_master_start < 0):
        raise ValueError('dissolve camera timeline is invalid')
    for segment, source in zip(segments, sources):
        if not 0 <= source < len(eligible) or segment.source_index != eligible[source]:
            raise ValueError('dissolve camera source mapping is invalid')
    measured = {}
    def observed(source, master):
        key = (source, master)
        if key not in measured:
            sample = geometry_provider(source, master)
            measured[key] = sample if (isinstance(sample, DirectMemberFaceSample)
                and sample.face_count == 1 and sample.alignment_geometry_basis == FOCAL_FACE_BASIS) else None
        return measured[key]
    rows = []
    for index, transition in enumerate(transitions):
        if transition.kind is not TransitionKind.FACE_DISSOLVE:
            continue
        row = dict(boundary_index=index, applied=False, quality_approval=False)
        rows.append(row)
        left, right = sources[index:index+2]
        if left == right:
            row['reason'] = 'same_source'; continue
        if any(s.crop is not None or s.pan_crop is not None or s.editorial_transform is not None
               for s in segments[index:index+2]):
            row['reason'] = 'existing_camera_preserved'; continue
        cut = common_master_start + segments[index+1].master_start
        count = transition.frame_count
        if count < 1:
            row['reason'] = 'empty_overlap'; continue
        overlap_times = tuple(cut + n/fps for n in range(count))
        pairs = tuple((observed(left,t),observed(right,t)) for t in overlap_times)
        if any(a is None or b is None for a,b in pairs):
            row['reason'] = 'overlap_focal_face_missing'; continue
        boxes = tuple((map_box_to_portrait_fill(a.bbox_xywh,source_geometries[left]),
                       map_box_to_portrait_fill(b.bbox_xywh,source_geometries[right])) for a,b in pairs)
        baseline = fmean(sum(attention_mismatch(a,b)) for a,b in boxes)
        row['baseline_mean_position_plus_scale'] = baseline
        if baseline <= 1e-9:
            row['reason'] = 'natural_match'; continue
        candidates = []
        for side, source in ((0,left),(1,right)):
            segment_index = index+side
            segment = segments[segment_index]
            # Median-free constant fit: every measured overlap pair contributes.
            zoom = fmean(pair[1-side][3]/pair[side][3] for pair in boxes)
            capacity = source_geometries[source].lossless_zoom_capacity
            limit = min(settings.maximum_boundary_zoom, capacity*settings.maximum_boundary_upscale_ratio)
            zoom = max(1., zoom)
            if zoom > limit + 1e-9:
                continue
            center_x = fmean(pair[side][0]+pair[side][2]/2-(pair[1-side][0]+pair[1-side][2]/2-.5)/zoom for pair in boxes)
            center_y = fmean(pair[side][1]+pair[side][3]/2-(pair[1-side][1]+pair[1-side][3]/2-.5)/zoom for pair in boxes)
            if math.hypot(center_x-.5,center_y-.5) > settings.maximum_center_shift+1e-9:
                continue
            try:
                state = Keyframe(0.,zoom,center_x,center_y)
            except ValueError:
                continue
            projected = tuple(project_box(pair[side],zoom=zoom,center_x=center_x,center_y=center_y) for pair in boxes)
            residuals = tuple(attention_mismatch(p,pair[1-side]) for p,pair in zip(projected,boxes))
            if any(position > settings.maximum_face_residual_position_mismatch+1e-9
                   or scale > settings.maximum_face_residual_scale_mismatch+1e-9 for position,scale in residuals):
                continue
            residual = fmean(sum(item) for item in residuals)
            if 1.-residual/baseline < settings.minimum_mismatch_improvement_fraction:
                continue
            overlap = count/fps
            ramp = settings.ramp_seconds
            if segment.duration <= overlap+ramp+1/fps:
                continue
            if side == 0:
                hold_start = segment.duration-1/fps
                active_start = hold_start-ramp
                points = (Keyframe(0.,1.),Keyframe(active_start,1.),replace(state,time_seconds=hold_start),replace(state,time_seconds=segment.duration))
                local_times = tuple(n/fps for n in range(max(0,math.floor(active_start*fps)), math.ceil((segment.duration+overlap)*fps)+1))
            else:
                active_start = 0.
                points = (state,replace(state,time_seconds=overlap),Keyframe(overlap+ramp,1.),Keyframe(segment.duration,1.))
                local_times = tuple(n/fps for n in range(math.ceil((overlap+ramp)*fps)+1))
            try:
                track = Track(points,reason='dense_focal_face_registered_dissolve_v1')
            except ValueError:
                continue
            valid = True
            for local in local_times:
                sample = observed(source,common_master_start+segment.master_start+local)
                if sample is None:
                    valid = False; break
                at = track.evaluate(local)
                face = map_box_to_portrait_fill(sample.bbox_xywh,source_geometries[source])
                if box_visible_fraction(face,zoom=at.zoom,center_x=at.center_x,center_y=at.center_y) < settings.minimum_dense_face_visible_fraction:
                    valid = False; break
                widened = (at.zoom > settings.legacy_face_only_maximum_zoom or
                    math.hypot(at.center_x-.5,at.center_y-.5) > settings.legacy_face_only_maximum_center_shift)
                if widened:
                    if sample.subject_bbox_xywh is None or sample.subject_confidence < settings.minimum_widened_subject_confidence:
                        valid = False; break
                    subject = map_box_to_portrait_fill(sample.subject_bbox_xywh,source_geometries[source])
                    if box_visible_fraction(subject,zoom=at.zoom,center_x=at.center_x,center_y=at.center_y) < settings.minimum_face_associated_subject_visible_fraction:
                        valid = False; break
            if valid:
                candidates.append((residual,segment_index,track,dict(transformed_side='outgoing' if side==0 else 'incoming',
                    zoom=zoom,center_x=center_x,center_y=center_y,upscale_ratio=max(1.,zoom/capacity),
                    overlap_frame_count=count,dense_frame_count=len(local_times),residual_mean_position_plus_scale=residual)))
        if not candidates:
            row['reason'] = 'no_dense_verified_registration'; continue
        _, segment_index, track, proof = min(candidates,key=lambda item:item[0])
        segments[segment_index] = replace(segments[segment_index],editorial_transform=track)
        row.update(proof,applied=True,reason='registered_overlap_and_verified_ramp')
    return tuple(segments), dict(contract='member_dissolve_camera_v1',boundaries=rows,
        applied_count=sum(row['applied'] for row in rows),measurement_count=len(measured),
        cut_selection_changed=False,effects_changed=False,quality_approval=False)
