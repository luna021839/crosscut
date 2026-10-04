"""Bounded camera authoring from actual static observations, for review only.

This is a V2 measurement consumer, not the legacy uniform-role admission path.
Original role epochs are read-only. A descriptor is visual salience, never a
named/persistent member or a required cast. UNKNOWN motion stays UNKNOWN.
Every proposed source/camera still requires fresh dense/parent pixel checks.
"""
from dataclasses import dataclass
import math

from .group_original_visible_support import (
    ARM_INDICES, measured_visible_arm_support, support_preserved_coordinates)
from .render import EditorialTransformKeyframe, EditorialTransformTrack

CONTRACT = 'measured_two_sided_group_camera_review_repair_v2'


@dataclass(frozen=True, slots=True)
class FramingDescriptor:
    head_center: tuple[float, float] | None
    face_height: float | None
    face_width: float | None
    pose_shape: tuple | None
    frame_local_head_id: int | None
    owner_known: bool
    body_box: tuple | None = None
    native_face_pixels: float | None = None
    output_face_pixels: float | None = None

    def to_dict(self):
        return dict(head_center=None if self.head_center is None else list(self.head_center), face_height=self.face_height,
            face_width=self.face_width, pose_shape=self.pose_shape,
            frame_local_head_id=self.frame_local_head_id, owner_known=self.owner_known,
            body_box=self.body_box,native_face_pixels=self.native_face_pixels,
            output_face_pixels=self.output_face_pixels,effective_face_detail_pixels=effective_face_detail_pixels(self),
            body_box_is_required_cast=False,persistent_identity_certified=False, motion_observed=False)


def framing_descriptor(measurement,*,head_id=None):
    """One actual face/owned-body hypothesis, never a semantic actor ID.

    The optional frame-local ID is for a separately measured appearance path.
    It cannot fall back to a different face if that hypothesis is unavailable.
    A face descriptor must not borrow the largest unrelated body's geometry.
    """
    row = measurement['projection']
    heads = [h for h in row['heads'] if h['visible_fraction'] >= .995
        and all(math.isfinite(v) for v in h['projected_box_xywh'])
        and min(h['projected_box_xywh'][2:]) > 0]
    native=measurement['native_observation']
    by_id={p['observation_id']:p for p in native['people']}
    bodies=[p for p in row['people'] if by_id.get(p['observation_id'],{}).get('confidence',0.)>=.50
        and min(p['projected_box_xywh'][2:])>0 and p['visible_fraction']>=.80]
    head=(next((h for h in heads if h['observation_id']==head_id),None) if head_id is not None
        else max(heads,key=lambda h:h['projected_box_xywh'][2]*h['projected_box_xywh'][3]) if heads else None)
    if head_id is not None and head is None:return None
    x,y,w,h=head['projected_box_xywh'] if head else (None,)*4
    owners=[] if head is None else [p for p in row['people'] if p['unique_head_observation_id']==head['observation_id']]
    body=(next((p for p in bodies if len(owners)==1 and p['observation_id']==owners[0]['observation_id']),None)
        if head is not None else max(bodies,key=lambda p:p['projected_box_xywh'][2]*p['projected_box_xywh'][3]) if bodies else None)
    if head is None and body is None:return None
    shape = None
    if len(owners) == 1:
        p = owners[0]; joints = p['projected_upper_body_xyc']
        if (len(joints) == 8 and p['part_states'].get('shoulder_anchor') == 'observed'
                and min(joints[0][2], joints[1][2]) >= .42):
            center = tuple((joints[0][i]+joints[1][i])/2 for i in range(2))
            shoulder = math.hypot(joints[0][0]-joints[1][0], (joints[0][1]-joints[1][1])*9/16)
            if shoulder > 1e-6:
                allowed = {0,1}
                for name, indices in ARM_INDICES.items():
                    if p['part_states'].get(name) == 'observed': allowed.update(indices)
                shape = tuple((index, (point[0]-center[0])/shoulder,
                    (point[1]-center[1])*9/16/shoulder)
                    for index,point in enumerate(joints[:6]) if index in allowed and point[2] >= .42)
    head_native=None if head is None else next(p for p in native['heads'] if p['observation_id']==head['observation_id'])
    pixels=None if head_native is None else head_native['box_xywh'][3]*native['frame_size_wh'][1]
    size=measurement.get('query',{}).get('camera',{}).get('output_size')
    output_pixels=None if head is None or size is None else h*size[1]
    return FramingDescriptor(None if head is None else (x+w/2,y+h/2),h,w,shape,
        None if head is None else head['observation_id'],len(owners)==1,
        None if body is None else tuple(body['projected_box_xywh']),pixels,output_pixels)


def effective_face_detail_pixels(descriptor):
    """Geometric legibility proxy, never measured texture or sharpness.

    A 4K wide face can be downsized to 35 output pixels, while zooming that
    35-pixel native face to 100 output pixels cannot create native detail.
    Both actual inventories must be known; missing output size is not guessed.
    """
    if descriptor is None or descriptor.native_face_pixels is None or descriptor.output_face_pixels is None:
        return None
    return min(descriptor.native_face_pixels,descriptor.output_face_pixels)


def geometry_components(left,right):
    """Independent actually observed components; missing face stays missing."""
    if left is None or right is None:return {}
    values={}
    if left.face_height is not None and right.face_height is not None:
        values['face_scale']=1.5*abs(math.log(right.face_height/left.face_height))
        values['face_attention']=3.*math.hypot(right.head_center[0]-left.head_center[0],
            (right.head_center[1]-left.head_center[1])*9/16)
        values['face_aspect']=.2*abs(math.log((right.face_width/right.face_height)/(left.face_width/left.face_height)))
    if left.body_box is not None and right.body_box is not None:
        a,b=left.body_box,right.body_box
        values['body_scale']=.7*abs(math.log(b[3]/a[3]))
        values['body_span']=.25*abs(math.log(b[2]/a[2]))
        values['body_attention']=1.5*math.hypot((b[0]+b[2]/2)-(a[0]+a[2]/2),
            ((b[1]+b[3]/2)-(a[1]+a[3]/2))*9/16)
    return values


def geometry_cost(left, right):
    """Unknown face geometry has no numeric cost; it is not a perfect match."""
    values=geometry_components(left,right)
    return sum(values.values()) if values else None


def shape_cost(left, right):
    if left is None or right is None or left.pose_shape is None or right.pose_shape is None:
        return None
    a={p[0]:p[1:] for p in left.pose_shape};b={p[0]:p[1:] for p in right.pose_shape}
    common=tuple(sorted(set(a)&set(b)))
    if len(common)<4:return None
    return sum(math.dist(a[i],b[i]) for i in common)/len(common)


def _point(point, key):
    # These search cameras have no roll/base crop. Actual registered query
    # projection is independently measured after a numeric plan is built.
    return (.5+(point[0]-key.center_x)*key.zoom,
        .5+(point[1]-key.center_y)*key.zoom)


def _inside(point):
    return all(-1e-9 <= v <= 1.+1e-9 for v in point)


def measured_native_support_preserved(measurement, key):
    """Do not purchase scale matching by cutting off measured native support.

    Face ROI is not whole-head anatomy. Partial source-native arms retain only
    their actually visible vertices/bone portions. Body boxes are NOT a cast.
    Unbound/unknown arm ownership is not silently promoted to required limbs.
    """
    raw=measurement['native_observation']
    for head in raw['heads']:
        x,y,w,h=head['box_xywh']
        for corner in ((max(0.,x),max(0.,y)),(min(1.,x+w),min(1.,y+h))):
            if not _inside(_point(corner,key)):return False
    for person in raw['people']:
        parts=dict(person.get('part_states',()))
        joints=person['upper_body_xyc']
        if len(joints)!=8:continue
        for name,indices in ARM_INDICES.items():
            if parts.get(name)!='observed':continue
            points=tuple(tuple(joints[i][:2]) for i in indices)
            support=measured_visible_arm_support(native_points=points,project_point=lambda p:p,
                base_crop=(0.,0.,1.,1.))
            if not support_preserved_coordinates(support,native_points=points,
                    projected_points=tuple(_point(p,key) for p in points),base_crop=(0.,0.,1.,1.)):
                return False
    return True


def transformed_descriptor(descriptor,key):
    if descriptor is None:return None
    body=descriptor.body_box
    projected_body=None if body is None else (*_point(body[:2],key),body[2]*key.zoom,body[3]*key.zoom)
    return FramingDescriptor(None if descriptor.head_center is None else _point(descriptor.head_center,key),
        None if descriptor.face_height is None else descriptor.face_height*key.zoom,
        None if descriptor.face_width is None else descriptor.face_width*key.zoom,
        descriptor.pose_shape,descriptor.frame_local_head_id,descriptor.owner_known,projected_body,
        descriptor.native_face_pixels,None if descriptor.output_face_pixels is None else descriptor.output_face_pixels*key.zoom)


def relax_measured_camera_track(track, measurements, *, sample_times, required_support_check,
        maximum_trials=21):
    """One bounded whole-track correction from already measured native rows.

    The caller supplies immutable original-part duties, not an all-context
    body-box gate. UNKNOWN correspondence is not certified PASS. This numeric
    correction performs no new inference and cannot certify editing quality.
    Measurements must be projected through the source-native identity camera;
    the callback applies the proposed key to that same geometry.
    """
    if (not isinstance(track, EditorialTransformTrack) or not measurements
            or len(measurements) != len(sample_times) or not callable(required_support_check)
            or type(maximum_trials) is not int or not 2 <= maximum_trials <= 65
            or any(not math.isfinite(t) or t < 0. or t > track.keyframes[-1].time_seconds+1e-8
                for t in sample_times)
            or any(a >= b for a,b in zip(sample_times,sample_times[1:]))
            or any(k.roll_degrees != 0. for k in track.keyframes)):
        raise ValueError('measured camera correction needs bounded ordered native samples and frozen duties')
    for trial_index in range(maximum_trials):
        alpha = 1.-trial_index/(maximum_trials-1)
        keys = tuple(EditorialTransformKeyframe(k.time_seconds,1.+(k.zoom-1.)*alpha,
            .5+(k.center_x-.5)*alpha,.5+(k.center_y-.5)*alpha) for k in track.keyframes)
        candidate = EditorialTransformTrack(keys,reason='dense_measured_original_part_camera_correction')
        if all(required_support_check(row,candidate.evaluate(t))
                for row,t in zip(measurements,sample_times,strict=True)):
            return candidate, dict(trials=trial_index+1,retained_intervention_fraction=alpha,
                measured_ticks=len(measurements),fresh_inference_performed=False,
                unknown_correspondence_certified=False,quality_approved=False)
    return None, dict(trials=maximum_trials,retained_intervention_fraction=None,
        measured_ticks=len(measurements),fresh_inference_performed=False,
        unknown_correspondence_certified=False,quality_approved=False)


def fit_endpoint_camera(measurement,target,*,time_seconds,maximum_zoom=1.75,head_id=None):
    if not 1. <= maximum_zoom <= 2.:raise ValueError('review camera zoom cap must be declared within renderer range')
    descriptor=framing_descriptor(measurement,head_id=head_id)
    if descriptor is None:return None
    if target is not None and descriptor.face_height is not None and target.face_height is not None:
        target_zoom=target.face_height/descriptor.face_height
    elif target is not None and descriptor.body_box is not None and target.body_box is not None:
        target_zoom=target.body_box[3]/descriptor.body_box[3]
    else:target_zoom=1.
    zooms=tuple(sorted({1.,1.08,1.16,1.28,1.4,maximum_zoom,
        max(1.,min(maximum_zoom,target_zoom))}))
    options=[]
    for zoom in zooms:
        x,y=descriptor.head_center if descriptor.head_center is not None else (
            descriptor.body_box[0]+descriptor.body_box[2]/2,descriptor.body_box[1]+descriptor.body_box[3]/2)
        destination=(.5,.45) if target is None else target.head_center if target.head_center is not None else (
            target.body_box[0]+target.body_box[2]/2,target.body_box[1]+target.body_box[3]/2)
        desired=(x-(destination[0]-.5)/zoom,y-(destination[1]-.5)/zoom)
        inset=.5/zoom
        centers=((.5,.5),(max(inset,min(1-inset,desired[0])),max(inset,min(1-inset,desired[1]))))
        for cx,cy in centers:
            try:key=EditorialTransformKeyframe(time_seconds,zoom,cx,cy)
            except ValueError:continue
            if not measured_native_support_preserved(measurement,key):continue
            value=transformed_descriptor(descriptor,key)
            cost=geometry_cost(target,value)
            if cost is None:
                # No invented perfect UNKNOWN target. This independent camera
                # is simply the least intervention until a neighbor is known.
                cost=math.log(zoom)
            options.append((cost+.025*math.log(zoom),key))
    return min(options,key=lambda row:row[0])[1] if options else None


def fit_whole_camera_track(measurements,left,right,*,duration,maximum_zoom=1.75,
        sample_times=None,head_ids=None):
    """Fit a single continuous camera, then verify all prefix measurements."""
    if len(measurements)<3 or duration<=0:raise ValueError('whole camera needs a measured prefix and positive duration')
    if sample_times is None:
        camera=measurements[0].get('query',{}).get('camera',{})
        fps=camera.get('output_fps')
        sample_times=(tuple((row['output_tick']-measurements[0]['output_tick'])/fps for row in measurements)
            if fps else tuple(duration*i/(len(measurements)-1) for i in range(len(measurements))))
    if (len(sample_times)!=len(measurements) or sample_times[0]!=0.
            or abs(sample_times[-1]-duration)>1e-8
            or any(not math.isfinite(t) for t in sample_times)
            or any(a>=b for a,b in zip(sample_times,sample_times[1:]))):
        raise ValueError('review camera fit needs actual ordered output sample phases')
    if head_ids is None:head_ids=(None,)*len(measurements)
    if len(head_ids)!=len(measurements):raise ValueError('review camera actor samples differ')
    first=fit_endpoint_camera(measurements[0],left,time_seconds=0.,maximum_zoom=maximum_zoom,head_id=head_ids[0])
    last=fit_endpoint_camera(measurements[-1],right,time_seconds=duration,maximum_zoom=maximum_zoom,head_id=head_ids[-1])
    if first is None or last is None:return None
    # Neutral/intermediate framing is blended, not another physical edit.
    midpoint=EditorialTransformKeyframe(duration/2,(first.zoom+last.zoom)/2,
        (first.center_x+last.center_x)/2,(first.center_y+last.center_y)/2)
    try:track=EditorialTransformTrack((first,midpoint,last),reason='actual_two_sided_measured_review_camera')
    except ValueError:return None
    # Endpoint-optimal cameras can violate measured interior support. Solve
    # that WHOLE-camera constraint from the same prefix, instead of declaring
    # the source unusable before testing even its legal identity camera.
    options=[]
    for alpha in (1.,.85,.7,.55,.4,.25,0.):
        keys=tuple(EditorialTransformKeyframe(k.time_seconds,1+(k.zoom-1)*alpha,
            .5+(k.center_x-.5)*alpha,.5+(k.center_y-.5)*alpha) for k in track.keyframes)
        try:trial=EditorialTransformTrack(keys,reason=track.reason)
        except ValueError:continue
        if not all(measured_native_support_preserved(row,trial.evaluate(t))
                for t,row in zip(sample_times,measurements,strict=True)):continue
        endpoints=(transformed_descriptor(framing_descriptor(measurements[0],head_id=head_ids[0]),trial.evaluate(0.)),
            transformed_descriptor(framing_descriptor(measurements[-1],head_id=head_ids[-1]),trial.evaluate(duration)))
        costs=(geometry_cost(left,endpoints[0]),geometry_cost(endpoints[1],right))
        options.append((sum(c for c in costs if c is not None),trial))
    return min(options,key=lambda item:item[0])[1] if options else None


def compare_two_sided_geometry(*,left,right,original_first,original_last,
        candidate_first,candidate_last):
    before_components=(geometry_components(left,original_first),geometry_components(original_last,right))
    after_components=(geometry_components(left,candidate_first),geometry_components(candidate_last,right))
    if any(not set(a)<=set(b) for a,b in zip(before_components,after_components,strict=True)):
        return dict(contract=CONTRACT,review_geometry_gain=False,reason='known_geometry_incomplete',
            original_costs=[sum(v.values()) if v else None for v in before_components],
            candidate_costs=[sum(v.values()) if v else None for v in after_components],
            original_components=list(before_components),candidate_components=list(after_components),
            unknown_imputed_as_zero=False,motion_nonregression_certified=False,quality_approved=False)
    # Additional observations are NOT improvement. Compare only original
    # known components, keeping the other original seam unknown explicitly.
    before=tuple(sum(v.values()) if v else None for v in before_components)
    after=tuple(sum(b[k] for k in a) if a else None for a,b in zip(before_components,after_components,strict=True))
    # Each incident known seam is protected: no average can pay for a new
    # gross jump at the other side of this same realizable whole camera.
    pairs=tuple((a,b) for a,b in zip(before,after,strict=True) if a is not None)
    nonregression=all(b<=a+1e-7 for a,b in pairs)
    gain=bool(pairs) and nonregression and sum(b for _,b in pairs)<sum(a for a,_ in pairs)-1e-5
    return dict(contract=CONTRACT,review_geometry_gain=gain,
        reason='two_sided_known_geometry_gain' if gain else 'known_seam_worse_or_no_gain',
        original_costs=list(before),candidate_costs=list(after),
        original_components=list(before_components),candidate_components=list(after_components),
        comparison_kind='partial_known_original_geometry' if len(pairs)<2 else 'known_original_geometry',
        original_pose_shape_costs=[shape_cost(left,original_first),shape_cost(original_last,right)],
        candidate_pose_shape_costs=[shape_cost(left,candidate_first),shape_cost(candidate_last,right)],
        unknown_imputed_as_zero=False,actor_identity_certified=False,
        motion_nonregression_certified=False,candidate_admitted=False,quality_approved=False)
