"""Two-source bounded roll/crop construction from real focal observations.

Review geometry only: it cannot certify anatomy ownership or editing quality.
The feasible output target is clipped against BOTH native rotated canvases;
no black/reflected fill or independently clamped fake face match is accepted.
"""
import math
import numpy as np

from .group_review_camera_repair import _rotated_point
from .group_review_observed_programme import (
    coupled_focal_cameras,focal_extent,project_focal_observation,
)


def covered_roll_focal_camera(camera, *, maximum_zoom=2.4, maximum_roll_degrees=6.,
        raster_size=(1920,1080)):
    values=np.asarray(camera,dtype=float)
    if values.shape!=(4,) or not np.isfinite(values).all():return False
    z,x,y,roll=values
    if not 1<=z<=maximum_zoom or abs(roll)>maximum_roll_degrees:return False
    margin=0. if roll==0 else 2/min(raster_size)
    for a in (-.5,.5):
        for b in (-.5,.5):
            working=(x+a/z,y+b/z)
            if any(v < -1e-10 or v > 1+1e-10 for v in working):return False
            native=_rotated_point(working,-roll)
            if any(v < margin-1e-10 or v > 1-margin+1e-10 for v in native):return False
    return True


def project_roll_focal_observation(row,camera):
    if camera[3]==0:return project_focal_observation(row,camera)
    z,x,y,roll=map(float,camera)
    def point(p):
        a,b=_rotated_point(p,roll)
        return (float((a-x)*z+.5),float((b-y)*z+.5))
    face=dict(row['face'],landmarks_xy=[point(p) for p in row['face']['landmarks_xy']])
    body,points=row.get('body'),{}
    if body and body['reason']=='observed' and body['face_anchor_consistent'] is True:
        for j,(bx,by,confidence) in enumerate(body['upper_body_xyc'][:6]):
            if confidence>=.3 and 0<=bx<=1 and 0<=by<=1:
                p=point((bx,by))
                if all(0<=v<=1 for v in p):points[j]=p
    return face,points


def focal_eye_roll_degrees(face):
    points=np.asarray(face['landmarks_xy'],float)
    focal_extent(face)
    delta=(points[1]-points[0])*np.array((1.,9/16))
    if np.linalg.norm(delta)<=1e-6:return None
    return math.degrees(math.atan2(delta[1],delta[0]))


def observed_independent_head_reference(row, *, minimum_confidence=.3):
    """Expose real COCO head points without relabeling them as mouth landmarks.

    The face detector's five-point regression can miss tilted eyes even when
    its face box is correct. A caller may independently compare an existing
    RTMPose nose/eyes/ears observation. This is cache-only: unavailable points
    stay unknown, and no point, confidence or private identity is invented.
    The original detector observation is retained unchanged in the view.
    """
    if (type(minimum_confidence) not in (int,float)
            or not math.isfinite(minimum_confidence) or not 0<=minimum_confidence<=1):
        raise ValueError('independent head reference needs its finite confidence floor')
    body=row.get('body')
    if (not body or body.get('reason')!='observed'
            or body.get('face_anchor_consistent') is not True
            or not body.get('raw_keypoints_xyc')):
        return None
    points=np.asarray(body['raw_keypoints_xyc'],float)
    if points.shape!=(17,3) or not np.isfinite(points).all():
        raise ValueError('independent head reference requires actual finite raw17 observations')
    # COCO order: nose, subject-left eye, subject-right eye, left ear, right ear.
    # Use a separate, explicitly recorded head basis; ears are NOT mouth points.
    ids=(2,1,0,4,3)
    selected=points[list(ids)]
    if (np.any(selected[:,2]<minimum_confidence) or np.any(selected[:,:2]<0)
            or np.any(selected[:,:2]>1) or selected[1,0]<=selected[0,0]):
        return None
    face=dict(row['face'],landmarks_xy=selected[:,:2].tolist(),
        landmark_basis='independent_rtmpose_coco_nose_eyes_ears_not_yunet_mouth5')
    focal_extent(face)
    return dict(row,face=face,original_face_detector_observation=row['face'],
        actual_head_reference_coco_ids=list(ids),head_points_recomputed=False,
        head_reference_is_not_named_identity=True)


def _clip(poly,normal,bound):
    result=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        da,db=float(normal@a-bound),float(normal@b-bound)
        if da<=1e-12:result.append(a)
        if (da<=1e-12)!=(db<=1e-12):
            result.append(a+(b-a)*(da/(da-db)))
    return result


def _target_polygon(centers,sizes,rolls,extent,raster_size,*,edge_margin=None):
    poly=[np.array(p,float) for p in ((0.,0.),(1.,0.),(1.,1.),(0.,1.))]
    for center,size,roll in zip(centers,sizes,rolls,strict=True):
        zoom=extent/size
        rc=np.asarray(_rotated_point(center,roll))
        constraints=[]
        # The renderer first rotates in a fixed native canvas then crops it.
        for axis in range(2):
            normal=np.eye(2)[axis]
            constraints.extend(((normal,zoom*rc[axis]),(-normal,zoom*(1-rc[axis])-1)))
        angle=math.radians(-roll);c,s=math.cos(angle),math.sin(angle)
        inverse=np.array(((c,-s*9/16),(s*16/9,c)))
        margin=0. if roll==0 else (2/min(raster_size) if edge_margin is None else edge_margin)
        for corner in (np.array(p) for p in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5))):
            base=center+inverse@(np.array((.5,.5))+corner)/zoom
            for axis in range(2):
                normal=inverse[axis]/zoom
                constraints.extend(((normal,base[axis]-margin),(-normal,1-margin-base[axis])))
        for normal,bound in constraints:
            if not poly:return []
            poly=_clip(poly,normal,bound)
    return poly


def _closest_target(poly,target):
    # All constraints are convex. Check the target against the clipped edges,
    # then choose the nearest feasible edge point in actual 16:9 pixel space.
    cross=lambda a,b:float(a[0]*b[1]-a[1]*b[0])
    area=abs(sum(cross(a,b) for a,b in zip(poly,poly[1:]+poly[:1])))
    if len(poly)>=3 and area>1e-12:
        signs=[cross(b-a,target-a)/max(float(np.linalg.norm(b-a)),1e-12)
            for a,b in zip(poly,poly[1:]+poly[:1])]
        if all(v>=-1e-10 for v in signs) or all(v<=1e-10 for v in signs):return target
    ratio=np.array((1.,9/16));options=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        edge=(b-a)*ratio;delta=(target-a)*ratio
        u=0. if edge@edge<=1e-20 else float(np.clip((delta@edge)/(edge@edge),0.,1.))
        options.append(a+u*(b-a))
    return min(options,key=lambda p:float(np.linalg.norm((p-target)*ratio)))


def coupled_roll_focal_cameras(a,b,*,rolls,maximum_zoom=2.4,maximum_roll_degrees=6.,
        raster_size=(1920,1080),minimum_extent=0.):
    if (not isinstance(rolls,(tuple,list)) or len(rolls)!=2
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in rolls)
            or type(maximum_zoom) not in (int,float) or not math.isfinite(maximum_zoom)
            or not 1<=maximum_zoom<=2.4
            or type(maximum_roll_degrees) not in (int,float) or not math.isfinite(maximum_roll_degrees)
            or not 0<=maximum_roll_degrees<=6
            or any(abs(v)>maximum_roll_degrees for v in rolls)
            or raster_size!=(1920,1080)):
        raise ValueError('paired focal roll needs its finite declared 1080p geometry domain')
    if (type(minimum_extent) not in (int,float) or not math.isfinite(minimum_extent)
            or minimum_extent<0):
        raise ValueError('authored focal extent floor must be a finite nonnegative measurement')
    if rolls[0]==rolls[1]==0:
        return coupled_focal_cameras(a,b,maximum_zoom=maximum_zoom,minimum_extent=minimum_extent)
    centers,sizes=zip(*(focal_extent(row['face']) for row in (a,b)),strict=True)
    low,high=max(minimum_extent,1.12*max(sizes)),maximum_zoom*min(sizes)
    if high<low:return None
    # Feasibility expands monotonically with the common face extent. Freeze
    # one shared target/extent before constructing any rendered trajectory.
    poly=_target_polygon(centers,sizes,rolls,high,raster_size)
    if not poly:return None
    if _target_polygon(centers,sizes,rolls,low,raster_size):
        extent=low
    else:
        for _ in range(32):
            middle=(low+high)/2
            if _target_polygon(centers,sizes,rolls,middle,raster_size):high=middle
            else:low=middle
        extent=high
    poly=_target_polygon(centers,sizes,rolls,extent,raster_size)
    target=_closest_target(poly,np.array((.5,.42)))
    cameras=[]
    for center,size,roll in zip(centers,sizes,rolls,strict=True):
        zoom=extent/size
        center=np.asarray(_rotated_point(center,roll))+(np.array((.5,.5))-target)/zoom
        camera=[float(zoom),*map(float,center),float(roll)]
        if not covered_roll_focal_camera(camera,maximum_zoom=maximum_zoom,
                maximum_roll_degrees=maximum_roll_degrees,raster_size=raster_size):return None
        cameras.append(camera)
    return cameras


def coupled_rendered_face_height_cameras(faces,cameras,*,maximum_zoom=2.4):
    """Match selected DECODED heads, retaining native roll and legibility.

    COCO nose/eyes/ears RMS is useful for independent head orientation, but
    is not interchangeable with the actual displayed face-box height. This
    correction uses two explicitly selected encoded face boxes and their
    exact native cameras. It never picks the largest face, changes the actor,
    phases or roll, or shrinks either head to obtain a cheap residual.

    The caller must bind each measured face to the selected native parent
    before calling. This geometry operation does not establish that binding,
    named identity, body ownership or action continuity. It may be used once
    on a finite declared diagnostic inventory, not as an NN/re-render loop.
    """
    from .ive import AspectRatio
    from .render import (EDITORIAL_TRANSFORM_EDGE_MARGIN,EditorialTransformKeyframe,
        _rotated_crop_is_inside_frame)
    if (not isinstance(faces,(tuple,list)) or len(faces)!=2
            or not isinstance(cameras,(tuple,list)) or len(cameras)!=2
            or type(maximum_zoom) not in (int,float) or not math.isfinite(maximum_zoom)
            or not 1<=maximum_zoom<=2.4):
        raise ValueError('decoded face correction needs one finite selected two-head camera pair')
    centers=[];sizes=[];displayed=[];rolls=[];targets=[]
    for face,camera in zip(faces,cameras,strict=True):
        if not isinstance(face,dict):
            raise ValueError('decoded face correction needs measured face boxes')
        box=np.asarray(face.get('box_xywh'),float)
        values=np.asarray(camera,float)
        if (box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[:2]<0)
                or np.any(box[2:]<=0) or np.any(box[:2]+box[2:]>1)
                or values.shape!=(4,) or not np.isfinite(values).all()):
            raise ValueError('decoded face correction needs actual finite contained face boxes and cameras')
        z,x,y,roll=map(float,values)
        if not covered_roll_focal_camera(values,maximum_zoom=maximum_zoom):
            raise ValueError('decoded face correction cannot start from an uncovered native camera')
        center=box[:2]+box[2:]/2
        rotated_native=np.array((x,y))+(center-.5)/z
        centers.append(_rotated_point(rotated_native,-roll))
        sizes.append(float(box[3]*9/16/z))
        displayed.append(float(box[3]*9/16))
        rolls.append(roll);targets.append(center)
    low=max(displayed);high=maximum_zoom*min(sizes)
    if high+1e-12<low:return None
    # The same production margin is retained; infeasibility is not repaired
    # by changing rollout limits, inventing fill or clamping one camera.
    def polygon(extent):
        return _target_polygon(centers,sizes,rolls,extent,(1920,1080),
            edge_margin=EDITORIAL_TRANSFORM_EDGE_MARGIN)
    if not polygon(high):return None
    if polygon(low):extent=low
    else:
        for _ in range(32):
            middle=(low+high)/2
            if polygon(middle):high=middle
            else:low=middle
        extent=high
    target=_closest_target(polygon(extent),np.mean(targets,axis=0))
    result=[]
    for center,size,roll in zip(centers,sizes,rolls,strict=True):
        zoom=extent/size
        pan=np.asarray(_rotated_point(center,roll))+(np.array((.5,.5))-target)/zoom
        camera=[float(zoom),float(pan[0]),float(pan[1]),float(roll)]
        try:EditorialTransformKeyframe(0.,*camera)
        except ValueError:return None
        if not _rotated_crop_is_inside_frame(zoom=camera[0],center_x=camera[1],center_y=camera[2],
                roll_degrees=camera[3],aspect_ratio=AspectRatio.LANDSCAPE):return None
        result.append(camera)
    return result
