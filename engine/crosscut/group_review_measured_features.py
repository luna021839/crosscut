"""Measured near-shot edge hypotheses, not actor or quality certificates."""
import math
import numpy as np

from .group_review_camera_repair import geometry_cost,shape_cost
from .group_review_native_window import cosine


def face_shape(observation):
    if observation is None or observation['descriptor'] is None: return None
    identity = observation['descriptor'].frame_local_head_id
    head = next((h for h in observation['row']['projection']['heads'] if h['observation_id'] == identity),None)
    if head is None: return None
    points = np.asarray(head['projected_landmarks_xy'],float)*np.asarray((1.,9/16))
    center = points[:2].mean(axis=0); scale = np.linalg.norm(points[0]-points[1])
    return None if scale < 1e-6 else (points-center)/scale


def face_shape_risk(a,b):
    aa,bb = face_shape(a),face_shape(b)
    return None if aa is None or bb is None else float(np.linalg.norm(aa-bb,axis=1).mean())


def local_motion(a,b,fps):
    if a is None or b is None or a['descriptor'] is None or b['descriptor'] is None: return None
    ra,rb = a['row'],b['row']
    if (ra['projection']['camera_sha256'] != rb['projection']['camera_sha256']
            or rb['output_tick'] <= ra['output_tick']): return None
    sim = cosine(a['vector'],b['vector'])
    if sim is None or sim < .25: return None
    dt = (rb['output_tick']-ra['output_tick'])/fps
    sa,sb = a['descriptor'].pose_shape,b['descriptor'].pose_shape
    if sa is not None and sb is not None:
        aa,bb = {p[0]:p[1:] for p in sa},{p[0]:p[1:] for p in sb}
        keys = sorted(set(aa)&set(bb))
        if len(keys) >= 4:
            vector = np.asarray([(np.asarray(bb[k])-aa[k])/dt for k in keys]).flatten()
            return dict(vector=vector,indices=keys,basis='owned_upper_body_shape_velocity',appearance_hypothesis=sim)
    aa,bb = face_shape(a),face_shape(b)
    if aa is None or bb is None: return None
    return dict(vector=((bb-aa)/dt).flatten(),indices=list(range(5)),
        basis='five_face_landmark_shape_velocity',appearance_hypothesis=sim)


def pair_risk(a,b,ma=None,mb=None):
    da = None if a is None else a['descriptor']; db = None if b is None else b['descriptor']
    geometry = geometry_cost(da,db); shape = shape_cost(da,db)
    if shape is None: shape = face_shape_risk(a,b)
    appearance = None if a is None or b is None else cosine(a['vector'],b['vector'])
    mutual = False
    if appearance is not None and a['appearance'] is not None and b['appearance'] is not None:
        av,bv = a['vector'],b['vector']
        forward = sorted([c for h in b['appearance']['heads']
            if (c:=cosine(av,h['embedding'])) is not None],reverse=True)
        reverse = sorted([c for h in a['appearance']['heads']
            if (c:=cosine(bv,h['embedding'])) is not None],reverse=True)
        mutual = bool(forward and reverse and abs(forward[0]-appearance) < 1e-7
            and abs(reverse[0]-appearance) < 1e-7
            and (len(forward)<2 or forward[0]-forward[1] >= .025)
            and (len(reverse)<2 or reverse[0]-reverse[1] >= .025))
    motion = None; kinetic = 0.
    if ma is not None and mb is not None and (ma['basis'],ma['indices']) == (mb['basis'],mb['indices']):
        va,vb = ma['vector'],mb['vector']; na,nb = np.linalg.norm(va),np.linalg.norm(vb)
        # Adapter boundary owns conversion of numpy scalar producer results.
        # Strict scalar contracts in the graph must not be weakened globally.
        kinetic = float(min(1.,min(na,nb)))
        motion = float(np.linalg.norm(va-vb)/(1.+max(na,nb)))
    unknown = int(geometry is None)+int(shape is None)+int(appearance is None)+int(motion is None)
    risk = ((1.2 if geometry is None else geometry)+.7*(1. if shape is None else shape)
        +.65*(1. if appearance is None else max(0.,.5-appearance))
        +.35*(.5 if motion is None else motion)+.1*unknown)
    near = min(0. if a is None else a['near'],0. if b is None else b['near'])
    choreography_motion = (ma is not None and mb is not None
        and ma['basis'] == mb['basis'] == 'owned_upper_body_shape_velocity')
    # Mouth/eye landmark deformation may be useful face-pose evidence, but it
    # is not measured arm/choreography continuity. Do not buy an authored
    # moving-dance-cut bonus with that fallback when the body is unobserved.
    matched = (near*kinetic*max(0.,min(1.,(appearance-.25)/.35))*math.exp(-2.*shape-motion)
        if choreography_motion and mutual and appearance is not None
        and shape is not None and motion is not None else 0.)
    return dict(risk=float(risk),moving_match=float(matched),face_geometry_risk=geometry,
        pose_or_face_shape_risk=shape,appearance_cosine=appearance,mutual_appearance_hypothesis=mutual,
        motion_direction_phase_risk=motion,kinetic_strength=kinetic,
        choreography_motion_observed=choreography_motion,same_actor_certified=False,
        unknown_metric_count=unknown)
