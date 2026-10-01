"""Source-local ROI/pose/head ownership, not named identity or head visibility.

Keep competing hypotheses instead of NMS-merging real overlapping dancers.
Arms may leave a tight detector box but must lie in the actual padded pose
support. Optical texture agreement is not a replacement for these checks.
"""
from dataclasses import dataclass, replace
import math
import hashlib
from pathlib import Path

import numpy as np

from .cut_anchored_containment import _iou
from .group_joint_formation import ObservedFormationPerson


@dataclass(frozen=True,slots=True)
class OwnedPersonObservation(ObservedFormationPerson):
    pose_crop_xywh: tuple | None = None
    source_pixel_to_pose_input_affine: tuple = ()
    raw_pose_xyc: tuple = ()
    pose_reason: str = 'pose_unavailable'
    candidate_head_observation_ids: tuple[int,...] = ()
    unique_head_observation_id: int | None = None
    competing_person_observation_ids: tuple[int,...] = ()
    body_ownership_state: str = 'unknown'
    part_states: tuple[tuple[str,str],...] = ()
    face_anchor_consistent: bool | None = None
    upper_body_ownership_state: str = 'unknown'

    def __post_init__(self):
        ObservedFormationPerson.__post_init__(self)
        if any(state not in ('observed','ambiguous','unknown') for state in
                (self.body_ownership_state, self.upper_body_ownership_state)):
            raise ValueError('person ownership must retain an explicit observed/ambiguous/unknown state')
        for box in (() if self.pose_crop_xywh is None else (self.pose_crop_xywh,)):
            if (not isinstance(box,tuple) or len(box)!=4
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in box)
                    or min(box[2:])<=0):
                raise ValueError('pose support must preserve the actual unclipped padded crop')
        if (len(set(self.candidate_head_observation_ids))!=len(self.candidate_head_observation_ids)
                or self.unique_head_observation_id is not None
                    and self.candidate_head_observation_ids!=(self.unique_head_observation_id,)
                or any(state not in ('observed','ambiguous','unknown') for _,state in self.part_states)):
            raise ValueError('head and part ownership hypotheses are inconsistent')
        if (not isinstance(self.raw_pose_xyc,tuple) or len(self.raw_pose_xyc) not in (0,17)
                or any(not isinstance(row,tuple) or len(row)!=3
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in row) for row in self.raw_pose_xyc)
                or not isinstance(self.source_pixel_to_pose_input_affine,tuple)
                or self.source_pixel_to_pose_input_affine and (len(self.source_pixel_to_pose_input_affine)!=2
                    or any(not isinstance(row,tuple) or len(row)!=3
                        or any(type(v) not in (int,float) or not math.isfinite(v) for v in row)
                        for row in self.source_pixel_to_pose_input_affine))):
            raise ValueError('owned pose must retain finite original output and exact crop-coordinate affine')


def _inside(point,box,*,padding=0.):
    x,y,w,h=box
    return x-padding<=point[0]<=x+w+padding and y-padding<=point[1]<=y+h+padding


def torso_owned_by_detector(person):
    """Reject an adjacent person's skeleton without clipping legitimate arms."""
    if len(person.upper_body_xyc)!=8:
        return False
    torso=tuple(person.upper_body_xyc[n] for n in (0,1,6,7))
    if any(confidence<.3 or not _inside((x,y),(0.,0.,1.,1.))
            for x,y,confidence in torso):
        return False
    centers=(np.mean(np.asarray(torso[:2])[:,:2],axis=0),
        np.mean(np.asarray(torso[2:])[:,:2],axis=0))
    return all(_inside(center,person.box_xywh,padding=.005) for center in centers)


def _shoulder_anchor_owned(person):
    """A close-up anchor, NOT full torso/person/head-extent ownership.

    Hips outside native pixels remain outside/unknown. Visible shoulders plus
    a separately detected unique face can still own actual upper-body parts.
    """
    if (person.pose_reason != 'observed' or person.pose_crop_xywh is None
            or not person.source_pixel_to_pose_input_affine or len(person.upper_body_xyc) != 8
            or len(person.raw_pose_xyc) != 17 or person.face_anchor_consistent is False):
        return False
    nose = person.raw_pose_xyc[0]
    shoulders = person.upper_body_xyc[:2]
    if any(p[2] < .3 or not _inside(p[:2], (0., 0., 1., 1.))
            or not _inside(p[:2], person.pose_crop_xywh) for p in (*shoulders, nose)):
        return False
    center = np.mean(np.asarray(shoulders)[:, :2], axis=0)
    return _inside(center, person.box_xywh, padding=.005) and _inside(nose[:2], person.box_xywh, padding=.005)


def ownership_policy_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def trackable_person_ownership(person):
    if isinstance(person,OwnedPersonObservation):
        return (not person.competing_person_observation_ids
            and person.body_ownership_state != 'ambiguous' and person.upper_body_ownership_state != 'ambiguous'
            and 'observed' in (person.body_ownership_state, person.upper_body_ownership_state))
    # Legacy synthetic/diagnostic callers have no owned-head authority. Still
    # reject a skeleton outside its claimed detector before LK assignment.
    return torso_owned_by_detector(person)


def bind_person_ownership(people,heads,frame_size_wh):
    """Resolve only explicit support/anchor facts, preserving all observations."""
    size=np.asarray(frame_size_wh,dtype=float)
    duplicates={person.observation_id:set() for person in people}
    for i,left in enumerate(people):
        for right in people[i+1:]:
            if (len(left.upper_body_xyc)!=8 or len(right.upper_body_xyc)!=8
                    or _iou(left.box_xywh,right.box_xywh)<=0):
                continue
            a,b=np.asarray(left.upper_body_xyc),np.asarray(right.upper_body_xyc)
            common=tuple(n for n in (0,1,6,7) if min(a[n,2],b[n,2])>=.3)
            same_torso = False
            if len(common) >= 3:
                distances=np.linalg.norm((a[list(common),:2]-b[list(common),:2])*size,axis=1)
                scale=min(float(np.linalg.norm((a[:2,:2].mean(0)-a[6:,:2].mean(0))*size)),
                    float(np.linalg.norm((b[:2,:2].mean(0)-b[6:,:2].mean(0))*size)))
                same_torso = float(np.median(distances)) <= max(3., .06*scale)
            same_upper = False
            if _shoulder_anchor_owned(left) and _shoulder_anchor_owned(right):
                anchors_a = np.asarray((*left.upper_body_xyc[:2], left.raw_pose_xyc[0]))
                anchors_b = np.asarray((*right.upper_body_xyc[:2], right.raw_pose_xyc[0]))
                shoulder_scale = min(float(np.linalg.norm((a[1, :2]-a[0, :2])*size)),
                    float(np.linalg.norm((b[1, :2]-b[0, :2])*size)))
                distances = np.linalg.norm((anchors_a[:, :2]-anchors_b[:, :2])*size, axis=1)
                same_upper = float(distances.max()) <= max(3., .06*shoulder_scale)
            if same_torso or same_upper:
                duplicates[left.observation_id].add(right.observation_id)
                duplicates[right.observation_id].add(left.observation_id)
    head_candidates={person.observation_id:[] for person in people}
    owners={head.observation_id:[] for head in heads}
    for person in people:
        if len(person.raw_pose_xyc)!=17 or person.pose_crop_xywh is None:
            continue
        x,y,confidence=person.raw_pose_xyc[0]
        if confidence<.3:
            continue
        for head in heads:
            fx,fy,fw,fh=head.box_xywh
            distance=math.hypot((x-(fx+fw/2))/fw,(y-(fy+fh/2))/fh)
            if distance<=.8 and _inside((x,y),person.pose_crop_xywh):
                head_candidates[person.observation_id].append(head.observation_id)
                owners[head.observation_id].append(person.observation_id)
    result=[]
    for person in people:
        competing=tuple(sorted(duplicates[person.observation_id]))
        body_valid=(person.pose_reason=='observed' and person.pose_crop_xywh is not None
            and bool(person.source_pixel_to_pose_input_affine) and torso_owned_by_detector(person)
            and all(_inside(person.upper_body_xyc[n][:2],person.pose_crop_xywh) for n in (0,1,6,7)))
        body_state='ambiguous' if competing and body_valid else 'observed' if body_valid else 'unknown'
        candidates=tuple(head_candidates[person.observation_id])
        unique_face = len(candidates) == 1 and len(owners[candidates[0]]) == 1
        upper_valid = _shoulder_anchor_owned(person)
        upper_state = ('ambiguous' if upper_valid and (competing or candidates and not unique_face) else
            'observed' if upper_valid and unique_face else 'unknown')
        head_id=(candidates[0] if unique_face and not competing
            and (body_state=='observed' or upper_state=='observed')
            and person.face_anchor_consistent is not False else None)
        states=[('torso',body_state),('shoulder_anchor',upper_state),('face_roi','observed' if head_id is not None else
            'ambiguous' if candidates else 'unknown'),('head_extent','unknown'),('head_motion','unknown')]
        for name,joints in (('left_arm',(0,2,4)),('right_arm',(1,3,5))):
            observed=(not competing and (body_state=='observed' or upper_state=='observed')
                and all(person.upper_body_xyc[n][2]>=.3
                and _inside(person.upper_body_xyc[n][:2],person.pose_crop_xywh)
                and _inside(person.upper_body_xyc[n][:2],(0.,0.,1.,1.)) for n in joints))
            states.append((name,'observed' if observed else 'ambiguous' if
                'ambiguous' in (body_state, upper_state) else 'unknown'))
        result.append(replace(person,candidate_head_observation_ids=candidates,unique_head_observation_id=head_id,
            competing_person_observation_ids=competing,body_ownership_state=body_state,
            upper_body_ownership_state=upper_state,part_states=tuple(states)))
    return tuple(result),owners
