"""Measured local articulation paths, not a rigid transform of a whole person.

The native body detector may include moving arms, background and another
person in one rectangle. A single affine fit of that rectangle is therefore
not a source-native identity tracker. Here feature patches are attached to
the actually observed joints and checked in both pixel-flow directions.
These diagnostic observations do not establish cross-camera or named identity.
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from .group_joint_person_ownership import trackable_person_ownership, OwnedPersonObservation


@dataclass(frozen=True, slots=True)
class NativeJointPixelFlow:
    observation_id: int
    # joint index, original x/y, measured predicted x/y, FB error, features
    joints: tuple[tuple[int, float, float, float, float, float, int], ...]

    def to_dict(self):
        return dict(contract='source_native_joint_local_forward_backward_pixel_flow_v1',
            **asdict(self), rigid_whole_person_transform=False,
            cross_camera_correspondence_verified=False, quality_approved=False)


def native_joint_pixel_flows(previous_pixels, current_pixels, people):
    """One pair of LK calls for all bodies, with separate joint-owned patches.

    Non-textured, off-screen, missing and inconsistent features stay absent.
    No observed joint is clipped into visibility or replaced by zero motion.
    The consumer must still resolve competing current-body assignments.
    """
    if previous_pixels.shape != current_pixels.shape:
        return tuple(None for _ in people)
    previous = cv2.cvtColor(previous_pixels, cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(current_pixels, cv2.COLOR_BGR2GRAY)
    height, width = previous.shape
    feature_rows, groups = [], []
    for owner, person in enumerate(people):
        if not trackable_person_ownership(person):
            continue
        for joint, (x, y, confidence) in enumerate(person.upper_body_xyc):
            if confidence < .3 or not 0 <= x < 1 or not 0 <= y < 1:
                continue
            radius = max(6, min(20, int(min(width, height)*.015)))
            xx, yy = int(round(x*width)), int(round(y*height))
            left, top = max(0, xx-radius), max(0, yy-radius)
            right, bottom = min(width, xx+radius+1), min(height, yy+radius+1)
            patch = previous[top:bottom, left:right]
            if min(patch.shape, default=0) < 3:
                continue
            corners = cv2.goodFeaturesToTrack(patch, maxCorners=8, qualityLevel=.012,
                minDistance=3., blockSize=3)
            if corners is None or len(corners) < 3:
                continue
            points = corners.reshape(-1, 2)+np.asarray((left, top), dtype=np.float32)
            start = len(feature_rows)
            feature_rows.extend(points)
            groups.append((owner, joint, x, y, start, len(feature_rows)))
    if not feature_rows:
        return tuple(None for _ in people)
    original = np.asarray(feature_rows, dtype=np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(previous, current, original, None, **options)
    if forward is None or forward_status is None:
        return tuple(None for _ in people)
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **options)
    if backward is None or backward_status is None:
        return tuple(None for _ in people)
    positions, moved, returned = original.reshape(-1, 2), forward.reshape(-1, 2), backward.reshape(-1, 2)
    errors = np.linalg.norm(returned-positions, axis=1)
    valid = (forward_status.reshape(-1).astype(bool) & backward_status.reshape(-1).astype(bool)
        & np.isfinite(errors) & (errors <= 1.25)
        & np.all(np.isfinite(moved), axis=1)
        & (moved[:, 0] >= 0) & (moved[:, 0] < width)
        & (moved[:, 1] >= 0) & (moved[:, 1] < height))
    rows = [[] for _ in people]
    for owner, joint, x, y, start, end in groups:
        indices = np.flatnonzero(valid[start:end])+start
        if len(indices) < 3:
            continue
        shifts = moved[indices]-positions[indices]
        shift = np.median(shifts, axis=0)
        # Different motion inside one small patch cannot become one joint's
        # motion just because its arithmetic average happens to look plausible.
        coherent = np.linalg.norm(shifts-shift, axis=1) <= 2.
        if int(coherent.sum()) < 3 or coherent.mean() < .7:
            continue
        indices = indices[coherent]
        shift = np.median(moved[indices]-positions[indices], axis=0)
        dx, dy = float(shift[0]/width), float(shift[1]/height)
        if math.hypot(dx, dy) > .08 or not 0 <= x+dx < 1 or not 0 <= y+dy < 1:
            continue
        rows[owner].append((joint, float(x), float(y), x+dx, y+dy,
            float(np.max(errors[indices])), len(indices)))
    result = []
    for person, joints in zip(people, rows, strict=True):
        core = {row[0] for row in joints} & {0, 1, 6, 7}
        result.append(NativeJointPixelFlow(person.observation_id, tuple(joints))
            if len(joints) >= 3 and len(core) >= 2 else None)
    return tuple(result)


def native_joint_assignment_cost(flow, person, frame_size_wh):
    """Actual predicted joint -> measured same-source joint discrepancy.

    This is a local tracking policy, not calibrated cross-camera confidence.
    It never borrows the .015 margin of the single-time formation diagnostic.
    """
    if flow is None or len(person.upper_body_xyc) != 8 or not trackable_person_ownership(person):
        return None
    common = tuple(row for row in flow.joints if person.upper_body_xyc[row[0]][2] >= .3)
    if len(common) < 3 or len({row[0] for row in common} & {0, 1, 6, 7}) < 2:
        return None
    size = np.asarray(frame_size_wh, dtype=float)
    points = np.asarray(person.upper_body_xyc, dtype=float)
    shoulder = np.mean(points[:2, :2], axis=0)*size
    hip = np.mean(points[6:8, :2], axis=0)*size
    if isinstance(person, OwnedPersonObservation) and person.body_ownership_state != 'observed':
        if person.upper_body_ownership_state != 'observed' or min(points[:2, 2]) < .3:
            return None
        # Actual source-local shoulder span for a measured cropped upper body,
        # never a fabricated hip or a cross-camera identity threshold.
        scale = float(np.linalg.norm((points[1, :2]-points[0, :2])*size))
    else:
        if min(points[[0, 1, 6, 7], 2]) < .3:
            return None
        scale = float(np.linalg.norm(hip-shoulder))
    if scale < 8.:
        return None
    errors = tuple(float(np.linalg.norm((np.asarray(row[3:5])-points[row[0], :2])*size))/scale
        for row in common)
    return float(np.median(errors)) if max(errors) <= .35 else None
