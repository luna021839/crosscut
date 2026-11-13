"""Real continuity producer with small synthetic descriptors; no media claim."""
from pathlib import Path

from crosscut.direct_member_rendered_continuity import evaluate_direct_member_rendered_continuity
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor


def member_continuity_report(plan=None):
    descriptor = FrameVisualDescriptor(
        quality_score=.95, composition_score=.95, shot_scale=.5, shot_type=ShotType.MEDIUM,
        structure_embedding=(.2, .1, .9), face_count=1, primary_face_bbox=(.3, .2, .2, .48),
        pose_embedding=(.2, .1, .9), pose_confidence=.95, pose_backend="synthetic_verified_pose",
        silhouette_embedding=(.2, .1, .9), silhouette_bbox_xywh=(.3, .2, .2, .48),
        silhouette_centroid_xy=(.4, .44), silhouette_area_ratio=.096, silhouette_confidence=.95,
    )
    plan = plan or {"fps": 30., "output_frame_count": 120,
        "segments": [{"source_index": index, "duration": 1.} for index in range(4)],
        "transitions": [{"kind": "hard_cut", "frame_count": 0} for _ in range(3)]}
    return evaluate_direct_member_rendered_continuity(
        Path("synthetic-member.mp4"), plan, descriptor_provider=lambda _seconds: descriptor,
    ).to_dict()
