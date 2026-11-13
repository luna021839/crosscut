"""Numeric cache/clock contracts only; no fabricated detector or NN outputs."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from crosscut.group_cached_native_observations import (
    observation_from_cached_descriptor, load_cached_native_inventory,
)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.timeline import ShotType
from crosscut.visual_feature_cache import CanonicalFrameIdentity, DescriptorRequestIdentity
from crosscut.visual_score import FrameVisualDescriptor


def inputs():
    clock = VerifiedRenderFrameClock(Fraction(30000, 1001), 500, "a"*64)
    request = DescriptorRequestIdentity(CanonicalFrameIdentity(90, 30000, 1001),
        None, "b"*64, None, True, True)
    descriptor = FrameVisualDescriptor(.8, .6, .3, ShotType.MEDIUM, 0, (0.,)*96,
        pose_embedding=(0.,)*96, pose_backend="structure_hog_fallback",
        shot_scale_basis="unknown_no_subject_geometry")
    return dict(source_index=2, source_sha256="c"*64, source_clock=clock,
        request=request, descriptor=descriptor, producer_fingerprint="d"*64,
        record_path=Path("record.json"))


def test_unknown_face_and_hog_remain_partial_observation_not_a_pose():
    row = observation_from_cached_descriptor(**inputs())
    assert row.native_frame_index == 90
    assert row.native_seconds == float(Fraction(90*1001, 30000))
    assert row.pose is None and row.pose_confidence == 0 and row.face_bbox is None
    assert row.pose_is_face_owned is False


def test_numeric_pose_still_does_not_prove_face_ownership():
    args = inputs()
    # Typed synthetic numbers exercise transfer semantics, not inference.
    args["descriptor"] = replace(args["descriptor"], pose_embedding=(.2,)*60,
        pose_confidence=.7, pose_backend="opencv_zoo_mediapipe_pose_2023mar")
    row = observation_from_cached_descriptor(**args)
    assert row.pose == (.2,)*60 and row.pose_confidence == .7
    assert not row.pose_is_face_owned


def test_motion_owns_exact_following_native_frame():
    args = inputs()
    args["request"] = replace(args["request"],
        following=CanonicalFrameIdentity(93, 30000, 1001), following_working_frame_sha256="e"*64)
    args["descriptor"] = replace(args["descriptor"], motion=(.03, -.01))
    row = observation_from_cached_descriptor(**args)
    assert row.following_native_frame == 93 and row.image_motion == (.03, -.01)
    args["request"] = inputs()["request"]
    assert observation_from_cached_descriptor(**args).image_motion is None


@pytest.mark.parametrize("frame,rate", [(500, Fraction(30000,1001)), (90,Fraction(30,1))])
def test_other_clock_or_out_of_coverage_rejected(frame, rate):
    args = inputs()
    args["request"] = replace(args["request"], primary=CanonicalFrameIdentity(frame, rate.numerator, rate.denominator))
    with pytest.raises(ValueError):
        observation_from_cached_descriptor(**args)


def test_backward_motion_pair_rejected():
    args = inputs()
    args["request"] = replace(args["request"], following=CanonicalFrameIdentity(89,30000,1001),
        following_working_frame_sha256="e"*64)
    with pytest.raises(ValueError):
        observation_from_cached_descriptor(**args)


@pytest.mark.parametrize("bad", [-1, True, 64])
def test_invalid_source_rejected_even_before_empty_inventory_read(bad):
    with pytest.raises(ValueError):
        load_cached_native_inventory(None, source_index=bad, source_clock=inputs()["source_clock"])


def test_unverified_source_clock_rejected_before_any_inventory_read():
    with pytest.raises(ValueError):
        load_cached_native_inventory(None, source_index=0, source_clock=None)
