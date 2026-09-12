"""The real joint producers must agree with the rendered grammar consumer."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.editorial_transform_events import rendered_transform_event_records
from crosscut.group_boundary_grammar_adapter import _validated_transforms
from crosscut.ive import AspectRatio
from crosscut.joint_editorial_motion_overlay import merge_joint_interior_motion_overlay
from crosscut.joint_visual_candidate_adapter import _joint_editorial_motion_plan
from crosscut.render import (
    EditorialTransformKeyframe as Keyframe,
    EditorialTransformTrack as Track,
    RenderPlan,
    RenderSegment,
    RenderTransition,
)
from crosscut.editorial_motion import EditorialMotionPlan, EditorialMotionSegment


def motion_plan(tracks, *, pulse=False):
    return EditorialMotionPlan(
        segments=tuple(
            EditorialMotionSegment(
                index, 0, track,
                ("reference_calibrated_same_source_interior_zoom_pulse",)
                if pulse else (),
            )
            for index, track in enumerate(tracks)
        ),
        boundary_match_count=0,
        zoom_event_count=int(pulse),
        pan_event_count=0,
        roll_event_count=0,
        blocked_dissolve_segment_count=0,
        interior_zoom_pulse_count=int(pulse),
    )


def render_plan(segments, fps=10.0):
    return RenderPlan(
        sources=(Path("source.mp4"),),
        segments=tuple(segments),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path("output.mp4"),
        master_audio_start=0.0,
        width=640,
        height=360,
        fps=fps,
    )


def validate(plan, motion):
    return _validated_transforms(
        plan=plan,
        editorial_motion=motion,
        eligible_to_request=(0,),
        dynamic_crop_version=None,
        dynamic_crop_motion_v3=None,
    )


class JointMotionEventContractTests(unittest.TestCase):
    def test_constant_tracks_do_not_evaluate_every_output_frame(self):
        track = Track((Keyframe(0.0, 1.2), Keyframe(120.0, 1.2)))
        segments = (RenderSegment(0, 0.0, 120.0, 0.0, editorial_transform=track),)
        with patch.object(Track, "evaluate", side_effect=AssertionError("unneeded sample")):
            self.assertEqual(rendered_transform_event_records(segments, 60.0),
                             {"zoom": (), "pan": (), "roll": ()})

    def test_three_components_share_the_same_first_visible_sample(self):
        track = Track((Keyframe(0.0, 1.2), Keyframe(0.9, 1.21, 0.51, 0.5, 0.1)))
        segments = (RenderSegment(0, 0.0, 1.0, 0.0, editorial_transform=track),)
        original = Track.evaluate
        calls = []
        def observed(value, seconds):
            calls.append(seconds)
            return original(value, seconds)
        with patch.object(Track, "evaluate", observed):
            result = rendered_transform_event_records(segments, 10.0)
        self.assertEqual(result, {name: ((0.1, 0),) for name in ("zoom", "pan", "roll")})
        self.assertEqual(calls, [0.1])

    def test_split_rising_pulse_keeps_one_authored_pulse_and_exact_event_count(self):
        # One 0.6 -> 1.4 -> 2.2 pulse, sliced BEFORE its peak. The grammar
        # contract counts frame-visible trajectory targets, not authored pulses.
        tracks = (
            Track((Keyframe(0.0, 1.0), Keyframe(0.6, 1.0),
                   Keyframe(0.9, 1.024375))),
            Track((Keyframe(0.0, 1.0325), Keyframe(0.4, 1.065),
                   Keyframe(1.2, 1.0), Keyframe(1.9, 1.0))),
        )
        segments = (
            RenderSegment(0, 0.0, 1.0, 0.0),
            RenderSegment(0, 1.0, 2.0, 1.0),
        )
        result = merge_joint_interior_motion_overlay(
            provider_plan=motion_plan((None, None)),
            provider_segments=segments,
            transitions=(RenderTransition(
                tier="D", reason="joint_decoded_same_source_same_shot_hold"
            ),),
            proposal_plan=motion_plan(tracks, pulse=True),
            output_fps=10.0,
        )
        self.assertTrue(result.applied)
        self.assertEqual(result.applied_pulse_count, 1)
        self.assertEqual(result.plan.interior_zoom_pulse_count, 1)
        evidence = validate(render_plan(result.segments), result.plan)
        self.assertEqual(result.plan.zoom_event_count, 2)
        self.assertEqual(evidence.zoom_event_times_seconds, (0.7, 1.1))
        for actual, original in zip(result.segments, tracks, strict=True):
            self.assertIs(actual.editorial_transform, original)

    def test_constant_provider_crop_is_not_a_visible_zoom_event(self):
        tracks = tuple(
            Track((Keyframe(0.0, 1.1), Keyframe(0.9, 1.1)))
            for _ in range(2)
        )
        segments = tuple(
            RenderSegment(0, float(index), 1.0, float(index),
                          editorial_transform=track)
            for index, track in enumerate(tracks)
        )
        transform = SimpleNamespace(crop_xywh=(0.05, 0.05, 0.9, 0.9),
                                    rotation_degrees=0.0)
        proof = SimpleNamespace(
            boundary_proofs=(SimpleNamespace(outgoing_transform=transform,
                                            incoming_transform=transform),),
            segment_path_proofs=tuple(
                SimpleNamespace(editorial_transform_track=track,
                                candidate=SimpleNamespace(segment_index=index),
                                source_index=0)
                for index, track in enumerate(tracks)
            ),
        )
        # This is the actual joint materialization producer, not a handwritten
        # metadata patch at the QA boundary.
        motion = _joint_editorial_motion_plan(proof, segments, fps=10.0)
        evidence = validate(render_plan(segments), motion)
        self.assertEqual(motion.boundary_match_count, 1)
        self.assertEqual(motion.zoom_event_count, 0)
        self.assertEqual(evidence.zoom_event_times_seconds, ())

    def test_forged_motion_count_still_fails(self):
        track = Track((Keyframe(0.0, 1.0), Keyframe(0.9, 1.05)))
        plan = render_plan((RenderSegment(0, 0.0, 1.0, 0.0,
                                         editorial_transform=track),))
        with self.assertRaisesRegex(ValueError, "zoom event count"):
            validate(plan, replace(motion_plan((track,)), zoom_event_count=7))


if __name__ == "__main__":
    unittest.main()
