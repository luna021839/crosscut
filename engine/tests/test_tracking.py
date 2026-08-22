from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from crosscut.ive import IVEMember
from crosscut.tracking import (
    AnchoredTrackerConfig,
    MemberAnchor,
    TrackerUpdate,
    TrackingFailure,
    TrackingState,
    available_opencv_trackers,
    create_opencv_tracker_backend,
    track_anchored_member,
)


class ScriptedBackend:
    """Deterministic backend that also records forbidden post-loss updates."""

    name = "scripted"

    def __init__(
        self,
        updates: list[TrackerUpdate],
        *,
        initialize_success: bool = True,
    ) -> None:
        self.updates = list(updates)
        self.initialize_success = initialize_success
        self.initialized_box: tuple[float, float, float, float] | None = None
        self.update_calls = 0

    def initialize(
        self, frame: np.ndarray, bbox_xywh: tuple[float, float, float, float]
    ) -> bool:
        self.initialized_box = bbox_xywh
        return self.initialize_success

    def update(self, frame: np.ndarray) -> TrackerUpdate:
        self.update_calls += 1
        if not self.updates:
            raise AssertionError("tracker was updated after the scripted track ended")
        return self.updates.pop(0)


class SmokeTracker:
    def __init__(self, *, update_success: bool = True) -> None:
        self.update_success = update_success

    def init(self, _frame: np.ndarray, _box: tuple[int, int, int, int]) -> None:
        return None

    def update(
        self,
        _frame: np.ndarray,
    ) -> tuple[bool, tuple[int, int, int, int]]:
        return (self.update_success, (24, 24, 40, 40))


def subject_frame(
    box_xyxy: tuple[float, float, float, float], *, size: int = 100
) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    left, top, right, bottom = box_xyxy
    frame[
        int(round(top * size)) : int(round(bottom * size)),
        int(round(left * size)) : int(round(right * size)),
    ] = 255
    return frame


def pixel_box(box_xyxy: tuple[float, float, float, float], size: int = 100):
    left, top, right, bottom = box_xyxy
    return (
        left * size,
        top * size,
        (right - left) * size,
        (bottom - top) * size,
    )


class AnchoredTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor_box = (0.10, 0.20, 0.30, 0.80)
        self.anchor = MemberAnchor(
            frame_index=0,
            bbox_xyxy=self.anchor_box,
            member="가을",
            source_id="music-bank",
        )

    def test_successful_continuous_track_produces_timed_crop_keyframes(self) -> None:
        moved_once = (0.15, 0.20, 0.35, 0.80)
        moved_twice = (0.20, 0.20, 0.40, 0.80)
        backend = ScriptedBackend(
            [
                TrackerUpdate(True, pixel_box(moved_once), confidence=0.96),
                TrackerUpdate(True, pixel_box(moved_twice), confidence=0.94),
            ]
        )
        frames = [
            (0, subject_frame(self.anchor_box)),
            (1, subject_frame(moved_once)),
            (2, subject_frame(moved_twice)),
        ]

        result = track_anchored_member(
            frames,
            anchor=self.anchor,
            fps=25.0,
            backend=backend,
        )

        self.assertIs(result.member, IVEMember.GAEUL)
        self.assertFalse(result.is_terminally_lost)
        self.assertEqual(result.backend_name, "scripted")
        self.assertEqual(
            [item.state for item in result.observations],
            [TrackingState.TRACKED] * 3,
        )
        self.assertAlmostEqual(result.observations[2].timestamp_seconds, 0.08)
        self.assertEqual(backend.initialized_box, pixel_box(self.anchor_box))

        keyframes = result.to_dynamic_crop_keyframes(minimum_confidence=0.8)
        self.assertEqual([item.frame_index for item in keyframes], [0, 1, 2])
        self.assertAlmostEqual(keyframes[2].center[0], 0.30)
        self.assertAlmostEqual(keyframes[2].size[1], 0.60)

        camera_observations = result.to_virtual_camera_observations()
        self.assertEqual(len(camera_observations), 3)
        self.assertIs(camera_observations[2].member, IVEMember.GAEUL)
        self.assertFalse(camera_observations[2].occluded)

    def test_backend_failure_holds_briefly_then_stays_lost_without_reentry(self) -> None:
        backend = ScriptedBackend(
            [
                TrackerUpdate(False),
                # This tempting later box must never be consumed automatically.
                TrackerUpdate(True, pixel_box((0.70, 0.20, 0.90, 0.80)), 1.0),
            ]
        )
        frames = [(index, subject_frame(self.anchor_box)) for index in range(5)]

        result = track_anchored_member(
            frames,
            anchor=self.anchor,
            fps=30.0,
            backend=backend,
            config=AnchoredTrackerConfig(hold_frames=2),
        )

        self.assertEqual(
            [item.state for item in result.observations],
            [
                TrackingState.TRACKED,
                TrackingState.HOLDING,
                TrackingState.HOLDING,
                TrackingState.LOST,
                TrackingState.LOST,
            ],
        )
        self.assertIs(result.terminal_failure, TrackingFailure.BACKEND_UPDATE)
        self.assertEqual(backend.update_calls, 1)
        self.assertEqual(
            [item.frame_index for item in result.to_dynamic_crop_keyframes()], [0]
        )
        self.assertEqual(
            [
                item.frame_index
                for item in result.to_dynamic_crop_keyframes(include_holding=True)
            ],
            [0, 1, 2],
        )
        held_camera = result.to_virtual_camera_observations(include_holding=True)
        self.assertTrue(held_camera[1].occluded)

    def test_scene_change_is_immediately_lost_and_never_updates_new_shot(self) -> None:
        backend = ScriptedBackend(
            [TrackerUpdate(True, pixel_box(self.anchor_box), confidence=1.0)]
        )
        frames = [
            (0, subject_frame(self.anchor_box)),
            (1, np.full((100, 100, 3), 80, dtype=np.uint8)),
            (2, subject_frame((0.70, 0.20, 0.90, 0.80))),
        ]

        result = track_anchored_member(
            frames,
            anchor=self.anchor,
            fps=30.0,
            backend=backend,
            scene_change_frames={1},
        )

        self.assertEqual(
            [item.state for item in result.observations],
            [TrackingState.TRACKED, TrackingState.LOST, TrackingState.LOST],
        )
        self.assertIs(result.terminal_failure, TrackingFailure.SCENE_CHANGE)
        self.assertEqual(backend.update_calls, 0)

    def test_large_jump_fails_confidence_gate_and_cannot_switch_members(self) -> None:
        wrong_member_box = (0.70, 0.20, 0.90, 0.80)
        backend = ScriptedBackend(
            [
                TrackerUpdate(True, pixel_box(wrong_member_box), confidence=0.99),
                TrackerUpdate(True, pixel_box(self.anchor_box), confidence=0.99),
            ]
        )
        frames = [
            (0, subject_frame(self.anchor_box)),
            (1, subject_frame(wrong_member_box)),
            (2, subject_frame(self.anchor_box)),
            (3, subject_frame(self.anchor_box)),
        ]

        result = track_anchored_member(
            frames,
            anchor=self.anchor,
            fps=30.0,
            backend=backend,
            config=AnchoredTrackerConfig(hold_frames=1, maximum_center_jump=0.15),
        )

        self.assertEqual(
            [item.state for item in result.observations],
            [
                TrackingState.TRACKED,
                TrackingState.HOLDING,
                TrackingState.LOST,
                TrackingState.LOST,
            ],
        )
        self.assertIs(result.terminal_failure, TrackingFailure.CONFIDENCE_GATE)
        self.assertEqual(backend.update_calls, 1)

    def test_backend_confidence_drop_is_terminal_even_if_box_looks_plausible(self) -> None:
        backend = ScriptedBackend(
            [
                TrackerUpdate(True, pixel_box(self.anchor_box), confidence=0.10),
                TrackerUpdate(True, pixel_box(self.anchor_box), confidence=0.99),
            ]
        )
        frames = [(index, subject_frame(self.anchor_box)) for index in range(3)]

        result = track_anchored_member(
            frames,
            anchor=self.anchor,
            fps=30.0,
            backend=backend,
            config=AnchoredTrackerConfig(hold_frames=0),
        )

        self.assertEqual(
            [item.state for item in result.observations],
            [TrackingState.TRACKED, TrackingState.LOST, TrackingState.LOST],
        )
        self.assertIs(result.terminal_failure, TrackingFailure.CONFIDENCE_GATE)
        self.assertEqual(backend.update_calls, 1)

    def test_missing_anchor_frame_is_rejected(self) -> None:
        backend = ScriptedBackend([])
        with self.assertRaisesRegex(ValueError, "anchor frame"):
            track_anchored_member(
                [(1, subject_frame(self.anchor_box))],
                anchor=self.anchor,
                fps=30.0,
                backend=backend,
            )

    def test_opencv_availability_query_is_environment_independent(self) -> None:
        names = available_opencv_trackers()
        self.assertIsInstance(names, tuple)
        self.assertTrue(set(names).issubset({"CSRT", "KCF", "MIL"}))

    def test_broken_preferred_factory_falls_through_to_working_backend(self) -> None:
        def broken_factory() -> object:
            raise RuntimeError("compiled tracker is unavailable")

        def factories(name: str):  # type: ignore[no-untyped-def]
            if name == "CSRT":
                return (broken_factory,)
            if name == "KCF":
                return (lambda: SmokeTracker(),)
            return ()

        with patch(
            "crosscut.tracking._opencv_tracker_factories",
            side_effect=factories,
        ):
            self.assertEqual(available_opencv_trackers(), ("KCF",))
            backend = create_opencv_tracker_backend(("CSRT", "KCF", "MIL"))

        self.assertEqual(backend.name, "KCF")

    def test_factory_with_failed_real_update_is_not_advertised(self) -> None:
        with patch(
            "crosscut.tracking._opencv_tracker_factories",
            return_value=(lambda: SmokeTracker(update_success=False),),
        ):
            self.assertEqual(available_opencv_trackers(), ())


if __name__ == "__main__":
    unittest.main()
