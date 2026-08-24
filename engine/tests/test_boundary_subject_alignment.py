from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import cv2

from crosscut.boundary_subject_alignment import (
    BoundarySubjectAlignmentConfig,
    BoundarySubjectObservation,
    build_boundary_subject_alignment_plan,
)
from crosscut.face_match import FaceMatchPlan, FaceMatchSegment
from crosscut.editorial_motion import EditorialMotionConfig, build_editorial_motion_plan
from crosscut.render import (
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderTransition,
    TransitionKind,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def _choice(
    segment_index: int,
    source_index: int,
    *,
    start: float,
    end: float,
    internal_cut: bool = False,
    entry_cut: bool = False,
    exit_cut: bool = False,
) -> VisualTimelineChoice:
    box = (0.42, 0.28, 0.16, 0.30)
    return VisualTimelineChoice(
        segment_index=segment_index,
        source_index=source_index,
        canonical_master_start=start,
        canonical_master_end=end,
        source_local_start=start,
        source_local_end=end,
        quality_score=0.9,
        composition_score=0.9,
        boundary_score=0.9,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"{source_index}:shot",
        entry_scene_cut_near=entry_cut,
        exit_scene_cut_near=exit_cut,
        internal_scene_cut_master_seconds=((start + end) / 2.0,) if internal_cut else (),
        face_samples_xywh=(box, box, box),
        face_detector_backends=("test_face",) * 3,
    )


def _frame(source_index: int, marker: int | None = None) -> np.ndarray:
    value = 50 + 50 * source_index if marker is None else marker
    return np.full((360, 640, 3), value, dtype=np.uint8)


def _observations(
    *,
    source0_box: tuple[float, float, float, float],
    source1_box: tuple[float, float, float, float],
    source0_roll: float | None = None,
    source1_roll: float | None = None,
):
    def provider(frame: np.ndarray) -> BoundarySubjectObservation:
        marker = int(round(float(frame.mean())))
        if marker < 20:
            return BoundarySubjectObservation()
        if marker < 75:
            return BoundarySubjectObservation(
                face_count=1,
                face_box_xywh=source0_box,
                face_head_roll_radians=source0_roll,
                face_eye_scale=source0_box[3] * 0.42,
            )
        return BoundarySubjectObservation(
            face_count=1,
            face_box_xywh=source1_box,
            face_head_roll_radians=source1_roll,
            face_eye_scale=source1_box[3] * 0.42,
        )

    return provider


class BoundarySubjectAlignmentTests(unittest.TestCase):
    def test_different_stage_backgrounds_do_not_require_orb(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )
        small = (0.45, 0.35, 0.10, 0.16)
        large = (0.44, 0.33, 0.12, 0.18)
        calls: list[tuple[int, float]] = []

        def frames(source_index: int, seconds: float) -> np.ndarray:
            calls.append((source_index, seconds))
            # Flat, unrelated frames have no ORB evidence by construction.
            return _frame(source_index)

        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("stage-a.mp4"), Path("stage-b.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            frame_provider=frames,
            observation_provider=_observations(
                source0_box=small,
                source1_box=large,
            ),
            source_resolutions=((1920, 1080), (1920, 1080)),
        )

        self.assertEqual(plan.eligible_count, 1)
        decision = plan.decisions[0]
        self.assertEqual(decision.transformed_side, "outgoing")
        self.assertTrue(decision.partial_scale_alignment)
        self.assertGreater(decision.validated_output_frame_count, 0)
        track = plan.track_for(0)
        assert track is not None
        self.assertAlmostEqual(track.final_keyframe.zoom, 1.10)
        self.assertIsNone(plan.track_for(1))
        self.assertGreaterEqual(len(calls), 14)
        self.assertFalse(plan.to_dict()["policy"]["orb_required"])

    def test_exact_face_roll_is_used_when_renderer_safe(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )
        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("a.mp4"), Path("b.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            frame_provider=lambda source_index, _seconds: _frame(source_index),
            observation_provider=_observations(
                source0_box=(0.45, 0.35, 0.10, 0.16),
                source1_box=(0.44, 0.33, 0.12, 0.18),
                source0_roll=0.0,
                source1_roll=np.deg2rad(1.0),
            ),
            source_resolutions=((3840, 2160), (3840, 2160)),
        )

        decision = plan.decisions[0]
        self.assertTrue(decision.eligible)
        self.assertTrue(decision.roll_evidence_available)
        self.assertTrue(decision.roll_applied)
        self.assertAlmostEqual(decision.outgoing_state.roll_degrees, 1.0, places=5)
        self.assertEqual(plan.roll_event_count, 1)

    def test_missing_geometry_on_one_transformed_frame_fails_closed(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )

        def frames(source_index: int, seconds: float) -> np.ndarray:
            if source_index == 1 and abs(seconds - 1.2) < 0.01:
                return _frame(source_index, marker=0)
            return _frame(source_index)

        # Incoming is smaller, so its first 14 output frames own the settle.
        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("a.mp4"), Path("b.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            frame_provider=frames,
            observation_provider=_observations(
                source0_box=(0.44, 0.33, 0.12, 0.18),
                source1_box=(0.45, 0.35, 0.10, 0.16),
            ),
            source_resolutions=((3840, 2160), (3840, 2160)),
        )

        self.assertEqual(plan.eligible_count, 0)
        self.assertEqual(
            plan.decisions[0].rejection_reason,
            "per_frame_subject_geometry_unavailable",
        )
        self.assertIsNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))

    def test_high_confidence_silhouette_can_validate_a_transient_face_miss(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )

        def frames(source_index: int, seconds: float) -> np.ndarray:
            if source_index == 1 and abs(seconds - 1.2) < 0.01:
                return _frame(source_index, marker=0)
            return _frame(source_index)

        regular = _observations(
            source0_box=(0.44, 0.33, 0.12, 0.18),
            source1_box=(0.45, 0.35, 0.10, 0.16),
        )

        def observations(frame: np.ndarray) -> BoundarySubjectObservation:
            if float(frame.mean()) < 20.0:
                return BoundarySubjectObservation(
                    silhouette_box_xywh=(0.36, 0.18, 0.28, 0.70),
                    silhouette_confidence=0.93,
                )
            return regular(frame)

        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("a.mp4"), Path("b.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            frame_provider=frames,
            observation_provider=observations,
            source_resolutions=((3840, 2160), (3840, 2160)),
        )

        self.assertEqual(plan.eligible_count, 1)
        self.assertGreater(plan.decisions[0].validated_output_frame_count, 0)

    def test_forward_backward_optical_flow_validates_one_nearby_detector_miss(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )
        random = np.random.default_rng(811)
        texture = random.integers(0, 256, (360, 640, 3), dtype=np.uint8)

        def frames(source_index: int, seconds: float) -> np.ndarray:
            local_frame = int(round((seconds - source_index) * 30.0))
            shifted = cv2.warpAffine(
                texture,
                np.asarray([[1.0, 0.0, local_frame * 0.35], [0.0, 1.0, 0.0]]),
                (640, 360),
                borderMode=cv2.BORDER_REFLECT,
            )
            marker = 50 if source_index == 0 else 100
            if source_index == 1 and local_frame == 6:
                marker = 0
            shifted[:32, :96] = marker
            return shifted

        def observations(frame: np.ndarray) -> BoundarySubjectObservation:
            marker = float(frame[:32, :96].mean())
            if marker < 20.0:
                return BoundarySubjectObservation()
            box = (
                (0.44, 0.33, 0.12, 0.18)
                if marker < 75.0
                else (0.45, 0.35, 0.10, 0.16)
            )
            return BoundarySubjectObservation(
                face_count=1,
                face_box_xywh=box,
                face_eye_scale=box[3] * 0.42,
            )

        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("a.mp4"), Path("b.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            frame_provider=frames,
            observation_provider=observations,
            source_resolutions=((3840, 2160), (3840, 2160)),
        )

        self.assertEqual(plan.eligible_count, 1)
        self.assertEqual(plan.optical_flow_containment_frame_count, 1)
        self.assertEqual(
            plan.decisions[0].optical_flow_containment_frame_count,
            1,
        )

    def test_native_cut_and_dissolve_are_rejected_before_decode(self) -> None:
        cases = (
            (
                VisualTimelinePlan(
                    (
                        _choice(0, 0, start=0.0, end=1.0, exit_cut=True),
                        _choice(1, 1, start=1.0, end=2.0),
                    ),
                    1.0,
                ),
                RenderTransition(tier="C", reason="hard"),
                "broadcaster_native_internal_cut",
            ),
            (
                VisualTimelinePlan(
                    (
                        _choice(0, 0, start=0.0, end=1.0),
                        _choice(1, 1, start=1.0, end=2.0),
                    ),
                    1.0,
                ),
                RenderTransition(
                    kind=TransitionKind.FACE_DISSOLVE,
                    frame_count=11,
                    tier="A",
                    reason="reviewed elsewhere",
                ),
                "unreviewed_dissolve_overlap",
            ),
        )
        for timeline, transition, reason in cases:
            with self.subTest(reason=reason):
                calls = 0

                def frames(_source_index: int, _seconds: float) -> np.ndarray:
                    nonlocal calls
                    calls += 1
                    return _frame(0)

                plan = build_boundary_subject_alignment_plan(
                    visual_timeline=timeline,
                    sources=(Path("a.mp4"), Path("b.mp4")),
                    transitions=(transition,),
                    fps=30.0,
                    frame_provider=frames,
                    source_resolutions=((1920, 1080), (1920, 1080)),
                )
                self.assertEqual(plan.decisions[0].rejection_reason, reason)
                self.assertEqual(calls, 0)

    def test_base_pan_crop_is_included_without_cascading_to_same_source_neighbor(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
                _choice(2, 1, start=2.0, end=3.0),
            ),
            1.0,
        )
        crop = PanCropTrack(
            (
                PanCropKeyframe(0.0, NormalizedCrop(0.05, 0.05, 0.90, 0.90)),
                PanCropKeyframe(1.0, NormalizedCrop(0.06, 0.05, 0.90, 0.90)),
            )
        )
        face_plan = FaceMatchPlan(
            (
                FaceMatchSegment(0, 0, False, "raw", 0.18, 0.18, 1.0, None, None),
                FaceMatchSegment(1, 1, True, None, 0.16, 0.16, 1.1, 1.1, crop, "face"),
                FaceMatchSegment(2, 1, False, "raw", 0.16, 0.16, 1.0, None, None),
            ),
            0.18,
            1.35,
            (0.5, 0.42),
        )
        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("a.mp4"), Path("b.mp4")),
            transitions=(
                RenderTransition(tier="C", reason="hard"),
                RenderTransition(tier="D", reason="same-shot hold"),
            ),
            face_match_plan=face_plan,
            fps=30.0,
            frame_provider=lambda source_index, _seconds: _frame(source_index),
            observation_provider=_observations(
                source0_box=(0.44, 0.33, 0.12, 0.18),
                source1_box=(0.45, 0.35, 0.10, 0.16),
            ),
            source_resolutions=((3840, 2160), (3840, 2160)),
        )

        self.assertEqual(plan.eligible_count, 1)
        self.assertIsNone(plan.track_for(0))
        self.assertIsNotNone(plan.track_for(1))
        self.assertIsNone(plan.track_for(2))
        self.assertEqual(plan.segments[1].boundary_indices, (0,))

    def test_resolution_headroom_is_reported_without_relaxing_v1_schema(self) -> None:
        timeline = VisualTimelinePlan(
            (_choice(0, 0, start=0.0, end=1.0),),
            1.0,
        )
        for resolution, profile_max in (
            ((1920, 1080), 1.10),
            ((2560, 1440), 1.33),
            ((3840, 2160), 2.0),
        ):
            with self.subTest(resolution=resolution):
                plan = build_boundary_subject_alignment_plan(
                    visual_timeline=timeline,
                    sources=(Path("source.mp4"),),
                    transitions=(),
                    fps=30.0,
                    frame_provider=lambda _source, _seconds: self.fail(
                        "no boundary should decode"
                    ),
                    source_resolutions=(resolution,),
                )
                headroom = plan.source_headroom[0]
                self.assertAlmostEqual(headroom.quality_profile_max_zoom, profile_max)
                self.assertLessEqual(headroom.effective_max_zoom, 1.18)
                self.assertFalse(
                    plan.to_dict()["policy"]["legacy_editorial_schema_globally_relaxed"]
                )

    def test_editorial_motion_uses_validated_local_track_and_versions_analysis(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )
        transitions = (RenderTransition(tier="C", reason="hard"),)
        alignment = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("a.mp4"), Path("b.mp4")),
            transitions=transitions,
            fps=30.0,
            frame_provider=lambda source_index, _seconds: _frame(source_index),
            observation_provider=_observations(
                source0_box=(0.45, 0.35, 0.10, 0.16),
                source1_box=(0.44, 0.33, 0.12, 0.18),
            ),
            source_resolutions=((1920, 1080), (1920, 1080)),
        )

        motion = build_editorial_motion_plan(
            visual_timeline=timeline,
            transitions=transitions,
            boundary_subject_alignment_plan=alignment,
            fps=30.0,
            config=EditorialMotionConfig(
                target_zoom_events_per_minute=0.0,
                target_roll_events_per_minute=0.0,
            ),
        )

        self.assertEqual(motion.boundary_match_count, 1)
        self.assertEqual(motion.zoom_event_count, 1)
        self.assertTrue(motion.analysis_basis.endswith("virtual_camera_v4"))
        self.assertIs(motion.boundary_subject_alignment_plan, alignment)
        self.assertEqual(
            motion.track_for(0).reason,  # type: ignore[union-attr]
            alignment.track_for(0).reason,  # type: ignore[union-attr]
        )
        self.assertEqual(
            motion.to_dict()["boundary_subject_alignment"]["eligible_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
