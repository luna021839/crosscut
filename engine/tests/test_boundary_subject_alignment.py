from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from crosscut.boundary_subject_alignment import (
    BoundarySubjectAlignmentConfig,
    BoundarySubjectObservation,
    _optical_flow_box,
    _shot_policy,
    _subject,
    _TransformState,
    _project_box,
    _layout_mismatch,
    _native_cut_in_boundary_window,
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
    shot_type: ShotType = ShotType.MEDIUM,
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
        shot_type=shot_type,
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
    def test_native_cut_scope_uses_only_the_transformed_window(self):
        choice = replace(_choice(0, 0, start=10., end=12.),
            internal_scene_cut_master_seconds=(10.3,),
            face_match_candidate_failure="internal_camera_cut")
        def conflicts(item, outgoing):
            return _native_cut_in_boundary_window(item, outgoing=outgoing,
                frame_count=60, fps=30., window_frames=14)
        self.assertFalse(conflicts(choice, True))
        self.assertTrue(conflicts(choice, False))
        near_exit = replace(choice, internal_scene_cut_master_seconds=(11.7,))
        self.assertTrue(conflicts(near_exit, True))
        self.assertFalse(conflicts(near_exit, False))
        # Unknown clock, invalid clock, and the optical recovery guard do not
        # gain permission just because the endpoint detector sees a face.
        self.assertTrue(conflicts(replace(choice, internal_scene_cut_master_seconds=()), True))
        self.assertTrue(conflicts(replace(choice, internal_scene_cut_master_seconds=(float('nan'),)), True))
        self.assertTrue(conflicts(replace(choice, internal_scene_cut_master_seconds=(9.,)), True))
        self.assertTrue(conflicts(replace(choice, internal_scene_cut_master_seconds=(11.50,)), True))
        self.assertTrue(conflicts(replace(choice, internal_scene_cut_master_seconds=(10.50,)), False))
        self.assertTrue(conflicts(replace(choice, exit_scene_cut_near=True), True))

    def test_far_native_cut_does_not_discard_real_local_face_alignment(self):
        timeline = VisualTimelinePlan((
            replace(_choice(0, 0, start=0., end=2.),
                internal_scene_cut_master_seconds=(.3,),
                face_match_candidate_failure="internal_camera_cut"),
            replace(_choice(1, 1, start=2., end=4.),
                internal_scene_cut_master_seconds=(3.7,),
                face_match_candidate_failure="internal_camera_cut")), 1.)
        plan = build_boundary_subject_alignment_plan(visual_timeline=timeline,
            sources=(Path('a.mp4'), Path('b.mp4')),
            transitions=(RenderTransition(tier='C'),), fps=30.,
            frame_provider=lambda index, _seconds: _frame(index),
            observation_provider=_observations(source0_box=(.40,.26,.20,.24),
                source1_box=(.3875,.23,.225,.27)),
            source_resolutions=((1920,1080),(1920,1080)))
        self.assertEqual(plan.eligible_count, 1)
        self.assertEqual(plan.validated_output_frame_count, 13)
        track = plan.track_for(0)
        self.assertIsNotNone(track)
        self.assertEqual(track.evaluate(.3).zoom, 1.)
        self.assertGreater(track.evaluate(59/30.).zoom, 1.)

    def test_close_face_uses_head_envelope_but_medium_keeps_full_performer(self):
        settings = BoundarySubjectAlignmentConfig()
        observation = BoundarySubjectObservation(face_count=1,
            face_box_xywh=(.40, .26, .20, .24), detected_person_count=1,
            formation_boxes_xywh=((.20, .15, .60, .85),))
        close = _subject(observation, preferred_kind='face', settings=settings)
        self.assertEqual(_shot_policy(observation), 'close_face')
        self.assertGreater(close.containment_boxes[0][3], observation.face_box_xywh[3])
        state = _TransformState(1.15, .5, .5)
        self.assertGreaterEqual(_project_box(close.containment_boxes[0], state)[1], .995)
        self.assertLess(_project_box(observation.formation_boxes_xywh[0], state)[1], .995)
        medium = _subject(replace(observation, face_box_xywh=(.45, .35, .10, .16)),
            preferred_kind='face', settings=settings)
        self.assertEqual(medium.containment_boxes, observation.formation_boxes_xywh)
        jittered = replace(observation, face_box_xywh=(.40, .26, .20, .215))
        self.assertEqual(_shot_policy(jittered), 'medium_focus')
        fixed_contract = _subject(jittered, preferred_kind='face', settings=settings,
            framing_policy='close_face')
        self.assertLess(fixed_contract.containment_boxes[0][3], .4)

    def test_close_head_margin_cannot_be_cropped_just_because_face_fits(self):
        observation = BoundarySubjectObservation(face_count=1,
            face_box_xywh=(.40, .08, .20, .24))
        subject = _subject(observation, preferred_kind='face',
            settings=BoundarySubjectAlignmentConfig())
        state = _TransformState(1.18, .5, .5)
        self.assertGreaterEqual(_project_box(observation.face_box_xywh, state)[1], .995)
        self.assertLess(_project_box(subject.containment_boxes[0], state)[1], .995)

    def test_background_faces_need_explicit_dominance_evidence(self):
        observation = BoundarySubjectObservation(face_count=3,
            face_box_xywh=(.40, .26, .20, .24), detected_person_count=3)
        self.assertEqual(_shot_policy(observation), 'wide_formation')
        self.assertIsNone(_subject(observation, preferred_kind='face',
            settings=BoundarySubjectAlignmentConfig()))
        proved = replace(observation, face_is_unambiguous_dominant=True)
        self.assertEqual(_shot_policy(proved), 'close_face')
        self.assertIsNotNone(_subject(proved, preferred_kind='face',
            settings=BoundarySubjectAlignmentConfig()))
        with self.assertRaises(ValueError):
            BoundarySubjectObservation(face_count=0, face_is_unambiguous_dominant=True)

    def test_real_planner_accepts_close_scale_match_despite_source_truncated_torso(self):
        timeline = VisualTimelinePlan((
            _choice(0, 0, start=0., end=1.),
            _choice(1, 1, start=1., end=2.)), 1.)
        def observations(frame):
            face = ((.40, .26, .20, .24) if float(frame.mean()) < 75
                    else (.3875, .23, .225, .27))
            return BoundarySubjectObservation(face_count=1, face_box_xywh=face,
                face_eye_scale=face[3] * .42, detected_person_count=1,
                formation_boxes_xywh=((.20, .15, .60, .85),))
        plan = build_boundary_subject_alignment_plan(visual_timeline=timeline,
            sources=(Path('a.mp4'), Path('b.mp4')),
            transitions=(RenderTransition(tier='C'),), fps=30.,
            frame_provider=lambda index, _seconds: _frame(index),
            observation_provider=observations,
            source_resolutions=((1920,1080),(1920,1080)))
        self.assertEqual(plan.eligible_count, 1)
        self.assertGreater(plan.validated_output_frame_count, 0)
        self.assertAlmostEqual(plan.decisions[0].outgoing_state.zoom, 1.125)

    def test_large_face_precedes_background_people_but_small_face_does_not(self) -> None:
        formation = (
            (0.12, 0.30, 0.14, 0.48),
            (0.43, 0.25, 0.15, 0.55),
            (0.73, 0.31, 0.13, 0.47),
        )
        large = BoundarySubjectObservation(
            face_count=1,
            face_box_xywh=(0.40, 0.18, 0.20, 0.28),
            detected_person_count=3,
            formation_boxes_xywh=formation,
        )
        small = BoundarySubjectObservation(
            face_count=1,
            face_box_xywh=(0.46, 0.22, 0.08, 0.08),
            detected_person_count=3,
            formation_boxes_xywh=formation,
        )

        self.assertEqual(_shot_policy(large), "close_face")
        self.assertEqual(_shot_policy(small), "wide_formation")

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
        self.assertFalse(decision.partial_scale_alignment)
        self.assertGreater(decision.validated_output_frame_count, 0)
        track = plan.track_for(0)
        assert track is not None
        self.assertAlmostEqual(track.final_keyframe.zoom, 1.125)
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

    def test_dense_strength_search_keeps_a_proved_partial_match_without_redecoding(self):
        timeline = VisualTimelinePlan((
            _choice(0, 0, start=0., end=1.),
            _choice(1, 1, start=1., end=2.),
        ), 1.)
        calls = []

        def frames(source, seconds):
            calls.append((source, round(seconds * 30)))
            return _frame(source, marker=50 if source == 0 else 100 + round((seconds - 1) * 30))

        def observation(frame):
            marker = round(float(frame.mean()))
            if marker < 75:
                box = (.41, .365, .18, .27)
            else:
                index = marker - 100
                excursion = max(0., min(index / 5., (13. - index) / 8.))
                box = (.42 - .371 * excursion, .38, .16, .24)
            return BoundarySubjectObservation(face_count=1, face_box_xywh=box,
                                              face_eye_scale=box[3] * .42)

        plan = build_boundary_subject_alignment_plan(visual_timeline=timeline,
            sources=(Path('a'), Path('b')), transitions=(RenderTransition(tier='C'),),
            fps=30., frame_provider=frames, observation_provider=observation,
            source_resolutions=((1920, 1080),) * 2)
        self.assertEqual(plan.eligible_count, 1)
        self.assertGreater(plan.dense_track_candidate_attempt_count, 1)
        self.assertLessEqual(plan.dense_track_candidate_attempt_count, 5)
        self.assertEqual(len(calls), len(set(calls)), 'strengths must share exact decoded observations')
        decision = plan.decisions[0]
        self.assertTrue(decision.partial_scale_alignment)
        self.assertLess(decision.incoming_state.zoom, 1.125)
        self.assertGreater(decision.incoming_state.zoom, 1.)
        self.assertGreater(decision.baseline_geometry_mismatch - decision.transformed_geometry_mismatch, .015)
        self.assertEqual(decision.validated_output_frame_count, 13)
        track = plan.track_for(1)
        settings = BoundarySubjectAlignmentConfig()
        full_strength_failed_frames = 0
        for index in range(14):
            subject = _subject(observation(_frame(1, marker=100 + index)),
                preferred_kind='face', settings=settings, framing_policy='close_face')
            keyframe = track.evaluate(index / 30.)
            chosen = _TransformState(keyframe.zoom, keyframe.center_x, keyframe.center_y,
                                     keyframe.roll_degrees)
            full = _TransformState(1. + .125 * (1. - index / 13.))
            full_strength_failed_frames += min(_project_box(box, full)[1]
                for box in subject.containment_boxes) < settings.minimum_face_visible_fraction
            self.assertGreaterEqual(min(_project_box(box, chosen)[1]
                for box in subject.containment_boxes), settings.minimum_face_visible_fraction)
        self.assertGreater(full_strength_failed_frames, 0, 'the original first-choice curve must fail')

    def test_bad_entry_ramp_does_not_remove_a_separately_valid_exit_ramp(self):
        timeline = VisualTimelinePlan(tuple(
            _choice(index, index, start=float(index), end=float(index + 1))
            for index in range(3)), 1.)
        calls = []

        def frames(source, seconds):
            calls.append((source, round(seconds * 30)))
            missing = source == 1 and round((seconds - 1) * 30) == 6
            return _frame(source, marker=0 if missing else 100 if source == 1 else 50)

        observation = _observations(source0_box=(.41, .365, .18, .27),
                                    source1_box=(.42, .38, .16, .24))
        plan = build_boundary_subject_alignment_plan(visual_timeline=timeline,
            sources=tuple(Path(str(index)) for index in range(3)),
            transitions=(RenderTransition(tier='C'),) * 2, fps=30.,
            frame_provider=frames, observation_provider=observation,
            source_resolutions=((1920, 1080),) * 3)
        self.assertFalse(plan.decisions[0].eligible)
        self.assertEqual(plan.eligible_count, 1)
        self.assertTrue(plan.decisions[1].eligible)
        self.assertEqual(plan.segments[1].boundary_indices, (1,))
        self.assertEqual(plan.track_for(1).evaluate(0.).zoom, 1.)
        self.assertGreater(plan.track_for(1).evaluate(29 / 30.).zoom, 1.)
        self.assertLessEqual(plan.dense_track_candidate_attempt_count, 35)
        self.assertEqual(len(calls), len(set(calls)))

    def test_all_dense_strengths_still_reject_missing_geometry_with_bounded_work(self):
        timeline = VisualTimelinePlan(tuple(
            _choice(index, index, start=float(index), end=float(index + 1))
            for index in range(3)), 1.)
        calls = []

        def frames(source, seconds):
            calls.append((source, round(seconds * 30)))
            missing = source == 1 and round((seconds - 1) * 30) in (6, 23)
            return _frame(source, marker=0 if missing else 100 if source == 1 else 50)

        plan = build_boundary_subject_alignment_plan(visual_timeline=timeline,
            sources=tuple(Path(str(index)) for index in range(3)),
            transitions=(RenderTransition(tier='C'),) * 2, fps=30., frame_provider=frames,
            observation_provider=_observations(source0_box=(.41, .365, .18, .27),
                                              source1_box=(.42, .38, .16, .24)),
            source_resolutions=((1920, 1080),) * 3)
        self.assertEqual(plan.eligible_count, 0)
        self.assertIsNone(plan.track_for(1))
        self.assertTrue(all(decision.rejection_reason == 'per_frame_subject_geometry_unavailable'
                            for decision in plan.decisions))
        self.assertLessEqual(plan.dense_track_candidate_attempt_count, 35)
        self.assertEqual(len(calls), len(set(calls)))

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

    def test_moving_closeup_uses_adjacent_motion_not_a_frozen_other_camera(self) -> None:
        timeline = VisualTimelinePlan((
            _choice(0, 0, start=0.0, end=1.0),
            _choice(1, 1, start=1.0, end=2.0),
        ), 1.0)
        out_box = (.40, .29, .16, .27)

        def observation(frame):
            marker = int(round(float(frame.mean())))
            index = marker - 100
            box = out_box if marker < 75 else (.50 - .03 * index, .31, .16, .24)
            return BoundarySubjectObservation(face_count=1, face_box_xywh=box,
                                               face_eye_scale=box[3] * .42)

        def frames(source, seconds):
            return _frame(source, marker=50 if source == 0 else 100 + round((seconds - 1) * 30))

        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline, sources=(Path("a"), Path("b")),
            transitions=(RenderTransition(tier="C", reason="hard"),), fps=30,
            frame_provider=frames, observation_provider=observation,
            source_resolutions=((1920, 1080),) * 2,
        )
        self.assertEqual(plan.eligible_count, 1)
        self.assertGreater(plan.decisions[0].validated_output_frame_count, 0)
        track = plan.track_for(1)
        self.assertIsNotNone(track)
        settings = BoundarySubjectAlignmentConfig()
        target = _subject(observation(frames(0, 0)), preferred_kind="face",
                          settings=settings, framing_policy="close_face")
        old_frozen_reference_failures = 0
        for index in range(14):
            subject = _subject(observation(frames(1, 1 + index / 30)), preferred_kind="face",
                               settings=settings, framing_policy="close_face")
            keyframe = track.evaluate(index / 30)
            state = _TransformState(keyframe.zoom, keyframe.center_x,
                                    keyframe.center_y, keyframe.roll_degrees)
            baseline = _layout_mismatch(subject.layout_boxes, target.layout_boxes)
            transformed = _layout_mismatch(
                tuple(_project_box(box, state)[0] for box in subject.layout_boxes),
                target.layout_boxes,
            )
            old_frozen_reference_failures += transformed > baseline + settings.maximum_layout_worsening
        self.assertGreater(old_frozen_reference_failures, 0,
                           "the reproducer must fail the former frozen-endpoint test")

    def test_adjacent_layout_pop_is_still_rejected_inside_a_matched_settle(self) -> None:
        timeline = VisualTimelinePlan((
            _choice(0, 0, start=0.0, end=1.0),
            _choice(1, 1, start=1.0, end=2.0),
        ), 1.0)

        def observation(frame):
            marker = int(round(float(frame.mean())))
            box = ((.40, .29, .18, .27) if marker < 75 else
                   (.28, .35, .06, .08) if marker == 106 else (.50, .31, .16, .24))
            return BoundarySubjectObservation(face_count=1, face_box_xywh=box,
                                               face_eye_scale=box[3] * .42)

        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline, sources=(Path("a"), Path("b")),
            transitions=(RenderTransition(tier="C", reason="hard"),), fps=30,
            frame_provider=lambda source, seconds: _frame(
                source, marker=50 if source == 0 else 100 + round((seconds - 1) * 30)),
            observation_provider=observation, source_resolutions=((1920, 1080),) * 2,
        )
        self.assertEqual(plan.eligible_count, 0)
        self.assertEqual(plan.decisions[0].rejection_reason, "per_frame_layout_mismatch_gate")
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
            return replace(regular(frame),
                silhouette_box_xywh=(0.36, 0.18, 0.28, 0.70),
                silhouette_confidence=0.93)

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
                return (BoundarySubjectObservation(
                    silhouette_box_xywh=(0., 0., 1., 1.), silhouette_confidence=.96,
                ) if broad_silhouette else BoundarySubjectObservation())
            box = (
                (0.44, 0.33, 0.12, 0.18)
                if marker < 75.0
                else (0.45, 0.35, 0.10, 0.16)
            )
            return BoundarySubjectObservation(
                face_count=1,
                face_box_xywh=box,
                face_eye_scale=box[3] * 0.42,
                detected_person_count=1,
                formation_boxes_xywh=((0.34, 0.12, 0.32, 0.76),),
            )

        def build(config=None):
            return build_boundary_subject_alignment_plan(
                visual_timeline=timeline,
                sources=(Path("a.mp4"), Path("b.mp4")),
                transitions=(RenderTransition(tier="C", reason="hard"),),
                fps=30.0,
                frame_provider=frames,
                observation_provider=observations,
                source_resolutions=((3840, 2160), (3840, 2160)),
                config=config,
            )

        for broad_silhouette in (False, True):
            with self.subTest(broad_silhouette=broad_silhouette), patch(
                "crosscut.boundary_subject_alignment._optical_flow_box",
                wraps=_optical_flow_box,
            ) as optical_flow:
                plan = build()
            self.assertEqual(plan.eligible_count, 1)
            self.assertEqual(plan.optical_flow_containment_frame_count, 1)
            self.assertEqual(plan.decisions[0].optical_flow_containment_frame_count, 1)
            self.assertTrue(
                any(call.args[2][3] > 0.70 for call in optical_flow.call_args_list),
                "full performer containment must be optically tracked with the face",
            )
        # A high-confidence aggregate silhouette is not a measured head, and
        # cannot pass the crop just because the exact optical proof failed.
        with patch("crosscut.boundary_subject_alignment._optical_flow_box", return_value=None):
            untracked = build(BoundarySubjectAlignmentConfig(
                minimum_silhouette_visible_fraction=.999))
        self.assertEqual(untracked.eligible_count, 0)
        self.assertEqual(untracked.decisions[0].rejection_reason,
                         "per_frame_subject_containment_failed")

    def test_mixed_face_dropout_needs_common_adjacent_silhouettes(self):
        timeline = VisualTimelinePlan((
            _choice(0, 0, start=0., end=1.),
            _choice(1, 1, start=1., end=2.),
        ), 1.)
        silhouette = (0., .15, 1., .85)
        face_box = (.43, .29, .18, .27)
        face = _subject(BoundarySubjectObservation(face_count=1, face_box_xywh=face_box),
            preferred_kind='face', settings=BoundarySubjectAlignmentConfig())
        self.assertGreater(_layout_mismatch((silhouette,), face.layout_boxes), .34,
            'the reproducer must reject the former silhouette-to-frozen-head comparison')
        for mode, reason in (
            ('common', None),
            ('missing_previous', 'per_frame_common_geometry_unavailable'),
            ('layout_jump', 'per_frame_layout_mismatch_gate'),
        ):
            def observe(frame):
                marker = int(round(float(frame.mean())))
                dropout = marker == 111
                box = (.40, .26, .20, .30) if marker < 75 else face_box
                body = ((.04, .05, .25, .3) if dropout and mode == 'layout_jump'
                        else None if not dropout and mode == 'missing_previous' else silhouette)
                return BoundarySubjectObservation(
                    face_count=0 if dropout else 1,
                    face_box_xywh=None if dropout else box,
                    face_eye_scale=None if dropout else box[3] * .42,
                    silhouette_box_xywh=body, silhouette_confidence=.93 if body else 0.)
            with self.subTest(mode=mode):
                plan = build_boundary_subject_alignment_plan(
                    visual_timeline=timeline, sources=(Path('a'),Path('b')),
                    transitions=(RenderTransition(tier='C',reason='hard'),),fps=30.,
                    frame_provider=lambda source, seconds: _frame(source,
                        marker=50 if source==0 else 100+round((seconds-1)*30)),
                    observation_provider=observe, source_resolutions=((1920,1080),)*2)
                self.assertEqual(plan.decisions[0].rejection_reason, reason)
                self.assertEqual(plan.eligible_count, int(reason is None))
                self.assertEqual(plan.optical_flow_containment_frame_count, 0)

    def test_optical_face_without_full_performer_track_fails_closed(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )

        def frames(source_index: int, seconds: float) -> np.ndarray:
            local_frame = int(round((seconds - source_index) * 30.0))
            marker = 50 if source_index == 0 else 100
            if source_index == 1 and local_frame == 6:
                marker = 0
            frame = np.full((360, 640, 3), 118, dtype=np.uint8)
            frame[:32, :96] = marker
            return frame

        def observations(frame: np.ndarray) -> BoundarySubjectObservation:
            marker = float(frame[:32, :96].mean())
            if marker < 20.0:
                return BoundarySubjectObservation()
            face = (
                (0.44, 0.33, 0.12, 0.18)
                if marker < 75.0
                else (0.45, 0.35, 0.10, 0.16)
            )
            return BoundarySubjectObservation(
                face_count=1,
                face_box_xywh=face,
                face_eye_scale=face[3] * 0.42,
                detected_person_count=1,
                formation_boxes_xywh=((0.34, 0.12, 0.32, 0.76),),
            )

        def optical_flow(
            _reference: np.ndarray,
            _target: np.ndarray,
            box: tuple[float, float, float, float],
        ) -> tuple[float, float, float, float] | None:
            # Face evidence survives, but the full hands/feet envelope does
            # not.  The fallback must not return the face-only crop.
            return None if box[3] > 0.70 else box

        with patch(
            "crosscut.boundary_subject_alignment._optical_flow_box",
            side_effect=optical_flow,
        ):
            plan = build_boundary_subject_alignment_plan(
                visual_timeline=timeline,
                sources=(Path("a.mp4"), Path("b.mp4")),
                transitions=(RenderTransition(tier="C", reason="hard"),),
                fps=30.0,
                frame_provider=frames,
                observation_provider=observations,
                source_resolutions=((3840, 2160), (3840, 2160)),
            )

        self.assertEqual(plan.eligible_count, 0)
        self.assertEqual(
            plan.decisions[0].rejection_reason,
            "per_frame_subject_geometry_unavailable",
        )
        self.assertIsNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))

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

    def test_resolution_headroom_is_available_only_to_dense_local_alignment(self) -> None:
        timeline = VisualTimelinePlan(
            (_choice(0, 0, start=0.0, end=1.0),),
            1.0,
        )
        for resolution, profile_max in (
            ((1920, 1080), 1.18),
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
                self.assertAlmostEqual(headroom.effective_max_zoom, profile_max)
                self.assertFalse(
                    plan.to_dict()["policy"]["legacy_editorial_schema_globally_relaxed"]
                )

    def test_4k_face_scale_match_uses_native_pixels_but_1080_keeps_its_cap(self):
        timeline = VisualTimelinePlan((_choice(0, 0, start=0., end=1.),
            _choice(1, 1, start=1., end=2.)), 1.)
        for resolution, expected in (((3840,2160), 1.5), ((1920,1080), 1.18),
                                     ((2560,1440), 1.33), (None, 1.06)):
            with self.subTest(resolution=resolution):
                plan = build_boundary_subject_alignment_plan(visual_timeline=timeline,
                    sources=(Path('a.mp4'), Path('b.mp4')),
                    transitions=(RenderTransition(tier='C'),), fps=30.,
                    frame_provider=lambda index, _seconds: _frame(index),
                    observation_provider=_observations(source0_box=(.42,.33,.16,.24),
                        source1_box=(.38,.27,.24,.36)),
                    source_resolutions=(resolution,resolution))
                self.assertEqual(plan.eligible_count, 1)
                self.assertEqual(plan.validated_output_frame_count, 13)
                self.assertAlmostEqual(plan.decisions[0].outgoing_state.zoom, expected)

    def test_native_headroom_subtracts_existing_base_crop(self):
        from crosscut.boundary_subject_alignment import _headroom
        for crop, expected in ((1.,2.), (.9,1.8), (.65,1.3), (.55,1.18), (.4,1.06)):
            with self.subTest(crop=crop):
                proof = _headroom(0, (3840,2160), output_width=1920,
                    output_height=1080, base_crop_scale=crop,
                    settings=BoundarySubjectAlignmentConfig())
                self.assertAlmostEqual(proof.effective_max_zoom, expected)

    def test_exact_face_overrides_stale_timeline_wide_label(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(
                    0,
                    0,
                    start=0.0,
                    end=1.0,
                    shot_type=ShotType.GROUP_WIDE,
                ),
                _choice(
                    1,
                    1,
                    start=1.0,
                    end=2.0,
                    shot_type=ShotType.GROUP_WIDE,
                ),
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
                source0_box=(0.45, 0.34, 0.10, 0.17),
                source1_box=(0.43, 0.31, 0.13, 0.21),
            ),
            source_resolutions=((1920, 1080), (1920, 1080)),
        )

        self.assertEqual(plan.eligible_count, 1)
        self.assertEqual(plan.decisions[0].subject_kind, "face")
        self.assertEqual(plan.decisions[0].outgoing_shot_policy, "medium_focus")

    def test_medium_uses_upper_body_anchor_and_full_performer_containment(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )

        def observations(frame: np.ndarray) -> BoundarySubjectObservation:
            box = (
                (0.39, 0.20, 0.22, 0.58)
                if float(frame.mean()) < 75.0
                else (0.37, 0.15, 0.26, 0.68)
            )
            return BoundarySubjectObservation(
                detected_person_count=1,
                formation_boxes_xywh=(box,),
            )

        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("a.mp4"), Path("b.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            frame_provider=lambda source_index, _seconds: _frame(source_index),
            observation_provider=observations,
            source_resolutions=((1920, 1080), (1920, 1080)),
        )

        self.assertEqual(plan.eligible_count, 1)
        decision = plan.decisions[0]
        self.assertEqual(decision.subject_kind, "upper_body")
        self.assertLessEqual(decision.outgoing_state.zoom, 1.16)
        payload = plan.to_dict()
        self.assertEqual(
            payload["eligible_alignment_subject_counts"],
            [{"subject": "upper_body", "count": 1}],
        )
        self.assertEqual(payload["applied_boundary_density"], 1.0)

    def test_wide_uses_restricted_formation_anchor(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )

        def observations(frame: np.ndarray) -> BoundarySubjectObservation:
            if float(frame.mean()) < 75.0:
                boxes = (
                    (0.22, 0.28, 0.13, 0.48),
                    (0.43, 0.25, 0.14, 0.52),
                    (0.66, 0.29, 0.12, 0.47),
                )
            else:
                boxes = (
                    (0.20, 0.25, 0.14, 0.52),
                    (0.42, 0.22, 0.16, 0.57),
                    (0.67, 0.26, 0.13, 0.51),
                )
            return BoundarySubjectObservation(
                detected_person_count=3,
                formation_boxes_xywh=boxes,
            )

        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("a.mp4"), Path("b.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            frame_provider=lambda source_index, _seconds: _frame(source_index),
            observation_provider=observations,
            source_resolutions=((3840, 2160), (3840, 2160)),
        )

        self.assertEqual(plan.eligible_count, 1)
        decision = plan.decisions[0]
        self.assertEqual(decision.subject_kind, "formation")
        transformed = (
            decision.outgoing_state
            if decision.transformed_side == "outgoing"
            else decision.incoming_state
        )
        self.assertLessEqual(transformed.zoom, 1.08)
        self.assertLessEqual(
            np.hypot(transformed.center_x - 0.5, transformed.center_y - 0.5),
            0.018 + 1e-9,
        )

    def test_close_to_formation_family_mismatch_fails_closed(self) -> None:
        timeline = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, end=1.0),
                _choice(1, 1, start=1.0, end=2.0),
            ),
            1.0,
        )

        def observations(frame: np.ndarray) -> BoundarySubjectObservation:
            if float(frame.mean()) < 75.0:
                return BoundarySubjectObservation(
                    face_count=1,
                    face_box_xywh=(0.41, 0.25, 0.18, 0.26),
                )
            return BoundarySubjectObservation(
                detected_person_count=3,
                formation_boxes_xywh=(
                    (0.18, 0.30, 0.15, 0.50),
                    (0.42, 0.26, 0.16, 0.56),
                    (0.68, 0.31, 0.14, 0.49),
                ),
            )

        plan = build_boundary_subject_alignment_plan(
            visual_timeline=timeline,
            sources=(Path("a.mp4"), Path("b.mp4")),
            transitions=(RenderTransition(tier="C", reason="hard"),),
            fps=30.0,
            frame_provider=lambda source_index, _seconds: _frame(source_index),
            observation_provider=observations,
            source_resolutions=((3840, 2160), (3840, 2160)),
        )

        self.assertEqual(plan.eligible_count, 0)
        self.assertEqual(
            plan.decisions[0].rejection_reason,
            "shot_policy_family_mismatch",
        )
        self.assertIsNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))

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
