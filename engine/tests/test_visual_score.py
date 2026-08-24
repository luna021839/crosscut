from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.pose import FormationPersonDescriptor, SubjectGeometryDescriptor
from crosscut.timeline import (
    BoundarySideStability,
    ShotType,
    TimelineCandidate,
    TimelineChoice,
    TimelineMode,
    TimelineResult,
    select_timeline,
)
from crosscut.timeline_review import review_timeline
from crosscut.visual_score import (
    FrameVisualDescriptor,
    VisualDiversityConfig,
    YUNET_EXPECTED_SHA256,
    _DissolveEvidence,
    _DetectedFace,
    _dark_foreground_obstruction_score,
    _dissolve_evidence_window,
    _normalized_face_metadata,
    _safe_cadence_profiles,
    _select_safe_cadence_candidates,
    build_visual_timeline,
    describe_boundary_side_stability,
    describe_frame,
    yunet_model_asset_status,
)


def formation_person(center_x: float) -> FormationPersonDescriptor:
    return FormationPersonDescriptor(
        bbox_xywh=(center_x - 0.08, 0.14, 0.16, 0.72),
        centroid_xy=(center_x, 0.50),
        detection_confidence=0.9,
        upper_body_keypoints_xyc=(
            (-0.24, -0.36, 0.9),
            (0.24, -0.36, 0.9),
            (-0.38, -0.10, 0.9),
            (0.38, -0.10, 0.9),
            (-0.48, 0.05, 0.9),
            (0.48, 0.05, 0.9),
            (-0.16, 0.38, 0.9),
            (0.16, 0.38, 0.9),
        ),
    )


class VisualScoreTests(unittest.TestCase):
    @staticmethod
    def maximum_source_run(sources: list[int]) -> int:
        maximum = run = 0
        previous: int | None = None
        for source in sources:
            run = run + 1 if source == previous else 1
            maximum = max(maximum, run)
            previous = source
        return maximum

    @staticmethod
    def timeline_result(*source_ids: str) -> TimelineResult:
        choices = tuple(
            TimelineChoice(
                candidate=TimelineCandidate(
                    segment_index=index,
                    source_id=source_id,
                    start_seconds=float(index),
                    end_seconds=float(index + 1),
                    candidate_id=f"{index}:{source_id}",
                ),
                emission_score=1.0,
                transition_score=0.0,
                cumulative_score=float(index + 1),
                used_group_wide_fallback=False,
            )
            for index, source_id in enumerate(source_ids)
        )
        return TimelineResult(
            mode=TimelineMode.GENERAL,
            target_member=None,
            choices=choices,
            total_score=float(len(choices)),
        )

    def test_safe_cadence_profiles_change_only_requested_soft_controls(self) -> None:
        profiles = _safe_cadence_profiles(VisualDiversityConfig())

        self.assertEqual(
            [profile.name for profile in profiles],
            ["balanced", "reference", "dense_safe"],
        )
        expected = {
            "balanced": (0.30, 0.22, 0.48, 4.2, 7.2, 0.22, 0.20, 0.42, 0.65),
            "reference": (0.18, 0.18, 0.38, 3.8, 7.2, 0.32, 0.48, 0.52, 0.75),
            "dense_safe": (0.12, 0.14, 0.30, 3.2, 6.5, 0.44, 0.85, 0.68, 0.90),
        }
        for profile in profiles:
            config = profile.config
            self.assertEqual(config.quality_soft_floor, 0.84)
            self.assertEqual(config.quality_soft_floor_penalty, 0.65)
            self.assertEqual(
                (
                    config.continuity_bonus,
                    config.cut_penalty,
                    config.early_cut_penalty,
                    config.preferred_max_source_hold_seconds,
                    config.maximum_same_source_hold_seconds,
                    config.source_hold_penalty,
                    config.source_switch_bonus,
                    config.ping_pong_penalty,
                    config.maximum_switch_score_sacrifice,
                ),
                expected[profile.name],
            )

        soft_fields = {
            "continuity_bonus",
            "cut_penalty",
            "early_cut_penalty",
            "preferred_max_source_hold_seconds",
            "maximum_same_source_hold_seconds",
            "source_hold_penalty",
            "source_switch_bonus",
            "ping_pong_penalty",
            "maximum_switch_score_sacrifice",
        }
        balanced = profiles[0].config
        for profile in profiles[1:]:
            for item in fields(type(balanced)):
                if item.name not in soft_fields:
                    self.assertEqual(
                        getattr(profile.config, item.name),
                        getattr(balanced, item.name),
                        item.name,
                    )

    def test_safe_cadence_union_round_robins_and_dedupes_signatures(self) -> None:
        path_a = self.timeline_result("a")
        path_b = self.timeline_result("b")
        path_c = self.timeline_result("c")
        path_d = self.timeline_result("d")
        path_e = self.timeline_result("e")
        path_f = self.timeline_result("f")
        path_g = self.timeline_result("g")
        candidates_by_segment = ((path_a.candidates[0],),)
        pools = (
            (path_a, path_b, path_c),
            (path_a, path_d, path_e),
            (path_f, path_d, path_g),
        )

        with patch(
            "crosscut.visual_score.select_timeline_candidates",
            side_effect=pools,
        ) as select:
            results, cadence_profiles = _select_safe_cadence_candidates(
                candidates_by_segment,
                preferred_source_ids=None,
                diversity=VisualDiversityConfig(),
            )

        self.assertEqual(
            [result.candidates[0].source_id for result in results],
            ["a", "f", "b", "d", "c", "e", "g"],
        )
        self.assertEqual(
            cadence_profiles,
            (
                "balanced",
                "dense_safe",
                "balanced",
                "reference",
                "balanced",
                "reference",
                "dense_safe",
            ),
        )
        for call in select.call_args_list:
            config = call.kwargs["config"]
            self.assertEqual(config.quality_soft_floor, 0.84)
            self.assertEqual(config.quality_soft_floor_penalty, 0.65)
        self.assertEqual(select.call_count, 3)
        for call in select.call_args_list:
            self.assertIs(call.args[0], candidates_by_segment)
            self.assertEqual(call.kwargs["maximum_results"], 16)

    def test_safe_cadence_union_caps_at_32_without_profile_monopoly(self) -> None:
        pools = tuple(
            tuple(
                self.timeline_result(f"{profile}-{index}")
                for index in range(16)
            )
            for profile in ("balanced", "reference", "dense")
        )
        candidates_by_segment = ((pools[0][0].candidates[0],),)

        with patch(
            "crosscut.visual_score.select_timeline_candidates",
            side_effect=pools,
        ):
            results, cadence_profiles = _select_safe_cadence_candidates(
                candidates_by_segment,
                preferred_source_ids=None,
                diversity=VisualDiversityConfig(),
            )

        self.assertEqual(len(results), 32)
        self.assertEqual(
            cadence_profiles[:9],
            ("balanced", "reference", "dense_safe") * 3,
        )
        self.assertEqual(cadence_profiles.count("balanced"), 11)
        self.assertEqual(cadence_profiles.count("reference"), 11)
        self.assertEqual(cadence_profiles.count("dense_safe"), 10)

    def test_bbox_layout_gate_removes_the_all_review_failure_path(self) -> None:
        stable = BoundarySideStability(
            sample_span_seconds=3.0 / 29.97,
            scale_delta=0.0,
            structure_mismatch=0.0,
            pose_mismatch=None,
            shot_type_mismatch=0.0,
        )
        safe_box = (0.30, 0.18, 0.40, 0.72)
        trap_box = (0.72, 0.16, 0.24, 0.78)
        candidates_by_segment: list[list[TimelineCandidate]] = []
        for segment in range(24):
            segment_candidates = [
                TimelineCandidate(
                    segment_index=segment,
                    source_id=str(source),
                    start_seconds=float(segment),
                    end_seconds=float(segment + 1),
                    quality_score=0.12,
                    sync_score=0.6,
                    composition_score=0.4,
                    boundary_score=0.6,
                    entry_shot_scale=0.4,
                    exit_shot_scale=0.4,
                    entry_boundary_side_stability=stable,
                    exit_boundary_side_stability=stable,
                    entry_silhouette_embedding=(0.2, 0.2, 0.2),
                    exit_silhouette_embedding=(0.2, 0.2, 0.2),
                    entry_subject_bbox_xywh=safe_box,
                    exit_subject_bbox_xywh=safe_box,
                    entry_subject_centroid_xy=(0.5, 0.54),
                    exit_subject_centroid_xy=(0.5, 0.54),
                    shot_id=f"{source}:shot",
                    candidate_id=f"{segment}:{source}",
                )
                for source in range(9)
            ]
            if 9 <= segment <= 11:
                segment_candidates.append(
                    TimelineCandidate(
                        segment_index=segment,
                        source_id="trap",
                        start_seconds=float(segment),
                        end_seconds=float(segment + 1),
                        quality_score=1.0,
                        sync_score=1.0,
                        composition_score=1.0,
                        boundary_score=1.0,
                        entry_shot_scale=0.4,
                        exit_shot_scale=0.4,
                        entry_boundary_side_stability=stable,
                        exit_boundary_side_stability=stable,
                        entry_silhouette_embedding=(0.2, 0.2, 0.2),
                        exit_silhouette_embedding=(0.2, 0.2, 0.2),
                        entry_subject_bbox_xywh=trap_box,
                        exit_subject_bbox_xywh=trap_box,
                        entry_subject_centroid_xy=(0.84, 0.55),
                        exit_subject_centroid_xy=(0.84, 0.55),
                        shot_id="trap:shot",
                        candidate_id=f"{segment}:trap",
                    )
                )
            candidates_by_segment.append(segment_candidates)

        balanced = _safe_cadence_profiles(VisualDiversityConfig())[0].config
        old_gate_equivalent = select_timeline(
            candidates_by_segment,
            config=replace(
                balanced,
                maximum_cross_source_layout_mismatch=None,
            ),
        )
        self.assertIn("trap", {item.source_id for item in old_gate_equivalent.candidates})
        self.assertIn(
            "layout_mismatch_exceeded",
            {
                failure.code
                for failure in review_timeline(
                    old_gate_equivalent,
                    available_source_count=9,
                ).hard_failures
            },
        )

        results, _profiles = _select_safe_cadence_candidates(
            candidates_by_segment,
            preferred_source_ids=None,
            diversity=VisualDiversityConfig(),
            maximum_results=8,
        )
        self.assertTrue(results)
        for result in results:
            self.assertNotIn("trap", {item.source_id for item in result.candidates})
            review = review_timeline(result, available_source_count=9)
            self.assertNotIn(
                "layout_mismatch_exceeded",
                {failure.code for failure in review.hard_failures},
            )

    def write_constant_sources(
        self,
        root: Path,
        levels: tuple[int, ...],
    ) -> tuple[Path, ...]:
        paths = tuple(root / f"camera-{index}.mp4" for index in range(len(levels)))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers = [
            cv2.VideoWriter(str(path), fourcc, 10.0, (80, 48)) for path in paths
        ]
        self.assertTrue(all(writer.isOpened() for writer in writers))
        for frame_index in range(100):
            for writer, level in zip(writers, levels):
                frame = np.full((48, 80, 3), level, dtype=np.uint8)
                frame[:, frame_index % 80] = min(255, level + 20)
                writer.write(frame)
        for writer in writers:
            writer.release()
        return paths

    def test_dissolve_window_samples_future_handle_and_rejects_scene_cut(self) -> None:
        geometry = (0.45, 0.35, 0.55, 0.35, 0.10, 0.0, 0.0, 0.5, 1.0, 0.0)
        box = (0.38, 0.20, 0.24, 0.32)

        descriptor = FrameVisualDescriptor(
            quality_score=0.8,
            composition_score=0.8,
            shot_scale=0.4,
            shot_type=ShotType.MEDIUM,
            face_count=1,
            structure_embedding=(0.0,) * 96,
            primary_face_bbox=box,
            face_detector_backend="yunet",
            primary_face_landmarks_xy=(
                (0.44, 0.35),
                (0.56, 0.35),
                (0.50, 0.48),
                (0.45, 0.56),
                (0.55, 0.56),
            ),
            primary_face_geometry=geometry,
            pose_embedding=(0.1,) * 60,
            pose_confidence=0.9,
            pose_backend="opencv_zoo_mediapipe_pose_2023mar",
            silhouette_embedding=(0.2,) * 121,
            silhouette_confidence=0.9,
            silhouette_backend="opencv_zoo_pphumanseg_2023mar",
        )

        class Sampler:
            def __init__(self) -> None:
                self.times: list[float] = []

            def transition_descriptor_at(self, seconds: float):  # type: ignore[no-untyped-def]
                self.times.append(seconds)
                return descriptor

            def sface_embedding_at(  # type: ignore[no-untyped-def]
                self, _seconds: float, _descriptor: FrameVisualDescriptor
            ):
                return (1.0,) + (0.0,) * 127

        sampler = Sampler()
        evidence = _dissolve_evidence_window(
            sampler,  # type: ignore[arg-type]
            boundary_seconds=2.0,
            scene_cut_times=(),
        )
        blocked = _dissolve_evidence_window(
            sampler,  # type: ignore[arg-type]
            boundary_seconds=2.0,
            scene_cut_times=(2.2,),
        )

        self.assertEqual(evidence.face_geometry, (geometry,) * 3)
        self.assertEqual(evidence.face_boxes, (box,) * 3)
        self.assertEqual(len(evidence.face_identity_embeddings), 3)
        self.assertEqual(evidence.pose_backends, (descriptor.pose_backend,) * 3)
        self.assertAlmostEqual(sampler.times[0], 2.0)
        self.assertAlmostEqual(sampler.times[1], 2.0 + (11 / 29.97) / 2.0)
        self.assertAlmostEqual(sampler.times[2], 2.0 + 11 / 29.97)
        self.assertEqual(blocked, _DissolveEvidence())

    def test_sharp_structured_frame_scores_above_flat_blur(self) -> None:
        checker = np.zeros((180, 320, 3), dtype=np.uint8)
        checker[::12, :] = 255
        checker[:, ::12] = 255
        flat = np.full((180, 320, 3), 118, dtype=np.uint8)

        sharp = describe_frame(checker)
        blurred = describe_frame(flat)

        self.assertGreater(sharp.quality_score, blurred.quality_score)
        self.assertEqual(len(sharp.structure_embedding), 96)

    def test_dark_foreground_boom_is_detected_without_flagging_black_clothing(self) -> None:
        stage = np.full((853, 480, 3), 176, dtype=np.uint8)
        cv2.rectangle(stage, (188, 180), (292, 790), (8, 8, 8), thickness=-1)
        clothing = describe_frame(stage, include_subject_geometry=False)

        boom = np.full((853, 480, 3), 176, dtype=np.uint8)
        cv2.fillConvexPoly(
            boom,
            np.array(
                [
                    (0, 45),
                    (479, 245),
                    (479, 400),
                    (0, 190),
                ],
                dtype=np.int32,
            ),
            (0, 0, 0),
        )
        obstructed = describe_frame(boom, include_subject_geometry=False)

        self.assertLess(clothing.foreground_obstruction_score, 0.85)
        self.assertGreaterEqual(obstructed.foreground_obstruction_score, 0.85)
        self.assertEqual(
            obstructed.to_dict()["foreground_obstruction_score"],
            obstructed.foreground_obstruction_score,
        )

    def test_top_edge_camera_rig_requires_subject_attention_overlap(self) -> None:
        gray = np.full((853, 480), 176, dtype=np.uint8)
        cv2.fillConvexPoly(
            gray,
            np.array(
                [
                    (70, 0),
                    (400, 0),
                    (385, 280),
                    (245, 400),
                    (105, 300),
                ],
                dtype=np.int32,
            ),
            0,
        )

        without_attention = _dark_foreground_obstruction_score(
            gray,
            attention_bbox_xywh=None,
        )
        over_subject = _dark_foreground_obstruction_score(
            gray,
            attention_bbox_xywh=(0.25, 0.20, 0.42, 0.72),
        )

        self.assertLess(without_attention, 0.85)
        self.assertGreaterEqual(over_subject, 0.85)

    def test_full_width_large_rig_uses_secondary_silhouette_overlap(self) -> None:
        gray = np.full((853, 480), 176, dtype=np.uint8)
        # A broad top rail connected to a thick central camera housing has the
        # real large-rig geometry but is intentionally not boom-elongated.
        cv2.rectangle(gray, (0, 0), (479, 24), 0, thickness=-1)
        cv2.rectangle(gray, (160, 20), (320, 430), 0, thickness=-1)
        peripheral_face = (0.02, 0.75, 0.10, 0.15)

        face_only = _dark_foreground_obstruction_score(
            gray,
            attention_bbox_xywh=peripheral_face,
        )
        with_silhouette = _dark_foreground_obstruction_score(
            gray,
            attention_bbox_xywh=peripheral_face,
            secondary_attention_bbox_xywh=(0.15, 0.15, 0.70, 0.75),
        )

        self.assertLess(face_only, 0.85)
        self.assertGreaterEqual(with_silhouette, 0.85)

    def test_lower_edge_boom_is_detected_without_flagging_horizontal_stage_band(self) -> None:
        attention = (0.25, 0.15, 0.50, 0.75)
        lower_boom = np.full((853, 480), 176, dtype=np.uint8)
        # A diagonal foreground bar merged into a lower horizontal dark mass
        # reproduces the case where component PCA alone becomes horizontal.
        cv2.rectangle(lower_boom, (0, 500), (479, 650), 0, thickness=-1)
        cv2.fillConvexPoly(
            lower_boom,
            np.array(
                [
                    (0, 300),
                    (479, 460),
                    (479, 580),
                    (0, 420),
                ],
                dtype=np.int32,
            ),
            0,
        )
        horizontal_stage_band = np.full((853, 480), 176, dtype=np.uint8)
        cv2.rectangle(
            horizontal_stage_band,
            (0, 300),
            (479, 570),
            0,
            thickness=-1,
        )

        boom_score = _dark_foreground_obstruction_score(
            lower_boom,
            attention_bbox_xywh=attention,
        )
        stage_score = _dark_foreground_obstruction_score(
            horizontal_stage_band,
            attention_bbox_xywh=attention,
        )

        self.assertGreaterEqual(boom_score, 0.85)
        self.assertLess(stage_score, 0.85)

    def test_near_lens_rig_is_detected_by_central_silhouette_overlap(self) -> None:
        gray = np.full((853, 480), 176, dtype=np.uint8)
        cv2.rectangle(gray, (180, 0), (300, 760), 0, thickness=-1)
        cv2.rectangle(gray, (80, 250), (400, 450), 0, thickness=-1)
        peripheral_face = (0.02, 0.75, 0.10, 0.15)

        face_only = _dark_foreground_obstruction_score(
            gray,
            attention_bbox_xywh=peripheral_face,
        )
        with_silhouette = _dark_foreground_obstruction_score(
            gray,
            attention_bbox_xywh=peripheral_face,
            secondary_attention_bbox_xywh=(0.15, 0.12, 0.70, 0.82),
        )

        self.assertLess(face_only, 0.85)
        self.assertGreaterEqual(with_silhouette, 0.85)

    def test_irregular_edge_cable_and_full_width_loop_are_detected(self) -> None:
        attention = (0.12, 0.10, 0.75, 0.82)
        edge_cable = np.full((853, 480), 176, dtype=np.uint8)
        cv2.polylines(
            edge_cable,
            [
                np.array(
                    [(0, 40), (245, 40), (245, 450), (70, 450), (70, 220)],
                    dtype=np.int32,
                )
            ],
            isClosed=False,
            color=0,
            thickness=40,
            lineType=cv2.LINE_8,
        )
        large_loop = np.full((853, 480), 176, dtype=np.uint8)
        cv2.polylines(
            large_loop,
            [
                np.array(
                    [(0, 120), (479, 180), (20, 370), (450, 570), (20, 750)],
                    dtype=np.int32,
                )
            ],
            isClosed=False,
            color=0,
            thickness=40,
            lineType=cv2.LINE_8,
        )
        vertical_stage_truss = np.full((853, 480), 176, dtype=np.uint8)
        cv2.rectangle(vertical_stage_truss, (30, 0), (90, 650), 0, thickness=-1)

        edge_score = _dark_foreground_obstruction_score(
            edge_cable,
            attention_bbox_xywh=attention,
        )
        loop_score = _dark_foreground_obstruction_score(
            large_loop,
            attention_bbox_xywh=attention,
        )
        stage_score = _dark_foreground_obstruction_score(
            vertical_stage_truss,
            attention_bbox_xywh=attention,
        )

        self.assertGreaterEqual(edge_score, 0.85)
        self.assertGreaterEqual(loop_score, 0.85)
        self.assertLess(stage_score, 0.85)

    def test_reflective_soft_dark_top_rig_is_not_a_vertical_stage_light(self) -> None:
        attention = (0.10, 0.05, 0.80, 0.65)
        reflected_rig = np.full((853, 480), 176, dtype=np.uint8)
        # Value 25 is intentionally above the strict near-black threshold;
        # the irregular top-connected shape exercises the reflected-housing
        # fallback rather than any of the strict dark-component branches.
        cv2.polylines(
            reflected_rig,
            [np.array([(0, 0), (390, 120), (80, 230)], dtype=np.int32)],
            isClosed=False,
            color=25,
            thickness=20,
            lineType=cv2.LINE_8,
        )
        vertical_stage_light = np.full((853, 480), 176, dtype=np.uint8)
        cv2.rectangle(vertical_stage_light, (210, 0), (245, 500), 25, thickness=-1)

        rig_score = _dark_foreground_obstruction_score(
            reflected_rig,
            attention_bbox_xywh=attention,
        )
        stage_score = _dark_foreground_obstruction_score(
            vertical_stage_light,
            attention_bbox_xywh=attention,
        )

        self.assertGreaterEqual(rig_score, 0.85)
        self.assertLess(stage_score, 0.85)

    def test_highlight_clipping_ratio_is_a_serialized_scalar(self) -> None:
        frame = np.full((180, 320, 3), 118, dtype=np.uint8)
        frame[:, :80] = 255

        descriptor = describe_frame(frame, include_subject_geometry=False)

        self.assertAlmostEqual(descriptor.highlight_clipping_ratio, 0.25)
        self.assertEqual(
            descriptor.to_dict()["highlight_clipping_ratio"],
            descriptor.highlight_clipping_ratio,
        )

    def test_bundled_yunet_has_exact_digest_and_loads_with_face_detector(self) -> None:
        status = yunet_model_asset_status(verify_load=True)

        self.assertTrue(status["present"])
        self.assertTrue(status["digest_valid"])
        self.assertTrue(status["loadable"])
        self.assertEqual(status["actual_sha256"], YUNET_EXPECTED_SHA256)

    def test_corrupt_yunet_is_rejected_before_detector_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "face_detection_yunet_2023mar.onnx"
            path.write_bytes(b"not an onnx model")
            status = yunet_model_asset_status(
                model_path=path,
                verify_load=True,
            )

        self.assertTrue(status["present"])
        self.assertFalse(status["digest_valid"])
        self.assertIsNone(status["loadable"])

    def test_frame_descriptor_uses_pose_and_silhouette_models_when_available(self) -> None:
        frame = np.full((180, 320, 3), 118, dtype=np.uint8)
        subject = SubjectGeometryDescriptor(
            pose_embedding=(0.1,) * 60,
            pose_confidence=0.82,
            detected_person_count=1,
            formation_people=(formation_person(0.5),),
            pose_backend="opencv_zoo_mediapipe_pose_2023mar",
            silhouette_embedding=(0.2,) * 121,
            silhouette_bbox_xywh=(0.2, 0.1, 0.6, 0.8),
            silhouette_centroid_xy=(0.5, 0.52),
            silhouette_area_ratio=0.25,
            silhouette_confidence=0.91,
            silhouette_backend="opencv_zoo_pphumanseg_2023mar",
        )
        with patch(
            "crosscut.visual_score.describe_subject_geometry",
            return_value=subject,
        ):
            descriptor = describe_frame(frame, include_formation_pose=True)

        self.assertEqual(descriptor.pose_embedding, subject.pose_embedding)
        self.assertEqual(descriptor.pose_backend, subject.pose_backend)
        self.assertEqual(descriptor.formation_people, subject.formation_people)
        self.assertEqual(
            descriptor.silhouette_embedding,
            subject.silhouette_embedding,
        )
        self.assertEqual(descriptor.silhouette_bbox_xywh, (0.2, 0.1, 0.6, 0.8))
        persisted = descriptor.to_dict()
        self.assertEqual(persisted["formation_person_count"], 1)
        self.assertEqual(persisted["formation_upper_body_count"], 1)
        self.assertNotIn("formation_people", persisted)

    def test_no_face_uses_single_performer_geometry_instead_of_forcing_wide(self) -> None:
        frame = np.full((180, 320, 3), 118, dtype=np.uint8)
        subject = SubjectGeometryDescriptor(
            detected_person_count=1,
            formation_people=(formation_person(0.5),),
        )
        with patch(
            "crosscut.visual_score.describe_subject_geometry",
            return_value=subject,
        ), patch("crosscut.visual_score._detect_faces", return_value=((), "test")):
            descriptor = describe_frame(frame)

        self.assertEqual(descriptor.shot_type, ShotType.MEDIUM)
        self.assertEqual(
            descriptor.shot_scale_basis,
            "single_or_pair_performer_bbox",
        )
        self.assertEqual(
            descriptor.to_dict()["shot_scale_basis"],
            "single_or_pair_performer_bbox",
        )

    def test_no_face_multi_person_geometry_is_aggregate_wide(self) -> None:
        frame = np.full((180, 320, 3), 118, dtype=np.uint8)
        subject = SubjectGeometryDescriptor(
            detected_person_count=3,
            formation_people=(
                formation_person(0.25),
                formation_person(0.50),
                formation_person(0.75),
            ),
        )
        with patch(
            "crosscut.visual_score.describe_subject_geometry",
            return_value=subject,
        ), patch("crosscut.visual_score._detect_faces", return_value=((), "test")):
            descriptor = describe_frame(frame)

        self.assertEqual(descriptor.shot_type, ShotType.GROUP_WIDE)
        self.assertEqual(descriptor.shot_scale_basis, "aggregate_formation_people")

    def test_large_dominant_face_precedes_background_formation_people(self) -> None:
        frame = np.full((180, 320, 3), 118, dtype=np.uint8)
        subject = SubjectGeometryDescriptor(
            detected_person_count=3,
            formation_people=(
                formation_person(0.25),
                formation_person(0.50),
                formation_person(0.75),
            ),
        )
        face = _DetectedFace((120.0, 28.0, 64.0, 72.0))
        with patch(
            "crosscut.visual_score.describe_subject_geometry",
            return_value=subject,
        ), patch(
            "crosscut.visual_score._detect_faces",
            return_value=((face,), "test"),
        ):
            descriptor = describe_frame(frame)

        self.assertEqual(descriptor.shot_type, ShotType.CLOSE_UP)
        self.assertEqual(descriptor.shot_scale_basis, "dominant_face_height")

    def test_small_partial_face_does_not_demote_multi_person_formation(self) -> None:
        frame = np.full((180, 320, 3), 118, dtype=np.uint8)
        subject = SubjectGeometryDescriptor(
            detected_person_count=3,
            formation_people=(
                formation_person(0.25),
                formation_person(0.50),
                formation_person(0.75),
            ),
        )
        face = _DetectedFace((145.0, 50.0, 30.0, 24.0))
        with patch(
            "crosscut.visual_score.describe_subject_geometry",
            return_value=subject,
        ), patch(
            "crosscut.visual_score._detect_faces",
            return_value=((face,), "test"),
        ):
            descriptor = describe_frame(frame)

        self.assertEqual(descriptor.shot_type, ShotType.GROUP_WIDE)
        self.assertEqual(descriptor.shot_scale_basis, "aggregate_formation_people")

    def test_missing_face_and_body_geometry_is_explicit_unknown_not_observed_wide(self) -> None:
        frame = np.full((180, 320, 3), 118, dtype=np.uint8)
        with patch(
            "crosscut.visual_score.describe_subject_geometry",
            return_value=SubjectGeometryDescriptor(),
        ), patch("crosscut.visual_score._detect_faces", return_value=((), "test")):
            descriptor = describe_frame(frame)

        self.assertEqual(descriptor.shot_type, ShotType.MEDIUM)
        self.assertEqual(descriptor.shot_scale, 0.34)
        self.assertEqual(
            descriptor.shot_scale_basis,
            "unknown_no_subject_geometry",
        )

    def test_visual_plan_exposes_selected_body_transition_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.write_constant_sources(
                Path(temporary_directory),
                (118, 118),
            )
            descriptor = FrameVisualDescriptor(
                quality_score=0.8,
                composition_score=0.8,
                shot_scale=0.4,
                shot_type=ShotType.MEDIUM,
                face_count=0,
                structure_embedding=(0.0,) * 96,
                pose_embedding=(0.1,) * 60,
                pose_confidence=0.8,
                detected_person_count=1,
                person_layout_embedding=(0.25, 0.50, 0.75),
                formation_people=(formation_person(0.5),),
                pose_backend="opencv_zoo_mediapipe_pose_2023mar",
                silhouette_embedding=(0.2,) * 121,
                silhouette_bbox_xywh=(0.2, 0.1, 0.6, 0.8),
                silhouette_centroid_xy=(0.5, 0.5),
                silhouette_confidence=0.9,
                silhouette_backend="opencv_zoo_pphumanseg_2023mar",
            )
            with patch(
                "crosscut.visual_score.describe_frame",
                return_value=descriptor,
            ) as describe:
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=((1.0, 2.0), (2.0, 3.0)),
                    audio_offsets=(0.0, 0.0),
                    sync_scores=(0.9, 0.9),
                    scene_cut_times=((), ()),
                )

        self.assertTrue(describe.call_args_list[0].kwargs["include_formation_pose"])
        self.assertNotIn("include_formation_pose", describe.call_args_list[1].kwargs)
        self.assertTrue(describe.call_args_list[2].kwargs["include_formation_pose"])
        self.assertNotIn("include_formation_pose", describe.call_args_list[3].kwargs)
        self.assertNotIn("include_formation_pose", describe.call_args_list[4].kwargs)

        self.assertEqual(plan.choices[1].shot_scale_bases, ("legacy",) * 3)
        self.assertEqual(
            plan.choices[1].to_dict()["shot_scale_bases"],
            ["legacy", "legacy", "legacy"],
        )

        self.assertEqual(
            plan.choices[1].pose_transition_basis_from_previous,
            "opencv_zoo_mediapipe_pose",
        )
        self.assertEqual(plan.choices[1].pose_transition_mismatch_from_previous, 0.0)
        self.assertEqual(
            plan.choices[1].silhouette_transition_mismatch_from_previous,
            0.0,
        )
        self.assertEqual(
            plan.choices[1].layout_transition_mismatch_from_previous,
            0.0,
        )
        self.assertEqual(
            plan.choices[1].formation_transition_mismatch_from_previous,
            0.0,
        )
        self.assertEqual(plan.choices[1].formation_person_counts, (1, 1, 1))
        self.assertEqual(
            plan.choices[1].formation_upper_body_counts,
            (1, 1, 1),
        )
        self.assertEqual(
            plan.choices[1].silhouette_backends,
            ("opencv_zoo_pphumanseg_2023mar",) * 3,
        )
        summaries = plan.to_dict()["candidate_review_summaries"]
        quality_floor = plan.to_dict()["per_shot_quality_soft_floor"]
        self.assertEqual(quality_floor["floor"], 0.84)
        self.assertEqual(quality_floor["maximum_penalty"], 0.65)
        self.assertTrue(quality_floor["above_floor_emission_unchanged"])
        self.assertFalse(quality_floor["hard_rejection"])
        self.assertTrue(summaries)
        for summary in summaries:
            self.assertIn(
                summary["cadence_profile"],
                {"balanced", "reference", "dense_safe"},
            )
            self.assertIn("source_switches_per_minute", summary)
            self.assertIn("source_run_median_seconds", summary)
            self.assertIn("visual_shot_median_seconds", summary)

    def test_visual_plan_persists_the_bbox_layout_fallback_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.write_constant_sources(
                Path(temporary_directory),
                (96, 176),
            )

            def descriptor_for(
                frame: np.ndarray,
                **_kwargs: object,
            ) -> FrameVisualDescriptor:
                source_b = float(frame.mean()) > 140.0
                box = (
                    (0.34, 0.18, 0.42, 0.72)
                    if source_b
                    else (0.28, 0.20, 0.42, 0.70)
                )
                centroid = (0.55, 0.54) if source_b else (0.48, 0.54)
                return FrameVisualDescriptor(
                    quality_score=0.8,
                    composition_score=0.8,
                    shot_scale=0.4,
                    shot_type=ShotType.MEDIUM,
                    face_count=0,
                    structure_embedding=(0.0,) * 96,
                    person_layout_embedding=(),
                    silhouette_bbox_xywh=box,
                    silhouette_centroid_xy=centroid,
                )

            def selected_cross_source_path(
                candidates_by_segment: object,
                **_kwargs: object,
            ) -> tuple[tuple[TimelineResult, ...], tuple[str, ...]]:
                segments = candidates_by_segment
                assert isinstance(segments, list)
                selected = (segments[0][0], segments[1][1])
                choices = tuple(
                    TimelineChoice(
                        candidate=item,
                        emission_score=1.0,
                        transition_score=0.0,
                        cumulative_score=float(index + 1),
                        used_group_wide_fallback=False,
                    )
                    for index, item in enumerate(selected)
                )
                return (
                    (
                        TimelineResult(
                            mode=TimelineMode.GENERAL,
                            target_member=None,
                            choices=choices,
                            total_score=2.0,
                        ),
                    ),
                    ("balanced",),
                )

            with patch(
                "crosscut.visual_score.describe_frame",
                side_effect=descriptor_for,
            ), patch(
                "crosscut.visual_score._select_safe_cadence_candidates",
                side_effect=selected_cross_source_path,
            ):
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=((1.0, 2.0), (2.0, 3.0)),
                    audio_offsets=(0.0, 0.0),
                    sync_scores=(0.9, 0.9),
                    scene_cut_times=((), ()),
                )

        persisted = plan.choices[1].layout_transition_mismatch_from_previous
        reviewed = review_timeline(plan).boundaries[0].layout_mismatch
        self.assertIsNotNone(persisted)
        self.assertAlmostEqual(persisted or 0.0, reviewed or 0.0)
        self.assertLess(persisted or 1.0, 0.32)

    def test_visual_plan_serializes_selected_entry_exit_gate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.write_constant_sources(
                Path(temporary_directory),
                (118, 118),
            )

            def descriptor(scale: float, shot_type: ShotType) -> FrameVisualDescriptor:
                return FrameVisualDescriptor(
                    quality_score=0.8,
                    composition_score=0.8,
                    shot_scale=scale,
                    shot_type=shot_type,
                    face_count=0,
                    structure_embedding=(0.0,) * 96,
                    person_layout_embedding=(0.2, 0.4, 0.3),
                )

            # Each source now describes entry/middle/exit plus the already
            # decoded following-entry and preceding-exit frames used by the
            # multi-frame boundary-side stability gate.
            samples = (
                descriptor(0.18, ShotType.GROUP_WIDE),
                descriptor(0.40, ShotType.MEDIUM),
                descriptor(0.62, ShotType.CLOSE_UP),
                descriptor(0.18, ShotType.GROUP_WIDE),
                descriptor(0.62, ShotType.CLOSE_UP),
            ) * 2
            with patch(
                "crosscut.visual_score.describe_frame",
                side_effect=samples,
            ):
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=((1.0, 1.125),),
                    audio_offsets=(0.0, 0.0),
                    sync_scores=(0.9, 0.1),
                    # Both candidates change camera 115 ms *inside* the
                    # incoming segment so review remains source-neutral. The
                    # higher-sync selected source must serialize entry risk.
                    # It is within one decoded frame of the outgoing edge, so
                    # that side remains boundary-aligned rather than risky.
                    scene_cut_times=((1.115,), (1.115,)),
                )

        choice = plan.choices[0]
        self.assertAlmostEqual(choice.entry_shot_scale or 0.0, 0.18)
        self.assertAlmostEqual(choice.exit_shot_scale or 0.0, 0.62)
        self.assertEqual(choice.entry_shot_type, ShotType.GROUP_WIDE)
        self.assertEqual(choice.exit_shot_type, ShotType.CLOSE_UP)
        self.assertTrue(choice.entry_scene_cut_near)
        self.assertFalse(choice.exit_scene_cut_near)
        serialized = choice.to_dict()
        self.assertEqual(serialized["entry_shot_type"], "group_wide")
        self.assertEqual(serialized["exit_shot_type"], "close_up")
        self.assertTrue(serialized["entry_scene_cut_near"])
        self.assertEqual(
            serialized["entry_boundary_side_stability"]["scale_delta"],
            0.0,
        )
        self.assertEqual(
            serialized["exit_boundary_side_stability"]["scale_delta"],
            0.0,
        )

    def test_boundary_side_stability_detects_group_wide_to_closeup_flash(
        self,
    ) -> None:
        def descriptor(scale: float, shot_type: ShotType) -> FrameVisualDescriptor:
            return FrameVisualDescriptor(
                quality_score=0.8,
                composition_score=0.8,
                shot_scale=scale,
                shot_type=shot_type,
                face_count=0,
                structure_embedding=(scale,) * 96,
            )

        evidence = describe_boundary_side_stability(
            descriptor(0.1631326356, ShotType.GROUP_WIDE),
            descriptor(0.6768598338, ShotType.CLOSE_UP),
            sample_span_seconds=3.0 / 29.97,
        )

        self.assertAlmostEqual(evidence.scale_delta, 0.5137271982)
        self.assertEqual(evidence.shot_type_mismatch, 1.0)
        self.assertGreater(evidence.structure_mismatch or 0.0, 0.5)

    def test_hidden_side_camera_cuts_do_not_block_cross_source_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.write_constant_sources(
                Path(temporary_directory),
                (118, 118),
            )
            descriptor = FrameVisualDescriptor(
                quality_score=0.8,
                composition_score=0.8,
                shot_scale=0.4,
                shot_type=ShotType.MEDIUM,
                face_count=0,
                structure_embedding=(0.0,) * 96,
                person_layout_embedding=(0.2, 0.4, 0.3),
            )
            with patch(
                "crosscut.visual_score.describe_frame",
                return_value=descriptor,
            ):
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=((1.0, 1.4),),
                    audio_offsets=(0.0, 0.0),
                    sync_scores=(0.9, 0.1),
                    # A cut before the segment begins and another after it
                    # ends are hidden by the cross-edit; neither may suppress
                    # a professional boundary swap.
                    scene_cut_times=((0.92, 1.48), ()),
                )

        self.assertFalse(plan.choices[0].entry_scene_cut_near)
        self.assertFalse(plan.choices[0].exit_scene_cut_near)

    def test_exit_motion_samples_only_frames_before_rendered_boundary(self) -> None:
        descriptor = FrameVisualDescriptor(
            quality_score=0.8,
            composition_score=0.8,
            shot_scale=0.4,
            shot_type=ShotType.MEDIUM,
            face_count=0,
            structure_embedding=(0.0,) * 96,
            person_layout_embedding=(0.2, 0.4, 0.3),
            motion=(0.0, 0.0),
        )

        class Sampler:
            instances: list["Sampler"] = []

            def __init__(self, _path: Path) -> None:
                self.times: list[float] = []
                self.instances.append(self)

            def frame_at(self, seconds: float):  # type: ignore[no-untyped-def]
                self.times.append(seconds)
                return np.full((48, 80, 3), 118, dtype=np.uint8)

            def sface_embedding_at(  # type: ignore[no-untyped-def]
                self, _seconds: float, _descriptor: FrameVisualDescriptor
            ):
                return ()

            def close(self) -> None:
                pass

        with patch("crosscut.visual_score._VideoSampler", Sampler), patch(
            "crosscut.visual_score.describe_frame",
            return_value=descriptor,
        ):
            build_visual_timeline(
                sources=(Path("source-a.mp4"), Path("source-b.mp4")),
                canonical_spans=((1.0, 1.1),),
                audio_offsets=(0.0, 0.0),
                sync_scores=(0.9, 0.8),
                scene_cut_times=((), ()),
            )

        exit_time = 1.1 - min(0.05, 0.1 * 0.12)
        for sampler in Sampler.instances:
            self.assertTrue(
                any(abs(value - (exit_time - 0.08)) < 1e-9 for value in sampler.times)
            )
            self.assertFalse(
                any(abs(value - (exit_time + 0.08)) < 1e-9 for value in sampler.times)
            )

    def test_expensive_dissolve_models_run_only_after_dp_on_selected_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.write_constant_sources(
                Path(temporary_directory),
                (118, 118),
            )
            geometry = (
                0.44,
                0.40,
                0.56,
                0.40,
                0.12,
                0.0,
                0.0,
                0.75,
                0.0,
                1.45,
            )
            landmarks = (
                (0.44, 0.40),
                (0.56, 0.40),
                (0.50, 0.48),
                (0.46, 0.57),
                (0.54, 0.57),
            )
            descriptor = FrameVisualDescriptor(
                quality_score=0.8,
                composition_score=0.8,
                shot_scale=0.4,
                shot_type=ShotType.MEDIUM,
                face_count=1,
                structure_embedding=(0.0,) * 96,
                person_layout_embedding=(0.2, 0.4, 0.3),
                motion=(0.0, 0.0),
                primary_face_bbox=(0.4, 0.2, 0.2, 0.4),
                face_detector_backend="yunet",
                primary_face_landmarks_xy=landmarks,
                primary_face_head_roll_radians=0.0,
                primary_face_eye_scale=0.12,
                primary_face_geometry=geometry,
            )
            raw_embedding_sentinel = 0.731928417
            identity = (raw_embedding_sentinel,) * 128
            evidence = _DissolveEvidence(
                face_geometry=(geometry,) * 3,
                face_boxes=((0.4, 0.2, 0.2, 0.4),) * 3,
                face_identity_embeddings=(identity,) * 3,
                pose_embeddings=((0.1,) * 60,) * 3,
                pose_backends=("opencv_zoo_mediapipe_pose_2023mar",) * 3,
                pose_confidences=(0.9,) * 3,
            )
            with patch(
                "crosscut.visual_score.describe_frame",
                return_value=descriptor,
            ), patch(
                "crosscut.visual_score._dissolve_evidence_window",
                return_value=evidence,
            ) as evidence_window:
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=((1.0, 2.0), (2.0, 3.0)),
                    audio_offsets=(0.0, 0.0),
                    sync_scores=(0.9, 0.9),
                    scene_cut_times=((), ()),
                    preferred_source_indices=(0, 1),
                )

        self.assertEqual([choice.source_index for choice in plan.choices], [0, 1])
        # One selected switch has exactly two source handles. The four rejected
        # candidate handles were never sent through SFace/body inference.
        self.assertEqual(evidence_window.call_count, 2)
        self.assertEqual(
            plan.choices[0].dissolve_exit_face_identity_window,
            (identity,) * 3,
        )
        self.assertEqual(
            plan.choices[1].dissolve_entry_pose_embedding_window,
            ((0.1,) * 60,) * 3,
        )
        serialized = plan.choices[1].to_dict()
        self.assertNotIn("dissolve_entry_face_identity_window", serialized)
        evidence_payload = serialized["dissolve_entry_evidence"]
        self.assertEqual(evidence_payload["appearance_embedding_count"], 3)
        self.assertIn("appearance_compatibility_backend", evidence_payload)
        self.assertNotIn("identity_backend", evidence_payload)
        self.assertNotIn("identity_embedding_count", evidence_payload)
        self.assertFalse(evidence_payload["raw_embeddings_persisted"])
        policy = plan.to_dict()["dissolve_safety_window"]
        self.assertFalse(policy["raw_appearance_embeddings_persisted"])
        self.assertNotIn("raw_identity_embeddings_persisted", policy)

        def contains_raw_sentinel(value: object) -> bool:
            if isinstance(value, dict):
                return any(
                    contains_raw_sentinel(key) or contains_raw_sentinel(item)
                    for key, item in value.items()
                )
            if isinstance(value, (list, tuple)):
                return any(contains_raw_sentinel(item) for item in value)
            return value == raw_embedding_sentinel

        self.assertFalse(contains_raw_sentinel(plan.to_dict()))
        self.assertNotIn(str(raw_embedding_sentinel), repr(plan.choices[1]))

    def test_yunet_five_points_produce_normalized_roll_and_scale_metadata(self) -> None:
        face = _DetectedFace(
            bbox_xywh=(20.0, 10.0, 60.0, 75.0),
            landmarks_xy=(
                (65.0, 35.0),
                (35.0, 35.0),
                (50.0, 52.0),
                (62.0, 68.0),
                (38.0, 68.0),
            ),
            confidence=0.95,
        )

        bbox, landmarks, roll, scale, geometry = _normalized_face_metadata(
            face,
            frame_width=100,
            frame_height=100,
        )

        self.assertEqual(bbox, (0.2, 0.1, 0.6, 0.75))
        assert landmarks is not None and geometry is not None
        self.assertEqual(len(landmarks), 5)
        self.assertAlmostEqual(roll or 0.0, 0.0)
        self.assertAlmostEqual(scale or 0.0, 0.3)
        self.assertEqual(len(geometry), 10)
        self.assertAlmostEqual(geometry[0], 0.35)
        self.assertAlmostEqual(geometry[2], 0.65)

    def test_normalized_face_bbox_clips_detector_overflow_to_frame(self) -> None:
        face = _DetectedFace(
            bbox_xywh=(175.0, -8.0, 50.0, 120.0),
            landmarks_xy=None,
            confidence=0.95,
        )

        bbox, landmarks, roll, scale, geometry = _normalized_face_metadata(
            face,
            frame_width=200,
            frame_height=100,
        )

        self.assertEqual(bbox, (0.875, 0.0, 0.125, 1.0))
        self.assertIsNone(landmarks)
        self.assertIsNone(roll)
        self.assertIsNone(scale)
        self.assertIsNone(geometry)

    def test_visual_plan_samples_three_frame_face_minimax_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.write_constant_sources(
                Path(temporary_directory),
                (118, 118),
            )
            geometry = (
                0.44,
                0.40,
                0.56,
                0.40,
                0.12,
                0.0,
                0.0,
                0.75,
                0.0,
                1.45,
            )
            descriptor = FrameVisualDescriptor(
                quality_score=0.8,
                composition_score=0.8,
                shot_scale=0.4,
                shot_type=ShotType.MEDIUM,
                face_count=1,
                structure_embedding=(0.0,) * 96,
                person_layout_embedding=(0.2, 0.4, 0.3),
                motion=(0.0, 0.0),
                primary_face_bbox=(0.4, 0.2, 0.2, 0.4),
                primary_face_landmarks_xy=(
                    (0.44, 0.4),
                    (0.56, 0.4),
                    (0.5, 0.48),
                    (0.46, 0.57),
                    (0.54, 0.57),
                ),
                primary_face_head_roll_radians=0.0,
                primary_face_eye_scale=0.12,
                primary_face_geometry=geometry,
            )
            with patch(
                "crosscut.visual_score.describe_frame",
                return_value=descriptor,
            ):
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=((1.0, 2.0), (2.0, 3.0)),
                    audio_offsets=(0.0, 0.0),
                    sync_scores=(0.9, 0.9),
                    scene_cut_times=((), ()),
                )

        self.assertEqual(len(plan.choices[0].exit_face_geometry_window), 3)
        self.assertEqual(len(plan.choices[1].entry_face_geometry_window), 3)
        self.assertEqual(
            plan.choices[1].face_transition_minimax_mismatch_from_previous,
            0.0,
        )

    def test_visual_timeline_preserves_master_plus_offset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = (root / "soft.mp4", root / "sharp.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writers = [
                cv2.VideoWriter(str(path), fourcc, 20.0, (160, 90)) for path in paths
            ]
            self.assertTrue(all(writer.isOpened() for writer in writers))
            for frame_index in range(80):
                soft = np.full((90, 160, 3), 110, dtype=np.uint8)
                sharp = np.zeros((90, 160, 3), dtype=np.uint8)
                sharp[:, ::8] = 255
                sharp[::8, :] = 255
                cv2.circle(sharp, (20 + frame_index % 100, 45), 8, (0, 0, 255), -1)
                writers[0].write(soft)
                writers[1].write(sharp)
            for writer in writers:
                writer.release()

            plan = build_visual_timeline(
                sources=paths,
                canonical_spans=((1.0, 2.0), (2.0, 3.0)),
                audio_offsets=(-0.5, 0.75),
                sync_scores=(0.9, 0.9),
                scene_cut_times=((), ()),
            )

        self.assertEqual(len(plan.choices), 2)
        for choice in plan.choices:
            offset = (-0.5, 0.75)[choice.source_index]
            self.assertAlmostEqual(
                choice.source_local_start,
                choice.canonical_master_start + offset,
            )
        self.assertTrue(all(choice.source_index == 1 for choice in plan.choices))

    def test_nine_segment_visual_plan_rotates_comparable_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = (root / "camera-a.mp4", root / "camera-b.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writers = [
                cv2.VideoWriter(str(path), fourcc, 10.0, (80, 48)) for path in paths
            ]
            self.assertTrue(all(writer.isOpened() for writer in writers))
            frame = np.full((48, 80, 3), 118, dtype=np.uint8)
            for _ in range(100):
                for writer in writers:
                    writer.write(frame)
            for writer in writers:
                writer.release()

            descriptor = FrameVisualDescriptor(
                quality_score=0.8,
                composition_score=0.8,
                shot_scale=0.4,
                shot_type=ShotType.MEDIUM,
                face_count=1,
                structure_embedding=(0.0,) * 96,
                person_layout_embedding=(0.2, 0.4, 0.3),
                motion=(0.0, 0.0),
            )
            with patch(
                "crosscut.visual_score.describe_frame",
                return_value=descriptor,
            ):
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=tuple(
                        (float(index), float(index + 1)) for index in range(9)
                    ),
                    audio_offsets=(0.0, 0.0),
                    sync_scores=(0.9, 0.9),
                    # Source A changes camera inside every segment; this must
                    # not reset the same-source rotation clock. Exact boundary
                    # cuts are intentionally excluded because the production
                    # selector now rejects unstable double-cut edges.
                    scene_cut_times=(
                        tuple(index + 0.5 for index in range(9)),
                        (),
                    ),
                )

        sources = [choice.source_index for choice in plan.choices]
        self.assertEqual(len(sources), 9)
        self.assertEqual(set(sources), {0, 1})
        self.assertLessEqual(self.maximum_source_run(sources), 6)
        self.assertEqual(
            plan.analysis_basis,
            "reviewed_thirty_two_candidate_shot_hierarchy_boundary_stability_safe_v13",
        )

    def test_nine_equal_sources_follow_full_coverage_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.write_constant_sources(
                Path(temporary_directory),
                (118,) * 9,
            )
            descriptor = FrameVisualDescriptor(
                quality_score=0.8,
                composition_score=0.8,
                shot_scale=0.4,
                shot_type=ShotType.MEDIUM,
                face_count=1,
                structure_embedding=(0.0,) * 96,
                person_layout_embedding=(0.2, 0.4, 0.3),
                motion=(0.0, 0.0),
            )
            with patch(
                "crosscut.visual_score.describe_frame",
                return_value=descriptor,
            ):
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=tuple(
                        (float(index), float(index + 1)) for index in range(9)
                    ),
                    audio_offsets=(0.0,) * 9,
                    sync_scores=(0.9,) * 9,
                    scene_cut_times=((),) * 9,
                    preferred_source_indices=tuple(range(9)),
                )

        selected = [choice.source_index for choice in plan.choices]
        # One-second spans cannot safely show nine different sources while the
        # product enforces a two-span minimum hold. Coverage anchors therefore
        # spread across the maximum compatible set instead of forcing a cut on
        # every beat.
        self.assertGreaterEqual(len(set(selected)), 5)
        self.assertLessEqual(self.maximum_source_run(selected), 2)

    def test_very_poor_preferred_source_is_excluded_from_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.write_constant_sources(
                Path(temporary_directory),
                (118,) * 8 + (0,),
            )

            def descriptor_for(frame: np.ndarray, **_kwargs: object) -> FrameVisualDescriptor:
                poor = float(frame.mean()) < 20.0
                return FrameVisualDescriptor(
                    quality_score=0.02 if poor else 0.8,
                    composition_score=0.05 if poor else 0.8,
                    shot_scale=0.4,
                    shot_type=ShotType.MEDIUM,
                    face_count=1,
                    structure_embedding=(0.0,) * 96,
                    person_layout_embedding=(0.2, 0.4, 0.3),
                    motion=(0.0, 0.0),
                )

            with patch(
                "crosscut.visual_score.describe_frame",
                side_effect=descriptor_for,
            ):
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=tuple(
                        (float(index), float(index + 1)) for index in range(9)
                    ),
                    audio_offsets=(0.0,) * 9,
                    sync_scores=(0.9,) * 9,
                    scene_cut_times=((),) * 9,
                    preferred_source_indices=tuple(range(9)),
                )

        used = {choice.source_index for choice in plan.choices}
        self.assertNotIn(8, used)
        self.assertGreaterEqual(len(used), 4)
        self.assertTrue(used.issubset(set(range(8))))


if __name__ == "__main__":
    unittest.main()
