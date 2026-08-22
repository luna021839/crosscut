from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.pose import SubjectGeometryDescriptor
from crosscut.timeline import ShotType
from crosscut.visual_score import (
    FrameVisualDescriptor,
    YUNET_EXPECTED_SHA256,
    _DissolveEvidence,
    _DetectedFace,
    _dissolve_evidence_window,
    _normalized_face_metadata,
    build_visual_timeline,
    describe_frame,
    yunet_model_asset_status,
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
            descriptor = describe_frame(frame)

        self.assertEqual(descriptor.pose_embedding, subject.pose_embedding)
        self.assertEqual(descriptor.pose_backend, subject.pose_backend)
        self.assertEqual(
            descriptor.silhouette_embedding,
            subject.silhouette_embedding,
        )
        self.assertEqual(descriptor.silhouette_bbox_xywh, (0.2, 0.1, 0.6, 0.8))

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
            ):
                plan = build_visual_timeline(
                    sources=paths,
                    canonical_spans=((1.0, 2.0), (2.0, 3.0)),
                    audio_offsets=(0.0, 0.0),
                    sync_scores=(0.9, 0.9),
                    scene_cut_times=((), ()),
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
            plan.choices[1].silhouette_backends,
            ("opencv_zoo_pphumanseg_2023mar",) * 3,
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
                motion=(0.0, 0.0),
                primary_face_bbox=(0.4, 0.2, 0.2, 0.4),
                face_detector_backend="yunet",
                primary_face_landmarks_xy=landmarks,
                primary_face_head_roll_radians=0.0,
                primary_face_eye_scale=0.12,
                primary_face_geometry=geometry,
            )
            identity = (1.0,) + (0.0,) * 127
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
                    # Source A changes internal camera at each edit boundary;
                    # this must not reset the same-source rotation clock.
                    scene_cut_times=(tuple(float(index) for index in range(1, 9)), ()),
                )

        sources = [choice.source_index for choice in plan.choices]
        self.assertEqual(len(sources), 9)
        self.assertEqual(set(sources), {0, 1})
        self.assertLessEqual(self.maximum_source_run(sources), 4)
        self.assertEqual(
            plan.analysis_basis,
            "frame_quality_pose_silhouette_motion_color_face_minimax_coverage_v6",
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

        self.assertEqual(
            [choice.source_index for choice in plan.choices],
            list(range(9)),
        )

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
        self.assertEqual(used, set(range(8)))


if __name__ == "__main__":
    unittest.main()
