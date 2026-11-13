from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.face_match import FaceMatchPlan, FaceMatchSegment
from crosscut.group_transition import build_group_transition_plan
from crosscut.render import (
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    TransitionKind,
)
from crosscut.timeline import ShotType
from crosscut.visual_score import (
    _DISSOLVE_EVIDENCE_FIELD_NAMES,
    _DissolveEvidence,
    VisualTimelineChoice,
    VisualTimelinePlan,
    rehydrate_visual_timeline_dissolve_evidence,
)


_GEOMETRY = (
    0.45,
    0.35,
    0.55,
    0.35,
    0.10,
    0.0,
    0.0,
    0.55,
    1.0,
    0.0,
)
_BOX = (0.38, 0.20, 0.24, 0.32)
_LANDMARKS = (
    (0.44, 0.35),
    (0.56, 0.35),
    (0.50, 0.48),
    (0.45, 0.56),
    (0.55, 0.56),
)
_RAW_IDENTITY_SENTINEL = 0.731928417
_IDENTITY = (_RAW_IDENTITY_SENTINEL,) + (0.0,) * 127
_POSE = (0.1,) * 60
_SILHOUETTE = (0.2,) * 121
_POSE_BACKEND = "opencv_zoo_mediapipe_pose_2023mar"
_SILHOUETTE_BACKEND = "opencv_zoo_pphumanseg_2023mar"


def _choice(
    index: int,
    source: int,
    *,
    start: float | None = None,
    duration: float = 1.0,
    eligible_face: bool = True,
) -> VisualTimelineChoice:
    canonical_start = float(index) if start is None else float(start)
    canonical_end = canonical_start + duration
    return VisualTimelineChoice(
        segment_index=index,
        source_index=source,
        canonical_master_start=canonical_start,
        canonical_master_end=canonical_end,
        source_local_start=canonical_start,
        source_local_end=canonical_end,
        quality_score=0.8,
        composition_score=0.8,
        boundary_score=0.8,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"{source}:shot-{index}",
        face_detector_backends=(
            ("yunet", "yunet", "yunet")
            if eligible_face
            else ("none", "none", "none")
        ),
        face_landmark_samples_xy=(
            (_LANDMARKS, _LANDMARKS, _LANDMARKS)
            if eligible_face
            else (None, None, None)
        ),
    )


def _safe_evidence() -> _DissolveEvidence:
    return _DissolveEvidence(
        face_geometry=(_GEOMETRY,) * 3,
        face_boxes=(_BOX,) * 3,
        face_identity_embeddings=(_IDENTITY,) * 3,
        pose_embeddings=(_POSE,) * 3,
        pose_backends=(_POSE_BACKEND,) * 3,
        pose_confidences=(0.9,) * 3,
        silhouette_embeddings=(_SILHOUETTE,) * 3,
        silhouette_backends=(_SILHOUETTE_BACKEND,) * 3,
        silhouette_confidences=(0.9,) * 3,
    )


def _checkpoint_like_plan() -> VisualTimelinePlan:
    outgoing = replace(
        _choice(0, 0),
        dissolve_exit_face_geometry_window=(_GEOMETRY,) * 3,
        dissolve_exit_face_box_window=(_BOX,) * 3,
        dissolve_exit_pose_backend_window=(_POSE_BACKEND,) * 3,
        dissolve_exit_pose_confidence_window=(0.9,) * 3,
    )
    incoming = replace(
        _choice(1, 1),
        dissolve_entry_face_geometry_window=(_GEOMETRY,) * 3,
        dissolve_entry_face_box_window=(_BOX,) * 3,
        dissolve_entry_pose_backend_window=(_POSE_BACKEND,) * 3,
        dissolve_entry_pose_confidence_window=(0.9,) * 3,
    )
    return VisualTimelinePlan(
        (outgoing, incoming),
        2.0,
        analysis_basis="checkpoint_replay_v1+local_boundary_repair_v3",
    )


def _face_match(choices: tuple[VisualTimelineChoice, ...]) -> FaceMatchPlan:
    crop = NormalizedCrop(0.10, 0.10, 0.80, 0.80)
    return FaceMatchPlan(
        tuple(
            FaceMatchSegment(
                segment_index=choice.segment_index,
                source_index=choice.source_index,
                applied=True,
                fallback_reason=None,
                observed_face_height=0.32,
                target_face_height=0.32,
                required_zoom=1.25,
                applied_zoom=1.25,
                alignment_subject="face",
                pan_crop=PanCropTrack(
                    (
                        PanCropKeyframe(0.0, crop),
                        PanCropKeyframe(0.5, crop),
                        PanCropKeyframe(1.0, crop),
                    )
                ),
            )
            for choice in choices
        ),
        0.32,
        1.35,
        (0.5, 0.42),
    )


class _Sampler:
    instances: list["_Sampler"] = []

    def __init__(
        self,
        path: Path,
        *,
        feature_cache_directory: Path | None = None,
    ) -> None:
        self.path = path
        self.feature_cache_directory = feature_cache_directory
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


class VisualDissolveRehydrationTests(unittest.TestCase):
    def setUp(self) -> None:
        _Sampler.instances.clear()

    def test_rehydrates_repaired_clock_lazily_and_never_serializes_identity(
        self,
    ) -> None:
        boundary = 1.25
        plan = VisualTimelinePlan(
            (
                _choice(0, 0, start=0.0, duration=boundary),
                _choice(1, 1, start=boundary, duration=1.0),
            ),
            2.0,
            analysis_basis="checkpoint_replay_v1+local_boundary_repair_v3",
        )
        calls: list[tuple[Path, tuple[float, ...], tuple[float, ...]]] = []

        def evidence_window(sampler: _Sampler, **kwargs: object) -> _DissolveEvidence:
            calls.append(
                (
                    sampler.path,
                    tuple(kwargs["sample_times_seconds"]),  # type: ignore[arg-type]
                    tuple(kwargs["scene_cut_times"]),  # type: ignore[arg-type]
                )
            )
            return _safe_evidence()

        cache_directory = Path("descriptor-cache")
        with patch("crosscut.visual_score._VideoSampler", _Sampler), patch(
            "crosscut.visual_score._dissolve_evidence_window",
            side_effect=evidence_window,
        ) as mocked_window:
            rehydrated = rehydrate_visual_timeline_dissolve_evidence(
                plan,
                sources=(
                    Path("camera-0.mp4"),
                    Path("camera-1.mp4"),
                    Path("unused-camera-2.mp4"),
                ),
                audio_offsets=(0.25, -0.50, 7.0),
                scene_cut_times=((0.1,), (0.2,), (0.3,)),
                feature_cache_directory=cache_directory,
            )
            rehydrated_again = rehydrate_visual_timeline_dissolve_evidence(
                rehydrated,
                sources=(
                    Path("camera-0.mp4"),
                    Path("camera-1.mp4"),
                    Path("unused-camera-2.mp4"),
                ),
                audio_offsets=(0.25, -0.50, 7.0),
                scene_cut_times=((0.1,), (0.2,), (0.3,)),
                feature_cache_directory=cache_directory,
            )

        self.assertEqual(mocked_window.call_count, 4)
        self.assertEqual(len(calls), 4)
        handle = 11.0 / 29.97
        expected_master_times = (
            boundary,
            boundary + handle / 2.0,
            boundary + handle,
        )
        for actual, expected in zip(
            calls[0][1],
            (value + 0.25 for value in expected_master_times),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            calls[1][1],
            (value - 0.50 for value in expected_master_times),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(calls[0][2], (0.1,))
        self.assertEqual(calls[1][2], (0.2,))
        self.assertEqual(
            {sampler.path for sampler in _Sampler.instances},
            {Path("camera-0.mp4"), Path("camera-1.mp4")},
        )
        self.assertTrue(
            all(
                sampler.feature_cache_directory == cache_directory
                and sampler.closed
                for sampler in _Sampler.instances
            )
        )
        self.assertEqual(
            rehydrated.choices[0].dissolve_exit_face_identity_window,
            (_IDENTITY,) * 3,
        )
        self.assertEqual(
            rehydrated.choices[1].dissolve_entry_pose_embedding_window,
            (_POSE,) * 3,
        )
        self.assertEqual(
            rehydrated_again.analysis_basis.split("+").count(
                "post_repair_ram_dissolve_rehydration_v1"
            ),
            1,
        )
        serialized = rehydrated.choices[1].to_dict()
        self.assertNotIn("dissolve_entry_face_identity_window", serialized)
        self.assertEqual(
            serialized["dissolve_entry_evidence"][
                "appearance_embedding_count"
            ],
            3,
        )
        self.assertNotIn(str(_RAW_IDENTITY_SENTINEL), repr(rehydrated.choices[1]))
        self.assertNotIn(str(_RAW_IDENTITY_SENTINEL), str(serialized))

    def test_ineligible_boundaries_clear_every_stale_evidence_without_io(
        self,
    ) -> None:
        stale = {
            "dissolve_entry_face_geometry_window": (_GEOMETRY,),
            "dissolve_exit_face_geometry_window": (_GEOMETRY,),
            "dissolve_entry_face_box_window": (_BOX,),
            "dissolve_exit_face_box_window": (_BOX,),
            "dissolve_entry_face_identity_window": (_IDENTITY,),
            "dissolve_exit_face_identity_window": (_IDENTITY,),
            "dissolve_entry_pose_embedding_window": (_POSE,),
            "dissolve_exit_pose_embedding_window": (_POSE,),
            "dissolve_entry_pose_backend_window": (_POSE_BACKEND,),
            "dissolve_exit_pose_backend_window": (_POSE_BACKEND,),
            "dissolve_entry_pose_confidence_window": (0.9,),
            "dissolve_exit_pose_confidence_window": (0.9,),
            "dissolve_entry_silhouette_embedding_window": (_SILHOUETTE,),
            "dissolve_exit_silhouette_embedding_window": (_SILHOUETTE,),
            "dissolve_entry_silhouette_backend_window": (
                _SILHOUETTE_BACKEND,
            ),
            "dissolve_exit_silhouette_backend_window": (
                _SILHOUETTE_BACKEND,
            ),
            "dissolve_entry_silhouette_confidence_window": (0.9,),
            "dissolve_exit_silhouette_confidence_window": (0.9,),
        }
        self.assertEqual(set(stale), set(_DISSOLVE_EVIDENCE_FIELD_NAMES))
        choices = (
            replace(_choice(0, 0), **stale),
            replace(_choice(1, 0), **stale),
            replace(_choice(2, 1, eligible_face=False), **stale),
        )
        plan = VisualTimelinePlan(choices, 2.0)

        class ForbiddenSampler:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise AssertionError("ineligible boundaries must not open media")

        with patch("crosscut.visual_score._VideoSampler", ForbiddenSampler):
            rehydrated = rehydrate_visual_timeline_dissolve_evidence(
                plan,
                sources=(Path("camera-0.mp4"), Path("camera-1.mp4")),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
            )

        for choice in rehydrated.choices:
            for field_name in _DISSOLVE_EVIDENCE_FIELD_NAMES:
                self.assertEqual(getattr(choice, field_name), ())

    def test_checkpoint_missing_appearance_promotes_only_after_rehydration(
        self,
    ) -> None:
        checkpoint_plan = _checkpoint_like_plan()
        face_match = _face_match(checkpoint_plan.choices)
        before = build_group_transition_plan(
            visual_timeline=checkpoint_plan,
            face_match_plan=face_match,
            source_durations_seconds=(4.0, 4.0),
        )
        self.assertEqual(
            before.transitions[0].reason,
            "future_face_appearance_evidence_unavailable",
        )

        with patch("crosscut.visual_score._VideoSampler", _Sampler), patch(
            "crosscut.visual_score._dissolve_evidence_window",
            return_value=_safe_evidence(),
        ):
            rehydrated = rehydrate_visual_timeline_dissolve_evidence(
                checkpoint_plan,
                sources=(Path("camera-0.mp4"), Path("camera-1.mp4")),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
            )

        after = build_group_transition_plan(
            visual_timeline=rehydrated,
            face_match_plan=face_match,
            source_durations_seconds=(4.0, 4.0),
        )
        self.assertEqual(after.transitions[0].kind, TransitionKind.FACE_DISSOLVE)
        self.assertEqual(after.transitions[0].tier, "A")


if __name__ == "__main__":
    unittest.main()
