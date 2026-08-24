from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.ive import AspectRatio
from crosscut.professional_reference_gate import (
    ProfessionalReferenceGateResult,
    ProfessionalReferenceProfile,
)
from crosscut.render import (
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from crosscut.rendered_proxy_review import (
    BOUNDARY_FRAME_OFFSETS,
    INTRODUCED_CROSS_SOURCE_DOMAIN,
    NATIVE_SOURCE_CUT_DOMAIN,
    OutputSubjectGeometry,
    PROXY_LANDSCAPE_SIZE,
    PROXY_PORTRAIT_SIZE,
    ProxyReviewConfig,
    TIER_D_CONTINUITY_DOMAIN,
    _box_discontinuity,
    _motion_discontinuity,
    derive_proxy_render_plan,
    render_and_review_proxy,
    review_rendered_proxy,
)


class _FakeProxyCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.frames = frames
        self.width = width
        self.height = height
        self.fps = fps
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return (False, None)
        frame = self.frames[self.position]
        self.position += 1
        return (True, frame.copy())

    def release(self) -> None:
        self.released = True


class RenderedProxyReviewTests(unittest.TestCase):
    def _plan(self, aspect_ratio: AspectRatio) -> RenderPlan:
        width, height = (
            (1080, 1920)
            if aspect_ratio is AspectRatio.PORTRAIT
            else (1920, 1080)
        )
        return RenderPlan(
            sources=(Path("source.mp4"),),
            segments=(RenderSegment(0, 0.0, 0.5, 0.0),),
            reference_source=0,
            aspect_ratio=aspect_ratio,
            output=Path("full.mp4"),
            width=width,
            height=height,
            fps=30.0,
        )

    def _textured_frames(
        self,
        count: int,
        *,
        color: tuple[int, int, int] = (35, 90, 145),
        start_shift: int = 0,
    ) -> list[np.ndarray]:
        y_grid, x_grid = np.indices((48, 72))
        checker = (((x_grid // 4) + (y_grid // 4)) % 2) * 70
        frames: list[np.ndarray] = []
        for index in range(count):
            frame = np.empty((48, 72, 3), dtype=np.uint8)
            for channel, base in enumerate(color):
                frame[..., channel] = np.clip(
                    base + checker,
                    0,
                    255,
                ).astype(np.uint8)
            frames.append(
                np.roll(frame, start_shift + index, axis=1)
            )
        return frames

    def _two_segment_plan(
        self,
        *,
        same_source: bool,
        tier: str,
        reason: str,
    ) -> RenderPlan:
        frame_duration = 6.0 / 30.0
        sources = (
            (Path("same.mp4"),)
            if same_source
            else (Path("first.mp4"), Path("second.mp4"))
        )
        return RenderPlan(
            sources=sources,
            segments=(
                RenderSegment(0, 0.0, frame_duration, 0.0),
                RenderSegment(
                    0 if same_source else 1,
                    frame_duration if same_source else 0.0,
                    frame_duration,
                    frame_duration,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("synthetic-proxy.mp4"),
            width=854,
            height=480,
            fps=30.0,
            transitions=(RenderTransition(tier=tier, reason=reason),),
        )

    def _review_frames(
        self,
        plan: RenderPlan,
        frames: list[np.ndarray],
        *,
        geometry_provider=None,
        professional_reference_profile=None,
    ):
        capture = _FakeProxyCapture(
            frames,
            width=plan.width,
            height=plan.height,
            fps=plan.fps,
        )
        provider = geometry_provider or (
            lambda _frame: OutputSubjectGeometry()
        )
        with patch(
            "crosscut.rendered_proxy_review.cv2.VideoCapture",
            return_value=capture,
        ):
            review = review_rendered_proxy(
                plan,
                geometry_provider=provider,
                professional_reference_profile=(
                    professional_reference_profile
                ),
            )
        self.assertTrue(capture.released)
        return review

    def test_derives_exact_landscape_proxy_without_changing_edit_grammar(self) -> None:
        plan = self._plan(AspectRatio.LANDSCAPE)
        proxy = derive_proxy_render_plan(plan, Path("proxy.mp4"))

        self.assertEqual((proxy.width, proxy.height), PROXY_LANDSCAPE_SIZE)
        self.assertEqual(proxy.sources, plan.sources)
        self.assertEqual(proxy.segments, plan.segments)
        self.assertEqual(proxy.transitions, plan.transitions)
        self.assertEqual(proxy.reference_source, plan.reference_source)
        self.assertEqual(proxy.master_audio_start, plan.master_audio_start)
        self.assertEqual(proxy.fps, plan.fps)
        self.assertEqual(proxy.output, Path("proxy.mp4"))

    def test_derives_exact_portrait_proxy(self) -> None:
        proxy = derive_proxy_render_plan(
            self._plan(AspectRatio.PORTRAIT),
            Path("portrait-proxy.mp4"),
        )
        self.assertEqual((proxy.width, proxy.height), PROXY_PORTRAIT_SIZE)

    def test_configuration_and_geometry_validate_normalized_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalized"):
            ProxyReviewConfig(maximum_color_shock=1.1)
        with self.assertRaisesRegex(ValueError, "tolerance multiplier"):
            ProxyReviewConfig(dissolve_endpoint_tolerance_multiplier=1.26)
        with self.assertRaisesRegex(ValueError, "bounds"):
            OutputSubjectGeometry(face_bbox_xywh=(0.8, 0.2, 0.4, 0.3))

    def test_one_sided_and_group_face_geometry_is_not_fabricated_as_maximum(self) -> None:
        self.assertEqual(
            _box_discontinuity(None, (0.2, 0.2, 0.2, 0.3)),
            (None, None),
        )
        plan = self._two_segment_plan(
            same_source=False,
            tier="C",
            reason="synthetic_cross_source",
        )
        calls = 0

        def geometry_provider(_frame: np.ndarray) -> OutputSubjectGeometry:
            nonlocal calls
            calls += 1
            if calls <= 2:
                return OutputSubjectGeometry(
                    face_count=1,
                    face_bbox_xywh=(0.30, 0.20, 0.18, 0.28),
                    face_backend="synthetic",
                )
            return OutputSubjectGeometry(
                face_count=2,
                face_backend="synthetic_group",
            )

        review = self._review_frames(
            plan,
            self._textured_frames(12),
            geometry_provider=geometry_provider,
        )
        boundary = review.payload["boundaries"][0]
        self.assertEqual(
            boundary["review_domain"],
            INTRODUCED_CROSS_SOURCE_DOMAIN,
        )
        self.assertFalse(boundary["geometry_comparability"]["face"])
        self.assertIsNone(
            boundary["discontinuities"]["face_scale_discontinuity"]
        )
        self.assertIsNone(
            boundary["discontinuities"]["face_position_discontinuity"]
        )
        self.assertNotIn(
            "face_scale_discontinuity",
            boundary["issue_reasons"],
        )
        self.assertNotIn(
            "face_position_discontinuity",
            boundary["issue_reasons"],
        )

    def test_unavailable_motion_is_not_maximum_but_reliable_opposition_is(self) -> None:
        unavailable = {
            "direction_xy": [1.0, 0.0],
            "direction_available": False,
        }
        reliable_right = {
            "direction_xy": [1.0, 0.0],
            "direction_available": True,
        }
        reliable_left = {
            "direction_xy": [-1.0, 0.0],
            "direction_available": True,
        }
        self.assertIsNone(
            _motion_discontinuity(unavailable, reliable_right)
        )
        self.assertIsNone(
            _motion_discontinuity(
                {"direction_xy": None, "direction_available": False},
                reliable_right,
            )
        )
        self.assertEqual(
            _motion_discontinuity(reliable_right, reliable_left),
            1.0,
        )

    def test_tier_d_continuity_ignores_subject_dropout_on_smooth_frames(self) -> None:
        plan = self._two_segment_plan(
            same_source=True,
            tier="D",
            reason="same_source_hold_no_visible_switch",
        )
        calls = 0

        def geometry_provider(_frame: np.ndarray) -> OutputSubjectGeometry:
            nonlocal calls
            calls += 1
            return (
                OutputSubjectGeometry(
                    face_count=1,
                    face_bbox_xywh=(0.30, 0.20, 0.18, 0.28),
                )
                if calls <= 2
                else OutputSubjectGeometry(face_count=2)
            )

        review = self._review_frames(
            plan,
            self._textured_frames(12),
            geometry_provider=geometry_provider,
        )
        boundary = review.payload["boundaries"][0]
        self.assertEqual(boundary["review_domain"], TIER_D_CONTINUITY_DOMAIN)
        self.assertTrue(boundary["source_clock_contiguous"])
        self.assertTrue(boundary["passed"])
        self.assertEqual(boundary["issue_reasons"], [])
        self.assertEqual(
            review.payload["visible_boundary_score"]["distribution"]["count"],
            0,
        )
        self.assertEqual(
            review.payload["boundary_domains"][TIER_D_CONTINUITY_DOMAIN][
                "issue_count"
            ],
            0,
        )

    def test_tier_d_continuity_detects_concat_duplicate_or_freeze(self) -> None:
        plan = self._two_segment_plan(
            same_source=True,
            tier="D",
            reason="same_source_hold_no_visible_switch",
        )
        frames = self._textured_frames(12)
        frames[6] = frames[5].copy()
        review = self._review_frames(plan, frames)
        boundary = review.payload["boundaries"][0]
        self.assertFalse(boundary["passed"])
        self.assertTrue(
            boundary["temporal_integrity"]["duplicate_or_freeze_detected"]
        )
        self.assertIn(
            "tier_d_cross_boundary_duplicate_or_freeze",
            boundary["issue_reasons"],
        )
        self.assertFalse(review.passed)

    def test_tier_d_continuity_detects_cross_boundary_drop_or_reset(self) -> None:
        plan = self._two_segment_plan(
            same_source=True,
            tier="D",
            reason="same_source_hold_no_visible_switch",
        )
        frames = self._textured_frames(6, color=(20, 70, 120))
        frames.extend(
            self._textured_frames(
                6,
                color=(170, 25, 35),
                start_shift=6,
            )
        )
        review = self._review_frames(plan, frames)
        boundary = review.payload["boundaries"][0]
        self.assertTrue(
            boundary["temporal_integrity"]["drop_or_reset_detected"]
        )
        self.assertIn(
            "tier_d_cross_boundary_drop_or_reset",
            boundary["issue_reasons"],
        )
        self.assertFalse(review.passed)

    def test_same_source_native_cut_is_pass_through_not_editorial_mismatch(self) -> None:
        plan = self._two_segment_plan(
            same_source=True,
            tier="C",
            reason="same_source_internal_camera_cut",
        )
        frames = self._textured_frames(6, color=(20, 55, 170))
        frames.extend(
            self._textured_frames(6, color=(170, 55, 20), start_shift=6)
        )
        review = self._review_frames(plan, frames)
        boundary = review.payload["boundaries"][0]
        self.assertEqual(boundary["review_domain"], NATIVE_SOURCE_CUT_DOMAIN)
        self.assertFalse(boundary["editorial_continuity_applicable"])
        self.assertTrue(boundary["passed"])
        self.assertEqual(boundary["issue_reasons"], [])
        self.assertTrue(
            boundary["diagnostic_editorial_issue_reasons"]
        )
        self.assertTrue(review.passed)

    def test_same_source_native_cut_still_gates_black_and_freeze_artifacts(self) -> None:
        plan = self._two_segment_plan(
            same_source=True,
            tier="C",
            reason="same_source_internal_camera_cut",
        )
        black_frames = self._textured_frames(12)
        black_frames[6] = np.zeros_like(black_frames[6])
        black_review = self._review_frames(plan, black_frames)
        black_boundary = black_review.payload["boundaries"][0]
        self.assertIn(
            "native_source_cut_black_frame",
            black_boundary["issue_reasons"],
        )
        self.assertIn(
            "native_source_cut_boundary_frame_quality",
            black_boundary["issue_reasons"],
        )

        freeze_frames = self._textured_frames(12)
        freeze_frames[6] = freeze_frames[5].copy()
        freeze_review = self._review_frames(plan, freeze_frames)
        freeze_boundary = freeze_review.payload["boundaries"][0]
        self.assertIn(
            "native_source_cut_cross_boundary_duplicate_or_freeze",
            freeze_boundary["issue_reasons"],
        )

    def test_same_source_native_cut_requires_contiguous_pass_through_clock(self) -> None:
        plan = self._two_segment_plan(
            same_source=True,
            tier="C",
            reason="same_source_internal_camera_cut",
        )
        discontinuous = RenderPlan(
            sources=plan.sources,
            segments=(
                plan.segments[0],
                RenderSegment(0, 1.0, 0.2, 0.2),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("native-clock-gap.mp4"),
            width=854,
            height=480,
            fps=30.0,
            transitions=plan.transitions,
        )
        review = self._review_frames(
            discontinuous,
            self._textured_frames(12),
        )
        boundary = review.payload["boundaries"][0]
        self.assertFalse(boundary["source_clock_contiguous"])
        self.assertIn(
            "native_source_cut_source_clock_discontinuity",
            boundary["issue_reasons"],
        )
        self.assertFalse(review.passed)

    def test_cross_source_internal_camera_cut_reason_remains_editorial_gate(self) -> None:
        plan = self._two_segment_plan(
            same_source=False,
            tier="C",
            reason="internal_camera_cut",
        )
        frames = self._textured_frames(6, color=(20, 55, 170))
        frames.extend(
            self._textured_frames(6, color=(170, 55, 20), start_shift=6)
        )
        review = self._review_frames(plan, frames)
        boundary = review.payload["boundaries"][0]
        self.assertEqual(
            boundary["review_domain"],
            INTRODUCED_CROSS_SOURCE_DOMAIN,
        )
        self.assertEqual(boundary["transition_reason"], "internal_camera_cut")
        self.assertTrue(boundary["editorial_continuity_applicable"])
        self.assertIn("color_shock", boundary["issue_reasons"])
        self.assertEqual(
            review.payload[
                "introduced_cross_source_boundary_issue_indices"
            ],
            [0],
        )
        self.assertFalse(review.passed)

    def test_cross_source_black_and_duplicate_frames_remain_zero_tolerance(self) -> None:
        plan = self._two_segment_plan(
            same_source=False,
            tier="C",
            reason="synthetic_cross_source",
        )
        black_frames = self._textured_frames(12)
        black_frames[6] = np.zeros_like(black_frames[6])
        black_review = self._review_frames(plan, black_frames)
        self.assertIn(
            "introduced_cross_source_black_frame",
            black_review.payload["boundaries"][0][
                "technical_integrity_issue_reasons"
            ],
        )
        self.assertFalse(black_review.passed)

        duplicate_frames = self._textured_frames(12)
        duplicate_frames[6] = duplicate_frames[5].copy()
        duplicate_review = self._review_frames(plan, duplicate_frames)
        self.assertIn(
            "introduced_cross_source_duplicate_or_freeze",
            duplicate_review.payload["boundaries"][0][
                "technical_integrity_issue_reasons"
            ],
        )
        self.assertFalse(duplicate_review.passed)

    def test_professional_gate_replaces_only_absolute_visible_selection_gate(self) -> None:
        plan = self._two_segment_plan(
            same_source=False,
            tier="C",
            reason="synthetic_cross_source",
        )
        frames = self._textured_frames(6, color=(20, 55, 170))
        frames.extend(
            self._textured_frames(6, color=(170, 55, 20), start_shift=6)
        )
        profile = ProfessionalReferenceProfile(
            profile_path=Path("profile.json"),
            profile_sha256="0" * 64,
            artist="IVE",
            song="SYNTHETIC",
            references=(),
            payload={},
        )
        relative_pass = ProfessionalReferenceGateResult(
            {
                "passed": True,
                "technical_integrity": {"passed": True},
            }
        )

        with patch(
            "crosscut.professional_reference_gate."
            "evaluate_professional_reference_gate",
            return_value=relative_pass,
        ):
            review = self._review_frames(
                plan,
                frames,
                professional_reference_profile=profile,
            )

        self.assertFalse(
            review.payload["boundary_domains"][
                INTRODUCED_CROSS_SOURCE_DOMAIN
            ]["passed"]
        )
        self.assertFalse(
            review.payload["visible_boundary_score"]["selection_gate"]
        )
        self.assertTrue(review.payload["professional_reference_gate"]["passed"])
        self.assertTrue(review.passed)

    def test_domain_summaries_use_only_introduced_cross_source_denominator(self) -> None:
        frame_duration = 6.0 / 30.0
        plan = RenderPlan(
            sources=(Path("same.mp4"), Path("other.mp4")),
            segments=(
                RenderSegment(0, 0.0, frame_duration, 0.0),
                RenderSegment(0, frame_duration, frame_duration, frame_duration),
                RenderSegment(
                    0,
                    frame_duration * 2,
                    frame_duration,
                    frame_duration * 2,
                ),
                RenderSegment(
                    1,
                    0.0,
                    frame_duration,
                    frame_duration * 3,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("mixed-domains.mp4"),
            width=854,
            height=480,
            fps=30.0,
            transitions=(
                RenderTransition(
                    tier="D",
                    reason="same_source_hold_no_visible_switch",
                ),
                RenderTransition(
                    tier="C",
                    reason="same_source_internal_camera_cut",
                ),
                RenderTransition(tier="C", reason="internal_camera_cut"),
            ),
        )
        frames = self._textured_frames(18, color=(20, 55, 170))
        frames.extend(
            self._textured_frames(6, color=(170, 55, 20), start_shift=18)
        )
        review = self._review_frames(plan, frames)
        domains = review.payload["boundary_domains"]
        self.assertEqual(
            {
                name: domains[name]["boundary_count"]
                for name in (
                    INTRODUCED_CROSS_SOURCE_DOMAIN,
                    TIER_D_CONTINUITY_DOMAIN,
                    NATIVE_SOURCE_CUT_DOMAIN,
                )
            },
            {
                INTRODUCED_CROSS_SOURCE_DOMAIN: 1,
                TIER_D_CONTINUITY_DOMAIN: 1,
                NATIVE_SOURCE_CUT_DOMAIN: 1,
            },
        )
        self.assertEqual(
            review.payload["visible_boundary_score"]["distribution"]["count"],
            1,
        )
        self.assertEqual(
            review.payload["boundary_score"]["distribution"]["count"],
            3,
        )
        self.assertEqual(
            review.payload["visible_boundary_score"]["issue_fraction"],
            1.0,
        )
        self.assertAlmostEqual(
            review.payload["boundary_score"]["issue_fraction"],
            1.0 / 3.0,
        )
        self.assertEqual(
            review.payload["partial_reedit_boundary_issue_indices"],
            [2],
        )

    def test_dissolve_reviews_completed_endpoint_frames_and_can_fail(self) -> None:
        plan = RenderPlan(
            sources=(Path("first.mp4"), Path("second.mp4")),
            segments=(
                RenderSegment(0, 0.0, 10.0 / 30.0, 0.0),
                RenderSegment(1, 10.0 / 30.0, 10.0 / 30.0, 10.0 / 30.0),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("dissolve-proxy.mp4"),
            width=854,
            height=480,
            fps=30.0,
            transitions=(
                RenderTransition(
                    kind=TransitionKind.FACE_DISSOLVE,
                    frame_count=3,
                    reason="synthetic dissolve",
                ),
            ),
        )
        capture = _FakeProxyCapture(
            [np.full((32, 48, 3), 112, dtype=np.uint8) for _ in range(20)],
            width=854,
            height=480,
            fps=30.0,
        )
        geometry_calls = 0

        def geometry_provider(_frame: np.ndarray) -> OutputSubjectGeometry:
            nonlocal geometry_calls
            geometry_calls += 1
            left = geometry_calls <= 2
            x = 0.08 if left else 0.72
            return OutputSubjectGeometry(
                face_count=1,
                face_bbox_xywh=(x, 0.20, 0.18, 0.24),
                face_backend="synthetic",
                silhouette_bbox_xywh=(x, 0.10, 0.24, 0.76),
                silhouette_centroid_xy=(x + 0.12, 0.48),
                silhouette_confidence=1.0,
                silhouette_backend="synthetic",
            )

        with patch(
            "crosscut.rendered_proxy_review.cv2.VideoCapture",
            return_value=capture,
        ):
            review = review_rendered_proxy(
                plan,
                geometry_provider=geometry_provider,
            )

        self.assertTrue(capture.released)
        self.assertEqual(geometry_calls, 4)
        self.assertEqual(review.payload["schema_version"], 3)
        self.assertEqual(review.payload["transition_count"], 1)
        self.assertEqual(review.payload["reviewed_boundary_count"], 1)
        self.assertEqual(review.payload["hard_cut_count"], 0)
        self.assertEqual(review.payload["dissolve_count"], 1)
        boundary = review.payload["boundaries"][0]
        self.assertEqual(boundary["transition_kind"], "face_dissolve")
        self.assertEqual(boundary["transition_frame_count"], 3)
        self.assertAlmostEqual(boundary["transition_duration_seconds"], 0.1)
        self.assertEqual(boundary["sample_offsets"], [-2, -1, 3, 4])
        self.assertEqual(boundary["frame_indices"], [8, 9, 13, 14])
        self.assertEqual(
            [item["frame_index"] for item in boundary["frame_metrics"]],
            [8, 9, 13, 14],
        )
        self.assertEqual(
            boundary["endpoint_review_policy"]["threshold_multiplier"],
            1.20,
        )
        self.assertIn("face_position_discontinuity", boundary["issue_reasons"])
        self.assertFalse(boundary["passed"])
        self.assertEqual(review.payload["boundary_issue_indices"], [0])
        self.assertLess(review.payload["boundary_score"]["p10"], 100.0)

    def test_dissolve_without_two_post_frames_is_a_failed_boundary(self) -> None:
        plan = RenderPlan(
            sources=(Path("first.mp4"), Path("second.mp4")),
            segments=(
                RenderSegment(0, 0.0, 10.0 / 30.0, 0.0),
                RenderSegment(1, 10.0 / 30.0, 4.0 / 30.0, 10.0 / 30.0),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("short-dissolve-proxy.mp4"),
            width=854,
            height=480,
            fps=30.0,
            transitions=(
                RenderTransition(
                    kind=TransitionKind.FACE_DISSOLVE,
                    frame_count=3,
                    reason="short synthetic dissolve",
                ),
            ),
        )
        capture = _FakeProxyCapture(
            [np.full((32, 48, 3), 112, dtype=np.uint8) for _ in range(14)],
            width=854,
            height=480,
            fps=30.0,
        )

        def unexpected_geometry(_frame: np.ndarray) -> OutputSubjectGeometry:
            self.fail("an unreviewable dissolve must not sample another shot")

        with patch(
            "crosscut.rendered_proxy_review.cv2.VideoCapture",
            return_value=capture,
        ):
            review = review_rendered_proxy(
                plan,
                geometry_provider=unexpected_geometry,
            )

        self.assertTrue(capture.released)
        self.assertEqual(review.payload["reviewed_boundary_count"], 1)
        boundary = review.payload["boundaries"][0]
        self.assertEqual(boundary["frame_indices"], [8, 9, 13, 14])
        self.assertEqual(boundary["frame_metrics"], [])
        self.assertEqual(boundary["weighted_score"], 0.0)
        self.assertEqual(
            boundary["issue_reasons"],
            ["insufficient_post_transition_review_handle"],
        )
        self.assertEqual(review.payload["boundary_issue_indices"], [0])
        self.assertEqual(review.payload["boundary_score"]["p10"], 0.0)
        self.assertFalse(review.passed)


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
class RenderedProxyReviewIntegrationTests(unittest.TestCase):
    def _source(self, path: Path, *, hue: int) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=s=320x180:r=30:d=0.8,hue=h={hue}",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=0.8",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(path),
            ],
            check=True,
        )

    def test_renders_decodes_and_serializes_four_frame_hard_cut_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mp4"
            second = root / "second.mp4"
            self._source(first, hue=0)
            self._source(second, hue=120)
            full_plan = RenderPlan(
                sources=(first, second),
                segments=(
                    RenderSegment(0, 0.0, 0.3, 0.0),
                    RenderSegment(1, 0.3, 0.3, 0.3),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=root / "unused-full.mp4",
                width=1920,
                height=1080,
                fps=30.0,
            )
            calls = 0

            def geometry_provider(_frame):
                nonlocal calls
                calls += 1
                # The first two calls are outgoing frames and the next two are
                # incoming frames.  This deliberately creates a screen-space
                # subject jump that the decoded-output reviewer must surface.
                center_x = 0.22 if calls <= 2 else 0.78
                return OutputSubjectGeometry(
                    face_count=1,
                    face_bbox_xywh=(center_x - 0.08, 0.25, 0.16, 0.22),
                    face_backend="synthetic_geometry",
                    silhouette_bbox_xywh=(center_x - 0.15, 0.12, 0.30, 0.72),
                    silhouette_centroid_xy=(center_x, 0.48),
                    silhouette_confidence=1.0,
                    silhouette_backend="synthetic_geometry",
                )

            proxy_plan, review = render_and_review_proxy(
                full_plan,
                root / "proxy.mp4",
                geometry_provider=geometry_provider,
                video_codec="libx264",
            )

            self.assertTrue(proxy_plan.output.is_file())
            self.assertEqual(
                (proxy_plan.width, proxy_plan.height),
                PROXY_LANDSCAPE_SIZE,
            )
            self.assertEqual(review.payload["proxy"]["decoded_frame_count"], 18)
            self.assertEqual(review.payload["transition_count"], 1)
            self.assertEqual(review.payload["reviewed_boundary_count"], 1)
            self.assertEqual(review.payload["hard_cut_count"], 1)
            self.assertEqual(review.payload["dissolve_count"], 0)
            self.assertEqual(calls, 4)
            boundary = review.payload["boundaries"][0]
            self.assertEqual(boundary["transition_kind"], "hard_cut")
            self.assertEqual(boundary["transition_frame_count"], 0)
            self.assertEqual(boundary["transition_duration_seconds"], 0.0)
            self.assertEqual(
                boundary["sample_offsets"],
                list(BOUNDARY_FRAME_OFFSETS),
            )
            self.assertEqual(boundary["frame_indices"], [7, 8, 9, 10])
            self.assertEqual(
                [item["frame_index"] for item in boundary["frame_metrics"]],
                [7, 8, 9, 10],
            )
            self.assertIn(
                "face_position_discontinuity",
                boundary["issue_reasons"],
            )
            self.assertIn(
                "silhouette_position_discontinuity",
                boundary["issue_reasons"],
            )
            self.assertEqual(review.payload["boundary_issue_indices"], [0])
            self.assertEqual(
                len(review.payload["source_quality"]["per_source"]),
                2,
            )
            self.assertIsInstance(review.weighted_score, float)
            self.assertFalse(review.passed)
            storage = review.payload["storage_policy"]
            self.assertFalse(storage["frame_images_persisted"])
            self.assertFalse(storage["original_pixels_serialized"])
            self.assertFalse(
                storage["appearance_or_identity_embeddings_serialized"]
            )
            report_path = review.write(root / "proxy-review.json")
            self.assertTrue(report_path.is_file())

    def test_actual_ffmpeg_dissolve_reviews_frames_after_completed_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mp4"
            second = root / "second.mp4"
            self._source(first, hue=0)
            self._source(second, hue=45)
            full_plan = RenderPlan(
                sources=(first, second),
                segments=(
                    RenderSegment(0, 0.0, 0.3, 0.0),
                    RenderSegment(1, 0.3, 0.3, 0.3),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=root / "unused-full.mp4",
                width=1920,
                height=1080,
                fps=30.0,
                transitions=(
                    RenderTransition(
                        kind=TransitionKind.FACE_DISSOLVE,
                        frame_count=3,
                        reason="actual synthetic dissolve",
                    ),
                ),
            )

            proxy_plan, review = render_and_review_proxy(
                full_plan,
                root / "dissolve-proxy.mp4",
                geometry_provider=lambda _frame: OutputSubjectGeometry(),
                video_codec="libx264",
            )

            self.assertEqual(proxy_plan.output_frame_count, 18)
            self.assertEqual(review.payload["proxy"]["decoded_frame_count"], 18)
            self.assertEqual(review.payload["hard_cut_count"], 0)
            self.assertEqual(review.payload["dissolve_count"], 1)
            self.assertEqual(review.payload["reviewed_boundary_count"], 1)
            boundary = review.payload["boundaries"][0]
            self.assertEqual(boundary["sample_offsets"], [-2, -1, 3, 4])
            self.assertEqual(boundary["frame_indices"], [7, 8, 12, 13])
            self.assertEqual(
                [item["frame_index"] for item in boundary["frame_metrics"]],
                [7, 8, 12, 13],
            )


if __name__ == "__main__":
    unittest.main()
