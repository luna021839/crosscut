from __future__ import annotations

import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.audio import AlignmentResult, SourceAlignment
from crosscut.ive import AspectRatio, IVEMember
from crosscut.member_draft import MemberDraftConfig, build_member_draft
from crosscut.pipeline import _member_anchor_trim, _member_render_segments
from crosscut.render import (
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    aligned_render_segment,
    build_filter_graph,
)
from crosscut.tracking import (
    MemberTrackObservation,
    MemberTrackingResult,
    TrackingState,
)
from crosscut.timeline import ShotType
from crosscut.timewarp import (
    TimeWarpAnchor,
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)
from crosscut.visual_boundary_repair import repair_visual_timeline_boundaries
from crosscut.visual_score import (
    FrameVisualDescriptor,
    VisualTimelineChoice,
    VisualTimelinePlan,
    build_visual_timeline,
)


def _timewarp(global_offset: float, rate: float) -> TimeWarpMap:
    config = TimeWarpConfig(maximum_drift_rate=0.01)
    masters = (12.0, 108.0)
    anchors = tuple(
        TimeWarpAnchor(
            window_index=index,
            window_start_seconds=master - 12.0,
            window_end_seconds=master + 12.0,
            master_seconds=master,
            source_seconds=global_offset + rate * master,
            offset_seconds=global_offset + (rate - 1.0) * master,
            score=0.9,
            second_best_score=0.2,
            score_margin=0.7,
            accepted=True,
            trend_residual_seconds=0.0,
        )
        for index, master in enumerate(masters)
    )
    return TimeWarpMap(
        global_offset_seconds=global_offset,
        master_duration_seconds=120.0,
        source_duration_seconds=global_offset + rate * 120.0 + 1.0,
        control_points=tuple(
            TimeWarpControlPoint(anchor.master_seconds, anchor.source_seconds)
            for anchor in anchors
        ),
        anchors=anchors,
        residuals=TimeWarpResidualDiagnostics(0.0, 0.0, 0.0, 2),
        config=config,
        scalar_fallback=False,
        fallback_reason=None,
    )


def _scalar_timewarp(global_offset: float) -> TimeWarpMap:
    return TimeWarpMap(
        global_offset_seconds=global_offset,
        master_duration_seconds=120.0,
        source_duration_seconds=121.0,
        control_points=(),
        anchors=(),
        residuals=TimeWarpResidualDiagnostics(None, None, None, 0),
        config=TimeWarpConfig(),
        scalar_fallback=True,
        fallback_reason="test_scalar_fallback",
    )


def _descriptor() -> FrameVisualDescriptor:
    return FrameVisualDescriptor(
        quality_score=0.8,
        composition_score=0.8,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        face_count=0,
        structure_embedding=(0.1, 0.2, 0.3),
        person_layout_embedding=(0.2, 0.4, 0.3),
        motion=(0.0, 0.0),
    )


class _RecordingSampler:
    calls: dict[str, list[float]] = {}

    def __init__(self, path: Path) -> None:
        self.path = path
        self.frame_rate = 30.0
        self.calls.setdefault(path.name, [])

    def frame_at(self, seconds: float) -> np.ndarray:
        self.calls[self.path.name].append(float(seconds))
        return np.zeros((16, 16, 3), dtype=np.uint8)

    def descriptor_at(self, seconds: float) -> FrameVisualDescriptor:
        self.calls[self.path.name].append(float(seconds))
        return _descriptor()

    def transition_descriptor_at(self, seconds: float) -> FrameVisualDescriptor:
        self.calls[self.path.name].append(float(seconds))
        return _descriptor()

    def sface_embedding_at(
        self, _seconds: float, _descriptor_value: FrameVisualDescriptor
    ) -> tuple[float, ...]:
        return ()

    def close(self) -> None:
        pass


def _choice(
    index: int,
    source_index: int,
    timewarp: TimeWarpMap,
) -> VisualTimelineChoice:
    master_start = float(index)
    master_end = float(index + 1)
    source_start, source_end = timewarp.master_span_to_source(
        master_start, master_end
    )
    return VisualTimelineChoice(
        segment_index=index,
        source_index=source_index,
        canonical_master_start=master_start,
        canonical_master_end=master_end,
        source_local_start=source_start,
        source_local_end=source_end,
        quality_score=0.8,
        composition_score=0.8,
        boundary_score=0.5,
        shot_scale=0.4,
        shot_type=ShotType.MEDIUM,
        shot_id=f"{source_index}:shot",
    )


class VisualTimeWarpConsumptionTests(unittest.TestCase):
    def setUp(self) -> None:
        _RecordingSampler.calls = {}

    def test_visual_sampling_and_persisted_span_use_local_clock_map(self) -> None:
        mapped = _timewarp(0.5, 1.004)
        internal_cut_master = 21.25
        internal_cut_source = mapped.master_to_source(internal_cut_master)
        with patch(
            "crosscut.visual_score._VideoSampler", _RecordingSampler
        ), patch(
            "crosscut.visual_score.describe_frame", return_value=_descriptor()
        ):
            plan = build_visual_timeline(
                sources=(Path("scalar.mp4"), Path("warped.mp4")),
                canonical_spans=((20.0, 22.0),),
                audio_offsets=(0.0, 0.5),
                sync_scores=(0.0, 1.0),
                # Give both source candidates the same canonical detector cut
                # so candidate review remains neutral; the higher-sync mapped
                # source must still win and prove exact inverse serialization.
                scene_cut_times=((internal_cut_master,), (internal_cut_source,)),
                source_timewarps=(None, mapped),
            )

        self.assertEqual(plan.source_clock_basis, "local_monotonic_timewarp_with_per_source_scalar_fallback")
        self.assertEqual(len(plan.choices), 1)
        choice = plan.choices[0]
        self.assertEqual(choice.source_index, 1)
        self.assertAlmostEqual(choice.source_local_start, mapped.master_to_source(20.0))
        self.assertAlmostEqual(choice.source_local_end, mapped.master_to_source(22.0))
        self.assertAlmostEqual(choice.source_playback_rate, 1.004)
        self.assertEqual(len(choice.internal_scene_cut_master_seconds), 1)
        self.assertAlmostEqual(
            choice.internal_scene_cut_master_seconds[0],
            internal_cut_master,
            places=9,
        )
        self.assertAlmostEqual(
            choice.to_dict()["internal_scene_cut_master_seconds"][0],
            internal_cut_master,
            places=9,
        )

        sampled = _RecordingSampler.calls["warped.mp4"]
        for master_time in (20.05, 21.0, 21.95):
            expected = mapped.master_to_source(master_time)
            self.assertTrue(
                any(math.isclose(value, expected, abs_tol=1e-9) for value in sampled),
                f"mapped sample {expected} was not decoded",
            )

    def test_visual_timeline_rejects_misaligned_map_arrays_and_offsets(self) -> None:
        mapped = _timewarp(0.5, 1.004)
        common = {
            "sources": (Path("a"), Path("b")),
            "canonical_spans": ((1.0, 2.0),),
            "audio_offsets": (0.0, 0.5),
            "sync_scores": (1.0, 1.0),
            "scene_cut_times": ((), ()),
        }
        with self.assertRaisesRegex(ValueError, "match the source count"):
            build_visual_timeline(**common, source_timewarps=(mapped,))
        with self.assertRaisesRegex(ValueError, "does not match audio_offsets"):
            build_visual_timeline(**common, source_timewarps=(mapped, None))

    def test_boundary_repair_persists_exact_mapped_boundary_clocks(self) -> None:
        outgoing_map = _timewarp(0.2, 1.004)
        incoming_map = _timewarp(0.5, 0.998)
        plan = VisualTimelinePlan(
            (
                _choice(0, 0, outgoing_map),
                _choice(1, 1, incoming_map),
            ),
            2.0,
            source_clock_basis="local_monotonic_timewarp_with_per_source_scalar_fallback",
        )

        repaired = repair_visual_timeline_boundaries(
            visual_timeline=plan,
            sources=(Path("a"), Path("b")),
            audio_offsets=(0.2, 0.5),
            scene_cut_times=((), ()),
            descriptor_provider=lambda _source, _seconds: _descriptor(),
            source_timewarps=(outgoing_map, incoming_map),
        )

        boundary = repaired.timeline.choices[0].canonical_master_end
        self.assertAlmostEqual(
            repaired.timeline.choices[0].source_local_end,
            outgoing_map.master_to_source(boundary),
        )
        self.assertAlmostEqual(
            repaired.timeline.choices[1].source_local_start,
            incoming_map.master_to_source(boundary),
        )
        self.assertEqual(
            repaired.timeline.source_clock_basis, plan.source_clock_basis
        )


class RenderTimeWarpConsumptionTests(unittest.TestCase):
    def test_monotonic_map_inverse_round_trips_source_tracking_times(self) -> None:
        mapped = _timewarp(0.5, 1.004)
        for master_time in (0.0, 12.0, 37.25, 108.0, 120.0):
            self.assertAlmostEqual(
                mapped.source_to_master(mapped.master_to_source(master_time)),
                master_time,
                places=9,
            )

    def test_filter_consumes_mapped_source_span_and_restores_canonical_duration(self) -> None:
        mapped = _timewarp(0.5, 1.004)
        segment = aligned_render_segment(
            source_index=0,
            reference_start=10.0,
            common_master_start=10.0,
            source_offset=0.5,
            duration=2.0,
            timewarp=mapped,
        )
        plan = RenderPlan(
            sources=(Path("source.mp4"),),
            segments=(segment,),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )

        self.assertAlmostEqual(segment.source_start, 10.54)
        self.assertAlmostEqual(segment.effective_source_duration, 2.008)
        self.assertAlmostEqual(segment.source_playback_rate, 1.004)
        self.assertEqual(plan.output_frame_count, 60)
        graph = build_filter_graph(plan)
        self.assertIn("trim=start=10.540000:duration=2.008000", graph)
        self.assertIn("setpts=0.996015936255*(PTS-STARTPTS)", graph)
        serialized = plan.to_json()["segments"]
        assert isinstance(serialized, list)
        self.assertAlmostEqual(serialized[0]["source_playback_rate"], 1.004)  # type: ignore[arg-type,index]

    def test_dissolve_handle_uses_same_local_rate_and_exact_source_boundary(self) -> None:
        first = RenderSegment(
            0,
            0.5,
            1.0,
            0.0,
            source_duration=1.004,
        )
        second = RenderSegment(1, 0.0, 1.0, 1.0)
        plan = RenderPlan(
            sources=(Path("first.mp4"), Path("second.mp4")),
            segments=(first, second),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(TransitionKind.FACE_DISSOLVE, 11, "safe"),
            ),
        )

        graph = build_filter_graph(plan)
        self.assertIn("trim=start=1.504000:duration=0.368133", graph)
        self.assertGreaterEqual(
            graph.count("setpts=0.996015936255*(PTS-STARTPTS)"), 2
        )

    def test_scalar_fallback_retains_legacy_setpts_and_offset_contract(self) -> None:
        scalar = _scalar_timewarp(0.75)
        segment = aligned_render_segment(
            source_index=0,
            reference_start=12.5,
            common_master_start=12.5,
            source_offset=0.75,
            duration=2.0,
            timewarp=scalar,
        )
        plan = RenderPlan(
            sources=(Path("source.mp4"),),
            segments=(segment,),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
        )

        self.assertEqual(segment.source_start, 13.25)
        self.assertIsNone(segment.source_duration)
        self.assertEqual(segment.source_playback_rate, 1.0)
        graph = build_filter_graph(plan)
        self.assertIn("setpts=PTS-STARTPTS", graph)
        self.assertNotIn("*(PTS-STARTPTS)", graph)

    def test_render_segment_rejects_large_or_mismatched_timewarp(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety bound"):
            RenderSegment(
                0,
                0.0,
                1.0,
                0.0,
                source_duration=1.03,
            )
        with self.assertRaisesRegex(ValueError, "does not match source_offset"):
            aligned_render_segment(
                source_index=0,
                reference_start=10.0,
                common_master_start=10.0,
                source_offset=0.4,
                duration=1.0,
                timewarp=_timewarp(0.5, 1.004),
            )


class MemberTimeWarpConsumptionTests(unittest.TestCase):
    @staticmethod
    def _tracking_at(times: tuple[float, ...]) -> MemberTrackingResult:
        return MemberTrackingResult(
            member=IVEMember.GAEUL,
            source_id="mapped-member",
            fps=30.0,
            backend_name="synthetic",
            observations=tuple(
                MemberTrackObservation(
                    frame_index=index,
                    timestamp_seconds=timestamp,
                    bbox_xyxy=(0.30, 0.15, 0.55, 0.90),
                    confidence=0.98,
                    state=TrackingState.TRACKED,
                    member=IVEMember.GAEUL,
                    source_id="mapped-member",
                )
                for index, timestamp in enumerate(times)
            ),
            terminal_failure=None,
        )

    def test_member_tracking_crop_and_render_share_mapped_segment_clock(self) -> None:
        mapped = _timewarp(0.5, 1.004)
        master_start, master_end = 20.0, 22.0
        local_start, local_end = mapped.master_span_to_source(
            master_start, master_end
        )
        local_middle = (local_start + local_end) / 2.0
        timeline = build_member_draft(
            canonical_spans=((master_start, master_end),),
            audio_offsets=(0.5,),
            tracking_by_source=(
                self._tracking_at((local_start, local_middle, local_end)),
            ),
            target_member=IVEMember.GAEUL,
            output_aspect=AspectRatio.LANDSCAPE,
            source_timewarps=(mapped,),
            config=MemberDraftConfig(
                minimum_interval_coverage=0.0,
                maximum_untracked_gap_seconds=10.0,
            ),
        )

        member_segment = timeline.segments[0]
        self.assertTrue(member_segment.source_timewarp_applied)
        self.assertAlmostEqual(member_segment.source_local_start, local_start)
        self.assertAlmostEqual(member_segment.source_local_end, local_end)
        self.assertTrue(
            any(
                math.isclose(sample.time_seconds, 1.0, abs_tol=1e-9)
                for sample in member_segment.crop_samples
            )
        )
        timing_segment = type(
            "TimingSegment",
            (),
            {
                "segment_index": 0,
                "canonical_master_start": master_start,
                "canonical_master_end": master_end,
                "source_index": 0,
                "audio_offset_seconds": 0.5,
                "source_local_start": master_start + 0.5,
                "source_local_end": master_end + 0.5,
            },
        )()
        rendered, _transitions, _fallbacks = _member_render_segments(
            member_segments=timeline.segments,
            timing_segments=(timing_segment,),
            eligible_indices=(0,),
            common_master_start=master_start,
            source_timewarps=(mapped,),
        )
        self.assertAlmostEqual(rendered[0].source_start, local_start)
        self.assertAlmostEqual(
            rendered[0].effective_source_duration,
            local_end - local_start,
        )
        assert rendered[0].pan_crop is not None
        self.assertTrue(
            any(
                math.isclose(keyframe.time_seconds, 1.0, abs_tol=1e-9)
                for keyframe in rendered[0].pan_crop.keyframes
            )
        )

    def test_manual_anchor_trim_inverts_the_checked_source_clock(self) -> None:
        mapped = _timewarp(0.5, 1.004)
        source_anchor = mapped.master_to_source(20.0)
        alignment = AlignmentResult(
            reference="mapped-member",
            sample_rate=4_000,
            alignments=(
                SourceAlignment(
                    "mapped-member", False, 0, 0.5, 0.9, 40_000, 10.0
                ),
            ),
        )
        effective_start, payload = _member_anchor_trim(
            common_master_start=0.0,
            common_master_end=120.0,
            alignment=alignment,
            eligible_indices=(0,),
            tracking_results={0: self._tracking_at((source_anchor,))},
            source_timewarps=(mapped,),
        )

        self.assertAlmostEqual(effective_start, 20.0)
        self.assertAlmostEqual(
            payload["anchors"][0]["canonical_anchor_time_seconds"],
            20.0,
        )
        self.assertEqual(
            payload["anchors"][0]["source_clock_basis"],
            "local_monotonic_timewarp",
        )

if __name__ == "__main__":
    unittest.main()
