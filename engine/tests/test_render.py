from __future__ import annotations

from pathlib import Path
import math
import unittest

from crosscut.ive import AspectRatio
from crosscut.render import (
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RENDER_PLAN_SOURCE_INDEX_DOMAIN,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    aligned_render_segment,
    build_filter_graph,
    output_size,
    segment_frame_counts,
)


class RenderTests(unittest.TestCase):
    def test_portrait_output_size(self) -> None:
        self.assertEqual(output_size(AspectRatio.PORTRAIT), (1080, 1920))

    def test_filter_uses_continuous_reference_audio(self) -> None:
        plan = RenderPlan(
            sources=(Path("one.mp4"), Path("two.mp4")),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0),
                RenderSegment(1, 1.2, 1.5, 1.0, NormalizedCrop(0.25, 0.0, 0.5, 1.0)),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.PORTRAIT,
            output=Path("out.mp4"),
            master_audio_start=3.25,
            width=1080,
            height=1920,
        )
        graph = build_filter_graph(plan)
        self.assertIn(
            f"[0:a]atrim=start=3.250000:duration={plan.render_duration:.6f}",
            graph,
        )
        self.assertNotIn("[1:a]", graph)
        self.assertIn("crop=", graph)
        self.assertIn("setsar=1", graph)
        self.assertIn("concat=n=2", graph)

    def test_global_frame_quantization_does_not_accumulate_per_cut_rounding(self) -> None:
        duration = 1.0 / 12.0
        segments = tuple(
            RenderSegment(index % 2, index * duration, duration, index * duration)
            for index in range(12)
        )
        plan = RenderPlan(
            sources=(Path("one.mp4"), Path("two.mp4")),
            segments=segments,
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )
        counts = segment_frame_counts(plan)

        self.assertEqual(sum(counts), 30)
        self.assertEqual(plan.output_frame_count, 30)
        self.assertIn(
            "[vcat]fps=30.000000,tpad=stop_mode=clone:stop_duration=1,trim=end_frame=30",
            build_filter_graph(plan),
        )

    def test_mixed_dissolve_runs_extend_outgoing_handles_then_concat(self) -> None:
        segments = tuple(
            RenderSegment(index, index * 0.5, 0.5, index * 0.5)
            for index in range(4)
        )
        plan = RenderPlan(
            sources=tuple(Path(f"{index}.mp4") for index in range(4)),
            segments=segments,
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(TransitionKind.FACE_DISSOLVE, 11, "safe", 0.1, "A"),
                RenderTransition(tier="C", reason="ordinary_hard_cut"),
                RenderTransition(TransitionKind.FACE_DISSOLVE, 11, "safe", 0.1, "A"),
            ),
        )

        graph = build_filter_graph(plan)

        self.assertEqual(plan.output_frame_count, 60)
        self.assertIn("[v0][v1]xfade=transition=fade:duration=0.366666667:offset=0.500000000,format=yuv420p[vx0]", graph)
        self.assertIn("[v2][v3]xfade=transition=fade:duration=0.366666667:offset=0.500000000,format=yuv420p[vx1]", graph)
        self.assertIn("[vx0][vx1]concat=n=2:v=1:a=0[vcat]", graph)
        # Nominal shots retain their own quantization; only outgoing shots get
        # a separately-seeked 11-frame canonical-time handle.
        self.assertEqual(graph.count("trim=end_frame=11"), 2)
        self.assertEqual(graph.count("trim=end_frame=15"), 4)
        self.assertIn("trim=start=0.500000:duration=0.366667", graph)

    def test_transition_metadata_defaults_to_hard_cuts_and_validates_overlap(self) -> None:
        plan = RenderPlan(
            sources=(Path("one.mp4"), Path("two.mp4")),
            segments=(
                RenderSegment(0, 0.0, 0.5, 0.0),
                RenderSegment(1, 0.5, 0.5, 0.5),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )
        self.assertEqual(plan.transitions[0].kind, TransitionKind.HARD_CUT)
        self.assertEqual(plan.to_json()["transitions"][0]["kind"], "hard_cut")  # type: ignore[index]

        source_index_domains = plan.to_json()["source_index_domains"]
        self.assertEqual(
            source_index_domains,
            {
                "sources[]": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
                "reference_source": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
                "segments[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            },
        )

        with self.assertRaisesRegex(ValueError, "incoming segment"):
            RenderPlan(
                sources=(Path("one.mp4"), Path("two.mp4")),
                segments=plan.segments,
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
                fps=30.0,
                transitions=(
                    RenderTransition(TransitionKind.FACE_DISSOLVE, 16, "unsafe"),
                ),
            )

    def test_consecutive_dissolves_form_one_duration_neutral_xfade_chain(self) -> None:
        plan = RenderPlan(
            sources=(Path("0.mp4"), Path("1.mp4"), Path("2.mp4")),
            segments=(
                RenderSegment(0, 0.0, 0.5, 0.0),
                RenderSegment(1, 0.5, 0.5, 0.5),
                RenderSegment(2, 1.0, 0.5, 1.0),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(TransitionKind.FACE_DISSOLVE, 11, "safe"),
                RenderTransition(TransitionKind.FACE_DISSOLVE, 11, "safe"),
            ),
        )

        graph = build_filter_graph(plan)

        self.assertIn("[v0][v1]xfade=", graph)
        self.assertIn("[vx0][v2]xfade=", graph)
        self.assertIn("offset=1.000000000,format=yuv420p[vx1]", graph)
        self.assertIn("[vx1]null[vcat]", graph)
        self.assertEqual(plan.output_frame_count, 45)

    def test_dissolve_handle_holds_final_pan_crop_at_exact_source_boundary(self) -> None:
        track = PanCropTrack(
            (
                PanCropKeyframe(0.0, NormalizedCrop(0.0, 0.0, 0.8, 0.8)),
                PanCropKeyframe(0.5, NormalizedCrop(0.2, 0.1, 0.8, 0.8)),
            )
        )
        plan = RenderPlan(
            sources=(Path("0.mp4"), Path("1.mp4")),
            segments=(
                RenderSegment(0, 0.0, 0.5, 0.0, pan_crop=track),
                RenderSegment(1, 0.5, 0.5, 0.5),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(TransitionKind.FACE_DISSOLVE, 11, "safe"),
            ),
        )

        graph = build_filter_graph(plan)

        handle = graph.split("[vh0]", 1)[0].rsplit(";\n", 1)[-1]
        self.assertIn("trim=start=0.500000:duration=0.366667", handle)
        self.assertIn("x=iw*0.20000000:y=ih*0.10000000", handle)
        self.assertNotIn("if(lt(t", handle)

    def test_pan_crop_generates_frame_evaluated_linear_motion(self) -> None:
        track = PanCropTrack(
            (
                PanCropKeyframe(0.0, NormalizedCrop(0.0, 0.0, 0.30, 0.90)),
                PanCropKeyframe(0.5, NormalizedCrop(0.25, 0.05, 0.30, 0.90)),
                PanCropKeyframe(1.0, NormalizedCrop(0.60, 0.10, 0.30, 0.90)),
            )
        )
        plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(RenderSegment(0, 0.0, 1.0, 0.0, pan_crop=track),),
            reference_source=0,
            aspect_ratio=AspectRatio.PORTRAIT,
            output=Path("out.mp4"),
            width=1080,
            height=1920,
        )

        graph = build_filter_graph(plan)

        self.assertIn("if(lt(t,0.5000000000)", graph)
        self.assertIn("if(lt(t,1.0000000000)", graph)
        self.assertIn("x='iw*(", graph)
        self.assertIn("y='ih*(", graph)

    def test_pan_crop_rejects_zoom_breathing_and_out_of_segment_keyframes(self) -> None:
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            PanCropTrack(
                (
                    PanCropKeyframe(0.0, NormalizedCrop(0.0, 0.0, 0.3, 0.9)),
                    PanCropKeyframe(1.0, NormalizedCrop(0.0, 0.0, 0.4, 0.9)),
                )
            )
        track = PanCropTrack(
            (
                PanCropKeyframe(0.0, NormalizedCrop(0.0, 0.0, 0.3, 0.9)),
                PanCropKeyframe(2.0, NormalizedCrop(0.2, 0.0, 0.3, 0.9)),
            )
        )
        with self.assertRaisesRegex(ValueError, "inside the segment"):
            RenderSegment(0, 0.0, 1.0, 0.0, pan_crop=track)

    def test_dense_pan_crop_is_simplified_below_ffmpeg_parser_limit(self) -> None:
        keyframes = tuple(
            PanCropKeyframe(
                index / 30.0,
                NormalizedCrop(
                    0.0 if index % 2 == 0 else 0.68,
                    0.0,
                    0.30,
                    0.90,
                ),
            )
            for index in range(120)
        )
        dense = PanCropTrack(keyframes)

        simplified = dense.simplified()

        self.assertLessEqual(len(simplified.keyframes), 80)
        self.assertEqual(simplified.keyframes[0], dense.keyframes[0])
        self.assertEqual(simplified.keyframes[-1], dense.keyframes[-1])

    def test_non_contiguous_plan_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            RenderPlan(
                sources=(Path("one.mp4"),),
                segments=(RenderSegment(0, 0.0, 1.0, 0.2),),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
            )

    def test_alignment_mapping_keeps_common_start_out_of_source_offset(self) -> None:
        positive = aligned_render_segment(
            source_index=1,
            reference_start=12.5,
            common_master_start=10.0,
            source_offset=0.75,
            duration=2.0,
        )
        negative = aligned_render_segment(
            source_index=0,
            reference_start=12.5,
            common_master_start=10.0,
            source_offset=-1.25,
            duration=2.0,
        )

        self.assertEqual(positive.master_start, 2.5)
        self.assertEqual(positive.source_start, 13.25)
        self.assertEqual(negative.master_start, 2.5)
        self.assertEqual(negative.source_start, 11.25)

    def test_plan_rejects_non_integer_indices_and_source_overwrite(self) -> None:
        segment = RenderSegment(0, 0.0, 1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "source_index"):
            RenderSegment(0.5, 0.0, 1.0, 0.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "reference_source"):
            RenderPlan(
                sources=(Path("one.mp4"),),
                segments=(segment,),
                reference_source=0.5,  # type: ignore[arg-type]
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
            )
        with self.assertRaisesRegex(ValueError, "positive even integers"):
            RenderPlan(
                sources=(Path("one.mp4"),),
                segments=(segment,),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
                width=1920.0,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "overwrite"):
            RenderPlan(
                sources=(Path("one.mp4"),),
                segments=(segment,),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("one.mp4"),
            )

    def test_non_finite_values_and_invalid_output_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            NormalizedCrop(math.nan, 0.0, 0.5, 0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            RenderSegment(0, math.nan, 1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            RenderPlan(
                sources=(Path("one.mp4"),),
                segments=(RenderSegment(0, 0.0, 1.0, 0.0),),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
                fps=math.nan,
            )
        self.assertEqual(output_size(AspectRatio.LANDSCAPE, 1280), (1280, 720))


if __name__ == "__main__":
    unittest.main()
