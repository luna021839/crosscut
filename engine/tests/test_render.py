from __future__ import annotations

from pathlib import Path
import math
import unittest

from crosscut.ive import AspectRatio
from crosscut.render import (
    BoundaryLookKeyframe,
    BoundaryLookTrack,
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RENDER_PLAN_SOURCE_INDEX_DOMAIN,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    SourceLookAdjustment,
    TransitionKind,
    aligned_render_segment,
    build_filter_graph,
    output_size,
    minimum_roll_safe_zoom,
    segment_frame_counts,
)


class RenderTests(unittest.TestCase):
    def test_portrait_output_size(self) -> None:
        self.assertEqual(output_size(AspectRatio.PORTRAIT), (1080, 1920))

    def test_render_rate_arguments_must_be_nonempty(self) -> None:
        plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(RenderSegment(0, 0.0, 1.0, 0.0),),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
        )
        from crosscut.render import render

        with self.assertRaisesRegex(ValueError, "video_bitrate"):
            render(plan, video_bitrate="")

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

    def test_bounded_source_look_filters_run_after_geometry_without_retiming(self) -> None:
        adjustment = SourceLookAdjustment(
            brightness=0.02,
            contrast=1.04,
            saturation=0.96,
            temperature_balance=0.015,
            tint_balance=-0.012,
            detail_enhancement=0.35,
        )
        plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(
                RenderSegment(
                    0,
                    0.0,
                    1.0,
                    0.0,
                    crop=NormalizedCrop(0.1, 0.1, 0.8, 0.8),
                    look_adjustment=adjustment,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )

        graph = build_filter_graph(plan)

        eq = "eq=brightness=0.020000:contrast=1.040000:saturation=0.960000"
        chroma = (
            "geq=lum='lum(X,Y)':"
            "cb='clip(cb(X,Y)-0.522000,0,255)':"
            "cr='clip(cr(X,Y)+2.070000,0,255)'"
        )
        self.assertIn(eq, graph)
        self.assertIn(chroma, graph)
        self.assertIn("cas=strength=0.350000", graph)
        self.assertNotIn("colorbalance=", graph)
        self.assertLess(graph.index("crop="), graph.index(eq))
        self.assertLess(graph.index("crop="), graph.index(chroma))
        self.assertEqual(plan.output_frame_count, 30)
        payload = plan.to_json()["segments"][0]  # type: ignore[index]
        self.assertEqual(
            payload["look_adjustment"],
            {
                "brightness": 0.02,
                "contrast": 1.04,
                "saturation": 0.96,
                "temperature_balance": 0.015,
                "tint_balance": -0.012,
                "detail_enhancement": 0.35,
            },
        )

    def test_identity_source_look_is_filter_free_and_contract_is_tightly_bounded(self) -> None:
        plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(
                RenderSegment(
                    0,
                    0.0,
                    1.0,
                    0.0,
                    look_adjustment=SourceLookAdjustment(),
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
        )
        graph = build_filter_graph(plan)
        self.assertNotIn("eq=brightness", graph)
        self.assertNotIn("colorbalance=", graph)
        self.assertNotIn("geq=lum='lum(X,Y)'", graph)
        with self.assertRaisesRegex(ValueError, "brightness"):
            SourceLookAdjustment(brightness=0.181)
        with self.assertRaisesRegex(ValueError, "contrast"):
            SourceLookAdjustment(contrast=1.401)
        with self.assertRaisesRegex(ValueError, "saturation"):
            SourceLookAdjustment(saturation=0.799)
        with self.assertRaisesRegex(ValueError, "temperature"):
            SourceLookAdjustment(temperature_balance=0.181)
        with self.assertRaisesRegex(ValueError, "tint"):
            SourceLookAdjustment(tint_balance=-0.101)
        with self.assertRaisesRegex(ValueError, "detail enhancement"):
            SourceLookAdjustment(detail_enhancement=1.001)
        with self.assertRaisesRegex(TypeError, "look_adjustment"):
            RenderSegment(
                0,
                0.0,
                1.0,
                0.0,
                look_adjustment={"brightness": 0.01},  # type: ignore[arg-type]
            )

    def test_boundary_look_is_a_separate_post_global_smooth_eq_layer(self) -> None:
        global_look = SourceLookAdjustment(brightness=0.01, saturation=0.98)
        local_track = BoundaryLookTrack(
            (
                BoundaryLookKeyframe(
                    0.0,
                    brightness=-0.025,
                    saturation=1.04,
                    temperature_balance=0.01,
                ),
                BoundaryLookKeyframe(0.2),
                BoundaryLookKeyframe(0.9),
            )
        )
        plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(
                RenderSegment(
                    0,
                    0.0,
                    1.0,
                    0.0,
                    crop=NormalizedCrop(0.1, 0.1, 0.8, 0.8),
                    look_adjustment=global_look,
                    boundary_look_track=local_track,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )

        graph = build_filter_graph(plan)

        global_filter = (
            "eq=brightness=0.010000:contrast=1.000000:saturation=0.980000"
        )
        self.assertIn(global_filter, graph)
        self.assertIn(":eval=frame", graph)
        self.assertIn("if(lt(t,0.2000000000)", graph)
        self.assertIn("geq=lum='lum(X,Y)'", graph)
        self.assertIn("cb='clip(cb(X,Y)-128*", graph)
        self.assertIn("cr='clip(cr(X,Y)+128*", graph)
        self.assertIn("if(lt(T,0.2000000000)", graph)
        self.assertLess(graph.index(global_filter), graph.index(":eval=frame"))
        self.assertLess(graph.index(":eval=frame"), graph.index("geq="))
        self.assertLess(graph.index("crop="), graph.index(global_filter))
        payload = plan.to_json()["segments"][0]  # type: ignore[index]
        self.assertEqual(
            payload["boundary_look_track"]["reason"],
            "boundary_local_post_global_look_residual",
        )
        self.assertEqual(
            payload["boundary_look_track"]["keyframes"][0][
                "temperature_balance"
            ],
            0.01,
        )

    def test_boundary_look_contract_rejects_unsafe_or_misplaced_tracks(self) -> None:
        with self.assertRaisesRegex(ValueError, "brightness"):
            BoundaryLookKeyframe(0.0, brightness=0.051)
        with self.assertRaisesRegex(ValueError, "start"):
            BoundaryLookTrack((BoundaryLookKeyframe(0.1, brightness=0.01),))
        with self.assertRaisesRegex(ValueError, "inside the segment"):
            RenderSegment(
                0,
                0.0,
                0.5,
                0.0,
                boundary_look_track=BoundaryLookTrack(
                    (
                        BoundaryLookKeyframe(0.0),
                        BoundaryLookKeyframe(0.6, brightness=0.01),
                    )
                ),
            )

    def test_dissolve_handle_reuses_the_same_static_source_look(self) -> None:
        plan = RenderPlan(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            segments=(
                RenderSegment(
                    0,
                    0.0,
                    0.5,
                    0.0,
                    look_adjustment=SourceLookAdjustment(brightness=0.01),
                ),
                RenderSegment(1, 0.5, 0.5, 0.5),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(TransitionKind.FACE_DISSOLVE, 3, "safe"),
            ),
        )

        graph = build_filter_graph(plan)

        self.assertEqual(
            graph.count("eq=brightness=0.010000:contrast=1.000000:saturation=1.000000"),
            2,
        )
        self.assertEqual(plan.output_frame_count, 30)

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

    def test_identity_same_source_tier_d_and_native_boundaries_share_one_clock(self) -> None:
        plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(
                RenderSegment(0, 0.0, 0.4, 0.0),
                RenderSegment(0, 0.4, 0.3, 0.4),
                RenderSegment(0, 0.7, 0.5, 0.7),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(tier="D", reason="same_source_hold_no_visible_switch"),
                RenderTransition(tier="C", reason="same_source_internal_camera_cut"),
            ),
        )

        graph = build_filter_graph(plan)

        self.assertEqual(graph.count("[0:v]trim="), 1)
        self.assertIn("trim=start=0.000000:duration=1.200000", graph)
        self.assertIn("trim=end_frame=36", graph)
        self.assertIn("[v0]null[vcat]", graph)
        # Coalescing is a renderer implementation detail.  The authored
        # boundaries remain available to QA and provenance.
        self.assertEqual(len(plan.to_json()["transitions"]), 2)  # type: ignore[arg-type]

    def test_same_source_coalescing_fails_closed_for_clock_or_geometry_change(self) -> None:
        clock_gap = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(
                RenderSegment(0, 0.0, 0.5, 0.0),
                RenderSegment(0, 0.501, 0.5, 0.5),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("clock-gap.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(tier="C", reason="same_source_internal_camera_cut"),
            ),
        )
        geometry_change = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(
                RenderSegment(0, 0.0, 0.5, 0.0),
                RenderSegment(
                    0,
                    0.5,
                    0.5,
                    0.5,
                    crop=NormalizedCrop(0.1, 0.0, 0.8, 1.0),
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("geometry-change.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(tier="C", reason="same_source_internal_camera_cut"),
            ),
        )

        self.assertEqual(build_filter_graph(clock_gap).count("[0:v]trim="), 2)
        self.assertEqual(build_filter_graph(geometry_change).count("[0:v]trim="), 2)

    def test_cross_cut_look_ramps_merge_with_adjacent_same_source_clocks(self) -> None:
        last_frame = 5.0 / 30.0
        outgoing = BoundaryLookTrack(
            (
                BoundaryLookKeyframe(0.0),
                BoundaryLookKeyframe(last_frame, brightness=0.02),
            )
        )
        incoming = BoundaryLookTrack(
            (
                BoundaryLookKeyframe(0.0, brightness=-0.02),
                BoundaryLookKeyframe(last_frame),
            )
        )
        plan = RenderPlan(
            sources=(Path("a.mp4"), Path("b.mp4")),
            segments=(
                RenderSegment(0, 0.0, 0.2, 0.0),
                RenderSegment(0, 0.2, 0.2, 0.2),
                RenderSegment(
                    0,
                    0.4,
                    0.2,
                    0.4,
                    boundary_look_track=outgoing,
                ),
                RenderSegment(
                    1,
                    0.6,
                    0.2,
                    0.6,
                    boundary_look_track=incoming,
                ),
                RenderSegment(1, 0.8, 0.2, 0.8),
                RenderSegment(1, 1.0, 0.2, 1.0),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("look-ramp.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(tier="D", reason="same_source_hold_no_visible_switch"),
                RenderTransition(tier="C", reason="same_source_internal_camera_cut"),
                RenderTransition(tier="C", reason="introduced_cross_source"),
                RenderTransition(tier="D", reason="same_source_hold_no_visible_switch"),
                RenderTransition(tier="C", reason="same_source_internal_camera_cut"),
            ),
        )

        graph = build_filter_graph(plan)

        self.assertEqual(graph.count("[0:v]trim="), 1)
        self.assertEqual(graph.count("[1:v]trim="), 1)
        self.assertIn("if(lt(t,0.4000000000)", graph)
        self.assertIn("if(lt(t,0.5666666667)", graph)
        self.assertIn("if(lt(t,0.1666666667)", graph)
        self.assertEqual(len(plan.transitions), 5)

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
        self.assertIn("[v0][v1]xfade=transition=custom:", graph)
        self.assertIn("[v2][v3]xfade=transition=custom:", graph)
        self.assertEqual(graph.count("exp(-12*(P-0.5))"), 4)
        self.assertIn("duration=0.366666667:offset=0.500000000,format=yuv420p[vx0]", graph)
        self.assertIn("duration=0.366666667:offset=0.500000000,format=yuv420p[vx1]", graph)
        self.assertEqual(
            plan.to_json()["transitions"][0]["alpha_curve"],  # type: ignore[index]
            "normalized_logistic_sigmoid_k12_v1",
        )
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
                "source_usage.used_source_indices[]": (
                    RENDER_PLAN_SOURCE_INDEX_DOMAIN
                ),
                "source_usage.sources[].source_index": (
                    RENDER_PLAN_SOURCE_INDEX_DOMAIN
                ),
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

    def test_plan_serializes_exact_local_endpoints_and_actual_source_usage(self) -> None:
        plan = RenderPlan(
            sources=(Path("zero.mp4"), Path("one.mp4"), Path("two.mp4")),
            segments=(
                RenderSegment(
                    source_index=0,
                    source_start=1.0,
                    duration=2.0,
                    master_start=0.0,
                    source_duration=2.008,
                ),
                RenderSegment(
                    source_index=2,
                    source_start=4.0,
                    duration=3.0,
                    master_start=2.0,
                    source_duration=2.988,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
        )

        payload = plan.to_json()
        self.assertAlmostEqual(payload["segments"][0]["source_local_start"], 1.0)  # type: ignore[index]
        self.assertAlmostEqual(payload["segments"][0]["source_local_end"], 3.008)  # type: ignore[index]
        self.assertAlmostEqual(payload["segments"][1]["source_local_end"], 6.988)  # type: ignore[index]
        usage = payload["source_usage"]  # type: ignore[assignment]
        self.assertEqual(usage["distinct_source_count"], 2)  # type: ignore[index]
        self.assertEqual(usage["used_source_indices"], [0, 2])  # type: ignore[index]
        self.assertEqual(usage["total_segment_count"], 2)  # type: ignore[index]
        self.assertAlmostEqual(usage["total_duration_seconds"], 5.0)  # type: ignore[index]
        self.assertEqual(
            [item["segment_count"] for item in usage["sources"]],  # type: ignore[index]
            [1, 0, 1],
        )
        self.assertEqual(
            [item["duration_seconds"] for item in usage["sources"]],  # type: ignore[index]
            [2.0, 0.0, 3.0],
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

    def test_group_transform_emits_dynamic_zoom_pan_and_corner_safe_roll(self) -> None:
        roll_zoom = minimum_roll_safe_zoom(1.0)
        track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, roll_zoom, 0.50, 0.50, -1.0),
                EditorialTransformKeyframe(0.5, 1.10, 0.50, 0.50, 0.0),
                EditorialTransformKeyframe(1.0, 1.16, 0.52, 0.48, 1.0),
            ),
            reason="matched_subject_scale_with_phrase_push",
        )
        plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0, editorial_transform=track),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
        )

        graph = build_filter_graph(plan)

        self.assertIn(
            "scale=2228:1254:force_original_aspect_ratio=increase,crop=2228:1254",
            graph,
        )
        self.assertIn("fps=30.000000,rotate=angle=", graph)
        self.assertIn("zoompan=z=", graph)
        self.assertIn("on/30.0000000000", graph)
        self.assertIn("d=1:s=1920x1080:fps=30.000000", graph)
        payload = plan.to_json()["segments"][0]  # type: ignore[index]
        self.assertEqual(
            payload["editorial_transform"]["reason"],  # type: ignore[index]
            "matched_subject_scale_with_phrase_push",
        )

    def test_group_transform_endpoint_tolerance_never_extrapolates_bounds(self) -> None:
        fps = 29.97
        exact_output_frame_time = 22.0 / fps
        track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.065),
                EditorialTransformKeyframe(
                    exact_output_frame_time - 1.5e-15,
                    1.0,
                ),
            )
        )

        endpoint = track.evaluate(exact_output_frame_time)

        self.assertEqual(endpoint.time_seconds, exact_output_frame_time)
        self.assertEqual(endpoint.zoom, 1.0)
        self.assertEqual(endpoint.center_x, 0.5)
        self.assertEqual(endpoint.center_y, 0.5)
        self.assertEqual(endpoint.roll_degrees, 0.0)
        plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(
                RenderSegment(
                    0,
                    0.0,
                    23.0 / fps,
                    0.0,
                    editorial_transform=track,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=fps,
        )
        self.assertIs(plan.segments[0].editorial_transform, track)

    def test_group_transform_rejects_black_corner_roll_and_mixed_crop_modes(self) -> None:
        widened_pan = EditorialTransformKeyframe(0.0, 2.0, 0.75, 0.50)
        self.assertEqual(widened_pan.center_x, 0.75)
        with self.assertRaisesRegex(ValueError, "quarter-frame"):
            EditorialTransformKeyframe(0.0, 2.0, 0.751, 0.50)
        with self.assertRaisesRegex(ValueError, "zoom"):
            EditorialTransformKeyframe(0.0, 2.001)
        with self.assertRaisesRegex(ValueError, "frame edges"):
            EditorialTransformKeyframe(0.0, 1.0, roll_degrees=1.0)
        track = EditorialTransformTrack((EditorialTransformKeyframe(0.0, 1.05),))
        with self.assertRaisesRegex(ValueError, "both static and panning"):
            RenderSegment(
                0,
                0.0,
                1.0,
                0.0,
                crop=NormalizedCrop(0.0, 0.0, 0.9, 0.9),
                pan_crop=PanCropTrack(
                    (PanCropKeyframe(0.0, NormalizedCrop(0.0, 0.0, 0.9, 0.9)),)
                ),
                editorial_transform=track,
            )
        with self.assertRaisesRegex(ValueError, "inside the segment"):
            RenderSegment(
                0,
                0.0,
                0.5,
                0.0,
                editorial_transform=EditorialTransformTrack(
                    (
                        EditorialTransformKeyframe(0.0, 1.05),
                        EditorialTransformKeyframe(1.0, 1.10),
                    )
                ),
            )

    def test_unreviewed_editorial_transform_is_forbidden_next_to_dissolve(self) -> None:
        track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.0),
                EditorialTransformKeyframe(0.5, 1.06),
            )
        )
        with self.assertRaisesRegex(ValueError, "unreviewed editorial"):
            RenderPlan(
                sources=(Path("zero.mp4"), Path("one.mp4")),
                segments=(
                    RenderSegment(0, 0.0, 0.5, 0.0, editorial_transform=track),
                    RenderSegment(1, 0.5, 0.5, 0.5),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
                fps=30.0,
                transitions=(
                    RenderTransition(
                        TransitionKind.FACE_DISSOLVE,
                        11,
                        "safe_without_transform",
                        tier="A",
                    ),
                ),
            )

    def test_frame_neutral_transform_is_allowed_outside_dissolve_overlap(
        self,
    ) -> None:
        outgoing = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.06),
                EditorialTransformKeyframe(0.5, 1.0),
                EditorialTransformKeyframe(2.0, 1.0),
            ),
            reason="settled_before_overlap",
        )
        incoming = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.0),
                EditorialTransformKeyframe(0.5, 1.0),
                EditorialTransformKeyframe(2.0, 1.06),
            ),
            reason="starts_after_overlap",
        )

        plan = RenderPlan(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            segments=(
                RenderSegment(
                    0,
                    0.0,
                    2.0,
                    0.0,
                    editorial_transform=outgoing,
                ),
                RenderSegment(
                    1,
                    2.0,
                    2.0,
                    2.0,
                    editorial_transform=incoming,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(
                    TransitionKind.FACE_DISSOLVE,
                    11,
                    "frame_neutral_overlap",
                    tier="A",
                ),
            ),
        )

        self.assertIs(plan.segments[0].editorial_transform, outgoing)
        self.assertIs(plan.segments[1].editorial_transform, incoming)

    def test_incoming_transform_inside_dissolve_overlap_is_forbidden(self) -> None:
        incoming = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.0),
                EditorialTransformKeyframe(0.2, 1.06),
                EditorialTransformKeyframe(1.0, 1.06),
            ),
            reason="starts_inside_overlap",
        )

        with self.assertRaisesRegex(ValueError, "unreviewed editorial"):
            RenderPlan(
                sources=(Path("zero.mp4"), Path("one.mp4")),
                segments=(
                    RenderSegment(0, 0.0, 1.0, 0.0),
                    RenderSegment(
                        1,
                        1.0,
                        1.0,
                        1.0,
                        editorial_transform=incoming,
                    ),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
                fps=30.0,
                transitions=(
                    RenderTransition(
                        TransitionKind.FACE_DISSOLVE,
                        11,
                        "unsafe_incoming_transform",
                        tier="A",
                    ),
                ),
            )

    def test_output_verified_member_dissolve_authorizes_transformed_overlap(
        self,
    ) -> None:
        incoming = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.02),
                EditorialTransformKeyframe(0.2, 1.06),
                EditorialTransformKeyframe(1.0, 1.06),
            ),
            reason="decoded_output_face_match",
        )
        plan = RenderPlan(
            sources=(Path("zero.mp4"), Path("one.mp4")),
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0),
                RenderSegment(
                    1,
                    1.0,
                    1.0,
                    1.0,
                    editorial_transform=incoming,
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            fps=30.0,
            transitions=(
                RenderTransition(
                    TransitionKind.FACE_DISSOLVE,
                    7,
                    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
                    tier="A",
                    transform_overlap_authority=(
                        MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY
                    ),
                ),
            ),
            member_pose_dissolve_requested=True,
            member_pose_dissolve_effective=True,
            member_pose_dissolve_count=1,
        )

        transition_payload = plan.to_json()["transitions"][0]  # type: ignore[index]
        self.assertEqual(
            transition_payload["transform_overlap_authority"],
            MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
        )
        self.assertIn("xfade=transition=custom", build_filter_graph(plan))

    def test_tier_d_same_source_hold_rejects_editorial_camera_reset(self) -> None:
        transformed = EditorialTransformTrack(
            (EditorialTransformKeyframe(0.0, 1.10),),
            reason="constant_reframe",
        )
        with self.assertRaisesRegex(ValueError, "same-source hold"):
            RenderPlan(
                sources=(Path("one.mp4"),),
                segments=(
                    RenderSegment(
                        0,
                        0.0,
                        1.0,
                        0.0,
                        editorial_transform=transformed,
                    ),
                    RenderSegment(0, 1.0, 1.0, 1.0),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
                fps=30.0,
                transitions=(RenderTransition(tier="D", reason="same shot"),),
            )

        with self.assertRaisesRegex(ValueError, "same source"):
            RenderPlan(
                sources=(Path("one.mp4"), Path("two.mp4")),
                segments=(
                    RenderSegment(0, 0.0, 1.0, 0.0),
                    RenderSegment(1, 1.0, 1.0, 1.0),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
                fps=30.0,
                transitions=(RenderTransition(tier="D", reason="invalid hold"),),
            )
        with self.assertRaisesRegex(ValueError, "one of A"):
            RenderTransition(tier="X", reason="invalid tier")
        with self.assertRaisesRegex(ValueError, "contiguous source clock"):
            RenderPlan(
                sources=(Path("one.mp4"),),
                segments=(
                    RenderSegment(0, 0.0, 1.0, 0.0),
                    RenderSegment(0, 10.0, 1.0, 1.0),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
                fps=30.0,
                transitions=(RenderTransition(tier="D", reason="clock jump"),),
            )
        with self.assertRaisesRegex(ValueError, "base crop"):
            RenderPlan(
                sources=(Path("one.mp4"),),
                segments=(
                    RenderSegment(
                        0,
                        0.0,
                        1.0,
                        0.0,
                        crop=NormalizedCrop(0.0, 0.0, 0.8, 0.8),
                    ),
                    RenderSegment(
                        0,
                        1.0,
                        1.0,
                        1.0,
                        crop=NormalizedCrop(0.2, 0.2, 0.8, 0.8),
                    ),
                ),
                reference_source=0,
                aspect_ratio=AspectRatio.LANDSCAPE,
                output=Path("out.mp4"),
                fps=30.0,
                transitions=(RenderTransition(tier="D", reason="crop reset"),),
            )

    def test_tier_d_accepts_exact_crop_endpoint_across_fractional_frame_cut(
        self,
    ) -> None:
        duration = 0.6122383722162043
        width = 0.9666666666666668
        height = 0.9666666666666668
        top = 0.006
        shared_x = 0.016666666666666607
        plan = RenderPlan(
            sources=(Path("one.mp4"),),
            segments=(
                RenderSegment(
                    0,
                    0.0,
                    duration,
                    0.0,
                    pan_crop=PanCropTrack(
                        (
                            PanCropKeyframe(
                                0.0,
                                NormalizedCrop(0.0, top, width, height),
                            ),
                            PanCropKeyframe(
                                duration,
                                NormalizedCrop(shared_x, top, width, height),
                            ),
                        )
                    ),
                ),
                RenderSegment(
                    0,
                    duration,
                    duration,
                    duration,
                    pan_crop=PanCropTrack(
                        (
                            PanCropKeyframe(
                                0.0,
                                NormalizedCrop(shared_x, top, width, height),
                            ),
                            PanCropKeyframe(
                                duration,
                                NormalizedCrop(
                                    1.0 - width,
                                    top,
                                    width,
                                    height,
                                ),
                            ),
                        )
                    ),
                ),
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("out.mp4"),
            transitions=(RenderTransition(tier="D", reason="same shot"),),
        )

        self.assertEqual(segment_frame_counts(plan), (18, 19))

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
