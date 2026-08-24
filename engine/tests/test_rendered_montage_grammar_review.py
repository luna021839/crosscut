from __future__ import annotations

import unittest

from crosscut.rendered_montage_grammar_review import (
    RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS,
    review_rendered_montage_grammar,
)


def _analysis(
    shot_types: tuple[str, ...],
    sources: tuple[int, ...],
    *,
    motion_indices: tuple[int, ...] = (1,),
    role: str = "chorus_like",
    establish: bool = True,
    face_proxy: bool = True,
) -> dict[str, object]:
    if len(shot_types) != len(sources):
        raise ValueError("fixture arrays differ")
    choices = [
        {
            "segment_index": index,
            "source_index": sources[index],
            "canonical_master_start": float(index),
            "canonical_master_end": float(index + 1),
            "duration_seconds": 1.0,
            "source_local_start": float(index),
            "source_local_end": float(index + 1),
            "shot_type": shot_type,
            "entry_shot_type": shot_type,
            "exit_shot_type": shot_type,
            "shot_id": f"shot-{index}",
            "detected_person_counts": (
                [6, 6, 6]
                if shot_type in {"group_wide", "full_shot"}
                else [2, 2, 2]
                if shot_type == "medium"
                else [1, 1, 1]
            ),
            "formation_person_counts": (
                [6, 6, 6]
                if shot_type in {"group_wide", "full_shot"}
                else [0, 0, 0]
            ),
            "formation_upper_body_counts": (
                [5, 5, 5]
                if shot_type in {"group_wide", "full_shot"}
                else [0, 0, 0]
            ),
        }
        for index, shot_type in enumerate(shot_types)
    ]
    geometry = {
        "face_count": 1 if face_proxy else 0,
        "face_bbox_xywh": [0.35, 0.18, 0.30, 0.35] if face_proxy else None,
    }
    boundaries = [
        {
            "boundary_index": index,
            "cut_frame_index": (index + 1) * 30,
            "outgoing_source_index": sources[index],
            "incoming_source_index": sources[index + 1],
            "transition_kind": "hard_cut",
            "transition_frame_count": 0,
            "sample_offsets": [-2, -1, 0, 1],
            "frame_indices": [
                (index + 1) * 30 + offset for offset in (-2, -1, 0, 1)
            ],
            "frame_metrics": [
                {
                    "offset": offset,
                    "frame_index": (index + 1) * 30 + offset,
                    "subject_geometry": dict(geometry),
                }
                for offset in (-2, -1, 0, 1)
            ],
        }
        for index in range(len(choices) - 1)
    ]
    duration = float(len(choices))
    source_count = max(sources) + 1
    return {
        "stage_edit_exact_shot_clock": {
            "contract": (
                "serialized_inference_token_section_and_phrase_progress_"
                "per_selected_shot_v1"
            ),
            "source_index_domain": "eligible_source_index",
            "shots": [
                {
                    "shot_index": index,
                    "source_index": sources[index],
                    "section_role": role,
                    "section_progress": index / max(1, len(choices) - 1),
                    "phrase_progress": (index % 3) / 3.0,
                }
                for index in range(len(choices))
            ],
        },
        "source_index_mapping": {
            "request_index_domain": "request_source_index",
            "eligible_index_domain": "eligible_source_index",
            "render_plan_index_domain": "render_plan_sources_array_index",
            "domain_equivalences": {
                "render_plan_sources_array_index": "request_source_index",
            },
            "eligible_to_request": list(range(source_count)),
            "request_to_eligible": list(range(source_count)),
            "eligible_sources": [
                {
                    "eligible_source_index": index,
                    "request_source_index": index,
                    "source": f"source-{index}.mp4",
                }
                for index in range(source_count)
            ],
            "payload_index_domains": {
                "visual_timeline.choices[].source_index": (
                    "eligible_source_index"
                ),
                (
                    "rendered_proxy_review.boundaries[]"
                    ".outgoing_source_index"
                ): "request_source_index",
                (
                    "rendered_proxy_review.boundaries[]"
                    ".incoming_source_index"
                ): "request_source_index",
                "render_plan.segments[].source_index": (
                    "render_plan_sources_array_index"
                ),
            },
        },
        "source_derivative_diversity": {
            "index_domain": "eligible_source_index",
            "source_count": source_count,
            "family_ids": list(range(source_count)),
            "production_ready": True,
            "clock_contract_complete": True,
            "uncertain_pair_count": 0,
            "family_consistency_conflict_count": 0,
        },
        "section_plan": {
            "sections": [
                {
                    "section_index": 0,
                    "role": role,
                    "start_seconds": 0.0,
                    "end_seconds": duration,
                    "camera_motion_budget": 0.82,
                    "establish_with_wide_shot": establish,
                }
            ]
        },
        "visual_timeline": {
            "index_domain": "eligible_source_index",
            "choices": choices,
        },
        "editorial_motion": {
            "segments": [
                {
                    "segment_index": index,
                    "applied": index in motion_indices,
                    "keyframes": (
                        [
                            {
                                "time_seconds": 0.0,
                                "zoom": 1.0,
                                "center_x": 0.5,
                                "center_y": 0.5,
                                "roll_degrees": 0.0,
                            },
                            {
                                "time_seconds": 0.75,
                                "zoom": 1.05,
                                "center_x": 0.5,
                                "center_y": 0.5,
                                "roll_degrees": 0.0,
                            },
                        ]
                        if index in motion_indices
                        else []
                    ),
                }
                for index in range(len(choices))
            ]
        },
        "rendered_proxy_review": {
            "schema_version": 3,
            "proxy": {
                "fps": 30.0,
                "decoded_frame_count": len(choices) * 30,
            },
            "transition_count": len(boundaries),
            "reviewed_boundary_count": len(boundaries),
            "hard_cut_count": len(boundaries),
            "dissolve_count": 0,
            "boundaries": boundaries,
        },
    }


def _render_plan(fixture: dict[str, object]) -> dict[str, object]:
    choices = fixture["visual_timeline"]["choices"]
    source_count = fixture["source_derivative_diversity"]["source_count"]
    duration = float(len(choices))
    return {
        "sources": [f"source-{index}.mp4" for index in range(source_count)],
        "source_index_domains": {
            "sources[]": "render_plan_sources_array_index",
            "reference_source": "render_plan_sources_array_index",
            "segments[].source_index": "render_plan_sources_array_index",
        },
        "reference_source": 0,
        "master_audio_start": 0.0,
        "fps": 30.0,
        "segments": [
            {
                "source_index": choice["source_index"],
                "master_start": choice["canonical_master_start"],
                "duration": (
                    choice["canonical_master_end"]
                    - choice["canonical_master_start"]
                ),
            }
            for choice in choices
        ],
        "transitions": [
            {"kind": "hard_cut", "frame_count": 0}
            for _ in range(max(0, len(choices) - 1))
        ],
        "render_duration": duration,
        "output_frame_count": int(duration * 30),
    }


def _review(
    fixture: dict[str, object],
    *,
    render_plan: dict[str, object] | None = None,
) -> dict[str, object]:
    return review_rendered_montage_grammar(
        fixture,
        render_plan=_render_plan(fixture) if render_plan is None else render_plan,
    )


def _configure_dissolve(
    fixture: dict[str, object],
    plan: dict[str, object],
    *,
    boundary_index: int,
    frame_count: int = 3,
) -> dict[str, object]:
    plan["transitions"][boundary_index] = {
        "kind": "face_dissolve",
        "frame_count": frame_count,
    }
    proxy = fixture["rendered_proxy_review"]
    proxy["hard_cut_count"] -= 1
    proxy["dissolve_count"] += 1
    boundary = proxy["boundaries"][boundary_index]
    cut_frame_index = boundary["cut_frame_index"]
    offsets = (-2, -1, frame_count, frame_count + 1)
    boundary["transition_kind"] = "face_dissolve"
    boundary["transition_frame_count"] = frame_count
    boundary["sample_offsets"] = list(offsets)
    boundary["frame_indices"] = [cut_frame_index + value for value in offsets]
    boundary["frame_metrics"] = [
        {
            "offset": offset,
            "frame_index": cut_frame_index + offset,
            "subject_geometry": {
                "face_count": 1,
                "face_bbox_xywh": [0.35, 0.18, 0.30, 0.35],
            },
        }
        for offset in offsets
    ]
    return boundary


class RenderedMontageGrammarReviewTests(unittest.TestCase):
    def test_wide_anchor_then_two_short_closeup_bursts_passes(self) -> None:
        result = _review(
            _analysis(
                (
                    "group_wide",
                    "medium",
                    "close_up",
                    "full_shot",
                    "medium",
                    "close_up",
                ),
                (0, 1, 2, 3, 1, 2),
                motion_indices=(1, 4),
            )
        )

        self.assertTrue(result["passed"])
        self.assertEqual(
            RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS,
            result["analysis_basis"],
        )
        section = result["sections"][0]
        self.assertTrue(section["formation_wide_anchor_present"])
        self.assertAlmostEqual(2.0 / 6.0, section["closeup_duration_fraction"])
        self.assertAlmostEqual(4.0 / 6.0, section["nonwide_duration_fraction"])
        self.assertEqual(2, section["nonwide_burst_boundary_count"])
        self.assertEqual(
            1.0,
            result["rendered_closeup_proxy"]["face_coverage_fraction"],
        )
        self.assertEqual(
            1.0,
            result["exact_anchor_metrics"][
                "chorus_section_early_wide_anchor_fraction"
            ],
        )
        self.assertEqual(
            1.0,
            result["exact_anchor_metrics"][
                "chorus_phrase_entry_wide_anchor_fraction"
            ],
        )

    def test_exact_phrase_entry_wide_floor_rejects_nonwide_phrase_entries(self) -> None:
        fixture = _analysis(
            ("medium", "close_up", "group_wide", "medium", "close_up", "full_shot"),
            (0, 1, 2, 3, 1, 2),
            establish=False,
        )

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "highlight_phrase_entry_wide_anchor_fraction_below_reference_floor",
            result["failure_codes"],
        )

    def test_exact_shot_clock_source_mismatch_is_rejected(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        fixture["stage_edit_exact_shot_clock"]["shots"][2]["source_index"] = 0

        with self.assertRaisesRegex(ValueError, "source differs"):
            _review(fixture)

    def test_unknown_in_first_three_consumes_early_anchor_slot(self) -> None:
        fixture = _analysis(
            ("medium", "close_up", "unknown", "group_wide", "close_up", "full_shot"),
            (0, 1, 2, 3, 1, 2),
            establish=False,
        )

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertEqual(
            0.0,
            result["exact_anchor_metrics"][
                "chorus_section_early_wide_anchor_fraction"
            ],
        )
        self.assertIn(
            "highlight_early_wide_anchor_fraction_below_reference_floor",
            result["failure_codes"],
        )

    def test_flat_all_wide_highlight_fails_even_when_timing_is_contiguous(self) -> None:
        result = _review(
            _analysis(
                ("group_wide",) * 6,
                (0, 1, 2, 0, 1, 2),
                motion_indices=(2,),
            )
        )

        self.assertFalse(result["passed"])
        failures = set(result["failure_codes"])
        self.assertIn("section_0:highlight_closeup_fraction_below_floor", failures)
        self.assertIn("section_0:highlight_nonwide_fraction_below_floor", failures)
        self.assertIn("section_0:highlight_nonwide_burst_missing", failures)

    def test_closeup_plan_claim_without_decoded_face_geometry_fails(self) -> None:
        result = _review(
            _analysis(
                (
                    "group_wide",
                    "medium",
                    "close_up",
                    "full_shot",
                    "medium",
                    "close_up",
                ),
                (0, 1, 2, 3, 1, 2),
                face_proxy=False,
            )
        )

        self.assertFalse(result["passed"])
        self.assertIn(
            "rendered_closeup_face_coverage_below_floor",
            result["failure_codes"],
        )

    def test_same_native_shot_slices_are_collapsed_before_fraction_review(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "group_wide",
                "medium",
                "close_up",
                "group_wide",
                "medium",
            ),
            (0, 0, 1, 2, 3, 1),
            role="verse_like",
        )
        choices = fixture["visual_timeline"]["choices"]
        choices[1]["shot_id"] = choices[0]["shot_id"]
        result = _review(fixture)

        self.assertTrue(result["passed"])
        self.assertEqual(5, result["visible_run_count"])
        self.assertEqual(2.0, result["visible_runs"][0]["duration_seconds"])

    def test_proxy_boundary_count_mismatch_is_rejected(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        fixture["rendered_proxy_review"]["boundaries"].pop()
        with self.assertRaisesRegex(ValueError, "boundary count"):
            _review(fixture)

    def test_generic_wide_without_group_evidence_is_not_a_formation_anchor(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        for choice in fixture["visual_timeline"]["choices"]:
            if choice["shot_type"] in {"group_wide", "full_shot"}:
                choice["detected_person_counts"] = [0, 0, 0]
                choice["formation_person_counts"] = [0, 0, 0]
                choice["formation_upper_body_counts"] = [0, 0, 0]

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "section_0:formation_wide_anchor_evidence_missing",
            result["failure_codes"],
        )

    def test_strong_multi_person_wide_survives_tiny_pose_failure(self) -> None:
        fixture = _analysis(
            (
                "group_wide",
                "medium",
                "close_up",
                "full_shot",
                "medium",
                "close_up",
            ),
            (0, 1, 2, 3, 1, 2),
        )
        for choice in fixture["visual_timeline"]["choices"]:
            if choice["shot_type"] in {"group_wide", "full_shot"}:
                choice["detected_person_counts"] = [4, 4, 4]
                choice["formation_person_counts"] = [4, 4, 4]
                choice["formation_upper_body_counts"] = [0, 0, 0]

        result = _review(fixture)

        self.assertTrue(result["passed"], result["failure_codes"])
        self.assertTrue(
            result["sections"][0]["formation_wide_anchor_present"]
        )

    def test_declared_section_must_be_fully_covered(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        fixture["section_plan"]["sections"][0]["end_seconds"] = 7.0
        with self.assertRaisesRegex(ValueError, "fully cover|end does not match"):
            _review(fixture)

    def test_central_close_claim_cannot_escape_proxy_face_review(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
            face_proxy=False,
        )
        for choice in fixture["visual_timeline"]["choices"]:
            if choice["shot_type"] == "close_up":
                choice["entry_shot_type"] = "medium"
                choice["exit_shot_type"] = "medium"

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertGreater(
            result["rendered_closeup_proxy"]["required_closeup_endpoint_count"],
            0,
        )
        self.assertIn(
            "rendered_closeup_face_coverage_below_floor",
            result["failure_codes"],
        )

    def test_nonwide_burst_requires_later_formation_reset(self) -> None:
        result = _review(
            _analysis(
                ("group_wide", "medium", "close_up", "medium", "close_up", "medium"),
                (0, 1, 2, 3, 1, 2),
            )
        )

        self.assertFalse(result["passed"])
        self.assertIn(
            "section_0:post_burst_formation_wide_reset_missing",
            result["failure_codes"],
        )

    def test_identity_transform_does_not_count_as_motion_accent(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        for segment in fixture["editorial_motion"]["segments"]:
            segment["applied"] = True
            segment["keyframes"] = [
                {
                    "time_seconds": 0.0,
                    "zoom": 1.0,
                    "center_x": 0.5,
                    "center_y": 0.5,
                    "roll_degrees": 0.0,
                }
            ]

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "section_0:highlight_motion_accent_missing",
            result["failure_codes"],
        )

    def test_same_derivative_camera_family_does_not_create_a_multicam_burst(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        fixture["source_derivative_diversity"]["family_ids"] = [0, 0, 0, 0]

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertIn(
            "section_0:highlight_nonwide_burst_missing",
            result["failure_codes"],
        )

    def test_render_plan_is_bound_to_selected_visual_timeline(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        passing = _review(fixture, render_plan=plan)
        self.assertTrue(passing["render_plan_binding"]["passed"])

        plan["segments"][2]["source_index"] = 0
        with self.assertRaisesRegex(ValueError, "source differs"):
            _review(fixture, render_plan=plan)

    def test_render_plan_is_required_for_strict_proxy_identity(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )

        with self.assertRaisesRegex(ValueError, "render_plan is required"):
            review_rendered_montage_grammar(fixture)

    def test_missing_explicit_source_mapping_fails_closed(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        fixture.pop("source_index_mapping")

        with self.assertRaisesRegex(ValueError, "source_index_mapping"):
            _review(fixture, render_plan=plan)

    def test_eligible_request_gap_is_applied_to_plan_and_proxy_sources(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        mapping = fixture["source_index_mapping"]
        mapping["eligible_to_request"] = [0, 2, 3, 4]
        mapping["request_to_eligible"] = [0, None, 1, 2, 3]
        plan["sources"] = [f"source-{index}.mp4" for index in range(5)]
        translated = [0, 2, 3, 4, 2, 3]
        mapping["eligible_sources"] = [
            {
                "eligible_source_index": eligible_index,
                "request_source_index": request_index,
                "source": plan["sources"][request_index],
            }
            for eligible_index, request_index in enumerate(
                mapping["eligible_to_request"]
            )
        ]
        for segment, source_index in zip(
            plan["segments"], translated, strict=True
        ):
            segment["source_index"] = source_index
        boundaries = fixture["rendered_proxy_review"]["boundaries"]
        for index, boundary in enumerate(boundaries):
            boundary["outgoing_source_index"] = translated[index]
            boundary["incoming_source_index"] = translated[index + 1]

        result = _review(fixture, render_plan=plan)

        self.assertTrue(result["passed"])
        self.assertEqual(
            translated,
            result["render_plan_binding"]["source_index_binding"][
                "translated_choice_source_indices"
            ],
        )

    def test_dissolve_closeup_uses_completed_transition_endpoint(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        boundary = _configure_dissolve(fixture, plan, boundary_index=1)
        for metric in boundary["frame_metrics"]:
            if metric["offset"] != 3:
                metric["subject_geometry"]["face_bbox_xywh"] = None

        result = _review(fixture, render_plan=plan)

        self.assertTrue(result["passed"])
        close_segment = result["rendered_closeup_proxy"]["records"][0]
        incoming = close_segment["endpoints"][0]
        self.assertEqual("incoming", incoming["side"])
        self.assertEqual("face_dissolve", incoming["transition_kind"])
        self.assertEqual(3, incoming["offset"])
        self.assertEqual(boundary["cut_frame_index"] + 3, incoming["frame_index"])
        self.assertTrue(incoming["face_geometry_present"])

    def test_dissolve_forged_offset_zero_metric_is_rejected(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        boundary = _configure_dissolve(fixture, plan, boundary_index=1)
        boundary["frame_metrics"][2]["offset"] = 0
        boundary["frame_metrics"][2]["frame_index"] = boundary["cut_frame_index"]

        with self.assertRaisesRegex(ValueError, "exact sample"):
            _review(fixture, render_plan=plan)

    def test_dissolve_wrong_declared_sample_offsets_are_rejected(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        boundary = _configure_dissolve(fixture, plan, boundary_index=1)
        boundary["sample_offsets"] = [-2, -1, 2, 3]

        with self.assertRaisesRegex(ValueError, "sample offsets"):
            _review(fixture, render_plan=plan)

    def test_proxy_boundary_source_pair_is_bound_to_render_plan(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        fixture["rendered_proxy_review"]["boundaries"][1][
            "incoming_source_index"
        ] = 0

        with self.assertRaisesRegex(ValueError, "incoming_source_index"):
            _review(fixture, render_plan=plan)

    def test_proxy_boundary_transition_kind_is_bound_to_render_plan(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        plan = _render_plan(fixture)
        fixture["rendered_proxy_review"]["boundaries"][1][
            "transition_kind"
        ] = "face_dissolve"

        with self.assertRaisesRegex(ValueError, "transition kind"):
            _review(fixture, render_plan=plan)

    def test_proxy_boundary_frame_window_is_exactly_bound(self) -> None:
        mutations = (
            (
                "boundary_index",
                lambda boundary: boundary.__setitem__("boundary_index", 4),
                "indices",
            ),
            (
                "cut_frame_index",
                lambda boundary: boundary.__setitem__("cut_frame_index", 61),
                "cut_frame_index",
            ),
            (
                "transition_frame_count",
                lambda boundary: boundary.__setitem__(
                    "transition_frame_count", 1
                ),
                "transition_frame_count",
            ),
            (
                "frame_indices",
                lambda boundary: boundary["frame_indices"].__setitem__(2, 61),
                "frame indices",
            ),
        )
        for label, mutate, error in mutations:
            with self.subTest(field=label):
                fixture = _analysis(
                    (
                        "group_wide",
                        "medium",
                        "close_up",
                        "full_shot",
                        "medium",
                        "close_up",
                    ),
                    (0, 1, 2, 3, 1, 2),
                )
                plan = _render_plan(fixture)
                mutate(fixture["rendered_proxy_review"]["boundaries"][1])

                with self.assertRaisesRegex(ValueError, error):
                    _review(fixture, render_plan=plan)

    def test_tiny_face_box_does_not_prove_a_closeup(self) -> None:
        fixture = _analysis(
            ("group_wide", "medium", "close_up", "full_shot", "medium", "close_up"),
            (0, 1, 2, 3, 1, 2),
        )
        for boundary in fixture["rendered_proxy_review"]["boundaries"]:
            for metric in boundary["frame_metrics"]:
                metric["subject_geometry"]["face_bbox_xywh"] = [0.4, 0.4, 0.05, 0.05]

        result = _review(fixture)

        self.assertFalse(result["passed"])
        self.assertEqual(
            0.0,
            result["rendered_closeup_proxy"]["face_coverage_fraction"],
        )


if __name__ == "__main__":
    unittest.main()
