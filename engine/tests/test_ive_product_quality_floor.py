from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import unittest

from crosscut.ive_product_quality_floor import (
    APP_RESULT_EXPOSURE_CONTRACT,
    LEGACY_I_AM_ANALYSIS_SHA256,
    PRE_REVIEW_REPAIR_LINEAGE_CONTRACT,
    ROLE_CONDITIONED_HIGHLIGHT_AGGREGATE_CONTRACT,
    evaluate_ive_product_quality_floor,
)
from crosscut.rendered_montage_grammar_review import (
    RENDERED_SEGMENT_INTERIOR_GEOMETRY_REVIEW_CONTRACT,
)
from crosscut.rendered_proxy_review import (
    SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS,
    SEGMENT_INTERIOR_GEOMETRY_CONTRACT,
)


_I_AM = {
    "close_duration_fraction": 0.32152109124843137,
    "nonwide_duration_fraction": 0.5143116148397875,
    "maximum_consecutive_wide_run_seconds": 7.331084476681127,
}
_BANG = {
    "close_duration_fraction": 0.14735084064807696,
    "nonwide_duration_fraction": 0.24510078543285077,
    "maximum_consecutive_wide_run_seconds": 15.822888179098527,
}


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rendered_geometry_evidence(
    metrics: dict[str, float],
    *,
    passed: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    source_core: dict[str, object] = {
        "schema_version": 1,
        "analysis_basis": SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS,
        "contract": SEGMENT_INTERIOR_GEOMETRY_CONTRACT,
        "sampling_schedule_sha256": "1" * 64,
        "render_timeline_binding_sha256": "2" * 64,
    }
    source_sha = _sha256(source_core)
    source = {**source_core, "evidence_sha256": source_sha}
    failures = [] if passed else [
        "rendered_nonwide_claim_contradicted_by_wide_interior_geometry"
    ]
    summary_core: dict[str, object] = {
        "contract": RENDERED_SEGMENT_INTERIOR_GEOMETRY_REVIEW_CONTRACT,
        "required": True,
        "available": True,
        "source_evidence_sha256": source_sha,
        "source_sampling_schedule_sha256": "1" * 64,
        "source_render_timeline_binding_sha256": "2" * 64,
        "whole_song_observed": {
            "total_duration_seconds": 100.0,
            "known_duration_seconds": 100.0,
            "known_duration_fraction": 1.0,
            "close_duration_fraction": metrics["close_duration_fraction"],
            "nonwide_duration_fraction": metrics["nonwide_duration_fraction"],
            "maximum_consecutive_wide_run_seconds": metrics[
                "maximum_consecutive_wide_run_seconds"
            ],
        },
        "failure_codes": failures,
        "passed": passed,
    }
    summary = {
        **summary_core,
        "review_evidence_sha256": _sha256(summary_core),
    }
    return source, summary


def _choice(start: float, duration: float, shot_type: str) -> dict[str, object]:
    return {
        "canonical_master_start": start,
        "canonical_master_end": start + duration,
        "duration_seconds": duration,
        "shot_type": shot_type,
    }


def _legacy_analysis(metrics: dict[str, float]) -> dict[str, object]:
    total = 100.0
    close_duration = metrics["close_duration_fraction"] * total
    medium_duration = (
        metrics["nonwide_duration_fraction"]
        - metrics["close_duration_fraction"]
    ) * total
    wide_duration = (1.0 - metrics["nonwide_duration_fraction"]) * total
    maximum_wide = metrics["maximum_consecutive_wide_run_seconds"]
    wide_chunk_count = math.ceil(wide_duration / maximum_wide)
    separator_duration = close_duration / max(1, wide_chunk_count - 1)
    remaining_close = close_duration
    remaining_wide = wide_duration
    cursor = 0.0
    choices: list[dict[str, object]] = []
    for index in range(wide_chunk_count):
        chunk = min(maximum_wide, remaining_wide)
        choices.append(_choice(cursor, chunk, "group_wide"))
        cursor += chunk
        remaining_wide -= chunk
        if index < wide_chunk_count - 1:
            separator = min(separator_duration, remaining_close)
            choices.append(_choice(cursor, separator, "close_up"))
            cursor += separator
            remaining_close -= separator
    if remaining_close > 1.0e-12:
        choices.append(_choice(cursor, remaining_close, "close_up"))
        cursor += remaining_close
    choices.append(_choice(cursor, medium_duration, "medium"))
    return {
        "request": {
            "group": "IVE",
            "edit_mode": "group",
            "source_mode": "automatic",
        },
        "stage_edit_generation": None,
        "rendered_montage_grammar_review": None,
        "common_master_range": {
            "start_seconds": 0.0,
            "end_seconds": cursor + medium_duration,
            "duration_seconds": cursor + medium_duration,
        },
        "visual_timeline": {"choices": choices},
    }


def _modern_analysis(
    *,
    passed: bool = True,
    metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    selected_metrics = _I_AM if metrics is None else metrics
    whole_song = _legacy_analysis(selected_metrics)
    source_geometry, reviewed_geometry = _rendered_geometry_evidence(
        selected_metrics
    )
    return {
        "request": {
            "group": "IVE",
            "edit_mode": "group",
            "source_mode": "automatic",
        },
        "stage_edit_generation": {"activation": "explicit_request_bundle"},
        "common_master_range": whole_song["common_master_range"],
        "visual_timeline": whole_song["visual_timeline"],
        "rendered_proxy_review": {
            "segment_interior_geometry": source_geometry,
        },
        "rendered_montage_grammar_review": {
            "passed": passed,
            "rendered_segment_interior_geometry": reviewed_geometry,
            "highlight_aggregate": {
                "contract": ROLE_CONDITIONED_HIGHLIGHT_AGGREGATE_CONTRACT,
                "strict_highlight_section_count": 2,
                "passed": passed,
                "observed": {
                    "closeup_duration_fraction": 0.31,
                    "nonwide_duration_fraction": 0.55,
                    "maximum_consecutive_wide_run_seconds": 4.0,
                },
            },
        },
    }


def _set_rendered_geometry_metrics(
    analysis: dict[str, object],
    metrics: dict[str, float],
) -> None:
    summary = analysis["rendered_montage_grammar_review"][
        "rendered_segment_interior_geometry"
    ]
    summary["whole_song_observed"].update(
        {
            "close_duration_fraction": metrics["close_duration_fraction"],
            "nonwide_duration_fraction": metrics["nonwide_duration_fraction"],
            "maximum_consecutive_wide_run_seconds": metrics[
                "maximum_consecutive_wide_run_seconds"
            ],
        }
    )
    core = dict(summary)
    core.pop("review_evidence_sha256", None)
    summary["review_evidence_sha256"] = _sha256(core)


def _passing_selection() -> dict[str, object]:
    return {
        "selected_candidate_index": 0,
        "complete_candidates": [
            {
                "candidate_index": 0,
                "visual_timeline": {
                    "selected_candidate_index": 0,
                    "candidate_review_summaries": [
                        {
                            "candidate_index": 0,
                            "selected": True,
                            "passed": True,
                            "hard_failure_codes": [],
                        }
                    ],
                },
                "candidate_lineage": [],
            }
        ],
    }


class IVEProductQualityFloorTests(unittest.TestCase):
    def test_immutable_i_am_baseline_identity_and_shaped_timeline_pass(self) -> None:
        result = evaluate_ive_product_quality_floor(_legacy_analysis(_I_AM))

        self.assertTrue(result.required)
        self.assertTrue(result.passed)
        self.assertEqual(
            result.evaluation_mode,
            "legacy_whole_song_i_am_positive_regression_fallback",
        )
        self.assertEqual(result.baseline["analysis_sha256"], LEGACY_I_AM_ANALYSIS_SHA256)
        self.assertAlmostEqual(
            result.metrics["close_duration_fraction"],
            _I_AM["close_duration_fraction"],
        )
        self.assertAlmostEqual(
            result.metrics["nonwide_duration_fraction"],
            _I_AM["nonwide_duration_fraction"],
        )
        self.assertAlmostEqual(
            result.metrics["maximum_consecutive_wide_run_seconds"],
            _I_AM["maximum_consecutive_wide_run_seconds"],
        )

    def test_failed_bang_shaped_timeline_is_rejected_by_all_three_regressions(self) -> None:
        result = evaluate_ive_product_quality_floor(_legacy_analysis(_BANG))

        self.assertFalse(result.passed)
        self.assertEqual(
            set(result.failure_codes),
            {
                "legacy_close_duration_fraction_below_i_am_floor",
                "legacy_nonwide_duration_fraction_below_i_am_floor",
                "legacy_maximum_consecutive_wide_run_exceeds_i_am_floor",
            },
        )

    def test_modern_i_am_timeline_passes_role_review_then_absolute_floor(self) -> None:
        result = evaluate_ive_product_quality_floor(
            _modern_analysis(),
            selection=_passing_selection(),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.evaluation_mode, "modern_role_conditioned_profile_gate")
        self.assertFalse(result.metrics["legacy_whole_song_fallback_applied"])
        self.assertTrue(result.metrics["whole_song_i_am_absolute_floor_applied"])
        whole_song = result.metrics["whole_song_i_am_absolute_floor"]
        self.assertTrue(whole_song["passed"])
        self.assertAlmostEqual(
            whole_song["close_duration_fraction"],
            _I_AM["close_duration_fraction"],
        )
        self.assertAlmostEqual(
            whole_song["nonwide_duration_fraction"],
            _I_AM["nonwide_duration_fraction"],
        )
        self.assertAlmostEqual(
            whole_song["maximum_consecutive_wide_run_seconds"],
            _I_AM["maximum_consecutive_wide_run_seconds"],
        )
        self.assertEqual(
            result.baseline["usage"],
            "secondary_absolute_minimum_applied_after_role_conditioned_review",
        )

    def test_modern_bang_rendered_geometry_fails_all_three_absolute_metrics(self) -> None:
        analysis = _modern_analysis(metrics=_BANG)
        analysis["rendered_montage_grammar_review"]["highlight_aggregate"][
            "observed"
        ] = dict(_I_AM)

        result = evaluate_ive_product_quality_floor(
            analysis,
            selection=_passing_selection(),
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            set(result.metrics["whole_song_i_am_absolute_floor"]["failure_codes"]),
            {
                "modern_close_duration_fraction_below_i_am_floor",
                "modern_nonwide_duration_fraction_below_i_am_floor",
                "modern_maximum_consecutive_wide_run_exceeds_i_am_floor",
            },
        )
        self.assertEqual(
            set(result.failure_codes),
            set(result.metrics["whole_song_i_am_absolute_floor"]["failure_codes"]),
        )

    def test_failed_role_review_cannot_be_rescued_by_i_am_timeline(self) -> None:
        result = evaluate_ive_product_quality_floor(
            _modern_analysis(passed=False),
            selection=_passing_selection(),
        )

        self.assertFalse(result.passed)
        self.assertIn(
            "modern_role_conditioned_montage_review_failed",
            result.failure_codes,
        )
        self.assertIn("modern_highlight_aggregate_failed", result.failure_codes)
        self.assertTrue(result.metrics["whole_song_i_am_absolute_floor"]["passed"])

    def test_global_wide_run_crosses_conceptual_section_split(self) -> None:
        analysis = _modern_analysis()
        _set_rendered_geometry_metrics(
            analysis,
            {
                **_I_AM,
                "maximum_consecutive_wide_run_seconds": 8.0,
            },
        )

        result = evaluate_ive_product_quality_floor(
            analysis,
            selection=_passing_selection(),
        )

        self.assertFalse(result.passed)
        whole_song = result.metrics["whole_song_i_am_absolute_floor"]
        self.assertEqual(whole_song["maximum_consecutive_wide_run_seconds"], 8.0)
        self.assertEqual(
            whole_song["failure_codes"],
            ["modern_maximum_consecutive_wide_run_exceeds_i_am_floor"],
        )

    def test_modern_whole_song_evidence_fails_closed_when_malformed(self) -> None:
        missing = _modern_analysis()
        del missing["rendered_proxy_review"]["segment_interior_geometry"]
        with self.assertRaisesRegex(ValueError, "segment-interior geometry"):
            evaluate_ive_product_quality_floor(
                missing,
                selection=_passing_selection(),
            )

        source_tamper = _modern_analysis()
        source_tamper["rendered_proxy_review"]["segment_interior_geometry"][
            "sampling_schedule_sha256"
        ] = "3" * 64
        with self.assertRaisesRegex(ValueError, "evidence SHA-256 mismatch"):
            evaluate_ive_product_quality_floor(
                source_tamper,
                selection=_passing_selection(),
            )

        summary_tamper = _modern_analysis()
        summary_tamper["rendered_montage_grammar_review"][
            "rendered_segment_interior_geometry"
        ]["whole_song_observed"]["close_duration_fraction"] = 0.99
        with self.assertRaisesRegex(ValueError, "review SHA-256 mismatch"):
            evaluate_ive_product_quality_floor(
                summary_tamper,
                selection=_passing_selection(),
            )

        mismatched_reference = _modern_analysis()
        summary = mismatched_reference["rendered_montage_grammar_review"][
            "rendered_segment_interior_geometry"
        ]
        summary["source_evidence_sha256"] = "4" * 64
        core = dict(summary)
        core.pop("review_evidence_sha256")
        summary["review_evidence_sha256"] = _sha256(core)
        with self.assertRaisesRegex(ValueError, "references different"):
            evaluate_ive_product_quality_floor(
                mismatched_reference,
                selection=_passing_selection(),
            )

    def test_modern_generation_cannot_demote_to_legacy_fallback(self) -> None:
        analysis = _legacy_analysis(_I_AM)
        analysis["stage_edit_generation"] = {"activation": "explicit_request_bundle"}

        result = evaluate_ive_product_quality_floor(analysis)

        self.assertFalse(result.passed)
        self.assertIn(
            "modern_generation_missing_role_conditioned_highlight_evidence",
            result.failure_codes,
        )
        self.assertFalse(result.metrics["legacy_whole_song_fallback_applied"])

    def test_current_delivery_without_generation_cannot_claim_legacy_fallback(
        self,
    ) -> None:
        analysis = _legacy_analysis(_I_AM)
        analysis["artifact_role"] = "delivery"
        analysis["app_result_exposure"] = {
            "contract": APP_RESULT_EXPOSURE_CONTRACT,
            "video_output_exposed": True,
            "delivery_completion_message_allowed": True,
        }

        result = evaluate_ive_product_quality_floor(analysis)

        self.assertFalse(result.passed)
        self.assertIn(
            "current_delivery_missing_role_conditioned_highlight_evidence",
            result.failure_codes,
        )
        self.assertFalse(result.metrics["legacy_whole_song_fallback_applied"])

    def test_modern_role_evidence_without_selected_pre_review_proof_fails(self) -> None:
        result = evaluate_ive_product_quality_floor(_modern_analysis())

        self.assertFalse(result.passed)
        self.assertIn(
            "modern_selected_pre_review_evidence_missing",
            result.failure_codes,
        )

    def test_legacy_fallback_rejects_gap_unknown_and_duration_tampering(self) -> None:
        gap = _legacy_analysis(_I_AM)
        gap_choices = gap["visual_timeline"]["choices"]
        gap_choices[1]["canonical_master_start"] += 0.01
        gap_choices[1]["canonical_master_end"] += 0.01
        with self.assertRaisesRegex(ValueError, "exactly contiguous"):
            evaluate_ive_product_quality_floor(gap)

        unknown = _legacy_analysis(_I_AM)
        unknown["visual_timeline"]["choices"][0]["shot_type"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown shot-scale"):
            evaluate_ive_product_quality_floor(unknown)

        duration = _legacy_analysis(_I_AM)
        duration["visual_timeline"]["choices"][0]["duration_seconds"] += 0.01
        with self.assertRaisesRegex(ValueError, "duration disagrees"):
            evaluate_ive_product_quality_floor(duration)

    def test_legacy_fallback_rejects_common_master_range_omission(self) -> None:
        analysis = _legacy_analysis(_I_AM)
        analysis["common_master_range"]["end_seconds"] += 1.0
        analysis["common_master_range"]["duration_seconds"] += 1.0
        with self.assertRaisesRegex(ValueError, "complete common master range"):
            evaluate_ive_product_quality_floor(analysis)

    def test_other_artist_adapter_is_noop(self) -> None:
        analysis = _legacy_analysis(_BANG)
        analysis["request"]["group"] = "OTHER"

        result = evaluate_ive_product_quality_floor(analysis)

        self.assertFalse(result.required)
        self.assertIsNone(result.passed)
        self.assertEqual(result.evaluation_mode, "not_applicable")

    def test_only_explicit_app_hidden_comparison_is_exempt(self) -> None:
        analysis = _legacy_analysis(_BANG)
        analysis["artifact_role"] = "comparison"
        analysis["app_result_exposure"] = {
            "contract": APP_RESULT_EXPOSURE_CONTRACT,
            "video_output_exposed": False,
            "delivery_completion_message_allowed": False,
        }
        analysis["render"] = {
            "artifact_role": "comparison",
            "allow_failed_comparison_render": True,
        }

        exempt = evaluate_ive_product_quality_floor(analysis)

        self.assertFalse(exempt.required)
        self.assertIsNone(exempt.passed)
        self.assertEqual(
            exempt.evaluation_mode,
            "explicit_non_delivery_artifact",
        )

        analysis["app_result_exposure"]["video_output_exposed"] = True
        exposed = evaluate_ive_product_quality_floor(analysis)
        self.assertTrue(exposed.required)
        self.assertFalse(exposed.passed)

        analysis["app_result_exposure"]["video_output_exposed"] = False
        analysis["render"]["allow_failed_comparison_render"] = False
        unbound = evaluate_ive_product_quality_floor(analysis)
        self.assertTrue(unbound.required)
        self.assertFalse(unbound.passed)

    def test_selected_failed_pre_review_requires_exact_successful_lineage(self) -> None:
        selection = {
            "selected_candidate_index": 5,
            "complete_candidates": [
                {
                    "candidate_index": 5,
                    "visual_timeline": {
                        "selected_candidate_index": 5,
                        "candidate_review_summaries": [
                            {
                                "candidate_index": 5,
                                "selected": True,
                                "passed": False,
                                "hard_failure_codes": ["focus_mismatch_exceeded"],
                            }
                        ],
                    },
                    "candidate_lineage": [],
                }
            ],
        }
        rejected = evaluate_ive_product_quality_floor(
            _legacy_analysis(_I_AM), selection=selection
        )
        self.assertFalse(rejected.passed)
        self.assertIn(
            "selected_failed_pre_review_lacks_successful_repair_lineage",
            rejected.failure_codes,
        )

        selection["complete_candidates"][0]["candidate_lineage"] = [
            {
                "stage": "pre_review_failure_repair_validation",
                "contract": PRE_REVIEW_REPAIR_LINEAGE_CONTRACT,
                "candidate_index": 5,
                "pre_review_passed": False,
                "pre_review_hard_failure_codes": ["focus_mismatch_exceeded"],
                "boundary_repair_changed_count": 1,
                "post_repair_timeline_review_passed": True,
                "delivery_eligible": True,
            }
        ]
        accepted = evaluate_ive_product_quality_floor(
            _legacy_analysis(_I_AM), selection=selection
        )
        self.assertTrue(accepted.passed)
        self.assertTrue(
            accepted.repair_lineage["successful_repair_lineage_found"]
        )

        selection["complete_candidates"][0]["candidate_lineage"][0][
            "candidate_index"
        ] = 6
        wrong_candidate = evaluate_ive_product_quality_floor(
            _legacy_analysis(_I_AM), selection=selection
        )
        self.assertFalse(wrong_candidate.passed)
        self.assertIn(
            "selected_failed_pre_review_lacks_successful_repair_lineage",
            wrong_candidate.failure_codes,
        )

    def test_selected_pre_review_summary_identity_cannot_be_tampered(self) -> None:
        selection = _passing_selection()
        summary = selection["complete_candidates"][0]["visual_timeline"][
            "candidate_review_summaries"
        ][0]
        summary["candidate_index"] = 4

        result = evaluate_ive_product_quality_floor(
            _modern_analysis(),
            selection=selection,
        )

        self.assertFalse(result.passed)
        self.assertIn(
            "selected_pre_review_evidence_invalid",
            result.failure_codes,
        )

    def test_real_i_am_and_failed_bang_regressions_when_archives_are_mounted(self) -> None:
        i_am = Path(
            "/Users/developer/Movies/CrossCut QA/"
            "IVE-I-AM-20260823-054846-editorial-v2-final-analysis.json"
        )
        bang = Path(
            "/Volumes/CrossCutMedia/CrossCut-Validation/BANG-BANG/group/output/"
            "IVE-BANG-BANG-20260823-161511-analysis.json"
        )
        if not i_am.is_file() or not bang.is_file():
            self.skipTest("local archived regression analyses are not mounted")
        i_am_result = evaluate_ive_product_quality_floor(
            json.loads(i_am.read_text(encoding="utf-8"))
        )
        bang_analysis = json.loads(bang.read_text(encoding="utf-8"))
        bang_result = evaluate_ive_product_quality_floor(
            bang_analysis,
            selection=bang_analysis.get("rendered_candidate_selection"),
        )
        self.assertTrue(i_am_result.passed)
        self.assertFalse(bang_result.passed)


if __name__ == "__main__":
    unittest.main()
