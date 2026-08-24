from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import crosscut_cli
from crosscut.professional_reference_gate import (
    DISCONTINUITY_COMPONENTS,
    PROFILE_ANALYSIS_BASIS,
    ProfessionalReferenceCut,
    ProfessionalReferenceProfile,
    _summarize_cuts,
    evaluate_professional_reference_gate,
    evaluate_professional_reference_review,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cut(reference_index: int, cut_index: int) -> ProfessionalReferenceCut:
    # Five intentionally different but overlapping professional distributions.
    phase = ((cut_index * 7 + reference_index * 3) % 19) / 18.0
    discontinuities = {
        component: min(
            1.0,
            0.08
            + 0.025 * reference_index
            + 0.28 * phase
            + 0.005 * component_index,
        )
        for component_index, component in enumerate(DISCONTINUITY_COMPONENTS)
    }
    return ProfessionalReferenceCut(
        score=72.0 - 1.5 * reference_index - 8.0 * phase,
        discontinuities=discontinuities,
        # A high absolute warning rate is deliberate: it remains diagnostic.
        absolute_issue_reasons=("color_shock",) if cut_index % 2 == 0 else (),
    )


def _cut_mapping(cut: ProfessionalReferenceCut, frame_index: int) -> dict:
    return {
        "frame_index": frame_index,
        "seconds": frame_index / 30.0,
        "score": cut.score,
        "discontinuities": dict(cut.discontinuities),
        "issue_reasons": list(cut.absolute_issue_reasons),
    }


def _build_profile(root: Path) -> Path:
    references = []
    all_cuts: list[ProfessionalReferenceCut] = []
    for reference_index in range(5):
        video_id = f"human-{reference_index}"
        media = root / f"{video_id}.mp4"
        media.write_bytes((f"media-{reference_index}-" * 30).encode())
        cuts = tuple(_cut(reference_index, index) for index in range(20))
        audit = root / f"{video_id}-audit.json"
        summary = _summarize_cuts(cuts)
        audit.write_text(
            json.dumps(
                {
                    "analysis_basis": (
                        "same_song_human_stage_mix_cut_quality_calibration_v1"
                    ),
                    "reference": str(media.resolve()),
                    "summary": summary,
                    "cuts": [
                        _cut_mapping(cut, index + 10)
                        for index, cut in enumerate(cuts)
                    ],
                }
            ),
            encoding="utf-8",
        )
        references.append(
            {
                "video_id": video_id,
                "path": str(media.resolve()),
                "sha256": _sha256(media),
                "fps": 30.0,
                "audit_path": str(audit.resolve()),
                "summary": summary,
            }
        )
        all_cuts.extend(cuts)
    profile = root / "professional-profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analysis_basis": PROFILE_ANALYSIS_BASIS,
                "artist": "IVE",
                "song": "SYNTHETIC",
                "reference_count": 5,
                "reference_cut_count": len(all_cuts),
                "reference_videos": references,
                "aggregate_human_visible_cut_profile": _summarize_cuts(all_cuts),
                "policy_note": (
                    "technical decode, black, duplicate and freeze remain strict"
                ),
            }
        ),
        encoding="utf-8",
    )
    return profile


class ProfessionalReferenceGateTests(unittest.TestCase):
    def test_validates_schema_reference_count_audits_and_media_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _build_profile(root)

            profile = ProfessionalReferenceProfile.from_file(path)

            self.assertEqual(profile.reference_count, 5)
            self.assertEqual(profile.reference_cut_count, 100)
            self.assertEqual(len(profile.profile_sha256), 64)
            self.assertTrue(
                all(len(reference.audit_sha256) == 64 for reference in profile.references)
            )

            profile.references[0].path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                ProfessionalReferenceProfile.from_file(path)

    def test_each_human_reference_passes_other_four_reference_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = ProfessionalReferenceProfile.from_file(
                _build_profile(Path(directory))
            )

            for held_out in profile.references:
                with self.subTest(held_out=held_out.video_id):
                    result = evaluate_professional_reference_gate(
                        held_out.cuts,
                        profile,
                        excluded_reference_id=held_out.video_id,
                    )
                    self.assertTrue(result.passed)
                    self.assertEqual(
                        result.payload["evaluation_fold"][
                            "training_reference_count"
                        ],
                        4,
                    )

    def test_synthetically_worsened_candidate_fails_distribution_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = ProfessionalReferenceProfile.from_file(
                _build_profile(Path(directory))
            )
            degraded = tuple(
                ProfessionalReferenceCut(
                    score=8.0,
                    discontinuities={
                        component: 0.99 for component in DISCONTINUITY_COMPONENTS
                    },
                    absolute_issue_reasons=("color_shock",),
                )
                for _ in range(20)
            )

            result = evaluate_professional_reference_gate(degraded, profile)

            self.assertFalse(result.passed)
            self.assertFalse(result.payload["scalar_score"]["passed"])
            self.assertTrue(
                any(
                    not component["passed"]
                    for component in result.payload["components"].values()
                )
            )

    def test_absolute_background_reasons_are_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = ProfessionalReferenceProfile.from_file(
                _build_profile(Path(directory))
            )
            cuts = tuple(
                ProfessionalReferenceCut(
                    score=cut.score,
                    discontinuities=cut.discontinuities,
                    absolute_issue_reasons=(
                        "color_shock",
                        "exposure_discontinuity",
                    ),
                )
                for cut in profile.references[2].cuts
            )

            result = evaluate_professional_reference_gate(cuts, profile)

            self.assertTrue(result.passed)
            diagnostics = result.payload["absolute_editorial_diagnostics"]
            self.assertEqual(diagnostics["issue_fraction"], 1.0)
            self.assertFalse(diagnostics["selection_gate"])

    def test_technical_failure_is_zero_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = ProfessionalReferenceProfile.from_file(
                _build_profile(Path(directory))
            )

            result = evaluate_professional_reference_gate(
                profile.references[0].cuts,
                profile,
                technical_issue_reasons=(
                    "introduced_cross_source_black_frame",
                ),
            )

            self.assertFalse(result.passed)
            self.assertFalse(result.payload["technical_integrity"]["passed"])
            self.assertEqual(result.payload["technical_integrity"]["issue_count"], 1)

    def test_review_adapter_uses_only_cross_source_and_keeps_tier_d_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = ProfessionalReferenceProfile.from_file(
                _build_profile(Path(directory))
            )
            visible = [
                {
                    "boundary_index": index,
                    "outgoing_source_index": 0,
                    "incoming_source_index": 1,
                    "review_domain": "introduced_cross_source",
                    "diagnostic_editorial_weighted_score": cut.score,
                    "discontinuities": dict(cut.discontinuities),
                    "diagnostic_editorial_issue_reasons": ["color_shock"],
                    "issue_reasons": ["color_shock"],
                }
                for index, cut in enumerate(profile.references[1].cuts)
            ]
            tier_d = {
                "boundary_index": 20,
                "outgoing_source_index": 0,
                "incoming_source_index": 0,
                "review_domain": "tier_d_continuity",
                "diagnostic_editorial_weighted_score": 100.0,
                "discontinuities": {
                    component: 0.0 for component in DISCONTINUITY_COMPONENTS
                },
                "diagnostic_editorial_issue_reasons": [],
                "issue_reasons": [],
            }
            review = {
                "proxy": {"decoded_frame_count": 600},
                "boundaries": [*visible, tier_d],
                "boundary_domains": {
                    "tier_d_continuity": {"passed": True},
                    "native_source_cut": {"passed": True},
                },
            }

            passing = evaluate_professional_reference_review(review, profile)
            self.assertTrue(passing.passed)
            self.assertEqual(passing.payload["candidate_visible_cut_count"], 20)
            self.assertEqual(
                passing.payload["absolute_editorial_diagnostics"]["issue_fraction"],
                1.0,
            )

            tier_d["issue_reasons"] = [
                "tier_d_cross_boundary_duplicate_or_freeze"
            ]
            failing = evaluate_professional_reference_review(review, profile)
            self.assertFalse(failing.passed)
            self.assertFalse(failing.payload["technical_integrity"]["passed"])

    def test_cli_validates_profile_and_writes_relative_gate_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = _build_profile(root)
            profile = ProfessionalReferenceProfile.from_file(profile_path)
            review_path = root / "review.json"
            output_path = root / "professional-gate.json"
            review_path.write_text(
                json.dumps(
                    {
                        "proxy": {"decoded_frame_count": 600},
                        "boundaries": [
                            {
                                "boundary_index": index,
                                "outgoing_source_index": 0,
                                "incoming_source_index": 1,
                                "review_domain": "introduced_cross_source",
                                "diagnostic_editorial_weighted_score": cut.score,
                                "discontinuities": dict(cut.discontinuities),
                                "diagnostic_editorial_issue_reasons": [
                                    "color_shock"
                                ],
                                "issue_reasons": ["color_shock"],
                            }
                            for index, cut in enumerate(
                                profile.references[0].cuts
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with patch(
                "crosscut_cli._prepare_analysis_process_group",
                return_value=False,
            ), redirect_stdout(stdout):
                exit_code = crosscut_cli.main(
                    [
                        "professional-reference-gate",
                        "--profile",
                        str(profile_path),
                        "--review",
                        str(review_path),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["reference_count"], 5)
            self.assertEqual(summary["candidate_visible_cut_count"], 20)
            self.assertTrue(output_path.is_file())
            self.assertTrue(
                json.loads(output_path.read_text(encoding="utf-8"))["passed"]
            )

            replay_path = root / "candidate-replay-audit.json"
            replay_output_path = root / "candidate-replay-gate.json"
            replay_path.write_text(
                json.dumps(
                    {
                        "analysis_basis": (
                            "strict_failed_complete_candidate_qa_replay_v1"
                        ),
                        "candidate_index": 13,
                        "decoded_proxy_review": json.loads(
                            review_path.read_text(encoding="utf-8")
                        ),
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with patch(
                "crosscut_cli._prepare_analysis_process_group",
                return_value=False,
            ), redirect_stdout(stdout):
                replay_exit_code = crosscut_cli.main(
                    [
                        "professional-reference-gate",
                        "--profile",
                        str(profile_path),
                        "--review",
                        str(replay_path),
                        "--output",
                        str(replay_output_path),
                    ]
                )

            self.assertEqual(replay_exit_code, 0)
            replay_gate = json.loads(
                replay_output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                replay_gate["candidate_evidence"]["review_container_field"],
                "decoded_proxy_review",
            )
            self.assertEqual(
                replay_gate["candidate_evidence"]["document_sha256"],
                _sha256(replay_path),
            )
            self.assertTrue(replay_gate["candidate_evidence"]["no_rerender"])


if __name__ == "__main__":
    unittest.main()
