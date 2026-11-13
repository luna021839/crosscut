from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.audio import OffsetEstimate
from crosscut.professional_reference_cadence import (
    AUDIT_ANALYSIS_BASIS,
    CadenceAlignmentError,
    CadenceConfig,
    ProfessionalReferenceCadenceProfile,
    ReferenceCadenceInput,
    SectionSpan,
    build_mapped_cadence_model,
    build_professional_reference_cadence_profile,
    candidate_source_switch_boundaries_from_selection_report,
    discover_reference_inputs,
)


def _reference_times() -> dict[str, tuple[float, ...]]:
    canonical = (4.0, 8.0, 12.0, 16.0, 20.0, 24.0)
    jitters = (-0.055, -0.025, 0.0, 0.030, 0.050)
    return {
        f"human-{index}": tuple(value + jitters[index] for value in canonical)
        for index in range(5)
    }


def _audit(path: Path, media: Path, times: tuple[float, ...]) -> None:
    cuts = [
        {
            "frame_index": int(round(seconds * 30.0)),
            "seconds": seconds,
            "score": 70.0,
            "discontinuities": {},
            "issue_reasons": [],
        }
        for seconds in times
    ]
    path.write_text(
        json.dumps(
            {
                "analysis_basis": AUDIT_ANALYSIS_BASIS,
                "reference": str(media.resolve()),
                "summary": {"cut_count": len(cuts)},
                "cuts": cuts,
            }
        ),
        encoding="utf-8",
    )


class _IdentityWarp:
    scalar_fallback = False
    fallback_reason = None

    def source_to_master(self, source_seconds: float) -> float:
        return source_seconds

    def to_dict(self) -> dict:
        return {
            "backend": "synthetic_identity_for_test",
            "global_offset_seconds": 0.0,
            "master_duration_seconds": 30.0,
            "source_duration_seconds": 30.0,
            "scalar_fallback": False,
            "fallback_reason": None,
            "control_points": [
                {
                    "master_seconds": 0.0,
                    "source_seconds": 0.0,
                    "offset_seconds": 0.0,
                },
                {
                    "master_seconds": 30.0,
                    "source_seconds": 30.0,
                    "offset_seconds": 0.0,
                },
            ],
        }


def _good_offset() -> OffsetEstimate:
    return OffsetEstimate(
        offset_samples=0,
        offset_seconds=0.0,
        score=0.94,
        overlap_samples=3_000,
        overlap_seconds=30.0,
        sample_rate=100,
        second_best_score=0.62,
        score_margin=0.32,
        overlap_ratio=1.0,
        backend="spectral_flux_ncc",
    )


class MappedCadenceModelTests(unittest.TestCase):
    def test_leave_one_reference_out_rewards_each_unseen_human_timing(self) -> None:
        references = _reference_times()
        model = build_mapped_cadence_model(
            references,
            master_duration_seconds=28.0,
            sections=(
                SectionSpan("first", "first_half", 0.0, 14.0),
                SectionSpan("second", "second_half", 14.0, 28.0),
            ),
        )

        for reference_id, held_out_times in references.items():
            with self.subTest(reference_id=reference_id):
                held_out = model.score_candidate_boundaries(
                    held_out_times,
                    hard_safety_passed=True,
                    excluded_reference_ids=(reference_id,),
                )
                shifted = model.score_candidate_boundaries(
                    tuple(value + 0.55 for value in held_out_times),
                    hard_safety_passed=True,
                    excluded_reference_ids=(reference_id,),
                )
                self.assertGreater(held_out.raw_prior, 0.65)
                self.assertGreater(held_out.raw_prior, shifted.raw_prior)
                self.assertEqual(held_out.payload["training_reference_count"], 4)

    def test_on_consensus_beats_shifted_but_prior_is_capped_and_safety_preserved(self) -> None:
        references = _reference_times()
        model = build_mapped_cadence_model(
            references,
            master_duration_seconds=28.0,
            config=CadenceConfig(maximum_prior_bonus_points=2.5),
        )
        consensus = tuple(sum(values) / len(values) for values in zip(*references.values()))
        on_consensus = model.score_candidate_boundaries(
            consensus,
            hard_safety_passed=True,
            base_score=80.0,
        )
        shifted = model.score_candidate_boundaries(
            tuple(value + 0.60 for value in consensus),
            hard_safety_passed=True,
            base_score=80.0,
        )
        unsafe = model.score_candidate_boundaries(
            consensus,
            hard_safety_passed=False,
            base_score=80.0,
        )

        self.assertGreater(on_consensus.raw_prior, shifted.raw_prior)
        self.assertLessEqual(on_consensus.applied_bonus_points, 2.5)
        self.assertEqual(on_consensus.payload["adjusted_score"], 82.5)
        self.assertEqual(shifted.payload["adjusted_score"], 80.0)
        self.assertEqual(unsafe.applied_bonus_points, 0.0)
        self.assertEqual(unsafe.payload["adjusted_score"], 80.0)
        self.assertFalse(unsafe.payload["eligible_for_ranking_bonus"])

    def test_clusters_are_diameter_bounded_and_have_unique_reference_support(self) -> None:
        references = _reference_times()
        model = build_mapped_cadence_model(
            references,
            master_duration_seconds=28.0,
            config=CadenceConfig(cluster_tolerance_seconds=0.12),
        )

        clusters = model.clusters()

        self.assertEqual(len(clusters), 6)
        self.assertTrue(all(cluster.support_count == 5 for cluster in clusters))
        self.assertTrue(all(cluster.spread_seconds <= 0.12 for cluster in clusters))
        self.assertTrue(
            all(
                len({member.reference_id for member in cluster.members})
                == cluster.support_count
                for cluster in clusters
            )
        )
        distributions = model.distributions()
        self.assertEqual(distributions["whole_song"]["visible_cut_rate_per_minute"]["count"], 5)
        self.assertEqual(len(distributions["by_section"]), 1)


class CadenceProfileTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, tuple[ReferenceCadenceInput, ...]]:
        master = root / "master-source.bin"
        master.write_bytes(b"master-media")
        evidence = root / "evidence"
        evidence.mkdir()
        inputs: list[ReferenceCadenceInput] = []
        for index, jitter in enumerate((-0.02, 0.02)):
            media = root / f"video-{index}-IVE-stage-mix.mp4"
            media.write_bytes(f"media-{index}".encode())
            audit = evidence / f"audit-{index}-human-cut-quality-audit.json"
            _audit(audit, media, tuple(value + jitter for value in (5.0, 10.0, 15.0)))
            inputs.append(ReferenceCadenceInput(f"video-{index}", media, audit))
        # This duplicate has no declared reference and must not create ambiguity.
        (evidence / "legacy-human-cut-quality-audit.json").write_text(
            json.dumps({"summary": {}, "cuts": []}), encoding="utf-8"
        )
        return master, tuple(inputs)

    def test_profile_round_trip_recomputes_hash_mapping_clusters_and_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master, inputs = self._inputs(root)
            pcm = np.zeros(3_000, dtype=np.float32)
            config = CadenceConfig(minimum_coarse_overlap_seconds=10.0)
            with (
                patch(
                    "crosscut.professional_reference_cadence.extract_mono_pcm",
                    return_value=pcm,
                ),
                patch(
                    "crosscut.professional_reference_cadence.spectral_flux_alignment_offset",
                    return_value=_good_offset(),
                ),
                patch(
                    "crosscut.professional_reference_cadence.estimate_local_timewarp",
                    return_value=_IdentityWarp(),
                ),
            ):
                profile = build_professional_reference_cadence_profile(
                    master,
                    inputs,
                    artist="IVE",
                    song="SYNTHETIC",
                    config=config,
                    sample_rate=100,
                )
            profile_path = profile.write_json(root / "cadence-profile.json")

            loaded = ProfessionalReferenceCadenceProfile.from_file(profile_path)

            self.assertEqual(loaded.reference_count, 2)
            self.assertEqual(loaded.to_dict()["consensus_summary"]["consensus_cluster_count"], 3)
            self.assertEqual(len(loaded.file_sha256 or ""), 64)
            inputs[0].media_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "media_sha256 mismatch"):
                ProfessionalReferenceCadenceProfile.from_file(profile_path)

    def test_low_or_ambiguous_alignment_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            master, inputs = self._inputs(Path(directory))
            pcm = np.zeros(3_000, dtype=np.float32)
            weak = OffsetEstimate(
                offset_samples=0,
                offset_seconds=0.0,
                score=0.40,
                overlap_samples=3_000,
                overlap_seconds=30.0,
                sample_rate=100,
                second_best_score=0.39,
                score_margin=0.01,
                overlap_ratio=1.0,
                backend="spectral_flux_ncc",
            )
            with (
                patch(
                    "crosscut.professional_reference_cadence.extract_mono_pcm",
                    return_value=pcm,
                ),
                patch(
                    "crosscut.professional_reference_cadence.spectral_flux_alignment_offset",
                    return_value=weak,
                ),
            ):
                with self.assertRaisesRegex(CadenceAlignmentError, "low_score"):
                    build_professional_reference_cadence_profile(
                        master,
                        inputs,
                        artist="IVE",
                        song="SYNTHETIC",
                        config=CadenceConfig(minimum_coarse_overlap_seconds=10.0),
                        sample_rate=100,
                    )

    def test_discovery_matches_declared_paths_not_hyphen_splitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _master, expected = self._inputs(root)

            discovered = discover_reference_inputs(root)

            self.assertEqual(
                {item.media_path for item in discovered},
                {item.media_path for item in expected},
            )

    def test_selection_report_reader_keeps_only_actual_source_switches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "selection.json"
            report.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_index": 13,
                                "review": {
                                    "proxy": {"fps": 30.0},
                                    "boundaries": [
                                        {
                                            "cut_frame_index": 30,
                                            "outgoing_source_index": 2,
                                            "incoming_source_index": 2,
                                        },
                                        {
                                            "cut_frame_index": 90,
                                            "outgoing_source_index": 2,
                                            "incoming_source_index": 4,
                                        },
                                    ],
                                },
                            }
                        ],
                        "complete_candidates": [
                            {
                                "candidate_index": 13,
                                "final_render_segments": [
                                    {
                                        "canonical_master_start": 2.0,
                                        "canonical_master_end": 12.0,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = candidate_source_switch_boundaries_from_selection_report(
                report, candidate_indices=(13,)
            )

            self.assertEqual(result[0].boundary_times_seconds, (5.0,))
            self.assertEqual(result[0].canonical_master_start_seconds, 2.0)
            self.assertEqual(len(result[0].selection_report_sha256), 64)


if __name__ == "__main__":
    unittest.main()
