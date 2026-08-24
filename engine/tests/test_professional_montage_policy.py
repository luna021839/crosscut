from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.professional_montage_grammar import ROLE_NAMES
from crosscut.professional_montage_policy import (
    FORMATION_EVIDENCE_THRESHOLD_NORMALIZED,
    FORMATION_PERSON_COUNT_SEMANTICS,
    PRODUCT_EDITORIAL_REQUIREMENT_BASIS,
    ProfessionalMontagePolicyError,
    load_professional_montage_policy,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _reference_distribution(
    value: float,
    *,
    count: int = 20,
    median: float | None = None,
) -> dict[str, object]:
    resolved_median = value if median is None else median
    return {
        "count": count,
        "minimum": 0.0 if value == 0.0 else value * 0.5,
        "p10": value,
        "p25": value,
        "median": resolved_median,
        "p75": max(value, resolved_median),
        "p90": max(value, resolved_median),
        "maximum": max(value, resolved_median),
        "reference_floor": {
            "quantile": 0.1,
            "value": value,
            "basis": "lower_quantile_of_contributing_human_reference_edits",
        },
    }


def _event_distribution(value: float, *, count: int) -> dict[str, object]:
    return {
        "count": count,
        "minimum": value * 0.5,
        "p10": value,
        "p25": value,
        "median": value,
        "p75": value,
        "p90": value,
        "maximum": value * 2.0,
    }


def _profile() -> dict[str, object]:
    chorus = {
        "contributing_reference_count": 20,
        "pooled_counts": {
            "nonwide_multi_source_burst_count": 16,
            "closeup_multi_source_burst_count": 6,
        },
        "reference_distributions": {
            "shot_size_fractions": {
                "wide": _reference_distribution(0.2381818181818182),
                "medium": _reference_distribution(0.0),
                "close": _reference_distribution(0.15096153846153848),
            },
            "entry_wide_anchor_fraction": _reference_distribution(0.0),
            "early_wide_anchor_fraction": _reference_distribution(1.0 / 3.0),
            "phrase_entry_wide_anchor_fraction": _reference_distribution(
                0.2785714285714286,
                count=19,
            ),
            "source_switch_with_wide_fraction": _reference_distribution(
                0.37333333333333335,
                count=17,
                median=0.8,
            ),
            "nonwide_multi_source_burst_rate_per_minute": (
                _reference_distribution(0.0)
            ),
            "closeup_multi_source_burst_rate_per_minute": (
                _reference_distribution(0.0)
            ),
        },
        "burst_event_distributions": {
            "nonwide_multi_source_shot_count": _event_distribution(2.0, count=16),
            "nonwide_multi_source_duration_seconds": _event_distribution(
                2.299924872715586,
                count=16,
            ),
            "nonwide_multi_source_switch_fraction": _event_distribution(
                5.0 / 6.0,
                count=16,
            ),
            "closeup_multi_source_shot_count": _event_distribution(2.0, count=6),
            "closeup_multi_source_duration_seconds": _event_distribution(
                2.1503940144767864,
                count=6,
            ),
            "closeup_multi_source_switch_fraction": _event_distribution(
                2.0 / 3.0,
                count=6,
            ),
        },
        "wide_shot_person_count_evidence": {
            "detected_person_count_normalized": _event_distribution(
                1.0 / 6.0,
                count=82,
            ),
            "formation_person_count_normalized": _event_distribution(
                1.0 / 6.0,
                count=82,
            ),
            "semantics": FORMATION_PERSON_COUNT_SEMANTICS,
        },
    }
    songs = [
        {
            "song_id": f"IVE-SONG-{song_index}",
            "reference_count": 5,
            "source_count": 10,
            "references": [
                {"reference_id": f"REF-{song_index}-{reference_index}"}
                for reference_index in range(5)
            ],
        }
        for song_index in range(4)
    ]
    profile: dict[str, object] = {
        "schema_version": 1,
        "analysis_basis": "ive_human_reference_visual_montage_grammar_v1",
        "configuration": {
            "minimum_song_count": 4,
            "minimum_references_per_song": 5,
            "minimum_sources_per_song": 8,
            "early_anchor_shot_count": 2,
            "reference_floor_quantile": 0.1,
            "short_shot_basis": (
                "duration_at_or_below_each_reference_edits_observed_median"
            ),
            "burst_connectivity": (
                "contiguous_known_shots_with_cross_source_switches"
            ),
        },
        "distributional_policy": {
            "hand_authored_visual_quality_thresholds_present": False,
            "reference_floors_are_lower_distribution_quantiles": True,
        },
        "evidence_limitations": {
            "generic_wide_is_proven_group_formation": False,
            "person_count_threshold_used_to_assert_group_formation": False,
            "formation_identity_or_member_labels_present": False,
        },
        "provenance": {
            "content_hash_algorithm": (
                "sha256_of_canonical_json_with_profile_content_sha256_null"
            ),
            "profile_content_sha256": None,
            "generator": {
                "module": "crosscut.professional_montage_grammar",
                "module_version": "1",
            },
            "songs": [
                {"song_id": song["song_id"]} for song in songs
            ],
        },
        "scope": {
            "artist": "IVE",
            "song_count": 4,
            "reference_count": 20,
        },
        "songs": songs,
        "aggregate_by_acoustic_role": {
            role: chorus if role == "chorus_like" else {}
            for role in ROLE_NAMES
        },
    }
    profile["provenance"]["profile_content_sha256"] = _canonical_sha256(profile)  # type: ignore[index]
    return profile


def _rehash(profile: dict[str, object]) -> None:
    provenance = profile["provenance"]
    assert isinstance(provenance, dict)
    provenance["profile_content_sha256"] = None
    provenance["profile_content_sha256"] = _canonical_sha256(profile)


def _write(path: Path, profile: dict[str, object]) -> bytes:
    raw = (json.dumps(profile, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(raw)
    return raw


class ProfessionalMontagePolicyTests(unittest.TestCase):
    def test_loads_hash_bound_reference_thresholds_and_separate_product_rules(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            raw = _write(path, _profile())
            file_sha256 = hashlib.sha256(raw).hexdigest()

            policy = load_professional_montage_policy(
                path,
                expected_file_sha256=file_sha256,
            )

            self.assertEqual(
                policy.profile_identity,
                {
                    "analysis_basis": (
                        "ive_human_reference_visual_montage_grammar_v1"
                    ),
                    "schema_version": 1,
                    "path": str(path),
                    "file_sha256": file_sha256,
                    "content_sha256": _profile()["provenance"][
                        "profile_content_sha256"
                    ],
                },
            )
            expected = {
                "minimum_close_duration_fraction": 0.15096153846153848,
                "minimum_early_wide_anchor_fraction": 1.0 / 3.0,
                "minimum_phrase_entry_wide_anchor_fraction": (
                    0.2785714285714286
                ),
                "minimum_wide_involved_source_switch_fraction": (
                    0.37333333333333335
                ),
                "minimum_nonwide_burst_shot_count": 2.0,
                "minimum_nonwide_burst_duration_seconds": 2.299924872715586,
                "minimum_nonwide_burst_source_switch_fraction": 5.0 / 6.0,
                "minimum_close_burst_shot_count": 2.0,
                "minimum_close_burst_duration_seconds": 2.1503940144767864,
                "minimum_close_burst_source_switch_fraction": 2.0 / 3.0,
            }
            for name, value in expected.items():
                threshold = getattr(policy, name)
                self.assertAlmostEqual(threshold.value, value)
                self.assertEqual(
                    threshold.provenance.profile_identity.to_dict(),
                    policy.profile_identity,
                )
                self.assertIn("chorus_like", threshold.provenance.statistic_path)
                self.assertEqual(threshold.provenance.quantile, 0.1)

            preferred = (
                policy.preferred_wide_involved_source_switch_fraction
            )
            self.assertEqual(preferred.value, 0.8)
            self.assertEqual(preferred.reference_count, 17)
            self.assertEqual(preferred.provenance.quantile, 0.5)
            self.assertEqual(
                preferred.provenance.statistic,
                "profile_reference_median_soft_preferred_target",
            )
            serialized_preferred = policy.to_dict()[
                "profile_derived_preferred_targets"
            ]["preferred_wide_involved_source_switch_fraction"]
            self.assertFalse(serialized_preferred["hard_gate"])
            self.assertEqual(serialized_preferred["reference_count"], 17)

            self.assertIsNone(policy.minimum_nonwide_duration_fraction)
            self.assertEqual(
                policy.nonwide_duration_fraction_policy,
                "unavailable_never_derived_from_one_minus_wide",
            )
            zero_paths = {
                item.statistic_path
                for item in policy.unusable_zero_reference_floors
            }
            self.assertIn(
                "aggregate_by_acoustic_role.chorus_like.reference_distributions."
                "shot_size_fractions.medium",
                zero_paths,
            )
            self.assertIn(
                "aggregate_by_acoustic_role.chorus_like.reference_distributions."
                "nonwide_multi_source_burst_rate_per_minute",
                zero_paths,
            )
            serialized = policy.to_dict()
            self.assertTrue(
                all(
                    not item["usable_as_positive_production_minimum"]
                    for item in serialized["unusable_zero_reference_floors"]
                )
            )

            product = policy.product_editorial_requirements
            self.assertEqual(
                product.requirement_basis,
                PRODUCT_EDITORIAL_REQUIREMENT_BASIS,
            )
            self.assertEqual(product.minimum_nonwide_multi_family_burst_count, 1)
            self.assertEqual(
                product.minimum_post_burst_formation_evidenced_wide_reset_count,
                1,
            )
            self.assertEqual(
                product.formation_evidence_threshold_normalized,
                FORMATION_EVIDENCE_THRESHOLD_NORMALIZED,
            )
            self.assertEqual(
                product.formation_evidence_semantics,
                FORMATION_PERSON_COUNT_SEMANTICS,
            )
            self.assertIn("weak_observed_person_evidence", product.formation_evidence_label)
            self.assertFalse(product.formation_evidence_proves_full_group_formation)
            self.assertFalse(product.missing_or_zero_formation_evidence_passes)
            self.assertFalse(product.is_reference_floor)

            from crosscut.pipeline import (
                _rendered_montage_config_from_policy,
                _stage_edit_semimarkov_config_from_montage_policy,
            )

            decode_config = _stage_edit_semimarkov_config_from_montage_policy(
                policy
            )
            rendered_config = _rendered_montage_config_from_policy(policy)
            self.assertEqual(
                decode_config.highlight_preferred_wide_involved_source_switch_fraction,
                0.8,
            )
            self.assertEqual(
                decode_config.highlight_wide_involved_source_switch_reference_count,
                17,
            )
            self.assertEqual(
                rendered_config.preferred_wide_involved_source_switch_fraction,
                0.8,
            )
            self.assertEqual(
                rendered_config.wide_involved_source_switch_reference_count,
                17,
            )

    def test_expected_raw_file_hash_is_optional_but_strict_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            _write(path, _profile())
            self.assertEqual(
                load_professional_montage_policy(path).identity.path,
                str(path),
            )
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "file SHA-256 mismatch",
            ):
                load_professional_montage_policy(
                    path,
                    expected_file_sha256="0" * 64,
                )
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "lowercase SHA-256",
            ):
                load_professional_montage_policy(
                    path,
                    expected_file_sha256="invalid",
                )

    def test_content_tampering_fails_even_without_expected_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            profile = _profile()
            profile["scope"]["artist"] = "OTHER"  # type: ignore[index]
            _write(path, profile)
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "content SHA-256 mismatch",
            ):
                load_professional_montage_policy(path)

    def test_semantic_tampering_fails_even_when_attacker_rehashes_profile(self) -> None:
        mutations = (
            ("schema", lambda value: value.__setitem__("schema_version", 2)),
            ("basis", lambda value: value.__setitem__("analysis_basis", "other")),
            (
                "generator",
                lambda value: value["provenance"]["generator"].__setitem__(  # type: ignore[index,union-attr]
                    "module_version", "2"
                ),
            ),
            (
                "song count",
                lambda value: value["scope"].__setitem__("song_count", 3),  # type: ignore[union-attr]
            ),
            (
                "reference count",
                lambda value: value["aggregate_by_acoustic_role"][  # type: ignore[index]
                    "chorus_like"
                ].__setitem__("contributing_reference_count", 19),
            ),
            (
                "formation semantics",
                lambda value: value["aggregate_by_acoustic_role"][  # type: ignore[index]
                    "chorus_like"
                ]["wide_shot_person_count_evidence"].__setitem__(
                    "semantics", "claimed_full_formation"
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (name, mutate) in enumerate(mutations):
                with self.subTest(name=name):
                    profile = _profile()
                    mutate(profile)
                    _rehash(profile)
                    path = Path(temporary) / f"profile-{index}.json"
                    _write(path, profile)
                    with self.assertRaises(ProfessionalMontagePolicyError):
                        load_professional_montage_policy(path)

    def test_named_thresholds_fail_closed_on_forged_or_zero_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mismatched = _profile()
            close = mismatched["aggregate_by_acoustic_role"]["chorus_like"][  # type: ignore[index]
                "reference_distributions"
            ]["shot_size_fractions"]["close"]
            close["reference_floor"]["value"] = 0.14
            _rehash(mismatched)
            mismatch_path = Path(temporary) / "mismatch.json"
            _write(mismatch_path, mismatched)
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "not a profile p10 reference floor",
            ):
                load_professional_montage_policy(mismatch_path)

            zero = _profile()
            early = zero["aggregate_by_acoustic_role"]["chorus_like"][  # type: ignore[index]
                "reference_distributions"
            ]["early_wide_anchor_fraction"]
            early["p10"] = 0.0
            early["reference_floor"]["value"] = 0.0
            _rehash(zero)
            zero_path = Path(temporary) / "zero.json"
            _write(zero_path, zero)
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "cannot supply a positive production minimum",
            ):
                load_professional_montage_policy(zero_path)

            inverted = _profile()
            wide_switch = inverted["aggregate_by_acoustic_role"][  # type: ignore[index]
                "chorus_like"
            ]["reference_distributions"]["source_switch_with_wide_fraction"]
            wide_switch["median"] = 0.2
            _rehash(inverted)
            inverted_path = Path(temporary) / "inverted-median.json"
            _write(inverted_path, inverted)
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "reference-median target",
            ):
                load_professional_montage_policy(inverted_path)

    def test_event_distribution_count_must_match_pooled_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = _profile()
            profile["aggregate_by_acoustic_role"]["chorus_like"][  # type: ignore[index]
                "pooled_counts"
            ]["nonwide_multi_source_burst_count"] = 15
            _rehash(profile)
            path = Path(temporary) / "profile.json"
            _write(path, profile)
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "disagrees with its event distribution",
            ):
                load_professional_montage_policy(path)

    def test_rejects_relative_path_and_symlink(self) -> None:
        with self.assertRaisesRegex(
            ProfessionalMontagePolicyError,
            "path must be absolute",
        ):
            load_professional_montage_policy(Path("profile.json"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "profile.json"
            _write(path, _profile())
            link = root / "profile-link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "non-symlink",
            ):
                load_professional_montage_policy(link)

    def test_rejects_duplicate_keys_and_nonfinite_json_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "duplicate JSON key",
            ):
                load_professional_montage_policy(duplicate)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"schema_version":NaN}')
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "non-finite JSON number",
            ):
                load_professional_montage_policy(nonfinite)


if __name__ == "__main__":
    unittest.main()
