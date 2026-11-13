from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from crosscut.professional_montage_grammar import (
    PROFILE_ANALYSIS_BASIS,
    PROFILE_SCHEMA_VERSION,
    ROLE_NAMES,
)
from crosscut.professional_montage_policy import (
    DURATION_PREFERRED_TARGET_QUANTILE,
    DURATION_PRODUCT_MINIMUM_QUANTILE,
    FORMATION_EVIDENCE_THRESHOLD_NORMALIZED,
    FORMATION_PERSON_COUNT_SEMANTICS,
    PRODUCT_EDITORIAL_REQUIREMENT_BASIS,
    ProfessionalMontagePolicyError,
    WIDE_RUN_HARD_MAXIMUM_QUANTILE,
    WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE,
    load_professional_montage_policy,
)
from crosscut.stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from crosscut.stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
)
from crosscut.stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
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


def _linear_quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _reference_distribution_from_values(
    values: list[float],
) -> dict[str, object]:
    return {
        "count": len(values),
        "minimum": min(values),
        "p10": _linear_quantile(values, 0.10),
        "p25": _linear_quantile(values, 0.25),
        "median": _linear_quantile(values, 0.50),
        "p75": _linear_quantile(values, 0.75),
        "p90": _linear_quantile(values, 0.90),
        "maximum": max(values),
        "reference_floor": {
            "quantile": 0.1,
            "value": _linear_quantile(values, 0.10),
            "basis": "lower_quantile_of_contributing_human_reference_edits",
        },
    }


def _event_distribution_from_values(values: list[float]) -> dict[str, object]:
    result = _reference_distribution_from_values(values)
    result.pop("reference_floor")
    return result


def _nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


_ACTUAL_PROFILE_CHORUS_CADENCE_INPUTS = (
    (9, 25.809134238516712),
    (4, 11.916519656435916),
    (10, 29.495805653528365),
    (6, 18.933705533026185),
    (13, 38.622289783810324),
    (11, 31.24760389005099),
    (12, 36.33653711649983),
    (14, 54.83347015657381),
    (45, 54.899775343268956),
    (3, 8.733291253498138),
    (8, 20.670925027346737),
    (15, 30.748324427174193),
    (4, 7.608044066362048),
    (12, 28.078470190044857),
    (8, 18.985874443270276),
    (10, 28.927961054939985),
    (11, 21.420438393614155),
    (6, 18.201056422958985),
    (5, 9.692450497403534),
    (9, 33.29892498582262),
)

_CLOSE_DURATION_FRACTIONS = [0.20 + 0.01 * index for index in range(20)]
_NONWIDE_DURATION_FRACTIONS = [0.50 + 0.01 * index for index in range(20)]
_ROLE_DURATIONS = [
    duration for _shot_count, duration in _ACTUAL_PROFILE_CHORUS_CADENCE_INPUTS
]
_WIDE_RUN_DURATIONS = [
    duration * (1.0 - nonwide)
    for duration, nonwide in zip(
        _ROLE_DURATIONS,
        _NONWIDE_DURATION_FRACTIONS,
        strict=True,
    )
]


def _profile() -> dict[str, object]:
    close_duration_seconds = [
        duration * fraction
        for duration, fraction in zip(
            _ROLE_DURATIONS,
            _CLOSE_DURATION_FRACTIONS,
            strict=True,
        )
    ]
    nonwide_duration_seconds = [
        duration * fraction
        for duration, fraction in zip(
            _ROLE_DURATIONS,
            _NONWIDE_DURATION_FRACTIONS,
            strict=True,
        )
    ]
    medium_duration_seconds = [
        nonwide - close
        for nonwide, close in zip(
            nonwide_duration_seconds,
            close_duration_seconds,
            strict=True,
        )
    ]
    chorus = {
        "contributing_reference_count": 20,
        "pooled_counts": {
            "shot_count": sum(
                shot_count
                for shot_count, _duration in _ACTUAL_PROFILE_CHORUS_CADENCE_INPUTS
            ),
            "nonwide_multi_source_burst_count": 16,
            "closeup_multi_source_burst_count": 6,
            "consecutive_wide_run_count": 20,
        },
        "pooled_durations_seconds": {
            "role": sum(_ROLE_DURATIONS),
            "observed_shot_scale": sum(_ROLE_DURATIONS),
            "unknown_shot_scale": 0.0,
            "shot_size": {
                "wide": sum(_WIDE_RUN_DURATIONS),
                "medium": sum(medium_duration_seconds),
                "close": sum(close_duration_seconds),
            },
            "nonwide": sum(nonwide_duration_seconds),
        },
        "reference_distributions": {
            # Legacy, explicitly count-based audit evidence.  These values are
            # deliberately different from the duration distributions below.
            "shot_size_fractions": {
                "wide": _reference_distribution(0.2381818181818182),
                "medium": _reference_distribution(0.0),
                "close": _reference_distribution(0.15096153846153848),
            },
            "shot_size_duration_fractions": {
                "wide": _reference_distribution_from_values(
                    [1.0 - value for value in _NONWIDE_DURATION_FRACTIONS]
                ),
                "medium": _reference_distribution_from_values(
                    [
                        nonwide - close
                        for nonwide, close in zip(
                            _NONWIDE_DURATION_FRACTIONS,
                            _CLOSE_DURATION_FRACTIONS,
                            strict=True,
                        )
                    ]
                ),
                "close": _reference_distribution_from_values(
                    _CLOSE_DURATION_FRACTIONS
                ),
            },
            "nonwide_duration_fraction": _reference_distribution_from_values(
                _NONWIDE_DURATION_FRACTIONS
            ),
            "maximum_consecutive_wide_run_seconds": (
                _reference_distribution_from_values(_WIDE_RUN_DURATIONS)
            ),
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
        "wide_run_event_distributions": {
            "consecutive_wide_run_shot_count": _event_distribution_from_values(
                [1.0] * 20
            ),
            "consecutive_wide_run_duration_seconds": (
                _event_distribution_from_values(_WIDE_RUN_DURATIONS)
            ),
            "semantics": (
                "maximal_contiguous_known_wide_shots_within_one_"
                "acoustic_role_single_wide_shot_is_a_run"
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
    songs = []
    for song_index in range(4):
        references = []
        for reference_index in range(5):
            cadence_index = song_index * 5 + reference_index
            chorus_shot_count, chorus_duration = (
                _ACTUAL_PROFILE_CHORUS_CADENCE_INPUTS[cadence_index]
            )
            close_fraction = _CLOSE_DURATION_FRACTIONS[cadence_index]
            nonwide_fraction = _NONWIDE_DURATION_FRACTIONS[cadence_index]
            wide_fraction = 1.0 - nonwide_fraction
            medium_fraction = nonwide_fraction - close_fraction
            wide_duration = chorus_duration * wide_fraction
            medium_duration = chorus_duration * medium_fraction
            close_duration = chorus_duration * close_fraction
            references.append(
                {
                    "reference_id": f"REF-{song_index}-{reference_index}",
                    "by_acoustic_role": {
                        role: {
                            "role": role,
                            "shot_count": (
                                chorus_shot_count if role == "chorus_like" else 0
                            ),
                            "role_duration_seconds": (
                                chorus_duration if role == "chorus_like" else 0.0
                            ),
                            **(
                                {
                                    "observed_shot_scale_duration_seconds": (
                                        chorus_duration
                                    ),
                                    "unknown_shot_scale_duration_seconds": 0.0,
                                    "shot_size_duration_seconds": {
                                        "wide": wide_duration,
                                        "medium": medium_duration,
                                        "close": close_duration,
                                    },
                                    "shot_size_duration_fractions": {
                                        "wide": wide_fraction,
                                        "medium": medium_fraction,
                                        "close": close_fraction,
                                    },
                                    "nonwide_duration_seconds": (
                                        medium_duration + close_duration
                                    ),
                                    "nonwide_duration_fraction": (
                                        nonwide_fraction
                                    ),
                                    "consecutive_wide_run_count": 1,
                                    "consecutive_wide_run_shot_counts": [1],
                                    "consecutive_wide_run_durations_seconds": [
                                        wide_duration
                                    ],
                                    "maximum_consecutive_wide_run_seconds": (
                                        wide_duration
                                    ),
                                }
                                if role == "chorus_like"
                                else {}
                            ),
                        }
                        for role in ROLE_NAMES
                    },
                }
            )
        songs.append(
            {
                "song_id": f"IVE-SONG-{song_index}",
                "reference_count": 5,
                "source_count": 10,
                "references": references,
            }
        )
    profile: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "analysis_basis": PROFILE_ANALYSIS_BASIS,
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
            "shot_size_count_fractions_are_count_only_audit_metrics": True,
            "shot_size_duration_fractions_use_observed_duration_only": True,
            "nonwide_duration_is_explicit_medium_plus_close_never_one_minus_wide": True,
            "consecutive_wide_runs_are_maximal_contiguous_known_wide_shots_within_role": True,
            "single_wide_shot_is_a_consecutive_wide_run": True,
            "reference_maximum_wide_run_is_zero_when_no_wide_is_observed": True,
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
                "module_version": "2",
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


def _install_positive_duration_role(
    profile: dict[str, object],
    *,
    role_name: str,
    close_values: list[float],
    nonwide_values: list[float],
) -> None:
    if len(close_values) != 20 or len(nonwide_values) != 20:
        raise ValueError("role fixture needs twenty references")
    maximum_wide_runs: list[float] = []
    for index, reference in enumerate(
        ref
        for song in profile["songs"]  # type: ignore[index]
        for ref in song["references"]
    ):
        close = close_values[index]
        nonwide = nonwide_values[index]
        if not 0.0 <= close <= nonwide < 1.0:
            raise ValueError("invalid synthetic role fractions")
        duration = 10.0
        medium = nonwide - close
        wide = 1.0 - nonwide
        wide_duration = duration * wide
        maximum_wide_runs.append(wide_duration)
        reference["by_acoustic_role"][role_name] = {
            "role": role_name,
            "shot_count": 3,
            "role_duration_seconds": duration,
            "observed_shot_scale_duration_seconds": duration,
            "unknown_shot_scale_duration_seconds": 0.0,
            "shot_size_duration_seconds": {
                "wide": wide_duration,
                "medium": duration * medium,
                "close": duration * close,
            },
            "shot_size_duration_fractions": {
                "wide": wide,
                "medium": medium,
                "close": close,
            },
            "nonwide_duration_seconds": duration * nonwide,
            "nonwide_duration_fraction": nonwide,
            "consecutive_wide_run_count": 1,
            "consecutive_wide_run_shot_counts": [1],
            "consecutive_wide_run_durations_seconds": [wide_duration],
            "maximum_consecutive_wide_run_seconds": wide_duration,
        }
    chorus = profile["aggregate_by_acoustic_role"]["chorus_like"]  # type: ignore[index]
    aggregate = deepcopy(chorus)
    distributions = aggregate["reference_distributions"]
    distributions["shot_size_duration_fractions"] = {
        "wide": _reference_distribution_from_values(
            [1.0 - value for value in nonwide_values]
        ),
        "medium": _reference_distribution_from_values(
            [
                nonwide - close
                for nonwide, close in zip(
                    nonwide_values, close_values, strict=True
                )
            ]
        ),
        "close": _reference_distribution_from_values(close_values),
    }
    distributions["nonwide_duration_fraction"] = (
        _reference_distribution_from_values(nonwide_values)
    )
    distributions["maximum_consecutive_wide_run_seconds"] = (
        _reference_distribution_from_values(maximum_wide_runs)
    )
    aggregate["contributing_reference_count"] = 20
    profile["aggregate_by_acoustic_role"][role_name] = aggregate  # type: ignore[index]
    _rehash(profile)


class ProfessionalMontagePolicyTests(unittest.TestCase):
    def test_role_conditioned_duration_policy_omits_zero_p10_without_bypass(
        self,
    ) -> None:
        profile = _profile()
        build_close = [0.0, 0.0, 0.0] + [0.24] * 17
        build_nonwide = [0.0, 0.0, 0.0] + [0.54] * 17
        bridge_close = [0.0, 0.0, 0.0] + [0.32] * 17
        bridge_nonwide = [0.60] * 20
        _install_positive_duration_role(
            profile,
            role_name="build",
            close_values=build_close,
            nonwide_values=build_nonwide,
        )
        _install_positive_duration_role(
            profile,
            role_name="bridge_like",
            close_values=bridge_close,
            nonwide_values=bridge_nonwide,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "role-conditioned-profile.json"
            _write(path, profile)
            policy = load_professional_montage_policy(path)

        roles = {
            item.role: item
            for item in policy.role_conditioned_duration_policies
        }
        self.assertEqual(
            {"build", "chorus_like", "bridge_like"}, set(roles)
        )
        self.assertIsNone(
            roles["build"].local_minimum_close_duration_fraction
        )
        self.assertIsNone(
            roles["build"].local_minimum_nonwide_duration_fraction
        )
        self.assertIsNone(
            roles["bridge_like"].local_minimum_close_duration_fraction
        )
        self.assertAlmostEqual(
            _linear_quantile(bridge_nonwide, 0.10),
            roles["bridge_like"].local_minimum_nonwide_duration_fraction.value,
        )
        self.assertAlmostEqual(
            _linear_quantile(build_close, 0.25),
            roles[
                "build"
            ].pooled_product_minimum_close_duration_fraction.value,
        )
        self.assertAlmostEqual(
            _linear_quantile(bridge_close, 0.50),
            roles["bridge_like"].preferred_close_duration_fraction.value,
        )
        serialized = policy.to_dict()["role_conditioned_duration_policies"]
        self.assertIsNone(
            serialized["build"][
                "local_minimum_close_duration_fraction"
            ]
        )
        self.assertTrue(
            serialized["build"][
                "zero_local_p10_is_omitted_not_promoted"
            ]
        )

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
                    "analysis_basis": PROFILE_ANALYSIS_BASIS,
                    "schema_version": PROFILE_SCHEMA_VERSION,
                    "path": str(path),
                    "file_sha256": file_sha256,
                    "content_sha256": _profile()["provenance"][
                        "profile_content_sha256"
                    ],
                },
            )
            expected = {
                "minimum_close_duration_fraction": _linear_quantile(
                    _CLOSE_DURATION_FRACTIONS, 0.10
                ),
                "minimum_nonwide_duration_fraction": _linear_quantile(
                    _NONWIDE_DURATION_FRACTIONS, 0.10
                ),
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
                if "close_burst" in name:
                    self.assertEqual(6, threshold.reference_count)

            duration_targets = {
                "highlight_pooled_product_minimum_close_duration_fraction": (
                    _nearest_rank(
                        _CLOSE_DURATION_FRACTIONS,
                        DURATION_PRODUCT_MINIMUM_QUANTILE,
                    )
                ),
                "highlight_pooled_product_minimum_nonwide_duration_fraction": (
                    _nearest_rank(
                        _NONWIDE_DURATION_FRACTIONS,
                        DURATION_PRODUCT_MINIMUM_QUANTILE,
                    )
                ),
                "preferred_close_duration_fraction": _nearest_rank(
                    _CLOSE_DURATION_FRACTIONS,
                    DURATION_PREFERRED_TARGET_QUANTILE,
                ),
                "preferred_nonwide_duration_fraction": _nearest_rank(
                    _NONWIDE_DURATION_FRACTIONS,
                    DURATION_PREFERRED_TARGET_QUANTILE,
                ),
            }
            for name, expected_value in duration_targets.items():
                target = getattr(policy, name)
                self.assertAlmostEqual(target.value, expected_value)
                self.assertEqual(target.reference_count, 20)
                self.assertIn("duration", target.provenance.statistic)
                self.assertIsNone(target.provenance.reference_floor_basis)
                expected_quantile = (
                    DURATION_PRODUCT_MINIMUM_QUANTILE
                    if "product_minimum" in name
                    else DURATION_PREFERRED_TARGET_QUANTILE
                )
                self.assertEqual(target.provenance.quantile, expected_quantile)
            self.assertIn(
                "shot_size_duration_fractions.close",
                policy.minimum_close_duration_fraction.provenance.statistic_path,
            )
            self.assertNotIn(
                ".shot_size_fractions.close",
                policy.minimum_close_duration_fraction.provenance.statistic_path,
            )

            hard_wide_maximum = policy.maximum_consecutive_wide_run_seconds
            preferred_wide_maximum = (
                policy.preferred_maximum_consecutive_wide_run_seconds
            )
            self.assertAlmostEqual(
                hard_wide_maximum.value,
                _nearest_rank(
                    _WIDE_RUN_DURATIONS,
                    WIDE_RUN_HARD_MAXIMUM_QUANTILE,
                ),
            )
            self.assertAlmostEqual(
                preferred_wide_maximum.value,
                _nearest_rank(
                    _WIDE_RUN_DURATIONS,
                    WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE,
                ),
            )
            self.assertEqual(hard_wide_maximum.reference_count, 20)
            self.assertEqual(preferred_wide_maximum.reference_count, 20)
            self.assertEqual(hard_wide_maximum.provenance.quantile, 1.0)
            self.assertEqual(preferred_wide_maximum.provenance.quantile, 0.75)

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

            cadence_minimum = (
                policy.highlight_product_minimum_source_switches_per_minute
            )
            cadence_preferred = (
                policy.highlight_preferred_source_switches_per_minute
            )
            self.assertEqual(policy.highlight_cadence_reference_count, 20)
            self.assertEqual(cadence_minimum.reference_count, 20)
            self.assertEqual(
                cadence_minimum.value,
                18.642084765823732,
            )
            self.assertEqual(cadence_minimum.provenance.quantile, 0.5)
            self.assertIsNone(
                cadence_minimum.provenance.reference_floor_basis
            )
            self.assertIn(
                "derived_shot_boundary_rate_per_minute",
                cadence_minimum.provenance.statistic_path,
            )
            self.assertEqual(
                cadence_preferred.value,
                23.505554096533405,
            )
            self.assertEqual(cadence_preferred.reference_count, 20)
            self.assertEqual(cadence_preferred.provenance.quantile, 0.75)
            self.assertIn(
                "lower_median",
                cadence_minimum.provenance.statistic,
            )
            self.assertIn(
                "nearest_rank_p75",
                cadence_preferred.provenance.statistic,
            )
            serialized = policy.to_dict()
            serialized_cadence_minimum = serialized[
                "profile_informed_product_minimums"
            ]["highlight_product_minimum_source_switches_per_minute"]
            self.assertTrue(serialized_cadence_minimum["hard_gate"])
            self.assertFalse(
                serialized_cadence_minimum["is_reference_floor"]
            )
            serialized_cadence_preferred = serialized[
                "profile_derived_preferred_targets"
            ]["highlight_preferred_source_switches_per_minute"]
            self.assertFalse(serialized_cadence_preferred["hard_gate"])
            self.assertEqual(serialized["highlight_cadence_reference_count"], 20)

            self.assertEqual(
                policy.nonwide_duration_fraction_policy,
                "explicit_observed_duration_p10_never_derived_from_one_minus_wide",
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
                serialized["profile_informed_product_maximums"]
                ["maximum_consecutive_wide_run_seconds"]["hard_gate"]
            )
            self.assertEqual(
                serialized["profile_informed_product_maximums"]
                ["maximum_consecutive_wide_run_seconds"]["comparison"],
                "less_than_or_equal",
            )
            self.assertFalse(
                serialized["profile_derived_preferred_maximums"]
                ["preferred_maximum_consecutive_wide_run_seconds"]["hard_gate"]
            )
            self.assertTrue(
                serialized["count_based_audit_minimums"]
                ["never_used_as_duration_policy"]
            )
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
            self.assertEqual(product.minimum_nonwide_multi_family_burst_count, 2)
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
                policy,
                minimum_unique_sources=8,
                maximum_source_occupancy_fraction=0.45,
            )
            rendered_config = _rendered_montage_config_from_policy(policy)
            learned_decode_config = (
                _stage_edit_semimarkov_config_from_montage_policy(
                    policy,
                    continuity_authority=(
                        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
                    ),
                    continuity_scorer_contract=(
                        STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT
                    ),
                    continuity_scorer_fingerprint="a" * 64,
                )
            )
            self.assertEqual(
                decode_config.highlight_preferred_wide_involved_source_switch_fraction,
                0.8,
            )
            self.assertEqual(
                decode_config.highlight_wide_involved_source_switch_reference_count,
                17,
            )
            self.assertEqual(
                decode_config.highlight_product_minimum_source_switches_per_minute,
                18.642084765823732,
            )
            self.assertEqual(
                decode_config.highlight_preferred_source_switches_per_minute,
                23.505554096533405,
            )
            self.assertEqual(decode_config.highlight_cadence_reference_count, 20)
            self.assertEqual(
                decode_config.continuity_authority,
                HANDCRAFTED_CONTINUITY_AUTHORITY,
            )
            self.assertEqual(
                decode_config.continuity_scorer_contract,
                HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            )
            self.assertEqual(
                decode_config.continuity_scorer_fingerprint,
                HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
            )
            self.assertEqual(
                learned_decode_config.continuity_authority,
                PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
            )
            self.assertEqual(
                learned_decode_config.continuity_scorer_contract,
                STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
            )
            self.assertEqual(
                learned_decode_config.continuity_scorer_fingerprint,
                "a" * 64,
            )
            self.assertEqual(
                decode_config.highlight_minimum_nonwide_burst_boundaries,
                2,
            )
            self.assertEqual(
                decode_config.highlight_minimum_close_burst_shot_count,
                policy.minimum_close_burst_shot_count.value,
            )
            self.assertEqual(
                decode_config.highlight_minimum_close_burst_duration_seconds,
                policy.minimum_close_burst_duration_seconds.value,
            )
            self.assertEqual(
                decode_config.highlight_minimum_close_burst_source_switch_fraction,
                policy.minimum_close_burst_source_switch_fraction.value,
            )
            self.assertEqual(
                decode_config.highlight_close_burst_event_reference_count,
                6,
            )
            self.assertEqual(
                rendered_config.minimum_close_burst_shot_count,
                policy.minimum_close_burst_shot_count.value,
            )
            self.assertEqual(
                rendered_config.minimum_close_burst_duration_seconds,
                policy.minimum_close_burst_duration_seconds.value,
            )
            self.assertEqual(
                rendered_config.minimum_close_burst_source_switch_fraction,
                policy.minimum_close_burst_source_switch_fraction.value,
            )
            self.assertEqual(rendered_config.close_burst_event_reference_count, 6)
            self.assertEqual(
                rendered_config.close_burst_policy_binding_sha256,
                decode_config.highlight_close_burst_policy_binding_sha256,
            )
            self.assertEqual(decode_config.minimum_unique_sources, 8)
            self.assertEqual(
                decode_config.maximum_source_occupancy_fraction,
                0.45,
            )
            self.assertEqual(
                rendered_config.preferred_wide_involved_source_switch_fraction,
                0.8,
            )
            self.assertEqual(
                rendered_config.wide_involved_source_switch_reference_count,
                17,
            )
            self.assertEqual(
                decode_config.highlight_minimum_closeup_fraction,
                policy.minimum_close_duration_fraction.value,
            )
            self.assertEqual(
                decode_config.highlight_minimum_nonwide_fraction,
                policy.minimum_nonwide_duration_fraction.value,
            )
            self.assertEqual(
                decode_config.highlight_pooled_product_minimum_closeup_fraction,
                policy.highlight_pooled_product_minimum_close_duration_fraction.value,
            )
            self.assertEqual(
                decode_config.highlight_pooled_product_minimum_nonwide_fraction,
                policy.highlight_pooled_product_minimum_nonwide_duration_fraction.value,
            )
            self.assertEqual(
                decode_config.highlight_preferred_closeup_fraction,
                policy.preferred_close_duration_fraction.value,
            )
            self.assertEqual(
                decode_config.highlight_preferred_nonwide_fraction,
                policy.preferred_nonwide_duration_fraction.value,
            )
            self.assertEqual(
                decode_config.highlight_maximum_consecutive_wide_run_seconds,
                policy.maximum_consecutive_wide_run_seconds.value,
            )
            self.assertEqual(
                decode_config.highlight_preferred_maximum_consecutive_wide_run_seconds,
                policy.preferred_maximum_consecutive_wide_run_seconds.value,
            )
            self.assertEqual(
                rendered_config.minimum_closeup_fraction,
                decode_config.highlight_minimum_closeup_fraction,
            )
            self.assertEqual(
                rendered_config.minimum_nonwide_fraction,
                decode_config.highlight_minimum_nonwide_fraction,
            )
            self.assertEqual(
                rendered_config.pooled_product_minimum_closeup_fraction,
                decode_config.highlight_pooled_product_minimum_closeup_fraction,
            )
            self.assertEqual(
                rendered_config.pooled_product_minimum_nonwide_fraction,
                decode_config.highlight_pooled_product_minimum_nonwide_fraction,
            )
            self.assertEqual(
                rendered_config.preferred_closeup_fraction,
                decode_config.highlight_preferred_closeup_fraction,
            )
            self.assertEqual(
                rendered_config.preferred_nonwide_fraction,
                decode_config.highlight_preferred_nonwide_fraction,
            )
            self.assertEqual(
                rendered_config.maximum_consecutive_wide_run_seconds,
                decode_config.highlight_maximum_consecutive_wide_run_seconds,
            )
            self.assertEqual(
                rendered_config.preferred_maximum_consecutive_wide_run_seconds,
                decode_config.highlight_preferred_maximum_consecutive_wide_run_seconds,
            )
            self.assertEqual(
                rendered_config.policy_fingerprint_sha256,
                decode_config.highlight_policy_fingerprint_sha256,
            )
            self.assertRegex(
                str(rendered_config.policy_fingerprint_sha256),
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                rendered_config.policy_profile_analysis_basis,
                policy.identity.analysis_basis,
            )
            self.assertEqual(
                rendered_config.policy_profile_schema_version,
                policy.identity.schema_version,
            )
            self.assertEqual(
                rendered_config.policy_profile_path,
                policy.identity.path,
            )
            self.assertEqual(
                rendered_config.policy_profile_file_sha256,
                policy.identity.file_sha256,
            )
            self.assertEqual(
                rendered_config.policy_profile_content_sha256,
                policy.identity.content_sha256,
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
            ("schema", lambda value: value.__setitem__("schema_version", 1)),
            ("basis", lambda value: value.__setitem__("analysis_basis", "other")),
            (
                "generator",
                lambda value: value["provenance"]["generator"].__setitem__(  # type: ignore[index,union-attr]
                    "module_version", "1"
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
            ]["shot_size_duration_fractions"]["close"]
            close["reference_floor"]["value"] = 0.14
            _rehash(mismatched)
            mismatch_path = Path(temporary) / "mismatch.json"
            _write(mismatch_path, mismatched)
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "reference-floor provenance mismatch",
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

    def test_v1_and_count_as_duration_profiles_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = _profile()
            legacy["schema_version"] = 1
            legacy["analysis_basis"] = (
                "ive_human_reference_visual_montage_grammar_v1"
            )
            legacy["provenance"]["generator"]["module_version"] = "1"  # type: ignore[index]
            _rehash(legacy)
            legacy_path = root / "legacy-v1.json"
            _write(legacy_path, legacy)
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "unsupported profile schema version",
            ):
                load_professional_montage_policy(legacy_path)

            count_as_duration = _profile()
            chorus = count_as_duration["aggregate_by_acoustic_role"][  # type: ignore[index]
                "chorus_like"
            ]
            distributions = chorus["reference_distributions"]
            distributions["shot_size_duration_fractions"]["close"] = deepcopy(  # type: ignore[index]
                distributions["shot_size_fractions"]["close"]  # type: ignore[index]
            )
            _rehash(count_as_duration)
            misuse_path = root / "count-as-duration.json"
            _write(misuse_path, count_as_duration)
            with self.assertRaisesRegex(
                ProfessionalMontagePolicyError,
                "disagrees with duration evidence",
            ):
                load_professional_montage_policy(misuse_path)

    def test_explicit_nonwide_and_wide_run_evidence_fail_closed(self) -> None:
        mutations = (
            (
                "nonwide value",
                "explicit nonwide duration mismatch",
                lambda role: role.__setitem__(
                    "nonwide_duration_fraction",
                    1.0
                    - role["shot_size_duration_fractions"]["wide"]
                    + 0.01,
                ),
            ),
            (
                "wide maximum",
                "maximum consecutive wide-run mismatch",
                lambda role: role.__setitem__(
                    "maximum_consecutive_wide_run_seconds", 999.0
                ),
            ),
            (
                "wide run duration",
                "consecutive wide-run evidence mismatch",
                lambda role: role[
                    "consecutive_wide_run_durations_seconds"
                ].__setitem__(0, 0.01),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (name, error, mutate) in enumerate(mutations):
                with self.subTest(name=name):
                    profile = _profile()
                    role = profile["songs"][0]["references"][0][  # type: ignore[index]
                        "by_acoustic_role"
                    ]["chorus_like"]
                    mutate(role)
                    _rehash(profile)
                    path = Path(temporary) / f"duration-{index}.json"
                    _write(path, profile)
                    with self.assertRaisesRegex(
                        ProfessionalMontagePolicyError, error
                    ):
                        load_professional_montage_policy(path)

    def test_chorus_cadence_evidence_fails_closed_when_rehashed_but_invalid(
        self,
    ) -> None:
        def first_chorus(profile: dict[str, object]) -> dict[str, object]:
            songs = profile["songs"]
            assert isinstance(songs, list)
            song = songs[0]
            assert isinstance(song, dict)
            references = song["references"]
            assert isinstance(references, list)
            reference = references[0]
            assert isinstance(reference, dict)
            roles = reference["by_acoustic_role"]
            assert isinstance(roles, dict)
            chorus = roles["chorus_like"]
            assert isinstance(chorus, dict)
            return chorus

        mutations = (
            (
                "zero duration",
                "invalid duration coverage",
                lambda profile: first_chorus(profile).__setitem__(
                    "role_duration_seconds", 0.0
                ),
            ),
            (
                "fractional shot count",
                "shot_count must be an integer",
                lambda profile: first_chorus(profile).__setitem__(
                    "shot_count", 9.5
                ),
            ),
            (
                "wrong role label",
                "chorus_like role label mismatch",
                lambda profile: first_chorus(profile).__setitem__(
                    "role", "build"
                ),
            ),
            (
                "pooled mismatch",
                "pooled shot_count disagrees",
                lambda profile: profile["aggregate_by_acoustic_role"][  # type: ignore[index]
                    "chorus_like"
                ]["pooled_counts"].__setitem__("shot_count", 214),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (name, error, mutate) in enumerate(mutations):
                with self.subTest(name=name):
                    profile = _profile()
                    mutate(profile)
                    _rehash(profile)
                    path = Path(temporary) / f"cadence-{index}.json"
                    _write(path, profile)
                    with self.assertRaisesRegex(
                        ProfessionalMontagePolicyError,
                        error,
                    ):
                        load_professional_montage_policy(path)

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
