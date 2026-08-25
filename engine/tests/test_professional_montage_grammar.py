from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.professional_montage_grammar import (
    EditorialTransformEvidence,
    IntroducedBoundaryContinuityEvidence,
    MontageGrammarConfig,
    MontageGrammarEvidenceError,
    MontageSectionWindow,
    MontageSongInput,
    PROFILE_ANALYSIS_BASIS,
    PROFILE_SCHEMA_VERSION,
    ProfessionalBoundaryGrammarEvidence,
    ProfessionalBoundaryGrammarPolicy,
    ProfessionalMontageGrammarProfile,
    assess_professional_boundary_grammar,
    build_professional_montage_grammar_profile,
)
from crosscut.stage_edit_feature_materializer import (
    FEATURE_MATERIALIZER_CONTRACT,
    FEATURE_MATERIALIZER_SCHEMA_VERSION,
    StageEditFeatureProvenance,
    StageEditFeatureSnapshot,
)
from crosscut.stage_edit_feature_snapshot_job import (
    FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS,
    FEATURE_SNAPSHOT_JOB_INPUT_POLICY,
    FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditSequence,
    StageEditToken,
)
from engine.tests.stage_edit_causal_sensing_test_support import (
    synthetic_production_snapshot,
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


def _snapshot_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw


class _SyntheticSong:
    def __init__(
        self,
        root: Path,
        song_id: str,
        *,
        cut_clock_offset: bool = False,
        missing_first_scale: bool = False,
        mismatched_source_catalog: bool = False,
        mismatched_snapshot_schema: bool = False,
    ) -> None:
        self.root = root / song_id
        self.song_id = song_id
        self.reference_ids = ("REF_A", "REF_B")
        self.source_ids = ("A", "B", "C")
        source_catalog_ids = (
            ("A", "B", "D") if mismatched_source_catalog else self.source_ids
        )
        self.source_catalog = [
            {
                "video_id": source_id,
                "path": f"/synthetic/{song_id}/{source_id}.mp4",
                "clock": {},
                "provenance": {},
                "media_probe": {},
            }
            for source_id in source_catalog_ids
        ]
        self.supervisions: dict[str, Path] = {}
        self.snapshots: dict[str, Path] = {}
        self.snapshot_objects: dict[str, dict[str, object]] = {}
        supervision_records: list[dict[str, object]] = []
        snapshot_records: list[dict[str, object]] = []

        for reference_index, reference_id in enumerate(self.reference_ids):
            snapshot = self._snapshot(
                reference_id,
                missing_first_scale=(missing_first_scale and reference_index == 0),
            )
            snapshot_payload = snapshot.to_dict()
            if mismatched_snapshot_schema and reference_index == 0:
                snapshot_payload["token_schema_fingerprint"] = "0" * 64
                unsigned = deepcopy(snapshot_payload)
                unsigned.pop("snapshot_sha256")
                snapshot_payload["snapshot_sha256"] = _snapshot_sha256(unsigned)
            snapshot_hash = str(snapshot_payload["snapshot_sha256"])
            snapshot_path = self.root / (
                f"{reference_id}-stageedit-feature-snapshot-"
                f"{snapshot_hash[:16]}.json"
            )
            _write_json(snapshot_path, snapshot_payload)
            self.snapshots[reference_id] = snapshot_path
            self.snapshot_objects[reference_id] = snapshot_payload
            snapshot_records.append(
                {
                    "reference_id": reference_id,
                    "path": str(snapshot_path.resolve()),
                    "snapshot_sha256": snapshot_hash,
                    "token_count": 6,
                    "source_count": 3,
                    "performance_start_seconds": 0.0,
                    "performance_end_seconds": 6.0,
                }
            )

            supervision = self._supervision(
                reference_id,
                cut_clock_offset=(cut_clock_offset and reference_index == 0),
            )
            supervision_path = self.root / f"{reference_id}-supervision.json"
            supervision_raw = _write_json(supervision_path, supervision)
            supervision_sha = hashlib.sha256(supervision_raw).hexdigest()
            self.supervisions[reference_id] = supervision_path
            supervision_records.append(
                {
                    "reference_id": reference_id,
                    "path": str(supervision_path.resolve()),
                    "sha256": supervision_sha,
                }
            )

        contract = {
            "analysis_basis": "immutable_reference_training_materialization_v1",
            "style_id": f"{song_id}-style",
            "reference_dataset_schema_version": 4,
            "reference_supervision_schema_version": 3,
            "expected_reference_count": 2,
            "expected_source_count": 3,
            "inputs": [
                {
                    "reference_id": reference_id,
                    "source_count": 3,
                    "known_shot_count": 6,
                }
                for reference_id in self.reference_ids
            ],
        }
        fingerprint = _canonical_sha256(contract)
        outputs = [
            {
                "reference_id": item["reference_id"],
                "supervision_path": item["path"],
                "supervision_file_sha256": item["sha256"],
            }
            for item in supervision_records
        ]
        materialization = {
            "schema_version": 1,
            "analysis_basis": "immutable_reference_training_materialization_v1",
            "created_at": "2026-08-24T00:00:00+00:00",
            "input_fingerprint_sha256": fingerprint,
            "input_contract": contract,
            "outputs": outputs,
            "statistics": {
                "reference_count": 2,
                "known_shot_label_count": 12,
            },
            "immutable_inputs_modified": False,
        }
        self.materialization_path = self.root / (
            f"reference-training-materialization-{fingerprint[:16]}.json"
        )
        _write_json(self.materialization_path, materialization)

        source_content = [
            {
                "source_id": source_id,
                "path": f"/synthetic/{song_id}/{source_id}.mp4",
                "content_sha256": hashlib.sha256(source_id.encode()).hexdigest(),
            }
            for source_id in self.source_ids
        ]
        unsigned_manifest = {
            "schema_version": FEATURE_MATERIALIZER_SCHEMA_VERSION,
            "analysis_basis": FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS,
            "song_id": song_id,
            "input_policy": dict(FEATURE_SNAPSHOT_JOB_INPUT_POLICY),
            "sampling": {},
            "source_catalog_sha256": _canonical_sha256(self.source_catalog),
            "source_count": 3,
            "source_content": source_content,
            "supervision_inputs": supervision_records,
            "feature_provenance": {
                "producer": FEATURE_MATERIALIZER_CONTRACT,
                "contracts": list(snapshot.provenance.contracts),
            },
            "causal_sensing_binding": (
                snapshot.causal_sensing_binding.to_dict()
            ),
            "production_causal_sensing_gate": (
                snapshot.require_production_causal_sensing(edit_mode="group")
            ),
            "snapshots": snapshot_records,
            "statistics": {
                "reference_count": 2,
                "source_count": 3,
            },
        }
        manifest_sha = _canonical_sha256(unsigned_manifest)
        feature_manifest = {**unsigned_manifest, "manifest_sha256": manifest_sha}
        self.feature_path = self.root / (
            f"stageedit-feature-snapshot-job-manifest-{manifest_sha[:16]}.json"
        )
        _write_json(self.feature_path, feature_manifest)

    @property
    def input(self) -> MontageSongInput:
        return MontageSongInput(
            self.song_id,
            self.feature_path,
            self.materialization_path,
        )

    def _snapshot(
        self,
        reference_id: str,
        *,
        missing_first_scale: bool,
    ) -> StageEditFeatureSnapshot:
        roles = ("verse_like", "verse_like", "chorus_like", "chorus_like", "chorus_like", "chorus_like")
        section_progress = (0.0, 0.5, 0.0, 0.25, 0.50, 0.75)
        phrase_progress = (0.0, 0.7, 0.0, 0.4, 0.1, 0.6)
        selected_sources = ("A", "B", "C", "A", "B", "C")
        selected_scales = (0.10, 0.20, 0.35, 0.70, 0.80, 0.15)
        role_index = {
            "intro": 0,
            "verse_like": 1,
            "build": 2,
            "chorus_like": 3,
            "bridge_like": 4,
            "outro": 5,
        }
        scale_index = DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names.index(
            "shot_scale"
        )
        detected_index = DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names.index(
            "detected_person_count_normalized"
        )
        formation_index = DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names.index(
            "formation_person_count_normalized"
        )
        phrase_index = DEFAULT_STAGE_EDIT_SCHEMA.music_feature_names.index(
            "phrase_progress"
        )
        tokens: list[StageEditToken] = []
        for index in range(6):
            music = [0.0] * len(DEFAULT_STAGE_EDIT_SCHEMA.music_feature_names)
            music[phrase_index] = phrase_progress[index]
            section = [0.0] * len(DEFAULT_STAGE_EDIT_SCHEMA.section_feature_names)
            section[role_index[roles[index]]] = 0.8
            section[6] = section_progress[index]
            candidates: list[SourceCandidateToken] = []
            for source_id in self.source_ids:
                features = [0.0] * DEFAULT_STAGE_EDIT_SCHEMA.raw_source_dimension
                missing = [True] * DEFAULT_STAGE_EDIT_SCHEMA.raw_source_dimension
                features[scale_index] = (
                    selected_scales[index]
                    if source_id == selected_sources[index]
                    else 0.4
                )
                missing[scale_index] = False
                features[detected_index] = (
                    1.0 if features[scale_index] < 0.22 else 1.0 / 6.0
                )
                features[formation_index] = features[detected_index]
                missing[detected_index] = False
                missing[formation_index] = False
                if missing_first_scale and index == 0 and source_id == "A":
                    features[scale_index] = 0.0
                    missing[scale_index] = True
                candidates.append(
                    SourceCandidateToken(
                        source_id=source_id,
                        features=tuple(features),
                        missing_mask=tuple(missing),
                        available=True,
                        hard_gate_passed=True,
                    )
                )
            tokens.append(
                StageEditToken(
                    master_time_seconds=float(index),
                    music_features=tuple(music),
                    section_features=tuple(section),
                    source_candidates=tuple(candidates),
                    recent_edit_features=(0.0,)
                    * len(DEFAULT_STAGE_EDIT_SCHEMA.recent_edit_feature_names),
                    style_features=(0.0,)
                    * len(DEFAULT_STAGE_EDIT_SCHEMA.style_feature_names),
                    missing_mask=(False,)
                    * DEFAULT_STAGE_EDIT_SCHEMA.raw_context_dimension,
                )
            )
        return synthetic_production_snapshot(
            StageEditSequence(
                f"stageedit-features:{self.song_id}:{reference_id}",
                tuple(tokens),
            ),
            provenance=StageEditFeatureProvenance(
                contracts=FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS,
            ),
        )

    def _supervision(
        self,
        reference_id: str,
        *,
        cut_clock_offset: bool,
    ) -> dict[str, object]:
        selected_sources = ("A", "B", "C", "A", "B", "C")
        labels: list[dict[str, object]] = []
        for index, source_id in enumerate(selected_sources):
            start = float(index)
            if cut_clock_offset and index == 1:
                start += 0.001
            # Make the final wide chorus shot three seconds long.  The count
            # distribution therefore differs materially from the duration
            # distribution and catches count-as-duration regressions.
            end = 8.0 if index == 5 else float(index + 1)
            labels.append(
                {
                    "record_id": f"{reference_id}:{index}",
                    "style_id": f"{self.song_id}-style",
                    "reference_id": reference_id,
                    "shot_index": index,
                    "selected_source_id": source_id,
                    "selected_source_time": start,
                    "edit_labels": {
                        "cut": {
                            "boundary_label": "reference_start" if index == 0 else "cut",
                            "nominal_cut_master_time": start,
                            "cut_master_time": start,
                            "end_master_time": end,
                            "duration_seconds": end - start,
                        },
                        "source_transition": {},
                        "boundary_event": {},
                        "dense_attribution": {},
                        "crop": {},
                        "geometry_evidence": {},
                    },
                    "attribution_confidence": {},
                }
            )
        return {
            "schema_version": 3,
            "dataset_kind": "reference_stage_mix_pairwise_listwise_supervision",
            "style_id": f"{self.song_id}-style",
            "reference_id": reference_id,
            "reference_path": f"/synthetic/{reference_id}.mp4",
            "source_catalog": deepcopy(self.source_catalog),
            "base_reference_dataset_schema_version": 4,
            "canonical_shot_schema": {},
            "dense_attribution_schema": {},
            "crop_supervision_schema": {},
            "label_policy": {},
            "attribution_analysis": {},
            "reference_evidence_provenance": {},
            "shot_labels": labels,
            "detected_boundary_records": [],
            "detected_boundary_statistics": {},
            "pairwise_records": [],
            "listwise_records": [],
            "statistics": {},
        }


class ProfessionalMontageGrammarTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = MontageGrammarConfig(
            minimum_song_count=2,
            minimum_references_per_song=2,
            minimum_sources_per_song=3,
            early_anchor_shot_count=2,
            reference_floor_quantile=0.10,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _profile(self, **first_song_options: object) -> ProfessionalMontageGrammarProfile:
        first = _SyntheticSong(
            self.root,
            "IVE-SONG-A",
            **first_song_options,
        )
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        return build_professional_montage_grammar_profile(
            (first.input, second.input),
            config=self.config,
        )

    def test_measures_wide_switches_anchors_and_closeup_bursts(self) -> None:
        profile = self._profile()
        payload = profile.to_dict()
        self.assertEqual(2, PROFILE_SCHEMA_VERSION)
        self.assertEqual(
            "ive_human_reference_visual_montage_grammar_v2",
            PROFILE_ANALYSIS_BASIS,
        )
        self.assertEqual(PROFILE_SCHEMA_VERSION, payload["schema_version"])
        self.assertEqual(PROFILE_ANALYSIS_BASIS, payload["analysis_basis"])
        self.assertEqual(2, profile.song_count)
        self.assertEqual(4, profile.reference_count)
        verse = payload["aggregate_by_acoustic_role"]["verse_like"]
        chorus = payload["aggregate_by_acoustic_role"]["chorus_like"]
        self.assertEqual(
            1.0,
            verse["reference_distributions"]
            ["wide_to_wide_source_switch_fraction"]["median"],
        )
        self.assertEqual(
            1.0,
            verse["reference_distributions"]["entry_wide_anchor_fraction"][
                "median"
            ],
        )
        self.assertEqual(
            0.0,
            chorus["reference_distributions"]["entry_wide_anchor_fraction"][
                "median"
            ],
        )
        self.assertEqual(4, chorus["pooled_counts"]["short_closeup_burst_count"])
        self.assertEqual(
            2.0,
            chorus["burst_event_distributions"]["short_closeup_shot_count"][
                "median"
            ],
        )
        self.assertEqual(
            0.0,
            chorus["reference_distributions"]
            ["phrase_entry_wide_anchor_fraction"]["reference_floor"]["value"],
        )
        self.assertEqual(
            0.25,
            chorus["reference_distributions"]["shot_size_fractions"]["wide"]
            ["median"],
        )
        self.assertEqual(
            0.5,
            chorus["reference_distributions"]["shot_size_duration_fractions"]
            ["wide"]["median"],
        )
        self.assertEqual(
            0.5,
            chorus["reference_distributions"]["nonwide_duration_fraction"]
            ["median"],
        )
        self.assertEqual(
            3.0,
            chorus["reference_distributions"]
            ["maximum_consecutive_wide_run_seconds"]["median"],
        )
        self.assertEqual(
            3.0,
            chorus["wide_run_event_distributions"]
            ["consecutive_wide_run_duration_seconds"]["maximum"],
        )
        self.assertEqual(
            12.0,
            chorus["pooled_durations_seconds"]["shot_size"]["wide"],
        )
        self.assertEqual(
            12.0,
            chorus["pooled_durations_seconds"]["nonwide"],
        )

    def test_person_count_is_evidence_not_a_formation_assertion(self) -> None:
        payload = self._profile().to_dict()
        limitations = payload["evidence_limitations"]
        self.assertFalse(limitations["generic_wide_is_proven_group_formation"])
        evidence = payload["aggregate_by_acoustic_role"]["verse_like"][
            "wide_shot_person_count_evidence"
        ]
        self.assertEqual(8, evidence["detected_person_count_normalized"]["count"])
        self.assertEqual(
            "identity_free_observed_counts_only_no_group_formation_threshold",
            evidence["semantics"],
        )

    def test_missing_scale_remains_unknown_and_never_becomes_wide(self) -> None:
        payload = self._profile(missing_first_scale=True).to_dict()
        first = payload["songs"][0]["references"][0]
        verse = first["by_acoustic_role"]["verse_like"]
        self.assertEqual(1, verse["unknown_shot_scale_count"])
        self.assertEqual(1, verse["unknown_shot_scale_reasons"]["missing_mask"])
        self.assertEqual(1, verse["shot_size_counts"]["wide"])
        self.assertEqual(1.0, verse["shot_size_duration_seconds"]["wide"])
        self.assertEqual(
            1.0,
            verse["unknown_shot_scale_duration_seconds"],
        )
        self.assertEqual(
            1.0,
            verse["maximum_consecutive_wide_run_seconds"],
        )

    def test_duration_and_wide_run_tampering_fails_closed(self) -> None:
        mutations = (
            (
                "duration fraction",
                lambda metric: metric["shot_size_duration_fractions"].__setitem__(
                    "close", metric["shot_size_fractions"]["close"]
                ),
                "duration fraction mismatch",
            ),
            (
                "explicit nonwide",
                lambda metric: metric.__setitem__(
                    "nonwide_duration_fraction",
                    1.0 - metric["shot_size_fractions"]["wide"],
                ),
                "explicit nonwide duration fraction mismatch",
            ),
            (
                "wide run maximum",
                lambda metric: metric.__setitem__(
                    "maximum_consecutive_wide_run_seconds", 99.0
                ),
                "maximum consecutive wide-run duration mismatch",
            ),
        )
        for name, mutate, error in mutations:
            with self.subTest(name=name):
                payload = self._profile().to_dict()
                metric = payload["songs"][0]["references"][0][
                    "by_acoustic_role"
                ]["chorus_like"]
                mutate(metric)
                with self.assertRaisesRegex(MontageGrammarEvidenceError, error):
                    ProfessionalMontageGrammarProfile.from_mapping(payload)

    def test_round_trip_rebuilds_from_immutable_inputs(self) -> None:
        profile = self._profile()
        path = profile.write_json(self.root / "profile.json")
        restored = ProfessionalMontageGrammarProfile.from_file(path)
        self.assertEqual(profile.profile_content_sha256, restored.profile_content_sha256)
        self.assertIsNotNone(restored.file_sha256)

    def test_round_trip_detects_later_input_mutation(self) -> None:
        first = _SyntheticSong(self.root, "IVE-SONG-A")
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        profile = build_professional_montage_grammar_profile(
            (first.input, second.input),
            config=self.config,
        )
        path = profile.write_json(self.root / "profile.json")
        supervision = first.supervisions["REF_A"]
        supervision.write_bytes(supervision.read_bytes() + b" ")
        with self.assertRaisesRegex(MontageGrammarEvidenceError, "SHA-256 mismatch"):
            ProfessionalMontageGrammarProfile.from_file(path)
        unverified = ProfessionalMontageGrammarProfile.from_file(
            path,
            verify_input_hashes=False,
        )
        self.assertEqual(profile.profile_content_sha256, unverified.profile_content_sha256)

    def test_profile_hash_rejects_metric_tampering(self) -> None:
        payload = self._profile().to_dict()
        payload["songs"][0]["references"][0]["shot_count"] += 1
        with self.assertRaisesRegex(
            MontageGrammarEvidenceError,
            "shot count mismatch|known-gap coverage mismatch|content SHA-256 mismatch",
        ):
            ProfessionalMontageGrammarProfile.from_mapping(payload)

    def test_feature_manifest_hash_is_fail_closed(self) -> None:
        first = _SyntheticSong(self.root, "IVE-SONG-A")
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        payload = json.loads(first.feature_path.read_text(encoding="utf-8"))
        payload["statistics"]["reference_count"] = 999
        _write_json(first.feature_path, payload)
        with self.assertRaisesRegex(MontageGrammarEvidenceError, "content hash"):
            build_professional_montage_grammar_profile(
                (first.input, second.input),
                config=self.config,
            )

    def test_legacy_v2_feature_manifest_is_not_silently_accepted(self) -> None:
        first = _SyntheticSong(self.root, "IVE-SONG-A")
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        payload = json.loads(first.feature_path.read_text(encoding="utf-8"))
        payload["schema_version"] = 2
        payload["analysis_basis"] = (
            "synchronized_source_causal_stage_edit_feature_snapshot_v2"
        )
        payload.pop("manifest_sha256")
        digest = _canonical_sha256(payload)
        payload["manifest_sha256"] = digest
        first.feature_path = first.root / (
            f"stageedit-feature-snapshot-job-manifest-{digest[:16]}.json"
        )
        _write_json(first.feature_path, payload)

        with self.assertRaisesRegex(
            MontageGrammarEvidenceError,
            "feature manifest schema version mismatch",
        ):
            build_professional_montage_grammar_profile(
                (first.input, second.input),
                config=self.config,
            )

    def test_duplicate_snapshot_record_is_rejected_even_with_valid_hash(self) -> None:
        first = _SyntheticSong(self.root, "IVE-SONG-A")
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        payload = json.loads(first.feature_path.read_text(encoding="utf-8"))
        payload["snapshots"].append(deepcopy(payload["snapshots"][0]))
        payload.pop("manifest_sha256")
        digest = _canonical_sha256(payload)
        payload["manifest_sha256"] = digest
        first.feature_path = first.root / (
            f"stageedit-feature-snapshot-job-manifest-{digest[:16]}.json"
        )
        _write_json(first.feature_path, payload)
        with self.assertRaisesRegex(
            MontageGrammarEvidenceError,
            "snapshot reference set mismatch",
        ):
            build_professional_montage_grammar_profile(
                (first.input, second.input),
                config=self.config,
            )

    def test_materialization_and_feature_source_counts_must_match(self) -> None:
        first = _SyntheticSong(self.root, "IVE-SONG-A")
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        payload = json.loads(first.materialization_path.read_text(encoding="utf-8"))
        payload["input_contract"]["expected_source_count"] = 4
        fingerprint = _canonical_sha256(payload["input_contract"])
        payload["input_fingerprint_sha256"] = fingerprint
        first.materialization_path = first.root / (
            f"reference-training-materialization-{fingerprint[:16]}.json"
        )
        _write_json(first.materialization_path, payload)
        with self.assertRaisesRegex(
            MontageGrammarEvidenceError,
            "materialization/feature source count mismatch",
        ):
            build_professional_montage_grammar_profile(
                (first.input, second.input),
                config=self.config,
            )

    def test_cut_clock_requires_exact_stageedit_token(self) -> None:
        first = _SyntheticSong(
            self.root,
            "IVE-SONG-A",
            cut_clock_offset=True,
        )
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        with self.assertRaisesRegex(
            MontageGrammarEvidenceError,
            "no exact StageEdit token",
        ):
            build_professional_montage_grammar_profile(
                (first.input, second.input),
                config=self.config,
            )

    def test_source_catalog_mismatch_is_rejected(self) -> None:
        first = _SyntheticSong(
            self.root,
            "IVE-SONG-A",
            mismatched_source_catalog=True,
        )
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        with self.assertRaisesRegex(MontageGrammarEvidenceError, "source set mismatch"):
            build_professional_montage_grammar_profile(
                (first.input, second.input),
                config=self.config,
            )

    def test_snapshot_schema_fingerprint_mismatch_is_rejected(self) -> None:
        first = _SyntheticSong(
            self.root,
            "IVE-SONG-A",
            mismatched_snapshot_schema=True,
        )
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        with self.assertRaisesRegex(
            MontageGrammarEvidenceError,
            "token schema mismatch|snapshot is invalid",
        ):
            build_professional_montage_grammar_profile(
                (first.input, second.input),
                config=self.config,
            )

    def test_ive_only_and_multi_song_coverage_are_mandatory(self) -> None:
        only = _SyntheticSong(self.root, "IVE-SONG-A")
        with self.assertRaisesRegex(ValueError, "not enough songs"):
            build_professional_montage_grammar_profile(
                (only.input,),
                config=self.config,
            )
        second = _SyntheticSong(self.root, "IVE-SONG-B")
        with self.assertRaisesRegex(ValueError, "IVE-only"):
            build_professional_montage_grammar_profile(
                (only.input, second.input),
                artist="OTHER",
                config=self.config,
            )


class ProfessionalBoundaryGrammarAssessmentTests(unittest.TestCase):
    _SECTIONS = (
        MontageSectionWindow("general-a", "general", 0.0, 20.0),
        MontageSectionWindow("chorus-a", "chorus", 20.0, 40.0),
        MontageSectionWindow("highlight-a", "highlight", 40.0, 46.0),
        MontageSectionWindow("general-b", "general", 46.0, 60.0),
    )
    _SWITCHES = (
        2.0,
        6.0,
        10.0,
        14.0,
        18.0,
        20.5,
        23.5,
        26.5,
        29.5,
        32.5,
        35.5,
        38.5,
        40.5,
        42.5,
        44.5,
        48.0,
        52.0,
        56.0,
    )
    _NATIVE_CUTS = (
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.0,
        17.0,
        19.0,
        21.0,
        22.0,
        24.0,
        25.0,
        27.0,
        28.0,
        30.0,
        31.0,
        33.0,
        34.0,
        36.0,
        37.0,
        39.0,
        41.0,
        43.0,
        45.0,
        47.0,
        49.0,
        50.0,
        51.0,
        53.0,
        54.0,
        55.0,
        57.0,
        58.0,
        59.0,
    )

    def _transforms(
        self,
        *,
        roll: tuple[float, ...] = (42.0,),
    ) -> EditorialTransformEvidence:
        return EditorialTransformEvidence(
            segment_count=100,
            applied_segment_count=8,
            zoom_event_times_seconds=(3.0, 23.0, 32.0, 42.0, 52.0),
            pan_event_times_seconds=(6.0, 26.0, 44.0),
            roll_event_times_seconds=roll,
        )

    def _evidence(
        self,
        *,
        switches: tuple[float, ...] | None = None,
        native_cuts: tuple[float, ...] | None = None,
        sections: tuple[MontageSectionWindow, ...] | None = None,
        issue_count: int = 2,
        score_p10: float | None = 82.0,
        transforms: EditorialTransformEvidence | None = None,
    ) -> ProfessionalBoundaryGrammarEvidence:
        selected_switches = self._SWITCHES if switches is None else switches
        if not selected_switches:
            issue_count = 0
            score_p10 = None
        return ProfessionalBoundaryGrammarEvidence(
            performance_id="IVE-SYNTHETIC-BOUNDARY",
            duration_seconds=60.0,
            source_switch_times_seconds=selected_switches,
            native_internal_cut_times_seconds=(
                self._NATIVE_CUTS if native_cuts is None else native_cuts
            ),
            sections=self._SECTIONS if sections is None else sections,
            introduced_boundary_continuity=(
                IntroducedBoundaryContinuityEvidence.from_counts(
                    boundary_count=len(selected_switches),
                    issue_count=issue_count,
                    score_p10=score_p10,
                )
            ),
            editorial_transforms=transforms or self._transforms(),
        )

    def test_native_cuts_do_not_mask_cross_source_cadence_failure(self) -> None:
        native_only = tuple(float(index) for index in range(1, 60))
        assessment = assess_professional_boundary_grammar(
            self._evidence(
                switches=(),
                native_cuts=native_only,
                sections=(
                    MontageSectionWindow("ordinary", "general", 0.0, 60.0),
                ),
                transforms=self._transforms(roll=()),
            )
        )
        self.assertFalse(assessment.passed)
        self.assertIn(
            "whole_song_source_switch_density_below_minimum",
            assessment.failure_reasons,
        )
        self.assertEqual(0, assessment.source_switch_count)
        self.assertEqual(59, assessment.native_internal_cut_count)
        self.assertEqual(59, assessment.total_visible_cut_count)
        self.assertFalse(
            assessment.to_dict()["cadence"][
                "native_cuts_satisfy_source_switch_policy"
            ]
        )

    def test_highlight_burst_passes_and_sparse_highlight_fails(self) -> None:
        passing = assess_professional_boundary_grammar(self._evidence())
        self.assertTrue(passing.passed, passing.failure_reasons)
        highlight = next(
            item
            for item in passing.section_assessments
            if item.kind == "highlight"
        )
        self.assertEqual(30.0, highlight.source_switches_per_minute)
        self.assertEqual(2.0, highlight.maximum_no_switch_gap_seconds)

        sparse_switches = tuple(
            item for item in self._SWITCHES if not 40.0 <= item < 46.0
        )
        sparse = assess_professional_boundary_grammar(
            self._evidence(
                switches=sparse_switches,
                issue_count=1,
            )
        )
        self.assertFalse(sparse.passed)
        self.assertIn(
            "highlight_source_switch_density_below_minimum",
            sparse.failure_reasons,
        )
        self.assertIn(
            "highlight_no_switch_gap_above_maximum",
            sparse.failure_reasons,
        )

    def test_introduced_boundary_continuity_failure_is_independent(self) -> None:
        assessment = assess_professional_boundary_grammar(
            self._evidence(issue_count=9, score_p10=60.0)
        )
        self.assertFalse(assessment.passed)
        self.assertIn(
            "introduced_boundary_issue_fraction_above_maximum",
            assessment.failure_reasons,
        )
        self.assertIn(
            "introduced_boundary_score_p10_below_minimum",
            assessment.failure_reasons,
        )
        self.assertEqual(0.5, assessment.introduced_boundary_issue_fraction)

    def test_roll_abuse_and_roll_outside_highlight_or_chorus_fail(self) -> None:
        assessment = assess_professional_boundary_grammar(
            self._evidence(
                transforms=self._transforms(roll=(5.0, 10.0, 15.0, 42.0))
            )
        )
        self.assertFalse(assessment.passed)
        self.assertIn("roll_fraction_above_maximum", assessment.failure_reasons)
        self.assertIn("roll_outside_allowed_section", assessment.failure_reasons)

    def test_ordinary_sections_do_not_inherit_the_60_per_minute_exemplar(self) -> None:
        ordinary_switches = tuple(2.5 + 5.0 * index for index in range(12))
        assessment = assess_professional_boundary_grammar(
            self._evidence(
                switches=ordinary_switches,
                native_cuts=(),
                sections=(
                    MontageSectionWindow("ordinary", "general", 0.0, 60.0),
                ),
                issue_count=0,
                transforms=self._transforms(roll=()),
            )
        )
        self.assertTrue(assessment.passed, assessment.failure_reasons)
        ordinary = assessment.section_assessments[0]
        self.assertIsNone(ordinary.minimum_required_source_switches_per_minute)
        self.assertIsNone(ordinary.maximum_allowed_no_switch_gap_seconds)
        exemplar = assessment.to_dict()["policy"][
            "highlight_burst_exemplar"
        ]
        self.assertEqual(60.0, exemplar["cuts_per_minute"])
        self.assertFalse(exemplar["used_as_general_section_minimum"])
        self.assertFalse(exemplar["used_as_whole_song_minimum"])

    def test_transform_coverage_is_bounded(self) -> None:
        too_sparse = EditorialTransformEvidence(
            segment_count=100,
            applied_segment_count=1,
            zoom_event_times_seconds=(42.0,),
            pan_event_times_seconds=(),
            roll_event_times_seconds=(),
        )
        sparse = assess_professional_boundary_grammar(
            self._evidence(transforms=too_sparse)
        )
        self.assertIn(
            "editorial_transform_coverage_below_minimum",
            sparse.failure_reasons,
        )
        self.assertIn("zoom_pan_coverage_below_minimum", sparse.failure_reasons)

        too_dense = EditorialTransformEvidence(
            segment_count=100,
            applied_segment_count=50,
            zoom_event_times_seconds=tuple(float(index) for index in range(1, 51)),
            pan_event_times_seconds=(),
            roll_event_times_seconds=(),
        )
        dense = assess_professional_boundary_grammar(
            self._evidence(transforms=too_dense)
        )
        self.assertIn(
            "editorial_transform_coverage_above_maximum",
            dense.failure_reasons,
        )

    def test_transform_events_preserve_zero_clock_and_multiple_pulses(self) -> None:
        transforms = EditorialTransformEvidence(
            segment_count=100,
            applied_segment_count=8,
            # A learned trajectory may start transformed at output frame zero
            # and return to neutral before beginning another pulse in the same
            # segment.  Neither exact event may be discarded or merged.
            zoom_event_times_seconds=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
            pan_event_times_seconds=(),
            roll_event_times_seconds=(),
        )
        assessment = assess_professional_boundary_grammar(
            self._evidence(transforms=transforms)
        )
        self.assertTrue(assessment.passed, assessment.failure_reasons)
        self.assertEqual(
            transforms.zoom_event_times_seconds,
            ProfessionalBoundaryGrammarEvidence.from_mapping(
                self._evidence(transforms=transforms).to_dict()
            ).editorial_transforms.zoom_event_times_seconds,
        )

        with self.assertRaisesRegex(
            MontageGrammarEvidenceError,
            r"\[0, performance duration\)",
        ):
            self._evidence(
                transforms=EditorialTransformEvidence(
                    segment_count=100,
                    applied_segment_count=8,
                    zoom_event_times_seconds=(60.0,),
                )
            )

    def test_nonfinite_and_serialized_tampering_are_rejected(self) -> None:
        with self.assertRaisesRegex(MontageGrammarEvidenceError, "finite"):
            ProfessionalBoundaryGrammarPolicy(
                minimum_highlight_source_switches_per_minute=float("nan")
            )

        evidence = self._evidence()
        restored = ProfessionalBoundaryGrammarEvidence.from_mapping(
            evidence.to_dict()
        )
        self.assertEqual(evidence.evidence_sha256, restored.evidence_sha256)

        payload = evidence.to_dict()
        payload["duration_seconds"] = 61.0
        with self.assertRaisesRegex(
            MontageGrammarEvidenceError,
            "content hash mismatch",
        ):
            ProfessionalBoundaryGrammarEvidence.from_mapping(payload)

        payload = evidence.to_dict()
        payload["introduced_boundary_continuity"]["issue_fraction"] = 0.99
        unsigned = deepcopy(payload)
        unsigned.pop("evidence_sha256")
        payload["evidence_sha256"] = _canonical_sha256(unsigned)
        with self.assertRaisesRegex(
            MontageGrammarEvidenceError,
            "issue_fraction does not match",
        ):
            ProfessionalBoundaryGrammarEvidence.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
