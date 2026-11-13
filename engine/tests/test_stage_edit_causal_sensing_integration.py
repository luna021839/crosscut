from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.causal_shooting_grammar import (
    ShootingGrammarObservation,
    ShootingGrammarProvenance,
    build_causal_shooting_grammar_artifact,
    write_causal_shooting_grammar_artifact,
)
from crosscut.ive import IVEMember
from crosscut.ive_performer_state import (
    IVEPerformerStateProvenance,
    IVEPerformerStateSourceProvenance,
    build_ive_performer_state_artifact,
    write_ive_performer_state_artifact,
)
from crosscut.member_state_graph import (
    CausalMemberObservation,
    MemberProbabilityMass,
    MemberStateArtifactProvenance,
    MemberStateSourceProvenance,
    build_member_state_graph,
    write_member_state_graph_artifact,
)
from crosscut.stage_edit_feature_materializer import (
    FEATURE_MATERIALIZER_SCHEMA_VERSION,
    FEATURE_SNAPSHOT_LEGACY_SCHEMA_VERSION,
    STAGE_EDIT_MEMBER_FEATURE_MODE_GROUP_NO_TARGET,
    STAGE_EDIT_MEMBER_FEATURE_MODE_TARGET_MEMBER,
    SourceFeatureObservation,
    StageEditCausalSensingArtifacts,
    StageEditCausalSensingBinding,
    StageEditFeatureFrame,
    StageEditFeatureProvenance,
    StageEditFeatureSnapshot,
    materialize_stage_edit_feature_snapshot,
    materialize_stage_edit_token,
)
from crosscut.stage_edit_feature_snapshot_job import (
    build_stage_edit_feature_snapshots,
)
from crosscut.stage_edit_inference_snapshot import (
    materialize_stage_edit_inference_snapshot,
)
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES,
    STAGE_EDIT_TARGET_MEMBER_SOURCE_FEATURE_NAMES,
    TOKEN_SCHEMA_VERSION,
)


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _member_mass() -> MemberProbabilityMass:
    values = {member: 0.0 for member in IVEMember}
    values[IVEMember.GAEUL] = 0.9
    return MemberProbabilityMass(values, unknown_mass=0.1)


def _member_observation(source_id: str, order: int) -> CausalMemberObservation:
    return CausalMemberObservation(
        observation_id=f"member-{source_id}",
        source_id=source_id,
        shot_id=f"shot-{source_id}",
        timestamp_seconds=0.5,
        frame_index=15,
        causal_order=order,
        member_probabilities=_member_mass(),
        face_bbox=(0.42, 0.20, 0.58, 0.38),
        body_bbox=(0.30, 0.10, 0.70, 0.95),
        face_identity_id=f"face-{source_id}",
        tracker_tracklet_id=f"track-{source_id}",
        occlusion_probability=0.0,
        geometry_confidence=1.0,
        shot_confidence=1.0,
    )


def _grammar_observation(
    source_id: str,
    timestamp: float,
) -> ShootingGrammarObservation:
    return ShootingGrammarObservation(
        source_id=source_id,
        master_time_seconds=timestamp,
        shot_closeup_probability=0.6,
        shot_medium_probability=0.3,
        shot_wide_probability=0.1,
        camera_pan_velocity=0.2,
        camera_tilt_velocity=-0.1,
        camera_zoom_in_probability=0.7,
        camera_zoom_out_probability=0.1,
        camera_static_probability=0.2,
        camera_roll_degrees_normalized=0.1,
        subject_headroom_normalized=0.2,
        subject_lead_room_normalized=0.3,
        detected_person_count=6,
        formation_person_count=4,
        internal_native_cut_since_previous=(timestamp == 0.5),
        shooting_grammar_confidence=0.9,
    )


def _source(
    source_id: str,
    *,
    roll: float = 0.1,
    detected_person_count_normalized: float = 1.0,
    formation_pose_coverage: float = 4.0 / 6.0,
) -> SourceFeatureObservation:
    return SourceFeatureObservation(
        source_id=source_id,
        camera_roll_degrees_normalized=roll,
        detected_person_count_normalized=detected_person_count_normalized,
        formation_pose_coverage=formation_pose_coverage,
    )


class StageEditCausalSensingIntegrationTests(unittest.TestCase):
    def _artifacts(
        self,
        root: Path,
        *,
        grammar_times: tuple[float, ...] = (0.5, 1.0),
        target_member: IVEMember | None = IVEMember.GAEUL,
    ) -> StageEditCausalSensingArtifacts:
        source_ids = ("A", "B")
        member_graph = build_member_state_graph(
            tuple(
                _member_observation(source_id, index)
                for index, source_id in enumerate(source_ids)
            )
        )
        member_provenance = MemberStateArtifactProvenance(
            producer_id="stageedit-integration-test",
            producer_version="1",
            created_at_utc="2026-08-25T00:00:00Z",
            sources=tuple(
                MemberStateSourceProvenance(
                    source_id=source_id,
                    source_content_sha256=marker * 64,
                    observation_manifest_sha256=marker.upper().casefold() * 64,
                )
                for source_id, marker in zip(source_ids, "ab", strict=True)
            ),
        )
        member_artifact = write_member_state_graph_artifact(
            root / "member",
            member_graph,
            member_provenance,
        )
        grammar = build_causal_shooting_grammar_artifact(
            tuple(
                _grammar_observation(source_id, timestamp)
                for timestamp in grammar_times
                for source_id in source_ids
            ),
            provenance=ShootingGrammarProvenance(
                upstream_sha256=(("source_catalog", "c" * 64),),
                analyzer_fingerprint_sha256="d" * 64,
                contracts=("same_causal_snapshot_test_v1",),
            ),
        )
        grammar_path = write_causal_shooting_grammar_artifact(
            grammar,
            root / "grammar",
        )
        performer = build_ive_performer_state_artifact(
            member_graph,
            grammar.records,
            provenance=IVEPerformerStateProvenance(
                analyzer_fingerprint_sha256="e" * 64,
                sources=tuple(
                    IVEPerformerStateSourceProvenance(
                        source_id=source_id,
                        source_content_sha256=marker * 64,
                    )
                    for source_id, marker in zip(source_ids, "ab", strict=True)
                ),
                contracts=("member_and_grammar_scalar_v1",),
            ),
        )
        performer_path = write_ive_performer_state_artifact(
            root / "performer",
            performer,
        )
        return StageEditCausalSensingArtifacts(
            shooting_grammar_path=grammar_path,
            shooting_grammar_content_sha256=(
                grammar.artifact_content_sha256
            ),
            member_state_graph_path=member_artifact.path,
            member_state_graph_content_sha256=member_artifact.content_sha256,
            target_member=target_member,
            ive_performer_state_path=performer_path,
            ive_performer_state_content_sha256=performer.content_sha256,
        )

    def test_schema_v3_has_one_copy_of_overlap_and_new_columns(self) -> None:
        self.assertEqual(3, TOKEN_SCHEMA_VERSION)
        self.assertEqual(3, DEFAULT_STAGE_EDIT_SCHEMA.schema_version)
        names = DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names
        for name in (
            "camera_roll_degrees_normalized",
            "detected_person_count_normalized",
            "formation_pose_coverage",
        ):
            self.assertEqual(1, names.count(name))
        for name in (
            "shot_closeup_probability",
            "native_cut_proximity",
            "shooting_grammar_confidence",
            "target_member_probability",
            "target_member_uncertainty_probability",
            "target_member_state_observed",
            "ive_center_member_unknown_probability",
            "ive_active_performer_unknown_probability",
            "ive_center_member_known_mass",
        ):
            self.assertIn(name, names)

    def test_bound_artifacts_materialize_and_round_trip_exact_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            frames = (
                StageEditFeatureFrame(0.5, (_source("A"), _source("B"))),
                StageEditFeatureFrame(1.0, (_source("A"), _source("B"))),
            )
            provenance = StageEditFeatureProvenance(
                upstream_sha256=(("source_alignment", "e" * 64),),
                contracts=("shared_train_inference_test_v1",),
            )
            snapshot = materialize_stage_edit_feature_snapshot(
                "IVE:sensing:test",
                frames,
                performance_end_seconds=2.0,
                provenance=provenance,
                causal_sensing=artifacts,
                production_edit_mode="member",
                production_target_member=IVEMember.GAEUL,
            )
            self.assertEqual(FEATURE_MATERIALIZER_SCHEMA_VERSION, snapshot.schema_version)
            self.assertEqual(
                artifacts.binding,
                snapshot.causal_sensing_binding,
            )
            upstream = dict(snapshot.provenance.upstream_sha256)
            self.assertEqual(
                artifacts.binding.shooting_grammar_content_sha256,
                upstream["causal_shooting_grammar"],
            )
            self.assertEqual(
                artifacts.binding.member_state_graph_content_sha256,
                upstream["member_state_graph"],
            )
            self.assertEqual(
                artifacts.binding.ive_performer_state_content_sha256,
                upstream["ive_performer_state"],
            )

            indexes = {
                name: index
                for index, name in enumerate(
                    DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names
                )
            }
            observed = snapshot.sequence.tokens[0].source_candidates[0]
            self.assertFalse(
                observed.missing_mask[indexes["target_member_probability"]]
            )
            self.assertAlmostEqual(
                0.9,
                observed.features[indexes["target_member_probability"]],
            )
            self.assertEqual(
                1.0,
                observed.features[indexes["target_member_state_observed"]],
            )
            missing = snapshot.sequence.tokens[1].source_candidates[0]
            self.assertEqual(
                0.0,
                missing.features[indexes["target_member_probability"]],
            )
            self.assertEqual(
                1.0,
                missing.features[
                    indexes["target_member_uncertainty_probability"]
                ],
            )
            self.assertEqual(
                0.0,
                missing.features[indexes["target_member_state_observed"]],
            )
            self.assertTrue(missing.missing_mask[indexes["target_center_x"]])
            self.assertTrue(missing.missing_mask[indexes["target_center_y"]])
            self.assertFalse(
                missing.missing_mask[
                    indexes["ive_center_member_unknown_probability"]
                ]
            )
            self.assertEqual(
                1.0,
                missing.features[
                    indexes["ive_center_member_unknown_probability"]
                ],
            )
            self.assertEqual(
                1.0,
                missing.features[
                    indexes["ive_active_performer_unknown_probability"]
                ],
            )
            self.assertTrue(
                missing.missing_mask[indexes["ive_vocal_probability"]]
            )
            restored = StageEditFeatureSnapshot.from_dict(snapshot.to_dict())
            self.assertEqual(snapshot, restored)

            direct = materialize_stage_edit_token(
                master_time_seconds=0.5,
                source_observations=frames[0].source_observations,
                performance_end_seconds=2.0,
                causal_sensing=artifacts,
            )
            self.assertEqual(snapshot.sequence.tokens[0], direct)

    def test_group_mode_uses_all_member_performer_state_without_target_bias(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(
                Path(temporary),
                target_member=None,
            )
            snapshot = materialize_stage_edit_feature_snapshot(
                "IVE:group-no-target",
                (StageEditFeatureFrame(0.5, (_source("A"), _source("B"))),),
                performance_end_seconds=1.0,
                causal_sensing=artifacts,
                production_edit_mode="group",
            )
            binding = snapshot.causal_sensing_binding
            self.assertEqual(
                STAGE_EDIT_MEMBER_FEATURE_MODE_GROUP_NO_TARGET,
                binding.member_feature_mode,
            )
            self.assertIsNone(binding.target_member)
            indexes = {
                name: index
                for index, name in enumerate(
                    DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names
                )
            }
            for candidate in snapshot.sequence.tokens[0].source_candidates:
                for name in STAGE_EDIT_TARGET_MEMBER_SOURCE_FEATURE_NAMES:
                    index = indexes[name]
                    self.assertTrue(candidate.missing_mask[index])
                    self.assertEqual(0.0, candidate.features[index])
                for name in STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES:
                    if name == "ive_vocal_probability":
                        continue
                    self.assertFalse(candidate.missing_mask[indexes[name]])

            payload = binding.to_dict()
            payload["member_state_graph"]["member_feature_mode"] = (
                STAGE_EDIT_MEMBER_FEATURE_MODE_TARGET_MEMBER
            )
            with self.assertRaisesRegex(ValueError, "member feature mode"):
                StageEditCausalSensingBinding.from_dict(payload)

    def test_equivalent_construction_order_has_identical_snapshot_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            fresh = StageEditCausalSensingArtifacts(
                shooting_grammar_path=artifacts.shooting_grammar_path,
                shooting_grammar_content_sha256=(
                    artifacts.shooting_grammar_content_sha256
                ),
                member_state_graph_path=artifacts.member_state_graph_path,
                member_state_graph_content_sha256=(
                    artifacts.member_state_graph_content_sha256
                ),
                target_member=IVEMember.GAEUL,
                ive_performer_state_path=artifacts.ive_performer_state_path,
                ive_performer_state_content_sha256=(
                    artifacts.ive_performer_state_content_sha256
                ),
            )
            forward = materialize_stage_edit_feature_snapshot(
                "IVE:deterministic",
                (StageEditFeatureFrame(0.5, (_source("A"), _source("B"))),),
                performance_end_seconds=1.0,
                provenance=StageEditFeatureProvenance(
                    upstream_sha256=(
                        ("source_alignment", "1" * 64),
                        ("visual_score", "2" * 64),
                    ),
                    contracts=("z_contract", "a_contract"),
                ),
                causal_sensing=artifacts,
                production_edit_mode="member",
                production_target_member=IVEMember.GAEUL,
            )
            reverse = materialize_stage_edit_feature_snapshot(
                "IVE:deterministic",
                (StageEditFeatureFrame(0.5, (_source("B"), _source("A"))),),
                performance_end_seconds=1.0,
                provenance=StageEditFeatureProvenance(
                    upstream_sha256=(
                        ("visual_score", "2" * 64),
                        ("source_alignment", "1" * 64),
                    ),
                    contracts=("a_contract", "z_contract"),
                ),
                causal_sensing=fresh,
                production_edit_mode="member",
                production_target_member=IVEMember.GAEUL,
            )
            self.assertEqual(forward.to_dict(), reverse.to_dict())
            self.assertEqual(forward.snapshot_sha256, reverse.snapshot_sha256)

    def test_exact_timestamp_and_typed_overlap_policy_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = self._artifacts(root, grammar_times=(0.5,))
            with self.assertRaisesRegex(ValueError, "exact.*source/timestamp"):
                materialize_stage_edit_feature_snapshot(
                    "IVE:missing-time",
                    (StageEditFeatureFrame(1.0, (_source("A"), _source("B"))),),
                    performance_end_seconds=2.0,
                    causal_sensing=artifacts,
                    production_edit_mode="member",
                    production_target_member=IVEMember.GAEUL,
                )

            complete = self._artifacts(root / "complete")
            authoritative = materialize_stage_edit_feature_snapshot(
                "IVE:authoritative-overlaps",
                (
                    StageEditFeatureFrame(
                        0.5,
                        (
                            _source(
                                "A",
                                roll=-0.7,
                                formation_pose_coverage=0.5,
                            ),
                            _source("B"),
                        ),
                    ),
                ),
                performance_end_seconds=2.0,
                causal_sensing=complete,
                production_edit_mode="member",
                production_target_member=IVEMember.GAEUL,
            )
            indexes = {
                name: index
                for index, name in enumerate(
                    DEFAULT_STAGE_EDIT_SCHEMA.source_feature_names
                )
            }
            candidate_a = next(
                item
                for item in authoritative.sequence.tokens[0].source_candidates
                if item.source_id == "A"
            )
            self.assertEqual(
                -0.7,
                candidate_a.features[
                    indexes["camera_roll_degrees_normalized"]
                ],
            )
            self.assertEqual(
                0.5,
                candidate_a.features[indexes["formation_pose_coverage"]],
            )

            composite_count_homonym = materialize_stage_edit_feature_snapshot(
                "IVE:composite-person-count-homonym",
                (
                    StageEditFeatureFrame(
                        0.5,
                        (
                            _source(
                                "A",
                                detected_person_count_normalized=0.5,
                            ),
                            _source("B"),
                        ),
                    ),
                ),
                performance_end_seconds=2.0,
                causal_sensing=complete,
                production_edit_mode="member",
                production_target_member=IVEMember.GAEUL,
            )
            composite_candidate_a = next(
                item
                for item in composite_count_homonym.sequence.tokens[
                    0
                ].source_candidates
                if item.source_id == "A"
            )
            self.assertEqual(
                0.5,
                composite_candidate_a.features[
                    indexes["detected_person_count_normalized"]
                ],
            )
            self.assertEqual(
                [],
                composite_count_homonym.causal_sensing_binding.to_dict()[
                    "known_overlap_policy"
                ]["exact_same_snapshot_corroboration"],
            )
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                materialize_stage_edit_feature_snapshot(
                    "IVE:source-pool-conflict",
                    (StageEditFeatureFrame(0.5, (_source("A"),)),),
                    performance_end_seconds=2.0,
                    causal_sensing=complete,
                    production_edit_mode="member",
                    production_target_member=IVEMember.GAEUL,
                )

    def test_expected_artifact_hash_and_legacy_snapshot_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            with self.assertRaisesRegex(ValueError, "expected content hash mismatch"):
                StageEditCausalSensingArtifacts(
                    shooting_grammar_path=artifacts.shooting_grammar_path,
                    shooting_grammar_content_sha256="0" * 64,
                )

            snapshot = materialize_stage_edit_feature_snapshot(
                "IVE:legacy-reject",
                (StageEditFeatureFrame(0.5, (_source("A"), _source("B"))),),
                performance_end_seconds=1.0,
                inspection_only_without_causal_sensing=True,
            )
            legacy = deepcopy(snapshot.to_dict())
            legacy["schema_version"] = FEATURE_SNAPSHOT_LEGACY_SCHEMA_VERSION
            with self.assertRaisesRegex(ValueError, "inspection-only"):
                StageEditFeatureSnapshot.from_dict(legacy)

            tampered = snapshot.to_dict()
            tampered["causal_sensing_binding"]["shooting_grammar"] = {
                "artifact_content_sha256": "f" * 64,
                "artifact_adapter": "forged",
                "feature_names": [],
            }
            unsigned = deepcopy(tampered)
            unsigned.pop("snapshot_sha256")
            tampered["snapshot_sha256"] = _sha256_payload(unsigned)
            with self.assertRaisesRegex(ValueError, "adapter binding changed"):
                StageEditFeatureSnapshot.from_dict(tampered)

    def test_production_gate_rejects_empty_partial_and_wrong_member_mode(self) -> None:
        frame = StageEditFeatureFrame(0.5, (_source("A"), _source("B")))
        with self.assertRaisesRegex(ValueError, "complete causal-sensing triple"):
            materialize_stage_edit_feature_snapshot(
                "IVE:empty-default-rejected",
                (frame,),
                performance_end_seconds=1.0,
            )
        inspection = materialize_stage_edit_feature_snapshot(
            "IVE:empty-inspection",
            (frame,),
            performance_end_seconds=1.0,
            inspection_only_without_causal_sensing=True,
        )
        with self.assertRaisesRegex(ValueError, "performer-state triple"):
            inspection.require_production_causal_sensing(edit_mode="group")

        partial_binding = StageEditCausalSensingBinding(
            shooting_grammar_content_sha256="1" * 64,
        )
        partial = StageEditFeatureSnapshot.build(
            inspection.sequence,
            provenance=StageEditFeatureProvenance(
                upstream_sha256=(("causal_shooting_grammar", "1" * 64),),
            ),
            causal_sensing_binding=partial_binding,
            inspection_only_without_causal_sensing=True,
        )
        with self.assertRaisesRegex(ValueError, "performer-state triple"):
            partial.require_production_causal_sensing(edit_mode="group")

        group_binding = StageEditCausalSensingBinding(
            shooting_grammar_content_sha256="1" * 64,
            member_state_graph_content_sha256="2" * 64,
            ive_performer_state_content_sha256="3" * 64,
        )
        group_gate = group_binding.require_production_scope(edit_mode="group")
        self.assertEqual("group_no_target", group_gate["member_feature_mode"])
        with self.assertRaisesRegex(ValueError, "does not match MEMBER"):
            group_binding.require_production_scope(
                edit_mode="member",
                target_member=IVEMember.GAEUL,
            )

        member_binding = StageEditCausalSensingBinding(
            shooting_grammar_content_sha256="1" * 64,
            member_state_graph_content_sha256="2" * 64,
            ive_performer_state_content_sha256="3" * 64,
            target_member=IVEMember.GAEUL.value,
        )
        with self.assertRaisesRegex(ValueError, "does not match GROUP"):
            member_binding.require_production_scope(edit_mode="group")
        with self.assertRaisesRegex(ValueError, "does not match the request"):
            member_binding.require_production_scope(
                edit_mode="member",
                target_member=IVEMember.REI,
            )

    def test_training_and_inference_entrypoints_share_typed_sensing_input(self) -> None:
        for entrypoint in (
            build_stage_edit_feature_snapshots,
            materialize_stage_edit_inference_snapshot,
        ):
            parameter = inspect.signature(entrypoint).parameters["causal_sensing"]
            self.assertIsNone(parameter.default)


if __name__ == "__main__":
    unittest.main()
