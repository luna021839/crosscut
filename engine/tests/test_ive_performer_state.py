from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from crosscut.causal_shooting_grammar import CausalShootingGrammarRecord
from crosscut.ive import IVEMember
from crosscut.ive_performer_state import (
    CENTER_DISTRIBUTION,
    LEGACY_CENTER_DISTRIBUTION,
    IVEPerformerStateArtifact,
    _artifact_content_sha256,
    CausalPerformerEvidence,
    IVEPerformerStateArtifactError,
    IVEPerformerStateError,
    IVEPerformerStateProvenance,
    IVEPerformerStateSourceProvenance,
    ObservationActivityEvidence,
    STAGE_EDIT_IVE_PERFORMER_STATE_ADAPTER,
    STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES,
    adapt_ive_performer_state_for_stage_edit,
    build_ive_performer_state_artifact,
    build_ive_performer_state_input_binding,
    derive_ive_performer_state_records,
    load_ive_performer_state_artifact,
    write_ive_performer_state_artifact,
)
from crosscut.member_state_graph import (
    CausalMemberObservation,
    MemberProbabilityMass,
    build_member_state_graph,
)


def _mass(
    member: IVEMember,
    *,
    known: float = 0.96,
    unknown: float = 0.04,
) -> MemberProbabilityMass:
    probabilities = {candidate: 0.0 for candidate in IVEMember}
    probabilities[member] = known
    remainder = 1.0 - known - unknown
    if remainder > 0.0:
        fallback = next(candidate for candidate in IVEMember if candidate is not member)
        probabilities[fallback] = remainder
    return MemberProbabilityMass(probabilities, unknown_mass=unknown)


def _observation(
    observation_id: str,
    *,
    member: IVEMember = IVEMember.GAEUL,
    timestamp: float = 0.0,
    order: int = 0,
    center_x: float = 0.5,
    width: float = 0.24,
    occlusion: float = 0.0,
    geometry_confidence: float = 1.0,
    shot_confidence: float = 0.98,
    source_id: str = "source-a",
) -> CausalMemberObservation:
    return CausalMemberObservation(
        observation_id=observation_id,
        source_id=source_id,
        shot_id="shot-1",
        timestamp_seconds=timestamp,
        frame_index=round(timestamp * 30.0),
        causal_order=order,
        member_probabilities=_mass(member),
        body_bbox=(
            center_x - width / 2.0,
            0.08,
            center_x + width / 2.0,
            0.92,
        ),
        occlusion_probability=occlusion,
        geometry_confidence=geometry_confidence,
        shot_confidence=shot_confidence,
    )


def _missing_observation(
    observation_id: str,
    *,
    timestamp: float,
    order: int,
) -> CausalMemberObservation:
    return CausalMemberObservation(
        observation_id=observation_id,
        source_id="source-a",
        shot_id="shot-1",
        timestamp_seconds=timestamp,
        frame_index=round(timestamp * 30.0),
        causal_order=order,
        member_probabilities=MemberProbabilityMass.unknown(),
        occlusion_probability=1.0,
        geometry_confidence=0.0,
        shot_confidence=0.8,
    )


def _grammar(
    timestamp: float,
    *,
    closeup: float = 0.90,
    medium: float = 0.08,
    wide: float = 0.02,
    count: int = 1,
    headroom: float | None = 0.18,
    confidence: float | None = 0.95,
    source_id: str = "source-a",
) -> CausalShootingGrammarRecord:
    states = ("closeup", "medium", "wide")
    shot_state = states[max(range(3), key=(closeup, medium, wide).__getitem__)]
    return CausalShootingGrammarRecord(
        source_id=source_id,
        master_time_seconds=timestamp,
        shot_closeup_probability=closeup,
        shot_medium_probability=medium,
        shot_wide_probability=wide,
        shot_scale_state=shot_state,
        camera_pan_velocity=None,
        camera_pan_direction="missing",
        camera_tilt_velocity=None,
        camera_tilt_direction="missing",
        camera_zoom_in_probability=None,
        camera_zoom_out_probability=None,
        camera_static_probability=None,
        camera_zoom_state="missing",
        camera_roll_degrees_normalized=None,
        subject_headroom_normalized=headroom,
        subject_lead_room_normalized=None,
        detected_person_count=count,
        formation_person_count=count,
        native_cut_proximity=0.0,
        shooting_grammar_confidence=confidence,
    )


def _evidence(
    timestamp: float,
    *activities: ObservationActivityEvidence,
    vocal: float | None = 0.9,
) -> CausalPerformerEvidence:
    return CausalPerformerEvidence(
        source_id="source-a",
        master_time_seconds=timestamp,
        vocal_probability=vocal,
        observation_activity=tuple(sorted(activities, key=lambda item: item.observation_id)),
    )


def _provenance() -> IVEPerformerStateProvenance:
    return IVEPerformerStateProvenance(
        analyzer_fingerprint_sha256="a" * 64,
        sources=(
            IVEPerformerStateSourceProvenance(
                source_id="source-a",
                source_content_sha256="b" * 64,
            ),
        ),
        contracts=("mouth_activity_v1", "vocal_context_v1"),
    )


def _solo_inputs(
    *,
    vocal: float | None = 0.9,
) -> tuple[
    object,
    tuple[CausalShootingGrammarRecord, ...],
    tuple[CausalPerformerEvidence, ...],
]:
    graph = build_member_state_graph((_observation("solo"),))
    grammar = (_grammar(0.0),)
    evidence = (
        _evidence(
            0.0,
            ObservationActivityEvidence(
                observation_id="solo",
                mouth_activity_probability=0.95,
                body_activity_probability=0.82,
                activity_confidence=0.97,
            ),
            vocal=vocal,
        ),
    )
    return graph, grammar, evidence


class IVEPerformerStateDerivationTests(unittest.TestCase):
    def test_center_identity_and_visibility_are_each_counted_once(self) -> None:
        graph = build_member_state_graph((_observation(
            "solo", geometry_confidence=.8, shot_confidence=.9, occlusion=.2,
        ),))
        result = derive_ive_performer_state_records(graph, (_grammar(0.),))[0]
        expected = .96 * .8 * .9 * (1 - .2)
        self.assertAlmostEqual(expected, result.center_member_known_mass)
        self.assertAlmostEqual(expected, result.center_member.probability_for(IVEMember.GAEUL))
        self.assertAlmostEqual(1 - expected, result.center_member.unknown_mass)

    def test_two_visible_known_people_remain_a_mixture_not_artificial_unknown(self) -> None:
        graph = build_member_state_graph((
            _observation("left", member=IVEMember.GAEUL, center_x=.4, shot_confidence=.8),
            _observation("right", member=IVEMember.REI, center_x=.6, shot_confidence=.8, order=1),
        ))
        result = derive_ive_performer_state_records(
            graph, (_grammar(0., closeup=0., medium=0., wide=1., count=2),),
        )[0]
        self.assertAlmostEqual(.96 * .8, result.center_member_known_mass)
        self.assertAlmostEqual(.96 * .8 / 2, result.center_member.probability_for(IVEMember.GAEUL))
        self.assertAlmostEqual(.96 * .8 / 2, result.center_member.probability_for(IVEMember.REI))

    def test_solo_closeup_is_stronger_than_six_person_wide_but_never_hard(self) -> None:
        solo_graph, solo_grammar, solo_evidence = _solo_inputs()
        solo = derive_ive_performer_state_records(
            solo_graph,
            solo_grammar,
            performer_evidence=solo_evidence,
        )[0]

        centers = (0.10, 0.26, 0.42, 0.58, 0.74, 0.90)
        observations = tuple(
            _observation(
                f"member-{index}",
                member=member,
                order=index,
                center_x=center,
                width=0.12,
                shot_confidence=0.85,
            )
            for index, (member, center) in enumerate(zip(IVEMember, centers, strict=True))
        )
        group_graph = build_member_state_graph(observations)
        group_grammar = (
            _grammar(
                0.0,
                closeup=0.05,
                medium=0.15,
                wide=0.80,
                count=6,
                confidence=0.82,
            ),
        )
        group_evidence = (
            _evidence(
                0.0,
                *(
                    ObservationActivityEvidence(
                        observation_id=item.observation_id,
                        mouth_activity_probability=0.75,
                        body_activity_probability=0.85,
                    )
                    for item in observations
                ),
                vocal=0.9,
            ),
        )
        group = derive_ive_performer_state_records(
            group_graph,
            group_grammar,
            performer_evidence=group_evidence,
        )[0]

        self.assertGreater(solo.center_member_known_mass, group.center_member_known_mass)
        self.assertGreater(
            solo.active_performer_known_mass,
            group.active_performer_known_mass,
        )
        self.assertGreater(
            solo.center_member.probability_for(IVEMember.GAEUL),
            solo.center_member.probability_for(IVEMember.REI),
        )
        self.assertGreater(
            solo.active_performer.probability_for(IVEMember.GAEUL),
            solo.active_performer.probability_for(IVEMember.REI),
        )
        self.assertGreaterEqual(solo.center_member.unknown_mass, 0.05)
        self.assertGreaterEqual(solo.active_performer.unknown_mass, 0.05)
        self.assertAlmostEqual(
            1.0,
            sum(solo.center_member.member_probabilities.values())
            + solo.center_member.unknown_mass,
        )
        self.assertAlmostEqual(
            1.0,
            sum(group.active_performer.member_probabilities.values())
            + group.active_performer.unknown_mass,
        )

    def test_future_append_does_not_change_existing_exact_time_record(self) -> None:
        prefix_graph, prefix_grammar, prefix_evidence = _solo_inputs()
        prefix = derive_ive_performer_state_records(
            prefix_graph,
            prefix_grammar,
            performer_evidence=prefix_evidence,
        )

        full_graph = build_member_state_graph(
            (
                _observation("solo"),
                _observation(
                    "future",
                    member=IVEMember.REI,
                    timestamp=1.0,
                    order=1,
                    center_x=0.5,
                ),
            )
        )
        full_grammar = prefix_grammar + (
            _grammar(1.0, closeup=0.98, medium=0.01, wide=0.01),
        )
        full_evidence = prefix_evidence + (
            _evidence(
                1.0,
                ObservationActivityEvidence(
                    observation_id="future",
                    mouth_activity_probability=1.0,
                    body_activity_probability=1.0,
                ),
                vocal=1.0,
            ),
        )
        full = derive_ive_performer_state_records(
            full_graph,
            full_grammar,
            performer_evidence=full_evidence,
        )

        self.assertEqual(prefix[0].to_dict(), full[0].to_dict())
        self.assertEqual(2, len(full))

    def test_missing_exact_time_and_geometry_remain_explicit_unknown(self) -> None:
        graph = build_member_state_graph(
            (
                _observation("visible", timestamp=0.0, order=0),
                _missing_observation("missing", timestamp=1.0, order=1),
            )
        )
        grammar = (_grammar(0.5), _grammar(1.0))
        records = derive_ive_performer_state_records(graph, grammar)

        for record in records:
            self.assertEqual(1.0, record.center_member.unknown_mass)
            self.assertEqual(1.0, record.active_performer.unknown_mass)
            self.assertTrue(
                all(value == 0.0 for value in record.center_member.member_probabilities.values())
            )
        adapted = adapt_ive_performer_state_for_stage_edit(records[0])
        self.assertEqual(STAGE_EDIT_IVE_PERFORMER_STATE_ADAPTER, adapted.adapter)
        self.assertEqual(
            STAGE_EDIT_IVE_PERFORMER_STATE_FEATURE_NAMES,
            adapted.feature_names,
        )
        self.assertEqual(len(adapted.values), len(adapted.missing_mask))
        self.assertEqual((True, False, False), adapted.missing_mask[-3:])
        self.assertEqual(1.0, adapted.named_values()["ive_center_member_unknown_probability"][0])

    def test_occlusion_reduces_known_mass_and_complete_occlusion_is_unknown(self) -> None:
        visible_graph = build_member_state_graph((_observation("member"),))
        occluded_graph = build_member_state_graph(
            (_observation("member", occlusion=0.9),)
        )
        grammar = (_grammar(0.0),)
        visible = derive_ive_performer_state_records(visible_graph, grammar)[0]
        occluded = derive_ive_performer_state_records(occluded_graph, grammar)[0]

        self.assertGreater(
            visible.center_member_known_mass,
            occluded.center_member_known_mass,
        )
        self.assertGreater(occluded.center_member.unknown_mass, 0.9)

        missing_graph = build_member_state_graph(
            (_missing_observation("member", timestamp=0.0, order=0),)
        )
        missing = derive_ive_performer_state_records(missing_graph, grammar)[0]
        self.assertEqual(1.0, missing.center_member.unknown_mass)

    def test_explicit_nonvocal_and_uncorroborated_context_cannot_name_performer(self) -> None:
        graph, grammar, nonvocal_evidence = _solo_inputs(vocal=0.0)
        nonvocal = derive_ive_performer_state_records(
            graph,
            grammar,
            performer_evidence=nonvocal_evidence,
        )[0]
        self.assertGreater(nonvocal.center_member_known_mass, 0.0)
        self.assertEqual(0.0, nonvocal.active_performer_known_mass)
        self.assertEqual(1.0, nonvocal.active_performer.unknown_mass)
        self.assertEqual(0, nonvocal.corroborated_observation_count)

        vocal_only = (
            CausalPerformerEvidence(
                source_id="source-a",
                master_time_seconds=0.0,
                vocal_probability=1.0,
            ),
        )
        uncorroborated = derive_ive_performer_state_records(
            graph,
            grammar,
            performer_evidence=vocal_only,
        )[0]
        self.assertEqual(1.0, uncorroborated.active_performer.unknown_mass)

    def test_order_and_exact_source_observation_binding_fail_closed(self) -> None:
        graph = build_member_state_graph(
            (
                _observation("current", timestamp=0.0, order=0),
                _observation("later", timestamp=1.0, order=1),
            )
        )
        with self.assertRaisesRegex(IVEPerformerStateError, "strict"):
            derive_ive_performer_state_records(
                graph,
                (_grammar(1.0), _grammar(0.0)),
            )
        wrong_time = (
            _evidence(
                0.0,
                ObservationActivityEvidence(
                    observation_id="later",
                    mouth_activity_probability=1.0,
                ),
            ),
        )
        with self.assertRaisesRegex(IVEPerformerStateError, "exact source/time"):
            derive_ive_performer_state_records(
                graph,
                (_grammar(0.0), _grammar(1.0)),
                performer_evidence=wrong_time,
            )


class IVEPerformerStateArtifactTests(unittest.TestCase):
    def test_legacy_center_policy_keeps_its_original_hash_and_is_not_silently_upgraded(self) -> None:
        graph, grammar, evidence = _solo_inputs()
        artifact = build_ive_performer_state_artifact(
            graph, grammar, performer_evidence=evidence, provenance=_provenance(),
        )
        self.assertEqual(CENTER_DISTRIBUTION, artifact.center_distribution_contract)
        payload = artifact.to_dict()
        payload['computation_policy']['center_distribution'] = LEGACY_CENTER_DISTRIBUTION
        payload['content_sha256'] = _artifact_content_sha256(payload)
        restored = IVEPerformerStateArtifact.from_dict(payload)
        self.assertEqual(LEGACY_CENTER_DISTRIBUTION, restored.center_distribution_contract)
        self.assertEqual(payload, restored.to_dict())
        self.assertNotEqual(artifact.content_sha256, restored.content_sha256)

    def test_content_addressed_round_trip_privacy_and_named_adapter(self) -> None:
        graph, grammar, evidence = _solo_inputs()
        artifact = build_ive_performer_state_artifact(
            graph,
            grammar,
            performer_evidence=evidence,
            provenance=_provenance(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = write_ive_performer_state_artifact(Path(temporary), artifact)
            loaded = load_ive_performer_state_artifact(
                path,
                expected_content_sha256=artifact.content_sha256,
                expected_input_binding=artifact.input_binding,
                expected_source_content_sha256={"source-a": "b" * 64},
            )

            self.assertEqual(artifact.to_dict(), loaded.to_dict())
            self.assertEqual(f"{artifact.content_sha256}.json", path.name)
            self.assertEqual(0, path.stat().st_mode & 0o222)
            payload = artifact.to_dict()
            self.assertFalse(payload["storage_policy"]["raw_frames_persisted"])
            self.assertFalse(payload["storage_policy"]["pixels_persisted"])
            self.assertFalse(
                payload["storage_policy"][
                    "identity_or_appearance_embeddings_persisted"
                ]
            )
            serialized_records = json.dumps(payload["records"], sort_keys=True)
            for forbidden in ("image_bytes", "face_embedding", "body_embedding"):
                self.assertNotIn(forbidden, serialized_records)
            adapted = adapt_ive_performer_state_for_stage_edit(loaded.records[0])
            self.assertEqual(17, len(adapted.values))
            self.assertEqual((False,) * 17, adapted.missing_mask)

    def test_tamper_and_cross_input_replay_are_rejected(self) -> None:
        graph, grammar, evidence = _solo_inputs()
        artifact = build_ive_performer_state_artifact(
            graph,
            grammar,
            performer_evidence=evidence,
            provenance=_provenance(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = write_ive_performer_state_artifact(Path(temporary), artifact)
            os.chmod(path, 0o644)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["records"][0]["vocal_probability"] = 0.8
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(
                IVEPerformerStateArtifactError,
                "content hash mismatch",
            ):
                load_ive_performer_state_artifact(path)

        alternate_graph = build_member_state_graph(
            (_observation("solo", member=IVEMember.REI),)
        )
        alternate_binding = build_ive_performer_state_input_binding(
            alternate_graph,
            grammar,
            performer_evidence=evidence,
        )
        self.assertNotEqual(artifact.input_binding, alternate_binding)
        with tempfile.TemporaryDirectory() as temporary:
            path = write_ive_performer_state_artifact(Path(temporary), artifact)
            with self.assertRaisesRegex(
                IVEPerformerStateArtifactError,
                "replay/input binding mismatch",
            ):
                load_ive_performer_state_artifact(
                    path,
                    expected_input_binding=alternate_binding,
                )

    def test_provenance_source_binding_is_exact(self) -> None:
        graph, grammar, evidence = _solo_inputs()
        wrong = IVEPerformerStateProvenance(
            analyzer_fingerprint_sha256="a" * 64,
            sources=(
                IVEPerformerStateSourceProvenance(
                    source_id="source-b",
                    source_content_sha256="c" * 64,
                ),
            ),
        )
        with self.assertRaisesRegex(IVEPerformerStateError, "exactly bind"):
            build_ive_performer_state_artifact(
                graph,
                grammar,
                performer_evidence=evidence,
                provenance=wrong,
            )


if __name__ == "__main__":
    unittest.main()
