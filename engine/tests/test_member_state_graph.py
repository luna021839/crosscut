from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.ive import IVEMember
from crosscut.member_state_graph import (
    CausalMemberObservation,
    CorroboratedIdentityEvidence,
    IdentityEvidenceKind,
    MemberProbabilityMass,
    MemberStateArtifactError,
    MemberStateArtifactProvenance,
    MemberStateEdgeKind,
    MemberStateFeatureAdapter,
    MemberStateGraphConfig,
    MemberStateGraphError,
    MemberStateSourceProvenance,
    NormalizedBBox,
    ReIdentificationClaim,
    adapt_target_member_features,
    build_member_state_graph,
    load_member_state_graph_artifact,
    write_member_state_graph_artifact,
)


def _mass(
    target: IVEMember,
    probability: float = 0.9,
    unknown: float = 0.1,
) -> MemberProbabilityMass:
    values = {member: 0.0 for member in IVEMember}
    values[target] = probability
    remainder = 1.0 - probability - unknown
    if remainder < -1e-12:
        raise ValueError("test probability mass is invalid")
    if remainder > 0.0:
        fallback = next(member for member in IVEMember if member is not target)
        values[fallback] = remainder
    return MemberProbabilityMass(values, unknown_mass=unknown)


def _observation(
    observation_id: str,
    *,
    source: str = "source-a",
    shot: str = "shot-1",
    timestamp: float = 0.0,
    frame: int = 0,
    order: int = 0,
    center_x: float = 0.3,
    member: IVEMember = IVEMember.GAEUL,
    probability: float = 0.9,
    unknown: float = 0.1,
    tracker: str | None = None,
    pose: str | None = None,
    occlusion: float = 0.0,
    geometry_confidence: float = 1.0,
    shot_confidence: float = 1.0,
    face: bool = True,
    body: bool = True,
) -> CausalMemberObservation:
    body_bbox = (
        (center_x - 0.10, 0.10, center_x + 0.10, 0.90) if body else None
    )
    face_bbox = (
        (center_x - 0.04, 0.18, center_x + 0.04, 0.34) if face else None
    )
    return CausalMemberObservation(
        observation_id=observation_id,
        source_id=source,
        shot_id=shot,
        timestamp_seconds=timestamp,
        frame_index=frame,
        causal_order=order,
        member_probabilities=_mass(member, probability, unknown),
        face_bbox=face_bbox,
        body_bbox=body_bbox,
        face_identity_id=f"face:{observation_id}" if face else None,
        pose_tracklet_id=pose,
        tracker_tracklet_id=tracker,
        occlusion_probability=occlusion,
        geometry_confidence=geometry_confidence,
        shot_confidence=shot_confidence,
    )


def _missing_observation(
    observation_id: str,
    *,
    timestamp: float,
    frame: int,
    order: int,
    tracker: str,
) -> CausalMemberObservation:
    return CausalMemberObservation(
        observation_id=observation_id,
        source_id="source-a",
        shot_id="shot-1",
        timestamp_seconds=timestamp,
        frame_index=frame,
        causal_order=order,
        member_probabilities=MemberProbabilityMass.unknown(),
        tracker_tracklet_id=tracker,
        occlusion_probability=1.0,
        geometry_confidence=0.0,
    )


def _evidence(
    kind: IdentityEvidenceKind,
    marker: str,
    *,
    confidence: float = 0.96,
) -> CorroboratedIdentityEvidence:
    return CorroboratedIdentityEvidence(
        kind=kind,
        confidence=confidence,
        provider_id=f"provider-{marker}",
        evidence_sha256=marker * 64,
    )


class MemberProbabilityMassTests(unittest.TestCase):
    def test_requires_all_six_members_and_explicit_normalized_unknown_mass(self) -> None:
        probabilities = {member: 0.0 for member in IVEMember}
        probabilities[IVEMember.GAEUL] = 0.7
        probabilities[IVEMember.REI] = 0.1
        mass = MemberProbabilityMass(probabilities, unknown_mass=0.2)

        self.assertEqual(tuple(mass.member_probabilities), tuple(IVEMember))
        self.assertAlmostEqual(mass.probability_for("gaeul"), 0.7)
        self.assertAlmostEqual(mass.probability_for("레이"), 0.1)
        self.assertAlmostEqual(mass.unknown_mass, 0.2)
        with self.assertRaises(TypeError):
            mass.member_probabilities[IVEMember.GAEUL] = 0.8  # type: ignore[index]

        probabilities.pop(IVEMember.LEESEO)
        with self.assertRaisesRegex(MemberStateGraphError, "all six"):
            MemberProbabilityMass(probabilities, unknown_mass=0.2)

        with self.assertRaisesRegex(MemberStateGraphError, "sum to 1"):
            MemberProbabilityMass(
                {member: 0.0 for member in IVEMember}, unknown_mass=0.9
            )

    def test_missing_geometry_cannot_retain_or_invent_member_identity(self) -> None:
        with self.assertRaisesRegex(MemberStateGraphError, "unknown_mass must be 1"):
            CausalMemberObservation(
                observation_id="missing",
                source_id="source-a",
                shot_id="shot-1",
                timestamp_seconds=0.0,
                frame_index=0,
                causal_order=0,
                member_probabilities=_mass(IVEMember.GAEUL),
                tracker_tracklet_id="track-1",
                occlusion_probability=1.0,
                geometry_confidence=0.0,
            )

        missing = _missing_observation(
            "missing-safe", timestamp=0.0, frame=0, order=0, tracker="track-1"
        )
        self.assertEqual(missing.member_probabilities.unknown_mass, 1.0)
        self.assertFalse(missing.has_geometry)

    def test_rejects_non_normalized_or_inconsistent_face_body_geometry(self) -> None:
        with self.assertRaisesRegex(MemberStateGraphError, "bbox"):
            NormalizedBBox(-0.1, 0.1, 0.5, 0.9)

        with self.assertRaisesRegex(MemberStateGraphError, "contained"):
            CausalMemberObservation(
                observation_id="bad-face",
                source_id="source-a",
                shot_id="shot-1",
                timestamp_seconds=0.0,
                frame_index=0,
                causal_order=0,
                member_probabilities=_mass(IVEMember.GAEUL),
                face_bbox=(0.7, 0.1, 0.8, 0.2),
                body_bbox=(0.1, 0.1, 0.3, 0.9),
            )


class MemberStateAssociationTests(unittest.TestCase):
    def test_geometry_association_is_deterministic_for_same_source_frame_batches(self) -> None:
        observations = (
            _observation(
                "left-0", frame=0, timestamp=0.0, order=0, center_x=0.25, face=False
            ),
            _observation(
                "right-0", frame=0, timestamp=0.0, order=1, center_x=0.75, face=False
            ),
            # Detection order intentionally reverses at the next frame.
            _observation(
                "right-1", frame=3, timestamp=0.1, order=2, center_x=0.73, face=False
            ),
            _observation(
                "left-1", frame=3, timestamp=0.1, order=3, center_x=0.27, face=False
            ),
        )

        first = build_member_state_graph(observations)
        second = build_member_state_graph(observations)

        self.assertEqual(first, second)
        self.assertEqual(len(first.tracklets), 2)
        tracks = {frozenset(item.observation_ids) for item in first.tracklets}
        self.assertEqual(
            tracks,
            {
                frozenset(("left-0", "left-1")),
                frozenset(("right-0", "right-1")),
            },
        )
        self.assertTrue(
            all(
                edge.kind is MemberStateEdgeKind.WITHIN_TRACKLET
                and not edge.probability_mass_propagated
                for edge in first.edges
            )
        )

    def test_bounded_occlusion_recovery_keeps_gap_unknown(self) -> None:
        observations = (
            _observation(
                "visible-before",
                timestamp=0.0,
                frame=0,
                order=0,
                tracker="track-local-7",
            ),
            _missing_observation(
                "occluded",
                timestamp=0.1,
                frame=3,
                order=1,
                tracker="track-local-7",
            ),
            _observation(
                "visible-after",
                timestamp=0.2,
                frame=6,
                order=2,
                center_x=0.32,
                tracker="track-local-7",
            ),
        )

        graph = build_member_state_graph(observations)

        self.assertEqual(len(graph.tracklets), 1)
        self.assertEqual(graph.tracklets[0].occlusion_recovery_count, 1)
        recovery = [
            edge
            for edge in graph.edges
            if edge.kind is MemberStateEdgeKind.OCCLUSION_RECOVERY
        ]
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0].left_observation_id, "occluded")
        self.assertFalse(recovery[0].probability_mass_propagated)
        occluded = next(
            item for item in graph.observations if item.observation_id == "occluded"
        )
        self.assertEqual(occluded.member_probabilities.unknown_mass, 1.0)

    def test_recovery_beyond_bound_starts_a_new_tracklet(self) -> None:
        observations = (
            _observation(
                "visible-before",
                timestamp=0.0,
                frame=0,
                order=0,
                tracker="track-local-7",
            ),
            _missing_observation(
                "occluded",
                timestamp=0.4,
                frame=10,
                order=1,
                tracker="track-local-7",
            ),
            _observation(
                "too-late",
                timestamp=1.2,
                frame=40,
                order=2,
                center_x=0.32,
                tracker="track-local-7",
            ),
        )

        graph = build_member_state_graph(observations)

        self.assertEqual(len(graph.tracklets), 2)
        self.assertFalse(
            any(
                edge.kind is MemberStateEdgeKind.OCCLUSION_RECOVERY
                for edge in graph.edges
            )
        )

    def test_fails_closed_on_source_order_and_local_token_errors(self) -> None:
        with self.assertRaisesRegex(MemberStateGraphError, "timestamp order"):
            build_member_state_graph(
                (
                    _observation("later", timestamp=0.2, frame=6, order=0),
                    _observation("earlier", timestamp=0.1, frame=3, order=1),
                )
            )

        with self.assertRaisesRegex(MemberStateGraphError, "reuses"):
            build_member_state_graph(
                (
                    _observation(
                        "person-1",
                        timestamp=0.0,
                        frame=0,
                        order=0,
                        tracker="same-token",
                    ),
                    _observation(
                        "person-2",
                        timestamp=0.0,
                        frame=0,
                        order=1,
                        center_x=0.7,
                        tracker="same-token",
                    ),
                )
            )


class ConservativeReIdentificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.left = _observation(
            "left",
            source="source-a",
            center_x=0.3,
            member=IVEMember.GAEUL,
            probability=0.96,
            unknown=0.04,
        )
        self.right = _observation(
            "right",
            source="source-b",
            center_x=0.35,
            member=IVEMember.GAEUL,
            probability=0.97,
            unknown=0.03,
        )

    def test_cross_source_edges_are_never_created_without_an_explicit_claim(self) -> None:
        graph = build_member_state_graph((self.left, self.right))
        self.assertEqual(len(graph.tracklets), 2)
        self.assertFalse(
            any(
                edge.kind is MemberStateEdgeKind.CORROBORATED_REIDENTIFICATION
                for edge in graph.edges
            )
        )

    def test_one_modality_or_low_confidence_fails_closed(self) -> None:
        one_modality = ReIdentificationClaim(
            "left",
            "right",
            IVEMember.GAEUL,
            (_evidence(IdentityEvidenceKind.FACE_IDENTITY_MATCH, "a"),),
        )
        with self.assertRaisesRegex(MemberStateGraphError, "corroborating"):
            build_member_state_graph(
                (self.left, self.right), reidentification_claims=(one_modality,)
            )

        low = ReIdentificationClaim(
            "left",
            "right",
            IVEMember.GAEUL,
            (
                _evidence(IdentityEvidenceKind.FACE_IDENTITY_MATCH, "a"),
                _evidence(
                    IdentityEvidenceKind.BODY_APPEARANCE_MATCH,
                    "b",
                    confidence=0.90,
                ),
            ),
        )
        with self.assertRaisesRegex(MemberStateGraphError, "threshold"):
            build_member_state_graph(
                (self.left, self.right), reidentification_claims=(low,)
            )

    def test_two_high_confidence_modalities_create_only_a_scalar_reid_edge(self) -> None:
        claim = ReIdentificationClaim(
            "right",
            "left",
            IVEMember.GAEUL,
            (
                _evidence(IdentityEvidenceKind.BODY_APPEARANCE_MATCH, "b"),
                _evidence(IdentityEvidenceKind.FACE_IDENTITY_MATCH, "a"),
            ),
        )

        graph = build_member_state_graph(
            (self.left, self.right), reidentification_claims=(claim,)
        )

        edges = [
            edge
            for edge in graph.edges
            if edge.kind is MemberStateEdgeKind.CORROBORATED_REIDENTIFICATION
        ]
        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].member, IVEMember.GAEUL)
        self.assertEqual(len(edges[0].evidence), 2)
        self.assertFalse(edges[0].probability_mass_propagated)
        serialized = json.dumps(graph.to_dict(), ensure_ascii=False).casefold()
        self.assertNotIn("sface_embedding", serialized)
        self.assertNotIn("appearance_embedding", serialized)


class MemberStateFeatureAdapterTests(unittest.TestCase):
    def test_exposes_exact_timestamp_visibility_center_prominence_and_uncertainty(self) -> None:
        target = _observation(
            "gaeul",
            timestamp=0.0,
            frame=0,
            order=0,
            center_x=0.25,
            member=IVEMember.GAEUL,
            probability=1.0,
            unknown=0.0,
            occlusion=0.2,
        )
        other = _observation(
            "rei",
            timestamp=0.0,
            frame=0,
            order=1,
            center_x=0.75,
            member=IVEMember.REI,
            probability=1.0,
            unknown=0.0,
        )
        ambiguous = _observation(
            "ambiguous",
            timestamp=0.1,
            frame=3,
            order=2,
            center_x=0.4,
            member=IVEMember.GAEUL,
            probability=0.5,
            unknown=0.0,
        )
        graph = build_member_state_graph((target, other, ambiguous))
        adapter = MemberStateFeatureAdapter(graph)

        features = adapter.at_timestamp(
            source_id="source-a",
            timestamp_seconds=0.0,
            target_member=IVEMember.GAEUL,
        )

        self.assertAlmostEqual(features.target_member_probability, 1.0)
        self.assertAlmostEqual(features.visibility_probability, 0.8)
        self.assertAlmostEqual(features.center_x or 0.0, 0.25)
        self.assertGreater(features.prominence_probability, 0.4)
        self.assertLess(features.prominence_probability, 0.5)
        self.assertAlmostEqual(features.uncertainty_probability, 0.0)
        self.assertEqual(
            set(features.to_stage_edit_features()),
            {
                "target_member_probability",
                "target_visibility_probability",
                "target_center_x",
                "target_center_y",
                "target_prominence_probability",
                "target_member_uncertainty_probability",
                "target_member_state_observed",
            },
        )

        ambiguous_features = adapter.at_timestamp(
            source_id="source-a",
            timestamp_seconds=0.1,
            target_member=IVEMember.GAEUL,
        )
        self.assertAlmostEqual(ambiguous_features.uncertainty_probability, 1.0)

        missing = adapter.at_timestamp(
            source_id="source-a",
            timestamp_seconds=0.2,
            target_member=IVEMember.GAEUL,
        )
        self.assertFalse(missing.state_observed)
        self.assertIsNone(missing.center_xy)
        self.assertEqual(missing.visibility_probability, 0.0)
        self.assertEqual(missing.uncertainty_probability, 1.0)
        self.assertEqual(len(adapt_target_member_features(graph, "가을")), 2)

    def test_adapter_rejects_unknown_sources_instead_of_cross_source_fallback(self) -> None:
        graph = build_member_state_graph((_observation("known"),))
        with self.assertRaisesRegex(MemberStateGraphError, "unknown graph source"):
            MemberStateFeatureAdapter(graph).at_timestamp(
                source_id="wrong-source",
                timestamp_seconds=0.0,
                target_member=IVEMember.GAEUL,
            )


class MemberStateArtifactTests(unittest.TestCase):
    def _provenance(self, *source_ids: str) -> MemberStateArtifactProvenance:
        return MemberStateArtifactProvenance(
            producer_id="unit-test-caller",
            producer_version="1.0",
            created_at_utc="2026-08-25T00:00:00Z",
            sources=tuple(
                MemberStateSourceProvenance(
                    source_id=source_id,
                    source_content_sha256=marker * 64,
                    observation_manifest_sha256=marker.upper().casefold() * 64,
                )
                for source_id, marker in zip(source_ids, "cdef", strict=False)
            ),
        )

    def test_content_addressed_round_trip_is_immutable_and_privacy_declared(self) -> None:
        graph = build_member_state_graph(
            (
                _observation("a", source="source-a"),
                _observation("b", source="source-b"),
            )
        )
        provenance = self._provenance("source-a", "source-b")
        with tempfile.TemporaryDirectory() as temporary:
            first = write_member_state_graph_artifact(
                Path(temporary), graph, provenance
            )
            first_bytes = first.path.read_bytes()
            second = write_member_state_graph_artifact(
                Path(temporary), graph, provenance
            )

            self.assertEqual(first.path, second.path)
            self.assertEqual(first_bytes, second.path.read_bytes())
            self.assertEqual(first.path.stem, first.content_sha256)
            self.assertEqual(first.path.stat().st_mode & 0o222, 0)

            loaded = load_member_state_graph_artifact(
                first.path, expected_content_sha256=first.content_sha256
            )
            self.assertEqual(loaded.graph, graph)
            self.assertEqual(loaded.provenance, provenance)
            payload = json.loads(first.path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["privacy"],
                {
                    "raw_frames_persisted": False,
                    "compressed_frames_persisted": False,
                    "face_identity_embeddings_persisted": False,
                    "body_appearance_embeddings_persisted": False,
                },
            )
            graph_payload = json.dumps(payload["graph"]).casefold()
            self.assertNotIn("sface_embedding", graph_payload)
            self.assertNotIn("appearance_embedding", graph_payload)

    def test_checksum_tamper_and_source_provenance_mismatch_fail_closed(self) -> None:
        graph = build_member_state_graph((_observation("a"),))
        with tempfile.TemporaryDirectory() as temporary:
            artifact = write_member_state_graph_artifact(
                Path(temporary), graph, self._provenance("source-a")
            )
            payload = json.loads(artifact.path.read_text(encoding="utf-8"))
            payload["graph"]["observations"][0]["shot_confidence"] = 0.5
            artifact.path.chmod(0o644)
            artifact.path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MemberStateArtifactError, "checksum"):
                load_member_state_graph_artifact(artifact.path)

        wrong_provenance = self._provenance("source-b")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(MemberStateGraphError, "exactly cover"):
                write_member_state_graph_artifact(
                    Path(temporary), graph, wrong_provenance
                )

    def test_semantically_forged_topology_fails_even_with_a_matching_digest(self) -> None:
        graph = build_member_state_graph(
            (
                _observation(
                    "first", timestamp=0.0, frame=0, order=0, face=False
                ),
                _observation(
                    "second", timestamp=0.1, frame=3, order=1, face=False
                ),
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            artifact = write_member_state_graph_artifact(
                Path(temporary), graph, self._provenance("source-a")
            )
            payload = json.loads(artifact.path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["graph"]["edges"]), 1)
            payload["graph"]["edges"][0]["confidence"] = 0.123
            payload["content_sha256"] = None
            canonical_without_digest = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest = hashlib.sha256(canonical_without_digest).hexdigest()
            payload["content_sha256"] = digest
            forged_path = Path(temporary) / f"{digest}.json"
            forged_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                MemberStateArtifactError, "topology is not reproducible"
            ):
                load_member_state_graph_artifact(forged_path)


if __name__ == "__main__":
    unittest.main()
