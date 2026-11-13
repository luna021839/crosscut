from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.causal_artifact_io import CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES
from crosscut import causal_shooting_grammar as grammar_module
from crosscut.causal_shooting_grammar import (
    ShootingGrammarObservation,
    ShootingGrammarProvenance,
    build_causal_shooting_grammar_artifact,
    load_causal_shooting_grammar_artifact,
    write_causal_shooting_grammar_artifact,
)
from crosscut.ive import IVEMember
from crosscut import ive_performer_state as performer_module
from crosscut.ive_performer_state import (
    IVEPerformerStateProvenance,
    IVEPerformerStateSourceProvenance,
    build_ive_performer_state_artifact,
    load_ive_performer_state_artifact,
    write_ive_performer_state_artifact,
)
from crosscut import member_state_graph as member_module
from crosscut.member_state_graph import (
    CausalMemberObservation,
    MemberProbabilityMass,
    MemberStateArtifactProvenance,
    MemberStateSourceProvenance,
    build_member_state_graph,
    load_member_state_graph_artifact,
    write_member_state_graph_artifact,
)
from crosscut.stage_edit_feature_materializer import StageEditCausalSensingArtifacts


_LEGACY_LIMIT = 32 * 1024 * 1024
_SOURCE_COUNT = 8
_MAXIMUM_STAGE_EDIT_TIMESTAMPS = 2048
_IVE_PERSON_COUNT = 6
_MAXIMUM_DETECTIONS_PER_FRAME = 12
_FORBIDDEN_KEY_PARTS = (
    "raw_frame",
    "frame_bytes",
    "image_bytes",
    "pixels",
    "face_embedding",
    "body_embedding",
    "appearance_embedding",
    "identity_embedding",
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _member_mass(member_index: int) -> MemberProbabilityMass:
    members = tuple(IVEMember)
    if member_index >= len(members):
        return MemberProbabilityMass(
            {member: 0.0 for member in members},
            unknown_mass=1.0,
        )
    return MemberProbabilityMass(
        {
            member: 0.9 if index == member_index else 0.0
            for index, member in enumerate(members)
        },
        unknown_mass=0.1,
    )


def _build_member_graph(
    *,
    source_count: int,
    timestamp_count: int,
    people_per_frame: int,
):
    sources = tuple(f"S{index}" for index in range(source_count))
    masses = tuple(_member_mass(index) for index in range(people_per_frame))
    observations: list[CausalMemberObservation] = []
    for source_id in sources:
        for time_index in range(timestamp_count):
            timestamp = time_index * 0.25
            for person_index, mass in enumerate(masses):
                left = 0.01 + person_index * (0.96 / people_per_frame)
                width = 0.80 / people_per_frame
                observations.append(
                    CausalMemberObservation(
                        observation_id=(
                            f"member-{source_id}-{time_index}-{person_index}"
                        ),
                        source_id=source_id,
                        shot_id=f"shot-{source_id}",
                        timestamp_seconds=timestamp,
                        frame_index=time_index,
                        causal_order=time_index * people_per_frame + person_index,
                        member_probabilities=mass,
                        face_bbox=(
                            left + width * 0.30,
                            0.20,
                            left + width * 0.70,
                            0.29,
                        ),
                        body_bbox=(left, 0.10, left + width, 0.95),
                        face_identity_id=(
                            f"face-{person_index}"
                            if person_index < _IVE_PERSON_COUNT
                            else None
                        ),
                        tracker_tracklet_id=f"track-{source_id}-{person_index}",
                        occlusion_probability=0.0,
                        geometry_confidence=1.0,
                        shot_confidence=1.0,
                    )
                )
    return sources, build_member_state_graph(observations)


def _build_grammar(*, sources: tuple[str, ...], timestamp_count: int):
    observations = tuple(
        ShootingGrammarObservation(
            source_id=source_id,
            master_time_seconds=time_index * 0.25,
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
            internal_native_cut_since_previous=(
                time_index > 0 and time_index % 16 == 0
            ),
            shooting_grammar_confidence=0.9,
        )
        for time_index in range(timestamp_count)
        for source_id in sources
    )
    return build_causal_shooting_grammar_artifact(
        observations,
        provenance=ShootingGrammarProvenance(
            upstream_sha256=(("source_catalog", _sha("catalog")),),
            analyzer_fingerprint_sha256=_sha("shooting-grammar"),
            contracts=("causal_artifact_scaling_v1",),
        ),
    )


def _member_provenance(sources: tuple[str, ...]) -> MemberStateArtifactProvenance:
    return MemberStateArtifactProvenance(
        producer_id="causal-artifact-scaling-test",
        producer_version="1",
        created_at_utc="2026-08-25T00:00:00Z",
        sources=tuple(
            MemberStateSourceProvenance(
                source_id=source_id,
                source_content_sha256=_sha(f"source:{source_id}"),
                observation_manifest_sha256=_sha(f"manifest:{source_id}"),
            )
            for source_id in sources
        ),
    )


def _performer_provenance(
    sources: tuple[str, ...],
) -> IVEPerformerStateProvenance:
    return IVEPerformerStateProvenance(
        analyzer_fingerprint_sha256=_sha("performer-state"),
        sources=tuple(
            IVEPerformerStateSourceProvenance(
                source_id=source_id,
                source_content_sha256=_sha(f"source:{source_id}"),
            )
            for source_id in sources
        ),
        contracts=("causal_artifact_scaling_v1",),
    )


def _reject_private_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).casefold()
            privacy_declaration = lowered.endswith("_persisted") and isinstance(
                nested, bool
            )
            if privacy_declaration and nested is not False:
                raise AssertionError(f"unsafe privacy declaration {key!r}")
            if not privacy_declaration and any(
                part in lowered for part in _FORBIDDEN_KEY_PARTS
            ):
                raise AssertionError(f"forbidden persisted key {key!r}")
            _reject_private_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_private_keys(nested)


class CausalArtifactScalingTests(unittest.TestCase):
    def test_all_three_artifacts_declare_one_finite_128_mib_cap(self) -> None:
        self.assertEqual(128 * 1024 * 1024, CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES)
        self.assertEqual(
            CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES,
            member_module.MEMBER_STATE_ARTIFACT_MAXIMUM_BYTES,
        )
        self.assertEqual(
            CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES,
            grammar_module.SHOOTING_GRAMMAR_ARTIFACT_MAXIMUM_BYTES,
        )
        self.assertEqual(
            CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES,
            performer_module.IVE_PERFORMER_STATE_ARTIFACT_MAXIMUM_BYTES,
        )

    def test_three_artifacts_roundtrip_privacy_and_stage_loader(self) -> None:
        sources, graph = _build_member_graph(
            source_count=2,
            timestamp_count=3,
            people_per_frame=_IVE_PERSON_COUNT,
        )
        grammar = _build_grammar(sources=sources, timestamp_count=3)
        performer = build_ive_performer_state_artifact(
            graph,
            grammar.records,
            provenance=_performer_provenance(sources),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = write_member_state_graph_artifact(
                root / "member",
                graph,
                _member_provenance(sources),
            )
            grammar_path = write_causal_shooting_grammar_artifact(
                grammar,
                root / "grammar",
            )
            performer_path = write_ive_performer_state_artifact(
                root / "performer",
                performer,
            )
            self.assertEqual(
                graph,
                load_member_state_graph_artifact(
                    member.path,
                    expected_content_sha256=member.content_sha256,
                ).graph,
            )
            self.assertEqual(
                grammar,
                load_causal_shooting_grammar_artifact(
                    grammar_path,
                    expected_content_sha256=grammar.artifact_content_sha256,
                ),
            )
            self.assertEqual(
                performer,
                load_ive_performer_state_artifact(
                    performer_path,
                    expected_content_sha256=performer.content_sha256,
                ),
            )
            for path in (member.path, grammar_path, performer_path):
                _reject_private_keys(json.loads(path.read_text(encoding="utf-8")))
            stage = StageEditCausalSensingArtifacts(
                shooting_grammar_path=grammar_path,
                shooting_grammar_content_sha256=(
                    grammar.artifact_content_sha256
                ),
                member_state_graph_path=member.path,
                member_state_graph_content_sha256=member.content_sha256,
                target_member=IVEMember.GAEUL,
                ive_performer_state_path=performer_path,
                ive_performer_state_content_sha256=performer.content_sha256,
            )
            self.assertTrue(stage.binding.production_complete)
            self.assertEqual(
                performer.content_sha256,
                stage.binding.ive_performer_state_content_sha256,
            )

    def test_each_writer_and_loader_fails_closed_over_declared_cap(self) -> None:
        sources, graph = _build_member_graph(
            source_count=2,
            timestamp_count=3,
            people_per_frame=_IVE_PERSON_COUNT,
        )
        grammar = _build_grammar(sources=sources, timestamp_count=3)
        performer = build_ive_performer_state_artifact(
            graph,
            grammar.records,
            provenance=_performer_provenance(sources),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(member_module, "_MAXIMUM_ARTIFACT_BYTES", 1024):
                with self.assertRaisesRegex(ValueError, "finite byte cap"):
                    write_member_state_graph_artifact(
                        root / "member-small-cap",
                        graph,
                        _member_provenance(sources),
                    )
            with patch.object(grammar_module, "_MAXIMUM_ARTIFACT_BYTES", 1024):
                with self.assertRaisesRegex(ValueError, "finite byte cap"):
                    write_causal_shooting_grammar_artifact(
                        grammar,
                        root / "grammar-small-cap",
                    )
            with patch.object(performer_module, "_MAXIMUM_ARTIFACT_BYTES", 1024):
                with self.assertRaisesRegex(ValueError, "finite byte cap"):
                    write_ive_performer_state_artifact(
                        root / "performer-small-cap",
                        performer,
                    )

            member = write_member_state_graph_artifact(
                root / "member",
                graph,
                _member_provenance(sources),
            )
            grammar_path = write_causal_shooting_grammar_artifact(
                grammar,
                root / "grammar",
            )
            performer_path = write_ive_performer_state_artifact(
                root / "performer",
                performer,
            )
            with patch.object(member_module, "_MAXIMUM_ARTIFACT_BYTES", 1024):
                with self.assertRaisesRegex(ValueError, "outside the finite cap"):
                    load_member_state_graph_artifact(member.path)
            with patch.object(grammar_module, "_MAXIMUM_ARTIFACT_BYTES", 1024):
                with self.assertRaisesRegex(ValueError, "outside the finite cap"):
                    load_causal_shooting_grammar_artifact(grammar_path)
            with patch.object(performer_module, "_MAXIMUM_ARTIFACT_BYTES", 1024):
                with self.assertRaisesRegex(ValueError, "outside the finite cap"):
                    load_ive_performer_state_artifact(performer_path)

    @unittest.skipUnless(
        os.environ.get("CROSSCUT_RUN_CAUSAL_SCALE_TESTS") == "1",
        "set CROSSCUT_RUN_CAUSAL_SCALE_TESTS=1 for the M4 production envelope",
    )
    def test_m4_exact_8_by_2048_six_ive_person_envelope_roundtrips(self) -> None:
        sources, graph = _build_member_graph(
            source_count=_SOURCE_COUNT,
            timestamp_count=_MAXIMUM_STAGE_EDIT_TIMESTAMPS,
            people_per_frame=_IVE_PERSON_COUNT,
        )
        grammar = _build_grammar(
            sources=sources,
            timestamp_count=_MAXIMUM_STAGE_EDIT_TIMESTAMPS,
        )
        performer = build_ive_performer_state_artifact(
            graph,
            grammar.records,
            provenance=_performer_provenance(sources),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = write_member_state_graph_artifact(
                root / "member",
                graph,
                _member_provenance(sources),
            )
            grammar_path = write_causal_shooting_grammar_artifact(
                grammar,
                root / "grammar",
            )
            performer_path = write_ive_performer_state_artifact(
                root / "performer",
                performer,
            )
            sizes = tuple(
                path.stat().st_size
                for path in (member.path, grammar_path, performer_path)
            )
            self.assertGreater(sizes[0], _LEGACY_LIMIT)
            self.assertTrue(
                all(0 < size <= CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES for size in sizes)
            )
            self.assertEqual(
                graph,
                load_member_state_graph_artifact(member.path).graph,
            )
            self.assertEqual(
                grammar,
                load_causal_shooting_grammar_artifact(grammar_path),
            )
            self.assertEqual(
                performer,
                load_ive_performer_state_artifact(performer_path),
            )

    @unittest.skipUnless(
        os.environ.get("CROSSCUT_RUN_CAUSAL_SCALE_TESTS") == "1",
        "set CROSSCUT_RUN_CAUSAL_SCALE_TESTS=1 for the M4 production envelope",
    )
    def test_m4_8_by_881_twelve_detection_envelope_is_finite(self) -> None:
        sources, graph = _build_member_graph(
            source_count=_SOURCE_COUNT,
            timestamp_count=881,
            people_per_frame=_MAXIMUM_DETECTIONS_PER_FRAME,
        )
        with tempfile.TemporaryDirectory() as temporary:
            member = write_member_state_graph_artifact(
                Path(temporary) / "member",
                graph,
                _member_provenance(sources),
            )
            size = member.path.stat().st_size
            self.assertGreater(size, _LEGACY_LIMIT)
            self.assertLessEqual(size, CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES)
            self.assertEqual(
                graph,
                load_member_state_graph_artifact(member.path).graph,
            )
            # The media job permits 12 detections per frame. Extending that
            # literal worst case to 2,048 timestamps is intentionally not
            # truncated to six people; it exceeds the scalar v1 byte envelope
            # and must fail closed until a future aggregation schema exists.
            projected = size * _MAXIMUM_STAGE_EDIT_TIMESTAMPS / 881
            self.assertGreater(projected, CAUSAL_SCALAR_ARTIFACT_MAXIMUM_BYTES)


if __name__ == "__main__":
    unittest.main()
