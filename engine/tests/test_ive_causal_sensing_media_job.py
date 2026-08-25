from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.causal_shooting_grammar import (
    load_causal_shooting_grammar_artifact,
)
from crosscut.ive import EditMode, IVEMember
from crosscut.ive_causal_sensing_media_job import (
    CausalFrameSample,
    IVECausalSensingMediaJobError,
    IVECausalSensingMediaRequest,
    IVECausalSensingRuntime,
    IVECausalSensingSource,
    OpenCVSequentialCausalFrameProvider,
    materialize_ive_causal_sensing_from_media,
    reject_attitude_before_ive_causal_sensing_access,
)
from crosscut.ive_face_identity_gallery import (
    AuthorizedFaceEmbeddingSample,
    OfficialTargetMemberFancamProvenance,
    build_ive_face_identity_gallery,
    write_ive_face_identity_gallery_artifact,
)
from crosscut.ive_performer_state import load_ive_performer_state_artifact
from crosscut.member_state_graph import load_member_state_graph_artifact
from crosscut.pose import FormationPersonDescriptor
from crosscut.reference_evidence_builder import ReferenceSourceClock
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor


def _sha(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vector(member_index: int, marker: int = 0) -> tuple[float, ...]:
    vector = np.zeros(128, dtype=np.float64)
    vector[member_index] = 1.0
    vector[16 + marker % 96] = 0.02
    vector /= np.linalg.norm(vector)
    return tuple(float(value) for value in vector)


def _ambiguous_vector(left: int, right: int) -> tuple[float, ...]:
    vector = np.zeros(128, dtype=np.float64)
    vector[left] = 1.0
    vector[right] = 1.0
    vector /= np.linalg.norm(vector)
    return tuple(float(value) for value in vector)


def _gallery_provenance(
    member: IVEMember,
    video_number: int,
) -> OfficialTargetMemberFancamProvenance:
    member_index = tuple(IVEMember).index(member)
    marker = f"{member_index}-{video_number}"
    return OfficialTargetMemberFancamProvenance(
        song="TEST SONG",
        target_member=member,
        video_id=f"T{member_index}{video_number:02d}abc",
        title=f"IVE {member.value} Fancam TEST SONG",
        broadcast_date=f"2026-0{video_number + 1}-01",
        broadcast_program=f"Test Music Show {video_number}",
        channel_id=f"UCTEST{member_index}{video_number}",
        metadata_authority_sha256=_sha(f"metadata-{marker}"),
        acquisition_manifest_sha256=_sha(f"manifest-{marker}"),
        media_content_sha256=_sha(f"media-{marker}"),
        media_size_bytes=10_000 + video_number,
        media_width=1080,
        media_height=1920,
    )


def _gallery_artifact(root: Path):
    samples: list[AuthorizedFaceEmbeddingSample] = []
    for member_index, member in enumerate(IVEMember):
        for video_number in range(2):
            provenance = _gallery_provenance(member, video_number)
            for sample_number in range(3):
                samples.append(
                    AuthorizedFaceEmbeddingSample(
                        provenance=provenance,
                        timestamp_seconds=sample_number * 0.6,
                        frame_index=sample_number * 18,
                        embedding=_vector(
                            member_index,
                            member_index * 20 + video_number * 3 + sample_number,
                        ),
                        face_observation_sha256=_sha(
                            f"face-{member_index}-{video_number}-{sample_number}"
                        ),
                    )
                )
    gallery = build_ive_face_identity_gallery(samples)
    return write_ive_face_identity_gallery_artifact(root, gallery)


def _descriptor(current: object, _following: object) -> FrameVisualDescriptor:
    assert isinstance(current, dict)
    closeup = current["source_id"] == "A"
    bbox = (0.38, 0.12, 0.24, 0.30)
    return FrameVisualDescriptor(
        quality_score=0.90,
        composition_score=0.88,
        shot_scale=0.67 if closeup else 0.40,
        shot_type=ShotType.CLOSE_UP if closeup else ShotType.MEDIUM,
        face_count=1,
        structure_embedding=(0.1, 0.2),
        motion=(0.12, -0.04),
        primary_face_bbox=bbox,
        face_detector_backend="yunet",
        primary_face_landmarks_xy=(
            (0.43, 0.22),
            (0.56, 0.22),
            (0.50, 0.29),
            (0.45, 0.36),
            (0.55, 0.36),
        ),
        detected_person_count=1,
        formation_people=(
            FormationPersonDescriptor(
                bbox_xywh=(0.24, 0.08, 0.52, 0.86),
                centroid_xy=(0.50, 0.51),
                detection_confidence=0.92,
            ),
        ),
        pose_confidence=0.85,
        pose_backend="synthetic_actual_pose",
        shot_scale_basis="dominant_face_height",
    )


def _embedding(current: object, _descriptor_value: FrameVisualDescriptor):
    assert isinstance(current, dict)
    if current["source_id"] == "A":
        return _vector(tuple(IVEMember).index(IVEMember.GAEUL), 90)
    return _ambiguous_vector(
        tuple(IVEMember).index(IVEMember.GAEUL),
        tuple(IVEMember).index(IVEMember.REI),
    )


class _FakeFrameProvider:
    def __init__(self, *, omit_last: bool = False) -> None:
        self.omit_last = omit_last
        self.calls: list[str] = []
        self.requested_by_source: dict[str, tuple[float, ...]] = {}
        self.active = 0
        self.maximum_active = 0

    def iter_source_samples(self, source, requested_source_seconds):
        self.calls.append(source.source_id)
        self.requested_by_source[source.source_id] = tuple(
            requested_source_seconds
        )
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            values = tuple(requested_source_seconds)
            if self.omit_last:
                values = values[:-1]
            for timestamp in values:
                frame_index = int(round(timestamp * 10.0))
                yield CausalFrameSample(
                    requested_source_seconds=timestamp,
                    decoded_source_seconds=frame_index / 10.0,
                    frame_index=frame_index,
                    following_frame_index=frame_index + 1,
                    frame_rate=10.0,
                    current_frame={
                        "source_id": source.source_id,
                        "timestamp": timestamp,
                    },
                    following_frame={
                        "source_id": source.source_id,
                        "timestamp": timestamp + 0.1,
                    },
                )
        finally:
            self.active -= 1


class _FakeCapture:
    def __init__(self, frame_count: int = 8) -> None:
        self.frames = [
            np.full((8, 8, 3), index, dtype=np.uint8)
            for index in range(frame_count)
        ]
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return 10.0
        return 0.0

    def read(self):
        if self.position >= len(self.frames):
            return (False, None)
        frame = self.frames[self.position]
        self.position += 1
        return (True, frame)

    def release(self) -> None:
        self.released = True


def _runtime(provider: _FakeFrameProvider) -> IVECausalSensingRuntime:
    return IVECausalSensingRuntime(
        frame_provider=provider,
        descriptor_provider=_descriptor,
        face_embedding_provider=_embedding,
        analyzer_fingerprint_sha256=_sha("synthetic-real-media-runtime-v1"),
    )


class IVECausalSensingMediaJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.gallery_artifact = _gallery_artifact(cls.root / "gallery")
        cls.media: dict[str, tuple[Path, str]] = {}
        for source_id in ("A", "B"):
            path = cls.root / f"source-{source_id}.bin"
            content = (f"verified-source-{source_id}" * 32).encode("utf-8")
            path.write_bytes(content)
            cls.media[source_id] = (path, _sha(content))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _sources(
        self,
        *,
        first_clock: ReferenceSourceClock | None = None,
        first_sha256: str | None = None,
    ) -> tuple[IVECausalSensingSource, ...]:
        source_a = IVECausalSensingSource(
            source_id="A",
            path=self.media["A"][0],
            media_content_sha256=first_sha256 or self.media["A"][1],
            clock=first_clock
            or ReferenceSourceClock.scalar(
                0.0,
                basis="synthetic_checked_identity_clock_v1",
            ),
            native_cut_source_seconds=(0.75,),
            native_cut_evidence_sha256=_sha("native-cuts-A"),
        )
        source_b = IVECausalSensingSource(
            source_id="B",
            path=self.media["B"][0],
            media_content_sha256=self.media["B"][1],
            clock=ReferenceSourceClock.scalar(
                0.1,
                basis="synthetic_checked_offset_clock_v1",
            ),
            native_cut_source_seconds=(),
            native_cut_evidence_sha256=_sha("native-cuts-B"),
        )
        return (source_a, source_b)

    def _request(
        self,
        output: Path,
        *,
        song: str = "TEST SONG",
        edit_mode: EditMode = EditMode.MEMBER,
        target_member: IVEMember | None = IVEMember.GAEUL,
        sources: tuple[IVECausalSensingSource, ...] | None = None,
        gallery_sha256: str | None = None,
    ) -> IVECausalSensingMediaRequest:
        return IVECausalSensingMediaRequest(
            song=song,
            edit_mode=edit_mode,
            target_member=target_member,
            sources=sources or self._sources(),
            master_decision_times=(0.5, 1.0),
            gallery_artifact_path=self.gallery_artifact.path,
            gallery_content_sha256=(
                gallery_sha256 or self.gallery_artifact.content_sha256
            ),
            artifact_directory=output,
            created_at_utc="2026-08-25T00:00:00Z",
        )

    def test_materializes_hash_bound_member_triple_with_unknown_on_ambiguity(self) -> None:
        provider = _FakeFrameProvider()
        request = self._request(self.root / "out-member")
        result = materialize_ive_causal_sensing_from_media(
            request,
            runtime=_runtime(provider),
        )

        self.assertEqual(provider.calls, ["A", "B"])
        self.assertEqual(provider.requested_by_source["A"], (0.5, 1.0))
        self.assertEqual(provider.requested_by_source["B"], (0.6, 1.1))
        self.assertEqual(provider.maximum_active, 1)
        self.assertEqual(result.source_count, 2)
        self.assertEqual(result.decision_count, 2)
        self.assertEqual(result.accepted_identity_observation_count, 2)
        result.stage_edit_artifacts.binding.require_production_scope(
            edit_mode="member",
            target_member=IVEMember.GAEUL,
        )

        grammar = load_causal_shooting_grammar_artifact(
            result.shooting_grammar_path,
            expected_content_sha256=result.shooting_grammar_content_sha256,
        )
        member = load_member_state_graph_artifact(
            result.member_state_graph_path,
            expected_content_sha256=result.member_state_graph_content_sha256,
        )
        performer = load_ive_performer_state_artifact(
            result.ive_performer_state_path,
            expected_content_sha256=result.ive_performer_state_content_sha256,
            expected_source_content_sha256={
                source.source_id: source.media_content_sha256
                for source in request.sources
            },
        )
        expected_keys = {
            (source_id, timestamp)
            for source_id in ("A", "B")
            for timestamp in (0.5, 1.0)
        }
        self.assertEqual(
            {(item.source_id, item.master_time_seconds) for item in grammar.records},
            expected_keys,
        )
        grammar_by_key = {
            (item.source_id, item.master_time_seconds): item
            for item in grammar.records
        }
        self.assertEqual(grammar_by_key[("A", 1.0)].native_cut_proximity, 1.0)
        self.assertEqual(grammar_by_key[("B", 1.0)].native_cut_proximity, 0.0)
        self.assertEqual(
            {(item.source_id, item.master_time_seconds) for item in performer.records},
            expected_keys,
        )
        a_observations = [
            item for item in member.graph.observations if item.source_id == "A"
        ]
        b_observations = [
            item for item in member.graph.observations if item.source_id == "B"
        ]
        self.assertTrue(
            all(
                item.member_probabilities.probability_for(IVEMember.GAEUL) > 0.8
                for item in a_observations
            )
        )
        self.assertTrue(
            all(item.member_probabilities.unknown_mass == 1.0 for item in b_observations)
        )
        self.assertTrue(all(item.face_identity_id is None for item in b_observations))
        self.assertTrue(
            all(
                record.active_performer.unknown_mass == 1.0
                for record in performer.records
            )
        )

    def test_group_scope_reuses_target_independent_triple(self) -> None:
        member_provider = _FakeFrameProvider()
        member = materialize_ive_causal_sensing_from_media(
            self._request(self.root / "out-same-member"),
            runtime=_runtime(member_provider),
        )
        group_provider = _FakeFrameProvider()
        group = materialize_ive_causal_sensing_from_media(
            self._request(
                self.root / "out-same-group",
                edit_mode=EditMode.GROUP,
                target_member=None,
            ),
            runtime=_runtime(group_provider),
        )

        group.stage_edit_artifacts.binding.require_production_scope(
            edit_mode="group",
            target_member=None,
        )
        self.assertIsNone(group.stage_edit_artifacts.binding.target_member)
        self.assertEqual(
            member.shooting_grammar_content_sha256,
            group.shooting_grammar_content_sha256,
        )
        self.assertEqual(
            member.member_state_graph_content_sha256,
            group.member_state_graph_content_sha256,
        )
        self.assertEqual(
            member.ive_performer_state_content_sha256,
            group.ive_performer_state_content_sha256,
        )

    def test_outputs_do_not_persist_media_paths_frames_descriptors_or_queries(self) -> None:
        result = materialize_ive_causal_sensing_from_media(
            self._request(self.root / "out-privacy"),
            runtime=_runtime(_FakeFrameProvider()),
        )
        encoded = b"\n".join(
            path.read_bytes()
            for path in (
                result.shooting_grammar_path,
                result.member_state_graph_path,
                result.ive_performer_state_path,
            )
        )
        for source_path, _digest in self.media.values():
            self.assertNotIn(str(source_path).encode("utf-8"), encoded)
        lowered = encoded.lower()
        for forbidden_key in (
            b'"raw_frame":',
            b'"frame_bytes":',
            b'"visual_descriptor":',
            b'"query_embedding":',
            b'"face_embedding":',
        ):
            self.assertNotIn(forbidden_key, lowered)
        self.assertNotIn(
            json.dumps(list(_vector(1, 90))).encode("utf-8"),
            encoded,
        )

    def test_attitude_is_rejected_before_gallery_files_models_or_provider(self) -> None:
        with self.assertRaisesRegex(
            IVECausalSensingMediaJobError,
            "ATTITUDE is strictly quarantined",
        ):
            reject_attitude_before_ive_causal_sensing_access(
                song="unrelated",
                source_ids=("opaque-source",),
                source_paths=("/private/ATTITUDE-source.mp4",),
            )
        provider = _FakeFrameProvider()
        request = self._request(
            self.root / "out-attitude",
            song="ATTITUDE",
            gallery_sha256="0" * 64,
        )
        with self.assertRaisesRegex(
            IVECausalSensingMediaJobError,
            "ATTITUDE is strictly quarantined",
        ):
            materialize_ive_causal_sensing_from_media(
                request,
                runtime=_runtime(provider),
            )
        self.assertEqual(provider.calls, [])
        self.assertFalse((self.root / "out-attitude").exists())

    def test_media_or_gallery_tamper_fails_before_frame_analysis(self) -> None:
        media_provider = _FakeFrameProvider()
        tampered_sources = self._sources(first_sha256=_sha("not-source-A"))
        with self.assertRaisesRegex(
            IVECausalSensingMediaJobError,
            "content SHA-256 mismatch",
        ):
            materialize_ive_causal_sensing_from_media(
                self._request(
                    self.root / "out-media-tamper",
                    sources=tampered_sources,
                ),
                runtime=_runtime(media_provider),
            )
        self.assertEqual(media_provider.calls, [])

        gallery_provider = _FakeFrameProvider()
        with self.assertRaisesRegex(ValueError, "gallery content SHA-256 mismatch"):
            materialize_ive_causal_sensing_from_media(
                self._request(
                    self.root / "out-gallery-tamper",
                    gallery_sha256="f" * 64,
                ),
                runtime=_runtime(gallery_provider),
            )
        self.assertEqual(gallery_provider.calls, [])

    def test_invalid_clock_and_incomplete_frame_coverage_fail_closed(self) -> None:
        negative_provider = _FakeFrameProvider()
        negative_clock = ReferenceSourceClock.scalar(
            -1.0,
            basis="synthetic_negative_clock_v1",
        )
        with self.assertRaisesRegex(
            IVECausalSensingMediaJobError,
            "clock does not cover",
        ):
            materialize_ive_causal_sensing_from_media(
                self._request(
                    self.root / "out-negative-clock",
                    sources=self._sources(first_clock=negative_clock),
                ),
                runtime=_runtime(negative_provider),
            )
        self.assertEqual(negative_provider.calls, [])

        incomplete_provider = _FakeFrameProvider(omit_last=True)
        with self.assertRaisesRegex(
            IVECausalSensingMediaJobError,
            "did not cover every exact source/time",
        ):
            materialize_ive_causal_sensing_from_media(
                self._request(self.root / "out-incomplete"),
                runtime=_runtime(incomplete_provider),
            )
        self.assertEqual(incomplete_provider.active, 0)

    def test_scope_source_and_registered_grid_contracts(self) -> None:
        with self.assertRaisesRegex(
            IVECausalSensingMediaJobError,
            "GROUP.*must not bind",
        ):
            self._request(
                self.root / "out-bad-group",
                edit_mode=EditMode.GROUP,
                target_member=IVEMember.GAEUL,
            )
        with self.assertRaisesRegex(
            IVECausalSensingMediaJobError,
            "MEMBER.*requires",
        ):
            self._request(
                self.root / "out-bad-member",
                edit_mode=EditMode.MEMBER,
                target_member=None,
            )
        with self.assertRaisesRegex(
            IVECausalSensingMediaJobError,
            "source-ID ordered",
        ):
            replace(
                self._request(self.root / "out-bad-source-order"),
                sources=tuple(reversed(self._sources())),
            )
        source_template = self._sources()[0]
        maximum_sources = tuple(
            replace(source_template, source_id=f"S{index:02d}")
            for index in range(24)
        )
        maximum_grid = tuple(index / 4.0 for index in range(2048))
        eight_source_request = replace(
            self._request(self.root / "out-eight-source-registered-grid"),
            sources=maximum_sources[:8],
            master_decision_times=maximum_grid,
        )
        self.assertEqual(
            len(eight_source_request.sources)
            * len(eight_source_request.master_decision_times),
            16_384,
        )
        maximum_request = replace(
            self._request(self.root / "out-maximum-registered-grid"),
            sources=maximum_sources,
            master_decision_times=maximum_grid,
        )
        self.assertEqual(
            len(maximum_request.sources) * len(maximum_request.master_decision_times),
            49_152,
        )

        oversized_sources = tuple(
            replace(source_template, source_id=f"T{index:02d}")
            for index in range(25)
        )
        with self.assertRaisesRegex(
            IVECausalSensingMediaJobError,
            "bounded M4-memory",
        ):
            replace(
                self._request(self.root / "out-oversized-grid"),
                sources=oversized_sources,
                master_decision_times=tuple(
                    index / 4.0 for index in range(2001)
                ),
            )

    def test_eight_source_actual_blackhole_union_grid_is_request_admissible(
        self,
    ) -> None:
        manifest_path = Path(
            "/Volumes/CrossCutMedia/CrossCut-Scratch/reference-edits/blackhole/"
            "stageedit-features-v3/"
            "stageedit-feature-snapshot-job-manifest-0a27cb8651902549.json"
        )
        if not manifest_path.is_file():
            self.skipTest("captured BLACKHOLE StageEdit grid is not mounted")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["manifest_sha256"],
            "0a27cb8651902549b3dc3d6bdda60f6485c3f7abd2985b1dd572421659d2a1cd",
        )
        decision_times = tuple(
            sorted(
                {
                    token["master_time_seconds"]
                    for snapshot in manifest["snapshots"]
                    for token in json.loads(
                        Path(snapshot["path"]).read_text(encoding="utf-8")
                    )["tokens"]
                }
            )
        )
        self.assertEqual(len(decision_times), 881)
        source_template = self._sources()[0]
        sources = tuple(
            replace(source_template, source_id=f"BLACKHOLE-{index:02d}")
            for index in range(8)
        )
        request = replace(
            self._request(self.root / "out-blackhole-grid"),
            song="BLACKHOLE",
            sources=sources,
            master_decision_times=decision_times,
        )
        self.assertEqual(
            len(request.sources) * len(request.master_decision_times),
            7_048,
        )

    def test_materializer_reuses_one_native_frame_for_strict_decision_times(
        self,
    ) -> None:
        provider = _FakeFrameProvider()
        descriptor_calls: list[object] = []
        embedding_calls: list[object] = []

        def descriptor_provider(current, following):
            descriptor_calls.append(current)
            return _descriptor(current, following)

        def embedding_provider(current, descriptor):
            embedding_calls.append(current)
            return _embedding(current, descriptor)

        runtime = IVECausalSensingRuntime(
            frame_provider=provider,
            descriptor_provider=descriptor_provider,
            face_embedding_provider=embedding_provider,
            analyzer_fingerprint_sha256=_sha("same-native-frame-runtime-v1"),
        )
        request = replace(
            self._request(self.root / "out-same-native-frame"),
            sources=(self._sources()[0],),
            master_decision_times=(0.01, 0.02),
        )
        result = materialize_ive_causal_sensing_from_media(
            request,
            runtime=runtime,
        )

        self.assertEqual(len(descriptor_calls), 1)
        self.assertEqual(len(embedding_calls), 1)
        member = load_member_state_graph_artifact(
            result.member_state_graph_path,
            expected_content_sha256=result.member_state_graph_content_sha256,
        )
        self.assertEqual(
            [item.timestamp_seconds for item in member.graph.observations],
            [0.01, 0.02],
        )
        self.assertEqual(
            [item.frame_index for item in member.graph.observations],
            [0, 1],
        )

    def test_default_decoder_uses_engine_ties_to_even_and_adjacent_following_frame(
        self,
    ) -> None:
        capture = _FakeCapture()
        provider = OpenCVSequentialCausalFrameProvider()
        with patch(
            "crosscut.ive_causal_sensing_media_job.cv2.VideoCapture",
            return_value=capture,
        ):
            samples = tuple(
                provider.iter_source_samples(
                    self._sources()[0],
                    (0.01, 0.02, 0.05, 0.15),
                )
            )

        self.assertEqual([item.frame_index for item in samples], [0, 0, 0, 2])
        self.assertEqual(
            [item.following_frame_index for item in samples],
            [1, 1, 1, 3],
        )
        self.assertEqual(
            [int(item.current_frame[0, 0, 0]) for item in samples],
            [0, 0, 0, 2],
        )
        self.assertEqual(
            [int(item.following_frame[0, 0, 0]) for item in samples],
            [1, 1, 1, 3],
        )
        self.assertIs(samples[0].current_frame, samples[1].current_frame)
        self.assertIs(samples[1].current_frame, samples[2].current_frame)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
