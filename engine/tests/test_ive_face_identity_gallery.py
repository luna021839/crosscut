from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.ive import IVEMember
from crosscut.member_state_graph import IdentityEvidenceKind
from crosscut.ive_face_identity_gallery import (
    CLOSEUP_FACECAM_MEDIA_CONTRACT,
    FACECAM_TITLE_MEMBER_AUTHORIZATION_CONTRACT,
    IDENTITY_PROVIDER_ID,
    PRIVATE_LOCAL_DISTRIBUTION_POLICY,
    AuthorizedFaceEmbeddingSample,
    IVEFaceGalleryAmbiguityConflict,
    IVEFaceGalleryAmbiguityError,
    IVEFaceGalleryArtifactError,
    IVEFaceGalleryConfig,
    IVEFaceGalleryError,
    IVEFaceGallerySourceAuthority,
    OfficialTargetMemberFancamProvenance,
    adapt_gallery_identity_for_member_state,
    build_ive_face_identity_gallery,
    classify_open_set_ive_identity,
    load_ive_face_identity_gallery_artifact,
    write_ive_face_identity_gallery_artifact,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _vector(member_index: int, marker: int, *, noise: float = 0.025) -> tuple[float, ...]:
    vector = np.zeros(128, dtype=np.float64)
    vector[member_index] = 1.0
    vector[16 + marker % 96] = noise
    vector /= np.linalg.norm(vector)
    return tuple(float(value) for value in vector)


def _mixed_vector(left_index: int, right_index: int) -> tuple[float, ...]:
    vector = np.zeros(128, dtype=np.float64)
    vector[left_index] = 1.0
    vector[right_index] = 1.0
    vector /= np.linalg.norm(vector)
    return tuple(float(value) for value in vector)


def _weighted_vector(
    left_index: int,
    right_index: int,
    *,
    left_weight: float,
    right_weight: float,
) -> tuple[float, ...]:
    vector = np.zeros(128, dtype=np.float64)
    vector[left_index] = left_weight
    vector[right_index] = right_weight
    vector /= np.linalg.norm(vector)
    return tuple(float(value) for value in vector)


def _provenance(member: IVEMember, video_number: int) -> OfficialTargetMemberFancamProvenance:
    member_index = tuple(IVEMember).index(member)
    marker = f"member-{member_index}-video-{video_number}"
    return OfficialTargetMemberFancamProvenance(
        song="TEST SONG",
        target_member=member,
        video_id=f"M{member_index}{video_number:02d}abc",
        title=f"IVE {member.value} Fancam TEST SONG",
        broadcast_date=f"2026-{video_number % 12 + 1:02d}-01",
        broadcast_program=f"Music Show {video_number}",
        channel_id=f"UCIVE{video_number:02d}",
        metadata_authority_sha256=_sha(f"metadata-{marker}"),
        acquisition_manifest_sha256=_sha(f"manifest-{marker}"),
        media_content_sha256=_sha(f"media-{marker}"),
        media_size_bytes=10_000 + video_number,
        media_width=1080,
        media_height=1920,
    )


def _facecam_provenance(
    member: IVEMember,
    video_number: int,
    *,
    title: str | None = None,
    width: int = 3840,
    height: int = 2160,
) -> OfficialTargetMemberFancamProvenance:
    member_index = tuple(IVEMember).index(member)
    marker = f"facecam-member-{member_index}-video-{video_number}"
    return OfficialTargetMemberFancamProvenance(
        song="TEST SONG",
        target_member=member,
        video_id=f"F{member_index}{video_number:02d}xyz",
        title=title or f"IVE {member.value} FaceCam TEST SONG",
        broadcast_date=f"2026-0{video_number + 1}-02",
        broadcast_program=f"Music Show {video_number}",
        channel_id=f"UCFACE{video_number:02d}",
        metadata_authority_sha256=_sha(f"metadata-{marker}"),
        acquisition_manifest_sha256=_sha(f"manifest-{marker}"),
        media_content_sha256=_sha(f"media-{marker}"),
        media_size_bytes=20_000 + video_number,
        media_width=width,
        media_height=height,
        source_authority=(
            IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_CLOSEUP_FACECAM
        ),
        source_media_contract=CLOSEUP_FACECAM_MEDIA_CONTRACT,
        title_member_authorization_contract=(
            FACECAM_TITLE_MEMBER_AUTHORIZATION_CONTRACT
        ),
    )


def _sample(
    provenance: OfficialTargetMemberFancamProvenance,
    sample_number: int,
    *,
    vector: tuple[float, ...] | None = None,
    timestamp_seconds: float | None = None,
) -> AuthorizedFaceEmbeddingSample:
    member_index = tuple(IVEMember).index(provenance.target_member)
    marker = (
        member_index * 30
        + int(provenance.video_id[2:4]) * 6
        + sample_number
    )
    return AuthorizedFaceEmbeddingSample(
        provenance=provenance,
        timestamp_seconds=(
            sample_number * 0.55
            if timestamp_seconds is None
            else timestamp_seconds
        ),
        frame_index=sample_number * 17,
        embedding=vector or _vector(member_index, marker),
        face_observation_sha256=_sha(
            f"observation-{provenance.provenance_sha256}-{sample_number}"
        ),
    )


def _complete_samples(
    *, videos_per_member: int = 3, samples_per_video: int = 4
) -> list[AuthorizedFaceEmbeddingSample]:
    samples: list[AuthorizedFaceEmbeddingSample] = []
    for member in IVEMember:
        for video_number in range(videos_per_member):
            provenance = _provenance(member, video_number)
            samples.extend(
                _sample(provenance, sample_number)
                for sample_number in range(samples_per_video)
            )
    return samples


def _mixed_authority_samples() -> list[AuthorizedFaceEmbeddingSample]:
    samples = _complete_samples()
    portrait = _provenance(IVEMember.GAEUL, 0)
    samples = [item for item in samples if item.provenance != portrait]
    facecam = _facecam_provenance(IVEMember.GAEUL, 0)
    samples.extend(_sample(facecam, index) for index in range(4))
    return samples


class AuthorizedProvenanceTests(unittest.TestCase):
    def test_ambiguity_conflict_is_typed_distinct_and_canonical(self) -> None:
        conflict = IVEFaceGalleryAmbiguityConflict(
            left_member=IVEMember.REI,
            right_member="가을",
            left_source_provenance_sha256=_sha("rei-provenance"),
            right_source_provenance_sha256=_sha("gaeul-provenance"),
            left_source_media_sha256=_sha("rei-media"),
            right_source_media_sha256=_sha("gaeul-media"),
            similarity=0.81,
            maximum_allowed_similarity=0.78,
        )

        self.assertEqual(conflict.left_member, IVEMember.GAEUL)
        self.assertEqual(conflict.right_member, IVEMember.REI)
        self.assertEqual(
            conflict.left_source_media_sha256,
            _sha("gaeul-media"),
        )
        with self.assertRaisesRegex(IVEFaceGalleryError, "two different"):
            replace(conflict, right_member=IVEMember.GAEUL)
        with self.assertRaisesRegex(IVEFaceGalleryError, "must be distinct"):
            replace(
                conflict,
                right_source_media_sha256=conflict.left_source_media_sha256,
            )

    def test_title_member_authority_and_media_hash_are_mandatory(self) -> None:
        valid = _provenance(IVEMember.GAEUL, 0)
        self.assertEqual(valid.group, "IVE")
        self.assertEqual(valid.target_member, IVEMember.GAEUL)
        self.assertNotIn("path", valid.to_dict())
        self.assertEqual(len(valid.media_content_sha256), 64)

        with self.assertRaisesRegex(IVEFaceGalleryError, "title lacks"):
            replace(valid, title="random downloaded file.mp4")
        with self.assertRaisesRegex(IVEFaceGalleryError, "SHA-256"):
            replace(valid, media_content_sha256="not-a-hash")
        with self.assertRaisesRegex(IVEFaceGalleryError, "IVE sources only"):
            replace(valid, group="OTHER")
        with self.assertRaisesRegex(IVEFaceGalleryError, "metadata contract"):
            replace(valid, metadata_contract="filename_guess_v1")

    def test_verified_landscape_facecam_is_a_distinct_identity_authority(self) -> None:
        source = _facecam_provenance(IVEMember.GAEUL, 0)

        self.assertEqual((source.media_width, source.media_height), (3840, 2160))
        self.assertEqual(
            source.source_authority,
            IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_CLOSEUP_FACECAM,
        )
        self.assertEqual(source.source_media_contract, CLOSEUP_FACECAM_MEDIA_CONTRACT)
        self.assertIn("FaceCam", source.title)

        gallery = build_ive_face_identity_gallery(_mixed_authority_samples())
        self.assertTrue(
            any(
                item.source_authority
                is IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_CLOSEUP_FACECAM
                for item in gallery.sources
            )
        )
        self.assertEqual(len(gallery.prototypes_for(IVEMember.GAEUL)), 3)

    def test_ordinary_landscape_fancam_and_fullcam_remain_rejected(self) -> None:
        with self.assertRaisesRegex(IVEFaceGalleryError, "FaceCam|얼빡|title lacks"):
            _facecam_provenance(
                IVEMember.GAEUL,
                0,
                title="IVE 가을 Fancam TEST SONG",
            )
        with self.assertRaisesRegex(IVEFaceGalleryError, "FaceCam|얼빡|title lacks"):
            _facecam_provenance(
                IVEMember.GAEUL,
                0,
                title="IVE 가을 FullCam TEST SONG",
            )
        portrait = _provenance(IVEMember.GAEUL, 0)
        with self.assertRaisesRegex(IVEFaceGalleryError, "9:16"):
            replace(portrait, media_width=3840, media_height=2160)

    def test_source_contract_and_single_yunet_face_seal_fail_closed(self) -> None:
        facecam = _facecam_provenance(IVEMember.GAEUL, 0)
        with self.assertRaisesRegex(IVEFaceGalleryError, "FaceCam media contract"):
            replace(facecam, source_media_contract="ordinary_landscape_fullcam_v1")
        with self.assertRaisesRegex(IVEFaceGalleryError, "authorization contract"):
            replace(
                facecam,
                title_member_authorization_contract=(
                    "unverified_filename_member_guess_v1"
                ),
            )
        sample = _sample(facecam, 0)
        with self.assertRaisesRegex(IVEFaceGalleryError, "exactly one"):
            replace(sample, detected_face_count=2)
        with self.assertRaisesRegex(IVEFaceGalleryError, "YuNet model"):
            replace(sample, face_detector_model_sha256=_sha("wrong-yunet"))

    def test_nonfinite_or_wrong_dimension_embeddings_fail_closed(self) -> None:
        provenance = _provenance(IVEMember.LIZ, 0)
        with self.assertRaisesRegex(IVEFaceGalleryError, "128"):
            _sample(provenance, 0, vector=(1.0, 0.0))
        bad = list(_vector(4, 1))
        bad[12] = float("nan")
        with self.assertRaisesRegex(IVEFaceGalleryError, "finite"):
            _sample(provenance, 0, vector=tuple(bad))


class GalleryBuildTests(unittest.TestCase):
    def test_ambiguity_outlier_margin_cannot_weaken_sealed_minimum(self) -> None:
        with self.assertRaisesRegex(
            IVEFaceGalleryError,
            "cannot weaken",
        ):
            IVEFaceGalleryConfig(
                minimum_ambiguity_outlier_cohesion_margin=0.119,
            )

    def test_same_member_multi_video_clusters_build_six_explicit_members(self) -> None:
        gallery = build_ive_face_identity_gallery(_complete_samples())

        self.assertEqual(
            {prototype.member for prototype in gallery.prototypes},
            set(IVEMember),
        )
        self.assertEqual(len(gallery.prototypes), 18)
        for member in IVEMember:
            prototypes = gallery.prototypes_for(member)
            self.assertEqual(len(prototypes), 3)
            self.assertEqual(
                len({item.source_media_sha256 for item in prototypes}), 3
            )
            self.assertTrue(
                all(len(item.contributing_sample_sha256) == 4 for item in prototypes)
            )

    def test_mislabeled_video_contamination_is_rejected_as_member_outlier(self) -> None:
        samples = _complete_samples()
        contaminated = _provenance(IVEMember.GAEUL, 3)
        # A whole title-authorized source accidentally contains Rei faces.  It
        # cannot pull Gaeul's robust video medoid because the other three
        # independently authorized videos agree with each other.
        samples.extend(
            _sample(
                contaminated,
                sample_number,
                vector=_vector(2, 80 + sample_number),
            )
            for sample_number in range(4)
        )

        gallery = build_ive_face_identity_gallery(samples)

        self.assertEqual(len(gallery.prototypes_for(IVEMember.GAEUL)), 3)
        contaminated_digest = contaminated.provenance_sha256
        reasons = {
            item.reason
            for item in gallery.rejections
            if item.source_provenance_sha256 == contaminated_digest
        }
        self.assertEqual(reasons, {"same_member_video_outlier"})

    def test_one_cross_member_ambiguous_source_is_removed_maximally(self) -> None:
        samples = _complete_samples()
        contaminated = _provenance(IVEMember.REI, 2)
        ambiguous = _weighted_vector(
            1,
            2,
            left_weight=0.8,
            right_weight=0.6,
        )
        rewritten = [
            (
                _sample(
                    item.provenance,
                    item.frame_index // 17,
                    vector=ambiguous,
                )
                if item.provenance == contaminated
                else item
            )
            for item in samples
        ]

        gallery = build_ive_face_identity_gallery(rewritten)
        reversed_gallery = build_ive_face_identity_gallery(
            list(reversed(rewritten))
        )

        self.assertEqual(len(gallery.prototypes), 17)
        self.assertEqual(len(gallery.prototypes_for(IVEMember.REI)), 2)
        self.assertEqual(
            {
                item.reason
                for item in gallery.rejections
                if item.source_provenance_sha256
                == contaminated.provenance_sha256
            },
            {"inter_member_ambiguous_outlier_source"},
        )
        self.assertEqual(
            gallery.fingerprint_sha256,
            reversed_gallery.fingerprint_sha256,
        )

    def test_ambiguous_outlier_is_not_removed_without_independent_floor_peers(
        self,
    ) -> None:
        samples = _complete_samples(videos_per_member=2)
        contaminated = _provenance(IVEMember.REI, 1)
        ambiguous = _weighted_vector(
            1,
            2,
            left_weight=0.8,
            right_weight=0.6,
        )
        rewritten = [
            (
                _sample(
                    item.provenance,
                    item.frame_index // 17,
                    vector=ambiguous,
                )
                if item.provenance == contaminated
                else item
            )
            for item in samples
        ]

        with self.assertRaisesRegex(
            IVEFaceGalleryAmbiguityError,
            "inter-member ambiguity",
        ):
            build_ive_face_identity_gallery(rewritten)

    def test_global_cap_can_replace_a_capped_in_ambiguous_source(self) -> None:
        samples = _complete_samples()
        contaminated = _provenance(IVEMember.REI, 2)
        ambiguous = _weighted_vector(
            1,
            2,
            left_weight=0.8,
            right_weight=0.6,
        )
        rewritten = [
            (
                _sample(
                    item.provenance,
                    item.frame_index // 17,
                    vector=ambiguous,
                )
                if item.provenance == contaminated
                else item
            )
            for item in samples
        ]

        gallery = build_ive_face_identity_gallery(
            rewritten,
            config=IVEFaceGalleryConfig(maximum_videos_per_member=2),
        )

        self.assertEqual(len(gallery.prototypes), 12)
        self.assertEqual(len(gallery.prototypes_for(IVEMember.REI)), 2)
        self.assertNotIn(
            contaminated.provenance_sha256,
            {
                item.source_provenance_sha256
                for item in gallery.prototypes_for(IVEMember.REI)
            },
        )
        ambiguous_audit = [
            item.to_dict()
            for item in gallery.rejections
            if item.source_provenance_sha256
            == contaminated.provenance_sha256
        ]
        self.assertTrue(ambiguous_audit)
        self.assertEqual(
            {item["reason"] for item in ambiguous_audit},
            {"inter_member_ambiguous_outlier_source"},
        )
        for item in ambiguous_audit:
            self.assertNotIn("embedding", item)
            self.assertNotIn("path", item)
            self.assertNotIn("video_id", item)

    def test_duplicate_immutable_video_identities_fail_before_floor_counting(
        self,
    ) -> None:
        samples = _complete_samples(videos_per_member=2)
        first = _provenance(IVEMember.GAEUL, 0)
        second = _provenance(IVEMember.GAEUL, 1)

        duplicate_media = replace(
            second,
            media_content_sha256=first.media_content_sha256,
        )
        media_rewritten = [
            (
                _sample(
                    duplicate_media,
                    item.frame_index // 17,
                    vector=item.embedding,
                )
                if item.provenance == second
                else item
            )
            for item in samples
        ]
        with self.assertRaisesRegex(
            IVEFaceGalleryError,
            "duplicate media content.*independent videos",
        ):
            build_ive_face_identity_gallery(media_rewritten)

        duplicate_video_id = replace(second, video_id=first.video_id)
        video_rewritten = [
            (
                _sample(
                    duplicate_video_id,
                    item.frame_index // 17,
                    vector=item.embedding,
                )
                if item.provenance == second
                else item
            )
            for item in samples
        ]
        with self.assertRaisesRegex(
            IVEFaceGalleryError,
            "duplicate video ID.*independent videos",
        ):
            build_ive_face_identity_gallery(video_rewritten)

    def test_policy_maximum_of_72_candidates_is_bounded_and_deterministic(
        self,
    ) -> None:
        samples = _complete_samples(videos_per_member=12)

        started = time.monotonic()
        gallery = build_ive_face_identity_gallery(samples)
        elapsed = time.monotonic() - started
        reversed_gallery = build_ive_face_identity_gallery(
            tuple(reversed(samples))
        )

        self.assertEqual(len(gallery.prototypes), 72)
        self.assertLess(elapsed, 5.0)
        self.assertEqual(gallery.to_dict(), reversed_gallery.to_dict())

    def test_excessive_pairwise_resolution_states_fail_closed(self) -> None:
        samples = _complete_samples(videos_per_member=26)

        started = time.monotonic()
        with self.assertRaisesRegex(
            IVEFaceGalleryError,
            "deterministic state budget",
        ):
            build_ive_face_identity_gallery(samples)

        self.assertLess(time.monotonic() - started, 15.0)

    def test_global_member_video_cap_is_applied_after_source_validation(
        self,
    ) -> None:
        samples = _complete_samples(videos_per_member=4)

        gallery = build_ive_face_identity_gallery(
            samples,
            config=IVEFaceGalleryConfig(maximum_videos_per_member=3),
        )

        self.assertEqual(len(gallery.prototypes), 18)
        self.assertEqual(
            sum(item.reason == "member_video_cap" for item in gallery.rejections),
            6 * 4,
        )

    def test_inter_member_lookalike_prototypes_reject_gallery(self) -> None:
        samples = _complete_samples()
        rewritten: list[AuthorizedFaceEmbeddingSample] = []
        for sample in samples:
            if sample.provenance.target_member is IVEMember.REI:
                rewritten.append(
                    _sample(
                        sample.provenance,
                        sample.frame_index // 17,
                        vector=_vector(1, 60 + sample.frame_index // 17),
                    )
                )
            else:
                rewritten.append(sample)
        with self.assertRaisesRegex(
            IVEFaceGalleryAmbiguityError,
            "inter-member ambiguity",
        ) as captured:
            build_ive_face_identity_gallery(rewritten)
        diagnostics = captured.exception.diagnostics
        self.assertEqual(
            diagnostics["contract"],
            "ive_face_gallery_scalar_ambiguity_diagnostics_v1",
        )
        self.assertFalse(diagnostics["feasible_safe_subset_found"])
        self.assertTrue(diagnostics["conflicts"])
        self.assertEqual(
            diagnostics["privacy"],
            {
                "contains_raw_embeddings": False,
                "contains_raw_frames": False,
                "contains_local_paths": False,
                "contains_plain_video_ids": False,
            },
        )
        for conflict in diagnostics["conflicts"]:
            self.assertEqual(len(conflict["left_source_provenance_sha256"]), 64)
            self.assertEqual(len(conflict["right_source_provenance_sha256"]), 64)
            self.assertGreaterEqual(
                conflict["similarity"],
                conflict["maximum_allowed_similarity"],
            )
            self.assertNotIn("video_id", conflict)
            self.assertNotIn("path", conflict)
            self.assertNotIn("embedding", conflict)

    def test_video_and_temporal_caps_prevent_one_video_from_dominating(self) -> None:
        samples = _complete_samples()
        target = _provenance(IVEMember.GAEUL, 0)
        samples = [item for item in samples if item.provenance != target]
        # Thirty temporally independent samples from one source still produce
        # one equal-weighted prototype and at most the configured sample cap.
        samples.extend(_sample(target, index) for index in range(30))
        # A near-duplicate temporal observation is rejected independently.
        close = _sample(
            target,
            90,
            timestamp_seconds=0.05,
            vector=_vector(1, 91),
        )
        samples.append(close)

        gallery = build_ive_face_identity_gallery(samples)
        target_prototype = next(
            item
            for item in gallery.prototypes_for(IVEMember.GAEUL)
            if item.source_media_sha256 == target.media_content_sha256
        )

        self.assertLessEqual(
            len(target_prototype.contributing_sample_sha256),
            gallery.config.maximum_samples_per_video,
        )
        self.assertEqual(len(gallery.prototypes_for(IVEMember.GAEUL)), 3)
        reasons = {item.reason for item in gallery.rejections}
        self.assertIn("per_video_sample_cap", reasons)
        self.assertIn("temporal_separation_cap", reasons)

    def test_build_is_deterministic_for_reordered_inputs(self) -> None:
        samples = _complete_samples()

        first = build_ive_face_identity_gallery(samples)
        second = build_ive_face_identity_gallery(tuple(reversed(samples)))

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.fingerprint_sha256, second.fingerprint_sha256)


class OpenSetClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gallery = build_ive_face_identity_gallery(_complete_samples())

    def test_confident_identity_returns_probabilities_opaque_token_and_adapter(self) -> None:
        result = classify_open_set_ive_identity(
            self.gallery, _vector(1, 90, noise=0.01)
        )

        self.assertEqual(result.accepted_member, IVEMember.GAEUL)
        self.assertLessEqual(result.member_probabilities.unknown_mass, 0.20)
        self.assertGreater(
            result.member_probabilities.probability_for(IVEMember.GAEUL), 0.72
        )
        self.assertTrue(result.opaque_identity_token.startswith("iveface:"))
        scalar = result.to_scalar_dict()
        self.assertNotIn("embedding", scalar)
        self.assertFalse(scalar["raw_query_embedding_persisted"])
        self.assertFalse(scalar["gallery_prototype_embeddings_persisted"])
        adapter = adapt_gallery_identity_for_member_state(result)
        self.assertEqual(adapter.face_identity_id, result.opaque_identity_token)
        self.assertEqual(adapter.member_probabilities, result.member_probabilities)
        self.assertIsNotNone(adapter.evidence)
        assert adapter.evidence is not None
        self.assertEqual(adapter.evidence.kind, IdentityEvidenceKind.FACE_IDENTITY_MATCH)
        self.assertEqual(adapter.evidence.provider_id, IDENTITY_PROVIDER_ID)
        self.assertEqual(adapter.evidence.evidence_sha256, result.evidence_sha256)

    def test_lookalike_query_with_small_top_two_margin_remains_unknown(self) -> None:
        result = classify_open_set_ive_identity(
            self.gallery, _mixed_vector(1, 2)
        )

        self.assertIsNone(result.accepted_member)
        self.assertEqual(result.decision_reason, "below_top_two_margin_gate")
        self.assertEqual(result.member_probabilities.unknown_mass, 1.0)
        self.assertIsNone(result.opaque_identity_token)
        adapter = adapt_gallery_identity_for_member_state(result)
        self.assertIsNone(adapter.evidence)
        self.assertIsNone(adapter.face_identity_id)

    def test_weak_similarity_query_remains_unknown(self) -> None:
        query = np.zeros(128, dtype=np.float64)
        query[127] = 1.0
        result = classify_open_set_ive_identity(self.gallery, query)

        self.assertIsNone(result.accepted_member)
        self.assertEqual(result.decision_reason, "below_absolute_similarity_gate")
        self.assertEqual(result.member_probabilities.unknown_mass, 1.0)


class PrivateArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gallery = build_ive_face_identity_gallery(_mixed_authority_samples())

    def test_content_addressed_round_trip_is_deterministic_and_private_local(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_ive_face_identity_gallery_artifact(root, self.gallery)
            second_gallery = build_ive_face_identity_gallery(
                tuple(reversed(_mixed_authority_samples()))
            )
            second = write_ive_face_identity_gallery_artifact(root, second_gallery)
            restored = load_ive_face_identity_gallery_artifact(
                first.path, expected_content_sha256=first.content_sha256
            )

            self.assertEqual(first.path, second.path)
            self.assertEqual(first.path.read_bytes(), second.path.read_bytes())
            self.assertEqual(restored.gallery.to_dict(), self.gallery.to_dict())
            payload = json.loads(first.path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["distribution_policy"], PRIVATE_LOCAL_DISTRIBUTION_POLICY
            )
            self.assertTrue(payload["privacy"]["private_local_only"])
            self.assertTrue(payload["privacy"]["contains_sface_prototype_vectors"])
            self.assertFalse(
                payload["privacy"]["stageedit_snapshot_export_permitted"]
            )
            self.assertFalse(
                payload["privacy"]["general_audit_json_export_permitted"]
            )
            self.assertNotIn("source_path", first.path.read_text(encoding="utf-8"))

    def test_exfat_without_hard_links_uses_locked_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "crosscut.ive_face_identity_gallery.os.link",
                side_effect=OSError(errno.ENOTSUP, "hard links unsupported"),
            ):
                artifact = write_ive_face_identity_gallery_artifact(
                    root,
                    self.gallery,
                )

            restored = load_ive_face_identity_gallery_artifact(
                artifact.path,
                expected_content_sha256=artifact.content_sha256,
            )
            self.assertEqual(restored.gallery.to_dict(), self.gallery.to_dict())
            self.assertFalse(
                any(
                    path.name.endswith(".publish-lock")
                    or path.name.endswith(".tmp")
                    for path in root.rglob("*")
                )
            )

    def test_tamper_symlink_duplicate_key_and_nonfinite_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = write_ive_face_identity_gallery_artifact(root, self.gallery)

            alias = root / "alias.json"
            alias.symlink_to(artifact.path)
            with self.assertRaisesRegex(IVEFaceGalleryArtifactError, "non-symlink"):
                load_ive_face_identity_gallery_artifact(alias)

            duplicate = root / "duplicate.json"
            duplicate.write_text('{"contract":"a","contract":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(IVEFaceGalleryArtifactError, "repeats key"):
                load_ive_face_identity_gallery_artifact(duplicate)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(IVEFaceGalleryArtifactError, "non-finite"):
                load_ive_face_identity_gallery_artifact(nonfinite)

            payload = json.loads(artifact.path.read_text(encoding="utf-8"))
            payload["distribution_policy"] = "public_export_allowed"
            artifact.path.chmod(0o600)
            artifact.path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(IVEFaceGalleryArtifactError, "policy|checksum"):
                load_ive_face_identity_gallery_artifact(artifact.path)

    def test_rehashed_facecam_contract_tamper_is_semantically_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = write_ive_face_identity_gallery_artifact(root, self.gallery)
            payload = json.loads(artifact.path.read_text(encoding="utf-8"))
            source = next(
                item
                for item in payload["gallery"]["sources"]
                if item["source_authority"]
                == IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_CLOSEUP_FACECAM.value
            )
            source["source_media_contract"] = "ordinary_landscape_fullcam_v1"
            payload["content_sha256"] = None
            encoded_without_digest = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest = hashlib.sha256(encoded_without_digest).hexdigest()
            payload["content_sha256"] = digest
            tampered = root / f"{digest}.json"
            tampered.write_text(
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
                IVEFaceGalleryArtifactError, "FaceCam media contract"
            ):
                load_ive_face_identity_gallery_artifact(tampered)

    def test_symlink_artifact_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(IVEFaceGalleryArtifactError, "root is a symlink"):
                write_ive_face_identity_gallery_artifact(alias, self.gallery)


if __name__ == "__main__":
    unittest.main()
