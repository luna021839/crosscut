from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.acquisition import _issue_member_acquisition_identity_authority
from crosscut.ive import IVEMember
from crosscut.ive_face_identity_gallery import (
    AuthorizedFaceEmbeddingSample,
    IVEFaceGalleryError,
    IVEFaceGallerySourceAuthority,
    load_ive_face_identity_gallery_artifact,
)
from crosscut.ive_face_identity_gallery_materializer import (
    FACECAM_REJECTION_AUTHORITY_BASIS,
    PORTRAIT_ACQUISITION_AUTHORITY_BASIS,
    IVEFaceGalleryMaterializerConfig,
    IVEFaceGalleryMaterializerError,
    materialize_ive_face_identity_gallery_job,
    seal_closeup_facecam_sources_from_rejection_manifest,
    seal_portrait_gallery_sources_from_acquisition,
)
from crosscut.media import AudioStreamInfo, MediaInfo, VideoStreamInfo


def _sha(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _envelope(payload: dict[str, object]) -> dict[str, object]:
    return {
        "algorithm": "sha256",
        "sha256": _canonical_sha(payload),
        "payload": payload,
    }


def _media(
    path: Path,
    *,
    width: int = 3840,
    height: int = 2160,
    duration: float = 20.0,
) -> MediaInfo:
    return MediaInfo(
        path=str(path),
        format_name="mp4",
        duration_seconds=duration,
        start_time_seconds=0.0,
        size_bytes=path.stat().st_size,
        bit_rate=8_000_000,
        audio_streams=(
            AudioStreamInfo(1, "aac", 48_000, 2, "stereo", duration, 128_000),
        ),
        video_streams=(
            VideoStreamInfo(
                0,
                "h264",
                width,
                height,
                30.0,
                duration,
                8_000_000,
            ),
        ),
        tags={},
    )


def _write_facecam_rejection_manifest(
    root: Path,
    member: IVEMember,
    *,
    count: int = 2,
    title_camera: str = "FaceCam",
    width: int = 3840,
    height: int = 2160,
) -> tuple[object, tuple[str, ...], tuple[Path, ...]]:
    cache = root / f"member-{tuple(IVEMember).index(member)}"
    videos = cache / "videos"
    videos.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    paths: list[Path] = []
    ids: list[str] = []
    for index in range(count):
        member_index = tuple(IVEMember).index(member)
        video_id = f"FC{member_index}{index:02d}xy"
        ids.append(video_id)
        path = videos / f"{video_id}.mp4.invalid.3"
        path.write_bytes(f"fixture-{member.value}-{index}".encode("utf-8"))
        paths.append(path)
        title = f"IVE {member.value} {title_camera} TEST SONG Music Bank 26010{index + 1}"
        metadata_payload: dict[str, object] = {
            "schema_version": 1,
            "video_id": video_id,
            "title": title,
            "channel": "KBS Kpop",
            "channel_id": "UCeLPm9yH_a_QH8n6445G-Ow",
            "duration_seconds": 20.0,
            "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
            "broadcast_date": f"2026-01-0{index + 1}",
            "broadcast_program": "Music Bank",
            "broadcast_date_basis": "title",
            "official_channel_signal": "channel_id_allowlist",
            "current_metadata_observation_sha256": [
                _sha(f"metadata-observation-{member.value}-{index}")
            ],
        }
        request_payload: dict[str, object] = {
            "schema_version": 1,
            "group": "IVE",
            "song": "TEST SONG",
            "source_mode": "automatic",
            "edit_mode": "member",
            "aspect_ratio": "9:16",
            "target_member": member.value,
            "automatic_media_quality_contract_version": 1,
            "member_portrait_aspect_contract_version": 1,
            "expected_width_over_height": 9.0 / 16.0,
            "aspect_ratio_tolerance": 0.01,
            "minimum_short_edge": 720,
            "minimum_long_edge": 1280,
            "minimum_frame_rate": 23.0,
            "download_profile": "fixture",
            "download_format": None,
        }
        diagnostics: dict[str, object] = {
            "required": True,
            "analysis_basis": "first_video_stream_width_over_height",
            "expected_width_over_height": 9.0 / 16.0,
            "tolerance": 0.01,
            "minimum_short_edge": 720,
            "minimum_long_edge": 1280,
            "minimum_frame_rate": 23.0,
            "width": width,
            "height": height,
            "actual_short_edge": min(width, height),
            "actual_long_edge": max(width, height),
            "frame_rate": 30.0,
            "actual_width_over_height": width / height,
            "passed": False,
            "hard_rejection_reason": (
                "automatic_member_portrait_aspect_ratio_mismatch"
            ),
        }
        entries.append(
            {
                "schema_version": 1,
                "evidence_origin": "downloaded_media_contract_failure",
                "video_id": video_id,
                "candidate_media_metadata_identity": _envelope(metadata_payload),
                "request_media_contract_identity": _envelope(request_payload),
                "hard_rejection_reason": (
                    "automatic_member_portrait_aspect_ratio_mismatch"
                ),
                "diagnostics": diagnostics,
                "diagnostics_sha256": _canonical_sha(diagnostics),
                "quarantine_artifact": {
                    "relative_path": f"videos/{path.name}",
                    "content_integrity": {
                        "contract_version": 1,
                        "algorithm": "sha256",
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha(path.read_bytes()),
                    },
                },
                "recorded_at": "2026-01-01T00:00:00+00:00",
            }
        )
    member_index = tuple(IVEMember).index(member)
    selected_source = videos / f"selected-{member_index}.mp4"
    selected_source.write_bytes(f"selected-{member.value}".encode("utf-8"))
    selected_digest = _sha(selected_source.read_bytes())
    selected = {
        "source_index": 0,
        "video_id": f"SEL{member_index}xyz",
        "title": f"IVE {member.value} TEST SONG Fancam Music Bank 260101",
        "channel": "KBS Kpop",
        "channel_id": "UCeLPm9yH_a_QH8n6445G-Ow",
        "duration_seconds": 20.0,
        "webpage_url": f"https://www.youtube.com/watch?v=SEL{member_index}xyz",
        "query": "fixture",
        "queries": ["fixture"],
        "discovery_sources": ["query_search"],
        "broadcast_date": "2026-01-01",
        "broadcast_program": "music_bank",
        "broadcast_date_basis": "title",
        "official_channel_signal": "allowlisted_channel_id",
        "fallback_reason": None,
        "download_status": "downloaded",
        "local_path": str(selected_source.resolve()),
        "media_validation": {"passed": True},
        "aspect_validation": {"passed": True},
        "content_integrity": {
            "contract_version": 1,
            "algorithm": "sha256",
            "size_bytes": selected_source.stat().st_size,
            "sha256": selected_digest,
        },
    }
    manifest = {
        "schema_version": 1,
        "content_integrity_contract_version": 1,
        "automatic_media_quality_contract_version": 1,
        "persistent_media_contract_rejection_contract_version": 1,
        "member_portrait_aspect_contract_version": 1,
        "group": "IVE",
        "song": "TEST SONG",
        "edit_mode": "member",
        "aspect_ratio": "9:16",
        "target_member": member.value,
        "persistent_media_contract_rejections": entries,
        "summary": {"status": "complete"},
        "selected": [selected],
    }
    manifest_path = cache / "acquisition-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    authority = _issue_member_acquisition_identity_authority(
        source_paths=(selected_source,),
        manifest_path=manifest_path,
        manifest=manifest,
    )
    if authority is None:
        raise AssertionError("fixture did not issue member acquisition authority")
    return (authority, tuple(ids), tuple(paths))


def _write_portrait_authority(root: Path):  # type: ignore[no-untyped-def]
    source = root / "portrait.mp4"
    source.write_bytes(b"portrait-fixture")
    digest = _sha(source.read_bytes())
    selected = {
        "source_index": 0,
        "video_id": "PORTRT01",
        "title": "IVE 가을 TEST SONG Fancam Music Bank 260101",
        "channel": "KBS Kpop",
        "channel_id": "UCeLPm9yH_a_QH8n6445G-Ow",
        "duration_seconds": 20.0,
        "webpage_url": "https://www.youtube.com/watch?v=PORTRT01",
        "query": "fixture",
        "queries": ["fixture"],
        "discovery_sources": ["query_search"],
        "broadcast_date": "2026-01-01",
        "broadcast_program": "music_bank",
        "broadcast_date_basis": "title",
        "official_channel_signal": "channel_id_allowlist",
        "fallback_reason": None,
        "download_status": "downloaded",
        "local_path": str(source.resolve()),
        "media_validation": {"passed": True},
        "aspect_validation": {"passed": True},
        "content_integrity": {
            "contract_version": 1,
            "algorithm": "sha256",
            "size_bytes": source.stat().st_size,
            "sha256": digest,
        },
    }
    manifest = {
        "schema_version": 1,
        "content_integrity_contract_version": 1,
        "automatic_media_quality_contract_version": 1,
        "member_portrait_aspect_contract_version": 1,
        "group": "IVE",
        "song": "TEST SONG",
        "edit_mode": "member",
        "target_member": "가을",
        "summary": {"status": "complete"},
        "selected": [selected],
    }
    manifest_path = root / "portrait-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    authority = _issue_member_acquisition_identity_authority(
        source_paths=(source,),
        manifest_path=manifest_path,
        manifest=manifest,
    )
    if authority is None:
        raise AssertionError("fixture did not issue portrait authority")
    return authority, source


def _frames(_path: Path):  # type: ignore[no-untyped-def]
    for index in range(20):
        yield (
            index,
            float(index),
            np.full((12, 12, 3), index % 255, dtype=np.uint8),
        )


def _fake_extract(
    _frame: np.ndarray,
    *,
    provenance,  # type: ignore[no-untyped-def]
    timestamp_seconds: float,
    frame_index: int,
    face_observation_sha256: str,
) -> AuthorizedFaceEmbeddingSample:
    member_index = tuple(IVEMember).index(provenance.target_member)
    vector = np.zeros(128, dtype=np.float64)
    vector[member_index] = 1.0
    vector[16 + (frame_index % 80)] = 0.02
    return AuthorizedFaceEmbeddingSample(
        provenance=provenance,
        timestamp_seconds=timestamp_seconds,
        frame_index=frame_index,
        embedding=tuple(float(value) for value in vector),
        face_observation_sha256=face_observation_sha256,
    )


class SourceAuthorityTests(unittest.TestCase):
    def test_invalid3_landscape_facecam_is_sealed_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            acquisition, video_ids, paths = _write_facecam_rejection_manifest(
                root, IVEMember.GAEUL
            )
            before = tuple((path.read_bytes(), path.stat().st_mtime_ns) for path in paths)

            authorities = seal_closeup_facecam_sources_from_rejection_manifest(
                acquisition,
                video_ids=video_ids,
                media_probe=lambda path: _media(path),
            )

            self.assertEqual(len(authorities), 2)
            self.assertTrue(all(item.media_path.name.endswith(".invalid.3") for item in authorities))
            self.assertTrue(
                all(
                    item.acquisition_authority_basis
                    == FACECAM_REJECTION_AUTHORITY_BASIS
                    for item in authorities
                )
            )
            self.assertTrue(
                all(
                    item.provenance.source_authority
                    is IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_CLOSEUP_FACECAM
                    for item in authorities
                )
            )
            after = tuple((path.read_bytes(), path.stat().st_mtime_ns) for path in paths)
            self.assertEqual(before, after)

    def test_ordinary_landscape_fancam_and_fullcam_are_not_facecam_authorities(self) -> None:
        for title_camera in ("Fancam", "FullCam"):
            with self.subTest(title_camera=title_camera):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    acquisition, video_ids, _paths = (
                        _write_facecam_rejection_manifest(
                            root,
                            IVEMember.GAEUL,
                            title_camera=title_camera,
                        )
                    )
                    with self.assertRaisesRegex(
                        IVEFaceGalleryMaterializerError,
                        "FaceCam|얼빡|title lacks",
                    ):
                        seal_closeup_facecam_sources_from_rejection_manifest(
                            acquisition,
                            video_ids=video_ids,
                            media_probe=lambda path: _media(path),
                        )

    def test_manifest_or_media_tamper_fails_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            acquisition, video_ids, paths = _write_facecam_rejection_manifest(
                root, IVEMember.GAEUL
            )
            manifest = Path(acquisition.manifest_path)
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                IVEFaceGalleryMaterializerError, "sealed authority"
            ):
                seal_closeup_facecam_sources_from_rejection_manifest(
                    acquisition,
                    video_ids=video_ids,
                    media_probe=lambda path: _media(path),
                )
            # Restore exact manifest bytes so the independent media-content
            # tamper reaches its own gate.
            manifest.write_text(
                manifest.read_text(encoding="utf-8").rstrip(), encoding="utf-8"
            )
            self.assertEqual(_sha(manifest.read_bytes()), acquisition.manifest_sha256)
            paths[0].write_bytes(b"changed-after-acquisition")
            with self.assertRaisesRegex(
                IVEFaceGalleryMaterializerError, "content no longer matches"
            ):
                seal_closeup_facecam_sources_from_rejection_manifest(
                    acquisition,
                    video_ids=video_ids,
                    media_probe=lambda path: _media(path),
                )

    def test_process_sealed_portrait_authority_rebinds_current_sha_and_9x16(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            acquisition, source = _write_portrait_authority(root)

            sealed = seal_portrait_gallery_sources_from_acquisition(
                acquisition,
                media_probe=lambda path: _media(
                    path, width=1080, height=1920
                ),
            )

            self.assertEqual(len(sealed), 1)
            self.assertEqual(
                sealed[0].acquisition_authority_basis,
                PORTRAIT_ACQUISITION_AUTHORITY_BASIS,
            )
            self.assertEqual(sealed[0].media_sha256, _sha(source.read_bytes()))
            self.assertEqual(
                sealed[0].provenance.source_authority,
                IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_PORTRAIT_FANCAM,
            )

            with self.assertRaisesRegex(IVEFaceGalleryMaterializerError, "9:16"):
                seal_portrait_gallery_sources_from_acquisition(
                    acquisition,
                    media_probe=lambda path: _media(
                        path, width=3840, height=2160
                    ),
                )


class GalleryMaterializerJobTests(unittest.TestCase):
    def _all_member_authorities(self, root: Path):  # type: ignore[no-untyped-def]
        authorities = []
        paths: list[Path] = []
        for member in IVEMember:
            acquisition, video_ids, member_paths = (
                _write_facecam_rejection_manifest(root, member)
            )
            authorities.extend(
                seal_closeup_facecam_sources_from_rejection_manifest(
                    acquisition,
                    video_ids=video_ids,
                    media_probe=lambda path: _media(path),
                )
            )
            paths.extend(member_paths)
        return tuple(authorities), tuple(paths)

    def test_complete_job_is_deterministic_private_and_preserves_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorities, paths = self._all_member_authorities(root)
            before = {
                path: (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
                for path in paths
            }
            policy = IVEFaceGalleryMaterializerConfig(
                start_margin_seconds=1.0,
                end_margin_seconds=1.0,
                minimum_sample_interval_seconds=2.0,
                failed_attempt_retry_seconds=0.5,
                minimum_samples_per_source=3,
                maximum_samples_per_source=4,
                maximum_scanned_frames_per_source=100,
            )
            with patch(
                "crosscut.ive_face_identity_gallery_materializer.extract_authorized_face_sample",
                side_effect=_fake_extract,
            ):
                first = materialize_ive_face_identity_gallery_job(
                    authorities,
                    artifact_directory=root / "private-gallery",
                    config=policy,
                    frame_iterator_factory=_frames,
                )
                second = materialize_ive_face_identity_gallery_job(
                    tuple(reversed(authorities)),
                    artifact_directory=root / "private-gallery",
                    config=policy,
                    frame_iterator_factory=_frames,
                )

            self.assertEqual(first.artifact.path, second.artifact.path)
            self.assertEqual(first.artifact.content_sha256, second.artifact.content_sha256)
            restored = load_ive_face_identity_gallery_artifact(first.artifact.path)
            self.assertEqual(len(restored.gallery.prototypes), 12)
            self.assertEqual(
                {item.member for item in restored.gallery.prototypes}, set(IVEMember)
            )
            audit = first.to_public_audit_dict()
            serialized_audit = json.dumps(audit, ensure_ascii=False)
            self.assertNotIn("embedding", serialized_audit.casefold())
            self.assertNotIn(".invalid", serialized_audit)
            self.assertNotIn(str(root), serialized_audit)
            self.assertFalse(audit["privacy"]["contains_raw_frames"])
            self.assertFalse(audit["privacy"]["contains_identity_vectors"])
            self.assertFalse(audit["privacy"]["stageedit_feature_source"])
            self.assertTrue(
                all(item["sampled_face_count"] == 4 for item in audit["sources"])
            )
            self.assertTrue(
                all(item["first_sample_seconds"] == 1.0 for item in audit["sources"])
            )
            self.assertTrue(
                all(item["last_sample_seconds"] == 7.0 for item in audit["sources"])
            )
            after = {
                path: (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
                for path in paths
            }
            self.assertEqual(before, after)

    def test_changed_media_after_sealing_fails_before_frame_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorities, _paths = self._all_member_authorities(root)
            changed = authorities[0].media_path
            changed.write_bytes(b"changed-after-seal")
            decode_calls = 0

            def frames(path: Path):  # type: ignore[no-untyped-def]
                nonlocal decode_calls
                decode_calls += 1
                return _frames(path)

            with self.assertRaisesRegex(
                IVEFaceGalleryMaterializerError, "changed after sealing"
            ):
                materialize_ive_face_identity_gallery_job(
                    authorities,
                    artifact_directory=root / "private-gallery",
                    frame_iterator_factory=frames,
                )
            self.assertEqual(decode_calls, 0)

    def test_six_member_independent_video_floor_is_checked_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorities, _paths = self._all_member_authorities(root)
            removed = next(
                item
                for item in authorities
                if item.provenance.target_member is IVEMember.LEESEO
            )
            incomplete = tuple(
                item
                for item in authorities
                if item.authority_content_sha256 != removed.authority_content_sha256
            )
            decode_calls = 0

            def frames(path: Path):  # type: ignore[no-untyped-def]
                nonlocal decode_calls
                decode_calls += 1
                return _frames(path)

            with self.assertRaisesRegex(
                IVEFaceGalleryMaterializerError,
                "이서 lacks independent sealed source authorities",
            ):
                materialize_ive_face_identity_gallery_job(
                    incomplete,
                    artifact_directory=root / "private-gallery",
                    frame_iterator_factory=frames,
                )
            self.assertEqual(decode_calls, 0)

    def test_nonsequential_frames_or_insufficient_faces_fail_without_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorities, _paths = self._all_member_authorities(root)
            output = root / "private-gallery"

            def nonsequential(_path: Path):  # type: ignore[no-untyped-def]
                frame = np.zeros((8, 8, 3), dtype=np.uint8)
                yield (0, 1.0, frame)
                yield (1, 0.5, frame)

            with self.assertRaisesRegex(
                IVEFaceGalleryMaterializerError, "not sequential"
            ):
                materialize_ive_face_identity_gallery_job(
                    authorities,
                    artifact_directory=output,
                    config=IVEFaceGalleryMaterializerConfig(
                        start_margin_seconds=0.0,
                        end_margin_seconds=1.0,
                        minimum_sample_interval_seconds=0.4,
                        failed_attempt_retry_seconds=0.1,
                        minimum_samples_per_source=3,
                        maximum_samples_per_source=3,
                        maximum_scanned_frames_per_source=100,
                    ),
                    frame_iterator_factory=nonsequential,
                )

            def reject_face(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                raise IVEFaceGalleryError("no single dominant face")

            with patch(
                "crosscut.ive_face_identity_gallery_materializer.extract_authorized_face_sample",
                side_effect=reject_face,
            ):
                with self.assertRaisesRegex(
                    IVEFaceGalleryMaterializerError, "enough verified"
                ):
                    materialize_ive_face_identity_gallery_job(
                        authorities,
                        artifact_directory=output,
                        config=IVEFaceGalleryMaterializerConfig(
                            start_margin_seconds=0.0,
                            end_margin_seconds=1.0,
                            minimum_sample_interval_seconds=0.4,
                            failed_attempt_retry_seconds=0.1,
                            minimum_samples_per_source=3,
                            maximum_samples_per_source=3,
                            maximum_scanned_frames_per_source=100,
                        ),
                        frame_iterator_factory=_frames,
                    )
            self.assertFalse(any(output.rglob("*.json")) if output.exists() else False)


if __name__ == "__main__":
    unittest.main()
