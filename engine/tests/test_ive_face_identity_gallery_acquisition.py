from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from crosscut.acquisition import (
    YTDLPClient,
    _IVE_STAGE_PLAYLIST_ID,
    _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
)
from crosscut.ive import IVEMember
from crosscut.ive_face_identity_gallery_acquisition import (
    FULL_HD_AVC_COMPATIBILITY_PROFILE,
    IVEFaceGalleryAcquisitionError,
    IVEFaceGalleryAcquisitionSpec,
    IVEFaceGalleryMemberSourceSpec,
    IVEFaceGalleryPinnedSourceSpec,
    acquire_ive_face_identity_gallery_sources,
    coordinate_ive_face_identity_gallery_acquisition,
    is_dedicated_facecam_acquisition_authority,
    revalidate_dedicated_facecam_acquisition_authority,
    _prepare_external_destination,
)
from crosscut.ive_face_identity_gallery_materializer import (
    DEDICATED_FACECAM_ACQUISITION_AUTHORITY_BASIS,
    IVEFaceGalleryMaterializerError,
    materialize_ive_face_identity_gallery_job,
)
from crosscut.media import AudioStreamInfo, MediaInfo, VideoStreamInfo


_SONGS = ("BLACKHOLE", "BANG BANG", "XOXZ")
_PROGRAMS = (
    ("M COUNTDOWN", "M2", "UCbD8EppRX3ZwJSou-TVo90A"),
    ("Music Bank", "KBS Kpop", "UCeLPm9yH_a_QH8n6445G-Ow"),
    ("Inkigayo", "SBS KPOP", "UCM3jwNRfl5-W8VzgT6DsaEQ"),
)


def _video_id(member_index: int, source_index: int) -> str:
    return f"FG{member_index}{source_index}video{member_index}"


def _spec(*, count: int = 3, songs: tuple[str, ...] = _SONGS):
    return IVEFaceGalleryAcquisitionSpec(
        gallery_id="ive-universal-face-gallery-2026-v1",
        members=tuple(
            IVEFaceGalleryMemberSourceSpec(
                member=member,
                sources=tuple(
                    IVEFaceGalleryPinnedSourceSpec(
                        video_id=_video_id(member_index, source_index),
                        song=songs[source_index],
                    )
                    for source_index in range(count)
                ),
            )
            for member_index, member in enumerate(IVEMember)
        ),
    )


def _metadata_for_spec(spec: IVEFaceGalleryAcquisitionSpec):
    result: dict[str, dict[str, object]] = {}
    for member_spec in spec.members:
        for source_index, source in enumerate(member_spec.sources):
            program, channel, channel_id = _PROGRAMS[source_index]
            result[source.video_id] = {
                "id": source.video_id,
                "title": (
                    f"[FaceCam] IVE 아이브 {member_spec.member.value} "
                    f"{source.song} | {program} 2026.08.{source_index + 1:02d}"
                ),
                "channel": channel,
                "channel_id": channel_id,
                "duration": 180.0,
                "upload_date": f"202608{source_index + 1:02d}",
                "availability": "public",
                "live_status": "not_live",
                "has_drm": False,
                "webpage_url": (
                    f"https://www.youtube.com/watch?v={source.video_id}"
                ),
            }
    return result


class _FakeYTDLPRunner:
    def __init__(self, metadata: dict[str, dict[str, object]]) -> None:
        self.metadata = metadata
        self.calls: list[tuple[str, ...]] = []
        self.catalog_calls = 0
        self.metadata_calls = 0
        self.download_calls = 0
        self.fail_download_video_id: str | None = None

    def __call__(self, command, **kwargs):
        command = tuple(str(item) for item in command)
        self.calls.append(command)
        if "--flat-playlist" in command:
            self.catalog_calls += 1
            payload = {
                "id": _IVE_STAGE_PLAYLIST_ID,
                "title": "IVE STAGE",
                "channel": "IVE",
                "channel_id": _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
                "entries": [
                    {
                        "id": video_id,
                        "title": metadata["title"],
                        "channel": metadata["channel"],
                        "channel_id": metadata["channel_id"],
                        "duration": metadata["duration"],
                        "webpage_url": metadata["webpage_url"],
                    }
                    for video_id, metadata in sorted(self.metadata.items())
                ],
            }
            return subprocess.CompletedProcess(
                command, 0, json.dumps(payload), ""
            )
        url = command[-1]
        video_id = url.rsplit("=", 1)[-1]
        if "--skip-download" in command:
            self.metadata_calls += 1
            payload = self.metadata.get(video_id)
            if payload is None:
                return subprocess.CompletedProcess(command, 1, "", "missing")
            return subprocess.CompletedProcess(
                command, 0, json.dumps(payload), ""
            )
        if "--output" in command:
            self.download_calls += 1
            template = command[command.index("--output") + 1]
            output = Path(template.replace("%(ext)s", "mp4"))
            output.parent.mkdir(parents=True, exist_ok=True)
            if video_id == self.fail_download_video_id:
                output.with_suffix(".mp4.part").write_bytes(b"partial")
                return subprocess.CompletedProcess(command, 1, "", "forced")
            output.write_bytes((f"media-{video_id}-" * 32).encode("utf-8"))
            return subprocess.CompletedProcess(command, 0, f"{output}\n", "")
        raise AssertionError(f"unexpected yt-dlp command: {command!r}")


def _client(runner: _FakeYTDLPRunner) -> YTDLPClient:
    return YTDLPClient(
        executable="fake-yt-dlp",
        runner=runner,
        download_profile=FULL_HD_AVC_COMPATIBILITY_PROFILE,
    )


def _media(path: Path) -> MediaInfo:
    duration = 180.0
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
            VideoStreamInfo(0, "h264", 1920, 1080, 29.97, duration, 8_000_000),
        ),
        tags={},
    )


def _fixture_external_destination(destination: Path) -> Path:
    private = destination / ".ive-face-gallery-acquisition-v1"
    private.mkdir(parents=True, exist_ok=True)
    return private.resolve(strict=True)


class IVEFaceIdentityGalleryAcquisitionTests(unittest.TestCase):
    def test_attitude_is_rejected_before_any_runner_or_catalog_call(self) -> None:
        runner = _FakeYTDLPRunner({})
        with self.assertRaisesRegex(
            IVEFaceGalleryAcquisitionError, "ATTITUDE is strictly quarantined"
        ):
            IVEFaceGalleryPinnedSourceSpec(
                video_id="ATT000video", song=" At Ti Tu De "
            )
        self.assertEqual(runner.calls, [])
        self.assertEqual(runner.catalog_calls, 0)

    def test_spec_requires_six_members_two_sources_and_unique_ownership(self) -> None:
        valid = _spec(count=2)
        self.assertEqual(len(valid.members), 6)
        too_few = list(valid.members)
        with self.assertRaisesRegex(IVEFaceGalleryAcquisitionError, "at least two"):
            IVEFaceGalleryMemberSourceSpec(
                member=IVEMember.GAEUL,
                sources=(
                    IVEFaceGalleryPinnedSourceSpec(
                        video_id="onevideo01", song="BLACKHOLE"
                    ),
                ),
            )
        duplicate = list(valid.members)
        duplicate[1] = IVEFaceGalleryMemberSourceSpec(
            member=IVEMember.GAEUL,
            sources=(
                duplicate[0].sources[0],
                IVEFaceGalleryPinnedSourceSpec(
                    video_id="uniquevid02", song="BANG BANG"
                ),
            ),
        )
        with self.assertRaisesRegex(
            IVEFaceGalleryAcquisitionError, "cannot authorize multiple members"
        ):
            replace(valid, members=tuple(duplicate))

    def test_external_destination_rejects_internal_and_symlink_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                IVEFaceGalleryAcquisitionError, "external /Volumes"
            ):
                _prepare_external_destination(root / "not-external")
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(IVEFaceGalleryAcquisitionError):
                _prepare_external_destination(link / "gallery")

    def test_full_six_member_three_song_acquisition_resume_and_coordinator(self) -> None:
        spec = _spec()
        metadata = _metadata_for_spec(spec)
        closeup_id = spec.members[0].sources[0].video_id
        metadata[closeup_id]["title"] = str(
            metadata[closeup_id]["title"]
        ).replace("[FaceCam]", "[Close-Up]")
        runner = _FakeYTDLPRunner(metadata)
        progress = []
        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            destination = Path(temporary) / "external-fixture"
            coordinated = coordinate_ive_face_identity_gallery_acquisition(
                spec,
                destination=destination,
                client=_client(runner),
                media_probe=_media,
                progress=progress.append,
            )
            first = coordinated.acquisition
            self.assertEqual(len(first.authorities), 18)
            self.assertEqual(len(coordinated.materializer_authorities), 18)
            self.assertEqual(first.downloaded_source_count, 18)
            self.assertEqual(first.reused_source_count, 0)
            self.assertEqual(runner.download_calls, 18)
            self.assertTrue(
                all(
                    authority.acquisition_authority_basis
                    == DEDICATED_FACECAM_ACQUISITION_AUTHORITY_BASIS
                    for authority in coordinated.materializer_authorities
                )
            )
            second = acquire_ive_face_identity_gallery_sources(
                spec,
                destination=destination,
                client=_client(runner),
                media_probe=_media,
            )
            self.assertEqual(second.reused_source_count, 18)
            self.assertEqual(second.downloaded_source_count, 0)
            self.assertEqual(runner.download_calls, 18)
            self.assertEqual(second.manifest_path, first.manifest_path)
            self.assertEqual(
                second.manifest_content_sha256, first.manifest_content_sha256
            )
            self.assertEqual(first.manifest_path.stem, first.manifest_content_sha256)
            self.assertEqual(first.manifest_path.stat().st_mode & 0o777, 0o400)
            self.assertTrue(
                all(
                    is_dedicated_facecam_acquisition_authority(authority)
                    for authority in second.authorities
                )
            )
            revalidate_dedicated_facecam_acquisition_authority(
                second.authorities[0], media_probe=_media
            )
            manifest = json.loads(second.manifest_path.read_text("utf-8"))
            self.assertEqual(manifest["raw_frame_count"], 0)
            self.assertEqual(manifest["raw_embedding_count"], 0)
            self.assertEqual(
                {source["song"] for source in manifest["sources"]}, set(_SONGS)
            )
        fractions = [item.fraction for item in progress]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)
        self.assertEqual(
            {item.song for item in progress if item.video_id is not None},
            set(_SONGS),
        )
        self.assertEqual(runner.catalog_calls, 2)
        self.assertEqual(runner.metadata_calls, 36)
        single_video_calls = [
            call
            for call in runner.calls
            if "--skip-download" in call or "--output" in call
        ]
        self.assertTrue(single_video_calls)
        for call in single_video_calls:
            self.assertIn("--no-cookies", call)
            self.assertIn("--no-playlist", call)
            self.assertNotIn("--cookies-from-browser", call)
            self.assertNotIn("--yes-playlist", call)

    def test_coordinator_rejects_wrong_three_song_corpus_before_runner(self) -> None:
        invalid = _spec(songs=("BLACKHOLE", "BANG BANG", "BANG BANG"))
        runner = _FakeYTDLPRunner(_metadata_for_spec(invalid))
        with self.assertRaisesRegex(
            IVEFaceGalleryAcquisitionError,
            "BLACKHOLE/BANG BANG/XOXZ",
        ):
            coordinate_ive_face_identity_gallery_acquisition(
                invalid,
                destination=Path("/never/reached"),
                client=_client(runner),
                media_probe=_media,
            )
        self.assertEqual(runner.calls, [])

    def test_duplicate_legacy_video_prefers_fresh_dedicated_and_core_rejects_it(self) -> None:
        legacy_spec = replace(
            _spec(count=2), gallery_id="ive-gallery-legacy-transcode-fixture"
        )
        current_spec = replace(
            _spec(count=2), gallery_id="ive-gallery-current-dedicated-fixture"
        )
        runner = _FakeYTDLPRunner(_metadata_for_spec(current_spec))
        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            destination = Path(temporary)
            legacy = coordinate_ive_face_identity_gallery_acquisition(
                legacy_spec,
                destination=destination,
                client=_client(runner),
                media_probe=_media,
            )
            legacy_paths = tuple(
                authority.media_path
                for authority in legacy.materializer_authorities
            )
            legacy_bytes = tuple(path.read_bytes() for path in legacy_paths)
            current = coordinate_ive_face_identity_gallery_acquisition(
                current_spec,
                destination=destination,
                client=_client(runner),
                media_probe=_media,
                legacy_materializer_authorities=(
                    legacy.materializer_authorities
                ),
            )
            self.assertEqual(len(current.materializer_authorities), 12)
            self.assertEqual(
                {
                    authority.provenance.video_id
                    for authority in current.materializer_authorities
                },
                {
                    source.video_id
                    for member_spec in current_spec.members
                    for source in member_spec.sources
                },
            )
            self.assertTrue(
                all(
                    authority.manifest_path == current.acquisition.manifest_path
                    for authority in current.materializer_authorities
                )
            )
            self.assertEqual(
                tuple(path.read_bytes() for path in legacy_paths), legacy_bytes
            )
            with self.assertRaisesRegex(
                IVEFaceGalleryMaterializerError, "provenance video_id"
            ):
                materialize_ive_face_identity_gallery_job(
                    (
                        *current.materializer_authorities,
                        legacy.materializer_authorities[0],
                    ),
                    artifact_directory=destination / "must-not-materialize",
                )

    def test_legacy_video_cross_member_ownership_is_rejected(self) -> None:
        legacy_spec = replace(
            _spec(count=2), gallery_id="ive-gallery-legacy-owner-fixture"
        )
        swapped_members = list(legacy_spec.members)
        swapped_members[0] = IVEFaceGalleryMemberSourceSpec(
            member=legacy_spec.members[0].member,
            sources=legacy_spec.members[1].sources,
        )
        swapped_members[1] = IVEFaceGalleryMemberSourceSpec(
            member=legacy_spec.members[1].member,
            sources=legacy_spec.members[0].sources,
        )
        current_spec = replace(
            legacy_spec,
            gallery_id="ive-gallery-current-owner-fixture",
            members=tuple(swapped_members),
        )
        runner = _FakeYTDLPRunner(_metadata_for_spec(legacy_spec))
        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            destination = Path(temporary)
            legacy = coordinate_ive_face_identity_gallery_acquisition(
                legacy_spec,
                destination=destination,
                client=_client(runner),
                media_probe=_media,
            )
            runner.metadata = _metadata_for_spec(current_spec)
            with self.assertRaisesRegex(
                IVEFaceGalleryAcquisitionError,
                "cannot be owned by two members",
            ):
                coordinate_ive_face_identity_gallery_acquisition(
                    current_spec,
                    destination=destination,
                    client=_client(runner),
                    media_probe=_media,
                    legacy_materializer_authorities=(
                        legacy.materializer_authorities
                    ),
                )

    def test_ordinary_landscape_fullcam_is_rejected_before_download(self) -> None:
        spec = _spec(count=2)
        metadata = _metadata_for_spec(spec)
        bad_id = spec.members[0].sources[0].video_id
        metadata[bad_id]["title"] = (
            "[FANCAM] IVE 아이브 안유진 BLACKHOLE FullCam "
            "M COUNTDOWN 2026.08.01"
        )
        runner = _FakeYTDLPRunner(metadata)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            with self.assertRaisesRegex(
                IVEFaceGalleryAcquisitionError, "ordinary landscape fullcam"
            ):
                acquire_ive_face_identity_gallery_sources(
                    spec,
                    destination=Path(temporary),
                    client=_client(runner),
                    media_probe=_media,
                )
        self.assertEqual(runner.download_calls, 0)

    def test_generic_landscape_fancam_without_closeup_signal_is_rejected(self) -> None:
        spec = _spec(count=2)
        metadata = _metadata_for_spec(spec)
        bad_id = spec.members[0].sources[0].video_id
        metadata[bad_id]["title"] = (
            "[FANCAM] IVE 아이브 안유진 BLACKHOLE "
            "M COUNTDOWN 2026.08.01"
        )
        runner = _FakeYTDLPRunner(metadata)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            with self.assertRaisesRegex(
                IVEFaceGalleryAcquisitionError, "lacks an explicit FaceCam"
            ):
                acquire_ive_face_identity_gallery_sources(
                    spec,
                    destination=Path(temporary),
                    client=_client(runner),
                    media_probe=_media,
                )
        self.assertEqual(runner.download_calls, 0)

    def test_missing_catalog_id_rejects_without_single_video_or_download(self) -> None:
        spec = _spec(count=2)
        metadata = _metadata_for_spec(spec)
        metadata.pop(spec.members[-1].sources[-1].video_id)
        runner = _FakeYTDLPRunner(metadata)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            with self.assertRaisesRegex(
                IVEFaceGalleryAcquisitionError, "absent from pinned"
            ):
                acquire_ive_face_identity_gallery_sources(
                    spec,
                    destination=Path(temporary),
                    client=_client(runner),
                    media_probe=_media,
                )
        self.assertEqual(runner.metadata_calls, 0)
        self.assertEqual(runner.download_calls, 0)

    def test_failed_partial_is_quarantined_with_immutable_evidence(self) -> None:
        spec = _spec(count=2)
        metadata = _metadata_for_spec(spec)
        runner = _FakeYTDLPRunner(metadata)
        runner.fail_download_video_id = spec.members[0].sources[0].video_id
        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            destination = Path(temporary)
            with self.assertRaisesRegex(
                IVEFaceGalleryAcquisitionError, "forced"
            ):
                acquire_ive_face_identity_gallery_sources(
                    spec,
                    destination=destination,
                    client=_client(runner),
                    media_probe=_media,
                )
            job = (
                destination
                / ".ive-face-gallery-acquisition-v1"
                / spec.request_sha256
            )
            quarantined = list((job / "quarantine").iterdir())
            failures = list((job / "failures").glob("*/*.json"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(len(failures), 1)
            self.assertIn(".part.invalid.", quarantined[0].name)
            evidence = json.loads(failures[0].read_text("utf-8"))
            self.assertEqual(len(evidence["quarantined_artifacts"]), 1)

    def test_manifest_and_media_tamper_are_rejected_without_redownload(self) -> None:
        spec = _spec(count=2)
        metadata = _metadata_for_spec(spec)
        runner = _FakeYTDLPRunner(metadata)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            destination = Path(temporary)
            first = acquire_ive_face_identity_gallery_sources(
                spec,
                destination=destination,
                client=_client(runner),
                media_probe=_media,
            )
            downloads = runner.download_calls
            media_path = first.authorities[0].media_path
            media_path.write_bytes(media_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(
                IVEFaceGalleryAcquisitionError, "no longer matches current media"
            ):
                revalidate_dedicated_facecam_acquisition_authority(
                    first.authorities[0], media_probe=_media
                )
            self.assertEqual(runner.download_calls, downloads)

            os.chmod(first.manifest_path, 0o600)
            first.manifest_path.write_bytes(
                first.manifest_path.read_bytes() + b"\n"
            )
            with self.assertRaises(IVEFaceGalleryAcquisitionError):
                revalidate_dedicated_facecam_acquisition_authority(
                    first.authorities[1], media_probe=_media
                )
            self.assertEqual(runner.download_calls, downloads)

    def test_active_reference_contract_tamper_is_rejected(self) -> None:
        spec = _spec(count=2)
        runner = _FakeYTDLPRunner(_metadata_for_spec(spec))
        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            destination = Path(temporary)
            first = acquire_ive_face_identity_gallery_sources(
                spec,
                destination=destination,
                client=_client(runner),
                media_probe=_media,
            )
            downloads = runner.download_calls
            active = first.manifest_path.parents[2] / "active-manifest.json"
            reference = json.loads(active.read_text("utf-8"))
            reference["contract"] = "forged-active-reference-v1"
            active.write_text(json.dumps(reference), encoding="utf-8")
            with self.assertRaisesRegex(
                IVEFaceGalleryAcquisitionError, "active acquisition reference"
            ):
                acquire_ive_face_identity_gallery_sources(
                    spec,
                    destination=destination,
                    client=_client(runner),
                    media_probe=_media,
                )
            self.assertEqual(runner.download_calls, downloads)

    def test_media_mutation_during_ffprobe_is_rejected_and_quarantined(self) -> None:
        spec = _spec(count=2)
        runner = _FakeYTDLPRunner(_metadata_for_spec(spec))
        mutated = False

        def mutating_probe(path: Path) -> MediaInfo:
            nonlocal mutated
            info = _media(path)
            if not mutated:
                path.write_bytes(path.read_bytes() + b"probe-race")
                mutated = True
            return info

        with tempfile.TemporaryDirectory() as temporary, patch(
            "crosscut.ive_face_identity_gallery_acquisition._prepare_external_destination",
            side_effect=_fixture_external_destination,
        ):
            destination = Path(temporary)
            with self.assertRaisesRegex(
                IVEFaceGalleryAcquisitionError, "changed while ffprobe"
            ):
                acquire_ive_face_identity_gallery_sources(
                    spec,
                    destination=destination,
                    client=_client(runner),
                    media_probe=mutating_probe,
                )
            job = (
                destination
                / ".ive-face-gallery-acquisition-v1"
                / spec.request_sha256
            )
            self.assertEqual(len(list((job / "quarantine").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
