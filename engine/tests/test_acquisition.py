from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from crosscut.acquisition import (
    AcquisitionError,
    YTDLPClient,
    _IVE_STAGE_PLAYLIST_ID,
    _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
    _IVE_STAGE_PLAYLIST_URL,
    _CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT,
    _load_official_ive_stage_catalog,
    _merge_discovery,
    _metadata_candidate,
    _metadata_queries,
    _is_member_closeup_candidate,
    _manifest_matches_cache_binding_contract,
    _ordered_for_download,
    _broadcast_date,
    acquire_sources,
)
from crosscut.media import AudioStreamInfo, MediaInfo, VideoStreamInfo
from crosscut.request import AnalysisRequest


def _request(
    root: Path,
    *,
    member: bool = False,
    desired: int | None = None,
    song: str = "I AM",
) -> AnalysisRequest:
    payload: dict[str, object] = {
        "song": song,
        "edit_mode": "member" if member else "group",
        "aspect_ratio": "9:16" if member else "16:9",
        "target_member": "가을" if member else None,
        "source_mode": "automatic",
        "desired_source_count": desired if desired is not None else (10 if member else 9),
        "sources": [],
        "output_directory": str(root / "out"),
        "source_cache_directory": str(root / "cache"),
    }
    return AnalysisRequest.from_dict(payload)


def _entry(index: int, *, title: str | None = None) -> dict[str, object]:
    day = 10 + index
    return {
        "id": f"stage{index:06d}",
        "title": title or f"IVE - I AM | Music Bank Stage 2304{day:02d}",
        "channel": "KBS Kpop",
        "channel_id": "UCeLPm9yH_a_QH8n6445G-Ow",
        "duration": 185 + index % 3,
        "webpage_url": f"https://www.youtube.com/watch?v=stage{index:06d}",
    }


def _catalog_payload(
    entries: list[dict[str, object]],
    *,
    owner_channel_id: str = _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
) -> dict[str, object]:
    return {
        "id": _IVE_STAGE_PLAYLIST_ID,
        "title": "IVE STAGE",
        "channel": "IVE",
        "channel_id": owner_channel_id,
        "entries": deepcopy(entries),
    }


def _portrait_media(path: Path) -> MediaInfo:
    return MediaInfo(
        path=str(path),
        format_name="mp4",
        duration_seconds=185.0,
        start_time_seconds=0.0,
        size_bytes=1,
        bit_rate=1,
        audio_streams=(
            AudioStreamInfo(1, "aac", 48_000, 2, "stereo", 185.0, 128_000),
        ),
        video_streams=(
            VideoStreamInfo(0, "h264", 1080, 1920, 30.0, 185.0, 4_000_000),
        ),
        tags={},
    )


def _landscape_media(path: Path) -> MediaInfo:
    return MediaInfo(
        path=str(path),
        format_name="mp4",
        duration_seconds=185.0,
        start_time_seconds=0.0,
        size_bytes=1,
        bit_rate=1,
        audio_streams=(
            AudioStreamInfo(1, "aac", 48_000, 2, "stereo", 185.0, 128_000),
        ),
        video_streams=(
            VideoStreamInfo(0, "h264", 1920, 1080, 30.0, 185.0, 4_000_000),
        ),
        tags={},
    )


class _FakeClient:
    executable = "/mock/yt-dlp"

    def __init__(self, entries: list[dict[str, object]], failed_ids: set[str] | None = None) -> None:
        self.entries = entries
        self.failed_ids = failed_ids or set()
        self.download_attempts: list[str] = []

    def search(self, query: str, limit: int) -> list[dict[str, object]]:
        del query, limit
        return list(self.entries)

    def playlist_catalog(self, playlist_id: str) -> dict[str, object]:
        if playlist_id != _IVE_STAGE_PLAYLIST_ID:
            raise AssertionError(f"unexpected playlist: {playlist_id}")
        return _catalog_payload(self.entries)

    def download(self, candidate, directory: Path) -> Path:
        self.download_attempts.append(candidate.video_id)
        if candidate.video_id in self.failed_ids:
            raise AcquisitionError(f"mock failure {candidate.video_id}")
        path = directory / "videos" / f"{candidate.video_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path.resolve()


class _CatalogClient(_FakeClient):
    def __init__(
        self,
        entries: list[dict[str, object]],
        *,
        catalog_entries: list[dict[str, object]] | None = None,
        catalog_payload: dict[str, object] | None = None,
        catalog_error: str | None = None,
        failed_ids: set[str] | None = None,
    ) -> None:
        super().__init__(entries, failed_ids=failed_ids)
        self.catalog_payload = catalog_payload or _catalog_payload(
            catalog_entries or []
        )
        self.catalog_error = catalog_error
        self.catalog_attempts = 0

    def playlist_catalog(self, playlist_id: str) -> dict[str, object]:
        self.catalog_attempts += 1
        if playlist_id != _IVE_STAGE_PLAYLIST_ID:
            raise AssertionError(f"unexpected playlist: {playlist_id}")
        if self.catalog_error is not None:
            raise AcquisitionError(self.catalog_error)
        return deepcopy(self.catalog_payload)


class AcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Production requires a plausibly complete 1,500-entry catalog. Small
        # fixtures exercise the same logic with a one-entry floor; dedicated
        # anti-regression tests below restore the production threshold.
        floor_patch = patch(
            "crosscut.acquisition._CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT",
            1,
        )
        floor_patch.start()
        self.addCleanup(floor_patch.stop)

    def test_broadcast_date_parses_compact_dotted_and_underscored_titles(self) -> None:
        self.assertEqual(_broadcast_date("Music Bank 230414"), "2023-04-14")
        self.assertEqual(
            _broadcast_date("@MCOUNTDOWN_2023.4.13"),
            "2023-04-13",
        )
        self.assertEqual(
            _broadcast_date("SBS Inkigayo_2023.04.16"),
            "2023-04-16",
        )

    def test_ytdlp_search_uses_no_config_no_cookies_and_ytsearch(self) -> None:
        calls: list[tuple[str, ...]] = []

        def runner(command, **kwargs):
            calls.append(tuple(command))
            self.assertFalse(kwargs["check"])
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"entries": [_entry(1)]}),
                stderr="",
            )

        client = YTDLPClient("/mock/yt-dlp", runner=runner)
        entries = client.search('IVE "I AM" stage', 20)
        self.assertEqual(len(entries), 1)
        self.assertIn("--no-config", calls[0])
        self.assertIn("--no-cookies", calls[0])
        self.assertTrue(calls[0][-1].startswith("ytsearch20:"))

    def test_ytdlp_playlist_catalog_is_flat_cookie_free_and_pinned(self) -> None:
        calls: list[tuple[str, ...]] = []

        def runner(command, **kwargs):
            calls.append(tuple(command))
            self.assertFalse(kwargs["check"])
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(_catalog_payload([_entry(1)])),
                stderr="",
            )

        client = YTDLPClient("/mock/yt-dlp", runner=runner)
        payload = client.playlist_catalog(_IVE_STAGE_PLAYLIST_ID)
        self.assertEqual(payload["id"], _IVE_STAGE_PLAYLIST_ID)
        self.assertEqual(calls[0][-1], _IVE_STAGE_PLAYLIST_URL)
        self.assertIn("--no-config", calls[0])
        self.assertIn("--no-cookies", calls[0])
        self.assertIn("--ignore-errors", calls[0])
        self.assertIn("--flat-playlist", calls[0])
        self.assertIn("--dump-single-json", calls[0])
        self.assertNotIn("--no-playlist", calls[0])

    def test_ytdlp_nonzero_partial_playlist_payload_is_not_cacheable(self) -> None:
        def runner(command, **kwargs):
            del kwargs
            return subprocess.CompletedProcess(
                command,
                1,
                stdout=json.dumps(_catalog_payload([_entry(1)])),
                stderr="one playlist entry failed",
            )

        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp), desired=8)
            entries, provenance = _load_official_ive_stage_catalog(
                request=request,
                client=YTDLPClient("/mock/yt-dlp", runner=runner),
                now=datetime(2026, 8, 24, 1, 0, tzinfo=UTC),
            )

        self.assertEqual(entries, [])
        self.assertEqual(provenance["network_status"], "incomplete_or_regressed")
        self.assertIn("nonzero", provenance["network_error"])

    def test_ytdlp_client_refuses_blind_preexisting_cache_reuse(self) -> None:
        calls: list[tuple[str, ...]] = []

        def runner(command, **kwargs):
            del kwargs
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root)
            candidate = _metadata_candidate(
                _entry(1),
                request=request,
                query="cache-contract",
            )
            assert candidate is not None
            videos = root / "cache" / "videos"
            videos.mkdir(parents=True)
            (videos / f"{candidate.video_id}.mp4").write_bytes(b"untrusted")
            client = YTDLPClient("/mock/yt-dlp", runner=runner)

            with self.assertRaisesRegex(AcquisitionError, "인증되지 않은"):
                client.download(candidate, root / "cache")

        self.assertEqual(calls, [])

    def test_ytdlp_professional_profile_prefers_4k_pixels_with_fhd_fallback(self) -> None:
        calls: list[tuple[str, ...]] = []

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root)
            candidate = _metadata_candidate(
                _entry(1),
                request=request,
                query="professional-download-profile",
            )
            assert candidate is not None
            output = root / "cache" / "videos" / f"{candidate.video_id}.mp4"

            def runner(command, **kwargs):
                del kwargs
                calls.append(tuple(command))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=f"{output}\n",
                    stderr="",
                )

            client = YTDLPClient("/mock/yt-dlp", runner=runner)
            actual = client.download(candidate, root / "cache")

        self.assertEqual(output.resolve(), actual)
        self.assertIn("--no-config", calls[0])
        self.assertIn("--no-cookies", calls[0])
        self.assertIn("--no-playlist", calls[0])
        format_value = calls[0][calls[0].index("--format") + 1]
        self.assertIn("width<=3840", format_value)
        self.assertIn("height<=3840", format_value)
        self.assertIn("vcodec^=avc1", format_value)
        self.assertLess(
            format_value.index("width<=3840"),
            format_value.index("vcodec^=avc1"),
        )

    def test_queries_include_each_major_music_show(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            queries = _metadata_queries(_request(Path(temp)))
        joined = " ".join(queries).casefold()
        for marker in (
            "m countdown",
            "music bank",
            "music core",
            "inkigayo",
            "the show",
            "show champion",
        ):
            self.assertIn(marker, joined)

    def test_song_alias_registry_covers_every_korean_title_and_promoted_track(
        self,
    ) -> None:
        cases = (
            ("ELEVEN", "일레븐"),
            ("LOVE DIVE", "러브다이브"),
            ("After LIKE", "애프터라이크"),
            ("Kitsch", "키치"),
            ("I AM", "아이 엠"),
            ("Either Way", "이더웨이"),
            ("Off The Record", "오프더레코드"),
            ("Baddie", "배디"),
            ("HEYA", "해야"),
            ("Accendio", "아센디오"),
            ("REBEL HEART", "레벨 하트"),
            ("ATTITUDE", "애티튜드"),
            ("XOXZ", "엑스오엑스지"),
            ("BANG BANG", "방방"),
            ("BLACKHOLE", "블랙홀"),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, (song, title_alias) in enumerate(cases):
                with self.subTest(song=song, title_alias=title_alias):
                    request = _request(root, song=song)
                    candidate = _metadata_candidate(
                        _entry(
                            index,
                            title=(
                                f"IVE {title_alias} Music Bank Stage 230414"
                            ),
                        ),
                        request=request,
                        query="alias-contract",
                    )
                    assert candidate is not None
                    self.assertTrue(candidate.selectable)

    def test_unknown_song_alias_fallback_remains_exact_and_conservative(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp), song="Imaginary Song")
            exact = _metadata_candidate(
                _entry(
                    30,
                    title="IVE Imaginary Song Music Bank Stage 230414",
                ),
                request=request,
                query="alias-contract",
            )
            near_match = _metadata_candidate(
                _entry(
                    31,
                    title="IVE Imaginary Songs Music Bank Stage 230421",
                ),
                request=request,
                query="alias-contract",
            )
        assert exact is not None and near_match is not None
        self.assertTrue(exact.selectable)
        self.assertIn("song_title_signal_missing", near_match.rejection_reasons)

    def test_filters_mv_practice_stagemix_and_concert(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp))
            for index, label in enumerate(
                ("Official MV", "Dance Practice", "Stage Mix", "Festival Concert")
            ):
                candidate = _metadata_candidate(
                    _entry(index, title=f"IVE I AM {label} stage 23041{index}"),
                    request=request,
                    query="test",
                )
                assert candidate is not None
                self.assertFalse(candidate.selectable, label)
                self.assertTrue(
                    any(reason.startswith("non_music_show_") for reason in candidate.rejection_reasons)
                )

    def test_ive_song_alias_and_member_title_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp), member=True)
            accepted = _metadata_candidate(
                {
                    **_entry(1),
                    "title": "아이브 가을 아이엠 직캠 Music Bank 230414",
                },
                request=request,
                query="test",
            )
            wrong_member = _metadata_candidate(
                {
                    **_entry(2),
                    "title": "아이브 레이 아이엠 직캠 Music Bank 230421",
                },
                request=request,
                query="test",
            )
        assert accepted is not None and wrong_member is not None
        self.assertTrue(accepted.selectable)
        self.assertIn("target_member_title_signal_missing", wrong_member.rejection_reasons)
        self.assertIn(
            "other_member_title_signal_without_target",
            wrong_member.rejection_reasons,
        )

    def test_order_prefers_clean_canonical_and_unique_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp))
            raw = [
                _entry(1, title="IVE I AM Fullcam Music Bank 230414"),
                _entry(2, title="IVE I AM Stage Music Bank 230414"),
                _entry(3, title="IVE I AM Stage Music Bank 230421"),
            ]
            candidates = [
                _metadata_candidate(item, request=request, query="test") for item in raw
            ]
        ordered = _ordered_for_download([item for item in candidates if item is not None])
        self.assertEqual(len(ordered), 3)
        self.assertEqual(ordered[0].video_id, "stage000002")
        self.assertEqual(ordered[1].broadcast_date, "2023-04-21")
        self.assertEqual(ordered[2].video_id, "stage000001")

    def test_order_demotes_skycam_until_after_clean_broadcasts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp))
            raw = [
                _entry(
                    1,
                    title="IVE I AM Sky Cam Inkigayo 230416",
                ),
                _entry(
                    2,
                    title="IVE I AM Stage Music Bank 230414",
                ),
                _entry(
                    3,
                    title="IVE I AM Stage Music Bank 230421",
                ),
            ]
            candidates = [
                _metadata_candidate(item, request=request, query="test")
                for item in raw
            ]

        ordered = _ordered_for_download(
            [item for item in candidates if item is not None]
        )
        self.assertNotEqual(ordered[0].video_id, "stage000001")
        self.assertEqual(ordered[-1].video_id, "stage000001")

    def test_download_failures_continue_and_manifest_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=9)
            entries = [_entry(index) for index in range(11)]
            client = _FakeClient(entries, {"stage000000", "stage000001"})
            artifact = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(artifact.source_paths), 9)
        self.assertEqual(manifest["summary"]["downloaded_source_count"], 9)
        self.assertEqual(manifest["summary"]["failed_download_count"], 2)
        self.assertEqual(len(manifest["selected"]), 9)
        selected = manifest["selected"][0]
        for key in (
            "video_id",
            "title",
            "channel",
            "duration_seconds",
            "webpage_url",
            "query",
            "broadcast_date",
            "download_status",
            "local_path",
        ):
            self.assertIn(key, selected)

    def test_fails_clearly_when_requested_count_cannot_be_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=9)
            client = _FakeClient([_entry(index) for index in range(9)], {"stage000000"})
            with self.assertRaisesRegex(AcquisitionError, "최소 음방 소스 9개"):
                acquire_sources(
                    request,
                    reporter=_MemoryReporter(),
                    client=client,  # type: ignore[arg-type]
                    media_validator=lambda path: None,
                    media_probe=_landscape_media,
                )
            manifest = json.loads(
                (root / "cache" / "acquisition-manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(manifest["summary"]["status"], "insufficient_downloaded_sources")

    def test_member_acquisition_defers_empty_anchors_to_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, member=True, desired=10)
            entries = [
                {
                    **_entry(index),
                    "title": f"IVE 가을 I AM Fancam Music Bank 2304{10 + index:02d}",
                }
                for index in range(10)
            ]

            artifact = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=_FakeClient(entries),  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_portrait_media,
            )
            materialized = request.with_materialized_sources(
                artifact.source_paths,
                member_anchors=artifact.member_anchors,
            )
            manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact.member_anchors, ())
        self.assertEqual(materialized.member_anchors, ())
        self.assertEqual(len(manifest["selected"]), 10)
        self.assertTrue(
            all("automatic_member_anchor" not in item for item in manifest["selected"])
        )
        self.assertNotIn("auto_anchor_rejected_count", manifest["summary"])

    def test_member_closeup_title_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp), member=True)
            cases = {
                "IVE 가을 I AM Fancam Music Bank 230414": False,
                "IVE 가을 I AM 직캠 Music Bank 230415": False,
                "IVE 가을 I AM FaceCam Music Bank 230416": True,
                "IVE 가을 I AM 얼빡직캠 Music Bank 230417": True,
                "IVE 가을 I AM Close-Up Fancam Music Bank 230418": True,
                "IVE 가을 I AM 클로즈업 직캠 Music Bank 230419": True,
            }
            for index, (title, expected) in enumerate(cases.items()):
                with self.subTest(title=title):
                    candidate = _metadata_candidate(
                        _entry(index, title=title),
                        request=request,
                        query="test",
                    )
                    assert candidate is not None
                    self.assertEqual(
                        _is_member_closeup_candidate(candidate),
                        expected,
                    )

    def test_member_acquisition_prefers_ten_standard_fancams(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, member=True, desired=10)
            standard_entries = [
                _entry(
                    index,
                    title=(
                        f"IVE 가을 I AM Fancam Music Bank "
                        f"2304{10 + index:02d}"
                    ),
                )
                for index in range(10)
            ]
            closeup_entries = [
                _entry(
                    20 + index,
                    title=(
                        f"IVE 가을 I AM FaceCam Music Bank "
                        f"2304{20 + index:02d}"
                    ),
                )
                for index in range(3)
            ]
            client = _FakeClient([*closeup_entries, *standard_entries])

            artifact = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_portrait_media,
            )

        expected = {f"stage{index:06d}" for index in range(10)}
        self.assertEqual(set(client.download_attempts), expected)
        self.assertEqual(
            {item["video_id"] for item in artifact.manifest["selected"]},
            expected,
        )
        self.assertEqual(
            artifact.manifest["selection_policy"]["member_camera_preference"],
            "standard_fancam_before_facecam_or_closeup_fallback",
        )

    def test_member_closeups_are_used_only_after_standard_shortage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, member=True, desired=10)
            standard_entries = [
                _entry(
                    index,
                    title=(
                        f"IVE 가을 I AM Fancam Music Bank "
                        f"2304{10 + index:02d}"
                    ),
                )
                for index in range(8)
            ]
            closeup_entries = [
                _entry(
                    20 + index,
                    title=(
                        f"IVE 가을 I AM FaceCam Music Bank "
                        f"2304{20 + index:02d}"
                    ),
                )
                for index in range(3)
            ]
            client = _FakeClient([*closeup_entries, *standard_entries])

            acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_portrait_media,
            )

        standard_ids = {f"stage{index:06d}" for index in range(8)}
        self.assertEqual(set(client.download_attempts[:8]), standard_ids)
        self.assertTrue(
            all(video_id.startswith("stage00002") for video_id in client.download_attempts[8:])
        )

    def test_complete_manifest_reuses_exact_validated_source_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=9)
            first_client = _FakeClient([_entry(index) for index in range(10)])
            first = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=first_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            second_client = _FakeClient([])
            second = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=second_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

        self.assertEqual(second.source_paths, first.source_paths)
        self.assertEqual(second_client.download_attempts, [])
        self.assertEqual(second.manifest["summary"]["cache_reuse_count"], 1)
        self.assertTrue(
            all(
                item["content_integrity"]["algorithm"] == "sha256"
                and len(item["content_integrity"]["sha256"]) == 64
                for item in second.manifest["selected"]
            )
        )

    def test_cache_binding_contract_accepts_only_exact_v4_or_v5_scope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=9)
            artifact = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=_FakeClient(  # type: ignore[arg-type]
                    [_entry(index) for index in range(9)]
                ),
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            current = deepcopy(artifact.manifest)

        self.assertTrue(
            _manifest_matches_cache_binding_contract(current, request)
        )
        legacy_partial = deepcopy(current)
        legacy_partial["selection_policy_version"] = 4
        legacy_partial["summary"]["status"] = "downloading"
        self.assertTrue(
            _manifest_matches_cache_binding_contract(legacy_partial, request)
        )

        mismatches: list[tuple[str, dict[str, object]]] = []
        for label, field, value in (
            ("unsupported_policy", "selection_policy_version", 3),
            ("integrity_contract", "content_integrity_contract_version", 2),
            (
                "media_quality_contract",
                "automatic_media_quality_contract_version",
                None,
            ),
            ("group", "group", "OTHER"),
            ("song", "song", "BLACKHOLE"),
            ("edit_mode", "edit_mode", "member"),
            ("target_member", "target_member", "가을"),
            ("desired_count", "desired_source_count", 8),
        ):
            mismatched = deepcopy(current)
            mismatched[field] = value
            mismatches.append((label, mismatched))
        invalid_summary = deepcopy(current)
        invalid_summary["summary"] = "complete"
        mismatches.append(("summary_type", invalid_summary))

        for label, mismatched in mismatches:
            with self.subTest(label=label):
                self.assertFalse(
                    _manifest_matches_cache_binding_contract(
                        mismatched,
                        request,
                    )
                )

    def test_clean_completed_v4_cache_survives_v5_title_audit_without_download(
        self,
    ) -> None:
        cases = (
            ("I AM", True, _portrait_media),
            ("BLACKHOLE", False, _landscape_media),
        )
        for song, member, media_probe in cases:
            with self.subTest(song=song, member=member):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    request = _request(
                        root,
                        member=member,
                        desired=10,
                        song=song,
                    )
                    entries = [
                        _entry(
                            index,
                            title=(
                                f"IVE 가을 {song} Fancam Music Bank "
                                if member
                                else f"IVE - {song} | Music Bank Stage "
                            )
                            + (
                                f"2304{10 + index:02d}"
                            ),
                        )
                        for index in range(10)
                    ]
                    first = acquire_sources(
                        request,
                        reporter=_MemoryReporter(),
                        client=_FakeClient(entries),  # type: ignore[arg-type]
                        media_validator=lambda path: None,
                        media_probe=media_probe,
                    )
                    legacy = json.loads(
                        first.manifest_path.read_text(encoding="utf-8")
                    )
                    legacy["selection_policy_version"] = 4
                    first.manifest_path.write_text(
                        json.dumps(legacy, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    second_client = _FakeClient([])
                    second = acquire_sources(
                        request,
                        reporter=_MemoryReporter(),
                        client=second_client,  # type: ignore[arg-type]
                        media_validator=lambda path: None,
                        media_probe=media_probe,
                    )

                self.assertEqual(second.source_paths, first.source_paths)
                self.assertEqual(second_client.download_attempts, [])
                self.assertEqual(second.manifest["selection_policy_version"], 4)
                audit = second.manifest[
                    "legacy_selection_policy_current_title_audit"
                ]
                self.assertTrue(audit["passed"])
                self.assertEqual(audit["current_selection_policy_version"], 5)
                self.assertEqual(audit["audited_selected_count"], 10)

    def test_dirty_completed_v4_or_v5_cache_reuses_nine_and_replaces_bad_clip(
        self,
    ) -> None:
        cases = (
            (
                4,
                "[MPD직캠] IVE XOXZ 1위 앵콜직캠 | "
                "M Countdown Encore 260819",
                "non_music_show_encore",
            ),
            (
                5,
                "[#음중직캠] IVE XOXZ 1위-직캠 | "
                "Show! MusicCore 260819",
                "non_music_show_winner_afterstage",
            ),
            (
                5,
                "XOXZ - IVE (아이브) [뮤직뱅크 인 리스본] | "
                "KBS 251010 방송",
                "non_music_show_music_bank_global_special",
            ),
        )
        for prior_policy_version, dirty_title, expected_reason in cases:
            with self.subTest(prior_policy_version=prior_policy_version):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    request = _request(root, desired=10, song="XOXZ")
                    initial_entries = [
                        _entry(
                            index,
                            title=(
                                "IVE XOXZ Music Bank Stage "
                                f"2608{10 + index:02d}"
                            ),
                        )
                        for index in range(10)
                    ]
                    first = acquire_sources(
                        request,
                        reporter=_MemoryReporter(),
                        client=_FakeClient(initial_entries),  # type: ignore[arg-type]
                        media_validator=lambda path: None,
                        media_probe=_landscape_media,
                    )
                    prior = json.loads(
                        first.manifest_path.read_text(encoding="utf-8")
                    )
                    prior["selection_policy_version"] = prior_policy_version
                    dirty_item = prior["selected"][-1]
                    dirty_video_id = dirty_item["video_id"]
                    dirty_path = Path(dirty_item["local_path"])
                    dirty_item["title"] = dirty_title
                    first.manifest_path.write_text(
                        json.dumps(prior, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    fresh_entries: list[dict[str, object]] = []
                    for entry in initial_entries:
                        refreshed = dict(entry)
                        if refreshed["id"] == dirty_video_id:
                            refreshed["title"] = dirty_item["title"]
                        fresh_entries.append(refreshed)
                    fresh_entries.append(
                        _entry(
                            10,
                            title="IVE XOXZ Music Bank Stage 260821",
                        )
                    )
                    second_client = _FakeClient(fresh_entries)
                    second = acquire_sources(
                        request,
                        reporter=_MemoryReporter(),
                        client=second_client,  # type: ignore[arg-type]
                        media_validator=lambda path: None,
                        media_probe=_landscape_media,
                    )

                    self.assertTrue(
                        dirty_path.with_suffix(".mp4.invalid").is_file()
                    )
                    self.assertEqual(
                        second_client.download_attempts,
                        ["stage000010"],
                    )
                    self.assertEqual(
                        second.manifest["selection_policy_version"],
                        5,
                    )
                    audit = second.manifest[
                        "prior_selection_policy_title_audit"
                    ]
                    self.assertEqual(
                        audit["from_selection_policy_version"],
                        prior_policy_version,
                    )
                    self.assertEqual(audit["hard_rejected_count"], 1)
                    self.assertEqual(
                        audit["hard_rejected_entries"][0][
                            "hard_rejection_reason"
                        ],
                        expected_reason,
                    )
                    self.assertNotIn(
                        dirty_video_id,
                        {
                            item["video_id"]
                            for item in second.manifest["selected"]
                        },
                    )
                    self.assertEqual(
                        sum(
                            item["download_status"] == "cached"
                            for item in second.manifest["selected"]
                        ),
                        9,
                    )

    def test_partial_v4_cache_reuses_only_authenticated_clean_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=10, song="BANG BANG")
            initial_entries = [
                _entry(
                    index,
                    title=(
                        f"IVE BANG BANG Music Bank Stage "
                        f"2608{10 + index:02d}"
                    ),
                )
                for index in range(10)
            ]
            entries_by_id = {
                str(entry["id"]): entry for entry in initial_entries
            }
            first = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=_FakeClient(initial_entries),  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            legacy = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            legacy["selection_policy_version"] = 4
            legacy["selected"] = legacy["selected"][:7]
            legacy["summary"]["status"] = "downloading"
            legacy["summary"]["downloaded_source_count"] = 7
            changed_item = legacy["selected"][0]
            cached_items = legacy["selected"][1:5]
            winner_afterstage_item = legacy["selected"][5]
            encore_item = legacy["selected"][6]
            changed_path = Path(changed_item["local_path"])
            winner_afterstage_path = Path(winner_afterstage_item["local_path"])
            encore_path = Path(encore_item["local_path"])
            changed_path.write_bytes(b"tampered cache object")
            winner_afterstage_item["title"] = (
                "[쇼! 음악중심 미방분] 2월 4주차 1위 "
                "IVE - BANG BANG | MusicCore 260228"
            )
            encore_item["title"] = (
                "IVE BANG BANG Winner EncoreCam | Music Bank 260819"
            )
            first.manifest_path.write_text(
                json.dumps(legacy, ensure_ascii=False),
                encoding="utf-8",
            )
            fresh_entries = [
                dict(entries_by_id[str(changed_item["video_id"])]),
                *[
                    dict(entries_by_id[str(item["video_id"])])
                    for item in cached_items
                ],
                {
                    **entries_by_id[
                        str(winner_afterstage_item["video_id"])
                    ],
                    "title": winner_afterstage_item["title"],
                },
                {
                    **entries_by_id[str(encore_item["video_id"])],
                    "title": encore_item["title"],
                },
                *[
                    _entry(
                        20 + index,
                        title=(
                            "IVE BANG BANG Music Bank Stage "
                            f"2608{20 + index:02d}"
                        ),
                    )
                    for index in range(5)
                ],
            ]
            assert request.catalog_cache_directory is not None
            catalog_index_path = next(
                request.catalog_cache_directory.glob("*index*.json")
            )
            # This regression targets partial per-song media authentication,
            # not shared catalog retention. Force a cache miss so the supplied
            # fresh metadata remains the complete catalog fixture for the run.
            catalog_index_path.unlink()
            second_client = _FakeClient(fresh_entries)
            second = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=second_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

            expected_downloads = {
                str(changed_item["video_id"]),
                *(f"stage{20 + index:06d}" for index in range(5)),
            }
            self.assertEqual(set(second_client.download_attempts), expected_downloads)
            for cached_item in cached_items:
                self.assertNotIn(
                    str(cached_item["video_id"]),
                    second_client.download_attempts,
                )
            self.assertTrue(changed_path.with_suffix(".mp4.invalid").is_file())
            self.assertTrue(
                winner_afterstage_path.with_suffix(".mp4.invalid").is_file()
            )
            self.assertTrue(encore_path.with_suffix(".mp4.invalid").is_file())
            selected_by_id = {
                item["video_id"]: item for item in second.manifest["selected"]
            }
            for cached_item in cached_items:
                self.assertEqual(
                    selected_by_id[str(cached_item["video_id"])][
                        "download_status"
                    ],
                    "cached",
                )
            self.assertEqual(second.manifest["selection_policy_version"], 5)
            self.assertEqual(
                second.manifest["prior_selection_policy_title_audit"][
                    "hard_rejected_count"
                ],
                2,
            )

    def test_changed_cache_content_is_quarantined_and_reacquired(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=9)
            entries = [_entry(index) for index in range(10)]
            first = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=_FakeClient(entries),  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            changed = first.source_paths[0]
            changed.write_bytes(b"locally replaced media")
            second_client = _FakeClient(entries)
            second = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=second_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            quarantined_exists = changed.with_suffix(".mp4.invalid").is_file()

        self.assertTrue(quarantined_exists)
        self.assertIn(changed.stem, second_client.download_attempts)
        self.assertEqual(second.manifest["selection_policy_version"], 5)
        self.assertEqual(second.manifest["content_integrity_contract_version"], 1)
        self.assertEqual(second.manifest["summary"]["status"], "complete")

    def test_untrusted_cache_entry_symlinks_quarantine_the_link_not_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=9)
            entries = [_entry(index) for index in range(10)]
            videos = root / "cache" / "videos"
            videos.mkdir(parents=True)
            outside = root / "outside-user-video.mp4"
            outside.write_bytes(b"must remain untouched")
            for index in range(10):
                (videos / f"stage{index:06d}.mp4").symlink_to(outside)
            client = _FakeClient(entries)

            artifact = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

            self.assertEqual(outside.read_bytes(), b"must remain untouched")
            for video_id in client.download_attempts:
                quarantined = videos / f"{video_id}.mp4.invalid"
                self.assertTrue(quarantined.is_symlink())
                self.assertEqual(quarantined.resolve(), outside.resolve())
            self.assertTrue(all(not path.is_symlink() for path in artifact.source_paths))

    def test_cache_videos_directory_symlink_fails_without_external_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=9)
            external_directory = root / "external-videos"
            external_directory.mkdir()
            sentinel = external_directory / "sentinel.mp4"
            sentinel.write_bytes(b"must remain untouched")
            cache = root / "cache"
            cache.mkdir()
            (cache / "videos").symlink_to(external_directory, target_is_directory=True)

            with self.assertRaisesRegex(AcquisitionError, "심볼릭 링크"):
                acquire_sources(
                    request,
                    reporter=_MemoryReporter(),
                    client=_FakeClient([_entry(index) for index in range(9)]),  # type: ignore[arg-type]
                    media_validator=lambda path: None,
                    media_probe=_landscape_media,
                )

            self.assertEqual(sentinel.read_bytes(), b"must remain untouched")
            self.assertEqual(tuple(external_directory.iterdir()), (sentinel,))

    def test_missing_or_invalid_manifest_cannot_authenticate_preexisting_cache(
        self,
    ) -> None:
        for manifest_text in (None, "{not-json"):
            with self.subTest(manifest_text=manifest_text):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    request = _request(root, desired=9)
                    cache = root / "cache"
                    videos = cache / "videos"
                    videos.mkdir(parents=True)
                    for index in range(10):
                        (videos / f"stage{index:06d}.mp4").write_bytes(
                            b"untrusted-preexisting-content"
                        )
                    if manifest_text is not None:
                        (cache / "acquisition-manifest.json").write_text(
                            manifest_text,
                            encoding="utf-8",
                        )
                    client = _FakeClient([_entry(index) for index in range(10)])

                    artifact = acquire_sources(
                        request,
                        reporter=_MemoryReporter(),
                        client=client,  # type: ignore[arg-type]
                        media_validator=lambda path: None,
                        media_probe=_landscape_media,
                    )

                    self.assertEqual(len(client.download_attempts), 9)
                    self.assertTrue(
                        all(
                            path.read_bytes() == b""
                            for path in artifact.source_paths
                        )
                    )
                    for video_id in client.download_attempts:
                        quarantined = videos / f"{video_id}.mp4.invalid"
                        self.assertEqual(
                            quarantined.read_bytes(),
                            b"untrusted-preexisting-content",
                        )
                    self.assertTrue(
                        all(
                            item["download_status"] == "downloaded"
                            for item in artifact.manifest["selected"]
                        )
                    )

    def test_stale_selection_policy_manifest_is_reselected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=9)
            entries = [_entry(index) for index in range(10)]
            first = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=_FakeClient(entries),  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            stale = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            stale["selection_policy_version"] = 2
            stale["summary"]["cache_reuse_count"] = 99
            first.manifest_path.write_text(
                json.dumps(stale, ensure_ascii=False),
                encoding="utf-8",
            )
            second_client = _FakeClient(entries)
            second = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=second_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

        self.assertEqual(second.manifest["selection_policy_version"], 5)
        self.assertGreaterEqual(len(second_client.download_attempts), 9)
        self.assertNotEqual(second.manifest["summary"].get("cache_reuse_count"), 100)

    def test_playlist_owner_never_grants_source_channel_trust(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            query_entries = [_entry(index) for index in range(8)]
            untrusted = _entry(
                99,
                title="IVE - I AM | Music Bank Stage 230425",
            )
            untrusted.update(
                {
                    "id": "owner000001",
                    "channel": "IVE",
                    "channel_id": _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
                    "duration": 214,
                }
            )
            artifact = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=_CatalogClient(
                    query_entries,
                    catalog_entries=[untrusted],
                ),  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

        record = next(
            item
            for item in artifact.manifest["candidates"]
            if item["video_id"] == "owner000001"
        )
        self.assertIn(
            "official_broadcast_channel_id_unverified",
            record["rejection_reasons"],
        )
        self.assertEqual(
            record["discovery_sources"],
            ["official_ive_stage_playlist"],
        )
        provenance = artifact.manifest[
            "official_ive_stage_playlist_catalog"
        ]
        self.assertTrue(provenance["playlist_owner_proves_catalog_only"])
        self.assertTrue(
            provenance["playlist_owner_never_grants_source_channel_trust"]
        )
        self.assertEqual(provenance["metadata_filter_rejected_count"], 1)

    def test_catalog_dedupes_after_query_and_preserves_richer_query_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            query_entries = [_entry(index) for index in range(8)]
            degraded_duplicate = {
                "id": query_entries[0]["id"],
                "title": query_entries[0]["title"],
                "channel": "KBS Kpop",
                "duration": query_entries[0]["duration"],
            }
            artifact = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=_CatalogClient(
                    query_entries,
                    catalog_entries=[degraded_duplicate],
                ),  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

        duplicate_id = str(query_entries[0]["id"])
        matching = [
            item
            for item in artifact.manifest["candidates"]
            if item["video_id"] == duplicate_id
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(
            matching[0]["channel_id"],
            "UCeLPm9yH_a_QH8n6445G-Ow",
        )
        self.assertEqual(
            matching[0]["discovery_sources"],
            ["query_search", "official_ive_stage_playlist"],
        )
        provenance = artifact.manifest[
            "official_ive_stage_playlist_catalog"
        ]
        self.assertEqual(
            provenance["exact_video_id_duplicates_with_query_or_cache"], 1
        )
        self.assertEqual(
            provenance["merge_order"],
            "after_query_search_richer_whole_candidate_preserved",
        )

    def test_catalog_cache_tamper_is_rejected_when_refresh_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            fetched_at = datetime(2026, 8, 24, 1, 0, tzinfo=UTC)
            first_client = _CatalogClient(
                [],
                catalog_entries=[_entry(1)],
            )
            first_entries, first_provenance = (
                _load_official_ive_stage_catalog(
                    request=request,
                    client=first_client,  # type: ignore[arg-type]
                    now=fetched_at,
                )
            )
            self.assertEqual(len(first_entries), 1)
            object_path = Path(first_provenance["object_path"])
            object_path.write_bytes(object_path.read_bytes() + b" ")

            failed_client = _CatalogClient(
                [],
                catalog_error="network unavailable",
            )
            entries, provenance = _load_official_ive_stage_catalog(
                request=request,
                client=failed_client,  # type: ignore[arg-type]
                now=fetched_at + timedelta(hours=1),
            )

        self.assertEqual(entries, [])
        self.assertEqual(failed_client.catalog_attempts, 1)
        self.assertEqual(provenance["cache_state"], "invalid")
        self.assertIn("SHA-256", provenance["cache_error"])
        self.assertEqual(
            provenance["result"],
            "catalog_unavailable_query_search_only",
        )
        self.assertFalse(provenance["stale_fallback_used"])

    def test_catalog_rejects_unpinned_playlist_owner_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            client = _CatalogClient(
                [],
                catalog_payload=_catalog_payload(
                    [_entry(1)],
                    owner_channel_id="UC-not-the-pinned-ive-owner",
                ),
            )
            entries, provenance = _load_official_ive_stage_catalog(
                request=request,
                client=client,  # type: ignore[arg-type]
                now=datetime(2026, 8, 24, 1, 0, tzinfo=UTC),
            )

        self.assertEqual(entries, [])
        self.assertEqual(provenance["network_status"], "failed_or_untrusted")
        self.assertIn("소유자 채널 ID", provenance["network_error"])
        self.assertEqual(
            provenance["result"],
            "catalog_unavailable_query_search_only",
        )

    def test_fresh_hash_valid_catalog_cache_skips_network_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            fetched_at = datetime(2026, 8, 24, 1, 0, tzinfo=UTC)
            first_client = _CatalogClient(
                [],
                catalog_entries=[_entry(1)],
            )
            first_entries, _ = _load_official_ive_stage_catalog(
                request=request,
                client=first_client,  # type: ignore[arg-type]
                now=fetched_at,
            )
            offline_client = _CatalogClient(
                [],
                catalog_error="must not be called",
            )
            entries, provenance = _load_official_ive_stage_catalog(
                request=request,
                client=offline_client,  # type: ignore[arg-type]
                now=fetched_at + timedelta(hours=1),
            )

        self.assertEqual(entries, first_entries)
        self.assertEqual(offline_client.catalog_attempts, 0)
        self.assertEqual(provenance["cache_state"], "fresh")
        self.assertEqual(provenance["result"], "fresh_hash_valid_cache")

    def test_catalog_cache_uses_hash_valid_stale_fallback_on_network_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            fetched_at = datetime(2026, 8, 20, 1, 0, tzinfo=UTC)
            first_client = _CatalogClient(
                [],
                catalog_entries=[_entry(1)],
            )
            first_entries, _ = _load_official_ive_stage_catalog(
                request=request,
                client=first_client,  # type: ignore[arg-type]
                now=fetched_at,
            )
            failed_client = _CatalogClient(
                [],
                catalog_error="network unavailable",
            )
            entries, provenance = _load_official_ive_stage_catalog(
                request=request,
                client=failed_client,  # type: ignore[arg-type]
                now=fetched_at + timedelta(days=2),
            )

        self.assertEqual(entries, first_entries)
        self.assertEqual(failed_client.catalog_attempts, 1)
        self.assertEqual(provenance["cache_state"], "stale")
        self.assertTrue(provenance["stale_fallback_used"])
        self.assertEqual(
            provenance["result"],
            "stale_hash_valid_cache_after_refresh_failure",
        )

    def test_catalog_policy_upgrade_reuses_authenticated_media_without_redownload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            entries = [_entry(index) for index in range(8)]
            first = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=_FakeClient(entries),  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            legacy = json.loads(
                first.manifest_path.read_text(encoding="utf-8")
            )
            legacy["latest_song_acquisition_policy_version"] = 1
            legacy["promotion_date_cluster_selection"]["policy_version"] = 1
            legacy.pop("official_ive_stage_playlist_catalog", None)
            first.manifest_path.write_text(
                json.dumps(legacy, ensure_ascii=False),
                encoding="utf-8",
            )
            assert request.catalog_cache_directory is not None
            catalog_index_path = next(
                request.catalog_cache_directory.glob("*index*.json")
            )
            catalog_index = json.loads(
                catalog_index_path.read_text(encoding="utf-8")
            )
            catalog_index["fetched_at"] = "2020-01-01T00:00:00+00:00"
            catalog_index_path.write_text(
                json.dumps(catalog_index), encoding="utf-8"
            )
            second_client = _CatalogClient(
                entries,
                catalog_entries=entries,
            )
            second = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=second_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

        self.assertEqual(second_client.catalog_attempts, 1)
        self.assertEqual(second_client.download_attempts, [])
        self.assertEqual(
            second.manifest["latest_song_acquisition_policy_version"], 2
        )
        self.assertEqual(
            second.manifest["cache_refresh_upgrade"]["new_download_count"],
            0,
        )
        self.assertEqual(
            second.manifest["official_ive_stage_playlist_catalog"]["result"],
            "validated_network_catalog_cached",
        )

    def test_query_only_completion_retries_catalog_without_media_redownload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            query_entries = [_entry(index) for index in range(8)]
            first = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=_CatalogClient(
                    query_entries,
                    catalog_error="transient outage",
                ),  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            self.assertEqual(
                first.manifest["official_ive_stage_playlist_catalog"]["result"],
                "catalog_unavailable_query_search_only",
            )
            self.assertTrue(
                first.manifest["cache_refresh_upgrade"]["refresh_required"]
            )
            self.assertTrue(
                first.manifest["cache_refresh_upgrade"][
                    "catalog_retry_required"
                ]
            )

            recovered_client = _CatalogClient(
                query_entries,
                catalog_entries=query_entries,
            )
            recovered = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=recovered_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

        self.assertEqual(recovered_client.catalog_attempts, 1)
        self.assertEqual(recovered_client.download_attempts, [])
        self.assertEqual(
            recovered.manifest["official_ive_stage_playlist_catalog"]["result"],
            "validated_network_catalog_cached",
        )
        self.assertFalse(
            recovered.manifest["cache_refresh_upgrade"][
                "catalog_retry_required"
            ]
        )

    def test_smaller_refresh_cannot_replace_larger_hash_valid_stale_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            fetched_at = datetime(2026, 8, 20, 1, 0, tzinfo=UTC)
            original_entries = [_entry(index) for index in range(3)]
            first_entries, first_provenance = (
                _load_official_ive_stage_catalog(
                    request=request,
                    client=_CatalogClient(
                        [], catalog_entries=original_entries
                    ),  # type: ignore[arg-type]
                    now=fetched_at,
                )
            )
            regressed_client = _CatalogClient(
                [],
                catalog_entries=original_entries[:2],
            )
            entries, provenance = _load_official_ive_stage_catalog(
                request=request,
                client=regressed_client,  # type: ignore[arg-type]
                now=fetched_at + timedelta(days=2),
            )

        self.assertEqual(entries, first_entries)
        self.assertEqual(provenance["network_status"], "incomplete_or_regressed")
        self.assertTrue(provenance["stale_fallback_used"])
        self.assertEqual(
            provenance["object_sha256"], first_provenance["object_sha256"]
        )
        self.assertIn("3", provenance["network_error"])

    def test_nonzero_partial_refresh_uses_larger_hash_valid_stale_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            fetched_at = datetime(2026, 8, 20, 1, 0, tzinfo=UTC)
            original_entries = [_entry(index) for index in range(3)]
            first_entries, _ = _load_official_ive_stage_catalog(
                request=request,
                client=_CatalogClient(
                    [], catalog_entries=original_entries
                ),  # type: ignore[arg-type]
                now=fetched_at,
            )
            partial_payload = _catalog_payload(original_entries[:2])
            partial_payload["_crosscut_ytdlp_returncode"] = 1
            entries, provenance = _load_official_ive_stage_catalog(
                request=request,
                client=_CatalogClient(
                    [], catalog_payload=partial_payload
                ),  # type: ignore[arg-type]
                now=fetched_at + timedelta(days=2),
            )

        self.assertEqual(entries, first_entries)
        self.assertEqual(provenance["network_status"], "incomplete_or_regressed")
        self.assertTrue(provenance["stale_fallback_used"])
        self.assertIn("nonzero", provenance["network_error"])

    def test_initial_catalog_below_production_plausibility_floor_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp, patch(
            "crosscut.acquisition._CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT",
            _CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT,
        ):
            root = Path(temp)
            request = _request(root, desired=8)
            partial_entries = [
                {
                    "id": f"catalog{index:06d}",
                    "title": f"IVE catalog item {index}",
                }
                for index in range(
                    _CATALOG_MINIMUM_PLAUSIBLE_ENTRY_COUNT - 1
                )
            ]
            entries, provenance = _load_official_ive_stage_catalog(
                request=request,
                client=_CatalogClient(
                    [], catalog_entries=partial_entries
                ),  # type: ignore[arg-type]
                now=datetime(2026, 8, 24, 1, 0, tzinfo=UTC),
            )

        self.assertEqual(entries, [])
        self.assertEqual(provenance["network_status"], "incomplete_or_regressed")
        self.assertIn("1500", provenance["network_error"])
        self.assertEqual(
            provenance["result"],
            "catalog_unavailable_query_search_only",
        )

    def test_exact_id_merge_only_fills_nontrust_date_and_program(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _request(Path(temp), desired=8)
            trusted = _metadata_candidate(
                {
                    "id": "merge000001",
                    "title": "IVE I AM Official Stage",
                    "channel": "KBS Kpop",
                    "channel_id": "UCeLPm9yH_a_QH8n6445G-Ow",
                    "duration": 180,
                },
                request=request,
                query="query-search",
            )
            alternate = _metadata_candidate(
                {
                    "id": "merge000001",
                    "title": "IVE I AM Music Bank Stage",
                    "channel": "IVE",
                    "channel_id": _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
                    "duration": 180,
                    "upload_date": "20230414",
                },
                request=request,
                query=f"playlist:{_IVE_STAGE_PLAYLIST_ID}",
                discovery_source="official_ive_stage_playlist",
            )
            assert trusted is not None and alternate is not None
            merged = _merge_discovery(trusted, alternate)

        self.assertEqual(merged.title, trusted.title)
        self.assertEqual(merged.channel, trusted.channel)
        self.assertEqual(merged.channel_id, trusted.channel_id)
        self.assertEqual(merged.trust_score, trusted.trust_score)
        self.assertEqual(merged.rejection_reasons, trusted.rejection_reasons)
        self.assertEqual(merged.broadcast_date, "2023-04-14")
        self.assertEqual(merged.broadcast_program, "music_bank")

    def test_playlist_candidate_wrong_media_is_quarantined_and_next_is_used(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root, desired=8)
            catalog_entries = [_entry(index) for index in range(9)]
            client = _CatalogClient([], catalog_entries=catalog_entries)
            probed: list[Path] = []

            def first_portrait_then_landscape(path: Path) -> MediaInfo:
                probed.append(path)
                return _portrait_media(path) if len(probed) == 1 else _landscape_media(path)

            artifact = acquire_sources(
                request,
                reporter=_MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=first_portrait_then_landscape,
            )
            rejected_id = client.download_attempts[0]
            quarantined = root / "cache" / "videos" / f"{rejected_id}.mp4.invalid"
            rejected_record = next(
                item
                for item in artifact.manifest["candidates"]
                if item["video_id"] == rejected_id
            )
            quarantined_exists = quarantined.is_file()

        self.assertEqual(len(artifact.source_paths), 8)
        self.assertEqual(len(client.download_attempts), 9)
        self.assertTrue(quarantined_exists)
        self.assertIn(
            "automatic_group_landscape_aspect_ratio_mismatch",
            rejected_record["download_hard_rejection_reasons"],
        )
        self.assertEqual(
            rejected_record["discovery_sources"],
            ["official_ive_stage_playlist"],
        )


class _MemoryReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event_type: str, **payload) -> None:
        self.events.append((event_type, payload))

    def status(self, message: str) -> None:
        self.emit("status", message=message)

    def progress(self, value: float, message: str) -> None:
        self.emit("progress", value=value, message=message)

    def log(self, message: str) -> None:
        self.emit("log", message=message)


if __name__ == "__main__":
    unittest.main()
