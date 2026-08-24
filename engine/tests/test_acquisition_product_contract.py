from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.acquisition import (
    AcquisitionError,
    AcquisitionArtifact,
    _metadata_candidate,
    _ordered_for_download,
    _quarantine_invalid_media,
    acquire_sources,
)
from crosscut.media import (
    AudioStreamInfo,
    MediaInfo,
    MediaProbeError,
    VideoStreamInfo,
)
from crosscut.request import AnalysisRequest


_OFFICIAL_CHANNEL_ID = "UCeLPm9yH_a_QH8n6445G-Ow"


def _automatic_request(
    root: Path,
    *,
    member: bool = False,
    desired: int | None = None,
    acquire_only: bool = False,
) -> AnalysisRequest:
    payload: dict[str, object] = {
        "song": "I AM",
        "edit_mode": "member" if member else "group",
        "aspect_ratio": "9:16" if member else "16:9",
        "target_member": "가을" if member else None,
        "source_mode": "automatic",
        "sources": [],
        "output_directory": str(root / "out"),
        "source_cache_directory": str(root / "cache"),
        "acquire_only": acquire_only,
    }
    if desired is not None:
        payload["desired_source_count"] = desired
    return AnalysisRequest.from_dict(payload)


def _official_entry(
    index: int,
    *,
    title: str | None = None,
    channel: str = "KBS Kpop",
    channel_id: str | None = _OFFICIAL_CHANNEL_ID,
) -> dict[str, object]:
    date = f"2304{10 + index:02d}"
    video_id = f"contract{index:04d}"
    return {
        "id": video_id,
        "title": title or f"IVE I AM Music Bank Stage {date}",
        "channel": channel,
        "channel_id": channel_id,
        "duration": 184 + index % 4,
        "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
    }


def _media_with_dimensions(
    path: Path,
    *,
    width: int | None = 1080,
    height: int | None = 1920,
    frame_rate: float | None = 30.0,
) -> MediaInfo:
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
            VideoStreamInfo(
                0,
                "h264",
                width,
                height,
                frame_rate,
                185.0,
                4_000_000,
            ),
        ),
        tags={},
    )


def _landscape_media(path: Path) -> MediaInfo:
    return _media_with_dimensions(path, width=1920, height=1080)


def _member_entries(count: int) -> list[dict[str, object]]:
    return [
        _official_entry(
            index,
            title=f"IVE Gaeul I AM Fancam Music Bank 2304{10 + index:02d}",
        )
        for index in range(count)
    ]


class _FakeClient:
    executable = "/test-double/yt-dlp"

    def __init__(self, entries: list[dict[str, object]]) -> None:
        self.entries = list(entries)
        self.search_calls: list[tuple[str, int]] = []
        self.download_attempts: list[str] = []

    def search(self, query: str, limit: int) -> list[dict[str, object]]:
        self.search_calls.append((query, limit))
        return list(self.entries)

    def download(self, candidate, directory: Path) -> Path:  # type: ignore[no-untyped-def]
        self.download_attempts.append(candidate.video_id)
        path = directory / "videos" / f"{candidate.video_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path.resolve()


class _MemoryReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event_type: str, **payload: object) -> None:
        self.events.append((event_type, payload))

    def status(self, message: str) -> None:
        self.emit("status", message=message)

    def progress(self, value: float, message: str) -> None:
        self.emit("progress", value=value, message=message)

    def log(self, message: str) -> None:
        self.emit("log", message=message)


class AutomaticAcquisitionProductContractTests(unittest.TestCase):
    def test_failed_comparison_render_requires_explicit_cli_flag(self) -> None:
        import crosscut_cli

        default = crosscut_cli._parser().parse_args(
            ["analyze", "--request", "request.json"]
        )
        opted_in = crosscut_cli._parser().parse_args(
            [
                "analyze",
                "--request",
                "request.json",
                "--allow-failed-comparison-render",
            ]
        )

        self.assertFalse(default.allow_failed_comparison_render)
        self.assertTrue(opted_in.allow_failed_comparison_render)

    def test_empty_automatic_request_materializes_exact_default_ten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root)
            client = _FakeClient([_official_entry(index) for index in range(12)])

            artifact = acquire_sources(
                request,
                _MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

            self.assertEqual(request.sources, ())
            self.assertEqual(request.desired_source_count, 10)
            self.assertEqual(len(artifact.source_paths), 10)
            self.assertEqual(len(client.download_attempts), 10)
            self.assertTrue(all(limit == 20 for _, limit in client.search_calls))
            self.assertEqual(artifact.manifest["desired_source_count"], 10)
            self.assertEqual(
                [item["source_index"] for item in artifact.manifest["selected"]],
                list(range(10)),
            )
            self.assertEqual(artifact.manifest["summary"]["status"], "complete")

    def test_group_default_target_ten_completes_and_materializes_with_eight(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root)
            client = _FakeClient([_official_entry(index) for index in range(8)])

            artifact = acquire_sources(
                request,
                _MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

            self.assertEqual(request.desired_source_count, 10)
            self.assertEqual(len(artifact.source_paths), 8)
            self.assertEqual(len(client.download_attempts), 8)
            self.assertEqual(
                artifact.manifest["minimum_acceptable_source_count"],
                8,
            )
            self.assertEqual(
                artifact.manifest["source_count_policy"],
                {
                    "policy": "default_group_target_ten_minimum_eight",
                    "target_source_count": 10,
                    "minimum_acceptable_source_count": 8,
                    "fallback_allowed": True,
                    "explicit_nondefault_group_count_is_exact": False,
                },
            )
            summary = artifact.manifest["summary"]
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["requested_source_count"], 10)
            self.assertEqual(summary["minimum_acceptable_source_count"], 8)
            self.assertEqual(summary["downloaded_source_count"], 8)
            self.assertTrue(summary["source_count_fallback_applied"])

            materialized = request.with_materialized_sources(
                artifact.source_paths
            )
            self.assertEqual(len(materialized.sources), 8)

    def test_completed_group_nine_of_ten_cache_reuses_without_web_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root)
            first_client = _FakeClient(
                [_official_entry(index) for index in range(9)]
            )
            first = acquire_sources(
                request,
                _MemoryReporter(),
                client=first_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            second_client = _FakeClient([])
            second = acquire_sources(
                request,
                _MemoryReporter(),
                client=second_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )

            self.assertEqual(second.source_paths, first.source_paths)
            self.assertEqual(second_client.search_calls, [])
            self.assertEqual(second_client.download_attempts, [])
            self.assertEqual(
                second.manifest["source_count_policy"]["policy"],
                "default_group_target_ten_minimum_eight",
            )
            summary = second.manifest["summary"]
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["cache_reuse_count"], 1)
            self.assertEqual(summary["downloaded_source_count"], 9)
            self.assertTrue(summary["source_count_fallback_applied"])

    def test_explicit_group_counts_do_not_inherit_default_ten_fallback(
        self,
    ) -> None:
        cases = ((8, 7), (9, 8), (11, 10))
        for desired, available in cases:
            with self.subTest(desired=desired, available=available):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    request = _automatic_request(root, desired=desired)
                    client = _FakeClient(
                        [_official_entry(index) for index in range(available)]
                    )

                    with self.assertRaisesRegex(
                        AcquisitionError,
                        f"\ucd5c\uc18c {desired}\uac1c",
                    ):
                        acquire_sources(
                            request,
                            _MemoryReporter(),
                            client=client,  # type: ignore[arg-type]
                            media_validator=lambda path: None,
                            media_probe=_landscape_media,
                        )

                    manifest = json.loads(
                        (
                            root
                            / "cache"
                            / "acquisition-manifest.json"
                        ).read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        manifest["minimum_acceptable_source_count"],
                        desired,
                    )
                    self.assertEqual(
                        manifest["source_count_policy"]["policy"],
                        "explicit_group_count_required",
                    )
                    self.assertEqual(
                        manifest["summary"]["status"],
                        "insufficient_metadata_candidates",
                    )
                    self.assertEqual(client.download_attempts, [])

    def test_member_target_ten_completes_with_eight_verified_portrait_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root, member=True)
            client = _FakeClient(_member_entries(8))

            artifact = acquire_sources(
                request,
                _MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_media_with_dimensions,
            )

            self.assertEqual(request.desired_source_count, 10)
            self.assertEqual(len(artifact.source_paths), 8)
            self.assertEqual(len(client.download_attempts), 8)
            self.assertEqual(
                artifact.manifest["minimum_acceptable_source_count"],
                8,
            )
            summary = artifact.manifest["summary"]
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["downloaded_source_count"], 8)
            self.assertTrue(summary["source_count_fallback_applied"])

            materialized = request.with_materialized_sources(artifact.source_paths)
            self.assertEqual(len(materialized.sources), 8)

    def test_selection_requires_ive_show_title_and_verified_official_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            accepted = _metadata_candidate(
                _official_entry(0), request=request, query="contract"
            )
            non_ive = _metadata_candidate(
                _official_entry(1, title="FIVE I AM Music Bank Stage 230411"),
                request=request,
                query="contract",
            )
            wrong_song = _metadata_candidate(
                _official_entry(4, title="IVE MIAMI Music Bank Stage 230414"),
                request=request,
                query="contract",
            )
            non_show = _metadata_candidate(
                _official_entry(2, title="IVE I AM Performance Video"),
                request=request,
                query="contract",
            )
            unofficial = _metadata_candidate(
                _official_entry(
                    3,
                    channel="Unofficial Reupload",
                    channel_id="UC-not-allowlisted",
                ),
                request=request,
                query="contract",
            )

        assert accepted is not None
        assert non_ive is not None
        assert wrong_song is not None
        assert non_show is not None
        assert unofficial is not None
        self.assertTrue(accepted.selectable)
        self.assertEqual(accepted.official_channel_signal, "allowlisted_channel_id")
        self.assertIn("ive_title_signal_missing", non_ive.rejection_reasons)
        self.assertIn("song_title_signal_missing", wrong_song.rejection_reasons)
        self.assertIn("music_show_title_signal_missing", non_show.rejection_reasons)
        self.assertIn(
            "official_broadcast_channel_id_unverified",
            unofficial.rejection_reasons,
        )
        self.assertFalse(non_ive.selectable)
        self.assertFalse(wrong_song.selectable)
        self.assertFalse(non_show.selectable)
        self.assertFalse(unofficial.selectable)

    def test_partial_unknown_song_cannot_impersonate_a_known_ive_title(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            partial_request = AnalysisRequest.from_dict(
                {
                    "song": "AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 9,
                    "sources": [],
                    "output_directory": str(Path(temp) / "out"),
                }
            )
            partial = _metadata_candidate(
                _official_entry(0, title="IVE I AM Music Bank Stage 230414"),
                request=partial_request,
                query="contract",
            )
            future_request = AnalysisRequest.from_dict(
                {
                    "song": "Future Song",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 9,
                    "sources": [],
                    "output_directory": str(Path(temp) / "future-out"),
                }
            )
            future = _metadata_candidate(
                _official_entry(
                    1,
                    title="IVE Future Song Music Bank Stage 230421",
                ),
                request=future_request,
                query="contract",
            )

        assert partial is not None
        assert future is not None
        self.assertIn(
            "requested_song_conflicts_with_known_ive_title",
            partial.rejection_reasons,
        )
        self.assertFalse(partial.selectable)
        self.assertTrue(future.selectable)

    def test_non_music_show_formats_are_hard_rejected_with_specific_reasons(self) -> None:
        cases = {
            "[MV] Music Bank Stage": "non_music_show_music_video",
            "Dance Practice Music Bank Stage": "non_music_show_dance_practice",
            "Concert Music Bank Stage": "non_music_show_concert_or_festival",
            "Festival Music Bank Stage": "non_music_show_concert_or_festival",
            "Stage Mix Music Bank Stage": "non_music_show_stage_mix",
        }
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            for index, (format_label, reason) in enumerate(cases.items()):
                with self.subTest(format_label=format_label):
                    candidate = _metadata_candidate(
                        _official_entry(
                            index,
                            title=f"IVE I AM {format_label} 2304{10 + index:02d}",
                        ),
                        request=request,
                        query="contract",
                    )
                    assert candidate is not None
                    self.assertFalse(candidate.selectable)
                    self.assertIn(reason, candidate.rejection_reasons)

    def test_explicit_encore_and_winner_afterstage_titles_are_hard_rejected(
        self,
    ) -> None:
        rejected_titles = {
            (
                "[MPD직캠] 아이브 1위 앵콜 직캠 4K 'I AM' "
                "(IVE FanCam No.1 Encore) | @MCOUNTDOWN_2023.4.27"
            ): "non_music_show_encore",
            (
                "[4K] 아이브 'I AM' 뮤직뱅크 1위 앵콜직캠 "
                "(IVE Encore Facecam) @Music Bank 230428"
            ): "non_music_show_encore",
            (
                "[앵콜캠4K] 아이브 'I AM' 인기가요 1위 앵콜 직캠 "
                "(IVE Encore Fancam) @SBS Inkigayo 230430"
            ): "non_music_show_encore",
            (
                "IVE I AM Winner Encore Stage | Music Bank 230428"
            ): "non_music_show_encore",
            (
                "IVE I AM Winner EncoreCam4K | Music Bank 230428"
            ): "non_music_show_encore",
            (
                "[쇼! 음악중심 미방분] 4월 4주차 1위 IVE - I AM | "
                "Show! MusicCore | MBC230429방송"
            ): "non_music_show_winner_afterstage",
            (
                "[Show! MusicCore Unreleased] IVE I AM No. 1 | MBC230429"
            ): "non_music_show_winner_afterstage",
            (
                "[#음중직캠] IVE - I AM 1위 직캠 | "
                "Show! MusicCore | MBC230429"
            ): "non_music_show_winner_afterstage",
            (
                "IVE I AM 1 위—[4K]—직 캠 | Music Bank 230428"
            ): "non_music_show_winner_afterstage",
        }
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            for index, (title, reason) in enumerate(rejected_titles.items()):
                with self.subTest(title=title):
                    candidate = _metadata_candidate(
                        _official_entry(index, title=title),
                        request=request,
                        query="encore-contract",
                    )
                    assert candidate is not None
                    self.assertFalse(candidate.selectable)
                    self.assertIn(reason, candidate.rejection_reasons)

    def test_music_bank_global_special_titles_are_hard_rejected(self) -> None:
        rejected_titles = (
            "XOXZ - IVE (아이브) [뮤직뱅크 인 리스본] | KBS 251010 방송",
            "IVE I AM [Music Bank in Madrid] | KBS WORLD TV 230722",
            "IVE I AM [MusicBank IN Paris] Special Stage 230722",
            "IVE I AM [뮤직 뱅크 in Berlin] | KBS 230722",
        )
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            for index, title in enumerate(rejected_titles):
                with self.subTest(title=title):
                    candidate = _metadata_candidate(
                        _official_entry(index, title=title),
                        request=request,
                        query="global-special-contract",
                    )
                    assert candidate is not None
                    self.assertFalse(candidate.selectable)
                    self.assertIn(
                        "non_music_show_music_bank_global_special",
                        candidate.rejection_reasons,
                    )

    def test_winner_words_do_not_reject_an_ordinary_main_stage(self) -> None:
        accepted_titles = (
            "IVE I AM Music Bank Main Stage 230414 | Weekly Winner",
            "IVE I AM Music Bank Stage in 4K | KBS 230414",
            "IVE I AM at Music Bank | KBS WORLD TV 230414",
            "IVE I AM 뮤직뱅크 본방 무대 | KBS 230414",
            "[쇼! 음악중심] 4월 3주차 1위 후보 IVE - I AM | MBC230415",
            "[쇼! 음악중심 미방분] IVE - I AM 스페셜 무대 | MBC230422",
            "IVE I AM No. 1 Nominee Main Stage | M Countdown 230413",
            (
                "[#음중직캠] IVE - I AM 1위 후보 직캠 | "
                "Show! MusicCore | MBC230415"
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            for index, title in enumerate(accepted_titles):
                with self.subTest(title=title):
                    candidate = _metadata_candidate(
                        _official_entry(20 + index, title=title),
                        request=request,
                        query="encore-false-positive-contract",
                    )
                    assert candidate is not None
                    self.assertTrue(candidate.selectable)
                    self.assertNotIn(
                        "non_music_show_encore",
                        candidate.rejection_reasons,
                    )
                    self.assertNotIn(
                        "non_music_show_winner_afterstage",
                        candidate.rejection_reasons,
                    )

    def test_generic_stage_labels_do_not_substitute_for_a_named_music_show(self) -> None:
        titles = (
            "IVE I AM MAMA Awards Stage 230414",
            "IVE I AM Relay Dance Stage 230414",
            "IVE I AM Studio Live Stage 230414",
            "IVE I AM 무대 풀캠 230414",
        )
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            for index, title in enumerate(titles):
                with self.subTest(title=title):
                    candidate = _metadata_candidate(
                        _official_entry(index, title=title),
                        request=request,
                        query="contract",
                    )
                    assert candidate is not None
                    self.assertFalse(candidate.selectable)
                    self.assertIn(
                        "music_show_title_signal_missing",
                        candidate.rejection_reasons,
                    )

    def test_allowlisted_channel_id_is_required_and_name_only_is_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            name_only = _metadata_candidate(
                _official_entry(
                    0,
                    title="IVE I AM Music Bank Stage 230414",
                    channel_id="UC-name-only",
                ),
                request=request,
                query="contract",
            )
            allowlisted = _metadata_candidate(
                _official_entry(
                    1,
                    title="IVE I AM Music Bank Stage 230414",
                    channel="Unhelpful Display Name",
                ),
                request=request,
                query="contract",
            )

        assert name_only is not None and allowlisted is not None
        self.assertEqual(name_only.official_channel_signal, "trusted_channel_name")
        self.assertEqual(allowlisted.official_channel_signal, "allowlisted_channel_id")
        self.assertFalse(name_only.selectable)
        self.assertIn(
            "official_broadcast_channel_id_unverified",
            name_only.rejection_reasons,
        )
        self.assertTrue(allowlisted.selectable)

    def test_show_champion_and_simply_kpop_official_channel_ids_are_accepted(self) -> None:
        cases = (
            (
                "IVE I AM | Show Champion | EP.470 230419",
                "ALL THE K-POP",
                "UCPde4guD9yFBRzkxk2PatoA",
            ),
            (
                "IVE I AM | Simply K-Pop CON-TOUR | EP.570 230420",
                "ARIRANG K-POP",
                "UC_6zPf4uuteHFFctnQtEaXQ",
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            for index, (title, channel, channel_id) in enumerate(cases):
                with self.subTest(channel=channel):
                    candidate = _metadata_candidate(
                        _official_entry(
                            index,
                            title=title,
                            channel=channel,
                            channel_id=channel_id,
                        ),
                        request=request,
                        query="contract",
                    )
                    assert candidate is not None
                    self.assertTrue(candidate.selectable)
                    self.assertEqual(
                        candidate.official_channel_signal,
                        "allowlisted_channel_id",
                    )

    def test_unique_broadcast_dates_precede_same_date_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            entries = (
                _official_entry(
                    0,
                    title="IVE I AM Fullcam Music Bank 230414",
                ),
                _official_entry(
                    1,
                    title="IVE I AM Music Bank Stage 230414",
                ),
                _official_entry(
                    2,
                    title="IVE I AM Music Bank Stage 230421",
                ),
                _official_entry(
                    3,
                    title="IVE I AM Music Bank Stage 230428",
                ),
            )
            candidates = [
                _metadata_candidate(entry, request=request, query="contract")
                for entry in entries
            ]

        ordered = _ordered_for_download(
            tuple(candidate for candidate in candidates if candidate is not None)
        )
        first_dates = [candidate.broadcast_date for candidate in ordered[:3]]
        self.assertEqual(
            set(first_dates),
            {"2023-04-14", "2023-04-21", "2023-04-28"},
        )
        self.assertEqual(len(set(first_dates)), 3)
        self.assertEqual(ordered[3].broadcast_date, "2023-04-14")
        self.assertIn("fullcam", ordered[3].title.casefold())

    def test_clean_later_broadcast_precedes_earlier_one_take_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            request = _automatic_request(Path(temp), desired=9)
            candidates = tuple(
                _metadata_candidate(entry, request=request, query="contract")
                for entry in (
                    _official_entry(
                        0,
                        title="IVE I AM ONE TAKE STAGE SBS Inkigayo 230416",
                    ),
                    _official_entry(
                        1,
                        title="IVE I AM Music Bank 230428",
                    ),
                    _official_entry(
                        2,
                        title="IVE I AM MusicCore 230429",
                    ),
                )
            )
        selectable = [candidate for candidate in candidates if candidate is not None]
        ordered = _ordered_for_download(selectable)
        self.assertEqual(
            [candidate.broadcast_date for candidate in ordered],
            ["2023-04-28", "2023-04-29", "2023-04-16"],
        )

    def test_member_sources_require_target_member_and_fancam_title_signals(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            member_request = _automatic_request(Path(temp), member=True, desired=10)
            group_request = _automatic_request(Path(temp), desired=9)
            accepted = _metadata_candidate(
                _official_entry(
                    0,
                    title="IVE Gaeul I AM Fancam Music Bank 230414",
                ),
                request=member_request,
                query="contract",
            )
            missing_fancam = _metadata_candidate(
                _official_entry(
                    1,
                    title="IVE Gaeul I AM Music Bank Stage 230421",
                ),
                request=member_request,
                query="contract",
            )
            wrong_member = _metadata_candidate(
                _official_entry(
                    2,
                    title="IVE Rei I AM Fancam Music Bank 230428",
                ),
                request=member_request,
                query="contract",
            )
            group_fancam = _metadata_candidate(
                _official_entry(
                    3,
                    title="IVE Gaeul I AM Fancam Music Bank 230429",
                ),
                request=group_request,
                query="contract",
            )

        assert accepted is not None
        assert missing_fancam is not None
        assert wrong_member is not None
        assert group_fancam is not None
        self.assertTrue(accepted.selectable)
        self.assertIn("fancam_title_signal_missing", missing_fancam.rejection_reasons)
        self.assertIn(
            "target_member_title_signal_missing",
            wrong_member.rejection_reasons,
        )
        self.assertIn(
            "other_member_title_signal_without_target",
            wrong_member.rejection_reasons,
        )
        self.assertIn(
            "single_member_fancam_not_group_stage",
            group_fancam.rejection_reasons,
        )

    def test_korean_member_aliases_require_complete_hangul_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rei_request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "member",
                    "aspect_ratio": "9:16",
                    "target_member": "레이",
                    "source_mode": "automatic",
                    "sources": [],
                    "output_directory": str(root / "out"),
                    "source_cache_directory": str(root / "cache"),
                }
            )
            accepted = _metadata_candidate(
                _official_entry(
                    4,
                    title="아이브 레이 I AM 직캠 M COUNTDOWN 230413",
                ),
                request=rei_request,
                query="contract",
            )
            relay_only = _metadata_candidate(
                _official_entry(
                    5,
                    title="[릴레이 직캠] 아이브 I AM FanCam M COUNTDOWN 230420",
                ),
                request=rei_request,
                query="contract",
            )
            relay_with_liz = _metadata_candidate(
                _official_entry(
                    6,
                    title="[릴레이 직캠] 아이브 리즈 I AM FanCam M COUNTDOWN 230427",
                ),
                request=rei_request,
                query="contract",
            )

        assert accepted is not None
        assert relay_only is not None
        assert relay_with_liz is not None
        self.assertTrue(accepted.selectable)
        self.assertFalse(relay_only.selectable)
        self.assertIn(
            "target_member_title_signal_missing",
            relay_only.rejection_reasons,
        )
        self.assertFalse(relay_with_liz.selectable)
        self.assertIn(
            "target_member_title_signal_missing",
            relay_with_liz.rejection_reasons,
        )
        self.assertIn(
            "other_member_title_signal_without_target",
            relay_with_liz.rejection_reasons,
        )

    def test_manifest_keeps_policy_provenance_and_rejection_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root, desired=9)
            valid = [_official_entry(index) for index in range(9)]
            rejected = [
                _official_entry(
                    20,
                    title="FIVE I AM Music Bank Stage 230510",
                ),
                _official_entry(
                    21,
                    title="IVE I AM [MV] Music Bank Stage 230511",
                ),
                _official_entry(
                    22,
                    title="IVE I AM Music Bank Stage 230512",
                    channel="Fan Reupload",
                    channel_id="UC-fan-reupload",
                ),
            ]
            client = _FakeClient([*valid, *rejected])

            artifact = acquire_sources(
                request,
                _MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=_landscape_media,
            )
            persisted = json.loads(
                artifact.manifest_path.read_text(encoding="utf-8")
            )

            records = {
                record["video_id"]: record for record in persisted["candidates"]
            }
            non_ive = records["contract0020"]
            music_video = records["contract0021"]
            unofficial = records["contract0022"]
            self.assertIn("ive_title_signal_missing", non_ive["rejection_reasons"])
            self.assertIn(
                "non_music_show_music_video",
                music_video["rejection_reasons"],
            )
            self.assertIn(
                "official_broadcast_channel_id_unverified",
                unofficial["rejection_reasons"],
            )
            for record in (non_ive, music_video, unofficial):
                self.assertFalse(record["selected"])
                self.assertEqual(record["download_status"], "not_selected")
                self.assertGreater(len(record["queries"]), 1)

            self.assertEqual(persisted["group"], "IVE")
            self.assertEqual(persisted["network_policy"]["cookies"], "disabled")
            self.assertFalse(persisted["network_policy"]["drm_bypass"])
            self.assertFalse(persisted["network_policy"]["uploads"])
            self.assertEqual(
                persisted["selection_policy"]["primary"],
                "trusted_official_broadcast_channel_and_music_show_title",
            )
            self.assertEqual(persisted["selection_policy_version"], 5)
            self.assertIn(
                "encore",
                persisted["selection_policy"]["hard_rejections"],
            )
            self.assertIn(
                "winner_afterstage",
                persisted["selection_policy"]["hard_rejections"],
            )
            self.assertIn(
                "music_bank_global_special",
                persisted["selection_policy"]["hard_rejections"],
            )
            self.assertEqual(len(persisted["selected"]), 9)
            self.assertEqual(persisted, artifact.manifest)

    def test_invalid_download_is_quarantined_and_replaced_by_next_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root, desired=9)
            client = _FakeClient([_official_entry(index) for index in range(10)])
            validation_calls: list[Path] = []

            def validate(path: Path) -> None:
                validation_calls.append(path)
                if len(validation_calls) == 1:
                    raise ValueError("synthetic corrupt media")

            artifact = acquire_sources(
                request,
                _MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=validate,
                media_probe=_landscape_media,
            )

            self.assertEqual(len(client.download_attempts), 10)
            self.assertEqual(len(artifact.source_paths), 9)
            failed_id = client.download_attempts[0]
            failed_record = next(
                record
                for record in artifact.manifest["candidates"]
                if record["video_id"] == failed_id
            )
            self.assertEqual(failed_record["download_status"], "failed")
            self.assertFalse(failed_record["selected"])
            self.assertIn("synthetic corrupt media", failed_record["download_error"])
            self.assertTrue(
                (root / "cache" / "videos" / f"{failed_id}.mp4.invalid").is_file()
            )
            self.assertNotIn(
                failed_id,
                [selected["video_id"] for selected in artifact.manifest["selected"]],
            )
            self.assertEqual(
                artifact.manifest["summary"]["failed_download_count"], 1
            )
            self.assertEqual(artifact.manifest["summary"]["status"], "complete")

    def test_group_quality_gate_rejects_resolution_fps_and_portrait_media(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root, desired=9)
            client = _FakeClient([_official_entry(index) for index in range(12)])
            probe_calls = 0

            def probe(path: Path) -> MediaInfo:
                nonlocal probe_calls
                probe_calls += 1
                if probe_calls == 1:
                    return _media_with_dimensions(
                        path,
                        width=854,
                        height=480,
                    )
                if probe_calls == 2:
                    return _media_with_dimensions(
                        path,
                        width=1920,
                        height=1080,
                        frame_rate=22.0,
                    )
                if probe_calls == 3:
                    return _media_with_dimensions(path)
                return _landscape_media(path)

            artifact = acquire_sources(
                request,
                _MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=probe,
            )

            self.assertEqual(len(client.download_attempts), 12)
            expected_reasons = (
                "automatic_video_resolution_below_minimum",
                "automatic_video_frame_rate_below_minimum",
                "automatic_group_landscape_aspect_ratio_mismatch",
            )
            records = {
                item["video_id"]: item for item in artifact.manifest["candidates"]
            }
            for video_id, reason in zip(
                client.download_attempts[:3],
                expected_reasons,
                strict=True,
            ):
                self.assertIn(reason, records[video_id]["rejection_reasons"])
                self.assertTrue(
                    (root / "cache" / "videos" / f"{video_id}.mp4.invalid").is_file()
                )
            self.assertEqual(
                artifact.manifest["summary"][
                    "media_contract_hard_rejection_count"
                ],
                3,
            )
            quality = artifact.manifest["selection_policy"][
                "automatic_media_quality_contract"
            ]
            self.assertEqual(quality["minimum_short_edge"], 720)
            self.assertEqual(quality["minimum_long_edge"], 1280)
            self.assertEqual(quality["minimum_frame_rate"], 23.0)
            self.assertEqual(quality["group_ratio_tolerance"], 0.03)

    def test_group_quality_gate_accepts_exact_resolution_and_fps_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root, desired=9)
            client = _FakeClient([_official_entry(index) for index in range(9)])

            artifact = acquire_sources(
                request,
                _MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=lambda path: _media_with_dimensions(
                    path,
                    width=1280,
                    height=720,
                    frame_rate=23.0,
                ),
            )

            self.assertEqual(len(artifact.source_paths), 9)
            self.assertTrue(
                all(
                    item["media_validation"]["passed"]
                    and item["media_validation"]["actual_short_edge"] == 720
                    and item["media_validation"]["actual_long_edge"] == 1280
                    and item["media_validation"]["frame_rate"] == 23.0
                    for item in artifact.manifest["selected"]
                )
            )

    def test_member_landscape_download_is_quarantined_and_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root, member=True, desired=10)
            client = _FakeClient(_member_entries(11))
            probe_calls = 0

            def probe(path: Path) -> MediaInfo:
                nonlocal probe_calls
                probe_calls += 1
                if probe_calls == 1:
                    return _media_with_dimensions(path, width=1920, height=1080)
                return _media_with_dimensions(path)

            artifact = acquire_sources(
                request,
                _MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                # The legacy one-argument validator remains a valid injection.
                media_validator=lambda path: None,
                media_probe=probe,
            )

            self.assertEqual(len(client.download_attempts), 11)
            self.assertEqual(len(set(client.download_attempts)), 11)
            self.assertEqual(len(artifact.source_paths), 10)
            failed_id = client.download_attempts[0]
            failed = next(
                item
                for item in artifact.manifest["candidates"]
                if item["video_id"] == failed_id
            )
            self.assertEqual(failed["download_status"], "failed")
            self.assertIn(
                "automatic_member_portrait_aspect_ratio_mismatch",
                failed["rejection_reasons"],
            )
            self.assertEqual(
                failed["aspect_validation"]["actual_width_over_height"],
                16.0 / 9.0,
            )
            self.assertFalse(failed["aspect_validation"]["passed"])
            self.assertTrue(
                (root / "cache" / "videos" / f"{failed_id}.mp4.invalid").is_file()
            )
            self.assertEqual(
                artifact.manifest["summary"]["aspect_hard_rejection_count"],
                1,
            )
            policy = artifact.manifest["selection_policy"]
            self.assertEqual(
                policy["member_portrait_aspect_contract"][
                    "expected_width_over_height"
                ],
                9.0 / 16.0,
            )
            self.assertIn(
                "automatic_member_portrait_aspect_ratio_mismatch",
                policy["hard_rejections"],
            )
            self.assertTrue(
                all(
                    selected["aspect_validation"]["passed"]
                    for selected in artifact.manifest["selected"]
                )
            )

    def test_member_complete_cache_reuse_rejects_landscape_and_replaces_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root, member=True, desired=10)
            entries = _member_entries(11)
            first = acquire_sources(
                request,
                _MemoryReporter(),
                client=_FakeClient(entries),  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=lambda path: _media_with_dimensions(path),
            )
            bad_path = first.source_paths[0]
            second_client = _FakeClient(entries)

            def changed_cache_probe(path: Path) -> MediaInfo:
                if path.name == bad_path.name:
                    return _media_with_dimensions(path, width=1920, height=1080)
                return _media_with_dimensions(path)

            second = acquire_sources(
                request,
                _MemoryReporter(),
                client=second_client,  # type: ignore[arg-type]
                media_validator=lambda path: None,
                media_probe=changed_cache_probe,
            )

            self.assertNotEqual(second.source_paths, first.source_paths)
            self.assertNotIn(bad_path, second.source_paths)
            self.assertTrue(bad_path.with_suffix(".mp4.invalid").is_file())
            self.assertEqual(len(second.source_paths), 10)
            # The nine still-valid prior hash bindings are retained. Only
            # one new candidate is downloaded to replace the media-contract
            # rejection; the bad cached bytes themselves are never rebound.
            self.assertEqual(len(set(second_client.download_attempts)), 1)
            self.assertEqual(
                second.manifest["summary"]["aspect_hard_rejection_count"],
                1,
            )
            failed = next(
                item
                for item in second.manifest["candidates"]
                if item["video_id"] == bad_path.stem
            )
            self.assertEqual(
                failed["download_hard_rejection_reasons"],
                ["automatic_member_portrait_aspect_ratio_mismatch"],
            )

    def test_repeated_invalid_cache_files_get_unique_quarantine_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            media = root / "source.mp4"
            media.write_bytes(b"first")
            first = _quarantine_invalid_media(media)
            media.write_bytes(b"second")
            second = _quarantine_invalid_media(media)

            self.assertEqual(first, root / "source.mp4.invalid")
            self.assertEqual(second, root / "source.mp4.invalid.1")
            self.assertFalse(media.exists())
            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(second.read_bytes(), b"second")

    def test_real_media_probe_error_is_quarantined_and_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _automatic_request(root, desired=9)
            client = _FakeClient([_official_entry(index) for index in range(10)])
            validation_calls = 0

            def validate(path: Path) -> None:
                nonlocal validation_calls
                del path
                validation_calls += 1
                if validation_calls == 1:
                    raise MediaProbeError("synthetic ffprobe failure")

            artifact = acquire_sources(
                request,
                _MemoryReporter(),
                client=client,  # type: ignore[arg-type]
                media_validator=validate,
                media_probe=_landscape_media,
            )

            self.assertEqual(len(client.download_attempts), 10)
            self.assertEqual(len(artifact.source_paths), 9)
            failed_id = client.download_attempts[0]
            failed_record = next(
                record
                for record in artifact.manifest["candidates"]
                if record["video_id"] == failed_id
            )
            self.assertEqual(failed_record["download_status"], "failed")
            self.assertFalse(failed_record["selected"])
            self.assertIn("synthetic ffprobe failure", failed_record["download_error"])
            self.assertTrue(
                (root / "cache" / "videos" / f"{failed_id}.mp4.invalid").is_file()
            )

    def test_acquire_only_cli_returns_sources_without_starting_analysis(self) -> None:
        import crosscut_cli

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps(
                    {
                        "song": "I AM",
                        "edit_mode": "group",
                        "source_mode": "automatic",
                        "sources": [],
                        "output_directory": str(root / "out"),
                        "source_cache_directory": str(root / "cache"),
                        "acquire_only": True,
                    }
                ),
                encoding="utf-8",
            )
            sources = tuple(root / f"source-{index:02d}.mp4" for index in range(10))
            manifest_path = root / "cache" / "acquisition-manifest.json"
            acquisition = AcquisitionArtifact(
                source_paths=sources,
                member_anchors=(),
                manifest_path=manifest_path,
                manifest={"summary": {"status": "complete"}},
            )
            stream = io.StringIO()

            with patch(
                "crosscut_cli.acquire_sources",
                return_value=acquisition,
            ) as acquire_mock, patch("crosscut_cli.analyze") as analyze_mock:
                with redirect_stdout(stream):
                    exit_code = crosscut_cli.main(
                        ["analyze", "--request", str(request_path)]
                    )

            self.assertEqual(exit_code, 0)
            acquire_mock.assert_called_once()
            analyze_mock.assert_not_called()
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
            result = next(event for event in events if event["type"] == "result")
            expected_paths = [str(path) for path in sources]
            self.assertEqual(result["outputs"], expected_paths)
            self.assertEqual(result["source_paths"], expected_paths)
            self.assertEqual(result["acquisition_manifest"], str(manifest_path))
            self.assertTrue(
                any(
                    event["type"] == "progress" and event["value"] == 1.0
                    for event in events
                )
            )


if __name__ == "__main__":
    unittest.main()
