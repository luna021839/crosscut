from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.acquisition import acquire_sources
from crosscut.group_source_quality import (
    GROUP_RELATIVE_QUALITY_SAMPLE_FRACTIONS,
    GroupSourceQualityAssessment,
    RankedGroupSource,
    analyze_group_source_quality,
    select_group_quality_pool,
)
from crosscut.media import AudioStreamInfo, MediaInfo, VideoStreamInfo
from crosscut.request import AnalysisRequest


_OFFICIAL_CHANNEL_ID = "UCeLPm9yH_a_QH8n6445G-Ow"


def _media(path: Path, *, portrait: bool = False) -> MediaInfo:
    width, height = ((1080, 1920) if portrait else (1920, 1080))
    return MediaInfo(
        path=str(path),
        format_name="mp4",
        duration_seconds=180.0,
        start_time_seconds=0.0,
        size_bytes=1,
        bit_rate=4_100_000,
        audio_streams=(
            AudioStreamInfo(1, "aac", 48_000, 2, "stereo", 180.0, 128_000),
        ),
        video_streams=(
            VideoStreamInfo(0, "h264", width, height, 29.97, 180.0, 4_000_000),
        ),
        tags={},
    )


def _assessment(
    score: float,
    *,
    structure: float = 100.0,
    duplicate_key: str | None = None,
) -> GroupSourceQualityAssessment:
    distribution = {
        "p10": score,
        "median": score,
        "p90": score,
        "mean": score,
        "std": 0.0,
    }
    return GroupSourceQualityAssessment(
        rankable=True,
        analysis_error=None,
        sample_count=8,
        requested_sample_fractions=GROUP_RELATIVE_QUALITY_SAMPLE_FRACTIONS,
        visual_quality_index_0_100=score,
        metadata_score_0_100=100.0,
        structure_score_0_100=structure,
        ranking_score_0_100=score,
        distributions={"quality_0_100": distribution},
        structure_signals=("test",),
        duplicate_feed_key=duplicate_key,
    )


class _FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def isOpened(self) -> bool:
        return True

    def set(self, _property: int, _value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return (True, self.frame.copy())

    def release(self) -> None:
        return None


class _FakeClient:
    executable = "/test/yt-dlp"

    def __init__(self, entries: list[dict[str, object]]) -> None:
        self.entries = entries
        self.download_attempts: list[str] = []
        self.search_calls = 0

    def search(self, query: str, limit: int) -> list[dict[str, object]]:
        del query, limit
        self.search_calls += 1
        return list(self.entries)

    def download(self, candidate, directory: Path) -> Path:
        self.download_attempts.append(candidate.video_id)
        path = directory / "videos" / f"{candidate.video_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path.resolve()


class _Reporter:
    def status(self, _message: str) -> None:
        return None

    def log(self, _message: str) -> None:
        return None

    def progress(self, _progress: float, _message: str) -> None:
        return None

    def emit(self, *_args, **_kwargs) -> None:
        return None


def _request(root: Path, *, member: bool = False, desired: int | None = None) -> AnalysisRequest:
    payload: dict[str, object] = {
        "song": "I AM",
        "edit_mode": "member" if member else "group",
        "aspect_ratio": "9:16" if member else "16:9",
        "target_member": "가을" if member else None,
        "source_mode": "automatic",
        "sources": [],
        "output_directory": str(root / "output"),
        "source_cache_directory": str(root / "cache"),
    }
    if desired is not None:
        payload["desired_source_count"] = desired
    return AnalysisRequest.from_dict(payload)


def _entries(count: int, *, member: bool = False) -> list[dict[str, object]]:
    return [
        {
            "id": f"quality{index:04d}",
            "title": (
                f"IVE 가을 I AM Fancam Music Bank 2304{10 + index:02d}"
                if member
                else f"IVE I AM Music Bank Stage 2304{10 + index:02d}"
            ),
            "channel": "KBS Kpop",
            "channel_id": _OFFICIAL_CHANNEL_ID,
            "duration": 180 + index % 2,
            "webpage_url": f"https://youtube.test/quality{index:04d}",
        }
        for index in range(count)
    ]


class GroupSourceQualityTests(unittest.TestCase):
    def test_eight_frame_analysis_prefers_sharp_consistent_source(self) -> None:
        generator = np.random.default_rng(8)
        sharp = generator.integers(24, 232, (270, 480, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(sharp, (41, 41), 12.0)
        info = _media(Path("source.mp4"))
        with patch(
            "crosscut.group_source_quality.cv2.VideoCapture",
            return_value=_FakeCapture(sharp),
        ):
            sharp_result = analyze_group_source_quality(
                Path("sharp.mp4"),
                info,
                title="IVE I AM Music Bank Stage 230410",
                channel="KBS Kpop",
                broadcast_date="2023-04-10",
            )
        with patch(
            "crosscut.group_source_quality.cv2.VideoCapture",
            return_value=_FakeCapture(blurred),
        ):
            blurred_result = analyze_group_source_quality(
                Path("blurred.mp4"),
                info,
                title="IVE I AM Music Bank Stage 230411",
                channel="KBS Kpop",
                broadcast_date="2023-04-11",
            )

        self.assertTrue(sharp_result.rankable)
        self.assertEqual(sharp_result.sample_count, 8)
        self.assertGreater(
            sharp_result.ranking_score_0_100 or 0.0,
            blurred_result.ranking_score_0_100 or 0.0,
        )

    def test_top_eight_prefers_unique_stage_feeds_and_deprioritizes_fancam(self) -> None:
        sources = tuple(
            RankedGroupSource(
                acquisition_index=index,
                video_id=f"video{index}",
                assessment=_assessment(
                    100.0 - index,
                    structure=5.0 if index == 9 else 100.0,
                    duplicate_key=("kbs:date:standard" if index in {7, 8} else None),
                ),
            )
            for index in range(10)
        )
        decision = select_group_quality_pool(sources)

        self.assertTrue(decision.gate["applied"])
        self.assertEqual(decision.selected_acquisition_indices, tuple(range(8)))
        self.assertIn(
            "duplicate_broadcast_feed_deprioritized",
            decision.exclusion_reasons[8],
        )
        self.assertIn(
            "group_structure_deprioritized",
            decision.exclusion_reasons[9],
        )

    def test_exactly_eight_are_kept_without_relative_rejection(self) -> None:
        sources = tuple(
            RankedGroupSource(
                acquisition_index=index,
                video_id=f"video{index}",
                assessment=GroupSourceQualityAssessment.unavailable("not sampled"),
            )
            for index in range(8)
        )
        decision = select_group_quality_pool(sources)

        self.assertFalse(decision.gate["applied"])
        self.assertEqual(decision.selected_acquisition_indices, tuple(range(8)))
        self.assertEqual(decision.exclusion_reasons, {})

    def test_automatic_group_manifest_records_scores_selection_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root)
            client = _FakeClient(_entries(10))

            def analyzer(path: Path, _info: MediaInfo, **_kwargs) -> GroupSourceQualityAssessment:
                index = int(path.stem[-4:])
                return _assessment(
                    100.0 - index,
                    structure=5.0 if index == 9 else 100.0,
                    duplicate_key=("kbs:date:standard" if index in {7, 8} else None),
                )

            artifact = acquire_sources(
                request,
                _Reporter(),  # type: ignore[arg-type]
                client=client,  # type: ignore[arg-type]
                media_validator=lambda _path: None,
                media_probe=_media,
                group_quality_analyzer=analyzer,
            )

            self.assertEqual(len(client.download_attempts), 10)
            self.assertEqual(len(artifact.source_paths), 8)
            self.assertEqual(
                [path.stem for path in artifact.source_paths],
                [f"quality{index:04d}" for index in range(8)],
            )
            gate = artifact.manifest["group_relative_quality_gate"]
            self.assertTrue(gate["applied"])
            self.assertEqual(len(gate["ranking"]), 10)
            summary = artifact.manifest["summary"]
            self.assertEqual(summary["materialized_source_count"], 10)
            self.assertEqual(summary["eligible_source_count"], 8)
            self.assertEqual(summary["group_quality_excluded_source_count"], 2)
            self.assertEqual(len(artifact.manifest["selected"]), 8)
            excluded = {
                record["video_id"]: record
                for record in artifact.manifest["candidates"]
                if record["download_status"] == "downloaded_quality_excluded"
            }
            self.assertEqual(set(excluded), {"quality0008", "quality0009"})
            self.assertTrue(
                all(record["group_relative_quality"] for record in excluded.values())
            )

            cached = acquire_sources(
                request,
                _Reporter(),  # type: ignore[arg-type]
                client=_FakeClient([]),  # type: ignore[arg-type]
                media_validator=lambda _path: None,
                media_probe=_media,
                group_quality_analyzer=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("completed relative ranking must be reused")
                ),
            )
            self.assertEqual(cached.source_paths, artifact.source_paths)
            self.assertEqual(
                cached.manifest["summary"]["materialized_source_count"],
                10,
            )

    def test_missing_group_gate_upgrades_ten_bound_files_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = _request(root)
            first = acquire_sources(
                request,
                _Reporter(),  # type: ignore[arg-type]
                client=_FakeClient(_entries(10)),  # type: ignore[arg-type]
                media_validator=lambda _path: None,
                media_probe=_media,
                group_quality_analyzer=lambda *_args, **_kwargs: (
                    GroupSourceQualityAssessment.unavailable(
                        "legacy_gate_not_available"
                    )
                ),
            )
            self.assertEqual(len(first.source_paths), 10)

            legacy = deepcopy(first.manifest)
            legacy.pop("group_relative_quality_gate_version")
            legacy.pop("group_relative_quality_gate")
            legacy["selection_policy"].pop("group_relative_quality_gate")
            legacy["minimum_acceptable_source_count"] = 9
            legacy["minimum_full_quality_source_count"] = 9
            legacy["source_count_policy"] = {
                "policy": "default_group_target_ten_minimum_nine",
                "target_source_count": 10,
                "minimum_acceptable_source_count": 9,
                "fallback_allowed": True,
                "explicit_nondefault_group_count_is_exact": False,
            }
            for field in (
                "materialized_source_count",
                "eligible_source_count",
                "group_quality_excluded_source_count",
            ):
                legacy["summary"].pop(field, None)
            legacy["summary"]["minimum_acceptable_source_count"] = 9
            # Array order is not trusted; the persisted source_index contract
            # controls both local quality input order and the upgraded order.
            legacy["selected"].reverse()
            first.manifest_path.write_text(
                json.dumps(legacy, ensure_ascii=False),
                encoding="utf-8",
            )

            before = {
                path: (
                    path.name,
                    path.stat().st_ino,
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                )
                for path in first.source_paths
            }
            second_client = _FakeClient([])

            def analyzer(
                path: Path,
                _info: MediaInfo,
                **_kwargs,
            ) -> GroupSourceQualityAssessment:
                return _assessment(100.0 - int(path.stem[-4:]))

            upgraded = acquire_sources(
                request,
                _Reporter(),  # type: ignore[arg-type]
                client=second_client,  # type: ignore[arg-type]
                media_validator=lambda _path: None,
                media_probe=_media,
                group_quality_analyzer=analyzer,
            )

            self.assertEqual(second_client.search_calls, 0)
            self.assertEqual(second_client.download_attempts, [])
            self.assertEqual(
                [path.stem for path in upgraded.source_paths],
                [f"quality{index:04d}" for index in range(8)],
            )
            self.assertEqual(
                [item["source_index"] for item in upgraded.manifest["selected"]],
                list(range(8)),
            )
            self.assertEqual(
                [
                    item["acquisition_source_index"]
                    for item in upgraded.manifest["selected"]
                ],
                list(range(8)),
            )
            self.assertEqual(
                upgraded.manifest["group_relative_quality_gate_version"],
                1,
            )
            self.assertEqual(
                upgraded.manifest["group_relative_quality_gate"]["status"],
                "applied_top_eight_selected",
            )
            self.assertEqual(
                upgraded.manifest["source_count_policy"]["policy"],
                "default_group_target_ten_minimum_eight",
            )
            self.assertEqual(
                upgraded.manifest["minimum_acceptable_source_count"],
                8,
            )
            summary = upgraded.manifest["summary"]
            self.assertEqual(summary["materialized_source_count"], 10)
            self.assertEqual(summary["eligible_source_count"], 8)
            self.assertEqual(summary["group_quality_excluded_source_count"], 2)
            self.assertTrue(summary["source_count_fallback_applied"])
            upgrade_audit = upgraded.manifest[
                "group_relative_quality_gate_upgrade"
            ]
            self.assertFalse(upgrade_audit["network_search_performed"])
            self.assertFalse(upgrade_audit["media_download_performed"])
            self.assertFalse(upgrade_audit["existing_media_files_modified"])
            self.assertEqual(
                {
                    path: (
                        path.name,
                        path.stat().st_ino,
                        path.stat().st_size,
                        path.stat().st_mtime_ns,
                    )
                    for path in first.source_paths
                },
                before,
            )

            third_client = _FakeClient([])
            reused = acquire_sources(
                request,
                _Reporter(),  # type: ignore[arg-type]
                client=third_client,  # type: ignore[arg-type]
                media_validator=lambda _path: None,
                media_probe=_media,
                group_quality_analyzer=lambda *_args, **_kwargs: (
                    _ for _ in ()
                ).throw(AssertionError("upgraded gate must be reused")),
            )
            self.assertEqual(reused.source_paths, upgraded.source_paths)
            self.assertEqual(third_client.search_calls, 0)
            self.assertEqual(third_client.download_attempts, [])

    def test_member_and_explicit_group_counts_never_call_relative_analyzer(self) -> None:
        def forbidden(*_args, **_kwargs):
            raise AssertionError("relative GROUP analyzer crossed its scope")

        cases = ((True, 10, True), (False, 9, False))
        for member, desired, portrait in cases:
            with self.subTest(member=member, desired=desired):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    artifact = acquire_sources(
                        _request(root, member=member, desired=desired),
                        _Reporter(),  # type: ignore[arg-type]
                        client=_FakeClient(_entries(desired, member=member)),  # type: ignore[arg-type]
                        media_validator=lambda _path: None,
                        media_probe=lambda path: _media(path, portrait=portrait),
                        group_quality_analyzer=forbidden,
                    )
                self.assertEqual(len(artifact.source_paths), desired)
                self.assertIsNone(
                    artifact.manifest["group_relative_quality_gate"]
                )


if __name__ == "__main__":
    unittest.main()
