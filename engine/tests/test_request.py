from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from crosscut.ive import EditMode, IVEMember, SourceMode
from crosscut.request import AnalysisRequest, MemberAnchorRequest


class RequestTests(unittest.TestCase):
    def test_stage_edit_production_inference_request_is_explicit_and_round_trips(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "production-checkpoint"
            checkpoint.mkdir()
            report = checkpoint / "training-report.json"
            report.write_text("{}", encoding="utf-8")
            sources = tuple(root / f"source-{index}.mp4" for index in range(8))
            for source in sources:
                source.touch()
            legacy = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "sources": [],
                    "output_directory": str(root / "legacy-out"),
                }
            )
            self.assertNotIn(
                "stage_edit_checkpoint_directory", legacy.as_dict()
            )
            self.assertNotIn(
                "stage_edit_production_training_report", legacy.as_dict()
            )
            self.assertNotIn(
                "stage_edit_learned_reviewer_checkpoint_directory",
                legacy.as_dict(),
            )
            self.assertNotIn(
                "stage_edit_learned_reviewer_production_training_report",
                legacy.as_dict(),
            )

            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "sources": [],
                    "output_directory": str(root / "out"),
                    "stage_edit_checkpoint_directory": str(checkpoint),
                    "stage_edit_production_training_report": str(report),
                }
            )
            self.assertEqual(
                request.stage_edit_checkpoint_directory, checkpoint.resolve()
            )
            self.assertEqual(
                request.stage_edit_production_training_report, report.resolve()
            )
            self.assertEqual(
                request.as_dict()["stage_edit_checkpoint_directory"],
                str(checkpoint.resolve()),
            )
            materialized = request.with_materialized_sources(sources)
            self.assertEqual(
                materialized.stage_edit_checkpoint_directory,
                checkpoint.resolve(),
            )
            self.assertEqual(
                materialized.stage_edit_production_training_report,
                report.resolve(),
            )

    def test_stage_edit_production_inference_request_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            report = checkpoint / "training-report.json"
            report.touch()
            second_source = root / "second.mp4"
            second_source.touch()
            common = {
                "song": "BANG BANG",
                "edit_mode": "group",
                "source_mode": "automatic",
                "sources": [],
            }
            with self.assertRaisesRegex(ValueError, "supplied together"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "stage_edit_checkpoint_directory": str(checkpoint),
                    }
                )
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "stage_edit_checkpoint_directory": "",
                        "stage_edit_production_training_report": str(report),
                    }
                )
            with self.assertRaisesRegex(ValueError, "existing directory"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "stage_edit_checkpoint_directory": str(root / "missing"),
                        "stage_edit_production_training_report": str(report),
                    }
                )
            with self.assertRaisesRegex(ValueError, "existing file"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "stage_edit_checkpoint_directory": str(checkpoint),
                        "stage_edit_production_training_report": str(
                            root / "missing-report.json"
                        ),
                    }
                )
            with self.assertRaisesRegex(ValueError, "automatic group"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "source_mode": "local",
                        "sources": [str(report), str(second_source)],
                        "stage_edit_checkpoint_directory": str(checkpoint),
                        "stage_edit_production_training_report": str(report),
                    }
                )
            with self.assertRaisesRegex(ValueError, "acquire_only"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "acquire_only": True,
                        "stage_edit_checkpoint_directory": str(checkpoint),
                        "stage_edit_production_training_report": str(report),
                    }
                )

    def test_learned_reviewer_request_pair_round_trips_and_materializes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "learned-reviewer"
            checkpoint.mkdir()
            report = checkpoint / "training-report.json"
            report.write_text("{}", encoding="utf-8")
            sources = tuple(root / f"source-{index}.mp4" for index in range(8))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "sources": [],
                    "output_directory": str(root / "out"),
                    "stage_edit_learned_reviewer_checkpoint_directory": str(
                        checkpoint
                    ),
                    "stage_edit_learned_reviewer_production_training_report": str(
                        report
                    ),
                }
            )

            self.assertEqual(
                request.stage_edit_learned_reviewer_checkpoint_directory,
                checkpoint.resolve(),
            )
            self.assertEqual(
                request.as_dict()[
                    "stage_edit_learned_reviewer_production_training_report"
                ],
                str(report.resolve()),
            )
            materialized = request.with_materialized_sources(sources)
            self.assertEqual(
                materialized.stage_edit_learned_reviewer_checkpoint_directory,
                checkpoint.resolve(),
            )
            self.assertEqual(
                materialized.stage_edit_learned_reviewer_production_training_report,
                report.resolve(),
            )

    def test_learned_reviewer_request_pair_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "learned-reviewer"
            checkpoint.mkdir()
            report = checkpoint / "training-report.json"
            report.touch()
            common = {
                "song": "BANG BANG",
                "edit_mode": "group",
                "source_mode": "automatic",
                "sources": [],
            }
            with self.assertRaisesRegex(ValueError, "supplied together"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "stage_edit_learned_reviewer_checkpoint_directory": str(
                            checkpoint
                        ),
                    }
                )
            with self.assertRaisesRegex(ValueError, "existing directory"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "stage_edit_learned_reviewer_checkpoint_directory": str(
                            root / "missing"
                        ),
                        "stage_edit_learned_reviewer_production_training_report": str(
                            report
                        ),
                    }
                )
            with self.assertRaisesRegex(ValueError, "automatic group"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "edit_mode": "member",
                        "aspect_ratio": "9:16",
                        "target_member": "가을",
                        "stage_edit_learned_reviewer_checkpoint_directory": str(
                            checkpoint
                        ),
                        "stage_edit_learned_reviewer_production_training_report": str(
                            report
                        ),
                    }
                )
            with self.assertRaisesRegex(ValueError, "acquire_only"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "acquire_only": True,
                        "stage_edit_learned_reviewer_checkpoint_directory": str(
                            checkpoint
                        ),
                        "stage_edit_learned_reviewer_production_training_report": str(
                            report
                        ),
                    }
                )
    def test_professional_reference_profile_round_trips_and_survives_materialization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = root / "human-profile.json"
            profile.write_text("{}", encoding="utf-8")
            cadence_profile = root / "human-cadence-profile.json"
            cadence_profile.write_text("{}", encoding="utf-8")
            sources = tuple(root / f"source-{index}.mp4" for index in range(8))
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "sources": [],
                    "professional_reference_profile": str(profile),
                    "professional_reference_cadence_profile": str(
                        cadence_profile
                    ),
                    "output_directory": str(root / "out"),
                }
            )

            self.assertEqual(
                request.professional_reference_profile,
                profile.resolve(),
            )
            self.assertEqual(
                request.as_dict()["professional_reference_profile"],
                str(profile.resolve()),
            )
            self.assertEqual(
                request.professional_reference_cadence_profile,
                cadence_profile.resolve(),
            )
            self.assertEqual(
                request.as_dict()["professional_reference_cadence_profile"],
                str(cadence_profile.resolve()),
            )
            materialized = request.with_materialized_sources(sources)
            self.assertEqual(
                materialized.professional_reference_profile,
                profile.resolve(),
            )
            self.assertEqual(
                materialized.professional_reference_cadence_profile,
                cadence_profile.resolve(),
            )

    def test_professional_reference_profile_path_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            common = {
                "song": "BANG BANG",
                "edit_mode": "group",
                "source_mode": "automatic",
                "sources": [],
            }
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                AnalysisRequest.from_dict(
                    {**common, "professional_reference_profile": ""}
                )
            with self.assertRaisesRegex(ValueError, "existing file"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "professional_reference_profile": str(
                            root / "missing.json"
                        ),
                    }
                )
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                AnalysisRequest.from_dict(
                    {**common, "professional_reference_cadence_profile": ""}
                )
            with self.assertRaisesRegex(ValueError, "existing file"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "professional_reference_cadence_profile": str(
                            root / "missing-cadence.json"
                        ),
                    }
                )

    def test_legacy_request_defaults_to_local_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / "one.mp4", root / "two.mp4"]
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "sources": [str(path) for path in sources],
                    "output_directory": str(root / "out"),
                }
            )
        self.assertEqual(request.source_mode, SourceMode.LOCAL)
        self.assertFalse(request.acquire_only)

    def test_automatic_request_accepts_zero_sources_and_defaults_to_ten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                    "sources": [],
                    "output_directory": str(root / "out"),
                }
            )
        self.assertEqual(request.source_mode, SourceMode.AUTOMATIC)
        self.assertEqual(request.sources, ())
        self.assertEqual(request.desired_source_count, 10)
        self.assertTrue(
            str(request.source_cache_directory).endswith(
                "CrossCut Sources/IVE/I AM/group"
            )
        )
        self.assertTrue(
            str(request.catalog_cache_directory).endswith(
                "CrossCut Sources/IVE/_catalog"
            )
        )
        self.assertEqual(
            request.as_dict()["catalog_cache_directory"],
            str(request.catalog_cache_directory),
        )

    def test_catalog_cache_directory_is_optional_but_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            requested = root / "shared-catalog"
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                    "sources": [],
                    "catalog_cache_directory": str(requested),
                }
            )
            self.assertEqual(
                request.catalog_cache_directory,
                requested.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                AnalysisRequest.from_dict(
                    {
                        "song": "I AM",
                        "edit_mode": "group",
                        "source_mode": "automatic",
                        "sources": [],
                        "catalog_cache_directory": "",
                    }
                )

    def test_group_default_floor_is_eight_but_explicit_counts_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / f"stage-{index}.mp4" for index in range(11)]
            for source in sources:
                source.touch()

            default_request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "sources": [],
                    "output_directory": str(root / "default-out"),
                }
            )
            self.assertEqual(default_request.automatic_source_count_floor, 8)
            self.assertEqual(
                len(
                    default_request.with_materialized_sources(
                        sources[:8]
                    ).sources
                ),
                8,
            )

            exact_nine = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 9,
                    "sources": [],
                    "output_directory": str(root / "nine-out"),
                }
            )
            self.assertEqual(exact_nine.automatic_source_count_floor, 9)
            with self.assertRaisesRegex(ValueError, "between 9 and 9"):
                exact_nine.with_materialized_sources(sources[:8])

            exact_eleven = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 11,
                    "sources": [],
                    "output_directory": str(root / "eleven-out"),
                }
            )
            self.assertEqual(exact_eleven.automatic_source_count_floor, 11)
            with self.assertRaisesRegex(ValueError, "between 11 and 11"):
                exact_eleven.with_materialized_sources(sources[:10])
            with self.assertRaisesRegex(ValueError, "between 11 and 11"):
                AnalysisRequest.from_dict(
                    {
                        "song": "I AM",
                        "edit_mode": "group",
                        "aspect_ratio": "16:9",
                        "source_mode": "automatic",
                        "desired_source_count": 11,
                        "sources": [str(path) for path in sources[:10]],
                        "output_directory": str(root / "direct-out"),
                    }
                )

    def test_group_request_discards_irrelevant_member_selection(self) -> None:
        request = AnalysisRequest.from_dict(
            {
                "song": "I AM",
                "edit_mode": "group",
                "source_mode": "automatic",
                "sources": [],
                "target_member": "가을",
            }
        )

        self.assertIsNone(request.target_member)
        self.assertIsNone(request.as_dict()["target_member"])

    def test_automatic_count_and_acquire_only_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 8 and 24"):
            AnalysisRequest.from_dict(
                {
                    "song": "HEYA",
                    "edit_mode": "group",
                    "source_mode": "automatic",
                    "desired_source_count": 7,
                }
            )
        valid_eight = AnalysisRequest.from_dict(
            {
                "song": "HEYA",
                "edit_mode": "group",
                "source_mode": "automatic",
                "desired_source_count": 8,
            }
        )
        self.assertEqual(valid_eight.automatic_source_count_floor, 8)
        with tempfile.TemporaryDirectory() as temp:
            sources = [Path(temp) / "a.mp4", Path(temp) / "b.mp4"]
            for source in sources:
                source.touch()
            with self.assertRaisesRegex(ValueError, "automatic source mode"):
                AnalysisRequest.from_dict(
                    {
                        "song": "HEYA",
                        "edit_mode": "group",
                        "sources": [str(path) for path in sources],
                        "acquire_only": True,
                    }
                )

    def test_automatic_member_can_defer_anchors_to_safe_inference(self) -> None:
        request = AnalysisRequest.from_dict(
            {
                "song": "I AM",
                "edit_mode": "member",
                "aspect_ratio": "9:16",
                "target_member": "가을",
                "source_mode": "automatic",
                "sources": [],
                "member_anchors": [],
            }
        )
        self.assertEqual(request.target_member, IVEMember.GAEUL)
        self.assertEqual(request.member_anchors, ())

    def test_automatic_member_keeps_target_ten_with_four_source_floor(self) -> None:
        request = AnalysisRequest.from_dict(
            {
                "song": "I AM",
                "edit_mode": "member",
                "aspect_ratio": "9:16",
                "target_member": "가을",
                "source_mode": "automatic",
                "desired_source_count": 10,
                "sources": [],
            }
        )
        self.assertEqual(request.desired_source_count, 10)
        self.assertEqual(request.automatic_source_count_floor, 4)
        with tempfile.TemporaryDirectory() as temp:
            sources = [Path(temp) / f"fancam-{index}.mp4" for index in range(4)]
            for source in sources:
                source.touch()
            self.assertEqual(
                len(request.with_materialized_sources(sources).sources),
                4,
            )
            with self.assertRaisesRegex(ValueError, "between 4 and 10"):
                request.with_materialized_sources(sources[:3])

        for requested_count in (4, 8, 9, 11, 24):
            with self.subTest(requested_count=requested_count), self.assertRaisesRegex(
                ValueError,
                "must remain 10",
            ):
                AnalysisRequest.from_dict(
                    {
                        "song": "I AM",
                        "edit_mode": "member",
                        "aspect_ratio": "9:16",
                        "target_member": "가을",
                        "source_mode": "automatic",
                        "desired_source_count": requested_count,
                        "sources": [],
                    }
                )

    def test_automatic_member_rejects_landscape_before_legacy_tracking(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires original 9:16 fancams"):
            AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "member",
                    "aspect_ratio": "16:9",
                    "target_member": "가을",
                    "source_mode": "automatic",
                    "sources": [],
                }
            )

    def test_automatic_member_rejects_confirmed_anchors(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not accept member anchors"):
            AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "member",
                    "aspect_ratio": "9:16",
                    "target_member": "가을",
                    "source_mode": "automatic",
                    "sources": [],
                    "member_anchors": [
                        {
                            "source_index": 0,
                            "time_seconds": 1.0,
                            "bbox_xyxy": [0.2, 0.1, 0.6, 0.9],
                        }
                    ],
                }
            )

    def test_materialized_automatic_member_accepts_four_to_ten_with_empty_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / f"fancam-{index}.mp4" for index in range(11)]
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "member",
                    "aspect_ratio": "9:16",
                    "target_member": "가을",
                    "source_mode": "automatic",
                    "sources": [],
                }
            )

            deferred = request.with_materialized_sources(sources[:10], member_anchors=())
            self.assertEqual(deferred.member_anchors, ())
            self.assertEqual(
                len(request.with_materialized_sources(sources[:4]).sources),
                4,
            )
            self.assertEqual(
                len(request.with_materialized_sources(sources[:9]).sources),
                9,
            )

            partial = (
                MemberAnchorRequest(
                    source_index=0,
                    time_seconds=1.0,
                    bbox_xyxy=(0.2, 0.1, 0.6, 0.9),
                ),
            )
            with self.assertRaisesRegex(ValueError, "does not accept member anchors"):
                request.with_materialized_sources(sources[:10], member_anchors=partial)

            complete = tuple(
                MemberAnchorRequest(
                    source_index=index,
                    time_seconds=1.0,
                    bbox_xyxy=(0.2, 0.1, 0.6, 0.9),
                )
                for index in range(10)
            )
            with self.assertRaisesRegex(ValueError, "does not accept member anchors"):
                request.with_materialized_sources(
                    sources[:10],
                    member_anchors=complete,
                )

            with self.assertRaisesRegex(ValueError, "between 4 and 10"):
                request.with_materialized_sources(sources[:3], member_anchors=())
            with self.assertRaisesRegex(ValueError, "between 4 and 10"):
                request.with_materialized_sources(sources, member_anchors=())

    def test_member_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / "one.mp4", root / "two.mp4"]
            for source in sources:
                source.touch()
            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "member",
                    "aspect_ratio": "9:16",
                    "target_member": "리즈",
                    "sources": [str(path) for path in sources],
                    "member_anchors": [
                        {
                            "source_index": index,
                            "time_seconds": 0.5,
                            "bbox_xyxy": [0.2, 0.1, 0.5, 0.9],
                        }
                        for index in range(len(sources))
                    ],
                    "output_directory": str(root / "out"),
                }
            )
        self.assertEqual(request.edit_mode, EditMode.MEMBER)
        self.assertEqual(request.target_member, IVEMember.LIZ)

    def test_member_is_required_for_member_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / "one.mp4", root / "two.mp4"]
            for source in sources:
                source.touch()
            with self.assertRaisesRegex(ValueError, "target_member"):
                AnalysisRequest.from_dict(
                    {
                        "song": "LOVE DIVE",
                        "edit_mode": "member",
                        "sources": [str(path) for path in sources],
                        "member_anchors": [
                            {
                                "source_index": index,
                                "time_seconds": 0.0,
                                "bbox_xyxy": [0.1, 0.1, 0.4, 0.9],
                            }
                            for index in range(len(sources))
                        ],
                    }
                )

    def test_strict_values_and_unique_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "one.mp4"
            second_source = Path(temp) / "two.mp4"
            source.touch()
            second_source.touch()
            common = {
                "song": "I AM",
                "edit_mode": "group",
                "aspect_ratio": "16:9",
                "sources": [str(source), str(source)],
            }
            with self.assertRaisesRegex(ValueError, "unique"):
                AnalysisRequest.from_dict(common)

            with self.assertRaisesRegex(ValueError, "song must be a string"):
                AnalysisRequest.from_dict({**common, "song": None})

            with self.assertRaisesRegex(ValueError, "reference_source must be an integer"):
                AnalysisRequest.from_dict(
                    {
                        **common,
                        "sources": [str(source), str(second_source)],
                        "reference_source": True,
                    }
                )

    def test_group_mode_rejects_portrait_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / "one.mp4", root / "two.mp4"]
            for source in sources:
                source.touch()
            with self.assertRaisesRegex(ValueError, "original 16:9"):
                AnalysisRequest.from_dict(
                    {
                        "song": "HEYA",
                        "edit_mode": "group",
                        "aspect_ratio": "9:16",
                        "sources": [str(path) for path in sources],
                    }
                )

    def test_member_mode_requires_one_valid_anchor_per_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / "one.mp4", root / "two.mp4"]
            for source in sources:
                source.touch()
            payload = {
                "song": "ATTITUDE",
                "edit_mode": "member",
                "aspect_ratio": "9:16",
                "target_member": "레이",
                "sources": [str(path) for path in sources],
                "member_anchors": [
                    {
                        "source_index": 0,
                        "time_seconds": 1.0,
                        "bbox_xyxy": [0.1, 0.1, 0.4, 0.9],
                    }
                ],
            }
            with self.assertRaisesRegex(ValueError, "every source"):
                AnalysisRequest.from_dict(payload)

            payload["member_anchors"] = [
                {
                    "source_index": index,
                    "time_seconds": 1.0,
                    "bbox_xyxy": [0.1, 0.1, 1.2, 0.9],
                }
                for index in range(2)
            ]
            with self.assertRaisesRegex(ValueError, "normalized"):
                AnalysisRequest.from_dict(payload)

    def test_member_mode_accepts_nine_synchronized_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / f"music-show-{index}.mp4" for index in range(9)]
            for source in sources:
                source.touch()

            request = AnalysisRequest.from_dict(
                {
                    "song": "I AM",
                    "edit_mode": "member",
                    "aspect_ratio": "9:16",
                    "target_member": "가을",
                    "sources": [str(path) for path in sources],
                    "member_anchors": [
                        {
                            "source_index": index,
                            "time_seconds": index / 10,
                            "bbox_xyxy": [0.2, 0.1, 0.5, 0.9],
                        }
                        for index in range(9)
                    ],
                }
            )

        self.assertEqual(len(request.sources), 9)
        self.assertEqual(len(request.member_anchors), 9)


if __name__ == "__main__":
    unittest.main()
