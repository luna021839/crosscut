from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from crosscut.ive import EditMode, IVEMember
from crosscut.request import AnalysisRequest


class RequestTests(unittest.TestCase):
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
