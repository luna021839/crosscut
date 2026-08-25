from pathlib import Path
from types import SimpleNamespace
import unittest

from crosscut.ive import AspectRatio, EditMode, SourceMode
from crosscut.pipeline import (
    _attach_direct_member_portrait_fill_geometry,
    _automatic_member_portrait_source_ratios,
    _lossless_member_portrait_fill_crop,
)
from crosscut.render import RenderSegment


def _request(source_count: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        edit_mode=EditMode.MEMBER,
        source_mode=SourceMode.AUTOMATIC,
        aspect_ratio=AspectRatio.PORTRAIT,
        member_anchors=(),
        desired_source_count=10,
        sources=tuple(Path(f"source-{index}.mp4") for index in range(source_count)),
    )


def _media(width: int, height: int) -> SimpleNamespace:
    return SimpleNamespace(
        video_streams=(SimpleNamespace(width=width, height=height),)
    )


class AutomaticMemberPortraitFillTests(unittest.TestCase):
    def test_4k_landscape_facecam_passes_lossless_portrait_output_gate(self) -> None:
        ratios = _automatic_member_portrait_source_ratios(
            _request(),
            (
                _media(1080, 1920),
                _media(3840, 2160),
                _media(2160, 3840),
                _media(1440, 2560),
            ),
        )

        self.assertIsNotNone(ratios)
        assert ratios is not None
        self.assertAlmostEqual(ratios[1], 16.0 / 9.0)

    def test_full_hd_landscape_facecam_fails_without_portrait_headroom(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot cover 1080x1920"):
            _automatic_member_portrait_source_ratios(
                _request(),
                (
                    _media(1080, 1920),
                    _media(1920, 1080),
                    _media(2160, 3840),
                    _media(1440, 2560),
                ),
            )

    def test_landscape_fill_crop_is_centered_and_exactly_portrait(self) -> None:
        crop = _lossless_member_portrait_fill_crop(16.0 / 9.0)

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertAlmostEqual(crop.width, 81.0 / 256.0)
        self.assertAlmostEqual(crop.height, 1.0)
        self.assertAlmostEqual(crop.x, (1.0 - 81.0 / 256.0) / 2.0)
        self.assertAlmostEqual(crop.y, 0.0)

    def test_fill_geometry_only_claims_untransformed_landscape_span(self) -> None:
        selection = SimpleNamespace(source_indices=(0, 1))
        segments = (
            RenderSegment(2, 0.0, 1.0, 0.0),
            RenderSegment(5, 1.0, 1.0, 1.0),
        )

        attached = _attach_direct_member_portrait_fill_geometry(
            visual_selection=selection,
            eligible_indices=(2, 5),
            eligible_source_aspect_ratios=(9.0 / 16.0, 16.0 / 9.0),
            render_segments=segments,
        )

        self.assertIsNone(attached[0].crop)
        self.assertIsNotNone(attached[1].crop)
        assert attached[1].crop is not None
        self.assertAlmostEqual(
            attached[1].crop.width / attached[1].crop.height,
            81.0 / 256.0,
        )


if __name__ == "__main__":
    unittest.main()
