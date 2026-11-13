from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.transition_contact_sheets import (
    ContactSheetConfig,
    OpenCVFrameDecoder,
    VideoMetadata,
    _atomic_write_bytes,
    _ascii_sheet_label,
    _letterbox,
    generate_transition_contact_sheets,
)


class _SyntheticDecoder:
    def __init__(
        self,
        metadata: VideoMetadata,
        *,
        omitted_indices: tuple[int, ...] = (),
        frame_shape: tuple[int, int] = (90, 160),
    ) -> None:
        self._metadata = metadata
        self.omitted_indices = set(omitted_indices)
        self.frame_shape = frame_shape
        self.requested: tuple[int, ...] = ()
        self.closed = False

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def iter_frames(self, frame_indices):  # type: ignore[no-untyped-def]
        self.requested = tuple(frame_indices)
        height, width = self.frame_shape
        for frame_index in frame_indices:
            if frame_index in self.omitted_indices:
                continue
            color = (
                (frame_index * 17) % 256,
                (frame_index * 29) % 256,
                (frame_index * 43) % 256,
            )
            yield (
                frame_index,
                np.full((height, width, 3), color, dtype=np.uint8),
            )

    def close(self) -> None:
        self.closed = True


def _fixture(
    root: Path,
    *,
    segments: tuple[tuple[int, float], ...] = (
        (0, 0.4),
        (0, 0.3),
        (1, 0.5),
        (2, 0.4),
    ),
    fps: float = 10.0,
) -> tuple[Path, Path, int]:
    video = root / "IVE-TEST-20260823-120000-group-crosscut.mp4"
    video.write_bytes(b"synthetic-final-render")
    master_start = 0.0
    serialized_segments: list[dict[str, object]] = []
    for source_index, duration in segments:
        serialized_segments.append(
            {
                "source_index": source_index,
                "source_start": master_start,
                "duration": duration,
                "master_start": master_start,
            }
        )
        master_start += duration
    output_frame_count = int(round(master_start * fps))
    plan = root / "IVE-TEST-20260823-120000-render-plan.json"
    plan.write_text(
        json.dumps(
            {
                "sources": [
                    "/private/source-zero.mp4",
                    "/private/source-one.mp4",
                    "/private/source-two.mp4",
                ],
                "segments": serialized_segments,
                "output": str(video.resolve()),
                "fps": fps,
                "width": 160,
                "height": 90,
                "output_frame_count": output_frame_count,
                "transitions": [
                    {"kind": "hard_cut", "frame_count": 0}
                    for _ in range(len(segments) - 1)
                ],
            }
        ),
        encoding="utf-8",
    )
    return (plan, video, output_frame_count)


class ContactSheetConfigTests(unittest.TestCase):
    def test_requires_ordered_outgoing_and_boundary_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            ContactSheetConfig(frame_offsets=(-1, -2, 0))
        with self.assertRaisesRegex(ValueError, "outgoing"):
            ContactSheetConfig(frame_offsets=(0, 1))
        with self.assertRaisesRegex(ValueError, "include"):
            ContactSheetConfig(frame_offsets=(-2, -1, 1))

    def test_letterbox_preserves_four_by_three_aspect(self) -> None:
        source = np.full((90, 120, 3), (10, 80, 210), dtype=np.uint8)
        fitted = _letterbox(source, 160, 90)
        self.assertEqual(fitted.shape, (90, 160, 3))
        self.assertTrue(np.all(fitted[:, :20] == 12))
        self.assertTrue(np.all(fitted[:, 20:140] == source))
        self.assertTrue(np.all(fitted[:, 140:] == 12))

    def test_non_ascii_sheet_labels_fall_back_to_distinct_source_ids(self) -> None:
        self.assertEqual(_ascii_sheet_label("가을.mp4", "source-0"), "source-0")
        self.assertEqual(_ascii_sheet_label("레이.mp4", "source-1"), "source-1")
        self.assertEqual(
            _ascii_sheet_label("source-zero.mp4", "source-0"),
            "source-zero.mp4",
        )


class OpenCVFrameDecoderTests(unittest.TestCase):
    def test_reads_from_zero_sequentially_without_random_seek(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.position = 0
                self.released = False
                self.seek_count = 0

            def isOpened(self) -> bool:
                return True

            def release(self) -> None:
                self.released = True

            def get(self, property_id: int) -> float:
                return {
                    cv2.CAP_PROP_FRAME_COUNT: 6.0,
                    cv2.CAP_PROP_FRAME_WIDTH: 160.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 90.0,
                    cv2.CAP_PROP_FPS: 10.0,
                }[property_id]

            def read(self):  # type: ignore[no-untyped-def]
                frame_index = self.position
                self.position += 1
                if frame_index >= 6:
                    return (False, None)
                return (
                    True,
                    np.full((90, 160, 3), frame_index, dtype=np.uint8),
                )

            def set(self, _property_id: int, _value: float) -> bool:
                self.seek_count += 1
                return True

        capture = FakeCapture()
        with patch(
            "crosscut.transition_contact_sheets.cv2.VideoCapture",
            return_value=capture,
        ):
            decoder = OpenCVFrameDecoder(Path("synthetic.mp4"))
            decoded = list(decoder.iter_frames((1, 3)))
            decoder.close()

        self.assertEqual([index for index, _frame in decoded], [1, 3])
        self.assertTrue(np.all(decoded[0][1] == 1))
        self.assertTrue(np.all(decoded[1][1] == 3))
        # Six declared frames plus one EOF read prove the entire final stream.
        self.assertEqual(capture.position, 7)
        self.assertEqual(capture.seek_count, 0)
        self.assertTrue(capture.released)

    def test_inconsistent_backend_eof_is_rejected(self) -> None:
        class InconsistentEOFCapture:
            def __init__(self) -> None:
                self.position = 0

            def isOpened(self) -> bool:
                return True

            def release(self) -> None:
                pass

            def get(self, property_id: int) -> float:
                return {
                    cv2.CAP_PROP_FRAME_COUNT: 2.0,
                    cv2.CAP_PROP_FRAME_WIDTH: 160.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 90.0,
                    cv2.CAP_PROP_FPS: 10.0,
                }[property_id]

            def read(self):  # type: ignore[no-untyped-def]
                frame_index = self.position
                self.position += 1
                if frame_index >= 2:
                    return (True, None)
                return (
                    True,
                    np.full((90, 160, 3), frame_index, dtype=np.uint8),
                )

        with patch(
            "crosscut.transition_contact_sheets.cv2.VideoCapture",
            return_value=InconsistentEOFCapture(),
        ):
            decoder = OpenCVFrameDecoder(Path("inconsistent-eof.mp4"))
            with self.assertRaisesRegex(RuntimeError, "exact clean EOF"):
                list(decoder.iter_frames((0,)))
            decoder.close()

    def test_truncated_tail_raises_at_the_exact_missing_frame(self) -> None:
        class TruncatedCapture:
            def __init__(self) -> None:
                self.position = 0
                self.released = False

            def isOpened(self) -> bool:
                return True

            def release(self) -> None:
                self.released = True

            def get(self, property_id: int) -> float:
                return {
                    cv2.CAP_PROP_FRAME_COUNT: 6.0,
                    cv2.CAP_PROP_FRAME_WIDTH: 160.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 90.0,
                    cv2.CAP_PROP_FPS: 10.0,
                }[property_id]

            def read(self):  # type: ignore[no-untyped-def]
                frame_index = self.position
                self.position += 1
                if frame_index == 4:
                    return (False, None)
                return (
                    True,
                    np.full((90, 160, 3), frame_index, dtype=np.uint8),
                )

        capture = TruncatedCapture()
        with patch(
            "crosscut.transition_contact_sheets.cv2.VideoCapture",
            return_value=capture,
        ):
            decoder = OpenCVFrameDecoder(Path("truncated.mp4"))
            with self.assertRaisesRegex(RuntimeError, "exact frame 4"):
                list(decoder.iter_frames((1, 3)))
            decoder.close()

        self.assertTrue(capture.released)


class TransitionContactSheetTests(unittest.TestCase):
    def test_samples_only_real_source_switches_and_paginates_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, frame_count = _fixture(root)
            decoder = _SyntheticDecoder(
                VideoMetadata(fps=10.0, width=160, height=90, frame_count=frame_count)
            )
            output = root / "contact-sheets"

            result = generate_transition_contact_sheets(
                plan,
                video,
                output,
                config=ContactSheetConfig(
                    boundaries_per_page=1,
                    tile_width=160,
                    tile_height=90,
                ),
                decoder_factory=lambda path: decoder,
            )

            self.assertTrue(decoder.closed)
            # The first segment boundary keeps source 0 and must not be sampled.
            self.assertEqual(decoder.requested, (5, 6, 7, 8, 10, 11, 12, 13))
            self.assertEqual(result.source_switch_count, 2)
            self.assertEqual(result.sampled_frame_count, 8)
            self.assertEqual(
                [path.name for path in result.page_paths],
                [
                    f"{video.stem}-source-switches-page-001-of-002.png",
                    f"{video.stem}-source-switches-page-002-of-002.png",
                ],
            )
            for page_path in result.page_paths:
                page = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
                self.assertIsNotNone(page)
                assert page is not None
                self.assertEqual(page.shape[:2], (274, 712))
            assert result.manifest_path is not None
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["source_switch_count"], 2)
        self.assertEqual(manifest["page_count"], 2)
        self.assertTrue(manifest["sampling"]["source_switch_only"])
        self.assertFalse(manifest["sampling"]["raw_source_frames_persisted"])
        self.assertFalse(manifest["sampling"]["embeddings_persisted"])
        self.assertEqual(
            [item["plan_boundary_index"] for item in manifest["boundaries"]],
            [1, 2],
        )
        self.assertEqual(
            [item["boundary_frame_index"] for item in manifest["boundaries"]],
            [7, 12],
        )
        self.assertEqual(
            [sample["frame_index"] for sample in manifest["boundaries"][0]["samples"]],
            [5, 6, 7, 8],
        )
        self.assertNotIn("/private/source-zero.mp4", json.dumps(manifest))

    def test_missing_exact_frame_fails_before_any_output_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, frame_count = _fixture(root)
            decoder = _SyntheticDecoder(
                VideoMetadata(fps=10.0, width=160, height=90, frame_count=frame_count),
                omitted_indices=(13,),
            )
            output = root / "contact-sheets"

            with self.assertRaisesRegex(RuntimeError, "did not decode exact frames"):
                generate_transition_contact_sheets(
                    plan,
                    video,
                    output,
                    decoder_factory=lambda path: decoder,
                )

            self.assertTrue(decoder.closed)
            self.assertFalse(output.exists())

    def test_decoded_frame_dimensions_must_match_the_final_video_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, frame_count = _fixture(root)
            decoder = _SyntheticDecoder(
                VideoMetadata(fps=10.0, width=160, height=90, frame_count=frame_count),
                frame_shape=(100, 100),
            )
            output = root / "contact-sheets"

            with self.assertRaisesRegex(RuntimeError, "do not match final video"):
                generate_transition_contact_sheets(
                    plan,
                    video,
                    output,
                    decoder_factory=lambda path: decoder,
                )

            self.assertTrue(decoder.closed)
            self.assertFalse(output.exists())

    def test_video_contract_mismatch_fails_closed_and_closes_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, frame_count = _fixture(root)
            for metadata, message in (
                (
                    VideoMetadata(
                        fps=10.049,
                        width=160,
                        height=90,
                        frame_count=frame_count,
                    ),
                    "FPS",
                ),
                (
                    VideoMetadata(
                        fps=10.0,
                        width=160,
                        height=90,
                        frame_count=frame_count + 1,
                    ),
                    "frame count",
                ),
                (
                    VideoMetadata(
                        fps=10.0,
                        width=320,
                        height=180,
                        frame_count=frame_count,
                    ),
                    "dimensions",
                ),
            ):
                with self.subTest(message=message):
                    decoder = _SyntheticDecoder(metadata)
                    with self.assertRaisesRegex(RuntimeError, message):
                        generate_transition_contact_sheets(
                            plan,
                            video,
                            root / "out",
                            decoder_factory=lambda path, item=decoder: item,
                        )
                    self.assertTrue(decoder.closed)

    def test_plan_must_name_the_exact_supplied_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, _frame_count = _fixture(root)
            different = root / "different.mp4"
            different.write_bytes(b"not-the-plan-output")
            factory = patch("crosscut.transition_contact_sheets.OpenCVFrameDecoder")

            with factory as decoder_factory, self.assertRaisesRegex(
                ValueError,
                "exact output path",
            ):
                generate_transition_contact_sheets(
                    plan,
                    different,
                    root / "out",
                    decoder_factory=decoder_factory,
                )
            decoder_factory.assert_not_called()
            self.assertTrue(video.exists())

    def test_malformed_transition_contracts_fail_before_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, _frame_count = _fixture(root)
            original = json.loads(plan.read_text(encoding="utf-8"))
            malformed: list[tuple[str, dict[str, object], str]] = []

            missing = dict(original)
            missing.pop("transitions")
            malformed.append(("missing", missing, "one transition"))

            hard_cut_overlap = json.loads(json.dumps(original))
            hard_cut_overlap["transitions"][1]["frame_count"] = 1
            malformed.append(("hard-cut overlap", hard_cut_overlap, "zero overlap"))

            short_dissolve = json.loads(json.dumps(original))
            short_dissolve["transitions"][1] = {
                "kind": "face_dissolve",
                "frame_count": 2,
            }
            malformed.append(("short dissolve", short_dissolve, "at least three"))

            same_source_dissolve = json.loads(json.dumps(original))
            same_source_dissolve["transitions"][0] = {
                "kind": "face_dissolve",
                "frame_count": 3,
            }
            malformed.append(
                ("same-source dissolve", same_source_dissolve, "real source change")
            )

            long_dissolve = json.loads(json.dumps(original))
            long_dissolve["transitions"][1] = {
                "kind": "face_dissolve",
                "frame_count": 6,
            }
            malformed.append(("long dissolve", long_dissolve, "incoming segment"))

            unknown_kind = json.loads(json.dumps(original))
            unknown_kind["transitions"][1]["kind"] = "mystery"
            malformed.append(("unknown kind", unknown_kind, "kind is invalid"))

            for name, payload, message in malformed:
                with self.subTest(name=name):
                    plan.write_text(json.dumps(payload), encoding="utf-8")
                    factory = patch(
                        "crosscut.transition_contact_sheets.OpenCVFrameDecoder"
                    )
                    with factory as decoder_factory, self.assertRaisesRegex(
                        ValueError,
                        message,
                    ):
                        generate_transition_contact_sheets(
                            plan,
                            video,
                            root / "out",
                            decoder_factory=decoder_factory,
                        )
                    decoder_factory.assert_not_called()

    def test_dissolve_samples_are_labeled_as_transition_until_overlap_ends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, frame_count = _fixture(root)
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["transitions"][1] = {
                "kind": "face_dissolve",
                "frame_count": 3,
            }
            plan.write_text(json.dumps(payload), encoding="utf-8")
            decoder = _SyntheticDecoder(
                VideoMetadata(fps=10.0, width=160, height=90, frame_count=frame_count)
            )

            result = generate_transition_contact_sheets(
                plan,
                video,
                root / "out",
                decoder_factory=lambda path: decoder,
            )
            assert result.manifest_path is not None
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        first_switch = manifest["boundaries"][0]
        self.assertEqual(first_switch["transition"]["kind"], "face_dissolve")
        self.assertEqual(
            [sample["side"] for sample in first_switch["samples"]],
            ["outgoing", "outgoing", "transition", "transition"],
        )

    def test_generated_manifest_path_cannot_overwrite_the_render_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_plan, video, _frame_count = _fixture(root)
            colliding_plan = root / f"{video.stem}-source-switches-manifest.json"
            original_plan.replace(colliding_plan)
            original_bytes = colliding_plan.read_bytes()
            factory = patch(
                "crosscut.transition_contact_sheets.OpenCVFrameDecoder"
            )

            with factory as decoder_factory, self.assertRaisesRegex(
                ValueError,
                "cannot overwrite an input artifact",
            ):
                generate_transition_contact_sheets(
                    colliding_plan,
                    video,
                    root,
                    decoder_factory=decoder_factory,
                )

            decoder_factory.assert_not_called()
            self.assertEqual(colliding_plan.read_bytes(), original_bytes)

    def test_all_pages_are_encoded_before_the_first_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, frame_count = _fixture(root)
            decoder = _SyntheticDecoder(
                VideoMetadata(fps=10.0, width=160, height=90, frame_count=frame_count)
            )
            output = root / "out"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("unchanged", encoding="utf-8")

            with patch(
                "crosscut.transition_contact_sheets._encode_png",
                side_effect=(b"first-page", RuntimeError("synthetic encode failure")),
            ), self.assertRaisesRegex(RuntimeError, "synthetic encode failure"):
                generate_transition_contact_sheets(
                    plan,
                    video,
                    output,
                    config=ContactSheetConfig(boundaries_per_page=1),
                    decoder_factory=lambda path: decoder,
                )

            self.assertEqual([path.name for path in output.iterdir()], ["keep.txt"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_repeat_generation_produces_identical_png_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, frame_count = _fixture(root)
            output = root / "out"

            first = generate_transition_contact_sheets(
                plan,
                video,
                output,
                decoder_factory=lambda path: _SyntheticDecoder(
                    VideoMetadata(
                        fps=10.0,
                        width=160,
                        height=90,
                        frame_count=frame_count,
                    )
                ),
            )
            first_pages = [path.read_bytes() for path in first.page_paths]
            second = generate_transition_contact_sheets(
                plan,
                video,
                output,
                decoder_factory=lambda path: _SyntheticDecoder(
                    VideoMetadata(
                        fps=10.0,
                        width=160,
                        height=90,
                        frame_count=frame_count,
                    )
                ),
            )

            self.assertEqual(
                [path.read_bytes() for path in second.page_paths],
                first_pages,
            )

    def test_manifest_output_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, frame_count = _fixture(root)
            result = generate_transition_contact_sheets(
                plan,
                video,
                root / "out",
                decoder_factory=lambda path: _SyntheticDecoder(
                    VideoMetadata(
                        fps=10.0,
                        width=160,
                        height=90,
                        frame_count=frame_count,
                    )
                ),
                write_manifest=False,
            )

            self.assertIsNone(result.manifest_path)
            self.assertTrue(result.page_paths)
            self.assertFalse(list((root / "out").glob("*.json")))

    def test_atomic_replace_failure_preserves_existing_file_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "page.png"
            destination.write_bytes(b"existing-page")

            with patch(
                "crosscut.transition_contact_sheets.os.replace",
                side_effect=OSError("synthetic replace failure"),
            ), self.assertRaisesRegex(OSError, "synthetic replace failure"):
                _atomic_write_bytes(destination, b"replacement-page")

            self.assertEqual(destination.read_bytes(), b"existing-page")
            self.assertEqual(list(root.glob(".page.png.*.tmp")), [])

    def test_input_replacement_during_decode_fails_before_output_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, frame_count = _fixture(root)

            class MutatingDecoder(_SyntheticDecoder):
                def iter_frames(self, frame_indices):  # type: ignore[no-untyped-def]
                    yield from super().iter_frames(frame_indices)
                    video.write_bytes(b"replacement-final-render-with-new-size")

            decoder = MutatingDecoder(
                VideoMetadata(fps=10.0, width=160, height=90, frame_count=frame_count)
            )
            output = root / "out"

            with self.assertRaisesRegex(RuntimeError, "input artifact changed"):
                generate_transition_contact_sheets(
                    plan,
                    video,
                    output,
                    decoder_factory=lambda path: decoder,
                )

            self.assertTrue(decoder.closed)
            self.assertFalse(output.exists())

    def test_no_source_switch_is_rejected_without_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, video, _frame_count = _fixture(
                root,
                segments=((0, 0.4), (0, 0.4)),
            )
            decoder_factory = patch(
                "crosscut.transition_contact_sheets.OpenCVFrameDecoder"
            )
            with decoder_factory as factory, self.assertRaisesRegex(
                ValueError,
                "no source-switch",
            ):
                generate_transition_contact_sheets(
                    plan,
                    video,
                    root / "out",
                    decoder_factory=factory,
                )
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
