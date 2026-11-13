from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from crosscut.request import AspectRatio
from crosscut.timeline_interchange import (
    ANALYSIS_BASIS,
    build_timeline_interchange_bundle,
    write_timeline_interchange_bundle,
)


def _plan(root: Path, *, second_duration: float = 2.0) -> RenderPlan:
    sources = (root / "방송 A.mp4", root / "stage-B.mp4")
    for source in sources:
        source.write_bytes(b"synthetic-media")
    return RenderPlan(
        sources=sources,
        segments=(
            RenderSegment(
                source_index=0,
                source_start=1.25,
                source_duration=1.001,
                master_start=0.0,
                duration=1.0,
            ),
            RenderSegment(
                source_index=1,
                source_start=2.5,
                master_start=1.0,
                duration=1.0,
            ),
            RenderSegment(
                source_index=1,
                source_start=3.5,
                source_duration=second_duration,
                master_start=2.0,
                duration=second_duration,
                editorial_transform=EditorialTransformTrack(
                    keyframes=(
                        EditorialTransformKeyframe(0.0, 1.05),
                        EditorialTransformKeyframe(0.75, 1.0),
                    ),
                    reason="test_camera_ramp",
                ),
            ),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        output=root / "final.mp4",
        master_audio_start=1.25,
        width=1920,
        height=1080,
        fps=30000 / 1001,
        transitions=(
            RenderTransition(
                kind=TransitionKind.FACE_DISSOLVE,
                frame_count=5,
                reason="verified_face_match",
                tier="A",
            ),
            RenderTransition(reason="camera_ramp_hard_cut"),
        ),
    )


class TimelineInterchangeTests(unittest.TestCase):
    def test_bundle_preserves_video_clocks_master_audio_and_nonportable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = _plan(Path(directory))
            bundle = build_timeline_interchange_bundle(
                plan,
                project_name="IVE 교차편집",
            )

            xml = bundle.fcpxml.decode("utf-8")
            parsed = ET.fromstring(xml.split("<!DOCTYPE fcpxml>\n", 1)[1])
            clips = parsed.findall("./library/event/project/sequence/spine/asset-clip")
            self.assertEqual(len(clips), 4)
            self.assertEqual(clips[-1].attrib["name"], "Continuous master audio")
            self.assertEqual(clips[-1].attrib["lane"], "-1")
            self.assertEqual(len(clips[1].findall("marker")), 1)
            metadata_value = clips[2].find("metadata/md").attrib["value"]
            self.assertEqual(
                json.loads(metadata_value)["editorial_transform"]["reason"],
                "test_camera_ramp",
            )

            otio = json.loads(bundle.otio)
            tracks = otio["tracks"]["children"]
            self.assertEqual([item["kind"] for item in tracks], ["Video", "Audio"])
            self.assertEqual(len(tracks[0]["children"]), 3)
            self.assertEqual(len(tracks[1]["children"]), 1)
            self.assertEqual(
                tracks[0]["children"][0]["source_range"]["duration"]["value"],
                30,
            )
            self.assertAlmostEqual(
                tracks[0]["children"][0]["effects"][0]["time_scalar"],
                1.001,
            )
            self.assertEqual(
                otio["metadata"]["crosscut"]["transitions"][0]["frame_count"],
                5,
            )

            edl = bundle.edl.decode("utf-8")
            self.assertIn("FCM: NON-DROP FRAME", edl)
            self.assertIn("D 005", edl)
            self.assertIn("MASTER AUDIO SOURCE", edl)

            audit = json.loads(bundle.audit)
            self.assertEqual(audit["analysis_basis"], ANALYSIS_BASIS)
            self.assertEqual(
                audit["fidelity"]["dynamic_crop_and_editorial_camera"],
                "metadata_only",
            )
            self.assertEqual(len(audit["content_sha256"]), 64)

    def test_writer_is_idempotent_but_refuses_different_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _plan(root)
            paths = write_timeline_interchange_bundle(
                plan,
                output_directory=root / "output",
                base_filename="IVE-TEST-timeline",
                project_name="IVE TEST",
            )
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(
                paths,
                write_timeline_interchange_bundle(
                    plan,
                    output_directory=root / "output",
                    base_filename="IVE-TEST-timeline",
                    project_name="IVE TEST",
                ),
            )

            changed = _plan(root, second_duration=2.5)
            with self.assertRaisesRegex(FileExistsError, "differs"):
                write_timeline_interchange_bundle(
                    changed,
                    output_directory=root / "output",
                    base_filename="IVE-TEST-timeline",
                    project_name="IVE TEST",
                )


if __name__ == "__main__":
    unittest.main()
