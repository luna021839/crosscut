from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut.stage_edit_source_pair_production_cli import (
    _song_map,
    build_parser,
    main,
)


class StageEditSourcePairProductionCliTests(unittest.TestCase):
    def test_song_map_is_exact_and_rejects_duplicates(self) -> None:
        self.assertEqual(
            {"ref-a": "IVE-A", "ref-b": "IVE-B"},
            _song_map(("ref-a=IVE-A", "ref-b=IVE-B")),
        )
        with self.assertRaisesRegex(ValueError, "duplicate song id"):
            _song_map(("ref-a=IVE-A", "ref-a=IVE-B"))

    def test_parser_keeps_production_gate_defaults_explicit(self) -> None:
        parsed = build_parser().parse_args(
            [
                "--input",
                "/tmp/reference.json",
                "--song-id",
                "ref-a=IVE-A",
                "--feature-snapshot",
                "ref-a=/tmp/features.json",
                "--destination",
                "/tmp/source-pair",
            ]
        )
        self.assertEqual(250, parsed.steps)
        self.assertEqual(0.25, parsed.learning_rate)
        self.assertEqual(0.05, parsed.margin)
        self.assertEqual(1.0e-12, parsed.activation_tolerance)

    def test_main_routes_the_exact_corpus_and_publication_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            destination = root / "source-pair"
            corpus = SimpleNamespace(data_id="a" * 64)
            production_result = SimpleNamespace(
                checkpoint_path=destination / "source-pair-scorer.json",
                report_path=destination / "source-pair-production-report.json",
                report=SimpleNamespace(
                    checkpoint=SimpleNamespace(
                        checkpoint_sha256="b" * 64,
                        corpus_sha256="c" * 64,
                    ),
                    report_sha256="d" * 64,
                    cross_validation=SimpleNamespace(
                        corpus=SimpleNamespace(
                            song_count=4,
                            reference_count=20,
                            group_count=100,
                        )
                    ),
                ),
            )
            with (
                patch(
                    "crosscut.stage_edit_source_pair_production_cli._snapshot_map",
                    return_value={"ref-a": object()},
                ),
                patch(
                    "crosscut.stage_edit_source_pair_production_cli."
                    "load_reference_training_corpus",
                    return_value=corpus,
                ) as load,
                patch(
                    "crosscut.stage_edit_source_pair_production_cli."
                    "produce_stage_edit_source_pair_checkpoint",
                    return_value=production_result,
                ) as produce,
                patch("builtins.print") as output,
            ):
                code = main(
                    [
                        "--input",
                        str(root / "reference.json"),
                        "--song-id",
                        "ref-a=IVE-A",
                        "--feature-snapshot",
                        f"ref-a={root / 'features.json'}",
                        "--destination",
                        str(destination),
                        "--checkpoint-id",
                        "pair-release-v1",
                        "--created-at",
                        "2026-08-24T00:00:00+00:00",
                    ]
                )

            self.assertEqual(0, code)
            self.assertEqual({"ref-a": "IVE-A"}, load.call_args.kwargs["song_id_by_reference"])
            produce.assert_called_once_with(
                corpus,
                destination,
                checkpoint_id="pair-release-v1",
                created_at="2026-08-24T00:00:00+00:00",
                steps=250,
                learning_rate=0.25,
                margin=0.05,
                tolerance=1.0e-12,
            )
            printed = json.loads(output.call_args.args[0])
            self.assertTrue(printed["production_authorized"])
            self.assertEqual(4, printed["song_count"])

    def test_relative_destination_fails_before_loading_data(self) -> None:
        with (
            patch(
                "crosscut.stage_edit_source_pair_production_cli."
                "load_reference_training_corpus"
            ) as load,
            self.assertRaisesRegex(ValueError, "absolute path"),
        ):
            main(
                [
                    "--input",
                    "/tmp/reference.json",
                    "--song-id",
                    "ref-a=IVE-A",
                    "--feature-snapshot",
                    "ref-a=/tmp/features.json",
                    "--destination",
                    "relative-output",
                ]
            )
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
