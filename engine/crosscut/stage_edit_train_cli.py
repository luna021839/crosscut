"""Command-line entry point for the offline sparse StageEdit calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_edit_reference_adapter import load_reference_training_corpus
from .stage_edit_training import (
    sparse_calibration_stage_edit_config,
    train_sparse_stage_edit,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - diagnosed by the training runner.
    mx = None


def _song_map(values: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        reference, separator, song = value.partition("=")
        reference = reference.strip()
        song = song.strip()
        if not separator or not reference or not song:
            raise ValueError("--song-id must use REFERENCE_ID=SONG_ID")
        if reference in output and output[reference] != song:
            raise ValueError(f"conflicting song id for {reference}")
        output[reference] = song
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the small, offline StageEdit calibration model from "
            "reference-supervision JSON. This is not production readiness."
        )
    )
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--song-id", action="append", default=[])
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--adapted-manifest-path", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--negative-samples-per-hold", type=int, default=1)
    parser.add_argument("--boundary-guard-seconds", type=float, default=0.18)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--max-sources", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = load_reference_training_corpus(
        args.input,
        song_id_by_reference=_song_map(args.song_id),
        negative_samples_per_hold=args.negative_samples_per_hold,
        boundary_guard_seconds=args.boundary_guard_seconds,
        max_sources=args.max_sources,
    )
    if args.adapted_manifest_path is not None:
        corpus.write_manifest(args.adapted_manifest_path)
    config = sparse_calibration_stage_edit_config(
        max_sequence_length=args.max_sequence_length,
        max_sources=args.max_sources,
    )
    device = None
    if args.device != "auto":
        if mx is None:
            raise RuntimeError("MLX is required to select a training device")
        device = mx.cpu if args.device == "cpu" else mx.gpu
    result = train_sparse_stage_edit(
        corpus,
        checkpoint_directory=args.checkpoint_dir,
        report_path=args.report_path,
        model_config=config,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=device,
    )
    print(
        json.dumps(
            {
                "checkpoint_directory": str(result.checkpoint_directory),
                "report_path": str(result.report_path),
                "best_epoch": result.best_epoch,
                "best_training_step": result.best_training_step,
                "production_ready": False,
                "production_supervision_gate": result.report[
                    "production_supervision_gate"
                ],
                "best_validation_metrics": result.best_validation_metrics,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
