"""CLI for song-disjoint StageEdit source-pair production training."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Sequence

from .stage_edit_feature_materializer import StageEditFeatureSnapshot
from .stage_edit_reference_adapter import load_reference_training_corpus
from .stage_edit_source_pair_production import (
    produce_stage_edit_source_pair_checkpoint,
)
from .stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA


def _song_map(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        reference_id, separator, song_id = raw.partition("=")
        reference_id = reference_id.strip()
        song_id = song_id.strip()
        if not separator or not reference_id or not song_id:
            raise ValueError("--song-id must use REFERENCE_ID=SONG_ID")
        if reference_id in result:
            raise ValueError(f"duplicate song id for {reference_id}")
        result[reference_id] = song_id
    return result


def _snapshot_map(values: Sequence[str]) -> dict[str, StageEditFeatureSnapshot]:
    result: dict[str, StageEditFeatureSnapshot] = {}
    for raw in values:
        reference_id, separator, raw_path = raw.partition("=")
        reference_id = reference_id.strip()
        raw_path = raw_path.strip()
        if not separator or not reference_id or not raw_path:
            raise ValueError(
                "--feature-snapshot must use REFERENCE_ID=/absolute/path.json"
            )
        if reference_id in result:
            raise ValueError(f"duplicate feature snapshot for {reference_id}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError(
                f"feature snapshot must be an absolute non-symlink file: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"feature snapshot root must be an object: {path}")
        snapshot = StageEditFeatureSnapshot.from_dict(
            payload,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
        )
        snapshot.require_production_causal_sensing(
            edit_mode="group",
            target_member=None,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
        )
        result[reference_id] = snapshot
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize causal source pairs once, run leave-one-song-out "
            "ranking gates, then publish a full-corpus scorer only if every "
            "held-out fold beats both fixed baselines."
        )
    )
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--song-id", action="append", required=True)
    parser.add_argument(
        "--feature-snapshot",
        action="append",
        required=True,
        metavar="REFERENCE_ID=/ABSOLUTE/PATH.json",
    )
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--created-at")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--activation-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--negative-samples-per-hold", type=int, default=2)
    parser.add_argument("--boundary-guard-seconds", type=float, default=0.18)
    parser.add_argument("--max-sources", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = args.destination.expanduser()
    if not destination.is_absolute():
        raise ValueError("--destination must be an absolute path")
    if destination.is_symlink():
        raise ValueError("--destination must not be a symlink")
    corpus = load_reference_training_corpus(
        args.input,
        feature_snapshot_by_reference=_snapshot_map(args.feature_snapshot),
        song_id_by_reference=_song_map(args.song_id),
        negative_samples_per_hold=args.negative_samples_per_hold,
        boundary_guard_seconds=args.boundary_guard_seconds,
        max_sources=args.max_sources,
        production_edit_mode="group",
        production_target_member=None,
    )
    checkpoint_id = str(args.checkpoint_id or "").strip()
    if not checkpoint_id:
        checkpoint_id = f"stageedit-source-pair-{corpus.data_id[:16]}"
    created_at = str(args.created_at or "").strip()
    if not created_at:
        created_at = datetime.now(UTC).isoformat()
    result = produce_stage_edit_source_pair_checkpoint(
        corpus,
        destination.resolve(),
        checkpoint_id=checkpoint_id,
        created_at=created_at,
        steps=args.steps,
        learning_rate=args.learning_rate,
        margin=args.margin,
        tolerance=args.activation_tolerance,
    )
    print(
        json.dumps(
            {
                "checkpoint_path": str(result.checkpoint_path),
                "report_path": str(result.report_path),
                "checkpoint_sha256": result.report.checkpoint.checkpoint_sha256,
                "corpus_sha256": result.report.checkpoint.corpus_sha256,
                "report_sha256": result.report.report_sha256,
                "song_count": result.report.cross_validation.corpus.song_count,
                "reference_count": (
                    result.report.cross_validation.corpus.reference_count
                ),
                "group_count": result.report.cross_validation.corpus.group_count,
                "production_authorized": True,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
