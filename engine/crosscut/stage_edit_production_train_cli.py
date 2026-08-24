"""CLI for strict song-held-out StageEdit production training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .stage_edit_dynamic_crop_training import (
    build_stage_edit_dynamic_crop_training_corpus_v2,
)
from .reference_dynamic_crop_job import (
    validate_reference_dynamic_crop_manifest,
)
from .reference_dynamic_crop_supervision import (
    load_reference_dynamic_crop_supervision,
)
from .stage_edit_feature_materializer import StageEditFeatureSnapshot
from .stage_edit_production_training import train_production_stage_edit
from .stage_edit_production_registry import (
    register_stage_edit_production_bundle,
)
from .stage_edit_reference_adapter import load_reference_training_corpus
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    production_stage_edit_config,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - diagnosed by the training function.
    mx = None


def _training_result_exit_code(
    *,
    activation_authorized: bool,
    production_ready: bool,
    quality_gate_passed: bool,
) -> int:
    """Separate evaluation acceptance from registry activation.

    The fixed fold launcher intentionally runs without activation authority so
    both unseen-song reports can be reviewed before one bundle is installed.
    A passed held-out gate is therefore a successful evaluation run even though
    ``production_ready`` correctly remains false.  Authorized runs retain the
    stricter production-ready requirement.
    """

    if activation_authorized:
        return 0 if production_ready else 2
    return 0 if quality_gate_passed else 2


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


def _snapshot_map(values: Sequence[str]) -> dict[str, StageEditFeatureSnapshot]:
    output: dict[str, StageEditFeatureSnapshot] = {}
    for value in values:
        reference, separator, raw_path = value.partition("=")
        reference = reference.strip()
        raw_path = raw_path.strip()
        if not separator or not reference or not raw_path:
            raise ValueError(
                "--feature-snapshot must use REFERENCE_ID=/absolute/path.json"
            )
        if reference in output:
            raise ValueError(f"duplicate feature snapshot for {reference}")
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"feature snapshot root must be an object: {path}")
        output[reference] = StageEditFeatureSnapshot.from_dict(
            payload,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
        )
    return output


def _artifact_map(values: Sequence[str]) -> dict[str, tuple[Path, ...]]:
    output: dict[str, list[Path]] = {}
    for value in values:
        reference, separator, raw_path = value.partition("=")
        reference = reference.strip()
        raw_path = raw_path.strip()
        if not separator or not reference or not raw_path:
            raise ValueError(
                "--dynamic-crop-artifact must use REFERENCE_ID=/absolute/path.json"
            )
        output.setdefault(reference, []).append(Path(raw_path))
    return {reference: tuple(paths) for reference, paths in output.items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dynamic_artifact_map(
    values: Sequence[str],
    manifest_paths: Sequence[Path],
) -> dict[str, tuple[Path, ...]]:
    """Resolve explicit artifacts plus complete immutable crop-v3 manifests."""

    combined: dict[str, list[Path]] = {
        reference: list(paths) for reference, paths in _artifact_map(values).items()
    }
    seen_paths = {
        path.expanduser().resolve()
        for paths in combined.values()
        for path in paths
    }
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                f"dynamic crop manifest root must be an object: {manifest_path}"
            )
        # Validate the immutable manifest and the small per-shot JSON here, but
        # leave provenance-media hashing to the corpus builder.  Its one cache
        # spans all three manifests, so shared source/reference media is never
        # hashed once per manifest and then again during corpus assembly.
        validate_reference_dynamic_crop_manifest(
            payload,
            verify_artifact_files=False,
        )
        records = payload.get("artifacts")
        assert isinstance(records, list)  # established by the strict validator
        for record in records:
            assert isinstance(record, dict)
            artifact = record.get("artifact")
            assert isinstance(artifact, dict)
            reference = str(record.get("reference_id") or "").strip()
            path = Path(str(artifact.get("path") or "")).resolve()
            if path in seen_paths:
                raise ValueError(
                    f"duplicate dynamic crop artifact input: {path}"
                )
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"dynamic crop artifact is unavailable: {path}")
            if _file_sha256(path) != artifact.get("file_sha256"):
                raise ValueError(f"dynamic crop artifact file hash mismatch: {path}")
            restored = load_reference_dynamic_crop_supervision(
                path,
                verify_input_files=False,
            )
            if (
                restored.payload["provenance"]["artifact_content_sha256"]
                != artifact.get("content_sha256")
            ):
                raise ValueError(
                    f"dynamic crop artifact content hash mismatch: {path}"
                )
            seen_paths.add(path)
            combined.setdefault(reference, []).append(path)
    return {
        reference: tuple(sorted(paths, key=lambda path: str(path)))
        for reference, paths in sorted(combined.items())
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the real long-context StageEdit Transformer with complete "
            "song-held-out evaluation. A failing head gate remains inactive."
        )
    )
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--song-id", action="append", required=True)
    parser.add_argument("--held-out-song", action="append", required=True)
    parser.add_argument(
        "--required-training-song",
        action="append",
        default=[],
        help=(
            "Fail closed unless this complete song is admitted to training. "
            "Use for a newly materialized song such as IVE-REBEL-HEART."
        ),
    )
    parser.add_argument(
        "--feature-snapshot",
        action="append",
        required=True,
        metavar="REFERENCE_ID=PATH",
    )
    parser.add_argument(
        "--dynamic-crop-artifact",
        action="append",
        default=[],
        metavar="REFERENCE_ID=PATH",
        help=(
            "Verified crop-v2 target artifact. Repeat for every retained shot; "
            "activation cannot pass without held-out trajectory gates. May be "
            "combined with complete --dynamic-crop-manifest inputs."
        ),
    )
    parser.add_argument(
        "--dynamic-crop-manifest",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help=(
            "Immutable reference-dynamic-crop-v3 manifest. Repeat once per "
            "song instead of expanding hundreds of per-shot arguments."
        ),
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--adapted-manifest-path", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--maximum-cut-positive-weight",
        type=float,
        default=8.0,
        help=(
            "Upper bound for the positive cut weight derived only from the "
            "training partition negative/positive ratio."
        ),
    )
    parser.add_argument(
        "--cut-pretrain-steps",
        type=int,
        default=500,
        help=(
            "Balanced causal cut-only pretraining steps. Use 0 only for an "
            "explicit diagnostic ablation; enabled runs require at least 500."
        ),
    )
    parser.add_argument(
        "--cut-pretrain-learning-rate", type=float, default=3.0e-4
    )
    parser.add_argument("--cut-pretrain-window-length", type=int, default=32)
    parser.add_argument("--cut-pretrain-warmup-steps", type=int, default=50)
    parser.add_argument("--finetune-warmup-steps", type=int, default=8)
    parser.add_argument(
        "--minimum-learning-rate-ratio", type=float, default=0.10
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--negative-samples-per-hold", type=int, default=2)
    parser.add_argument("--boundary-guard-seconds", type=float, default=0.18)
    parser.add_argument("--max-sources", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--authorize-production-activation",
        action="store_true",
        help=(
            "Allow activation only if every fixed unseen-song quality gate "
            "passes. This flag cannot override a failed metric."
        ),
    )
    parser.add_argument(
        "--production-registry",
        type=Path,
        help=(
            "Optional active-registry destination. With production activation "
            "authorization, a passing checkpoint is registered here; otherwise "
            "the external-SSD-first machine default is used."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.production_registry is not None and not (
        args.authorize_production_activation
    ):
        raise ValueError(
            "--production-registry requires --authorize-production-activation"
        )
    checkpoint_directory = args.checkpoint_dir.expanduser().resolve()
    if checkpoint_directory.exists() or checkpoint_directory.is_symlink():
        raise FileExistsError(
            "refusing to overwrite production checkpoint directory: "
            f"{checkpoint_directory}"
        )
    args.checkpoint_dir = checkpoint_directory
    corpus = load_reference_training_corpus(
        args.input,
        feature_snapshot_by_reference=_snapshot_map(args.feature_snapshot),
        song_id_by_reference=_song_map(args.song_id),
        negative_samples_per_hold=args.negative_samples_per_hold,
        boundary_guard_seconds=args.boundary_guard_seconds,
        max_sources=args.max_sources,
    )
    if args.adapted_manifest_path is not None:
        corpus.write_manifest(args.adapted_manifest_path)
    if not args.dynamic_crop_artifact and not args.dynamic_crop_manifest:
        raise ValueError(
            "production Dynamic Crop v2 training requires artifacts or manifests"
        )
    dynamic_crop_corpus = build_stage_edit_dynamic_crop_training_corpus_v2(
        corpus.examples,
        artifact_paths_by_reference=_dynamic_artifact_map(
            args.dynamic_crop_artifact,
            args.dynamic_crop_manifest,
        ),
    )
    config = production_stage_edit_config()
    device = None
    if args.device != "auto":
        if mx is None:
            raise RuntimeError("MLX is required to select a training device")
        device = mx.cpu if args.device == "cpu" else mx.gpu
    result = train_production_stage_edit(
        corpus,
        checkpoint_directory=args.checkpoint_dir,
        held_out_song_ids=args.held_out_song,
        required_training_song_ids=args.required_training_song,
        model_config=config,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        maximum_cut_positive_weight=args.maximum_cut_positive_weight,
        cut_pretrain_steps=args.cut_pretrain_steps,
        cut_pretrain_learning_rate=args.cut_pretrain_learning_rate,
        cut_pretrain_window_length=args.cut_pretrain_window_length,
        cut_pretrain_warmup_steps=args.cut_pretrain_warmup_steps,
        finetune_warmup_steps=args.finetune_warmup_steps,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        seed=args.seed,
        device=device,
        dynamic_crop_corpus=dynamic_crop_corpus,
        authorize_production_activation=args.authorize_production_activation,
    )
    registered_production_bundle = None
    if result.production_ready and args.authorize_production_activation:
        registered_production_bundle = register_stage_edit_production_bundle(
            result.checkpoint_directory,
            registry_path=args.production_registry,
        )
    print(
        json.dumps(
            {
                "checkpoint_directory": str(result.checkpoint_directory),
                "report_path": str(result.report_path),
                "best_epoch": result.best_epoch,
                "best_training_step": result.best_training_step,
                "production_ready": result.production_ready,
                "quality_gate_passed": result.quality_gate.passed,
                "registered_production_bundle": (
                    str(registered_production_bundle)
                    if registered_production_bundle is not None
                    else None
                ),
                "quality_gate": result.quality_gate.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return _training_result_exit_code(
        activation_authorized=args.authorize_production_activation,
        production_ready=result.production_ready,
        quality_gate_passed=result.quality_gate.passed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
