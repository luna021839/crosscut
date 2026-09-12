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
from .stage_edit_dynamic_crop_head_v2 import DynamicCropV2LossContract
from .stage_edit_dynamic_crop_head_v3 import DynamicCropV3LossContract
from .stage_edit_dynamic_crop_training_v3 import (
    build_stage_edit_dynamic_crop_training_corpus_v3,
)
from .reference_dynamic_crop_job import (
    validate_reference_dynamic_crop_manifest,
)
from .reference_dynamic_crop_supervision import (
    load_reference_dynamic_crop_supervision,
)
from .professional_montage_policy import (
    ProfessionalMontagePolicyError,
    load_professional_montage_policy,
)
from .stage_edit_feature_materializer import StageEditFeatureSnapshot
from .stage_edit_production_training import train_production_stage_edit
from .stage_edit_training_progress import configure_training_progress
from .stage_edit_production_registry import (
    STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_BEAM_QA_MANIFEST_CONTRACT,
    _validated_contextual_source_pair_identity,
    register_stage_edit_production_bundle,
)
from .stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
    STAGE_EDIT_CONTINUITY_AUTHORITIES,
)
from .stage_edit_reference_adapter import load_reference_training_corpus
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    StageEditLossContract,
    StageEditTransformerConfig,
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


def _snapshot_map(
    values: Sequence[str],
    *,
    edit_mode: object = "group",
    target_member: object | None = None,
) -> dict[str, StageEditFeatureSnapshot]:
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
        snapshot = StageEditFeatureSnapshot.from_dict(
            payload,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
        )
        snapshot.require_production_causal_sensing(
            edit_mode=edit_mode,
            target_member=target_member,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
        )
        output[reference] = snapshot
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


def _json_object_argument(raw: str, name: str) -> dict[str, object]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _contextual_activation_evidence(
    bundle_directory: Path,
    beam_qa_manifest_path: Path,
) -> tuple[object, ...]:
    """Replay registry-format Beam-QA evidence before any expensive training."""

    for path, option, is_directory in (
        (bundle_directory, "--contextual-source-pair-bundle", True),
        (beam_qa_manifest_path, "--contextual-source-pair-beam-qa-manifest", False),
    ):
        if not path.is_absolute():
            raise ValueError(f"{option} must be an explicit absolute path")
        if path.is_symlink() or not (path.is_dir() if is_directory else path.is_file()):
            raise ValueError(f"{option} is missing or unsafe")

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate Beam-QA manifest key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite Beam-QA manifest value: {value}")

    payload = json.loads(
        beam_qa_manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject_nonfinite,
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != {"contract", "cases"}
        or payload.get("contract")
        != STAGE_EDIT_PRODUCTION_REGISTRY_CONTEXTUAL_BEAM_QA_MANIFEST_CONTRACT
        or not isinstance(payload.get("cases"), list)
        or not payload["cases"]
    ):
        raise ValueError(
            "--contextual-source-pair-beam-qa-manifest requires a nonempty "
            "registry-format Beam-QA manifest with contract and cases"
        )
    for entry in payload["cases"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"case", "derivation"}
            or not isinstance(entry["case"], dict)
            or not isinstance(entry["derivation"], dict)
        ):
            raise ValueError("Beam-QA manifest entries require case and derivation objects")

    from .stage_edit_contextual_beam_qa_adapter_v2 import (
        rehydrate_stage_edit_contextual_beam_qa_case_v2,
    )

    try:
        cases = tuple(
            rehydrate_stage_edit_contextual_beam_qa_case_v2(
                case_payload=entry["case"],
                derivation_payload=entry["derivation"],
            )
            for entry in payload["cases"]
        )
        # Use the same artifact, model-readiness, and adapter-seal authority as
        # registration. A JSON DTO alone must never authorize activation.
        _validated_contextual_source_pair_identity(bundle_directory, cases)
    except Exception as exc:
        raise ValueError(f"contextual-v2 activation evidence failed preflight: {exc}") from exc
    return cases


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
    parser.add_argument("--edit-mode", choices=("group", "member"), default="group")
    parser.add_argument("--target-member")
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
    parser.add_argument(
        "--dynamic-crop-version",
        type=int,
        choices=(2, 3),
        default=2,
        help=(
            "Dynamic Crop checkpoint/corpus contract to train. Version 2 is "
            "kept as the compatibility default; production v3 launchers must "
            "request --dynamic-crop-version 3 explicitly."
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
    parser.add_argument(
        "--source-pretrain-steps",
        type=int,
        default=500,
        help=(
            "Song/reference-balanced source-choice pretraining steps. Use 0 "
            "only for an explicit diagnostic ablation; enabled runs require "
            "at least 500."
        ),
    )
    parser.add_argument(
        "--source-pretrain-learning-rate", type=float, default=3.0e-4
    )
    parser.add_argument("--source-pretrain-window-length", type=int, default=32)
    parser.add_argument("--source-pretrain-warmup-steps", type=int, default=50)
    parser.add_argument(
        "--dynamic-crop-v3-prior-pretrain-steps",
        type=int,
        default=0,
        help=(
            "Training-only song/reference-balanced normalized-coordinate "
            "prior steps. Production v3 schedules enable 500; v2 rejects "
            "any nonzero value."
        ),
    )
    parser.add_argument(
        "--dynamic-crop-v3-prior-pretrain-learning-rate",
        type=float,
        default=3.0e-4,
    )
    parser.add_argument(
        "--dynamic-crop-v3-prior-pretrain-warmup-steps",
        type=int,
        default=50,
    )
    parser.add_argument("--finetune-warmup-steps", type=int, default=8)
    parser.add_argument(
        "--minimum-finetune-optimizer-steps", type=int, default=120
    )
    parser.add_argument(
        "--minimum-checkpoint-selection-finetune-steps",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--minimum-learning-rate-ratio", type=float, default=0.10
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--negative-samples-per-hold", type=int, default=2)
    parser.add_argument("--boundary-guard-seconds", type=float, default=0.18)
    parser.add_argument("--max-sources", type=int, default=32)
    parser.add_argument(
        "--model-config-json",
        help=(
            "Canonical StageEditTransformerConfig JSON. Release launchers pass "
            "this explicitly so model capacity cannot drift to a CLI default."
        ),
    )
    parser.add_argument(
        "--base-loss-contract-json",
        help=(
            "Canonical StageEditLossContract JSON, including the exact source "
            "objective. Must be paired with the other explicit contracts."
        ),
    )
    parser.add_argument(
        "--dynamic-crop-loss-contract-json",
        help=(
            "Canonical DynamicCropV2LossContract or DynamicCropV3LossContract "
            "JSON matching --dynamic-crop-version. Must be paired with the "
            "other explicit release contracts."
        ),
    )
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
    parser.add_argument(
        "--professional-montage-grammar-profile",
        type=Path,
        help=(
            "Explicit absolute non-symlink IVE professional montage grammar "
            "profile. Required for production activation and hash-bound into "
            "the active registry."
        ),
    )
    parser.add_argument(
        "--continuity-authority",
        choices=tuple(sorted(STAGE_EDIT_CONTINUITY_AUTHORITIES)),
        help=(
            "Explicit Dynamic Crop v3 boundary-continuity authority to bind "
            "into the active production registry. Required only for authorized "
            "v3 activation; v2 registries preserve their legacy schema."
        ),
    )
    parser.add_argument(
        "--source-pair-scorer",
        type=Path,
        help=(
            "Exact production source-pair checkpoint. Allowed only with v3, "
            "authorized activation, and learned continuity authority."
        ),
    )
    parser.add_argument(
        "--source-pair-production-report",
        type=Path,
        help=(
            "Exact all-fold source-pair production report paired with "
            "--source-pair-scorer."
        ),
    )
    parser.add_argument(
        "--contextual-source-pair-bundle",
        type=Path,
        help=(
            "Exact contextual-v2 production bundle directory. Required with "
            "contextual-v2 continuity authority and verified Beam-QA evidence."
        ),
    )
    parser.add_argument(
        "--contextual-source-pair-beam-qa-manifest",
        type=Path,
        help=(
            "Registry-format JSON with contract and case/derivation records. "
            "Its bound artifacts are replayed before training, then the "
            "verified envelopes are passed to contextual-v2 registration."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_training_progress()
    explicit_contract_values = (
        args.model_config_json,
        args.base_loss_contract_json,
        args.dynamic_crop_loss_contract_json,
    )
    if any(value is not None for value in explicit_contract_values) and not all(
        value is not None for value in explicit_contract_values
    ):
        raise ValueError(
            "explicit model, base-loss, and dynamic-crop-loss contracts must "
            "be supplied together"
        )
    if all(value is not None for value in explicit_contract_values):
        model_config = StageEditTransformerConfig.from_dict(
            _json_object_argument(args.model_config_json, "--model-config-json")
        )
        base_loss_contract = StageEditLossContract(
            **_json_object_argument(
                args.base_loss_contract_json, "--base-loss-contract-json"
            )
        )
        dynamic_payload = _json_object_argument(
            args.dynamic_crop_loss_contract_json,
            "--dynamic-crop-loss-contract-json",
        )
        dynamic_base_payload = dynamic_payload.get("base")
        if not isinstance(dynamic_base_payload, dict):
            raise ValueError(
                "--dynamic-crop-loss-contract-json base must be an object"
            )
        dynamic_loss_type = (
            DynamicCropV3LossContract
            if args.dynamic_crop_version == 3
            else DynamicCropV2LossContract
        )
        dynamic_crop_loss_contract = dynamic_loss_type(
            **{
                **dynamic_payload,
                "base": StageEditLossContract(**dynamic_base_payload),
            }
        )
        if dynamic_crop_loss_contract.base != base_loss_contract:
            raise ValueError(
                "dynamic crop and base loss/source objective contracts differ"
            )
        if model_config.transition_authority != base_loss_contract.transition_authority:
            raise ValueError(
                "model and loss transition authority contracts differ"
            )
    else:
        model_config = production_stage_edit_config()
        base_loss_contract = StageEditLossContract()
        dynamic_crop_loss_contract = (
            DynamicCropV3LossContract()
            if args.dynamic_crop_version == 3
            else DynamicCropV2LossContract()
        )
    if args.production_registry is not None and not (
        args.authorize_production_activation
    ):
        raise ValueError(
            "--production-registry requires --authorize-production-activation"
        )
    continuity_inputs = (
        args.continuity_authority,
        args.source_pair_scorer,
        args.source_pair_production_report,
        args.contextual_source_pair_bundle,
        args.contextual_source_pair_beam_qa_manifest,
    )
    if any(value is not None for value in continuity_inputs) and not (
        args.authorize_production_activation
    ):
        raise ValueError(
            "continuity-authority and source-pair registry evidence require "
            "--authorize-production-activation"
        )
    if args.dynamic_crop_version == 2 and any(
        value is not None for value in continuity_inputs
    ):
        raise ValueError(
            "Dynamic Crop v2 activation cannot bind v3 continuity authority "
            "or source-pair evidence"
        )
    if (
        args.authorize_production_activation
        and args.dynamic_crop_version == 3
        and args.continuity_authority is None
    ):
        raise ValueError(
            "authorized Dynamic Crop v3 activation requires "
            "--continuity-authority"
        )
    contextual_inputs = (
        args.contextual_source_pair_bundle,
        args.contextual_source_pair_beam_qa_manifest,
    )
    if args.continuity_authority == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2:
        if args.source_pair_scorer is not None or args.source_pair_production_report is not None:
            raise ValueError("contextual-v2 continuity rejects learned-v1 source-pair artifacts")
        if any(value is None for value in contextual_inputs):
            raise ValueError(
                "contextual-v2 continuity requires both --contextual-source-pair-bundle "
                "and --contextual-source-pair-beam-qa-manifest before training"
            )
    elif any(value is not None for value in contextual_inputs):
        raise ValueError("contextual-v2 artifacts require contextual-v2 continuity authority")
    if args.continuity_authority == HANDCRAFTED_CONTINUITY_AUTHORITY and (
        args.source_pair_scorer is not None
        or args.source_pair_production_report is not None
    ):
        raise ValueError(
            "handcrafted continuity authority rejects source-pair artifacts"
        )
    if args.continuity_authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY and (
        args.source_pair_scorer is None
        or args.source_pair_production_report is None
    ):
        raise ValueError(
            "learned continuity authority requires both --source-pair-scorer "
            "and --source-pair-production-report"
        )
    if args.continuity_authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
        for attribute, option in (
            ("source_pair_scorer", "--source-pair-scorer"),
            (
                "source_pair_production_report",
                "--source-pair-production-report",
            ),
        ):
            raw_path = Path(getattr(args, attribute))
            if not raw_path.is_absolute():
                raise ValueError(f"{option} must be an explicit absolute path")
            if raw_path.is_symlink() or not raw_path.is_file():
                raise ValueError(f"{option} is missing or unsafe")
            setattr(args, attribute, raw_path)
    if args.authorize_production_activation and (
        args.professional_montage_grammar_profile is None
    ):
        raise ValueError(
            "--authorize-production-activation requires "
            "--professional-montage-grammar-profile"
        )
    if args.professional_montage_grammar_profile is not None and not (
        args.authorize_production_activation
    ):
        raise ValueError(
            "--professional-montage-grammar-profile requires "
            "--authorize-production-activation"
        )
    if args.authorize_production_activation:
        montage_profile_path = Path(args.professional_montage_grammar_profile)
        if not montage_profile_path.is_absolute():
            raise ValueError(
                "--professional-montage-grammar-profile must be an explicit "
                "absolute path"
            )
        try:
            load_professional_montage_policy(montage_profile_path)
        except ProfessionalMontagePolicyError as exc:
            raise ValueError(
                "--professional-montage-grammar-profile is unsafe or invalid: "
                f"{exc}"
            ) from exc
        args.professional_montage_grammar_profile = montage_profile_path
    verified_contextual_cases = None
    if args.continuity_authority == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2:
        verified_contextual_cases = _contextual_activation_evidence(
            args.contextual_source_pair_bundle,
            args.contextual_source_pair_beam_qa_manifest,
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
        feature_snapshot_by_reference=_snapshot_map(
            args.feature_snapshot,
            edit_mode=args.edit_mode,
            target_member=args.target_member,
        ),
        song_id_by_reference=_song_map(args.song_id),
        negative_samples_per_hold=args.negative_samples_per_hold,
        boundary_guard_seconds=args.boundary_guard_seconds,
        max_sources=args.max_sources,
        production_edit_mode=args.edit_mode,
        production_target_member=args.target_member,
    )
    if args.adapted_manifest_path is not None:
        corpus.write_manifest(args.adapted_manifest_path)
    if not args.dynamic_crop_artifact and not args.dynamic_crop_manifest:
        raise ValueError(
            "production Dynamic Crop training requires artifacts or manifests"
        )
    dynamic_crop_v2_corpus = build_stage_edit_dynamic_crop_training_corpus_v2(
        corpus.examples,
        artifact_paths_by_reference=_dynamic_artifact_map(
            args.dynamic_crop_artifact,
            args.dynamic_crop_manifest,
        ),
    )
    dynamic_crop_corpus = (
        build_stage_edit_dynamic_crop_training_corpus_v3(dynamic_crop_v2_corpus)
        if args.dynamic_crop_version == 3
        else dynamic_crop_v2_corpus
    )
    config = model_config
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
        source_pretrain_steps=args.source_pretrain_steps,
        source_pretrain_learning_rate=args.source_pretrain_learning_rate,
        source_pretrain_window_length=args.source_pretrain_window_length,
        source_pretrain_warmup_steps=args.source_pretrain_warmup_steps,
        dynamic_crop_v3_prior_pretrain_steps=(
            args.dynamic_crop_v3_prior_pretrain_steps
        ),
        dynamic_crop_v3_prior_pretrain_learning_rate=(
            args.dynamic_crop_v3_prior_pretrain_learning_rate
        ),
        dynamic_crop_v3_prior_pretrain_warmup_steps=(
            args.dynamic_crop_v3_prior_pretrain_warmup_steps
        ),
        finetune_warmup_steps=args.finetune_warmup_steps,
        minimum_finetune_optimizer_steps=args.minimum_finetune_optimizer_steps,
        minimum_checkpoint_selection_finetune_steps=(
            args.minimum_checkpoint_selection_finetune_steps
        ),
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        seed=args.seed,
        device=device,
        loss_contract=base_loss_contract,
        dynamic_crop_corpus=dynamic_crop_corpus,
        **(
            {"dynamic_crop_v3_loss_contract": dynamic_crop_loss_contract}
            if args.dynamic_crop_version == 3
            else {"dynamic_crop_loss_contract": dynamic_crop_loss_contract}
        ),
        authorize_production_activation=args.authorize_production_activation,
    )
    registered_production_bundle = None
    if result.production_ready and args.authorize_production_activation:
        registered_production_bundle = register_stage_edit_production_bundle(
            result.checkpoint_directory,
            professional_montage_grammar_profile_path=(
                args.professional_montage_grammar_profile
            ),
            registry_path=args.production_registry,
            dynamic_crop_version=args.dynamic_crop_version,
            continuity_authority=args.continuity_authority,
            source_pair_scorer_path=args.source_pair_scorer,
            source_pair_production_report_path=(
                args.source_pair_production_report
            ),
            **(
                {
                    "contextual_source_pair_bundle_directory": args.contextual_source_pair_bundle,
                    "contextual_source_pair_verified_beam_qa_cases": verified_contextual_cases,
                }
                if verified_contextual_cases is not None
                else {}
            ),
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
