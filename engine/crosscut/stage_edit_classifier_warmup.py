"""Classifier warmup that preserves the trajectory feature representation.

Single-task cut/source warmup must not overwrite features needed by the crop
task before that task receives any updates. Train the real classifiers first;
restore every model parameter to joint trainability before trajectory fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


CLASSIFIER_WARMUP_CONTRACT = 'classifier_heads_only_preserved_features_then_full_joint_v1'
PHASE_MODULES = {
    'cut': ('cut_linear_baseline', 'cut_timing_head'),
    'source': ('source_query', 'source_key', 'source_bias'),
}


def _leaves(model: Any, *, trainable: bool = False) -> dict[str, Any]:
    from mlx.utils import tree_flatten
    return dict(tree_flatten(model.trainable_parameters() if trainable else model.parameters()))


def _digest(leaves: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(leaves.items()):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ClassifierWarmupState:
    phase: str
    all_keys: frozenset[str]
    classifier_keys: frozenset[str]
    frozen_sha256: str
    classifier_sha256: str
    trainable_parameter_count: int


def begin_classifier_warmup(model: Any, phase: str) -> ClassifierWarmupState:
    """Limit a fresh, completely trainable model to the declared classifiers."""
    if phase not in PHASE_MODULES:
        raise ValueError('Classifier warmup phase must be cut or source')
    before = _leaves(model)
    if not before or set(_leaves(model, trainable=True)) != set(before):
        raise ValueError('Classifier warmup requires a fully trainable owned model')
    base = getattr(model, 'base', model)
    prefix = 'base.' if base is not model else ''
    modules = PHASE_MODULES[phase]
    selected = frozenset(name for name in before if any(name.startswith(prefix + module + '.') for module in modules))
    if not selected:
        raise ValueError('Declared classifier parameters are absent')
    state = ClassifierWarmupState(phase, frozenset(before), selected,
        _digest({k: v for k, v in before.items() if k not in selected}),
        _digest({k: before[k] for k in selected}), sum(int(before[k].size) for k in selected))
    model.freeze()
    for module in modules:
        getattr(base, module).unfreeze()
    if set(_leaves(model, trainable=True)) != selected:
        model.unfreeze()
        raise RuntimeError('Classifier warmup trainable parameter scope differs')
    return state


def finish_classifier_warmup(model: Any, state: ClassifierWarmupState) -> dict[str, Any]:
    """Verify the real optimizer's frozen-weight invariance, then restore all."""
    try:
        current = _leaves(model)
        frozen_sha = _digest({k: v for k, v in current.items() if k not in state.classifier_keys})
        classifier_sha = _digest({k: current[k] for k in state.classifier_keys})
        unchanged = frozen_sha == state.frozen_sha256 and set(current) == state.all_keys
        if not unchanged:
            raise RuntimeError('Classifier warmup changed frozen feature or auxiliary parameters')
    finally:
        model.unfreeze()
    restored = set(_leaves(model, trainable=True)) == state.all_keys
    if not restored:
        raise RuntimeError('Classifier warmup did not restore full joint trainability')
    return dict(contract=CLASSIFIER_WARMUP_CONTRACT, phase=state.phase,
        trainable_modules=list(PHASE_MODULES[state.phase]),
        classifier_trainable_parameter_count=state.trainable_parameter_count,
        frozen_parameter_sha256_before=state.frozen_sha256,
        frozen_parameter_sha256_after=frozen_sha,
        all_nonclassifier_parameters_byte_identical=unchanged,
        classifier_parameters_changed=classifier_sha != state.classifier_sha256,
        full_joint_trainability_restored=restored,
        encoder_updated_during_classifier_warmup=False,
        joint_encoder_and_source_projection_training_required=True,
        validation_or_heldout_data_consumed=False)
