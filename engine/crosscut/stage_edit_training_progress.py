"""Scalar-only progress emitted after completed training work, never a gate."""
from __future__ import annotations

import json
import logging
import math
import sys
from time import monotonic

LOGGER = logging.getLogger('crosscut.stage_edit.training_progress')


def configure_training_progress() -> None:
    """CLI-owned stderr handler; library callers can install their own handler."""
    if not any(getattr(handler, '_stage_edit_progress', False) for handler in LOGGER.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._stage_edit_progress = True
        handler.setFormatter(logging.Formatter('%(message)s'))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def training_step_completed(phase: str, *, step: int, optimizer_step: int,
                            loss: float, total_steps: int | None = None,
                            epoch: int | None = None) -> None:
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    if step != 1 and step % 25 != 0 and step != total_steps:
        return
    numeric_loss = float(loss)
    LOGGER.info(json.dumps(dict(
        type='stage_edit_training_progress',
        event='optimizer_step_completed', phase=phase,
        phase_step=step, optimizer_step=optimizer_step,
        configured_phase_steps=total_steps, epoch=epoch,
        loss=numeric_loss if math.isfinite(numeric_loss) else None,
        loss_is_finite=math.isfinite(numeric_loss),
        monotonic_seconds=monotonic(),
        device_evaluation_completed=True, final_quality_authority=False,
    ), allow_nan=False))


def training_epoch_evaluated(*, epoch: int, optimizer_step: int,
                             training_loss: float, validation_loss: float | None,
                             checkpoint_selection_eligible: bool,
                             validation_improved: bool) -> None:
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    def finite(value):
        return float(value) if value is not None and math.isfinite(float(value)) else None
    LOGGER.info(json.dumps(dict(
        type='stage_edit_training_progress', event='training_reference_validation_completed',
        epoch=epoch, optimizer_step=optimizer_step,
        training_loss=finite(training_loss), validation_loss=finite(validation_loss),
        checkpoint_selection_eligible=checkpoint_selection_eligible,
        validation_improved=validation_improved,
        validation_scope='training_song_reference_split_not_held_out_song',
        monotonic_seconds=monotonic(), final_quality_authority=False,
    ), allow_nan=False))
