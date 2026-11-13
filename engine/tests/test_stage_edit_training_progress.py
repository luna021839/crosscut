import io
import json
import unittest
from unittest.mock import patch

from crosscut.stage_edit_training_progress import (
    LOGGER, configure_training_progress, training_step_completed,
    training_epoch_evaluated,
)


class TrainingProgressTests(unittest.TestCase):
    def test_completed_step_cadence_and_no_quality_authority(self):
        with self.assertLogs(LOGGER, level='INFO') as captured:
            for step in range(1, 54):
                training_step_completed('cut_pretraining', step=step,
                                        optimizer_step=step, loss=1. / step,
                                        total_steps=53)
        values = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual([item['phase_step'] for item in values], [1, 25, 50, 53])
        self.assertTrue(all(item['device_evaluation_completed'] for item in values))
        self.assertFalse(any(item['final_quality_authority'] for item in values))

    def test_nonfinite_loss_does_not_break_reporting_or_claim_finite(self):
        with self.assertLogs(LOGGER, level='INFO') as captured:
            training_step_completed('joint_finetuning', step=1, optimizer_step=1001,
                                    loss=float('nan'), epoch=1)
        value = json.loads(captured.records[0].getMessage())
        self.assertIsNone(value['loss'])
        self.assertFalse(value['loss_is_finite'])

    def test_validation_not_presented_as_heldout_acceptance(self):
        with self.assertLogs(LOGGER, level='INFO') as captured:
            training_epoch_evaluated(epoch=1, optimizer_step=1008, training_loss=.2,
                                     validation_loss=.3, checkpoint_selection_eligible=False,
                                     validation_improved=False)
        value = json.loads(captured.records[0].getMessage())
        self.assertEqual(value['validation_scope'], 'training_song_reference_split_not_held_out_song')
        self.assertFalse(value['checkpoint_selection_eligible'])
        self.assertFalse(value['final_quality_authority'])

    def test_cli_handler_uses_stderr_and_is_idempotent(self):
        old_handlers, old_level, old_propagate = LOGGER.handlers[:], LOGGER.level, LOGGER.propagate
        output = io.StringIO()
        try:
            LOGGER.handlers = []
            with patch('sys.stderr', output):
                configure_training_progress()
                configure_training_progress()
                training_step_completed('source_pretraining', step=1, optimizer_step=501, loss=.5)
            self.assertEqual(len(output.getvalue().splitlines()), 1)
            self.assertEqual(json.loads(output.getvalue())['optimizer_step'], 501)
        finally:
            LOGGER.handlers, LOGGER.level, LOGGER.propagate = old_handlers, old_level, old_propagate

    def test_real_v3_smoke_reports_completed_optimizer_and_validation(self):
        from engine.tests.test_stage_edit_production_training import (
            StageEditProductionTrainingSmokeTests, mlx_runtime_version,
        )
        if mlx_runtime_version() is None:
            self.skipTest('MLX unavailable')
        # Existing tiny CPU fixture verifies save/reload and optimizer histories.
        # It is instrumentation testing, never the actual production model fit.
        case = StageEditProductionTrainingSmokeTests(
            'test_v3_trains_saves_reloads_and_reports_distinct_contracts')
        with self.assertLogs(LOGGER, level='INFO') as captured:
            case.test_v3_trains_saves_reloads_and_reports_distinct_contracts()
        events = [json.loads(record.getMessage()) for record in captured.records]
        prior = [item for item in events if item.get('phase') == 'dynamic_crop_v3_prior']
        self.assertEqual(prior[0]['phase_step'], 1)
        self.assertEqual(prior[-1]['phase_step'], 500)
        self.assertTrue(any(item.get('phase') == 'joint_finetuning' for item in events))
        self.assertTrue(any(item['event'] == 'training_reference_validation_completed' for item in events))


if __name__ == '__main__':
    unittest.main()
