from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from crosscut.source_first_execution import resolve_body_execution_device, body_observation_namespace
from crosscut.body_appearance import BodyAppearanceEncoder
from crosscut.mps_body_appearance import MPSBodyAppearanceEncoder


class SourceFirstExecutionTests(unittest.TestCase):
    def test_cpu_namespace_is_byte_identical_and_does_not_need_gpu_model(self):
        encoder = Mock()
        self.assertEqual(body_observation_namespace('a'*64, encoder, 'cpu'), 'a'*64)
        encoder.fingerprint.assert_not_called()

    def test_gpu_namespace_binds_adapter_and_does_not_reuse_cpu_evidence(self):
        encoder = Mock()
        encoder.fingerprint.return_value = {'model': 'fixed', 'device': 'mps', 'adapter': 'one'}
        first = body_observation_namespace('a'*64, encoder, 'mps')
        self.assertNotEqual(first, 'a'*64)
        self.assertEqual(first, body_observation_namespace('a'*64, encoder, 'mps'))
        encoder.fingerprint.return_value['adapter'] = 'two'
        self.assertNotEqual(first, body_observation_namespace('a'*64, encoder, 'mps'))

    def test_auto_resolves_before_observation_and_explicit_gpu_cannot_fallback(self):
        with patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(resolve_body_execution_device('auto'), 'mps')
            self.assertEqual(resolve_body_execution_device('cpu'), 'cpu')
        with patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(resolve_body_execution_device('auto'), 'cpu')
            with self.assertRaises(RuntimeError):
                resolve_body_execution_device('mps')
        with self.assertRaises(ValueError):
            resolve_body_execution_device('cuda')

    def test_explicit_startup_override_binds_auto_but_not_explicit_backend(self):
        with patch.dict('os.environ', {'IVE_CROSSCUT_BODY_EXECUTION_DEVICE': 'cpu'}), \
                patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(resolve_body_execution_device('auto'), 'cpu')
            self.assertEqual(resolve_body_execution_device('mps'), 'mps')
        with patch.dict('os.environ', {'IVE_CROSSCUT_BODY_EXECUTION_DEVICE': 'invalid'}):
            with self.assertRaises(ValueError):
                resolve_body_execution_device('auto')

    def test_failed_gpu_transfer_does_not_leave_cpu_model_as_a_successful_retry(self):
        encoder = MPSBodyAppearanceEncoder(Path('not-read'))
        cpu_model = Mock()
        cpu_model.to.side_effect = RuntimeError('GPU transfer failed')
        def base_load(instance):
            instance._model = cpu_model
            return cpu_model
        with patch('torch.backends.mps.is_available', return_value=True), \
                patch.object(BodyAppearanceEncoder, '_load', base_load):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, 'GPU transfer failed'):
                    encoder._load()
                self.assertIsNone(encoder._model)
        self.assertEqual(cpu_model.to.call_count, 2)


if __name__ == '__main__':
    unittest.main()
