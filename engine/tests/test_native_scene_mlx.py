from unittest import TestCase
from unittest.mock import Mock
import numpy as np

from crosscut.native_scene_mlx import CalibratedMLXSceneModel, expected_weight_shapes


class NativeSceneMLXTests(TestCase):
    def setUp(self):
        self.frames = np.zeros((1, 100, 27, 48, 3), np.float32)
        self.single = np.full((1, 100, 1), 2.0, np.float32)
        self.many = np.full((1, 100, 1), -3.0, np.float32)
        self.reference = Mock(return_value=(self.single, {"many_hot": self.many}))
        self.port = Mock(return_value=(self.single + 1e-6, self.many - 1e-6))

    def model(self):
        return CalibratedMLXSceneModel(self.reference, port=self.port)

    def test_first_window_uses_reference_then_calibrated_port(self):
        model = self.model()
        first, _ = model(self.frames)
        np.testing.assert_array_equal(first, self.single)
        second, heads = model(self.frames)
        np.testing.assert_array_equal(second, self.port.return_value[0])
        np.testing.assert_array_equal(heads['many_hot'], self.port.return_value[1])
        self.assertEqual(model.reference_windows, 1)
        self.assertEqual(model.accelerated_windows, 1)
        self.assertTrue(model.calibrated)
        self.assertEqual(self.port.call_args.args[0].dtype, np.uint8)

    def test_threshold_uncertainty_in_either_head_uses_reference(self):
        for head in (0, 1):
            with self.subTest(head=head):
                model = self.model(); model(self.frames)
                pair = [self.single.copy(), self.many.copy()]
                pair[head][0, 60, 0] = 0.0001
                self.port.return_value = tuple(pair)
                output, _ = model(self.frames)
                np.testing.assert_array_equal(output, self.single)
                self.assertEqual(model.reference_windows, 2)
                self.assertEqual(model.accelerated_windows, 0)
                self.port.return_value = (self.single + 1e-6, self.many - 1e-6)

    def test_bad_calibration_disables_port_for_following_windows(self):
        self.port.return_value = (self.single + .01, self.many)
        model = self.model()
        model(self.frames); model(self.frames)
        self.assertEqual(model.disabled_reason, 'reference_calibration_mismatch')
        self.assertEqual(self.port.call_count, 1)
        self.assertEqual(model.reference_windows, 2)

    def test_runtime_failure_and_nonfinite_output_preserve_reference(self):
        for outcome in ('error', 'nan', 'shape'):
            with self.subTest(outcome=outcome):
                self.port.reset_mock(side_effect=True)
                if outcome == 'error': self.port.side_effect = RuntimeError('GPU unavailable')
                elif outcome == 'nan': self.port.return_value = (self.single * np.nan, self.many)
                else: self.port.return_value = (self.single[:, :10], self.many)
                model = self.model(); model(self.frames); model(self.frames)
                self.assertIsNotNone(model.disabled_reason)
                self.assertEqual(model.reference_windows, 2)
                self.assertEqual(self.port.call_count, 1)

    def test_cancellation_is_not_a_backend_fallback(self):
        self.port.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.model()(self.frames)
        self.reference.assert_not_called()

    def test_noninteger_or_wrong_size_input_cannot_be_silently_quantized(self):
        for frames in (self.frames + .25, self.frames - 1, self.frames + 256,
                       self.frames + np.nan, self.frames[:, :99]):
            with self.assertRaises(ValueError): self.model()(frames)
        self.port.assert_not_called()
        self.reference.assert_not_called()

    def test_pinned_architecture_has_complete_shape_contract(self):
        shapes = expected_weight_shapes()
        self.assertEqual(len(shapes), 84)
        self.assertEqual(shapes['SDDCNN_3/DDCNN_1/Conv3D_8/conv_temporal/kernel'], (3, 1, 1, 128, 64))
        self.assertEqual(shapes['dense/kernel'], (4864, 1024))
