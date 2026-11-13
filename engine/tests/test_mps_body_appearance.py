from pathlib import Path
import unittest
from unittest.mock import patch
from crosscut.body_appearance import BodyAppearanceEncoder
from crosscut.mps_body_appearance import MPSBodyAppearanceEncoder


class MPSBodyAppearanceTests(unittest.TestCase):
    def test_gpu_evidence_does_not_claim_cpu_namespace_equivalence(self):
        encoder=MPSBodyAppearanceEncoder(Path('not-loaded'))
        self.assertEqual(encoder.fingerprint()['model_sha256'],BodyAppearanceEncoder(Path('not-loaded')).fingerprint()['model_sha256'])
        self.assertFalse(encoder.fingerprint()['cpu_cache_interchangeable'])
        self.assertEqual(encoder.fingerprint()['execution_device'],'mps')
        self.assertIsNone(encoder._model)

    def test_explicit_gpu_cannot_silently_fall_back_to_cpu(self):
        encoder=MPSBodyAppearanceEncoder(Path('not-loaded'))
        with patch('torch.backends.mps.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'available Apple GPU'):encoder._load()
        self.assertIsNone(encoder._model)


if __name__=='__main__':unittest.main()
