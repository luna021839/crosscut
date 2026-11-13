from copy import deepcopy
import hashlib
from pathlib import Path
import unittest

from crosscut.joint_boundary_provider import DecodedBoundaryOptionProvider
from crosscut.joint_complete_containment_compatibility import (
    RECOVERY_PROVIDER, RECOVERY_CONTAINMENT, RECOVERY_IMPLEMENTATION,
    PREVIOUS_CONTAINMENT, previous_complete_containment_analyzer,
)
from crosscut.joint_scalar_reader_compatibility import MEASURED_FACE_ANALYZER, NATIVE_TRACKING_READER_PROVIDER


class CompleteContainmentCompatibilityTests(unittest.TestCase):
    def analyzer(self):
        return dict(analysis_contract='joint-bridge-disposition-scalar-shard-v3',
            visual_analyzer_sha256s=[MEASURED_FACE_ANALYZER]*2,source_content_sha256s=['a'*64,'b'*64],
            provider_configuration_sha256='c'*64,segment_index=115,
            implementation_sha256=dict(provider_class_ast=RECOVERY_PROVIDER,
                cut_anchored_containment_sha256=RECOVERY_CONTAINMENT,
                observed_face_containment_sha256=RECOVERY_IMPLEMENTATION,
                other_unchanged='d'*64))

    def test_exact_old_complete_lookup_preserves_every_other_field(self):
        current=self.analyzer()
        original=deepcopy(current)
        previous=previous_complete_containment_analyzer(current)
        expected=deepcopy(current)
        expected['implementation_sha256']['provider_class_ast']=NATIVE_TRACKING_READER_PROVIDER
        expected['implementation_sha256']['cut_anchored_containment_sha256']=PREVIOUS_CONTAINMENT
        del expected['implementation_sha256']['observed_face_containment_sha256']
        self.assertEqual(previous,expected)
        self.assertEqual(current,original)
        self.assertIsNone(previous_complete_containment_analyzer(previous))

    def test_future_recovery_or_numeric_analyzer_never_migrates(self):
        for key in ('provider_class_ast','cut_anchored_containment_sha256','observed_face_containment_sha256'):
            current=self.analyzer()
            current['implementation_sha256'][key]='f'*64
            self.assertIsNone(previous_complete_containment_analyzer(current))
        current=self.analyzer()
        current['visual_analyzer_sha256s'][0]='f'*64
        self.assertIsNone(previous_complete_containment_analyzer(current))

    def test_incomplete_old_flow_disposition_cannot_be_imported(self):
        decode=DecodedBoundaryOptionProvider._decode_persisted_scalar_bridge_entry
        self.assertIsNone(decode([False,False,False,['segment_cut_containment_pixel_flow_failed'],0.]))
        self.assertEqual(decode([True,True,True,[],0.]),(True,True,True,(),0.))
        self.assertEqual(decode([True,False,False,['segment_source_scene_cut_breaks_track'],0.]),
                         (True,False,False,('segment_source_scene_cut_breaks_track',),0.))

    def test_exact_current_implementation_pins(self):
        from crosscut.visual_feature_cache import python_symbol_dependency_sha256
        root=Path(__file__).parents[1]/'crosscut'
        self.assertEqual(python_symbol_dependency_sha256(root/'joint_boundary_provider.py',
                          root_symbols=('DecodedBoundaryOptionProvider',)),RECOVERY_PROVIDER)
        self.assertEqual(hashlib.sha256((root/'cut_anchored_containment.py').read_bytes()).hexdigest(),RECOVERY_CONTAINMENT)
        self.assertEqual(hashlib.sha256((root/'observed_face_containment.py').read_bytes()).hexdigest(),RECOVERY_IMPLEMENTATION)


if __name__=='__main__':
    unittest.main()
