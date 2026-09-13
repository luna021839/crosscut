from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from crosscut.joint_scalar_reader_compatibility import (
    H264_READER_ANALYZER, PREVIOUS_READER_ANALYZER, previous_reader_scalar_analyzer,
)
from crosscut.source_scalar_cache import PersistentSourceScalarCache
from crosscut.joint_observation_reuse import JointObservationBank, ReusingDecodedBoundaryOptionProvider
from crosscut.visual_boundary_repair import BoundaryDescriptorCache


class ScalarReaderCompatibilityTests(unittest.TestCase):
    def analyzer(self):
        return dict(analysis_contract="joint-bridge-disposition-scalar-shard-v3",
                    visual_analyzer_sha256s=[H264_READER_ANALYZER]*2,
                    source_content_sha256s=["a"*64,"b"*64],
                    implementation_sha256={"provider_class_ast":"c"*64},
                    provider_configuration_sha256="d"*64,segment_index=1)

    def test_exact_forward_edge_preserves_other_fields_and_input(self):
        current=self.analyzer()
        original=deepcopy(current)
        previous=previous_reader_scalar_analyzer(current)
        self.assertEqual(current,original)
        expected=deepcopy(original)
        expected['visual_analyzer_sha256s']=[PREVIOUS_READER_ANALYZER]*2
        self.assertEqual(previous,expected)
        previous['implementation_sha256']['provider_class_ast']='changed'
        self.assertEqual(current,original)
        self.assertIsNone(previous_reader_scalar_analyzer(expected))

    def test_unknown_mixed_and_other_contracts_never_migrate(self):
        for overrides in (
            {'analysis_contract':'future-contract'},
            {'visual_analyzer_sha256s':['e'*64]*2},
            {'visual_analyzer_sha256s':[H264_READER_ANALYZER,PREVIOUS_READER_ANALYZER]},
            {'source_content_sha256s':['a'*64]},
            {'visual_analyzer_sha256s':[]},
        ):
            self.assertIsNone(previous_reader_scalar_analyzer({**self.analyzer(),**overrides}))

    def test_ordinary_loader_still_rejects_nonreader_changes(self):
        with tempfile.TemporaryDirectory(prefix='scalar-reader-parity-') as directory:
            source=Path(directory)/'source.mp4'
            source.touch()
            arguments=dict(source_cache_directory=Path(directory)/'cache',
                           source_path=source,namespace='joint-bridge-disposition-v3')
            original=self.analyzer()
            previous=previous_reader_scalar_analyzer(original)
            writer=PersistentSourceScalarCache(analyzer=previous,**arguments)
            self.assertTrue(writer.put({'entries':{'a':[True,True,True,[],0.]}}))
            self.assertIsNotNone(PersistentSourceScalarCache(analyzer=previous_reader_scalar_analyzer(original),**arguments).get())
            for field,value in (
                ('implementation_sha256',{'provider_class_ast':'e'*64}),
                ('source_content_sha256s',['f'*64,'b'*64]),
                ('provider_configuration_sha256','1'*64),('segment_index',2),
            ):
                changed=previous_reader_scalar_analyzer({**original,field:value})
                self.assertIsNone(PersistentSourceScalarCache(analyzer=changed,**arguments).get())

    def test_production_adapter_reads_legacy_shard_but_fresh_conflict_wins(self):
        with tempfile.TemporaryDirectory(prefix='scalar-reader-adapter-') as directory:
            root=Path(directory)
            sources=(root/'a.mp4',root/'b.mp4')
            for source in sources:
                source.touch()
            derived=root/'derived'
            derived.mkdir()
            with ReusingDecodedBoundaryOptionProvider(
                observation_bank=JointObservationBank(), sources=sources,
                audio_offsets=(0.,0.),scene_cut_times=((),()),source_timewarps=None,
                descriptor_cache=BoundaryDescriptorCache(),output_fps=30.,total_segment_count=3,
            ) as provider:
                # Supply an already built analyzer to isolate real shard I/O
                # from feature/model initialization, not to mock the loader.
                analyzer={**self.analyzer(),
                          'provider_configuration_sha256':provider._configuration_identity}
                analyzer.pop('segment_index')
                provider._scalar_bridge_persistent_analyzer_base=analyzer
                provider.require_persistent_feature_cache=True
                provider.feature_derived_cache_directory=derived
                old=PersistentSourceScalarCache(
                    source_cache_directory=derived,derived_cache_directory=derived,
                    source_path=sources[0],namespace='joint-bridge-disposition-v3',
                    analyzer=previous_reader_scalar_analyzer({**analyzer,'segment_index':1}))
                payload=dict(contract='joint-bridge-disposition-scalar-shard-v3',
                             provider_configuration_sha256=provider._configuration_identity,
                             segment_index=1,entries={'a'*64:[True,True,True,[],0.]})
                self.assertTrue(old.put(payload))
                provider._load_scalar_bridge_persistent_cache(1)
                self.assertEqual(provider._scalar_bridge_persistent_entries_by_segment[1]['a'*64],
                                 (True,True,True,(),0.))
                self.assertEqual(provider._legacy_scalar_entries_admitted,1)
                self.assertFalse(provider._materialization_by_proof)
                current=provider._scalar_bridge_persistent_caches[1]
                self.assertIsNone(current.get())  # Migration itself is RAM-only.
                fresh={**payload,'entries':{'a'*64:[True,False,False,['fresh_failure'],0.]}}
                self.assertTrue(current.put(fresh))
                provider._scalar_bridge_persistent_cache_loaded_segments.clear()
                provider._load_scalar_bridge_persistent_cache(1)
                self.assertEqual(provider._scalar_bridge_persistent_entries_by_segment[1]['a'*64],
                                 (True,False,False,('fresh_failure',),0.))
                self.assertEqual(provider._legacy_scalar_conflicts,1)


if __name__ == '__main__':
    unittest.main()
