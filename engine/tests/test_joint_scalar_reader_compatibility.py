from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from crosscut.joint_scalar_reader_compatibility import (
    H264_READER_ANALYZER, PREVIOUS_READER_ANALYZER, previous_reader_scalar_analyzer,
    MEASURED_FACE_ANALYZER, NATIVE_TRACKING_READER_PROVIDER,
    TIMESTAMP_TRACKING_READER_PROVIDER,
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


class NativeTrackingReaderCompatibilityTests(unittest.TestCase):
    def analyzer(self):
        original = ScalarReaderCompatibilityTests().analyzer()
        original['visual_analyzer_sha256s'] = [MEASURED_FACE_ANALYZER] * 2
        original['implementation_sha256']['provider_class_ast'] = NATIVE_TRACKING_READER_PROVIDER
        return original

    # Exercise real source/config/checksum-bound shard I/O for this separate
    # equivalence as well; a fresh conflict must still beat the old scalar.
    test_production_adapter_reads_legacy_shard_but_fresh_conflict_wins = (
        ScalarReaderCompatibilityTests.test_production_adapter_reads_legacy_shard_but_fresh_conflict_wins)

    def test_changes_only_the_exact_tracking_reader_provider(self):
        original = self.analyzer()
        current = deepcopy(original)
        previous = previous_reader_scalar_analyzer(current)
        expected = deepcopy(original)
        expected['implementation_sha256']['provider_class_ast'] = TIMESTAMP_TRACKING_READER_PROVIDER
        self.assertEqual(previous, expected)
        self.assertEqual(current, original)
        self.assertIsNone(previous_reader_scalar_analyzer(previous))

    def test_unknown_provider_or_changed_visual_numbers_are_not_migrated(self):
        for field, value in (
            ('implementation_sha256', {'provider_class_ast': 'e'*64}),
            ('implementation_sha256', None),
            ('visual_analyzer_sha256s', [MEASURED_FACE_ANALYZER, H264_READER_ANALYZER]),
            ('visual_analyzer_sha256s', ['e'*64]*2),
        ):
            self.assertIsNone(previous_reader_scalar_analyzer({**self.analyzer(), field:value}))

    def test_reader_only_equivalence_refuses_new_recovery_policy(self):
        from crosscut.visual_feature_cache import python_symbol_dependency_sha256
        actual = python_symbol_dependency_sha256(
            Path(__file__).parents[1] / 'crosscut/joint_boundary_provider.py',
            root_symbols=('DecodedBoundaryOptionProvider',))
        self.assertNotEqual(actual, NATIVE_TRACKING_READER_PROVIDER)
        analyzer=self.analyzer()
        analyzer['implementation_sha256']['provider_class_ast']=actual
        self.assertIsNone(previous_reader_scalar_analyzer(analyzer))

    def test_other_context_changes_cannot_load_the_previous_shard(self):
        with tempfile.TemporaryDirectory(prefix='tracking-reader-cache-') as directory:
            source = Path(directory) / 'source.mp4'
            source.touch()
            arguments = dict(source_cache_directory=Path(directory)/'cache',
                             source_path=source, namespace='joint-bridge-disposition-v3')
            original = self.analyzer()
            previous = previous_reader_scalar_analyzer(original)
            self.assertTrue(PersistentSourceScalarCache(analyzer=previous, **arguments).put({'ok':True}))
            for field,value in (
                ('source_content_sha256s', ['f'*64,'b'*64]),
                ('provider_configuration_sha256', 'e'*64),
                ('segment_index', 2),
                ('appearance_model_asset', {'actual_sha256':'f'*64}),
                ('implementation_sha256', {**original['implementation_sha256'], 'face_vector_ast':'f'*64}),
            ):
                changed = previous_reader_scalar_analyzer({**original, field:value})
                self.assertIsNotNone(changed)
                self.assertIsNone(PersistentSourceScalarCache(analyzer=changed, **arguments).get())


if __name__ == '__main__':
    unittest.main()
