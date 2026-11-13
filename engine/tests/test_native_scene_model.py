from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

from crosscut.native_scene_model import prediction_intervals, predict_native_frames, detect_native_scene_changes_cached, detect_native_scene_changes, ensure_native_scene_model
from crosscut.vision import detect_scene_changes_in_video_cached, SceneChange


class NativeSceneModelTests(unittest.TestCase):
    def test_only_exact_source_and_reference_analyzer_can_reuse_pre_gpu_scalars(self):
        from crosscut.native_scene_model import _REFERENCE_ONLY_IMPLEMENTATION_SHA256
        from crosscut.source_scalar_cache import PersistentSourceScalarCache
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'; source.write_bytes(b'source-one')
            identity = {'contract':'test', 'implementation_sha256':'new',
                        'optional_executor_sha256':'gpu', 'threshold':.5}
            legacy = {key:value for key,value in identity.items() if key != 'optional_executor_sha256'}
            legacy['implementation_sha256'] = _REFERENCE_ONLY_IMPLEMENTATION_SHA256
            cache = PersistentSourceScalarCache(source_cache_directory=root,
                source_path=source, namespace='native-transnet-scene-changes-v1', analyzer=legacy)
            cache.put({'changes':[dict(frame_index=10,score=.8,frame_difference=.7,histogram_distance=.6)]})
            with patch('crosscut.native_scene_model.native_scene_analyzer_identity',return_value=identity), \
                 patch('crosscut.native_scene_model.ensure_native_scene_model',side_effect=RuntimeError('cache miss')):
                changes, report = detect_native_scene_changes_cached(source,source_cache_directory=root,derived_cache_directory=None)
                self.assertEqual(changes[0].frame_index,10)
                self.assertTrue(report['reference_executor_cache_reused'])
                self.assertEqual(report['detection_policy'],legacy)
                source.write_bytes(b'source-two')
                with self.assertRaisesRegex(RuntimeError,'cache miss'):
                    detect_native_scene_changes_cached(source,source_cache_directory=root,derived_cache_directory=None)
                source.write_bytes(b'source-one')
                identity['threshold'] = .6
                with self.assertRaisesRegex(RuntimeError,'cache miss'):
                    detect_native_scene_changes_cached(source,source_cache_directory=root,derived_cache_directory=None)

    def test_measured_native_cut_reaches_existing_joint_guard(self):
        from crosscut.joint_boundary_provider import DecodedBoundaryOptionProvider
        boundary=15.135858389858278
        source_start=28.50004038726173
        rate=.9999766050872786
        provider=SimpleNamespace(
            output_fps=29.97,scene_cut_times=((),(857/(30000/1001),)),
            _source_time=lambda source,t: source_start+(t-boundary)*rate if source else t,
        )
        arguments=dict(outgoing_source=0,incoming_source=1,
                       outgoing_candidate=SimpleNamespace(start_seconds=13.844648245359146),
                       incoming_candidate=SimpleNamespace(end_seconds=16.293601734223955),
                       boundary_master_seconds=boundary,outgoing_boundary_local=boundary,
                       incoming_boundary_local=source_start)
        self.assertEqual(DecodedBoundaryOptionProvider._scene_cut_failures(provider,**arguments),
                         ('incoming_source_local_camera_cut_near_boundary',))
        provider.scene_cut_times=((),())
        self.assertEqual(DecodedBoundaryOptionProvider._scene_cut_failures(provider,**arguments),())

    def test_model_transition_frame_becomes_first_incoming_frame(self):
        frames=[np.full((27,48,3),i*20,dtype=np.uint8) for i in range(8)]
        capture=Mock()
        capture.isOpened.return_value=True
        capture.read.side_effect=[(True,f) for f in frames]+[(False,None)]
        p=np.array([.9,0.,.6,.7,0.,.8,0.,.9])
        with patch('crosscut.native_scene_model.cv2.VideoCapture',return_value=capture), \
             patch('crosscut.native_scene_model._load_model'), \
             patch('crosscut.native_scene_model.predict_native_frames',return_value=(p,p)):
            changes,observations=detect_native_scene_changes(Path('source'),Path('model'))
        self.assertEqual([c.frame_index for c in changes],[4,6])
        self.assertEqual(observations['decoded_frame_count'],8)
        self.assertEqual(observations['transition_intervals'][0]['first_transition_frame'],2)
        capture.release.assert_called_once()

    def test_existing_model_mismatch_is_preserved_and_missing_storage_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            bad=root/'models/transnetv2-official/saved_model.pb'
            bad.parent.mkdir(parents=True)
            bad.write_bytes(b'not-the-model')
            with patch('crosscut.native_scene_model._download') as download:
                with self.assertRaisesRegex(ValueError,'preserved'): ensure_native_scene_model(root)
                download.assert_not_called()
                self.assertEqual(bad.read_bytes(),b'not-the-model')
                with self.assertRaises(OSError): ensure_native_scene_model(root/'missing-ssd')
                self.assertFalse((root/'missing-ssd').exists())

    def test_transition_intervals_preserve_exact_frames_and_last_positive(self):
        self.assertEqual(prediction_intervals([0,.6,.7,0,.5,.9,0]), ((1,2),(5,5)))
        self.assertEqual(prediction_intervals([.8,.9]), ((0,1),))
        self.assertEqual(prediction_intervals([]), ())
        for value in ([float('nan')], [-.1], [1.1], [[.5]]):
            with self.assertRaises(ValueError): prediction_intervals(value)

    def test_windowed_predictions_preserve_length_and_frame_identity(self):
        def model(values):
            p=values[:,:,0,0,:1]/255.
            return p,{'many_hot':p}
        tf=SimpleNamespace(float32=np.float32,cast=lambda a,t:a.astype(t),
                           sigmoid=lambda a:SimpleNamespace(numpy=lambda:a))
        with patch('crosscut.native_scene_model._tensorflow',return_value=tf):
            for length in (1,49,50,51,99,100,101,150):
                frames=np.repeat(np.arange(length,dtype=np.uint8)[:,None,None,None],27*48*3,axis=1).reshape(length,27,48,3)
                a,b=predict_native_frames(frames,model)
                np.testing.assert_array_equal(a,np.arange(length,dtype=np.float32)/255.)
                np.testing.assert_array_equal(a,b)

    def test_backend_dispatch_never_silently_skips_frames_or_falls_back(self):
        with patch('crosscut.native_scene_model.detect_native_scene_changes_cached',return_value=((),{})) as native:
            detect_scene_changes_in_video_cached(Path('source'),backend='transnet_v2',derived_cache_directory=Path('cache'))
            native.assert_called_once()
            with self.assertRaisesRegex(ValueError,'every native frame'):
                detect_scene_changes_in_video_cached(Path('source'),backend='transnet_v2',sample_every=2)
            with self.assertRaisesRegex(ValueError,'unknown'):
                detect_scene_changes_in_video_cached(Path('source'),backend='other')

    def test_source_bound_numeric_cache_reuses_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            payload={'changes':[dict(frame_index=857,score=.58,frame_difference=.79,histogram_distance=.18)]}
            with patch('crosscut.native_scene_model.native_scene_analyzer_identity',return_value={'contract':'test'}), \
                 patch('crosscut.source_scalar_cache.PersistentSourceScalarCache') as cache, \
                 patch('crosscut.native_scene_model.ensure_native_scene_model') as model:
                cache.return_value.get.return_value=payload
                cache.return_value.diagnostics.return_value={}
                result,report=detect_native_scene_changes_cached(root/'source',source_cache_directory=None,derived_cache_directory=root)
                self.assertEqual(result[0].frame_index,857)
                self.assertTrue(report['analysis_reused'])
                model.assert_not_called()
                self.assertEqual(cache.call_args.kwargs['namespace'],'native-transnet-scene-changes-v1')

    def test_cache_miss_publishes_measured_changes_and_storage_errors_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('crosscut.native_scene_model.native_scene_analyzer_identity',return_value={'contract':'test'}), \
                 patch('crosscut.source_scalar_cache.PersistentSourceScalarCache') as cache, \
                 patch('crosscut.native_scene_model.ensure_native_scene_model',return_value=root), \
                 patch('crosscut.native_scene_model.detect_native_scene_changes',return_value=((SceneChange(857,.58,.79,.18),),{})):
                cache.return_value.get.return_value=None
                cache.return_value.diagnostics.return_value={}
                result,report=detect_native_scene_changes_cached(root/'source',source_cache_directory=None,derived_cache_directory=root)
                self.assertFalse(report['analysis_reused'])
                self.assertEqual(cache.return_value.put.call_args.args[0]['changes'][0]['frame_index'],857)
                cache.side_effect=OSError('SSD unavailable')
                with self.assertRaisesRegex(OSError,'SSD unavailable'):
                    detect_native_scene_changes_cached(root/'source',source_cache_directory=None,derived_cache_directory=root)


if __name__=='__main__': unittest.main()
