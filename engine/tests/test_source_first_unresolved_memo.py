from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut.source_first_unresolved_memo import (
    SourceFirstUnresolvedMemo, install_source_first_unresolved_memo, _verified_empty_native_detection)


class UnresolvedMemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name)/'source.mp4'
        self.source.write_bytes(b'fixed source identity')
        self.sampler = SimpleNamespace(frame_rate=30., capture=SimpleNamespace(get=lambda _:3000.),
            frame_identity=lambda t:SimpleNamespace(frame_index=int(t*30+1e-7)))
        identity = lambda path:(path.stat().st_dev,path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns)
        self.faces = SimpleNamespace(_samplers=(self.sampler,self.sampler),_focal_candidate_cache={},
            _identity_key=lambda source,sampler,t:(source,int(t*30+1e-7),30,1))
        self.bodies = SimpleNamespace(_references={0:(1,2,3),1:(1,2,3)},_track_cache={},_frame_cache={})
        self.provider = SimpleNamespace(sources=(self.source,self.source),_identity=identity,
            _identities=(identity(self.source),identity(self.source)),observation_namespace='cpu-v1',
            faces=self.faces,bodies=self.bodies,_persistent=None,_persistent_key=lambda source,key:str(key),
            close=Mock(),diagnostics=lambda:{'base':True})
        def original(source,seconds,descriptor=None):
            if isinstance(source,bool) or not isinstance(source,int) or source not in (0,1):
                raise ValueError('invalid source')
            if not isinstance(seconds,(int,float)) or isinstance(seconds,bool) or not 0<=seconds<100:
                raise ValueError('invalid time')
            return FocalSelectorGeometry(source,seconds,None)
        self.provider.selector_geometry=Mock(side_effect=original)
        self.populate(0,10.)

    def populate(self,source,seconds):
        key=self.faces._identity_key(source,self.sampler,seconds)
        self.bodies._track_cache[key]=SimpleNamespace(observation=None,reason='independent_face_body_anchors_missing')
        center=int(seconds*30+1e-7)//4*4
        times=[(center+i*4)/30 for i in range(-10,11)] + [seconds]
        observation=SimpleNamespace(reference_similarity=.8,pose_screen_xyc=((.2,.3,.9),)*8)
        for time in times:
            frame_key=self.faces._identity_key(source,self.sampler,time)
            self.bodies._frame_cache[frame_key]=SimpleNamespace(observations=(observation,))
            self.faces._focal_candidate_cache[frame_key]=(object(),)
        return key

    def test_completed_missing_query_survives_body_cache_churn(self):
        memo=SourceFirstUnresolvedMemo(self.provider)
        first=memo(0,10.)
        self.bodies._frame_cache.clear();self.bodies._track_cache.clear()
        self.faces._focal_candidate_cache.clear()
        self.assertEqual(memo(0,10.),first)
        self.provider.selector_geometry.assert_called_once()
        self.assertEqual(memo.diagnostics()['hits'],1)
        self.assertFalse(memo.diagnostics()['negative_observations_persisted'])

    def test_same_native_frame_retains_exact_requested_time(self):
        memo=SourceFirstUnresolvedMemo(self.provider)
        memo(0,10.)
        self.assertEqual(memo(0,10.01).source_seconds,10.01)
        self.assertEqual(self.provider.selector_geometry.call_count,1)
        memo(0,10.04)
        self.assertEqual(self.provider.selector_geometry.call_count,2)

    def test_source_namespace_and_source_index_do_not_alias(self):
        memo=SourceFirstUnresolvedMemo(self.provider)
        memo(0,10.);self.populate(1,10.);memo(1,10.)
        self.provider.observation_namespace='mps-v1'
        memo(0,10.)
        self.assertEqual(self.provider.selector_geometry.call_count,3)

    def test_replaced_source_is_not_hidden_by_hit(self):
        memo=SourceFirstUnresolvedMemo(self.provider);memo(0,10.)
        self.source.write_bytes(b'different source bytes')
        with self.assertRaises(OSError):memo(0,10.)

    def test_missing_source_is_not_hidden_by_hit(self):
        memo=SourceFirstUnresolvedMemo(self.provider);memo(0,10.)
        self.source.rename(self.source.with_suffix('.moved'))
        with self.assertRaises(OSError):memo(0,10.)

    def test_incomplete_native_face_decode_is_retryable(self):
        for value in (None,()):
            with self.subTest(value=value):
                self.populate(0,10.)
                key=next(iter(self.faces._focal_candidate_cache))
                if value is None:self.faces._focal_candidate_cache.pop(key)
                else:self.faces._focal_candidate_cache[key]=value
                memo=SourceFirstUnresolvedMemo(self.provider)
                memo(0,10.);memo(0,10.)
                self.assertEqual(memo.diagnostics()['records'],0)

    def test_missing_body_frame_or_failed_pose_is_retryable(self):
        for kind in ('absent','empty','pose_failed'):
            with self.subTest(kind=kind):
                self.populate(0,10.)
                key=next(iter(self.bodies._frame_cache))
                if kind=='absent':self.bodies._frame_cache.pop(key)
                elif kind=='empty':self.bodies._frame_cache[key]=SimpleNamespace(observations=())
                else:self.bodies._frame_cache[key]=SimpleNamespace(observations=(
                    SimpleNamespace(reference_similarity=.8,pose_screen_xyc=()),))
                memo=SourceFirstUnresolvedMemo(self.provider)
                memo(0,10.);self.assertEqual(memo.diagnostics()['records'],0)

    def test_observed_zero_face_frame_differs_from_failed_native_decode(self):
        key=next(iter(self.faces._focal_candidate_cache))
        self.faces._focal_candidate_cache[key]=()
        memo=SourceFirstUnresolvedMemo(self.provider)
        memo.completed_face_frames[key]=True
        memo(0,10.)
        self.assertEqual(memo.diagnostics()['records'],1)

    def test_failed_pose_on_low_clothing_frozen_face_is_not_completed_negative(self):
        key = next(iter(self.bodies._frame_cache))
        self.bodies._frame_cache[key] = SimpleNamespace(observations=(SimpleNamespace(
            reference_similarity=.585, frozen_face_match=True, pose_screen_xyc=()),))
        memo = SourceFirstUnresolvedMemo(self.provider)
        memo(0, 10.)
        self.assertEqual(memo.records, 0)
        self.assertEqual(memo.incomplete_reasons, {'pose_inference_unproven': 1})

    def test_native_completion_trace_does_not_record_failure_or_empty_cache_hit(self):
        self.provider.native_faces=SimpleNamespace(inference_frame_count=0)
        def candidates(source,seconds):
            if seconds==10.:
                self.provider.native_faces.inference_frame_count+=1
            return ()
        self.faces._external_focal_candidate_provider=candidates
        install_source_first_unresolved_memo(self.provider)
        memo=self.provider._source_first_unresolved_memo
        with patch('crosscut.source_first_unresolved_memo._verified_empty_native_detection',return_value=True):
            self.faces._external_focal_candidate_provider(0,10.)
        self.faces._external_focal_candidate_provider(0,11.)
        self.assertIn((0,300,30,1),memo.completed_face_frames)
        self.assertNotIn((0,330,30,1),memo.completed_face_frames)

    def test_haar_or_failed_sface_is_not_a_verified_empty_face(self):
        self.provider.native_faces=SimpleNamespace(inference_frame_count=0)
        def candidates(source,seconds):
            self.provider.native_faces.inference_frame_count+=1
            return ()
        self.faces._external_focal_candidate_provider=candidates
        install_source_first_unresolved_memo(self.provider)
        with patch('crosscut.source_first_unresolved_memo._verified_empty_native_detection',return_value=False):
            self.faces._external_focal_candidate_provider(0,10.)
        self.assertFalse(self.provider._source_first_unresolved_memo.completed_face_frames)

    def test_empty_detection_proof_requires_yunet_and_no_detected_faces(self):
        import numpy as np
        self.provider.native_faces=SimpleNamespace(maximum_detection_width=960,
            frame_at=Mock(return_value=np.zeros((20,20,3),dtype=np.uint8)))
        for backend,faces,expected in (('yunet',(),True),('haar',(),False),
                                       ('none',(),False),('yunet',(object(),),False)):
            with patch('crosscut.visual_score._detect_faces',return_value=(faces,backend)):
                self.assertEqual(_verified_empty_native_detection(self.provider,0,10.),expected)
        self.provider.native_faces.frame_at.return_value=None
        self.assertFalse(_verified_empty_native_detection(self.provider,0,10.))

    def test_unavailable_reference_bank_is_not_memoized(self):
        self.bodies._references[0]=(1,2)
        memo=SourceFirstUnresolvedMemo(self.provider);memo(0,10.)
        self.assertEqual(memo.diagnostics()['records'],0)

    def test_positive_geometry_and_exceptions_are_not_negative_entries(self):
        memo=SourceFirstUnresolvedMemo(self.provider)
        self.provider.selector_geometry.side_effect=OSError('decode')
        with self.assertRaises(OSError):memo(0,10.)
        self.provider.selector_geometry.side_effect=None
        self.provider.selector_geometry.return_value=FocalSelectorGeometry(0,10.,(.1,.1,.1,.1))
        memo(0,10.);self.assertEqual(len(memo.entries),0)

    def test_external_positive_publication_overrides_run_local_negative(self):
        memo=SourceFirstUnresolvedMemo(self.provider);memo(0,10.)
        positive=FocalSelectorGeometry(0,10.,(.1,.1,.1,.1))
        self.provider._persistent=SimpleNamespace(get=Mock(return_value=positive))
        self.assertEqual(memo(0,10.),positive)
        self.assertEqual(len(memo.entries),0)

    def test_external_cache_error_is_not_hidden_by_hit(self):
        memo=SourceFirstUnresolvedMemo(self.provider);memo(0,10.)
        self.provider._persistent=SimpleNamespace(get=Mock(side_effect=OSError('SSD')))
        with self.assertRaises(OSError):memo(0,10.)

    def test_capacity_is_bounded_and_evicted_query_is_retryable(self):
        memo=SourceFirstUnresolvedMemo(self.provider,maximum_entries=1)
        memo(0,10.);self.populate(0,11.);memo(0,11.);memo(0,10.)
        self.assertEqual(len(memo.entries),1)
        self.assertEqual(self.provider.selector_geometry.call_count,3)
        self.assertGreaterEqual(memo.evictions,1)

    def test_install_uses_one_shared_resolver_and_closes_once(self):
        original_close=self.provider.close
        install_source_first_unresolved_memo(self.provider)
        resolver=self.provider.selector_geometry
        install_source_first_unresolved_memo(self.provider)
        self.assertIs(self.provider.selector_geometry,resolver)
        self.provider.selector_geometry(0,10.)
        self.assertEqual(self.provider.diagnostics()['unresolved_query_memo']['records'],1)
        self.provider.close();self.provider.close()
        original_close.assert_called_once()
        self.assertEqual(len(resolver.entries),0)
        with self.assertRaises(RuntimeError):resolver(0,10.)

    def test_invalid_queries_retain_validation(self):
        memo=SourceFirstUnresolvedMemo(self.provider)
        for source,seconds in ((True,10.),(-1,10.),(0,True),(0,float('nan')),(0,-1.)):
            with self.assertRaises(ValueError):memo(source,seconds)
        for capacity in (False,0,-1,1.5):
            with self.assertRaises(ValueError):SourceFirstUnresolvedMemo(self.provider,maximum_entries=capacity)


if __name__=='__main__':unittest.main()
