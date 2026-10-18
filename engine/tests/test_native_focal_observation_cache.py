from collections import OrderedDict
from pathlib import Path
import json
import sqlite3

import pytest

from crosscut.direct_member_target_face import FocalFaceCandidate
from crosscut.native_focal_observation_cache import PersistentNativeFocalFaces,producer_components,producer_namespace


class Producer:
    def __init__(self,source,*,faces=None,complete=True,basis='native_test_basis'):
        self.sources=(source,);self._cache=OrderedDict();self._candidate_completeness={}
        self.basis=basis;self.complete=complete;self.calls=0;self.inference_frame_count=0;self.closed=False
        self.faces=(FocalFaceCandidate((.4,.2,.1,.12),(1.,)+127*(0.,),.03,.04),) if faces is None else faces

    def fingerprint(self):return dict(unchanged_sensor=self.basis)

    def candidates_at(self,s,t):
        if (s,t) not in self._cache:
            self.calls+=1;self.inference_frame_count+=1
            self._cache[s,t]=self.faces;self._candidate_completeness[s,t]=self.complete
        return self._cache[s,t]

    def candidate_observation_complete(self,s,t):return self._candidate_completeness.get((s,t),False)

    def close(self):self.closed=True;self._cache.clear();self._candidate_completeness.clear()


@pytest.fixture
def source(tmp_path):
    path=tmp_path/'private-original.mp4';path.write_bytes(b'not decoded by these cache-only tests');return path


def test_fresh_provider_reuses_actual_descriptor_bytes_and_completeness_without_inference(tmp_path,source):
    first=Producer(source);cache=PersistentNativeFocalFaces(first,tmp_path)
    old_fingerprint=first.fingerprint();expected=cache.candidates_at(0,1.)
    assert first.calls==1 and cache.writes==1 and cache.fingerprint()==old_fingerprint
    cache.close();cache.close();assert first.closed
    second=Producer(source,faces=());warm=PersistentNativeFocalFaces(second,tmp_path)
    assert warm.candidates_at(0,1.)==expected and second.calls==0 and warm.hits==1
    assert warm.candidate_observation_complete(0,1.) and warm.inference_frame_count==0
    assert warm.fingerprint()==old_fingerprint and warm.diagnostics()['private_query_embeddings_persisted']
    warm.close()


def test_complete_empty_is_cached_but_failed_or_partial_model_result_is_not(tmp_path,source):
    for complete in (False,True):
        root=tmp_path/str(complete);root.mkdir();first=Producer(source,faces=(),complete=complete)
        cache=PersistentNativeFocalFaces(first,root);assert cache.candidates_at(0,1.)==()
        assert cache.writes==int(complete);cache.close()
        second=Producer(source,faces=(),complete=complete);warm=PersistentNativeFocalFaces(second,root)
        assert warm.candidates_at(0,1.)==() and second.calls==int(not complete)
        warm.close()


def test_exact_vfr_query_seconds_and_sensor_basis_are_not_aliased(tmp_path,source):
    first=Producer(source);cache=PersistentNativeFocalFaces(first,tmp_path)
    cache.candidates_at(0,1.);cache.close()
    second=Producer(source);cache=PersistentNativeFocalFaces(second,tmp_path)
    cache.candidates_at(0,1.000001);assert second.calls==1 and cache.hits==0;cache.close()
    third=Producer(source,basis='different_model_or_decoder');cache=PersistentNativeFocalFaces(third,tmp_path)
    cache.candidates_at(0,1.);assert third.calls==1 and cache.hits==0;cache.close()


def test_changed_source_or_unavailable_cache_is_not_a_silent_fallback(tmp_path,source):
    cache=PersistentNativeFocalFaces(Producer(source),tmp_path);cache.candidates_at(0,1.)
    source.write_bytes(b'changed original')
    with pytest.raises(OSError,match='changed/disconnected'):cache.candidates_at(0,1.)
    cache.close()
    cache=PersistentNativeFocalFaces(Producer(source),tmp_path);cache.path.rename(tmp_path/'moved.sqlite3')
    with pytest.raises(OSError):cache.candidates_at(0,1.)
    cache.close()


def test_corrupt_cache_row_recomputes_instead_of_inventing_an_empty_face(tmp_path,source):
    cache=PersistentNativeFocalFaces(Producer(source),tmp_path);cache.candidates_at(0,1.)
    cache.connection.execute('UPDATE observations SET payload=?',(b'corrupt',));cache.connection.commit();cache.close()
    second=Producer(source);cache=PersistentNativeFocalFaces(second,tmp_path)
    assert len(cache.candidates_at(0,1.))==1 and second.calls==1 and cache.invalid==1
    cache.close()


def test_unsupported_descriptor_family_stays_uncached_not_relabelled_as_sface(tmp_path,source):
    producer=Producer(source,faces=(FocalFaceCandidate((.4,.2,.1,.12),(1.,0.),.03,.04),))
    cache=PersistentNativeFocalFaces(producer,tmp_path)
    assert cache.candidates_at(0,1.)==producer.faces and cache.writes==0
    cache.close()


def test_boolean_and_nonfinite_inputs_cannot_alias_a_cached_source(tmp_path,source):
    cache=PersistentNativeFocalFaces(Producer(source),tmp_path)
    for s,t in ((True,1.),(0,True),(0,float('nan')),(0,-1.)):
        with pytest.raises(ValueError):cache.candidates_at(s,t)
    assert cache.writes==0;cache.close()


def test_completed_row_publication_does_not_hold_a_lock_over_the_next_model_call(tmp_path,source):
    first=PersistentNativeFocalFaces(Producer(source),tmp_path)
    second=PersistentNativeFocalFaces(Producer(source),tmp_path)
    first.candidates_at(0,1.);second.candidates_at(0,2.)
    assert first.writes==1 and second.writes==1
    assert first.candidates_at(0,2.)==second.producer.faces and first.hits==1
    first.close();second.close()


def test_unrelated_visual_editor_policy_is_not_a_face_sensor_dependency(source,monkeypatch):
    producer=Producer(source);before=producer_namespace(producer);components=producer_components(producer)
    assert 'visual_score.py' not in components['decoder_dependencies']
    assert 'visual._detect_faces' in components['kernels'] and 'visual._normalized_face_metadata' in components['kernels']
    # Appending/changing an editorial function in the same huge module must
    # not invalidate every measured raw face. Model kernels remain bound.
    import crosscut.visual_score as visual
    monkeypatch.setattr(visual,'_unrelated_editorial_policy_test_only',lambda:123,raising=False)
    assert producer_namespace(producer)==before
