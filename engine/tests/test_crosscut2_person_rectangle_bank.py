from fractions import Fraction
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from crosscut2 import person_rectangle_bank as people
from crosscut2.grid_perception import file_identity


def world(tmp_path, monkeypatch, *, sources=1, empty=False):
    paths = [tmp_path / f'source-{s}.mp4' for s in range(sources)]
    for path in paths:
        path.write_bytes(b'bound source')
    model = tmp_path / 'model.onnx'; model.write_bytes(b'pinned test model')
    log = dict(models=0, probe=[], decoded=[], infer=[])
    state = SimpleNamespace(paths=paths, model=model, cache=tmp_path / 'cache', log=log,
        fail=set(), empty=empty, runtime='test-runtime', wrong_clock=False)

    def runtime(path):
        return dict(test=state.runtime), file_identity(path)

    def probe(path, *, check, maximum_seconds=10.):
        check(); s = paths.index(path); log['probe'].append(s)
        return people.sparse.SeekClock(Fraction(30), 299 if state.wrong_clock else 300,
            (0,), file_identity(path), 'c' * 64)

    def frames(path, natives, *, clock, check, width, height, receipt, maximum_seconds):
        s = paths.index(path)
        for f in natives:
            check()
            if (s, f) in state.fail:
                raise OSError('failed next verified sparse window')
            # Each yield represents an already EOF-validated sparse window.
            log['decoded'].append((s, f))
            yield f, np.full((height, width, 3), f, np.uint8)

    class Detector:
        def infer(self, pixels):
            log['infer'].append(int(pixels[0, 0, 0]))
            return (np.empty((0, 5), np.float32) if state.empty
                else np.array([[160, 36, 480, 324, .9]], np.float32))

    def detector(path):
        log['models'] += 1; return Detector()

    monkeypatch.setattr(people, '_runtime', runtime)
    monkeypatch.setattr(people, '_detector', detector)
    monkeypatch.setattr(people, '_raster', lambda p: (640, 360, 300))
    monkeypatch.setattr(people.sparse, 'probe', probe)
    monkeypatch.setattr(people.sparse, 'frames', frames)
    state.bound = [dict(source=s, source_file_identity=list(file_identity(p)),
        source_sha256='d' * 64, fps=30., frame_count=300, native_event_rows=[])
        for s, p in enumerate(paths)]
    return state


def run(w, requests, **kw):
    return people.observe_person_rectangles(sources=w.paths, bound=w.bound, requests=requests,
        cache=w.cache, model=w.model, check=lambda: None, **kw)


def forbid_cold(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('warm must not create session, probe or decode')
    for owner, name in ((people, '_detector'), (people, '_raster'),
                        (people.sparse, 'probe'), (people.sparse, 'frames')):
        monkeypatch.setattr(owner, name, forbidden)


def test_cold_exact_normalized_boxes_then_warm_no_probe_decode_or_session(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    cold, receipt = run(w, [{20: 1, 60: 0, 61: 0}])
    assert receipt[0]['new_person_forwards'] == 3 and w.log['models'] == 1
    assert w.log['decoded'] == [(0, 60), (0, 61), (0, 20)]
    np.testing.assert_allclose(cold[0][20], [[.25, .1, .75, .9, .9]])
    assert receipt[0]['status'] == 'complete' and len(list(w.cache.glob('*.npz'))) == 3
    forbid_cold(monkeypatch)
    warm, again = run(w, [{20: 1, 60: 0, 61: 0}], maximum_new_frames=0)
    assert again[0]['reused_frames'] == 3
    assert again[0]['new_person_forwards'] == again[0]['probe_calls'] == again[0]['sparse_decode_calls'] == 0
    for f in cold[0]:
        assert np.array_equal(warm[0][f], cold[0][f])


def test_global_budget_prioritizes_later_source_native_over_earlier_grid(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, sources=2)
    output, receipts = run(w, [{20: 1, 30: 1}, {40: 0, 41: 0}], maximum_new_frames=2)
    assert output[0] == {} and set(output[1]) == {40, 41}
    assert receipts[0]['deferred_frame_budget'] == 2
    assert sum(r['new_person_forwards'] for r in receipts) == 2
    assert w.log['probe'] == [1]


def test_requests_use_actually_exposed_skipped_pair_not_nominal_adjacent_flank():
    heads = np.zeros((2, 2, 8, 4), np.float32)
    heads[0, 1, 0] = [.5, .4, .1, .9]
    bank = dict(heads=heads, native_indices=np.array([[10, 11], [20, 21]]))
    bound = dict(source=0, native_event_rows=[dict(output_frame=40,
        source_native_first_incoming=80, exposed_native_ordinals=[77, 80])])
    native = dict(frames=np.array([40]), observed=np.array([False]), heads=np.zeros((1, 2, 8, 4)))
    requests = people.requested_unknown_endpoints([bank], [bound], [native])
    assert requests == [{10: 1, 20: 1, 21: 1, 77: 0, 80: 0}]
    assert 79 not in requests[0] and 11 not in requests[0]


def test_zero_box_attempt_is_reusable_not_certified_absence(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, empty=True)
    output, receipt = run(w, [{20: 1}])
    assert output[0][20].shape == (0, 5) and receipt[0]['zero_box_attempts'] == 1
    assert not receipt[0]['actor_identity_or_absence_certified']
    forbid_cold(monkeypatch)
    again, report = run(w, [{20: 1}])
    assert report[0]['reused_frames'] == report[0]['zero_box_attempts'] == 1
    assert again[0][20].shape == (0, 5)


def test_later_failed_window_keeps_only_completed_verified_raster(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch); w.fail.add((0, 60))
    output, receipt = run(w, [{20: 1, 60: 1}])
    assert set(output[0]) == {20} and len(list(w.cache.glob('*.npz'))) == 1
    assert receipt[0]['failed_frames'] == 1 and receipt[0]['status'] == 'partial'
    assert receipt[0]['new_person_forwards'] == 1
    w.fail.clear()
    output, receipt = run(w, [{20: 1, 60: 1}])
    assert receipt[0]['reused_frames'] == receipt[0]['new_person_forwards'] == 1
    assert set(output[0]) == {20, 60}


def test_unverified_clock_cannot_create_model_or_cache(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch); w.wrong_clock = True
    output, receipts = run(w, [{20: 1}])
    assert output == [{}] and w.log['models'] == 0 and not w.cache.exists()
    assert receipts[0]['failed_frames'] == 1 and 'clock' in receipts[0]['errors'][0]


def test_changed_runtime_does_not_mix_numeric_cache(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    run(w, [{20: 1}]); w.runtime = 'new-runtime'
    _, receipts = run(w, [{20: 1}])
    assert receipts[0]['reused_frames'] == 0 and receipts[0]['new_person_forwards'] == 1
    assert len(list(w.cache.glob('*.npz'))) == 2


def test_source_changed_after_warm_read_is_fatal(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch); run(w, [{20: 1}]); original = people._load
    def changed(*a, **kw):
        value = original(*a, **kw); w.paths[0].write_bytes(b'changed source'); return value
    monkeypatch.setattr(people, '_load', changed)
    with pytest.raises(OSError, match='changed'):
        run(w, [{20: 1}])


def test_model_changed_during_inference_cannot_commit(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    class Changed:
        def infer(self, pixels):
            w.model.write_bytes(b'changed model'); return np.array([[1, 1, 20, 20, .9]], np.float32)
    monkeypatch.setattr(people, '_detector', lambda p: Changed())
    with pytest.raises(OSError, match='changed'):
        run(w, [{20: 1}])
    assert not list(w.cache.glob('*.npz'))
    assert not list(w.cache.glob('.partial-*'))


def test_malformed_cache_is_not_silently_replaced(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch); run(w, [{20: 1}])
    target, = w.cache.glob('*.npz')
    with np.load(target, allow_pickle=False) as record:
        arrays = {k: record[k] for k in record.files}
    arrays['rectangles'] = np.array([[0., 0., np.nan, 1., .9]], np.float32)
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError, match='malformed'):
        run(w, [{20: 1}])


@pytest.mark.parametrize('requests', [[{True: 0}], [{-1: 0}], [{300: 0}], [{20: -1}], [{20: True}]])
def test_invalid_physical_domain_never_decodes(tmp_path, monkeypatch, requests):
    w = world(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='physical ordinal'):
        run(w, requests)
    assert not w.log['probe'] and not w.log['infer']


def test_time_limit_defers_without_session_or_retry(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch); calls = iter([0., 2., 2., 2., 2.])
    monkeypatch.setattr(people, 'monotonic', lambda: next(calls, 2.))
    output, receipt = run(w, [{20: 1}], maximum_seconds=1.)
    assert output == [{}] and receipt[0]['deferred_time_budget'] == 1
    assert not w.log['probe'] and not w.log['models']


def test_real_fingerprint_pins_asset_runtime_reader_without_loading_detector(tmp_path, monkeypatch):
    model = tmp_path / 'model.onnx'; model.write_bytes(b'fixture model')
    monkeypatch.setattr(people.native_person_onnx, 'PINNED_YOLO11L_ONNX_SHA256',
                        hashlib.sha256(model.read_bytes()).hexdigest())
    monkeypatch.setattr(people, 'version', lambda name: 'test-ort')
    def forbidden(*a, **kw):
        raise AssertionError('fingerprinting must not instantiate or probe')
    monkeypatch.setattr(people, '_detector', forbidden)
    monkeypatch.setattr(people.sparse, 'probe', forbidden)
    original, identity = people._runtime(model)
    assert len(original['implementation']) == 6 and original['onnxruntime'] == 'test-ort'
    assert identity == file_identity(model)
    monkeypatch.setattr(people, 'function_text', lambda *a: 'changed reader or native clock')
    changed, _ = people._runtime(model)
    assert changed['implementation'] != original['implementation']
    model.write_bytes(b'corrupt fixture model')
    with pytest.raises(ValueError, match='pinned YOLO'):
        people._runtime(model)


def test_decoder_wait_timeout_is_partial_receipt_not_pass_abort(tmp_path, monkeypatch):
    import subprocess
    w = world(tmp_path, monkeypatch, sources=2); original = people.sparse.frames
    def hung(path, natives, **kw):
        if w.paths.index(path) == 0:
            raise subprocess.TimeoutExpired('ffmpeg', 10.)
        yield from original(path, natives, **kw)
    monkeypatch.setattr(people.sparse, 'frames', hung)
    output, receipts = run(w, [{20: 0}, {40: 0}])
    assert receipts[0]['failed_frames'] == 1 and receipts[0]['status'] == 'partial'
    assert set(output[1]) == {40}


def test_window_deadline_is_failure_not_global_budget(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, sources=2); original = people.sparse.frames
    def stuck(path, natives, **kw):
        if w.paths.index(path) == 0:
            raise TimeoutError('single sparse decode window exhausted')
        yield from original(path, natives, **kw)
    monkeypatch.setattr(people.sparse, 'frames', stuck)
    output, receipts = run(w, [{20: 0}, {40: 0}])
    assert receipts[0]['failed_frames'] == 1 and receipts[0]['deferred_time_budget'] == 0
    assert set(output[1]) == {40}


def test_extracted_measurement_is_bitwise_initial_inline_arithmetic():
    rng = np.random.default_rng(42)
    for count in (0, 1, 2, 12, 100):
        xy = rng.random((count, 4), dtype=np.float32)
        xy[:, [0, 2]] = np.sort(xy[:, [0, 2]], axis=1)
        xy[:, [1, 3]] = np.sort(xy[:, [1, 3]], axis=1)
        raw = np.column_stack((xy * [1920, 1080, 1920, 1080], np.full(count, .9))).astype(np.float32)
        class Detector:
            def infer(self, pixels):return raw.copy()
        original = np.asarray(raw, np.float32).copy()
        original[:, :4] /= np.array([1920, 1080, 1920, 1080], np.float32)
        actual = people._measure(None, Detector(), np.array([1920, 1080], np.int64))
        assert np.array_equal(actual, original)
    text = '\n'.join(people.function_text(people.__file__, f)
        for f in ('_measure', '_validate', '_detector', '_raster'))
    assert hashlib.sha256(text.encode()).hexdigest() == people.QUALIFIED_RAW_RECTANGLE_SEMANTICS_SHA256


def test_qualified_initial_numeric_cache_reused_without_rewriting_its_provenance(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    runtime = dict(implementation=[people.LEGACY_RAW_BANK_IMPLEMENTATION_SHA256, *['unchanged']*5],
                   runtime='same-model-decoder-and-libraries')
    monkeypatch.setattr(people, '_runtime', lambda p: (dict(runtime), file_identity(p)))
    cold, _ = run(w, [{20: 1}])
    target, = w.cache.glob('*.npz'); before = target.read_bytes()
    runtime['implementation'] = [people.QUALIFIED_RAW_RECTANGLE_SEMANTICS_SHA256, *['unchanged']*5]
    forbid_cold(monkeypatch)
    warm, receipts = run(w, [{20: 1}], maximum_new_frames=0)
    assert receipts[0]['qualified_legacy_raw_frames_reused'] == receipts[0]['reused_frames'] == 1
    assert receipts[0]['new_person_forwards'] == 0 and np.array_equal(cold[0][20], warm[0][20])
    assert target.read_bytes() == before and len(list(w.cache.glob('*.npz'))) == 1


@pytest.mark.parametrize('index', range(1, 6))
def test_legacy_bridge_cannot_hide_changed_model_or_reader_seals(tmp_path, monkeypatch, index):
    w = world(tmp_path, monkeypatch)
    runtime = dict(implementation=[people.LEGACY_RAW_BANK_IMPLEMENTATION_SHA256, *['unchanged']*5])
    monkeypatch.setattr(people, '_runtime', lambda p: (dict(runtime), file_identity(p)))
    run(w, [{20: 1}])
    runtime['implementation'] = [people.QUALIFIED_RAW_RECTANGLE_SEMANTICS_SHA256, *['unchanged']*5]
    runtime['implementation'][index] = 'changed'
    forbid_cold(monkeypatch)
    output, receipts = run(w, [{20: 1}], maximum_new_frames=0)
    assert output == [{}] and receipts[0]['reused_frames'] == 0
    assert receipts[0]['deferred_frame_budget'] == 1


def test_one_unseekable_window_cannot_drop_other_supported_source_frames(tmp_path,monkeypatch):
    w=world(tmp_path,monkeypatch);original=people.sparse.frames
    def real_planning(path,natives,**kw):
        people.sparse.windows(natives,kw['clock'])
        yield from original(path,natives,**kw)
    monkeypatch.setattr(people.sparse,'frames',real_planning)
    output,receipts=run(w,[{20:0,240:0,270:0}])
    assert set(output[0])=={20}
    assert receipts[0]['failed_frames']==2 and receipts[0]['new_person_forwards']==1
    assert receipts[0]['status']=='partial' and not receipts[0]['whole_source_scan_fallback']


def test_combined_preroll_limit_can_split_before_any_decode_or_model_retry(tmp_path,monkeypatch):
    from dataclasses import replace
    w=world(tmp_path,monkeypatch);probe=people.sparse.probe;frames=people.sparse.frames
    monkeypatch.setattr(people.sparse,'probe',lambda *a,**kw:replace(probe(*a,**kw),keyframes=(0,220)))
    def real_planning(path,natives,**kw):
        people.sparse.windows(natives,kw['clock'])
        yield from frames(path,natives,**kw)
    monkeypatch.setattr(people.sparse,'frames',real_planning)
    output,receipts=run(w,[{200:0,250:0}],maximum_new_frames=2)
    assert set(output[0])=={200,250}
    assert receipts[0]['new_person_forwards']==2 and receipts[0]['failed_frames']==0
    assert w.log['infer']==[200,250] and receipts[0]['status']=='complete'
