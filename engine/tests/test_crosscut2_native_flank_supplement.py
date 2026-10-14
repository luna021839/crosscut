from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2 import native_flank_supplement as nfs
from crosscut2.grid_perception import file_identity


class Detector:
    def __init__(self, log, faces):
        self.log, self.faces = log, faces

    def detect(self, image):
        assert image.shape == (540, 960, 3)
        self.log['detect'] += 1
        if not self.faces:
            return 1, None
        face = np.zeros(15, np.float32)
        face[:4], face[-1] = [400, 100, 160, 200], .95
        return 1, face[None]


class Recognizer:
    def __init__(self, log):
        self.log = log

    def alignCrop(self, pixels, face):
        assert pixels.shape == (360, 640, 3)
        return np.zeros((112, 112, 3), np.uint8)

    def feature(self, crop):
        self.log['feature'] += 1
        return np.ones((1, 128), np.float32)


def world(tmp_path, monkeypatch, *, faces=True, count=300):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'bound media')
    models = tmp_path / 'models'
    models.mkdir()
    for name in nfs.FILES:
        (models / name).write_bytes(name.encode())
    log, fail = dict(probe=0, frames=0, decoded=0, detect=0, feature=0, models=0), set()
    clock = nfs.sparse.SeekClock(Fraction(30), count, (0,), file_identity(source), 'c' * 64)

    def probe(path, *, check, maximum_seconds=10.):
        log['probe'] += 1
        return clock

    def frames(path, natives, *, clock, check, width, height, maximum_seconds,receipt):
        log['frames'] += 1
        receipt['strategy']='verified_original_PTS_bounded_native_seek_v1'
        for n in natives:
            if n in fail:
                raise OSError('sparse exact-native decode incomplete: injected')
            log['decoded'] += 1
            yield n, np.full((height, width, 3), n % 251, np.uint8)

    def face_models(assets):
        log['models'] += 1
        return Detector(log, faces), Recognizer(log)
    monkeypatch.setattr(nfs.sparse, 'probe', probe)
    monkeypatch.setattr(nfs.sparse, 'frames', frames)
    monkeypatch.setattr(nfs, '_raster', lambda path: (640, 360, 300))
    monkeypatch.setattr(nfs, '_face_models', face_models)
    return SimpleNamespace(source=source, models=models, log=log, fail=fail, cache=tmp_path / 'cache')


def inputs(w, events):
    """events: (output tick, physical native keys, exposed override or None)."""
    rows = [dict(output_frame=t, source_native_first_incoming=f, exposed_native_ordinals=e or [f - 1, f])
            for t, keys, e in events for f in keys]
    n = len(events)
    bound = [dict(source=0, source_file_identity=list(file_identity(w.source)), source_sha256='d' * 64,
                  fps=30., frame_count=300, native_event_rows=rows)]
    registry = [dict(frames=np.array([t for t, _, _ in events]), heads=np.zeros((n, 2, 8, 4), np.float32),
                     vectors=np.zeros((n, 2, 3, 128), np.float32), observed=np.zeros(n, bool),
                     native_event_keys=[list(k) for _, k, _ in events], actor_binding_certified=False,
                     contract='physical_native_event_registry_v1')]
    return bound, registry


def run(w, bound, registry, **kw):
    return nfs.augment_missing_native_flanks(sources=[w.source], bound=bound, registry=registry,
                                             cache=w.cache, models=w.models, check=lambda: None, **kw)


def forbid(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('warm path must not probe, decode or create face models')
    for owner, name in ((nfs.sparse, 'probe'), (nfs.sparse, 'frames'), (nfs, '_face_models'), (nfs, '_raster')):
        monkeypatch.setattr(owner, name, forbidden)


def saved(w):
    return sorted(w.cache.glob('*')) if w.cache.exists() else []


def test_cold_fills_only_missing_exact_pairs_then_warm_avoids_models_probe_and_decode(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    bound, registry = inputs(w, [(10, [20], None), (40, [80], None), (70, [130], None)])
    registry[0]['observed'][0] = True
    registry[0]['heads'][0, :, 0] = [.1, .2, .3, .9]
    original = {k: registry[0][k].copy() for k in ('heads', 'vectors', 'observed')}
    cold, receipts = run(w, bound, registry)
    assert w.log == dict(probe=1, frames=2, decoded=4, detect=4, feature=4, models=1)
    assert receipts[0]['measured_pairs'] == 2 and receipts[0]['new_face_input_frames'] == 4
    assert receipts[0]['detector_calls'] == 4 and receipts[0]['recognizer_calls'] == 4
    assert receipts[0]['supplemented_output_frames'] == [40, 70] and receipts[0]['status'] == 'complete'
    assert cold[0]['observed'].tolist() == [True, True, True]
    assert np.array_equal(cold[0]['heads'][0], original['heads'][0])
    for key, value in original.items():
        assert np.array_equal(registry[0][key], value)  # caller arrays are not mutated
    assert len(saved(w)) == 2
    forbid(monkeypatch)
    warm, again = run(w, bound, registry)
    assert again[0]['reused_pairs'] == 2 and again[0]['measured_pairs'] == 0
    assert again[0]['new_face_input_frames'] == again[0]['probe_calls'] == again[0]['sparse_decode_calls'] == 0
    for key in ('heads', 'vectors', 'observed'):
        assert np.array_equal(warm[0][key], cold[0][key])


def test_exact_physical_pair_join_rejects_skipped_nearby_and_multiple_cuts(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    bound, registry = inputs(w, [(10, [20], [18, 20]), (20, [40, 41], None), (30, [60], None)])
    nearby = bound[0]['native_event_rows'][-1]
    nearby.update(source_native_first_incoming=61, exposed_native_ordinals=[60, 61])
    result, receipts = run(w, bound, registry)
    assert not result[0]['observed'].any() and not result[0]['heads'].any()
    assert receipts[0]['ineligible_not_exact_adjacent_pair'] == 2
    assert receipts[0]['ineligible_multiple_or_missing_keys'] == 1 and receipts[0]['still_unknown_events'] == 3
    assert w.log == dict(probe=0, frames=0, decoded=0, detect=0, feature=0, models=0) and saved(w) == []


def test_decoder_error_after_partial_window_cannot_save_or_fill_cells(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    bound, registry = inputs(w, [(40, [80], None), (45, [90], None)])
    w.fail.add(90)
    result, receipts = run(w, bound, registry)
    assert w.log['detect'] == receipts[0]['detector_calls'] == 3  # honest spent calls
    assert receipts[0]['failed_pairs'] == 2 and receipts[0]['measured_pairs'] == 0
    assert receipts[0]['status'] == 'partial' and 'decode incomplete' in receipts[0]['errors'][0]
    assert not result[0]['observed'].any() and saved(w) == []
    w.fail.clear()
    result, receipts = run(w, bound, registry)
    assert receipts[0]['measured_pairs'] == 2 and result[0]['observed'].all() and len(saved(w)) == 2


@pytest.mark.parametrize('changed', ['implementation_seal', 'decoder_identity'])
def test_changed_reader_or_runtime_cannot_reuse_pair(tmp_path, monkeypatch, changed):
    assert len(nfs.implementation_seal()) == 64
    w = world(tmp_path, monkeypatch)
    bound, registry = inputs(w, [(40, [80], None)])
    run(w, bound, registry)
    monkeypatch.setattr(nfs, changed, lambda: 'changed-' + changed)
    _, receipts = run(w, bound, registry)
    assert receipts[0]['reused_pairs'] == 0 and receipts[0]['measured_pairs'] == 1
    assert w.log['probe'] == 2 and len(saved(w)) == 2


def test_face_frame_budget_counts_unique_physical_rasters_and_defers_rest(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    bound, registry = inputs(w, [(10, [30], None), (11, [31], None), (40, [90], None)])
    result, receipts = run(w, bound, registry, maximum_new_face_frames=3)
    assert w.log['decoded'] == w.log['detect'] == receipts[0]['new_face_input_frames'] == 3
    assert receipts[0]['deferred_face_frame_budget'] == 1 and receipts[0]['status'] == 'partial'
    assert result[0]['observed'].tolist() == [True, True, False]


def test_no_face_pair_is_cached_but_stays_unknown(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, faces=False)
    bound, registry = inputs(w, [(40, [80], None)])
    result, receipts = run(w, bound, registry)
    assert receipts[0]['kept_unknown_without_faces_on_both_sides'] == 1 and not result[0]['observed'].any()
    forbid(monkeypatch)
    result, receipts = run(w, bound, registry)
    assert receipts[0]['reused_pairs'] == 1 and receipts[0]['still_unknown_events'] == 1
    assert not result[0]['observed'].any()


def test_unverified_clock_is_reported_and_never_fabricated(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, count=299)
    bound, registry = inputs(w, [(40, [80], None)])
    result, receipts = run(w, bound, registry)
    assert receipts[0]['failed_pairs'] == 1 and 'clock' in receipts[0]['errors'][0]
    assert w.log['probe'] == 1 and w.log['models'] == w.log['detect'] == 0
    assert not result[0]['observed'].any() and saved(w) == []


def test_tampered_cache_record_is_rejected_not_reused(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    bound, registry = inputs(w, [(40, [80], None)])
    run(w, bound, registry)
    target, = saved(w)
    with np.load(target, allow_pickle=False) as record:
        data = {k: record[k] for k in record.files}
    data['heads'] = np.full_like(data['heads'], np.nan)
    np.savez_compressed(target, **data)
    with pytest.raises(ValueError, match='malformed'):
        run(w, bound, registry)


def test_overlapping_pair_at_the_chunk_limit_cannot_exceed_real_face_input_budget(tmp_path,monkeypatch):
    w=world(tmp_path,monkeypatch)
    bound,registry=inputs(w,[(t,[t],None) for t in range(30,47)])
    result,receipts=run(w,bound,registry,maximum_new_face_frames=18)
    assert w.log['detect']==receipts[0]['new_face_input_frames']==18
    assert result[0]['observed'].all()


def test_source_changed_after_warm_read_cannot_return_old_measurements_as_current(tmp_path,monkeypatch):
    w=world(tmp_path,monkeypatch);bound,registry=inputs(w,[(40,[80],None)])
    run(w,bound,registry);original=nfs._load
    def changed(*a,**kw):
        result=original(*a,**kw);w.source.write_bytes(b'changed source during warm cache read');return result
    monkeypatch.setattr(nfs,'_load',changed)
    with pytest.raises(OSError,match='source changed'):run(w,bound,registry)
