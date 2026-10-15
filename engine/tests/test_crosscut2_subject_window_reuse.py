from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from crosscut2.subject_window_reuse import retrieve,covered_offsets
from test_crosscut2_dense_subject_catalogue import prepared


def test_adjacent_cut_can_reuse_actual_native_window_without_renaming_track():
    _,window,_=prepared()
    before=window.body.copy()
    result=retrieve([('old.npz',window)],source=0,seed_native=103,seed_head=window.heads[17],needed=np.arange(96,110))
    assert result[0]=='old.npz' and result[1] is window
    np.testing.assert_array_equal(window.body,before)
    clock=SimpleNamespace(output_frames=np.arange(-14,15),native_indices=np.arange(89,118))
    assert covered_offsets(window,window,clock,clock,list(range(-6,7)))==list(range(-6,6))


def test_wrong_source_unseen_seed_or_uncovered_needed_frames_are_not_cache_hits():
    _,window,_=prepared();kw=dict(source=0,seed_native=100,seed_head=window.heads[14],needed=np.arange(93,107))
    assert retrieve([('old',window)],**dict(kw,source=1)) is None
    assert retrieve([('old',window)],**dict(kw,seed_head=np.array([.9,.9,.02,.99]))) is None
    assert retrieve([('old',window)],**dict(kw,needed=np.arange(50,107))) is None


def test_conflicting_overlapping_windows_cannot_win_by_cache_file_order():
    _,window,_=prepared();body=window.body.copy();body[15,4,0]+=.04
    other=replace(window,body=body,track='another_seed')
    kw=dict(source=0,seed_native=100,seed_head=window.heads[14],needed=np.arange(93,107))
    for rows in ([('a',window),('b',other)],[('b',other),('a',window)]):assert retrieve(rows,**kw) is None
