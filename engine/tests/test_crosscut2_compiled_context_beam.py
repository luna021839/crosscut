"""Packed beam ownership/rank tests without images, model fits or AV labels."""
import numpy as np
import pytest
from crosscut2.compiled_context_beam import CompiledBeam
from test_crosscut2_playback_suffix_arena import arena,descriptor
from crosscut2.playback_suffix_arena import SuffixRoot


def fields(width=8):
    values=[np.full(width,np.inf),np.zeros(width,np.int64),np.full(width,-1,np.int64),
        np.full(width,-1,np.int64),np.zeros(width),np.zeros(width),np.zeros(width,np.uint64)]
    return tuple(values)


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_beam_rank_rejection_never_constructs_suffix_and_alias_keeps_original_code_cost_backtrace(kernel):
    a=arena(kernel=kernel,history_frames=12,root_capacity=32);empty=a.empty();beam=CompiledBeam(a);state=fields()
    for variant in range(8):
        with descriptor(a,0,12,lambda k,v=variant:k.__setitem__((-1,2),v+10)) as current:
            beam.offer(state,(float(variant),0,variant,variant%6,0.,.1),empty.handle,current)
    before=a.receipt()['tables']['roots']['requests']
    with descriptor(a,0,12,lambda k:k.__setitem__((-1,2),999)) as current:
        beam.offer(state,(100.,0,0,0,0.,0.),empty.handle,current)
    assert a.receipt()['tables']['roots']['requests']==before
    # Same physical playback but a strictly better original camera-code/base
    # record replaces the old candidate; identity does not discard its code.
    with descriptor(a,0,12,lambda k:k.__setitem__((-1,2),10)) as current:
        beam.offer(state,(-np.finfo(float).eps,0,3,5,.025,.1),empty.handle,current)
    at=int(np.argmin(state[0]));assert state[3][at]==5 and state[2][at]==3 and state[4][at]==.025
    assert a.receipt()['tables']['roots']['live']==9
    for handle in state[6]:a.release(SuffixRoot(a.owner,int(handle)))
    a.release(empty);a.close()


def test_compiled_4096_improving_offers_keep_only_live_eight_label_roots():
    pytest.importorskip('numba')
    a=arena(kernel='numba',history_frames=12,frame_capacity=128,cell_capacity=64,root_capacity=16)
    empty=a.empty();beam=CompiledBeam(a);state=fields()
    for variant in range(4096):
        with descriptor(a,0,12,lambda k,v=variant:k.__setitem__((-1,2),v+10)) as current:
            beam.offer(state,(-float(variant),0,variant,variant%6,0.,0.),empty.handle,current)
    receipt=a.receipt()
    assert receipt['tables']['roots']['live']==9 and receipt['tables']['roots']['peak_live']<=10
    assert receipt['tables']['frames']['peak_live']<=21 and receipt['tables']['cells']['peak_live']<=11
    assert beam.discarded==4096-8
    for handle in state[6]:a.release(SuffixRoot(a.owner,int(handle)))
    a.release(empty)
    assert all(t['live']==0 for t in a.receipt()['tables'].values())
    a.close()
