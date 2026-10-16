"""Complete metadata-owner envelope, NOT original descriptor or disk timing.

Uses the ACTUAL index constructor/publisher/release of the fixed-slot store.
Payload IO/transactions are tested separately on small real files. This test
does not create a2.7GB internal file or claim original producer peak/150s/AV.
The source reservation is the reported byte claim, not loaded observations.
"""
import numpy as np
import pytest
from crosscut2.bounded_suffix_spool import _make_index,_publish,_resolve,_release,capacity_requirements,HEADER_BYTES
from crosscut2.playback_workspace import PlaybackWorkspace


def _new(meta,ordinal,scratch):
    mask=len(meta[6])-1;slot=np.int64(meta[2][meta[7][0]-np.uint64(1)])
    scratch[0,0]=ordinal;scratch[0,1]=ordinal^np.uint64(0xdeadbeef)
    scratch[1,0]=ordinal;scratch[1,1]=ordinal^np.uint64(0xabcdef)
    at=np.int64(scratch[0,0]&np.uint64(mask))
    while meta[6][at]:at=(at+1)&mask
    return _publish(meta,slot,at,4357,scratch[0],scratch[1])


def _schedule(meta,future,incoming,transactions,scratch):
    ordinal=np.uint64(1);peak=0;replacements=0
    for k in range(len(future)):future[k]=_new(meta,ordinal,scratch);ordinal+=np.uint64(1)
    for k in range(len(incoming)):incoming[k]=_new(meta,ordinal,scratch);ordinal+=np.uint64(1)
    for k in range(len(transactions)):transactions[k]=_new(meta,ordinal,scratch);ordinal+=np.uint64(1)
    peak=int(meta[7][1])
    # One transaction slot is reused for the next successfully stored record.
    # Three other transaction owners remain held throughout every replacement.
    _release(meta,transactions[-1]);transactions[-1]=0
    for k in range(len(future)):
        old=future[k];candidate=_new(meta,ordinal,scratch);ordinal+=np.uint64(1)
        # Before publication the old record is STILL owned and resolvable.
        _resolve(meta,old);_resolve(meta,candidate)
        future[k]=candidate;_release(meta,old);replacements+=1
        if int(meta[7][1])!=peak-1:raise AssertionError('replacement owner leaked or victim released early')
    for k in range(len(future)):_release(meta,future[k]);future[k]=0
    for k in range(len(incoming)):_release(meta,incoming[k]);incoming[k]=0
    for k in range(len(transactions)):
        if transactions[k]:_release(meta,transactions[k]);transactions[k]=0
    return peak,replacements


def test_all274_by40_by2_by8_future_slots_origin_and_four_transactions_fit_the_actual_index():
    numba=pytest.importorskip('numba')
    from numba.extending import register_jitable
    for f in (_publish,_resolve,_release,_new):register_jitable(f)
    schedule=numba.njit(cache=False,fastmath=False,parallel=False)(_schedule)
    N,W,L,H,D=274,40,8,120,8;C=(N*W+W)*2*L+4
    budget=PlaybackWorkspace(200*2**20)
    # Explicit REPORT-DERIVED source reservation; no fake observation data.
    budget.claim(86123190+422712,'reported-source-and-index-reservation-not-new-source-data')
    arrays=[budget.empty((2,N*W,2,L),dtype,owner='full-seven-field-DP-owner-envelope') for dtype in
        (np.float64,np.int64,np.int64,np.int64,np.float64,np.float64,np.uint64)]
    for a in arrays:a[:]=0
    raw=budget.empty((W*2*L,H,8),np.uint64,owner='complete-origin-raw-bank-envelope');raw[:]=0
    values=budget.empty((W*2*L,H,D),np.float64,owner='complete-origin-embedding-bank-envelope');values[:]=0
    budget.claim(4*2**20,'conservative-origin-IO-candidate-and-bookkeeping-envelope')
    index=_make_index(budget,C);transactions=budget.empty((4,),np.uint64,owner='transaction-root-owners')
    scratch=budget.empty((2,2),np.uint64,owner='owner-envelope-constant-digest-scratch')
    future=arrays[6][1].ravel();incoming=arrays[6][0,:W].ravel()
    peak,replacements=schedule(index,future,incoming,transactions,scratch)
    assert peak==C==176004 and replacements==175360
    assert int(index[7][1])==0 and int(index[7][0])==C and not index[0].any() and not index[6].any()
    assert not future.any() and not incoming.any() and not transactions.any()
    need=capacity_requirements(embedding_width=D,history_frames=H,capacity=C)
    last=HEADER_BYTES+(C-1)*need['slot_bytes']
    assert last+need['slot_bytes']==need['backing_bytes']==2703425536
    assert budget.peak_bytes<140*2**20 and budget.current_bytes==budget.peak_bytes
    assert schedule.nopython_signatures
