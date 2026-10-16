"""Exact long factorization and weak ownership, not learned/film approval."""
import gc
import numpy as np
import pytest
from crosscut2.compiled_context_beam import CompiledBeam
from crosscut2.playback_suffix_arena import SuffixRoot
from test_crosscut2_playback_suffix_arena import arena,descriptor
from test_crosscut2_compiled_context_beam import fields
from test_crosscut2_compiled_context_port_dp import fixture


@pytest.mark.parametrize('kernel',['numpy','numba'])
@pytest.mark.parametrize('final',[158,159,180,181])
def test_factored_all_prefix_costs_bits_and_suffixes_equal_actual_full_input(kernel,final):
    context,c=fixture(np.array([0,30,final,final+30]),kernel=kernel)
    a=c.arena;empty=a.empty();P=len(context.entries[0]);Q=len(context.exits[2]);L=8
    ids=np.full((P,2,L),empty.handle,np.uint64);costs=np.zeros(ids.shape)
    costs[:]=np.arange(P)[:,None,None]*1e-15+np.arange(L)[None,None]*.001
    cuts=np.broadcast_to(np.arange(L,dtype=np.int64),ids.shape).copy()
    benefit=np.full((P,Q),.025)
    with c.open_origin(0,ids,costs) as bank:
        base,allowed=context.arcs.camera_alternatives(0,2)
        with c.compile_hold(0,2,base,allowed) as hold:
            # Actual frozen current inputs and all prefixes, before sharing.
            expected={};exact_roots={}
            for p,q,code in zip(*np.nonzero(allowed[:,:,:4])):
                p,q,code=map(int,(p,q,code))
                with hold.current(p,q,code) as current:
                    expected[p,q,code]=np.stack([c.score(bank,current,p,bit,np.arange(L)) for bit in (0,1)])
                    root=a.append(empty,current.lease)
                    if (q,code) in exact_roots:
                        assert root==exact_roots[q,code];a.release(root)
                    else:exact_roots[q,code]=root
            before=dict(c.factored_piece_compilations)
            handled,values=c.long_candidates(hold,bank,base,allowed,costs,cuts,benefit,same_episode=False,width=max(P,Q))
            assert np.array_equal(handled,allowed[:,:,:4])
            # Every head was populated by the independent full recipe pass.
            assert c.factored_piece_compilations['head']==before['head']
            for p,q,code in zip(*np.nonzero(handled)):
                p,q,code=map(int,(p,q,code));candidates=[]
                for bit in (0,1):
                    for k in range(L):
                        for claim in (False,True) if bit==0 else (False,):
                            paid=benefit[p,q] if claim else 0.
                            extra=expected[p,q,code][bit,k]
                            back=((p*2+bit)*L+k)
                            rank=(((costs[p,bit,k]+base[p,q,code])+extra)-paid,cuts[p,bit,k],back,code,-paid)
                            candidates.append((rank,bit,k,extra))
                winner=min(candidates)
                assert (values[0][p,q,code,0],values[1][p,q,code,0],values[2][p,q,code,0],code,
                    -values[5][p,q,code,0])==winner[0]
                assert values[6][p,q,code,0]==winner[3]
                assert not np.isfinite(values[0][p,q,code,1])
                with hold.current(p,q,code,part='suffix') as current:
                    assert not len(current.head_ends) and not len(current.tail_ends)
                    assert len(current.frames)==context.history_frames and current.lease is not None
                    root=a.append(empty,current.lease);assert root==exact_roots[q,code];a.release(root)
            # Same destination tails are reused on another proved long origin.
            tail_calls=c.factored_piece_compilations['tail']
            again=c.long_candidates(hold,bank,base,allowed,costs,cuts,benefit,same_episode=True,width=max(P,Q))
            assert c.factored_piece_compilations['tail']==tail_calls
            del again,handled,values
            for root in exact_roots.values():a.release(root)
    a.release(empty);assert all(t['live']==0 for t in a.receipt()['tables'].values());a.close()


def batch_values(P=32):
    shape=(P,1,4,2)
    result=tuple(np.empty(shape,dtype) for dtype in
        (np.float64,np.int64,np.int64,np.int64,np.int64,np.float64,np.float64))
    for p in range(P):
        for code in range(4):
            for bit in (0,1):
                # Improving ranks, code aliases and sub-tolerance differences.
                for column,value in zip(result,(-p*.1+code*np.finfo(float).eps,1,p*16+bit,
                        bit,p%8,.025 if bit else 0.,.123)):
                    column[p,0,code,bit]=value
    return result


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_compiled_long_resume_matches_original_saturated_beam_order_and_weak_hints_do_not_pin(kernel):
    a=arena(kernel=kernel,history_frames=12,root_capacity=32,cell_capacity=128,frame_capacity=256)
    empty=a.empty();batch=batch_values();beam=CompiledBeam(a);reference=CompiledBeam(a)
    actual=tuple(np.stack((np.stack((v,v.copy())),)) for v in fields())
    expected=tuple(v.copy() for v in actual);hints=np.zeros((1,4),np.uint64);builds=[]
    def build(p,q,code,bit,slot):
        builds.append((p,q,code,bit,slot))
        with descriptor(a,0,12,lambda keys:keys.__setitem__((-1,2),10+code%2)) as current:
            return a.append(empty,current)
    for bit in (0,1):
        for variant in range(8):
            with descriptor(a,0,12,lambda keys,v=variant:keys.__setitem__((-1,2),100+v)) as current:
                record=(10.+variant,0,variant,variant%6,0.,.1)
                reference.offer(tuple(v[0,bit] for v in expected),record,empty.handle,current)
                beam.offer(tuple(v[0,bit] for v in actual),record,empty.handle,current)
    for p in range(batch[0].shape[0]):
        for code in range(4):
            for bit in (0,1):
                with descriptor(a,0,12,lambda keys:keys.__setitem__((-1,2),10+code%2)) as current:
                    record=(batch[0][p,0,code,bit],int(batch[1][p,0,code,bit]),int(batch[2][p,0,code,bit]),
                        code,batch[5][p,0,code,bit],batch[6][p,0,code,bit])
                    reference.offer(tuple(v[0,bit] for v in expected),record,empty.handle,current)
    beam.long_batch(actual,batch,hints,build)
    assert len(builds)<=4 and beam.long_suffix_misses==len(builds)
    for bit in (0,1):
        def records(state):return sorted(tuple(v[0,bit,k].item() for v in state) for k in range(8))
        assert records(actual)==records(expected)
    stale=hints.copy()
    for state in (actual,expected):
        for h in state[6].ravel():
            if h:a.release(SuffixRoot(a.owner,int(h)))
        state[6][:]=0;state[0][:]=np.inf
    assert a.receipt()['tables']['roots']['live']==1
    # All hints are now stale, but still nonzero; recycled slots must rebuild.
    before=len(builds);beam.long_batch(actual,batch,hints,build)
    assert len(builds)>before and np.all(hints!=stale)
    for h in actual[6].ravel():
        if h:a.release(SuffixRoot(a.owner,int(h)))
    a.release(empty);assert all(t['live']==0 for t in a.receipt()['tables'].values());a.close()


def test_long_batch_exception_releases_built_but_not_consumed_root_and_actual_buffers_remain_charged():
    a=arena(history_frames=12,root_capacity=16);empty=a.empty();beam=CompiledBeam(a)
    destination=tuple(np.stack((np.stack((v,v.copy())),)) for v in fields())
    batch=batch_values(1);hints=np.zeros((1,4),np.uint64);old=beam._long;calls=0
    def fail_after_build(*args):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('finite post-build interruption')
        return old(*args)
    beam._long=fail_after_build
    def build(*unused):
        with descriptor(a,0,12) as current:return a.append(empty,current)
    with pytest.raises(RuntimeError,match='post-build'):beam.long_batch(destination,batch,hints,build)
    assert a.receipt()['tables']['roots']['live']==1
    a.release(empty);a.close()


def test_factored_relaxation_outputs_use_managed_owners_including_retained_small_views():
    context,c=fixture(np.array([0,30,180,210]));a=c.arena;w=a.workspace;empty=a.empty()
    P=len(context.entries[0]);ids=np.zeros((P,2,8),np.uint64);costs=np.full(ids.shape,np.inf)
    ids[0,0,0]=empty.handle;costs[0,0,0]=0.
    with c.open_origin(0,ids,costs) as bank:
        base,allowed=context.arcs.camera_alternatives(0,2)
        with c.compile_hold(0,2,base,allowed) as hold:
            result=c.long_candidates(hold,bank,base,allowed,costs,np.zeros(costs.shape,np.int64),
                np.zeros(base.shape[:2]),same_episode=True,width=max(base.shape[:2]))
            held=result[1][0].ravel()[:1];allocated=w.current_bytes
            del result;gc.collect();assert w.current_bytes<allocated
            retained=w.current_bytes;del held;gc.collect();assert w.current_bytes<retained
    a.release(empty);a.close()
