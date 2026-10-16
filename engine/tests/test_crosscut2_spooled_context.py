"""Finite original-producer parity; no original whole graph/fit/film proof."""
import gc
from itertools import combinations
import numpy as np
import pytest
from crosscut2.bounded_suffix_spool import capacity_requirements
from crosscut2.compiled_played_context import CompiledPlayback
from crosscut2.compiled_context_port_dp import solve as compiled_solve
from crosscut2.context_port_dp import solve as reference_solve,ArcRef
from crosscut2.playback_workspace import PlaybackWorkspace
from crosscut2.playback_suffix_arena import SuffixRoot
from crosscut2.spooled_playback_arena import SpooledPlaybackArena
from crosscut2.spooled_context_beam import SpooledBeam
from test_crosscut2_compiled_context_port_dp import fixture,problem
from test_crosscut2_compiled_context_beam import fields
from test_crosscut2_playback_suffix_arena import descriptor


def spooled(context,tmp_path,*,kernel='numpy'):
    W=max(map(len,context.entries+context.exits));C=(len(context.nodes)*W+W)*2*8+4
    H=context.history_frames;D=context.model.embedding_width
    need=capacity_requirements(embedding_width=D,history_frames=H,capacity=C)
    a=SpooledPlaybackArena(namespace=CompiledPlayback.namespace(context),embedding_width=D,history_frames=H,
        root_capacity=C,workspace=PlaybackWorkspace(200*2**20),directory=tmp_path,
        maximum_spool_bytes=need['backing_bytes'],kernel=kernel)
    return CompiledPlayback(context,arena=a)


@pytest.mark.parametrize('kernel',['numpy','numba'])
@pytest.mark.parametrize('long',[False,True])
@pytest.mark.parametrize('nonlinear',[False,True])
def test_actual_all_camera_native_target_cost_code_paid_bit_and_replay_match_with_spooled_suffixes(tmp_path,kernel,long,nonlinear):
    nodes=np.array([0,30,180,210,240]) if long else np.array([0,18,36,54])
    context,previous=fixture(nodes,kernel=kernel,nonlinear=nonlinear)
    previous.arena.close();del previous;gc.collect()
    entry,exit,kw=problem(context)
    old=reference_solve(nodes,entry,exit,context_cost=context.incremental,history_key=context.equivalent_key,
        context_lower_bound=context.lower_bound,long_context_batch=context.long_candidates,**kw)
    c=spooled(context,tmp_path,kernel=kernel);a=c.arena;path=a.store.path
    def forbidden(*args,**kwargs):raise AssertionError('raw history reconstruction in spooled executor')
    context._encoded_history=context._blocks=forbidden
    new=compiled_solve(nodes,entry,exit,compiler=c,**kw)
    assert new[:2]==old[:2]
    for key in ('selected_arc_codes','opportunity_claims','camera_alternatives_scored'):
        assert new[2][key]==old[2][key]
    np.testing.assert_array_equal(new[2]['selected_context_costs'],old[2]['selected_context_costs'])
    refs=[ArcRef(*run,code) for run,code in zip(new[0],new[2]['selected_arc_codes'])]
    parts,total=context.replay(refs)
    assert parts==pytest.approx(new[2]['selected_context_costs'],abs=2e-13)
    assert sum(parts)==pytest.approx(total,abs=2e-13)
    receipt=a.receipt()
    assert receipt['origin_banks_bound']==c.origin_decode_calls
    assert receipt['backing']['physical_space_reserved'] and not receipt['hidden_resident_frame_cache']
    assert all(v['live']==0 for v in receipt['tables'].values())
    assert c.active_origin is None and a._bank is None
    assert a.workspace.peak_bytes<=200*2**20
    assert not receipt['whole_runtime_or_media_approved']
    a.close();assert not path.exists()


def test_after_actual_current_compile_short_extension_uses_bound_raw_bank_not_source_or_per_prefix_file_read(tmp_path,monkeypatch):
    context,old=fixture(np.array([0,18,36]));old.arena.close();del old
    c=spooled(context,tmp_path);a=c.arena;empty=a.empty()
    def bank(i,p,root):
        ids=np.zeros((len(context.entries[i]),2,8),np.uint64);costs=np.full(ids.shape,np.inf)
        ids[p,0,0]=root.handle;costs[p,0,0]=0.
        return c.open_origin(i,ids,costs)
    p=next(k for k,v in enumerate(context.entries[0]) if v.source==0)
    q=next(k for k,v in enumerate(context.exits[1]) if v.source==0)
    with bank(0,p,empty):
        base,allowed=context.arcs.camera_alternatives(0,1)
        with c.compile_hold(0,1,base,allowed) as hold:
            with hold.current(p,q,4) as current:past=a.append(empty,current.lease)
    p=next(k for k,v in enumerate(context.entries[1]) if v.source==1)
    q=next(k for k,v in enumerate(context.exits[2]) if v.source==1)
    with bank(1,p,past) as prefix:
        base,allowed=context.arcs.camera_alternatives(1,2)
        with c.compile_hold(1,2,base,allowed) as hold:
            with hold.current(p,q,4) as current:
                def forbidden(*args,**kwargs):raise AssertionError('post-compile raw reconstruction or prefix file read')
                monkeypatch.setattr(context,'_native',forbidden)
                monkeypatch.setattr(context,'_intent_projection',forbidden)
                monkeypatch.setattr(context.arcs,'sample_camera',forbidden)
                monkeypatch.setattr(context.model,'encode',forbidden)
                monkeypatch.setattr('crosscut2.compiled_played_context.pack',forbidden)
                with monkeypatch.context() as patch:
                    patch.setattr(a.store,'_payload',forbidden)
                    # Unique new raw suffix; no equality-collision reread is
                    # needed. The prior prefix is already in the origin bank.
                    assert np.isfinite(c.score(prefix,current,p,0,[0])).all()
                    root=a.append(past,current.lease)
                with a.decode([root],end=36) as data:
                    np.testing.assert_array_equal(data.keys[0,:18],prefix.decoded.keys[0])
                    np.testing.assert_array_equal(data.keys[0,18:],current.keys)
                a.release(root)
    a.release(past);a.release(empty);a.close()


@pytest.mark.parametrize('length',[12,13,14,37,38,157,158])
def test_actual_boundary_lengths_store_every_admitted_camera_with_raw_embedding_bits_and_masks_unchanged(tmp_path,length):
    context,old=fixture(np.array([0,length,length+13]));old.arena.close();del old
    c=spooled(context,tmp_path);a=c.arena;empty=a.empty();P=len(context.entries[0])
    ids=np.zeros((P,2,8),np.uint64);costs=np.full(ids.shape,np.inf)
    ids[:,0,0]=empty.handle;costs[:,0,0]=0.;seen=set()
    with c.open_origin(0,ids,costs):
        with c.reserve_hold(0,1) as producer:
            base,allowed=context.arcs.camera_alternatives(0,1);expected_base=base.copy();expected_mask=allowed.copy()
            producer.observe(base,allowed,np.zeros(base.shape[:2]))
            with c.compile_hold(0,1,base,allowed) as hold:
                for p,q,code in zip(*np.nonzero(allowed)):
                    p,q,code=map(int,(p,q,code));seen.add(code)
                    with hold.current(p,q,code) as current:
                        root=a.append(empty,current.lease)
                        with a.decode([root],end=length) as bank:
                            np.testing.assert_array_equal(bank.keys[0],current.keys[-120:])
                            np.testing.assert_array_equal(bank.values[0].view(np.uint64),current.encoded[-120:].view(np.uint64))
                        a.release(root)
                np.testing.assert_array_equal(base,expected_base);np.testing.assert_array_equal(allowed,expected_mask)
    assert {0,1,2,3}<=seen
    if length<38:assert 4 in seen
    if length>=14:assert 5 in seen
    a.release(empty);a.close()


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_rejected_or_failed_store_never_releases_or_rewrites_existing_eight_labels(tmp_path,kernel,monkeypatch):
    need=capacity_requirements(embedding_width=2,history_frames=12,capacity=16)
    a=SpooledPlaybackArena(namespace=('finite-numeric-only',),embedding_width=2,history_frames=12,root_capacity=16,
        workspace=PlaybackWorkspace(2**20),directory=tmp_path,maximum_spool_bytes=need['backing_bytes'],kernel=kernel)
    empty=a.empty();beam=SpooledBeam(a);state=fields()
    for i in range(8):
        with descriptor(a,0,12,lambda keys,v=i:keys.__setitem__((-1,2),100+v)) as current:
            beam.offer(state,(float(i),0,i,i%6,0.,.1),empty.handle,current)
    snapshot=tuple(v.copy() for v in state);before=a.store.receipt();calls=[]
    def broken(*unused):calls.append(True);raise OSError('finite backing write failure')
    with descriptor(a,0,12,lambda keys:keys.__setitem__((-1,2),999)) as current:
        monkeypatch.setattr(a.store,'_write',broken)
        beam.offer(state,(100.,0,0,0,0.,.1),empty.handle,current);assert not calls
        with pytest.raises(OSError,match='backing write'):
            beam.offer(state,(-1.,0,0,0,0.,.1),empty.handle,current)
    assert len(calls)==1 and a.store.live==before['live_roots']
    for value,expected in zip(state,snapshot):np.testing.assert_array_equal(value,expected)
    for handle in state[6]:a.release(SuffixRoot(a.owner,int(handle)))
    a.release(empty);a.close()


def test_cancelled_actual_search_releases_future_roots_banks_and_transient_currents(tmp_path):
    context,old=fixture(np.array([0,18,36,54]));old.arena.close();del old
    c=spooled(context,tmp_path);entry,exit,kw=problem(context);calls=0
    def cancelled():
        nonlocal calls
        calls+=1
        if calls==5:raise TimeoutError('finite spooled search cancellation')
    with pytest.raises(TimeoutError,match='search cancellation'):
        compiled_solve(context.nodes,entry,exit,compiler=c,check=cancelled,**kw)
    assert c.active_origin is None and c.arena._bank is None
    assert all(v['live']==0 for v in c.arena.receipt()['tables'].values())
    c.arena.close()


def test_proved_current_hold_tail_is_not_sampled_packed_or_encoded_again_for_viable_future_suffix(tmp_path,monkeypatch):
    context,old=fixture(np.array([0,30,180,210]));old.arena.close();del old
    c=spooled(context,tmp_path);a=c.arena;empty=a.empty();P=len(context.entries[0])
    ids=np.zeros((P,2,8),np.uint64);costs=np.full(ids.shape,np.inf)
    ids[:,0,0]=empty.handle;costs[:,0,0]=0.
    with c.open_origin(0,ids,costs) as bank:
        base,allowed=context.arcs.camera_alternatives(0,2)
        with c.compile_hold(0,2,base,allowed) as hold:
            handled,values=c.long_candidates(hold,bank,base,allowed,costs,np.zeros(costs.shape,np.int64),
                np.zeros(base.shape[:2]),same_episode=True,width=max(base.shape[:2]))
            before=(c.camera_sample_calls,c.projection_batches,c.encoded_frames)
            def forbidden(*args,**kwargs):raise AssertionError('prepared current tail was recomputed after expansion')
            monkeypatch.setattr(type(hold.inputs),'sample',forbidden)
            monkeypatch.setattr(context,'_intent_projection',forbidden)
            monkeypatch.setattr(context.model,'encode',forbidden)
            monkeypatch.setattr('crosscut2.compiled_played_context.pack',forbidden)
            count=0
            for p,q,code in zip(*np.nonzero(handled)):
                with hold.current(int(p),int(q),int(code),part='suffix') as current:
                    assert not current.keys.flags.writeable and not current.encoded.flags.writeable
                    root=a.append(empty,current.lease);a.release(root);count+=1
            assert count>0 and c.prepared_suffix_reuses==count
            assert (c.camera_sample_calls,c.projection_batches,c.encoded_frames)==before
    a.release(empty);a.close()


def test_64_virtual_partitions_share_exact_canonical_suffix_even_with_forced_digest_collision(tmp_path):
    need=capacity_requirements(embedding_width=2,history_frames=120,capacity=8)
    a=SpooledPlaybackArena(namespace=('finite-partitions-only',),embedding_width=2,history_frames=120,root_capacity=8,
        workspace=PlaybackWorkspace(2**20),directory=tmp_path,maximum_spool_bytes=need['backing_bytes'])
    a.store._digest=lambda keys:bytes(16);empty=a.empty()
    with descriptor(a,0,157) as current:expected=a.append(empty,current)
    count=0;cuts=(5,6,7,119,120,121)
    for length in range(7):
        for split in combinations(cuts,length):
            parent=a.retain(empty);first=0
            for final in (*split,157):
                with a.decode([parent],end=first) as bank:
                    a.bind_origin(np.array([parent.handle],np.uint64),bank)
                    try:
                        with descriptor(a,first,final) as current:child=a.append(parent,current)
                    finally:a.unbind_origin()
                a.release(parent);parent=child;first=final
            assert parent==expected;a.release(parent);count+=1
    assert count==64;a.release(expected);a.release(empty);a.close()


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_4096_improving_spooled_beam_offers_keep_only_eight_roots_and_original_alias_rank(tmp_path,kernel):
    need=capacity_requirements(embedding_width=2,history_frames=12,capacity=16)
    a=SpooledPlaybackArena(namespace=('finite-improving-numeric-only',),embedding_width=2,history_frames=12,root_capacity=16,
        workspace=PlaybackWorkspace(2**20),directory=tmp_path,maximum_spool_bytes=need['backing_bytes'],kernel=kernel)
    empty=a.empty();beam=SpooledBeam(a);state=fields();backing=a.store.maximum_bytes
    for variant in range(4096):
        with descriptor(a,0,12,lambda keys,v=variant:keys.__setitem__((-1,2),v+100)) as current:
            beam.offer(state,(-float(variant),0,variant,variant%6,0.,.1),empty.handle,current)
    assert a.store.live==9 and a.store.peak_live<=10 and a.store.path.stat().st_size==backing
    assert beam.discarded==4096-8
    with descriptor(a,0,12,lambda keys:keys.__setitem__((-1,2),4095+100)) as current:
        beam.offer(state,(-4096.,0,3,5,.025,.1),empty.handle,current)
    at=int(np.argmin(state[0]));assert state[2][at]==3 and state[3][at]==5 and state[4][at]==.025
    assert a.store.live==9
    for handle in state[6]:a.release(SuffixRoot(a.owner,int(handle)))
    a.release(empty);a.close()
