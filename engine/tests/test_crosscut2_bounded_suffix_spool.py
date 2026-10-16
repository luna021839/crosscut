"""Small backing fixtures; not original-domain runtime, fit or film proof."""
import gc
import os
import numpy as np
import pytest
from crosscut2.bounded_suffix_spool import BoundedSuffixSpool,capacity_requirements
from crosscut2.playback_workspace import PlaybackWorkspace


def store(tmp_path,*,capacity=8,H=12,D=2,workspace=None,**kwargs):
    return BoundedSuffixSpool(namespace=('numeric-model','exact-bound-source-and-observation-inputs'),
        embedding_width=D,history_frames=H,capacity=capacity,workspace=workspace or PlaybackWorkspace(2**20),
        directory=tmp_path,maximum_spool_bytes=capacity*H*(8+D)*8,**kwargs)


def payload(final,H=12,D=2,variant=0):
    ticks=np.arange(max(0,final-H),final,dtype=np.uint64);keys=np.zeros((len(ticks),8),np.uint64)
    keys[:,0]=ticks;keys[:,1]=ticks+100;keys[:,2]=1;keys[:,5]=np.float64(1.1+variant*.001).view(np.uint64)
    values=np.column_stack([np.sin(ticks.astype(float)/(i+7)) for i in range(D)])
    return keys,values


@pytest.mark.parametrize('final',[0,11,12,13,119,120,121,4357])
def test_exact_suffix_alias_roundtrip_and_fixed_backing_size(tmp_path,final):
    s=store(tmp_path);keys,values=payload(final);root=s.intern(keys,values,final=final)
    same=s.intern(keys.copy(),values.copy(),final=final);assert same==root
    with s.read(root) as read:
        np.testing.assert_array_equal(read.keys,keys)
        np.testing.assert_array_equal(read.values.view(np.uint64),values.view(np.uint64))
    receipt=s.receipt();assert receipt['backing_logical_bytes']==s.maximum_bytes
    assert receipt['live_roots']==1 and not receipt['append_only_candidate_history']
    path=s.path;s.release(same);s.release(root);s.close();assert not path.exists()


def test_full_digest_collision_compares_original_words_and_embedding_bits(tmp_path):
    s=store(tmp_path);s._digest=lambda keys:bytes(32);roots=[]
    for variant in range(8):
        keys,values=payload(12,variant=variant);roots.append(s.intern(keys,values,final=12))
    assert len(set(roots))==8
    keys,values=payload(12);bad=values.copy();bad[0,0]=-0. # first value was +0
    assert np.array_equal(bad,values)
    with pytest.raises(ValueError,match='different model embeddings'):s.intern(keys,bad,final=12)
    for root in reversed(roots):s.release(root)
    assert s.live==0;s.close()


def test_recycled_slot_checks_generation_and_old_weak_id_cannot_revive(tmp_path):
    s=store(tmp_path,capacity=1);keys,values=payload(12);old=s.intern(keys,values,final=12);s.release(old)
    new=s.intern(keys,values,final=12);assert old!=new
    with pytest.raises(ValueError,match='stale'):s.read(old)
    with pytest.raises(ValueError,match='live roots'):s.close()
    s.release(new);s.close()


def test_backing_is_fixed_under4096_replacements_and_zero_ref_hash_entries_leave(tmp_path):
    s=store(tmp_path,capacity=2);size=s.path.stat().st_size
    for i in range(4096):
        keys,values=payload(12,variant=i);root=s.intern(keys,values,final=12);s.release(root)
    assert s.path.stat().st_size==size and s.live==0 and not s.buckets.any()
    assert s.peak_live==1;s.close()


def test_all274_synthetic_future_ends_can_remain_owned_without_expanding_backing(tmp_path):
    # One root/end, NOT the original40port16label/camera domain. This is a
    # lifetime/storage primitive check and is not the next whole-DP gate.
    s=store(tmp_path,capacity=274,H=120,D=8,workspace=PlaybackWorkspace(2**20));roots=[]
    for final in np.linspace(158,4357,274).astype(int):
        keys,values=payload(int(final),H=120,D=8,variant=int(final))
        roots.append(s.intern(keys,values,final=int(final)))
    assert s.live==274 and s.path.stat().st_size==274*120*16*8
    for root in roots:
        with s.read(root) as data:assert len(data.keys)==120
        s.release(root)
    s.close()


def test_preallocation_memory_and_disk_refusals_create_no_backing_file(tmp_path):
    initial=list(tmp_path.iterdir())
    with pytest.raises(MemoryError,match='metadata/scratch'):
        store(tmp_path,capacity=1024,workspace=PlaybackWorkspace(2**16))
    assert list(tmp_path.iterdir())==initial
    with pytest.raises(MemoryError,match='disk budget'):
        BoundedSuffixSpool(namespace=('test',),embedding_width=2,history_frames=12,capacity=8,
            workspace=PlaybackWorkspace(2**20),directory=tmp_path,maximum_spool_bytes=1)
    assert list(tmp_path.iterdir())==initial


def test_failed_partial_write_publishes_no_root_or_lost_slot(tmp_path,monkeypatch):
    s=store(tmp_path);old=os.pwrite;calls=0
    def broken(fd,data,offset):
        nonlocal calls
        calls+=1
        if calls==2:raise OSError('finite interrupted partial write')
        return old(fd,data[:8],offset)
    keys,values=payload(12)
    with monkeypatch.context() as patch:
        patch.setattr(os,'pwrite',broken)
        with pytest.raises(OSError,match='interrupted'):s.intern(keys,values,final=12)
    assert s.live==0 and s.free_count==s.capacity and not s.buckets.any()
    root=s.intern(keys,values,final=12)
    with s.read(root) as read:np.testing.assert_array_equal(read.keys,keys)
    s.release(root);s.close()


def test_backing_corruption_is_not_returned_as_valid_embeddings(tmp_path):
    s=store(tmp_path);keys,values=payload(12);root=s.intern(keys,values,final=12)
    os.pwrite(s.fd,b'\xff',s.H*64)
    with pytest.raises(OSError,match='checksum'):s.read(root)
    s.release(root);s.close()


def test_cancellation_cannot_prevent_reference_cleanup_or_report(tmp_path):
    cancelled=False
    def check():
        if cancelled:raise TimeoutError('finite cancellation')
    s=store(tmp_path,check=check);keys,values=payload(12);root=s.intern(keys,values,final=12)
    cancelled=True
    with pytest.raises(TimeoutError):s.read(root)
    s.release(root);assert s.receipt()['live_roots']==0;s.close()


def test_loaded_small_views_and_internal_io_views_remain_charged_after_store_close(tmp_path):
    w=PlaybackWorkspace(2**20);s=store(tmp_path,workspace=w);keys,values=payload(12)
    root=s.intern(keys,values,final=12);loaded=s.read(root);outside=loaded.values[:1];io=s._io[:1]
    loaded.close();s.release(root);s.close();gc.collect();both=w.current_bytes
    del outside;gc.collect();assert w.current_bytes<both
    one=w.current_bytes;del io;gc.collect();assert w.current_bytes<one


def test_replaced_path_is_not_removed_as_our_temporary_backing(tmp_path):
    s=store(tmp_path);path=s.path;original=path.with_suffix('.saved');path.rename(original)
    path.write_bytes(b'user replacement, preserve')
    with pytest.raises(OSError,match='another file'):s.close()
    assert path.read_bytes()==b'user replacement, preserve' and original.is_file()
    # Both are tiny pytest-owned fixtures; no user filesystem cleanup here.


@pytest.mark.parametrize('end',[0,119,120,121,4357])
def test_complete_prefix_bank_decodes_original_words_and_embeddings_once_without_source_callbacks(tmp_path,end):
    s=store(tmp_path,H=120,D=8);roots=[]
    for i in range(8):
        keys,values=payload(end,H=120,D=8,variant=i);roots.append(s.intern(keys,values,final=end))
    ids=np.array([r.handle for r in roots],np.uint64)
    with s.decode_handles(ids,end=end) as bank:
        np.testing.assert_array_equal(bank.frames,np.arange(max(0,end-120),end))
        for i in range(8):
            expected=payload(end,H=120,D=8,variant=i)
            np.testing.assert_array_equal(bank.keys[i],expected[0])
            np.testing.assert_array_equal(bank.values[i].view(np.uint64),expected[1].view(np.uint64))
    for root in roots:s.release(root)
    s.close()


def test_prefix_bank_refuses_mixed_ends_without_losing_owned_payload(tmp_path):
    s=store(tmp_path);keys,values=payload(12);root=s.intern(keys,values,final=12)
    with pytest.raises(ValueError,match='different finalized ends'):
        s.decode_handles(np.array([root.handle],np.uint64),end=13)
    assert s.live==1;s.release(root);s.close()


def test_original_domain_sized_resource_arithmetic_requires_no_large_internal_file(tmp_path):
    before=list(tmp_path.iterdir());N,W,L,H,D=274,40,8,120,8
    # Conservative N*W root bound, not observed port counts or a retained
    # original whole graph. In particular this creates NO2.7GB test file.
    capacity=(N*W+W)*2*L+4
    need=capacity_requirements(embedding_width=D,history_frames=H,capacity=capacity)
    assert need['backing_bytes']==2703421440
    assert need['total_managed_index_and_io_bytes']<24*2**20
    assert list(tmp_path.iterdir())==before
    s=store(tmp_path);w=s.workspace
    assert w.current_bytes-w.bookkeeping_bytes==capacity_requirements(
        embedding_width=2,history_frames=12,capacity=8)['total_managed_index_and_io_bytes']
    s.close()
