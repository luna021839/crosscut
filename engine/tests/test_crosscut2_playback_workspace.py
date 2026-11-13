import gc
import sys
import weakref
import numpy as np
import pytest
from crosscut2.playback_workspace import PlaybackWorkspace
from crosscut2.compiled_played_context import _owned_inputs
from types import MappingProxyType


def test_frozen_mapping_proxy_arrays_and_shared_bases_are_counted_not_only_the_proxy_header():
    array=np.zeros((100,8));view=array[-1:]
    amount,buffers=_owned_inputs(MappingProxyType({'native':array,'tail':view,'again':array}))
    assert amount>=array.nbytes and {id(v) for v in buffers}=={id(array),id(view)}


def test_preallocated_producer_claim_is_transferred_to_real_buffer_lifetime():
    w=PlaybackWorkspace(2**20);baseline=w.current_bytes;claim=w.claim(100000,'producer-before-allocation')
    array=np.zeros((100,8));view=array[-1:]
    w.pin_group(claim,array.nbytes+128,[array]);after=w.current_bytes
    assert baseline<after<baseline+100000
    del array;gc.collect();assert w.current_bytes==after
    del view;gc.collect();assert w.current_bytes==baseline


def test_unknown_or_oversized_claim_cannot_be_freed_or_silently_expanded():
    w=PlaybackWorkspace(70000);key=w.claim(100,'current')
    with pytest.raises(ValueError,match='not live'):w.release(key+1)
    with pytest.raises(MemoryError,match='reservation'):w.pin_group(key,100000,[np.zeros(1)])
    w.release(key);assert w.current_bytes==w.bookkeeping_bytes


def test_many_frozen_source_arrays_share_one_real_lifetime_without_per_array_finalizers(monkeypatch):
    w=PlaybackWorkspace(4*2**20);arrays=tuple(np.zeros((2,)) for _ in range(5000))
    amount=sum(sys.getsizeof(value) for value in arrays);before=w.current_bytes
    def forbidden(*a,**k):raise AssertionError('one weak callback per frozen input is unnecessary')
    monkeypatch.setattr(weakref,'finalize',forbidden)
    w.pin_frozen_inputs(amount,arrays,owner='finite-prepared-source-inputs')
    assert w.current_bytes==before+amount+sys.getsizeof(arrays)+1024
    reference=weakref.ref(arrays[0]);del arrays;gc.collect()
    assert reference() is not None and not reference().flags.writeable
    assert w.receipt()['immutable_source_groups_pinned_for_workspace_lifetime']==1
    del w;gc.collect();assert reference() is None


def test_compact_numeric_source_exclusion_counts_new_views_without_recharging_the_frozen_parent():
    source=np.zeros((100,8));view=source[-3:]
    excluded=np.array([id(source)],np.uint64)
    amount,buffers=_owned_inputs({'tail':view},exclude=excluded)
    assert {id(v) for v in buffers}=={id(view)} and amount<source.nbytes
