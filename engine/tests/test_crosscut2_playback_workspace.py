import gc
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
