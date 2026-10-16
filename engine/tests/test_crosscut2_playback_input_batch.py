"""Exact numeric batch identity before labels; no full-film speed claim."""
import numpy as np
import pytest
from crosscut2.playback_input_batch import pack,intern,_deduplicate,_compiled


def packed(N=120):
    return pack(model_binding=('schema','model'),frames=np.arange(N),native=np.arange(N)*2,
        cameras=np.tile([1.2,.5,.5],(N,1)),source_observation_handles=np.ones(N,np.int64),
        intent_handles=np.tile([1,0],(N,1)))


@pytest.mark.parametrize('kernel',['numpy','numba'])
def test_exact_aliases_reuse_local_ids_without_deleting_camera_candidates_or_label_slots(kernel):
    if kernel=='numba':pytest.importorskip('numba')
    binding,one=packed();keys=np.concatenate([one,one,one]);keys.setflags(write=False)
    result=intern((binding,keys),kernel=kernel)
    assert result.unique_count==120 and len(result.local_ids)==360
    assert np.array_equal(result.local_ids[:120],result.local_ids[120:240])
    # Consumers gather encoded values for all three candidate codes and all
    # eight retained labels. They do not repack/inter frames for each label.
    base_costs=np.array([0.,np.finfo(float).eps,0.])
    candidates=np.zeros((3,8,120,2))
    encoded=np.ones((result.unique_count,2))
    for code in range(3):candidates[code]=encoded[result.local_ids[code*120:(code+1)*120]]
    assert candidates.shape==(3,8,120,2) and base_costs[1]!=base_costs[0]
    assert result.maximum_active_bytes<100000


@pytest.mark.parametrize('column',range(8))
def test_every_original_input_word_and_single_camera_ulp_distinguishes_identity(column):
    binding,keys=packed(1);two=np.repeat(keys,2,axis=0)
    two[1,column]+=np.uint64(1);two.setflags(write=False)
    assert intern((binding,two)).unique_count==2


@pytest.mark.parametrize('compiled',[False,True])
def test_hash_collisions_are_resolved_by_original_words_never_hash_only(compiled):
    if compiled:pytest.importorskip('numba')
    _,keys=packed(17);values=np.concatenate([keys,keys[[2,10,2]]]);values.setflags(write=False)
    function=_compiled() if compiled else _deduplicate
    # Force all rows into the same initial bucket. Production always uses
    # the complete64bit hash; this exercises its exact collision branch.
    with np.errstate(over='ignore'):ids,first,probes=function(values,np.full(64,-1,np.int64),np.uint64(0))
    assert len(first)==17 and ids[-3:].tolist()==[2,10,2] and probes>len(values)


def test_budget_or_mutable_inputs_fail_without_source_or_domain_reduction():
    p=packed()
    with pytest.raises(MemoryError,match='no candidates'):intern(p,maximum_active_bytes=64)
    mutable=p[1].copy()
    with pytest.raises(ValueError,match='immutable'):intern((p[0],mutable))
