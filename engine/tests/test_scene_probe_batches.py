"""Work ordering only, not physical observations or editorial admission."""
from types import SimpleNamespace as NS

import pytest

from crosscut import group_scene_source_owner as scene


def prepared(count=35):
    return tuple((tuple(NS(first_output_frame=30*(index+slot),
        end_output_frame=30*(index+slot+1)) for slot in range(3)),(),
        dict(observed_wide_repair_priority=((index*17)%23)/23)) for index in range(count))


def test_remaining_unknown_windows_survive_the_initial_fixed_probe_batch():
    values=prepared()
    batches=scene.fixed_window_probe_batches(values,batch_size=12)
    assert tuple(map(len,batches))==(12,12,11)
    assert batches[0]==scene.fixed_window_probe_order(values,12)
    ordered=[index for batch in batches for index,item in batch]
    assert len(ordered)==len(set(ordered))==35
    assert set(ordered)==set(range(35))
    assert all(item is values[index] for batch in batches for index,item in batch)


def test_small_and_empty_inventories_do_not_create_a_retry_or_duplicate_window():
    assert scene.fixed_window_probe_batches((),batch_size=12)==()
    values=prepared(2);batches=scene.fixed_window_probe_batches(values,batch_size=12)
    assert len(batches)==1 and len(batches[0])==2


@pytest.mark.parametrize('size',[0,-1,True,1.5])
def test_probe_batch_size_is_explicit_and_integral(size):
    with pytest.raises(ValueError):scene.fixed_window_probe_batches(prepared(),batch_size=size)
