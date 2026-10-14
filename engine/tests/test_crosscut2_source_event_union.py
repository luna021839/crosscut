import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from crosscut2.source_event_union import merge_frames,nearest_indices,combine


def test_independent_transnet_cut_survives_adaptive_miss_without_duplicate_micro_shots():
    assert merge_frames([101,400],[100,856],count=1000).tolist()==[100,400,856]
    assert merge_frames([],[],count=100).size==0
    with pytest.raises(ValueError):merge_frames([10,10],[],count=100)


def test_first_incoming_mapping_uses_nearest_native_sampler_not_rounded_timestamp():
    warp=SimpleNamespace(master_to_source=lambda value:value+.49/30.)
    native=nearest_indices(warp,master_start=0.,fps=30.,final_frame=100,native_fps=30.,native_count=101)
    assert np.searchsorted(native,40)==40
    warp=SimpleNamespace(master_to_source=lambda value:value+.51/30.)
    native=nearest_indices(warp,master_start=0.,fps=30.,final_frame=100,native_fps=30.,native_count=101)
    assert np.searchsorted(native,40)==39


def fixture(tmp_path):
    path=tmp_path/'source.mp4';path.write_bytes(b'exact-source-pixels')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    record=dict(source=str(path),source_index=0,detection_policy=dict(contract='transnetv2_source_native_first_incoming_frame_v1'),
        persistent_scalar_cache=dict(source_sha256=sha),changes=[dict(frame_index=20)])
    return dict(sources=[path],manifest=dict(scene_analysis=[record]),
        census=[dict(fps=30.,frame_count=100,frames=[50])],warps=[SimpleNamespace(master_to_source=lambda x:x)],
        master_start=0.,fps=30.,final_frame=90)


def test_digest_bound_cut_union_needs_no_source_decode_or_neural_producer(tmp_path):
    args=fixture(tmp_path);events,bound,receipts=combine(**args)
    assert events[0].tolist()==[20,50] and bound[0]['frames']==[20,50]
    assert receipts[0]['new_neural_calls']==receipts[0]['new_source_decoded_frames']==0
    assert receipts[0]['source_sha256_verified']
    args['sources'][0].write_bytes(b'changed-same-path')
    with pytest.raises(OSError,match='current source pixels'):combine(**args)


def test_unknown_legacy_frame_domain_is_not_silently_treated_as_native(tmp_path):
    args=fixture(tmp_path);args['manifest']['scene_analysis'][0]['detection_policy']['contract']='timestamps-only'
    with pytest.raises(ValueError,match='unrecognized'):combine(**args)
