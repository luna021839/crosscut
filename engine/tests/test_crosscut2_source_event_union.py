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


def test_confirmed_raw_cut_enters_the_same_union_and_prevents_new_micro_fragment(tmp_path,monkeypatch):
    from crosscut2 import source_cut_refinement
    from crosscut2.played_event_support import PlayedEventSupport
    from crosscut2.native_fragment_integrity import NativeFragmentIntegrity
    monkeypatch.setattr(source_cut_refinement,'observed_native_cuts',lambda sources,cache:[[53]])
    args=fixture(tmp_path)
    events,bound,receipts=combine(**args,supplemental_cut_cache=tmp_path/'confirmed')
    assert bound[0]['frames']==[20,50,53]
    assert bound[0]['contract']=='bound_source_cut_union_with_confirmed_output_observations_v2'
    assert bound[0]['supplemental_confirmed_native_cuts']==[53]
    assert receipts[0]['new_neural_calls']==receipts[0]['new_source_decoded_frames']==0
    integrity=NativeFragmentIntegrity(PlayedEventSupport(events,final=90,fps=30.),minimum_frames=12)
    assert not integrity.allowed(51,90,np.array([0]))[0]


def test_supplemental_observation_does_not_override_source_pixel_binding(tmp_path,monkeypatch):
    from crosscut2 import source_cut_refinement
    monkeypatch.setattr(source_cut_refinement,'observed_native_cuts',lambda sources,cache:[[53]])
    args=fixture(tmp_path);args['sources'][0].write_bytes(b'changed original')
    with pytest.raises(OSError,match='current source pixels'):combine(**args,supplemental_cut_cache=tmp_path/'confirmed')
