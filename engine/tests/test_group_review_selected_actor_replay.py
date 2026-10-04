from copy import deepcopy
import pytest
from crosscut.group_review_selected_actor_replay import bind_review_selected_actor
from engine.tests.test_group_review_motion_seam import view


def actual(tick,head_id=7,box=(.4,.2,.2,.3),vector=None):
    result=view(tick,0.)
    from dataclasses import replace
    result['descriptor']=replace(result['descriptor'],frame_local_head_id=head_id)
    result['row']['native_observation'].update(native_pixel_sha256='a'*64,
        heads=[dict(observation_id=head_id,box_xywh=list(box))])
    if vector is not None:result['vector']=vector
    return result


def test_same_raster_selected_head_survives_detector_id_permutation():
    template=actual(4,7);fresh=actual(4,13)
    result=bind_review_selected_actor(template,[fresh])
    assert result['state']=='SAME_RASTER_OCCURRENCE'
    assert result['selected']['descriptor'].frame_local_head_id==13
    assert not result['frame_local_ids_treated_as_persistent']


def test_unmatched_is_not_recovered_as_the_largest_face():
    assert bind_review_selected_actor(None,[actual(4)])['selected'] is None
    template=actual(4)
    wrong=actual(5,vector=[0.,1.]+[0.]*126)
    assert bind_review_selected_actor(template,[wrong])['state']=='UNMATCHED'


def test_same_id_with_other_head_geometry_does_not_skip_appearance_match():
    template=actual(4)
    wrong=actual(4,box=(.1,.1,.1,.1),vector=[0.,1.]+[0.]*126)
    assert bind_review_selected_actor(template,[wrong])['selected'] is None


def test_appearance_competition_remains_ambiguous_not_best_face_success():
    template=actual(4);one=actual(5,7);two=deepcopy(one)
    from dataclasses import replace
    two['descriptor']=replace(two['descriptor'],frame_local_head_id=8)
    result=bind_review_selected_actor(template,[one,two])
    assert result['state']=='AMBIGUOUS' and result['selected'] is None


def test_same_native_address_cannot_reuse_changed_pixels_or_source():
    for key,value in (('native_pixel_sha256','b'*64),('request_source_index',3)):
        wrong=actual(4);wrong['row']['native_observation'][key]=value
        with pytest.raises(ValueError):bind_review_selected_actor(actual(4),[wrong])
