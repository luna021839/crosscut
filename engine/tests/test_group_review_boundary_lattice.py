import pytest
from crosscut.group_review_boundary_lattice import (review_boundary_options,inserted_flash_collisions,
    coalesce_review_mutable_intervals)


def test_mutable_lattice_union_coalesces_only_overlap_or_adjacency():
    spans=((29,41),(10,20),(18,30),(50,55),(55,60))
    result=coalesce_review_mutable_intervals(spans,output_frame_count=100)
    assert result==((10,41),(50,60))
    assert {t for a,b in result for t in range(a,b)}=={t for a,b in spans for t in range(a,b)}


def test_mutable_union_does_not_bridge_a_protected_gap():
    assert coalesce_review_mutable_intervals(((620,630),(750,765)),
        output_frame_count=4357)==((620,630),(750,765))


def event(tick,kind):return dict(first_incoming_output_frame=tick,kind=kind)


def test_actual_native_event_is_a_boundary_option_before_arbitrary_offset():
    assert review_boundary_options(745,output_frame_count=4357,observed_event_ticks=(744,))==(743,744,745)


def test_protected_internal_or_incident_boundary_does_not_move():
    for tick in (630,744,750):
        assert review_boundary_options(tick,output_frame_count=4357,
            observed_event_ticks=(744,),protected_intervals=((630,750),))==(tick,)
    assert 750 in review_boundary_options(751,output_frame_count=4357,
        observed_event_ticks=(750,),protected_intervals=((630,750),))


@pytest.mark.parametrize('a,b',[(744,745),(3571,3572),(4195,4197)])
def test_new_source_native_collision_cannot_make_an_unintended_flash(a,b):
    result=inserted_flash_collisions((event(a,'source_switch'),event(b,'native_camera_cut')),())
    assert result[0]['frame_count']==b-a


def test_unchanged_short_original_shot_is_not_subject_to_a_global_minimum():
    events=(event(20,'source_switch'),event(21,'native_camera_cut'))
    assert not inserted_flash_collisions(events,events)


def test_same_tick_native_and_authored_events_do_not_make_a_zero_frame_shot():
    assert not inserted_flash_collisions((event(20,'source_switch'),event(20,'native_camera_cut')),())


def test_new_two_frame_seek_is_not_a_continuous_original_shot():
    assert inserted_flash_collisions((event(20,'same_source_clock_jump'),
        event(22,'native_camera_cut')),(event(20,'native_camera_cut'),))


def test_retained_switch_tick_with_a_new_nearby_native_cut_is_still_a_new_flash():
    assert inserted_flash_collisions((event(20,'source_switch'),event(21,'native_camera_cut')),
        (event(20,'source_switch'),))
