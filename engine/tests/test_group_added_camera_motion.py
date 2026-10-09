from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from crosscut.group_added_camera_motion import measure_added_camera_motion
from crosscut.render import (EditorialTransformKeyframe as Key,EditorialTransformTrack,
    RenderPlan,RenderSegment,RenderTransition,NormalizedCrop,PanCropKeyframe,PanCropTrack)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock


def plan(keys,*,fps=30.,pan=None):
    return RenderPlan((Path('unused-a.mp4'),Path('unused-b.mp4')),
        (RenderSegment(0,0.,1.,0.,pan_crop=pan,editorial_transform=EditorialTransformTrack(tuple(keys))),
         RenderSegment(1,0.,1.,1.)),0,'16:9',Path('unused-output.mp4'),
        fps=fps,transitions=(RenderTransition(),))


def measure(p,*,cuts=None):
    clocks={i:VerifiedRenderFrameClock(Fraction(30),300,str(i)*64) for i in (0,1)}
    return measure_added_camera_motion(p,segment_indices=(0,1),source_clocks=clocks,
        scene_cut_times=cuts or {0:(),1:()})


def test_constant_zoom_pan_roll_is_not_a_dizzy_camera_and_edit_jump_is_separate():
    m=measure(plan([Key(0.,1.6,.55,.48,2.)]))
    assert m.ranking_cost==0 and m.measured_output_ticks==60
    assert m.excluded_native_or_edit_cut_steps==1
    assert not m.to_dict()['source_or_performer_motion_penalized']
    assert not m.to_dict()['quality_approved']


def test_hidden_middle_zoom_and_pan_cannot_be_missed_by_endpoint_sampling():
    steady=measure(plan([Key(0.,1.6),Key(29/30.,1.6)]))
    moving=measure(plan([Key(0.,1.6),Key(.5,1.9,.6,.55),Key(29/30.,1.6)]))
    assert moving.ranking_cost>steady.ranking_cost
    assert moving.maximum_control_velocity>0 and moving.maximum_control_acceleration>0
    assert moving.controls_sha256!=steady.controls_sha256


def test_zoom_does_not_invent_pan_and_faster_roll_reversal_is_costlier():
    zoom=measure(plan([Key(0.,1.2),Key(29/30.,1.8)]))
    assert zoom.maximum_control_velocity==pytest.approx(30*__import__('math').log((1.2+.6/29)/1.2))
    slow=measure(plan([Key(0.,1.6,roll_degrees=-4.),Key(29/30.,1.6,roll_degrees=4.)]))
    fast=measure(plan([Key(0.,1.6,roll_degrees=-4.),Key(.4,1.6,roll_degrees=4.),
        Key(29/30.,1.6,roll_degrees=4.)]))
    assert fast.ranking_cost>slow.ranking_cost


def test_base_tracking_pan_is_measured_even_without_editorial_pan():
    pan=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.1,.1,.7,.7)),
        PanCropKeyframe(29/30.,NormalizedCrop(.3,.1,.7,.7))))
    assert measure(plan([Key(0.,1.)],pan=pan)).ranking_cost>0


def test_native_cut_reset_not_smeared_into_camera_velocity():
    p=plan([Key(0.,1.4),Key(14/30.,1.4),Key(15/30.,1.8),Key(29/30.,1.8)])
    continuous=measure(p)
    native_cut=measure(p,cuts={0:(.5,),1:()})
    assert native_cut.ranking_cost==0 and continuous.ranking_cost>0
    assert native_cut.excluded_native_or_edit_cut_steps==2


def test_actual_output_clock_and_interior_controls_bind_cost_identity():
    p=plan([Key(0.,1.4),Key(29/30.,1.8)])
    first=measure(p);second=measure(replace(p,fps=60.))
    assert second.measured_output_ticks==120
    assert first.controls_sha256!=second.controls_sha256
