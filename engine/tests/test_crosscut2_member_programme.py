from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crosscut.ive import AspectRatio
from crosscut.render import RenderPlan,RenderSegment
from crosscut.render_frame_schedule import sampler_frame_schedule
from crosscut2.member_stage_mix import Source
from crosscut2.member_programme import fit_cameras,replay_audio_warp
from crosscut2.member_body_context import ObservedBodyContext


def inputs():
    sources=[Source(Path(f'{s}.mp4'),30.,1080,1920,120) for s in range(2)]
    plan=RenderPlan(tuple(s.path for s in sources),(RenderSegment(0,0.,1.,0.),RenderSegment(1,1.,1.,1.)),
        0,AspectRatio.PORTRAIT,Path('out.mp4'),width=540,height=960,fps=30.)
    rows=[{n:dict(face_bbox_xywh=b,pose_screen_xyc=[]) for n in range(120)}
        for b in ((.35,.2,.1,.08),(.45,.25,.08,.07))]
    return sources,plan,rows


def test_paired_cameras_preserve_every_source_clock_and_cut():
    sources,plan,rows=inputs();new,receipt=fit_cameras(plan,sources,rows)
    assert [(s.source_index,s.source_start,s.source_duration,s.master_start,s.duration) for s in new.segments]==[
        (s.source_index,s.source_start,s.source_duration,s.master_start,s.duration) for s in plan.segments]
    assert new.output_frame_count==plan.output_frame_count
    pair=receipt['face_pair_fits'][0]
    assert pair['measured_face_pair'] and pair['after']['geometry']<pair['before']['geometry']
    assert pair['after']['measured_arm_joint_count']==0
    assert all(s.editorial_transform.covers_aspect_ratio(new.aspect_ratio) for s in new.segments)
    with pytest.raises(ValueError,match='before adding'):
        fit_cameras(new,sources,rows)


def test_unknown_flank_never_removes_a_source_or_holds_instead_of_cutting():
    sources,plan,rows=inputs();rows[1]={}
    new,receipt=fit_cameras(plan,sources,rows)
    assert len(new.segments)==2 and {s.source_index for s in new.segments}=={0,1}
    assert not receipt['face_pair_fits'][0]['measured_face_pair']


def test_replay_repairs_interior_ordinals_not_just_the_last_frame():
    sources,plan,_=inputs()
    segment=replace(plan.segments[0],source_start=.0165)
    plan=replace(plan,segments=(segment,),transitions=())
    warps=(SimpleNamespace(master_to_source=lambda t:.0165+1.001*t),None)
    new,receipt=replay_audio_warp(plan,sources,warps,master_start=0.)
    assert receipt['all_nominal_native_ordinals_checked']==30
    assert receipt['affine_quantization_repairs']
    actual=sampler_frame_schedule(source_start=new.segments[0].source_start,
        source_playback_rate=new.segments[0].source_playback_rate,native_fps=30.,output_fps=30.,
        frame_count=30,source_frame_count=120).source_frame_indices
    wanted=np.floor((.0165+1.001*np.arange(30)/30)*30+.5).astype(int)
    assert np.array_equal(actual,wanted)
    assert new.segments[0].master_start==0 and new.segments[0].duration==1.


def test_nonlinear_native_sequence_is_not_silently_certified_affine():
    sources,plan,_=inputs();plan=replace(plan,segments=plan.segments[:1],transitions=())
    warps=(SimpleNamespace(master_to_source=lambda t:t+(0.1 if t>.5 else 0.)),None)
    with pytest.raises(ValueError,match='piecewise'):
        replay_audio_warp(plan,sources,warps,master_start=0.)


def test_body_context_is_finite_and_closer_positive_face_takes_precedence():
    empty=dict(keypoints17=[],scores=[])
    # Keep the original observed empty tensor shape, not an absent query.
    empty['keypoints17']=np.empty((0,17,3))
    context=ObservedBodyContext([{10:(10,empty)},{}],[{10:False,11:True},{}],fps=10.)
    crops=np.tile([0.,0.,1.,1.],(20,1));unary=np.full(20,.9);natives=np.arange(20)
    result=context.content(0,crops,unary,natives)
    assert result[10]==4. and result[11]==.9
    assert np.array_equal(context.content(1,crops,unary,natives),unary)
    assert context.metadata['named_actor_identity_or_pose_credit_supplied'] is False
