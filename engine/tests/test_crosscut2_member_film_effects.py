from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from crosscut.ive import AspectRatio
from crosscut.render import (FACE_DISSOLVE_SMOOTHSTEP_ALPHA_CURVE, RenderPlan, RenderSegment,
    RenderTransition, TransitionKind, EditorialTransformTrack, EditorialTransformKeyframe)
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut2.member_film_effects import apply, evaluate, windows
from crosscut2.member_stage_mix import Source


def inputs():
    sources = [Source(Path(f'source-{s}.mp4'), 30., 1080, 1920, 300) for s in range(2)]
    camera = EditorialTransformTrack((EditorialTransformKeyframe(0.,1.,.5,.5),
        EditorialTransformKeyframe(1.9,1.,.5,.5)), reason='measured_paired_camera')
    plan = RenderPlan(tuple(s.path for s in sources), (RenderSegment(0, 0., 2., 0., editorial_transform=camera),
        RenderSegment(1, 2., 2., 2., editorial_transform=camera)), 0, AspectRatio.PORTRAIT, Path('new.mp4'), width=1080, height=1920, fps=30.)
    faces = [{n: dict(faces=[dict(member='가을', member_scores={'가을': .8}, bbox_xywh=[.45,.2,.1,.1])])
              for n in (60,64,67)} for _ in sources]
    geometry = [{n: dict(face_bbox_xywh=[.45,.2,.1,.1], pose_screen_xyc=[]) for n in (60,64,67)} for _ in sources]
    return plan, sources, faces, geometry


def test_effect_requests_reuse_current_native_reads_and_do_not_select_a_new_cut():
    plan, sources, faces, geometry = inputs()
    request = windows(plan, sources, frame_count=8, current_faces=faces)
    assert request['requested_native_rasters'] == 0 and request['existing_query_reuses'] == 6
    assert request['ticks'] == [0,4,7]
    assert [v['native'] for v in request['boundaries'][0]['samples']] == [60,60,64,64,67,67]
    assert not request['source_or_cut_sequence_changed']


def test_only_missing_pictures_are_requested_not_the_whole_source():
    plan, sources, faces, geometry = inputs()
    del faces[0][64]
    request = windows(plan, sources, frame_count=8, current_faces=faces)
    assert request['source_requests'] == [dict(source=0,natives=[64]),dict(source=1,natives=[])]
    assert request['requested_native_rasters'] == 1


def test_measured_effect_keeps_all_nominal_source_frames_crops_and_master_clocks():
    plan, sources, faces, geometry = inputs()
    request = windows(plan, sources, frame_count=8, current_faces=faces)
    report = evaluate(request['boundaries'][0], current_faces=faces, original_geometry=geometry,
                      sources=sources, member='가을', output_size=(1080,1920))
    revised, receipt = apply(plan, [report], frame_count=8, alpha_curve=FACE_DISSOLVE_SMOOTHSTEP_ALPHA_CURVE)
    assert tuple(replace(s,outgoing_overlap_transform=None) for s in revised.segments) == plan.segments
    assert receipt['actual_effect_count'] == 1 and receipt['effect_seconds'] == [2.]
    assert revised.transitions[0].frame_count == 8
    assert revised.transitions[0].kind is TransitionKind.FACE_DISSOLVE
    assert revised.transitions[0].face_dissolve_curve == FACE_DISSOLVE_SMOOTHSTEP_ALPHA_CURVE
    assert render_plan_from_json(revised.to_json()) == revised


def test_missing_current_faces_decline_only_the_effect_without_any_source_veto():
    plan, sources, faces, geometry = inputs()
    request = windows(plan, sources, frame_count=8, current_faces=faces)
    del faces[0][64]
    report = evaluate(request['boundaries'][0], current_faces=faces, original_geometry=geometry,
                      sources=sources, member='가을', output_size=(1080,1920))
    revised, receipt = apply(plan, [report], frame_count=8)
    assert not report['eligible'] and receipt['actual_effect_count'] == 0
    assert revised.segments == plan.segments and revised.sources == plan.sources


def test_user_effect_off_needs_no_new_selection_or_observation():
    plan, sources, faces, geometry = inputs()
    request = windows(plan, sources, frame_count=8, current_faces=faces)
    report = evaluate(request['boundaries'][0], current_faces=faces, original_geometry=geometry,
                      sources=sources, member='가을', output_size=(1080,1920))
    enabled, _ = apply(plan, [report], frame_count=8)
    disabled, receipt = apply(enabled, [], frame_count=8, enabled=False)
    assert disabled.segments == plan.segments and disabled.transitions == plan.transitions
    assert disabled.member_pose_dissolve_requested is False and receipt['actual_effect_count'] == 0


def test_short_incoming_hold_does_not_authorize_an_effect_through_the_next_cut():
    plan, sources, faces, geometry = inputs()
    short_camera = replace(plan.segments[1].editorial_transform,
                           keyframes=plan.segments[1].editorial_transform.keyframes[:1])
    plan = replace(plan, segments=(plan.segments[0], replace(plan.segments[1], duration=.2, editorial_transform=short_camera)))
    request = windows(plan, sources, frame_count=8, current_faces=faces)
    assert not request['boundaries'][0]['physical_overlap_available'] and not request['requested_native_rasters']


def test_effect_reports_cannot_be_reused_at_another_master_cut():
    plan, sources, faces, geometry = inputs()
    request = windows(plan, sources, frame_count=8, current_faces=faces)
    report = evaluate(request['boundaries'][0], current_faces=faces, original_geometry=geometry,
                      sources=sources, member='가을', output_size=(1080,1920))
    report['seconds'] += 1.
    with pytest.raises(ValueError, match='fixed programme boundary'):
        apply(plan, [report], frame_count=8)
