from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from crosscut.render_frame_schedule import sampler_frame_schedule
from crosscut2.member_stage_mix import Source, select
from crosscut2.visible_shot_duration import Model, exposure_tail_artifact, fit


class Warp:
    def master_to_source(self, t):
        return t


def inputs(count=2, frames=240):
    sources = [Source(Path(f'source-{i}.mp4'), 30., 1080, 1920, frames + 1) for i in range(count)]
    pose = [[.4, .4, .9], [.6, .4, .9], [.3, .5, .9], [.7, .5, .9],
            [.2, .6, .9], [.8, .6, .9], [.4, .8, .9], [.6, .8, .9]]
    rows = [{n: dict(face_bbox_xywh=[.45, .15, .1, .1], pose_screen_xyc=pose)
             for n in range(frames)} for _ in sources]
    model = Model(exposure_tail_artifact(fit([dict(reference_id='editor', split='train', input_sha256='a'*64,
                        cut_seconds=[0., 1., 3., 6., 10.])]), fps=30.))
    kwargs = dict(warps=[Warp() for _ in sources], master_start=0., duration_frames=frames, fps=30.,
                  beat_times=np.arange(0., frames/30., .5), output=Path('new.mp4'), duration_model=model)
    return sources, rows, kwargs


def test_fresh_complete_programme_is_deterministic_and_does_not_read_a_previous_movie():
    sources, rows, kwargs = inputs()
    a, receipt = select(sources, rows, **kwargs)
    b, second = select(sources, rows, **kwargs)
    assert a.to_json() == b.to_json() and receipt == second
    assert a.sources == tuple(s.path for s in sources)
    assert sum(round(s.duration * a.fps) for s in a.segments) == 240
    assert receipt['exact_native_endpoint_comparisons']
    assert not receipt['professional_quality_approved']
    assert not receipt['final_member_identity_all_frames_certified']
    assert all(a.source_index != b.source_index for a, b in zip(a.segments, a.segments[1:]))


def test_selected_camera_is_the_exact_scored_ten_frame_interpolation():
    sources, rows, kwargs = inputs()
    plan, receipt = select(sources, rows, **kwargs)
    for segment in plan.segments:
        assert segment.pan_crop is not None
        times = [k.time_seconds for k in segment.pan_crop.keyframes]
        assert times[0] == 0.
        assert times[-1] == pytest.approx(segment.duration - 1/plan.fps)
        count = round(segment.duration * plan.fps)
        actual = sampler_frame_schedule(source_start=segment.source_start,
            source_playback_rate=segment.source_playback_rate, native_fps=30.,
            output_fps=plan.fps, frame_count=count, source_frame_count=241)
        assert list(actual.source_frame_indices) == list(range(round(segment.master_start*30.),
            round(segment.master_start*30.) + count))
    assert receipt['affine_interior_native_differences_from_piecewise_warp'] == 0


def test_single_source_and_missing_faces_still_cover_the_whole_film():
    sources, rows, kwargs = inputs(count=1)
    plan, receipt = select(sources, [{}], **kwargs)
    assert len(plan.segments) == 1 and plan.segments[0].duration == 8.
    assert receipt['unknown_face_or_pose_not_a_source_veto']
    assert receipt['seams'] == []


def test_sparse_content_faces_are_not_relabelled_as_exact_cut_evidence():
    sources, rows, kwargs = inputs()
    rows = [{n: r for n, r in source.items() if n % 15 == 0} for source in rows]
    plan, receipt = select(sources, rows, **kwargs)
    assert sum(round(s.duration*30.) for s in plan.segments) == 240
    assert all(not r['exact_face_flanks'] for r in receipt['seams'])


def test_body_context_callback_uses_physical_state_cameras_without_source_exclusion():
    sources, rows, kwargs = inputs()
    rows[1] = {}
    calls = []

    def context(source, crops, unary, native):
        calls.append((source, crops.copy(), native.copy()))
        return np.zeros_like(unary) if source == 1 else np.full_like(unary, 10.)

    plan, receipt = select(sources, rows, content_loss=context, **kwargs)
    assert len(calls) == 8 and set(s.source_index for s in plan.segments) == {1}
    assert receipt['unknown_face_or_pose_not_a_source_veto']
    assert not receipt['final_member_identity_all_frames_certified']


def test_callbacks_cannot_return_nonfinite_or_wrong_length_loss():
    sources, rows, kwargs = inputs()
    for value in (np.full(240, np.nan), np.zeros(1)):
        with pytest.raises(ValueError, match='finite per-frame'):
            select(sources, rows, content_loss=lambda *args: value, **kwargs)


def test_cancel_callback_is_used_before_cache_processing_and_selection():
    sources, rows, kwargs = inputs()
    calls = []

    def cancelled():
        calls.append(1)
        raise InterruptedError('cancelled')

    with pytest.raises(InterruptedError, match='cancelled'):
        select(sources, rows, check=cancelled, **kwargs)
    assert calls == [1]


def test_unknown_configuration_experiment_is_explicit_and_off_by_default():
    sources, rows, kwargs = inputs()
    _, receipt = select(sources, rows, **kwargs)
    assert receipt['articulated_configuration'] is False
    _, changed = select(sources, rows, articulated_configuration=True, **kwargs)
    assert changed['articulated_configuration'] is True


@pytest.mark.parametrize('update', [dict(fps=0.), dict(fps=float('nan')), dict(width=0), dict(frame_count=True)])
def test_invalid_source_metadata_is_not_a_physical_clock(update):
    with pytest.raises(ValueError, match='physical source'):
        Source(**dict(dict(path=Path('s.mp4'), fps=30., width=1080, height=1920, frame_count=241), **update))


def test_nonphysical_source_times_and_native_observations_fail_before_a_plan():
    sources, rows, kwargs = inputs()
    with pytest.raises(ValueError, match='exceeds its physical'):
        select([replace(s, frame_count=20) for s in sources], rows, **kwargs)
    rows[0][999] = rows[0][0]
    with pytest.raises(ValueError, match='native indices'):
        select(sources, rows, **kwargs)


def test_a_short_reference_paced_film_needs_no_arbitrary_minimum_source_count():
    sources, rows, kwargs = inputs(count=1, frames=30)
    plan, _ = select(sources, rows, **kwargs)
    assert len(plan.segments) == 1 and plan.segments[0].duration == 1.
