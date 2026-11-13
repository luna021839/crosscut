from dataclasses import replace

import numpy as np
import pytest

from crosscut2.subject_cut_moments import search
from crosscut2.subject_action_match import clock
from crosscut2.subject_window_bank import SubjectWindow


def window(source, cut=()):
    ordinals = np.arange(86, 115)
    n = len(ordinals)
    body = np.tile([[.4, .4, .9], [.6, .4, .9], [.35, .5, .9], [.65, .5, .9],
                    [.3, .55, .9], [.7, .55, .9], [.43, .7, .9], [.57, .7, .9]], (n, 1, 1))
    body[:, 4:6, 1] += .08 * np.sin(2 * np.pi * (ordinals - 100) / 6)[:, None]
    vectors = np.zeros((n, 128)); vectors[:, 0] = 1.
    return SubjectWindow(source, str(source), 'subject' + str(source), ordinals,
                         np.tile([.5, .3, .2, .99], (n, 1)), vectors, body,
                         np.ones((n, 8), bool), np.ones(n, bool),
                         tuple(np.zeros((0, 4)) for _ in ordinals), cut)


def hypothesis(w, phase=0):
    return clock(source=w.source, binding=w.binding, relative_frames=np.arange(-12, 13),
                 source_start=100 / 30, playback_rate=1., output_fps=30.,
                 native_fps=30., native_count=500, phase_frames=phase)


def run(a, b, **kwargs):
    return search(a, b, hypothesis(a), hypothesis(b), candidate_frames=[-6, 0, 6],
                  outgoing_cap=2., incoming_cap=2., **kwargs)


def test_cut_moment_moves_both_sources_without_retiming_either_one():
    a, b = window(0), window(1)
    result = run(a, b)
    for row in result['supported']:
        center = row['cut_offset_frames']
        assert row['native_A'][6] == 100 + center
        assert row['native_B'][6] == 100 + center
        assert row['both_music_clocks_moved_together'] and not row['source_retimed']
        assert row['clock_hypothesis_frames'] == 0


def test_preserved_readable_camera_pair_competes_with_matched_crop_at_the_same_moment():
    a,b=window(0),window(1)
    heads=b.heads.copy();heads[:,2]*=.75;b=replace(b,heads=heads)
    original=np.array([[1.,.5,.5],[1.,.5,.5]])
    result=search(a,b,hypothesis(a),hypothesis(b),candidate_frames=[0],
        outgoing_cap=2.,incoming_cap=2.,preserved_cameras=original)
    assert len(result['supported'])==2
    by_kind={r['camera_proposal_kind']:r for r in result['supported']}
    np.testing.assert_array_equal(by_kind['preserved']['outgoing_camera'],original[0])
    np.testing.assert_array_equal(by_kind['preserved']['incoming_camera'],original[1])
    assert by_kind['joint']['incoming_camera']!=original[1].tolist()
    for row in result['supported']:
        assert row['seam_action_supported'] and row['native_A'][6]==100 and row['native_B'][6]==100
        assert row['both_music_clocks_moved_together'] and not row['source_retimed']
    np.testing.assert_array_equal(original,[[1.,.5,.5],[1.,.5,.5]])


def test_duplicate_physical_camera_pair_is_not_a_new_cut_moment_candidate():
    a,b=window(0),window(1)
    result=search(a,b,hypothesis(a),hypothesis(b),candidate_frames=[0],
        outgoing_cap=2.,incoming_cap=2.,preserved_cameras=np.array([[1.,.5,.5]]*2))
    assert len(result['candidates'])==1


def test_good_window_average_cannot_hide_wrist_error_at_actual_seam():
    a, b = window(0), window(1)
    body = b.body.copy()
    body[np.isin(b.ordinals, [99, 100]), 4, 0] += .25
    b = replace(b, body=body)
    result = run(a, b)
    by_moment = {r['cut_offset_frames']: r for r in result['candidates']}
    assert by_moment[0]['seam_pose_residual'] > .5
    assert result['supported'][0]['cut_offset_frames'] != 0
    assert by_moment[0]['cut_moment_score'] > by_moment[6]['cut_moment_score']


def test_shoulder_only_observations_never_become_a_good_arm_connection():
    a, b = window(0), window(1)
    support = b.support.copy()
    support[:, 2:] = False
    result = run(a, replace(b, support=support))
    assert not result['supported']
    assert all(r['reason'] == 'no_complete_measured_arm_at_seam' for r in result['candidates'])


def test_configuration_has_priority_over_lower_position_or_velocity_sum():
    a, b = window(0), window(1)
    body = b.body.copy()
    # At the anchor, changing camera/body scale preserves arm directions.
    at_anchor = np.isin(b.ordinals, [99, 100])
    root = body[at_anchor, :2, :2].mean(axis=1)
    body[at_anchor, :, :2] = root[:, None] + 1.35 * (body[at_anchor, :, :2] - root[:, None])
    # A very small positional change near the next seam still rotates an
    # actual forearm. It must not beat the matching articulated configuration.
    at_next = np.isin(b.ordinals, [105, 106])
    body[at_next, 4, 0] += .04
    body[at_next, 5, 0] -= .04
    result = run(a, replace(b, body=body))
    rows = {r['cut_offset_frames']: r for r in result['candidates']}
    assert rows[0]['seam_arm_direction_residual'] < rows[6]['seam_arm_direction_residual']
    assert rows[0]['cut_moment_score'] > rows[6]['cut_moment_score']
    assert result['supported'].index(rows[0]) < result['supported'].index(rows[6])


def test_real_synchronous_choreography_reversal_is_not_a_motion_failure():
    a, b = window(0), window(1)
    result = run(a, b)
    assert len(result['supported']) == 3
    assert all(r['seam_motion_residual'] < 1e-12 for r in result['supported'])


def test_new_cut_moment_checks_both_subjects_not_only_incoming_attention():
    a,b=window(0),window(1)
    others=list(a.competitors);others[int(np.flatnonzero(a.ordinals==100)[0])]=np.array([[.5,.3,.5,.99]])
    a=replace(a,competitors=tuple(others))
    result=search(a,b,hypothesis(a),hypothesis(b),candidate_frames=[0,1],outgoing_cap=2.,incoming_cap=2.)
    rows={v['cut_offset_frames']:v for v in result['candidates']}
    assert rows[0]['status']=='SUPPORTED_CANDIDATE'
    assert rows[1]['outgoing_target_attention_ratio']==pytest.approx(.16)
    assert rows[1]['target_attention_ratio']==1.
    assert rows[1]['status']=='TARGET_OCCLUDED_OR_NOT_ATTENDED'
    assert rows[1]['cut_moment_score'] is None
    assert [v['cut_offset_frames'] for v in result['supported']]==[0]


def test_original_camera_cut_inside_context_is_not_a_subject_continuation():
    a, b = window(0), window(1, cut=(100,))
    result = run(a, b)
    at_cut=next(r for r in result['candidates'] if r['cut_offset_frames'] == 0)
    assert at_cut['cut_moment_score'] is not None
    assert at_cut['common_source_shot_sample_mask']==[False]*6+[True]*7
    # The outgoing and incoming physical endpoints remain real observations;
    # only the unavailable unplayed pre-cut B camera was excluded.
    assert at_cut['seam_action_supported']


def test_cannot_turn_a_cut_moment_search_into_a_source_phase_shift():
    a, b = window(0), window(1)
    with pytest.raises(ValueError, match='zero-phase'):
        search(a, b, hypothesis(a), hypothesis(b, 1), candidate_frames=[0],
               outgoing_cap=2., incoming_cap=2.)


def test_rank_never_claims_viewing_or_user_approval_and_does_not_mutate_inputs():
    a, b = window(0), window(1)
    before = a.body.copy(), b.body.copy()
    result = run(a, b)
    np.testing.assert_array_equal(a.body, before[0])
    np.testing.assert_array_equal(b.body, before[1])
    assert not result['professional_quality_approved']
    assert not result['phase_search_performed']
    assert all(not r['preview_eligible'] for r in result['supported'])
