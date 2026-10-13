import numpy as np
import pytest

from crosscut2.stage_mix import camera_track, choose_programme, native_event_cost, realize_programme, resume_plan, semi_markov
from crosscut2.paired_geometry import Camera


def test_dp_selects_entire_programme_not_retained_local_patches():
    nodes = np.arange(0, 121, 10)
    unary = np.zeros((2, 12))
    unary[0, 6:] = 8; unary[1, :6] = 8
    seams = np.ones((13, 2, 2))
    runs, score = semi_markov(nodes, fps=10, unary=unary, seam_cost=seams)
    assert runs[0][0] == 0 and runs[-1][1] == 12
    assert runs[0][2] == 0 and runs[-1][2] == 1
    assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))
    assert np.isfinite(score)


def test_missing_geometry_remains_finite_and_can_cover_sparse_sources():
    runs, _ = semi_markov(np.arange(0, 61, 10), fps=10,
        unary=np.full((2, 6), 1.2), seam_cost=np.full((7, 2, 2), 4.))
    assert runs[0][0] == 0 and runs[-1][1] == 6


def test_whole_hold_camera_does_not_bounce_back_to_neutral():
    a, b = Camera(1.2, .48, .49), Camera(1.3, .52, .51)
    track = camera_track(a, b, 3.)
    mid = track.evaluate(1.5)
    assert mid.zoom == pytest.approx(1.25)
    assert mid.center_x == pytest.approx(.5)
    assert track.evaluate(0).zoom == a.zoom
    assert track.evaluate(3).zoom == b.zoom


def test_one_source_has_full_coverage_without_artificial_editorial_cuts():
    runs, score = semi_markov(np.arange(0, 201, 10), fps=10,
        unary=np.ones((1, 20)), seam_cost=np.zeros((21, 1, 1)))
    assert runs == ((0, 20, 0),)
    assert score == pytest.approx(20.)


def test_resume_uses_exact_saved_plan_and_refuses_changed_sources(tmp_path):
    from crosscut.render import AspectRatio, RenderPlan, RenderSegment
    from crosscut2.grid_perception import file_identity
    source = tmp_path / 'source.mp4'
    source.touch()
    output = tmp_path / 'output.mp4'
    original = RenderPlan((source,), (RenderSegment(0, 0., 6., 0.),), 0,
        AspectRatio.LANDSCAPE, output)
    record = dict(render_plan=original.to_json(), sparse_perception=[
        dict(source=0, source_file_identity=file_identity(source))])
    resumed = resume_plan(record, original, output)
    assert resumed.to_json() == original.to_json()
    with pytest.raises(ValueError, match='differs'):
        resume_plan(record, original, tmp_path / 'another.mp4')
    source.write_bytes(b'changed')
    with pytest.raises(OSError, match='source changed'):
        resume_plan(record, original, output)


def test_native_cut_cost_distinguishes_a_boundary_from_a_tiny_retained_fragment():
    assert native_event_cost(20, 40, np.array([20, 40]), 10) == 0.
    assert native_event_cost(20, 40, np.array([30]), 10) == pytest.approx(.25)
    assert native_event_cost(20, 40, np.array([21]), 10) > 3.
    assert native_event_cost(20, 40, np.array([39]), 10) > 3.


def test_dp_can_avoid_native_cut_bursts_without_reducing_editorial_cuts():
    nodes = np.arange(0, 61, 10)
    unary = np.array([[5, 5, 0, 0, 5, 5], [0, 0, 0, 0, 0, 0], [5, 5, .1, .1, 5, 5]])
    seams = np.full((7, 3, 3), .1)
    original, _ = semi_markov(nodes, fps=10, unary=unary, seam_cost=seams)
    adjusted, _ = semi_markov(nodes, fps=10, unary=unary, seam_cost=seams,
        source_events=(np.array([21]), np.array([]), np.array([])))
    assert [s for _, _, s in original] == [1, 0, 1]
    assert [s for _, _, s in adjusted] == [1, 2, 1]


def test_native_cut_clock_mismatch_is_not_silently_ignored():
    with pytest.raises(ValueError, match='output-clock'):
        semi_markov(np.arange(0, 61, 10), fps=10, unary=np.zeros((2, 6)),
            seam_cost=np.zeros((7, 2, 2)), source_events=(np.array([61]), np.array([])))


def test_three_frame_native_burst_inside_a_long_hold_is_not_free():
    isolated = native_event_cost(0, 150, np.array([60]), 30)
    burst = native_event_cost(0, 150, np.array([60, 63]), 30)
    assert isolated == pytest.approx(.25)
    assert burst > 3.


def test_realized_pose_recipe_is_finite_json_with_real_float32_bank(tmp_path):
    import json
    from crosscut.render import AspectRatio, RenderPlan, RenderSegment
    from crosscut2.paired_geometry import match_heads
    heads = np.zeros((8, 4), np.float32)
    heads[0] = [.42, .35, .14, .95]
    right = heads.copy(); right[0] = [.49, .36, .15, .95]
    vectors = np.zeros((3, 128), np.float32); vectors[:, 0] = 1
    pose = np.zeros((2, 3, 8, 3), np.float32)
    match = match_heads(heads, right, vectors, vectors, left_pose=pose, right_pose=pose)
    class IdentityWarp:
        def master_to_source(self, value): return value
    original = RenderPlan((tmp_path / 'a.mp4', tmp_path / 'b.mp4'),
        (RenderSegment(0, 0., 6., 0.),), 0, AspectRatio.LANDSCAPE, tmp_path / 'old.mp4')
    plan, seams = realize_programme(original, (IdentityWarp(), IdentityWarp()),
        np.array([0, 90, 180]), ((0, 1, 0), (1, 2, 1)), {(1, 0, 1): match},
        output=tmp_path / 'new.mp4')
    restored = json.loads(json.dumps(dict(render_plan=plan.to_json(), seams=seams), allow_nan=False))
    assert restored['seams'][0]['conditional_pose_loss'] is None
    assert restored['seams'][0]['comparable_action_joints'] == 0


def test_all_unknown_action_keeps_complete_graph_ranking_and_rhythm():
    nodes = np.arange(0, 121, 10)
    rng = np.random.default_rng(541)
    banks = []
    for source in range(3):
        heads = np.zeros((len(nodes), 2, 8, 4), np.float32)
        heads[:, :, :3, 0] = [.3, .5, .7]
        heads[:, :, :3, 1] = .3
        heads[:, :, :3, 2] = [.10 + source * .01, .09, .08]
        heads[:, :, :3, 3] = .95
        vectors = rng.normal(size=(len(nodes), 2, 3, 128)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        banks.append(dict(heads=heads, vectors=vectors, zoom_cap=1.35))
    disabled = choose_programme(nodes, banks, fps=10, check=lambda: None)
    enabled = choose_programme(nodes, banks, fps=10, check=lambda: None,
        poses=[np.zeros((len(nodes), 4, 3, 8, 3), np.float32) for _ in banks])
    assert enabled[0] == disabled[0]
    assert enabled[2] == disabled[2]
    for key, base in disabled[1].items():
        refined = enabled[1][key]
        assert refined.cost == base.cost
        assert (refined.role, refined.left, refined.right) == (base.role, base.left, base.right)
