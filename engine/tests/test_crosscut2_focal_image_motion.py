from types import SimpleNamespace
import unittest
import numpy as np
import pytest

from crosscut2.focal_image_motion import TICKS, describe, matrix


def observation(x, *, side, camera=None, height=None):
    heads = np.zeros((14, 4))
    heads[:, 0], heads[:, 1], heads[:, 2], heads[:, 3] = x, .5, .1 if height is None else height, .99
    at = np.arange(14)
    window = SimpleNamespace(source=0, binding='actual-source', ordinals=at, heads=heads, associated=np.ones(14, bool))
    window.support=np.zeros((14,8),bool)
    clock = SimpleNamespace(source=0, binding='actual-source', output_frames=TICKS, native_indices=at, fps=30.)
    context = dict(window=window, clock=clock, at=at, same_shot=np.ones(14, bool), contiguous=np.ones(13, bool))
    return dict(side=side, camera_strip=np.tile([1., .5, .5], (7, 1)) if camera is None else camera,
                dense_subject_context=context)


class FocalImageMotionTests(unittest.TestCase):
    def score(self, a, b):
        return matrix([describe(a, fps=30.)], [describe(b, fps=30.)])

    def test_matching_face_endpoints_do_not_hide_opposed_image_motion(self):
        a = observation(.5 + .005 * (TICKS + 1), side=0)
        b = observation(.5 - .005 * TICKS, side=1)
        self.assertEqual(a['dense_subject_context']['window'].heads[6, 0], .5)
        self.assertEqual(b['dense_subject_context']['window'].heads[7, 0], .5)
        risk, known, count = self.score(a, b)
        self.assertTrue(known[0, 0])
        self.assertEqual(count[0, 0], 7)
        self.assertGreater(risk[0, 0], .5)

    def test_common_choreography_reversal_is_not_a_false_velocity_discontinuity(self):
        x = .5 + .005 * np.abs(TICKS)
        risk, known, _ = self.score(observation(x, side=0), observation(x, side=1))
        self.assertTrue(known[0, 0])
        self.assertAlmostEqual(risk[0, 0], 0.)

    def test_real_digital_tracking_cancels_native_image_pan_before_scoring(self):
        x = .5 + .003 * TICKS
        camera = np.c_[np.full(7, 1.2), x[:7], np.full(7, .5)]
        a = observation(x, side=0, camera=camera)
        b = observation(np.full(14, .5), side=1, camera=np.tile([1.2, .5, .5], (7, 1)))
        risk, known, _ = self.score(a, b)
        self.assertTrue(known[0, 0])
        self.assertAlmostEqual(risk[0, 0], 0., places=12)

    def test_measured_native_zoom_and_opposite_virtual_zoom_are_composed(self):
        height = .1 * 1.005 ** TICKS
        camera = np.c_[.12 / height[:7], np.full(7, .5), np.full(7, .5)]
        risk, known, _ = self.score(observation(np.full(14, .5), side=0, camera=camera, height=height),
                                   observation(np.full(14, .5), side=1, camera=np.tile([1.2, .5, .5], (7, 1))))
        self.assertTrue(known[0, 0])
        self.assertAlmostEqual(risk[0, 0], 0., places=12)

    def test_missing_association_is_unknown_not_a_stationary_face(self):
        a, b = observation(np.full(14, .5), side=0), observation(np.full(14, .5), side=1)
        a['dense_subject_context']['contiguous'][5] = False
        risk, known, count = self.score(a, b)
        self.assertFalse(known[0, 0])
        self.assertEqual(count[0, 0], 0)
        self.assertEqual(risk[0, 0], .5)

    def test_native_break_and_changed_clock_never_get_interpolated(self):
        a = observation(np.full(14, .5), side=0)
        b = observation(np.full(14, .5), side=1)
        a['dense_subject_context']['same_shot'][:] = False
        self.assertFalse(self.score(a, b)[1][0, 0])
        a['dense_subject_context']['clock'].native_indices = np.arange(1, 15)
        with self.assertRaises(ValueError):
            describe(a, fps=30.)


def test_actual_played_features_add_finite_motion_without_changing_cut_admission():
    from crosscut2.played_partial_pose import features, matrix as played_matrix
    from crosscut2.editorial_roles import apply, event_energy
    from test_crosscut2_played_partial_pose import observation as body_observation
    a,b=body_observation(0),body_observation(1)
    a.update(observation(.5+.005*(TICKS+1),side=0))
    b.update(observation(.5-.005*TICKS,side=1))
    base=played_matrix(features([a],fps=30.),features([b],fps=30.),details=True)
    actual=played_matrix(features([a],fps=30.,focal_motion=True),
                        features([b],fps=30.,focal_motion=True),details=True)
    np.testing.assert_array_equal(actual['allowed'],base['allowed'])
    assert actual['focal_image_motion_measured'][0,0]
    assert actual['focal_image_motion_rate'][0,0]>.5
    for strength in (0.,1.):
        old,new=apply(base,strength),apply(actual,strength)
        np.testing.assert_allclose(event_energy(new,.01,fps=30.,point_geometry=True)
            -event_energy(old,.01,fps=30.,point_geometry=True),.2*actual['focal_image_motion_rate'])


def test_missing_native_and_authored_motion_have_identical_point_cost():
    from crosscut2.boundary_arc_costs import ArcCosts
    from crosscut2.played_partial_pose import features, matrix as played_matrix
    from crosscut2.editorial_roles import event_energy
    from test_crosscut2_played_partial_pose import observation as body_observation
    a,b=body_observation(0),body_observation(1)
    for o in (a,b):o['body'][:]=0.;o['evidence'][:]=0
    actual=played_matrix(features([a],fps=30.,focal_motion=True),
                        features([b],fps=30.,focal_motion=True),details=True)
    assert actual['focal_image_motion_rate'][0,0]==.5
    assert not actual['focal_image_motion_measured'][0,0]
    arcs=ArcCosts.__new__(ArcCosts);arcs.point_geometry=True;arcs.focal_motion=True;arcs.fps=30.
    heads=np.zeros((1,2,8,4));heads[0,0,0]=a['head'];heads[0,1,0]=b['head']
    arcs.native=[dict(frames=np.array([10]),heads=heads)]
    cameras=np.array([[[1.,.5,.5]]])
    body,point=arcs.native_rates(0,np.array([10]),cameras,cameras,components=True)
    for width in (.01,.2):
        np.testing.assert_allclose(point+width*body,event_energy(actual,width,fps=30.,point_geometry=True))


def test_motion_is_explicit_and_cli_rejects_an_unused_flag(monkeypatch):
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior
    from crosscut2.stage_mix import main
    assert not PlayedPosePortPrior().artifact['composed_focal_image_motion']
    with pytest.raises(ValueError,match='actual played interval'):
        PlayedPosePortPrior(focal_motion=True)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,focal_motion=True)
    assert prior.artifact['focal_image_motion_changes_admission'] is False
    monkeypatch.setattr('sys.argv',['stage_mix','--source-plan','unused','--analysis','unused',
        '--cache','unused','--output','unused','--models','unused','--focal-image-motion'])
    with pytest.raises(ValueError,match='actual played interval'):main()


def test_real_selector_records_the_used_cost_and_replays_its_actual_objective():
    from crosscut2.played_pose_port_programme import PlayedPosePortPrior,select
    from test_crosscut2_pair_camera_options import counterexample
    _,_,_,_,arcs,_=counterexample(False)
    prior=PlayedPosePortPrior(editorial_opportunities=True,played_interval_utility=True,
        recycle_static_capacity=True,retain_pair_cameras=True,focal_motion=True)
    _,_,_,_,receipt,selected=select(arcs.nodes,[c.bank for c in arcs.cats],[c.body for c in arcs.cats],
        [c.style for c in arcs.cats],fps=30.,source_events=[[]]*4,body_states=None,
        native_ports=arcs.native,music=np.array([0.,1.,1.,0.]),prior=prior)
    assert receipt['event_preference']['fixed_connection_prior']['composed_focal_image_motion']
    assert receipt['frozen_selected_camera_arcs']
    assert receipt['port_pair_statistics']
    assert all(s['focal_image_motion_cost_used'] for s in receipt['port_pair_statistics'])
    for seam in selected.details.values():
        assert seam['focal_image_motion_cost_used']
        assert seam['focal_image_motion_rate']==.5
        assert not seam['focal_image_motion_measured']


def test_native_prefix_and_exact_replay_pay_the_same_unknown_motion():
    from crosscut2.boundary_arc_costs import ArcCosts
    from test_crosscut2_boundary_arc_costs import arc_fixture
    nodes,entry,exit,old,fragments=arc_fixture(True)
    arcs=ArcCosts(nodes,entry,exit,old.cats,old.dense,old.native,old.support,fragments,
        fps=30.,point_geometry=True,focal_motion=True)
    for i in range(len(nodes)-1):
        for j in range(i+1,len(nodes)):
            values,allowed,codes=arcs(i,j)
            for p,q in zip(*np.nonzero(allowed)):
                path=arcs.replay(i,j,p,q,int(codes[p,q]));s=entry[i][p].source
                assert values[p,q]==pytest.approx(arcs.direct_cost(s,int(nodes[i]),path[None])[0]+fragments(i,j)[s])


if __name__ == '__main__':
    unittest.main()
