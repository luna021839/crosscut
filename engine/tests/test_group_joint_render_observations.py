from dataclasses import replace
import unittest
from pathlib import Path
from unittest.mock import patch

from crosscut.group_joint_render_observations import refresh_joint_group_render_observations
from crosscut.group_joint_state_factory import candidate_joint_group_state
from crosscut.render import EditorialTransformTrack, EditorialTransformKeyframe
from crosscut.visual_score import VisualTimelinePlan
from tests import test_group_joint_state_factory as factory_fixture
from tests.test_sparse_source_lattice import Sampler
from tests.test_visual_boundary_repair import _choice


class JointSampler(Sampler):
    def __init__(self,path,**kwargs):
        super().__init__(Path(path.stem.rsplit('-',1)[-1]+'.mp4'),**kwargs)


class JointRenderedObservationsTests(unittest.TestCase):
    def fixture(self):
        plan,domain,states=factory_fixture.JointStateFactoryTests().fixture()
        timeline=VisualTimelinePlan(tuple(replace(_choice(i,i),quality_score=.99,
            face_match_candidate_failure='internal_camera_cut') for i in range(4)),1.,
            candidate_review_summaries=({'passed':True},))
        b=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],eligible_source_index=2)
        c=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[2],eligible_source_index=1)
        return plan,timeline,domain,(states[0],b,c,states[3])

    def run_refresh(self,**changes):
        plan,timeline,domain,path=self.fixture()
        arguments=dict(plan=plan,timeline=timeline,domain=domain,path=path,sync_scores=(1.,)*4)
        arguments.update(changes)
        Sampler.instances=[];Sampler.unreadable_source=None
        with patch('crosscut.visual_score._VideoSampler',JointSampler):
            return refresh_joint_group_render_observations(**arguments)

    def test_atomic_nonidentity_swap_refreshes_new_sources_and_every_dependent_edge(self):
        result=self.run_refresh()
        self.assertEqual(tuple(s.source_index for s in result.plan.segments),(0,3,2,5))
        self.assertEqual(tuple(c.source_index for c in result.timeline.choices),(0,2,1,3))
        self.assertEqual(result.changed_segment_indices,(1,2))
        self.assertEqual(result.refreshed_segment_indices,(0,1,2,3))
        self.assertEqual(result.timeline.candidate_review_summaries,())
        for choice in result.timeline.choices:
            self.assertAlmostEqual(choice.quality_score,.8)
            self.assertNotEqual(choice.face_match_candidate_failure,'internal_camera_cut')
        self.assertEqual([s.source for s in Sampler.instances],[0,2,3,5])
        self.assertTrue(all(s.closed for s in Sampler.instances))
        self.assertFalse(result.to_dict()['production_selected'])
        self.assertFalse(result.to_dict()['dense_role_evidence_verified'])

    def test_same_source_camera_change_is_also_refreshed_not_reused_as_raw_evidence(self):
        plan,timeline,domain,path=self.fixture()
        camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),
            EditorialTransformKeyframe(1.,1.2)))
        states=factory_fixture.JointStateFactoryTests().fixture()[2]
        b=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],
            eligible_source_index=1,camera_track=camera)
        # The materializer must receive the exact proposed camera, not an old
        # source's neutral segment. Its pixel projection is tested separately.
        with patch('crosscut.group_joint_render_observations.refresh_fixed_group_render_observations',
                return_value=timeline) as materialize:
            result=refresh_joint_group_render_observations(plan=plan,timeline=timeline,domain=domain,
                path=(states[0],b,states[2],states[3]),sync_scores=(1.,)*4)
        self.assertEqual(result.changed_segment_indices,(1,))
        self.assertEqual(result.refreshed_segment_indices,(0,1,2))
        self.assertEqual(materialize.call_args.kwargs['plan'].segments[1].editorial_transform,
            b.segments[0].editorial_transform)

    def test_stale_incumbent_rejects_before_source_sampler_creation(self):
        plan,timeline,domain,path=self.fixture()
        timeline=replace(timeline,choices=(replace(timeline.choices[0],source_index=3),*timeline.choices[1:]))
        with patch('crosscut.visual_score._VideoSampler') as sampler,self.assertRaisesRegex(ValueError,'stale incumbent'):
            refresh_joint_group_render_observations(plan=plan,timeline=timeline,domain=domain,path=path,
                sync_scores=(1.,)*4)
        sampler.assert_not_called()

    def test_no_change_preserves_exact_owned_objects_and_does_no_media_work(self):
        plan,timeline,domain,_=self.fixture();states=factory_fixture.JointStateFactoryTests().fixture()[2]
        with patch('crosscut.visual_score._VideoSampler') as sampler:
            result=refresh_joint_group_render_observations(plan=plan,timeline=timeline,domain=domain,
                path=states,sync_scores=(1.,)*4)
        self.assertIs(result.plan,plan);self.assertIs(result.timeline,timeline)
        self.assertEqual(result.refreshed_segment_indices,())
        sampler.assert_not_called()

    def test_invalid_sync_domain_rejects_before_sampler_creation(self):
        for scores in ((1.,),(1.,1.,True,1.),(1.,1.,float('nan'),1.)):
            with self.assertRaisesRegex(ValueError,'sync scores'):
                self.run_refresh(sync_scores=scores)
            self.assertEqual(Sampler.instances,[])


if __name__=='__main__':unittest.main()
