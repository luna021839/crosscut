from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest

from crosscut.group_joint_source_camera import GroupOutputRole,joint_state_from_render_plan
from crosscut.group_joint_state_factory import candidate_joint_group_state,assemble_joint_group_states
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import RenderPlan,RenderSegment,RenderTransition,EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.group_cut_camera_proposal import UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON


class JointStateFactoryTests(unittest.TestCase):
    def fixture(self,fps=30.,native_fps=Fraction(30)):
        sources=tuple(Path(f'unopened-{i}.mp4') for i in range(6))
        clock=VerifiedRenderFrameClock(native_fps,10000,'a'*64)
        domain=GroupSourceDomain(sources,(0,2,3,5),(0.,)*4,(None,)*4,((),)*4,(clock,)*4)
        plan=RenderPlan(sources=sources,segments=tuple(RenderSegment(s,float(i),1.,float(i))
            for i,s in enumerate(domain.eligible_indices)),reference_source=0,aspect_ratio='16:9',
            output=Path('not-rendered.mp4'),fps=fps,transitions=(RenderTransition(),)*3)
        from crosscut.render import segment_frame_counts
        counts=segment_frame_counts(plan);cursor=0;states=[]
        for i,count in enumerate(counts):
            states.append(joint_state_from_render_plan(plan=plan,domain=domain,slot=i,
                first_output_frame=cursor,end_output_frame=cursor+count,role=GroupOutputRole.FORMATION))
            cursor+=count
        return plan,domain,tuple(states)

    def test_both_sources_can_swap_in_one_transaction_without_prior_solo_success(self):
        plan,domain,states=self.fixture()
        b=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],eligible_source_index=2)
        c=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[2],eligible_source_index=1)
        changed,indices=assemble_joint_group_states(plan=plan,domain=domain,path=(states[0],b,c,states[3]))
        self.assertEqual(indices,(1,2))
        self.assertEqual(tuple(s.source_index for s in changed.segments),(0,3,2,5))
        self.assertEqual(changed.segments[0],plan.segments[0])
        self.assertEqual(changed.segments[3],plan.segments[3])
        self.assertEqual(changed.output_frame_count,plan.output_frame_count)
        self.assertTrue(all(t.reason==UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON for t in changed.transitions))

    def test_pair_cannot_erase_existing_cut_by_aliasing_the_middle_source(self):
        plan,domain,states=self.fixture()
        b=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],eligible_source_index=2)
        with self.assertRaisesRegex(ValueError,'visible cut cadence'):
            assemble_joint_group_states(plan=plan,domain=domain,path=(states[0],b,states[2],states[3]))

    def test_complete_camera_key_and_every_native_sample_survive_2997_5994_renderer_assembly(self):
        plan,domain,states=self.fixture(29.97,Fraction(60000,1001))
        camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),
            EditorialTransformKeyframe(.5,1.22,.51,.5),EditorialTransformKeyframe(1.,1.15)),
            reason='joint_test_whole_camera')
        b=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],
            eligible_source_index=2,camera_track=camera)
        c=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[2],eligible_source_index=1)
        changed,_=assemble_joint_group_states(plan=plan,domain=domain,path=(states[0],b,c,states[3]))
        actual=joint_state_from_render_plan(plan=changed,domain=domain,slot=1,
            first_output_frame=b.first_output_frame,end_output_frame=b.end_output_frame,role=b.role)
        self.assertEqual(actual.key,b.key)
        self.assertEqual(actual.native_frame_indices,b.native_frame_indices)

    def test_wrong_native_grid_or_fixed_neighbor_camera_rejects_complete_transaction(self):
        plan,domain,states=self.fixture()
        for path in ((states[0],replace(states[1],native_frame_indices=tuple(n+1 for n in states[1].native_frame_indices)),
                states[2],states[3]),
                (replace(states[0],segments=(replace(states[0].segments[0],editorial_transform=
                    EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),))),)),*states[1:])):
            with self.assertRaises(ValueError):assemble_joint_group_states(plan=plan,domain=domain,path=path)
        self.assertEqual(tuple(s.source_index for s in plan.segments),(0,2,3,5))

    def test_no_change_returns_same_owned_plan_and_no_changed_cells(self):
        plan,domain,states=self.fixture()
        proposed,changed=assemble_joint_group_states(plan=plan,domain=domain,path=states)
        self.assertIs(proposed,plan)
        self.assertEqual(changed,())

    def test_replacement_native_cut_or_excluded_source_rejects_before_any_pixels(self):
        plan,domain,states=self.fixture()
        domain=replace(domain,scene_cut_times=((),(),(1.5,),()))
        with self.assertRaisesRegex(ValueError,'actual native camera cut'):
            candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],eligible_source_index=2)
        with self.assertRaises(ValueError):candidate_joint_group_state(plan=plan,domain=domain,
            incumbent_state=states[1],eligible_source_index=4)

    def test_coalesced_whole_camera_is_sliced_without_restart_at_internal_beat_cell(self):
        plan,domain,states=self.fixture()
        split=(plan.segments[0],replace(plan.segments[1],duration=.4),
            replace(plan.segments[1],source_start=1.4,master_start=1.4,duration=.6),*plan.segments[2:])
        plan=replace(plan,segments=split,transitions=(RenderTransition(),)*4)
        original=joint_state_from_render_plan(plan=plan,domain=domain,slot=1,
            first_output_frame=30,end_output_frame=60,role=GroupOutputRole.FORMATION)
        camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),EditorialTransformKeyframe(1.,1.3)))
        b=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=original,
            eligible_source_index=2,camera_track=camera)
        c=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[2],eligible_source_index=1)
        proposed,_=assemble_joint_group_states(plan=plan,domain=domain,path=(states[0],b,c,states[3]))
        self.assertGreater(proposed.segments[2].editorial_transform.evaluate(0.).zoom,1.1)
        self.assertEqual(b.segment_output_frame_counts,(12,18))
        self.assertEqual(len(b.native_frame_indices),30)
