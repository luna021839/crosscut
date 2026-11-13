from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest

from crosscut.group_joint_source_camera import GroupOutputRole,joint_state_from_render_plan
from crosscut.group_joint_state_factory import (candidate_joint_group_state,assemble_joint_group_states,
    assemble_single_group_source_camera_state)
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import RenderPlan,RenderSegment,RenderTransition,EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.group_cut_camera_proposal import UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON
from crosscut.timewarp import TimeWarpMap,TimeWarpConfig,TimeWarpControlPoint,TimeWarpResidualDiagnostics


class JointStateFactoryTests(unittest.TestCase):
    def test_joint_continuous_run_does_not_require_invalid_solo_tier_d_transition(self):
        plan,domain,_ = self.fixture()
        plan = replace(plan,segments=(plan.segments[0], plan.segments[1],
            replace(plan.segments[2],source_index=plan.segments[1].source_index), plan.segments[3]),
            transitions=(RenderTransition(),RenderTransition(tier='D',reason='same_native_run'),RenderTransition()))
        states = tuple(joint_state_from_render_plan(plan=plan,domain=domain,slot=slot,
            first_output_frame=slot*30,end_output_frame=(slot+1)*30,role=GroupOutputRole.FOCAL_HEAD_ARMS)
            for slot in range(4))
        # Neither first solo edit is a valid full old plan: its old tier-D
        # seam refers to a source that is also being changed in this transaction.
        b = candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],eligible_source_index=2)
        c = candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[2],eligible_source_index=2)
        proposed,changed = assemble_joint_group_states(plan=plan,domain=domain,path=(states[0],b,c,states[3]))
        self.assertEqual(changed,(1,2))
        self.assertEqual(tuple(s.source_index for s in proposed.segments),(0,3,3,5))
        self.assertEqual(proposed.output_frame_count,plan.output_frame_count)

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

    def test_single_cell_cannot_bypass_the_whole_shot_300_frame_work_budget(self):
        plan,domain,states=self.fixture()
        plan=replace(plan,segments=tuple(replace(s,source_start=11.*i,master_start=11.*i,duration=11.)
            for i,s in enumerate(plan.segments)))
        old=joint_state_from_render_plan(plan=plan,domain=domain,slot=1,
            first_output_frame=330,end_output_frame=660,role=GroupOutputRole.FORMATION)
        with self.assertRaisesRegex(ValueError,'whole-shot frame budget'):
            candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=old,eligible_source_index=2)

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

    def test_piecewise_source_warp_cannot_be_observed_as_one_incorrect_linear_shot_clock(self):
        plan,domain,_=self.fixture()
        segments=tuple(replace(s,source_start=10.*i,master_start=10.*i,duration=10.)
            for i,s in enumerate(plan.segments))
        plan=replace(plan,segments=(segments[0],replace(segments[1],duration=4.),
            replace(segments[1],source_start=14.,master_start=14.,duration=6.),*segments[2:]),
            transitions=(RenderTransition(),)*4)
        old=joint_state_from_render_plan(plan=plan,domain=domain,slot=1,
            first_output_frame=300,end_output_frame=600,role=GroupOutputRole.FORMATION)
        warp=TimeWarpMap(0.,40.,40.,tuple(TimeWarpControlPoint(a,b) for a,b in
            ((0.,0.),(10.,10.),(15.,15.03),(20.,20.),(40.,40.))),(),
            TimeWarpResidualDiagnostics(None,None,None,0),TimeWarpConfig(),False,None)
        domain=replace(domain,source_timewarps=(None,None,warp,None))
        with self.assertRaisesRegex(ValueError,'not one complete linear renderer clock'):
            candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=old,
                eligible_source_index=2)

    def single_shot_fixture(self):
        plan,domain,_=self.fixture(29.97,Fraction(60000,1001))
        plan=replace(plan,segments=(plan.segments[0],
            replace(plan.segments[1],duration=.4),
            replace(plan.segments[1],source_start=1.4,master_start=1.4,duration=.6),
            replace(plan.segments[3],source_start=2.,master_start=2.)),
            transitions=(RenderTransition(),)*3)
        states=tuple(joint_state_from_render_plan(plan=plan,domain=domain,slot=slot,
            first_output_frame=slot*30,end_output_frame=(slot+1)*30,role=GroupOutputRole.FOCAL_HEAD_ARMS)
            for slot in range(3))
        return plan,domain,states

    def test_single_actual_shot_spanning_two_cells_keeps_only_two_visible_cuts(self):
        from crosscut.group_actual_shots import visible_group_cut_inventory
        plan,domain,states=self.single_shot_fixture()
        camera=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),
            EditorialTransformKeyframe(.97,1.25,.52,.48)),sampling_canvas_zoom=1.25)
        whole=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],
            eligible_source_index=2,camera_track=camera)
        proposed,changed=assemble_single_group_source_camera_state(plan=plan,domain=domain,
            path=(states[0],whole,states[2]))
        self.assertEqual(changed,(1,2))
        self.assertEqual(tuple(e.first_incoming_output_frame for e in visible_group_cut_inventory(proposed,domain)),(30,60))
        self.assertEqual(whole.segment_output_frame_counts,(12,18))
        self.assertEqual(tuple(s.editorial_transform.sampling_canvas_zoom for s in proposed.segments[1:3]),(1.25,1.25))
        self.assertGreater(proposed.segments[2].editorial_transform.evaluate(0.).zoom,1.1)
        rebound=joint_state_from_render_plan(plan=proposed,domain=domain,slot=1,
            first_output_frame=30,end_output_frame=60,role=whole.role)
        self.assertEqual(rebound.key,whole.key)
        self.assertEqual(proposed.segments[0],plan.segments[0])
        self.assertEqual(proposed.segments[3],plan.segments[3])

    def test_single_shot_factory_rejects_changed_right_neighbor_or_erased_outer_cut(self):
        plan,domain,states=self.single_shot_fixture()
        with self.assertRaisesRegex(ValueError,'fixed outer neighbor'):
            assemble_single_group_source_camera_state(plan=plan,domain=domain,path=(states[0],states[1],
                replace(states[2],segments=(replace(states[2].segments[0],editorial_transform=
                    EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),))),))))
        aliased=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],eligible_source_index=3)
        with self.assertRaisesRegex(ValueError,'visible cut cadence'):
            assemble_single_group_source_camera_state(plan=plan,domain=domain,path=(states[0],aliased,states[2]))

    def test_three_and_four_state_factory_contracts_are_not_interchangeable(self):
        plan,domain,states=self.single_shot_fixture()
        with self.assertRaisesRegex(ValueError,'declared state path'):
            assemble_joint_group_states(plan=plan,domain=domain,path=states)
        four_plan,four_domain,four_states=self.fixture()
        with self.assertRaisesRegex(ValueError,'declared state path'):
            assemble_single_group_source_camera_state(plan=four_plan,domain=four_domain,path=four_states)
