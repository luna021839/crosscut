from dataclasses import replace
import unittest

from crosscut.group_joint_source_camera import (GroupOutputRole, JointGroupEdgeObservation,
    JointGroupDenseVerification, JointGroupWorkBudget, SingleGroupDenseVerification,
    select_single_group_source_camera_path, select_joint_group_source_camera_path)
from crosscut.group_joint_state_factory import candidate_joint_group_state, assemble_single_group_source_camera_state


class SingleGroupSourceCameraTests(unittest.TestCase):
    def fixture(self):
        from engine.tests.test_group_joint_editorial_contract import GroupJointEditorialContractTests
        plan,domain,baseline,contract=GroupJointEditorialContractTests().single_contract()
        candidate=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=baseline[1],
            eligible_source_index=2)
        observed,dense_calls=[],[]
        def edge(a,b,budget):
            observed.append((a.key,b.key))
            cost=.4 if baseline[1].key in (a.key,b.key) else .1
            return JointGroupEdgeObservation(a.key,b.key,b.first_output_frame,cost,True,True)
        def dense(path,edges,budget):
            dense_calls.append(path)
            return SingleGroupDenseVerification(tuple(s.key for s in path),
                tuple(range(path[1].first_output_frame,path[1].end_output_frame)),True,
                'all_fixture_ticks_not_real_media')
        args=dict(domain=domain,left=baseline[0],right=baseline[2],states=(candidate,),
            declared_roles=(GroupOutputRole.FOCAL_HEAD_ARMS,)*3,observe_edge=edge,verify_dense=dense,
            budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.),
            editorial_contract=contract,baseline_path=baseline)
        return plan,args,baseline,candidate,observed,dense_calls

    def test_two_cells_are_one_whole_candidate_and_two_actual_boundaries(self):
        plan,args,baseline,candidate,observed,dense_calls=self.fixture()
        result=select_single_group_source_camera_path(**args)
        self.assertEqual(result.path,(baseline[0],candidate,baseline[2]))
        self.assertEqual(len(candidate.segments),2)
        self.assertEqual(result.edges[0].right_state_key,result.edges[1].left_state_key)
        self.assertEqual(result.baseline_edge_costs,(.4,.4))
        self.assertEqual(len(observed),4)
        self.assertEqual(len(dense_calls),1)
        proposed,changed=assemble_single_group_source_camera_state(plan=plan,domain=args['domain'],path=result.path)
        self.assertEqual(changed,(1,2))
        self.assertEqual(proposed.segments[0],plan.segments[0])
        self.assertEqual(proposed.segments[-1],plan.segments[-1])
        self.assertEqual(result.to_dict()['actual_boundary_count'],2)
        self.assertFalse(result.to_dict()['production_selected'])

    def test_endpoint_only_or_missing_internal_tick_cannot_pass_whole_x(self):
        for ticks in ((30,59),tuple(range(30,42))+tuple(range(43,60))):
            _,args,_,_,_,_=self.fixture()
            args['verify_dense']=lambda path,edges,budget:SingleGroupDenseVerification(
                tuple(s.key for s in path),ticks,True,'incomplete_actual_shot')
            self.assertIsNone(select_single_group_source_camera_path(**args).path)

    def test_better_sum_cannot_buy_a_worse_incident_cut(self):
        _,args,baseline,candidate,_,dense_calls=self.fixture()
        def edge(a,b,budget):
            cost=.4 if baseline[1].key in (a.key,b.key) else .01 if a.slot==0 else .5
            return JointGroupEdgeObservation(a.key,b.key,b.first_output_frame,cost,True,True)
        args['observe_edge']=edge
        result=select_single_group_source_camera_path(**args)
        self.assertIsNone(result.path)
        self.assertEqual(result.attempts[-1]['regressed_edge_slots'],[1])
        self.assertEqual(dense_calls,[])

    def test_unknown_baseline_or_missing_required_candidate_motion_stays_unknown(self):
        for baseline_unknown in (True,False):
            _,args,baseline,candidate,_,dense_calls=self.fixture()
            original=args['observe_edge']
            def edge(a,b,budget):
                if baseline_unknown and (a.key,b.key)==(baseline[0].key,baseline[1].key):
                    return None
                measured=original(a,b,budget)
                return replace(measured,motion_preservation_observed=False) if not baseline_unknown and a.key==candidate.key else measured
            args['observe_edge']=edge
            result=select_single_group_source_camera_path(**args)
            self.assertIsNone(result.path)
            self.assertEqual(dense_calls,[])
            if baseline_unknown:self.assertEqual(result.status,'baseline_edge_unknown_retained')

    def test_wrong_whole_camera_edge_and_foreign_contract_fail_before_adoption(self):
        _,args,baseline,candidate,_,_=self.fixture()
        args['observe_edge']=lambda a,b,budget:JointGroupEdgeObservation(
            'a'*64,b.key,b.first_output_frame,.1,True,True)
        with self.assertRaisesRegex(ValueError,'another camera'):
            select_single_group_source_camera_path(**args)
        _,args,baseline,candidate,observed,_=self.fixture()
        args['states']=(replace(candidate,editorial_contract_sha256='c'*64),)
        with self.assertRaisesRegex(ValueError,'pre-search'):
            select_single_group_source_camera_path(**args)
        self.assertEqual(observed,[])

    def test_three_and_four_state_verification_types_cannot_be_swapped(self):
        _,args,baseline,candidate,_,_=self.fixture()
        args['verify_dense']=lambda path,edges,budget:JointGroupDenseVerification(
            tuple(s.key for s in path),((30,59),()),True,'wrong_two_mutable_shot_payload')
        with self.assertRaisesRegex(ValueError,'typed evidence'):
            select_single_group_source_camera_path(**args)
        args.pop('states');args['states_by_slot']=((candidate,),())
        with self.assertRaisesRegex(ValueError,'declared actual-shot slots'):
            select_joint_group_source_camera_path(**args)

    def test_state_edge_and_dense_work_share_the_same_absolute_budget(self):
        for limits in (dict(maximum_states=2),dict(maximum_edges=3),dict(absolute_deadline=0.)):
            _,args,_,_,_,_=self.fixture()
            args['budget']=JointGroupWorkBudget(**dict(dict(absolute_deadline=10.,clock=lambda:0.),**limits))
            result=select_single_group_source_camera_path(**args)
            self.assertIsNone(result.path)
            self.assertEqual(result.status,'budget_retained_baseline')

    def test_duplicate_whole_candidates_do_not_repeat_native_edge_or_dense_work(self):
        _,args,baseline,candidate,observed,dense_calls=self.fixture()
        args['states']=(candidate,candidate,candidate)
        result=select_single_group_source_camera_path(**args)
        self.assertEqual(result.budget['counts']['states'],3)
        self.assertEqual(result.budget['counts']['edges'],4)
        self.assertEqual(result.budget['counts']['dense_paths'],1)
        self.assertEqual(len(observed),4)
        self.assertEqual(len(dense_calls),1)
