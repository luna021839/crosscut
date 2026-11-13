from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import unittest
import numpy as np

from crosscut.group_joint_source_camera import (GroupOutputRole,GroupSourceCameraState,
    JointGroupEdgeObservation,JointGroupDenseVerification,JointGroupWorkBudget,
    JointGroupBudgetExhausted,JointProjectedPixelCache,select_joint_group_source_camera_path)
from crosscut.group_joint_source_camera import joint_state_from_render_plan
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import RenderPlan,RenderSegment,RenderTransition,EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.boundary_subject_alignment import _project_box,_TransformState


class JointGroupSourceCameraTests(unittest.TestCase):
    def fixture(self):
        sources=tuple(Path(f'{i}.mp4') for i in range(6))
        clocks=(VerifiedRenderFrameClock(Fraction(30),1000,'a'*64),)*4
        domain=GroupSourceDomain(sources,(0,2,3,5),(0.,)*4,(None,)*4,((),)*4,clocks)
        clock_hash=hashlib.sha256(json.dumps(clocks[0].to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        def state(slot,camera=False,raw=None):
            center=.4 if slot==1 else .6
            track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.25,center,.5,0.),
                EditorialTransformKeyframe(29/30.,1.25,center,.5,0.))) if camera else None
            return GroupSourceCameraState(slot,30*slot,30*(slot+1),slot,
                (RenderSegment(domain.request_index(slot),float(slot),1.,float(slot),editorial_transform=track),),
                tuple(range(30*slot,30*(slot+1))),GroupOutputRole.FOCAL_HEAD,clock_hash,raw_shot_family=raw)
        raw=tuple(state(i,raw='wide_formation' if i==1 else 'close_face') for i in range(4))
        b,c=state(1,True,'wide_formation'),state(2,True,'medium_focus')
        def endpoint(value,outgoing):
            box=(.44,.3,.12,.25) if value.slot in (0,3) else (
                (.34,.3,.12,.2) if value.slot==1 else (.54,.3,.12,.2))
            track=value.segments[0].editorial_transform
            if track is not None:
                key=track.evaluate(29/30. if outgoing else 0.)
                box=_project_box(box,_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees))[0]
            return box
        def observe(a,b,budget):
            aa,bb=endpoint(a,True),endpoint(b,False)
            delta=abs((aa[0]+aa[2]/2)-(bb[0]+bb[2]/2))
            scale=abs(math.log(aa[3]/bb[3]))
            if delta>.051 or scale>.03:return None
            return JointGroupEdgeObservation(a.key,b.key,b.first_output_frame,min(1.,delta+scale),True,True)
        def dense(path,edges,budget):
            return JointGroupDenseVerification(tuple(s.key for s in path),
                tuple(tuple(range(s.first_output_frame,s.end_output_frame)) for s in path[1:3]),True,
                'all_fixture_native_ticks_verified_not_real_media')
        kwargs=dict(domain=domain,left=raw[0],right=raw[3],
            states_by_slot=((raw[1],b),(raw[2],c)),declared_roles=(GroupOutputRole.FOCAL_HEAD,)*4,
            observe_edge=observe,verify_dense=dense,budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.))
        return kwargs,raw,b,c

    def test_both_solo_sets_empty_but_two_complete_cameras_share_a_valid_middle_edge(self):
        args,raw,b,c=self.fixture();observe=args['observe_edge'];budget=args['budget']
        self.assertIsNone(observe(b,raw[2],budget))
        self.assertIsNone(observe(raw[1],c,budget))
        selected=select_joint_group_source_camera_path(**args)
        self.assertEqual(selected.path,(raw[0],b,c,raw[3]))
        self.assertEqual(selected.edges[0].right_state_key,selected.edges[1].left_state_key)
        self.assertEqual(selected.edges[1].right_state_key,selected.edges[2].left_state_key)
        self.assertFalse(selected.to_dict()['prior_solo_success_required'])
        self.assertFalse(selected.to_dict()['production_selected'])

    def test_raw_family_mismatch_does_not_prune_a_projected_valid_complete_state(self):
        args,raw,b,c=self.fixture()
        self.assertNotEqual(raw[0].raw_shot_family,b.raw_shot_family)
        selected=select_joint_group_source_camera_path(**args)
        self.assertIsNotNone(selected.path)
        self.assertFalse(selected.to_dict()['raw_family_reachability_gate'])

    def test_state_keys_bind_whole_camera_clock_role_and_output_grid(self):
        args,raw,b,c=self.fixture();keys={b.key}
        variants=(replace(b,role=GroupOutputRole.FOCAL_HEAD_ARMS),
            replace(b,source_clock_identity='another_clock'),
            replace(b,native_frame_indices=tuple(n+1 for n in b.native_frame_indices)),
            replace(b,output_size=(1280,720)),replace(b,output_fps=29.97),
            replace(b,segments=(replace(b.segments[0],editorial_transform=replace(b.segments[0].editorial_transform,
                keyframes=(b.segments[0].editorial_transform.keyframes[0],EditorialTransformKeyframe(.5,1.3,.4,.5),
                    b.segments[0].editorial_transform.keyframes[-1]))),)))
        keys.update(v.key for v in variants)
        self.assertEqual(len(keys),1+len(variants))
        self.assertEqual(b.key,replace(b,raw_shot_family='arbitrary_raw_label').key)

    def test_different_native_tick_middle_edges_cannot_form_an_inconsistent_path(self):
        args,raw,b,c=self.fixture();args['states_by_slot']=((b,),(
            replace(c,first_output_frame=61,end_output_frame=91),))
        self.assertIsNone(select_joint_group_source_camera_path(**args).path)

    def test_missing_motion_visibility_and_ambiguous_correspondence_are_not_zero_cost(self):
        for flags in (dict(motion_preservation_observed=False),dict(required_visibility_observed=False),
                dict(correspondence_ambiguous=True)):
            args,raw,b,c=self.fixture();args['observe_edge']=lambda a,b,_:JointGroupEdgeObservation(
                a.key,b.key,b.first_output_frame,.0,**dict(dict(required_visibility_observed=True,
                    motion_preservation_observed=True),**flags))
            self.assertIsNone(select_joint_group_source_camera_path(**args).path)

    def test_endpoint_only_dense_payload_cannot_authorize_two_whole_shots(self):
        args,raw,b,c=self.fixture()
        args['verify_dense']=lambda path,edges,_:JointGroupDenseVerification(tuple(s.key for s in path),
            ((30,59),(60,89)),True,'endpoints_only')
        self.assertIsNone(select_joint_group_source_camera_path(**args).path)

    def test_role_cannot_be_changed_after_rejection_or_native_cut_hidden_inside_state(self):
        args,raw,b,c=self.fixture()
        args['states_by_slot']=((replace(b,role=GroupOutputRole.FORMATION),),(c,))
        with self.assertRaisesRegex(ValueError,'predeclared role'):select_joint_group_source_camera_path(**args)
        args,raw,b,c=self.fixture();domain=args['domain']
        args['domain']=replace(domain,scene_cut_times=((),(1.5,),(),()))
        with self.assertRaisesRegex(ValueError,'actual native camera cut'):select_joint_group_source_camera_path(**args)

    def test_non_contiguous_source_mapping_and_wrong_edge_ownership_fail_before_adoption(self):
        args,raw,b,c=self.fixture();self.assertEqual(b.segments[0].source_index,2)
        self.assertIs(b.bind_source(args['domain']),b)
        with self.assertRaisesRegex(ValueError,'domain differs'):
            replace(b,segments=(replace(b.segments[0],source_index=1),)).bind_source(args['domain'])
        args['observe_edge']=lambda a,b,_:JointGroupEdgeObservation('another_camera',b.key,
            b.first_output_frame,.0,True,True)
        with self.assertRaisesRegex(ValueError,'another camera'):select_joint_group_source_camera_path(**args)

    def test_each_work_dimension_and_absolute_deadline_returns_no_partial_path(self):
        for limits in (dict(maximum_states=3),dict(maximum_edges=1),dict(absolute_deadline=0.)):
            args,raw,b,c=self.fixture()
            args['budget']=JointGroupWorkBudget(**dict(dict(absolute_deadline=10.,clock=lambda:0.),**limits))
            result=select_joint_group_source_camera_path(**args)
            self.assertIsNone(result.path);self.assertEqual(result.status,'budget_retained_baseline')
        args,raw,b,c=self.fixture()
        def dense(path,edges,budget):
            budget.spend('new_decodes',1201)
        args['verify_dense']=dense
        result=select_joint_group_source_camera_path(**args)
        self.assertIsNone(result.path);self.assertIn('new_decodes',result.attempts[-1]['reason'])

    def test_identical_states_and_edges_are_deduplicated_within_one_owned_search(self):
        args,raw,b,c=self.fixture();args['states_by_slot']=((b,b,b),(c,c,c))
        result=select_joint_group_source_camera_path(**args)
        self.assertIsNotNone(result.path)
        self.assertEqual(result.budget['counts']['states'],4)
        self.assertEqual(result.budget['counts']['edges'],3)
        self.assertEqual(result.budget['counts']['dense_paths'],1)

    def comparison(self, candidate_costs, baseline_costs=(.1,.6,.1)):
        args,raw,b,c=self.fixture();dense_calls=[]
        before={(raw[i].key,raw[i+1].key):cost for i,cost in enumerate(baseline_costs)}
        after={(a.key,z.key):cost for a,z,cost in zip((raw[0],b,c),(b,c,raw[3]),candidate_costs)}
        def observe(a,z,budget):
            cost={**before,**after}.get((a.key,z.key))
            return None if cost is None else JointGroupEdgeObservation(a.key,z.key,
                z.first_output_frame,cost,True,True)
        verify=args['verify_dense']
        def dense(path,edges,budget):
            dense_calls.append(path)
            return verify(path,edges,budget)
        args.update(observe_edge=observe,verify_dense=dense,baseline_path=raw,
            states_by_slot=((b,),(c,)))
        return args,raw,dense_calls

    def test_lower_total_cannot_buy_a_worse_fixed_neighbor_edge(self):
        args,raw,calls=self.comparison((.3,.2,.1))
        result=select_joint_group_source_camera_path(**args)
        self.assertIsNone(result.path)
        self.assertEqual(calls,[])
        self.assertEqual(result.attempts[-1]['reason'],'protected_edge_framing_regression')
        self.assertEqual(result.attempts[-1]['regressed_edge_slots'],[0])

    def test_all_three_edges_are_observed_before_individual_nonregression_and_gain(self):
        args,raw,calls=self.comparison((.1,.2,.1))
        result=select_joint_group_source_camera_path(**args)
        self.assertIsNotNone(result.path)
        self.assertEqual(len(calls),1)
        self.assertEqual(result.baseline_edge_costs,(.1,.6,.1))
        self.assertTrue(result.to_dict()['per_edge_nonregression_checked'])
        args,raw,calls=self.comparison((.1,.6,.1))
        result=select_joint_group_source_camera_path(**args)
        self.assertIsNone(result.path)
        self.assertEqual(result.attempts[-1]['reason'],'no_observed_edge_gain')
        self.assertEqual(calls,[])

    def test_unknown_baseline_edge_is_not_zero_or_a_worst_numeric_cost(self):
        args,raw,calls=self.comparison((.1,.2,.1))
        observe=args['observe_edge']
        args['observe_edge']=lambda a,b,budget:None if (a.key,b.key)==(raw[1].key,raw[2].key) \
            else observe(a,b,budget)
        result=select_joint_group_source_camera_path(**args)
        self.assertIsNone(result.path)
        self.assertEqual(result.status,'baseline_edge_unknown_retained')
        self.assertEqual(calls,[])

    def test_byte_lru_reuses_same_geometry_but_not_another_camera_on_the_same_raw_frame(self):
        args,raw,b,c=self.fixture();budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,
            maximum_projected_cache_bytes=100)
        cache=JointProjectedPixelCache(budget);calls=[]
        def observe():calls.append(1);return np.zeros((5,5,3),np.uint8)
        x=cache.get_or_observe(state=b,output_tick=30,raw_identity=('file',30),observer=observe)
        y=cache.get_or_observe(state=b,output_tick=30,raw_identity=('file',30),observer=observe)
        self.assertIs(x,y);self.assertFalse(x.flags.writeable)
        cache.get_or_observe(state=raw[1],output_tick=30,raw_identity=('file',30),observer=observe)
        self.assertEqual(len(calls),2);self.assertLessEqual(cache.bytes,100)
        self.assertEqual(len(cache.rows),1)

    def test_model_counts_are_named_and_fail_closed_without_consuming_an_invalid_call(self):
        args,raw,b,c=self.fixture();budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,maximum_model_calls=2)
        with self.assertRaises(ValueError):budget.spend('model_calls')
        self.assertEqual(budget.counts['model_calls'],0)
        budget.spend('model_calls',model='face');budget.spend('model_calls',model='pose')
        with self.assertRaises(JointGroupBudgetExhausted):budget.spend('model_calls',model='pose')
        self.assertEqual(budget.model_calls,{'face':1,'pose':1})

    def test_state_builder_uses_full_shared_renderer_clock_not_rounded_per_cell_seeks(self):
        args,raw,b,c=self.fixture();domain=args['domain']
        plan=RenderPlan(sources=domain.request_sources,reference_source=0,aspect_ratio='16:9',
            output=Path('unused.mp4'),fps=30.,segments=(
                RenderSegment(0,.51,.4,0.),RenderSegment(0,.91,.6,.4),
                RenderSegment(2,1.51,1.,1.)),transitions=(RenderTransition(tier='C'),)*2)
        state=joint_state_from_render_plan(plan=plan,domain=domain,slot=1,
            first_output_frame=0,end_output_frame=30,role=GroupOutputRole.FOCAL_HEAD)
        expected=native_frame_clocks_for_output(plan,domain.clocks_by_request,range(30))
        self.assertEqual(state.native_frame_indices,tuple(r.native_frame_index for r in expected.values()))
        self.assertEqual(len(state.segments),2)
        for first,end in ((1,30),(0,29)):
            with self.assertRaisesRegex(ValueError,'owned splitting required'):
                joint_state_from_render_plan(plan=plan,domain=domain,slot=1,
                    first_output_frame=first,end_output_frame=end,role=GroupOutputRole.FOCAL_HEAD)


if __name__=='__main__':unittest.main()
