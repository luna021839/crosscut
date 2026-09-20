from dataclasses import replace
from unittest import TestCase

from crosscut.joint_switch_requirements import SourceSwitchRequirement, SourceSwitchCounter
from crosscut.joint_timeline import BoundaryMeasurementKey, select_joint_timeline_candidates
from crosscut.timeline import NoCompleteTimelinePath
from engine.tests.test_joint_timeline import candidate, isolated_score_config, option_set, valid_bridge


class JointSwitchRequirementsTests(TestCase):
    def setUp(self):
        self.rows=tuple(tuple(candidate(i,s,.9 if s=='A' else .1) for s in ('A','B')) for i in range(6))

    def select(self,requirements=(),**kwargs):
        config=replace(isolated_score_config(),quality_weight=1.,source_switch_bonus=0.)
        return select_joint_timeline_candidates(self.rows,
            option_provider=kwargs.pop('option_provider',lambda key:option_set(key,authorized={0:0.})),
            bridge_provider=valid_bridge, config=config,maximum_results=1,
            source_switch_requirements=requirements,**kwargs)[0]

    def switches(self,result):
        path=result.timeline.candidates
        return [b.start_seconds for a,b in zip(path,path[1:]) if a.source_id!=b.source_id]

    def test_empty_requirements_leave_highest_score_path_unchanged(self):
        self.assertEqual(self.switches(self.select()),[])

    def test_section_count_is_enforced_before_result_not_only_soft_bonus(self):
        result=self.select((SourceSwitchRequirement(0.,6.,4),))
        self.assertGreaterEqual(len(self.switches(result)),4)

    def test_multiple_sections_keep_distinct_count_states(self):
        result=self.select((SourceSwitchRequirement(0.,3.,2),SourceSwitchRequirement(3.,6.,2)))
        cuts=self.switches(result)
        self.assertEqual(sum(0<t<3 for t in cuts),2)
        self.assertEqual(sum(3<t<6 for t in cuts),2)

    def test_impossible_domain_fails_before_pixel_measurement(self):
        domain=frozenset(BoundaryMeasurementKey.from_candidates(i,a,b)
            for i,(left,right) in enumerate(zip(self.rows,self.rows[1:]))
            for a in left for b in right if a.source_id==b.source_id)
        with self.assertRaises(NoCompleteTimelinePath) as caught:
            self.select((SourceSwitchRequirement(0.,6.,1),),boundary_search_domain=domain,
                option_provider=lambda key:self.fail('impossible cadence must not decode pixels'))
        self.assertEqual(caught.exception.reason_code,'section_source_switch_capacity_insufficient')

    def test_disconnected_switch_edges_cannot_fake_two_switches(self):
        rows=self.rows[:3]
        domain=set()
        for boundary,(left,right) in enumerate(zip(rows,rows[1:])):
            a_left=next(item for item in left if item.source_id=='A')
            b_left=next(item for item in left if item.source_id=='B')
            a_right=next(item for item in right if item.source_id=='A')
            b_right=next(item for item in right if item.source_id=='B')
            # Both positions contain an A->B switch, but no complete path can
            # use both because the second position cannot reconnect B to A.
            domain.add(BoundaryMeasurementKey.from_candidates(
                boundary,a_left,b_right))
            domain.add(BoundaryMeasurementKey.from_candidates(
                boundary,b_left,b_right))
            domain.add(BoundaryMeasurementKey.from_candidates(
                boundary,a_left,a_right))
        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_timeline_candidates(rows,
                option_provider=lambda key:self.fail('disconnected capacity must not decode'),
                bridge_provider=valid_bridge,
                config=replace(isolated_score_config(),source_switch_bonus=0.),
                maximum_results=1,
                boundary_search_domain=frozenset(domain),
                source_switch_requirements=(SourceSwitchRequirement(0.,3.,2),))
        self.assertEqual(caught.exception.reason_code,
                         'section_source_switch_capacity_insufficient')
        self.assertIsNotNone(caught.exception.segment_index)

    def test_duplicate_camera_family_does_not_fake_a_source_switch(self):
        with self.assertRaises(NoCompleteTimelinePath):
            self.select((SourceSwitchRequirement(0.,6.,1),),source_switch_families={'A':'same','B':'same'})

    def test_impossible_pixel_edges_are_not_authorized_by_count_target(self):
        with self.assertRaises(NoCompleteTimelinePath):
            self.select((SourceSwitchRequirement(0.,6.,2),),
                option_provider=lambda key:option_set(key,authorized={0:0.} if key.outgoing.source_id==key.incoming.source_id else {}))

    def test_shifted_boundary_uses_quantized_output_time(self):
        counter=SourceSwitchCounter((SourceSwitchRequirement(.9,1.1,1),),self.rows,
            fps=30.,maximum_offset_frames=8)
        inside=counter.advance((0,),left='A',right='B',nominal_boundary_seconds=1.,frame_offset=0)
        outside=counter.advance((0,),left='A',right='B',nominal_boundary_seconds=1.,frame_offset=4)
        self.assertEqual(inside,(1,));self.assertEqual(outside,(0,))

    def test_exact_section_edge_does_not_count_as_two_overlapping_runs(self):
        window=SourceSwitchRequirement(1.,2.,1)
        self.assertFalse(window.contains(1.));self.assertFalse(window.contains(2.))
        self.assertTrue(window.contains(1.5))

    def test_satisfied_count_caps_state_growth(self):
        counter=SourceSwitchCounter((SourceSwitchRequirement(0.,6.,1),),self.rows,
            fps=30.,maximum_offset_frames=8)
        self.assertEqual(counter.advance((1,),left='A',right='B',nominal_boundary_seconds=2.,frame_offset=0),(1,))

    def test_invalid_overlapping_or_unbound_family_input_rejected(self):
        for args in ((SourceSwitchRequirement(0.,3.,1),SourceSwitchRequirement(2.,5.,1)),):
            with self.assertRaises(ValueError):self.select(args)
        with self.assertRaises(ValueError):
            self.select((SourceSwitchRequirement(0.,6.,1),),source_switch_families={'A':'one'})
        for count in (True,0,-1):
            with self.assertRaises(ValueError):SourceSwitchRequirement(0.,1.,count)
