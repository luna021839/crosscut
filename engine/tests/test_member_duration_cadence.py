"""A held shot is bounded in seconds, not by its music-grid subdivisions."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, DirectMemberVisualSelection, DirectMemberVisualSelectionError,
    _bounded_same_source_hold_visual_path, _heterogeneous_cinematic_visual_path,
    _effective_consecutive_hold_limit,
)
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from engine.tests.test_direct_member_visual import _optimizer_mismatch


def config(**kwargs):
    return DirectMemberVisualConfig(hold_cadence_basis='duration',
        maximum_same_source_hold_fraction=1., **kwargs)


def domain(durations=(1.9,1.9,1.9,1.9)):
    path=(0,)*(len(durations)-1)+(1,)
    safe=_optimizer_mismatch(.1)
    bad=_optimizer_mismatch(10.,severe=True)
    matrices=[]
    for left,right in zip(path,path[1:]):
        matrix=[[bad,bad],[bad,bad]];matrix[left][right]=safe
        matrices.append(matrix)
    return path,matrices,tuple(tuple(i==s for i in range(2)) for s in path)


def solve(*, settings=None, durations=(1.9,1.9,1.9,1.9), milp=False,
          priority=(), priority_holds=0, forbidden=(), severe_cut=False):
    path,matrices,validity=domain(durations)
    if severe_cut:matrices[-1][0][1]=_optimizer_mismatch(10.,severe=True)
    with patch('crosscut.direct_member_visual._prefer_direct_member_milp',return_value=milp):
        try:
            return _bounded_same_source_hold_visual_path(matrices,
                source_quotas=(len(path)-1,1),source_order=(0,1),candidate_validity=validity,
                span_durations=durations,priority_cut_boundary_indices=priority,
                maximum_priority_cut_hold_count=priority_holds,
                forbidden_same_source_hold_boundary_indices=forbidden,config=settings or config())
        except DirectMemberVisualSelectionError:
            return None


class MemberDurationCadenceTests(unittest.TestCase):
    def test_time_policy_accepts_short_subdivisions_in_dp_and_milp(self):
        for milp in (False,True):
            with self.subTest(milp=milp):
                result=solve(milp=milp)
                self.assertIsNotNone(result)
                self.assertEqual(result[0],(0,0,0,1))
                self.assertIsNone(solve(milp=milp,settings=replace(config(),hold_cadence_basis='count_and_duration')))

    def test_same_shot_time_survives_different_grid_subdivision(self):
        # 5.7 seconds of source 0, represented by two or three grid spans.
        for durations in ((2.85,2.85,1.9),(1.9,1.9,1.9,1.9)):
            # This test isolates cadence from the independent count-derived
            # source exposure targets, which differ across these grids.
            result=solve(durations=durations,settings=config(maximum_source_duration_relative_deviation=.5))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(sum(d for d,s in zip(durations,result[0]) if s==0),5.7)

    def test_long_run_is_not_admitted_under_time_policy(self):
        for milp in (False,True):
            self.assertIsNone(solve(milp=milp,settings=config(maximum_same_source_run_seconds=5.5)))

    def test_global_count_and_density_are_still_hard(self):
        for settings in (config(maximum_same_source_hold_count=1),
                replace(config(),maximum_same_source_hold_fraction=.20)):
            for milp in (False,True):
                self.assertIsNone(solve(milp=milp,settings=settings))

    def test_explicit_zero_still_forbids_holds(self):
        settings=config(maximum_consecutive_same_source_holds=0)
        self.assertEqual(_effective_consecutive_hold_limit(settings,10),0)
        for milp in (False,True):self.assertIsNone(solve(milp=milp,settings=settings))

    def test_highlight_and_forbidden_boundary_requirements_survive(self):
        for milp in (False,True):
            self.assertIsNone(solve(milp=milp,priority=(0,1),priority_holds=1))
            self.assertIsNone(solve(milp=milp,forbidden=(0,)))

    def test_bad_cross_source_cut_is_never_relabelled_as_hold(self):
        for milp in (False,True):self.assertIsNone(solve(milp=milp,severe_cut=True))

    def test_role_cinematic_solver_keeps_time_limit_and_all_sources(self):
        for durations in ((1.9,1.9,1.9,1.9),(3.,3.,3.,3.)):
            path,matrices,validity=domain(durations)
            args=dict(source_order=(0,1),candidate_validity=validity,span_durations=durations,
                source_geometries=(direct_member_source_frame_geometry((2160,3840)),)*2,
                priority_cut_boundary_indices=(),maximum_priority_cut_hold_count=0,
                require_cinematic_bridge=False,config=config())
            if durations[0]==1.9:
                result=_heterogeneous_cinematic_visual_path(matrices,**args)
                self.assertEqual(result[0],path)
                self.assertTrue(result[2]['solver']['postsolve_path_audit_passed'])
            else:
                with self.assertRaises(DirectMemberVisualSelectionError):
                    _heterogeneous_cinematic_visual_path(matrices,**args)

    def artifact(self):
        path=(0,0,0,1);times=(0.,1.9,3.8,5.7,12.)
        safe=_optimizer_mismatch(.1)
        return DirectMemberVisualSelection(source_indices=path,source_quotas=(3,1),source_counts=(3,1),
            source_durations_seconds=(5.7,6.3),source_target_durations_seconds=(5.7,6.3),
            source_duration_relative_deviations=(0.,0.),span_validity=tuple(SimpleNamespace(
                canonical_master_start=a,canonical_master_end=b,sources=(SimpleNamespace(valid=True),)*2)
                for a,b in zip(times,times[1:])),boundaries=tuple(SimpleNamespace(
                cut_applied=l!=r,mismatch=safe) for l,r in zip(path,path[1:])),
            selected_boundary_master_seconds=times[1:-1],optimizer_cost=0.,sample_count=0,
            failed_sample_count=0,native_scene_cut_evidence_available=False,
            continuous_official_portrait_fancam_contract=True,config=config())

    def test_final_artifact_rechecks_offset_adjusted_run_duration(self):
        artifact=self.artifact()
        self.assertEqual(artifact.source_indices,(0,0,0,1))
        with self.assertRaisesRegex(ValueError,'run limit exceeded'):
            replace(artifact,selected_boundary_master_seconds=(1.9,3.8,8.))

    def test_existing_explicit_local_audition_extension_is_preserved(self):
        artifact=self.artifact()
        settings=replace(artifact.config,boundary_search_radius_frames=6)
        adjusted=replace(artifact,config=settings,selected_boundary_master_seconds=(1.9,3.8,7.35))
        self.assertEqual(adjusted.selected_boundary_master_seconds[-1],7.35)
        with self.assertRaisesRegex(ValueError,'run limit exceeded'):
            replace(adjusted,selected_boundary_master_seconds=(1.9,3.8,7.5))

    def test_serialized_policy_does_not_report_legacy_count_as_current_limit(self):
        payload=config().to_dict()['same_source_hold']
        self.assertEqual(payload['cadence_basis'],'duration')
        self.assertIsNone(payload['maximum_consecutive'])
        self.assertEqual(config(maximum_consecutive_same_source_holds=0).to_dict()['same_source_hold']['maximum_consecutive'],0)
        self.assertEqual(DirectMemberVisualConfig().to_dict()['same_source_hold']['maximum_consecutive'],1)
        with self.assertRaises(ValueError):DirectMemberVisualConfig(hold_cadence_basis='ignore_everything')


if __name__=='__main__':unittest.main()
