from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast
import unittest
from unittest.mock import Mock

from crosscut.member_joint_camera_preflight import MemberJointCameraPreflight
from crosscut.automatic_joint_dissolve import register_automatic_member_dissolves
from crosscut.direct_member_visual import DirectMemberVisualSelectionError
import engine.tests.test_automatic_joint_dissolve as automatic_tests


def fixture(success=False):
    args=automatic_tests.AutomaticJointDissolveTests().inputs()
    selection=args['visual_selection']
    selection.boundaries=(SimpleNamespace(boundary_index=0,
        mismatch=SimpleNamespace(cinematic_bridge=SimpleNamespace(style='pose_dissolve'))),)
    selection.selected_boundary_master_seconds=(103.,)
    if success:
        segments,transitions,audit=register_automatic_member_dissolves(**args)
    else:
        segments,transitions=args['render_segments'],args['transitions']
        audit={'boundaries':[{'boundary_index':0,'applied':False,'reason':'joint_camera_containment_not_verified'}]}
    return dict(visual_selection=selection,render_segments=segments,transitions=transitions,camera_audit=audit)


class MemberJointCameraPreflightTests(unittest.TestCase):
    def test_unknown_camera_changes_ranking_not_source_validity(self):
        args=fixture();gate=MemberJointCameraPreflight()
        out=gate.review(**args)
        self.assertFalse(out['passed'])
        self.assertTrue(out['reselection_requested'])
        self.assertTrue(out['proxy_encoding_skipped'])
        self.assertEqual(out['ranking_penalties'],((103.,0,1,2.75),))
        self.assertFalse(out['source_exclusion_authorized'])
        self.assertFalse(out['evidence_persisted_as_infeasible'])
        self.assertFalse(out['hard_cut_fallback_authorized'])

    def test_repeated_failed_plan_stops_without_escalating_cost_or_encoding(self):
        args=fixture();gate=MemberJointCameraPreflight()
        first=gate.review(**args);second=gate.review(**args)
        self.assertEqual(second['reason'],'unchanged_failed_camera_plan')
        self.assertFalse(second['reselection_requested'])
        self.assertEqual(first['ranking_penalties'],second['ranking_penalties'])

    def test_budget_is_bounded_even_when_clocks_keep_changing(self):
        args=fixture();gate=MemberJointCameraPreflight(maximum_reselections=1)
        gate.review(**args)
        args['render_segments']=(replace(args['render_segments'][0],source_start=10.01),args['render_segments'][1])
        out=gate.review(**args)
        self.assertEqual(out['reason'],'joint_camera_reselection_budget_exhausted')
        self.assertFalse(out['reselection_requested'])

    def test_changed_evidence_may_pass_previously_failed_source_plan(self):
        gate=MemberJointCameraPreflight();gate.review(**fixture())
        passed=gate.review(**fixture(True))
        self.assertTrue(passed['passed'])
        self.assertFalse(passed['proxy_encoding_skipped'])
        self.assertFalse(passed['quality_approval'])

    def test_applied_flag_without_renderer_tracks_is_not_success(self):
        args=fixture();args['camera_audit']=fixture(True)['camera_audit']
        self.assertFalse(MemberJointCameraPreflight().review(**args)['passed'])

    def test_other_effect_styles_and_optional_dissolves_are_not_rejected(self):
        for bridge in (None,SimpleNamespace(style='zoom_in')):
            args=fixture();args['visual_selection'].boundaries[0].mismatch.cinematic_bridge=bridge
            self.assertTrue(MemberJointCameraPreflight().review(**args)['passed'])

    def test_duplicate_camera_audit_is_programming_error_not_retry(self):
        args=fixture();args['camera_audit']['boundaries']*=2
        with self.assertRaises(ValueError):MemberJointCameraPreflight().review(**args)

    def run_pipeline_branch(self,success):
        # Execute the actual production preflight branch, without importing or
        # running the protected broad pipeline test suite or real acquisition.
        path=Path(__file__).parents[1]/'crosscut/pipeline.py'
        tree=ast.parse(path.read_text())
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='joint_camera_preflight_report' for t in n.targets))
        conditional=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
            and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.operand,ast.Subscript)
            and isinstance(n.test.operand.value,ast.Name)
            and n.test.operand.value.id=='joint_camera_preflight_report')
        args=fixture(success);selector=Mock(return_value=(args['render_segments'],args['transitions'],{}, {},args['visual_selection']))
        proxy=Mock();external_penalties=[]
        env=dict(joint_camera_preflight=MemberJointCameraPreflight(),
            direct_visual_selection=args['visual_selection'],attached_segments=args['render_segments'],
            candidate_member_transitions=args['transitions'],candidate_joint_dissolve_audit=args['camera_audit'],
            request=SimpleNamespace(output_directory=Path('/not-written'),target_member=SimpleNamespace(value='test')),
            base_filename='proof',_write_analysis_manifest_atomically=Mock(),decoded_proxy_render_count=0,
            dense_camera_reselection_count=0,dense_render_path_reselection_count=0,
            _DIRECT_MEMBER_MAXIMUM_DENSE_RENDER_PATH_RESELECTIONS=6,
            DirectMemberVisualSelectionError=DirectMemberVisualSelectionError,
            joint_camera_ranking_penalties={},render_penalized_transition_edges=external_penalties,
            render_infeasible_transition_edges=[],reporter=SimpleNamespace(log=Mock()),
            _select_direct_member_render_path=selector,_selected_partition_unit_counts_for_retry=lambda:(1,1),
            proxy=proxy)
        loop=ast.For(target=ast.Name('_path_selection_attempt',ast.Store()),iter=ast.Constant((0,)),
            body=[assignment,conditional,ast.Expr(ast.Call(ast.Name('proxy',ast.Load()),[],[]))],orelse=[])
        code=ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[]))
        exec(compile(code,str(path),'exec'),env)
        self.assertEqual(external_penalties,[])  # never persist camera uncertainty as decoded QA
        return selector,proxy,env

    def test_production_branch_reselects_before_proxy_for_failed_camera(self):
        selector,proxy,env=self.run_pipeline_branch(False)
        selector.assert_called_once();proxy.assert_not_called()
        self.assertEqual(env['decoded_proxy_render_count'],0)
        self.assertEqual(env['joint_camera_ranking_penalties'],{(103.,0,1):(103.,0,1,2.75)})

    def test_production_branch_keeps_proxy_qa_for_applied_camera(self):
        selector,proxy,_=self.run_pipeline_branch(True)
        selector.assert_not_called();proxy.assert_called_once()


if __name__=='__main__':unittest.main()
