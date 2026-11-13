"""Counterexamples for source-owner reachability and uncertainty contracts."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut import group_source_changing_owner as owner
from crosscut.group_joint_source_camera import GroupOutputRole, JointGroupWorkBudget, JointGroupEdgeObservation, select_joint_group_source_camera_path
from crosscut.group_joint_native_observer import NativeGroupHead
from engine.tests import test_group_joint_source_camera as numeric


class SourceOwnerReachabilityTests(unittest.TestCase):
    @staticmethod
    def static_fixture():
        from engine.tests.test_group_joint_state_factory import JointStateFactoryTests
        from crosscut.render import SourceLookAdjustment
        from crosscut.group_joint_source_camera import joint_state_from_render_plan
        plan,domain,_=JointStateFactoryTests().fixture()
        grades=(SourceLookAdjustment(brightness=-.01),SourceLookAdjustment(brightness=.02),
            SourceLookAdjustment(contrast=.98),SourceLookAdjustment(saturation=.97))
        plan=replace(plan,segments=tuple(replace(s,look_adjustment=g)
            for s,g in zip(plan.segments,grades,strict=True)))
        state=joint_state_from_render_plan(plan=plan,domain=domain,slot=1,
            first_output_frame=30,end_output_frame=60,role=GroupOutputRole.FOCAL_HEAD)
        return SimpleNamespace(incumbent=SimpleNamespace(render_plan=plan),domain=domain),state,grades

    def test_source_change_uses_destination_grade_in_request_not_eligible_domain(self):
        context,state,grades=self.static_fixture()
        new=owner._hypothesis(context,state,2)
        self.assertEqual(new.segments[0].source_index,3)
        self.assertEqual(new.segments[0].look_adjustment,grades[2])
        self.assertNotEqual(new.segments[0].look_adjustment,grades[1])
        self.assertEqual(new.native_frame_indices,state.native_frame_indices)
        self.assertTrue(all(row['fresh_color_quality_approved'] is False
            for row in owner.source_static_look_bindings(context)))

    def test_missing_static_grade_is_not_imputed_to_neutral(self):
        context,state,_=self.static_fixture()
        plan=context.incumbent.render_plan
        context.incumbent.render_plan=replace(plan,segments=tuple(
            replace(s,source_index=0) if s.source_index==3 else s for s in plan.segments))
        with self.assertRaisesRegex(ValueError,'static_look_unavailable'):
            owner._hypothesis(context,state,2)

    def test_locally_varying_destination_grade_is_not_reused_as_global(self):
        context,state,grades=self.static_fixture();plan=context.incumbent.render_plan
        context.incumbent.render_plan=replace(plan,segments=(
            replace(plan.segments[0],source_index=3,look_adjustment=grades[0]),*plan.segments[1:]))
        with self.assertRaisesRegex(ValueError,'locally_varying'):
            owner._hypothesis(context,state,2)

    def test_installed_native_assets_use_pinned_package_detector_not_pose_parent(self):
        from crosscut import pose,visual_score
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'package';cache=root/'cache'
            package.mkdir();cache.mkdir()
            person=package/pose.FORMATION_PERSON_DETECTOR_MODEL
            head=package/'yunet.onnx';model=cache/'rtmpose.onnx'
            payloads=((person,b'person-fixture'),(head,b'head-fixture'),(model,b'pose-fixture'))
            for path,value in payloads:path.write_bytes(value)
            # Explicit tiny fixtures verify path/digest policy, not NN models.
            expected={pose.FORMATION_PERSON_DETECTOR_MODEL:sha256(payloads[0][1]).hexdigest()}
            context=SimpleNamespace(model_path=model,model_sha256=sha256(payloads[2][1]).hexdigest())
            with patch.object(pose,'_MODEL_DIRECTORY',package),patch.object(pose,'_EXPECTED_SHA256',expected), \
                    patch.object(visual_score,'_YUNET_MODEL',head), \
                    patch.object(visual_score,'YUNET_EXPECTED_SHA256',sha256(payloads[1][1]).hexdigest()):
                self.assertEqual(owner.installed_native_assets(context),dict(
                    person_model_path=person,pose_model_path=model,head_model_path=head))
                person.write_bytes(b'tampered-detector')
                with self.assertRaisesRegex(ValueError,'pinned native asset unavailable'):
                    owner.installed_native_assets(context)

    def fixture(self):
        args,raw,b,c=numeric.JointGroupSourceCameraTests().fixture()
        args['baseline_path']=raw
        return args,raw,b,c

    def test_default_preserves_unknown_but_opt_in_reaches_coupled_path_without_solo(self):
        args,raw,b,c=self.fixture()
        self.assertEqual(select_joint_group_source_camera_path(**args).status,'baseline_edge_unknown_retained')
        args,raw,b,c=self.fixture();args['baseline_unknown_policy']='observe_candidate'
        result=select_joint_group_source_camera_path(**args)
        self.assertEqual(result.path,(raw[0],b,c,raw[3]))
        self.assertEqual(result.baseline_edge_costs,(None,None,None))
        self.assertTrue(any(a.get('reason')=='baseline_unknown_preserved_not_imputed' for a in result.attempts))

    def test_known_outer_cost_cannot_regress_when_another_incumbent_edge_is_unknown(self):
        args,raw,b,c=self.fixture();original=args['observe_edge']
        def edge(a,d,budget):
            if (a.key,d.key)==(raw[0].key,raw[1].key):
                return JointGroupEdgeObservation(a.key,d.key,d.first_output_frame,.0,True,True)
            result=original(a,d,budget)
            if result and (a.key,d.key)==(raw[0].key,b.key):return replace(result,framing_cost=.01)
            return result
        args.update(observe_edge=edge,baseline_unknown_policy='observe_candidate')
        self.assertIsNone(select_joint_group_source_camera_path(**args).path)

    def test_observable_replacement_still_needs_candidate_motion_and_visibility(self):
        for missing in ('motion_preservation_observed','required_visibility_observed'):
            args,raw,b,c=self.fixture();original=args['observe_edge']
            def edge(a,d,budget):
                result=original(a,d,budget)
                return replace(result,**{missing:False}) if result else None
            args.update(observe_edge=edge,baseline_unknown_policy='observe_candidate')
            with self.subTest(missing=missing):self.assertIsNone(select_joint_group_source_camera_path(**args).path)

    @staticmethod
    def native(*,ambiguous=False,arms=False):
        h=NativeGroupHead(0,(.3,.2,.2,.3),((.35,.3),(.45,.3),(.4,.35),(.35,.4),(.45,.4)),.95)
        heads=(h,replace(h,observation_id=1,box_xywh=(.6,.2,.19,.29))) if ambiguous else (h,)
        p=SimpleNamespace(unique_head_observation_id=0,part_states=(('left_arm','observed'),('right_arm','observed')))
        return SimpleNamespace(heads=heads,people=(p,) if arms else (),frame_size_wh=(1920,1080))

    def test_equal_native_competing_heads_cannot_be_hidden_by_camera(self):
        with self.assertRaisesRegex(ValueError,'ambiguous_not_cropped_away'):
            owner._dominant(self.native(ambiguous=True),GroupOutputRole.FOCAL_HEAD)

    def test_missing_required_arms_and_formation_cannot_be_relabelled(self):
        with self.assertRaisesRegex(ValueError,'parts_unknown'):
            owner._dominant(self.native(),GroupOutputRole.FOCAL_HEAD_ARMS)
        self.assertIsNotNone(owner._dominant(self.native(arms=True),GroupOutputRole.FOCAL_HEAD_ARMS)[1])
        with self.assertRaisesRegex(ValueError,'formation_refresh_not_implemented'):
            owner._dominant(self.native(arms=True),GroupOutputRole.FORMATION)

    def test_budget_exhaustion_does_not_return_partially_observed_coupled_camera(self):
        args,raw,b,c=self.fixture();args['baseline_unknown_policy']='observe_candidate'
        args['budget']=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,maximum_edges=1)
        result=select_joint_group_source_camera_path(**args)
        self.assertIsNone(result.path)
        self.assertEqual(result.status,'budget_retained_baseline')
