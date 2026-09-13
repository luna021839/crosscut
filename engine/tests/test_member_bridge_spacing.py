from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.direct_member_visual import (
    DirectMemberCinematicBridge, DirectMemberVisualConfig, DirectMemberVisualSelectionError,
    FOCAL_POSE_MOTION_BRIDGE_BASIS, _DirectBoundarySearchCandidate,
    _joint_boundary_search_assignment, _heterogeneous_cinematic_visual_path,
    verified_focal_pose_dissolve_spacing,
)
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.direct_member_transition import (
    _RequiredCinematicBridgeCandidate, build_direct_member_pose_dissolve_plan,
    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT, DirectMemberPoseDissolveConfig,
)
from crosscut.member_bridge_contract import (
    RequiredMemberBridgeAuthoringError, required_member_bridge_spacing_seconds,
)
from crosscut.render import RenderSegment, TransitionKind, DIRECT_MEMBER_POSE_DISSOLVE_REASON
from engine.tests.test_direct_member_visual import _optimizer_mismatch, _descriptor
from engine.tests.test_focal_articulation import evidence, window
from engine.tests.test_focal_selector_geometry import sample


def edge(*, focal=True, cost=.3):
    bridge = DirectMemberCinematicBridge(style='pose_dissolve',
        outgoing_is_landscape=False, incoming_is_landscape=False,
        authorized_severe_flags=('attention_scale',), pose_mismatch=.1,
        silhouette_mismatch=None if focal else .2, base_mismatch=.2,
        evidence_basis=FOCAL_POSE_MOTION_BRIDGE_BASIS if focal else 'pose_and_aggregate_silhouette_v1')
    return replace(_optimizer_mismatch(cost, severe=True),
        severe_flags=('attention_scale',), cinematic_bridge=bridge,
        motion_connection=evidence(window()))


def state(cut, mismatch, offset=0):
    samples = lambda time: tuple(sample(source=s, seconds=time) for s in range(3))
    return _DirectBoundarySearchCandidate(offset_frames=offset, cut_master_seconds=cut,
        outgoing_previous=samples(cut-2/29.97), outgoing=samples(cut-1/29.97), incoming=samples(cut),
        incoming_next=samples(cut+1/29.97), incoming_future=samples(cut+.2),
        matrix=((mismatch,) * 3,) * 3)


def assign(first, second, *, gap=1.9):
    return _joint_boundary_search_assignment(candidates_by_boundary=(first, second),
        path=(0, 1, 2), spans=((0., 3.), (3., 3.+gap), (3.+gap, 8.+gap)),
        source_duration_targets=(3., gap, 5.), source_duration_minimums=(0.,)*3,
        source_duration_maximums=(20.,)*3, maximum_same_source_run_seconds=10.,
        usage_policy='heterogeneous_role_aware_cinematic_v1',
        priority_boundary_indices=frozenset((0,1)), config=DirectMemberVisualConfig())


class MemberBridgeSpacingTests(unittest.TestCase):
    def test_only_two_verified_focal_dissolves_use_established_pose_cadence(self):
        self.assertEqual(required_member_bridge_spacing_seconds(True, True), 1.6)
        for flags in ((True,False),(False,True),(False,False)):
            self.assertEqual(required_member_bridge_spacing_seconds(*flags),3.)
        self.assertEqual(required_member_bridge_spacing_seconds(True,True,pose_minimum_seconds=.2),1.6)
        self.assertEqual(required_member_bridge_spacing_seconds(True,True,pose_minimum_seconds=2.),2.)

    def test_spacing_classification_requires_actual_articulation_basis(self):
        good=edge()
        self.assertTrue(verified_focal_pose_dissolve_spacing(good))
        for invalid in (edge(focal=False), replace(good, motion_connection=None),
                replace(good,motion_connection=replace(good.motion_connection,both_moving=False)),
                replace(good,motion_connection=replace(good.motion_connection,direction_cosine_audit_only=-1.)),
                replace(good,motion_connection=replace(good.motion_connection,motion_speed_ratio_mismatch=1.)),
                replace(good,motion_connection=replace(good.motion_connection,
                    motion_basis='aggregate_pose_silhouette_subject_v1')),
                replace(good,cinematic_bridge=replace(edge(focal=False).cinematic_bridge,style='zoom_in'))):
            self.assertFalse(verified_focal_pose_dissolve_spacing(invalid))

    def test_exact_short_pose_states_survive_joint_assignment(self):
        _,_,audit=assign((state(3.,edge()),),(state(4.9,edge()),))
        self.assertTrue(audit['required_effect_spacing_jointly_enforced'])

    def test_short_generic_and_mixed_effects_do_not_survive(self):
        for a,b in ((False,False),(True,False),(False,True)):
            with self.subTest(a=a,b=b),self.assertRaises(DirectMemberVisualSelectionError):
                assign((state(3.,edge(focal=a)),),(state(4.9,edge(focal=b)),))

    def test_under_floor_pose_dissolves_are_rejected_without_render(self):
        with self.assertRaises(DirectMemberVisualSelectionError):
            assign((state(3.,edge()),),(state(4.5,edge()),),gap=1.5)

    def test_local_offset_cannot_silently_violate_spacing(self):
        choices=(state(4.6-.0667334,edge(cost=.01),offset=-2),state(4.6,edge(cost=.5)))
        selected,_,audit=assign((state(3.,edge()),),choices,gap=1.6)
        self.assertEqual(selected[1].offset_frames,0)
        self.assertTrue(audit['solver']['invoked'])
        self.assertGreater(audit['required_effect_spacing_conflict_count'],0)

    def test_generic_offset_pair_keeps_three_second_floor(self):
        choices=(state(5.9332666,edge(focal=False,cost=.01),offset=-2),
                 state(6.,edge(focal=False,cost=.5)))
        selected,_,_=assign((state(3.,edge()),),choices,gap=3.)
        self.assertEqual(selected[1].offset_frames,0)

    def test_missing_solver_fallback_also_checks_effect_spacing(self):
        # Greedy violates spacing; without scipy, the zero-offset state is
        # still checked instead of declaring all fallback assignments safe.
        import builtins
        actual_import=builtins.__import__
        def importing(name,*args,**kwargs):
            if name=='scipy.optimize':raise ImportError('test unavailable solver')
            return actual_import(name,*args,**kwargs)
        choices=(state(4.5,edge(cost=.01)),state(4.53,edge(cost=.5),offset=1))
        with patch('builtins.__import__',side_effect=importing),self.assertRaises(DirectMemberVisualSelectionError):
            assign((state(3.,edge()),),choices,gap=1.5)

    def test_source_milp_and_independent_audit_accept_only_focal_short_pair(self):
        invalid=_optimizer_mismatch(20.,severe=True)
        for left_focal,right_focal in ((True,True),(False,False),(True,False),(False,True)):
            matrices=[[[invalid]*3 for _ in range(3)] for _ in range(2)]
            matrices[0][0][1]=edge(focal=left_focal);matrices[1][1][2]=edge(focal=right_focal)
            args=dict(source_order=(0,1,2),candidate_validity=((True,False,False),
                (False,True,False),(False,False,True)),span_durations=(3.,1.9,3.),
                source_geometries=(direct_member_source_frame_geometry((2160,3840)),)*3,
                priority_cut_boundary_indices=(0,1),maximum_priority_cut_hold_count=0,
                config=DirectMemberVisualConfig(),require_cinematic_bridge=True)
            if left_focal and right_focal:
                result=_heterogeneous_cinematic_visual_path(matrices,**args)
                self.assertEqual(result[0],(0,1,2))
                self.assertTrue(result[2]['solver']['postsolve_path_audit_passed'])
            else:
                with self.assertRaises(DirectMemberVisualSelectionError):
                    _heterogeneous_cinematic_visual_path(matrices,**args)

    def planner(self, *, focal=True, gap=1.9, enabled=True, config=None):
        renders=(RenderSegment(0,0.,3.,0.),RenderSegment(1,3.,gap,3.),RenderSegment(2,3.+gap,3.,3.+gap))
        def candidate(boundary_index,**kwargs):
            return _RequiredCinematicBridgeCandidate(boundary_index=boundary_index,
                canonical_master_seconds=3.+boundary_index*gap,style='pose_dissolve',
                transition_kind=TransitionKind.FACE_DISSOLVE,transition_reason=DIRECT_MEMBER_POSE_DISSOLVE_REASON,
                highlight_priority=True,both_moving=True,attention_basis='verified',
                endpoint_attention_position_mismatch=.1,endpoint_attention_scale_mismatch=.1,
                endpoint_pose_mismatch=.1,endpoint_formation_mismatch=None,endpoint_silhouette_mismatch=None,
                outgoing_handle_available_seconds=20.,required_outgoing_handle_seconds=.25,
                incoming_segment_frame_count=57,transition_frame_count=7,
                verified_focal_articulation=focal),()
        selection=SimpleNamespace(boundaries=(SimpleNamespace(mismatch=edge()),)*2)
        with patch('crosscut.direct_member_transition._required_cinematic_bridge_candidate',side_effect=candidate):
            return build_direct_member_pose_dissolve_plan(visual_selection=selection,render_segments=renders,
                eligible_request_source_indices=(0,1,2),eligible_source_durations_seconds=(50.,)*3,
                common_master_start=0.,source_metadata_contract=OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
                enabled=enabled,config=config)

    def test_actual_planner_spacing_stage_agrees_with_selector(self):
        result=self.planner()
        self.assertEqual([t.kind for t in result.transitions],[TransitionKind.FACE_DISSOLVE]*2)
        for kwargs in ({'focal':False},{'gap':1.5},{'enabled':False},
                {'config':DirectMemberPoseDissolveConfig(minimum_event_spacing_seconds=2.)}):
            with self.subTest(kwargs=kwargs),self.assertRaises(RequiredMemberBridgeAuthoringError):
                self.planner(**kwargs)


if __name__=='__main__':unittest.main()
