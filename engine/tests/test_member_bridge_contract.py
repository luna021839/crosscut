from dataclasses import replace
from types import SimpleNamespace
import unittest

from crosscut.member_bridge_contract import (
    MINIMUM_MEMBER_BRIDGE_HANDLE_QUALITY,
    RequiredMemberBridgeAuthoringError,
    member_bridge_handle_quality_failures,
)
from crosscut.member_bridge_recovery import select_authorable_member_path
from crosscut.direct_member_transition import DirectMemberPoseDissolveConfig
from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, DirectMemberVisualSelectionError,
    _DirectBoundarySearchCandidate, _best_boundary_search_edge,
)
from engine.tests.test_direct_member_visual import _optimizer_mismatch
from engine.tests.test_direct_member_transition import _cinematic_selection, _sample, _plan


class MemberBridgeContractTests(unittest.TestCase):
    def test_same_quality_floor_at_selection_and_authoring(self):
        self.assertEqual(DirectMemberPoseDissolveConfig().minimum_handle_quality_score,
                         MINIMUM_MEMBER_BRIDGE_HANDLE_QUALITY)
        self.assertEqual(member_bridge_handle_quality_failures((.68,.9),label='incoming'),())
        self.assertEqual(member_bridge_handle_quality_failures((.679999,.9),label='incoming'),
                         ('cinematic_bridge_incoming_future_quality_too_low',))

    def test_missing_and_nonfinite_handle_samples_do_not_authorize_effect(self):
        for values in ((None,.9),(float('nan'),.9),(float('inf'),.9),(True,.9)):
            with self.subTest(values=values):
                self.assertTrue(member_bridge_handle_quality_failures(values,label='outgoing'))
        self.assertIn('samples_incomplete',member_bridge_handle_quality_failures((.9,),label='incoming')[0])

    def test_selector_prefers_renderable_offset_before_full_path_search(self):
        mismatch=replace(_optimizer_mismatch(severe=True),
                         cinematic_bridge=SimpleNamespace(style='pose_dissolve'))
        def candidate(offset, quality):
            ordinary=SimpleNamespace(descriptor=SimpleNamespace(quality_score=.9))
            risky=SimpleNamespace(descriptor=SimpleNamespace(quality_score=quality))
            return _DirectBoundarySearchCandidate(offset_frames=offset,cut_master_seconds=4+offset/30,
                outgoing_previous=(),outgoing=(),incoming=(),incoming_next=(ordinary,ordinary),
                incoming_future=(ordinary,risky),matrix=((mismatch,mismatch),(mismatch,mismatch)))
        chosen, result=_best_boundary_search_edge((candidate(0,.5),candidate(2,.9)),
            outgoing_source_index=0,incoming_source_index=1,priority_boundary=True,
            config=DirectMemberVisualConfig())
        self.assertEqual(chosen.offset_frames,2)
        self.assertIsNotNone(result.cinematic_bridge)

    def test_authoring_failure_is_typed_and_keeps_exact_boundary_reason(self):
        selection=_cinematic_selection()
        boundary=selection.boundaries[0]
        a,b=_sample(4+1/30),_sample(4.2)
        boundary.outgoing_handle_samples=(a,b)
        boundary.incoming_handle_samples=(replace(a,descriptor=replace(a.descriptor,quality_score=.5)),b)
        with self.assertRaises(RequiredMemberBridgeAuthoringError) as caught:
            _plan((0,1),selection=selection)
        self.assertIn('cinematic_bridge_incoming_future_quality_too_low',caught.exception.failures[0])
        self.assertFalse(caught.exception.to_dict()['hard_cut_fallback_authorized'])

    def failure(self, time=4.):
        return RequiredMemberBridgeAuthoringError({0:('cinematic_bridge_incoming_future_quality_too_low',)},
            edges_by_boundary={0:(time,0,1)},partition_unit_counts=(3,5,4))

    def test_recovery_keeps_partition_and_exact_edge_with_same_callback(self):
        calls=[]
        events=[]
        expected=object()
        def selector(edges, partition):
            calls.append((edges,partition))
            if len(calls)==1:
                raise self.failure()
            return expected
        result,audit=select_authorable_member_path(selector,rejected_edges=((1.,2,3),),
            preferred_partition_unit_counts=(4,4,4),progress_callback=events.append)
        self.assertIs(result,expected)
        self.assertEqual(calls[1],(((1.,2,3),(4.,0,1)),(3,5,4)))
        self.assertEqual(audit['attempt_count'],2)
        self.assertEqual(len(events),1)
        self.assertTrue(audit['passed'])

    def test_repeated_failed_edge_stops_without_unbounded_reselection(self):
        calls=[]
        def selector(*args):
            calls.append(args)
            raise self.failure()
        with self.assertRaises(DirectMemberVisualSelectionError) as caught:
            select_authorable_member_path(selector)
        self.assertEqual(len(calls),2)
        self.assertEqual(caught.exception.diagnostics['failure'],'required_effect_feedback_made_no_progress')

    def test_recovery_has_small_hard_attempt_limit(self):
        calls=[]
        def selector(*args):
            calls.append(args)
            raise self.failure(float(len(calls)))
        with self.assertRaises(DirectMemberVisualSelectionError) as caught:
            select_authorable_member_path(selector,maximum_attempts=3)
        self.assertEqual(len(calls),3)
        self.assertEqual(caught.exception.diagnostics['failure'],'required_effect_authoring_attempt_budget_exhausted')

    def test_unbound_authoring_exception_cannot_reject_arbitrary_source_pairs(self):
        failure=RequiredMemberBridgeAuthoringError({0:('bad_handle',)})
        with self.assertRaises(RequiredMemberBridgeAuthoringError) as caught:
            select_authorable_member_path(lambda *_:(_ for _ in ()).throw(failure))
        self.assertIs(caught.exception,failure)

    def test_unrelated_runtime_error_is_not_treated_as_retryable_effect(self):
        failure=RuntimeError('decoder disconnected')
        with self.assertRaises(RuntimeError) as caught:
            select_authorable_member_path(lambda *_:(_ for _ in ()).throw(failure))
        self.assertIs(caught.exception,failure)

    def test_effect_feedback_requires_both_exact_partition_and_matching_edges(self):
        with self.assertRaises(ValueError):
            RequiredMemberBridgeAuthoringError({0:('bad_handle',)},edges_by_boundary={0:(4.,0,1)})
        with self.assertRaises(ValueError):
            RequiredMemberBridgeAuthoringError({0:('bad_handle',)},
                edges_by_boundary={1:(4.,0,1)},partition_unit_counts=(3,4))


if __name__=='__main__':
    unittest.main()
