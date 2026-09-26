import unittest
from unittest.mock import Mock

from crosscut.direct_member_visual import (
    DirectBoundaryMismatch, DirectMemberVisualConfig, DirectMemberVisualSelectionError,
    _milp_feasibility_lifeline,
)
from crosscut.member_observation_search import (
    MemberObservationSearch, MemberObservationSearchIncomplete, ObservationState,
    ProposalEdge, TransitionDemand, TransitionObservation,
)


def observed(demand, eligible=True, cost=.2):
    return TransitionObservation(demand, eligible, cost, ('exact-evidence', demand))


class MemberObservationSearchTests(unittest.TestCase):
    def search(self, spans=4, sources=3, observations=20, calls=5):
        return MemberObservationSearch(((True,)*sources,)*spans,
            maximum_observations=observations, maximum_solver_calls=calls)

    def test_only_proposed_edges_are_observed_before_returning_an_incumbent(self):
        search = self.search()
        observer = Mock(side_effect=observed)
        result = search.run(propose=lambda graph: (0, 1, 2, 0), observe=observer)
        self.assertEqual(result.source_indices, (0, 1, 2, 0))
        self.assertEqual(result.observation_count, 3)
        self.assertEqual(result.unobserved_transition_count, 15)
        self.assertTrue(all(item.eligible for item in result.transitions))
        self.assertEqual(observer.call_count, 3)
        self.assertIsInstance(search.proposal_graph()[0][0][1], ProposalEdge)
        self.assertNotIsInstance(search.proposal_graph()[0][0][1], DirectBoundaryMismatch)
        self.assertEqual(search.proposal_graph()[0][1][2].state, ObservationState.UNOBSERVED)

    def test_failure_repairs_path_without_observing_its_tail_or_repeating_good_edges(self):
        search = self.search()
        proposals = iter(((0, 1, 2, 0), (0, 1, 0, 2)))
        queries = []
        failed = TransitionDemand(1, 1, 2)
        def observe(demand):
            queries.append(demand)
            return observed(demand, eligible=demand != failed)
        def propose(graph):
            if queries:
                self.assertEqual(graph[1][1][2].state, ObservationState.REJECTED)
                self.assertEqual(graph[2][2][0].state, ObservationState.UNOBSERVED)
                self.assertLess(graph[0][0][1].cost, graph[0][2][1].cost)
            return next(proposals)
        result = search.run(propose=propose, observe=observe)
        self.assertEqual(result.source_indices, (0, 1, 0, 2))
        self.assertEqual(len(queries), 4)
        self.assertEqual(queries.count(TransitionDemand(0, 0, 1)), 1)
        self.assertNotIn(TransitionDemand(2, 2, 0), queries)

    def test_budget_exhaustion_is_inconclusive_and_never_a_partition_retry_signal(self):
        search = self.search(observations=1)
        observer = Mock(side_effect=observed)
        with self.assertRaises(MemberObservationSearchIncomplete) as caught:
            search.run(propose=lambda _: (0, 1, 2, 0), observe=observer)
        error = caught.exception
        self.assertNotIsInstance(error, DirectMemberVisualSelectionError)
        self.assertFalse(error.diagnostics['constraint_infeasibility_proven'])
        self.assertFalse(error.diagnostics['retry_as_another_partition'])
        self.assertFalse(error.diagnostics['render_authorized'])
        self.assertEqual(error.diagnostics['observation_count'], 1)
        observer.assert_called_once()

    def test_no_path_is_not_full_visual_infeasibility_even_without_observations(self):
        observer = Mock()
        with self.assertRaises(MemberObservationSearchIncomplete) as caught:
            self.search().run(propose=lambda _: None, observe=observer)
        self.assertFalse(caught.exception.diagnostics['constraint_infeasibility_proven'])
        observer.assert_not_called()

    def test_same_failed_transition_cannot_enter_a_retry_loop(self):
        observer = Mock(side_effect=lambda demand: observed(demand, False))
        with self.assertRaisesRegex(ValueError, 'repeats an observed rejected'):
            self.search().run(propose=lambda _: (0, 1, 2, 0), observe=observer)
        observer.assert_called_once()

    def test_source_clock_authority_errors_and_cancel_escape_without_retry(self):
        for failure in (OSError('SSD'), RuntimeError('cancel'), KeyboardInterrupt()):
            observer = Mock(side_effect=failure)
            with self.assertRaises(type(failure)):
                self.search().run(propose=lambda _: (0, 1, 2, 0), observe=observer)
            observer.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'different transition'):
            self.search().run(propose=lambda _: (0, 1, 2, 0),
                observe=lambda _: observed(TransitionDemand(0, 2, 1)))

    def test_proposer_must_honor_domain_and_no_hold_policy(self):
        for path in ((0, 1), (0, True, 2, 0), (0, 1, 3, 0), (0, 0, 1, 2)):
            observer = Mock()
            with self.assertRaises(ValueError):
                self.search().run(propose=lambda _, path=path: path, observe=observer)
            observer.assert_not_called()
        search = MemberObservationSearch(((True, False), (True, True)),
            maximum_observations=2, maximum_solver_calls=1)
        self.assertEqual(search.proposal_graph()[0][1][0].state, ObservationState.CONTENT_INELIGIBLE)
        with self.assertRaises(ValueError):
            search.run(propose=lambda _: (1, 0), observe=observed)

    def test_finite_solver_call_budget_cannot_reset_by_reentering_run(self):
        search = self.search(calls=1)
        observer = Mock(side_effect=lambda demand: observed(demand, False))
        with self.assertRaises(MemberObservationSearchIncomplete):
            search.run(propose=lambda _: (0, 1, 2, 0), observe=observer)
        proposer = Mock()
        with self.assertRaises(MemberObservationSearchIncomplete):
            search.run(propose=proposer, observe=observer)
        proposer.assert_not_called()
        observer.assert_called_once()

    def test_actual_path_milp_retains_quotas_durations_and_complete_path_no_good(self):
        # Exercise the existing solver, not a mock optimizer, over provisional
        # edge types. This is a synthetic policy/adapter check, not media QA.
        search = self.search(spans=6, sources=3, observations=30)
        durations = (1.,)*6
        forbidden = ((0, 1, 2, 0, 1, 2),)
        def propose(graph):
            result = _milp_feasibility_lifeline(graph, source_quotas=(2, 2, 2),
                source_order=(0, 1, 2), candidate_validity=search.validity,
                durations=durations, duration_targets=(2.,)*3, state_limit=10000,
                config=DirectMemberVisualConfig(), forbidden_source_paths=forbidden)
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.node)
            return result.node.path
        result = search.run(propose=propose, observe=observed)
        self.assertEqual(tuple(result.source_indices.count(source) for source in range(3)), (2, 2, 2))
        self.assertNotIn(result.source_indices, forbidden)
        self.assertEqual(result.observation_count, 5)
        self.assertEqual(result.unobserved_transition_count, 25)

    def test_invalid_budget_indices_and_observation_cannot_impersonate_evidence(self):
        for value in (0, -1, True, 1.2):
            with self.assertRaises(ValueError): self.search(observations=value)
            with self.assertRaises(ValueError): self.search(calls=value)
        with self.assertRaises(ValueError): TransitionDemand(True, 0, 1)
        demand = TransitionDemand(0, 0, 1)
        for cost in (True, float('nan'), float('inf')):
            with self.assertRaises(ValueError): observed(demand, cost=cost)
        with self.assertRaises(ValueError): TransitionObservation(demand, True, .2, None)


if __name__ == '__main__': unittest.main()
