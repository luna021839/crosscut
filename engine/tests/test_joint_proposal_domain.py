from dataclasses import dataclass, replace
from unittest import TestCase

from crosscut.joint_proposal_domain import (
    JointProposalPool, proposal_pool_from_results, select_with_joint_proposal_domains,
    build_joint_proposal_pool, proposal_only_config,
    add_local_proposal_detour,
    local_proposal_added_pair_budget,
)
from crosscut.joint_timeline import BoundaryMeasurementKey, select_joint_timeline_candidates
from crosscut.timeline import NoCompleteTimelinePath, TimelineChoice, TimelineMode, TimelineResult
from crosscut.visual_score import VisualTimelineOptimizerWorkspace, VisualDiversityConfig, _safe_cadence_profiles
from engine.tests.test_joint_timeline import (
    candidate, isolated_score_config, option_set, BatchedBridgeProvider, selection_scalar_option_set,
    valid_bridge,
)


@dataclass(frozen=True)
class Result:
    selection_retry_audit: dict


def pool_fixture():
    lattice = tuple(tuple(candidate(i, str(s), .5) for s in range(8)) for i in range(2))
    paths = tuple((lattice[0][i % 8], lattice[1][(i + 1) % 8]) for i in range(8))
    return lattice, JointProposalPool(paths, ('balanced',) * len(paths), 64)


class JointProposalDomainTests(TestCase):
    def test_unsearched_pairs_never_call_exact_provider(self):
        lattice, pool = pool_fixture()
        domain = pool.domain(3)
        measured = []
        def provider(key):
            self.assertIn(key, domain)
            measured.append(key)
            return option_set(key, authorized={0: 0.0})
        results = select_joint_timeline_candidates(
            lattice, option_provider=provider, config=isolated_score_config(),
            maximum_results=1, boundary_search_domain=domain,
        )
        self.assertTrue(results)
        self.assertEqual(set(measured), domain)
        self.assertEqual(len(measured), 3)

    def test_exact_failure_does_not_authorize_a_proposal(self):
        lattice, pool = pool_fixture()
        with self.assertRaises(NoCompleteTimelinePath):
            select_joint_timeline_candidates(
                lattice, option_provider=lambda key: option_set(key),
                config=isolated_score_config(), maximum_results=1,
                boundary_search_domain=pool.domain(3),
            )

    def test_domain_rejects_changed_clock_before_measurement(self):
        lattice, pool = pool_fixture()
        key = next(iter(pool.domain(1)))
        wrong = replace(key, outgoing=replace(key.outgoing, start_seconds=.02))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            select_joint_timeline_candidates(
                lattice, option_provider=lambda key: self.fail('unexpected measurement'),
                config=isolated_score_config(), maximum_results=1,
                boundary_search_domain=(wrong,),
            )

    def test_empty_domain_is_not_an_exhaustive_fallback(self):
        lattice, _ = pool_fixture()
        with self.assertRaises(NoCompleteTimelinePath):
            select_joint_timeline_candidates(
                lattice, option_provider=lambda key: self.fail('unexpected measurement'),
                config=isolated_score_config(), maximum_results=1,
                boundary_search_domain=(),
            )

    def test_failed_domain_expands_with_shared_remaining_proof_budget(self):
        _, pool = pool_fixture()
        requests = []
        def select(**kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                raise NoCompleteTimelinePath('local failure', reason_code='test',
                    rejection_counts={'proof_materialization_call_count': 2})
            return Result({'proof_materialization_call_count': 1})
        result = select_with_joint_proposal_domains(pool, maximum_proof_calls=4, select=select)
        self.assertEqual([r['maximum_proof_materialization_calls'] for r in requests], [4, 2])
        self.assertLess(requests[0]['boundary_search_domain'], requests[1]['boundary_search_domain'])
        self.assertEqual(result.selection_retry_audit['proof_materialization_call_count'], 3)
        self.assertFalse(result.selection_retry_audit['proposal_search_domain']['quality_gates_waived'])

    def test_exhaustion_is_finite_and_never_replenishes_budget(self):
        _, pool = pool_fixture()
        requests = []
        def select(**kwargs):
            requests.append(kwargs)
            raise NoCompleteTimelinePath('failed', reason_code='test',
                rejection_counts={'proof_materialization_call_count': 1})
        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_with_joint_proposal_domains(pool, maximum_proof_calls=2, select=select)
        self.assertEqual(len(requests), 2)
        self.assertEqual(caught.exception.reason_code, 'joint_proposal_domains_exhausted')
        self.assertEqual(caught.exception.rejection_counts['proof_materialization_call_count'], 2)

    def test_storage_error_does_not_trigger_proposal_expansion(self):
        _, pool = pool_fixture()
        calls = []
        def select(**kwargs):
            calls.append(kwargs)
            raise OSError('SSD disconnected')
        with self.assertRaises(OSError):
            select_with_joint_proposal_domains(pool, maximum_proof_calls=4, select=select)
        self.assertEqual(len(calls), 1)

    def test_proposals_must_cover_every_original_segment(self):
        lattice, _ = pool_fixture()
        def result(path):
            return TimelineResult(TimelineMode.GENERAL, None, tuple(
                TimelineChoice(c, .5, 0, .5, False) for c in path), 1.)
        first = result((lattice[0][0], lattice[1][1]))
        pool = proposal_pool_from_results(lattice, (first, first), ('balanced', 'dense_safe'))
        self.assertEqual(len(pool.paths), 1)
        with self.assertRaisesRegex(ValueError, 'complete lattice'):
            proposal_pool_from_results(lattice, (result((lattice[0][0],)),), ('balanced',))

    def test_repeated_same_domain_is_not_retried(self):
        lattice, pool = pool_fixture()
        pool = replace(pool, paths=pool.paths[:1], cadence_profiles=('balanced',))
        requests = []
        def select(**kwargs):
            requests.append(kwargs)
            raise NoCompleteTimelinePath('failed', reason_code='test')
        with self.assertRaises(NoCompleteTimelinePath):
            select_with_joint_proposal_domains(pool, maximum_proof_calls=8, select=select)
        self.assertEqual(len(requests), 1)

    def test_batched_bridge_priming_cannot_decode_outside_domain(self):
        lattice = tuple(tuple(candidate(i, str(s), .5) for s in range(4)) for i in range(3))
        pool = JointProposalPool(
            ((lattice[0][0], lattice[1][1], lattice[2][2]),
             (lattice[0][3], lattice[1][2], lattice[2][1])), ('balanced', 'reference'), 32,
        )
        domain = pool.domain(2)
        measured = []
        def provider(key):
            self.assertIn(key, domain)
            measured.append(key)
            return option_set(key, authorized={0: 0.0})
        bridge = BatchedBridgeProvider()
        results = select_joint_timeline_candidates(
            lattice, option_provider=provider, config=isolated_score_config(), maximum_results=1,
            boundary_search_domain=domain, bridge_provider=bridge.bridge_provider,
            bridge_disposition_batch_provider=bridge.bridge_disposition_batch,
        )
        self.assertTrue(results)
        self.assertEqual(set(measured), domain)
        self.assertEqual(len(measured), 4)

    def test_selected_proposal_still_needs_exact_rehydration(self):
        lattice, pool = pool_fixture()
        calls = []
        def rehydrate(option):
            calls.append(option)
            return None
        with self.assertRaises(NoCompleteTimelinePath):
            select_joint_timeline_candidates(
                lattice, option_provider=lambda key: selection_scalar_option_set(key, authorized={0: 0.0}),
                option_rehydrator=rehydrate, config=isolated_score_config(), maximum_results=1,
                boundary_search_domain=pool.domain(1),
            )
        self.assertEqual(len(calls), 1)

    def test_pretransform_proposals_do_not_mutate_exact_quality_config(self):
        strict = _safe_cadence_profiles(VisualDiversityConfig())[0].config
        proposal = proposal_only_config(strict)
        self.assertIsNotNone(strict.maximum_cross_source_scale_delta)
        self.assertIsNone(proposal.maximum_cross_source_scale_delta)
        self.assertTrue(strict.require_cross_source_boundary_side_stability)
        self.assertFalse(proposal.require_cross_source_boundary_side_stability)
        self.assertEqual(strict.minimum_member_confidence, proposal.minimum_member_confidence)
        self.assertEqual(strict.maximum_same_source_hold_seconds, proposal.maximum_same_source_hold_seconds)

    def test_real_descriptor_proposer_returns_complete_bounded_paths(self):
        lattice = tuple(tuple(candidate(i, str(s), .5) for s in range(4)) for i in range(4))
        workspace = VisualTimelineOptimizerWorkspace(lattice, 'a' * 64)
        pool = build_joint_proposal_pool(workspace)
        self.assertLessEqual(len(pool.paths), 9)
        self.assertTrue(pool.paths)
        self.assertTrue(all(len(path) == len(lattice) for path in pool.paths))
        continuation_count = (len(lattice) - 1) * len(lattice[0])
        self.assertLessEqual(len(pool.domain(3)), 3 * (len(lattice) - 1) + continuation_count)

    def test_local_failure_opens_boundary_then_adjacent_detours_only(self):
        lattice = tuple(tuple(candidate(i, str(s), .5) for s in range(3)) for i in range(6))
        pool = JointProposalPool((tuple(row[0] for row in lattice),), ('balanced',), 45, lattice)
        original = pool.domain(1)
        failure = NoCompleteTimelinePath('blocked', reason_code='test', segment_index=3)
        widened, audit = add_local_proposal_detour(pool, original, failure)
        self.assertLess(original, widened)
        self.assertEqual(audit['touched_boundary_indices'], [2])
        self.assertEqual(audit['repair_scope'], 'failed_boundary_only')
        self.assertEqual({key for key in widened if key.boundary_index != 2},
                         {key for key in original if key.boundary_index != 2})
        # If that boundary still cannot connect, a second local attempt can
        # change its neighbor without opening unrelated parts of the song.
        widened, audit = add_local_proposal_detour(pool, widened, failure)
        self.assertEqual(audit['touched_boundary_indices'], [1, 3])
        self.assertEqual(audit['repair_scope'], 'adjacent_reconnection')
        self.assertEqual({key for key in widened if key.boundary_index in (0, 4)},
                         {key for key in original if key.boundary_index in (0, 4)})
        # A replacement can enter the blocked area and rejoin the original path.
        self.assertIn(BoundaryMeasurementKey(1, lattice[1][0], lattice[2][2]), widened)
        self.assertIn(BoundaryMeasurementKey(2, lattice[2][2], lattice[3][0]), widened)
        self.assertFalse(audit['hard_no_goods_authored'])

    def test_boundary_first_has_same_exact_result_with_fewer_measured_pairs(self):
        lattice = tuple(tuple(candidate(i,str(s),.9 if s == (0 if i < 3 else 1) else .5)
                              for s in range(8)) for i in range(6))
        pool = JointProposalPool((tuple(row[0] for row in lattice),), ('balanced',), 320, lattice)
        failure = NoCompleteTimelinePath('hold cap',reason_code='test',segment_index=3)
        direct, _ = add_local_proposal_detour(pool,pool.domain(1),failure)
        adjacent, _ = add_local_proposal_detour(pool,direct,failure)
        counts, results = [], []
        for domain in (direct,adjacent):
            measured=[]
            def provider(key):
                measured.append(key)
                matched = key.outgoing.source_id == key.incoming.source_id or key.boundary_index == 2
                return option_set(key,authorized={0:0.} if matched else {})
            results.append(select_joint_timeline_candidates(lattice,option_provider=provider,
                bridge_provider=valid_bridge,maximum_results=1,boundary_search_domain=domain,
                config=replace(isolated_score_config(),maximum_same_source_hold_seconds=3.01))[0].timeline)
            counts.append(len(measured))
        self.assertEqual(results[0],results[1])
        self.assertEqual([c.source_id for c in results[0].candidates],['0']*3+['1']*3)
        self.assertEqual(counts,[96,208])

    def test_small_local_repairs_can_reach_later_song_without_more_pair_budget(self):
        lattice = tuple(tuple(candidate(i,str(s),.5) for s in range(2)) for i in range(30))
        pool = JointProposalPool((tuple(row[0] for row in lattice),),('balanced',),116,lattice)
        calls=[]
        def select(**kwargs):
            calls.append(kwargs)
            if len(calls) <= 7:
                raise NoCompleteTimelinePath('next frontier',reason_code='test',segment_index=len(calls))
            return Result({'proof_materialization_call_count':1})
        result = select_with_joint_proposal_domains(pool,maximum_proof_calls=2,select=select)
        audit = result.selection_retry_audit['proposal_search_domain']
        self.assertEqual(len(calls),8)
        self.assertEqual(audit['local_added_boundary_pair_count'],14)
        self.assertEqual(audit['maximum_local_added_boundary_pairs'],36)
        self.assertTrue(all(call['maximum_proof_materialization_calls']==2 for call in calls))

    def test_local_pair_budget_cannot_be_replenished_by_more_proposals(self):
        lattice = tuple(tuple(candidate(i,str(s),.5) for s in range(2)) for i in range(30))
        # Each larger pool really contributes new complete-path edges near the
        # end, so this exercises all 3/6/9 sizes rather than identical domains.
        paths = tuple(tuple(row[int(i == 21+j)] for i,row in enumerate(lattice)) for j in range(9))
        pool = JointProposalPool(paths,('balanced',)*9,116,lattice)
        calls=[]
        def select(**kwargs):
            calls.append(kwargs)
            raise NoCompleteTimelinePath('next frontier',reason_code='test',segment_index=len(calls))
        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_with_joint_proposal_domains(pool,maximum_proof_calls=2,select=select)
        self.assertEqual(local_proposal_added_pair_budget(pool),36)
        self.assertEqual(len(calls),21)
        self.assertLess(calls[-3]['boundary_search_domain'],calls[-2]['boundary_search_domain'])
        self.assertLess(calls[-2]['boundary_search_domain'],calls[-1]['boundary_search_domain'])
        self.assertEqual(caught.exception.rejection_counts['local_added_boundary_pair_count'],36)
        self.assertEqual(caught.exception.rejection_counts['proof_materialization_call_count'],0)

    def test_nested_graph_failure_repairs_before_unrelated_full_paths_expand(self):
        lattice = tuple(tuple(candidate(i, str(s), .5) for s in range(3)) for i in range(4))
        pool = JointProposalPool((tuple(row[0] for row in lattice),), ('balanced',), 27, lattice)
        calls = []
        def select(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                child = NoCompleteTimelinePath('blocked', reason_code='test', segment_index=2)
                raise NoCompleteTimelinePath('profile failed', reason_code='profile', profile_failures=(('balanced', child),))
            return Result({'proof_materialization_call_count': 1})
        result = select_with_joint_proposal_domains(pool, maximum_proof_calls=2, select=select)
        self.assertEqual(len(calls), 2)
        self.assertLess(calls[0]['boundary_search_domain'], calls[1]['boundary_search_domain'])
        self.assertEqual(result.selection_retry_audit['proof_materialization_call_count'], 1)
        self.assertEqual(len(result.selection_retry_audit['proposal_search_domain']['local_repairs']), 1)

    def test_global_failure_cannot_expand_every_boundary_as_a_detour(self):
        lattice = tuple(tuple(candidate(i, str(s), .5) for s in range(3)) for i in range(4))
        pool = JointProposalPool((tuple(row[0] for row in lattice),), ('balanced',), 27, lattice)
        self.assertIsNone(add_local_proposal_detour(pool, pool.domain(1),
            NoCompleteTimelinePath('global review', reason_code='test')))

    def test_continuation_reaches_a_later_matched_cut_without_relaxing_hold_cap(self):
        lattice = tuple(tuple(candidate(i, source, .9 if (source == 'A') == (i < 4) else .6)
                              for source in ('A', 'B')) for i in range(6))
        paths = tuple(tuple(lattice[i][s] for i, s in enumerate(path))
                      for path in ((0,0,1,1,0,0), (1,1,0,0,1,1)))
        fixed = JointProposalPool(paths, ('balanced', 'reference'), 20)
        flexible = replace(fixed, candidate_lattice=lattice)
        config = replace(isolated_score_config(), maximum_same_source_hold_segments=4)
        def provider(key):
            matched = key.outgoing.source_id == key.incoming.source_id or key.boundary_index == 3
            return option_set(key, authorized={0:0.0} if matched else {})
        def select(domain):
            return select_joint_timeline_candidates(lattice, option_provider=provider,
                bridge_provider=valid_bridge, boundary_search_domain=domain,
                config=config, maximum_results=1)
        with self.assertRaises(NoCompleteTimelinePath):
            select(fixed.domain(3))
        selected = select(flexible.domain(3))[0].timeline.candidates
        self.assertEqual([c.source_id for c in selected], ['A','A','A','A','B','B'])
        self.assertFalse(flexible.scalar_audit()['final_media_authority'])
