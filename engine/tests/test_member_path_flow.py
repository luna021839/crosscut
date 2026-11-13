"""Independent equivalence checks for source-integral / edge-continuous MILPs."""
from itertools import product
import random
import unittest

import numpy as np
from scipy.optimize import linprog

from crosscut.member_path_flow import member_path_flow_rows
from crosscut.direct_member_visual import DirectMemberVisualConfig, _milp_feasibility_lifeline
from engine.tests.test_direct_member_visual import _optimizer_mismatch


class MemberPathFlowTests(unittest.TestCase):
    def test_every_integer_source_path_has_exactly_its_edges_or_is_infeasible(self):
        # Includes holds, absent edges, dead ends and all source combinations.
        for source_count in (2, 3):
            spans = 4
            edges = tuple((b, left, right) for b in range(spans-1)
                for left in range(source_count) for right in range(source_count)
                if (b, left, right) not in {(0, 1, 0), (1, 0, 0), (2, 1, 1)})
            base = spans*source_count
            rows = tuple(member_path_flow_rows(edges, source_count=source_count,
                boundary_count=spans-1, edge_base=base))
            matrix = np.zeros((len(rows), base+len(edges)))
            for row, terms in enumerate(rows):
                for index, value in terms:
                    matrix[row, index] = value
            for path in product(range(source_count), repeat=spans):
                with self.subTest(sources=source_count, path=path):
                    chosen = {(b, path[b], path[b+1]) for b in range(spans-1)}
                    bounds = [(int(path[position] == source),)*2
                        for position in range(spans) for source in range(source_count)]
                    result = linprog(np.zeros(base+len(edges)), A_eq=matrix,
                        b_eq=np.zeros(len(rows)), bounds=bounds+[(0., 1.)]*len(edges), method='highs')
                    allowed = chosen.issubset(set(edges))
                    self.assertEqual(result.success, allowed)
                    if allowed:
                        np.testing.assert_allclose(result.x[base:],
                            [int(edge in chosen) for edge in edges], atol=1e-9)

    def test_fractional_source_mass_cannot_be_duplicated_over_multiple_edges(self):
        edges = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1))
        solution = (.5, .5, .5, .5, .5, .5, 0., 0.)
        # All former edge <= left/right rows and the edge-one-hot sum pass.
        self.assertEqual(sum(solution[4:]), 1.)
        self.assertTrue(all(solution[4+i] <= min(solution[left], solution[2+right])
                            for i, (_boundary, left, right) in enumerate(edges)))
        residuals = [sum(solution[index]*value for index, value in row)
            for row in member_path_flow_rows(edges, source_count=2, boundary_count=1, edge_base=4)]
        self.assertGreater(max(abs(value) for value in residuals), .49)

    def test_parallel_effect_edges_are_not_silently_fractionalized(self):
        with self.assertRaisesRegex(ValueError, 'one edge per ordered source pair'):
            tuple(member_path_flow_rows(((0, 0, 1, 'hard'), (0, 0, 1, 'dissolve')),
                source_count=2, boundary_count=1, edge_base=4))
        with self.assertRaisesRegex(ValueError, 'outside'):
            tuple(member_path_flow_rows(((2, 0, 1),),
                source_count=2, boundary_count=1, edge_base=4))

    def test_real_measured_milp_matches_exhaustive_quota_duration_and_severe_budget(self):
        rng = random.Random(791)
        spans, sources = 6, 3
        durations = (1., 1.4, .6, 1.2, .8, 1.)
        targets = (2.,)*sources
        quotas = (2,)*sources
        validity = ((True, True, False),) + ((True,)*sources,)*(spans-1)
        forbidden = ((0, 1, 2, 0, 1, 2),)
        config = DirectMemberVisualConfig()
        matrices = tuple(tuple(tuple(_optimizer_mismatch(rng.uniform(.01, .6),
            severe=(b, left, right) in {(1, 0, 2), (2, 2, 1), (4, 1, 0)})
            for right in range(sources)) for left in range(sources)) for b in range(spans-1))
        for budget in (None, 0, 1, 2):
            mask = None if budget is None else tuple(tuple(tuple(bool(edge.severe_flags)
                for edge in row) for row in matrix) for matrix in matrices)
            expected = []
            for path in product(range(sources), repeat=spans):
                if path in forbidden or tuple(path.count(s) for s in range(sources)) != quotas:
                    continue
                if any(not validity[i][s] for i, s in enumerate(path)) or any(a == b for a, b in zip(path, path[1:])):
                    continue
                selected = [matrices[i][a][b] for i, (a, b) in enumerate(zip(path, path[1:]))]
                severe_count = sum(bool(edge.severe_flags) for edge in selected)
                if (budget is None and severe_count) or (budget is not None and severe_count != budget):
                    continue
                totals = tuple(sum(d for d, s in zip(durations, path) if s == source) for source in range(sources))
                if any(abs(actual-target)/target > config.maximum_source_duration_relative_deviation+1e-9
                       for actual, target in zip(totals, targets)):
                    continue
                objective = (sum(edge.cost for edge in selected)
                    + config.ping_pong_penalty*sum(path[i] == path[i-2] for i in range(2, spans))
                    + sum(config.duration_balance_weight*abs(a-t)/t for a, t in zip(totals, targets))
                    + sum(path)*1e-9 + sum(path[1:])*1e-8)
                expected.append((objective, path))
            result = _milp_feasibility_lifeline(matrices, source_quotas=quotas,
                source_order=tuple(range(sources)), candidate_validity=validity,
                durations=durations, duration_targets=targets, state_limit=10000, config=config,
                severe_transition_budget_mask=mask, minimum_severe_transition_count=budget,
                maximum_severe_transition_count=budget, forbidden_source_paths=forbidden)
            with self.subTest(budget=budget):
                self.assertEqual(result.node is not None, bool(expected))
                self.assertEqual(result.solver_diagnostics['binary_variable_count'], spans*sources)
                if expected:
                    self.assertIn(result.node.path, [path for _, path in expected])
                    self.assertAlmostEqual(result.solver_diagnostics['objective_value'], min(expected)[0], places=7)
                    self.assertTrue(result.solver_diagnostics['postsolve_path_audit_passed'])
                else:
                    self.assertEqual(result.solver_diagnostics['status'], 2)


if __name__ == '__main__':
    unittest.main()
