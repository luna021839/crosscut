'''Boundary-port DP against an exhaustive oracle and its input contract.'''
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from crosscut2.boundary_port_dp import solve  # noqa: E402

MINIMUM = 12


def _graph(seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, 6))
    nodes = np.r_[0, np.cumsum(rng.integers(4, 11, count - 1))]
    ent = [rng.integers(0, 3, int(rng.integers(1 if i == 0 else 0, 4))) for i in range(count)]
    ext = [rng.integers(0, 3, int(rng.integers(1 if i == count - 1 else 0, 4))) for i in range(count)]
    start = rng.integers(0, 3, len(ent[0])).astype(float)
    holds = {}
    for i in range(count):
        for j in range(i + 1, count):
            shape = (len(ent[i]), len(ext[j]))
            holds[i, j] = (rng.integers(0, 5, shape).astype(float), rng.random(shape) < .8)
    cuts = {}
    for j in range(1, count - 1):
        shape = (len(ext[j]), len(ent[j]))
        cuts[j] = (rng.integers(0, 5, shape).astype(float), rng.random(shape) < .8)
    return nodes, ent, ext, start, holds, cuts


def _brute(nodes, ent, ext, start, holds, cuts):
    last = len(nodes) - 1
    best = None

    def visit(i, p, primary, count):
        nonlocal best
        for j in range(i + 1, last + 1):
            if nodes[j] - nodes[i] < MINIMUM and j != last:
                continue
            cost, mask = holds[i, j]
            for q in range(len(ext[j])):
                if not mask[p, q] or ent[i][p] != ext[j][q]:
                    continue
                total = primary + cost[p, q]
                if j == last:
                    if best is None or (total, count) < best:
                        best = (total, count)
                    continue
                kc, km = cuts[j]
                for r in range(len(ent[j])):
                    if km[q, r] and ext[j][q] != ent[j][r]:
                        visit(j, r, total + kc[q, r], count + 1)

    for p in range(len(ent[0])):
        visit(0, p, start[p], 0)
    return best


def _evaluate(runs, nodes, ent, ext, start, holds, cuts):
    last = len(nodes) - 1
    assert runs[0][0] == 0 and runs[-1][1] == last
    primary, count = float(start[runs[0][2]]), 0
    for k, (i, j, p, q) in enumerate(runs):
        assert i < j and (j == last or nodes[j] - nodes[i] >= MINIMUM)
        cost, mask = holds[i, j]
        assert mask[p, q] and ent[i][p] == ext[j][q]
        primary += cost[p, q]
        if k + 1 < len(runs):
            following, _, r, _ = runs[k + 1]
            assert following == j
            kc, km = cuts[j]
            assert km[q, r] and ext[j][q] != ent[j][r]
            primary += kc[q, r]
            count += 1
    return float(primary), count


def _solve(nodes, ent, ext, start, holds, cuts, **extra):
    return solve(nodes, ent, ext, start_cost=start, hold_cost=lambda i, j: holds[i, j],
                 cut_cost=lambda j: cuts[j], minimum_frames=MINIMUM, **extra)


def test_exhaustive_oracle_on_random_variable_port_graphs():
    feasible = 0
    for seed in range(300):
        case = _graph(seed)
        expected = _brute(*case)
        if expected is None:
            with pytest.raises(ValueError, match='no complete'):
                _solve(*case)
            continue
        feasible += 1
        runs, score, receipt = _solve(*case)
        assert score == (float(expected[0]), expected[1])
        assert _evaluate(runs, *case) == score
        assert receipt['arcs_stored'] is False
        assert receipt['peak_workspace_bytes'] <= receipt['workspace_limit_bytes']
    assert feasible >= 60


def _endpoint_case():
    holds = {(0, 1): (np.array([[0., 5.], [5., 0.]]), np.ones((2, 2), bool)),
             (0, 2): (np.zeros((2, 1)), np.ones((2, 1), bool)),
             (1, 2): (np.array([[2.]]), np.ones((1, 1), bool))}
    cuts = {1: (np.array([[10.], [0.]]), np.ones((2, 1), bool))}
    return [0, 12, 24], [[0, 0], [1], []], [[], [0, 0], [1]], np.array([0., 1.]), holds, cuts


def test_endpoint_conditioned_optimum_needs_exit_indexed_states():
    '''Entry port 1 is worse alone (start 1 vs 0), but only its matching exit
    avoids the expensive cut.  A DP keeping one best predecessor per
    (node, source), independent of the exit it serves, starts at port 0 and
    pays 7 or 12.  The cheap cross-source 0->2 hold is masked internally.'''
    case = _endpoint_case()
    runs, score, receipt = _solve(*case)
    assert runs == ((0, 1, 1, 1), (1, 2, 0, 0))
    assert score == (3.0, 1)
    assert _brute(*case) == (3.0, 1)
    assert receipt['hold_calls'] == 3 and receipt['cut_calls'] == 1


def _tie_case(mask_through):
    holds = {(0, 1): (np.array([[1.]]), np.ones((1, 1), bool)),
             (0, 2): (np.array([[2., 0.]]), np.array([[mask_through, True]])),
             (1, 2): (np.array([[0., 1.]]), np.ones((1, 2), bool))}
    cuts = {1: (np.array([[0.]]), np.ones((1, 1), bool))}
    return [0, 12, 24], [[0], [1], []], [[], [0], [0, 1]], np.array([0.]), holds, cuts


def test_equal_primary_prefers_fewer_cuts_without_multiplier():
    runs, score, _ = _solve(*_tie_case(True))
    assert runs == ((0, 2, 0, 0),) and score == (2.0, 0)


def test_masked_hold_is_never_selected():
    runs, score, _ = _solve(*_tie_case(False))
    assert runs == ((0, 1, 0, 0), (1, 2, 0, 1)) and score == (2.0, 1)


def test_same_source_cut_is_never_admitted_even_if_mask_allows_it():
    holds = {(0, 1): (np.zeros((1, 1)), np.ones((1, 1), bool)),
             (0, 2): (np.zeros((1, 1)), np.zeros((1, 1), bool)),
             (1, 2): (np.zeros((1, 1)), np.ones((1, 1), bool))}
    cuts = {1: (np.zeros((1, 1)), np.ones((1, 1), bool))}
    with pytest.raises(ValueError, match='no complete'):
        _solve([0, 12, 24], [[0], [0], []], [[], [0], [0]], np.zeros(1), holds, cuts)


def test_unreachable_states_stay_internal_and_skip_callbacks():
    holds = {(0, 1): (np.zeros((1, 1)), np.ones((1, 1), bool)),
             (0, 2): (np.array([[4.]]), np.ones((1, 1), bool))}
    cuts = {1: (np.zeros((1, 1)), np.zeros((1, 1), bool))}
    runs, score, receipt = _solve([0, 12, 24], [[0], [1], []], [[], [0], [0]],
                                  np.zeros(1), holds, cuts)
    assert runs == ((0, 2, 0, 0),) and score == (4.0, 0)
    assert receipt['unreachable_pairs_skipped'] == 1


def test_minimum_hold_applies_except_final():
    holds = {(0, 1): (np.zeros((1, 1)), np.ones((1, 1), bool)),
             (0, 2): (np.array([[9., 0.]]), np.ones((1, 2), bool)),
             (1, 2): (np.zeros((1, 2)), np.ones((1, 2), bool))}
    cuts = {1: (np.zeros((1, 1)), np.ones((1, 1), bool))}
    runs, score, receipt = _solve([0, 5, 10], [[0], [1], []], [[], [0], [0, 1]],
                                  np.zeros(1), holds, cuts)
    assert runs == ((0, 2, 0, 0),) and score == (9.0, 0)
    assert receipt['short_pairs_skipped'] == 1


@pytest.mark.parametrize('corrupt', [
    lambda c, m: (c[:, :1], m[:, :1]),
    lambda c, m: (c, m.astype(int)),
    lambda c, m: (c * np.nan, m),
    lambda c, m: (c + np.inf, m),
    lambda c, m: c,
    lambda c, m: (c.astype(complex), m),
])
def test_hold_callback_contract_is_validated(corrupt):
    nodes, ent, ext, start, holds, cuts = _endpoint_case()
    with pytest.raises(ValueError):
        solve(nodes, ent, ext, start_cost=start, hold_cost=lambda i, j: corrupt(*holds[i, j]),
              cut_cost=lambda j: cuts[j])


@pytest.mark.parametrize('corrupt', [
    lambda c, m: (c.T, m.T),
    lambda c, m: (c + np.inf, m),
    lambda c, m: (c, m.astype(float)),
])
def test_cut_callback_contract_is_validated(corrupt):
    nodes, ent, ext, start, holds, cuts = _endpoint_case()
    with pytest.raises(ValueError):
        solve(nodes, ent, ext, start_cost=start, hold_cost=lambda i, j: holds[i, j],
              cut_cost=lambda j: corrupt(*cuts[j]))


def test_inputs_are_validated():
    nodes, ent, ext, start, holds, cuts = _endpoint_case()
    good = dict(hold_cost=lambda i, j: holds[i, j], cut_cost=lambda j: cuts[j])
    cases = [
        (([0, 12, 12], ent, ext), dict(start_cost=start)),
        (([0., 12., 24.], ent, ext), dict(start_cost=start)),
        ((nodes, ent[:2], ext), dict(start_cost=start)),
        ((nodes, [[0.5, 0], [1], []], ext), dict(start_cost=start)),
        ((nodes, ent, ext), dict(start_cost=np.array([0., np.inf]))),
        ((nodes, ent, ext), dict(start_cost=start, start_allowed=np.ones(2))),
        ((nodes, ent, ext), dict(start_cost=start, minimum_frames=0)),
    ]
    for args, extra in cases:
        with pytest.raises(ValueError):
            solve(*args, **extra, **good)


def test_workspace_bound_is_enforced_not_silently_reduced():
    case = _endpoint_case()
    with pytest.raises(ValueError, match='workspace'):
        _solve(*case, workspace_limit_bytes=100)
    with pytest.raises(ValueError, match='workspace'):
        _solve(*case, workspace_limit_bytes=200)


def test_single_shared_check_can_abort():
    calls = []

    def check():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('deadline')

    with pytest.raises(RuntimeError, match='deadline'):
        _solve(*_endpoint_case(), check=check)
