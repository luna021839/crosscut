"""Bounded in-process re-selection after a required effect cannot be authored."""
from __future__ import annotations

from typing import Callable, Sequence, Any

from .direct_member_visual import DirectMemberVisualSelectionError
from .member_bridge_contract import RequiredMemberBridgeAuthoringError


def select_authorable_member_path(
    selector: Callable[[tuple, tuple | None], Any], *,
    rejected_edges: Sequence[tuple[float, int, int]] = (),
    preferred_partition_unit_counts: Sequence[int] | None = None,
    maximum_attempts: int = 8,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[Any, dict]:
    if isinstance(maximum_attempts,bool) or not isinstance(maximum_attempts,int) or not 1 <= maximum_attempts <= 8:
        raise ValueError('required effect recovery must be bounded to one through eight attempts')
    edges = list(dict.fromkeys(rejected_edges))
    preferred = tuple(preferred_partition_unit_counts) if preferred_partition_unit_counts else None
    failures = []
    for attempt in range(1, maximum_attempts+1):
        try:
            result = selector(tuple(edges), preferred)
        except RequiredMemberBridgeAuthoringError as exc:
            if not exc.edges_by_boundary or not exc.partition_unit_counts:
                # Unbound exceptions are programming errors, not authorization
                # to reject arbitrary cuts or to run another whole-song search.
                raise
            new_edges = [edge for edge in exc.edges_by_boundary.values() if edge not in edges]
            event = {'type':'direct_member_effect_authoring_recovery',
                     'contract':'bounded_exact_edge_effect_authoring_feedback_v1',
                     'attempt':attempt,'maximum_attempts':maximum_attempts,
                     'failure':exc.to_dict(),'new_rejected_edge_count':len(new_edges),
                     'selector_callback_reused':True,'visual_gates_relaxed':False}
            failures.append(event)
            if progress_callback is not None:
                progress_callback(event)
            if not new_edges:
                reason = 'required_effect_feedback_made_no_progress'
                break
            edges.extend(new_edges)
            preferred = exc.partition_unit_counts
        else:
            return result, {'contract':'bounded_exact_edge_effect_authoring_feedback_v1',
                            'attempt_count':attempt,'rejected_edge_count':len(edges),
                            'failures':failures,'rejected_transition_edges':list(edges),
                            'passed':True,'visual_gates_relaxed':False}
    else:
        reason = 'required_effect_authoring_attempt_budget_exhausted'
    raise DirectMemberVisualSelectionError('required member effect path could not be authored',
        {'failure':reason,'constraint_infeasibility_proven':False,
         'effect_authoring_recovery':{'attempt_count':len(failures),'failures':failures,
                                    'rejected_edge_count':len(edges),'passed':False},
         'visual_gates_relaxed':False,'hard_cut_fallback_authorized':False})
