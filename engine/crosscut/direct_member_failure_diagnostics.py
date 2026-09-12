"""Retain solver evidence when another fallback supplies the primary error."""
from collections.abc import Mapping


def find_member_infeasibility_relaxation(diagnostics):
    """Search only known selector failure envelopes, retaining provenance.

An effect domain may contain no solver at all while the preceding role-aware
solver already computed the requested counterfactual. Do not trigger a fresh
source analysis simply because that evidence is nested under a fallback.
    """
    if not isinstance(diagnostics, Mapping):
        return None
    for name in (None, 'role_aware_safe_hold_failure', 'heterogeneous_cinematic_failure'):
        envelope = diagnostics if name is None else diagnostics.get(name)
        if not isinstance(envelope, Mapping):
            continue
        search = envelope.get('search')
        solver = search.get('solver') if isinstance(search, Mapping) else None
        evidence = solver.get('infeasibility_relaxation') if isinstance(solver, Mapping) else None
        if isinstance(evidence, Mapping):
            return dict(evidence), ('search.solver.infeasibility_relaxation' if name is None
                                   else name+'.search.solver.infeasibility_relaxation')
    return None
