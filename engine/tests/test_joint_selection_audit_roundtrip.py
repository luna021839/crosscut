"""Exercise the real immutable success-result handoff, not a dict-only stub."""
from dataclasses import replace
import json
from types import MappingProxyType
import unittest
from unittest.mock import patch

from crosscut.joint_visual_candidate_adapter import _thaw_json_audit
from crosscut.joint_proposal_domain import select_with_joint_proposal_domains
from engine.tests.test_joint_visual_candidate_adapter import (
    _candidate_set, _limited_profile_pools, select_joint_visual_candidates,
    StubMaterializingProvider,
)
from engine.tests.test_joint_proposal_domain import pool_fixture


class JointSelectionAuditRoundtripTests(unittest.TestCase):
    def selection(self):
        candidates = _candidate_set(role_conditioned=True)
        pools = _limited_profile_pools(candidates, unique_path_count=1, alternate_paths=())
        with patch('crosscut.joint_visual_candidate_adapter.select_joint_timeline_candidates',
                   side_effect=pools):
            return select_joint_visual_candidates(candidates,
                provider=StubMaterializingProvider(fps=10.), fps=10.,
                minimum_unique_paths=1, minimum_required_candidates=1, maximum_candidates=8)

    def test_immutable_success_can_be_replaced_without_losing_proof(self):
        original = self.selection()
        replaced = replace(original)
        self.assertEqual(replaced.to_dict(), original.to_dict())
        self.assertIs(replaced.boundary_proof_bundles[0], original.boundary_proof_bundles[0])
        with self.assertRaises(TypeError):
            replaced.selection_retry_audit['profile_audition']['events'][0]['changed'] = True

    def test_proposal_domain_handoff_accepts_real_frozen_nested_audit(self):
        original = self.selection()
        _, pool = pool_fixture()
        result = select_with_joint_proposal_domains(pool, maximum_proof_calls=4,
                                                    select=lambda **kwargs: original)
        audit = result.to_dict()['selection_retry_audit']
        self.assertEqual(audit['proof_materialization_call_count'], 1)
        self.assertEqual(audit['proposal_search_domain']['events'][-1]['status'], 'exact_selection_passed')
        self.assertFalse(audit['proposal_search_domain']['quality_gates_waived'])
        json.dumps(result.to_dict(), allow_nan=False)

    def test_nested_lists_are_recursively_thawed_and_detached(self):
        source = {'items': [MappingProxyType({'nested': (MappingProxyType({'count': 2}),)})]}
        result = _thaw_json_audit(source)
        self.assertEqual(result, {'items': [{'nested': [{'count': 2}]}]})
        result['items'][0]['nested'][0]['count'] = 3
        self.assertEqual(source['items'][0]['nested'][0]['count'], 2)

    def test_non_json_and_nonfinite_audit_still_fail(self):
        original = self.selection()
        for bad in (object(), float('nan')):
            with self.assertRaises((TypeError, ValueError)):
                replace(original, selection_retry_audit={'nested': [{'bad': bad}]})


if __name__ == '__main__':
    unittest.main()
