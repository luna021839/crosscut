import unittest
from crosscut.direct_member_failure_diagnostics import find_member_infeasibility_relaxation


class MemberFailureDiagnosticsTests(unittest.TestCase):
    def test_nested_solver_survives_empty_effect_domain(self):
        evidence={'status':'diagnostic_only','feasible_without_quota':True}
        failure={'failure':'cinematic_bridge_edge_domain_exhausted',
                 'role_aware_safe_hold_failure':{'search':{'solver':{'infeasibility_relaxation':evidence}}}}
        found,path=find_member_infeasibility_relaxation(failure)
        self.assertEqual(found,evidence)
        self.assertIsNot(found,evidence)
        self.assertEqual(path,'role_aware_safe_hold_failure.search.solver.infeasibility_relaxation')

    def test_primary_has_precedence(self):
        found,_=find_member_infeasibility_relaxation({'search':{'solver':{'infeasibility_relaxation':{'primary':True}}},
            'role_aware_safe_hold_failure':{'search':{'solver':{'infeasibility_relaxation':{'primary':False}}}}})
        self.assertEqual(found,{'primary':True})

    def test_absent_diagnostic_is_not_fabricated(self):
        for value in (None,{}, {'search':None}, {'role_aware_safe_hold_failure':{'search':{'solver':{'status':2}}}}):
            self.assertIsNone(find_member_infeasibility_relaxation(value))

    def test_cinematic_nested_solver_is_retained(self):
        value={'heterogeneous_cinematic_failure':{'search':{'solver':{'infeasibility_relaxation':{'result':0}}}}}
        self.assertEqual(find_member_infeasibility_relaxation(value)[0],{'result':0})
