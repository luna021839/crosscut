from dataclasses import replace
import unittest
from unittest.mock import patch

from crosscut.direct_member_look import DirectMemberLookObservation, solve_direct_member_look_adjustments
from crosscut.member_look_preflight import (
    MemberLookProposalError, build_member_look_proxy_proposal,
    predicted_member_look_failure_reasons,
)


class MemberLookPreflightTests(unittest.TestCase):
    def test_prediction_failure_stays_explicit_without_changing_bounds(self):
        observations=tuple(DirectMemberLookObservation(i,i%2,1.,.1 if i%2 else .9,
            .1,.1,.15,5) for i in range(8))
        with self.assertRaisesRegex(ValueError,'cannot satisfy'):
            solve_direct_member_look_adjustments(observations)
        plan=solve_direct_member_look_adjustments(observations,require_predicted_pass=False)
        self.assertIn('source_luma_span',predicted_member_look_failure_reasons(plan))
        self.assertEqual(plan.config.source_luma_span,.18)
        self.assertTrue(all(abs(value.brightness)<=.18 for value in plan.segment_adjustments))

    def test_decoded_gate_is_explicitly_required_and_no_hidden_retry(self):
        with patch('crosscut.member_look_preflight.build_direct_member_look_plan',return_value='plan') as build:
            for flag in (False,None,1):
                with self.assertRaises(ValueError):
                    build_member_look_proxy_proposal(visual_selection=object(),render_segments=(),
                                                    mandatory_decoded_proxy=flag)
            build.assert_not_called()
            self.assertEqual(build_member_look_proxy_proposal(visual_selection='selection',render_segments=(),
                              mandatory_decoded_proxy=True),'plan')
            build.assert_called_once_with(visual_selection='selection',render_segments=(),require_predicted_pass=False)

    def test_malformed_color_evidence_has_its_own_stage_not_camera_fallback(self):
        with patch('crosscut.member_look_preflight.build_direct_member_look_plan',side_effect=ValueError('missing look evidence')):
            with self.assertRaisesRegex(MemberLookProposalError,'색 보정 입력 검증 실패'):
                build_member_look_proxy_proposal(visual_selection=object(),render_segments=(),
                                                mandatory_decoded_proxy=True)


if __name__=='__main__':
    unittest.main()
