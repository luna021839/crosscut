from copy import deepcopy
import unittest

from crosscut.proxy_review_contract import proxy_integrity_evidence
from crosscut.review_first_runtime import classify_review_first_qa, ReviewFirstStatus


CASES = (
    ("proxy_quality_overall_quality_p10", "overall_quality_p10", 0.4, 0.5),
    ("proxy_quality_source_quality_median_span", "source_quality_median_span", 0.4, 0.25),
    ("proxy_quality_boundary_score_p10", "boundary_score_p10", 50.0, 60.0),
    ("proxy_quality_proxy_weighted_score", "weighted_score", 47.1, 60.0),
    ("proxy_subject_geometry_coverage_incomplete", "subject_geometry_coverage", 0.5, 1.0),
    ("proxy_closeup_face_coverage_incomplete", "closeup_face_coverage", 0.5, 1.0),
)


def report_for(case):
    code, field, value, threshold = case
    return {
        "passed": False,
        "issues": [{"code": code, "value": value, "threshold": threshold}],
        "metrics": {"proxy_quality_and_geometry": {
            field: value, "boundary_sample_contract_valid": True,
            "closeup_relevant_boundary_count": 2,
            "technical_integrity": proxy_integrity_evidence({
                "schema_version": 3, "geometry_provider_failure_count": 0,
                "reviewed_boundary_count": 2,
                "introduced_cross_source_technical_issue_reasons": [],
                "boundaries": [{
                    "boundary_index": index, "passed": False,
                    "issue_reasons": ["subject_continuity_unobserved"],
                    "technical_integrity_issue_reasons": [],
                } for index in range(2)],
            }),
        }},
    }


def classify(report, **kwargs):
    return classify_review_first_qa(report, **{
        "content_binding_passed": True, "review_first_enabled": True,
        "member": False, **kwargs,
    })


class GroupReviewFirstMeasurementTests(unittest.TestCase):
    def test_measured_quality_misses_are_reviewable_but_never_approved(self):
        for case in CASES:
            with self.subTest(code=case[0]):
                result = classify(report_for(case))
                self.assertEqual(result.status, ReviewFirstStatus.REVIEW_READY)
                self.assertTrue(result.video_output_exposed)
                self.assertFalse(result.delivery_completion_message_allowed)
                self.assertEqual(result.technical_issue_codes, ())
                self.assertEqual(result.editorial_issue_codes, (case[0],))

    def test_missing_invalid_or_mismatched_measurements_still_block(self):
        for case in CASES:
            for bad in (None, True, float("nan"), float("inf"), -0.1, 101.0, "0.5"):
                with self.subTest(code=case[0], bad=bad):
                    report = report_for(case)
                    report["metrics"]["proxy_quality_and_geometry"][case[1]] = bad
                    report["issues"][0]["value"] = bad
                    result = classify(report)
                    self.assertEqual(result.status, ReviewFirstStatus.INPUT_UNUSABLE)
                    self.assertIn(case[0], result.technical_issue_codes)
            report = report_for(case)
            report["issues"][0]["value"] = case[2] + 0.01
            self.assertFalse(classify(report).video_output_exposed)

    def test_contradictory_gate_or_duplicate_evidence_still_blocks(self):
        for case in CASES:
            report = report_for(case)
            report["issues"][0]["threshold"] = case[2]
            self.assertFalse(classify(report).video_output_exposed)
            report = report_for(case)
            duplicate = deepcopy(report["issues"][0])
            duplicate["value"] = None
            report["issues"].append(duplicate)
            self.assertFalse(classify(report).video_output_exposed)

    def test_coverage_requires_real_count_and_valid_decoded_samples(self):
        for case in CASES[-2:]:
            for change in ({"boundary_sample_contract_valid": False},
                           {case[1]: 0.7}):
                report = report_for(case)
                report["metrics"]["proxy_quality_and_geometry"].update(change)
                report["issues"][0]["value"] = report["metrics"]["proxy_quality_and_geometry"][case[1]]
                self.assertFalse(classify(report).video_output_exposed)
        for count in (None, True, 0, -1, 3, 1.5):
            report = report_for(CASES[-1])
            report["metrics"]["proxy_quality_and_geometry"]["closeup_relevant_boundary_count"] = count
            self.assertFalse(classify(report).video_output_exposed)

    def test_nested_decode_or_provider_failure_still_blocks(self):
        for case in CASES:
            for failure in ("decode_failed", "new_unknown_boundary_failure"):
                report = report_for(case)
                integrity = report["metrics"]["proxy_quality_and_geometry"]["technical_integrity"]
                integrity["review"]["boundaries"][0]["issue_reasons"].append(failure)
                # A stale serialized passed=True is not trusted.
                self.assertFalse(classify(report).video_output_exposed)
            report = report_for(case)
            report["metrics"]["proxy_quality_and_geometry"]["technical_integrity"]["review"]["geometry_provider_failure_count"] = 1
            self.assertFalse(classify(report).video_output_exposed)

    def test_technical_top_level_codes_and_binding_remain_blocking(self):
        for case in CASES:
            report = report_for(case)
            self.assertFalse(classify(report, content_binding_passed=False).video_output_exposed)
            self.assertFalse(classify(report, review_first_enabled=False).video_output_exposed)
            for code in ("proxy_boundary_sample_contract_invalid", "full_decode_failed",
                         "proxy_quality_unknown_new_field"):
                damaged = deepcopy(report)
                damaged["issues"].append({"code": code})
                self.assertFalse(classify(damaged).video_output_exposed)
            report["passed"] = True
            self.assertFalse(classify(report).video_output_exposed)


if __name__ == "__main__":
    unittest.main()
