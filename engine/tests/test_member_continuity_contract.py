from copy import deepcopy
import unittest

from crosscut.member_continuity_contract import member_continuity_measurements_valid
from crosscut.review_first_runtime import classify_review_first_qa, ReviewFirstStatus
from tests.member_continuity_fixture import member_continuity_report


class MemberContinuityContractTests(unittest.TestCase):
    def test_actual_producer_accepts_hard_effect_and_sparse_source_switch_indices(self):
        for kind in ("hard_cut", "face_dissolve", "member_zoom_bridge", "member_blur_bridge"):
            with self.subTest(kind=kind):
                plan = {"fps": 30., "output_frame_count": 120,
                    "segments": [{"source_index": index, "duration": 1.} for index in (0, 0, 1, 1)],
                    "transitions": [{"kind": kind, "frame_count": 0 if kind == "hard_cut" else 3} for _ in range(3)]}
                report = member_continuity_report(plan)
                self.assertTrue(report["passed"], report["failure_reasons"])
                self.assertTrue(member_continuity_measurements_valid(report, plan=plan))
                self.assertEqual(report["boundaries"][0]["boundary_index"], 1)
                altered = deepcopy(plan)
                altered["segments"][1]["duration"] = 1.1
                self.assertFalse(member_continuity_measurements_valid(report, plan=altered))
                altered = deepcopy(plan)
                altered["transitions"][1]["frame_count"] += 1
                self.assertFalse(member_continuity_measurements_valid(report, plan=altered))

    def test_classifier_does_not_trust_bare_pass_or_hidden_nested_failure(self):
        for change in ({"metrics": {}}, {"boundaries": []}, {"schema_version": True},
                       {"passed": False, "failure_reasons": ["rendered_pose_p95_above_floor"]},
                       {"failure_boundary_indices": {"decode_failed": [0]}}):
            with self.subTest(change=change):
                report = member_continuity_report()
                report.update(change)
                disposition = classify_review_first_qa({"passed": True, "issues": [],
                    "metrics": {"rendered_continuity": report}},
                    content_binding_passed=True, review_first_enabled=True, member=True)
                self.assertEqual(disposition.status, ReviewFirstStatus.INPUT_UNUSABLE)
                self.assertFalse(disposition.video_output_exposed)

    def test_zero_switch_measurement_can_be_editorial_failure_but_never_pass(self):
        plan = {"fps": 30., "output_frame_count": 60,
            "segments": [{"source_index": 0, "duration": 1.} for _ in range(2)],
            "transitions": [{"kind": "hard_cut", "frame_count": 0}]}
        report = member_continuity_report(plan)
        self.assertFalse(report["passed"])
        self.assertTrue(member_continuity_measurements_valid(report, plan=plan))
        report.update(passed=True, failure_reasons=[])
        self.assertFalse(member_continuity_measurements_valid(report, plan=plan))

    def test_missing_observations_and_inconsistent_aggregates_are_not_measurements(self):
        mutations = (
            lambda r: r["boundaries"][0].pop("pose_mismatch"),
            lambda r: r["boundaries"][0].update(attention_position_mismatch=float("nan")),
            lambda r: r["boundaries"][0].update(sample_frame_indices=[0, 1, 2, 3]),
            lambda r: r["metrics"].update(actual_pose_coverage_fraction=.1),
            lambda r: r["metrics"].update(attention_boundary_count=100),
            lambda r: r["metrics"].update(pose_mismatch_p95=.7),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                report = member_continuity_report()
                mutate(report)
                self.assertFalse(member_continuity_measurements_valid(report))
