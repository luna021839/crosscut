from copy import deepcopy
import unittest

from crosscut.member_continuity_contract import member_continuity_measurements_valid
from crosscut.review_first_runtime import classify_review_first_qa, ReviewFirstStatus
from tests.member_continuity_fixture import member_continuity_report


class MemberContinuityContractTests(unittest.TestCase):
    def test_semantic_pass_mutations_reach_final_gate_and_classifier(self):
        from pathlib import Path
        from types import SimpleNamespace
        from crosscut.member_post_render_qa import _rendered_continuity_payload, _Issues, AutomaticMemberPostRenderQAConfig
        plan = {'fps': 30., 'output_frame_count': 120,
            'segments': [{'source_index': index, 'duration': 1.} for index in range(4)],
            'transitions': [{'kind': 'hard_cut', 'frame_count': 0} for _ in range(3)]}
        def remove_pose(report):
            for row in report['boundaries']: row['pose_mismatch'] = None
            report['metrics'].update(actual_pose_boundary_count=0, actual_pose_coverage_fraction=0., pose_mismatch_p95=None)
            report['thresholds']['minimum_actual_pose_coverage_fraction'] = 0.  # Not trusted.
        mutations = (
            remove_pose,
            lambda r: r['boundaries'][0]['motion'].clear(),
            lambda r: r['boundaries'][0]['motion'].update(timestamp_bound_adjacent=False),
            lambda r: r['boundaries'][0]['motion']['master_timestamps_seconds'].update(incoming_start=999.),
            lambda r: r['metrics'].update(focal_face_reference_guided_boundary_count=3,
                focal_face_reference_guided_fraction=1., focal_reference_authority_active=True),
            lambda r: r['boundaries'][0].update(focal_face_correspondence_similarity=float('nan')),
            lambda r: r['boundaries'][0].update(focal_face_reference_guided=True, body_reference_guided=True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                report = member_continuity_report(plan)
                mutate(report)
                issues = _Issues()
                payload = _rendered_continuity_payload(SimpleNamespace(video_path=Path('/synthetic/output')),
                    plan, AutomaticMemberPostRenderQAConfig(), lambda *a, **k: report, issues)
                self.assertFalse(member_continuity_measurements_valid(payload, plan=plan))
                disposition = classify_review_first_qa({'passed': True, 'issues': [],
                    'metrics': {'rendered_continuity': payload}}, content_binding_passed=True,
                    review_first_enabled=True, member=True)
                self.assertEqual(disposition.status, ReviewFirstStatus.INPUT_UNUSABLE)

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

    def test_nested_motion_requires_real_scalar_measurements_and_exact_frame_clock(self):
        mutations = (
            lambda m: m.clear(),
            lambda m: m.pop('motion_magnitudes'),
            lambda m: m['motion_magnitudes'].update(outgoing_pose=float('nan')),
            lambda m: m['master_timestamps_seconds'].update(incoming_start=99.),
            lambda m: m['source_local_timestamps_seconds'].pop('incoming_next'),
            lambda m: m.update(contract='unknown'),
            lambda m: m.update(both_moving='false'),
            lambda m: m.update(subject_direction_cosine=1.1),
            lambda m: m['motion_component_counts'].update(outgoing=True),
            lambda m: m['motion_magnitudes'].update(incoming_decision=.5),
            lambda m: m.update(direction_cosine=.2),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                report = member_continuity_report()
                mutate(report['boundaries'][0]['motion'])
                self.assertFalse(member_continuity_measurements_valid(report))
                disposition = classify_review_first_qa({'passed': True, 'issues': [],
                    'metrics': {'rendered_continuity': report}},
                    content_binding_passed=True, review_first_enabled=True, member=True)
                self.assertEqual(disposition.status, ReviewFirstStatus.INPUT_UNUSABLE)
                self.assertFalse(disposition.video_output_exposed)
