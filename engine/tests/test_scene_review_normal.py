"""Normal analyze preserves subject-owner time instead of auto-rendering a trial.

Explicit texture-proposal publication, pixel integrity and clock tests remain
in test_scene_review_proposal. These normal-entry tests now enforce the product
policy after the actual negative-gain trial was rejected on human quality.
Upstream authoring and sensors are synthetic; the renderer and final QA are not.
"""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from crosscut import group_scene_review_proposal as trial
from crosscut import group_performer_family_owner as family
from engine.tests import test_scene_normal_analyze as normal
from engine.tests.test_wide_composition_owner import (
    build_wide_fixture, encode_wide_chart, wide_chart_propose,
    wide_chart_final, wide_chart_sensors,
)


class SceneReviewNormalTests(unittest.TestCase):
    def run_normal_without_trial(self, *, corrupt_parent=False):
        events = []

        class CaptureReporter(normal.Reporter):
            def emit(self, *args, **kwargs):
                super().emit(*args, **kwargs)
                if args and args[0] == 'scene_review_proposal':
                    events.append(kwargs['scene_review_proposal'])

        with ExitStack() as stack:
            for name, value in dict(
                build_scene_fixture=build_wide_fixture,
                encode_scene_chart=encode_wide_chart,
                scene_chart_propose=wide_chart_propose,
                scene_chart_final=wide_chart_final,
                scene_chart_sensors=wide_chart_sensors,
                Reporter=CaptureReporter,
            ).items():
                stack.enter_context(patch.object(normal, name, new=value))
            preview = stack.enter_context(patch.object(
                trial, 'attempt_review_proposal', return_value=None))
            helper = normal.SceneSourceNormalAnalyzeTests(methodName='runTest')
            result = helper.run_normal(corrupt_parent=corrupt_parent)
        return events, preview.call_count, result

    def test_normal_positive_does_not_auto_encode_texture_review(self):
        events, preview_calls, result = self.run_normal_without_trial()
        _, manifest, reports, rendered, calls, failure = result
        self.assertEqual(preview_calls, 0)
        self.assertEqual(events, [])
        self.assertEqual(len(calls), 2, failure)  # Strict producer plus parent.
        self.assertGreaterEqual(len(rendered), 2, failure)
        self.assertTrue(reports, failure)
        self.assertFalse(reports[-1][1]['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
        self.assertTrue(any(i['code'] == family.QUALITY_CODE for i in reports[-1][1]['issues']))

    def test_rejected_strict_child_does_not_fall_back_to_texture_movie(self):
        events, preview_calls, result = self.run_normal_without_trial(corrupt_parent=True)
        baseline, manifest, reports, rendered, calls, failure = result
        self.assertEqual(preview_calls, 0)
        self.assertEqual(events, [])
        self.assertEqual(len(calls), 2, failure)
        self.assertEqual(baseline.render_plan.segments, rendered[-1].segments)
        self.assertTrue(reports, failure)
        self.assertFalse(reports[-1][1]['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])
