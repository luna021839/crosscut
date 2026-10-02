"""Normal GROUP analyze must not spend editing time on a texture-only movie.

The upstream authoring and subject sensors are explicit nonhuman fixtures.
The existing normal analyze harness still runs the real renderer and original
failing final QA. This proves dispatch/budget ownership, not human quality.
"""
import unittest
from unittest.mock import patch

from crosscut import group_scene_review_proposal as trial
from engine.tests import test_scene_normal_analyze as normal


class NoAutomaticTexturePreviewTests(unittest.TestCase):
    def test_normal_analyze_keeps_texture_trial_out_of_editing_budget(self):
        helper = normal.SceneSourceNormalAnalyzeTests(methodName='runTest')
        # A spy prevents the unwanted trial from consuming actual test time.
        # It must never be called at all by the ordinary request path; only an
        # explicit diagnostic caller may request that unapproved artifact.
        with patch.object(trial, 'attempt_review_proposal', return_value=None) as preview:
            _, manifest, reports, rendered, calls, failure = helper.run_normal()
        self.assertEqual(preview.call_count, 0,
            'normal analyze still dispatches a texture-only full render before subject editing')
        self.assertEqual(len(calls), 2, failure)  # Producer plus independent parent.
        self.assertGreaterEqual(len(rendered), 2, failure)
        self.assertTrue(reports, failure)
        self.assertFalse(reports[-1][1]['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])


if __name__ == '__main__':
    unittest.main()
