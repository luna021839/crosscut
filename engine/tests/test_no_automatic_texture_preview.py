"""Normal GROUP analyze must not spend editing time on a texture-only movie.

The upstream authoring and subject sensors are explicit nonhuman fixtures.
The existing normal analyze harness still runs the real renderer and original
failing final QA. This proves dispatch/budget ownership, not human quality.
"""
import unittest
from unittest.mock import patch

from crosscut import group_scene_review_proposal as trial
from crosscut import group_scene_selected_replay as selected_replay
from engine.tests import test_scene_normal_analyze as normal
from engine.tests import test_scene_source_owner as charts


class NoAutomaticTexturePreviewTests(unittest.TestCase):
    def test_normal_analyze_keeps_texture_trial_out_of_editing_budget(self):
        helper = normal.SceneSourceNormalAnalyzeTests(methodName='runTest')
        # A spy prevents the unwanted trial from consuming actual test time.
        # It must never be called at all by the ordinary request path; only an
        # explicit diagnostic caller may request that unapproved artifact.
        parent_proofs = []
        actual_parent = charts.scene_chart_selected_replay
        def observe_parent(context, candidate):
            proof = actual_parent(context, candidate)
            parent_proofs.append(proof)
            return proof
        with patch.object(trial, 'attempt_review_proposal', return_value=None) as preview, \
                patch.object(charts, 'scene_chart_selected_replay', side_effect=observe_parent):
            _, manifest, reports, rendered, calls, failure = helper.run_normal()
        self.assertEqual(preview.call_count, 0,
            'normal analyze still dispatches a texture-only full render before subject editing')
        self.assertEqual(len(calls), 1, failure)  # One producer, no repeated donor search.
        chosen = manifest['rendered_candidate_selection']['complete_candidates'][0]
        receipt = chosen['performer_family_review']
        program = receipt['selected_scene_program']
        selected_replay.validate_program(program, receipt['source_camera_refresh'])
        self.assertEqual(program['contract'], selected_replay.CONTRACT)
        self.assertEqual(len(parent_proofs), 1, failure)
        proof = parent_proofs[0]
        self.assertEqual(proof['selected_program_sha256'], program['program_sha256'])
        self.assertTrue(proof['fresh_complete_family_equal'])
        self.assertTrue(proof['selected_and_original_hypotheses_reobserved'])
        self.assertFalse(proof['unselected_donor_search_repeated'])
        self.assertFalse(proof['quality_approved'])
        self.assertGreaterEqual(len(rendered), 2, failure)
        self.assertTrue(reports, failure)
        self.assertFalse(reports[-1][1]['passed'])
        self.assertFalse(manifest['app_result_exposure']['delivery_completion_message_allowed'])


if __name__ == '__main__':
    unittest.main()
