"""Real chart producer/independent parent reject availability-only gain.

Decoders, ownership pipeline and selected-program replay are real. Comparison
numbers and model detections are declared NON-HUMAN sensor substitutes.
"""
from dataclasses import replace
from time import monotonic
from unittest.mock import patch
import shutil
import unittest

from crosscut import group_scene_observations as observations
from crosscut import group_scene_source_owner as scene
from crosscut import group_scene_selected_replay as selected
from crosscut import group_performer_family_owner as family
from crosscut.optional_native_worker import isolated_optional,optional_native_scope
from engine.tests import test_scene_source_owner as charts

_REAL_EDGE=observations.compare_scene_edge


def _coverage_tie_reader(context,candidate_cost=.2):
    # Keep the actual original source interval identity; no new donor/source
    # selection is introduced by this negative comparison-sensor substitute.
    original=scene.source.visible_group_shots(context.incumbent.render_plan,context.domain)
    owners={(shot.first_output_frame,shot.end_output_frame):shot.source_index for shot in original}
    def read(sensor,left,right):
        result=_REAL_EDGE(sensor,left,right)
        incumbent=all(owners[(s.first_output_frame,s.end_output_frame)]==
            context.domain.request_index(s.eligible_source_index) for s in (left,right))
        return dict(result,available=not incumbent,required_semantics_available=not incumbent,
            metrics={k:.2 if incumbent else candidate_cost
                for k in ('occupancy','attention','scale','span','screen_motion')},
            framing_cost=.2,comparison_sensor_substitute='NONHUMAN_AVAILABILITY_ONLY_GAIN_COUNTEREXAMPLE')
    return read


@isolated_optional()
def coverage_tie_selected_parent(context,candidate,candidate_cost=.2):
    with charts.scene_chart_sensors(),patch.object(observations,'compare_scene_edge',
            side_effect=_coverage_tie_reader(context,candidate_cost)):
        return selected.replay_selected_scene.__wrapped__(context,candidate)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'local chart tools required')
class PartialEvidenceOwnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        charts.SceneSourceIntegrationTests.setUpClass.__func__(cls)
        if isinstance(cls.result,family.PerformerFamilyRejection):
            cls.temp.cleanup()
            raise AssertionError(cls.result.error)

    @classmethod
    def tearDownClass(cls):
        charts.SceneSourceIntegrationTests.tearDownClass.__func__(cls)

    def test_normal_producer_keeps_incumbent_on_coverage_only_ties_before_dense_or_completion(self):
        for cost in (.2,.3):
            with self.subTest(candidate_geometry=cost):
                context=replace(self.context,deadline=monotonic()+90.)
                with optional_native_scope(context.deadline),charts.scene_chart_sensors(), \
                        patch.object(scene,'compare_scene_edge',side_effect=_coverage_tie_reader(context,cost)) as compare, \
                        patch.object(scene,'has_editorial_gain',wraps=observations.has_editorial_gain) as gain, \
                        patch.object(scene.source,'complete_source_candidate',
                            side_effect=AssertionError('tie/regression must stop before complete-family work')):
                    result=scene.propose_scene_changes(context)
                self.assertIsInstance(result,family.PerformerFamilyRejection)
                self.assertGreater(compare.call_count,0)
                if cost==.2:self.assertGreater(gain.call_count,0)
                self.assertTrue(all(before['available'] is False for before,after in
                    (call.args for call in gain.call_args_list)))
                evidence=family.plain(result.evidence)
                self.assertEqual(evidence['native_work']['counts']['dense_paths'],0)
        self.assertIs(self.controller.incumbent,self.bundle)

    def test_actual_independent_selected_worker_also_rejects_coverage_only_gain(self):
        self.assertFalse(family.plain(self.result.performer_family_review)['selected_scene_program']['composition_window'])
        for cost in (.2,.3):
            with self.subTest(candidate_geometry=cost):
                context=replace(self.context,deadline=monotonic()+90.)
                with optional_native_scope(context.deadline):
                    with self.assertRaisesRegex(RuntimeError,'no longer beats its actual incumbent reference'):
                        coverage_tie_selected_parent(context,self.result,cost)

    def test_normal_known_improvement_still_passes_actual_independent_parent(self):
        context=replace(self.context,deadline=monotonic()+90.)
        with optional_native_scope(context.deadline):
            proof=charts.scene_chart_selected_replay(context,self.result)
        self.assertTrue(proof['fresh_complete_family_equal'])
        self.assertFalse(proof['quality_approved'])
