from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from crosscut.ive_product_quality_floor import _repair_lineage_evidence
from crosscut.review_first_repair_lineage import review_first_repair_lineage
from crosscut.timeline_review import review_timeline
from crosscut.visual_boundary_repair import repair_visual_timeline_boundaries
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_visual_boundary_repair import _choice, _descriptor


class ReviewFirstRepairLineageTests(unittest.TestCase):
    def setUp(self):
        summary = dict(candidate_index=0, selected=True, passed=False,
                       hard_failure_codes=['focus_mismatch_exceeded'])
        self.before = VisualTimelinePlan((_choice(0,0), _choice(1,1)), 4.,
                                        candidate_review_summaries=(summary,))
        def provider(source, seconds):
            offset = round((seconds-(1.+(-.5 if source==0 else .5)/29.97))*29.97)
            return _descriptor(.4 if source==0 or offset in (2,5) else .8)
        self.repairs = repair_visual_timeline_boundaries(visual_timeline=self.before,
            sources=(Path('a'), Path('b')), audio_offsets=(0.,0.), scene_cut_times=((),()),
            descriptor_provider=provider)
        self.assertEqual(self.repairs.changed_count, 1)
        self.timeline = self.repairs.timeline
        # Fixture review authority tests bookkeeping, NOT real media quality.
        self.review = replace(review_timeline(self.timeline, available_source_count=2), hard_failures=())

    def event(self, **changes):
        return review_first_repair_lineage(**dict(
            dict(candidate_index=0, timeline=self.timeline, repairs=self.repairs, review=self.review), **changes))

    def test_typed_changed_repair_reaches_the_unchanged_product_floor_consumer(self):
        event = self.event()
        self.assertEqual(event['boundary_repair_changed_count'], 1)
        self.assertFalse(event['decoded_output_quality_approval'])
        selection = dict(selected_candidate_index=0, complete_candidates=[dict(candidate_index=0,
            visual_timeline=self.timeline.to_dict(), candidate_lineage=[event])])
        evidence, failures = _repair_lineage_evidence({}, selection)
        self.assertTrue(evidence['successful_repair_lineage_found'])
        self.assertEqual(failures, ())
        selection['complete_candidates'][0]['candidate_lineage'] = []
        self.assertIn('selected_failed_pre_review_lacks_successful_repair_lineage',
                      _repair_lineage_evidence({}, selection)[1])

    def test_failed_review_or_missing_or_unchanged_repair_never_creates_success(self):
        failed = review_timeline(self.timeline, available_source_count=2)
        self.assertFalse(failed.passed)
        self.assertIsNone(self.event(review=failed))
        self.assertIsNone(self.event(repairs=None))
        self.assertIsNone(self.event(repairs=replace(self.repairs, repairs=())))
        self.assertIsNone(self.event(repairs=replace(self.repairs,
            attempt_input_timelines=(self.timeline,))))

    def test_unrelated_clocks_or_source_path_cannot_claim_old_repairs(self):
        changed = replace(self.timeline.choices[0], source_local_start=.01)
        unrelated = replace(self.timeline, choices=(changed,self.timeline.choices[1]))
        with self.assertRaisesRegex(ValueError, 'selected timeline clocks'):
            self.event(timeline=unrelated)
        other_before = replace(self.before, choices=(replace(self.before.choices[0],source_index=1),
                                                   self.before.choices[1]))
        with self.assertRaisesRegex(ValueError, 'changed the source path'):
            self.event(repairs=replace(self.repairs,attempt_input_timelines=(other_before,)))

    def test_identity_malformed_or_duplicate_pre_review_is_not_repaired_by_a_label(self):
        for changes in (dict(candidate_index=True),dict(candidate_index=1),
                        dict(hard_failure_codes=[]),dict(hard_failure_codes=['x','x']),
                        dict(passed=True)):
            summary = dict(self.timeline.candidate_review_summaries[0], **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.event(timeline=replace(self.timeline,candidate_review_summaries=(summary,)))
        with self.assertRaisesRegex(ValueError,'one selected'):
            self.event(timeline=replace(self.timeline,
                candidate_review_summaries=self.timeline.candidate_review_summaries*2))

    def test_missing_or_successful_pre_review_does_not_invent_a_failure_history(self):
        self.assertIsNone(self.event(timeline=replace(self.timeline,candidate_review_summaries=())))
        summary = dict(self.timeline.candidate_review_summaries[0],passed=True,hard_failure_codes=[])
        self.assertIsNone(self.event(timeline=replace(self.timeline,candidate_review_summaries=(summary,))))

    def test_serialized_pass_flags_and_duck_typed_reports_are_not_authority(self):
        for changes in (dict(review={'passed': True}),dict(review=SimpleNamespace(passed=True)),
                        dict(repairs=SimpleNamespace(changed_count=1)),dict(candidate_index=False)):
            with self.subTest(changes=changes), self.assertRaises(TypeError):
                self.event(**changes)

    def test_event_does_not_mutate_the_preserved_pre_review_failure(self):
        before = deepcopy(self.timeline.candidate_review_summaries)
        event = self.event()
        event['pre_review_hard_failure_codes'].append('unrelated')
        self.assertEqual(self.timeline.candidate_review_summaries, before)


if __name__ == '__main__':
    unittest.main()
