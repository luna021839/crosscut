from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.ive import AspectRatio
from crosscut.pipeline import (_review_first_group_bundle, _validate_review_first_group_bundle,
    _stage_edit_montage_analysis_payload, _review_first_group_boundary_look_bundle)
from crosscut.boundary_look_match import analyze_boundary_look_match
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.visual_score import VisualTimelinePlan
from crosscut.visual_boundary_repair import _thaw_audit_value
from tests.test_visual_boundary_repair import _choice


class GroupFirstResultBundleTests(unittest.TestCase):
    def make_bundle(self):
        section = SectionPlan((SectionDirective(0, SectionRole.BUILD, 0., 2., .5, .5, 1,
            4, 15., .4, .1, False, False),))
        request = SimpleNamespace(sources=(Path('/synthetic/a'), Path('/synthetic/b')),
            reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE)
        with patch('crosscut.pipeline._review_visual_timeline_for_render', return_value=SimpleNamespace(weighted_score=.5)):
            return _review_first_group_bundle(timeline=VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 1.),
                request=request, eligible_indices=(0, 1), offsets=(0., 0.), timewarps=(None, None),
                common_start=0., section_plan=section, phrase_times=(0., 1., 2.),
                output=Path('/synthetic/out'), reporter=None, warnings=[])

    def validate(self, bundle):
        _validate_review_first_group_bundle(bundle, eligible_indices=(0, 1), offsets=(0., 0.),
            timewarps=(None, None), common_start=0.)

    def test_owned_clock_is_deeply_immutable_and_serializable(self):
        bundle = self.make_bundle()
        self.validate(bundle)
        clock = _thaw_audit_value(bundle.stage_edit_exact_shot_clock)
        owned = replace(bundle, stage_edit_exact_shot_clock=clock)
        clock['shots'][0]['source_index'] = 99
        self.assertEqual(owned.stage_edit_exact_shot_clock['shots'][0]['source_index'], 0)
        with self.assertRaises(TypeError):
            owned.stage_edit_exact_shot_clock['shots'][0]['source_index'] = 99
        self.assertIsInstance(_thaw_audit_value(owned.stage_edit_exact_shot_clock)['shots'], list)
        payload = _stage_edit_montage_analysis_payload(bundle=owned, proxy_review_payload={},
            section_plan=owned.section_plan, source_derivative_diversity=SimpleNamespace(to_dict=lambda: {}),
            source_index_mapping={})
        copied = json.loads(json.dumps(payload))
        self.assertEqual(copied['stage_edit_exact_shot_clock']['shots'][0]['source_index'], 0)
        payload['stage_edit_exact_shot_clock']['shots'][0]['source_index'] = 88
        self.assertEqual(owned.stage_edit_exact_shot_clock['shots'][0]['source_index'], 0)

    def test_look_bundle_keeps_timeline_geometry_review_and_exact_clock_owners(self):
        bundle = self.make_bundle()
        result = analyze_boundary_look_match(bundle.render_plan, group_16x9_safe=False)
        with patch('crosscut.review_first_boundary_look.refine_review_first_boundary_look',
                   return_value=(bundle.render_plan, result)) as worker:
            changed = _review_first_group_boundary_look_bundle(bundle, group_16x9_safe=False)
        self.validate(changed)
        worker.assert_called_once_with(bundle.render_plan, group_16x9_safe=False)
        self.assertEqual(changed.render_plan, bundle.render_plan)
        for name in ('visual_timeline', 'timeline_review', 'section_plan',
                     'stage_edit_exact_shot_clock', 'editorial_motion_plan'):
            self.assertEqual(getattr(changed, name), getattr(bundle, name))
        self.assertIs(changed.boundary_look_match, result)
        self.assertFalse(changed.candidate_lineage[-1]['whole_output_quality_approved'])
        self.assertEqual(bundle.candidate_lineage, ())

    def test_review_section_source_and_progress_faults_cannot_promote(self):
        bundle = self.make_bundle()
        candidates = [replace(bundle, timeline_review=None), replace(bundle, section_plan=None),
            replace(bundle, pre_render_score=float('nan')),
            replace(bundle, render_plan=replace(bundle.render_plan, segments=(
                replace(bundle.render_plan.segments[0], source_start=.1), bundle.render_plan.segments[1]))),
            replace(bundle, visual_timeline=replace(bundle.visual_timeline, choices=(
                replace(bundle.visual_timeline.choices[0], segment_index=99), bundle.visual_timeline.choices[1])))]
        for field, value in (('source_index', 99), ('phrase_progress', float('nan')), ('section_progress', 2.)):
            clock = _thaw_audit_value(bundle.stage_edit_exact_shot_clock)
            clock['shots'][0][field] = value
            candidates.append(replace(bundle, stage_edit_exact_shot_clock=clock))
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                self.validate(candidate)

    def test_in_range_but_stale_role_and_section_progress_cannot_promote(self):
        bundle=self.make_bundle()
        for field,value in (('section_role','chorus_like'),('section_progress',.9)):
            clock=_thaw_audit_value(bundle.stage_edit_exact_shot_clock)
            clock['shots'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.validate(replace(bundle,stage_edit_exact_shot_clock=clock))
