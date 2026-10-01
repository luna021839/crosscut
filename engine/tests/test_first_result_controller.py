from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.first_result_controller import FirstResultController, finalize_first_result_manifest
from crosscut.ive import AspectRatio
from crosscut.render import RenderPlan, RenderSegment
from crosscut.review_first_runtime import ReviewFirstRuntime, ReviewFirstSearchCutoff, ReviewFirstDeadlineExceeded
from crosscut.review_first_runtime import classify_review_first_qa, ReviewFirstStatus


class FirstResultControllerTests(unittest.TestCase):
    def test_journal_survives_failures_cancellation_and_bounded_attempts(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory, patch(
                'crosscut.first_result_controller.enforce_review_first_deadline', return_value=nullcontext()):
            _, controller, plan = self.make()
            controller.journal_path = Path(directory) / 'attempts.json'
            def failure():
                running = json.loads(controller.journal_path.read_text())
                self.assertEqual(running['attempts'][-1]['status'], 'running')
                raise ValueError('dense focal unavailable')
            controller.run_optional('camera', failure)
            journal = json.loads(controller.journal_path.read_text())
            self.assertEqual(journal['rejected_attempt_count'], 1)
            self.assertTrue(journal['baseline_retained'])
            def cancelled(): raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                controller.run_optional('cancel', cancelled)
            self.assertEqual(json.loads(controller.journal_path.read_text())['attempts'][-1]['status'], 'cancelled')
            for _ in range(70):
                controller.run_optional('bounded', lambda: None)
            journal = json.loads(controller.journal_path.read_text())
            self.assertEqual(len(journal['attempts']), 64)
            self.assertEqual(journal['omitted_attempt_count'], 8)
            self.assertEqual(journal['selected_authority'], 'verified_primitive')

    def make(self):
        clock = [0.]
        runtime = ReviewFirstRuntime(enabled=True, _clock=lambda: clock[0])
        plan = RenderPlan(sources=(Path('/synthetic/a.mp4'), Path('/synthetic/b.mp4')),
            segments=(RenderSegment(source_index=0, source_start=0., master_start=0., duration=1.),
                      RenderSegment(source_index=1, source_start=1., master_start=1., duration=1.)),
            reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE, output=Path('/synthetic/out.mp4'))
        controller = FirstResultController(runtime, sources=plan.sources, source_durations=(10., 10.),
            master_start=0., duration=2.)
        controller.install_baseline(plan, plan=plan, authority='verified_primitive')
        return clock, controller, plan

    def test_failures_and_control_cutoff_preserve_full_baseline(self):
        for failure in (ValueError('geometry'), RuntimeError('model'), ReviewFirstSearchCutoff('stop')):
            with self.subTest(failure=type(failure).__name__):
                _, controller, plan = self.make()
                def proposal(): raise failure
                with patch('crosscut.first_result_controller.enforce_review_first_deadline', return_value=nullcontext()):
                    self.assertIsNone(controller.run_optional('candidate', proposal))
                self.assertIs(controller.incumbent, plan)
                self.assertIs(controller.baseline, plan)
                self.assertEqual(controller.to_dict()['optional_attempt_count'], 1)
                self.assertNotEqual(controller.to_dict()['attempts'][0]['status'], 'running')

    def test_atomic_adoption_rejects_stale_worker_and_keeps_new_incumbent(self):
        _,controller,plan=self.make();token=controller.transaction_token()
        newer=replace(plan,segments=(replace(plan.segments[0],source_start=.1),plan.segments[1]))
        controller.promote(newer,plan=newer,authority='newer')
        calls=[]
        with self.assertRaisesRegex(ValueError,'baseline changed'):
            controller.compare_and_promote(plan,plan=plan,authority='stale-worker',token=token,
                validate=lambda value:calls.append(value))
        self.assertEqual(calls,[])
        self.assertIs(controller.incumbent,newer)

    def test_atomic_adoption_installs_full_owner_only_after_validation(self):
        _,controller,plan=self.make();token=controller.transaction_token()
        candidate=replace(plan,segments=(replace(plan.segments[0],source_start=.1),plan.segments[1]))
        def reject(value):
            self.assertIs(controller.incumbent,plan)
            raise ValueError('stale native observation')
        with self.assertRaisesRegex(ValueError,'stale native'):
            controller.compare_and_promote(candidate,plan=candidate,authority='candidate',token=token,validate=reject)
        self.assertIs(controller.incumbent,plan)
        self.assertEqual(controller.transaction_token(),token)
        controller.compare_and_promote(candidate,plan=candidate,authority='candidate',token=token,
            validate=lambda value:self.assertIs(controller.incumbent,plan))
        self.assertIs(controller.incumbent,candidate)
        self.assertEqual(controller.to_dict()['incumbent_revision'],token.revision+1)

    def test_atomic_adoption_detects_mutated_metadata_and_reentrant_change(self):
        _,controller,plan=self.make()
        owner={'music':[0.,1.]}
        controller.promote(owner,plan=plan,authority='mutable-test-only')
        token=controller.transaction_token();owner['music'][0]=99.
        with self.assertRaisesRegex(ValueError,'baseline changed'):
            controller.compare_and_promote(plan,plan=plan,authority='candidate',token=token,validate=lambda value:None)
        self.assertIs(controller.incumbent,owner)
        token=controller.transaction_token()
        def reentrant(value):controller.promote(plan,plan=plan,authority='new-owner')
        with self.assertRaisesRegex(ValueError,'during parent'):
            controller.compare_and_promote(owner,plan=plan,authority='stale-owner',token=token,validate=reentrant)
        self.assertIs(controller.incumbent,plan)

    def test_admission_stops_at_search_cutoff_without_extending_total_budget(self):
        for elapsed in (359., 360., 361., 599., 600.):
            with self.subTest(elapsed=elapsed):
                clock, controller, plan = self.make()
                clock[0] = elapsed
                calls = []
                with patch('crosscut.first_result_controller.enforce_review_first_deadline', return_value=nullcontext()):
                    if elapsed >= 600:
                        with self.assertRaises(ReviewFirstDeadlineExceeded):
                            controller.run_optional('candidate', lambda: calls.append(True))
                    else:
                        controller.run_optional('candidate', lambda: calls.append(True))
                self.assertEqual(len(calls), int(elapsed < 360))
                self.assertIs(controller.incumbent, plan)

    def test_foreign_sources_time_range_and_invalid_source_bounds_cannot_promote(self):
        _, controller, plan = self.make()
        for altered in (replace(plan, sources=(Path('/synthetic/foreign.mp4'), plan.sources[1])),
                        replace(plan, master_audio_start=1.),
                        replace(plan, reference_source=1), replace(plan, fps=60.),
                        replace(plan, width=1280, height=720),
                        replace(plan, segments=(replace(plan.segments[0], source_start=10.), plan.segments[1]))):
            with self.subTest(plan=altered):
                with self.assertRaises(ValueError):
                    controller.promote(altered, plan=altered, authority='unverified')
                self.assertIs(controller.incumbent, plan)

    def test_promoted_plan_cannot_use_an_ineligible_configured_source(self):
        _, original, plan = self.make()
        baseline = replace(plan, segments=tuple(replace(segment, source_index=0) for segment in plan.segments))
        controller = FirstResultController(original.runtime, sources=plan.sources, source_durations=(10., 10.),
            master_start=0., duration=2., allowed_source_indices=(0,))
        controller.install_baseline(baseline, plan=baseline, authority='verified-eligible-only')
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            controller.promote(plan, plan=plan, authority='foreign-source')
        self.assertIs(controller.incumbent, baseline)

    def test_search_cutoff_is_not_wrapped_by_generic_model_error_recovery(self):
        def adapter():
            try: raise ReviewFirstSearchCutoff('deadline')
            except Exception as error: raise ValueError('candidate rejected') from error
        with self.assertRaises(ReviewFirstSearchCutoff): adapter()

    def test_final_state_requires_bound_technical_disposition_and_render_event(self):
        for content_bound in (False, True):
            for passed in (False, True):
                with self.subTest(content_bound=content_bound, passed=passed):
                    _, controller, _ = self.make()
                    controller.render_complete()
                    report = {"passed": passed, "issues": [] if passed else [{"code": "timeline_review_failed"}]}
                    disposition = classify_review_first_qa(report, content_binding_passed=content_bound,
                        review_first_enabled=True, member=False)
                    manifest = {'review_first_visual_selection': {'first_result_controller': controller.to_dict()}}
                    finalize_first_result_manifest(manifest, disposition)
                    state = manifest['review_first_visual_selection']['first_result_controller']
                    self.assertEqual(state['stage'], disposition.status.value)
                    self.assertEqual(state['quality_approved'], disposition.status is ReviewFirstStatus.APPROVED)
                    self.assertEqual('TECHNICALLY_VERIFIED' in state['stage_history'], disposition.video_output_exposed)
                    with self.assertRaises(RuntimeError): controller.render_complete()
