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
    def test_realized_producer_camera_survives_normal_bundle_constructor_exactly(self):
        from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe
        from crosscut.editorial_transform_events import rendered_transform_event_records
        original=self.make_bundle()
        track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.2,.48,.46,1.),
            EditorialTransformKeyframe(.8,1.4,.51,.48,-1.)),reason='measured_producer_camera')
        plan=replace(original.render_plan,segments=(replace(original.render_plan.segments[0],
            editorial_transform=track),original.render_plan.segments[1]))
        records=rendered_transform_event_records(plan.segments,plan.fps)
        motion=replace(original.editorial_motion_plan,segments=(replace(original.editorial_motion_plan.segments[0],
            track=track),original.editorial_motion_plan.segments[1]),
            zoom_event_count=len(records['zoom']),pan_event_count=len(records['pan']),
            roll_event_count=len(records['roll']),authored_event_counts=(('zoom',0),('pan',0),('roll',0)))
        request=SimpleNamespace(sources=plan.sources,reference_source=plan.reference_source,aspect_ratio=plan.aspect_ratio)
        def build(**changes):
            return _review_first_group_bundle(timeline=original.visual_timeline,request=request,
                eligible_indices=(0,1),offsets=(0.,0.),timewarps=(None,None),common_start=0.,
                section_plan=original.section_plan,phrase_times=original.phrase_times,
                output=Path('/synthetic/realized-output'),reporter=SimpleNamespace(log=lambda *a,**k:None),warnings=[],
                **dict(dict(realized_plan=plan,realized_motion=motion),**changes))
        bundle=build();self.validate(bundle)
        self.assertEqual(bundle.render_plan,replace(plan,output=Path('/synthetic/realized-output')))
        self.assertIs(bundle.editorial_motion_plan,motion)
        self.assertEqual(bundle.render_plan.segments[0].editorial_transform,track)
        with self.assertRaisesRegex(ValueError,'both renderer and motion'):build(realized_motion=None)
        with self.assertRaises(ValueError):build(realized_motion=original.editorial_motion_plan)
        with self.assertRaisesRegex(ValueError,'source/reference clocks'):
            build(realized_plan=replace(plan,segments=(replace(plan.segments[0],source_start=.1),plan.segments[1])))

    def test_stale_or_nonfinite_source_local_descriptor_clock_cannot_promote(self):
        bundle = self.make_bundle()
        for field, value in (('source_local_start', .01), ('source_local_end', 1.01),
                ('source_local_start', float('nan')), ('source_local_end', float('inf')),
                ('source_local_start', False), ('canonical_master_start', False), ('segment_index', False)):
            timeline = replace(bundle.visual_timeline, choices=(replace(
                bundle.visual_timeline.choices[0], **{field: value}), *bundle.visual_timeline.choices[1:]))
            from crosscut.timeline_review import review_timeline
            review = review_timeline(timeline, available_source_count=2)
            candidate = replace(bundle, visual_timeline=timeline, timeline_review=review,
                pre_render_score=review.weighted_score)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.validate(candidate)

    def test_boolean_shot_ids_are_not_integer_music_authority(self):
        bundle = self.make_bundle()
        for field in ('shot_index', 'source_index'):
            clock = _thaw_audit_value(bundle.stage_edit_exact_shot_clock)
            clock['shots'][0][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(replace(bundle, stage_edit_exact_shot_clock=clock))

    def make_bundle(self):
        section = SectionPlan((SectionDirective(0, SectionRole.BUILD, 0., 2., .5, .5, 1,
            4, 15., .4, .1, False, False),))
        request = SimpleNamespace(sources=(Path('/synthetic/a'), Path('/synthetic/b')),
            reference_source=0, aspect_ratio=AspectRatio.LANDSCAPE)
        return _review_first_group_bundle(timeline=VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 1.),
            request=request, eligible_indices=(0, 1), offsets=(0., 0.), timewarps=(None, None),
            common_start=0., section_plan=section, phrase_times=(0., 1., 2.),
            output=Path('/synthetic/out'), reporter=SimpleNamespace(log=lambda *a,**k:None), warnings=[])

    def validate(self, bundle):
        _validate_review_first_group_bundle(bundle, eligible_indices=(0, 1), offsets=(0., 0.),
            timewarps=(None, None), common_start=0.,phrase_times=(0.,1.,2.))

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

    def test_in_range_stale_phrase_progress_and_forged_review_score_cannot_promote(self):
        bundle=self.make_bundle()
        clock=_thaw_audit_value(bundle.stage_edit_exact_shot_clock)
        clock['shots'][0]['phrase_progress']=.419
        for candidate in (replace(bundle,stage_edit_exact_shot_clock=clock),
                replace(bundle,pre_render_score=bundle.pre_render_score+999.),
                replace(bundle,timeline_review=replace(bundle.timeline_review,weighted_score=99.))):
            with self.subTest(candidate=candidate),self.assertRaises(ValueError):self.validate(candidate)

    def test_stale_motion_source_or_camera_cannot_promote(self):
        from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe
        bundle=self.make_bundle();motion=bundle.editorial_motion_plan
        for item in (replace(motion.segments[0],source_index=1),
                replace(motion.segments[0],track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),)))):
            changed=replace(bundle,editorial_motion_plan=replace(motion,segments=(item,motion.segments[1])))
            with self.subTest(item=item),self.assertRaises(ValueError):self.validate(changed)

    def test_forged_event_inventory_is_rejected_before_any_encode(self):
        bundle=self.make_bundle();motion=bundle.editorial_motion_plan
        for counters in ({'zoom_event_count':999},{'pan_event_count':1},{'roll_event_count':1}):
            with self.subTest(counters=counters),self.assertRaisesRegex(ValueError,'event inventory'):
                self.validate(replace(bundle,editorial_motion_plan=replace(motion,**counters)))

    def test_frame_visible_inventory_matches_actual_curves_not_claimed_counts(self):
        from crosscut.editorial_transform_events import validate_rendered_motion_event_inventory
        from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe
        bundle=self.make_bundle();motion=bundle.editorial_motion_plan
        track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.),EditorialTransformKeyframe(.5,1.1)))
        segments=(replace(bundle.render_plan.segments[0],editorial_transform=track),bundle.render_plan.segments[1])
        typed=replace(motion,zoom_event_count=1,authored_event_counts=(('zoom',1),))
        records=validate_rendered_motion_event_inventory(segments,bundle.render_plan.fps,typed)
        self.assertEqual(len(records['zoom']),1)
        with self.assertRaisesRegex(ValueError,'frame-visible'):
            validate_rendered_motion_event_inventory(segments,bundle.render_plan.fps,replace(typed,zoom_event_count=0))

    def test_phrase_inputs_and_section_are_serialized_without_mutable_aliases(self):
        bundle=self.make_bundle();payload=bundle.to_dict()
        self.assertEqual(payload['phrase_times'],[0.,1.,2.])
        self.assertEqual(payload['section_plan'],bundle.section_plan.to_dict())
        payload['phrase_times'][0]=999.
        self.assertEqual(bundle.phrase_times,(0.,1.,2.))
        from crosscut.pipeline import _deterministic_exact_shot_clock_from_timeline
        changed=replace(bundle,phrase_times=(0.,2.),stage_edit_exact_shot_clock=
            _deterministic_exact_shot_clock_from_timeline(bundle.visual_timeline,
                section_plan=bundle.section_plan,phrase_times=(0.,2.)))
        with self.assertRaisesRegex(ValueError,'external phrase authority'):self.validate(changed)

    def test_nonidentity_joint_source_change_cannot_promote_with_old_motion_or_review(self):
        from engine.tests.test_group_joint_state_factory import JointStateFactoryTests
        from crosscut.group_joint_state_factory import candidate_joint_group_state,assemble_joint_group_states
        from crosscut.group_transition import GroupTransitionPlan
        from crosscut.first_result_controller import FirstResultController
        from crosscut.review_first_runtime import ReviewFirstRuntime
        plan,domain,states=JointStateFactoryTests().fixture()
        timeline=VisualTimelinePlan(tuple(_choice(i,i) for i in range(4)),1.)
        section=SectionPlan((SectionDirective(0,SectionRole.BUILD,0.,4.,.5,.5,1,4,15.,.4,.1,False,False),))
        request=SimpleNamespace(sources=plan.sources,reference_source=0,aspect_ratio=plan.aspect_ratio)
        def produce(current_timeline):
            return _review_first_group_bundle(timeline=current_timeline,request=request,
                eligible_indices=domain.eligible_indices,offsets=domain.audio_offsets,
                timewarps=domain.source_timewarps,common_start=0.,section_plan=section,
                phrase_times=(0.,1.,2.,3.,4.),output=plan.output,
                reporter=SimpleNamespace(log=lambda *a,**k:None),warnings=[])
        baseline=produce(timeline)
        b=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[1],eligible_source_index=2)
        c=candidate_joint_group_state(plan=plan,domain=domain,incumbent_state=states[2],eligible_source_index=1)
        proposed,_=assemble_joint_group_states(plan=plan,domain=domain,path=(states[0],b,c,states[3]))
        # New synthetic source descriptors enter the real owned producer;
        # no old source's feature row or motion is copied into this bundle.
        fresh=produce(VisualTimelinePlan(tuple(_choice(i,s) for i,s in enumerate((0,2,1,3))),1.))
        fresh=replace(fresh,render_plan=proposed,group_transition_plan=GroupTransitionPlan(proposed.transitions))
        def validate(value):
            return _validate_review_first_group_bundle(value,eligible_indices=domain.eligible_indices,
                offsets=domain.audio_offsets,timewarps=domain.source_timewarps,common_start=0.,
                phrase_times=(0.,1.,2.,3.,4.))
        validate(fresh)
        self.assertEqual(tuple(s.source_index for s in fresh.render_plan.segments),(0,3,2,5))
        self.assertEqual(tuple(s.source_index for s in fresh.editorial_motion_plan.segments),(0,2,1,3))
        controller=FirstResultController(ReviewFirstRuntime(enabled=True),sources=plan.sources,
            source_durations=(60.,)*6,master_start=0.,duration=4.,allowed_source_indices=domain.eligible_indices)
        controller.install_baseline(baseline,plan=baseline.render_plan,authority='synthetic_complete_baseline')
        for candidate in (replace(fresh,editorial_motion_plan=baseline.editorial_motion_plan),
                replace(fresh,timeline_review=baseline.timeline_review,pre_render_score=baseline.pre_render_score)):
            self.assertIsNone(controller.run_optional('joint_source_camera',lambda:candidate,validate=validate))
            self.assertIs(controller.incumbent,baseline)
        self.assertFalse(controller.to_dict()['quality_approved'])

    def test_existing_joint_candidate_producer_retains_music_and_source_authority_for_owned_consumer(self):
        from crosscut.pipeline import _derive_complete_joint_group_candidate
        from crosscut.ive import SourceMode
        baseline=self.make_bundle()
        sources=(baseline.render_plan.sources[0],Path('/synthetic/excluded'),baseline.render_plan.sources[1])
        request=SimpleNamespace(sources=sources,reference_source=0,aspect_ratio=AspectRatio.PORTRAIT,
            source_mode=SourceMode.LOCAL)
        # Provider objects are synthetic. This checks the actual bundle
        # constructor/consumer, not decoded evidence or editorial quality.
        materialized=SimpleNamespace(common_master_start=0.,visual_plan=baseline.visual_timeline,
            editorial_motion_plan=baseline.editorial_motion_plan,
            segments=baseline.render_plan.segments,output_fps=baseline.render_plan.fps,
            candidate_index=0,review_rank=0,cadence_profile='synthetic_joint_producer',
            privacy_safe_signature='synthetic_only',candidate_history_entry={},
            boundary_proof_bundle=SimpleNamespace(selection_identity='synthetic_only',
                to_dict=lambda:dict(selection_identity='synthetic_only')))
        owned=_derive_complete_joint_group_candidate(materialized,request=request,eligible_indices=(0,2),
            source_aspect_ratios=(16/9,)*2,phrase_times=baseline.phrase_times,
            section_plan=baseline.section_plan,common_master_start=0.,width=1080,height=1920,
            output=Path('/synthetic/joint-not-rendered'),source_look_match=None,
            boundary_look_measurement_cache=None)
        _validate_review_first_group_bundle(owned,eligible_indices=(0,2),offsets=(0.,0.),
            timewarps=(None,None),common_start=0.,phrase_times=baseline.phrase_times)
        self.assertIs(owned.section_plan,baseline.section_plan)
        self.assertEqual(owned.phrase_times,baseline.phrase_times)
        self.assertEqual(owned.editorial_motion_plan.eligible_request_indices,(0,2))
        self.assertEqual([row['source_index'] for row in owned.to_dict()['editorial_motion']['segments']],[0,2])
        self.assertEqual([item.source_index for item in owned.editorial_motion_plan.segments],[0,1])
