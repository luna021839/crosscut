"""Synthetic numeric transport tests; no media/model/actor-quality claims."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest

from crosscut.boundary_look_match import BoundaryLookMatchResult
from crosscut.face_match import FaceMatchPlan, FaceMatchSegment
from crosscut.group_joint_bundle_metadata import (GroupNativeMotionProduction,
    SelectedNativeActorFrame, SelectedNativeMotionShot, seal_native_motion_production,
    validate_native_motion_production)
from crosscut.group_joint_native_preparation import _plan_identity
from crosscut.group_joint_source_camera import GroupOutputRole
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.pipeline import (_derive_complete_group_candidate, _review_first_group_bundle,
    _validate_review_first_group_bundle)
from crosscut.render import NormalizedCrop, PanCropKeyframe, PanCropTrack
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.section_director import SectionPlan, SectionDirective, SectionRole
from crosscut.timeline import ShotType
from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def synthetic_bundle():
    sources=tuple(Path(f'/synthetic/unopened-{i}.mp4') for i in range(6))
    clock=VerifiedRenderFrameClock(Fraction(30),10000,'a'*64)
    domain=GroupSourceDomain(sources,(0,2,3,5),(0.,)*4,(None,)*4,((),)*4,(clock,)*4)
    timeline=VisualTimelinePlan(tuple(VisualTimelineChoice(segment_index=i,source_index=i,
        canonical_master_start=float(i),canonical_master_end=float(i+1),
        source_local_start=float(i),source_local_end=float(i+1),
        quality_score=.8,composition_score=.8,boundary_score=.5,shot_scale=.4,
        shot_type=ShotType.MEDIUM,shot_id=f'synthetic-unobserved-{i}') for i in range(4)),1.)
    section=SectionPlan((SectionDirective(0,SectionRole.BUILD,0.,4.,.5,.5,1,4,15.,.4,.1,False,False),))
    request=SimpleNamespace(sources=sources,reference_source=0,aspect_ratio='16:9')
    bundle=_review_first_group_bundle(timeline=timeline,request=request,
        eligible_indices=domain.eligible_indices,offsets=domain.audio_offsets,
        timewarps=domain.source_timewarps,common_start=0.,section_plan=section,
        phrase_times=(0.,1.,2.,3.,4.),output=Path('/synthetic/not-rendered.mp4'),
        reporter=SimpleNamespace(log=lambda *a,**k:None),warnings=[])
    pan=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.2,.2,.6,.6)),))
    plan=replace(bundle.render_plan,segments=tuple(replace(s,pan_crop=pan) for s in bundle.render_plan.segments))
    face=FaceMatchPlan(tuple(FaceMatchSegment(i,i,True,None,.2,.22,1.1,1.1,pan,
        alignment_subject='face',alignment_purpose='SYNTHETIC_TRANSPORT_ONLY') for i in range(4)),
        .22,2.,(.5,.5),analysis_basis='SYNTHETIC_TRANSPORT_ONLY_NO_OBSERVATIONS')
    look=BoundaryLookMatchResult(False,'SYNTHETIC_UNOBSERVED_LOOK',(),(None,)*4,0,0)
    repairs=VisualBoundaryRepairPlan(timeline,(timeline,),(),0,(),
        analysis_basis='SYNTHETIC_TRANSPORT_ONLY_NO_REPAIR_SUCCESS')
    bundle=replace(bundle,render_plan=plan,face_match_plan=face,boundary_look_match=look,
        boundary_repairs=repairs,candidate_index=7,review_rank=3,
        cadence_profile='synthetic_native_program_consumer',candidate_origin='normal_native_motion_program',
        dynamic_crop_application={'contract':'SYNTHETIC_RETAIN_ONLY','quality_approved':False},
        diagnostics=('SYNTHETIC_NO_MODEL_OR_MEDIA',),candidate_lineage=({'stage':'synthetic_numeric_transport'},))
    return bundle,domain


def synthetic_production(*,state='SELECTED_FOR_REVIEW'):
    bundle,domain=synthetic_bundle()
    schedule=native_frame_clocks_for_output(bundle.render_plan,domain.clocks_by_request,range(30,90))
    # Unknown actor rows are retained, not fabricated successful detections.
    shots=tuple(SelectedNativeMotionShot(a,b,source,(0,10000),GroupOutputRole.FOCAL_HEAD_ARMS,
        0,None,tuple(SelectedNativeActorFrame(tick,source,schedule[tick].native_frame_index,None,None)
            for tick in range(a,b)),(i,)) for a,b,source,i in ((30,60,2,1),(60,90,3,2)))
    production=seal_native_motion_production(complete_bundle=bundle,source_domain=domain,
        shot_clock_context=None,selected_shots=shots if state!='KEEP_ORIGINAL' else (),
        original_plan_identity=_plan_identity(bundle.render_plan),protected_output_intervals=((0,30),(90,120)),
        state=state,selection_evidence={'contract':'SYNTHETIC_METADATA_TRANSPORT_ONLY',
            'actor_identity_state':'UNKNOWN','quality_approved':False})
    return production


def consumer_kwargs(production):
    bundle=production.complete_bundle;domain=production.source_domain;plan=bundle.render_plan
    return dict(request=SimpleNamespace(sources=plan.sources,reference_source=0,aspect_ratio=plan.aspect_ratio),
        eligible_indices=domain.eligible_indices,audio_offsets=domain.audio_offsets,
        source_timewarps=domain.source_timewarps,scene_cut_times=domain.scene_cut_times,
        source_aspect_ratios=(16/9,)*4,source_durations_seconds=(10.,)*4,beat_period_seconds=1.,
        phrase_times=bundle.phrase_times,section_plan=bundle.section_plan,common_master_start=0.,
        width=plan.width,height=plan.height,output=Path('/synthetic/relocated-only.mp4'),
        native_motion_production=production,native_motion_reporter=SimpleNamespace(log=lambda *a:None),
        native_motion_warnings=[])


class NativeMotionCompleteBundleTests(unittest.TestCase):
    def consume(self,production,**changes):
        selected=production.complete_bundle
        materialized=SimpleNamespace(visual_plan=selected.visual_timeline,candidate_index=selected.candidate_index,
            review_rank=selected.review_rank,cadence_profile=selected.cadence_profile)
        return _derive_complete_group_candidate(materialized,**dict(consumer_kwargs(production),**changes))

    def validate(self,bundle):
        domain=bundle.native_motion_production.source_domain
        _validate_review_first_group_bundle(bundle,eligible_indices=domain.eligible_indices,
            offsets=domain.audio_offsets,timewarps=domain.source_timewarps,common_start=0.,
            phrase_times=bundle.phrase_times)

    def test_complete_metadata_survives_normal_consumer_without_neutral_reconstruction(self):
        production=synthetic_production();selected=production.complete_bundle
        actual=self.consume(production)
        for field in ('face_match_plan','boundary_look_match','boundary_repairs','boundary_affine_plan',
                'editorial_motion_plan','group_transition_plan','section_plan','candidate_lineage'):
            self.assertIs(getattr(actual,field),getattr(selected,field),field)
        for field in ('stage_edit_exact_shot_clock','dynamic_crop_application'):
            self.assertEqual(getattr(actual,field),getattr(selected,field),field)
        self.assertEqual(actual.render_plan,replace(selected.render_plan,output=consumer_kwargs(production)['output']))
        self.assertIs(actual.native_motion_production,production)
        self.assertEqual((actual.candidate_index,actual.review_rank,actual.cadence_profile),(7,3,selected.cadence_profile))
        self.validate(actual)
        self.assertFalse(production.to_dict()['quality_approved'])
        self.assertIsNone(production.complete_bundle.native_motion_production)

    def test_missing_complete_bundle_cannot_be_replaced_by_neutral_metadata(self):
        production=synthetic_production()
        partial=replace(production,complete_bundle=None,selection_evidence_json=None,selection_sha256=None)
        with self.assertRaisesRegex(ValueError,'complete selected bundle'):
            self.consume(production,native_motion_production=partial)

    def test_face_look_repair_and_other_metadata_cannot_be_dropped_after_selection(self):
        actual=self.consume(synthetic_production())
        for field in ('face_match_plan','boundary_look_match','boundary_repairs','dynamic_crop_application'):
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'after motion/camera evaluation'):
                self.validate(replace(actual,**{field:None}))

    def test_frozen_stream_identity_cannot_be_relabelled_for_another_candidate(self):
        production=synthetic_production();selected=production.complete_bundle
        for field,value in (('candidate_index',8),('review_rank',4),('cadence_profile','another_profile')):
            materialized=SimpleNamespace(visual_plan=selected.visual_timeline,candidate_index=7,
                review_rank=3,cadence_profile=selected.cadence_profile)
            setattr(materialized,field,value)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'sealed stream identity'):
                _derive_complete_group_candidate(materialized,**consumer_kwargs(production))

    def test_render_header_and_source_clock_mutations_are_not_hidden_by_same_segments(self):
        actual=self.consume(synthetic_production())
        for field,value in (('reference_source',2),('fps',30.),('width',1280),('master_audio_start',.01)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.validate(replace(actual,render_plan=replace(actual.render_plan,**{field:value})))

    def test_music_authority_cannot_be_silently_reconstructed(self):
        production=synthetic_production()
        with self.assertRaisesRegex(ValueError,'music/section authority'):
            self.consume(production,phrase_times=(0.,.5,2.,3.,4.))

    def test_legacy_camera_repair_override_cannot_enter_native_consumer(self):
        production=synthetic_production()
        with self.assertRaisesRegex(ValueError,'corresponding materialized'):
            self.consume(production,visual_timeline_override=production.visual_timeline)

    def test_unknown_actor_rows_cannot_be_erased_or_replaced_by_wrong_native_frame(self):
        production=synthetic_production();shot=production.selected_shots[0]
        for changed in (replace(shot,actor_frames=shot.actor_frames[:-1]),
                replace(shot,actor_frames=(replace(shot.actor_frames[0],native_frame_index=31),*shot.actor_frames[1:]))):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                validate_native_motion_production(replace(production,selected_shots=(changed,production.selected_shots[1])))

    def test_mutating_nested_retained_metadata_breaks_selection_seal(self):
        production=synthetic_production()
        production.complete_bundle.candidate_lineage[0]['stage']='changed-after-selection'
        with self.assertRaisesRegex(ValueError,'after motion/camera evaluation'):
            validate_native_motion_production(production)

    def test_selection_evidence_and_protected_intervals_are_sealed_together(self):
        production=synthetic_production()
        for changed in (replace(production,selection_evidence_json='{}'),
                replace(production,protected_output_intervals=((0,29),(90,120)))):
            with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'after motion/camera evaluation'):
                validate_native_motion_production(changed)

    def test_keep_original_carries_the_same_full_metadata_without_inventing_actor_paths(self):
        production=synthetic_production(state='KEEP_ORIGINAL')
        self.validate(self.consume(production))
        self.assertEqual(production.selected_shots,())
        self.assertFalse(production.to_dict()['quality_approved'])

    def test_phase_claim_cannot_substitute_for_a_real_authorized_video_clock(self):
        production=synthetic_production()
        changed=replace(production.selected_shots[0],phase_native_frames=1)
        with self.assertRaisesRegex(ValueError,'authorized whole-shot clock'):
            validate_native_motion_production(replace(production,selected_shots=(changed,production.selected_shots[1])))

    def test_early_legacy_geometry_blocks_are_bypassed_for_a_retained_native_bundle(self):
        import ast
        import crosscut.pipeline as pipeline
        # Inspect only the allowed GROUP pipeline source. Evaluate its actual
        # branch conditions, without running acquisition, models or MEMBER QA.
        source=ast.parse(Path(pipeline.__file__).read_text())
        tests=[node.test for node in ast.walk(source) if isinstance(node,ast.If)
            and 'selected_native_group_bundle' in {n.id for n in ast.walk(node.test) if isinstance(n,ast.Name)}
            and 'geometry_basis' in {n.attr for n in ast.walk(node.test) if isinstance(n,ast.Attribute)}]
        self.assertEqual(len(tests),3)
        for condition in tests:
            for basis in ('source_pixels','prescribed_render_geometry_v1'):
                result=eval(compile(ast.Expression(condition),'<GROUP-only legacy condition>','eval'),{
                    'selected_native_group_bundle':object(),'visual_timeline':SimpleNamespace(geometry_basis=basis),
                    'review_first_runtime':SimpleNamespace(enabled=True,optional_refinement_allowed=True),
                    'group_first':object()})
                self.assertFalse(result)

    def test_phase_zero_parent_context_is_part_of_the_seal(self):
        from crosscut.group_joint_native_phase import NativePhaseSearchPolicy
        from crosscut.shot_video_clock import FrozenShotVideoClockContext
        old=synthetic_production();context=FrozenShotVideoClockContext(old.render_plan,old.source_domain,NativePhaseSearchPolicy())
        production=seal_native_motion_production(complete_bundle=old.complete_bundle,source_domain=old.source_domain,
            shot_clock_context=context,selected_shots=old.selected_shots,original_plan_identity=old.original_plan_identity,
            protected_output_intervals=old.protected_output_intervals,state=old.state,
            selection_evidence={'contract':'SYNTHETIC_CONTEXT_BINDING'})
        self.validate(self.consume(production))
        changed=replace(context,incumbent_plan=replace(context.incumbent_plan,output=Path('/synthetic/changed-parent.mp4')))
        with self.assertRaisesRegex(ValueError,'after motion/camera evaluation'):
            validate_native_motion_production(replace(production,shot_clock_context=changed))
