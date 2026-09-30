"""Bilateral camera states are an atomic, densely checked cut correction."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from crosscut.boundary_subject_alignment import (
    BoundarySubjectObservation, build_boundary_subject_alignment_plan,
)
from crosscut.editorial_motion import build_editorial_motion_plan
from crosscut.ive import AspectRatio
from crosscut.post_render_qa import (
    _bound_local_alignment_tracks, _transform_metrics, _IssueCollector, PostRenderQAConfig,
)
from crosscut.render import RenderPlan, RenderSegment, RenderTransition
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_boundary_subject_alignment import _choice, _frame


def observed(source):
    x,y,w,h = ((.4477,.3361,.0881,.2172), (.4454,.5005,.0840,.2176))[source]
    return BoundarySubjectObservation(face_count=1, face_box_xywh=(x,y,w,h),
        face_eye_scale=(.07148,.06944)[source],
        face_landmarks_xy=tuple((x+a*w,y+b*h) for a,b in ((.3,.3),(.7,.3),(.5,.5),(.3,.8),(.7,.8))),
        detected_person_count=1,
        formation_boxes_xywh=(((.30,.27,.35,.72),), ((.30,.445,.35,.543),))[source])


def produce(*, frames=None, observe=None, choices=None, resolution=(1920,1080)):
    choices=choices or (_choice(0,0,start=0.,end=1.), _choice(1,1,start=1.,end=2.))
    visual=VisualTimelinePlan(choices,1.)
    sources=(Path('/tmp/joint-source-a.mp4'),Path('/tmp/joint-source-b.mp4'))
    transitions=(RenderTransition(tier='C'),)*(len(choices)-1)
    alignment=build_boundary_subject_alignment_plan(visual_timeline=visual,
        sources=sources,transitions=transitions,fps=30.,
        frame_provider=frames or (lambda source,_seconds:_frame(source)),
        observation_provider=observe or (lambda frame:observed(int(frame.mean()>75))),
        source_resolutions=(resolution,)*2)
    motion=build_editorial_motion_plan(visual_timeline=visual,transitions=transitions,
        fps=30.,boundary_subject_alignment_plan=alignment)
    plan=RenderPlan(sources=sources,reference_source=0,aspect_ratio=AspectRatio.LANDSCAPE,
        output=Path('/tmp/joint-fixture.mp4'),fps=30.,transitions=transitions,
        segments=tuple(RenderSegment(item.source_index,item.source_local_start,
            item.duration_seconds,item.canonical_master_start,
            editorial_transform=motion.track_for(item.segment_index)) for item in choices))
    return alignment,motion,plan


class GroupJointFramingTests(unittest.TestCase):
    def test_equal_size_displaced_faces_use_two_complete_windows(self):
        alignment,motion,plan=produce()
        decision=alignment.decisions[0]
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.transformed_side,'both')
        self.assertLess(decision.transformed_geometry_mismatch,.7*decision.baseline_geometry_mismatch)
        self.assertEqual(decision.outgoing_validated_output_frame_count,13)
        self.assertEqual(decision.incoming_validated_output_frame_count,13)
        self.assertEqual(decision.validated_output_frame_count,26)
        self.assertTrue(all(segment.editorial_transform is not None for segment in plan.segments))
        self.assertEqual(_bound_local_alignment_tracks(motion.to_dict(),json.loads(json.dumps(plan.to_json()))),
            ({0,1},{0}))

    def test_current_one_sided_family_is_retained_before_paired_fallback(self):
        def observe(frame):
            value=observed(int(frame.mean()>75))
            return replace(value,face_box_xywh=(.4,.3,.16,.30),face_eye_scale=.105,
                formation_boxes_xywh=((.30,.23,.40,.77),)) if frame.mean()<75 else value
        with patch('crosscut.boundary_subject_alignment._joint_face_alignment_states',return_value=()):
            before,_,_=produce(observe=observe)
        after,_,_=produce(observe=observe)
        self.assertTrue(before.decisions[0].eligible)
        self.assertEqual(after.decisions[0],before.decisions[0])
        self.assertEqual(after.segments,before.segments)

    @staticmethod
    def _one_sided_fixture(frame):
        source=int(frame.mean()>75)
        x,y,w,h=((.42,.28,.15,.27),(.44,.40,.135,.24))[source]
        return replace(observed(source),face_box_xywh=(x,y,w,h),face_eye_scale=h*.35,
            face_landmarks_xy=tuple((x+a*w,y+b*h) for a,b in
                ((.3,.3),(.7,.3),(.5,.5),(.3,.8),(.7,.8))))

    def test_bilateral_candidate_can_improve_a_passing_isolated_one_sided_ramp(self):
        with patch('crosscut.boundary_subject_alignment._joint_face_alignment_states',return_value=()):
            before,_,_=produce(observe=self._one_sided_fixture,resolution=(3840,2160))
        after,motion,plan=produce(observe=self._one_sided_fixture,resolution=(3840,2160))
        old,new=before.decisions[0],after.decisions[0]
        self.assertTrue(old.eligible)
        self.assertIn(old.transformed_side,('outgoing','incoming'))
        self.assertEqual(new.transformed_side,'both')
        self.assertEqual(old.transformed_geometry_mismatch,new.replaced_one_sided_geometry_mismatch)
        self.assertLess(new.transformed_geometry_mismatch,old.transformed_geometry_mismatch-.015)
        self.assertLessEqual(new.transformed_layout_mismatch,old.transformed_layout_mismatch)
        self.assertEqual(new.validated_output_frame_count,26)
        self.assertEqual(_bound_local_alignment_tracks(motion.to_dict(),json.loads(json.dumps(plan.to_json()))),
                         ({0,1},{0}))

    def test_failed_bilateral_window_keeps_complete_valid_one_sided_incumbent(self):
        # The larger face is the untouched side of the initial one-sided crop.
        # Its missing interior frame must reject the two-sided replacement,
        # without erasing the already valid curve on the other side.
        frames=lambda index,time: None if index==0 and abs(time-25/30.)<1e-8 else _frame(index)
        with patch('crosscut.boundary_subject_alignment._joint_face_alignment_states',return_value=()):
            before,_,_=produce(frames=frames,observe=self._one_sided_fixture,resolution=(3840,2160))
        after,_,_=produce(frames=frames,observe=self._one_sided_fixture,resolution=(3840,2160))
        self.assertTrue(before.decisions[0].eligible)
        self.assertEqual(before.decisions,after.decisions)
        self.assertEqual(before.segments,after.segments)

    def test_bilateral_replacement_receipt_requires_a_measured_strict_improvement(self):
        alignment,_,_=produce(observe=self._one_sided_fixture,resolution=(3840,2160))
        decision=alignment.decisions[0]
        self.assertIsNotNone(decision.replaced_one_sided_geometry_mismatch)
        for previous in (True, float('nan'), float('inf'), -1., 1.1,
                         decision.transformed_geometry_mismatch):
            with self.subTest(previous=previous), self.assertRaises(ValueError):
                replace(decision,replaced_one_sided_geometry_mismatch=previous)
        with self.assertRaises(ValueError):
            replace(decision,transformed_side='incoming')
        self.assertEqual(decision.to_dict()['replaced_one_sided_geometry_mismatch'],
                         decision.replaced_one_sided_geometry_mismatch)

    def test_bilateral_competition_cannot_overwrite_a_neighboring_ramp(self):
        choices=(_choice(0,0,start=0.,end=1.), _choice(1,1,start=1.,end=2.),
                 _choice(2,0,start=2.,end=3.))
        with patch('crosscut.boundary_subject_alignment._joint_face_alignment_states',return_value=()):
            before,_,_=produce(choices=choices,observe=self._one_sided_fixture,resolution=(3840,2160))
        after,_,_=produce(choices=choices,observe=self._one_sided_fixture,resolution=(3840,2160))
        self.assertEqual(before.eligible_count,2)
        self.assertEqual(before.segments[1].boundary_indices,(0,1))
        self.assertEqual(before.decisions,after.decisions)
        self.assertEqual(before.segments,after.segments)

    def test_missing_frame_on_either_side_publishes_neither_transform(self):
        for source,seconds in ((0,25/30.),(1,1.+4/30.)):
            with self.subTest(source=source):
                alignment,_,_=produce(frames=lambda index,time:None
                    if index==source and abs(time-seconds)<1e-8 else _frame(index))
                self.assertFalse(alignment.decisions[0].eligible)
                self.assertTrue(all(segment.track is None for segment in alignment.segments))

    def test_paired_fallback_never_crosses_a_native_camera_cut(self):
        choices=(_choice(0,0,start=0.,end=1.,exit_cut=True),_choice(1,1,start=1.,end=2.))
        alignment,_,_=produce(choices=choices)
        self.assertFalse(alignment.decisions[0].eligible)
        self.assertTrue(all(segment.track is None for segment in alignment.segments))

    def test_missing_landmarks_do_not_create_a_bilateral_face_claim(self):
        alignment,_,_=produce(observe=lambda frame:replace(observed(int(frame.mean()>75)),
            face_landmarks_xy=None))
        self.assertFalse(alignment.decisions[0].eligible)

    def test_final_qa_requires_both_actual_tracks_and_each_side_count(self):
        _,motion,plan=produce()
        original=motion.to_dict()
        serialized=json.loads(json.dumps(plan.to_json()))
        for mode in ('left_count','right_count','total','endpoint','left_track','right_track','old_basis'):
            with self.subTest(mode=mode):
                payload=deepcopy(original); candidate=deepcopy(serialized)
                report=payload['boundary_subject_alignment']; decision=report['decisions'][0]
                if mode=='left_count': decision['outgoing_validated_output_frame_count']=0
                elif mode=='right_count': decision['incoming_validated_output_frame_count']=0
                elif mode=='total': decision['validated_output_frame_count']=1
                elif mode=='endpoint': decision['outgoing_state']['center_y']=.5
                elif mode=='old_basis': report['analysis_basis']='exact_cut_frame_shot_adaptive_subject_alignment_v4'
                else: candidate['segments'][0 if mode=='left_track' else 1]['editorial_transform']=None
                self.assertEqual(_bound_local_alignment_tracks(payload,candidate),(set(),set()))

    def test_larger_bilateral_pan_keeps_native_zoom_and_unbound_track_checks(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            alignment,motion,plan=produce(resolution=(3840,2160))
            sources=tuple(Path(directory)/f'{index}.mp4' for index in (0,1))
            for source in sources: source.write_bytes(b'explicit native probe fixture')
            plan=replace(plan,sources=sources)
            serialized=json.loads(json.dumps(plan.to_json()))
            payload=motion.to_dict()
            self.assertGreater(max(abs(state.center_y-.5) for state in (
                alignment.decisions[0].outgoing_state, alignment.decisions[0].incoming_state)),.04)
            def probe(command,_timeout):
                return subprocess.CompletedProcess(command,0,json.dumps({'streams':[
                    dict(width=3840,height=2160,sample_aspect_ratio='1:1')]}),'')
            collector=_IssueCollector()
            metrics=_transform_metrics({'editorial_motion':payload},serialized,PostRenderQAConfig(),collector,runner=probe)
            self.assertEqual(set(metrics['native_resolution_zoom_budgets']),{0,1})
            self.assertNotIn('editorial_transform_safety_bound_exceeded',[issue.code for issue in collector.issues])
            payload['boundary_subject_alignment']['decisions'][0]['incoming_validated_output_frame_count']=0
            rejected=_IssueCollector()
            _transform_metrics({'editorial_motion':payload},serialized,PostRenderQAConfig(),rejected,runner=probe)
            self.assertIn('editorial_transform_safety_bound_exceeded',[issue.code for issue in rejected.issues])


if __name__=='__main__': unittest.main()
