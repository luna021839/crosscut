from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from crosscut.boundary_subject_alignment import BoundarySubjectObservation, _close_face_head_envelope
from crosscut.shot_context_camera import build_shot_context_camera,fixed_neighbor_face_state,propose_shot_context_camera
from crosscut.render import RenderPlan,RenderSegment,RenderTransition
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.render_boundary_clock import native_frame_clocks_for_output


def face(h=.2,eye=.07,x=.44,y=.3):
    return BoundarySubjectObservation(face_count=1,face_box_xywh=(x,y,.12,h),
        face_landmarks_xy=((x+.03,y+.05),(x+.09,y+.05),(x+.06,y+.09),(x+.04,y+.13),(x+.08,y+.13)),
        face_head_roll_radians=0.,face_eye_scale=eye,face_is_unambiguous_dominant=True)


class ShotContextCameraTests(unittest.TestCase):
    def test_bounded_residual_can_keep_one_native_endpoint_while_matching_the_other(self):
        # The desired exit scale is physically impossible but the raw exit is
        # already close. Keep it native; do not discard the independent entry.
        source=face();target=face(.197,.069)
        with self.assertRaisesRegex(ValueError,'pixel budget'):
            fixed_neighbor_face_state(source,target,maximum_zoom=2.)
        fixed=fixed_neighbor_face_state(source,target,maximum_zoom=2.,endpoint_scale_policy='bounded_residual')
        self.assertEqual((fixed.zoom,fixed.center_x,fixed.center_y,fixed.roll_degrees),(1.,.5,.5,0.))
        heads=(_close_face_head_envelope(source.face_box_xywh),)*31
        camera=build_shot_context_camera(source_endpoints=(source,source),
            neighbor_endpoints=(face(.30,.1),target),head_envelopes=heads,fps=30.,maximum_zoom=2.,
            endpoint_scale_policy='bounded_residual')
        self.assertAlmostEqual(camera.entry.zoom,1.5)
        self.assertEqual(camera.exit.zoom,1.)
        self.assertEqual(camera.to_dict()['endpoint_scale_policy'],'bounded_residual')
        self.assertGreaterEqual(camera.minimum_head_visible_fraction,.995)

    def test_bounded_residual_does_not_increase_native_budget_or_claim_a_match(self):
        state=fixed_neighbor_face_state(face(),face(.30,.1),maximum_zoom=1.18,
            endpoint_scale_policy='bounded_residual')
        self.assertEqual(state.zoom,1.18)
        self.assertLess(state.zoom*face().face_box_xywh[3],face(.30,.1).face_box_xywh[3])
        with self.assertRaisesRegex(ValueError,'scale policy'):
            fixed_neighbor_face_state(face(),face(),maximum_zoom=2.,endpoint_scale_policy='always_pass')

    def test_face_height_and_eye_span_are_explicit_different_objectives(self):
        source,target=face(),face(.26,.077)
        head=fixed_neighbor_face_state(source,target,maximum_zoom=2.)
        eyes=fixed_neighbor_face_state(source,target,maximum_zoom=2.,scale_basis='eye_span')
        self.assertAlmostEqual(head.zoom,1.3)
        self.assertAlmostEqual(eyes.zoom,1.1)
        self.assertNotEqual(head.zoom,eyes.zoom)

    def test_camera_preserves_both_fixed_neighbor_states_and_has_no_neutral_reset(self):
        sources=(face(),face(.22,.08,x=.42,y=.27))
        targets=(face(.26,.09,x=.43),face(.29,.1,x=.46,y=.31))
        heads=tuple(_close_face_head_envelope((.44-.02*i/30,.3-.03*i/30,.12,.2+.02*i/30)) for i in range(31))
        result=build_shot_context_camera(source_endpoints=sources,neighbor_endpoints=targets,
            head_envelopes=heads,fps=30.,maximum_zoom=2.)
        self.assertEqual(result.track.evaluate(0.),result.entry.keyframe(0.))
        self.assertEqual(result.track.evaluate(1.),result.exit.keyframe(1.))
        self.assertGreater(result.track.evaluate(.5).zoom,1.25)
        self.assertGreaterEqual(result.minimum_head_visible_fraction,.995)
        self.assertFalse(result.to_dict()['production_binding_supported'])
        self.assertFalse(result.to_dict()['head_input_provenance_validated_by_this_geometry_function'])

    def test_unknown_endpoints_and_insufficient_native_pixels_stay_unavailable(self):
        source,target=face(),face(.3,.1)
        for absent in (replace(source,face_landmarks_xy=None),replace(source,face_eye_scale=None),
                replace(source,face_is_unambiguous_dominant=False)):
            with self.assertRaisesRegex(ValueError,'observed unambiguous'):
                fixed_neighbor_face_state(absent,target,maximum_zoom=2.)
        with self.assertRaisesRegex(ValueError,'pixel budget'):
            fixed_neighbor_face_state(source,target,maximum_zoom=1.18)
        with self.assertRaisesRegex(ValueError,'pixel budget'):
            fixed_neighbor_face_state(target,source,maximum_zoom=2.)
        with self.assertRaisesRegex(ValueError,'scale basis'):
            fixed_neighbor_face_state(source,target,maximum_zoom=2.,scale_basis='automatic_quality')

    def test_missing_or_clipped_head_cannot_be_hidden_by_endpoint_success(self):
        args=dict(source_endpoints=(face(),face()),neighbor_endpoints=(face(.3,.1),face(.3,.1)),fps=30.,maximum_zoom=2.)
        good=_close_face_head_envelope(face().face_box_xywh)
        for bad in (None,(.4,0.,.3,.7)):
            with self.assertRaises(ValueError):
                build_shot_context_camera(**args,head_envelopes=(good,good,bad,good,good))
        with self.assertRaisesRegex(ValueError,'2..300'):
            build_shot_context_camera(**args,head_envelopes=(good,)*301)


class ShotContextProposalTests(unittest.TestCase):
    def fixture(self):
        plan=RenderPlan(sources=tuple(Path(f'{i}.mp4') for i in range(3)),
            segments=tuple(RenderSegment(i,float(i),1.,float(i)) for i in range(3)),
            reference_source=0,aspect_ratio='16:9',output=Path('unused.mp4'),fps=30.,
            transitions=(RenderTransition(tier='C'),)*2)
        return dict(plan=plan,segment_index=1,scene_cut_times=((),)*3,experimental=True)

    def run_proposal(self, args, observer=None, **extra):
        capture=SimpleNamespace(close=lambda:None,
            frame_at_native_index=lambda source,*_:np.full((180,320,3),source,np.uint8))
        clock=VerifiedRenderFrameClock(Fraction(30),3000,'a'*64)
        def composed(frame,segment,time,size):
            from crosscut.boundary_subject_alignment import _project_box,_TransformState
            observed=face() if frame.mean()==1 else face(.26,.09)
            if segment.editorial_transform:
                key=segment.editorial_transform.evaluate(time)
                observed=replace(observed,face_box_xywh=_project_box(observed.face_box_xywh,
                    _TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees))[0])
            return observed
        with patch('crosscut.shot_context_camera._CaptureProvider',return_value=capture), \
                patch('crosscut.shot_context_camera._observe_composed_head',side_effect=composed), \
                patch('crosscut.shot_context_camera.probe_render_frame_clock',return_value=clock), \
                patch('crosscut.shot_context_camera.probe_media',return_value=SimpleNamespace(
                    video_streams=(SimpleNamespace(width=3840,height=2160),))), \
                patch('crosscut.shot_context_camera.observe_shot_head',side_effect=observer or (
                    lambda frame:face() if frame.mean()==1 else face(.26,.09))):
            return propose_shot_context_camera(**dict(args,**extra))

    def test_reusable_proposal_changes_one_camera_and_preserves_both_native_clock_neighbors(self):
        args=self.fixture();before=args['plan']
        proposal=self.run_proposal(args)
        self.assertEqual(proposal.plan.output_frame_count,before.output_frame_count)
        for i in (0,2):self.assertIs(proposal.plan.segments[i],before.segments[i])
        self.assertEqual(proposal.native_frame_ids,tuple(range(30,60)))
        self.assertEqual(proposal.plan.segments[1].source_start,before.segments[1].source_start)
        self.assertEqual(proposal.plan.segments[1].source_end,before.segments[1].source_end)
        self.assertTrue(proposal.to_dict()['audio_and_source_phase_unchanged'])
        self.assertFalse(proposal.to_dict()['production_binding_supported'])
        self.assertEqual(len(proposal.head_evidence),30)

    def test_opt_in_clock_and_scene_cut_checks_run_before_decoding(self):
        args=self.fixture()
        with self.assertRaisesRegex(ValueError,'experimental consumer'):
            self.run_proposal(args,experimental=False)
        with self.assertRaisesRegex(ValueError,'native camera cut'):
            self.run_proposal(args,scene_cut_times=((),(1.5,),()))
        for index in (0,2,True):
            with self.assertRaises(ValueError):self.run_proposal(args,segment_index=index)
        with self.assertRaises(ValueError):self.run_proposal(args,maximum_runtime_seconds=float('nan'))

    def test_missing_interior_head_does_not_become_success_from_good_endpoints(self):
        args=self.fixture();calls=0
        def observe(frame):
            nonlocal calls
            calls+=1
            if calls==17:return BoundarySubjectObservation()
            return face() if frame.mean()==1 else face(.26,.09)
        with patch('crosscut.shot_context_camera._optical_flow_box',return_value=None), \
                self.assertRaisesRegex(ValueError,'head containment is unobserved'):
            self.run_proposal(args,observer=observe)

    def test_two_real_head_tracks_are_not_serialized_as_new_face_landmarks(self):
        args=self.fixture();calls=0
        def observe(frame):
            nonlocal calls
            calls+=1
            if calls==17:return BoundarySubjectObservation()
            return face() if frame.mean()==1 else face(.26,.09)
        with patch('crosscut.shot_context_camera._optical_flow_box',return_value=(.41,.26,.18,.3)):
            result=self.run_proposal(args,observer=observe)
        tracked=[item for item in result.head_evidence if item['basis']!='observed_face_head_envelope']
        self.assertEqual(len(tracked),1)
        self.assertFalse(tracked[0]['face_landmarks_observed'])
        self.assertGreaterEqual(len(tracked[0]['reference_frames']),2)

    def test_geometry_success_does_not_hide_a_changed_native_clock(self):
        args=self.fixture();plan=args['plan']
        clocks={i:VerifiedRenderFrameClock(Fraction(30),3000,'a'*64) for i in range(3)}
        before=native_frame_clocks_for_output(plan,clocks,range(29,61))
        after=dict(before);after[30]=replace(after[30],native_frame_index=31)
        with patch('crosscut.shot_context_camera.native_frame_clocks_for_output',side_effect=(before,after)), \
                self.assertRaisesRegex(ValueError,'scheduled native frame clock'):
            self.run_proposal(args)


if __name__=='__main__':unittest.main()
