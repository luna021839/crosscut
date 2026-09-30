from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch, Mock
import numpy as np

from crosscut.boundary_subject_alignment import (
    BoundarySubjectObservation, BoundarySubjectAlignmentPlan, GLOBAL_ACTION_BASIS,
    ANALYSIS_BASIS, BoundarySubjectAlignmentConfig, build_boundary_subject_alignment_plan,
)
from crosscut.body_pose_2d import BodyPose2DEvidence
from crosscut.group_action_observation import observe_global_head_arms
from crosscut.group_focal_attention import GroupFocalSubject, GLOBAL_FOCAL_BASIS
from crosscut.pose import FormationPersonDescriptor
from crosscut.render import (
    EditorialTransformKeyframe as Key, EditorialTransformTrack as Track,
    RenderPlan, RenderSegment, RenderTransition,
)
from crosscut.post_render_qa import _bound_local_alignment_tracks
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_boundary_subject_alignment import _choice, _frame


def person():
    return FormationPersonDescriptor((.30,.12,.40,.82), (.50,.53), .95, ())


def pose(body):
    points=((.45,.48,.95),(.55,.48,.95),(.40,.38,.95),(.60,.38,.95),
        (.45,.33,.95),(.55,.33,.95),(.45,.75,.95),(.55,.75,.95))
    return BodyPose2DEvidence('observed',body.bbox_xywh,(854,480),points,True)


def observation(source):
    body=person();box=(.41,.26,.18,.20) if source==0 else (.40,.24,.20,.225)
    points=((.45,.31),(.55,.31),(.50,.34),(.47,.38),(.53,.38))
    return BoundarySubjectObservation(face_count=1,face_box_xywh=box,
        face_landmarks_xy=points,face_head_roll_radians=0.,face_eye_scale=box[3]*.42,
        face_is_unambiguous_dominant=True,detected_person_count=1,
        formation_boxes_xywh=(body.bbox_xywh,),
        selected_focal_subject=GroupFocalSubject(box,points,0.,box[3]*.42,body,1,GLOBAL_FOCAL_BASIS),
        action_pose=pose(body))


class GlobalActionObservationTests(unittest.TestCase):
    def descriptor(self, face=True, people=None):
        row=observation(0)
        return SimpleNamespace(face_count=1 if face else 0,
            primary_face_bbox=row.face_box_xywh if face else None,
            primary_face_landmarks_xy=row.face_landmarks_xy if face else None,
            primary_face_head_roll_radians=0. if face else None,
            primary_face_eye_scale=row.face_eye_scale if face else None,
            silhouette_bbox_xywh=None,silhouette_confidence=0.,
            detected_person_count=len(people if people is not None else (person(),)),
            formation_people=people if people is not None else (person(),))

    def test_known_global_face_keeps_its_geometry_and_exact_body_owner(self):
        analyzer=Mock();analyzer.describe.return_value=pose(person())
        with patch('crosscut.group_action_observation.describe_frame',return_value=self.descriptor()), \
                patch('crosscut.group_action_observation.detect_roll_aware_faces') as faces:
            observed=observe_global_head_arms(_frame(0),analyzer)
        faces.assert_not_called()
        self.assertEqual(observed.face_box_xywh,observation(0).face_box_xywh)
        self.assertEqual(observed.selected_focal_subject.basis,GLOBAL_FOCAL_BASIS)
        self.assertEqual(analyzer.describe.call_args.kwargs['bbox_xywh'],person().bbox_xywh)

    def test_ambiguous_or_missing_body_never_borrows_more_convenient_arms(self):
        for people in ((),(person(),person())):
            analyzer=Mock()
            with patch('crosscut.group_action_observation.describe_frame',return_value=self.descriptor(people=people)):
                observed=observe_global_head_arms(_frame(0),analyzer)
            analyzer.describe.assert_not_called()
            self.assertIsNone(observed.selected_focal_subject)
            self.assertIsNone(observed.action_pose)

    def test_ambiguous_fresh_faces_do_not_turn_centrality_into_global_dominance(self):
        analyzer=Mock()
        with patch('crosscut.group_action_observation.describe_frame',return_value=self.descriptor(face=False)), \
                patch('crosscut.group_action_observation.detect_roll_aware_faces',return_value=()), \
                patch('crosscut.group_action_observation._dominant_face',return_value=None):
            observed=observe_global_head_arms(_frame(0),analyzer)
        analyzer.describe.assert_not_called()
        self.assertIsNone(observed.face_box_xywh)
        self.assertFalse(observed.face_is_unambiguous_dominant)


class GlobalActionRetainedCameraTests(unittest.TestCase):
    def fixture(self):
        old=Track((Key(0.,1.),Key(1.50,1.),Key(59/30.,1.02)))
        return dict(visual_timeline=VisualTimelinePlan(tuple(_choice(i,i,start=2.*i,end=2.*(i+1))
            for i in range(2)),1.),sources=(Path('a.mp4'),Path('b.mp4')),
            transitions=(RenderTransition(),),fps=30.,frame_provider=lambda source,_:_frame(source),
            source_resolutions=((3840,2160),)*2,boundary_indices=(0,),
            preserved_editorial_tracks=(None,old),preserved_track_time_shifts=(0.,0.),
            preserved_boundary_frames=4,global_action_framing=True,action_framing_model_path=Path('unused.onnx'))

    def run_alignment(self,args,observer=None):
        with patch('crosscut.group_action_observation.observe_global_head_arms',
                side_effect=observer or (lambda frame,_:observation(0 if frame.mean()<75 else 1))):
            return build_boundary_subject_alignment_plan(**args)

    def test_retained_tail_receives_its_own_complete_dense_validation(self):
        args=self.fixture();result=self.run_alignment(args);decision=result.decision_for(0)
        self.assertTrue(decision.eligible,decision.rejection_reason)
        self.assertEqual(result.analysis_basis,GLOBAL_ACTION_BASIS)
        self.assertEqual(decision.transformed_side,'both')
        self.assertGreater(decision.outgoing_validated_output_frame_count,0)
        self.assertGreater(decision.incoming_validated_output_frame_count,0)
        self.assertEqual(decision.validated_output_frame_count,
            decision.outgoing_validated_output_frame_count+decision.incoming_validated_output_frame_count)
        for n in range(60):
            actual=result.track_for(1).evaluate(n/30.);expected=args['preserved_editorial_tracks'][1].evaluate(n/30.)
            self.assertEqual((actual.zoom,actual.center_x,actual.center_y,actual.roll_degrees),
                (expected.zoom,expected.center_x,expected.center_y,expected.roll_degrees))

    def test_unsupported_retained_endpoint_is_rejected_not_copied(self):
        args=self.fixture();args['preserved_editorial_tracks']=(None,Track((Key(0.,1.02),)))
        result=self.run_alignment(args)
        self.assertFalse(result.decision_for(0).eligible)
        self.assertEqual(result.applied_segment_count,0)

    def test_missing_head_in_retained_tail_revokes_the_entire_new_pair(self):
        args=self.fixture()
        args['frame_provider']=lambda source,seconds:_frame(source,
            120 if source==1 and seconds>=3.9 else 50+50*source)
        result=self.run_alignment(args,lambda frame,_:BoundarySubjectObservation()
            if frame.mean()==120 else observation(0 if frame.mean()<75 else 1))
        self.assertFalse(result.decision_for(0).eligible)
        self.assertEqual(result.applied_segment_count,0)

    def test_global_contract_cannot_accept_local_attention_or_injected_provider(self):
        args=self.fixture()
        for changed in ({'local_focal_attention':True},{'action_framing_model_path':None},
                {'observation_provider':lambda _:observation(0)},{'global_action_framing':1},
                {'config':BoundarySubjectAlignmentConfig(minimum_face_visible_fraction=.994)}):
            with self.subTest(changed=changed),patch('crosscut.boundary_subject_alignment._CaptureProvider') as capture, \
                    self.assertRaises(ValueError):
                build_boundary_subject_alignment_plan(**dict(args,**changed))
            capture.assert_not_called()
        with self.assertRaises(ValueError):
            BoundarySubjectAlignmentPlan((),(),(),14,analysis_basis=GLOBAL_ACTION_BASIS)
        with self.assertRaises(ValueError):
            BoundarySubjectAlignmentPlan((),(),(),14,analysis_basis=ANALYSIS_BASIS,global_action_framing=True)

    def test_independent_binding_requires_both_tracks_counts_and_exact_action_contract(self):
        result=self.run_alignment(self.fixture())
        plan=RenderPlan((Path('a.mp4'),Path('b.mp4')),tuple(RenderSegment(i,2.*i,2.,2.*i,
            editorial_transform=result.track_for(i)) for i in range(2)),0,'16:9',Path('unused.mp4'),
            fps=30.,transitions=(RenderTransition(),))
        raw={'boundary_subject_alignment':result.to_dict(),
            'segments':[track.to_dict() for track in (result.track_for(0),result.track_for(1))]}
        plan_json=json.loads(json.dumps(plan.to_json()));raw=json.loads(json.dumps(raw))
        self.assertEqual(_bound_local_alignment_tracks(raw,plan_json),({0,1},{0}))
        mutations=(('central_face_authority',True),('minimum_joint_confidence',.29),
            ('required_arm_joint_count',5),('hand_margin_face_heights',.39),
            ('minimum_head_arm_visible_fraction',.994),('validation','endpoint_only'))
        mutations=(*mutations,('maximum_action_center_shift',.11))
        for key,value in mutations:
            bad=deepcopy(raw);bad['boundary_subject_alignment']['global_action_framing'][key]=value
            self.assertEqual(_bound_local_alignment_tracks(bad,plan_json),(set(),set()),key)
        bad=deepcopy(raw);bad['boundary_subject_alignment']['decisions'][0]['incoming_validated_output_frame_count']=0
        self.assertEqual(_bound_local_alignment_tracks(bad,plan_json),(set(),set()))
        bad_plan=deepcopy(plan_json);bad_plan['segments'][1]['editorial_transform']=None
        self.assertEqual(_bound_local_alignment_tracks(raw,bad_plan),(set(),set()))


if __name__=='__main__':
    unittest.main()
