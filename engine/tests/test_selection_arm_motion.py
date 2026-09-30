from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.selection_arm_motion import (
    SelectionArmSampler, SelectionArmWindow, selection_arm_cost,
)
from crosscut.timeline import TimelineCandidate, TimelineConfig, select_timeline
from crosscut.timeline import ShotType
from crosscut.visual_score import (_candidate_review_summary, _optimizer_identity_value,
                                  FrameVisualDescriptor, build_visual_timeline)
from crosscut.timeline_review import review_timeline
from crosscut.pose import FormationPersonDescriptor
from crosscut.visual_feature_cache import CanonicalFrameIdentity


def window(source, times, angles, frames=(10, 13)):
    return SelectionArmWindow(source, times, frames,
        tuple((value, value, None, None) for value in angles))


class SelectionArmMotionTests(unittest.TestCase):
    def test_direction_distinguishes_identical_endpoint_poses(self):
        outgoing = window('0', (.85, .95), (-.1, 0.))
        forward = window('1', (1.05, 1.15), (0., .1))
        reverse = window('2', (1.05, 1.15), (0., -.1))
        self.assertGreater(selection_arm_cost(outgoing, reverse).cost,
                           selection_arm_cost(outgoing, forward).cost)

    def test_coherent_predicted_pose_and_speed_have_zero_cost(self):
        left = window('0', (.85, .95), (-.1, 0.))
        right = window('1', (1.05, 1.15), (.1, .2), frames=(50, 57))
        value = selection_arm_cost(left, right)
        self.assertAlmostEqual(value.cost, 0.)
        self.assertEqual(value.observed_bones, (0, 1))

    def test_angle_wrap_and_output_time_not_native_frame_delta(self):
        left = window('0', (.85, .95), (math.pi-.1, math.pi))
        right = window('1', (1.05, 1.15), (-math.pi+.1, -math.pi+.2), frames=(80, 86))
        self.assertAlmostEqual(selection_arm_cost(left, right).cost, 0.)

    def test_missing_disjoint_masks_and_same_source_stay_unknown(self):
        left = window('0', (.85, .95), (-.1, 0.))
        right = window('1', (1.05, 1.15), (.1, .2))
        for a,b in ((None,right), (left,None), (left,replace(right,source_id='0')),
                    (left,replace(right,angles=((None,None,.1,.1),(None,None,.2,.2)))),
                    (left,replace(right,master_times=(1.4,1.5)))):
            self.assertIsNone(selection_arm_cost(a,b))

    def test_window_rejects_clones_bad_clocks_and_nonfinite_angles(self):
        valid = window('0', (.85,.95), (-.1,0.))
        for changes in (dict(native_frames=(3,3)),dict(native_frames=(4,3)),
                        dict(master_times=(.9,.8)),dict(master_times=(True,2.)),
                        dict(angles=((math.nan,0.,None,None),(0.,0.,None,None))),
                        dict(angles=((0.,None,None,None),(0.,None,None,None)))):
            with self.assertRaises(ValueError):
                replace(valid,**changes)

    def test_candidate_cannot_reuse_window_after_clock_or_source_change(self):
        item = TimelineCandidate(0,'0',0.,1.,exit_arm_window=window('0',(.85,.95),(-.1,0.)))
        for changes in (dict(source_id='1'),dict(end_seconds=.9),dict(start_seconds=.9)):
            with self.assertRaises(ValueError):
                replace(item,**changes)

    def test_global_dp_prefers_moving_connection_over_equal_still_pose(self):
        left = TimelineCandidate(0,'0',0.,1.,exit_arm_window=window('0',(.85,.95),(-.1,0.)))
        reverse = TimelineCandidate(1,'1',1.,2.,entry_arm_window=window('1',(1.05,1.15),(0.,-.1)))
        forward = replace(reverse,source_id='2',
            entry_arm_window=window('2',(1.05,1.15),(0.,.1)))
        config = TimelineConfig(minimum_hold_segments=1)
        result = select_timeline(((left,),(reverse,forward)),config=config)
        self.assertEqual(result.choices[-1].candidate.source_id,'2')
        without = tuple(tuple(replace(c,entry_arm_window=None,exit_arm_window=None) for c in cells)
                        for cells in ((left,),(reverse,forward)))
        original = select_timeline(without,config=config)
        self.assertEqual(original.choices[-1].candidate.source_id,'1')
        audit = _candidate_review_summary(candidate_index=0,cadence_profile='test',selected=True,
            candidate=result,review=review_timeline(result,available_source_count=3))['selection_arm_motion']
        self.assertEqual(audit['measured_switch_count'],1)
        self.assertEqual(audit['unmeasured_switch_count'],0)
        self.assertNotIn('angles',json.dumps(audit))
        self.assertFalse(audit['edges'][0]['exact_render_frame_evidence'])
        self.assertNotEqual(_optimizer_identity_value(left),_optimizer_identity_value(without[0][0]))

    def test_observations_shared_across_candidate_windows_not_pair_reinference(self):
        class Sampler:
            path=Path('/synthetic/source.mp4')
            def frame_identity(self,seconds):
                return CanonicalFrameIdentity.from_seconds(seconds,frame_rate=30.)
            def frame_at(self,seconds):
                return np.zeros((96,160,3),np.uint8)
        body=FormationPersonDescriptor((.2,.1,.6,.8),(.5,.5),.9)
        descriptor=SimpleNamespace(primary_face_bbox=(.4,.12,.15,.16),formation_people=(body,))
        with patch('crosscut.body_pose_2d.BodyPose2DAnalyzer') as analyzer, patch(
                'crosscut.closeup_gesture.observed_arm_articulation',
                return_value=((1.,0.),(1.,0.),None,None)):
            sampler=SelectionArmSampler(Path('/not-loaded.onnx'))
            options=dict(source_id='0',master_times=(1.,1.1),source_times=(5.,5.1),
                         descriptors=(descriptor,descriptor),scene_cut_times=())
            one=sampler.window(Sampler(),**options)
            two=sampler.window(Sampler(),**options)
            self.assertEqual(one,two)
            self.assertEqual(sampler.inference_count,2)
            self.assertEqual(sampler.cache_hits,2)
            self.assertEqual(analyzer.return_value.describe.call_count,2)
            self.assertIsNone(sampler.window(Sampler(),**dict(options,scene_cut_times=(5.05,))))
            self.assertEqual(sampler.inference_count,2)

    def test_visual_builder_binds_source_and_master_clocks_before_three_dp_heads(self):
        descriptor=FrameVisualDescriptor(quality_score=.8,composition_score=.8,
            shot_scale=.4,shot_type=ShotType.MEDIUM,face_count=0,
            structure_embedding=(0.,)*96,person_layout_embedding=(.2,.4,.3))
        class Sampler:
            frame_rate=30.
            def __init__(self,path):self.path=path
            def frame_at(self,seconds):return np.zeros((48,80,3),np.uint8)
            def sface_embedding_at(self,*args):return ()
            def close(self):pass
        def observe(_sampler,**kwargs):
            return window(kwargs['source_id'],kwargs['master_times'],(0.,.1))
        with patch('crosscut.visual_score._VideoSampler',Sampler), patch(
                'crosscut.visual_score.describe_frame',return_value=descriptor), patch(
                'crosscut.visual_score._face_geometry_boundary_window',return_value=()), patch(
                'crosscut.visual_score._selected_dissolve_evidence',return_value=({},{})), patch(
                'crosscut.selection_arm_motion.SelectionArmSampler') as model:
            model.return_value.window.side_effect=observe
            plan=build_visual_timeline(sources=(Path('a.mp4'),Path('b.mp4')),
                canonical_spans=((1.,2.),(2.,3.)),audio_offsets=(0.,5.),sync_scores=(.9,.9),
                scene_cut_times=((),()),maximum_results=1,
                selection_arm_model_path=Path('/model.onnx'))
        model.assert_called_once_with(Path('/model.onnx'))
        self.assertEqual(model.return_value.window.call_count,8)
        for call in model.return_value.window.call_args_list:
            options=call.kwargs
            offset=5. if options['source_id']=='1' else 0.
            self.assertTrue(all(abs(source-master-offset)<1e-8 for source,master in zip(
                options['source_times'],options['master_times'])))
        self.assertIn('sampled_directional_focal_arm_selection_v1',plan.candidate_generation_basis)

    def test_optional_selector_and_exact_gesture_share_central_face_body_resolution(self):
        from crosscut.group_focal_attention import CENTRAL_FOCAL_BASIS, GroupFocalSubject
        class Sampler:
            path=Path('/synthetic/central.mp4')
            def frame_identity(self,seconds):
                return CanonicalFrameIdentity.from_seconds(seconds,frame_rate=30.)
            def frame_at(self,seconds):
                return np.zeros((96,160,3),np.uint8)
        body=FormationPersonDescriptor((.2,.1,.6,.8),(.5,.5),.9)
        descriptor=SimpleNamespace(primary_face_bbox=None,formation_people=(body,))
        focal=GroupFocalSubject((.4,.12,.15,.16),None,None,None,body,2,CENTRAL_FOCAL_BASIS)
        with patch('crosscut.body_pose_2d.BodyPose2DAnalyzer') as analyzer, patch(
                'crosscut.group_focal_attention.resolve_group_focal_subject',return_value=focal) as resolve, patch(
                'crosscut.closeup_gesture.observed_arm_articulation',
                return_value=((1.,0.),(1.,0.),None,None)):
            sampler=SelectionArmSampler(Path('/not-loaded.onnx'))
            first=sampler._observe(Sampler(),1.,descriptor)
            again=sampler._observe(Sampler(),1.,descriptor)
            self.assertEqual(first,again)
            resolve.assert_called_once()
            self.assertEqual(analyzer.return_value.describe.call_args.kwargs,
                dict(bbox_xywh=body.bbox_xywh,face_bbox_xywh=focal.bbox_xywh))
            self.assertEqual(sampler.inference_count,1)
            self.assertIsNone(descriptor.primary_face_bbox)


if __name__ == '__main__':
    unittest.main()
