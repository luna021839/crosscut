import unittest
from unittest.mock import patch

import numpy as np

from crosscut.cut_anchored_containment import track_cut_handle, track_around_cut_anchor


class CutAnchoredContainmentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17)
        patch_=rng.integers(0,256,(70,60,3),dtype=np.uint8)
        self.frames=[]
        self.boxes=[]
        for i in range(9):
            frame=np.zeros((180,320,3),dtype=np.uint8)
            frame[40:110,100+i:160+i]=patch_
            self.frames.append(frame)
            self.boxes.append(((100+i)/320,40/180,60/320,70/180))

    def run_case(self, **changes):
        arguments=dict(frames=self.frames,times=[i/30 for i in range(9)],
                       face_boxes=[(b,) for b in self.boxes],anchor_box=self.boxes[0],
                       anchor_at_end=False,output_fps=30.)
        arguments.update(changes)
        return track_cut_handle(**arguments)

    def test_real_forward_backward_pixel_track(self):
        for backwards in (False,True):
            result=self.run_case(anchor_at_end=backwards,anchor_box=self.boxes[-1 if backwards else 0])
            self.assertTrue(result.complete,result.failure)
            self.assertEqual([s.index for s in result.samples],list(range(9)))

    def test_two_detector_misses_recover_only_with_pixel_flow(self):
        boxes=[(b,) for b in self.boxes]
        boxes[3]=boxes[4]=()
        result=self.run_case(face_boxes=boxes)
        self.assertTrue(result.complete,result.failure)
        self.assertEqual(result.samples[4].consecutive_detector_misses,2)
        self.assertEqual(result.samples[5].consecutive_detector_misses,0)

    def test_third_missing_geometry_is_not_filled(self):
        boxes=[(b,) for b in self.boxes]
        boxes[3]=boxes[4]=boxes[5]=()
        self.assertEqual(self.run_case(face_boxes=boxes).failure,'detector_gap_exceeded')

    def test_background_face_cannot_replace_missing_target(self):
        boxes=[(b,) for b in self.boxes]
        boxes[3]=boxes[4]=boxes[5]=((.8,.1,.1,.2),)
        self.assertEqual(self.run_case(face_boxes=boxes).failure,'detector_gap_exceeded')

    def test_scene_cut_fails_before_any_flow(self):
        with patch('crosscut.cut_anchored_containment._optical_flow_box') as flow:
            result=self.run_case(scene_break_indices=(4,))
        self.assertEqual(result.failure,'native_scene_cut')
        flow.assert_not_called()

    def test_black_occluder_cannot_be_tracked(self):
        frames=list(self.frames)
        frames[4]=np.zeros_like(frames[4])
        self.assertEqual(self.run_case(frames=frames).failure,'pixel_flow_failed')

    def test_anchor_must_be_observed(self):
        boxes=[(b,) for b in self.boxes]
        boxes[0]=()
        self.assertEqual(self.run_case(face_boxes=boxes).failure,'anchor_not_observed')

    def test_duration_and_frame_gaps_are_bounded(self):
        self.assertEqual(self.run_case(times=[i/10 for i in range(9)]).failure,'handle_duration_exceeded')
        self.assertEqual(self.run_case(times=[i/13 for i in range(9)]).failure,'handle_frame_gap')

    def test_ambiguous_faces_are_not_picked_arbitrarily(self):
        boxes=[(b,) for b in self.boxes]
        boxes[4]=(self.boxes[4],self.boxes[4])
        self.assertEqual(self.run_case(face_boxes=boxes).failure,'face_geometry_ambiguous')

    def test_missing_real_frame_is_not_filled(self):
        frames=list(self.frames)
        frames[3]=None
        self.assertEqual(self.run_case(frames=frames).failure,'frame_unavailable')

    def test_middle_cut_anchor_checks_both_directions(self):
        result = track_around_cut_anchor(frames=self.frames,
            times=[i/30 for i in range(9)], face_boxes=[(b,) for b in self.boxes],
            anchor_box=self.boxes[4], anchor_index=4, output_fps=30.)
        self.assertTrue(result.complete, result.failure)
        self.assertEqual([s.index for s in result.samples], list(range(9)))
        self.assertEqual(result.samples[4].basis, 'observed_cut_anchor')

    def test_middle_cut_failure_has_actual_global_index(self):
        frames = list(self.frames)
        frames[7] = None
        result = track_around_cut_anchor(frames=frames,
            times=[i/30 for i in range(9)], face_boxes=[(b,) for b in self.boxes],
            anchor_box=self.boxes[4], anchor_index=4, output_fps=30.)
        self.assertEqual(result.failure, 'frame_unavailable')
        self.assertEqual(result.failed_index, 7)

    def test_middle_anchor_does_not_double_duration_budget(self):
        result = track_around_cut_anchor(frames=self.frames,
            times=[i/10 for i in range(9)], face_boxes=[(b,) for b in self.boxes],
            anchor_box=self.boxes[4], anchor_index=4, output_fps=10.)
        self.assertEqual(result.failure, 'handle_duration_exceeded')

    def test_invalid_numeric_types_are_rejected(self):
        for changes in ({'output_fps': True}, {'output_fps': '30'},
                        {'maximum_handle_seconds': False},
                        {'anchor_box': ('x', .1, .1, .1)},
                        {'times': [False] + [i/30 for i in range(1, 9)]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_case(**changes)


if __name__=='__main__':
    unittest.main()
