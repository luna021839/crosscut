from dataclasses import replace
import hashlib
import unittest

import cv2
import numpy as np

from crosscut.group_joint_part_surface_flow import NativePartSurfaceSupport, measure_part_surface_paths


class PartSurfaceFlowTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(442)
        first = np.zeros((180,240,3),np.uint8)
        first[50:100,80:140] = rng.integers(20,230,(50,60,3),dtype=np.uint8)
        frames = tuple(cv2.warpAffine(first,np.float32([[1,0,step*3],[0,1,0]]),(240,180)) for step in range(3))
        supports = []
        for step,frame in enumerate(frames):
            mask = np.zeros((180,240),bool)
            mask[49:101,79+step*3:141+step*3] = True
            supports.append(NativePartSurfaceSupport(2,109+step,'a'*64,
                hashlib.sha256(memoryview(frame)).hexdigest(),'source2/frozen-fragment','head_surface',
                mask,mask,(),'b'*64,'c'*64))
        return frames,tuple(supports)

    def test_actual_flow_retains_original_and_every_bidirectional_native_step(self):
        frames,rows = self.fixture()
        value = measure_part_surface_paths(native_pixels=frames,supports=rows)
        self.assertEqual(value['measurement_status'],'observed_prediction_supported_surface_paths')
        self.assertGreaterEqual(len(value['surviving_feature_ids']),3)
        for item in value['feature_attempts']:
            self.assertEqual(len(item['steps']),2)
            if item['survived']:
                self.assertAlmostEqual(item['steps'][-1]['forward_native_xy'][0]-item['original_native_xy'][0],6.,delta=.5)
                for step in item['steps']:
                    self.assertTrue(step['from_membership']['subject'])
                    self.assertTrue(step['to_membership']['part'])
                    self.assertTrue(step['forward_status'])
                    self.assertTrue(step['backward_status'])
                    self.assertLessEqual(step['forward_backward_error_pixels'],1.25)
        self.assertFalse(value['cross_camera_motion_certified'])
        self.assertFalse(value['quality_approved'])

    def test_background_texture_cannot_replace_missing_instance_or_semantic_support(self):
        frames,rows = self.fixture()
        empty = np.zeros((180,240),bool)
        for name in ('instance_support','semantic_part_support'):
            wrong = (replace(rows[0],**{name:empty}),*rows[1:])
            value = measure_part_surface_paths(native_pixels=frames,supports=wrong)
            self.assertEqual(value['feature_attempts'],[])
            self.assertTrue(value['measurement_status'].startswith('unknown'))

    def test_middle_mask_loss_is_retained_and_cannot_resurrect_at_final_frame(self):
        frames,rows = self.fixture()
        empty = np.zeros((180,240),bool)
        value = measure_part_surface_paths(native_pixels=frames,
            supports=(rows[0],replace(rows[1],semantic_part_support=empty),rows[2]))
        self.assertEqual(value['surviving_feature_ids'],[])
        self.assertTrue(value['feature_attempts'])
        self.assertTrue(all(not item['survived'] and len(item['steps']) == 2 for item in value['feature_attempts']))
        self.assertTrue(all(not item['steps'][0]['to_membership']['part'] for item in value['feature_attempts']))

    def test_competing_person_touching_the_same_part_remains_unknown_not_a_zero_cost(self):
        frames,rows = self.fixture()
        overlap = replace(rows[1],competing_subject_support=(('other',rows[1].instance_support),))
        value = measure_part_surface_paths(native_pixels=frames,supports=(rows[0],overlap,rows[2]))
        self.assertEqual(value['surviving_feature_ids'],[])
        self.assertTrue(all('other' in item['steps'][0]['to_membership']['competing_subjects']
            for item in value['feature_attempts']))

    def test_pixel_clock_subject_part_and_intervening_frame_bindings_cannot_be_forged(self):
        frames,rows = self.fixture()
        for wrong in (replace(rows[1],native_pixel_sha256='d'*64),
                replace(rows[1],source_clock_sha256='d'*64),
                replace(rows[1],subject_fragment_reference='different'),
                replace(rows[1],part='left_arm_surface'),replace(rows[1],native_frame_index=114)):
            with self.assertRaises(ValueError):
                measure_part_surface_paths(native_pixels=frames,supports=(rows[0],wrong,rows[2]))
        with self.assertRaisesRegex(ValueError,'intervening'):
            measure_part_surface_paths(native_pixels=(frames[0],frames[2]),supports=(rows[0],rows[2]))

    def test_flat_pixels_or_unavailable_foreground_keep_measurement_unknown(self):
        frames,rows = self.fixture()
        flat = tuple(np.zeros_like(frame) for frame in frames)
        rows = tuple(replace(row,native_pixel_sha256=hashlib.sha256(memoryview(frame)).hexdigest())
            for row,frame in zip(rows,flat))
        value = measure_part_surface_paths(native_pixels=flat,supports=rows)
        self.assertEqual(value['surviving_feature_ids'],[])
        self.assertTrue(value['measurement_status'].startswith('unknown'))

    def test_masks_are_snapshotted_and_foreign_competing_reference_is_rejected(self):
        _,rows = self.fixture()
        original = rows[0].instance_support.copy()
        row = replace(rows[0],instance_support=original)
        before = row.witness
        original[:] = False
        self.assertEqual(row.witness,before)
        self.assertFalse(row.instance_support.flags.writeable)
        for wrong in ((row.subject_fragment_reference,row.instance_support),('other',np.zeros((2,2),bool))):
            with self.assertRaises(ValueError):
                replace(row,competing_subject_support=(wrong,))
        with self.assertRaises(ValueError):
            replace(row,instance_support=row.instance_support.astype(np.uint8))
