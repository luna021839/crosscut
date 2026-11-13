from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.reference_crop_supervision import _estimate_sample, ReferenceCropSupervisionConfig
from crosscut.reference_identity_crop import (
    IDENTITY_OBSERVATION_POLICY, recover_verified_identity_sample, validate_identity_observation,
)
from crosscut.reference_dynamic_crop_supervision import build_reference_dynamic_crop_supervision
from tests.test_reference_crop_supervision import _fixture_frame
from tests.test_reference_dynamic_crop_supervision import (
    _write_dynamic_pair, _write_canonical_inputs, _request, _config,
)


class IdentityCropObservationTests(unittest.TestCase):
    def test_already_matching_pixels_get_noop_not_positive_crop(self):
        source=_fixture_frame(4,480,270)
        transform,reason,diagnostics=_estimate_sample(source,source,ReferenceCropSupervisionConfig())
        self.assertIsNone(transform)
        self.assertEqual(reason,'actual_warp_improvement_too_small')
        recovered=recover_verified_identity_sample(source,source,reason,diagnostics)
        self.assertIsNotNone(recovered)
        transform,audit=recovered
        self.assertEqual(transform.renderer_dict(),dict(zoom=1.,center_x=.5,center_y=.5,roll_degrees=0.))
        self.assertFalse(audit['identity_observation']['positive_crop_improvement_gate_relaxed'])
        self.assertGreaterEqual(audit['actual_warp_review']['warped_structure_ncc'],.999)
        validate_identity_observation(audit,transform.renderer_dict(),IDENTITY_OBSERVATION_POLICY)
        for changed_audit,changed_frame,policy in (
            (audit,transform.renderer_dict(),None),
            (audit,{**transform.renderer_dict(),'zoom':1.01},IDENTITY_OBSERVATION_POLICY),
            ({**audit,'actual_warp_review':{**audit['actual_warp_review'],'warped_structure_ncc':.9}},
             transform.renderer_dict(),IDENTITY_OBSERVATION_POLICY),
        ):
            with self.assertRaises(ValueError):
                validate_identity_observation(changed_audit,changed_frame,policy)

    def test_different_pixels_large_affine_and_prior_geometry_failures_are_not_rescued(self):
        source=_fixture_frame(4,480,270)
        _,reason,diagnostics=_estimate_sample(source,source,ReferenceCropSupervisionConfig())
        other=np.roll(source,20,axis=1)
        self.assertIsNone(recover_verified_identity_sample(other,source,reason,diagnostics))
        for failure in ('perspective_distortion_too_high','requires_zoom_out','affine_inlier_ratio_too_low',None):
            self.assertIsNone(recover_verified_identity_sample(source,source,failure,diagnostics))
        bad={**diagnostics,'affine_matrix_source_to_reference':[[1.,0.,2.],[0.,1.,0.]]}
        self.assertIsNone(recover_verified_identity_sample(source,source,reason,bad))
        blank=np.zeros_like(source)
        self.assertIsNone(recover_verified_identity_sample(blank,blank,reason,diagnostics))

    def test_real_dynamic_builder_retains_identity_clock_samples(self):
        with tempfile.TemporaryDirectory(prefix='identity-crop-builder-') as directory:
            root=Path(directory)
            identity=np.asarray(((1.,0.,0.),(0.,1.,0.)))
            reference,source=_write_dynamic_pair(root,[identity]*18)
            attribution,evidence,shot=_write_canonical_inputs(root,reference,source)
            request=_request(attribution,evidence,shot,reference,source)
            legacy=build_reference_dynamic_crop_supervision(
                request,config=replace(_config(),retain_verified_identity_observations=False))
            actual=build_reference_dynamic_crop_supervision(request,config=_config())
            self.assertTrue(actual.eligible)
            self.assertEqual(actual.payload['supervision']['motion_label'],'static')
            self.assertGreater(len(actual.keyframes),len(legacy.keyframes))
            for frame in actual.keyframes:
                self.assertEqual(frame['reference_frame_index'],frame['source_frame_index'])
                self.assertEqual(tuple(frame[k] for k in ('zoom','center_x','center_y','roll_degrees')),(1.,.5,.5,0.))
            self.assertNotEqual(actual.payload['provenance']['build_fingerprint_sha256'],
                                legacy.payload['provenance']['build_fingerprint_sha256'])


if __name__ == '__main__':
    unittest.main()
