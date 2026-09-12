from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from crosscut.body_pose_2d import BodyPose2DAnalyzer, BodyPose2DEvidence, _prepare_input, _decode_outputs


BOX = (.25, .1, .5, .8)
IMAGE = np.zeros((200, 200, 3), np.uint8)


def simcc():
    outputs = [np.zeros((1, 17, 384), np.float32), np.zeros((1, 17, 512), np.float32)]
    outputs[0][:, :, 192] = .9
    outputs[1][:, :, 256] = .8
    return outputs


def evidence():
    points = ((.4,.3,.9),(.6,.3,.9),(.3,.4,.9),(.7,.4,.9),
              (.2,.5,.9),(.8,.5,.9),(.4,.6,.9),(.6,.6,.9))
    return BodyPose2DEvidence('observed', BOX, (200, 200), points)


class BodyPose2DTests(unittest.TestCase):
    def test_rgb_normalization_matches_export_not_bgr_example(self):
        image = np.zeros((200, 200, 3), np.uint8)
        image[:, :, 2] = 255
        tensor, _, _ = _prepare_input(image, BOX)
        self.assertEqual(tensor.shape, (1,3,256,192))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[0,:,128,96],
                                   ((255-123.675)/58.395, -116.28/57.12, -103.53/57.375),rtol=1e-6)

    def test_simcc_center_round_trips_aspect_corrected_body_center(self):
        _, center, scale = _prepare_input(IMAGE, BOX)
        points, scores = _decode_outputs(simcc(), center, scale)
        np.testing.assert_allclose(points, np.tile(center,(17,1)))
        np.testing.assert_allclose(scores, .8)

    def test_non_finite_or_wrong_output_contract_is_rejected(self):
        wrong = simcc(); wrong[0][0,0,0] = np.nan
        for outputs in (wrong, [np.zeros((1,17,384))], [np.zeros((1,26,384)), np.zeros((1,26,512))]):
            with self.assertRaises(ValueError):
                _decode_outputs(outputs, np.zeros(2), np.ones(2))

    def test_visible_torso_keeps_eight_joint_basis(self):
        person = evidence().as_formation_person(.9)
        self.assertEqual(len(person.upper_body_keypoints_xyc), 8)
        self.assertEqual(person.bbox_xywh, BOX)

    def test_outside_hips_are_not_clamped_into_observed_torso(self):
        original = evidence()
        points = list(original.upper_body_xyc)
        points[6] = (.4, 1.1, .99)
        changed = replace(original, upper_body_xyc=tuple(points))
        self.assertEqual(changed.upper_body_xyc[6][1], 1.1)
        self.assertIsNone(changed.as_formation_person(.9))

    def test_unseen_hips_and_mismatching_face_cannot_supply_focal_pose(self):
        original = evidence()
        points = list(original.upper_body_xyc); points[6] = (.4,.6,.1)
        self.assertIsNone(replace(original,upper_body_xyc=tuple(points)).as_formation_person(.9))
        self.assertIsNone(replace(original,face_anchor_consistent=False).as_formation_person(.9))

    def test_model_missing_is_explicit_and_does_not_fabricate_evidence(self):
        result = BodyPose2DAnalyzer().describe(IMAGE,bbox_xywh=BOX)
        self.assertEqual(result.reason,'pose_model_unavailable')
        self.assertFalse(result.upper_body_xyc)

    def test_bad_model_digest_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)/'wrong.onnx'
            model.write_bytes(b'not the approved asset')
            analyzer = BodyPose2DAnalyzer(model)
            self.assertEqual(analyzer.describe(IMAGE,bbox_xywh=BOX).reason,'pose_model_digest_mismatch')
            self.assertIsNone(analyzer._net)

    def test_face_association_is_an_explicit_nonidentity_constraint(self):
        analyzer = BodyPose2DAnalyzer()
        analyzer._net = Mock()
        analyzer._net.forward.return_value = simcc()
        result = analyzer.describe(IMAGE,bbox_xywh=BOX,face_bbox_xywh=(.45,.15,.1,.1))
        self.assertEqual(result.reason,'observed')
        self.assertFalse(result.face_anchor_consistent)
        self.assertIsNone(result.as_formation_person(.9))
        self.assertEqual(len(result.upper_body_xyc),8)

    def test_invalid_input_never_runs_model(self):
        analyzer = BodyPose2DAnalyzer()
        analyzer._load = Mock()
        for box in ((True,.1,.5,.8),(-.1,.1,.5,.8),(.1,.1,0,.8),(.5,.1,.6,.8)):
            with self.assertRaises(ValueError):
                analyzer.describe(IMAGE,bbox_xywh=box)
        analyzer._load.assert_not_called()

    def test_fingerprint_binds_model_implementation_and_runtime(self):
        fingerprint = BodyPose2DAnalyzer.fingerprint()
        for key in ('implementation_sha256','model_sha256'):
            self.assertEqual(len(fingerprint[key]),64)
        self.assertEqual(fingerprint['feature_basis'],'eight_nonfacial_screen_xyc')


if __name__ == '__main__':
    unittest.main()
