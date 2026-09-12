from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.joint_boundary_provider import DecodedBoundaryOptionProvider
from crosscut.joint_timeline import BoundaryTransformState, BoundaryOffsetDisposition
from crosscut.visual_boundary_repair import BoundaryDescriptorCache
from engine.tests.test_joint_boundary_provider import _candidate, _descriptor, _provider, _key


def state(name, crop=None, basis=None):
    return BoundaryTransformState(transform_identity=name, crop_xywh=crop, track_id='track:one',
        subject_basis=basis or ('dominant_face' if crop else 'full_frame_identity'))


class BoundaryLocalProviderTests(unittest.TestCase):
    def assert_path_parity(self, provider, *, start=10., end=13., entry=None, exit=None):
        candidate = replace(_candidate(1, '1'), start_seconds=start, end_seconds=end)
        arguments = dict(candidate=candidate, entry_transform=entry or state('entry'),
            exit_transform=exit or state('exit'), start_master_seconds=start,
            end_master_seconds=end, duration_frames=(end-start)*30.,
            required_master_sample_times=(start+.5/30, start+3.5/30,
                                          end-3.5/30, end-.5/30))
        proof = provider._transform_path_evidence(**arguments, context_identity='scoped-regression')
        scalar = provider._transform_path_scalar_disposition(**arguments,
            descriptor_resolver=provider._sample_descriptor,
            face_identity_resolver=provider._sample_face_identity)
        self.assertEqual(scalar, (proof.evidence_complete, proof.crop_compatible,
                                 proof.track_compatible, proof.failure_codes, 0.))
        return proof

    def test_noop_face_basis_needs_no_recognition_or_face_geometry(self):
        face = unittest.mock.Mock(side_effect=AssertionError('identity raster cannot need recognition'))
        with _provider(lambda *_: _descriptor(.5), face_identity_provider=face) as provider:
            proof = self.assert_path_parity(provider,
                entry=state('noop-legacy', basis='dominant_face'))
        self.assertTrue(proof.full_frame_centers_included)
        self.assertTrue(proof.track_compatible)
        face.assert_not_called()

    def test_identity_middle_allows_native_member_change(self):
        def descriptor(_s, t):
            return _descriptor(.5, face_bbox=(.4,.3,.1,.15) if t < 10.6 or t > 12.4 else None)
        with _provider(descriptor, total_segment_count=3, scene_cut_times=((),(11.5,)),
                face_identity_provider=lambda _s,t,_d: (1.,0.) if t < 11.5 else (0.,1.)) as provider:
            proof = self.assert_path_parity(provider, entry=state('in',(.1,.1,.8,.8)),
                                            exit=state('out',(.1,.1,.8,.8)))
        self.assertTrue(proof.evidence_complete)
        self.assertTrue(proof.crop_compatible)
        self.assertTrue(proof.track_compatible)

    def test_native_cut_on_active_keyframe_is_not_missed(self):
        with _provider(lambda *_: _descriptor(.5,face_bbox=(.4,.3,.1,.15)),
                total_segment_count=3, scene_cut_times=((),(10.+3.5/30,)),
                face_identity_provider=lambda *_: (1.,0.)) as provider:
            proof = self.assert_path_parity(provider, entry=state('in',(.1,.1,.8,.8)))
        self.assertIn('segment_source_scene_cut_breaks_track',proof.failure_codes)

    def test_raw_spatial_failure_gets_actual_transformed_remeasurement(self):
        raw = (_descriptor(.25,shot_scale=.22,face_bbox=(.18,.30,.16,.22)),
               _descriptor(.65,shot_scale=.72,face_bbox=(.60,.22,.30,.40)))
        transformed = unittest.mock.Mock(return_value=_descriptor(.5,shot_scale=.72,
                                             face_bbox=(.35,.28,.30,.40)))
        with _provider(lambda s,_: raw[s], transformed_descriptor_provider=transformed,
                       face_identity_provider=lambda *_: (1.,0.)) as provider:
            audit = provider(_key()).audit_for(0)
        self.assertEqual(audit.disposition, BoundaryOffsetDisposition.AUTHORIZED)
        self.assertGreater(transformed.call_count,0)
        self.assertIsNotNone(audit.authorized_option.outgoing_transform.crop_xywh)

    def test_still_bad_transformed_layout_is_rejected(self):
        raw = (_descriptor(.25,shot_scale=.22,face_bbox=(.18,.30,.16,.22)),
               _descriptor(.65,shot_scale=.72,face_bbox=(.60,.22,.30,.40)))
        with _provider(lambda s,_: raw[s], transformed_descriptor_provider=lambda s,*_:raw[s],
                       face_identity_provider=lambda *_:(1.,0.)) as provider:
            audit = provider(_key()).audit_for(0)
        self.assertEqual(audit.disposition, BoundaryOffsetDisposition.HARD_INCOMPATIBLE)
        self.assertTrue(any(code.startswith('post_transform_') for code in audit.failure_codes))

    def test_real_pixel_containment_never_forges_face_cosines(self):
        self.pixel_case()

    def test_real_pixel_occlusion_still_fails_both_routes(self):
        self.pixel_case(occlusion=True)

    def pixel_case(self, occlusion=False):
        texture=np.random.default_rng(24).integers(0,256,(70,60,3),dtype=np.uint8)
        def pixels(_source, t):
            shift=int(round((t-10.)*30.))
            frame=np.zeros((180,320,3),dtype=np.uint8)
            box=((100+shift)/320,40/180,60/320,70/180)
            if not (occlusion and shift == 8):
                frame[40:110,100+shift:160+shift]=texture
            return frame,(box,),'yunet'
        def descriptor(source,t):
            return _descriptor(.5,face_bbox=pixels(source,t)[1][0])
        with DecodedBoundaryOptionProvider(sources=('unused-a.mp4','unused-b.mp4'),
                audio_offsets=(0.,0.),scene_cut_times=((),()),source_timewarps=None,descriptor_cache=BoundaryDescriptorCache(),
                output_fps=30.,total_segment_count=3,descriptor_provider=descriptor,
                face_identity_provider=lambda *_: (_ for _ in ()).throw(AssertionError('not recognition')),
                containment_frame_provider=lambda s,t:pixels(s,t)[0]) as provider:
            with patch.object(provider,'_cut_containment_pixels',side_effect=pixels):
                proof=self.assert_path_parity(provider,end=11.,entry=state('in',(.1,.1,.8,.8)))
            if occlusion:
                self.assertIn('segment_cut_containment_pixel_flow_failed',proof.failure_codes)
            else:
                self.assertTrue(proof.evidence_complete,proof.failure_codes)
                self.assertTrue(proof.track_compatible,proof.failure_codes)
                samples=provider._segment_materialization_by_proof[proof.proof_identity].samples
                supported=[s for s in samples if s.containment_basis]
                self.assertGreater(len(supported),10)
                self.assertTrue(all(s.face_similarity_to_anchor is None for s in supported))
                self.assertTrue(all(s.face_identity_embedding == () for s in supported))

    def test_native_pixel_cache_is_bounded_and_cleared_on_close(self):
        frame=np.zeros((540,960,3),dtype=np.uint8)
        provider=DecodedBoundaryOptionProvider(sources=('unused-a','unused-b'),
            audio_offsets=(0.,0.),scene_cut_times=((),()),source_timewarps=None,descriptor_cache=BoundaryDescriptorCache(),
            output_fps=30.,total_segment_count=2,descriptor_provider=lambda *_:_descriptor(.5),
            containment_frame_provider=lambda *_:frame)
        with patch('crosscut.joint_boundary_provider._detect_faces',return_value=((),'yunet')):
            for i in range(80):
                provider._cut_containment_pixels(0,i/30)
        self.assertEqual(len(provider._containment_pixels),64)
        self.assertLessEqual(provider._containment_pixel_bytes,128*1024**2)
        provider.close()
        self.assertEqual(provider._containment_pixel_bytes,0)
        self.assertFalse(provider._containment_pixels)


if __name__ == '__main__':
    unittest.main()
