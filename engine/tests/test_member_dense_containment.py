from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from crosscut.direct_member_editorial_motion import DirectMemberFaceSample
from crosscut.direct_member_target_face import FOCAL_FACE_BASIS
from crosscut.focal_bracket_containment import FocalBracketContainmentSupport, measure_bracketed_body_containment
from crosscut.member_dense_containment import (
    MemberDenseContainmentSample, dense_camera_sample, attach_member_dense_containment,
    selected_source_native_cuts, CONTAINMENT_BASIS,
)
from engine.tests import test_direct_member_bilateral_transform as paired_fixtures


BOX = (.3, .1, .4, .8)


def support(source=0, seconds=1., box=BOX):
    return FocalBracketContainmentSupport(source, seconds, (box,), 0,
                                          (seconds-.04, seconds+.04), (box, box))


def strict():
    return DirectMemberFaceSample((.42, .12, .12, .12), 1, 'yunet', BOX, .95,
                                 'face_associated_formation_person_box', FOCAL_FACE_BASIS)


class DenseContainmentAuthorityTests(unittest.TestCase):
    def test_support_is_not_identity_pose_or_an_independent_anchor(self):
        value = MemberDenseContainmentSample(support(), 1., .95)
        self.assertEqual(value.face_count, 0)
        self.assertEqual(value.alignment_geometry_basis, CONTAINMENT_BASIS)
        self.assertNotIsInstance(value, DirectMemberFaceSample)
        self.assertIsNone(measure_bracketed_body_containment(source_index=0, source_seconds=1.,
            observed_query_boxes=(BOX,), anchors=((0,.96,value),(0,1.04,value))))

    def test_strict_sample_wins_and_neither_endpoint_can_use_support(self):
        provider = Mock(return_value=None)
        provider.interior_containment_at = Mock(return_value=MemberDenseContainmentSample(support(), 1., .95))
        self.assertIsNone(dense_camera_sample(provider, 0, 1., interior=False))
        provider.interior_containment_at.assert_not_called()
        result = dense_camera_sample(provider, 0, 1., interior=True)
        self.assertEqual(result.bbox_xywh, BOX)
        provider.interior_containment_at.reset_mock()
        observed = strict()
        provider.return_value = observed
        self.assertIs(dense_camera_sample(provider, 0, 1., interior=True), observed)
        provider.interior_containment_at.assert_not_called()

    def test_wrong_source_clock_or_untyped_support_is_rejected(self):
        for value in (MemberDenseContainmentSample(support(source=1), 1., .95),
                      MemberDenseContainmentSample(support(), 1.1, .95), strict()):
            provider = Mock(return_value=None)
            provider.interior_containment_at = lambda *a: value
            with self.assertRaises(ValueError):
                dense_camera_sample(provider, 0, 1., interior=True)

    def fixture(self, *, competitors=(), cached=True, native=((),)):
        observed = SimpleNamespace(source_index=0, observations=tuple(SimpleNamespace(
            person=SimpleNamespace(bbox_xywh=b, detection_confidence=.95)) for b in (BOX, *competitors)))
        focal = SimpleNamespace(
            sample=Mock(side_effect=lambda s,t: strict() if abs(t-10.)>.001 else None),
            faces=SimpleNamespace(_samplers=(None,), _identity_key=lambda s,sampler,t:(s,t)),
            bodies=SimpleNamespace(_frame_cache={(0,10.):observed} if cached else {}))
        def provider(source, master): return None
        attach_member_dense_containment(provider, focal_provider=focal, source_local_time=lambda s,t:t+9.,
                                        native_scene_cut_times=native, fps=30.)
        return provider, focal

    def test_only_the_already_measured_query_body_is_used(self):
        provider, focal = self.fixture()
        result = dense_camera_sample(provider, 0, 1., interior=True)
        self.assertEqual(result.support.source_seconds, 10.)
        self.assertEqual(result.support.bbox_xywh, BOX)
        self.assertEqual(focal.sample.call_count, 2)
        self.assertIs(dense_camera_sample(provider, 0, 1., interior=True), result)
        self.assertEqual(focal.sample.call_count, 2)
        audit = provider.interior_containment_audit()
        self.assertFalse(audit[0]['focal_identity_or_pose_authority'])
        self.assertFalse(audit[0]['face_detection_claimed'])

    def test_missing_query_detection_ambiguous_rival_or_native_cut_stays_unknown(self):
        for args in (dict(cached=False), dict(competitors=(BOX,)), dict(native=((10.,),))):
            provider, _ = self.fixture(**args)
            self.assertIsNone(dense_camera_sample(provider, 0, 1., interior=True))
            self.assertEqual(provider.interior_containment_audit(), ())
        provider, focal = self.fixture(cached=False)
        self.assertIsNone(dense_camera_sample(provider, 0, 1., interior=True))
        focal.sample.assert_not_called()

    def test_unknown_native_cuts_disable_support_and_suppressed_cuts_still_block(self):
        selection = SimpleNamespace(native_scene_cut_evidence_available=False, source_indices=(0,))
        self.assertIsNone(selected_source_native_cuts(selection, 1))
        selection.native_scene_cut_evidence_available = True
        self.assertIsNone(selected_source_native_cuts(selection, 1))
        selection.span_validity = (SimpleNamespace(sources=(SimpleNamespace(eligible_source_index=0,
            native_scene_cut_source_seconds=(2.,), suppressed_portrait_fancam_scene_cut_source_seconds=(1.,)),)),)
        self.assertEqual(selected_source_native_cuts(selection, 1), ((1.,2.),))
        provider, _ = self.fixture(native=None)
        self.assertFalse(hasattr(provider, 'interior_containment_at'))

    def test_observed_interior_support_can_complete_both_dense_ramps(self):
        fixture = paired_fixtures.BilateralDensePublicationTests()
        fixture.setUp()
        missing = 1.5 + 10/30.
        def provider(source, seconds):
            return None if source == 1 and abs(seconds-missing)<1e-8 else fixture.sample(source, seconds)
        body = paired_fixtures.values()['incoming_subject_box']
        recovered = MemberDenseContainmentSample(support(1, missing, body), missing, .95)
        provider.interior_containment_at = lambda source, seconds: recovered if source == 1 and abs(seconds-missing)<1e-8 else None
        plan = fixture.build(provider)
        self.assertEqual(plan.accepted_boundary_indices, (0,))
        self.assertIsNotNone(plan.track_for(0))
        self.assertIsNotNone(plan.track_for(1))

    def test_missing_cross_source_endpoint_is_never_rescued(self):
        fixture = paired_fixtures.BilateralDensePublicationTests()
        fixture.setUp()
        def provider(source, seconds):
            return None if source == 1 and abs(seconds-1.5)<1e-8 else fixture.sample(source, seconds)
        provider.interior_containment_at = Mock()
        plan = fixture.build(provider)
        self.assertEqual(plan.accepted_boundary_indices, ())
        provider.interior_containment_at.assert_not_called()

    def test_containment_cannot_be_mislabeled_as_the_endpoint_provider(self):
        fixture = paired_fixtures.BilateralDensePublicationTests()
        fixture.setUp()
        body = paired_fixtures.values()['incoming_subject_box']
        def provider(source, seconds):
            if source == 1 and abs(seconds-1.5) < 1e-8:
                return MemberDenseContainmentSample(support(1, seconds, body), seconds, .95)
            return fixture.sample(source, seconds)
        plan = fixture.build(provider)
        self.assertEqual(plan.accepted_boundary_indices, ())
        self.assertIn('containment_only_endpoint_forbidden', plan.required_boundary_failures[0][1])


if __name__ == '__main__':
    unittest.main()
