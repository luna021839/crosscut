"""Selector and effect authoring share exact target-body attention geometry."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from crosscut.direct_member_visual import (
    _boundary_sample_with_focal_geometry, _focal_sample_in_body_attention_basis,
    FOCAL_POSE_MOTION_BRIDGE_BASIS, DirectMemberCinematicBridge,
    _scored_boundary_search_mismatch, _best_boundary_search_edge,
)
from crosscut.direct_member_transition import (
    _required_bridge_attention_basis, _required_cinematic_bridge_candidate,
    DirectMemberPoseDissolveConfig,
)
from crosscut.focal_body_track import FocalBodyObservation, FocalBodyTrackResult
from crosscut.focal_selector_geometry import (
    FOCAL_SELECTOR_BODY_BASIS, FOCAL_SELECTOR_FACE_BASIS, FOCAL_SELECTOR_POSE_BASIS,
    focal_pose_shape_mismatch,
)
from crosscut.render import RenderSegment
from engine.tests.test_focal_articulation import window, evidence, CONFIG
from engine.tests.test_focal_selector_geometry import sample


def hidden(value):
    focal = value.focal_geometry
    observed = FocalBodyObservation(focal.source_index, focal.source_seconds,
        focal.person, (1., 0.), .8)
    track = FocalBodyTrackResult(focal.source_index, focal.source_seconds,
        observed, 'observed_target_body_track',
        (focal.source_seconds-.2, focal.source_seconds+.2))
    focal = replace(focal, face_bbox_xywh=None, face_scale=None, body_track=track)
    return _boundary_sample_with_focal_geometry(master_seconds=value.master_seconds,
        source_local_seconds=value.source_local_seconds, source_index=focal.source_index,
        descriptor=value.descriptor, provider=SimpleNamespace(focal_geometry_at=lambda *a: focal))


def bridge(pose=.1, focal=True):
    return DirectMemberCinematicBridge(style='pose_dissolve',
        outgoing_is_landscape=False, incoming_is_landscape=False,
        authorized_severe_flags=('attention_scale',), pose_mismatch=pose,
        silhouette_mismatch=None if focal else .2, base_mismatch=.2,
        evidence_basis=FOCAL_POSE_MOTION_BRIDGE_BASIS if focal else 'pose_and_aggregate_silhouette_v1')


def fixture():
    values=list(window())
    values[1]=hidden(values[1])
    cut=values[2].master_seconds
    motion=evidence(values)
    effect=bridge(focal_pose_shape_mismatch(values[1].focal_geometry, values[2].focal_geometry))
    mismatch=SimpleNamespace(cinematic_bridge=effect, transform_repair=None,
        severe_flags=effect.authorized_severe_flags, pose_basis=FOCAL_SELECTOR_POSE_BASIS,
        motion_connection=motion, attention_position_mismatch=0., attention_scale_mismatch=0.)
    diagnostic=SimpleNamespace(boundary_index=0, canonical_master_seconds=cut,
        outgoing_eligible_source_index=0, incoming_eligible_source_index=1,
        outgoing_sample=values[1], incoming_sample=values[2], mismatch=mismatch,
        outgoing_handle_samples=tuple(sample(source=0,seconds=cut+d) for d in (1/29.97,.2)),
        incoming_handle_samples=tuple(sample(source=1,seconds=cut+d) for d in (1/29.97,.2)))
    valid=SimpleNamespace(sources=tuple(SimpleNamespace(eligible_source_index=s,valid=True,
        native_scene_cut_master_seconds=(), samples=()) for s in (0,1)))
    selection=SimpleNamespace(source_indices=(0,1),usage_policy='heterogeneous_role_aware_cinematic_v1',
        span_validity=(valid,valid))
    return dict(boundary_index=0,diagnostic=diagnostic,visual_selection=selection,
        render_segments=(RenderSegment(0,0.,cut,0.),RenderSegment(1,cut,3.,cut)),
        eligible_request_source_indices=(0,1), eligible_source_durations_seconds=(30.,30.),
        frame_counts=(32,90), common_master_start=0.,fps=29.97,
        highlight_intervals=((cut-.1,cut+.1),),config=DirectMemberPoseDissolveConfig())


class MemberBridgeCommonBasisTests(unittest.TestCase):
    def test_mixed_target_pair_uses_common_body_without_mutating_observations(self):
        left,right=hidden(sample()),sample(source=1)
        projected=_focal_sample_in_body_attention_basis(right)
        self.assertEqual(projected.attention.bbox_xywh,right.focal_geometry.person.bbox_xywh)
        self.assertIs(projected.focal_geometry,right.focal_geometry)
        self.assertIs(projected.descriptor,right.descriptor)
        self.assertEqual(right.attention.basis,FOCAL_SELECTOR_FACE_BASIS)
        self.assertEqual(_required_bridge_attention_basis(left,right,bridge()),FOCAL_SELECTOR_BODY_BASIS)
        self.assertIsNone(left.focal_geometry.face_bbox_xywh)

    def test_face_pair_keeps_face_basis(self):
        self.assertEqual(_required_bridge_attention_basis(sample(),sample(source=1),bridge()),
            FOCAL_SELECTOR_FACE_BASIS)

    def test_visible_handle_faces_use_fixed_body_basis_of_endpoint(self):
        self.assertEqual(_required_bridge_attention_basis(sample(),sample(source=1),bridge(),
            required_basis=FOCAL_SELECTOR_BODY_BASIS),FOCAL_SELECTOR_BODY_BASIS)

    def test_missing_target_and_stale_frame_cannot_borrow_raw_dominant_body(self):
        for invalid in (sample(source=1,missing=True),
                replace(sample(source=1),source_local_seconds=1.1),
                replace(sample(source=1),focal_geometry=None)):
            self.assertIsNone(_required_bridge_attention_basis(hidden(sample()),invalid,bridge()))

    def test_face_without_associated_body_cannot_supply_common_body(self):
        right=sample(source=1)
        right=replace(right,focal_geometry=replace(right.focal_geometry,person=None))
        self.assertIsNone(_required_bridge_attention_basis(hidden(sample()),right,bridge()))

    def test_untyped_bridge_does_not_gain_focal_projection_authority(self):
        self.assertIsNone(_required_bridge_attention_basis(hidden(sample()),sample(source=1),
            SimpleNamespace(pose_mismatch=.1,silhouette_mismatch=None)))

    def test_production_required_candidate_accepts_common_body_endpoint_and_face_handles(self):
        candidate,reasons=_required_cinematic_bridge_candidate(**fixture())
        self.assertEqual(reasons,())
        self.assertEqual(candidate.attention_basis,FOCAL_SELECTOR_BODY_BASIS)
        self.assertTrue(candidate.verified_focal_articulation)

    def test_production_candidate_still_rejects_missing_overlap_target(self):
        args=fixture();diagnostic=args['diagnostic']
        missing=sample(source=0,seconds=diagnostic.outgoing_handle_samples[0].master_seconds,missing=True)
        diagnostic.outgoing_handle_samples=(missing,diagnostic.outgoing_handle_samples[1])
        candidate,reasons=_required_cinematic_bridge_candidate(**args)
        self.assertIsNone(candidate)
        self.assertIn('focal_bridge_handle_pose_unverified',reasons)

    def test_selector_replaces_unobserved_effect_with_observed_alternate_offset(self):
        from engine.tests.test_member_bridge_spacing import state, edge
        bad=state(3.,edge(cost=.01))
        bad=replace(bad,incoming_future=(sample(source=0,seconds=3.2,missing=True),
            *bad.incoming_future[1:]))
        good=state(3.0667334,edge(cost=.5),offset=2)
        selected,mismatch=_best_boundary_search_edge((bad,good),outgoing_source_index=0,
            incoming_source_index=1,priority_boundary=True,config=CONFIG)
        self.assertIs(selected,good)
        self.assertIsNotNone(mismatch.cinematic_bridge)
        rejected=_scored_boundary_search_mismatch(bad,outgoing_source_index=0,
            incoming_source_index=1,candidate_count=2,priority_boundary=True,config=CONFIG)
        self.assertIsNone(rejected.cinematic_bridge)
        self.assertIn('focal_bridge_handle_pose_unverified',rejected.severe_flags)

    def test_wrong_source_overlap_cannot_pass_shared_contract(self):
        args=fixture();diagnostic=args['diagnostic']
        diagnostic.outgoing_handle_samples=diagnostic.incoming_handle_samples
        candidate,reasons=_required_cinematic_bridge_candidate(**args)
        self.assertIsNone(candidate)
        self.assertIn('focal_bridge_handle_pose_unverified',reasons)

    def test_existing_cinematic_subset_policies_keep_same_effect_proof(self):
        for policy in ('short_proof_quality_ranked_flexible_cinematic_v1',
                       'short_proof_feedback_adaptive_minimum_diversity_cinematic_v1'):
            args=fixture();args['visual_selection'].usage_policy=policy
            candidate,reasons=_required_cinematic_bridge_candidate(**args)
            self.assertEqual(reasons,())
            self.assertIsNotNone(candidate)
            args['diagnostic'].outgoing_handle_samples=()
            candidate,reasons=_required_cinematic_bridge_candidate(**args)
            self.assertIsNone(candidate)
            self.assertIn('focal_bridge_exact_handle_pose_missing',reasons)

    def test_safe_hold_or_unknown_policy_does_not_authorize_effect(self):
        for policy in ('heterogeneous_role_aware_safe_hold_v1','unknown'):
            args=fixture();args['visual_selection'].usage_policy=policy
            candidate,reasons=_required_cinematic_bridge_candidate(**args)
            self.assertIsNone(candidate)
            self.assertIn('cinematic_bridge_usage_authority_missing',reasons)


if __name__=='__main__':unittest.main()
