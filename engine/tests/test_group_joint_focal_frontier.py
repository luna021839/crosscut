from dataclasses import replace
from types import SimpleNamespace
import hashlib
import unittest
from unittest.mock import Mock, patch
import numpy as np

from crosscut.group_joint_focal_frontier import (OwnedFocalAppearance,
    NativeFaceAppearance, observe_native_face_appearances,
    InheritedFocalReference, NativeFocalLinkGraph, resolve_inherited_focal_reference, resolve_diagnostic_focal_reference,
    match_focal_trajectory,
    locally_associated_face_roi_target,
    match_focal_appearance_anchors, observe_owned_focal_appearances, _dominant_owned_face, _same_source_links)
from crosscut.group_joint_person_ownership import bind_person_ownership, ownership_policy_fingerprint
from engine.tests import test_group_joint_person_ownership as ownership_fixtures
from engine.tests import test_group_joint_native_tracks as track_fixtures


def appearance(person=0, face=7, axis=0):
    return OwnedFocalAppearance(person, face, tuple(1. if i == axis else 0. for i in range(128)))


class FocalFrontierTests(unittest.TestCase):
    def row(self):
        fixture = ownership_fixtures.PersonOwnershipTests()
        head = replace(fixture.head(), candidate_body_observation_ids=(0,))
        people, _ = bind_person_ownership((fixture.closeup(),), (head,), (1000, 800))
        _, rows = track_fixtures.NativeGroupTrackTests().fixture(((),))
        return replace(rows[0], frame_size_wh=(1000, 800), people=people, heads=(head,),
            person_ownership_policy_sha256=ownership_policy_fingerprint())

    def test_distinct_three_anchor_appearance_is_shortlist_not_identity(self):
        refs = (appearance(),) * 3
        values, reason, similarities, margins = match_focal_appearance_anchors(refs, ((appearance(4),),) * 3)
        self.assertEqual(tuple(row.person_observation_id for row in values), (4, 4, 4))
        self.assertEqual(reason, 'appearance_shortlist_only')
        self.assertEqual(similarities, (1., 1., 1.))
        self.assertEqual(margins, (None, None, None), 'single candidate has no fabricated confidence margin')

    def test_local_face_recovery_keeps_all_competitors_and_existing_geometry_limits(self):
        head = self.row().heads[0]
        moved = replace(head, observation_id=11, box_xywh=(head.box_xywh[0]+.003, *head.box_xywh[1:]))
        self.assertEqual(locally_associated_face_roi_target((head,), (moved,), head), moved)
        duplicate = replace(moved, observation_id=12)
        self.assertIsNone(locally_associated_face_roi_target((head,), (moved, duplicate), head))
        self.assertIsNone(locally_associated_face_roi_target((head, replace(head, observation_id=9)), (moved,), head))
        far = replace(moved, box_xywh=(.8, *moved.box_xywh[1:]))
        self.assertIsNone(locally_associated_face_roi_target((head,), (far,), head))
        changed_scale = replace(moved, box_xywh=(*moved.box_xywh[:2], moved.box_xywh[2]*1.3, moved.box_xywh[3]))
        self.assertIsNone(locally_associated_face_roi_target((head,), (changed_scale,), head))

    def test_missing_or_competing_same_face_remains_unavailable(self):
        refs = (appearance(),) * 3
        for values in (((appearance(),), (), (appearance(),)),
                ((appearance(), appearance(1)),) * 3, ((appearance(axis=1),),) * 3):
            self.assertIsNone(match_focal_appearance_anchors(refs, values)[0])
        with self.assertRaises(ValueError):
            match_focal_appearance_anchors(refs[:2], ((appearance(),),) * 2)

    def test_unrelated_clipped_edge_face_does_not_erase_an_in_frame_reference(self):
        head=replace(self.row().heads[0],box_xywh=(.37,.27,.15,.29))
        moved=replace(head,observation_id=11,box_xywh=(.371,.273,.14,.282))
        clipped=replace(head,observation_id=12,box_xywh=(.925,.31,.08,.26))
        self.assertEqual(locally_associated_face_roi_target((head,clipped),(moved,clipped),head),moved)

    def test_nearby_clipped_competitor_and_selected_outside_roi_still_fail(self):
        head=replace(self.row().heads[0],box_xywh=(.92,.2,.08,.2))
        clipped=replace(head,observation_id=12,box_xywh=(.921,.2,.08,.2))
        self.assertIsNone(locally_associated_face_roi_target((head,),(head,clipped),head))
        self.assertIsNone(locally_associated_face_roi_target((head,),(clipped,),head))

    def test_owned_cropped_hips_can_supply_actual_face_embedding_without_claiming_torso(self):
        row = self.row()
        session = SimpleNamespace(observe_native=Mock(return_value=appearance().embedding))
        values = observe_owned_focal_appearances(session, row)
        self.assertEqual(values, (appearance(),))
        self.assertEqual(row.people[0].body_ownership_state, 'unknown')
        self.assertEqual(dict(row.people[0].part_states)['head_extent'], 'unknown')
        self.assertEqual(session.observe_native.call_args.kwargs['model'], 'sface_owned_face')
        self.assertEqual(session.observe_native.call_args.args, (row.request_source_index, row.native_frame_index))

    def test_legacy_policy_and_forged_owned_parts_cannot_supply_embeddings(self):
        row = self.row()
        session = SimpleNamespace(observe_native=Mock(return_value=appearance().embedding))
        for wrong in (replace(row, person_ownership_policy_sha256='a'*64),
                replace(row, people=(replace(row.people[0], body_ownership_state='observed'),)),
                replace(row, heads=(replace(row.heads[0], candidate_body_observation_ids=(1,)),))):
            with self.assertRaises(ValueError):
                observe_owned_focal_appearances(session, wrong)
        session.observe_native.assert_not_called()

    def test_reference_is_global_face_dominance_not_convenient_body_or_order(self):
        row = self.row()
        self.assertEqual(_dominant_owned_face(row, (appearance(),)), appearance())
        head = replace(row.heads[0], observation_id=8, box_xywh=(.65, .27, .065, .06))
        ambiguous = replace(row, heads=(*row.heads, head))
        self.assertIsNone(_dominant_owned_face(ambiguous, (appearance(),)))
        self.assertIsNone(_dominant_owned_face(row, ()))

    def test_face_roi_sensing_preserves_all_ambiguous_body_hypotheses_without_selecting_one(self):
        row = self.row()
        other = replace(row.people[0], observation_id=1)
        head = replace(row.heads[0], candidate_body_observation_ids=(0, 1))
        people, _ = bind_person_ownership((row.people[0], other), (head,), row.frame_size_wh)
        row = replace(row, people=people, heads=(head,))
        self.assertEqual(tuple(p.upper_body_ownership_state for p in row.people), ('ambiguous', 'ambiguous'))
        session = SimpleNamespace(observe_native=Mock(return_value=appearance().embedding))
        self.assertEqual(observe_owned_focal_appearances(session, row), ())
        values = observe_native_face_appearances(session, row)
        self.assertEqual(values, (NativeFaceAppearance(7, appearance().embedding, (0, 1)),))
        self.assertIsNone(values[0].person_observation_id)
        self.assertEqual(values[0].candidate_body_observation_ids, (0, 1))

    def test_landmark_pixel_thresholds_use_detector_raster_not_arbitrary_native_resolution(self):
        row = self.row()
        rows = tuple(replace(row, native_frame_index=n, frame_size_wh=(1920, 1080)) for n in (0, 3, 6))
        face = NativeFaceAppearance(7, appearance().embedding, (0, 1))
        session = SimpleNamespace(domain=SimpleNamespace(eligible_index=lambda _: 0,
            render_frame_clocks=(SimpleNamespace(frame_rate=30.),), cuts_for_request=lambda _: ()),
            native_pixels=lambda *_: np.zeros((1080, 1920, 3), np.uint8))
        with patch('crosscut.group_joint_focal_frontier.focal_landmarks_track', return_value=True) as flow:
            self.assertTrue(_same_source_links(session, rows, (face,) * 3))
        self.assertEqual(flow.call_count, 2)
        self.assertEqual(flow.call_args.args[0].shape, (540, 960, 3))
        self.assertEqual(flow.call_args.args[2], row.heads[0].landmarks_xy)
        session.domain.cuts_for_request = lambda _: (.05,)
        with patch('crosscut.group_joint_focal_frontier.focal_landmarks_track') as flow:
            self.assertFalse(_same_source_links(session, rows, (face,) * 3))
            flow.assert_not_called()

    def test_reference_survives_background_face_growth_without_switching_or_lowering_area_margin(self):
        row = self.row()
        other = replace(row.heads[0], observation_id=8, box_xywh=(.65, .27, .061, .06))
        middle = replace(row, heads=(*row.heads, other))
        self.assertIsNone(_dominant_owned_face(middle, (appearance(),)))
        values = ((appearance(),), (appearance(), appearance(1, 8, axis=1)), (appearance(),))
        with patch('crosscut.group_joint_focal_frontier._same_source_links', return_value=True):
            self.assertEqual(resolve_diagnostic_focal_reference(None, (row, middle, row), values), (appearance(),)*3)
            self.assertIsNone(resolve_diagnostic_focal_reference(None, (middle, row, row), values),
                'no independent initial motive/seed may be fabricated')
        with patch('crosscut.group_joint_focal_frontier._same_source_links', return_value=False):
            self.assertIsNone(resolve_diagnostic_focal_reference(None, (row, middle, row), values))

    def test_seed_is_not_replaced_by_another_person_with_better_mean_appearance(self):
        row = self.row()
        other = replace(row.heads[0], observation_id=8, box_xywh=(.65, .27, .03, .03))
        row = replace(row, heads=(*row.heads, other))
        seed = appearance()
        later = OwnedFocalAppearance(0, 7, tuple(
            .5 if i == 0 else float(np.sqrt(.75)) if i == 2 else 0. for i in range(128)))
        wrong = OwnedFocalAppearance(1, 8, tuple(
            .8 if i == 0 else .6 if i == 1 else 0. for i in range(128)))
        values = ((seed, wrong), (later, wrong), (later, wrong))
        with patch('crosscut.group_joint_focal_frontier._same_source_links',
                side_effect=lambda _session, _rows, path: len({a.face_observation_id for a in path}) == 1):
            self.assertEqual(resolve_diagnostic_focal_reference(None, (row,)*3, values),
                (seed, later, later), 'a different person must not replace the declared start seed')

    def inherited(self, row, face=7):
        return InheritedFocalReference('parent:retained-focal-subject', 'parent:existing-head-arm-authority',
            row.request_source_index, row.native_frame_index, row.sha256, face)

    def test_inherited_subject_need_not_be_the_largest_face(self):
        row = self.row()
        large = replace(row.heads[0], observation_id=8, box_xywh=(.65, .27, .2, .3))
        row = replace(row, heads=(*row.heads, large))
        values = ((appearance(), appearance(1, 8, axis=1)),)*3
        self.assertEqual(_dominant_owned_face(row, values[0]).face_observation_id, 8)
        with patch('crosscut.group_joint_focal_frontier._same_source_links',
                side_effect=lambda _session, _rows, path: len({a.face_observation_id for a in path}) == 1):
            result = resolve_inherited_focal_reference(None, (row,)*3, values,
                inherited_reference=self.inherited(row))
        self.assertEqual(result, (appearance(),)*3)

    def test_inherited_missing_or_stale_seed_never_falls_back_to_a_convenient_face(self):
        row = self.row()
        reference = self.inherited(row)
        cases = (reference, replace(reference, request_source_index=row.request_source_index+1),
            replace(reference, native_frame_index=row.native_frame_index+1),
            replace(reference, observation_sha256='a'*64), replace(reference, face_observation_id=8))
        with patch('crosscut.group_joint_focal_frontier._same_source_links') as flow:
            self.assertIsNone(resolve_inherited_focal_reference(None, (row,)*3,
                ((), (appearance(),), (appearance(),)), inherited_reference=cases[0]))
            for wrong in cases[1:]:
                self.assertIsNone(resolve_inherited_focal_reference(None, (row,)*3,
                    ((appearance(),),)*3, inherited_reference=wrong))
            flow.assert_not_called()
        with self.assertRaises(ValueError):
            resolve_inherited_focal_reference(None, (row,)*3, ((appearance(),),)*3,
                inherited_reference=None)
        for name, value in (('request_source_index', True), ('native_frame_index', -1),
                ('face_observation_id', True), ('parent_authority_id', ''), ('observation_sha256', 'bad')):
            with self.assertRaises(ValueError):
                replace(reference, **{name: value})

    def test_native_trajectory_prevents_per_frame_appearance_switching_people(self):
        good = appearance(1, 8)
        alternative = appearance(2, 9)
        alternatives = ((alternative, good), (good,), (good,))
        path, info = match_focal_trajectory((appearance(),)*3, alternatives,
            path_available=lambda rows: len({row.face_observation_id for row in rows}) == 1)
        self.assertEqual(path, (good,)*3)
        self.assertEqual(info['hypotheses'][0]['face_observation_ids'], [8, 8, 8])
        self.assertFalse(info['identity_certified'])
        path, info = match_focal_trajectory((appearance(),)*3, alternatives, path_available=lambda _: True)
        self.assertEqual(len(info['hypotheses']), 2, 'retain all actual competing paths, no silent pruning')
        self.assertEqual(info['competing_trajectory_margin'], 0.)

    def test_hypothesis_overflow_remains_unknown_without_truncating_to_convenient_face(self):
        available = Mock(return_value=True)
        path, info = match_focal_trajectory((appearance(),)*3,
            ((appearance(), appearance(1)),)*3, path_available=available, maximum_hypotheses=7)
        self.assertIsNone(path)
        self.assertEqual(info['hypothesis_count'], 8)
        available.assert_not_called()

    def graph_fixture(self, size=(1920, 1080), cuts=()):
        session, rows = track_fixtures.NativeGroupTrackTests().fixture(((),)*3, cuts=cuts)
        base = self.row().heads[0]
        width, height = size
        pixels = {n: np.full((height, width, 3), n, np.uint8) for n in range(3)}
        heads = tuple(replace(base, observation_id=n) for n in range(4))
        rows = tuple(replace(row, frame_size_wh=size, heads=heads,
            native_pixel_sha256=hashlib.sha256(memoryview(pixels[row.native_frame_index])).hexdigest()) for row in rows)
        session.native_pixels = Mock(side_effect=lambda source, native: pixels[native])
        session._assert_source_identity = Mock(return_value=rows[0].source_file_identity)
        session._assert_model_assets = Mock()
        session.models = {name:(None, None, digest, None) for name, digest in rows[0].model_content_sha256}
        session.model_adapters = dict(rows[0].model_adapter_fingerprints)
        values = tuple(NativeFaceAppearance(n, appearance(axis=n).embedding, ()) for n in range(4))
        return session, rows, (values,)*3

    def test_four_by_four_by_four_paths_sense_only_32_pairs_at_1080p_and_4k(self):
        for size in ((1920, 1080), (3840, 2160)):
            with self.subTest(size=size):
                session, rows, values = self.graph_fixture(size)
                graph = NativeFocalLinkGraph(session)
                with patch('crosscut.group_joint_focal_frontier.focal_landmarks_track', return_value=True) as flow:
                    for a, b, va, vb in zip(rows, rows[1:], values, values[1:]):
                        graph.add_pair(a, b, va, vb)
                    self.assertEqual(flow.call_count, 32)
                    self.assertEqual(session.native_pixels.call_count, 4)
                    path, info = match_focal_trajectory((values[0][0],)*3, values,
                        path_available=lambda p: graph.path_available(rows, p))
                    self.assertEqual(len(info['hypotheses']), 64)
                    self.assertEqual(flow.call_count, 32, 'ranking must not repeat optical-flow sensing')
                    self.assertEqual(session.native_pixels.call_count, 4, 'ranking must not read native pixels')
                report = graph.to_dict()
                self.assertEqual(report['unique_pair_observations'], 32)
                self.assertEqual(report['cached_pair_lookups'], 128)
                self.assertEqual(report['peak_raw_pair_raster_bytes'], 2*size[0]*size[1]*3)
                self.assertEqual(report['peak_analysis_pair_raster_bytes'], 2*960*540*3)
                self.assertEqual(path, (values[0][0],)*3)
                self.assertFalse(report['part_feature_ownership_verified'])

    def test_failed_pair_is_cached_and_cut_or_repeated_native_is_not_linked(self):
        session, rows, values = self.graph_fixture((100, 50))
        graph = NativeFocalLinkGraph(session)
        with patch('crosscut.group_joint_focal_frontier.focal_landmarks_track', return_value=False) as flow, \
                patch('crosscut.group_joint_focal_frontier._optical_flow_box', return_value=None):
            graph.add_pair(rows[0], rows[1], values[0], values[1])
            for _ in range(3):
                self.assertFalse(graph.path_available(rows[:2], (values[0][0], values[1][0])))
            self.assertEqual(flow.call_count, 16)
        for cut, right in (((), rows[0]), ((1/30.,), rows[1])):
            session, own_rows, own_values = self.graph_fixture((100, 50), cuts=cut)
            right = own_rows[0] if not cut else own_rows[1]
            graph = NativeFocalLinkGraph(session)
            graph.add_pair(own_rows[0], right, own_values[0], own_values[1])
            self.assertFalse(graph.path_available((own_rows[0], right), (own_values[0][0], own_values[1][0])))
            session.native_pixels.assert_not_called()

    def test_pair_graph_rejects_changed_pixels_and_live_binding_even_with_recomputed_row_hash(self):
        session, rows, values = self.graph_fixture((100, 50))
        graph = NativeFocalLinkGraph(session)
        with self.assertRaisesRegex(ValueError, 'pixels differ'):
            graph.add_pair(replace(rows[0], native_pixel_sha256='a'*64), rows[1], values[0], values[1])
        graph = NativeFocalLinkGraph(session)
        with patch('crosscut.group_joint_focal_frontier.focal_landmarks_track', return_value=True):
            graph.add_pair(rows[0], rows[1], values[0], values[1])
        session._assert_source_identity.return_value = (2, 3, 4, 5, 6)
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            graph.path_available(rows[:2], (values[0][0], values[1][0]))
        session._assert_source_identity.return_value = rows[0].source_file_identity
        session.models['synthetic_test_only'] = (None, None, 'f'*64, None)
        with self.assertRaisesRegex(ValueError, 'model binding changed'):
            graph.path_available(rows[:2], (values[0][0], values[1][0]))

    def test_pair_budget_and_analysis_bytes_overflow_do_not_truncate_to_success(self):
        session, rows, values = self.graph_fixture((100, 50))
        for graph in (NativeFocalLinkGraph(session, maximum_pair_observations=15),
                NativeFocalLinkGraph(session, maximum_analysis_pair_bytes=10)):
            with self.assertRaisesRegex(ValueError, 'overflow|byte budget'):
                graph.add_pair(rows[0], rows[1], values[0], values[1])
            self.assertEqual(graph.pairs, {})
