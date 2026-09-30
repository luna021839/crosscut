import json
import unittest
from dataclasses import replace

from crosscut.direct_member_target_face import FocalFaceCandidate
from crosscut.focal_body_track import (FocalBodyFrame, FocalBodyObservation,
    body_reference_similarity, deduplicate_body_observations, focal_body_anchor_indices,
    resolve_focal_body_track)
from crosscut.pose import FormationPersonDescriptor


def observation(t, x=.2, *, anchor=False, score=.82, face=False, source=0, appearance=(1., 0.)):
    person = FormationPersonDescriptor((x, .1, .28, .75), (x+.14, .475), .95, ())
    candidate = FocalFaceCandidate((x+.08, .12, .1, .1), (1., 0.)) if anchor or face else None
    return FocalBodyObservation(source, t, person, appearance, score, candidate, anchor)


def frame(t, *observations):
    return FocalBodyFrame(0, t, tuple(observations))


class FocalBodyTrackTests(unittest.TestCase):
    @staticmethod
    def pose_owned(item):
        return replace(item,face_pose_consistent=True,
            pose_screen_xyc=tuple((.30+.02*i,.20+.04*i,.9) for i in range(8)))

    def window(self):
        return [frame(t, observation(t, anchor=t!=2.)) for t in (1.76, 1.88, 2., 2.12, 2.24)]

    def resolve(self, frames):
        return resolve_focal_body_track(frames, source_index=0, query_seconds=2.)

    def test_head_turned_body_track_does_not_invent_face(self):
        result = self.resolve(self.window())
        self.assertEqual(result.reason, 'observed_target_body_track')
        self.assertIsNotNone(result.observation)
        self.assertIsNone(result.observation.face)
        self.assertGreaterEqual(len(result.support_seconds), 2)
        self.assertFalse(result.to_dict()['face_geometry_extrapolated'])

    def test_wrong_frozen_face_on_conflicting_costume_cannot_be_anchor(self):
        frames = []
        for f in self.window():
            wrong = observation(f.source_seconds, x=.68, anchor=True, score=.55)
            frames.append(replace(f, observations=(*f.observations, wrong)))
        result = self.resolve(frames)
        self.assertEqual(result.observation.person.bbox_xywh[0], .2)

    def test_only_wrong_frozen_faces_leave_query_unresolved(self):
        frames = [frame(t, observation(t, x=.68, anchor=True, score=.55))
                  for t in (1.76, 1.88, 2., 2.12, 2.24)]
        self.assertIsNone(self.resolve(frames).observation)

    def test_two_exact_pose_owned_anchors_resolve_without_changing_observed_scores(self):
        frames=[]
        for t in (1.76,1.88,2.,2.12,2.24):
            target=observation(t,anchor=t!=2.,score=.64)
            if t!=2.: target=self.pose_owned(target)
            rival=observation(t,x=.68,score=.68,appearance=(0.,1.))
            frames.append(frame(t,target,rival))
        result=self.resolve(frames)
        self.assertIsNotNone(result.observation)
        self.assertEqual(result.observation.person.bbox_xywh[0],.2)
        self.assertEqual(result.observation.reference_similarity,.64)
        self.assertIsNone(result.observation.face)
        self.assertGreaterEqual(len(result.support_seconds),2)
        self.assertIn('unique_exact_face_pose',result.to_dict()['anchor_policy'])

    def test_new_anchor_family_cannot_replace_an_existing_resolved_identity(self):
        from crosscut.focal_body_track import _resolve_focal_body_track
        frames=[]
        for t in (1.76,1.88,2.,2.12,2.24):
            target=observation(t,anchor=t<2.,score=.82)
            rival=observation(t,x=.68,score=.95 if t==2. else .64,
                              anchor=t>2.,appearance=(0.,1.))
            if t>2.: rival=self.pose_owned(rival)
            frames.append(frame(t,target,rival))
        legacy=_resolve_focal_body_track(frames,source_index=0,query_seconds=2.,
                                         allow_exact_pose_anchors=False)
        expanded=_resolve_focal_body_track(frames,source_index=0,query_seconds=2.,
                                           allow_exact_pose_anchors=True)
        self.assertEqual(legacy.observation.person.bbox_xywh[0],.2)
        self.assertEqual(expanded.observation.person.bbox_xywh[0],.68)
        self.assertEqual(self.resolve(frames),legacy)

    def test_exact_pose_anchor_cannot_use_missing_ambiguous_or_conflicting_evidence(self):
        target=self.pose_owned(observation(1.76,anchor=True,score=.64))
        self.assertEqual(focal_body_anchor_indices((target,)),(0,))
        for value in (replace(target,face_pose_consistent=None),
                      replace(target,face_pose_consistent=False),
                      replace(target,reference_similarity=.599999),
                      replace(target,frozen_face_match=False)):
            with self.subTest(value=value):
                self.assertEqual(focal_body_anchor_indices((value,)),())
        rival=self.pose_owned(observation(1.76,x=.68,anchor=True,score=.64,appearance=(0.,1.)))
        self.assertEqual(focal_body_anchor_indices((target,rival)),())

    def test_exact_pose_anchors_do_not_bridge_empty_wrong_body_or_single_anchor(self):
        for failure in ('empty','wrong_body','single_anchor','weak_query_body'):
            frames=[]
            for t in (1.76,1.88,2.):
                target=observation(t,anchor=t!=2.,score=.64)
                if t!=2. and not (failure=='single_anchor' and t==1.76):
                    target=self.pose_owned(target)
                if t==2. and failure=='wrong_body':
                    target=replace(target,appearance=(0.,1.))
                if t==2. and failure=='weak_query_body':
                    target=replace(target,reference_similarity=.59)
                frames.append(frame(t,*(() if failure=='empty' and t==1.88 else (target,))))
            with self.subTest(failure=failure):
                self.assertIsNone(self.resolve(frames).observation)

    def test_occluded_query_never_substitutes_visible_neighbor(self):
        frames = self.window()
        frames[2] = frame(2., observation(2., x=.70, score=.72, face=True, appearance=(0., 1.)))
        self.assertEqual(self.resolve(frames).reason, 'target_trajectory_unresolved')

    def test_single_anchor_is_not_two_modal_temporal_proof(self):
        frames = [frame(t, observation(t, anchor=t==1.88)) for t in (1.76, 1.88, 2., 2.12, 2.24)]
        self.assertEqual(self.resolve(frames).reason, 'independent_face_body_anchors_missing')

    def test_empty_frame_breaks_trajectory_instead_of_disappearing(self):
        frames = [frame(1.64, observation(1.64, anchor=True)),
                  frame(1.76, observation(1.76, anchor=True)), frame(1.88),
                  frame(2., observation(2.))]
        self.assertEqual(self.resolve(frames).reason, 'target_trajectory_unresolved')

    def test_query_body_absence_remains_absence(self):
        frames = self.window()
        frames[2] = frame(2.)
        self.assertEqual(self.resolve(frames).reason, 'query_body_unobserved')

    def test_native_camera_jump_cannot_join_tracks(self):
        frames = [frame(1.76, observation(1.76, anchor=True)),
                  frame(1.88, observation(1.88, anchor=True)),
                  frame(2., observation(2., x=.70))]
        self.assertIsNone(self.resolve(frames).observation)

    def test_source_identity_is_not_crossed(self):
        with self.assertRaises(ValueError):
            self.resolve([FocalBodyFrame(1, 2., (observation(2., source=1),))])

    def test_duplicate_frame_is_not_independent_support(self):
        frames = self.window()
        with self.assertRaises(ValueError):
            self.resolve([*frames, frames[0]])

    def test_same_frame_duplicate_body_does_not_create_competitor(self):
        original = observation(2.)
        duplicate = replace(original, person=replace(original.person, detection_confidence=.85))
        self.assertEqual(deduplicate_body_observations((duplicate, original)), (original,))

    def test_overlapping_distinct_appearance_not_collapsed(self):
        original = observation(2.)
        other = replace(original, appearance=(0., 1.))
        self.assertEqual(len(deduplicate_body_observations((original, other))), 2)

    def test_frozen_face_flag_requires_actual_face(self):
        with self.assertRaises(ValueError):
            replace(observation(2.), frozen_face_match=True)

    def test_transient_vectors_not_serialized(self):
        result = self.resolve(self.window())
        serialized = json.dumps(result.to_dict())
        self.assertNotIn('embedding":', serialized)
        self.assertNotIn('appearance":', serialized)
        self.assertNotIn('probability', serialized)
        self.assertFalse(result.to_dict()['gallery_updated'])

    def test_body_reference_scores_are_not_probabilities(self):
        self.assertEqual(body_reference_similarity((1., 0.), ((-1., 0.),)*3), -1.)


if __name__ == '__main__':
    unittest.main()
