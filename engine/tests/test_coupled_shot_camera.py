from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from crosscut.boundary_subject_alignment import _track_for_segment,_TransformState,_project_box,_close_face_head_envelope
from crosscut.coupled_shot_camera import CoupledShotProposal,propose_coupled_shot_camera
from crosscut.coupled_shot_binding import (build_coupled_shot_binding,coupled_shot_binding_valid,
    coupled_geometry_identity,rebind_unchanged_coupled_shot)
from crosscut.cut_anchored_containment import CutAnchoredContainment,ContainmentSample
from crosscut.render import RenderSegment,RenderTransition
from crosscut.shot_context_refinement import attach_shot_context_proposal,attach_coupled_shot_proposal
from crosscut.post_render_qa import _bound_local_alignment_tracks,_transform_metrics,_IssueCollector,PostRenderQAConfig
from crosscut.shot_context_binding import _digest
from engine.tests import test_shot_context_camera as shot_fixtures
from engine.tests.test_shot_context_binding import motion_for

face=shot_fixtures.face


class CoupledShotCameraTests(unittest.TestCase):
    def fixture(self):
        source=shot_fixtures.ShotContextProposalTests();args=source.fixture();plan=args['plan']
        plan=replace(plan,segments=(*plan.segments,RenderSegment(0,3.,1.,3.)),
            transitions=(*plan.transitions,RenderTransition(tier='C')))
        args['plan']=plan;previous=source.run_proposal(args)
        motion=attach_shot_context_proposal(plan,motion_for(plan),previous)
        incoming=_track_for_segment(last_frame_time=29/30.,ramp_frames=14,fps=30.,entry=(1,_TransformState(1.1,.5,.5,0.)),exit=None)
        final=replace(previous.plan,segments=(previous.plan.segments[0],previous.plan.segments[1],
            replace(previous.plan.segments[2],editorial_transform=incoming),previous.plan.segments[3]))
        head=_close_face_head_envelope(face().face_box_xywh)
        evidence=tuple(dict(frame=i,face_box_xywh=list(face().face_box_xywh),head_envelope_xywh=list(head),
            basis='observed_cut_anchor' if i==0 else 'pixel_flow_plus_observed_geometry',consecutive_detector_misses=0) for i in range(14))
        pair=CoupledShotProposal(replace(previous,plan=final),evidence,tuple(range(60,74)),previous.source_clock,(),0.)
        return previous,motion,pair

    def test_atomic_pair_has_two_cameras_but_only_the_actual_two_boundaries(self):
        previous,motion,pair=self.fixture()
        changed=attach_coupled_shot_proposal(previous.plan,motion,pair)
        raw=changed.to_dict();plan=json.loads(json.dumps(pair.plan.to_json()))
        self.assertEqual(raw.get('shot_context_cameras',[]),[])
        self.assertEqual(len(raw['coupled_shot_cameras']),1)
        self.assertEqual(_bound_local_alignment_tracks(raw,plan),({1,2},{0,1}))
        self.assertNotIn(2,_bound_local_alignment_tracks(raw,plan)[1], 'neutral outgoing cut is not newly matched')
        collector=_IssueCollector();metrics=_transform_metrics({'editorial_motion':raw},plan,PostRenderQAConfig(),collector)
        self.assertTrue(all(metrics['count_reconciliation_checks'].values()),[x.to_dict() for x in collector.issues])
        self.assertFalse([x for x in collector.issues if x.code=='editorial_motion_shot_context_bindings_complete'])
        with self.assertRaises(TypeError):changed.coupled_shot_cameras[0]['segment_index']=2

    def test_missing_either_camera_revokes_pair_authority(self):
        previous,motion,pair=self.fixture();raw=attach_coupled_shot_proposal(previous.plan,motion,pair).to_dict()
        for index in (1,2):
            plan=pair.plan.to_json();plan['segments'][index]['editorial_transform']=None
            self.assertEqual(_bound_local_alignment_tracks(raw,json.loads(json.dumps(plan))),(set(),set()))
            damaged=deepcopy(raw);damaged['segments'][index]['keyframes']=[]
            self.assertEqual(_bound_local_alignment_tracks(damaged,json.loads(json.dumps(pair.plan.to_json()))),(set(),set()))

    def test_content_clock_subject_claims_and_cut_domain_are_not_forgeable_by_rehashing(self):
        _,_,pair=self.fixture();good=build_coupled_shot_binding(pair)
        mutations=(('incoming_native_frame_ids',[60]*14),('boundary_indices',[0,1,2]),
            ('professional_quality_approved',True),('incoming_neutral_from_frame',3),
            ('incoming_source_index',0),('incoming_native_cuts_in_active_window',1))
        for key,value in mutations:
            bad=deepcopy(good);bad[key]=value;bad['content_sha256']=_digest({k:v for k,v in bad.items() if k!='content_sha256'})
            self.assertFalse(coupled_shot_binding_valid(bad,pair.plan),key)
        bad=deepcopy(good);bad['incoming_head_evidence'][5]['face_box_xywh']=[.05,.05,.1,.2]
        bad['content_sha256']=_digest({k:v for k,v in bad.items() if k!='content_sha256'})
        self.assertFalse(coupled_shot_binding_valid(bad,pair.plan))

    def test_incoming_tail_must_be_neutral_even_with_updated_geometry_digest(self):
        _,_,pair=self.fixture();good=build_coupled_shot_binding(pair)
        track=_track_for_segment(last_frame_time=29/30.,ramp_frames=14,fps=30.,
            entry=(1,_TransformState(1.1)),exit=(2,_TransformState(1.1)))
        changed=replace(pair.plan,segments=(*pair.plan.segments[:2],replace(pair.plan.segments[2],editorial_transform=track),pair.plan.segments[3]))
        bad=deepcopy(good);bad['geometry_identity']=coupled_geometry_identity(changed,1)
        from crosscut.shot_context_binding import shot_context_geometry_identity
        bad['outgoing']['geometry_identity']=shot_context_geometry_identity(changed,1)
        bad['outgoing']['content_sha256']=_digest({k:v for k,v in bad['outgoing'].items() if k!='content_sha256'})
        bad['content_sha256']=_digest({k:v for k,v in bad.items() if k!='content_sha256'})
        self.assertFalse(coupled_shot_binding_valid(bad,changed))

    def test_attach_rejects_audio_source_or_fixed_neighbor_mutation_atomically(self):
        previous,motion,pair=self.fixture();before=motion.to_dict()
        mutations=[replace(pair.plan,master_audio_start=.2)]
        for index in range(len(pair.plan.segments)):
            segments=list(pair.plan.segments)
            segments[index]=replace(segments[index],source_start=segments[index].source_start+.1)
            mutations.append(replace(pair.plan,segments=tuple(segments)))
        for mutated in mutations:
            with self.assertRaises(ValueError):
                attach_coupled_shot_proposal(previous.plan,motion,
                    replace(pair,outgoing=replace(pair.outgoing,plan=mutated)))
            self.assertEqual(motion.to_dict(),before)

    def test_rehashed_evidence_cannot_skip_missing_detector_limit_or_anchor(self):
        _,_,pair=self.fixture();good=build_coupled_shot_binding(pair)
        for index,basis,misses in ((0,'pixel_flow_plus_observed_geometry',0),
                (3,'bounded_pixel_flow_containment',2),(4,'bounded_pixel_flow_containment',3),
                (2,'identity_certified',0)):
            bad=deepcopy(good)
            bad['incoming_head_evidence'][index].update(basis=basis,consecutive_detector_misses=misses)
            bad['content_sha256']=_digest({k:v for k,v in bad.items() if k!='content_sha256'})
            self.assertFalse(coupled_shot_binding_valid(bad,pair.plan))

    def run_proposal(self,previous,**extra):
        incoming=face(.26,.09,y=.35)
        def observed(frame):return face() if frame.mean()==1 else incoming if frame.mean()==2 else face(.26,.09)
        def composed(frame,segment,time,size):
            row=observed(frame)
            if segment.editorial_transform:
                key=segment.editorial_transform.evaluate(time)
                row=replace(row,face_box_xywh=_project_box(row.face_box_xywh,_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees))[0])
            return row
        capture=SimpleNamespace(frame_at_native_index=lambda source,*_:np.full((480,854,3),source,np.uint8),close=lambda:None)
        tracked=CutAnchoredContainment(tuple(ContainmentSample(i,incoming.face_box_xywh,
            'observed_cut_anchor' if i==0 else 'pixel_flow_plus_observed_geometry',0) for i in range(14)),True,None,None)
        with patch('crosscut.coupled_shot_camera.probe_render_frame_clock',return_value=previous.source_clock), \
                patch('crosscut.coupled_shot_camera._CaptureProvider',return_value=capture), \
                patch('crosscut.coupled_shot_camera.observe_shot_head',side_effect=observed), \
                patch('crosscut.coupled_shot_camera._observe_composed_head',side_effect=composed), \
                patch('crosscut.coupled_shot_camera.detect_roll_aware_faces',return_value=()), \
                patch('crosscut.coupled_shot_camera.track_cut_handle',return_value=tracked), \
                patch('crosscut.coupled_shot_camera.probe_media',return_value=SimpleNamespace(video_streams=(SimpleNamespace(width=3840,height=2160),))):
            return propose_coupled_shot_camera(previous=previous,**dict(dict(scene_cut_times=((),)*3),**extra))

    def test_general_producer_preserves_clocks_and_fixed_entry_without_song_ids(self):
        previous,_,_=self.fixture();pair=self.run_proposal(previous)
        self.assertEqual(pair.incoming_native_frame_ids,tuple(range(60,74)))
        self.assertEqual(pair.plan.segments[1].editorial_transform.evaluate(0.),previous.plan.segments[1].editorial_transform.evaluate(0.))
        self.assertEqual(pair.plan.master_audio_start,previous.plan.master_audio_start)
        self.assertEqual(pair.plan.output_frame_count,previous.plan.output_frame_count)
        for i in (0,3):self.assertIs(pair.plan.segments[i],previous.plan.segments[i])
        self.assertTrue(coupled_shot_binding_valid(build_coupled_shot_binding(pair),pair.plan))

    def test_native_scene_cut_and_runtime_budget_fail_without_silent_fallback(self):
        previous,_,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'native source cut'):
            self.run_proposal(previous,scene_cut_times=((),(),(2.2,)))
        for value in (0.,True,61.,float('nan')):
            with self.assertRaises(ValueError):self.run_proposal(previous,maximum_runtime_seconds=value)

    def test_automatic_stage_upgrades_once_and_keeps_single_when_pair_unavailable(self):
        from crosscut.shot_context_refinement import refine_shot_context_cameras
        previous,_,pair=self.fixture()
        before=replace(previous.plan,segments=(previous.plan.segments[0],
            replace(previous.plan.segments[1],editorial_transform=None),*previous.plan.segments[2:]))
        rows=deepcopy(previous.pixel_boundary_framing)
        rows[1].update(before=[.3,.3],after=[.01,.1])
        previous=replace(previous,pixel_boundary_framing=rows)
        for failed in (False,True):
            with patch('crosscut.shot_context_refinement.shot_context_shortlist',return_value=(1,)), \
                    patch('crosscut.shot_context_refinement.multi_cell_shot_context_shortlist',return_value=()), \
                    patch('crosscut.shot_context_refinement.propose_shot_context_camera',return_value=previous), \
                    patch('crosscut.coupled_shot_camera.propose_coupled_shot_camera',
                        side_effect=ValueError('tracked face unavailable') if failed else None,return_value=pair) as call:
                result=refine_shot_context_cameras(plan=before,timeline=None,motion=motion_for(before),scene_cut_times=((),)*3)
            call.assert_called_once()
            self.assertEqual(result.changed_segments,(1,) if failed else (1,2))
            self.assertEqual(len(result.motion.coupled_shot_cameras),0 if failed else 1)


    def test_unrelated_tail_retime_reuses_atomic_pair_without_changing_any_evidence(self):
        _,_,pair=self.fixture()
        before=replace(pair.plan,segments=(*pair.plan.segments,RenderSegment(0,4.,1.,4.),
            RenderSegment(0,5.,1.,5.)),transitions=(*pair.plan.transitions,RenderTransition(),RenderTransition()))
        pair=replace(pair,outgoing=replace(pair.outgoing,plan=before))
        good=build_coupled_shot_binding(pair);parts=list(before.segments);delta=4/30.
        parts[4]=replace(parts[4],duration=1.-delta)
        parts[5]=replace(parts[5],master_start=5.-delta,source_start=5.-delta,duration=1.+delta)
        after=replace(before,segments=tuple(parts))
        self.assertFalse(coupled_shot_binding_valid(good,after))
        rebound=rebind_unchanged_coupled_shot(good,before,after)
        self.assertTrue(coupled_shot_binding_valid(rebound,after))
        self.assertEqual(rebound['incoming_head_evidence'],good['incoming_head_evidence'])
        self.assertEqual(rebound['incoming_native_frame_ids'],good['incoming_native_frame_ids'])
        self.assertEqual(rebound['outgoing']['head_evidence'],good['outgoing']['head_evidence'])
        self.assertFalse(rebound['professional_quality_approved'])
        for index in (0,1,2,3):
            damaged=list(after.segments)
            damaged[index]=replace(damaged[index],source_start=damaged[index].source_start+.1)
            with self.subTest(index=index),self.assertRaisesRegex(ValueError,'context'):
                rebind_unchanged_coupled_shot(good,before,replace(after,segments=tuple(damaged)))
        damaged=deepcopy(good);damaged['incoming_head_evidence'][0]['basis']='pixel_flow_plus_observed_geometry'
        damaged['content_sha256']=_digest({k:v for k,v in damaged.items() if k!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'prior'):
            rebind_unchanged_coupled_shot(damaged,before,after)


if __name__=='__main__':unittest.main()
