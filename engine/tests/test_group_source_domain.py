from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.group_source_domain import GroupSourceDomain, bind_group_source_domain
from crosscut.group_source_span_proposal import build_group_source_span_proposal, refresh_group_source_span_observations
from crosscut.render import RenderPlan, RenderTransition, aligned_render_segment
from crosscut.render_boundary_clock import native_frame_clocks_for_output
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.visual_score import VisualTimelinePlan
from crosscut.timewarp import TimeWarpMap, TimeWarpConfig, TimeWarpControlPoint, TimeWarpResidualDiagnostics
from engine.tests.test_boundary_subject_alignment import _choice
from engine.tests.test_sparse_source_lattice import Sampler


class GroupSourceDomainTests(unittest.TestCase):
    def fixture(self):
        sources=tuple(Path(f'{i}.mp4') for i in range(6))
        domain=GroupSourceDomain(sources,(0,2,5),(.25,.5,.75),(None,)*3,((),)*3,
            tuple(VerifiedRenderFrameClock(Fraction(rate),1000,letter*64)
                for rate,letter in ((30,'a'),(25,'b'),(60,'c'))))
        local_path=(1,0,1,0,1)
        segments=tuple(aligned_render_segment(source_index=domain.request_index(local),
            reference_start=2.*i,common_master_start=0.,source_offset=domain.audio_offsets[local],
            duration=2.,timewarp=None) for i,local in enumerate(local_path))
        plan=RenderPlan(sources=sources,segments=segments,reference_source=0,aspect_ratio='16:9',
            output=Path('unused.mp4'),fps=30.,transitions=(RenderTransition(tier='C'),)*4)
        timeline=VisualTimelinePlan(tuple(replace(_choice(i,local,start=2.*i,end=2.*i+2.),
            source_local_start=segment.source_start,source_local_end=segment.source_end)
            for i,(local,segment) in enumerate(zip(local_path,segments))),1.)
        return dict(plan=plan,timeline=timeline,source_domain=domain,segment_index=2,
            source_index=2,entry_offset_frames=0,exit_offset_frames=0)

    def test_non_contiguous_domain_binds_files_offsets_clocks_and_local_timeline(self):
        args=self.fixture();result=build_group_source_span_proposal(**args);domain=args['source_domain']
        self.assertEqual([s.source_index for s in result.plan.segments],[2,0,5,0,2])
        self.assertEqual([c.source_index for c in result.clock_timeline.choices],[1,0,2,0,1])
        self.assertAlmostEqual(result.plan.segments[2].source_start,4.75)
        self.assertEqual(result.plan.sources[5],Path('5.mp4'))
        row=native_frame_clocks_for_output(result.plan,domain.clocks_by_request,(120,))[120]
        self.assertEqual((row.source_index,row.native_frame_index),(5,285))
        payload=result.to_dict()
        self.assertEqual(payload['contract'],'coupled_group_eligible_source_span_clock_proposal_v3')
        self.assertEqual(payload['source_index_domain'],'eligible_local')
        self.assertEqual((payload['source_index'],payload['request_source_index']),(2,5))
        self.assertEqual(payload['eligible_request_indices'],[0,2,5])
        self.assertFalse(payload['production_selected'])

    def test_replacement_and_incumbent_cuts_use_their_correct_local_array_rows(self):
        args=self.fixture();domain=args['source_domain']
        for cuts,reason in ((((),(),(5.4,)),'replacement.*broadcaster'),
                (((),(5.4,),()),'incumbent.*native cut')):
            with self.subTest(cuts=cuts),self.assertRaisesRegex(ValueError,reason):
                build_group_source_span_proposal(**dict(args,source_domain=replace(domain,scene_cut_times=cuts)))

    def test_non_contiguous_domain_maps_the_replacement_timewarp_not_another_source(self):
        args=self.fixture();domain=args['source_domain']
        warp=TimeWarpMap(.75,20.,21.,(TimeWarpControlPoint(0.,.75),
            TimeWarpControlPoint(20.,20.85)),(),TimeWarpResidualDiagnostics(None,None,None,0),
            TimeWarpConfig(),False,None)
        domain=replace(domain,source_timewarps=(None,None,warp))
        result=build_group_source_span_proposal(**dict(args,source_domain=domain))
        self.assertAlmostEqual(result.plan.segments[2].source_start,4.77)
        self.assertAlmostEqual(result.plan.segments[2].source_end,6.78)
        self.assertAlmostEqual(result.plan.segments[2].source_playback_rate,1.005)
        native=native_frame_clocks_for_output(result.plan,domain.clocks_by_request,(120,))[120]
        self.assertEqual((native.source_index,native.native_frame_index),(5,286))
        self.assertEqual(domain.to_dict()['source_timewarps'][2],warp.to_dict())

    def test_refresh_never_opens_or_senses_excluded_request_sources(self):
        args=self.fixture();proposal=build_group_source_span_proposal(**args)
        Sampler.instances=[];Sampler.unreadable_source=None
        with patch('crosscut.visual_score._VideoSampler',Sampler):
            fresh=refresh_group_source_span_observations(proposal=proposal,sync_scores=(.9,.8,.7))
        self.assertEqual([s.source for s in Sampler.instances],[0,2,5])
        self.assertEqual([len(s.times) for s in Sampler.instances],[10,0,5])
        self.assertTrue(all(s.closed for s in Sampler.instances))
        self.assertEqual([c.source_index for c in fresh.choices],[1,0,2,0,1])
        self.assertAlmostEqual(fresh.choices[2].source_local_start,4.75)
        self.assertIn('not_a_selected_timeline',fresh.analysis_basis)
        self.assertEqual(fresh.candidate_review_summaries,())

    def test_redundant_arrays_or_local_clock_keys_cannot_override_domain(self):
        args=self.fixture();domain=args['source_domain']
        for extra in (dict(audio_offsets=(0.,)*3),dict(source_timewarps=()),
                dict(scene_cut_times=((1.,),(),())),
                dict(source_clocks=dict(enumerate(domain.render_frame_clocks)))):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                build_group_source_span_proposal(**dict(args,**extra))
        correct=dict(audio_offsets=domain.audio_offsets,source_timewarps=domain.source_timewarps,
            scene_cut_times=domain.scene_cut_times,source_clocks=domain.clocks_by_request)
        self.assertEqual(build_group_source_span_proposal(**dict(args,**correct)).source_index,2)

    def test_malformed_mapping_or_mismatched_plan_fails_before_sensing(self):
        args=self.fixture();domain=args['source_domain']
        for indices in ((0,0,5),(0,True,5),(0,2,6),(0,2),()):
            with self.subTest(indices=indices),self.assertRaises(ValueError):
                replace(domain,eligible_indices=indices)
        with self.assertRaises(ValueError):domain.eligible_index(1)
        with self.assertRaises(ValueError):domain.request_index(3)
        with self.assertRaises(ValueError):domain.request_index(True)
        with self.assertRaisesRegex(ValueError,'source files/order'):
            build_group_source_span_proposal(**dict(args,plan=replace(args['plan'],
                sources=tuple(reversed(args['plan'].sources)))))
        with self.assertRaisesRegex(ValueError,'render clock'):
            build_group_source_span_proposal(**dict(args,timeline=replace(args['timeline'],
                choices=tuple(replace(c,source_index=0) for c in args['timeline'].choices))))

    def test_mapping_does_not_claim_file_identity_or_alignment_proof(self):
        domain=self.fixture()['source_domain'];payload=domain.to_dict()
        self.assertFalse(payload['file_identity_verified'])
        self.assertIn('caller_supplied',payload['source_reference_alignment_authority'])
        self.assertEqual(payload['eligible_sources'],['0.mp4','2.mp4','5.mp4'])
        self.assertEqual([r['frame_rate'] for r in payload['render_frame_clocks']],['30','25','60'])
        self.assertIs(bind_group_source_domain(plan=self.fixture()['plan'],source_domain=domain),domain)


if __name__=='__main__':unittest.main()
