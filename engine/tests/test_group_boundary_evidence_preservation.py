from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest

from crosscut.boundary_affine_plan import BoundaryAffinePlan,BoundaryAffineDecision
from crosscut.boundary_repair import BoundaryRepairConfig
from crosscut.group_boundary_evidence_preservation import preserve_boundary_evidence,BoundaryEvidencePreservationUnavailable
from crosscut.render import RenderPlan,RenderSegment,RenderTransition,EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.visual_boundary_repair import repair_visual_timeline_boundaries
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_visual_boundary_repair import _choice,_descriptor


def repair_fixture(source_order=(0,1,0,1,0,1)):
    timeline=VisualTimelinePlan(tuple(replace(_choice(i,source)) for i,source in enumerate(source_order)),1.)
    sources=(Path('/fixture/source0.mp4'),Path('/fixture/source1.mp4'))
    clocks=(VerifiedRenderFrameClock(Fraction(30),3000,'a'*64),)*2
    repaired=repair_visual_timeline_boundaries(visual_timeline=timeline,sources=sources,
        audio_offsets=(0.,0.),scene_cut_times=((),()),output_fps=30.,
        descriptor_provider=lambda *_:_descriptor(.5),render_frame_clocks=clocks,
        boundary_indices=(4,),config=BoundaryRepairConfig(search_radius_frames=3))
    plan=RenderPlan(sources=sources,segments=tuple(RenderSegment(c.source_index,c.source_local_start,
        c.duration_seconds,c.canonical_master_start,source_duration=c.source_local_end-c.source_local_start)
        for c in repaired.timeline.choices),reference_source=0,aspect_ratio='16:9',
        output=Path('/fixture/not-rendered.mp4'),fps=30.,transitions=(RenderTransition(tier='C'),)*5)
    return plan,repaired


class BoundaryEvidencePreservationTests(unittest.TestCase):
    def changed_camera(self,plan,index=0):
        segments=list(plan.segments)
        segments[index]=replace(segments[index],editorial_transform=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.25),)))
        return replace(plan,segments=tuple(segments))

    def invoke(self,plan,repair,proposed,affine=None,new_timeline=None):
        return preserve_boundary_evidence(old_plan=plan,new_plan=proposed,
            old_timeline=repair.timeline,new_timeline=new_timeline or repair.timeline,
            boundary_repairs=repair,boundary_affine_plan=affine)

    def test_real_repair_producer_declares_all_offsets_and_guard_samples(self):
        _,repair=repair_fixture()
        self.assertEqual(len(repair.repairs),1)
        row=repair.repairs[0];support=row.measurement_support
        self.assertTrue(support['renderer_bound'])
        # Search +/-3 and a 4-frame scene guard. Includes the candidate
        # brackets, not merely selected ticks 149/150.
        self.assertLessEqual(support['output_frame_intervals'][0][0],142)
        self.assertGreaterEqual(support['output_frame_intervals'][0][1],158)
        self.assertGreater(len(support['descriptor_queries']),4)
        self.assertEqual(row.to_dict()['measurement_support']['output_fps'],30.)
        self.assertFalse(support['projected_part_or_quality_certificate'])

    def test_distant_camera_preserves_all_raw_attempts_and_affine_payloads(self):
        plan,repair=repair_fixture();raw=repair.to_dict()
        affine=BoundaryAffinePlan((BoundaryAffineDecision(4,0,1,4.+29/30.,5.,None,None,'fixture_unavailable'),))
        new_timeline=replace(repair.timeline,optimizer_score=.9)
        kept,kept_affine,audit=self.invoke(plan,repair,self.changed_camera(plan),affine,new_timeline)
        self.assertIs(kept.timeline,new_timeline)
        self.assertIs(kept.repairs,repair.repairs)
        self.assertIs(kept.attempt_input_timelines,repair.attempt_input_timelines)
        self.assertEqual(kept.to_dict()['repairs'],raw['repairs'])
        self.assertIs(kept_affine,affine)
        self.assertEqual((audit['retained_repair_count'],audit['retained_affine_decision_count']),(1,1))
        self.assertFalse(audit['new_media_or_part_certificate_issued'])

    def test_long_declared_bracket_is_not_shortened_to_nearby_cells(self):
        plan,repair=repair_fixture();row=repair.repairs[0]
        support=dict(row.measurement_support,output_frame_intervals=((1,179),))
        repair=replace(repair,repairs=(replace(row,measurement_support=support),))
        with self.assertRaisesRegex(BoundaryEvidencePreservationUnavailable,'shared_renderer_unit_changed'):
            self.invoke(plan,repair,self.changed_camera(plan))

    def test_small_bracket_still_binds_entire_shared_clock_and_canvas_run(self):
        plan,repair=repair_fixture((0,0,0,0,0,1))
        self.assertGreater(repair.repairs[0].measurement_support['output_frame_intervals'][0][0],120)
        with self.assertRaisesRegex(BoundaryEvidencePreservationUnavailable,'shared_renderer_unit_changed'):
            self.invoke(plan,repair,self.changed_camera(plan))

    def test_overlapping_camera_or_affine_transition_cannot_preserve_stale_geometry(self):
        plan,repair=repair_fixture()
        with self.assertRaises(BoundaryEvidencePreservationUnavailable):
            self.invoke(plan,repair,self.changed_camera(plan,index=4))
        transitions=list(plan.transitions);transitions[4]=replace(transitions[4],reason='changed_edge_recipe')
        with self.assertRaises(BoundaryEvidencePreservationUnavailable):
            self.invoke(plan,repair,replace(plan,transitions=tuple(transitions)))

    def test_missing_stale_or_outside_support_is_not_freshly_relabelled(self):
        plan,repair=repair_fixture();row=repair.repairs[0]
        changes=(None,dict(row.measurement_support,output_fps=29.97),
            dict(row.measurement_support,output_frame_intervals=((150,900),)))
        for support in changes:
            with self.subTest(support=support),self.assertRaises(BoundaryEvidencePreservationUnavailable):
                self.invoke(plan,replace(repair,repairs=(replace(row,measurement_support=support),)),self.changed_camera(plan))
        stale_affine=BoundaryAffinePlan((BoundaryAffineDecision(4,0,1,4.5,5.,None,None,'fixture_unavailable'),))
        with self.assertRaisesRegex(BoundaryEvidencePreservationUnavailable,'preimage_stale'):
            self.invoke(plan,repair,self.changed_camera(plan),stale_affine)

    def test_unknown_support_contract_rejects_at_typed_payload_boundary(self):
        _,repair=repair_fixture();row=repair.repairs[0]
        with self.assertRaisesRegex(ValueError,'support contract'):
            replace(row,measurement_support=dict(row.measurement_support,contract='unknown_future_support'))

    def test_no_op_retains_legacy_payload_without_inventing_a_new_support_binding(self):
        plan,repair=repair_fixture();row=repair.repairs[0]
        legacy=replace(repair,repairs=(replace(row,measurement_support=None),))
        retained,_,audit=self.invoke(plan,legacy,plan)
        self.assertIs(retained,legacy)
        self.assertTrue(audit['unchanged_complete_incumbent'])
        self.assertIsNone(retained.repairs[0].measurement_support)
        self.assertFalse(audit['new_media_or_part_certificate_issued'])


if __name__=='__main__':unittest.main()
