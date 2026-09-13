import copy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from crosscut.render import (RenderSegment, RenderTransition, TransitionKind, NormalizedCrop,
    PanCropTrack, PanCropKeyframe, SourceLookAdjustment, BoundaryLookTrack, BoundaryLookKeyframe,
    EditorialTransformTrack, EditorialTransformKeyframe)
from crosscut.render_plan_serialization import render_plan_from_json
from crosscut.render_reference_clock import load_render_reference_clock
from crosscut.render_frame_schedule import FRAME_SCHEDULE_CONTRACT
from engine.tests.test_render_boundary_clock import plan, clocks


class RenderReferenceClockTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = replace(plan((RenderSegment(0,.0143,.084,0.),
            RenderSegment(0,.0983,.084,.084),RenderSegment(1,.021,1.,.168)),30.),
            output=Path(self.tmp.name)/'render.mp4')
        self.sidecar = self.p.output.with_suffix('.mp4.frame-clock.json')
        self.clocks = clocks(Fraction(60))

    def write(self, p=None, **overrides):
        p=p or self.p
        self.payload = dict(contract=FRAME_SCHEDULE_CONTRACT, enabled=True,
            source_clocks={str(i):self.clocks[i].to_dict() for i in {s.source_index for s in p.segments}},
            timestamp_fallback_source_indices=[],plan=p.to_json(),quality_approval=False)
        self.payload.update(overrides)
        self.sidecar.write_text(json.dumps(self.payload))

    def load(self, indices=(0,1,2,3,4,5), p=None, probe=None):
        p=p or self.p
        return load_render_reference_clock(p.output,p.to_json(),indices,
            probe=probe or (lambda path:self.clocks[p.sources.index(path)]))

    def test_exact_indices_use_coalesced_clock_not_legacy_segment_origin(self):
        self.write()
        self.assertEqual([f.native_frame_index for f in self.load().frames.values()],[1,3,5,7,9,1])

    def test_lossless_roundtrip_preserves_all_geometry_and_look_fields(self):
        segment=RenderSegment(0,.05,1.,0.,
            pan_crop=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.1,.1,.8,.8)),)),
            editorial_transform=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),)),
            look_adjustment=SourceLookAdjustment(brightness=.01),
            boundary_look_track=BoundaryLookTrack((BoundaryLookKeyframe(0.,brightness=.01),)))
        p=plan((segment,RenderSegment(1,.04,1.,1.)))
        self.assertEqual(render_plan_from_json(p.to_json()),p)
        p=replace(p,segments=(replace(segment,pan_crop=None,crop=NormalizedCrop(.1,.1,.8,.8)),p.segments[1]))
        self.assertEqual(render_plan_from_json(p.to_json()),p)

    def test_redundant_clock_tampering_and_unknown_fields_rejected(self):
        for mutate in (lambda p:p['segments'][0].update(source_playback_rate=1.01),
                       lambda p:p.update(unrecognized_field='not silently dropped')):
            raw=copy.deepcopy(self.p.to_json());mutate(raw)
            with self.assertRaisesRegex(ValueError,'round-trip'):
                render_plan_from_json(raw)

    def test_present_mismatched_or_corrupt_sidecar_does_not_use_legacy_clock(self):
        self.write(plan=replace(self.p,fps=24.).to_json())
        with self.assertRaises(ValueError):self.load()
        self.sidecar.write_text('broken')
        with self.assertRaises(ValueError):self.load()

    def test_current_source_clock_is_reverified(self):
        self.write()
        with self.assertRaisesRegex(ValueError,'no longer matches'):
            self.load(probe=lambda path:replace(self.clocks[0],frame_count=7000))

    def test_explicit_fallback_source_not_promoted_to_verified(self):
        self.write(source_clocks={'0':self.clocks[0].to_dict()},timestamp_fallback_source_indices=[1])
        result=self.load()
        self.assertEqual(set(result.frames),{0,1,2,3,4})
        self.assertEqual(result.timestamp_fallback_sources,(1,))

    def test_disabled_clock_requires_all_sources_in_fallback(self):
        self.write(enabled=False)
        with self.assertRaises(ValueError):self.load()
        self.write(enabled=False,source_clocks={},timestamp_fallback_source_indices=[0,1])
        self.assertEqual(self.load().frames,{})

    def test_missing_old_sidecar_returns_explicit_no_clock(self):
        self.assertIsNone(self.load())

    def test_overlap_samples_rejected_but_pure_samples_remain_available(self):
        p=replace(self.p,segments=(RenderSegment(0,0.,1.,0.),RenderSegment(1,0.,1.,1.)),
            transitions=(RenderTransition(TransitionKind.FACE_DISSOLVE,4),))
        self.write(p)
        self.assertEqual(set(self.load((29,34),p).frames),{29,34})
        with self.assertRaisesRegex(ValueError,'composited'):
            self.load((30,),p)
