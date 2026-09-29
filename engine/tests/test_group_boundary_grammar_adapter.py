from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

from engine.crosscut.editorial_motion import (
    EditorialMotionPlan,
    EditorialMotionSegment,
)
from engine.crosscut.group_boundary_grammar_adapter import (
    GroupBoundaryGrammarAdapterError,
    adapt_group_boundary_grammar_evidence,
    validate_group_boundary_grammar_plan,
)
from engine.crosscut.ive import AspectRatio
from engine.crosscut.professional_montage_grammar import (
    MontageGrammarEvidenceError,
    ProfessionalBoundaryGrammarEvidence,
)
from engine.crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
)
from engine.crosscut.rendered_proxy_review import (
    ANALYSIS_BASIS as PROXY_ANALYSIS_BASIS,
    INTRODUCED_CROSS_SOURCE_DOMAIN,
    NATIVE_SOURCE_CUT_DOMAIN,
    TIER_D_CONTINUITY_DOMAIN,
    RenderedProxyReview,
    _boundary_domain_summary, _boundary_evidence, ProxyReviewConfig,
)
from engine.crosscut.section_director import (
    SectionDirective,
    SectionPlan,
    SectionRole,
)
from engine.crosscut.stage_edit_dynamic_crop_motion_authority_v3 import (
    DynamicCropMotionAuthorityV3,
    apply_dynamic_crop_motion_authority_to_render_plan_v3,
    corroborate_dynamic_crop_motion_timeline_v3,
)
from engine.crosscut.stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropTrackDecodeV3,
)
from engine.crosscut.timeline import ShotType
from engine.crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(len(ordered) - 1, low + 1)
        alpha = position - low
        return ordered[low] * (1.0 - alpha) + ordered[high] * alpha

    return {
        "count": len(ordered),
        "minimum": ordered[0] if ordered else None,
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "maximum": ordered[-1] if ordered else None,
    }


def _domain_summary(
    domain: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    scores = [float(item["weighted_score"]) for item in records]
    issues = [int(item["boundary_index"]) for item in records if not item["passed"]]
    p10 = _distribution(scores)["p10"]
    return {
        "domain": domain,
        "count_basis": domain,
        "boundary_count": len(records),
        "issue_count": len(issues),
        "issue_fraction": len(issues) / len(records) if records else 0.0,
        "issue_indices": issues,
        "score": {
            "distribution": _distribution(scores),
            "p10": p10,
            "evidence_status": "OBSERVED" if records else "NOT_APPLICABLE",
        },
        "pass_policy": (
            "editorial_boundary_p10_and_issue_fraction_gate"
            if domain == INTRODUCED_CROSS_SOURCE_DOMAIN
            else "zero_technical_integrity_issues"
        ),
        "passed": not issues,
    }


class GroupBoundaryGrammarAdapterTests(unittest.TestCase):
    def test_all_domain_verdicts_are_derived_not_trusted(self):
        for domain in (INTRODUCED_CROSS_SOURCE_DOMAIN, TIER_D_CONTINUITY_DOMAIN, NATIVE_SOURCE_CUT_DOMAIN):
            with self.subTest(domain=domain):
                bundle = self._bundle()
                proxy = deepcopy(bundle["rendered_proxy_review"].payload)
                summary = proxy["boundary_domains"][domain]
                summary["passed"] = not summary["passed"]
                bundle["rendered_proxy_review"] = proxy
                with self.assertRaisesRegex(GroupBoundaryGrammarAdapterError, "derived passed mismatch"):
                    self._adapt(bundle)

    def test_real_producer_unknown_scores_flow_through_grammar_without_perfect_fallback(self):
        bundle = self._bundle()
        proxy = deepcopy(bundle["rendered_proxy_review"].payload)
        for boundary in proxy["boundaries"]:
            if boundary["review_domain"] == INTRODUCED_CROSS_SOURCE_DOMAIN:
                boundary.update(weighted_score=None, passed=False, evidence_coverage=_boundary_evidence({}))
        proxy["boundary_domains"] = {domain: _boundary_domain_summary(
            proxy["boundaries"], domain=domain, config=ProxyReviewConfig()) for domain in proxy["boundary_domains"]}
        proxy["introduced_cross_source_boundary_issue_indices"] = [0, 2]
        bundle["rendered_proxy_review"] = proxy
        result = self._adapt(bundle)
        self.assertIsNone(result.introduced_boundary_continuity.score_p10)
        self.assertEqual(result.introduced_boundary_continuity.issue_count, 2)

    @staticmethod
    def _rehash(payload: dict[str, object], field: str) -> None:
        unsigned = deepcopy(payload)
        unsigned[field] = None
        payload[field] = hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def _bundle(self) -> dict[str, object]:
        zoom_track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.0),
                EditorialTransformKeyframe(1.0, 1.05),
                EditorialTransformKeyframe(2.0, 1.0),
            ),
            reason="synthetic_zoom_pulse",
        )
        chorus_track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.0),
                EditorialTransformKeyframe(
                    1.0,
                    1.08,
                    center_x=0.51,
                    roll_degrees=0.3,
                ),
                EditorialTransformKeyframe(2.0, 1.0),
            ),
            reason="synthetic_chorus_motion",
        )
        segments = (
            RenderSegment(
                source_index=2,
                source_start=10.0,
                duration=5.0,
                master_start=0.0,
                source_duration=5.1,
                editorial_transform=zoom_track,
            ),
            RenderSegment(0, 20.0, 5.0, 5.0),
            RenderSegment(0, 25.0, 5.0, 10.0),
            RenderSegment(
                source_index=2,
                source_start=30.0,
                duration=5.0,
                master_start=15.0,
                editorial_transform=chorus_track,
            ),
        )
        plan = RenderPlan(
            sources=(
                Path("/tmp/synthetic-request-0.mp4"),
                Path("/tmp/synthetic-excluded-1.mp4"),
                Path("/tmp/synthetic-request-2.mp4"),
            ),
            segments=segments,
            reference_source=0,
            aspect_ratio=AspectRatio.LANDSCAPE,
            output=Path("/tmp/synthetic-output.mp4"),
            master_audio_start=100.0,
            width=1920,
            height=1080,
            fps=30.0,
            transitions=(
                RenderTransition(reason="synthetic_cross_source"),
                RenderTransition(
                    reason="same_source_hold_no_visible_switch",
                    tier="D",
                ),
                RenderTransition(reason="synthetic_cross_source"),
            ),
        )
        internal_canonical = (102.5, (), 112.0, ())
        choices = tuple(
            VisualTimelineChoice(
                segment_index=index,
                source_index={2: 0, 0: 1}[segment.source_index],
                canonical_master_start=100.0 + segment.master_start,
                canonical_master_end=100.0 + segment.master_start + segment.duration,
                source_local_start=segment.source_start,
                source_local_end=segment.source_end,
                quality_score=0.9,
                composition_score=0.9,
                boundary_score=0.9,
                shot_scale=0.4,
                shot_type=ShotType.MEDIUM,
                shot_id=f"synthetic-{index}",
                internal_scene_cut_master_seconds=(
                    (internal_canonical[index],)
                    if isinstance(internal_canonical[index], float)
                    else ()
                ),
            )
            for index, segment in enumerate(segments)
        )
        selected_timeline = VisualTimelinePlan(choices=choices, optimizer_score=88.0)

        section_plan = SectionPlan(
            sections=(
                SectionDirective(
                    section_index=0,
                    role=SectionRole.VERSE_LIKE,
                    start_seconds=100.0,
                    end_seconds=110.0,
                    energy_score=0.4,
                    confidence=0.9,
                    phrase_count=2,
                    preferred_source_hold_beats=4,
                    target_source_switches_per_minute=13.5,
                    camera_motion_budget=0.3,
                    target_roll_events_per_minute=0.0,
                    establish_with_wide_shot=False,
                    protect_last_beat_from_extra_cut=False,
                ),
                SectionDirective(
                    section_index=1,
                    role=SectionRole.CHORUS_LIKE,
                    start_seconds=110.0,
                    end_seconds=120.0,
                    energy_score=0.9,
                    confidence=0.9,
                    phrase_count=2,
                    preferred_source_hold_beats=2,
                    target_source_switches_per_minute=17.0,
                    camera_motion_budget=0.8,
                    target_roll_events_per_minute=0.35,
                    establish_with_wide_shot=False,
                    protect_last_beat_from_extra_cut=False,
                ),
            )
        )

        timeline_boundaries = [
            {
                "boundary_index": 0,
                "outgoing_segment_index": 0,
                "incoming_segment_index": 1,
                "at_seconds": 105.0,
                "outgoing_source_id": "0",
                "incoming_source_id": "1",
                "source_switch": True,
                "visual_boundary": True,
            },
            {
                "boundary_index": 1,
                "outgoing_segment_index": 1,
                "incoming_segment_index": 2,
                "at_seconds": 110.0,
                "outgoing_source_id": "1",
                "incoming_source_id": "1",
                "source_switch": False,
                "visual_boundary": False,
            },
            {
                "boundary_index": 2,
                "outgoing_segment_index": 2,
                "incoming_segment_index": 3,
                "at_seconds": 115.0,
                "outgoing_source_id": "1",
                "incoming_source_id": "0",
                "source_switch": True,
                "visual_boundary": True,
            },
        ]
        timeline_review: dict[str, object] = {
            "passed": True,
            "segment_count": 4,
            "boundary_count": 3,
            "internal_scene_cut_count": 2,
            "comparable_visual_boundary_count": 4,
            "visual_shot_count": 5,
            "total_duration_seconds": 20.0,
            "source_switch_count": 2,
            "source_switch_rate": 0.5,
            "source_switches_per_minute": 6.0,
            "unique_source_count": 2,
            "boundaries": timeline_boundaries,
        }

        proxy_boundaries: list[dict[str, object]] = [
            {
                "boundary_index": 0,
                "cut_frame_index": 150,
                "outgoing_source_index": 2,
                "incoming_source_index": 0,
                "review_domain": INTRODUCED_CROSS_SOURCE_DOMAIN,
                "transition_reason": "synthetic_cross_source",
                "transition_tier": None,
                "source_clock_contiguous": False,
                "transition_kind": "hard_cut",
                "transition_frame_count": 0,
                "transition_duration_seconds": 0.0,
                "weighted_score": 85.0,
                "passed": True,
            },
            {
                "boundary_index": 1,
                "cut_frame_index": 300,
                "outgoing_source_index": 0,
                "incoming_source_index": 0,
                "review_domain": TIER_D_CONTINUITY_DOMAIN,
                "transition_reason": "same_source_hold_no_visible_switch",
                "transition_tier": "D",
                "source_clock_contiguous": True,
                "transition_kind": "hard_cut",
                "transition_frame_count": 0,
                "transition_duration_seconds": 0.0,
                "weighted_score": 100.0,
                "passed": True,
            },
            {
                "boundary_index": 2,
                "cut_frame_index": 450,
                "outgoing_source_index": 0,
                "incoming_source_index": 2,
                "review_domain": INTRODUCED_CROSS_SOURCE_DOMAIN,
                "transition_reason": "synthetic_cross_source",
                "transition_tier": None,
                "source_clock_contiguous": False,
                "transition_kind": "hard_cut",
                "transition_frame_count": 0,
                "transition_duration_seconds": 0.0,
                "weighted_score": 75.0,
                "passed": False,
            },
        ]
        by_domain = {
            domain: [
                item for item in proxy_boundaries if item["review_domain"] == domain
            ]
            for domain in (
                INTRODUCED_CROSS_SOURCE_DOMAIN,
                TIER_D_CONTINUITY_DOMAIN,
                NATIVE_SOURCE_CUT_DOMAIN,
            )
        }
        proxy_payload: dict[str, object] = {
            "schema_version": 3,
            "review_policy": ProxyReviewConfig().to_dict(),
            "analysis": PROXY_ANALYSIS_BASIS,
            "transition_count": 3,
            "reviewed_boundary_count": 3,
            "introduced_cross_source_count": 2,
            "tier_d_continuity_count": 1,
            "native_source_cut_count": 0,
            "boundary_domains": {
                domain: _domain_summary(domain, records)
                for domain, records in by_domain.items()
            },
            "introduced_cross_source_boundary_issue_indices": [2],
            "boundaries": proxy_boundaries,
            "passed": False,
        }

        editorial_motion = EditorialMotionPlan(
            segments=tuple(
                EditorialMotionSegment(
                    segment_index=index,
                    source_index={2: 0, 0: 1}[segment.source_index],
                    track=segment.editorial_transform,
                    reasons=("synthetic",) if segment.editorial_transform else (),
                )
                for index, segment in enumerate(segments)
            ),
            boundary_match_count=0,
            zoom_event_count=2,
            pan_event_count=1,
            roll_event_count=1,
            blocked_dissolve_segment_count=0,
            section_directed=True,
        )
        source_index_mapping = {
            "request_index_domain": "request_source_index",
            "eligible_index_domain": "eligible_source_index",
            "render_plan_index_domain": "render_plan_sources_array_index",
            "domain_equivalences": {
                "render_plan_sources_array_index": "request_source_index",
            },
            "eligible_to_request": [2, 0],
            "request_to_eligible": [1, None, 0],
            "eligible_sources": [
                {
                    "eligible_source_index": 0,
                    "request_source_index": 2,
                    "source": str(plan.sources[2]),
                },
                {
                    "eligible_source_index": 1,
                    "request_source_index": 0,
                    "source": str(plan.sources[0]),
                },
            ],
            "payload_index_domains": {
                "scene_analysis[].source_index": "request_source_index",
                "visual_timeline.choices[].source_index": "eligible_source_index",
                "editorial_motion.segments[].source_index": "eligible_source_index",
                "render_plan.segments[].source_index": "render_plan_sources_array_index",
                "rendered_proxy_review.boundaries[].outgoing_source_index": "request_source_index",
                "rendered_proxy_review.boundaries[].incoming_source_index": "request_source_index",
            },
        }
        return {
            "performance_id": "synthetic-group",
            "render_plan": plan,
            "source_index_mapping": source_index_mapping,
            "selected_timeline": selected_timeline,
            "source_scene_cut_times": ((27.0,), (), (12.55,)),
            "timeline_review": timeline_review,
            "section_plan": section_plan,
            "rendered_proxy_review": RenderedProxyReview(proxy_payload),
            "editorial_motion": editorial_motion,
        }

    def _adapt(self, bundle: dict[str, object]) -> ProfessionalBoundaryGrammarEvidence:
        return adapt_group_boundary_grammar_evidence(**bundle)  # type: ignore[arg-type]

    def _attach_v3_motion(
        self,
        bundle: dict[str, object],
        *,
        motion_segment_index: int = 2,
    ) -> None:
        plan = bundle["render_plan"]
        authority = DynamicCropMotionAuthorityV3.create(
            release_freeze_content_sha256="a" * 64,
            release_freeze_file_sha256="b" * 64,
            source_commit_oid="c" * 40,
            candidate_id="group-grammar-v3-test",
        )
        decodes: list[DynamicCropTrackDecodeV3] = []
        provisional_segments: list[RenderSegment] = []
        for index, segment in enumerate(plan.segments):
            keyframes = (
                (
                    EditorialTransformKeyframe(0.0, 1.02),
                    EditorialTransformKeyframe(
                        4.0,
                        1.05,
                        center_x=0.52,
                    ),
                )
                if index == motion_segment_index
                else (EditorialTransformKeyframe(0.0, 1.02),)
            )
            original = EditorialTransformTrack(
                keyframes,
                reason="stage_edit_dynamic_crop_head_v3",
            )
            decodes.append(
                DynamicCropTrackDecodeV3(
                    source_id=f"source-{segment.source_index}",
                    source_index=segment.source_index,
                    eligible=True,
                    loss_masked=False,
                    track=original,
                    predicted_keyframe_count=len(keyframes),
                    applied_keyframe_count=len(keyframes),
                    minimum_safe_zoom_projection_count=0,
                    rejection_reasons=(),
                    motion_logit=20.0,
                    shot_duration_seconds=segment.duration,
                )
            )
            provisional_segments.append(
                replace(
                    segment,
                    editorial_transform=EditorialTransformTrack(
                        keyframes,
                        reason=(
                            "stage_edit_dynamic_crop_head_v3_"
                            "stage_edit_preserved"
                        ),
                    ),
                )
            )
        provisional_plan = replace(
            plan,
            segments=tuple(provisional_segments),
        )
        final_plan = apply_dynamic_crop_motion_authority_to_render_plan_v3(
            render_plan=provisional_plan,
            decodes=tuple(decodes),
            authority=authority,
        )
        evidence = corroborate_dynamic_crop_motion_timeline_v3(
            render_plan=final_plan,
            decodes=tuple(decodes),
            authority=authority,
        )
        bundle["render_plan"] = final_plan
        bundle["editorial_motion"] = None
        bundle["dynamic_crop_version"] = 3
        bundle["dynamic_crop_motion_v3"] = evidence.to_dict()

    def test_exact_typed_payloads_roundtrip_to_output_clock_evidence(self) -> None:
        evidence = self._adapt(self._bundle())
        self.assertEqual(evidence.source_switch_times_seconds, (5.0, 15.0))
        self.assertAlmostEqual(evidence.native_internal_cut_times_seconds[0], 2.5)
        self.assertAlmostEqual(evidence.native_internal_cut_times_seconds[1], 12.0)
        self.assertEqual(
            [(item.kind, item.start_seconds, item.end_seconds) for item in evidence.sections],
            [("general", 0.0, 10.0), ("chorus", 10.0, 20.0)],
        )
        self.assertEqual(evidence.introduced_boundary_continuity.boundary_count, 2)
        self.assertEqual(evidence.introduced_boundary_continuity.issue_count, 1)
        self.assertAlmostEqual(evidence.introduced_boundary_continuity.score_p10 or 0.0, 76.0)
        self.assertEqual(
            evidence.editorial_transforms.zoom_event_times_seconds,
            (1.0 / 30.0, 15.0 + 1.0 / 30.0),
        )
        self.assertEqual(
            evidence.editorial_transforms.pan_event_times_seconds,
            (15.0 + 1.0 / 30.0,),
        )
        self.assertEqual(
            evidence.editorial_transforms.roll_event_times_seconds,
            (15.0 + 1.0 / 30.0,),
        )
        self.assertEqual(
            ProfessionalBoundaryGrammarEvidence.from_mapping(evidence.to_dict()),
            evidence,
        )

    def test_constant_v3_style_tracks_are_not_counted_as_motion_on_every_shot(
        self,
    ) -> None:
        bundle = self._bundle()
        constant_track = EditorialTransformTrack(
            (EditorialTransformKeyframe(0.0, 1.02),),
            reason="stage_edit_dynamic_crop_head_v3",
        )
        plan = bundle["render_plan"]
        bundle["render_plan"] = replace(
            plan,
            segments=tuple(
                replace(segment, editorial_transform=constant_track)
                for segment in plan.segments
            ),
        )
        motion = bundle["editorial_motion"]
        bundle["editorial_motion"] = EditorialMotionPlan(
            segments=tuple(
                replace(
                    segment,
                    track=constant_track,
                    reasons=("stage_edit_dynamic_crop_head_v3",),
                )
                for segment in motion.segments
            ),
            boundary_match_count=0,
            zoom_event_count=0,
            pan_event_count=0,
            roll_event_count=0,
            blocked_dissolve_segment_count=0,
        )

        evidence = self._adapt(bundle)

        self.assertEqual(evidence.editorial_transforms.segment_count, 4)
        self.assertEqual(evidence.editorial_transforms.applied_segment_count, 0)
        self.assertEqual(
            evidence.editorial_transforms.zoom_event_times_seconds,
            (),
        )

    def test_active_to_active_v3_keyframe_value_changes_are_visible_events(
        self,
    ) -> None:
        bundle = self._bundle()
        constant_track = EditorialTransformTrack(
            (EditorialTransformKeyframe(0.0, 1.02),),
            reason="stage_edit_dynamic_crop_head_v3",
        )
        pulse_track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.02),
                EditorialTransformKeyframe(1.0, 1.08),
                EditorialTransformKeyframe(2.0, 1.02),
            ),
            reason="stage_edit_dynamic_crop_head_v3",
        )
        plan = bundle["render_plan"]
        tracks = (constant_track, constant_track, pulse_track, constant_track)
        bundle["render_plan"] = replace(
            plan,
            segments=tuple(
                replace(segment, editorial_transform=track)
                for segment, track in zip(plan.segments, tracks, strict=True)
            ),
        )
        motion = bundle["editorial_motion"]
        bundle["editorial_motion"] = EditorialMotionPlan(
            segments=tuple(
                replace(
                    segment,
                    track=track,
                    reasons=("stage_edit_dynamic_crop_head_v3",),
                )
                for segment, track in zip(
                    motion.segments,
                    tracks,
                    strict=True,
                )
            ),
            boundary_match_count=0,
            zoom_event_count=2,
            pan_event_count=0,
            roll_event_count=0,
            blocked_dissolve_segment_count=0,
        )

        evidence = self._adapt(bundle)

        self.assertEqual(evidence.editorial_transforms.applied_segment_count, 1)
        self.assertEqual(
            evidence.editorial_transforms.zoom_event_times_seconds,
            (10.0 + 1.0 / 30.0, 11.0 + 1.0 / 30.0),
        )

    def test_v3_uses_only_authorized_renderer_visible_component_clocks(
        self,
    ) -> None:
        bundle = self._bundle()
        self._attach_v3_motion(bundle)

        evidence = self._adapt(bundle)

        self.assertEqual(evidence.editorial_transforms.applied_segment_count, 1)
        self.assertEqual(
            evidence.editorial_transforms.zoom_event_times_seconds,
            (10.0 + 1.0 / 30.0,),
        )
        self.assertEqual(
            evidence.editorial_transforms.pan_event_times_seconds,
            (10.0 + 1.0 / 30.0,),
        )
        self.assertEqual(evidence.editorial_transforms.roll_event_times_seconds, ())

    def test_v3_missing_evidence_and_full_plan_hash_tamper_fail_closed(self) -> None:
        bundle = self._bundle()
        self._attach_v3_motion(bundle)
        missing = dict(bundle)
        missing["dynamic_crop_motion_v3"] = None
        with self.assertRaises(GroupBoundaryGrammarAdapterError):
            self._adapt(missing)

        tampered = dict(bundle)
        tampered["render_plan"] = replace(
            tampered["render_plan"],
            output=Path("/tmp/other-final-output.mp4"),
        )
        with self.assertRaises(GroupBoundaryGrammarAdapterError):
            self._adapt(tampered)

    def test_v3_self_hash_and_renderer_clock_component_tamper_fail_closed(
        self,
    ) -> None:
        for case in ("self_hash", "budget_rebind", "clock", "component"):
            with self.subTest(case=case):
                bundle = self._bundle()
                self._attach_v3_motion(bundle)
                payload = deepcopy(bundle["dynamic_crop_motion_v3"])
                segment = payload["segment_evidence"][2]
                if case == "self_hash":
                    payload["motion_event_applied_count"] = 0
                elif case == "budget_rebind":
                    payload["minimum_motion_applied_fraction"] = 0.0
                    payload["maximum_motion_applied_fraction"] = 1.0
                    payload["motion_budget_within_range"] = True
                    self._rehash(payload, "timeline_evidence_content_sha256")
                elif case == "clock":
                    clock = segment["renderer_visible_first_motion_clocks"][0]
                    clock["master_time_seconds"] += 1.0 / 30.0
                    self._rehash(segment, "evidence_content_sha256")
                    self._rehash(payload, "timeline_evidence_content_sha256")
                else:
                    clock = segment["renderer_visible_first_motion_clocks"][0]
                    clock["component"] = "roll_degrees"
                    self._rehash(segment, "evidence_content_sha256")
                    self._rehash(payload, "timeline_evidence_content_sha256")
                bundle["dynamic_crop_motion_v3"] = payload
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_v2_marker_keeps_ordinary_editorial_motion_path_unchanged(self) -> None:
        bundle = self._bundle()
        bundle["dynamic_crop_version"] = 2

        evidence = self._adapt(bundle)

        self.assertEqual(
            evidence.editorial_transforms.zoom_event_times_seconds,
            (1.0 / 30.0, 15.0 + 1.0 / 30.0),
        )

    def test_source_local_cut_authority_and_timeline_clock_are_both_required(self) -> None:
        for case in (
            "missing_timeline_clock",
            "wrong_source_clock",
            "count_only",
            "serialized_domain",
        ):
            with self.subTest(case=case):
                bundle = self._bundle()
                if case == "missing_timeline_clock":
                    timeline = bundle["selected_timeline"].to_dict()  # type: ignore[union-attr]
                    timeline["index_domain"] = "eligible_source_index"
                    del timeline["choices"][0]["internal_scene_cut_master_seconds"]
                    bundle["selected_timeline"] = timeline
                elif case == "wrong_source_clock":
                    bundle["source_scene_cut_times"] = ((27.0,), (), (12.6,))
                elif case == "count_only":
                    bundle["source_scene_cut_times"] = (1, 1, 1)
                else:
                    timeline = bundle["selected_timeline"].to_dict()  # type: ignore[union-attr]
                    timeline["index_domain"] = "request_source_index"
                    bundle["selected_timeline"] = timeline
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_source_timewarp_mapping_tamper_is_rejected(self) -> None:
        bundle = self._bundle()
        timeline = bundle["selected_timeline"].to_dict()  # type: ignore[union-attr]
        timeline["index_domain"] = "eligible_source_index"
        timeline["choices"][0]["internal_scene_cut_master_seconds"] = [102.55]
        bundle["selected_timeline"] = timeline
        with self.assertRaises(GroupBoundaryGrammarAdapterError):
            self._adapt(bundle)

    def test_static_clock_preflight_needs_no_proxy_and_rejects_clock_disagreement(self):
        bundle = self._bundle()
        arguments = {key:bundle[key] for key in ('render_plan', 'source_index_mapping',
            'selected_timeline', 'source_scene_cut_times', 'timeline_review', 'section_plan',
            'editorial_motion')}
        _mapping, native, _sections, _transforms = validate_group_boundary_grammar_plan(**arguments)
        self.assertEqual(len(native), 2)
        timeline = bundle['selected_timeline'].to_dict()
        timeline['index_domain'] = 'eligible_source_index'
        timeline['choices'][0]['internal_scene_cut_master_seconds'][0] += .0000033
        with self.assertRaisesRegex(GroupBoundaryGrammarAdapterError, 'render authority'):
            validate_group_boundary_grammar_plan(**{**arguments, 'selected_timeline':timeline})

    def test_non_identity_source_mapping_is_required_and_fail_closed(self) -> None:
        for case in (
            "missing",
            "duplicate",
            "out_of_range",
            "reverse_tamper",
            "source_identity",
            "domain",
        ):
            with self.subTest(case=case):
                bundle = self._bundle()
                mapping = deepcopy(bundle["source_index_mapping"])
                if case == "missing":
                    bundle["source_index_mapping"] = None
                elif case == "duplicate":
                    mapping["eligible_to_request"] = [2, 2]
                    bundle["source_index_mapping"] = mapping
                elif case == "out_of_range":
                    mapping["eligible_to_request"] = [3, 0]
                    bundle["source_index_mapping"] = mapping
                elif case == "reverse_tamper":
                    mapping["request_to_eligible"] = [1, None, None]
                    bundle["source_index_mapping"] = mapping
                elif case == "source_identity":
                    mapping["eligible_sources"][0]["source"] = "/tmp/wrong.mp4"
                    bundle["source_index_mapping"] = mapping
                else:
                    mapping["payload_index_domains"][
                        "visual_timeline.choices[].source_index"
                    ] = "request_source_index"
                    bundle["source_index_mapping"] = mapping
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_eligible_choice_and_motion_indices_must_translate_to_render_request(self) -> None:
        for case in ("choice", "motion"):
            with self.subTest(case=case):
                bundle = self._bundle()
                if case == "choice":
                    timeline = bundle["selected_timeline"].to_dict()  # type: ignore[union-attr]
                    timeline["index_domain"] = "eligible_source_index"
                    timeline["choices"][0]["source_index"] = 1
                    bundle["selected_timeline"] = timeline
                else:
                    motion = bundle["editorial_motion"].to_dict()  # type: ignore[union-attr]
                    motion["segments"][0]["source_index"] = 1
                    bundle["editorial_motion"] = motion
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_timeline_switch_clock_count_and_rate_tamper_fail_closed(self) -> None:
        mutations = (
            lambda payload: payload.__setitem__("source_switch_count", 3),
            lambda payload: payload.__setitem__("source_switch_rate", 0.75),
            lambda payload: payload["boundaries"][0].__setitem__("at_seconds", 105.2),
            lambda payload: payload["boundaries"][0].__setitem__("source_switch", False),
            lambda payload: payload.__setitem__("internal_scene_cut_count", 3),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                bundle = self._bundle()
                review = deepcopy(bundle["timeline_review"])
                mutate(review)
                bundle["timeline_review"] = review
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_section_partition_gap_offset_and_unknown_role_fail_closed(self) -> None:
        for case in ("gap", "offset", "unknown"):
            with self.subTest(case=case):
                bundle = self._bundle()
                section = bundle["section_plan"].to_dict()  # type: ignore[union-attr]
                if case == "gap":
                    section["sections"][1]["start_seconds"] = 110.1
                elif case == "offset":
                    section["start_seconds"] = 99.0
                else:
                    section["sections"][0]["role"] = "guessed_hook"
                bundle["section_plan"] = section
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_proxy_boundary_domain_and_frame_clock_tamper_fail_closed(self) -> None:
        for key, value in (
            ("review_domain", NATIVE_SOURCE_CUT_DOMAIN),
            ("cut_frame_index", 151),
            ("transition_reason", "tampered"),
            ("source_clock_contiguous", True),
        ):
            with self.subTest(key=key):
                bundle = self._bundle()
                proxy = deepcopy(bundle["rendered_proxy_review"].payload)  # type: ignore[union-attr]
                proxy["boundaries"][0][key] = value
                bundle["rendered_proxy_review"] = proxy
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_every_proxy_domain_summary_is_recomputed(self) -> None:
        for domain in (
            INTRODUCED_CROSS_SOURCE_DOMAIN,
            TIER_D_CONTINUITY_DOMAIN,
            NATIVE_SOURCE_CUT_DOMAIN,
        ):
            with self.subTest(domain=domain):
                bundle = self._bundle()
                proxy = deepcopy(bundle["rendered_proxy_review"].payload)  # type: ignore[union-attr]
                proxy["boundary_domains"][domain]["score"]["p10"] = (
                    proxy["boundary_domains"][domain]["score"]["p10"] or 0.0
                ) + 1.0
                bundle["rendered_proxy_review"] = proxy
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_introduced_issue_fraction_p10_and_top_count_tamper_fail_closed(self) -> None:
        for case in ("fraction", "distribution", "top_count", "issue_indices"):
            with self.subTest(case=case):
                bundle = self._bundle()
                proxy = deepcopy(bundle["rendered_proxy_review"].payload)  # type: ignore[union-attr]
                introduced = proxy["boundary_domains"][INTRODUCED_CROSS_SOURCE_DOMAIN]
                if case == "fraction":
                    introduced["issue_fraction"] = 0.0
                elif case == "distribution":
                    introduced["score"]["distribution"]["median"] = 82.0
                elif case == "top_count":
                    proxy["introduced_cross_source_count"] = 1
                else:
                    proxy["introduced_cross_source_boundary_issue_indices"] = []
                bundle["rendered_proxy_review"] = proxy
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_transform_keyframe_and_event_counts_are_recomputed(self) -> None:
        for case in ("keyframe", "event_count", "applied", "missing_segment"):
            with self.subTest(case=case):
                bundle = self._bundle()
                motion = bundle["editorial_motion"].to_dict()  # type: ignore[union-attr]
                if case == "keyframe":
                    motion["segments"][0]["keyframes"][1]["zoom"] = 1.06
                elif case == "event_count":
                    motion["zoom_event_count"] = 1
                elif case == "applied":
                    motion["segments"][0]["applied"] = False
                else:
                    motion["segments"].pop()
                bundle["editorial_motion"] = motion
                static_arguments = {key: bundle[key] for key in (
                    'render_plan', 'source_index_mapping', 'selected_timeline',
                    'source_scene_cut_times', 'timeline_review', 'section_plan',
                    'editorial_motion')}
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    validate_group_boundary_grammar_plan(**static_arguments)
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_stage_edit_dynamic_tracks_are_authoritative_when_motion_plan_is_none(self) -> None:
        bundle = self._bundle()
        bundle["editorial_motion"] = None
        evidence = self._adapt(bundle)
        self.assertEqual(evidence.editorial_transforms.applied_segment_count, 2)
        self.assertEqual(
            evidence.editorial_transforms.zoom_event_times_seconds,
            (1.0 / 30.0, 15.0 + 1.0 / 30.0),
        )
        self.assertEqual(
            evidence.editorial_transforms.pan_event_times_seconds,
            (15.0 + 1.0 / 30.0,),
        )
        self.assertEqual(
            evidence.editorial_transforms.roll_event_times_seconds,
            (15.0 + 1.0 / 30.0,),
        )

    def test_motion_none_without_final_render_tracks_is_unavailable(self) -> None:
        bundle = self._bundle()
        plan = bundle["render_plan"]
        bundle["render_plan"] = replace(
            plan,
            segments=tuple(
                replace(segment, editorial_transform=None)
                for segment in plan.segments
            ),
        )
        bundle["editorial_motion"] = None
        with self.assertRaises(GroupBoundaryGrammarAdapterError):
            self._adapt(bundle)

    def test_cut_time_reframe_is_excluded_while_interior_pulse_remains_exact(
        self,
    ) -> None:
        bundle = self._bundle()
        plan = bundle["render_plan"]
        multi_pulse = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.04),
                EditorialTransformKeyframe(1.0, 1.0),
                EditorialTransformKeyframe(2.0, 1.05),
                EditorialTransformKeyframe(3.0, 1.0),
            ),
            reason="synthetic_stage_edit_multi_pulse",
        )
        bundle["render_plan"] = replace(
            plan,
            segments=(
                replace(plan.segments[0], editorial_transform=multi_pulse),
                *plan.segments[1:],
            ),
        )
        bundle["editorial_motion"] = None
        evidence = self._adapt(bundle)
        self.assertEqual(
            evidence.editorial_transforms.zoom_event_times_seconds,
            (1.0 + 1.0 / 30.0, 15.0 + 1.0 / 30.0),
        )
        self.assertEqual(
            len(evidence.editorial_transforms.zoom_event_times_seconds),
            evidence.editorial_transforms.applied_segment_count,
        )

    def test_neutral_to_active_ramp_event_uses_first_visible_output_frame(self) -> None:
        bundle = self._bundle()
        plan = bundle["render_plan"]
        ramp = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(0.0, 1.0),
                EditorialTransformKeyframe(1.0, 1.05),
                EditorialTransformKeyframe(2.0, 1.0),
            ),
            reason="synthetic_frame_sampled_ramp",
        )
        bundle["render_plan"] = replace(
            plan,
            segments=(
                replace(plan.segments[0], editorial_transform=ramp),
                *(
                    replace(segment, editorial_transform=None)
                    for segment in plan.segments[1:]
                ),
            ),
        )
        bundle["editorial_motion"] = None

        evidence = self._adapt(bundle)

        self.assertEqual(
            evidence.editorial_transforms.zoom_event_times_seconds,
            (1.0 / plan.fps,),
        )
        self.assertNotEqual(
            evidence.editorial_transforms.zoom_event_times_seconds,
            (1.0,),
        )

    def test_keyframes_after_last_output_sample_do_not_count_as_visible_events(
        self,
    ) -> None:
        for active_time in (4.99, 5.0):
            with self.subTest(active_time=active_time):
                bundle = self._bundle()
                plan = bundle["render_plan"]
                invisible_tail = EditorialTransformTrack(
                    (
                        EditorialTransformKeyframe(0.0, 1.0),
                        EditorialTransformKeyframe(4.98, 1.0),
                        EditorialTransformKeyframe(active_time, 1.05),
                    ),
                    reason="synthetic_unrendered_tail_keyframe",
                )
                bundle["render_plan"] = replace(
                    plan,
                    segments=(
                        replace(
                            plan.segments[0],
                            editorial_transform=invisible_tail,
                        ),
                        *(
                            replace(segment, editorial_transform=None)
                            for segment in plan.segments[1:]
                        ),
                    ),
                )
                bundle["editorial_motion"] = None

                evidence = self._adapt(bundle)

                self.assertEqual(
                    evidence.editorial_transforms.zoom_event_times_seconds,
                    (),
                )
                self.assertEqual(
                    evidence.editorial_transforms.applied_segment_count,
                    0,
                )

    def test_missing_nonfinite_and_inspection_only_payloads_fail_closed(self) -> None:
        for case in ("missing", "nonfinite", "inspection"):
            with self.subTest(case=case):
                bundle = self._bundle()
                if case == "missing":
                    review = deepcopy(bundle["timeline_review"])
                    del review["boundaries"]
                    bundle["timeline_review"] = review
                elif case == "nonfinite":
                    review = deepcopy(bundle["timeline_review"])
                    review["source_switches_per_minute"] = float("nan")
                    bundle["timeline_review"] = review
                else:
                    proxy = deepcopy(bundle["rendered_proxy_review"].payload)  # type: ignore[union-attr]
                    proxy["inspection_only"] = True
                    bundle["rendered_proxy_review"] = proxy
                with self.assertRaises(GroupBoundaryGrammarAdapterError):
                    self._adapt(bundle)

    def test_generated_evidence_hash_tamper_is_rejected(self) -> None:
        evidence = self._adapt(self._bundle())
        payload = evidence.to_dict()
        payload["native_internal_cut_times_seconds"][0] = 2.6
        with self.assertRaises(MontageGrammarEvidenceError):
            ProfessionalBoundaryGrammarEvidence.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
