from __future__ import annotations

from pathlib import Path
import unittest

from crosscut.pipeline import (
    ELIGIBLE_SOURCE_INDEX_DOMAIN,
    REQUEST_SOURCE_INDEX_DOMAIN,
    _alignment_eligibility,
    _capabilities_payload,
    _common_master_range,
    _member_anchor_trim,
    _member_render_selection_basis,
    _safe_slug,
    _source_index_mapping,
    _with_index_domain,
)
from crosscut.audio import AlignmentResult, SourceAlignment
from crosscut.ive import EditMode, IVEMember
from crosscut.media import MediaInfo
from crosscut.render import RENDER_PLAN_SOURCE_INDEX_DOMAIN
from crosscut.tracking import (
    MemberTrackObservation,
    MemberTrackingResult,
    TrackingState,
)


def _media(path: str, duration: float) -> MediaInfo:
    return MediaInfo(
        path=path,
        format_name="mov,mp4",
        duration_seconds=duration,
        start_time_seconds=0.0,
        size_bytes=1,
        bit_rate=1,
        audio_streams=(),
        video_streams=(),
        tags={},
    )


def _tracking(source: str, source_local_time: float) -> MemberTrackingResult:
    return MemberTrackingResult(
        member=IVEMember.GAEUL,
        source_id=source,
        fps=30.0,
        backend_name="scripted",
        observations=(
            MemberTrackObservation(
                frame_index=round(source_local_time * 30.0),
                timestamp_seconds=source_local_time,
                bbox_xyxy=(0.1, 0.2, 0.3, 0.8),
                confidence=1.0,
                state=TrackingState.TRACKED,
                member=IVEMember.GAEUL,
                source_id=source,
            ),
        ),
        terminal_failure=None,
    )


class PipelineTests(unittest.TestCase):
    def test_source_index_mapping_documents_gap_after_excluded_source(self) -> None:
        sources = tuple(Path(f"/tmp/source-{index}.mp4") for index in range(4))

        mapping = _source_index_mapping(sources, (0, 2, 3))

        self.assertEqual(mapping["request_index_domain"], REQUEST_SOURCE_INDEX_DOMAIN)
        self.assertEqual(mapping["eligible_index_domain"], ELIGIBLE_SOURCE_INDEX_DOMAIN)
        self.assertEqual(
            mapping["render_plan_index_domain"],
            RENDER_PLAN_SOURCE_INDEX_DOMAIN,
        )
        self.assertEqual(
            mapping["domain_equivalences"],
            {RENDER_PLAN_SOURCE_INDEX_DOMAIN: REQUEST_SOURCE_INDEX_DOMAIN},
        )
        self.assertEqual(mapping["eligible_to_request"], [0, 2, 3])
        self.assertEqual(mapping["request_to_eligible"], [0, None, 1, 2])
        self.assertEqual(
            [
                (item["eligible_source_index"], item["request_source_index"])
                for item in mapping["eligible_sources"]
            ],
            [(0, 0), (1, 2), (2, 3)],
        )
        domains = mapping["payload_index_domains"]
        for payload_path in (
            "request.sources[]",
            "request.reference_source",
            "request.member_anchors[].source_index",
            "media[]",
            "alignment.alignments[]",
            "alignment_eligibility[].source_index",
            "scene_analysis[].source_index",
            "member_tracking[].source_index",
            "member_anchor_trim.anchors[].source_index",
            "member_anchor_trim.limiting_source_indices[]",
        ):
            self.assertEqual(domains[payload_path], REQUEST_SOURCE_INDEX_DOMAIN)
        for payload_path in (
            "render_plan.sources[]",
            "render_plan.reference_source",
            "render_plan.segments[].source_index",
        ):
            self.assertEqual(
                domains[payload_path],
                RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            )
        for payload_path in (
            "group_draft.segments[].source_index",
            "group_draft.audio_offsets[]",
            "group_draft.sync_scores[]",
            "visual_timeline.choices[].source_index",
            "face_match.segments[].source_index",
            "member_draft.segments[].source_index",
            "member_draft.audio_offsets[]",
            "member_draft.group_wide_fallback_source",
            "member_draft.selection_diagnostics.safe_source_indices[]",
            "member_draft.selection_diagnostics.quality_eligible_source_indices[]",
            "member_draft.selection_diagnostics.selected_source_indices[]",
            "member_draft.selection_diagnostics.no_safe_candidate_source_indices[]",
            "member_draft.selection_diagnostics.quality_guard_excluded_source_indices[]",
            "member_draft.selection_diagnostics.unused_quality_source_indices[]",
            "member_draft.selection_diagnostics.sources[].source_index",
        ):
            self.assertEqual(domains[payload_path], ELIGIBLE_SOURCE_INDEX_DOMAIN)
        for nonexistent_path in (
            "media[].position",
            "alignment.alignments[].position",
            "group_draft.audio_offsets[].position",
            "member_draft.audio_offsets[].position",
        ):
            self.assertNotIn(nonexistent_path, domains)

    def test_index_domain_helpers_reject_ambiguous_mappings(self) -> None:
        sources = (Path("/tmp/a.mp4"), Path("/tmp/b.mp4"))
        with self.assertRaises(ValueError):
            _source_index_mapping(sources, (0, 0))
        with self.assertRaises(ValueError):
            _source_index_mapping(sources, (0, 2))
        with self.assertRaises(ValueError):
            _with_index_domain({}, "unknown")

        original = {"segments": []}
        annotated = _with_index_domain(original, ELIGIBLE_SOURCE_INDEX_DOMAIN)
        self.assertNotIn("index_domain", original)
        self.assertEqual(annotated["index_domain"], ELIGIBLE_SOURCE_INDEX_DOMAIN)

    def test_alignment_eligibility_uses_request_index_domain(self) -> None:
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 0.0, 0.2, 1, 1.0),
            ),
        )

        payload = _alignment_eligibility(alignment)

        self.assertEqual([item["source_index"] for item in payload], [0, 1])
        self.assertTrue(
            all(
                item["index_domain"] == REQUEST_SOURCE_INDEX_DOMAIN
                for item in payload
            )
        )
        self.assertTrue(payload[0]["render_eligible"])
        self.assertFalse(payload[1]["render_eligible"])

    def test_common_range_respects_positive_and_negative_offsets(self) -> None:
        media = [_media("a", 10.0), _media("b", 12.0), _media("c", 9.0)]
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 2.0, 0.9, 1, 1.0),
                SourceAlignment("c", False, 0, -1.0, 0.9, 1, 1.0),
            ),
        )
        self.assertEqual(_common_master_range(media, alignment), (1.0, 10.0))

    def test_slug_keeps_korean_song_names(self) -> None:
        self.assertEqual(_safe_slug(" 해야 (HEYA) "), "해야-HEYA")

    def test_member_render_selection_basis_includes_global_path_policy(self) -> None:
        policy = (
            "safe_gate_then_local_best_quality_guard_then_"
            "maximum_distinct_balanced_global_path_v2"
        )

        self.assertEqual(
            _member_render_selection_basis(policy),
            "anchored_target_tracking_virtual_camera_" + policy,
        )
        with self.assertRaises(ValueError):
            _member_render_selection_basis("  ")

    def test_capabilities_describe_anchored_tracking_without_identity_claim(self) -> None:
        payload = _capabilities_payload(
            group_ready=True,
            edit_mode=EditMode.MEMBER,
            member_draft=None,
            rendered_video_path=None,
        )

        self.assertFalse(payload["member_tracking_ready"])
        self.assertIn("앵커", payload["member_tracking_note"])
        self.assertNotIn("member_identity_ready", payload)
        self.assertNotIn("member_identity_note", payload)

    def test_member_start_trims_to_latest_canonical_anchor(self) -> None:
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 2.0, 0.9, 1, 1.0),
                SourceAlignment("c", False, 0, -1.0, 0.9, 1, 1.0),
            ),
        )

        effective_start, payload = _member_anchor_trim(
            common_master_start=1.0,
            common_master_end=10.0,
            alignment=alignment,
            eligible_indices=[0, 1, 2],
            tracking_results={
                0: _tracking("a", 2.0),
                1: _tracking("b", 4.5),
                2: _tracking("c", 0.5),
            },
        )

        self.assertAlmostEqual(effective_start, 2.5)
        self.assertEqual(payload["index_domain"], REQUEST_SOURCE_INDEX_DOMAIN)
        self.assertTrue(payload["applied"])
        self.assertFalse(payload["reverse_tracking_used"])
        self.assertAlmostEqual(payload["trimmed_seconds"], 1.5)
        self.assertEqual(payload["limiting_source_indices"], [1])
        self.assertEqual(
            [item["canonical_anchor_time_seconds"] for item in payload["anchors"]],
            [2.0, 2.5, 1.5],
        )

    def test_member_start_does_not_expand_an_already_later_common_range(self) -> None:
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 1.0, 0.9, 1, 1.0),
            ),
        )

        effective_start, payload = _member_anchor_trim(
            common_master_start=3.0,
            common_master_end=10.0,
            alignment=alignment,
            eligible_indices=[0, 1],
            tracking_results={
                0: _tracking("a", 1.0),
                1: _tracking("b", 2.0),
            },
        )

        self.assertEqual(effective_start, 3.0)
        self.assertFalse(payload["applied"])
        self.assertEqual(payload["trimmed_seconds"], 0.0)
        self.assertTrue(payload["usable_for_render"])

    def test_member_start_marks_too_short_post_anchor_range_unusable(self) -> None:
        alignment = AlignmentResult(
            reference="a",
            sample_rate=4_000,
            alignments=(
                SourceAlignment("a", True, 0, 0.0, 1.0, 1, 1.0),
                SourceAlignment("b", False, 0, 0.0, 0.9, 1, 1.0),
            ),
        )

        effective_start, payload = _member_anchor_trim(
            common_master_start=0.0,
            common_master_end=10.0,
            alignment=alignment,
            eligible_indices=[0, 1],
            tracking_results={
                0: _tracking("a", 1.0),
                1: _tracking("b", 9.95),
            },
        )

        self.assertEqual(effective_start, 9.95)
        self.assertFalse(payload["usable_for_render"])
        self.assertAlmostEqual(payload["usable_duration_seconds"], 0.05)


if __name__ == "__main__":
    unittest.main()
