from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from crosscut.direct_member_boundary_proxy import (
    DirectMemberBoundaryProxyEvidence,
    build_direct_member_boundary_proxy_signature,
    direct_member_boundary_proxy_evidence_from_preflight,
    extract_direct_member_boundary_proxy_plan,
    load_direct_member_boundary_proxy_cache,
    persist_direct_member_boundary_proxy_cache,
)
from crosscut.direct_member_rendered_proxy import (
    DirectMemberRenderedProxyPreflight,
)
from crosscut.ive import AspectRatio
from crosscut.render import (
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    SourceLookAdjustment,
    TransitionKind,
)


_DENSE_BINDING = "a" * 64
_AUTHORITY_BINDING = "b" * 64


def _plan() -> RenderPlan:
    camera = EditorialTransformTrack(
        (
            EditorialTransformKeyframe(0.0, 1.0, 0.5, 0.5),
            EditorialTransformKeyframe(0.8, 1.08, 0.51, 0.49),
        )
    )
    return RenderPlan(
        sources=(Path("one.mp4"), Path("two.mp4"), Path("three.mp4")),
        segments=(
            RenderSegment(0, 10.0, 1.0, 0.0),
            RenderSegment(
                1,
                11.0,
                1.2,
                1.0,
                editorial_transform=camera,
                look_adjustment=SourceLookAdjustment(brightness=0.02),
            ),
            RenderSegment(2, 12.0, 1.4, 2.2),
        ),
        reference_source=0,
        aspect_ratio=AspectRatio.PORTRAIT,
        output=Path("full.mp4"),
        master_audio_start=8.0,
        width=1080,
        height=1920,
        fps=10.0,
        transitions=(
            RenderTransition(),
            RenderTransition(
                TransitionKind.FACE_DISSOLVE,
                3,
                DIRECT_MEMBER_POSE_DISSOLVE_REASON,
                transform_overlap_authority=(
                    MEMBER_OUTPUT_VERIFIED_TRANSFORM_OVERLAP_AUTHORITY
                ),
            ),
        ),
        member_pose_dissolve_requested=True,
        member_pose_dissolve_effective=True,
        member_pose_dissolve_count=1,
    )


def _preflight() -> DirectMemberRenderedProxyPreflight:
    return DirectMemberRenderedProxyPreflight(
        {
            "passed": False,
            "direct_member_rendered_continuity_passed": True,
            "member_output_quality_passed": False,
            "member_output_quality_failure_reasons": [
                "member_proxy_perceptual_source_admission"
            ],
            "rendered_continuity": {
                "passed": True,
                "failure_reasons": [],
                "metrics": {
                    "attention_position_mismatch_p95": 0.08,
                    "attention_scale_mismatch_p95": 0.06,
                    "pose_mismatch_p95": 0.14,
                },
                "boundaries": [
                    {
                        "endpoint_steps": [
                            {
                                "attention_position_step": 0.09,
                                "attention_scale_step": 0.07,
                                "pose_step": 0.12,
                            }
                        ]
                    }
                ],
            },
            "rendered_proxy_review": {
                "weighted_score": 61.0,
                "introduced_cross_source_technical_issue_reasons": [],
                "source_quality": {"median_span": 0.18},
                "perceptual_source_admission": {
                    "source_median_consistency": {
                        "sharpness_log_median_span": 0.30,
                        "luma_median_span": 0.20,
                        "chroma_median_span": 0.08,
                    }
                },
            },
        }
    )


class DirectMemberBoundaryProxyTests(unittest.TestCase):
    def test_window_preserves_pixel_decisions_and_rebases_master_only(self) -> None:
        original = _plan()
        window = extract_direct_member_boundary_proxy_plan(
            original,
            1,
            Path("boundary.mp4"),
        )

        self.assertEqual(window.sources, original.sources)
        self.assertEqual(window.segments[0].source_index, 1)
        self.assertEqual(window.segments[1].source_index, 2)
        self.assertEqual(window.segments[0].source_start, 11.0)
        self.assertEqual(window.segments[1].source_start, 12.0)
        self.assertEqual(window.segments[0].master_start, 0.0)
        self.assertEqual(window.segments[1].master_start, 1.2)
        self.assertEqual(
            window.segments[0].editorial_transform,
            original.segments[1].editorial_transform,
        )
        self.assertEqual(
            window.segments[0].look_adjustment,
            original.segments[1].look_adjustment,
        )
        self.assertEqual(window.transitions, (original.transitions[1],))
        self.assertEqual(window.master_audio_start, 9.0)
        self.assertTrue(window.member_pose_dissolve_effective)
        self.assertEqual(window.member_pose_dissolve_count, 1)

    def test_signature_changes_for_any_pixel_or_authority_change(self) -> None:
        window = extract_direct_member_boundary_proxy_plan(
            _plan(),
            1,
            Path("boundary.mp4"),
        )
        original = build_direct_member_boundary_proxy_signature(
            window,
            dense_binding_sha256=_DENSE_BINDING,
            authority_binding_sha256=_AUTHORITY_BINDING,
        )
        changed_segment = replace(
            window.segments[0],
            look_adjustment=SourceLookAdjustment(brightness=0.03),
        )
        changed_plan = replace(
            window,
            segments=(changed_segment, window.segments[1]),
        )
        changed_pixels = build_direct_member_boundary_proxy_signature(
            changed_plan,
            dense_binding_sha256=_DENSE_BINDING,
            authority_binding_sha256=_AUTHORITY_BINDING,
        )
        changed_authority = build_direct_member_boundary_proxy_signature(
            window,
            dense_binding_sha256=_DENSE_BINDING,
            authority_binding_sha256="c" * 64,
        )
        relocated_output = build_direct_member_boundary_proxy_signature(
            replace(window, output=Path("another-temporary-name.mp4")),
            dense_binding_sha256=_DENSE_BINDING,
            authority_binding_sha256=_AUTHORITY_BINDING,
        )

        self.assertNotEqual(
            original.signature_sha256,
            changed_pixels.signature_sha256,
        )
        self.assertNotEqual(
            original.signature_sha256,
            changed_authority.signature_sha256,
        )
        self.assertEqual(
            original.signature_sha256,
            relocated_output.signature_sha256,
        )

    def test_compact_evidence_and_cache_round_trip_are_binding_safe(self) -> None:
        window = extract_direct_member_boundary_proxy_plan(
            _plan(),
            1,
            Path("boundary.mp4"),
        )
        signature = build_direct_member_boundary_proxy_signature(
            window,
            dense_binding_sha256=_DENSE_BINDING,
            authority_binding_sha256=_AUTHORITY_BINDING,
        )
        evidence = direct_member_boundary_proxy_evidence_from_preflight(
            signature,
            _preflight(),
        )

        self.assertIsInstance(evidence, DirectMemberBoundaryProxyEvidence)
        self.assertTrue(evidence.continuity_passed)
        self.assertFalse(evidence.member_quality_passed)
        self.assertFalse(evidence.passed)
        self.assertGreater(evidence.finite_solver_penalty, 0.0)
        self.assertNotIn("boundaries", evidence.to_dict())
        with TemporaryDirectory() as directory:
            cache_directory = Path(directory)
            persisted = persist_direct_member_boundary_proxy_cache(
                cache_directory,
                (evidence,),
            )
            loaded, diagnostics = load_direct_member_boundary_proxy_cache(
                cache_directory,
                dense_binding_sha256=_DENSE_BINDING,
            )
            foreign, foreign_diagnostics = (
                load_direct_member_boundary_proxy_cache(
                    cache_directory,
                    dense_binding_sha256="d" * 64,
                )
            )

        self.assertTrue(persisted["persisted"])
        self.assertEqual(diagnostics["status"], "cache_loaded")
        self.assertEqual(loaded[signature.signature_sha256], evidence)
        self.assertEqual(foreign, {})
        self.assertEqual(foreign_diagnostics["foreign_binding_entry_count"], 1)

    def test_malformed_cache_entry_is_ignored_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            cache_directory = Path(directory)
            path = cache_directory / "direct-member-boundary-proxy-cache-v1.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "contract": "direct_member_exact_boundary_proxy_cache_v1",
                        "entries": [{"passed": True}],
                    }
                ),
                encoding="utf-8",
            )
            loaded, diagnostics = load_direct_member_boundary_proxy_cache(
                cache_directory,
                dense_binding_sha256=_DENSE_BINDING,
            )

        self.assertEqual(loaded, {})
        self.assertEqual(diagnostics["invalid_entry_count"], 1)

    def test_tampered_cache_evidence_is_ignored_fail_closed(self) -> None:
        window = extract_direct_member_boundary_proxy_plan(
            _plan(),
            1,
            Path("boundary.mp4"),
        )
        signature = build_direct_member_boundary_proxy_signature(
            window,
            dense_binding_sha256=_DENSE_BINDING,
            authority_binding_sha256=_AUTHORITY_BINDING,
        )
        evidence = direct_member_boundary_proxy_evidence_from_preflight(
            signature,
            _preflight(),
        )
        with TemporaryDirectory() as directory:
            cache_directory = Path(directory)
            persist_direct_member_boundary_proxy_cache(
                cache_directory,
                (evidence,),
            )
            path = (
                cache_directory
                / "direct-member-boundary-proxy-cache-v1.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["entries"][0]["finite_solver_penalty"] = 0.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, diagnostics = load_direct_member_boundary_proxy_cache(
                cache_directory,
                dense_binding_sha256=_DENSE_BINDING,
            )

        self.assertEqual(loaded, {})
        self.assertEqual(diagnostics["invalid_entry_count"], 1)


if __name__ == "__main__":
    unittest.main()
