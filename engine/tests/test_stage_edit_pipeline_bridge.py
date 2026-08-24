from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut.ive import AspectRatio, EditMode, SourceMode
from crosscut.pipeline import (
    _discover_automatic_stage_edit_production_bundle,
    _load_stage_edit_generation_preflight,
    _materialize_registered_stage_edit_preflight,
    analyze,
)
from crosscut.render import RenderSegment
from crosscut.request import AnalysisRequest
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit import StageEdit, StageShot
from crosscut.stage_edit_feature_materializer import StageEditFeatureSnapshot
from crosscut.stage_edit_pipeline_bridge import (
    StageEditPipelineBridgeError,
    StageEditPipelineBridgeInputError,
    StageEditPipelineBridgeNotReadyError,
    StageEditPipelineBridgePreflight,
    preflight_stage_edit_pipeline_bridge,
    run_stage_edit_pipeline_bridge,
)
from crosscut.stage_edit_semimarkov_beam import StageEditSemiMarkovConfig
from crosscut.stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditReadinessReport,
    StageEditSequence,
    StageEditToken,
)


def _candidate(source_id: str, value: float) -> SourceCandidateToken:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    return SourceCandidateToken(
        source_id=source_id,
        features=(value,) * schema.raw_source_dimension,
        missing_mask=(False,) * schema.raw_source_dimension,
        available=True,
        hard_gate_passed=True,
    )


def _snapshot() -> StageEditFeatureSnapshot:
    schema = DEFAULT_STAGE_EDIT_SCHEMA
    sequence = StageEditSequence(
        "ive:bridge",
        (
            StageEditToken(
                master_time_seconds=0.0,
                music_features=(0.1,) * len(schema.music_feature_names),
                section_features=(0.2,) * len(schema.section_feature_names),
                source_candidates=(
                    _candidate("source-a", 0.3),
                    _candidate("source-b", 0.4),
                ),
                recent_edit_features=(0.0,)
                * len(schema.recent_edit_feature_names),
                style_features=(0.0,) * len(schema.style_feature_names),
                missing_mask=(False,) * schema.raw_context_dimension,
            ),
        ),
    )
    return StageEditFeatureSnapshot.build(sequence)


def _ready(checkpoint: Path) -> StageEditReadinessReport:
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=True,
        status="production_checkpoint_ready",
        reason="test production evidence",
        mlx_version="test",
        checkpoint_path=str(checkpoint.resolve()),
        training_steps=10,
    )


def _not_ready(checkpoint: Path) -> StageEditReadinessReport:
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=False,
        status="production_training_report_missing",
        reason="test report missing",
        mlx_version="test",
        checkpoint_path=str(checkpoint.resolve()),
        training_steps=10,
    )


def _section_plan() -> SectionPlan:
    return SectionPlan(
        (
            SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=0.0,
                end_seconds=2.0,
                energy_score=0.8,
                confidence=0.9,
                phrase_count=1,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=20.0,
                camera_motion_budget=0.5,
                target_roll_events_per_minute=0.0,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _edit(sources: tuple[Path, ...]) -> StageEdit:
    section_plan = _section_plan()
    return StageEdit(
        sources=sources,
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        width=1920,
        height=1080,
        fps=30.0,
        master_audio_start=0.0,
        section_plan=section_plan,
        shots=(
            StageShot(
                shot_index=0,
                section_index=0,
                segment=RenderSegment(0, 0.0, 2.0, 0.0),
                decision_id="test-decision",
                decision_confidence=0.8,
            ),
        ),
        boundaries=(),
        output=Path("/tmp/stage-edit-bridge.mp4"),
    )


@dataclass(frozen=True)
class _FakeHeadResult:
    frames: tuple[object, ...] = (object(),)

    def to_dict(self):
        return {"frame_count": len(self.frames)}


@dataclass(frozen=True)
class _FakeDecodeResult:
    timelines: tuple[object, ...]
    audit: object

    def to_dict(self):
        return {
            "audit": {"request_fulfilled": self.audit.request_fulfilled},
            "timeline_count": len(self.timelines),
        }


class StageEditPipelineBridgeTests(unittest.TestCase):
    def test_automatic_group_strict_boundary_rejects_legacy_identity_default(self):
        snapshot = _snapshot()
        checkpoint = Path("/tmp/stageedit-bridge-ready").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=snapshot,
            readiness=_ready(checkpoint),
        )
        with self.assertRaisesRegex(
            StageEditPipelineBridgeInputError,
            "automatic GROUP.*verified source derivative metadata",
        ):
            run_stage_edit_pipeline_bridge(
                preflight=preflight,
                sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
                source_durations_seconds=(10.0, 10.0),
                source_start_offsets_seconds=(0.0, 0.0),
                internal_cut_times_seconds=((), ()),
                source_timewarps=(None, None),
                source_quality_log_priors=(0.0, 0.0),
                require_verified_source_derivative_metadata=True,
                section_plan=_section_plan(),
                output=Path("/tmp/out.mp4"),
                reference_source=0,
                config=StageEditSemiMarkovConfig(minimum_unique_sources=1),
            )

    def test_preflight_requires_exact_snapshot_sources_and_ready_v2_checkpoint(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            snapshot_path = snapshot.write_json(root / "snapshot.json")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "dynamic-crop-v2-metadata.json").write_text(
                "{}", encoding="utf-8"
            )
            with patch(
                "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                return_value=_ready(checkpoint),
            ):
                preflight = preflight_stage_edit_pipeline_bridge(
                    feature_snapshot_path=snapshot_path,
                    checkpoint_directory=checkpoint,
                    source_ids=("source-a", "source-b"),
                )
                self.assertEqual(snapshot.snapshot_sha256, preflight.snapshot.snapshot_sha256)

                with self.assertRaisesRegex(
                    StageEditPipelineBridgeInputError, "exactly match"
                ):
                    preflight_stage_edit_pipeline_bridge(
                        feature_snapshot_path=snapshot_path,
                        checkpoint_directory=checkpoint,
                        source_ids=("source-a", "source-c"),
                    )
            with patch(
                "crosscut.stage_edit_pipeline_bridge.stage_edit_readiness_report",
                return_value=_not_ready(checkpoint),
            ):
                with self.assertRaisesRegex(
                    StageEditPipelineBridgeNotReadyError, "not production-ready"
                ):
                    preflight_stage_edit_pipeline_bridge(
                        feature_snapshot_path=snapshot_path,
                        checkpoint_directory=checkpoint,
                        source_ids=("source-a", "source-b"),
                    )

    def test_generation_retains_parallel_candidates_and_never_selects_baseline(self):
        snapshot = _snapshot()
        checkpoint = Path("/tmp/stageedit-bridge-ready").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=snapshot,
            readiness=_ready(checkpoint),
        )
        sources = (
            Path("/tmp/stageedit-a.mp4"),
            Path("/tmp/stageedit-b.mp4"),
        )
        edit = _edit(sources)
        decision = SimpleNamespace(
            source_index=0,
            master_start_seconds=0.0,
            duration_seconds=2.0,
            editorial_transform=None,
        )
        timelines = tuple(
            SimpleNamespace(
                rank=rank,
                score=4.2 - rank * 0.01,
                decisions=(decision,),
                edit=edit,
            )
            for rank in range(1, 9)
        )
        decoded = _FakeDecodeResult(
            timelines=timelines,
            audit=SimpleNamespace(request_fulfilled=True),
        )
        config = StageEditSemiMarkovConfig(
            minimum_unique_sources=1,
        )
        learned_sections = SimpleNamespace(
            section_plan=_section_plan(),
            to_dict=lambda: {"contract": "test-learned-section-director"},
        )
        with (
            patch(
                "crosscut.stage_edit_pipeline_bridge.materialize_stage_edit_head_frames_v2",
                return_value=_FakeHeadResult(),
            ) as materialize,
            patch(
                "crosscut.stage_edit_pipeline_bridge.decode_stage_edit_semimarkov_beam",
                return_value=decoded,
            ) as decode,
            patch(
                "crosscut.stage_edit_pipeline_bridge.build_learned_section_plan_from_checkpoint",
                return_value=learned_sections,
            ) as direct_sections,
        ):
            result = run_stage_edit_pipeline_bridge(
                preflight=preflight,
                sources=sources,
                source_durations_seconds=(10.0, 10.0),
                source_start_offsets_seconds=(0.0, 0.0),
                internal_cut_times_seconds=((), ()),
                source_timewarps=(None, None),
                source_quality_log_priors=(0.0, 0.0),
                section_plan=_section_plan(),
                output=Path("/tmp/stageedit-retained.mp4"),
                reference_source=0,
                fps=30.0,
                config=config,
            )

        materialize.assert_called_once()
        direct_sections.assert_called_once()
        decode.assert_called_once()
        self.assertEqual(8, len(result.retained_candidates))
        self.assertFalse(result.selection_policy["selected_for_render"])
        self.assertFalse(result.selection_policy["visual_baseline_replaced"])
        self.assertEqual(
            "legacy_identity_default_not_automatic_group_safe",
            result.selection_policy["source_derivative_policy"],
        )
        history = result.retained_candidates[0].edit.candidate_history[-1]
        self.assertFalse(history["visual_baseline_replaced"])
        self.assertFalse(history["selected_for_render"])

    def test_requested_decode_shortfall_fails_without_visual_fallback(self):
        snapshot = _snapshot()
        checkpoint = Path("/tmp/stageedit-bridge-ready").resolve()
        preflight = StageEditPipelineBridgePreflight(
            snapshot_path=Path("/tmp/stageedit-feature.json").resolve(),
            checkpoint_directory=checkpoint,
            source_ids=("source-a", "source-b"),
            snapshot=snapshot,
            readiness=_ready(checkpoint),
        )
        decoded = _FakeDecodeResult(
            timelines=(),
            audit=SimpleNamespace(request_fulfilled=False),
        )
        learned_sections = SimpleNamespace(
            section_plan=_section_plan(),
            to_dict=lambda: {"contract": "test-learned-section-director"},
        )
        with (
            patch(
                "crosscut.stage_edit_pipeline_bridge.materialize_stage_edit_head_frames_v2",
                return_value=_FakeHeadResult(),
            ),
            patch(
                "crosscut.stage_edit_pipeline_bridge.decode_stage_edit_semimarkov_beam",
                return_value=decoded,
            ),
            patch(
                "crosscut.stage_edit_pipeline_bridge.build_learned_section_plan_from_checkpoint",
                return_value=learned_sections,
            ),
        ):
            with self.assertRaisesRegex(
                StageEditPipelineBridgeError, "silent"
            ):
                run_stage_edit_pipeline_bridge(
                    preflight=preflight,
                    sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
                    source_durations_seconds=(10.0, 10.0),
                    source_start_offsets_seconds=(0.0, 0.0),
                    internal_cut_times_seconds=((), ()),
                    source_timewarps=(None, None),
                    source_quality_log_priors=(0.0, 0.0),
                    section_plan=_section_plan(),
                    output=Path("/tmp/out.mp4"),
                    reference_source=0,
                    config=StageEditSemiMarkovConfig(
                        minimum_unique_sources=1,
                    ),
                )


class StageEditGenerationRequestTests(unittest.TestCase):
    def _request_payload(self, root: Path) -> dict[str, object]:
        sources = []
        for index in range(8):
            path = root / f"source-{index}.mp4"
            path.touch()
            sources.append(str(path))
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        snapshot = root / "snapshot.json"
        snapshot.write_text("{}", encoding="utf-8")
        return {
            "song": "BANG BANG",
            "edit_mode": "group",
            "aspect_ratio": "16:9",
            "source_mode": "automatic",
            "desired_source_count": 8,
            "sources": sources,
            "output_directory": str(root / "output"),
            "stage_edit_generation_checkpoint_directory": str(checkpoint),
            "stage_edit_generation_feature_snapshot": str(snapshot),
            "stage_edit_generation_source_ids": [
                f"video-{index}" for index in range(8)
            ],
        }

    def test_request_activation_is_explicit_three_part_and_round_trips(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            payload = self._request_payload(root)
            checkpoint = Path(
                str(payload["stage_edit_generation_checkpoint_directory"])
            )
            request = AnalysisRequest.from_dict(payload)
            restored = AnalysisRequest.from_dict(request.as_dict())
            self.assertEqual(
                request.stage_edit_generation_source_ids,
                restored.stage_edit_generation_source_ids,
            )
            self.assertEqual(
                checkpoint.resolve(),
                restored.stage_edit_generation_checkpoint_directory,
            )

            incomplete = dict(payload)
            incomplete.pop("stage_edit_generation_feature_snapshot")
            with self.assertRaisesRegex(ValueError, "supplied together"):
                AnalysisRequest.from_dict(incomplete)

    def test_pipeline_preflights_exact_activation_and_requires_proxy_review(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = AnalysisRequest.from_dict(self._request_payload(root))
            marker = object()
            with patch(
                "crosscut.pipeline.preflight_stage_edit_pipeline_bridge",
                return_value=marker,
            ) as preflight:
                self.assertIs(
                    marker,
                    _load_stage_edit_generation_preflight(request),
                )
            preflight.assert_called_once_with(
                feature_snapshot_path=request.stage_edit_generation_feature_snapshot,
                checkpoint_directory=(
                    request.stage_edit_generation_checkpoint_directory
                ),
                source_ids=request.stage_edit_generation_source_ids,
            )
            with self.assertRaisesRegex(
                RuntimeError, "requires decoded proxy review"
            ):
                analyze(
                    request,
                    SimpleNamespace(),
                    review_rendered_proxies=False,
                )

    def test_default_request_does_not_persist_or_activate_generation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources = []
            for index in range(8):
                path = root / f"source-{index}.mp4"
                path.touch()
                sources.append(str(path))
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 8,
                    "sources": sources,
                    "output_directory": str(root / "output"),
                }
            )
            self.assertIsNone(request.stage_edit_generation_checkpoint_directory)
            self.assertNotIn(
                "stage_edit_generation_checkpoint_directory", request.as_dict()
            )

    def test_simple_automatic_group_discovers_registered_bundle_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "sources": [],
                    "output_directory": str(root / "output"),
                }
            )
            marker = object()
            with patch(
                "crosscut.pipeline.discover_registered_stage_edit_production_bundle",
                return_value=marker,
            ) as discover:
                self.assertIs(
                    marker,
                    _discover_automatic_stage_edit_production_bundle(request),
                )
            discover.assert_called_once_with()

            local_sources = (root / "one.mp4", root / "two.mp4")
            for source in local_sources:
                source.touch()
            local = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "local",
                    "sources": [str(source) for source in local_sources],
                    "output_directory": str(root / "local-output"),
                }
            )
            with patch(
                "crosscut.pipeline.discover_registered_stage_edit_production_bundle"
            ) as discover:
                self.assertIsNone(
                    _discover_automatic_stage_edit_production_bundle(local)
                )
            discover.assert_not_called()

    def test_automatic_group_refuses_unregistered_visual_baseline(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 8,
                    "sources": [],
                    "output_directory": str(root / "output"),
                }
            )
            with patch(
                "crosscut.pipeline.discover_registered_stage_edit_production_bundle",
                return_value=None,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "requires a verified registered StageEdit production bundle",
                ):
                    analyze(
                        request,
                        SimpleNamespace(),
                        review_rendered_proxies=True,
                    )
            self.assertFalse((root / "output").exists())

    def test_registered_bundle_materializes_current_media_before_preflight(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources = []
            for index in range(8):
                source = root / f"source-{index}.mp4"
                source.touch()
                sources.append(str(source))
            request = AnalysisRequest.from_dict(
                {
                    "song": "BANG BANG",
                    "edit_mode": "group",
                    "aspect_ratio": "16:9",
                    "source_mode": "automatic",
                    "desired_source_count": 8,
                    "sources": sources,
                    "output_directory": str(root / "output"),
                    "source_cache_directory": str(root / "cache"),
                }
            )
            checkpoint = (root / "checkpoint").resolve()
            bundle = SimpleNamespace(
                checkpoint_directory=checkpoint,
                maximum_sequence_length=2048,
                maximum_sources=32,
            )
            snapshot = SimpleNamespace(
                snapshot_path=(root / "snapshot.json").resolve(),
                source_ids=tuple(f"source-{index}" for index in range(8)),
            )
            preflight = SimpleNamespace(
                readiness=SimpleNamespace(checkpoint_path=str(checkpoint))
            )
            alignment = SimpleNamespace(alignments=())
            eligibility = tuple(
                {"render_eligible": True} for _ in range(8)
            )
            timing = object()
            rhythm = object()
            sections = object()
            with (
                patch(
                    "crosscut.pipeline.materialize_stage_edit_inference_snapshot",
                    return_value=snapshot,
                ) as materialize,
                patch(
                    "crosscut.pipeline.preflight_stage_edit_pipeline_bridge",
                    return_value=preflight,
                ) as preflight_call,
            ):
                observed_snapshot, observed_preflight = (
                    _materialize_registered_stage_edit_preflight(
                        bundle=bundle,
                        request=request,
                        alignment=alignment,
                        alignment_eligibility=eligibility,
                        source_timewarps=(None,) * 8,
                        timing_draft=timing,
                        rhythm=rhythm,
                        section_plan=sections,
                    )
                )

            self.assertIs(observed_snapshot, snapshot)
            self.assertIs(observed_preflight, preflight)
            materialize.assert_called_once_with(
                song_id=request.song,
                sources=request.sources,
                alignment=alignment,
                alignment_hard_gates=(True,) * 8,
                source_timewarps=(None,) * 8,
                timing_draft=timing,
                rhythm=rhythm,
                section_plan=sections,
                output_directory=(
                    request.source_cache_directory
                    / "derived"
                    / "stage-edit-inference-v1"
                ),
                feature_cache_directory=request.source_cache_directory,
                maximum_sequence_length=2048,
                maximum_sources=32,
            )
            preflight_call.assert_called_once_with(
                feature_snapshot_path=snapshot.snapshot_path,
                checkpoint_directory=checkpoint,
                source_ids=snapshot.source_ids,
            )


if __name__ == "__main__":
    unittest.main()
