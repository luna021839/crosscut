from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut.ive import EditMode, IVEMember
from crosscut.ive_causal_sensing_installation import (
    register_ive_causal_sensing_installation,
)
from crosscut.ive_causal_sensing_media_job import (
    IVECausalSensingMediaJobError,
    IVECausalSensingMediaJobResult,
)
from crosscut.ive_causal_sensing_pipeline_inputs import (
    IVECausalSensingPipelineInputError,
    materialize_ive_causal_sensing_for_stage_edit_pipeline,
    prepare_ive_causal_sensing_pipeline_request,
    write_ive_causal_sensing_pipeline_audit,
)
from crosscut.stage_edit_inference_snapshot import (
    build_stage_edit_inference_decision_times,
    materialize_stage_edit_inference_snapshot,
)
from engine.tests.test_pipeline import _linear_timewarp
from engine.tests.test_stage_edit_inference_snapshot import (
    _alignment,
    _causal_sensing,
    _descriptor,
    _draft,
    _rhythm,
    _sections,
)
from tests.test_ive_causal_sensing_media_job import _gallery_artifact


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ExactHashSampler:
    hashes: dict[Path, str] = {}

    def __init__(self, path: Path, *, feature_cache_directory=None) -> None:
        del feature_cache_directory
        resolved = Path(path).resolve(strict=True)
        self.frame_rate = 30.0
        self.persistent_feature_cache = SimpleNamespace(
            source_sha256=self.hashes[resolved]
        )

    def feature_descriptor_at(self, *_args, **_kwargs):
        return _descriptor()

    def close(self) -> None:
        return None


class IVECausalSensingPipelineInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.sources = (self.root / "a.mp4", self.root / "b.mp4")
        self.sources[0].write_bytes(b"source-a-real-media")
        self.sources[1].write_bytes(b"source-b-real-media")
        gallery = _gallery_artifact(self.root / "gallery")
        self.installation = register_ive_causal_sensing_installation(
            gallery_artifact_path=gallery.path,
            gallery_content_sha256=gallery.content_sha256,
            pointer_path=self.root / "install" / "active-private-gallery.json",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _prepare(self, **overrides):
        values = {
            "song": "NEW IVE SONG",
            "edit_mode": EditMode.GROUP,
            "target_member": None,
            "sources": self.sources,
            "alignment": _alignment(self.sources),
            "alignment_hard_gates": (True, True),
            "source_timewarps": (None, None),
            "source_scene_cut_times": ((0.5, 2.0), (1.0, 3.0)),
            "timing_draft": _draft(),
            "rhythm": _rhythm(),
            "section_plan": _sections(),
            "maximum_sequence_length": 128,
            "maximum_sources": 8,
            "installation": self.installation,
            "artifact_directory": self.root / "derived" / "sensing",
            "created_at_utc": "2026-08-25T00:00:00Z",
        }
        values.update(overrides)
        return prepare_ive_causal_sensing_pipeline_request(**values)

    def test_prepared_request_reuses_exact_grid_media_clocks_and_native_cuts(self) -> None:
        prepared = self._prepare()
        expected_times = build_stage_edit_inference_decision_times(
            timing_draft=_draft(),
            rhythm=_rhythm(),
            section_plan=_sections(),
            maximum_sequence_length=128,
        )
        expected_hashes = tuple(_sha(path) for path in self.sources)
        expected_ids = tuple(
            f"source-{index:02d}-{digest[:16]}"
            for index, digest in enumerate(expected_hashes)
        )

        self.assertEqual(prepared.decision_times, expected_times)
        self.assertEqual(prepared.source_content_sha256, expected_hashes)
        self.assertEqual(prepared.source_ids, expected_ids)
        self.assertEqual(
            tuple(item.native_cut_source_seconds for item in prepared.request.sources),
            ((0.5, 2.0), (1.0, 3.0)),
        )
        for source, alignment in zip(
            prepared.request.sources,
            _alignment(self.sources).alignments,
            strict=True,
        ):
            for timestamp in expected_times:
                self.assertAlmostEqual(
                    source.clock.master_to_source(timestamp),
                    timestamp + alignment.offset_seconds,
                    places=12,
                )
        audit = prepared.public_audit_dict()
        self.assertFalse(audit["local_paths_persisted"])
        self.assertNotIn(str(self.root), str(audit))
        self.assertEqual(audit["source_ids"], list(expected_ids))

    def test_piecewise_clock_and_member_scope_are_preserved_exactly(self) -> None:
        timewarp = _linear_timewarp(duration=4.0, rate=1.004)
        prepared = self._prepare(
            edit_mode=EditMode.MEMBER,
            target_member=IVEMember.GAEUL,
            source_timewarps=(timewarp, None),
        )
        self.assertIs(prepared.request.edit_mode, EditMode.MEMBER)
        self.assertIs(prepared.request.target_member, IVEMember.GAEUL)
        first_clock = prepared.request.sources[0].clock
        self.assertEqual(first_clock.mapping_mode, "piecewise_linear")
        for timestamp in prepared.decision_times:
            self.assertAlmostEqual(
                first_clock.master_to_source(timestamp),
                timewarp.master_to_source(timestamp),
                places=12,
            )

    def test_quarantine_runs_before_discovery_or_any_filesystem_access(self) -> None:
        attitude_sources = (
            Path("/definitely/missing/ATTITUDE-a.mp4"),
            Path("/definitely/missing/b.mp4"),
        )
        with (
            patch(
                "crosscut.ive_causal_sensing_pipeline_inputs."
                "discover_ive_causal_sensing_installation"
            ) as discover,
            patch(
                "crosscut.ive_causal_sensing_pipeline_inputs._stable_media_sha256"
            ) as hasher,
            patch(
                "crosscut.ive_causal_sensing_pipeline_inputs."
                "materialize_ive_causal_sensing_from_media"
            ) as media_job,
        ):
            with self.assertRaisesRegex(
                IVECausalSensingMediaJobError,
                "ATTITUDE",
            ):
                materialize_ive_causal_sensing_for_stage_edit_pipeline(
                    song="ATTITUDE",
                    edit_mode=EditMode.GROUP,
                    target_member=None,
                    sources=attitude_sources,
                    alignment=_alignment(self.sources),
                    alignment_hard_gates=(True, True),
                    source_timewarps=(None, None),
                    source_scene_cut_times=((), ()),
                    timing_draft=_draft(),
                    rhythm=_rhythm(),
                    section_plan=_sections(),
                    maximum_sequence_length=128,
                    maximum_sources=8,
                    artifact_directory=self.root / "must-not-exist",
                    created_at_utc="2026-08-25T00:00:00Z",
                )
        discover.assert_not_called()
        hasher.assert_not_called()
        media_job.assert_not_called()
        self.assertFalse((self.root / "must-not-exist").exists())

    def test_invalid_gate_cut_order_and_clock_authority_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            IVECausalSensingPipelineInputError,
            "alignment hard gate",
        ):
            self._prepare(alignment_hard_gates=(True, False))
        with self.assertRaisesRegex(
            IVECausalSensingPipelineInputError,
            "scene-cut source times",
        ):
            self._prepare(source_scene_cut_times=((2.0, 0.5), (1.0,)))
        mismatched = _linear_timewarp(
            duration=4.0,
            global_offset=0.25,
            rate=1.004,
        )
        with self.assertRaisesRegex(
            IVECausalSensingPipelineInputError,
            "offset disagree",
        ):
            self._prepare(source_timewarps=(mismatched, None))

    def test_real_job_result_must_bind_exact_inference_snapshot(self) -> None:
        prepared = self._prepare()
        artifacts = _causal_sensing(
            self.root / "synthetic-job-artifacts",
            source_ids=prepared.source_ids,
            decision_times=prepared.decision_times,
        )
        binding = artifacts.binding
        job_result = IVECausalSensingMediaJobResult(
            shooting_grammar_path=artifacts.shooting_grammar_path,
            shooting_grammar_content_sha256=(
                binding.shooting_grammar_content_sha256
            ),
            member_state_graph_path=artifacts.member_state_graph_path,
            member_state_graph_content_sha256=(
                binding.member_state_graph_content_sha256
            ),
            ive_performer_state_path=artifacts.ive_performer_state_path,
            ive_performer_state_content_sha256=(
                binding.ive_performer_state_content_sha256
            ),
            stage_edit_artifacts=artifacts,
            source_count=2,
            decision_count=len(prepared.decision_times),
            member_observation_count=2,
            accepted_identity_observation_count=2,
        )
        with (
            patch(
                "crosscut.ive_causal_sensing_pipeline_inputs."
                "discover_ive_causal_sensing_installation",
                return_value=self.installation,
            ),
            patch(
                "crosscut.ive_causal_sensing_pipeline_inputs."
                "materialize_ive_causal_sensing_from_media",
                return_value=job_result,
            ),
        ):
            pipeline_result = materialize_ive_causal_sensing_for_stage_edit_pipeline(
                song="NEW IVE SONG",
                edit_mode=EditMode.GROUP,
                target_member=None,
                sources=self.sources,
                alignment=_alignment(self.sources),
                alignment_hard_gates=(True, True),
                source_timewarps=(None, None),
                source_scene_cut_times=((0.5, 2.0), (1.0, 3.0)),
                timing_draft=_draft(),
                rhythm=_rhythm(),
                section_plan=_sections(),
                maximum_sequence_length=128,
                maximum_sources=8,
                artifact_directory=self.root / "job-output",
                created_at_utc="2026-08-25T00:00:00Z",
            )

        _ExactHashSampler.hashes = {
            path.resolve(strict=True): _sha(path) for path in self.sources
        }
        with (
            patch(
                "crosscut.stage_edit_inference_snapshot._VideoSampler",
                _ExactHashSampler,
            ),
            patch(
                "crosscut.stage_edit_inference_snapshot."
                "visual_analyzer_fingerprint_sha256",
                return_value="a" * 64,
            ),
        ):
            inference = materialize_stage_edit_inference_snapshot(
                song_id="NEW IVE SONG",
                sources=self.sources,
                alignment=_alignment(self.sources),
                alignment_hard_gates=(True, True),
                source_timewarps=(None, None),
                timing_draft=_draft(),
                rhythm=_rhythm(),
                section_plan=_sections(),
                output_directory=self.root / "snapshots",
                feature_cache_directory=self.root / "cache",
                maximum_sequence_length=128,
                maximum_sources=8,
                causal_sensing=artifacts,
                edit_mode=EditMode.GROUP,
                target_member=None,
            )
        audit = pipeline_result.require_exact_inference_snapshot_binding(inference)
        self.assertTrue(audit["exact_inference_snapshot_binding_verified"])
        self.assertEqual(
            audit["source_content_sha256"],
            list(pipeline_result.prepared.source_content_sha256),
        )
        persisted = write_ive_causal_sensing_pipeline_audit(
            pipeline_result=pipeline_result,
            inference_snapshot=inference,
            artifact_directory=self.root / "pipeline-audits",
        )
        replayed = write_ive_causal_sensing_pipeline_audit(
            pipeline_result=pipeline_result,
            inference_snapshot=inference,
            artifact_directory=self.root / "pipeline-audits",
        )
        self.assertEqual(persisted, replayed)
        self.assertEqual(
            persisted.snapshot_sha256,
            inference.snapshot.snapshot_sha256,
        )
        encoded = persisted.path.read_text(encoding="utf-8")
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("prototypes", encoded)
        self.assertNotIn('"embedding": [', encoded)


if __name__ == "__main__":
    unittest.main()
