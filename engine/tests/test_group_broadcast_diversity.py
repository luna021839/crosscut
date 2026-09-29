from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from crosscut.acquisition import _build_acquisition_artifact
from crosscut.pipeline import _group_visual_diversity_config
from crosscut.timeline import TimelineCandidate, TimelineConfig, select_timeline
from crosscut.visual_score import (
    VisualDiversityConfig, _safe_cadence_profiles, _visual_optimizer_input_identity,
)


class GroupBroadcastDiversityTests(unittest.TestCase):
    def fixture(self, root: Path):
        root = root.resolve()
        sources = tuple(root / f"source-{i}.mp4" for i in range(3))
        selected = []
        for i, source in enumerate(sources):
            content = f"synthetic distinct camera {i}".encode()
            source.write_bytes(content)
            date = "2025-08-30" if i < 2 else "2025-09-06"
            selected.append({
                "source_index": i,
                "local_path": str(source),
                "title": f"IVE XOXZ Show! MusicCore MBC{date.replace('-', '')[2:]}방송",
                "broadcast_program": "music_core",
                "broadcast_date": date,
                "broadcast_date_basis": "title",
                "content_integrity": {
                    "contract_version": 1, "algorithm": "sha256",
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
            })
        manifest = {
            "schema_version": 1, "edit_mode": "group",
            "summary": {"status": "complete"}, "selected": selected,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return sources, path, manifest

    def test_artifact_keeps_all_sources_but_binds_same_stage_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources, path, manifest = self.fixture(Path(tmp))
            artifact = _build_acquisition_artifact(
                source_paths=sources, manifest_path=path, manifest=manifest,
            )
            self.assertEqual(artifact.source_paths, sources)
            self.assertIsNone(artifact.member_identity_authority)
            self.assertEqual(artifact.group_broadcast_slot_hints, (
                "2025-08-30:music_core", "2025-08-30:music_core",
                "2025-09-06:music_core",
            ))

    def test_stale_or_inconsistent_metadata_is_unknown_not_a_deleted_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources, path, manifest = self.fixture(Path(tmp))
            for key, value in (
                ("source_index", 1), ("source_index", False),
                ("local_path", str(sources[1])),
                ("local_path", str(Path(tmp) / "missing.mp4")),
                ("title", "IVE XOXZ Show! MusicCore without a performance date"),
                ("broadcast_date", "2025-08-31"),
                ("broadcast_program", "music_bank"),
                ("broadcast_date_basis", "upload_date"),
                ("content_integrity", {}),
            ):
                with self.subTest(key=key, value=value):
                    edited = deepcopy(manifest)
                    edited["selected"][0][key] = value
                    artifact = _build_acquisition_artifact(
                        source_paths=sources, manifest_path=path, manifest=edited,
                    )
                    self.assertEqual(artifact.source_paths, sources)
                    self.assertIsNone(artifact.group_broadcast_slot_hints[0])
                    self.assertEqual(artifact.group_broadcast_slot_hints[1],
                                     "2025-08-30:music_core")
            sources[0].write_bytes(b"changed media bytes")
            artifact = _build_acquisition_artifact(
                source_paths=sources, manifest_path=path, manifest=manifest,
            )
            self.assertIsNone(artifact.group_broadcast_slot_hints[0])

    def test_group_hints_do_not_repurpose_member_authority(self):
        from crosscut.acquisition import _group_broadcast_slot_hints
        self.assertEqual(_group_broadcast_slot_hints((Path("a"),),
                                                   {"edit_mode": "member"}), ())

    @staticmethod
    def candidate(segment, source, quality=1.0, scale=0.3):
        return TimelineCandidate(
            segment_index=segment, source_id=source,
            start_seconds=float(segment), end_seconds=float(segment + 1),
            quality_score=quality, shot_scale=scale,
        )

    @staticmethod
    def config(**kwargs):
        values = dict(
            quality_weight=1.0, continuity_bonus=0.0, cut_penalty=0.0,
            early_cut_penalty=0.0, source_switch_bonus=0.20,
            minimum_hold_segments=1,
            source_broadcast_slots=(("A", "same"), ("B", "same"), ("C", "other")),
        )
        values.update(kwargs)
        return TimelineConfig(**values)

    def test_near_equal_cross_broadcast_wins_without_excluding_same_stage(self):
        lattice = ((self.candidate(0, "A"),), (
            self.candidate(1, "B", .95), self.candidate(1, "C", .94),
        ))
        baseline = select_timeline(lattice, config=self.config(source_broadcast_slots=()))
        grouped = select_timeline(lattice, config=self.config())
        self.assertEqual(baseline.candidates[-1].source_id, "B")
        self.assertEqual(grouped.candidates[-1].source_id, "C")
        only_same = select_timeline((lattice[0], (lattice[1][0],)), config=self.config())
        self.assertEqual(only_same.candidates[-1].source_id, "B")
        self.assertAlmostEqual(only_same.choices[-1].transition_score, .05)

    def test_different_stage_cannot_bypass_visual_gate_or_large_quality_loss(self):
        first = (self.candidate(0, "A"),)
        for other, config in (
            (self.candidate(1, "C", 1.0, .9),
             self.config(maximum_cross_source_scale_delta=.1)),
            (self.candidate(1, "C", 0.1), self.config()),
        ):
            result = select_timeline(
                (first, (self.candidate(1, "B", .95), other)), config=config,
            )
            self.assertEqual(result.candidates[-1].source_id, "B")

    def test_unknown_stages_are_not_grouped_and_overrides_are_discounted(self):
        lattice = ((self.candidate(0, "A"),), (self.candidate(1, "B"),))
        unknown = select_timeline(lattice, config=self.config(source_broadcast_slots=()))
        grouped = select_timeline(lattice, config=self.config(),
                                  source_switch_bonus_by_segment={1: .48})
        self.assertAlmostEqual(unknown.choices[-1].transition_score, .20)
        self.assertAlmostEqual(grouped.choices[-1].transition_score, .12)

    def test_source_subset_mapping_and_all_cadence_heads_keep_hints(self):
        config = _group_visual_diversity_config((2, 0, 1), ("same", None, "other"))
        self.assertEqual(config.source_broadcast_slots, (("0", "other"), ("1", "same")))
        for profile in _safe_cadence_profiles(config):
            self.assertEqual(profile.config.source_broadcast_slots, config.source_broadcast_slots)
            self.assertEqual(profile.config.same_broadcast_switch_bonus_multiplier, .25)
        self.assertEqual(_group_visual_diversity_config((2, 0), ()).source_broadcast_slots, ())

    def test_optimizer_identity_changes_with_stage_hint(self):
        lattice = ((self.candidate(0, "A"),),)
        arguments = dict(sources=(Path("source.mp4"),), audio_offsets=(0.0,),
                         source_timewarps=(None,), source_clock_basis="synthetic",
                         scene_cut_times=((),))
        base = _visual_optimizer_input_identity(lattice, **arguments)
        hinted = _visual_optimizer_input_identity(lattice, **arguments,
            diversity=VisualDiversityConfig(source_broadcast_slots=(("A", "slot"),)))
        self.assertNotEqual(base, hinted)

    def test_invalid_config_hints_are_rejected(self):
        for hints in ((("A", ""),), (("A", "x"), ("A", "y")), [("A", "x")]):
            with self.assertRaises(ValueError):
                VisualDiversityConfig(source_broadcast_slots=hints)
        for value in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                replace(self.config(), same_broadcast_switch_bonus_multiplier=value)


if __name__ == "__main__":
    unittest.main()
