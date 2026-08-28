from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from crosscut.draft import DraftSegment
from crosscut.pipeline import (
    _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT,
    _DIRECT_MEMBER_PARTITION_WARM_START_FILENAME,
    _DIRECT_MEMBER_PARTITION_WARM_START_MAX_BYTES,
    _DIRECT_MEMBER_PARTITION_WARM_START_MAX_READ_BYTES,
    _DirectFullFramePartitionProfile,
    _direct_full_frame_spans_from_partition,
    _direct_member_partition_warm_start_binding,
    _direct_partition_signature_sha256,
    _load_direct_member_partition_warm_start_profiles,
    _persist_direct_member_partition_warm_start_profile,
)


def _timing_segments() -> tuple[DraftSegment, ...]:
    return tuple(
        DraftSegment(
            segment_index=index,
            source_index=0,
            canonical_master_start=float(index),
            canonical_master_end=float(index + 1),
            output_start_seconds=float(index),
            source_local_start=float(index),
            source_local_end=float(index + 1),
            audio_offset_seconds=0.0,
            sync_score=1.0,
            estimated_beat_count=1.0,
            start_boundary_reasons=("beat",),
            end_boundary_reasons=("beat",),
        )
        for index in range(76)
    )


def _profile(
    profile_id: str,
    unit_counts: tuple[int, ...],
) -> _DirectFullFramePartitionProfile:
    segments = _timing_segments()
    return _DirectFullFramePartitionProfile(
        profile_id=profile_id,
        unit_counts=unit_counts,
        spans=_direct_full_frame_spans_from_partition(segments, unit_counts),
    )


class DirectMemberPartitionWarmStartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = _timing_segments()
        self.base = _profile("base", (4,) * 19)
        self.candidate = _profile(
            "candidate",
            (2, 2, *((4,) * 18)),
        )

    def _write_manifest(
        self,
        cache_directory: Path,
        *,
        revision: int = 1,
        first_source_sha256: str = "a" * 64,
    ) -> None:
        (cache_directory / "acquisition-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "group": "IVE",
                    "song": "TEST",
                    "edit_mode": "member",
                    "target_member": "가을",
                    "revision": revision,
                    "selected": [
                        {
                            "source_index": source_index,
                            "video_id": f"source_{source_index:02d}",
                            "broadcast_date": f"2026-03-{source_index + 1:02d}",
                            "broadcast_program": f"show_{source_index:02d}",
                            "content_integrity": {
                                "algorithm": "sha256",
                                "sha256": (
                                    first_source_sha256
                                    if source_index == 0
                                    else f"{source_index + 1:064x}"
                                ),
                                "size_bytes": 1_000_000 + source_index,
                            },
                        }
                        for source_index in range(8)
                    ],
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    def test_loads_only_exact_manifest_and_base_bound_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            self._write_manifest(cache_directory)
            binding, diagnostics = _direct_member_partition_warm_start_binding(
                cache_directory,
                self.base,
                source_count=8,
            )
            self.assertIsNotNone(binding)
            self.assertEqual(diagnostics["status"], "binding_ready")
            assert binding is not None
            signature = _direct_partition_signature_sha256(
                self.candidate.unit_counts
            )
            (cache_directory / _DIRECT_MEMBER_PARTITION_WARM_START_FILENAME).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "contract": _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT,
                        "entries": [
                            {
                                **binding,
                                "candidate_partition_signature_sha256": signature,
                                "partition_unit_counts": list(
                                    self.candidate.unit_counts
                                ),
                                "origin": "test_fixture",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            profiles, load_diagnostics = (
                _load_direct_member_partition_warm_start_profiles(
                    cache_directory,
                    self.segments,
                    self.base,
                    source_count=8,
                )
            )

            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0].unit_counts, self.candidate.unit_counts)
            self.assertEqual(load_diagnostics["status"], "candidates_loaded")
            warm_start = profiles[0].to_dict()[
                "input_bound_partition_warm_start"
            ]
            self.assertEqual(
                warm_start["binding_sha256"],
                binding["binding_sha256"],
            )
            self.assertTrue(warm_start["full_selector_recheck_required"])
            self.assertFalse(warm_start["visual_gates_relaxed"])

            self._write_manifest(cache_directory, revision=2)
            refreshed_profiles, refreshed_diagnostics = (
                _load_direct_member_partition_warm_start_profiles(
                    cache_directory,
                    self.segments,
                    self.base,
                    source_count=8,
                )
            )
            self.assertEqual(len(refreshed_profiles), 1)
            self.assertEqual(
                refreshed_diagnostics["status"],
                "candidates_loaded",
            )

            self._write_manifest(
                cache_directory,
                revision=3,
                first_source_sha256="b" * 64,
            )
            stale_profiles, stale_diagnostics = (
                _load_direct_member_partition_warm_start_profiles(
                    cache_directory,
                    self.segments,
                    self.base,
                    source_count=8,
                )
            )
            self.assertEqual(stale_profiles, ())
            self.assertEqual(stale_diagnostics["status"], "no_match")
            self.assertEqual(
                stale_diagnostics["rejection_reason_counts"],
                {"input_binding_mismatch": 1},
            )

    def test_fully_selected_partition_is_loaded_before_newer_failures(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            self._write_manifest(cache_directory)
            binding, _diagnostics = (
                _direct_member_partition_warm_start_binding(
                    cache_directory,
                    self.base,
                    source_count=8,
                )
            )
            assert binding is not None
            generic_counts = (
                (1, 3, *((4,) * 18)),
                (2, 2, *((4,) * 18)),
                (3, 1, *((4,) * 18)),
                (1, 1, 2, *((4,) * 18)),
            )
            selected_counts = (2, 1, 1, *((4,) * 18))
            entries = []
            for counts in generic_counts:
                entries.append(
                    {
                        **binding,
                        "candidate_partition_signature_sha256": (
                            _direct_partition_signature_sha256(counts)
                        ),
                        "partition_unit_counts": list(counts),
                        "origin": "proof_bound_test_frontier",
                    }
                )
            entries.append(
                {
                    **binding,
                    "candidate_partition_signature_sha256": (
                        _direct_partition_signature_sha256(selected_counts)
                    ),
                    "partition_unit_counts": list(selected_counts),
                    "origin": "full_visual_selector_feasible",
                }
            )
            (
                cache_directory
                / _DIRECT_MEMBER_PARTITION_WARM_START_FILENAME
            ).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "contract": (
                            _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT
                        ),
                        "entries": entries,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            profiles, diagnostics = (
                _load_direct_member_partition_warm_start_profiles(
                    cache_directory,
                    self.segments,
                    self.base,
                    source_count=8,
                )
            )

            self.assertEqual(len(profiles), 4)
            self.assertEqual(profiles[0].unit_counts, selected_counts)
            self.assertTrue(
                profiles[0].profile_id.startswith("input-bound-warm-start-")
            )
            self.assertEqual(diagnostics["fully_selected_entry_count"], 1)
            self.assertEqual(
                diagnostics["entry_order_contract"],
                "fully_selected_profile_before_failure_frontier_then_cache_order_v1",
            )

    def test_persisted_candidate_is_reloaded_for_full_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            self._write_manifest(cache_directory)

            persistence = _persist_direct_member_partition_warm_start_profile(
                cache_directory,
                self.base,
                self.candidate,
                source_count=8,
            )
            profiles, diagnostics = (
                _load_direct_member_partition_warm_start_profiles(
                    cache_directory,
                    self.segments,
                    self.base,
                    source_count=8,
                )
            )

            self.assertTrue(persistence["persisted"])
            self.assertEqual(persistence["status"], "persisted")
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0].unit_counts, self.candidate.unit_counts)
            self.assertEqual(diagnostics["loaded_candidate_count"], 1)
            payload = json.loads(
                (
                    cache_directory
                    / _DIRECT_MEMBER_PARTITION_WARM_START_FILENAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["entries"][0]["origin"],
                "full_visual_selector_feasible",
            )

    def test_legacy_oversized_cache_loads_then_compacts_on_persistence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            self._write_manifest(cache_directory)
            initial = _persist_direct_member_partition_warm_start_profile(
                cache_directory,
                self.base,
                self.candidate,
                source_count=8,
            )
            self.assertTrue(initial["persisted"])
            cache_path = (
                cache_directory
                / _DIRECT_MEMBER_PARTITION_WARM_START_FILENAME
            )
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            padded_entry = {
                **payload["entries"][0],
                "legacy_padding": "x" * 9_000,
            }
            payload["entries"] = [dict(padded_entry) for _index in range(32)]
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            oversized_size = cache_path.stat().st_size
            self.assertGreater(
                oversized_size,
                _DIRECT_MEMBER_PARTITION_WARM_START_MAX_BYTES,
            )
            self.assertLessEqual(
                oversized_size,
                _DIRECT_MEMBER_PARTITION_WARM_START_MAX_READ_BYTES,
            )

            loaded, load_diagnostics = (
                _load_direct_member_partition_warm_start_profiles(
                    cache_directory,
                    self.segments,
                    self.base,
                    source_count=8,
                )
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(
                loaded[0].unit_counts,
                self.candidate.unit_counts,
            )
            self.assertTrue(load_diagnostics["cache_compaction_required"])

            second_candidate = _profile(
                "second-candidate",
                (1, 3, *((4,) * 18)),
            )
            persistence = _persist_direct_member_partition_warm_start_profile(
                cache_directory,
                self.base,
                second_candidate,
                source_count=8,
            )

            self.assertTrue(persistence["persisted"])
            self.assertTrue(
                persistence["existing_cache_compaction_required"]
            )
            self.assertGreater(persistence["dropped_entry_count"], 0)
            self.assertLessEqual(
                cache_path.stat().st_size,
                _DIRECT_MEMBER_PARTITION_WARM_START_MAX_BYTES,
            )
            compacted = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(
                compacted["entries"][0][
                    "candidate_partition_signature_sha256"
                ],
                _direct_partition_signature_sha256(
                    second_candidate.unit_counts
                ),
            )

    def test_missing_cache_is_a_nonfatal_optimization_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            self._write_manifest(cache_directory)

            profiles, diagnostics = (
                _load_direct_member_partition_warm_start_profiles(
                    cache_directory,
                    self.segments,
                    self.base,
                    source_count=8,
                )
            )

            self.assertEqual(profiles, ())
            self.assertEqual(diagnostics["status"], "cache_missing")
            self.assertTrue(diagnostics["full_selector_recheck_required"])
            self.assertFalse(diagnostics["visual_gates_relaxed"])


if __name__ == "__main__":
    unittest.main()
