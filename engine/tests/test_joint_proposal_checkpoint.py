import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from crosscut.joint_proposal_checkpoint import rebind_joint_proposal_checkpoint, find_joint_proposal_restart
from crosscut.joint_proposal_domain import JointProposalPool, select_with_joint_proposal_domains
from crosscut.timeline import NoCompleteTimelinePath
from crosscut.visual_candidate_checkpoint import canonical_json_sha256
from engine.tests.test_joint_timeline import candidate


@dataclass(frozen=True)
class Selection:
    selection_retry_audit: dict


class JointProposalCheckpointTests(TestCase):
    def setUp(self):
        self.lattice = tuple(tuple(candidate(i, str(s), .5) for s in range(3)) for i in range(3))
        self.path = tuple(row[i % 3] for i, row in enumerate(self.lattice))
        self.identity = {"sources": [{"sha256": "a" * 64}],
            "canonical_spans": [[row[0].start_seconds, row[0].end_seconds] for row in self.lattice],
            "audio_offsets": [0.0], "selector_code_sha256": "b" * 64}
        self.payload = {"status": "candidate_selection_complete_derivation_pending",
            "analysis_basis": "joint_decoded_visual_candidate_selection_v1",
            "input_identity": copy.deepcopy(self.identity),
            "input_identity_sha256": canonical_json_sha256(self.identity),
            "joint_visual_selection": {"selection_batches": [{"source_shot_paths": [
                [[c.source_id, c.shot_id] for c in self.path]]}]}}

    def load(self, payload=None, lattice=None, identity=None):
        return rebind_joint_proposal_checkpoint(payload or self.payload,
            expected_input_identity=identity or self.identity,
            lattice=lattice or self.lattice)

    def test_rebinds_only_current_objects_and_ignores_stored_approval(self):
        self.payload["boundary_proof_bundle"] = {"passed": True, "fake": "not loaded"}
        paths = self.load()
        for observed, current in zip(paths[0], self.path):
            self.assertIs(observed, current)

    def test_changed_file_clock_and_code_do_not_reuse_hint(self):
        for field, value in (("sources", [{"sha256": "c" * 64}]),
                             ("audio_offsets", [.001]), ("selector_code_sha256", "d" * 64)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                self.load(identity={**self.identity, field: value})

    def test_corrupt_identity_does_not_reuse_hint(self):
        self.payload["input_identity"]["audio_offsets"] = [.001]
        with self.assertRaisesRegex(ValueError, "identity"):
            self.load()

    def test_current_lattice_clock_and_ambiguous_ids_rejected(self):
        changed = ((replace(self.lattice[0][0], start_seconds=.001),) + self.lattice[0][1:],) + self.lattice[1:]
        with self.assertRaisesRegex(ValueError, "clock"):
            self.load(lattice=changed)
        ambiguous = ((self.path[0], self.path[0]),) + self.lattice[1:]
        with self.assertRaisesRegex(ValueError, "uniquely"):
            self.load(lattice=ambiguous)

    def test_missing_shot_or_incomplete_path_rejected(self):
        raw = self.payload["joint_visual_selection"]["selection_batches"][0]["source_shot_paths"][0]
        raw[0][1] = "unavailable"
        with self.assertRaisesRegex(ValueError, "uniquely"):
            self.load()
        raw.pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.load()

    def test_loader_skips_corrupt_newer_file_and_never_returns_proofs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "old-pre-derivation-candidate-selection.json"
            path.write_text(json.dumps(self.payload))
            (root / "new-pre-derivation-candidate-selection.json").write_text("broken")
            result = find_joint_proposal_restart(root, expected_input_identity=self.identity, lattice=self.lattice)
            self.assertEqual(result.paths, (self.path,))
            self.assertFalse(result.scalar_audit()["proofs_restored"])
            self.assertFalse(result.scalar_audit()["final_media_authority"])

    def test_symlink_not_used(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "target.json"
            path.write_text(json.dumps(self.payload))
            (root / "link-pre-derivation-candidate-selection.json").symlink_to(path)
            self.assertIsNone(find_joint_proposal_restart(root, expected_input_identity=self.identity, lattice=self.lattice))

    def pool(self):
        baseline = tuple(row[0] for row in self.lattice)
        return JointProposalPool((baseline,), ("balanced",), 18, self.lattice, (self.path,))

    def test_warm_domain_calls_exact_selector_and_preserves_gates(self):
        pool, calls = self.pool(), []
        def select(**kwargs):
            calls.append(kwargs)
            return Selection({"proof_materialization_call_count": 1})
        result = select_with_joint_proposal_domains(pool, maximum_proof_calls=3, select=select)
        self.assertEqual(calls[0]["boundary_search_domain"], pool.domain(0))
        self.assertFalse(result.selection_retry_audit["proposal_search_domain"]["quality_gates_waived"])

    def test_failed_warm_hint_returns_to_baseline_without_free_proof_budget(self):
        pool, calls = self.pool(), []
        # A cross-source baseline contributes an edge absent from the warm path.
        baseline = (self.lattice[0][2], self.lattice[1][0], self.lattice[2][1])
        pool = replace(pool, paths=(baseline,))
        def select(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise NoCompleteTimelinePath("revalidation failed", reason_code="test",
                    rejection_counts={"proof_materialization_call_count": 1})
            return Selection({"proof_materialization_call_count": 1})
        result = select_with_joint_proposal_domains(pool, maximum_proof_calls=3, select=select)
        self.assertEqual([c["maximum_proof_materialization_calls"] for c in calls], [3, 2])
        self.assertEqual(calls[1]["boundary_search_domain"], pool.domain(3))
        self.assertEqual(result.selection_retry_audit["proposal_search_domain"]["local_repairs"], [])

    def test_warm_path_cannot_bypass_failed_exact_proof(self):
        def select(**kwargs):
            raise NoCompleteTimelinePath("bad pixels", reason_code="test", rejection_counts={"proof_materialization_call_count": 1})
        with self.assertRaises(NoCompleteTimelinePath):
            select_with_joint_proposal_domains(self.pool(), maximum_proof_calls=1, select=select)
