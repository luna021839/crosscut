from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from crosscut.candidate_stream_verdict import (
    CandidateBlocker, CandidateBlockerKind, CandidateStreamAction,
    CandidateStreamReducer, CandidateVerdictStatus, TerminalCandidateVerdict,
)
from crosscut.events import EventReporter
from crosscut.joint_decoded_budget import JointDecodedBudgetLedger
from crosscut.joint_look_breadth import JointLookBreadthScheduler
from crosscut.joint_runtime_progress import (
    JointSelectionRuntime, joint_candidate_terminal_payload,
)


class JointRuntimeProgressTests(unittest.TestCase):
    def test_provider_observation_does_not_decode_or_build_proof_audit(self):
        provider = SimpleNamespace(
            descriptor_decode_count=12,
            scalar_bridge_persistent_cache_hit_count=90,
            scalar_bridge_persistent_cache_miss_count=2,
            to_dict=Mock(side_effect=AssertionError("full audit requested")),
        )
        ticks = iter((10.0, 13.25, 14.0))
        observer = JointSelectionRuntime(provider, batch_index=1, clock=lambda: next(ticks))
        first = observer.snapshot()
        self.assertEqual(first["batch_elapsed_seconds"], 3.25)
        self.assertEqual(first["persistent_scalar_cache"]["request_hit_count"], 90)
        self.assertIsNone(first["provider_work"]["face_identity_model_call_count"])
        provider.descriptor_decode_count += 1
        second = observer.snapshot()
        self.assertEqual(second["provider_work"]["descriptor_decode_count"], 13)
        self.assertEqual(first["provider_work"]["descriptor_decode_count"], 12)
        provider.to_dict.assert_not_called()
        json.dumps(second, allow_nan=False)

    def test_terminal_summary_retains_actual_issue_codes_without_private_data(self):
        verdict = TerminalCandidateVerdict(3, CandidateVerdictStatus.DECODED_REVIEW_FAILED, (
            CandidateBlocker(CandidateBlockerKind.DECODED_REVIEW, "decoded_proxy_quality_gate_failed"),
        ))
        review = {
            "candidate_index": 3, "stage": "decoded_proxy_quality_gate",
            "review": {"boundaries": [
                {"issue_reasons": ["structure_shock", "face_scale_jump"]},
                {"issue_reasons": ["structure_shock"], "path": "/private/media.mp4"},
            ]},
            "error": "/private/model/failed.bin",
        }
        review["cycle"] = review
        payload = joint_candidate_terminal_payload(
            verdict, elapsed_seconds=4.2567,
            failures=[review, {"candidate_index": 4, "issue_reasons": ["unrelated"]}],
            budget=JointDecodedBudgetLedger().report_payload(), decoded_pass_count=0,
        )
        self.assertEqual(payload["reported_issue_counts"], {"face_scale_jump": 1, "structure_shock": 2})
        self.assertEqual(payload["elapsed_seconds"], 4.257)
        self.assertFalse(payload["final_media_authority"])
        self.assertNotIn("private", json.dumps(payload))
        self.assertEqual(payload["status"], verdict.status.value)

    def _actual_pipeline_attempt(self, *, raises=False, broken_output=False):
        # Execute the production nested attempt function with small fakes at
        # media boundaries. This tests emitted/persisted terminal events and
        # duplicate suppression without importing unrelated pipeline jobs.
        source = Path(__file__).parents[1] / "crosscut" / "pipeline.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "derive_and_gate_candidate")
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ], type_ignores=[])
        ast.fix_missing_locations(module)
        indices = set()
        reducer = CandidateStreamReducer()
        verdict = TerminalCandidateVerdict(0, CandidateVerdictStatus.DECODED_REVIEW_FAILED, (
            CandidateBlocker(CandidateBlockerKind.DECODED_REVIEW, "decoded_proxy_quality_gate_failed"),
        ))

        def derive(*args, **kwargs):
            indices.add(0)
            if raises:
                raise ValueError("test decoder failure")
            return verdict

        stream = io.StringIO()
        reporter = EventReporter(stream)
        if broken_output:
            reporter.emit = Mock(side_effect=BrokenPipeError("app disconnected"))
        audits = []
        persisted = []
        ticks = iter((1.0, 5.0))
        namespace = {
            "TerminalCandidateVerdict": TerminalCandidateVerdict,
            "CandidateVerdictStatus": CandidateVerdictStatus,
            "CandidateBlocker": CandidateBlocker,
            "CandidateBlockerKind": CandidateBlockerKind,
            "CandidateStreamAction": CandidateStreamAction,
            "attempted_candidate_indices": indices,
            "candidate_stream_reducer": reducer,
            "_derive_and_gate_candidate_verdict": derive,
            "_underlying_joint_materialized_candidate": lambda value: None,
            "joint_visual_selection_active": True,
            "joint_candidate_terminal_payload": joint_candidate_terminal_payload,
            "joint_candidate_runtime_audits": audits,
            "joint_candidate_started_at": {},
            "joint_look_scheduler": JointLookBreadthScheduler(),
            "joint_decoded_budget": JointDecodedBudgetLedger(),
            "candidate_derivation_failures": [],
            "candidate_post_repair_review_rejections": [],
            "candidate_decoded_review_failures": [],
            "derived": [],
            "reporter": reporter,
            "monotonic": lambda: next(ticks),
            "persist_candidate_stream_progress": lambda **kw: persisted.append((list(audits), kw)),
        }
        exec(compile(module, str(source), "exec"), namespace)
        return namespace["derive_and_gate_candidate"], stream, audits, persisted

    def test_real_attempt_emits_once_and_persists_before_next_candidate(self):
        run, stream, audits, persisted = self._actual_pipeline_attempt()
        candidate = SimpleNamespace(candidate_index=0)
        verdict = run(candidate)
        self.assertIs(run(candidate), verdict)
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "joint_candidate_terminal")
        self.assertEqual(events[0]["elapsed_seconds"], 4.0)
        self.assertEqual(len(audits), 1)
        self.assertEqual(persisted[0][0], audits)

    def test_real_attempt_preserves_decoder_exception_and_records_system_error(self):
        run, stream, audits, persisted = self._actual_pipeline_attempt(raises=True)
        with self.assertRaisesRegex(ValueError, "test decoder failure"):
            run(SimpleNamespace(candidate_index=0))
        event = json.loads(stream.getvalue())
        self.assertEqual(event["status"], "system_error")
        self.assertEqual(len(audits), 1)
        self.assertEqual(persisted[0][1]["status"], "candidate_stream_system_error")

    def test_closed_output_still_persists_terminal_verdict(self):
        run, stream, audits, persisted = self._actual_pipeline_attempt(broken_output=True)
        with self.assertRaises(BrokenPipeError):
            run(SimpleNamespace(candidate_index=0))
        self.assertEqual(persisted[0][0], audits)
        self.assertEqual(persisted[0][0][0]["status"], "decoded_review_failed")

    def test_closed_output_preserves_original_decoder_failure(self):
        run, stream, audits, persisted = self._actual_pipeline_attempt(
            raises=True, broken_output=True,
        )
        with self.assertRaisesRegex(ValueError, "test decoder failure") as caught:
            run(SimpleNamespace(candidate_index=0))
        self.assertIn("BrokenPipeError", caught.exception.__notes__[0])
        self.assertEqual(persisted[0][0], audits)
        self.assertEqual(persisted[0][1]["status"], "candidate_stream_system_error")


if __name__ == "__main__":
    unittest.main()
