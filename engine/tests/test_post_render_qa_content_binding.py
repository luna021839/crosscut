from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from crosscut.ive_product_quality_floor import APP_RESULT_EXPOSURE_CONTRACT
from crosscut.pipeline import (
    _enforce_automatic_stage_edit_post_render_qa,
    _persist_qa_input_analysis_snapshot,
    _read_stable_regular_file,
    _write_analysis_manifest_atomically,
)
from crosscut.post_render_qa import (
    INPUT_ARTIFACT_FINGERPRINT_CONTRACT,
    PostRenderArtifactSet,
    fingerprint_post_render_artifacts,
)


class _Reporter:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.logs: list[str] = []

    def status(self, _message: str) -> None:
        self.statuses.append(_message)

    def log(self, _message: str) -> None:
        self.logs.append(_message)


def _qa_snapshot_directories(root: Path) -> tuple[Path, ...]:
    return tuple(root.glob(".IVE-*-qa-snapshot-*"))


def _artifact_shell(root: Path) -> tuple[PostRenderArtifactSet, dict[str, object]]:
    base = "IVE-fixture-20260824-120000"
    artifacts = PostRenderArtifactSet(
        base_name=base,
        timestamp="20260824-120000",
        analysis_path=root / f"{base}-analysis.json",
        render_plan_path=root / f"{base}-render-plan.json",
        selection_path=root / f"{base}-rendered-candidate-selection.json",
        video_path=root / f"{base}-group-crosscut.mp4",
    )
    manifest: dict[str, object] = {
        "request": {
            "group": "IVE",
            "edit_mode": "group",
            "source_mode": "automatic",
        },
        "artifact_role": "pending_qa",
        "app_result_exposure": {
            "contract": APP_RESULT_EXPOSURE_CONTRACT,
            "video_output_exposed": False,
            "delivery_completion_message_allowed": False,
        },
        "render": {"artifact_role": "pending_qa"},
        "stage_edit_generation": {},
    }
    artifacts.analysis_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifacts.render_plan_path.write_text("{}", encoding="utf-8")
    artifacts.selection_path.write_text("{}", encoding="utf-8")
    artifacts.video_path.write_bytes(b"AAAA")
    return artifacts, manifest


def _report(
    fingerprints: dict[str, object],
) -> SimpleNamespace:
    payload = {
        "passed": True,
        "issues": [],
        "issue_count": 0,
        "fail_issue_count": 0,
        "conditional_issue_count": 0,
        "input_artifact_fingerprints": fingerprints,
    }
    return SimpleNamespace(
        verdict=SimpleNamespace(value="PASS"),
        passed=True,
        analysis_basis="fixture_post_render_qa",
        to_dict=lambda: dict(payload),
    )


class PostRenderQAContentBindingTests(unittest.TestCase):
    def test_same_path_same_size_replacement_changes_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifacts, _manifest = _artifact_shell(Path(raw))
            before = fingerprint_post_render_artifacts(artifacts)
            artifacts.video_path.write_bytes(b"BBBB")
            after = fingerprint_post_render_artifacts(artifacts)

        self.assertEqual(
            INPUT_ARTIFACT_FINGERPRINT_CONTRACT, before["contract"]
        )
        self.assertEqual(
            before["video"]["size_bytes"], after["video"]["size_bytes"]
        )
        self.assertNotEqual(
            before["video"]["sha256"], after["video"]["sha256"]
        )

    def test_promotion_rejects_input_changed_after_passing_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifacts, manifest = _artifact_shell(root)
            reporter = _Reporter()

            def evaluate(artifact_set, **_kwargs):
                report = _report(fingerprint_post_render_artifacts(artifact_set))
                report_path = artifact_set.default_report_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                artifact_set.video_path.chmod(0o600)
                artifact_set.video_path.write_bytes(b"BBBB")
                return report, report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "artifact content binding failure"
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=manifest,
                        manifest_path=artifacts.analysis_path,
                        reporter=reporter,  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )

            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            self.assertEqual("rejected", persisted["artifact_role"])
            self.assertFalse(binding["current_inputs_match_evaluated_content"])
            self.assertFalse(binding["passed"])
            canonical_report = artifacts.default_report_path.resolve()
            self.assertEqual(
                str(canonical_report),
                persisted["post_render_qa"]["report_path"],
            )
            self.assertEqual(b"AAAA", artifacts.video_path.read_bytes())
            self.assertEqual((), _qa_snapshot_directories(root))
            self.assertTrue(reporter.logs)
            self.assertIn(str(canonical_report), reporter.logs[-1])

    def test_promotion_rejects_substituted_report_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifacts, manifest = _artifact_shell(Path(raw))

            def evaluate(artifact_set, **_kwargs):
                report = _report(fingerprint_post_render_artifacts(artifact_set))
                report_path = artifact_set.default_report_path
                report_path.write_text('{"verdict":"PASS"}', encoding="utf-8")
                return report, report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "artifact content binding failure"
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=manifest,
                        manifest_path=artifacts.analysis_path,
                        reporter=_Reporter(),  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )

            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            self.assertTrue(binding["current_inputs_match_evaluated_content"])
            self.assertFalse(binding["persisted_report_matches_evaluator_payload"])
            self.assertFalse(binding["passed"])

    def test_promotion_rejects_matching_report_from_substituted_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifacts, manifest = _artifact_shell(root)

            def evaluate(artifact_set, **_kwargs):
                report = _report(fingerprint_post_render_artifacts(artifact_set))
                report_path = root / "unbound-passing-report.json"
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                return report, report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "artifact content binding failure"
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=manifest,
                        manifest_path=artifacts.analysis_path,
                        reporter=_Reporter(),  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )

            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            self.assertFalse(binding["report_path_matches_bound_default"])
            self.assertFalse(
                binding["persisted_report_matches_evaluator_payload"]
            )
            self.assertEqual("rejected", persisted["artifact_role"])

    def test_promotion_rejects_symlinked_bound_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifacts, manifest = _artifact_shell(root)

            def evaluate(artifact_set, **_kwargs):
                report = _report(fingerprint_post_render_artifacts(artifact_set))
                target = root / "report-target.json"
                target.write_text(json.dumps(report.to_dict()), encoding="utf-8")
                artifact_set.default_report_path.symlink_to(target)
                return report, artifact_set.default_report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "artifact content binding failure"
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=manifest,
                        manifest_path=artifacts.analysis_path,
                        reporter=_Reporter(),  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )

            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            self.assertFalse(binding["report_path_matches_bound_default"])
            self.assertEqual("rejected", persisted["artifact_role"])

    def test_valid_promotion_preserves_exact_pre_promotion_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifacts, manifest = _artifact_shell(root)
            reporter = _Reporter()

            def evaluate(artifact_set, **_kwargs):
                report = _report(fingerprint_post_render_artifacts(artifact_set))
                report_path = artifact_set.default_report_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                return report, report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                returned_report_path = _enforce_automatic_stage_edit_post_render_qa(
                    registered_stage_edit_bundle=None,
                    manifest=manifest,
                    manifest_path=artifacts.analysis_path,
                    reporter=reporter,  # type: ignore[arg-type]
                    automatic_ive_group_delivery=True,
                )

            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            snapshot = binding["pre_promotion_analysis_snapshot"]
            snapshot_path = Path(snapshot["path"])
            self.assertEqual("delivery", persisted["artifact_role"])
            self.assertTrue(snapshot_path.is_file())
            self.assertEqual(
                hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                snapshot["sha256"],
            )
            canonical_report = artifacts.default_report_path.resolve()
            self.assertEqual(canonical_report, returned_report_path)
            self.assertEqual(
                str(canonical_report),
                persisted["post_render_qa"]["report_path"],
            )
            self.assertTrue(canonical_report.is_file())
            self.assertEqual((), _qa_snapshot_directories(root))
            self.assertTrue(reporter.logs)
            self.assertIn(str(canonical_report), reporter.logs[-1])

    def test_evaluator_exception_removes_private_snapshot_and_preserves_canonical_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifacts, manifest = _artifact_shell(root)
            observed_snapshot_directories: list[Path] = []

            def evaluate(snapshot_set, **_kwargs):
                observed_snapshot_directories.append(
                    snapshot_set.analysis_path.parent
                )
                self.assertEqual(b"AAAA", snapshot_set.video_path.read_bytes())
                raise RuntimeError("fixture evaluator failure")

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "fixture evaluator failure"
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=manifest,
                        manifest_path=artifacts.analysis_path,
                        reporter=_Reporter(),  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )

            self.assertEqual(1, len(observed_snapshot_directories))
            self.assertFalse(observed_snapshot_directories[0].exists())
            self.assertEqual((), _qa_snapshot_directories(root))
            self.assertEqual(b"AAAA", artifacts.video_path.read_bytes())
            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            self.assertEqual("pending_qa", persisted["artifact_role"])

    def test_promotion_rechecks_inputs_after_snapshot_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifacts, manifest = _artifact_shell(Path(raw))

            def evaluate(artifact_set, **_kwargs):
                report = _report(fingerprint_post_render_artifacts(artifact_set))
                report_path = artifact_set.default_report_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                return report, report_path

            def persist_then_replace_video(path, *, expected_sha256):
                snapshot = _persist_qa_input_analysis_snapshot(
                    path,
                    expected_sha256=expected_sha256,
                )
                frozen_video = path.parent / artifacts.video_path.name
                frozen_video.chmod(0o600)
                frozen_video.write_bytes(b"BBBB")
                return snapshot

            with (
                patch(
                    "crosscut.pipeline.evaluate_and_write_post_render_qa",
                    side_effect=evaluate,
                ),
                patch(
                    "crosscut.pipeline._persist_qa_input_analysis_snapshot",
                    side_effect=persist_then_replace_video,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "artifact content binding failure"
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=manifest,
                        manifest_path=artifacts.analysis_path,
                        reporter=_Reporter(),  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )

            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            self.assertFalse(binding["inputs_stable_through_promotion"])
            self.assertEqual("rejected", persisted["artifact_role"])

    def test_promotion_rechecks_report_after_initial_payload_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifacts, manifest = _artifact_shell(Path(raw))

            def evaluate(artifact_set, **_kwargs):
                report = _report(fingerprint_post_render_artifacts(artifact_set))
                report_path = artifact_set.default_report_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                return report, report_path

            report_read_count = 0

            def read_then_replace_report(path):
                nonlocal report_read_count
                content = _read_stable_regular_file(path)
                if Path(path).name.endswith("-post-render-qa.json"):
                    report_read_count += 1
                    if report_read_count == 1:
                        Path(path).write_text(
                            '{"substituted":true}', encoding="utf-8"
                        )
                return content

            with (
                patch(
                    "crosscut.pipeline.evaluate_and_write_post_render_qa",
                    side_effect=evaluate,
                ),
                patch(
                    "crosscut.pipeline._read_stable_regular_file",
                    side_effect=read_then_replace_report,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "artifact content binding failure"
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=manifest,
                        manifest_path=artifacts.analysis_path,
                        reporter=_Reporter(),  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )

            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            self.assertTrue(
                binding["persisted_report_matches_evaluator_payload"]
            )
            self.assertFalse(binding["report_stable_through_promotion"])
            self.assertEqual("rejected", persisted["artifact_role"])

    def test_canonical_aba_during_evaluation_cannot_change_snapshot_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifacts, manifest = _artifact_shell(Path(raw))
            original_plan = b'{"authority":"A"}'
            artifacts.render_plan_path.write_bytes(original_plan)
            canonical_transient_reads: list[bytes] = []
            evaluated_reads: list[bytes] = []

            def evaluate(snapshot_set, **_kwargs):
                evaluated_reads.append(snapshot_set.render_plan_path.read_bytes())
                artifacts.render_plan_path.write_bytes(b'{"authority":"B"}')
                canonical_transient_reads.append(
                    artifacts.render_plan_path.read_bytes()
                )
                artifacts.render_plan_path.write_bytes(original_plan)
                report = _report(
                    fingerprint_post_render_artifacts(snapshot_set)
                )
                report_path = snapshot_set.default_report_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                return report, report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                _enforce_automatic_stage_edit_post_render_qa(
                    registered_stage_edit_bundle=None,
                    manifest=manifest,
                    manifest_path=artifacts.analysis_path,
                    reporter=_Reporter(),  # type: ignore[arg-type]
                    automatic_ive_group_delivery=True,
                )

            self.assertEqual([original_plan], evaluated_reads)
            self.assertEqual([b'{"authority":"B"}'], canonical_transient_reads)
            self.assertEqual(original_plan, artifacts.render_plan_path.read_bytes())
            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            self.assertEqual("delivery", persisted["artifact_role"])
            self.assertTrue(binding["delivery_materialization"]["performed"])
            self.assertTrue(
                binding["delivery_materialization"]["input_artifacts"]
                ["render_plan"]["inode_identity_preserved"]
            )

    def test_final_canonical_video_swap_is_replaced_by_verified_snapshot_inode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifacts, manifest = _artifact_shell(Path(raw))

            def evaluate(snapshot_set, **_kwargs):
                report = _report(
                    fingerprint_post_render_artifacts(snapshot_set)
                )
                report_path = snapshot_set.default_report_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                # This replacement occurs after the evaluator has produced its
                # PASS payload. Delivery must not trust or re-hash this path;
                # it must overwrite it with the verified snapshot inode.
                artifacts.video_path.write_bytes(b"BBBB")
                return report, report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                _enforce_automatic_stage_edit_post_render_qa(
                    registered_stage_edit_bundle=None,
                    manifest=manifest,
                    manifest_path=artifacts.analysis_path,
                    reporter=_Reporter(),  # type: ignore[arg-type]
                    automatic_ive_group_delivery=True,
                )

            self.assertEqual(b"AAAA", artifacts.video_path.read_bytes())
            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            video = binding["delivery_materialization"]["input_artifacts"][
                "video"
            ]
            self.assertTrue(video["inode_identity_preserved"])
            self.assertEqual(
                video["snapshot_inode"],
                video["canonical_inode"],
            )
            threat_model = binding["evaluation_snapshot"]["threat_model"]
            self.assertFalse(
                threat_model[
                    "same_uid_malicious_snapshot_discovery_fully_prevented"
                ]
            )
            self.assertIn(
                binding["evaluation_snapshot"]["creation_methods"]["video"],
                {
                    "macos_clonefile_copy_on_write",
                    "byte_copy_fallback",
                },
            )

    def test_promotion_rejects_in_memory_manifest_not_evaluated_from_disk(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            artifacts, persisted_manifest = _artifact_shell(Path(raw))
            in_memory_manifest = dict(persisted_manifest)
            in_memory_manifest["unevaluated_tamper"] = True

            def evaluate(artifact_set, **_kwargs):
                report = _report(fingerprint_post_render_artifacts(artifact_set))
                report_path = artifact_set.default_report_path
                report_path.write_text(
                    json.dumps(report.to_dict()), encoding="utf-8"
                )
                return report, report_path

            with patch(
                "crosscut.pipeline.evaluate_and_write_post_render_qa",
                side_effect=evaluate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "artifact content binding failure"
                ):
                    _enforce_automatic_stage_edit_post_render_qa(
                        registered_stage_edit_bundle=None,
                        manifest=in_memory_manifest,
                        manifest_path=artifacts.analysis_path,
                        reporter=_Reporter(),  # type: ignore[arg-type]
                        automatic_ive_group_delivery=True,
                    )

            persisted = json.loads(
                artifacts.analysis_path.read_text(encoding="utf-8")
            )
            binding = persisted["post_render_qa"]["promotion_binding"]
            self.assertEqual("rejected", persisted["artifact_role"])
            self.assertNotIn("unevaluated_tamper", persisted)
            self.assertFalse(
                binding["in_memory_manifest_matches_evaluated_analysis"]
            )

    def test_artifact_fingerprint_rejects_symlinked_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifacts, _manifest = _artifact_shell(root)
            target = root / "video-target.mp4"
            target.write_bytes(artifacts.video_path.read_bytes())
            artifacts.video_path.unlink()
            artifacts.video_path.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "non-symlink"):
                fingerprint_post_render_artifacts(artifacts)

    def test_manifest_atomic_rewrite_does_not_follow_predictable_temp_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            destination = root / "IVE-fixture-20260824-120000-analysis.json"
            destination.write_text('{"old":true}', encoding="utf-8")
            victim = root / "victim.json"
            victim.write_text("do-not-touch", encoding="utf-8")
            predictable = destination.with_name(f".{destination.name}.tmp")
            predictable.symlink_to(victim)

            _write_analysis_manifest_atomically(destination, {"safe": True})

            self.assertEqual("do-not-touch", victim.read_text(encoding="utf-8"))
            self.assertEqual(
                {"safe": True},
                json.loads(destination.read_text(encoding="utf-8")),
            )
            self.assertTrue(predictable.is_symlink())


if __name__ == "__main__":
    unittest.main()
