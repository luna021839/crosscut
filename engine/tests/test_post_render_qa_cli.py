from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import crosscut_cli
from crosscut.post_render_qa import (
    PostRenderArtifactSet,
    PostRenderQAConfig,
    PostRenderQAReport,
    PostRenderQAVerdict,
    QAIssue,
    QAIssueSeverity,
)


def _complete_artifact_set(root: Path, timestamp: str) -> PostRenderArtifactSet:
    base = f"IVE-TEST-{timestamp}"
    paths = {
        "analysis_path": root / f"{base}-analysis.json",
        "render_plan_path": root / f"{base}-render-plan.json",
        "selection_path": root / f"{base}-rendered-candidate-selection.json",
        "video_path": root / f"{base}-group-crosscut.mp4",
    }
    for path in paths.values():
        path.write_bytes(b"fixture")
    return PostRenderArtifactSet(
        base_name=base,
        timestamp=timestamp,
        **paths,
    )


def _report(
    artifact_set: PostRenderArtifactSet,
    verdict: PostRenderQAVerdict,
) -> PostRenderQAReport:
    issues = ()
    if verdict is PostRenderQAVerdict.FAIL:
        issues = (
            QAIssue(
                code="fixture_failure",
                severity=QAIssueSeverity.FAIL,
                message="fixture failure",
            ),
        )
    elif verdict is PostRenderQAVerdict.CONDITIONAL:
        issues = (
            QAIssue(
                code="fixture_conditional",
                severity=QAIssueSeverity.CONDITIONAL,
                message="fixture conditional",
            ),
        )
    return PostRenderQAReport(
        artifact_set=artifact_set,
        verdict=verdict,
        metrics={},
        issues=issues,
        policy=PostRenderQAConfig(),
    )


def _events(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


class PostRenderQACLIParserTests(unittest.TestCase):
    def test_requires_exact_analysis_or_output_directory(self) -> None:
        parser = crosscut_cli._parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["post-render-qa"])
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "post-render-qa",
                    "--analysis",
                    "one-analysis.json",
                    "--output-directory",
                    "out",
                ]
            )


class PostRenderQACLITests(unittest.TestCase):
    def _run(
        self,
        argv: list[str],
        *,
        verdict: PostRenderQAVerdict,
    ) -> tuple[int, list[dict[str, object]], object]:
        stream = io.StringIO()

        def evaluate(artifact_set, *, output_path=None):  # type: ignore[no-untyped-def]
            destination = Path(output_path or artifact_set.default_report_path)
            return (_report(artifact_set, verdict), destination.resolve())

        with patch(
            "crosscut_cli._prepare_analysis_process_group",
            return_value=True,
        ), patch(
            "crosscut_cli.evaluate_and_write_post_render_qa",
            side_effect=evaluate,
        ) as evaluate_mock, redirect_stdout(stream):
            exit_code = crosscut_cli.main(argv)
        return (exit_code, _events(stream), evaluate_mock)

    def test_output_directory_selects_latest_complete_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _complete_artifact_set(root, "20260823-010101")
            latest = _complete_artifact_set(root, "20260823-020202")
            incomplete = root / "IVE-TEST-20260823-030303-analysis.json"
            incomplete.write_bytes(b"fixture")

            code, events, evaluate_mock = self._run(
                ["post-render-qa", "--output-directory", str(root)],
                verdict=PostRenderQAVerdict.PASS,
            )

        self.assertEqual(code, 0)
        selected = evaluate_mock.call_args.args[0]
        self.assertEqual(selected.base_name, latest.base_name)
        self.assertTrue(any(event["type"] == "progress" for event in events))
        result = next(event for event in events if event["type"] == "result")
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["video_path"], str(latest.video_path.resolve()))
        self.assertFalse(any(event["type"] == "error" for event in events))

    def test_exact_analysis_writes_custom_report_and_conditional_is_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _complete_artifact_set(root, "20260823-040404")
            destination = root / "reports" / "audit.json"

            code, events, evaluate_mock = self._run(
                [
                    "post-render-qa",
                    "--analysis",
                    str(artifacts.analysis_path),
                    "--report",
                    str(destination),
                ],
                verdict=PostRenderQAVerdict.CONDITIONAL,
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            evaluate_mock.call_args.kwargs["output_path"],
            destination,
        )
        result = next(event for event in events if event["type"] == "result")
        self.assertEqual(result["verdict"], "CONDITIONAL")
        self.assertEqual(result["outputs"], [str(destination.resolve())])

    def test_fail_report_is_emitted_and_process_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = _complete_artifact_set(root, "20260823-050505")

            code, events, _evaluate_mock = self._run(
                ["post-render-qa", "--analysis", str(artifacts.analysis_path)],
                verdict=PostRenderQAVerdict.FAIL,
            )

        self.assertEqual(code, 1)
        result = next(event for event in events if event["type"] == "result")
        error = next(event for event in events if event["type"] == "error")
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["fail_issue_count"], 1)
        self.assertIn("상세 보고서", error["message"])

    def test_incomplete_exact_set_fails_before_qa_with_jsonl_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            analysis = Path(directory) / "IVE-TEST-20260823-060606-analysis.json"
            analysis.write_bytes(b"fixture")
            stream = io.StringIO()
            with patch(
                "crosscut_cli._prepare_analysis_process_group",
                return_value=True,
            ), patch(
                "crosscut_cli.evaluate_and_write_post_render_qa",
            ) as evaluate_mock, redirect_stdout(stream):
                code = crosscut_cli.main(
                    ["post-render-qa", "--analysis", str(analysis)]
                )

        self.assertEqual(code, 1)
        evaluate_mock.assert_not_called()
        events = _events(stream)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("완성된", events[-1]["message"])

    def test_qa_runtime_error_uses_jsonl_error_and_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = _complete_artifact_set(
                Path(directory),
                "20260823-070707",
            )
            stream = io.StringIO()
            with patch(
                "crosscut_cli._prepare_analysis_process_group",
                return_value=True,
            ), patch(
                "crosscut_cli.evaluate_and_write_post_render_qa",
                side_effect=RuntimeError("synthetic ffmpeg launch failure"),
            ), redirect_stdout(stream):
                code = crosscut_cli.main(
                    [
                        "post-render-qa",
                        "--analysis",
                        str(artifacts.analysis_path),
                    ]
                )

        self.assertEqual(code, 1)
        events = _events(stream)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(
            events[-1]["message"],
            "synthetic ffmpeg launch failure",
        )


if __name__ == "__main__":
    unittest.main()
