#!/usr/bin/env python3
"""Command-line entry point used by the native CrossCut app."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from crosscut.doctor import runtime_report
from crosscut.events import EventReporter
from crosscut.pipeline import analyze
from crosscut.request import AnalysisRequest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crosscut_cli.py")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("doctor", help="check local media dependencies")
    analyze_parser = subcommands.add_parser("analyze", help="analyze IVE music-show sources")
    analyze_parser.add_argument("--request", required=True, type=Path)
    analyze_parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "doctor":
        report = runtime_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 1

    reporter = EventReporter()
    try:
        request = AnalysisRequest.from_json_file(arguments.request)
        reporter.status(
            f"IVE {request.song} · {request.edit_mode.value} · {request.aspect_ratio.value}"
        )
        artifact = analyze(request, reporter)
        warnings = artifact.manifest.get("warnings") or []
        for warning in warnings:
            reporter.emit("log", level="warning", message=warning)
        reporter.result(
            outputs=[str(path) for path in artifact.output_paths],
            message=(
                (
                    "IVE 멤버 교차직캠 초안이 완성되었습니다."
                    if request.edit_mode.value == "member"
                    else "IVE 그룹 교차편집 초안이 완성되었습니다."
                )
                if any(path.suffix.lower() == ".mp4" for path in artifact.output_paths)
                else "IVE 분석은 완료됐지만 안전 게이트 때문에 영상은 만들지 않았습니다."
            ),
        )
        return 0
    except KeyboardInterrupt:
        reporter.error("작업이 취소되었습니다.")
        return 130
    except Exception as exc:  # CLI boundary: convert failures to the JSONL contract.
        reporter.error(str(exc))
        if getattr(arguments, "debug", False):
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
