#!/usr/bin/env python3
"""Command-line entry point used by the native CrossCut app."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback

# Random-access analysis intentionally seeks between non-keyframes. OpenCV's
# bundled FFmpeg can print harmless H.264 reference warnings for those seeks;
# actual read failures are handled by the frame readers and full media probes.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

from crosscut.doctor import runtime_report
from crosscut.acquisition import acquire_sources
from crosscut.events import EventReporter, ProgressRangeReporter
from crosscut.ive import SourceMode
from crosscut.pipeline import analyze
from crosscut.post_render_qa import (
    PostRenderArtifactSet,
    PostRenderQAVerdict,
    discover_completed_artifact_sets,
    evaluate_and_write_post_render_qa,
)
from crosscut.professional_reference_gate import (
    ProfessionalReferenceProfile,
    evaluate_professional_reference_review,
)
from crosscut.reference_attribution import resolve_reference_attribution_file
from crosscut.reference_boundary_supervision import (
    augment_reference_boundary_requests_file,
)
from crosscut.reference_dataset import build_reference_dataset_file
from crosscut.request import AnalysisRequest


def _raise_analysis_cancellation(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _prepare_analysis_process_group() -> bool:
    """Make app termination cover this CLI and every media-tool child."""

    if os.name != "posix":
        return False
    if os.getpgrp() != os.getpid():
        os.setpgid(0, 0)
    signal.signal(signal.SIGTERM, _raise_analysis_cancellation)
    return os.getpgrp() == os.getpid()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_proxy_review_from_document(
    document: dict[str, object],
) -> tuple[dict[str, object], str]:
    """Accept either a raw review or a no-rerender QA replay audit."""

    if isinstance(document.get("boundaries"), list):
        return (document, "root")
    nested = document.get("decoded_proxy_review")
    if isinstance(nested, dict) and isinstance(nested.get("boundaries"), list):
        return (nested, "decoded_proxy_review")
    raise ValueError(
        "review JSON must be a decoded proxy review or contain "
        "decoded_proxy_review"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crosscut_cli.py")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("doctor", help="check local media dependencies")
    analyze_parser = subcommands.add_parser("analyze", help="analyze IVE music-show sources")
    analyze_parser.add_argument("--request", required=True, type=Path)
    analyze_parser.add_argument("--debug", action="store_true")
    analyze_parser.add_argument(
        "--allow-failed-comparison-render",
        action="store_true",
        help=(
            "explicitly render an automatic GROUP comparison even when visual "
            "timeline generation or final timeline QA fails"
        ),
    )
    dataset_parser = subcommands.add_parser(
        "build-reference-dataset",
        help="convert conservative stage-mix attribution into training records",
    )
    dataset_parser.add_argument("--attribution", required=True, type=Path)
    dataset_parser.add_argument("--output", required=True, type=Path)
    dataset_parser.add_argument("--style-id", required=True)
    dataset_parser.add_argument("--reference-id")
    attribution_parser = subcommands.add_parser(
        "resolve-reference-attribution",
        help=(
            "resolve synchronized visual evidence with UNKNOWN-aware temporal DP"
        ),
    )
    attribution_parser.add_argument("--evidence", required=True, type=Path)
    attribution_parser.add_argument("--output", required=True, type=Path)
    boundary_parser = subcommands.add_parser(
        "augment-reference-boundaries",
        help=(
            "add only UNKNOWN-aware frame-proven A-to-B boundary requests and "
            "rerun attribution"
        ),
    )
    boundary_parser.add_argument("--evidence", required=True, type=Path)
    boundary_parser.add_argument("--output", required=True, type=Path)
    boundary_parser.add_argument(
        "--attribution-output",
        type=Path,
        help="optional destination for the post-augmentation attribution",
    )
    professional_gate_parser = subcommands.add_parser(
        "professional-reference-gate",
        help=(
            "compare a decoded proxy review with five same-song human edits"
        ),
    )
    professional_gate_parser.add_argument(
        "--profile",
        required=True,
        type=Path,
    )
    professional_gate_parser.add_argument(
        "--review",
        required=True,
        type=Path,
    )
    professional_gate_parser.add_argument(
        "--output",
        type=Path,
        help="optional destination for the complete relative-gate report",
    )
    post_render_parser = subcommands.add_parser(
        "post-render-qa",
        help="fully decode and professionally audit one completed GROUP render",
    )
    artifact_source = post_render_parser.add_mutually_exclusive_group(required=True)
    artifact_source.add_argument(
        "--analysis",
        type=Path,
        help="exact timestamp-bound *-analysis.json artifact",
    )
    artifact_source.add_argument(
        "--output-directory",
        type=Path,
        help="directory whose latest completed timestamp-bound artifact set is audited",
    )
    post_render_parser.add_argument(
        "--report",
        type=Path,
        help="optional report destination (defaults beside the selected artifacts)",
    )
    post_render_parser.add_argument("--debug", action="store_true")
    return parser


def _completed_artifact_from_analysis(
    analysis_path: Path,
) -> PostRenderArtifactSet:
    requested = PostRenderArtifactSet.from_analysis_path(analysis_path)
    requested_path = requested.analysis_path.resolve()
    completed = discover_completed_artifact_sets(requested.output_directory)
    for artifact_set in completed:
        if artifact_set.analysis_path.resolve() == requested_path:
            return artifact_set
    raise ValueError(
        "지정한 analysis와 같은 타임스탬프의 완성된 "
        "analysis/render-plan/selection/MP4 세트를 찾지 못했습니다."
    )


def _resolve_post_render_artifact_set(
    *,
    analysis_path: Path | None,
    output_directory: Path | None,
) -> tuple[PostRenderArtifactSet, int]:
    if analysis_path is not None:
        return (_completed_artifact_from_analysis(analysis_path), 1)
    if output_directory is None:  # argparse enforces this; retain a safe API boundary.
        raise ValueError("analysis 또는 output directory가 필요합니다.")
    completed = discover_completed_artifact_sets(output_directory)
    if not completed:
        raise ValueError(
            "출력 폴더에서 같은 타임스탬프의 완성된 "
            "analysis/render-plan/selection/MP4 세트를 찾지 못했습니다."
        )
    return (completed[-1], len(completed))


def _run_post_render_qa(arguments: argparse.Namespace, reporter: EventReporter) -> int:
    reporter.status("완성 렌더의 후처리 QA 대상을 확인하는 중…")
    reporter.progress(0.02, "타임스탬프가 일치하는 산출물 세트를 찾는 중…")
    artifact_set, completed_count = _resolve_post_render_artifact_set(
        analysis_path=arguments.analysis,
        output_directory=arguments.output_directory,
    )
    reporter.progress(
        0.08,
        f"{artifact_set.base_name} 산출물 4종을 확인했습니다.",
    )
    if arguments.output_directory is not None and completed_count > 1:
        reporter.emit(
            "log",
            level="info",
            message=(
                f"완성 세트 {completed_count}개 중 최신 세트 "
                f"{artifact_set.base_name}을 선택했습니다."
            ),
        )
    reporter.progress(
        0.12,
        "ffprobe 검사와 최종 MP4 전체 프레임 디코드를 시작합니다.",
    )
    report, report_path = evaluate_and_write_post_render_qa(
        artifact_set,
        output_path=arguments.report,
    )
    payload = report.to_dict()
    reporter.progress(
        1.0,
        f"후처리 QA가 {report.verdict.value} 판정으로 완료됐습니다.",
    )
    reporter.result(
        outputs=[str(report_path)],
        qa_report=str(report_path),
        video_path=str(artifact_set.video_path.resolve()),
        artifact_set=artifact_set.to_dict(),
        verdict=report.verdict.value,
        passed=report.passed,
        issue_count=payload["issue_count"],
        fail_issue_count=payload["fail_issue_count"],
        conditional_issue_count=payload["conditional_issue_count"],
        message=(
            f"후처리 QA {report.verdict.value}: "
            f"{Path(report_path).name}"
        ),
    )
    if report.verdict is PostRenderQAVerdict.FAIL:
        reporter.error(
            "완성 렌더가 후처리 QA의 필수 품질 조건을 통과하지 못했습니다. "
            f"상세 보고서: {report_path}"
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    _prepare_analysis_process_group()
    if arguments.command == "doctor":
        report = runtime_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 1

    if arguments.command == "build-reference-dataset":
        output = build_reference_dataset_file(
            arguments.attribution,
            arguments.output,
            style_id=arguments.style_id,
            reference_id=arguments.reference_id,
        )
        print(output)
        return 0

    if arguments.command == "resolve-reference-attribution":
        output = resolve_reference_attribution_file(
            arguments.evidence,
            arguments.output,
        )
        print(output)
        return 0

    if arguments.command == "augment-reference-boundaries":
        augmentation, evidence_output, attribution_output = (
            augment_reference_boundary_requests_file(
                arguments.evidence,
                arguments.output,
                attribution_output_path=arguments.attribution_output,
            )
        )
        print(
            json.dumps(
                {
                    "analysis": "unknown_aware_reference_boundary_augmentation_v1",
                    "evidence_output": str(evidence_output),
                    "attribution_output": (
                        str(attribution_output)
                        if attribution_output is not None
                        else None
                    ),
                    "added_request_count": augmentation.added_request_count,
                    "added_resolved_hard_cut_count": (
                        augmentation.added_resolved_hard_cut_count
                    ),
                    "total_resolved_hard_cut_count": (
                        augmentation.resolved_hard_cut_count
                    ),
                    "skip_reason_counts": dict(
                        augmentation.skip_reason_counts
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if arguments.command == "professional-reference-gate":
        profile = ProfessionalReferenceProfile.from_file(arguments.profile)
        review_document = json.loads(arguments.review.read_text(encoding="utf-8"))
        if not isinstance(review_document, dict):
            raise ValueError("proxy review JSON root must be an object")
        raw_review, review_container_field = _decoded_proxy_review_from_document(
            review_document
        )
        gate = evaluate_professional_reference_review(raw_review, profile)
        gate.payload["candidate_evidence"] = {
            "document_path": str(arguments.review.expanduser().resolve()),
            "document_sha256": _sha256_file(arguments.review),
            "review_container_field": review_container_field,
            "no_rerender": True,
        }
        output = gate.write(arguments.output) if arguments.output else None
        print(
            json.dumps(
                {
                    "analysis": gate.payload["analysis"],
                    "passed": gate.passed,
                    "profile_path": str(profile.profile_path),
                    "profile_sha256": profile.profile_sha256,
                    "reference_count": profile.reference_count,
                    "reference_cut_count": profile.reference_cut_count,
                    "candidate_visible_cut_count": gate.payload[
                        "candidate_visible_cut_count"
                    ],
                    "technical_issue_count": gate.payload[
                        "technical_integrity"
                    ]["issue_count"],
                    "failed_components": [
                        name
                        for name, result in gate.payload["components"].items()
                        if not bool(result["passed"])
                    ],
                    "output": str(output) if output is not None else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if gate.passed else 1

    reporter = EventReporter()
    try:
        if arguments.command == "post-render-qa":
            return _run_post_render_qa(arguments, reporter)
        request = AnalysisRequest.from_json_file(arguments.request)
        reporter.status(
            f"IVE {request.song} · {request.edit_mode.value} · {request.aspect_ratio.value}"
        )
        acquisition = None
        if request.source_mode is SourceMode.AUTOMATIC:
            acquisition = acquire_sources(request, reporter)
            if request.acquire_only:
                reporter.progress(1.0, "IVE 소스 수집을 완료했습니다.")
                source_paths = [str(path) for path in acquisition.source_paths]
                reporter.result(
                    outputs=source_paths,
                    source_paths=source_paths,
                    acquisition_manifest=str(acquisition.manifest_path),
                    member_anchors=[
                        anchor.as_dict() for anchor in acquisition.member_anchors
                    ],
                    message=(
                        f"IVE {request.song} 소스 {len(source_paths)}개를 "
                        "로컬 캐시에 준비했습니다."
                    ),
                )
                return 0
            request = request.with_materialized_sources(
                acquisition.source_paths,
                member_anchors=acquisition.member_anchors,
            )
            pipeline_reporter = ProgressRangeReporter(
                reporter,
                start=0.20,
                end=1.0,
            )
        else:
            pipeline_reporter = reporter
        artifact = analyze(
            request,
            pipeline_reporter,
            allow_failed_comparison_render=(
                arguments.allow_failed_comparison_render
            ),
            review_rendered_proxies=(
                request.source_mode is SourceMode.AUTOMATIC
                and request.edit_mode.value == "group"
            ),
        )
        warnings = artifact.manifest.get("warnings") or []
        for warning in warnings:
            reporter.emit("log", level="warning", message=warning)
        reporter.result(
            outputs=[str(path) for path in artifact.output_paths],
            source_paths=(
                [str(path) for path in acquisition.source_paths]
                if acquisition is not None
                else [str(path) for path in request.sources]
            ),
            acquisition_manifest=(
                str(acquisition.manifest_path) if acquisition is not None else None
            ),
            member_anchors=(
                [anchor.as_dict() for anchor in acquisition.member_anchors]
                if acquisition is not None
                else [anchor.as_dict() for anchor in request.member_anchors]
            ),
            message=(
                (
                    "IVE 멤버 교차직캠이 완성되었습니다."
                    if request.edit_mode.value == "member"
                    else "IVE 그룹 교차편집이 완성되었습니다."
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
