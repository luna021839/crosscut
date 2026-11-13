"""Keep one completed failed proxy per run for reproducible pixel diagnosis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .render import RenderPlan, write_plan


class FailedProxyDiagnosticStore:
    def __init__(self, root: Path, *, maximum_video_bytes: int = 512 * 1024 * 1024):
        if type(maximum_video_bytes) is not int or maximum_video_bytes < 1:
            raise ValueError("diagnostic video limit must be positive")
        self.root = Path(root)
        self.maximum_video_bytes = maximum_video_bytes
        self._attempted = False

    def __call__(self, path: Path, plan: RenderPlan,
                 review: Mapping[str, Any]) -> dict[str, Any]:
        if review.get("passed") is not False:
            return {"retained": False, "reason": "not_a_failed_review"}
        if self._attempted:
            return {"retained": False, "reason": "one_diagnostic_per_run_limit"}
        self._attempted = True
        state = path.stat()
        if not 0 < state.st_size <= self.maximum_video_bytes:
            return {"retained": False, "reason": "video_outside_diagnostic_size_limit"}
        # Serialize before copying, so an invalid review cannot leave a large
        # orphaned video. This is diagnostic evidence, never production output.
        review_json = json.dumps(dict(review), ensure_ascii=False, indent=2)
        if len(review_json.encode("utf-8")) > 16 * 1024 * 1024:
            return {"retained": False, "reason": "review_outside_diagnostic_size_limit"}
        self.root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="failed-proxy-", dir=self.root))
        partial = directory / "diagnostic.mp4.partial"
        video = directory / "diagnostic.mp4"
        digest = hashlib.sha256()
        copied = 0
        with path.open("rb") as source, partial.open("xb") as output:
            while chunk := source.read(1024 * 1024):
                copied += len(chunk)
                if copied > self.maximum_video_bytes:
                    raise OSError("completed diagnostic proxy grew during retention")
                digest.update(chunk)
                output.write(chunk)
        after = path.stat()
        if (copied, after.st_size, after.st_mtime_ns) != (
            state.st_size, state.st_size, state.st_mtime_ns
        ):
            raise OSError("completed diagnostic proxy changed during retention")
        write_plan(plan, directory / "production-render-plan.json")
        (directory / "decoded-review.json").write_text(review_json, encoding="utf-8")
        partial.rename(video)
        audit = dict(contract="one_failed_decoded_proxy_diagnostic_v1",
                     retained=True, path=str(video), video_bytes=copied,
                     video_sha256=digest.hexdigest(),
                     review_path=str(directory / "decoded-review.json"),
                     render_plan_path=str(directory / "production-render-plan.json"),
                     quality_accepted=False, final_media_authority=False)
        (directory / "manifest.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8")
        return audit
