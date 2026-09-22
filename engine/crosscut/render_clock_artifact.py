"""The render clock is a required, content-bound input of modern output QA."""
import hashlib
import json
from pathlib import Path
import stat

RENDER_CLOCK_ARTIFACT_CONTRACT = "render_clock_video_plan_binding_v2"


def render_clock_path(video_path: Path) -> Path:
    video = Path(video_path)
    return video.with_suffix(video.suffix + ".frame-clock.json")


def video_content_sha256(path: Path) -> str:
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise ValueError("render clock requires a nonempty regular video")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise ValueError("video changed during render clock binding")
    return digest


def bound_render_clock_path(video_path: Path, plan_path: Path) -> Path | None:
    """Legacy artifact sets are explicit in their plan; modern clocks cannot vanish."""
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    required = isinstance(payload, dict) and payload.get("frame_clock_contract") == RENDER_CLOCK_ARTIFACT_CONTRACT
    sidecar = render_clock_path(video_path)
    if sidecar.exists() or sidecar.is_symlink():
        return sidecar
    if required:
        raise ValueError("modern render is missing its required frame clock artifact")
    return None
