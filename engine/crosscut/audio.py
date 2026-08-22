"""Audio decoding and coarse synchronization for cross-cut source videos."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_GROUP = "IVE"
DEFAULT_ANALYSIS_SAMPLE_RATE = 8_000


class AudioDecodeError(RuntimeError):
    """Raised when ffmpeg cannot decode a usable audio stream."""


@dataclass(frozen=True, slots=True)
class OffsetEstimate:
    """Normalized cross-correlation result for one source.

    ``offset_samples`` is a source-timeline coordinate.  A positive value means
    the matching content occurs later in the source than in the reference, so
    reference time ``t`` corresponds to source time ``t + offset_seconds``.
    """

    offset_samples: int
    offset_seconds: float
    score: float
    overlap_samples: int
    overlap_seconds: float
    sample_rate: int
    second_best_score: float | None
    score_margin: float
    overlap_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "offset_samples": self.offset_samples,
            "offset_seconds": self.offset_seconds,
            "score": self.score,
            "overlap_samples": self.overlap_samples,
            "overlap_seconds": self.overlap_seconds,
            "sample_rate": self.sample_rate,
            "second_best_score": self.second_best_score,
            "score_margin": self.score_margin,
            "overlap_ratio": self.overlap_ratio,
        }


@dataclass(frozen=True, slots=True)
class SourceAlignment:
    source: str
    is_reference: bool
    offset_samples: int
    offset_seconds: float
    score: float
    overlap_samples: int
    overlap_seconds: float
    second_best_score: float | None = None
    score_margin: float = 1.0
    overlap_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "is_reference": self.is_reference,
            "offset_samples": self.offset_samples,
            "offset_seconds": self.offset_seconds,
            "score": self.score,
            "overlap_samples": self.overlap_samples,
            "overlap_seconds": self.overlap_seconds,
            "second_best_score": self.second_best_score,
            "score_margin": self.score_margin,
            "overlap_ratio": self.overlap_ratio,
        }


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Alignment of a set of performances against one reference performance."""

    reference: str
    sample_rate: int
    alignments: tuple[SourceAlignment, ...]
    group: str = DEFAULT_GROUP

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "reference": self.reference,
            "sample_rate": self.sample_rate,
            "alignments": [alignment.to_dict() for alignment in self.alignments],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def extract_mono_pcm(
    path: str | os.PathLike[str],
    *,
    sample_rate: int = DEFAULT_ANALYSIS_SAMPLE_RATE,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> NDArray[np.float32]:
    """Decode a local media file to mono, normalized float32 PCM.

    Audio is returned as a one-dimensional NumPy array.  FFmpeg performs
    channel downmixing and resampling, which also makes recordings from
    different broadcasters directly comparable.
    """

    media_path = Path(path).expanduser()
    if not media_path.is_file():
        raise FileNotFoundError(media_path)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if start_seconds is not None and start_seconds < 0:
        raise ValueError("start_seconds cannot be negative")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    command = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start_seconds is not None:
        command.extend(("-ss", f"{start_seconds:.9f}"))
    command.extend(("-i", os.fspath(media_path), "-map", "0:a:0", "-vn"))
    if duration_seconds is not None:
        command.extend(("-t", f"{duration_seconds:.9f}"))
    command.extend(
        (
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        )
    )

    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise AudioDecodeError(f"could not execute {ffmpeg_bin!r}: {exc}") from exc
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(
            f"ffmpeg failed to decode {media_path}: {detail or 'unknown error'}"
        )

    # Copy so the result owns writable memory rather than retaining stdout bytes.
    pcm = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)
    if pcm.size == 0:
        raise AudioDecodeError(f"ffmpeg produced no audio samples for {media_path}")
    return pcm


def _validated_signal(signal: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional signal")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite samples")
    return np.ascontiguousarray(array)


def _fft_dot_correlation(
    reference: NDArray[np.float64], source: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return dot products for lags ``-(len(ref)-1) .. len(source)-1``."""

    output_size = reference.size + source.size - 1
    fft_size = 1 << (output_size - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(source, fft_size) * np.fft.rfft(reference[::-1], fft_size),
        fft_size,
    )
    return correlation[:output_size]


def normalized_cross_correlation_offset(
    reference: ArrayLike,
    source: ArrayLike,
    *,
    sample_rate: int,
    max_offset_seconds: float | None = None,
    min_overlap_seconds: float | None = None,
) -> OffsetEstimate:
    """Estimate a coarse source offset with overlap-normalized correlation.

    Means and variances are calculated independently for every candidate
    overlap.  The result is therefore robust to broadcaster gain changes and DC
    offsets.  FFT convolution supplies the dot products while cumulative sums
    normalize every lag without a Python loop.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if max_offset_seconds is not None and max_offset_seconds < 0:
        raise ValueError("max_offset_seconds cannot be negative")
    if min_overlap_seconds is not None and min_overlap_seconds <= 0:
        raise ValueError("min_overlap_seconds must be positive")

    ref = _validated_signal(reference, name="reference")
    src = _validated_signal(source, name="source")
    shortest = min(ref.size, src.size)
    if min_overlap_seconds is None:
        min_overlap_samples = max(8, shortest // 4)
    else:
        min_overlap_samples = max(2, int(math.ceil(min_overlap_seconds * sample_rate)))
    if min_overlap_samples > shortest:
        raise ValueError("minimum overlap exceeds the shorter signal")

    all_lags = np.arange(-(ref.size - 1), src.size, dtype=np.int64)
    if max_offset_seconds is None:
        selected = np.ones(all_lags.size, dtype=bool)
    else:
        max_offset_samples = int(round(max_offset_seconds * sample_rate))
        selected = np.abs(all_lags) <= max_offset_samples
    lags = all_lags[selected]
    if not lags.size:
        raise ValueError("no candidate offsets remain after applying bounds")

    ref_starts = np.maximum(0, -lags)
    src_starts = np.maximum(0, lags)
    overlaps = np.minimum(ref.size - ref_starts, src.size - src_starts)
    valid_overlap = overlaps >= min_overlap_samples
    if not np.any(valid_overlap):
        raise ValueError("no candidate offset has the requested minimum overlap")

    lags = lags[valid_overlap]
    ref_starts = ref_starts[valid_overlap]
    src_starts = src_starts[valid_overlap]
    overlaps = overlaps[valid_overlap]
    ref_ends = ref_starts + overlaps
    src_ends = src_starts + overlaps

    def cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values)))

    ref_sum_table = cumulative(ref)
    src_sum_table = cumulative(src)
    ref_square_table = cumulative(ref * ref)
    src_square_table = cumulative(src * src)
    ref_sums = ref_sum_table[ref_ends] - ref_sum_table[ref_starts]
    src_sums = src_sum_table[src_ends] - src_sum_table[src_starts]
    ref_squares = ref_square_table[ref_ends] - ref_square_table[ref_starts]
    src_squares = src_square_table[src_ends] - src_square_table[src_starts]

    raw_dot = _fft_dot_correlation(ref, src)[lags + ref.size - 1]
    overlap_float = overlaps.astype(np.float64)
    covariance = raw_dot - (ref_sums * src_sums / overlap_float)
    ref_energy = np.maximum(0.0, ref_squares - ref_sums * ref_sums / overlap_float)
    src_energy = np.maximum(0.0, src_squares - src_sums * src_sums / overlap_float)
    denominator = np.sqrt(ref_energy * src_energy)
    usable = denominator > np.finfo(np.float64).eps
    if not np.any(usable):
        raise ValueError("signals have no varying overlap to correlate")

    scores = np.full(lags.size, -np.inf, dtype=np.float64)
    scores[usable] = covariance[usable] / denominator[usable]
    scores[usable] = np.clip(scores[usable], -1.0, 1.0)
    best_index = int(np.argmax(scores))
    best_lag = int(lags[best_index])
    best_overlap = int(overlaps[best_index])
    # Nearby samples are the same broad correlation peak, not competing song
    # positions.  Excluding half a second exposes chorus/verse aliases that can
    # otherwise look like a confident but wrong synchronization.
    peak_exclusion_samples = max(1, int(round(0.5 * sample_rate)))
    competing = usable & (np.abs(lags - best_lag) > peak_exclusion_samples)
    if np.any(competing):
        second_best_score: float | None = float(np.max(scores[competing]))
        score_margin = float(scores[best_index] - second_best_score)
    else:
        second_best_score = None
        score_margin = float(scores[best_index] + 1.0)
    return OffsetEstimate(
        offset_samples=best_lag,
        offset_seconds=best_lag / sample_rate,
        score=float(scores[best_index]),
        overlap_samples=best_overlap,
        overlap_seconds=best_overlap / sample_rate,
        sample_rate=sample_rate,
        second_best_score=second_best_score,
        score_margin=score_margin,
        overlap_ratio=best_overlap / shortest,
    )


def align_signal_set(
    signals: Mapping[str, ArrayLike],
    *,
    reference: str | None = None,
    sample_rate: int,
    max_offset_seconds: float | None = None,
    min_overlap_seconds: float | None = None,
    group: str = DEFAULT_GROUP,
) -> AlignmentResult:
    """Align named in-memory signals and return a JSON-friendly result."""

    if not signals:
        raise ValueError("signals cannot be empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    reference_name = reference if reference is not None else next(iter(signals))
    if reference_name not in signals:
        raise KeyError(f"reference signal {reference_name!r} is missing")
    if not group.strip():
        raise ValueError("group cannot be blank")

    reference_signal = _validated_signal(signals[reference_name], name="reference")
    alignments: list[SourceAlignment] = [
        SourceAlignment(
            source=reference_name,
            is_reference=True,
            offset_samples=0,
            offset_seconds=0.0,
            score=1.0,
            overlap_samples=int(reference_signal.size),
            overlap_seconds=reference_signal.size / sample_rate,
        )
    ]
    for source_name, signal in signals.items():
        if source_name == reference_name:
            continue
        estimate = normalized_cross_correlation_offset(
            reference_signal,
            signal,
            sample_rate=sample_rate,
            max_offset_seconds=max_offset_seconds,
            min_overlap_seconds=min_overlap_seconds,
        )
        alignments.append(
            SourceAlignment(
                source=source_name,
                is_reference=False,
                offset_samples=estimate.offset_samples,
                offset_seconds=estimate.offset_seconds,
                score=estimate.score,
                overlap_samples=estimate.overlap_samples,
                overlap_seconds=estimate.overlap_seconds,
                second_best_score=estimate.second_best_score,
                score_margin=estimate.score_margin,
                overlap_ratio=estimate.overlap_ratio,
            )
        )
    return AlignmentResult(
        group=group,
        reference=reference_name,
        sample_rate=sample_rate,
        alignments=tuple(alignments),
    )


def align_audio_files(
    reference_path: str | os.PathLike[str],
    source_paths: Sequence[str | os.PathLike[str]],
    *,
    sample_rate: int = DEFAULT_ANALYSIS_SAMPLE_RATE,
    max_offset_seconds: float | None = 30.0,
    min_overlap_seconds: float | None = 5.0,
    group: str = DEFAULT_GROUP,
    ffmpeg_bin: str = "ffmpeg",
) -> AlignmentResult:
    """Decode and align several media files against ``reference_path``."""

    reference_name = os.fspath(Path(reference_path).expanduser().resolve())
    signals: dict[str, NDArray[np.float32]] = {
        reference_name: extract_mono_pcm(
            reference_name, sample_rate=sample_rate, ffmpeg_bin=ffmpeg_bin
        )
    }
    for source_path in source_paths:
        source_name = os.fspath(Path(source_path).expanduser().resolve())
        if source_name in signals:
            continue
        signals[source_name] = extract_mono_pcm(
            source_name, sample_rate=sample_rate, ffmpeg_bin=ffmpeg_bin
        )
    return align_signal_set(
        signals,
        reference=reference_name,
        sample_rate=sample_rate,
        max_offset_seconds=max_offset_seconds,
        min_overlap_seconds=min_overlap_seconds,
        group=group,
    )


# Concise aliases used by orchestration code.
estimate_coarse_offset = normalized_cross_correlation_offset
align_sources = align_audio_files
