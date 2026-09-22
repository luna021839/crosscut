"""Separate completing a render interval from completing the original song."""
from __future__ import annotations

import math


def render_coverage_payload(plan, *, common_start: float, common_end: float) -> dict:
    if not all(math.isfinite(value) for value in (common_start, common_end)) or common_end <= common_start:
        raise ValueError("common render interval must be finite and positive")
    start = plan.master_audio_start
    end = start + plan.render_duration
    tolerance = 1.0 / plan.fps + 1e-6
    return {
        "contract": "render_interval_coverage_v1",
        "basis": "selected_common_master_interval_not_verified_canonical_song",
        "common_master_start_seconds": common_start,
        "common_master_end_seconds": common_end,
        "rendered_master_start_seconds": start,
        "rendered_master_end_seconds": end,
        "covers_common_master_interval": (
            abs(start - common_start) <= tolerance and abs(end - common_end) <= tolerance
        ),
        "canonical_song_coverage_verified": False,
        "canonical_song_coverage_status": "UNKNOWN",
        "canonical_song_duration_seconds": None,
        "full_song_delivery_claim_allowed": False,
        "unknown_reason": "no_independent_canonical_song_range_was_supplied",
    }
