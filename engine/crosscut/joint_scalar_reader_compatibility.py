"""One audited, forward-only decoder equivalence for scalar bridge lookup.

Every other analyzer field is retained verbatim. The ordinary scalar loader
still verifies source content, full analyzer identity and payload checksum.
This does not transfer renderer proofs or authorize unknown model changes.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

H264_READER_ANALYZER = "0d665ac9a9b9ef183e0c666196d63cfd84c77f35fbe3d4c59fac1350c4fa1e8d"
PREVIOUS_READER_ANALYZER = "5e230050f8b6b83a4f208475ebbe0931e8078845cb830eaebd1333bd7f9b9509"


def previous_reader_scalar_analyzer(analyzer: Mapping[str, Any]) -> dict[str, Any] | None:
    """Change only the known byte-equivalent visual analyzer identity."""
    if analyzer.get("analysis_contract") != "joint-bridge-disposition-scalar-shard-v3":
        return None
    versions = analyzer.get("visual_analyzer_sha256s")
    sources = analyzer.get("source_content_sha256s")
    if not isinstance(versions, list) or not isinstance(sources, list):
        return None
    if len(versions) < 2 or len(versions) != len(sources):
        return None
    if any(version != H264_READER_ANALYZER for version in versions):
        return None
    result = deepcopy(dict(analyzer))
    result["visual_analyzer_sha256s"] = [PREVIOUS_READER_ANALYZER] * len(versions)
    return result
