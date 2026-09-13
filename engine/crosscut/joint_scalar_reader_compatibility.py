"""Audited, forward-only decoder equivalences for scalar bridge lookup.

Every other analyzer field is retained verbatim. The ordinary scalar loader
still verifies source content, full analyzer identity and payload checksum.
This does not transfer renderer proofs or authorize unknown model changes.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

H264_READER_ANALYZER = "0d665ac9a9b9ef183e0c666196d63cfd84c77f35fbe3d4c59fac1350c4fa1e8d"
PREVIOUS_READER_ANALYZER = "5e230050f8b6b83a4f208475ebbe0931e8078845cb830eaebd1333bd7f9b9509"
MEASURED_FACE_ANALYZER = "5a8c720efe6d42869df46685bb2aec9650308a1c42b455e0c1101349e072b048"
NATIVE_TRACKING_READER_PROVIDER = "32cee7eb75221d6075e3ba401fcdcb7a20854f168bd62c70c3693af666edd15b"
TIMESTAMP_TRACKING_READER_PROVIDER = "944386cf0a3598f6b5370d1ad5798b38987728d2057f94a7737e23f433a65513"


def previous_reader_scalar_analyzer(analyzer: Mapping[str, Any]) -> dict[str, Any] | None:
    """Change only one known byte-equivalent decoder implementation identity.

    Native tracking equivalence is pinned to both exact provider ASTs and the
    unchanged measured-face analyzer. All source/config/model/segment fields
    retain ordinary checksum-bound lookup. Numeric framing changes are NOT
    migrated. This imports scalar observations, never renderer proofs.
    """
    if analyzer.get("analysis_contract") != "joint-bridge-disposition-scalar-shard-v3":
        return None
    versions = analyzer.get("visual_analyzer_sha256s")
    sources = analyzer.get("source_content_sha256s")
    if not isinstance(versions, list) or not isinstance(sources, list):
        return None
    if len(versions) < 2 or len(versions) != len(sources):
        return None
    implementation = analyzer.get("implementation_sha256")
    if (all(version == MEASURED_FACE_ANALYZER for version in versions)
            and isinstance(implementation, Mapping)
            and implementation.get("provider_class_ast") == NATIVE_TRACKING_READER_PROVIDER):
        result = deepcopy(dict(analyzer))
        result["implementation_sha256"]["provider_class_ast"] = TIMESTAMP_TRACKING_READER_PROVIDER
        return result
    if any(version != H264_READER_ANALYZER for version in versions):
        return None
    result = deepcopy(dict(analyzer))
    result["visual_analyzer_sha256s"] = [PREVIOUS_READER_ANALYZER] * len(versions)
    return result
