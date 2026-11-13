"""Reuse explicitly supplied private photo references as IVE member hints.

This is NOT the metadata-authorized fancam gallery in ive_face_identity_gallery.
User-labelled photographs retain their own provenance and do not acquire that
gallery's source authority. References and query vectors remain local. A hint
can inform anonymous actor/cut selection, never certify choreography or quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import stat

import numpy as np

from .ive import IVEMember
from .native_face_descriptor import (
    ARCFACE_MODEL_NAME, ARCFACE_R50_SHA256, NativeFaceDescriptorSpace,
    native_descriptor_space,
)

CONTRACT = "user_labelled_local_ive_arcface_hint_v1"
SCORING_POLICY = "bounded_unit_vectors_explicit_einsum_topk_mean_v1"
_MAX_BANK_BYTES = 16 * 1024 * 1024
_MAX_REFERENCES = 2048


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _identity(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("local face references must be a regular nonsymlink file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _unique_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate local face reference JSON key")
        result[name] = value
    return result


def _read_bank(path):
    path = Path(path)
    before = _identity(path)
    if not 0 < before[2] <= _MAX_BANK_BYTES:
        raise ValueError("local face reference bank exceeds its declared byte bound")
    with path.open("rb") as stream:
        payload = stream.read(_MAX_BANK_BYTES + 1)
    if len(payload) != before[2] or _identity(path) != before:
        raise ValueError("local face reference bank changed during loading")
    value = json.loads(payload, object_pairs_hook=_unique_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("nonfinite local face reference JSON")))
    if not isinstance(value, dict):
        raise ValueError("local face references need a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _unit_vector(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 512
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError("local ArcFace vectors must contain exactly 512 finite values")
    peak = max(abs(v) for v in value)
    if peak == 0:
        raise ValueError("local ArcFace vector is empty")
    # Scaling first also handles large but finite JSON numbers without an
    # overflowing squared norm. Do not ask platform BLAS to validate input.
    scaled = tuple(v / peak for v in value)
    norm = math.sqrt(math.fsum(v * v for v in scaled))
    if peak * norm <= 1e-12:
        raise ValueError("local ArcFace vector is empty")
    return tuple(v / norm for v in scaled)


@dataclass(frozen=True, slots=True)
class LocalIVEArcFaceGallery:
    """Immutable six-member private input, never persisted by this adapter."""
    references: tuple = field(repr=False)
    bank_sha256: tuple[str, ...]
    minimum_similarity: float = .45
    minimum_member_margin: float = .08
    consensus_reference_count: int = 3
    _matrices: tuple = field(init=False, repr=False, compare=False)
    _identity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (type(self.minimum_similarity) not in (int, float)
                or type(self.minimum_member_margin) not in (int, float)
                or not 0 <= self.minimum_similarity <= 1
                or not 0 < self.minimum_member_margin <= 1
                or type(self.consensus_reference_count) is not int
                or not 1 <= self.consensus_reference_count <= 8
                or not isinstance(self.references, tuple)
                or tuple(member for member, _ in self.references) != tuple(IVEMember)
                or not isinstance(self.bank_sha256, tuple) or not self.bank_sha256
                or any(not isinstance(s, str) or len(s) != 64
                    or any(c not in "0123456789abcdef" for c in s) for s in self.bank_sha256)):
            raise ValueError("local IVE gallery needs six declared members and bounded hint gates")
        matrices = []; normalized = []; total = 0
        for member, vectors in self.references:
            if not isinstance(vectors, tuple) or not vectors:
                raise ValueError("local IVE gallery omitted a member reference")
            total += len(vectors)
            if total > _MAX_REFERENCES:
                raise ValueError("local IVE gallery exceeds its reference count bound")
            clean = tuple(_unit_vector(v) for v in vectors)
            matrix = np.asarray(clean, dtype=np.float64)
            matrix.setflags(write=False)
            matrices.append((member, matrix)); normalized.append((member, clean))
        object.__setattr__(self, "references", tuple(normalized))
        object.__setattr__(self, "_matrices", tuple(matrices))
        object.__setattr__(self, "_identity", _digest(dict(
            contract=CONTRACT, scoring_policy=SCORING_POLICY, banks=self.bank_sha256, references=[
                dict(member=member.value, vector_sha256=[_digest(v) for v in vectors])
                for member, vectors in normalized], gates=(self.minimum_similarity,
                    self.minimum_member_margin, self.consensus_reference_count))))

    @property
    def identity(self):
        return self._identity

    @classmethod
    def from_existing_photo_banks(cls, member_bank_path, yujin_bank_path, **gates):
        members, members_sha = _read_bank(member_bank_path)
        yujin, yujin_sha = _read_bank(yujin_bank_path)
        if yujin.get("model_sha256") != ARCFACE_R50_SHA256:
            raise ValueError("photo reference bank belongs to another ArcFace model")
        if members.get("model_sha256") not in (None, ARCFACE_R50_SHA256):
            raise ValueError("member photo reference bank belongs to another ArcFace model")
        galleries = members.get("galleries")
        if not isinstance(galleries, dict):
            raise ValueError("photo reference bank has no member galleries")
        by_member = {}
        for name, rows in galleries.items():
            member = IVEMember.parse(name)
            if member is IVEMember.ANYUJIN or member in by_member:
                raise ValueError("photo reference member aliases overlap")
            if not isinstance(rows, list) or len(rows) > _MAX_REFERENCES:
                raise ValueError("photo reference rows exceed the declared count bound")
            # Preserve the source tool's explicit rejected-reference decision.
            # No query-time threshold can promote its discarded outliers.
            by_member[member] = tuple(_unit_vector(row["embedding"]) for row in rows
                if row.get("reference_core") is True)
        rows = yujin.get("references")
        if not isinstance(rows, list) or len(rows) > _MAX_REFERENCES:
            raise ValueError("Yujin photo reference rows are absent or unbounded")
        by_member[IVEMember.ANYUJIN] = tuple(_unit_vector(row["embedding"]) for row in rows
            if row.get("reference_core") is True)
        if set(by_member) != set(IVEMember):
            raise ValueError("photo reference banks must cover all six IVE members")
        return cls(tuple((member, by_member[member]) for member in IVEMember),
            (members_sha, yujin_sha), **gates)

    def classify(self, vector, *, descriptor_space):
        if (not isinstance(descriptor_space, NativeFaceDescriptorSpace)
                or descriptor_space.model_name != ARCFACE_MODEL_NAME
                or descriptor_space.model_sha256 != ARCFACE_R50_SHA256
                or descriptor_space.dimension != 512):
            raise ValueError("local member hints cannot compare SFace and ArcFace coordinates")
        scores = []
        if vector is not None:
            query = np.asarray(_unit_vector(vector), dtype=np.float64)
            for member, references in self._matrices:
                # macOS Accelerate matmul can emit spurious floating-point
                # flags even for these bounded unit vectors. A small explicit
                # contraction avoids that backend, with a fail-closed result
                # check rather than clamping NaN into a false strong match.
                with np.errstate(all="raise"):
                    values = np.einsum("ij,j->i", references, query, optimize=False)
                if (not np.isfinite(values).all()
                        or np.any(np.abs(values) > 1. + 1e-8)):
                    raise ValueError("local member cosine calculation is invalid")
                count = min(self.consensus_reference_count, len(values))
                score = float(np.mean(np.sort(values)[-count:]))
                scores.append((member, max(-1., min(1., score))))
        ranked = sorted(scores, key=lambda row: (-row[1], row[0].value))
        margin = None if not ranked else ranked[0][1] - ranked[1][1]
        likely = (bool(ranked) and ranked[0][1] >= self.minimum_similarity
            and margin >= self.minimum_member_margin)
        return dict(contract=CONTRACT, state="LIKELY_MEMBER" if likely else "UNKNOWN",
            member=ranked[0][0].value if likely else None,
            best_similarity=None if not ranked else ranked[0][1], member_margin=margin,
            member_scores={member.value: score for member, score in scores},
            gallery_identity=self.identity, descriptor_space=descriptor_space.to_dict(),
            multi_reference_consensus=True, label_provenance="user_labelled_photo_references",
            stage_video_thresholds_calibrated=False, named_member_identity_certified=False,
            persistent_actor_identity_certified=False, quality_approved=False)

    def summary(self):
        return dict(contract=CONTRACT, gallery_identity=self.identity,
            member_reference_counts={member.value: len(rows) for member, rows in self.references},
            raw_references_or_embeddings_exported=False, references_copied=False,
            stage_video_thresholds_calibrated=False, quality_approved=False)


class LocalIVEArcFaceFeatureCache:
    """Bounded derived native features; no photograph/reference bank copies.

    The cache only saves the queries that were actually inferred. Import does
    not authorize a raw observation: get(raw) checks the complete parent,
    every head and the current model/adapter/gallery before reuse. Conflicting
    results are rejected, never resolved by choosing a better member score.
    Disk persistence is explicit and owned by the request, not this class.
    """
    def __init__(self, gallery, descriptor_space, *, maximum_rows=8192,
            maximum_bytes=64*1024*1024):
        if (not isinstance(gallery, LocalIVEArcFaceGallery)
                or not isinstance(descriptor_space, NativeFaceDescriptorSpace)
                or descriptor_space.model_name != ARCFACE_MODEL_NAME
                or type(maximum_rows) is not int or not 1 <= maximum_rows <= 16384
                or type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 64*1024*1024):
            raise ValueError("local member feature cache needs its exact bounded input spaces")
        self.gallery, self.space = gallery, descriptor_space
        self.maximum_rows, self.maximum_bytes = maximum_rows, maximum_bytes
        self.rows = {}; self.stored_bytes = 0

    def _frozen(self, features):
        if not isinstance(features, dict):
            raise ValueError("local member feature records must be explicit objects")
        parent = features.get("recorded_native_observation_sha256")
        if (not isinstance(parent, str) or len(parent) != 64
                or any(c not in "0123456789abcdef" for c in parent)
                or features.get("gallery_identity") != self.gallery.identity
                or features.get("contract") != CONTRACT
                or features.get("quality_approved") is not False
                or NativeFaceDescriptorSpace(**features.get("descriptor_space", {})) != self.space):
            raise ValueError("local member feature cache changed parent, gallery or descriptor space")
        heads = features.get("heads")
        if (not isinstance(heads, (list, tuple)) or len(heads) > 24
                or any(type(h.get("observation_id")) is not int or h["observation_id"] < 0 for h in heads)
                or len({h["observation_id"] for h in heads}) != len(heads)):
            raise ValueError("local member feature cache repeated or omitted its head addresses")
        for head in heads:
            if head.get("member_hint") != self.gallery.classify(head.get("embedding"), descriptor_space=self.space):
                raise ValueError("local member feature cache borrowed an uncomputed member hint")
        frozen = json.dumps(features, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return parent, frozen

    def import_features(self, records):
        if not isinstance(records, (tuple, list)) or len(records) > self.maximum_rows:
            raise ValueError("local member feature import exceeds its row budget")
        pending = dict(self.rows); size = self.stored_bytes
        for features in records:
            parent, frozen = self._frozen(features)
            prior = pending.get(parent)
            if prior is not None and prior != frozen:
                raise ValueError("conflicting local member feature records")
            if prior is None:
                pending[parent] = frozen; size += len(frozen.encode())
        if len(pending) > self.maximum_rows or size > self.maximum_bytes:
            raise ValueError("local member feature cache exceeds its row/byte budget")
        self.rows, self.stored_bytes = pending, size

    def add(self, raw, features):
        validate_recorded_member_features(raw, features, gallery=self.gallery, expected_space=self.space)
        self.import_features((features,))

    def get(self, raw):
        frozen = self.rows.get(_digest(raw))
        if frozen is None:
            return None
        value = json.loads(frozen)
        return validate_recorded_member_features(raw, value, gallery=self.gallery, expected_space=self.space)

    def to_dict(self):
        return dict(contract="bounded_local_ive_native_member_feature_cache_v1",
            gallery_identity=self.gallery.identity, descriptor_space=self.space.to_dict(),
            records=[json.loads(self.rows[key]) for key in sorted(self.rows)],
            photo_references_copied=False, original_raw_observations_rewritten=False,
            quality_approved=False)

    @classmethod
    def from_dict(cls, value, *, gallery, descriptor_space, **bounds):
        if (value.get("contract") != "bounded_local_ive_native_member_feature_cache_v1"
                or value.get("gallery_identity") != gallery.identity
                or value.get("descriptor_space") != descriptor_space.to_dict()
                or value.get("photo_references_copied") is not False
                or value.get("original_raw_observations_rewritten") is not False
                or value.get("quality_approved") is not False):
            raise ValueError("local member feature archive changed its explicit input bindings")
        result = cls(gallery, descriptor_space, **bounds)
        result.import_features(value.get("records"))
        return result


def validate_recorded_member_features(raw, features, *, gallery, expected_space):
    if not isinstance(gallery, LocalIVEArcFaceGallery):
        raise ValueError("recorded member hints need the actual local reference gallery")
    for name in ("request_source_index", "native_frame_index", "source_clock_identity",
            "source_file_identity", "native_pixel_sha256"):
        if features.get(name) != raw.get(name):
            raise ValueError("recorded member features changed their exact raw parent")
    if (features.get("contract") != CONTRACT
            or features.get("recorded_native_observation_sha256") != _digest(raw)
            or features.get("gallery_identity") != gallery.identity
            or features.get("quality_approved") is not False
            or features.get("named_member_identity_certified") is not False
            or features.get("persistent_actor_identity_certified") is not False
            or NativeFaceDescriptorSpace(**features.get("descriptor_space", {})) != expected_space
            or expected_space.model_name != ARCFACE_MODEL_NAME):
        raise ValueError("recorded member features changed gallery, descriptor space or authority")
    heads = features.get("heads")
    expected = {h["observation_id"] for h in raw["heads"]}
    if (not isinstance(heads, (list, tuple)) or len(heads) != len(expected)
            or {h.get("observation_id") for h in heads} != expected):
        raise ValueError("member features omitted or repeated observed head occurrences")
    for head in heads:
        wanted = gallery.classify(head.get("embedding"), descriptor_space=expected_space)
        if head.get("member_hint") != wanted:
            raise ValueError("recorded member hint differs from its actual ArcFace vector")
    return features


def observe_recorded_member_features(session, supply, gallery):
    """Count each actual native ArcFace query; never rerun head/body detection."""
    from .face_identity import _validate_normalized_face
    raw = supply.observation
    if raw is None or raw.native_pixel_sha256 is None or session.closed:
        raise ValueError("member feature completion needs a live, measured native raster")
    payload = json.loads(json.dumps(raw.to_dict(), sort_keys=True, allow_nan=False))
    source, native = raw.request_source_index, raw.native_frame_index
    current = session._assert_source_identity(source)
    if supply.source_equivalence_json is None:
        if tuple(current) != raw.source_file_identity:
            raise ValueError("member feature source changed")
    else:
        equivalence = json.loads(supply.source_equivalence_json)
        if (tuple(equivalence["current_file_identity"]) != tuple(current)
                or tuple(equivalence["recorded_file_identity"]) != raw.source_file_identity):
            raise ValueError("member features lost verified archived source equivalence")
    if raw.source_clock_identity != _digest(session.domain.clocks_by_request[source].to_dict()):
        raise ValueError("member feature source clock changed")
    space = native_descriptor_space(session, ARCFACE_MODEL_NAME)
    pixels = session.native_pixels(source, native)
    if pixels is None or hashlib.sha256(memoryview(pixels)).hexdigest() != raw.native_pixel_sha256:
        raise ValueError("member feature exact native raster changed or is unavailable")
    if len(raw.heads) > 24:
        raise ValueError("member feature head inventory is unbounded")
    rows = []
    for head in raw.heads:
        try:
            _validate_normalized_face(head.box_xywh, head.landmarks_xy)
        except ValueError:
            vector = None
        else:
            vector = session.observe_native(source, native, model=ARCFACE_MODEL_NAME,
                arguments=dict(bbox_xywh=head.box_xywh, landmarks_xy=head.landmarks_xy))
        rows.append(dict(observation_id=head.observation_id, embedding=vector,
            member_hint=gallery.classify(vector, descriptor_space=space)))
    if tuple(session._assert_source_identity(source)) != tuple(current):
        raise ValueError("member feature source changed during inference")
    session._assert_model_assets(ARCFACE_MODEL_NAME)
    result = dict(contract=CONTRACT, **{name: payload[name] for name in (
        "request_source_index", "native_frame_index", "source_clock_identity",
        "source_file_identity", "native_pixel_sha256")},
        recorded_native_observation_sha256=_digest(payload), descriptor_space=space.to_dict(),
        gallery_identity=gallery.identity, heads=rows, detector_input_is_recorded=True,
        head_detection_reexecuted=False, named_member_identity_certified=False,
        persistent_actor_identity_certified=False, quality_approved=False)
    return validate_recorded_member_features(payload, result, gallery=gallery, expected_space=space)


def local_member_hint_conflict(left, right):
    """Additional selection evidence; missing/weak hints never remove a head."""
    if left is None or right is None:
        return False
    for hint in (left, right):
        if (hint.get("contract") != CONTRACT or hint.get("quality_approved") is not False
                or hint.get("named_member_identity_certified") is not False):
            raise ValueError("local member hint changed its input provenance or authority")
    if left.get("state") != "LIKELY_MEMBER" or right.get("state") != "LIKELY_MEMBER":
        return False
    if (left.get("gallery_identity") != right.get("gallery_identity")
            or left.get("descriptor_space") != right.get("descriptor_space")):
        return False  # Incompatible evidence is not a measured identity conflict.
    return IVEMember.parse(left["member"]) is not IVEMember.parse(right["member"])
