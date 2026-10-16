"""Source-exact, uncalibrated six-member likeness inputs for StageEdit.

These are cosine measurements, not identity probabilities. UNKNOWN in the
older open-set classifier must not erase them. Existing schemas/checkpoints
are unchanged; an extended model explicitly owns these six extra columns.
No crop, label, source gate, or performer probability is modified here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re

from .ive import IVEMember
from .native_face_descriptor import NativeFaceDescriptorSpace, ARCFACE_MODEL_NAME
from .stage_edit_transformer import StageEditTokenSchema, DEFAULT_STAGE_EDIT_SCHEMA


CONTRACT = 'source_exact_uncalibrated_arcface_member_cosines_v1'
FEATURE_NAMES = tuple('ive_arcface_cosine_' + name for name in (
    'anyujin', 'gaeul', 'rei', 'jangwonyoung', 'liz', 'leeseo'))
_SHA = re.compile('[0-9a-f]{64}')


def _sha(value):
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError('face cues require explicit SHA-256 identities')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def schema_with_face_cues(base=DEFAULT_STAGE_EDIT_SCHEMA):
    if not isinstance(base, StageEditTokenSchema):
        raise TypeError('an explicit StageEdit token schema is required')
    present = set(FEATURE_NAMES).intersection(base.source_feature_names)
    if present:
        if tuple(base.source_feature_names[-6:]) != FEATURE_NAMES:
            raise ValueError('face cue columns must form one complete terminal block')
        return base
    return replace(base, source_feature_names=base.source_feature_names + FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class FaceCue:
    source_id: str
    master_seconds: float
    source_seconds: float
    native_frame: int
    member_cosines: tuple[float, ...]

    def __post_init__(self):
        if (not isinstance(self.source_id, str) or not self.source_id
                or type(self.native_frame) is not int or self.native_frame < 0
                or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                    for v in (self.master_seconds, self.source_seconds))
                or not isinstance(self.member_cosines, tuple) or len(self.member_cosines) != 6
                or any(type(v) not in (float, int) or not math.isfinite(v) or not -1 <= v <= 1
                    for v in self.member_cosines)):
            raise ValueError('source-exact face cues need two clocks, a native frame and six finite cosines')

    @property
    def key(self):
        return self.source_id, self.master_seconds

    def to_dict(self):
        return dict(source_id=self.source_id, master_seconds=self.master_seconds,
            source_seconds=self.source_seconds, native_frame=self.native_frame,
            member_cosines=list(self.member_cosines))


@dataclass(frozen=True, slots=True)
class FaceCueTable:
    """An immutable input table bound to one target-free snapshot, not a song name."""
    base_snapshot_sha256: str
    source_media_sha256: tuple[tuple[str, str], ...]
    gallery_identity: str
    descriptor_space: NativeFaceDescriptorSpace
    records: tuple[FaceCue, ...]

    def __post_init__(self):
        _sha(self.base_snapshot_sha256)
        _sha(self.gallery_identity)
        if (not isinstance(self.descriptor_space, NativeFaceDescriptorSpace)
                or self.descriptor_space.model_name != ARCFACE_MODEL_NAME
                or not isinstance(self.source_media_sha256, tuple)
                or not self.source_media_sha256 or len(self.source_media_sha256) > 32
                or not isinstance(self.records, tuple) or len(self.records) > 50000
                or any(not isinstance(row, FaceCue) for row in self.records)):
            raise ValueError('face table needs a pinned ArcFace space and bounded source-exact records')
        ids = []
        for source_id, digest in self.source_media_sha256:
            if not isinstance(source_id, str) or not source_id:
                raise ValueError('face table source ids must be explicit')
            ids.append(source_id)
            _sha(digest)
        if ids != sorted(set(ids)):
            raise ValueError('face table source domain must be unique and canonical')
        keys = [row.key for row in self.records]
        if keys != sorted(set(keys)) or any(source not in ids for source, _ in keys):
            raise ValueError('face table repeated, reordered or borrowed a source/time')

    @property
    def content_sha256(self):
        return _digest(self.to_dict())

    def to_dict(self):
        return dict(contract=CONTRACT, base_snapshot_sha256=self.base_snapshot_sha256,
            source_media_sha256=dict(self.source_media_sha256), gallery_identity=self.gallery_identity,
            descriptor_space=self.descriptor_space.to_dict(), records=[row.to_dict() for row in self.records],
            member_order=[member.value for member in IVEMember],
            values_are='uncalibrated_cosines_not_probabilities',
            raw_query_or_reference_embeddings_present=False)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError('face table must be an object')
        result = cls(value['base_snapshot_sha256'], tuple(sorted(value['source_media_sha256'].items())),
            value['gallery_identity'], NativeFaceDescriptorSpace(**value['descriptor_space']),
            tuple(FaceCue(row['source_id'], row['master_seconds'], row['source_seconds'],
                row['native_frame'], tuple(row['member_cosines'])) for row in value['records']))
        if value != result.to_dict():
            raise ValueError('face table changed its contract, member order or scalar-only policy')
        return result


def augment_snapshot(snapshot, table, *, base_schema=DEFAULT_STAGE_EDIT_SCHEMA):
    """Keep all old inputs/pose/clocks and append measured values + missing bits."""
    from .stage_edit_feature_materializer import StageEditFeatureSnapshot
    if not isinstance(table, FaceCueTable):
        raise TypeError('an explicit source-bound face cue table is required')
    snapshot.require_schema(base_schema)
    if set(FEATURE_NAMES).intersection(base_schema.source_feature_names):
        raise ValueError('a face cue block cannot be appended or replaced twice')
    if snapshot.snapshot_sha256 != table.base_snapshot_sha256:
        raise ValueError('face cues belong to a different target-free snapshot')
    sources = {c.source_id for token in snapshot.sequence.tokens for c in token.source_candidates}
    if sources != {source for source, _ in table.source_media_sha256}:
        raise ValueError('face cues do not bind the complete snapshot source domain')
    domain = {(c.source_id, t.master_time_seconds) for t in snapshot.sequence.tokens for c in t.source_candidates}
    cues = {row.key: row for row in table.records}
    if set(cues) - domain:
        raise ValueError('face cue has no exact candidate/timestamp in this snapshot')
    tokens = []
    for token in snapshot.sequence.tokens:
        candidates = []
        for candidate in token.source_candidates:
            cue = cues.get((candidate.source_id, token.master_time_seconds))
            values = (0.,)*6 if cue is None else cue.member_cosines
            candidates.append(replace(candidate, features=candidate.features + values,
                missing_mask=candidate.missing_mask + (cue is None,)*6))
        tokens.append(replace(token, source_candidates=tuple(candidates)))
    upstream = dict(snapshot.provenance.upstream_sha256)
    if 'face_member_cues' in upstream:
        raise ValueError('snapshot already has face cue provenance')
    upstream['face_member_cues'] = table.content_sha256
    return StageEditFeatureSnapshot.build(replace(snapshot.sequence, tokens=tuple(tokens)),
        schema=schema_with_face_cues(base_schema),
        provenance=replace(snapshot.provenance, upstream_sha256=tuple(sorted(upstream.items())),
            contracts=snapshot.provenance.contracts + (CONTRACT,)),
        causal_sensing_binding=snapshot.causal_sensing_binding,
        inspection_only_without_causal_sensing=not snapshot.causal_sensing_binding.production_complete)
